"""Local physical ordinary head followed by split overflow; no production edits."""

from __future__ import annotations

import functools
import inspect
import runpy
from pathlib import Path

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.mass_splitting.access import get_state_index
from newton._src.solvers.phoenx.constraints.constraint_container import constraint_get_body1, constraint_get_body2
from newton._src.solvers.phoenx.mass_splitting.slot_cache import _cache_slots_for_partition
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_phoenx_kernels import _make_singleworld_dispatch_func

_CONFIGURED = {}
_INSTALLED = False
_FACTOR = None
_SKIP_AVERAGE = set()


@wp.kernel(enable_backward=False, module="unique")
def stamp_physical_head(
    ids: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    active: wp.array[wp.int32],
    cap: wp.int32,
    copies: CopyStateContainer,
    constraints: ConstraintContainer,
    columns: ContactColumnContainer,
    contact_offset: wp.int32,
    batch: wp.int32,
):
    """Keep overflow caches intact; route all regular endpoints directly."""
    tid = wp.tid()
    if tid < active[0]:
        cid = ids[tid]
        if cid < contact_offset:
            pid = wp.int32(-1)
            if tid >= starts[cap]:
                pid = (tid - starts[cap]) / batch
            b0 = constraint_get_body1(constraints, cid)
            b1 = constraint_get_body2(constraints, cid)
            slot0, count0 = get_state_index(copies, b0, pid)
            slot1, count1 = get_state_index(copies, b1, pid)
            constraints.slot_cache[cid, 0] = slot0
            constraints.slot_cache[cid, 1] = slot1
            constraints.count_cache[cid, 0] = count0
            constraints.count_cache[cid, 1] = count1
        elif tid < starts[cap]:
            _cache_slots_for_partition(cid, wp.int32(-1), copies, constraints, columns, contact_offset)


def _factor_kernel():
    """Clone only scalar factor preparation and change its explicit count source."""
    global _FACTOR
    if _FACTOR is None:
        from newton._src.solvers.phoenx.constraints import bilateral_joint as source

        code = inspect.getsource(source.prepare_bilateral_joint_blocks.func)
        code = code.replace("def prepare_bilateral_joint_blocks(", "def prepare_physical_head_bilateral_blocks(")
        code = code.replace("@wp.kernel(enable_backward=False)", '@wp.kernel(enable_backward=False, module="unique")')
        for side in (0, 1):
            old = f"wp.max(copy_state.count_per_node[body{side}], wp.int32(1))"
            new = f"wp.max(constraints.count_cache[cid, {side}], wp.int32(1))"
            if old not in code:
                raise RuntimeError("Bilateral source changed; re-audit the local factor clone")
            code = code.replace(old, new)
        path = Path("/tmp/physical_head_overflow_factor.py")
        path.write_text(code)
        namespace = dict(vars(source))
        exec(compile(code, str(path), "exec"), namespace)
        _FACTOR = namespace["prepare_physical_head_bilateral_blocks"]
    return _FACTOR


@functools.cache
def _kernel(phase, flags, overflow):
    """Use native callbacks with one color per head launch or one lane per tail batch."""
    dispatch_flags = dict(flags)
    callback, _ = _make_singleworld_dispatch_func(
        **dispatch_flags,
        is_prepare=phase == "prepare",
        is_cached_prepare=False,
        use_bias=phase == "iterate",
    )

    @wp.kernel(enable_backward=False, module="unique")
    def execute(
        constraints: ConstraintContainer,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
        ids: wp.array[wp.int32],
        starts: wp.array[wp.int32],
        colors: wp.array[wp.int32],
        joint_enabled: wp.array[wp.int32],
        num_joints: wp.int32,
        num_bodies: wp.int32,
        color: wp.int32,
        batch: wp.int32,
        idt: wp.float32,
        sor: wp.float32,
    ):
        lane = wp.tid()
        if color < colors[0]:
            start = starts[color]
            count = starts[color + wp.int32(1)] - start
            steps = wp.int32(1)
            pid = wp.int32(-1)
            if wp.static(overflow):
                steps = batch
                pid = lane
            for inner in range(steps):
                offset = lane * steps + inner
                if offset < count:
                    callback(
                        constraints,
                        columns,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copies,
                        num_joints,
                        joint_enabled,
                        wp.int32(0),
                        wp.int32(0),
                        wp.int32(0),
                        wp.int32(0),
                        num_bodies,
                        idt,
                        sor,
                        ids[start + offset],
                        start + offset,
                        pid,
                    )

    return execute


def _validate(world):
    """Reject unsupported policy combinations before launching this diagnostic."""
    if world.step_layout != "single_world" or not world.mass_splitting_enabled:
        raise ValueError("Physical-head prototype requires single-world mass splitting")
    if world._color_group_data is not None:
        raise ValueError("Use mass_splitting_color_group_size=0")
    if world.max_colored_partitions is None or int(world.max_colored_partitions) < 1:
        raise ValueError("Use a positive ordinary color cap")
    if world._dispatch_specialization_flags()["cloth_support"] or world.num_particles:
        raise ValueError("Prototype supports rigid rows only")
    if world._partitioner._symmetric_sweep:
        raise ValueError("Reverse/symmetric sweeps need a separate phase schedule")
    if world._sleeping_enabled or world.prepare_refresh_stride != 1:
        raise ValueError("Sleeping and cached geometry reuse are not covered")
    if world._direct_contact_response is not None or world._maximal_tree_projector is not None:
        raise ValueError("Global contact/tree projectors are not covered")
    if world._reduced_articulation is not None:
        raise ValueError("Articulation-owned constraints are not covered")
    direct = getattr(world, "_direct_equality_system", None)
    if direct is not None:
        from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem

        if not isinstance(direct, BlockJointSystem) or direct.requires_global_projection:
            raise ValueError("Only unbounded local BlockJointSystem is covered")
    if world.num_joints and not world.constraints.bilateral.enabled:
        raise ValueError("Use actual block_pgs joint ownership")


def _flags(world):
    flags = world._dispatch_specialization_flags()
    return tuple(
        sorted(
            {
                **flags,
                "has_mass_splitting": True,
                "packed_contact_headers": world._colored_contact_headers,
                "patch_friction": world._contact_patch_enabled,
                "bilateral_joint_blocks": bool(world.constraints.bilateral.enabled),
            }.items()
        )
    )


def run_head(world, phase, idt, contact_container=None):
    """Expose the native physical-head boundary for independent phase tests."""
    _validate(world)
    for color in range(int(world.max_colored_partitions)):
        _launch(world, phase, idt, color, False, contact_container)


def run_overflow(world, phase, idt, contact_container=None):
    """Run native split tail only; caller owns broadcast and averaging."""
    _launch(world, phase, idt, int(world.max_colored_partitions), True, contact_container)


def _launch(world, phase, idt, color, overflow, cc):
    if cc is None:
        cc = world._contact_container_solve if world._colored_contact_rows else world._contact_container
    wp.launch(
        _kernel(phase, _flags(world), overflow),
        dim=max(1, world._constraint_capacity),
        inputs=[
            world.constraints,
            world._contact_cols_packed if world._colored_contact_headers else world._contact_cols,
            world.bodies,
            world._particles_or_sentinel(),
            cc,
            world._active_contact_views(),
            world._copy_state,
            world._partitioner.element_ids_by_color,
            world._partitioner.color_starts,
            world._partitioner.num_colors,
            world._joint_pgs_enabled,
            wp.int32(world.num_joints),
            wp.int32(world.num_bodies),
            wp.int32(color),
            wp.int32(world.mass_splitting_batch_size),
            idt,
            wp.float32(world.sor_boost),
        ],
        device=world.device,
    )


def configure_world(world):
    """Enable the local schedule on a preconstructed supported native world."""
    _install_hooks()
    _validate(world)
    _factor_kernel()
    _CONFIGURED[id(world)] = world
    return world


def _install_hooks():
    global _INSTALLED
    if _INSTALLED:
        return
    from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

    old_rebuild = PhoenXWorld._rebuild_mass_splitting_graph
    old_sweep = PhoenXWorld._singleworld_head_plus_tail_sweep
    old_factor = BlockJointSystem.prepare_and_factor
    old_average = PhoenXWorld._mass_splitting_average_and_broadcast

    def rebuild(world):
        old_rebuild(world)
        if id(world) in _CONFIGURED:
            wp.launch(
                stamp_physical_head,
                dim=max(1, world._constraint_capacity),
                inputs=[
                    world._partitioner.element_ids_by_color,
                    world._partitioner.color_starts,
                    world._num_active_constraints,
                    wp.int32(world.max_colored_partitions),
                    world._copy_state,
                    world.constraints,
                    world._contact_cols,
                    wp.int32(world._contact_offset),
                    wp.int32(world.mass_splitting_batch_size),
                ],
                device=world.device,
            )

    def factor(system, idt):
        world = system._block_world
        if id(world) not in _CONFIGURED:
            return old_factor(system, idt)
        _validate(world)
        if system.enabled:
            wp.launch(
                _factor_kernel(),
                dim=world.num_joints,
                inputs=[world.constraints, world.bodies, world._copy_state],
                device=world.device,
            )

    def sweep(world, head_kernel, tail_kernel, idt, contact_container=None):
        if id(world) not in _CONFIGURED:
            return old_sweep(world, head_kernel, tail_kernel, idt, contact_container)
        kernels = world._singleworld_kernels()
        phases = {kernels[0]: "prepare", kernels[2]: "iterate", kernels[4]: "relax"}
        if head_kernel not in phases:
            raise ValueError("Uncovered cached preparation callback")
        phase = phases[head_kernel]
        run_head(world, phase, idt, contact_container)
        world._mass_splitting_broadcast()
        run_overflow(world, phase, idt, contact_container)
        # Publish the tail mean before the next physical head, also for warm start.
        old_average(world, 1.0 / world.substep_dt)
        world._mass_splitting_writeback(already_averaged=True)
        _SKIP_AVERAGE.add(id(world))

    def average(world, inv_dt):
        if id(world) in _SKIP_AVERAGE:
            _SKIP_AVERAGE.remove(id(world))
            return
        return old_average(world, inv_dt)

    PhoenXWorld._rebuild_mass_splitting_graph = rebuild
    PhoenXWorld._singleworld_head_plus_tail_sweep = sweep
    BlockJointSystem.prepare_and_factor = factor
    PhoenXWorld._mass_splitting_average_and_broadcast = average
    _INSTALLED = True


def install():
    """Configure every subsequently constructed public solver in this process."""
    _install_hooks()
    from newton._src.solvers.phoenx.solver import SolverPhoenX

    original = SolverPhoenX.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        configure_world(self.world)

    SolverPhoenX.__init__ = initialize


if __name__ == "__main__":
    install()
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
