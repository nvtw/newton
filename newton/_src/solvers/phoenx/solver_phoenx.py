# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of PhoenX's ``Scene.Step`` driver -- contacts + actuated
double-ball-socket joints.

Supports :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` (ball-socket
/ revolute / prismatic with optional PD drives and limits) and
:data:`CONSTRAINT_TYPE_CONTACT`. Newton's :class:`CollisionPipeline`
produces contacts; this solver consumes them.

Per-step flow mirrors ``World.Step.cs``:
    1. Ingest contacts.
    2. Rebuild element view + Jones-Plassmann colouring once.
    3. Substep loop: integrate forces + gravity, main PGS solve
       (bias=True), integrate positions, position iterate, relax
       (bias=False).
    4. Damping + rotated-inertia refresh.
    5. Clear force accumulators.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_KINEMATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.body_or_particle import BodyOrParticleStore
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    JOINT_MODE_REVOLUTE,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    CLOTH_TRIANGLE_DWORDS,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    CONTACT_DWORDS,
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_pair_wrench_kernel,
    contact_per_contact_error_kernel,
    contact_per_contact_wrench_kernel,
    contact_views_make,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    CC_DERIVED_DWORDS_PER_CONTACT,
    CC_DWORDS_PER_CONTACT,
    ContactContainer,
    contact_container_swap_prev_current,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_ingest import (
    IngestScratch,
    gather_contact_warmstart,
    ingest_contacts,
    stamp_forward_contact_map,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    MAX_COLORS,
    IncrementalContactPartitioner,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int
from newton._src.solvers.phoenx.materials import MaterialData
from newton._src.solvers.phoenx.particle import (
    ParticleContainer,
    particle_container_zeros,
)
from newton._src.solvers.phoenx.solver_config import (
    FUSE_TAIL_BLOCK_DIM,
    FUSE_TAIL_MAX_COLOR_SIZE,
    NUM_INNER_WHILE_ITERATIONS,
    PHOENX_USE_GREEDY_COLORING,
)
from newton._src.solvers.phoenx.solver_kernels import (
    _accumulate_substep_velocity_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _PER_WORLD_COLORING_BLOCK_DIM,
    _STRAGGLER_BLOCK_DIM,
    _build_scatter_keys_kernel,
    _choose_fast_tail_worlds_per_block,
    _constraint_gather_errors_kernel,
    _constraint_gather_wrenches_kernel,
    _constraint_iterate_singleworld_cloth_kernel,
    _constraint_iterate_singleworld_fused_cloth_kernel,
    _constraint_iterate_singleworld_fused_kernel,
    _constraint_iterate_singleworld_fused_revolute_cloth_kernel,
    _constraint_iterate_singleworld_fused_revolute_kernel,
    _constraint_iterate_singleworld_kernel,
    _constraint_iterate_singleworld_revolute_cloth_kernel,
    _constraint_iterate_singleworld_revolute_kernel,
    _constraint_prepare_plus_iterate_fast_tail_kernel,
    _constraint_prepare_plus_iterate_fast_tail_revolute_kernel,
    _constraint_prepare_singleworld_cloth_kernel,
    _constraint_prepare_singleworld_fused_cloth_kernel,
    _constraint_prepare_singleworld_fused_kernel,
    _constraint_prepare_singleworld_fused_revolute_cloth_kernel,
    _constraint_prepare_singleworld_fused_revolute_kernel,
    _constraint_prepare_singleworld_kernel,
    _constraint_prepare_singleworld_revolute_cloth_kernel,
    _constraint_prepare_singleworld_revolute_kernel,
    _constraint_relax_fast_tail_kernel,
    _constraint_relax_fast_tail_revolute_kernel,
    _constraint_relax_singleworld_cloth_kernel,
    _constraint_relax_singleworld_fused_cloth_kernel,
    _constraint_relax_singleworld_fused_kernel,
    _constraint_relax_singleworld_fused_revolute_cloth_kernel,
    _constraint_relax_singleworld_fused_revolute_kernel,
    _constraint_relax_singleworld_kernel,
    _constraint_relax_singleworld_revolute_cloth_kernel,
    _constraint_relax_singleworld_revolute_kernel,
    _constraints_to_elements_cloth_kernel,
    _constraints_to_elements_kernel,
    _count_elements_per_world_kernel,
    _integrate_velocities_kernel,
    _kinematic_interpolate_substep_kernel,
    _kinematic_prepare_step_kernel,
    _per_world_greedy_coloring_kernel,
    _per_world_jp_coloring_kernel,
    _phoenx_apply_forces_and_gravity_kernel,
    _phoenx_apply_global_damping_kernel,
    _phoenx_init_cloth_triangle_rows_kernel,
    _phoenx_refresh_world_inertia_kernel,
    _phoenx_update_inertia_and_clear_forces_kernel,
    _pick_threads_per_world_kernel,
    _reduce_total_colours_kernel,
    _reset_head_active_kernel,
    _set_kinematic_pose_batch_kernel,
    _sync_num_active_constraints_kernel,
    pack_body_xforms_kernel,
)

__all__ = [
    "DEFAULT_SHAPE_GAP",
    "PhoenXWorld",
    "pack_body_xforms_kernel",
]


#: Default contact-detection gap [m] for shapes in PhoenX scenes.
#: Generous (5 cm) so contacts emit a few frames before impact;
#: PhoenX's speculative branch decelerates closing bodies while they
#: still have a gap, which is easier to stabilise than penetration
#: recovery. Scene-scale-sensitive: override for MEMS / vehicle scales.
DEFAULT_SHAPE_GAP: float = 0.05


def _build_gravity_array(gravity, num_worlds: int, device) -> wp.array[wp.vec3f]:
    """Coerce ``gravity`` into a ``wp.array[wp.vec3f]`` of length
    ``num_worlds``. Accepts a single 3-tuple (broadcast) or an
    iterable of ``num_worlds`` 3-tuples."""
    g_list = list(gravity) if hasattr(gravity, "__iter__") else None
    if g_list is not None and len(g_list) == 3 and not hasattr(g_list[0], "__iter__"):
        vec = (float(g_list[0]), float(g_list[1]), float(g_list[2]))
        data = [vec] * num_worlds
    else:
        if g_list is None:
            raise ValueError("gravity must be a tuple or an iterable of tuples")
        if len(g_list) != num_worlds:
            raise ValueError(f"gravity length {len(g_list)} != num_worlds {num_worlds}")
        data = []
        for row in g_list:
            r = tuple(row)
            if len(r) != 3:
                raise ValueError(f"per-world gravity entry must be 3 components; got {len(r)}")
            data.append((float(r[0]), float(r[1]), float(r[2])))
    arr_np = np.asarray(data, dtype=np.float32)
    return wp.array(arr_np, dtype=wp.vec3f, device=device)


#: Block dim used by the single-world PGS sweep kernels.
#: 256 threads / block matches Warp's default and NarrowPhase's choice;
#: the kernels are register-heavy (~168 registers/thread) so a smaller
#: block would not improve occupancy further.
_SINGLEWORLD_BLOCK_DIM: int = 256


def _singleworld_total_threads(
    constraint_capacity: int,
    device,
    max_thread_blocks: int | None = None,
) -> int:
    """Persistent grid size for the single-world PGS sweep kernels.

    Mirrors :class:`NarrowPhase`'s "saturate without overprovisioning"
    sizing: 4 blocks/SM on CUDA (capped by capacity and a 32-block
    floor) so the CUDA graph stays stable while avoiding the old
    one-thread-per-cid 900K-grid waste.

    Args:
        constraint_capacity: Upper bound on active cids per colour.
        device: Warp device (only ``sm_count`` is read).
        max_thread_blocks: Optional hard cap on the persistent grid's
            block count. When set, replaces the default ``4 *
            sm_count`` (CUDA) / ``256`` (CPU) cap *and* the 32-block
            floor; the grid is sized to ``min(capacity_blocks,
            max_thread_blocks)`` (still at least 1 block). Use this
            to share the GPU with a co-resident workload or to
            measure SM-occupancy effects. Only the single-world
            layout's PGS sweeps (prepare, main iterate, velocity
            relax) read this; the multi-world fast-tail path is
            unaffected.

    Returns:
        Persistent grid size in threads, multiple of
        :data:`_SINGLEWORLD_BLOCK_DIM`.
    """
    block_dim = _SINGLEWORLD_BLOCK_DIM
    capacity_blocks = (max(1, int(constraint_capacity)) + block_dim - 1) // block_dim
    if max_thread_blocks is not None:
        if int(max_thread_blocks) < 1:
            raise ValueError(f"max_thread_blocks must be >= 1 (got {max_thread_blocks})")
        # Opt-in mode: respect the user's cap verbatim, bypassing the
        # 32-block floor and SM-derived ceiling. Still clamped above
        # by ``capacity_blocks`` because launching more blocks than
        # there are cids strands work without benefit.
        num_blocks = max(1, min(capacity_blocks, int(max_thread_blocks)))
        return block_dim * num_blocks
    device_obj = wp.get_device(device)
    if device_obj.is_cuda:
        # 4 blocks/SM hits good occupancy without overprovisioning.
        max_blocks_limit = device_obj.sm_count * 4
    else:
        max_blocks_limit = 256
    # ``min_blocks = 32`` gives 8K threads minimum -- enough to keep
    # every SM warm on small GPUs without stranding work on large
    # ones. The ``capacity_blocks`` cap prevents a 900K-cid capacity
    # from demanding more threads than there are cids.
    min_blocks = 32
    num_blocks = max(min_blocks, min(capacity_blocks, max_blocks_limit))
    return block_dim * num_blocks


class PhoenXWorld:
    """PhoenX solver driver. Owns a :class:`BodyContainer`, a
    :class:`ConstraintContainer`, and per-step contact ingest /
    warm-start state. :meth:`step` advances every rigid body by ``dt``
    seconds.

    Joint columns occupy cids ``[0, num_joints)``; contact columns
    occupy ``[num_joints, num_joints + max_contact_columns)``. Only
    :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` joints are
    supported; call :meth:`initialize_actuated_double_ball_socket_joints`
    to populate them before the first :meth:`step`.
    """

    @dataclass
    class StepReport:
        """Diagnostic snapshot of the most recent :meth:`step`.

        All counters refer to the just-finished step; producing the
        report performs a handful of device-to-host copies and is not
        graph-capture safe -- only call it from host code outside the
        captured region.
        """

        num_colors: int
        """Number of graph colours used by the last PGS colouring.

        For ``step_layout="single_world"`` this is the global colour
        count. For ``"multi_world"`` it is the maximum colour count
        across all worlds (i.e. the depth of the per-world PGS sweep)."""

        color_sizes: list[int]
        """Element count per colour, length :attr:`num_colors`.

        Single-world: per-colour element counts of the global
        colouring. Multi-world: sum across worlds of the elements
        assigned to each colour index ``c`` (worlds with fewer than
        ``c+1`` colours contribute zero)."""

        per_world_num_colors: list[int] | None
        """Per-world colour counts, length ``num_worlds``. ``None`` for
        the single-world layout."""

        per_world_color_sizes: list[list[int]] | None
        """Per-world per-colour element counts. Outer list has one
        entry per world; inner list ``i`` has length
        ``per_world_num_colors[i]``. ``None`` for the single-world
        layout."""

        num_contact_columns: int
        """Active contact columns processed in the last step (zero if
        :meth:`step` was called without contacts)."""

        num_joints: int
        """Number of joint constraint columns; static for the lifetime
        of the world."""

        num_active_constraints: int
        """``num_joints + num_contact_columns`` -- total cids that the
        last colouring partitioned."""

        max_body_degree: int
        """Maximum number of active constraint columns incident to any
        single body in the last :meth:`step`. This is a hard lower
        bound on the number of colours any valid graph colouring of
        the constraint conflict graph can use, so comparing it against
        :attr:`num_colors` shows how close the colourer is to the
        theoretical optimum. ``0`` if :meth:`step` has not been called
        or there are no active constraints."""

    def __init__(
        self,
        bodies: BodyContainer,
        constraints: ConstraintContainer,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 1,
        gravity: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        rigid_contact_max: int = 0,
        num_joints: int = 0,
        num_particles: int = 0,
        num_cloth_triangles: int = 0,
        collision_filter_pairs: Iterable[tuple[int, int]] | None = None,
        default_friction: float = 0.5,
        num_worlds: int = 1,
        step_layout: str = "multi_world",
        threads_per_world: int | str = "auto",
        max_thread_blocks: int | None = None,
        enable_body_pair_grouping: bool = False,
        device: wp.context.Devicelike = None,
    ):
        """Take ownership of pre-built body and constraint containers.

        Args:
            bodies: Rigid body SoA. Caller sets initial pose, mass,
                inertia; solver reads/writes velocity, position,
                orientation, forces, rotated inertia.
            constraints: Shared container. Joints at cids
                ``[0, num_joints)``; contacts at
                ``[num_joints, num_joints + max_contact_columns)``.
                Per-constraint dword width must be at least
                ``max(CONTACT_DWORDS, ADBS_DWORDS)`` when
                ``num_joints > 0``. :meth:`make_constraint_container`
                handles the sizing.
            substeps: Substeps per :meth:`step` call.
            solver_iterations: PGS iterations per substep (main solve,
                bias=True).
            velocity_iterations: TGS-soft relax sweeps per substep
                (bias=False). Defaults to ``1``; ``0`` recovers raw
                PhoenX but accumulates Baumgarte drift on tall stacks.
            gravity: Constant world-space gravity [m/s^2]. A 3-tuple
                broadcasts; an iterable of 3-tuples gives each world
                its own vector.
            rigid_contact_max: Upper bound on the Newton ``Contacts``
                buffer's rigid-contact range. Sizes the contact-column
                capacity 1:1 (one column per ``(shape_a, shape_b)``
                pair); ``0`` disables contact paths entirely.
            num_joints: Actuated-DBS joint columns reserved at
                ``[0, num_joints)``. Populate via
                :meth:`initialize_actuated_double_ball_socket_joints`
                before the first step.
            collision_filter_pairs: Optional ``(body_a, body_b)`` pairs
                whose contacts are dropped on ingest.
            default_friction: Friction used when no per-shape material
                is registered (see :meth:`set_materials`).
            num_worlds: Number of independent sub-worlds. Gating uses
                ``BodyContainer.world_id``.
            step_layout: ``"multi_world"`` (default) uses the per-world
                fast-tail path (one warp per world; scales beyond ~256
                worlds). ``"single_world"`` drives the global
                Jones-Plassmann colouring with per-colour persistent
                grid launches via ``wp.capture_while``; wins when the
                scene is one or a few very big worlds.
            threads_per_world: Effective threads-per-world for the
                multi-world fast-tail kernels. ``"auto"`` (default)
                picks per-step from the colour-size histogram;
                ``32`` = one warp per world (legacy), ``16`` = two,
                ``8`` = four (rarely wins). Grid is always
                ``num_worlds * 32`` lanes with the surplus
                early-exiting, so this knob is graph-capture safe.
            max_thread_blocks: Optional hard cap on the persistent
                grid used by the single-world PGS sweeps (prepare,
                main iterate, velocity relax). When ``None``
                (default) the grid is sized as
                ``clamp(ceil(cap / 256), 32, 4 * sm_count)`` blocks
                of 256 threads. When set, replaces both the
                32-block floor and the SM-derived ceiling, so the
                grid becomes ``min(ceil(cap / 256),
                max_thread_blocks)`` blocks. Use this to share the
                GPU with a co-resident workload or to measure
                occupancy. No effect on
                ``step_layout="multi_world"``.
            device: Warp device. Defaults to ``bodies.position.device``.
        """
        if device is None:
            self.device = bodies.position.device
        else:
            self.device = wp.get_device(device)

        self.bodies: BodyContainer = bodies
        self.constraints: ConstraintContainer = constraints

        self.num_bodies: int = int(bodies.position.shape[0])
        # Count kinematic bodies once at construction so the per-substep
        # kinematic kernels can short-circuit when no body is scripted.
        # ``motion_type`` is written by the builder and never mutated by
        # the solver, so a single host count at init stays accurate.
        if self.num_bodies > 0:
            mt = bodies.motion_type.numpy()
            self._num_kinematic_bodies: int = int((mt == int(MOTION_KINEMATIC)).sum())
        else:
            self._num_kinematic_bodies = 0
        # One contact column covers an entire ``(shape_a, shape_b)``
        # pair regardless of contact count, so the column count equals
        # ``rigid_contact_max`` 1:1. The ``max(1, ...)`` keeps the
        # contact-free path sizing-safe (zero-length wp.array2d isn't
        # legal).
        self.rigid_contact_max: int = int(rigid_contact_max)
        self.max_contact_columns: int = max(1, self.rigid_contact_max) if self.rigid_contact_max > 0 else 0
        self.num_joints: int = int(num_joints)
        if self.num_joints < 0:
            raise ValueError(f"num_joints must be >= 0 (got {self.num_joints})")

        # Particles -- allocated lazily, ``None`` when ``num_particles
        # == 0`` so existing rigid-body-only scenes pay zero memory and
        # zero substep-loop overhead. The unified body-or-particle index
        # space (see :mod:`newton._src.solvers.phoenx.body_or_particle`)
        # places particle slot ``i_p`` at unified index ``num_bodies +
        # i_p``; constraint kernels that address either kind of "thing"
        # consume the unified index via the
        # :class:`BodyOrParticleStore` accessor helpers.
        self.num_particles: int = int(num_particles)
        if self.num_particles < 0:
            raise ValueError(f"num_particles must be >= 0 (got {self.num_particles})")
        self.particles: ParticleContainer | None = None
        if self.num_particles > 0:
            self.particles = particle_container_zeros(self.num_particles, device=device)

        # Cloth triangles -- count of XPBD-cloth-triangle constraint
        # rows the scene reserves in the joint-side
        # :class:`ConstraintContainer`. ``> 0`` selects the
        # ``cloth_support=True`` kernel binaries via
        # :meth:`_singleworld_kernels`; the cloth dispatch branch
        # reads each cid's ``constraint_type`` tag at dword 0 and
        # routes :data:`CONSTRAINT_TYPE_CLOTH_TRIANGLE` cids to
        # :func:`cloth_triangle_iterate_at`. Default ``0`` keeps
        # rigid-only scenes on the existing kernel binaries
        # bit-for-bit.
        self.num_cloth_triangles: int = int(num_cloth_triangles)
        if self.num_cloth_triangles < 0:
            raise ValueError(f"num_cloth_triangles must be >= 0 (got {self.num_cloth_triangles})")
        # Per-triangle particle indices (vec4i; 4th slot reserved for
        # tets). Length 0 placeholder when there are no triangles --
        # the contact kernels take ``tri_indices`` unconditionally.
        # :class:`SolverPhoenX` overwrites this with its own allocation
        # post-construction when cloth is present.
        with wp.ScopedDevice(device):
            self.tri_indices: wp.array = wp.zeros(
                max(self.num_cloth_triangles, 0), dtype=wp.vec4i, device=device
            )
        # Total cid count in the joint-side :class:`ConstraintContainer`.
        # Joints + cloth triangles all live in this container; the
        # dispatcher uses ``cid < num_joint_container_cids`` as the
        # boundary that separates joint-container cids (joints + cloth)
        # from contact-column cids. ``self.num_joints`` keeps its
        # original meaning -- just rigid joints -- so joint-init
        # kernels stay scoped correctly.
        self._joint_container_cids: int = self.num_joints + self.num_cloth_triangles
        # Cached :class:`BodyOrParticleStore` exposed via the
        # :attr:`body_or_particle` property; lazy so rigid-only scenes
        # never allocate the sentinel particle container.
        self._body_or_particle_store: BodyOrParticleStore | None = None

        self.substeps = int(substeps)
        if self.substeps <= 0:
            raise ValueError(f"substeps must be >= 1 (got {self.substeps})")
        self.solver_iterations = int(solver_iterations)
        if self.solver_iterations < 1:
            raise ValueError(f"solver_iterations must be >= 1 (got {self.solver_iterations})")
        self.velocity_iterations = int(velocity_iterations)
        if self.velocity_iterations < 0:
            raise ValueError(f"velocity_iterations must be >= 0 (got {self.velocity_iterations})")

        # Joint-type specialisation flag for the single-world kernels.
        # ``True`` means every joint is revolute (or there are none),
        # so :meth:`_singleworld_kernels` returns the revolute-only
        # variants that skip the ``joint_mode`` global read + dispatch
        # branch in the iterate hot loop. Defaults to ``True`` for
        # ``num_joints == 0`` (the joint branch is dead anyway, and
        # the smaller binary helps occupancy / icache); flipped to
        # ``False`` if :meth:`initialize_actuated_double_ball_socket_joints`
        # observes any non-revolute mode. Must be stable before
        # ``wp.ScopedCapture`` records the step.
        self._use_revolute_specialization: bool = True

        self.num_worlds: int = int(num_worlds)
        if self.num_worlds <= 0:
            raise ValueError(f"num_worlds must be >= 1 (got {self.num_worlds})")
        if step_layout not in ("multi_world", "single_world"):
            raise ValueError(f"step_layout must be 'multi_world' or 'single_world' (got {step_layout!r})")
        self.step_layout: str = step_layout
        # Threads-per-world. ``"auto"`` lets the GPU picker decide every
        # step from colour stats; an int forces a fixed value (validated
        # against the {8, 16, 32} set the kernels have been tuned for).
        # ``_tpw_choice`` is a 1-element GPU buffer that the fast-tail
        # kernels read at the top of every launch -- see
        # :func:`_pick_threads_per_world_kernel` for the heuristic.
        if isinstance(threads_per_world, str):
            if threads_per_world != "auto":
                raise ValueError(f"threads_per_world must be 'auto' or one of (8, 16, 32) (got {threads_per_world!r})")
            # Host-side fast-path: the picker's saturation gate for
            # tpw=16 is "num_worlds >= 8 * sm_count". Below that, the
            # picker would always emit tpw=32, so we skip the per-step
            # launch entirely and pin to 32. Avoids ~10us/step of
            # picker overhead on small fleets where it would never
            # pay off anyway.
            _sm = getattr(self.device, "sm_count", 0) or 1
            if self.num_worlds < 8 * _sm:
                self._tpw_auto: bool = False
                initial_tpw = _STRAGGLER_BLOCK_DIM
            else:
                self._tpw_auto = True
                initial_tpw = _STRAGGLER_BLOCK_DIM
        else:
            tpw_int = int(threads_per_world)
            if tpw_int not in (8, 16, 32):
                raise ValueError(f"threads_per_world must be 'auto' or one of (8, 16, 32) (got {tpw_int})")
            self._tpw_auto = False
            initial_tpw = tpw_int
        self._tpw_choice: wp.array[wp.int32] = wp.array([initial_tpw], dtype=wp.int32, device=self.device)
        # Scratch for the picker's parallel colour-count reduction.
        # Reset to 0 each step before the reduction kernel runs.
        self._tpw_total_colours: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.gravity: wp.array[wp.vec3f] = _build_gravity_array(gravity, self.num_worlds, self.device)
        self.default_friction = float(default_friction)

        # Global per-substep damping. ``None`` until the user opts in via
        # :meth:`set_global_linear_damping` / :meth:`set_global_angular_damping`;
        # while ``None`` the per-substep kernel is skipped entirely (zero
        # cost). Once allocated, kept allocated for the lifetime of the
        # solver -- changing the factor between graph replays is safe (the
        # captured graph reads from the same device slot), but **opting in
        # mid-simulation requires re-capturing any existing CUDA graph**
        # since the captured graph doesn't include the new kernel launch.
        self._global_damping: wp.array[wp.float32] | None = None
        self._global_damping_host: np.ndarray | None = None

        # ----- Step time bookkeeping -----
        self.step_dt: float = 0.0
        self.inv_step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # Joint cids at ``[0, num_joints)``; contact cids follow.
        self._constraint_capacity: int = max(1, self._joint_container_cids + self.max_contact_columns)

        # Persistent grid size for the single-world PGS sweep kernels.
        # Fixed at construction so every colour launch uses the same
        # ``dim`` -- required to keep the outer CUDA graph capture
        # stable across varying per-colour active counts. The
        # optional ``max_thread_blocks`` opt-in caps the persistent
        # grid (and bypasses the 32-block floor); see
        # :func:`_singleworld_total_threads`.
        if max_thread_blocks is not None and int(max_thread_blocks) < 1:
            raise ValueError(f"max_thread_blocks must be >= 1 (got {max_thread_blocks})")
        self._max_thread_blocks: int | None = int(max_thread_blocks) if max_thread_blocks is not None else None
        self._singleworld_total_threads: int = _singleworld_total_threads(
            self._constraint_capacity,
            self.device,
            max_thread_blocks=self._max_thread_blocks,
        )

        # Head capture-while predicate. Starts each sweep at 1; the
        # persistent-grid sweep zeroes it on the hand-off to the fused
        # tail kernel (colour size ``<= FUSE_TAIL_MAX_COLOR_SIZE``).
        # A separate flag (vs. reusing ``color_cursor``) is what lets
        # the tail kernel resume at the exact cursor position. When
        # the feature is disabled (``fuse_threshold = 0``), the hand-
        # off branch is unreachable; termination then comes from the
        # kernel's ``color_cursor == 0`` early-exit and the reset
        # kernel below is the only per-sweep bookkeeping needed.
        self._head_active: wp.array[wp.int32] = wp.ones(1, dtype=wp.int32, device=self.device)
        self._fuse_threshold: int = int(FUSE_TAIL_MAX_COLOR_SIZE)
        self._fuse_tail_block_dim: int = int(FUSE_TAIL_BLOCK_DIM)

        # ----- Partitioner + per-world CSR buffers -----
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            self._constraint_capacity, dtype=ElementInteractionData, device=self.device
        )
        # Joints are the only active cids until the first ingest.
        self._num_active_constraints: wp.array[int] = wp.array(
            [self._joint_container_cids], dtype=wp.int32, device=self.device
        )
        # Graph-coloring node count covers both bodies and particles
        # because cloth-triangle endpoints are unified indices in
        # ``[num_bodies, num_bodies + num_particles)``. The colourer
        # builds a CSR over interaction endpoints, so its node-count
        # bound has to span the entire unified-index range or the
        # particle-side adjacency lookups stomp out of bounds.
        self._partitioner = IncrementalContactPartitioner(
            max_num_interactions=self._constraint_capacity,
            max_num_nodes=max(1, self.num_bodies + self.num_particles),
            device=self.device,
            use_tile_scan=True,
        )
        # Tracks the live single-world coloring choice. Starts at the
        # config default; flipped to False (round-based JP) by
        # :meth:`step` if a non-captured greedy build overflows the
        # 64-color bitmask. Once flipped, we never flip back -- the
        # JP path has no chromatic-number bound and is the safe
        # fallback for any graph the greedy variant can't fit.
        self._use_greedy_coloring: bool = bool(PHOENX_USE_GREEDY_COLORING)

        cap = self._constraint_capacity
        nw = self.num_worlds
        self._world_element_ids_by_color: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._world_color_starts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_csr_offsets: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        # Shifted staging buffer for the cross-world prefix scan. Sized
        # ``num_worlds + 1`` so the inclusive scan input/output lengths
        # match and the scan result lands directly in
        # ``world_csr_offsets``.
        self._world_totals_shifted: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        self._world_num_colors: wp.array[wp.int32] = wp.zeros(nw, dtype=wp.int32, device=self.device)

        # ----- Per-world coloring scratch (for parallel JP path) -----
        #
        # The per-world coloring path parallelises the JP MIS loop over
        # worlds (one block per world); it's the only multi_world
        # coloring strategy now. Single-world layouts route through
        # ``partitioner.build_csr`` instead -- see the dispatch in
        # :meth:`step`.
        self._per_world_element_count: wp.array[wp.int32] = wp.zeros(nw, dtype=wp.int32, device=self.device)
        # Exclusive prefix of per-world counts; sized nw+1 so
        # ``world_element_offsets[w+1] - world_element_offsets[w]`` is
        # always a legal read.
        self._per_world_element_offsets: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        # Flat buffer that holds each world's active cids contiguously
        # (bucketed by world_id of ``bodies[0]``). Sized to ``2 * cap``
        # because the deterministic scatter uses
        # :func:`wp.utils.radix_sort_pairs`, which needs a ping-pong
        # second half. Only the first ``cap`` entries (the sort
        # output) are read by the JP coloring; the second half is
        # scratch.
        self._per_world_elements: wp.array[wp.int32] = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        # Per-cid world-id keys for the scatter sort. Same ping-pong
        # sizing as ``_per_world_elements``.
        self._per_world_scatter_keys: wp.array[wp.int32] = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        # Per-element colour assignment (1-based; 0 = unassigned). This
        # is the primary output of the per-world JP kernel and also
        # doubles as the scratch the kernel clears at the start of each
        # step.
        self._per_world_assigned: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self.device)
        # ----- Greedy variant scratch (per-world) ----------------------
        #
        # Used only by ``_per_world_greedy_coloring_kernel`` (the
        # PHOENX_USE_GREEDY_COLORING=True path). Each row holds one
        # world's per-colour histogram bucket and live scatter cursor.
        # ``_per_world_greedy_overflow`` is a single-element flag the
        # kernel sets if any world's coloring exceeds GREEDY_MAX_COLORS;
        # surfaced via :meth:`step_report` for debugging but not raised
        # mid-step (the greedy fallback is "use the round-based JP
        # path", which the user can pick by flipping the config flag).
        self._per_world_greedy_color_count: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS)), dtype=wp.int32, device=self.device
        )
        self._per_world_greedy_color_offsets: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS)), dtype=wp.int32, device=self.device
        )
        self._per_world_greedy_overflow: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self.device)

        # ----- Contact infrastructure -----
        # Contact state in two narrow containers (vs stuffing contacts
        # into the 154-dword joint container) -- saved ~1.3 GB at
        # h1_flat 4096 worlds.
        #
        #   :class:`ContactContainer`       -- keyed by contact index
        #     ``k`` into the rigid_contact_max buffer. Warm-start
        #     lambdas, prev-step lambdas, per-substep scratch.
        #   :class:`ContactColumnContainer` -- keyed by local column
        #     cid ``[0, max_contact_columns)``. 7-dword column header
        #     (type, body1/2, friction, friction_dynamic,
        #     contact_first, contact_count).
        if self.max_contact_columns > 0:
            self._contact_container: ContactContainer = contact_container_zeros(
                self.rigid_contact_max, device=self.device
            )
            self._contact_cols: ContactColumnContainer = contact_column_container_zeros(
                self.max_contact_columns, device=self.device
            )
            self._enable_body_pair_grouping: bool = bool(enable_body_pair_grouping)
            self._ingest_scratch: IngestScratch | None = IngestScratch(
                rigid_contact_max=self.rigid_contact_max,
                max_contact_columns=self.max_contact_columns,
                device=self.device,
                enable_body_pair_grouping=self._enable_body_pair_grouping,
            )
            # Forward map stamps cid (no slot); prev-frame state is
            # keyed by the contact's sorted-buffer index ``k``.
            self._cid_of_contact_cur = wp.full(self.rigid_contact_max, -1, dtype=wp.int32, device=self.device)
            self._cid_of_contact_prev = wp.full(self.rigid_contact_max, -1, dtype=wp.int32, device=self.device)
        else:
            self._contact_container = contact_container_zeros(1, device=self.device)
            self._contact_cols = contact_column_container_zeros(1, device=self.device)
            self._ingest_scratch = None
            self._cid_of_contact_cur = None
            self._cid_of_contact_prev = None
            self._enable_body_pair_grouping = False

        self._contact_views: ContactViews | None = None
        self._contact_views_placeholder: ContactViews = self._make_placeholder_contact_views()

        # ----- Pairwise contact filter (packed int64 keys) -----
        self._collision_filter_keys: wp.array[wp.int64]
        self._set_collision_filter_pairs_impl(collision_filter_pairs or ())

        # ----- Optional material table -----
        self._shape_material: wp.array[wp.int32] | None = None
        self._materials: wp.array[MaterialData] | None = None
        # Optional internally-stored shape_body. Set by
        # :class:`WorldBuilder` via :meth:`set_shape_body` when shapes
        # were declared through the builder API. When set, ``step()``
        # reads from here if the caller doesn't pass ``shape_body``
        # explicitly (Newton-Model callers still pass it through).
        self._shape_body_internal: wp.array[wp.int32] | None = None
        # Lazy zero-length sentinel substituted for optional
        # per-contact stiffness / damping / friction arrays when the
        # user's :class:`~newton.Contacts` didn't allocate them.
        # Allocated on first step that needs it.
        self._soft_contact_sentinel: wp.array[wp.float32] | None = None

        # Validate that the caller-supplied containers match the
        # solver's per-step sizing. Bypass-the-factory mistakes
        # (allocating ``ConstraintContainer`` with the wrong shape,
        # passing a body container of the wrong length) get caught
        # here at construction instead of producing silent
        # out-of-range reads in the kernels.
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        """Validate per-step buffer dimensions against the documented
        schema. Runs once at the end of ``__init__``; cost is a few
        Python ``assert`` checks on ``.shape`` tuples (no GPU work).

        Raises ``AssertionError`` with a descriptive message on
        mismatch; the message names which container, what shape it
        actually has, and what shape the kernels expected. Use
        :meth:`make_constraint_container` to build the constraint
        container -- the factory always emits the correct shape.
        """
        expected_constraint_dwords = self.required_constraint_dwords(self.num_joints, self.num_cloth_triangles)
        # The joint-side ConstraintContainer holds rigid joints AND
        # cloth triangles (mixed at any cid; per-cid type tag at
        # dword 0 routes the dispatcher), so its column capacity is
        # ``num_joints + num_cloth_triangles``.
        expected_constraint_cols = max(1, int(self._joint_container_cids))
        actual_constraint_shape = self.constraints.data.shape
        assert actual_constraint_shape == (expected_constraint_dwords, expected_constraint_cols), (
            f"ConstraintContainer.data has shape {actual_constraint_shape}, expected "
            f"({expected_constraint_dwords}, {expected_constraint_cols}); use "
            f"PhoenXWorld.make_constraint_container() to build it"
        )

        expected_col_cols = max(1, int(self.max_contact_columns))
        actual_col_shape = self._contact_cols.data.shape
        assert actual_col_shape == (CONTACT_DWORDS, expected_col_cols), (
            f"ContactColumnContainer.data has shape {actual_col_shape}, "
            f"expected ({CONTACT_DWORDS}, {expected_col_cols})"
        )

        # Contact container is sized to ``rigid_contact_max`` (or 1
        # when contacts are disabled).
        expected_cc_cols = max(1, int(self.rigid_contact_max))
        for name, expected_rows in (
            ("lambdas", CC_DWORDS_PER_CONTACT),
            ("prev_lambdas", CC_DWORDS_PER_CONTACT),
            ("derived", CC_DERIVED_DWORDS_PER_CONTACT),
        ):
            actual = getattr(self._contact_container, name).shape
            assert actual == (expected_rows, expected_cc_cols), (
                f"ContactContainer.{name} has shape {actual}, expected ({expected_rows}, {expected_cc_cols})"
            )

        gravity_n = int(self.gravity.shape[0])
        assert gravity_n == self.num_worlds, (
            f"gravity array has length {gravity_n}, expected num_worlds={self.num_worlds}"
        )

    # ------------------------------------------------------------------
    # Material system / collision filters / placeholder contact views
    # ------------------------------------------------------------------

    def set_materials(
        self,
        materials: wp.array | None,
        shape_material: wp.array | None,
    ) -> None:
        """Install per-shape friction materials. ``materials`` is a
        ``wp.array[MaterialData]``; ``shape_material`` is an
        ``wp.array[int32]`` mapping shape index -> material index."""
        self._materials = materials
        self._shape_material = shape_material

    def set_shape_body(self, shape_body: wp.array | None) -> None:
        """Install the shape -> body map used by contact ingest.

        Populated by :class:`WorldBuilder` after shapes are declared;
        once set, :meth:`step` doesn't need ``shape_body=...`` from
        the caller. Pass ``None`` to clear (e.g. when switching to a
        Newton-Model-driven flow that supplies the array per step).
        """
        self._shape_body_internal = shape_body

    # ------------------------------------------------------------------
    # Kinematic pose scripting
    # ------------------------------------------------------------------

    def set_kinematic_pose(
        self,
        body: int,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """Script a kinematic body's end-of-next-step pose.

        The next :meth:`step` snapshots the current pose as the lerp /
        slerp origin, infers the linear and angular velocity to land
        on the target by end-of-step, and lerps / slerps across
        substeps so contacts see smooth motion.

        Args:
            body: Kinematic body index (dynamic/static raises
                ``ValueError``).
            position: Target origin, world frame [m].
            orientation: Target quaternion ``(x, y, z, w)``.

        Use :meth:`set_kinematic_poses_batch` for many bodies to
        avoid per-call H2D round-trips.
        """
        # Host-side motion-type validation. Fetching one int per call
        # is not free, but it trades a GPU sync for a clear error when
        # users misaddress bodies, which matches the "hard to screw up"
        # ethos of the WorldBuilder API.
        if body < 0 or body >= self.num_bodies:
            raise IndexError(f"set_kinematic_pose: body index {body} out of range [0, {self.num_bodies})")
        mt = int(self.bodies.motion_type.numpy()[body])
        if mt != int(MOTION_KINEMATIC):
            raise ValueError(
                f"set_kinematic_pose(body={body}): body motion_type is "
                f"{mt} (not MOTION_KINEMATIC={int(MOTION_KINEMATIC)}). "
                "Only kinematic bodies can be pose-scripted."
            )
        body_arr = wp.array([int(body)], dtype=wp.int32, device=self.device)
        pos_arr = wp.array([tuple(float(c) for c in position)], dtype=wp.vec3f, device=self.device)
        orient_arr = wp.array(
            [tuple(float(c) for c in orientation)],
            dtype=wp.quatf,
            device=self.device,
        )
        wp.launch(
            _set_kinematic_pose_batch_kernel,
            dim=1,
            inputs=[self.bodies, body_arr, pos_arr, orient_arr],
            device=self.device,
        )

    def set_kinematic_poses_batch(
        self,
        body_ids: wp.array,
        positions: wp.array,
        orientations: wp.array,
    ) -> None:
        """Batched variant of :meth:`set_kinematic_pose`.

        Args:
            body_ids: ``wp.array[int32]`` of kinematic body indices.
            positions: ``wp.array[wp.vec3f]``, parallel to ``body_ids``.
            orientations: ``wp.array[wp.quatf]``, parallel to ``body_ids``.

        Non-kinematic entries in ``body_ids`` are silently ignored by
        the kernel -- the caller is expected to pre-filter on the
        host if strict validation matters. This matches the Newton
        adapter, which pushes *all* body indices through the same
        kernel and lets the motion-type check gate.
        """
        n = int(body_ids.shape[0])
        if n == 0:
            return
        if positions.shape[0] != n or orientations.shape[0] != n:
            raise ValueError(
                "set_kinematic_poses_batch: body_ids, positions, orientations "
                f"must share length (got {n}, {positions.shape[0]}, "
                f"{orientations.shape[0]})"
            )
        wp.launch(
            _set_kinematic_pose_batch_kernel,
            dim=n,
            inputs=[self.bodies, body_ids, positions, orientations],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Joint initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def required_constraint_dwords(num_joints: int, num_cloth_triangles: int = 0) -> int:
        """Dword width of the joint-side :class:`ConstraintContainer`.

        Contacts now live in a dedicated narrow
        :class:`ContactColumnContainer`; the
        :class:`ConstraintContainer` is sized for the widest of the
        joint-side constraint types actually present:

        * Rigid joints contribute ``ADBS_DWORDS`` (154).
        * Cloth triangles contribute ``CLOTH_TRIANGLE_DWORDS`` (~18).
        * Otherwise, the minimal ``CONTACT_DWORDS = 7`` placeholder
          satisfies the shared header contract on the unused
          single-row allocation.

        Mixed scenes get the max of all active types; that is wider
        than strictly necessary for either type alone but the joint
        container is column-sparse (one cid per joint / cloth row),
        so the extra unused dwords are negligible.
        """
        widest = int(CONTACT_DWORDS)
        if int(num_joints) > 0:
            widest = max(widest, int(ADBS_DWORDS))
        if int(num_cloth_triangles) > 0:
            widest = max(widest, int(CLOTH_TRIANGLE_DWORDS))
        return widest

    @staticmethod
    def make_constraint_container(
        num_joints: int,
        num_cloth_triangles: int = 0,
        device: wp.context.Devicelike = None,
    ) -> ConstraintContainer:
        """Factory for a correctly-sized joint-side
        :class:`ConstraintContainer`.

        Capacity is ``max(1, num_joints + num_cloth_triangles)`` -- the
        joint-side container holds rigid joints AND cloth triangles
        (mixed at any cid; per-cid ``constraint_type`` tag at dword 0
        routes the dispatcher). Contact columns live in their own
        :class:`ContactColumnContainer` and are sized separately. The
        dword width is the ADBS joint header (154 dwords) when joints
        exist; otherwise the contact placeholder width (7 dwords). See
        :meth:`required_constraint_dwords`.
        """
        cap = max(1, int(num_joints) + int(num_cloth_triangles))
        return constraint_container_zeros(
            num_constraints=cap,
            num_dwords=PhoenXWorld.required_constraint_dwords(num_joints, num_cloth_triangles),
            device=device,
        )

    def initialize_actuated_double_ball_socket_joints(
        self,
        body1: wp.array,
        body2: wp.array,
        anchor1: wp.array,
        anchor2: wp.array,
        hertz: wp.array,
        damping_ratio: wp.array,
        joint_mode: wp.array,
        drive_mode: wp.array,
        target: wp.array,
        target_velocity: wp.array,
        max_force_drive: wp.array,
        stiffness_drive: wp.array,
        damping_drive: wp.array,
        min_value: wp.array,
        max_value: wp.array,
        hertz_limit: wp.array,
        damping_ratio_limit: wp.array,
        stiffness_limit: wp.array,
        damping_limit: wp.array,
        armature: wp.array | None = None,
    ) -> None:
        """Pack ``num_joints`` actuated-DBS joint columns.

        One-shot launch over cids ``[0, num_joints)``. All input
        arrays must be length ``num_joints``, live on ``device``, and
        match the kernel's dtypes (see the kernel signature in
        :mod:`constraint_actuated_double_ball_socket`). Call once,
        after :meth:`__init__`, before the first :meth:`step`.

        Args:
            body1, body2: ``wp.array[int32]`` per-joint body indices.
            anchor1, anchor2: ``wp.array[vec3f]`` world-space anchors
                at init. Line from ``anchor1`` to ``anchor2`` defines
                the joint axis for revolute / prismatic modes.
            hertz, damping_ratio: Positional-block soft-constraint
                frequency [Hz] and damping ratio.
            joint_mode: Per-joint
                :class:`JointMode` enum (``wp.array[int32]``).
            drive_mode: Per-joint
                :class:`DriveMode` enum.
            target, target_velocity: PD setpoints [rad / rad/s for
                revolute, m / m/s for prismatic].
            max_force_drive, stiffness_drive, damping_drive: Drive
                impulse cap and PD gains. ``stiffness_drive ==
                damping_drive == 0`` disables the drive row.
            min_value, max_value: Limit window (``min > max``
                disables).
            hertz_limit, damping_ratio_limit: Soft-constraint limit
                knobs (used when both PD gains are zero).
            stiffness_limit, damping_limit: Limit PD gains (SI). Any
                strictly positive value selects the PD formulation.
            armature: Per-joint axial armature [kg*m^2 for revolute,
                kg for prismatic]. ``None`` (default) zero-fills, which
                disables armature on every joint -- callers that rely
                on PhoenX's pre-armature behaviour are unaffected.
                Folded into the axial drive / limit effective inertia
                only; rigid 5-row positional locks are unchanged.
        """
        if self.num_joints <= 0:
            return
        if armature is None:
            armature = wp.zeros(self.num_joints, dtype=wp.float32, device=self.device)
        # Detect whether the single-world solve can use the revolute-
        # only iterate kernels. ``joint_mode`` is a host-readable
        # ``wp.array[int32]`` of length ``num_joints``; one D2H copy
        # at init time is acceptable (this method runs once before
        # any graph capture).
        try:
            mode_np = joint_mode.numpy()
        except Exception:
            mode_np = None
        if mode_np is not None and mode_np.size > 0:
            self._use_revolute_specialization = bool((mode_np == int(JOINT_MODE_REVOLUTE)).all())
        wp.launch(
            actuated_double_ball_socket_initialize_kernel,
            dim=self.num_joints,
            inputs=[
                self.constraints,
                self.bodies,
                wp.int32(0),  # cid_offset -- joints start at cid 0
                body1,
                body2,
                anchor1,
                anchor2,
                hertz,
                damping_ratio,
                joint_mode,
                drive_mode,
                target,
                target_velocity,
                max_force_drive,
                stiffness_drive,
                damping_drive,
                min_value,
                max_value,
                hertz_limit,
                damping_ratio_limit,
                stiffness_limit,
                damping_limit,
                armature,
            ],
            device=self.device,
        )

    def populate_cloth_triangles_from_model(self, model: Any) -> None:
        """Stamp ``num_cloth_triangles`` cloth-triangle constraint rows
        from a Newton :class:`~newton.Model` and copy the model's
        particle state into the :class:`ParticleContainer`.

        Reads from the same VBD-compatible mesh API
        (``model.particle_q`` / ``particle_qd`` / ``particle_inv_mass``
        / ``tri_indices`` / ``tri_poses`` / ``tri_areas`` /
        ``tri_materials``) Newton's other cloth solvers consume, then
        packs into PhoenX's internal constraint-row format. Newton's
        ``tri_poses[t]`` is already a 2x2 inverse rest-pose matrix
        (``inv_dm`` in style3d / VBD), so it copies verbatim into the
        cloth row.

        Cloth cids occupy ``[num_joints, num_joints + num_cloth_triangles)``
        in the joint-side :class:`ConstraintContainer`. Contacts (when
        present) follow at ``cid >= num_joints + num_cloth_triangles``;
        the dispatcher's ``cid < num_joints_kernel_param`` branch (the
        kernel parameter is set to :attr:`_joint_container_cids` =
        ``num_joints + num_cloth_triangles``) sends both joints and
        cloth to the joint-side dispatch, where the per-cid type tag
        routes between ADBS and cloth.

        Args:
            model: A finalised :class:`~newton.Model` whose particle /
                triangle counts match :attr:`num_particles` /
                :attr:`num_cloth_triangles`. Cloth nodes must live in
                the model's particle store -- rigid-body cloth nodes
                are deferred (PLAN_CLOTH_TRIANGLE.md).

        Raises:
            ValueError: if the model's particle count differs from
                :attr:`num_particles`, or its triangle count differs
                from :attr:`num_cloth_triangles`.
        """
        if self.num_cloth_triangles == 0:
            # Nothing to do; rigid-only scenes don't reserve any cloth
            # rows. Still allowed (e.g. user calls this defensively
            # before checking ``model.tri_count``); silently no-op.
            return
        model_particle_count = int(model.particle_count)
        model_tri_count = int(model.tri_count)
        if model_particle_count != self.num_particles:
            raise ValueError(
                f"populate_cloth_triangles_from_model: model.particle_count "
                f"({model_particle_count}) != world.num_particles ({self.num_particles}). "
                "Construct PhoenXWorld with num_particles == model.particle_count."
            )
        if model_tri_count != self.num_cloth_triangles:
            raise ValueError(
                f"populate_cloth_triangles_from_model: model.tri_count "
                f"({model_tri_count}) != world.num_cloth_triangles ({self.num_cloth_triangles}). "
                "Construct PhoenXWorld with num_cloth_triangles == model.tri_count."
            )
        # Particle state: position / velocity / inverse_mass copy 1:1.
        if self.particles is None:  # pragma: no cover -- guarded above
            raise RuntimeError("self.particles is None despite num_particles > 0")
        wp.copy(self.particles.position, model.particle_q)
        wp.copy(self.particles.velocity, model.particle_qd)
        wp.copy(self.particles.inverse_mass, model.particle_inv_mass)
        # Stamp one cloth row per triangle. Cids start at
        # ``num_joints`` (immediately after the rigid joints), one
        # row per triangle.
        wp.launch(
            _phoenx_init_cloth_triangle_rows_kernel,
            dim=self.num_cloth_triangles,
            inputs=[
                self.constraints,
                wp.int32(self.num_joints),  # cid_offset
                wp.int32(self.num_bodies),
                model.tri_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
            ],
            device=self.device,
        )

    def set_collision_filter_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Replace the registered body-pair contact filter. Pairs are
        canonicalised ``(min, max)`` and deduped; self-pairs rejected;
        contact lookup is a device-side binary search."""
        self._set_collision_filter_pairs_impl(pairs)

    def _set_collision_filter_pairs_impl(self, pairs: Iterable[tuple[int, int]]) -> None:
        nb = int(self.num_bodies)
        packed: list[int] = []
        seen: set[tuple[int, int]] = set()
        for a, b in pairs:
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                raise ValueError(f"collision filter pair must have two distinct bodies (got both = {a_i})")
            if not (0 <= a_i < nb and 0 <= b_i < nb):
                raise IndexError(
                    f"collision filter pair ({a_i}, {b_i}) out of range [0, {nb}) for this World's body count"
                )
            lo = min(a_i, b_i)
            hi = max(a_i, b_i)
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            packed.append(lo * nb + hi)

        packed.sort()
        if not packed:
            arr = np.asarray([np.iinfo(np.int64).max], dtype=np.int64)
        else:
            arr = np.asarray(packed, dtype=np.int64)

        self._collision_filter_keys = wp.array(arr, dtype=wp.int64, device=self.device)
        self._collision_filter_count = int(len(packed))

    def _make_placeholder_contact_views(self) -> ContactViews:
        """Size-1 dummy :class:`ContactViews` for contact-free steps.
        Never actually read because the contact branch only fires for
        cids tagged :data:`CONSTRAINT_TYPE_CONTACT`."""
        dummy_int = wp.zeros(1, dtype=wp.int32, device=self.device)
        dummy_vec3 = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        dummy_float = wp.zeros(1, dtype=wp.float32, device=self.device)
        # Soft-contact stiffness / damping / friction arrays can be
        # sentinel length 0 when the caller's ``Contacts`` didn't
        # allocate them. The prepare kernel gates on the array
        # length per-contact.
        sentinel_float = wp.zeros(0, dtype=wp.float32, device=self.device)
        return contact_views_make(
            rigid_contact_count=dummy_int,
            rigid_contact_point0=dummy_vec3,
            rigid_contact_point1=dummy_vec3,
            rigid_contact_normal=dummy_vec3,
            rigid_contact_shape0=dummy_int,
            rigid_contact_shape1=dummy_int,
            rigid_contact_match_index=dummy_int,
            rigid_contact_margin0=dummy_float,
            rigid_contact_margin1=dummy_float,
            shape_body=dummy_int,
            rigid_contact_stiffness=sentinel_float,
            rigid_contact_damping=sentinel_float,
            rigid_contact_friction=sentinel_float,
        )

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        contacts=None,
        shape_body=None,
        picking=None,
        vel_accum: wp.array[wp.vec3f] | None = None,
        omega_accum: wp.array[wp.vec3f] | None = None,
    ) -> None:
        """Advance the world by ``dt`` seconds.

        Phases: ingest contacts -> rebuild elements + JP colouring ->
        substep loop (forces, gravity, main solve, integrate
        positions, position iterate, relax) -> damping + rotated
        inertia -> clear forces.

        Args:
            dt: Time step [s]. Non-positive values no-op.
            contacts: Newton :class:`Contacts` buffer. Must have been
                built with a non-disabled ``contact_matching`` mode
                (``"sticky"`` recommended for stable stacking).
                ``None`` runs a free-fall step with no contact
                constraints.
            shape_body: ``model.shape_body`` (shape id -> body id).
                Required when ``contacts`` is provided.
            picking: Optional :class:`Picking` instance. When
                provided, its ``apply_force()`` is called at the
                start of every substep so the pick spring stays stiff
                regardless of :attr:`substeps`. Callers driving
                picking themselves should leave this ``None``.
        """
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        if dt < 1e-7:
            return

        self.step_dt = dt
        self.inv_step_dt = 1.0 / dt
        self.substep_dt = dt / self.substeps

        self._ingest_and_warmstart_contacts(contacts, shape_body)
        if self._ingest_scratch is not None:
            self._partitioner.set_costs_from_contacts(
                # Contacts start at the cid past joints + cloth in
                # the unified joint-container cid space.
                self._joint_container_cids,
                self._ingest_scratch.num_contact_columns,
                self._contact_cols,
            )

        self._rebuild_elements()
        if self._constraint_capacity > 0:
            self._partitioner.reset(self._elements, self._num_active_constraints)
            if self.step_layout == "single_world":
                # Greedy gives 2-3x fewer colours on dense graphs but
                # is bounded at 64 colours (single int64 forbidden
                # mask). If the kernel raises its overflow flag on a
                # non-captured step we silently flip
                # ``_use_greedy_coloring`` and rebuild with the
                # round-based JP path (which has no such bound).
                if self._use_greedy_coloring:
                    # Greedy with in-graph JP fallback: the
                    # partitioner runs the round-based JP build
                    # inside a ``wp.capture_while`` keyed on the
                    # overflow flag, so a graph that overflows the
                    # 64-colour bitmask still gets a valid CSR within
                    # the same captured frame. No host probe needed.
                    self._partitioner.build_csr_greedy_with_jp_fallback()
                else:
                    self._partitioner.build_csr()
            else:
                # Multi-world: parallel per-world JP coloring. The
                # adjacency build is still global (per-body CSR of
                # element neighbours), but the MIS loop runs one
                # block per world and is the only multi_world
                # strategy now.
                self._build_per_world_coloring()

            # Per-step adaptive threads-per-world pick. Reads the freshly
            # built per-world colour stats and writes ``_tpw_choice[0]``;
            # the fast-tail kernels read it at every launch in the
            # substep loop below. Skipped when the user pinned a static
            # tpw at construction or when the single-world layout is in
            # use (its kernels don't read ``_tpw_choice``).
            if self._tpw_auto and self.step_layout != "single_world":
                self._pick_tpw()

        # Once-per-step kinematic prepare: snapshot ``position`` ->
        # ``position_prev``, resolve this step's pose target
        # (user-scripted or auto from constant velocity), and infer
        # the linear / angular velocity the solver exposes to contacts.
        # Dynamic and static bodies are a no-op.
        self._kinematic_prepare_step()

        # Substep loop ordering: solve with bias ON, integrate, position
        # iterate, then relax with bias OFF. Running relax before
        # integrate would throw away the positional bias's contribution
        # to penetration recovery. Kinematic bodies are slotted into
        # their lerp/slerp-interpolated pose at each substep's tail so
        # contacts prepared at the *next* substep see smooth motion.
        inv_n = 1.0 / float(self.substeps)
        for k in range(self.substeps):
            if picking is not None:
                picking.apply_force()
            self._integrate_forces_and_gravity()
            if self.step_layout == "single_world":
                self._solve_main_singleworld()
                self._integrate_positions()
                self._relax_velocities_singleworld()
            else:
                self._solve_main()
                self._integrate_positions()
                self._relax_velocities()
            # Inertia refresh: ``_integrate_positions`` rotated every
            # dynamic body via ``q_new = dq(omega * substep_dt) * q``;
            # the world-frame inverse inertia tensor stored in
            # ``bodies.inverse_inertia_world`` was rebuilt for the
            # PRIOR orientation, so any subsequent constraint solve in
            # this step (relax + the next substep) would project
            # impulses through a stale ``R * I^-1 * R^T``. For
            # anisotropic robot links this biases the angular impulse
            # direction over the substep loop. Refreshing here keeps
            # every solve consistent with the current pose.
            self._refresh_world_inertia()
            alpha = float(k + 1) * inv_n
            self._kinematic_interpolate_substep(alpha)
            self._apply_global_damping()
            if vel_accum is not None and omega_accum is not None and self.num_bodies > 0:
                # ``substep_average`` velocity readout: accumulate
                # post-substep velocities so the outer-step caller can
                # divide by ``step_dt`` and get the time-averaged value.
                wp.launch(
                    _accumulate_substep_velocity_kernel,
                    dim=int(self.num_bodies) - 1,  # slot 0 is the anchor
                    inputs=[
                        self.bodies.velocity,
                        self.bodies.angular_velocity,
                        wp.float32(self.substep_dt),
                    ],
                    outputs=[vel_accum, omega_accum],
                    device=self.device,
                )

        self._update_inertia_and_clear_forces()

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _ingest_and_warmstart_contacts(self, contacts, shape_body) -> None:
        """Translate Newton's ``Contacts`` buffer into contact columns.

        No contacts this step -> fall back to the joint-only active
        count. Otherwise swap prev/current per-cid state, run
        ingest -> warm-start -> forward-map stamp, then fuse joint +
        contact counts on-device.
        """
        if contacts is None or self.max_contact_columns == 0 or self._ingest_scratch is None:
            self._num_active_constraints.fill_(self._joint_container_cids)
            self._contact_views = None
            return

        if getattr(contacts, "contact_matching", False) is False:
            raise ValueError(
                "PhoenX solver requires the Contacts buffer to be built with "
                "a non-disabled contact_matching mode (needed for persistent "
                'warm-starting; use contact_matching="sticky" for stable '
                "stacking)."
            )
        if shape_body is None:
            shape_body = self._shape_body_internal
        if shape_body is None:
            raise ValueError(
                "step(dt, contacts=...) requires shape_body to resolve contact "
                "shape ids to rigid-body ids. Pass it explicitly (e.g. "
                "``model.shape_body``), or register shapes via "
                "``WorldBuilder.add_shape_*`` so it gets installed at finalize."
            )

        # Soft-contact per-contact arrays are optional on Newton's
        # ``Contacts``: only allocated when
        # ``per_contact_shape_properties=True``. When missing we pass
        # a length-0 sentinel so the Warp kernels' per-contact
        # ``array.shape[0] > k`` check short-circuits cleanly.
        if self._soft_contact_sentinel is None:
            self._soft_contact_sentinel = wp.zeros(0, dtype=wp.float32, device=self.device)
        contact_stiffness = (
            contacts.rigid_contact_stiffness
            if getattr(contacts, "rigid_contact_stiffness", None) is not None
            else self._soft_contact_sentinel
        )
        contact_damping = (
            contacts.rigid_contact_damping
            if getattr(contacts, "rigid_contact_damping", None) is not None
            else self._soft_contact_sentinel
        )
        contact_friction = (
            contacts.rigid_contact_friction
            if getattr(contacts, "rigid_contact_friction", None) is not None
            else self._soft_contact_sentinel
        )

        # Swap lambda buffers + forward map (pointer swap, O(1)). Also
        # swap the prev/curr inverse-permutation buffers when grouping
        # is on -- next frame's gather translates Newton-order match
        # indices through ``prev_inv_sort_perm`` (= this frame's
        # ``inv_sort_perm`` after the swap).
        contact_container_swap_prev_current(self._contact_container)
        self._cid_of_contact_cur, self._cid_of_contact_prev = (
            self._cid_of_contact_prev,
            self._cid_of_contact_cur,
        )
        if self._enable_body_pair_grouping and self._ingest_scratch.inv_sort_perm is not None:
            self._ingest_scratch.inv_sort_perm, self._ingest_scratch.prev_inv_sort_perm = (
                self._ingest_scratch.prev_inv_sort_perm,
                self._ingest_scratch.inv_sort_perm,
            )

        ingest_contacts(
            contacts=contacts,
            shape_body=shape_body,
            contact_cols=self._contact_cols,
            scratch=self._ingest_scratch,
            max_contact_columns=self.max_contact_columns,
            default_friction=self.default_friction,
            device=self.device,
            num_bodies=self.num_bodies,
            num_rigid_shapes=int(shape_body.shape[0]),
            filter_keys=self._collision_filter_keys,
            filter_count=self._collision_filter_count,
            shape_material=self._shape_material,
            materials=self._materials,
            enable_body_pair_grouping=self._enable_body_pair_grouping,
        )

        # Build ContactViews. In compound-grouping mode the views point
        # at PhoenX's sorted scratch (per-contact data is in sorted-k
        # order, match index is already prev-frame sorted-k); otherwise
        # they point at Newton's narrow-phase arrays directly.
        if self._enable_body_pair_grouping:
            self._contact_views = contact_views_make(
                rigid_contact_count=contacts.rigid_contact_count,
                rigid_contact_point0=self._ingest_scratch.sorted_point0,
                rigid_contact_point1=self._ingest_scratch.sorted_point1,
                rigid_contact_normal=self._ingest_scratch.sorted_normal,
                rigid_contact_shape0=self._ingest_scratch.sorted_shape0,
                rigid_contact_shape1=self._ingest_scratch.sorted_shape1,
                rigid_contact_match_index=self._ingest_scratch.sorted_match_index,
                rigid_contact_margin0=self._ingest_scratch.sorted_margin0,
                rigid_contact_margin1=self._ingest_scratch.sorted_margin1,
                shape_body=shape_body,
                rigid_contact_stiffness=self._ingest_scratch.sorted_stiffness,
                rigid_contact_damping=self._ingest_scratch.sorted_damping,
                rigid_contact_friction=self._ingest_scratch.sorted_friction,
            )
        else:
            self._contact_views = contact_views_make(
                rigid_contact_count=contacts.rigid_contact_count,
                rigid_contact_point0=contacts.rigid_contact_point0,
                rigid_contact_point1=contacts.rigid_contact_point1,
                rigid_contact_normal=contacts.rigid_contact_normal,
                rigid_contact_shape0=contacts.rigid_contact_shape0,
                rigid_contact_shape1=contacts.rigid_contact_shape1,
                rigid_contact_match_index=contacts.rigid_contact_match_index,
                rigid_contact_margin0=contacts.rigid_contact_margin0,
                rigid_contact_margin1=contacts.rigid_contact_margin1,
                shape_body=shape_body,
                rigid_contact_stiffness=contact_stiffness,
                rigid_contact_damping=contact_damping,
                rigid_contact_friction=contact_friction,
            )

        # Warm-start gather: in compound mode the match index is
        # already in prev-frame sorted-k space (translated during
        # ingest), so we pass the views' field directly. In
        # non-compound mode it is Newton-order match_index, which is
        # the same convention ``cc.prev_*`` was keyed by last frame
        # (no resort), so the gather unwinds correctly.
        gather_match_index = (
            self._ingest_scratch.sorted_match_index
            if self._enable_body_pair_grouping
            else contacts.rigid_contact_match_index
        )
        gather_contact_warmstart(
            scratch=self._ingest_scratch,
            rigid_contact_match_index=gather_match_index,
            prev_cid_of_contact=self._cid_of_contact_prev,
            bodies=self.bodies,
            contacts=self._contact_views,
            cc=self._contact_container,
            particles=self.body_or_particle.particles,
            tri_indices=self.tri_indices,
            num_rigid_shapes=int(shape_body.shape[0]),
            device=self.device,
        )

        stamp_forward_contact_map(
            rigid_contact_max=self.rigid_contact_max,
            # Contacts begin in the cid space *after* both rigid joints
            # and cloth triangles -- they live in the joint container.
            cid_base=self._joint_container_cids,
            scratch=self._ingest_scratch,
            cid_of_contact=self._cid_of_contact_cur,
            device=self.device,
        )

        self._sync_num_active_constraints()

    def _sync_num_active_constraints(self) -> None:
        """Fuse ``num_joints + num_contact_columns`` into
        ``_num_active_constraints`` on-device (graph-capture safe)."""
        wp.launch(
            _sync_num_active_constraints_kernel,
            dim=1,
            inputs=[
                self._ingest_scratch.num_contact_columns,
                # Joint-container cids = rigid joints + cloth triangles;
                # both are always active across substeps. The active
                # contact count is added on top by the kernel.
                wp.int32(self._joint_container_cids),
            ],
            outputs=[self._num_active_constraints],
            device=self.device,
        )

    def _rebuild_elements(self) -> None:
        """Project every active constraint into the partitioner's
        element view. Type-agnostic launch at
        ``dim = constraint_capacity``; threads beyond the device-held
        ``num_active_constraints`` early-out."""
        if self._constraint_capacity == 0:
            return
        elements_kernel = (
            _constraints_to_elements_cloth_kernel if self.num_cloth_triangles > 0 else _constraints_to_elements_kernel
        )
        wp.launch(
            elements_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._num_active_constraints,
                # Joint-container boundary; ``cid < num_joints`` in the
                # kernel routes joints + cloth to the joint dispatcher,
                # else to contacts.
                wp.int32(self._joint_container_cids),
                self._elements,
                self.body_or_particle,
            ],
            device=self.device,
        )

    def _maybe_fallback_from_per_world_greedy_overflow(self, nw: int) -> None:
        """Multi-world analogue of
        :meth:`_maybe_fallback_from_greedy_overflow`. Flips the
        instance flag and re-runs the per-world JP coloring if any
        world's greedy build hit the 64-colour cap.
        """
        flag = self._per_world_greedy_overflow
        device = flag.device
        if device.is_cuda and device.is_capturing:
            return
        if int(flag.numpy()[0]) == 0:
            return
        self._use_greedy_coloring = False
        wp.launch_tiled(
            _per_world_jp_coloring_kernel,
            dim=[nw],
            inputs=[
                self._per_world_element_offsets,
                self._per_world_element_count,
                self._per_world_elements,
                self._elements,
                self._partitioner._adjacency_section_end_indices,
                self._partitioner._vertex_to_adjacent_elements,
                self._partitioner._random_values,
                self._partitioner._cost_values,
                int(MAX_COLORS),
            ],
            outputs=[
                self._per_world_assigned,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_num_colors,
            ],
            block_dim=_PER_WORLD_COLORING_BLOCK_DIM,
            device=self.device,
        )

    def _build_per_world_coloring(self) -> None:
        """Parallel per-world Jones-Plassmann coloring.

        1. ``_count_elements_per_world_kernel`` -- atomic count of
           active cids per world (commutative atomics, deterministic).
        2. Inclusive scan -> ``per_world_element_offsets``.
        3. ``_build_scatter_keys_kernel`` + stable
           ``radix_sort_pairs`` -- bucket cids by world id; stable
           sort keeps order deterministic.
        4. ``_per_world_jp_coloring_kernel`` -- one block per world
           runs the full JP MIS loop on its bucket and emits into
           ``world_element_ids_by_color`` / ``world_color_starts`` /
           ``world_num_colors``. Intra-colour slots use
           ``wp.tile_scan_exclusive`` (deterministic).

        Reuses the adjacency CSR from ``partitioner.reset`` (worlds
        are disjoint after static-null-out, so all neighbours live in
        the same world).
        """
        nw = self.num_worlds
        cap = self._constraint_capacity

        # Phase 1: count elements per world (and seed the shifted-offset
        # buffer so the inclusive scan below produces exclusive offsets
        # + trailing total).
        self._per_world_element_count.zero_()
        self._world_totals_shifted.zero_()
        wp.launch(
            _count_elements_per_world_kernel,
            dim=cap,
            inputs=[self._elements, self._num_active_constraints, self.bodies],
            outputs=[self._per_world_element_count, self._world_totals_shifted],
            device=self.device,
        )

        # Phase 2: inclusive scan of shifted counts -> per_world_element_offsets[0]=0,
        # [w+1] = sum(counts[0..w]). Deterministic.
        wp.utils.array_scan(self._world_totals_shifted, self._per_world_element_offsets, inclusive=True)

        # Phase 3: deterministic scatter via stable sort by world id.
        # ``_build_scatter_keys_kernel`` populates per-cid (key=world_id,
        # value=cid) pairs (with INT32_MAX keys masking inactive
        # cids); ``sort_variable_length_int`` runs a stable
        # ``radix_sort_pairs`` so that the value array becomes the
        # per-world bucketed cid stream, ordered by cid within each
        # world. This replaces an atomic-cursor scatter whose order
        # depended on GPU thread scheduling -- the sort is a function
        # of (world_id, cid) and therefore bit-deterministic across
        # runs.
        wp.launch(
            _build_scatter_keys_kernel,
            dim=cap,
            inputs=[self._elements, self._num_active_constraints, self.bodies, wp.int32(cap)],
            outputs=[self._per_world_scatter_keys, self._per_world_elements],
            device=self.device,
        )
        sort_variable_length_int(
            self._per_world_scatter_keys,
            self._per_world_elements,
            self._num_active_constraints,
        )

        # Phase 4: per-world coloring. One block per world.
        self._per_world_assigned.zero_()
        # The output CSR base offsets (world_csr_offsets) match the
        # per-world element offsets (every active element belongs to
        # some colour in some world, so offsets[w] identifies both the
        # bucket start and the start of world w's colour sequence).
        wp.copy(self._world_csr_offsets, self._per_world_element_offsets)
        # ``launch_tiled`` returns (block_idx, lane) from ``wp.tid()`` --
        # exactly the (world, lane) pair the kernel expects. Plain
        # ``launch`` would return a global thread index instead, which
        # would silently mis-address every block-scoped read/write.
        if self._use_greedy_coloring:
            # Greedy variant: each MIS commit picks the smallest free
            # colour. Drops per-world colour counts toward the
            # max-body-degree lower bound (mirrors the single-world
            # path in IncrementalContactPartitioner.build_csr_greedy).
            wp.launch_tiled(
                _per_world_greedy_coloring_kernel,
                dim=[nw],
                inputs=[
                    self._per_world_element_offsets,
                    self._per_world_element_count,
                    self._per_world_elements,
                    self._elements,
                    self._partitioner._adjacency_section_end_indices,
                    self._partitioner._vertex_to_adjacent_elements,
                    self._partitioner._random_values,
                    self._partitioner._cost_values,
                    int(GREEDY_MAX_COLORS),
                ],
                outputs=[
                    self._per_world_assigned,
                    self._per_world_greedy_color_count,
                    self._per_world_greedy_color_offsets,
                    self._world_element_ids_by_color,
                    self._world_color_starts,
                    self._world_num_colors,
                    self._per_world_greedy_overflow,
                ],
                block_dim=_PER_WORLD_COLORING_BLOCK_DIM,
                device=self.device,
            )
            # Same overflow-fallback contract as the single-world
            # path: if the per-world greedy kernel exhausted the
            # 64-color bitmask in any world, flip the live mode to
            # JP and rerun *this* step's per-world coloring with it.
            self._maybe_fallback_from_per_world_greedy_overflow(nw)
        else:
            # Round-equals-colour Jones-Plassmann (legacy). Cost-biased
            # priorities: contacts use their per-column contact count
            # as the high priority word, while joints stay at cost 0.
            wp.launch_tiled(
                _per_world_jp_coloring_kernel,
                dim=[nw],
                inputs=[
                    self._per_world_element_offsets,
                    self._per_world_element_count,
                    self._per_world_elements,
                    self._elements,
                    self._partitioner._adjacency_section_end_indices,
                    self._partitioner._vertex_to_adjacent_elements,
                    self._partitioner._random_values,
                    self._partitioner._cost_values,
                    int(MAX_COLORS),
                ],
                outputs=[
                    self._per_world_assigned,
                    self._world_element_ids_by_color,
                    self._world_color_starts,
                    self._world_num_colors,
                ],
                block_dim=_PER_WORLD_COLORING_BLOCK_DIM,
                device=self.device,
            )

    def _integrate_forces_and_gravity(self) -> None:
        """One per-substep launch over the unified body-or-particle
        index space. The kernel internally dispatches on
        :func:`is_particle`:

        * Bodies: apply per-body force / torque accumulators and
          gravity to ``velocity`` / ``angular_velocity``. Static and
          kinematic slots early-return.
        * Particles: substep-entry access-mode transition
          (Velocity-level -> Position-level). Apply gravity +
          external force, snapshot pre-predict position into
          ``position_substep_start``, and advance ``position`` by
          ``velocity * dt`` so the cloth iterate sees the predicted
          pose. The substep-exit recovery happens inside
          :meth:`_integrate_positions`, which also runs over the
          unified index space.
        """
        n = self.num_bodies + self.num_particles
        if n == 0:
            return
        wp.launch(
            _phoenx_apply_forces_and_gravity_kernel,
            dim=n,
            inputs=[self.body_or_particle, self.gravity, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _solve_main(self) -> None:
        """Main PGS solve per substep: prepare + ``solver_iterations``
        iterate sweeps with bias ON.

        Launches the fused prepare+iterate kernel so the per-world
        setup (``world_id``, ``n_colors``, ``world_base``) is computed
        once and the one kernel-launch boundary between prepare and
        iterate is removed.
        """
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        kernel = (
            _constraint_prepare_plus_iterate_fast_tail_revolute_kernel
            if self._use_revolute_specialization
            else _constraint_prepare_plus_iterate_fast_tail_kernel
        )
        wp.launch(
            kernel,
            dim=self._fast_tail_launch_dim(),
            block_dim=self._fast_tail_block_dim(),
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                idt,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(self.solver_iterations),
                wp.int32(self.num_worlds),
                # Joint-container boundary (= rigid joints + cloth);
                # the multi-world dispatcher uses it the same way the
                # single-world ones do.
                wp.int32(self._joint_container_cids),
                self._tpw_choice,
                self.body_or_particle.particles,
                self.tri_indices,
            ],
            device=self.device,
        )

    def _relax_velocities(self) -> None:
        """TGS-soft relax sweeps with bias OFF. Removes the drift
        velocity the positional bias injected during the main solve."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        kernel = (
            _constraint_relax_fast_tail_revolute_kernel
            if self._use_revolute_specialization
            else _constraint_relax_fast_tail_kernel
        )
        self._launch_fast_iter(
            kernel,
            self.velocity_iterations,
            idt,
            contact_views,
        )

    # ------------------------------------------------------------------
    # Single-world dispatch (capture-while over the global colour CSR)
    # ------------------------------------------------------------------

    def _capture_singleworld_sweep(self, kernel, **kw) -> None:
        """``wp.capture_while`` body: sweep up to
        :data:`NUM_INNER_WHILE_ITERATIONS` colours on the
        persistent-grid ("head") path in one outer step.

        Uses a persistent fixed-size grid (``_singleworld_total_threads``)
        with an internal grid-stride loop over the colour's active cid
        range -- same strategy as
        :class:`newton._src.geometry.narrow_phase.NarrowPhase`. Thread
        0 decrements ``color_cursor`` and, on either convergence
        (``color_cursor == 0``) or tail-fuse hand-off
        (``count <= fuse_threshold``), clears ``head_active[0]`` to
        terminate the head capture-while.

        The body is unrolled ``NUM_INNER_WHILE_ITERATIONS`` times on
        the host to amortise the per-outer-iteration capture-while
        overhead. Each launch re-enters one of the kernel's early-exit
        branches once ``head_active`` has been cleared within the same
        outer iteration, so the tail launches are cheap no-ops.
        """
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        idt = kw.get("idt", wp.float32(0.0))
        store = self.body_or_particle
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            wp.launch(
                kernel,
                dim=self._singleworld_total_threads,
                inputs=[
                    self.constraints,
                    self._contact_cols,
                    self.bodies,
                    idt,
                    self._partitioner.element_ids_by_color,
                    self._partitioner.color_starts,
                    self._partitioner.num_colors,
                    self._partitioner.color_cursor,
                    self._contact_container,
                    contact_views,
                    wp.int32(self._joint_container_cids),
                    wp.int32(self._singleworld_total_threads),
                    wp.int32(self._fuse_threshold),
                    self._head_active,
                    store,
                    self.tri_indices,
                ],
                block_dim=_SINGLEWORLD_BLOCK_DIM,
                device=self.device,
            )

    def _capture_singleworld_tail_sweep(self, kernel, **kw) -> None:
        """``wp.capture_while`` body: drain the remaining small colours
        via the fused single-block kernel.

        Launched as ``wp.launch_tiled(dim=[1], block_dim=FUSE_TAIL_BLOCK_DIM)``
        so the whole sweep runs in one block and ``_sync_threads``
        (``__syncthreads``) can order body-velocity writes between
        consecutive colours. The kernel internally walks colours until
        either ``color_cursor`` hits 0 or a colour exceeds
        ``fuse_threshold`` (hand-off back to the head path); it always
        publishes the final cursor value, so the outer capture-while
        terminates naturally on the cursor == 0 predicate.
        """
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        idt = kw.get("idt", wp.float32(0.0))
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                idt,
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.num_colors,
                self._partitioner.color_cursor,
                self._contact_container,
                contact_views,
                wp.int32(self._joint_container_cids),
                wp.int32(self._fuse_threshold),
                self.body_or_particle,
                self.tri_indices,
            ],
            block_dim=self._fuse_tail_block_dim,
            device=self.device,
        )

    def _singleworld_head_plus_tail_sweep(self, head_kernel, tail_kernel, idt: wp.float32) -> None:
        """Drive a single PGS sweep as "persistent-grid head + fused
        single-block tail".

        The head ``capture_while`` is gated by :attr:`_head_active` so
        the head kernel can terminate by either (a) draining the
        cursor or (b) encountering a small colour that should be
        handled by the fused tail kernel. The tail ``capture_while``
        is gated by the colour cursor and runs the fused kernel which
        drains every remaining small colour in a single block using
        ``_sync_threads`` between colours.

        When :attr:`_fuse_threshold` is 0, the head kernel never takes
        the hand-off branch; it drains the cursor itself and the tail
        capture-while's predicate (``color_cursor``) is already 0, so
        the tail kernel is never launched -- behaviour identical to
        the pre-fuse path.
        """
        # Reset head_active = 1 so the head capture-while gets at
        # least one launch in which to decide (converge, hand off, or
        # do real work).
        wp.launch(_reset_head_active_kernel, dim=1, inputs=[self._head_active], device=self.device)

        wp.capture_while(
            self._head_active,
            self._capture_singleworld_sweep,
            kernel=head_kernel,
            idt=idt,
        )
        wp.capture_while(
            self._partitioner.color_cursor,
            self._capture_singleworld_tail_sweep,
            kernel=tail_kernel,
            idt=idt,
        )

    def _solve_main_singleworld(self) -> None:
        """Single-world prepare + main PGS iterate path.

        One head+tail sweep for prepare, then ``solver_iterations``
        more head+tail sweeps for the bias-on iterate. Each sweep
        runs the persistent-grid kernel for large colours then hands
        off to the single-block fused kernel for the tail of small
        colours.
        """
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)

        prepare_head, prepare_fused, iterate_head, iterate_fused, _, _ = self._singleworld_kernels()

        self._partitioner.begin_sweep()
        self._singleworld_head_plus_tail_sweep(prepare_head, prepare_fused, idt)

        for _ in range(self.solver_iterations):
            self._partitioner.begin_sweep()
            self._singleworld_head_plus_tail_sweep(iterate_head, iterate_fused, idt)

    def _relax_velocities_singleworld(self) -> None:
        """Single-world TGS-soft relax sweeps (bias OFF)."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        _, _, _, _, relax_head, relax_fused = self._singleworld_kernels()
        for _ in range(self.velocity_iterations):
            self._partitioner.begin_sweep()
            self._singleworld_head_plus_tail_sweep(relax_head, relax_fused, idt)

    def _singleworld_kernels(self):
        """Resolve the single-world kernel set for this solver.

        Returns ``(prepare_head, prepare_fused, iterate_head,
        iterate_fused, relax_head, relax_fused)``. When
        :attr:`_use_revolute_specialization` is set (no joints, or
        every joint is :data:`JointMode.REVOLUTE`), returns the
        revolute-only variants which skip the per-cid
        ``read_int(_OFF_JOINT_MODE)`` and four-way ``joint_mode``
        branch in :func:`actuated_double_ball_socket_iterate`. Otherwise
        returns the generic dispatchers."""
        cloth_on = self.num_cloth_triangles > 0
        if self._use_revolute_specialization:
            if cloth_on:
                return (
                    _constraint_prepare_singleworld_revolute_cloth_kernel,
                    _constraint_prepare_singleworld_fused_revolute_cloth_kernel,
                    _constraint_iterate_singleworld_revolute_cloth_kernel,
                    _constraint_iterate_singleworld_fused_revolute_cloth_kernel,
                    _constraint_relax_singleworld_revolute_cloth_kernel,
                    _constraint_relax_singleworld_fused_revolute_cloth_kernel,
                )
            return (
                _constraint_prepare_singleworld_revolute_kernel,
                _constraint_prepare_singleworld_fused_revolute_kernel,
                _constraint_iterate_singleworld_revolute_kernel,
                _constraint_iterate_singleworld_fused_revolute_kernel,
                _constraint_relax_singleworld_revolute_kernel,
                _constraint_relax_singleworld_fused_revolute_kernel,
            )
        if cloth_on:
            return (
                _constraint_prepare_singleworld_cloth_kernel,
                _constraint_prepare_singleworld_fused_cloth_kernel,
                _constraint_iterate_singleworld_cloth_kernel,
                _constraint_iterate_singleworld_fused_cloth_kernel,
                _constraint_relax_singleworld_cloth_kernel,
                _constraint_relax_singleworld_fused_cloth_kernel,
            )
        return (
            _constraint_prepare_singleworld_kernel,
            _constraint_prepare_singleworld_fused_kernel,
            _constraint_iterate_singleworld_kernel,
            _constraint_iterate_singleworld_fused_kernel,
            _constraint_relax_singleworld_kernel,
            _constraint_relax_singleworld_fused_kernel,
        )

    def _fast_tail_block_dim(self) -> int:
        """Resolve the fast-tail block size for this solver instance.

        Always ``_STRAGGLER_BLOCK_DIM * wpb`` so the block holds an
        integer number of warps (``__syncwarp`` correctness). ``wpb``
        depends on ``num_worlds`` -- see
        :func:`_choose_fast_tail_worlds_per_block`. The block_dim is
        independent of the adaptive ``threads_per_world`` choice; the
        latter only affects how many lanes inside each warp do real
        work.
        """
        return _STRAGGLER_BLOCK_DIM * _choose_fast_tail_worlds_per_block(self.num_worlds)

    def _fast_tail_launch_dim(self) -> int:
        """Launch dim for the fast-tail kernels, padded up to the block
        size so we can use a block_dim wider than a single warp.

        Sized for the maximum threads-per-world (``_STRAGGLER_BLOCK_DIM
        = 32``); when the adaptive picker drops the effective tpw to 16
        or 8, the surplus lanes resolve a ``world_id`` past
        ``num_worlds`` and early-exit. Keeping the launch shape fixed
        is what lets the per-step picker live inside the captured CUDA
        graph -- changing ``dim`` would require re-capturing.
        """
        block_dim = self._fast_tail_block_dim()
        raw = self.num_worlds * _STRAGGLER_BLOCK_DIM
        return ((raw + block_dim - 1) // block_dim) * block_dim

    def _pick_tpw(self) -> None:
        """Run the per-step threads-per-world picker on the GPU.

        Two-kernel pipeline so the cost stays ~constant in
        ``num_worlds``: a parallel atomic reduction over
        ``_world_num_colors`` totals the colour count, then a 1-thread
        kernel reads the precomputed total cid count
        (``_world_csr_offsets[num_worlds]``) plus the device's SM count
        and writes the chosen tpw to ``_tpw_choice[0]``. No host sync;
        the whole pipeline lands in the captured CUDA graph.
        """
        sm_count = getattr(self.device, "sm_count", 0) or 0
        self._tpw_total_colours.zero_()
        wp.launch(
            _reduce_total_colours_kernel,
            dim=self.num_worlds,
            inputs=[self._world_num_colors, wp.int32(self.num_worlds)],
            outputs=[self._tpw_total_colours],
            device=self.device,
        )
        wp.launch(
            _pick_threads_per_world_kernel,
            dim=1,
            inputs=[
                self._world_csr_offsets,
                self._tpw_total_colours,
                wp.int32(self.num_worlds),
                wp.int32(sm_count),
            ],
            outputs=[self._tpw_choice],
            device=self.device,
        )

    def _launch_fast_iter(
        self,
        kernel,
        num_iterations: int,
        idt: wp.float32,
        contact_views: ContactViews,
    ) -> None:
        """Launch an iterate / relax kernel that runs
        ``num_iterations`` sweeps internally, one warp per world."""
        wp.launch(
            kernel,
            dim=self._fast_tail_launch_dim(),
            block_dim=self._fast_tail_block_dim(),
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                idt,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(num_iterations),
                wp.int32(self.num_worlds),
                wp.int32(self.num_joints),
                self._tpw_choice,
                self.body_or_particle.particles,
                self.tri_indices,
            ],
            device=self.device,
        )

    def _integrate_positions(self) -> None:
        """One post-iterate launch over the unified body-or-particle
        index space. The kernel internally dispatches on
        :func:`is_particle`:

        * Bodies: ``x += v * dt`` and ``q = dq(w * dt) * q`` for
          dynamic bodies. Static and kinematic bodies are skipped --
          kinematic pose advances via
          :meth:`_kinematic_interpolate_substep`. Axis-angle
          quaternion form keeps unit norm over many substeps.
        * Particles: substep-exit access-mode transition
          (Position-level -> Velocity-level). The cloth iterate has
          already written the constraint-projected position into
          ``particles.position``; recover ``velocity = (position -
          position_substep_start) * inv_dt`` so the next substep
          starts from a consistent velocity-level state.
        """
        n = self.num_bodies + self.num_particles
        if n == 0:
            return
        wp.launch(
            _integrate_velocities_kernel,
            dim=n,
            inputs=[
                self.body_or_particle,
                wp.float32(self.substep_dt),
                wp.float32(1.0 / self.substep_dt),
            ],
            device=self.device,
        )

    def _kinematic_prepare_step(self) -> None:
        """Once-per-step kinematic prepare: snapshot prev pose,
        resolve target, infer velocity. No-op when no kinematic bodies
        are present in the model -- the kernel's per-thread guard
        would early-return on every thread anyway, so we can skip
        launching the grid entirely and save ~4us."""
        if self.num_bodies == 0 or self._num_kinematic_bodies == 0:
            return
        wp.launch(
            _kinematic_prepare_step_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.step_dt)],
            device=self.device,
        )

    def _kinematic_interpolate_substep(self, alpha: float) -> None:
        """Per-substep kinematic pose update via lerp / slerp. Skipped
        when no kinematic bodies exist (see :meth:`_kinematic_prepare_step`
        rationale); at substeps=4 this is ~16us / step reclaimed in
        the common dynamic-only case."""
        if self.num_bodies == 0 or self._num_kinematic_bodies == 0:
            return
        wp.launch(
            _kinematic_interpolate_substep_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(alpha)],
            device=self.device,
        )

    def _refresh_world_inertia(self) -> None:
        """Per-substep refresh of ``bodies.inverse_inertia_world``.

        Rebuilds ``R * I_local^-1 * R^T`` from the body's current
        orientation, no damping, no force-clear. Must run after
        :meth:`_integrate_positions` so the next solve / next substep
        sees an inertia rotation consistent with the body's pose.
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_refresh_world_inertia_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    def _update_inertia_and_clear_forces(self) -> None:
        """One end-of-step launch over the unified body-or-particle
        index space. The kernel internally dispatches on
        :func:`is_particle`:

        * Bodies: apply damping, rebuild ``inverse_inertia_world``
          from the final orientation (``R * I^-1 * R^T``), and zero
          the force / torque accumulators (every body slot, including
          kinematic / static).
        * Particles: zero the force accumulator. Particles have no
          orientation / inertia / damping fields, so this is the only
          work.
        """
        n = self.num_bodies + self.num_particles
        if n == 0:
            return
        wp.launch(
            _phoenx_update_inertia_and_clear_forces_kernel,
            dim=n,
            inputs=[self.body_or_particle],
            device=self.device,
        )

    def _apply_global_damping(self) -> None:
        """Per-substep tail kernel: apply :attr:`_global_damping` to every
        dynamic body's linear / angular velocity. No-op (no kernel
        launch) until the user opts in via
        :meth:`set_global_linear_damping` / :meth:`set_global_angular_damping`."""
        if self._global_damping is None or self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_apply_global_damping_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self._global_damping],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Global damping API
    # ------------------------------------------------------------------

    def _ensure_global_damping_allocated(self) -> None:
        """Lazily allocate the device array on first opt-in. Subsequent
        setter calls reuse the allocation."""
        if self._global_damping is None:
            self._global_damping = wp.zeros(2, dtype=wp.float32, device=self.device)
            self._global_damping_host = np.zeros(2, dtype=np.float32)

    def set_global_linear_damping(self, value: float) -> None:
        """Set the per-substep global linear-velocity damping factor.

        Applied as ``v *= 1 - value`` at the end of every substep, on
        top of the per-body :attr:`linear_damping`. ``value`` must lie
        in ``[0, 1]``: ``0`` is a no-op; ``1`` zeroes the linear velocity
        of every dynamic body each substep (useful for a settle
        warm-up before live simulation).

        **Opt-in semantics.** The first call to this (or
        :meth:`set_global_angular_damping`) lazily allocates the
        backing :class:`wp.array` and starts launching the per-substep
        damping kernel. While neither setter has been called, the
        kernel is skipped entirely (zero cost).

        **Graph-capture rules.** Once the array exists, the value lives
        in a device slot the captured graph reads at replay time, so
        changing the factor between replays is safe. **Opting in for
        the first time mid-simulation invalidates any already-captured
        graph** (the new kernel launch isn't in it) -- either
        re-capture, or call ``set_global_*_damping(0.0)`` once *before*
        capture to lock the kernel in upfront and toggle the value
        later without re-capture.
        """
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"global_linear_damping must be in [0, 1] (got {v})")
        self._ensure_global_damping_allocated()
        self._global_damping_host[0] = v
        self._global_damping.assign(self._global_damping_host)

    def set_global_angular_damping(self, value: float) -> None:
        """Set the per-substep global angular-velocity damping factor.

        Applied as ``w *= 1 - value`` at the end of every substep, on
        top of the per-body :attr:`angular_damping`. See
        :meth:`set_global_linear_damping` for opt-in and graph-capture
        rules.
        """
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"global_angular_damping must be in [0, 1] (got {v})")
        self._ensure_global_damping_allocated()
        self._global_damping_host[1] = v
        self._global_damping.assign(self._global_damping_host)

    def get_global_linear_damping(self) -> float:
        """Current global linear damping factor; ``0.0`` if the user
        has never opted in. Host-shadow read, no device sync."""
        if self._global_damping_host is None:
            return 0.0
        return float(self._global_damping_host[0])

    def get_global_angular_damping(self) -> float:
        """Current global angular damping factor; ``0.0`` if the user
        has never opted in. Host-shadow read, no device sync."""
        if self._global_damping_host is None:
            return 0.0
        return float(self._global_damping_host[1])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_constraints(self) -> int:
        """Total allocated cid capacity (joints + contact columns).
        Use this to size output arrays for
        :meth:`gather_constraint_wrenches` /
        :meth:`gather_constraint_errors`."""
        return self._constraint_capacity

    @property
    def body_or_particle(self) -> BodyOrParticleStore:
        """Unified body-or-particle store for kernels that address
        either kind of "thing" by a single integer index.

        Mirrors the joint-or-contact cid scheme: unified indices
        ``[0, num_bodies)`` resolve to body slots,
        ``[num_bodies, num_bodies + num_particles)`` to particle
        slots. The branch lives inside the
        :func:`~newton._src.solvers.phoenx.body_or_particle.get_position`
        / ``get_velocity`` / etc. accessors -- constraint kernels
        consume this store directly without knowing or caring which
        kind of thing they're addressing.

        Lazy-allocated; rigid-only scenes that never touch this
        property pay zero memory cost. When
        :attr:`num_particles == 0` the cached store wraps a length-1
        sentinel :class:`ParticleContainer` so the wp.struct fields
        are valid; the threshold compare in the accessors guarantees
        the sentinel is never read.
        """
        if self._body_or_particle_store is None:
            particles = self.particles
            if particles is None:
                # Length-1 sentinel keeps the wp.struct field valid;
                # the threshold compare in the accessors makes sure
                # nothing ever indexes past num_bodies in this case.
                particles = particle_container_zeros(1, device=self.device)
            store = BodyOrParticleStore()
            store.bodies = self.bodies
            store.particles = particles
            store.num_bodies = wp.int32(self.num_bodies)
            self._body_or_particle_store = store
        return self._body_or_particle_store

    def gather_constraint_wrenches(self, out: wp.array) -> None:
        """Per-cid world-frame wrench on ``body2`` averaged over the
        last substep. ``out`` is
        ``wp.array[wp.spatial_vector]`` of length :attr:`num_constraints`."""
        if self._constraint_capacity == 0:
            return
        out.zero_()
        if self.substep_dt <= 0.0:
            return
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            _constraint_gather_wrenches_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                wp.int32(self._constraint_capacity),
                # Joint-container boundary; cloth cids in the joint
                # range will route through the joint wrench helper
                # which reads ADBS-typed fields. For pure rigid scenes
                # this stays bit-for-bit unchanged; cloth wrench
                # readout is a follow-up.
                wp.int32(self._joint_container_cids),
                idt,
                self._contact_container,
                contact_views,
            ],
            outputs=[out],
            device=self.device,
        )

    def gather_constraint_errors(self, out: wp.array) -> None:
        """Per-cid position-level constraint residual. ``out`` is
        ``wp.array[wp.spatial_vector]`` of length :attr:`num_constraints`."""
        if self._constraint_capacity == 0:
            return
        out.zero_()
        wp.launch(
            _constraint_gather_errors_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                wp.int32(self._constraint_capacity),
                wp.int32(self._joint_container_cids),
            ],
            outputs=[out],
            device=self.device,
        )

    def num_colors_used(self) -> int:
        """Number of graph colours the last PGS colouring used.
        Performs a device-to-host copy -- do not call inside a
        :func:`wp.ScopedCapture` region.

        For richer diagnostics (per-colour element counts, active
        contact column count, per-world breakdown), use
        :meth:`step_report`.
        """
        if self.step_layout == "single_world":
            return int(self._partitioner.num_colors.numpy()[0])
        # Multi-world: report the maximum colour depth across worlds,
        # matching what step_report().num_colors returns.
        return int(self._world_num_colors.numpy().max(initial=0))

    def step_report(self) -> PhoenXWorld.StepReport:
        """Snapshot of colouring + active-constraint diagnostics for
        the last :meth:`step`.

        Performs a handful of device-to-host copies (all gated on this
        call -- the step itself never reads them on the host) so the
        steady-state path stays graph-capture clean. Do not call
        inside a :func:`wp.ScopedCapture` region.

        Returns:
            A :class:`StepReport` populated with the colour count,
            per-colour element histogram, and active-contact-column
            count from the last :meth:`step`. If :meth:`step` has not
            been called yet (or was called with ``contacts=None`` and
            no joints exist) the report counts are all zero.
        """
        num_contact_columns = (
            int(self._ingest_scratch.num_contact_columns.numpy()[0])
            if self._contact_views is not None and self._ingest_scratch is not None
            else 0
        )
        num_active = self.num_joints + num_contact_columns

        # Per-body degree from the partitioner's adjacency CSR end array.
        # After ``partitioning_adjacency_store_kernel`` runs, slot ``v``
        # holds the inclusive end of body ``v``'s element-id list, so
        # ``deg(v) = end[v] - end[v-1]`` (with a 0 sentinel for v == 0).
        # The lower bound on any valid graph colouring is the maximum
        # degree, so this is the right number to compare ``num_colors``
        # against.
        if num_active > 0 and self.num_bodies > 0:
            ends = self._partitioner._adjacency_section_end_indices.numpy()
            n_bodies = min(int(self.num_bodies), int(ends.shape[0]))
            if n_bodies > 0:
                degrees = ends[:n_bodies].astype(np.int64, copy=False)
                degrees[1:] = degrees[1:] - degrees[:-1]
                max_body_degree = int(degrees.max(initial=0))
            else:
                max_body_degree = 0
        else:
            max_body_degree = 0

        if self.step_layout == "single_world":
            nc = int(self._partitioner.num_colors.numpy()[0])
            if nc > 0:
                starts = self._partitioner.color_starts.numpy()
                color_sizes = [int(starts[c + 1] - starts[c]) for c in range(nc)]
            else:
                color_sizes = []
            return self.StepReport(
                num_colors=nc,
                color_sizes=color_sizes,
                per_world_num_colors=None,
                per_world_color_sizes=None,
                num_contact_columns=num_contact_columns,
                num_joints=self.num_joints,
                num_active_constraints=num_active,
                max_body_degree=max_body_degree,
            )

        # Multi-world: per-world coloring. ``_world_num_colors`` is
        # length ``num_worlds`` and ``_world_color_starts`` is shape
        # ``[num_worlds, MAX_COLORS + 1]``.
        nc_per_world = self._world_num_colors.numpy().astype(np.int32, copy=False)
        starts_2d = self._world_color_starts.numpy().astype(np.int32, copy=False)
        per_world_num_colors: list[int] = [int(n) for n in nc_per_world]
        per_world_color_sizes: list[list[int]] = []
        max_nc = 0
        for w, n in enumerate(per_world_num_colors):
            row = starts_2d[w]
            sizes = [int(row[c + 1] - row[c]) for c in range(n)]
            per_world_color_sizes.append(sizes)
            if n > max_nc:
                max_nc = n
        # Aggregate per-colour-index totals across worlds. Worlds with
        # fewer than ``max_nc`` colours contribute zero to higher
        # indices.
        agg = [0] * max_nc
        for sizes in per_world_color_sizes:
            for c, s in enumerate(sizes):
                agg[c] += s
        return self.StepReport(
            num_colors=max_nc,
            color_sizes=agg,
            per_world_num_colors=per_world_num_colors,
            per_world_color_sizes=per_world_color_sizes,
            num_contact_columns=num_contact_columns,
            num_joints=self.num_joints,
            num_active_constraints=num_active,
            max_body_degree=max_body_degree,
        )

    def gather_contact_wrenches(self, out: wp.array) -> None:
        """Per-individual-contact wrench (force + torque) from the
        last substep. ``out`` is ``wp.array[wp.spatial_vector]`` of
        length :attr:`rigid_contact_max`."""
        if self.max_contact_columns == 0:
            out.zero_()
            return
        out.zero_()
        if self.substep_dt <= 0.0 or self._contact_views is None:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            contact_per_contact_wrench_kernel,
            dim=self.max_contact_columns,
            inputs=[
                self._contact_cols,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(self.max_contact_columns),
                idt,
            ],
            outputs=[out],
            device=self.device,
        )

    def gather_contact_pair_wrenches(
        self,
        wrenches: wp.array,
        body1: wp.array,
        body2: wp.array,
        contact_count: wp.array,
    ) -> None:
        """Per-contact-column wrench summary."""
        if self.max_contact_columns == 0:
            return
        if self.substep_dt <= 0.0 or self._contact_views is None:
            wrenches.zero_()
            body1.fill_(-1)
            body2.fill_(-1)
            contact_count.zero_()
            return
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            contact_pair_wrench_kernel,
            dim=self.max_contact_columns,
            inputs=[
                self._contact_cols,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(self.max_contact_columns),
                idt,
            ],
            outputs=[wrenches, body1, body2, contact_count],
            device=self.device,
        )

    def gather_contact_errors(self, out: wp.array) -> None:
        """Per-individual-contact position-level residual."""
        if self.max_contact_columns == 0:
            out.zero_()
            return
        out.zero_()
        if self._contact_views is None:
            return
        wp.launch(
            contact_per_contact_error_kernel,
            dim=self.max_contact_columns,
            inputs=[
                self._contact_cols,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(self.max_contact_columns),
            ],
            outputs=[out],
            device=self.device,
        )
