"""Local deterministic support-star partitioning; production equations unchanged."""

import functools
import types

import warp as wp
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    BodyContainer,
    ConstraintContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
    _make_singleworld_dispatch_func,
    _sync_threads,
)

from newton._src.solvers.phoenx.constraints.constraint_contact import contact_get_body1, contact_get_body2
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups as topology
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import build_interaction_graph
from newton._src.solvers.phoenx.mass_splitting.slot_cache import build_partition_slot_cache_kernel


@wp.kernel(enable_backward=False)
def regroup(
    elements: wp.array[ElementInteractionData],
    active: wp.array[wp.int32],
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    joint_count: int,
    contact_offset: int,
    width: int,
    old_ids: wp.array[wp.int32],
    old_starts: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    row_partition: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    ids: wp.array[wp.int32],
    stages: wp.array[wp.int32],
    partitions: wp.array[wp.int32],
    num_partitions: wp.array[wp.int32],
    max_star_rows: int,
):
    # Static endpoints are excluded from element coloring. Use original body
    # headers to recognize support without treating the world as a graph hub.
    for body in range(bodies.inverse_mass.shape[0]):
        parent[body] = -1
        sizes[body] = 0
    for row in range(contact_offset, active[0]):
        a = contact_get_body1(columns, row - contact_offset)
        b = contact_get_body2(columns, row - contact_offset)
        if bodies.inverse_mass[a] > 0.0 and bodies.inverse_mass[b] == 0.0:
            parent[a] = a
        if bodies.inverse_mass[b] > 0.0 and bodies.inverse_mass[a] == 0.0:
            parent[b] = b
    # Connected supported bodies share one star, resolving joint ownership
    # deterministically by the smallest body index. No recursive absorption
    # of unsupported neighbors.
    for row in range(joint_count):
        a = elements[row].bodies[0]
        b = elements[row].bodies[1]
        if a >= 0 and b >= 0 and parent[a] >= 0 and parent[b] >= 0:
            root_a = a
            root_b = b
            while parent[root_a] != root_a:
                root_a = parent[root_a]
            while parent[root_b] != root_b:
                root_b = parent[root_b]
            parent[wp.max(root_a, root_b)] = wp.min(root_a, root_b)
    for body in range(bodies.inverse_mass.shape[0]):
        if parent[body] >= 0:
            root = body
            while parent[root] != root:
                root = parent[root]
            parent[body] = root
    original_partitions = (colors[0] + width - 1) / width
    for row in range(active[0]):
        owner = int(-1)
        if row < joint_count:
            for endpoint in range(8):
                body = elements[row].bodies[endpoint]
                if body >= 0 and parent[body] >= 0:
                    if owner < 0 or parent[body] < owner:
                        owner = parent[body]
        elif row >= contact_offset:
            a = contact_get_body1(columns, row - contact_offset)
            b = contact_get_body2(columns, row - contact_offset)
            if bodies.inverse_mass[a] > 0.0 and bodies.inverse_mass[b] == 0.0:
                owner = parent[a]
            elif bodies.inverse_mass[b] > 0.0 and bodies.inverse_mass[a] == 0.0:
                owner = parent[b]
        if owner >= 0:
            row_partition[row] = original_partitions + owner
            sizes[owner] += 1
    # Oversized components retain the original schedule as a whole. Every
    # contact and joint remains present, with no partial-star ownership rule.
    for color in range(colors[0]):
        for index in range(old_starts[color], old_starts[color + 1]):
            row = old_ids[index]
            pid = row_partition[row]
            if pid >= original_partitions and sizes[pid - original_partitions] > max_star_rows:
                row_partition[row] = color / width
    # Stable variable CSR: original color order, then original row order.
    # Empty original groups are retained to keep partition IDs unambiguous.
    out = int(0)
    stage = int(0)
    group_count = original_partitions + bodies.inverse_mass.shape[0]
    partitions[0] = 0
    for group in range(group_count):
        for color in range(colors[0]):
            begin = out
            for index in range(old_starts[color], old_starts[color + 1]):
                row = old_ids[index]
                if row_partition[row] == group:
                    ids[out] = row
                    out += 1
            if out > begin:
                stages[stage] = begin
                stage += 1
        partitions[group + 1] = stage
    stages[stage] = out
    num_partitions[0] = group_count


@functools.cache
def sweep_kernel(phase, soft_pd):
    dispatch, _ = _make_singleworld_dispatch_func(
        cloth_support=False,
        soft_tet_neohookean=False,
        enable_column_timers=False,
        has_joints=True,
        skip_joint_pgs=False,
        has_mass_splitting=True,
        packed_contact_headers=False,
        has_sleeping=False,
        has_soft_contact_pd=soft_pd,
        is_prepare=phase == "prepare",
        is_cached_prepare=phase == "cached_prepare",
        use_bias=phase == "iterate",
        patch_friction=False,
        bilateral_joint_blocks=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        c: ConstraintContainer,
        columns: ContactColumnContainer,
        b: BodyContainer,
        p: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
        joints: int,
        enabled: wp.array[wp.int32],
        bodies: int,
        idt: float,
        ids: wp.array[wp.int32],
        stages: wp.array[wp.int32],
        partitions: wp.array[wp.int32],
        count: wp.array[wp.int32],
    ):
        block, lane = wp.tid()
        for group in range(block, count[0], 64):
            for stage in range(partitions[group], partitions[group + 1]):
                for index in range(stages[stage] + lane, stages[stage + 1], 32):
                    dispatch(
                        c,
                        columns,
                        b,
                        p,
                        cc,
                        contacts,
                        copies,
                        joints,
                        enabled,
                        0,
                        0,
                        0,
                        0,
                        bodies,
                        idt,
                        1.0,
                        ids[index],
                        index,
                        group,
                    )
                _sync_threads()

    return sweep


def attach(world, max_star_rows=64):
    """Install bounded graph-capturable schedule hooks on one rigid world."""
    from newton._src.solvers.phoenx.solver_phoenx import _get_parallel_contact_prepare_kernel

    if world._color_group_data is None or world._colored_contact_rows or world.num_particles:
        raise ValueError("Support stars require existing rigid color-group dispatch")
    capacity = world._constraint_capacity
    size = capacity + world.num_bodies + 2
    extra = {
        name: wp.zeros(size, dtype=wp.int32, device=world.device)
        for name in ("ids", "stages", "partitions", "parent", "sizes")
    }
    extra["count"] = wp.zeros(1, dtype=wp.int32, device=world.device)
    world._support_star_data = extra

    def rebuild(self):
        data = self._color_group_data
        topology.build(
            data, self._elements, self._num_active_constraints, self.mass_splitting_color_group_size, self.device
        )
        wp.launch(
            regroup,
            1,
            [
                self._elements,
                self._num_active_constraints,
                self._contact_cols,
                self.bodies,
                self.num_joints,
                self._contact_offset,
                self.mass_splitting_color_group_size,
                data["ids"],
                data["starts"],
                data["num_colors"],
                data["row_partition"],
                extra["parent"],
                extra["sizes"],
                extra["ids"],
                extra["stages"],
                extra["partitions"],
                extra["count"],
                max_star_rows,
            ],
            device=self.device,
        )
        wp.launch(
            topology.emit_partition_pairs,
            capacity,
            [self._elements, self._num_active_constraints, data["row_partition"], self._interaction_graph_scratch],
            device=self.device,
        )
        build_interaction_graph(self._interaction_graph_scratch, self._copy_state)
        wp.launch(
            build_partition_slot_cache_kernel,
            capacity,
            [
                extra["ids"],
                data["row_partition"],
                self._num_active_constraints,
                self._copy_state,
                self.constraints,
                self._contact_cols,
                self._contact_offset,
            ],
            device=self.device,
        )

    def sweep(self, head, idt, contact_container=None):
        phase = ("prepare", "iterate", "relax")[self._singleworld_kernels()[::2].index(head)]
        cc = self._contact_container if contact_container is None else contact_container
        soft_pd = bool(self._dispatch_specialization_flags()["has_soft_contact_pd"])
        if phase == "prepare" and self.parallel_contact_prepare:
            if self.max_contact_columns:
                wp.launch(
                    _get_parallel_contact_prepare_kernel(True, soft_pd),
                    (self.max_contact_columns, min(128, self.contact_chunk_size or 128)),
                    [
                        self._contact_cols,
                        self._ingest_scratch.num_contact_columns,
                        self.bodies,
                        self._particles_or_sentinel(),
                        self.num_bodies,
                        idt,
                        cc,
                        self._active_contact_views(),
                        self._copy_state,
                    ],
                    device=self.device,
                )
            phase = "cached_prepare"
        wp.launch(
            sweep_kernel(phase, soft_pd),
            (64, 32),
            [
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._particles_or_sentinel(),
                cc,
                self._active_contact_views(),
                self._copy_state,
                self.num_joints,
                self._joint_pgs_enabled,
                self.num_bodies,
                idt,
                extra["ids"],
                extra["stages"],
                extra["partitions"],
                extra["count"],
            ],
            block_dim=32,
            device=self.device,
        )

    world._rebuild_mass_splitting_graph = types.MethodType(rebuild, world)
    world._color_group_sweep = types.MethodType(sweep, world)


def install():
    """Attach local scheduling before the public example captures its graph."""
    from newton._src.solvers.phoenx.solver import SolverPhoenX

    original = SolverPhoenX.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        attach(self.world)

    SolverPhoenX.__init__ = initialize
