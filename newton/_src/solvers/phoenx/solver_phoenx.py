# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PhoenX world driver: unified joint, deformable, and contact PGS."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations import (
    ArticulationDeviceSystem,
    ArticulationTopology,
    PrefactorizedArticulationSystem,
)
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.cloth_collision import (
    PhoenXClothShareVertexFilterData,
    _phoenx_pack_cloth_contact_barycentric_kernel,
    _phoenx_pack_cloth_contact_endpoints_kernel,
    _phoenx_populate_shape_endpoints_kernel,
    _phoenx_update_cloth_shape_geometry_kernel,
    _phoenx_update_soft_tet_shape_geometry_kernel,
    build_phoenx_share_vertex_filter_data,
    phoenx_cloth_share_vertex_filter,
    shape_endpoints_zeros,
)
from newton._src.solvers.phoenx.cloth_step import (
    cloth_init_triangle_rows_kernel,
    cloth_predict_kernel,
    cloth_recover_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_bending import (
    CLOTH_BENDING_DWORDS,
    CLOTH_BENDING_TIME_US_OFFSET,
    cloth_bending_init_rows_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    CLOTH_TRIANGLE_DWORDS,
    CLOTH_TRIANGLE_TIME_US_OFFSET,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    CONTACT_DWORDS,
    CONTACT_TIME_US_OFFSET,
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_pair_wrench_kernel,
    contact_per_contact_error_kernel,
    contact_per_contact_wrench_kernel,
    contact_views_make,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_MULTIPLIER_DWORDS,
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    ADBS_DWORDS,
    ADBS_TIME_US_OFFSET,
    JOINT_MODE_REVOLUTE,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_hexahedron import (
    SOFT_HEX_DWORDS,
    SOFT_HEX_STRAIN_MODEL_ARAP,
    SOFT_HEX_STRAIN_MODEL_TRACE,
    SOFT_HEX_TIME_US_OFFSET,
    soft_hex_init_rows_from_arrays_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import (
    SOFT_TET_NEOHOOKEAN_DWORDS,
    SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET,
    SoftBodyConstraintType,
    soft_tet_neohookean_init_rows_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    SOFT_TET_DWORDS,
    SOFT_TET_TIME_US_OFFSET,
    soft_tet_init_rows_kernel,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    CC_DERIVED_DWORDS_PER_CONTACT,
    CC_DWORDS_PER_CONTACT,
    CC_IMPULSE_DWORDS_PER_CONTACT,
    ContactContainer,
    contact_container_copy_current_to_prev,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_ingest import (
    IngestScratch,
    gather_contact_warmstart,
    ingest_contacts,
    stamp_forward_contact_map,
)
from newton._src.solvers.phoenx.dispatch.multi_world import MultiWorldDispatcher
from newton._src.solvers.phoenx.dispatch.single_world import SingleWorldDispatcher
from newton._src.solvers.phoenx.dispatch.single_world_mass_splitting import (
    SingleWorldMassSplittingDispatcher,
)
from newton._src.solvers.phoenx.dispatch.single_world_mass_splitting_unrolled import (
    SingleWorldMassSplittingUnrolledDispatcher,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    MAX_COLORS,
    IncrementalContactPartitioner,
)
from newton._src.solvers.phoenx.graph_coloring.luby_fixed import (
    FixedIterationLubyPartitioner,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int
from newton._src.solvers.phoenx.islands.island_builder import (
    MAX_BODIES_PER_INTERACTION,
    UnionFindIslandBuilder,
)
from newton._src.solvers.phoenx.mass_splitting import (
    CopyStateContainer,
    InteractionGraphScratch,
    build_constraint_slot_cache,
    build_interaction_graph,
    copy_state_container_zeros,
    interaction_graph_scratch_zeros,
    launch_average_and_broadcast,
    launch_average_and_broadcast_grouped,
    launch_average_and_broadcast_rigid_velocity,
    launch_broadcast_rigid_to_copy_states,
    launch_copy_state_into_rigids,
    record_all_interactions_kernel,
)
from newton._src.solvers.phoenx.materials import MaterialData
from newton._src.solvers.phoenx.particle import ParticleContainer, particle_container_zeros
from newton._src.solvers.phoenx.sleeping_kernels import (
    _phoenx_apply_island_wake_kernel,
    _phoenx_apply_wake_flag_kernel,
    _phoenx_collapse_sleeping_elements_kernel,
    _phoenx_compute_island_root_per_compact_kernel,
    _phoenx_copy_elements_to_int2d_kernel,
    _phoenx_detect_active_islands_kernel,
    _phoenx_finalize_body_aabb_diagonal_kernel,
    _phoenx_init_body_aabb_kernel,
    _phoenx_inject_chain_edges_kernel,
    _phoenx_island_fanin_external_input_kernel,
    _phoenx_island_max_velocity_kernel,
    _phoenx_mark_sleeping_islands_kernel,
    _phoenx_propagate_sleep_to_bodies_kernel,
    _phoenx_seed_uf_num_interactions_kernel,
    _phoenx_self_wake_fanin_kernel,
    _phoenx_shape_aabb_fanin_kernel,
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
    _PER_WORLD_FAST_FAMILIES,
    _STRAGGLER_BLOCK_DIM,
    _build_scatter_keys_kernel,
    _choose_fast_tail_worlds_per_block,
    _constraint_gather_errors_kernel,
    _constraint_gather_wrenches_kernel,
    _constraints_to_elements_kernel,
    _count_elements_per_world_kernel,
    _integrate_velocities_kernel,
    _kinematic_interpolate_substep_kernel,
    _kinematic_prepare_step_kernel,
    _per_world_greedy_coloring_kernel,
    _per_world_jp_coloring_kernel,
    _phoenx_apply_forces_and_gravity_kernel,
    _phoenx_apply_global_damping_kernel,
    _phoenx_refresh_world_inertia_kernel,
    _phoenx_update_inertia_and_clear_forces_kernel,
    _pick_threads_per_world_kernel,
    _reduce_constraint_time_us_kernel,
    _reduce_contact_time_us_kernel,
    _reduce_total_colours_kernel,
    _set_kinematic_pose_batch_kernel,
    _zero_constraint_time_us_kernel,
    _zero_contact_time_us_kernel,
    get_block_world_kernel,
    get_fast_tail_kernel,
    get_singleworld_kernel,
    pack_body_xforms_kernel,
)

__all__ = [
    "DEFAULT_SHAPE_GAP",
    "PhoenXWorld",
    "pack_body_xforms_kernel",
]


#: Default contact-detection gap [m]. 5 cm is generous so PhoenX's speculative
#: branch decelerates closing bodies while still apart. Override for MEMS/vehicles.
DEFAULT_SHAPE_GAP: float = 0.05


@wp.kernel(enable_backward=False)
def _update_contact_generation_reuse_kernel(
    contact_generation: wp.array[wp.int32],
    reuse_contact_indices: wp.array[wp.int32],
    last_contact_generation: wp.array[wp.int32],
):
    gen = contact_generation[0]
    if gen == last_contact_generation[0]:
        reuse_contact_indices[0] = wp.int32(1)
    else:
        reuse_contact_indices[0] = wp.int32(0)
    last_contact_generation[0] = gen


def _soft_hex_strain_model_value(strain_model: str | int) -> int:
    """Return the internal soft-hex strain model id."""
    if isinstance(strain_model, str):
        normalized = strain_model.strip().lower().replace("-", "_")
        if normalized in {"trace", "xpbd_fem", "xpbd_fem_trace", "invariant"}:
            return SOFT_HEX_STRAIN_MODEL_TRACE
        if normalized in {"arap", "integrated_arap"}:
            return SOFT_HEX_STRAIN_MODEL_ARAP
        raise ValueError(f"soft hex strain_model must be 'trace'/'xpbd_fem' or 'arap' (got {strain_model!r})")
    value = int(strain_model)
    if value in (SOFT_HEX_STRAIN_MODEL_TRACE, SOFT_HEX_STRAIN_MODEL_ARAP):
        return value
    raise ValueError(
        "soft hex strain_model must be SOFT_HEX_STRAIN_MODEL_TRACE "
        f"({SOFT_HEX_STRAIN_MODEL_TRACE}) or SOFT_HEX_STRAIN_MODEL_ARAP "
        f"({SOFT_HEX_STRAIN_MODEL_ARAP}) (got {strain_model!r})"
    )


def _build_gravity_array(gravity, num_worlds: int, device) -> wp.array[wp.vec3f]:
    """Coerce ``gravity`` into wp.array[wp.vec3f] of length ``num_worlds``.
    Accepts a 3-tuple (broadcast) or an iterable of num_worlds 3-tuples."""
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


def _choose_initial_threads_per_world(
    *,
    num_worlds: int,
    num_joints: int,
    max_contact_columns: int,
    sm_count: int,
) -> tuple[bool, int]:
    """Choose the graph-capture-stable initial fast-tail lane count.

    Returns ``(tpw_auto, initial_tpw)``. ``tpw_auto=True`` keeps the
    per-step GPU picker active; otherwise ``initial_tpw`` is the fixed
    fast-tail specialization used by captured graphs.
    """
    worlds = max(1, int(num_worlds))
    sm = max(1, int(sm_count) or 1)
    joints_per_world = float(num_joints) / float(worlds)
    contacts_capacity_per_world = float(max_contact_columns) / float(worlds)

    tpw_auto = worlds >= 8 * sm
    initial_tpw = _STRAGGLER_BLOCK_DIM

    sparse_joint_only_world = 0.0 < joints_per_world <= 40.0 and max_contact_columns == 0
    if sparse_joint_only_world and worlds >= 4 * sm:
        return False, 8
    if sparse_joint_only_world and worlds >= 2 * sm:
        return False, 16

    if tpw_auto:
        small_joint_world = 0.0 < joints_per_world <= 64.0 and contacts_capacity_per_world <= 512.0
        dense_joint_world = joints_per_world > 64.0
        dense_contact_only_world = num_joints == 0 and contacts_capacity_per_world > 256.0
        simple_saturated_joint_contact_world = (
            worlds >= 16 * sm and 0.0 < joints_per_world <= 32.0 and 128.0 <= contacts_capacity_per_world <= 384.0
        )
        if simple_saturated_joint_contact_world:
            return False, 8
        if small_joint_world:
            return False, 16
        if dense_contact_only_world:
            return False, 8
        if dense_joint_world:
            return False, _STRAGGLER_BLOCK_DIM

    return tpw_auto, initial_tpw


def _choose_auto_prepare_refresh_stride(
    *,
    substeps: int,
    contact_capacity_hint: int,
    cached_prepare_unsupported: bool,
) -> int:
    """Choose a graph-capture-stable cached-prepare refresh cadence."""
    if cached_prepare_unsupported or substeps < 8:
        return 1
    if contact_capacity_hint <= 0:
        return 1
    return 3


def _choose_multi_world_scheduler(
    *,
    block_world_supported: bool,
    num_worlds: int,
    num_joints: int,
    max_contact_columns: int,
) -> tuple[str, int]:
    """Choose a fixed multi-world scheduler from construction-time topology."""
    if not block_world_supported:
        return "fast_tail", 128

    inv_worlds = 1.0 / float(max(1, int(num_worlds)))
    joints_per_world = float(num_joints) * inv_worlds
    contacts_per_world = float(max_contact_columns) * inv_worlds
    rows_per_world = joints_per_world + contacts_per_world

    # Sparse toy worlds are better packed by fast-tail; one physical
    # block per world only starts to pay once each world has enough rows.
    if rows_per_world < 16.0:
        return "fast_tail", 128

    # Dense contact-only fleets are consistently limited by fast-tail's
    # per-world tail work; one full block per world keeps more lanes useful.
    if joints_per_world == 0.0 and contacts_per_world >= 512.0:
        return "block_world", 128

    # Robot fleets are currently too mixed for a static topology-only
    # block-world rule: solve-only sweeps can improve, but full-frame graph
    # replay regresses on DR-style scenes.
    return "fast_tail", 128


def _choose_fast_tail_solve_schedule(*, substeps: int) -> tuple[int, int, int]:
    """Select the fast-tail register reuse schedule.

    Returns ``(joint_inner_sweeps, contact_inner_sweeps, outer_iteration_chunk)``.
    """
    # High-substep robot fleets use the old two-by-two schedule: fewer outer
    # visits, but matching inner sweeps, so default even iteration counts keep
    # the same total row sweeps. Low-substep humanoid stiffness keeps the
    # conservative full outer cadence.
    if substeps >= 64:
        return 2, 2, 2
    return 3, 3, 1


def _choose_fast_tail_worlds_per_block_for_scene(
    *,
    num_worlds: int,
    num_joints: int,
    max_contact_columns: int,
    step_layout: str,
    tpw_launch_bound: int,
) -> int:
    """Choose fast-tail block packing from topology known at finalize time."""
    wpb = _choose_fast_tail_worlds_per_block(num_worlds)
    if step_layout == "single_world":
        return wpb

    inv_worlds = 1.0 / float(max(1, int(num_worlds)))
    joints_per_world = float(num_joints) * inv_worlds
    contacts_per_world = float(max_contact_columns) * inv_worlds

    # Dense contact-only worlds have long per-world colour loops. Packing
    # several worlds into one block made the contact-heavy tower fleet slower.
    if joints_per_world == 0.0 and contacts_per_world >= 512.0:
        return 1

    if int(tpw_launch_bound) <= 16 and int(num_worlds) >= 512:
        if joints_per_world <= 48.0 and contacts_per_world <= 512.0:
            wpb = min(wpb, 2)
    return wpb


def _choose_fast_tail_family_split_for_scene(
    *,
    step_layout: str,
    use_greedy_coloring: bool,
    num_worlds: int,
    num_joints: int,
    max_contact_columns: int,
    num_cloth_triangles: int,
    num_cloth_bending: int,
    num_soft_tetrahedra: int,
    num_soft_hexahedra: int,
) -> bool:
    """Choose whether fast-tail kernels consume per-family color ranges."""
    if step_layout == "single_world" or not use_greedy_coloring:
        return False

    deformable_family_count = 0
    if num_cloth_triangles > 0:
        deformable_family_count += 1
    if num_cloth_bending > 0:
        deformable_family_count += 1
    if num_soft_tetrahedra > 0:
        deformable_family_count += 1
    if num_soft_hexahedra > 0:
        deformable_family_count += 1

    if deformable_family_count == 0:
        return num_worlds >= 512 and num_joints > 0 and max_contact_columns > 0

    family_count = deformable_family_count
    if num_joints > 0:
        family_count += 1
    if max_contact_columns > 0:
        family_count += 1
    return family_count > 1


def _mass_splitting_copy_capacity(
    *,
    num_joints: int,
    num_cloth_triangles: int,
    num_cloth_bending: int,
    num_soft_tetrahedra: int,
    num_soft_hexahedra: int,
    num_particles: int,
    max_contact_columns: int,
) -> int:
    """Upper-bound emitted ``(node, partition)`` pairs for copy-state scratch."""
    contact_endpoints = 2
    if num_particles > 0 or num_cloth_triangles > 0 or num_soft_tetrahedra > 0 or num_soft_hexahedra > 0:
        # Cloth/soft contacts can touch up to two soft tets: 4 + 4 nodes.
        contact_endpoints = int(MAX_BODIES)
    capacity = (
        int(num_joints) * 2
        + int(num_cloth_triangles) * 3
        + int(num_cloth_bending) * 4
        + int(num_soft_tetrahedra) * 4
        + int(num_soft_hexahedra) * int(MAX_BODIES)
        + int(max_contact_columns) * contact_endpoints
    )
    return max(1, capacity)


#: Persistent-grid block dim for the single-world iterate / prepare /
#: relax kernels. One warp per block (32 threads) gives the most blocks
#: in flight per SM and decouples the per-block ``__syncthreads()`` to
#: a warp-sync. Kapla single-world FPS scales monotonically as block
#: dim decreases from 256 (baseline 63 FPS) -> 32 (124 FPS, +97 %),
#: with dragon flat-to-mildly-positive (+6 %). Bit-exact determinism +
#: tower/stacking tests preserved.
_SINGLEWORLD_BLOCK_DIM: int = 32


def _singleworld_total_threads(
    constraint_capacity: int,
    device,
    max_thread_blocks: int | None = None,
) -> int:
    """Persistent grid size for the single-world PGS kernels: ``clamp(ceil(cap /
    256), 32, 4 * sm_count)`` blocks of 256 threads. ``max_thread_blocks`` opts
    out of the floor + SM cap and uses ``min(capacity_blocks, max_thread_blocks)``."""
    block_dim = _SINGLEWORLD_BLOCK_DIM
    capacity_blocks = (max(1, int(constraint_capacity)) + block_dim - 1) // block_dim
    if max_thread_blocks is not None:
        if int(max_thread_blocks) < 1:
            raise ValueError(f"max_thread_blocks must be >= 1 (got {max_thread_blocks})")
        num_blocks = max(1, min(capacity_blocks, int(max_thread_blocks)))
        return block_dim * num_blocks
    device_obj = wp.get_device(device)
    if device_obj.is_cuda:
        max_blocks_limit = device_obj.sm_count * 4
    else:
        max_blocks_limit = 256
    min_blocks = 32
    num_blocks = max(min_blocks, min(capacity_blocks, max_blocks_limit))
    return block_dim * num_blocks


class PhoenXWorld:
    """PhoenX solver driver.

    Cids are joints, deformables, then contact columns. Contacts use
    :class:`ContactColumnContainer`; the rest use :class:`ConstraintContainer`.
    """

    @dataclass
    class StepReport:
        """Diagnostic snapshot. Triggers D2H copies; not graph-capture safe."""

        num_colors: int
        """Graph colour count from the last PGS. Multi-world: max across worlds."""

        color_sizes: list[int]
        """Element count per colour. Multi-world: sum across worlds per index."""

        per_world_num_colors: list[int] | None
        """Per-world colour counts; None for single-world."""

        per_world_color_sizes: list[list[int]] | None
        """Per-world per-colour element counts; None for single-world."""

        num_contact_columns: int
        """Active contact columns from the last step."""

        num_joints: int
        """Joint constraint columns (static for the world's lifetime)."""

        num_active_constraints: int
        """Active cids, including joints, deformables, and contacts."""

        max_body_degree: int
        """Max constraints incident to any body or particle node. Hard lower
        bound on the colour count any valid colouring can achieve."""

        time_us_total_joints: float | None = None
        """Total wall-clock microseconds spent in joint dispatches (sum of
        every constraint column's ``time_us`` slot). ``None`` unless
        :attr:`PhoenXWorld.enable_column_timers` is set."""

        time_us_total_cloth_triangles: float | None = None
        """Total wall-clock microseconds spent in cloth-triangle dispatches.
        ``None`` unless :attr:`PhoenXWorld.enable_column_timers` is set."""

        time_us_total_cloth_bending: float | None = None
        """Total wall-clock microseconds spent in cloth-bending dispatches.
        ``None`` unless :attr:`PhoenXWorld.enable_column_timers` is set."""

        time_us_total_soft_tetrahedra: float | None = None
        """Total wall-clock microseconds spent in soft-tet dispatches.
        ``None`` unless :attr:`PhoenXWorld.enable_column_timers` is set."""

        time_us_total_contacts: float | None = None
        """Total wall-clock microseconds spent in contact dispatches.
        ``None`` unless :attr:`PhoenXWorld.enable_column_timers` is set."""

        time_us_total_soft_hexahedra: float | None = None
        """Total wall-clock microseconds spent in soft-hex dispatches.
        ``None`` unless :attr:`PhoenXWorld.enable_column_timers` is set."""

    def __init__(
        self,
        bodies: BodyContainer,
        constraints: ConstraintContainer,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 1,
        gravity: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        rigid_contact_max: int = 0,
        max_contact_columns: int | None = None,
        num_joints: int = 0,
        num_particles: int = 0,
        num_cloth_triangles: int = 0,
        num_cloth_bending: int = 0,
        num_soft_tetrahedra: int = 0,
        num_soft_hexahedra: int = 0,
        collision_filter_pairs: Iterable[tuple[int, int]] | None = None,
        default_friction: float = 0.5,
        num_worlds: int = 1,
        step_layout: str = "multi_world",
        threads_per_world: int | str = "auto",
        multi_world_scheduler: str = "auto",
        max_thread_blocks: int | None = None,
        enable_body_pair_grouping: bool = False,
        mass_splitting: bool = False,
        max_colored_partitions: int = 12,
        mass_splitting_batch_size: int = 8,
        mass_splitting_unrolled: bool = False,
        partitioner_algorithm: str = "greedy",
        max_greedy_outer_iters: int | None = None,
        enable_warm_start_coloring: bool = True,
        symmetric_color_sweep: bool = False,
        # Defaults tuned for contact-heavy rigid scenes (Kapla, stacks).
        # cache-stir knobs break the warm-start colouring lock-in;
        # capture_while + speculative are perf wins with no quality
        # regression. See the matching ``set_*`` docstrings on the
        # partitioner.
        warm_start_invalidate_period: int | None = None,
        warm_start_rotate_skip_color: bool | None = None,
        warm_start_rotate_skip_width: int = 1,
        capture_while_greedy_coloring: bool = True,
        speculative_coloring: bool = True,
        sor_boost: float = 1.0,
        enable_column_timers: bool = False,
        articulation_dvi_host: bool = False,
        articulation_dvi_replaces_joint_pgs: bool | None = None,
        articulation_dvi_host_solver: str = "block_sparse",
        cache_articulation_topology: bool = True,
        sleeping_velocity_threshold: float = 0.0,
        sleeping_frames_required: int = 30,
        prepare_refresh_stride: int | str = "auto",
        device: wp.context.Devicelike = None,
    ):
        """Take ownership of pre-built body and constraint containers.

        Args:
            bodies, constraints: Pre-built containers. Joints occupy
                cid ``[0, num_joints)``; contacts ``[num_joints, ...)``.
            substeps, solver_iterations, velocity_iterations: PGS
                schedule. ``velocity_iterations=1`` enables TGS-soft
                relax (recommended for tall stacks).
            articulation_dvi_host: Run the host-validation full-coordinate
                DVI articulation solve inside each substep. Defaults to
                ``False`` because this path performs host/device transfers
                and is not CUDA-graph capturable.
            articulation_dvi_replaces_joint_pgs: When ``True``, PhoenX keeps
                joint CIDs in the coloring/contact layout but skips joint PGS
                dispatches so the DVI articulation solve owns those rows.
                Defaults to the value of ``articulation_dvi_host``.
            articulation_dvi_host_solver: DVI numeric solve method.
                ``"device_block_sparse"`` uses the Warp block-sparse solve.
                ``"block_sparse"`` uses the host prefactorized block LDLT
                validation path and is the most robust option for cyclic
                full-coordinate mechanisms, while ``"dense"`` uses the dense
                host LDLT fallback.
            cache_articulation_topology: Build and store DVI articulation
                topology during joint initialization. Disable this for normal
                PGS-only SolverPhoenX worlds to avoid DVI setup work.
            prepare_refresh_stride: Refresh cached per-row prepare data
                every N substeps in rigid contact/joint scenes without
                deformables, mass splitting, or sleeping. ``"auto"``
                chooses a conservative stride from the substep count and
                falls back to ``1`` when cached prepare is unsupported.
                Pass ``1`` to force exact per-substep rebuilds. Contact
                worlds currently support up to ``3``; joint-only worlds
                may use larger values.
            gravity: 3-tuple or iterable of ``num_worlds`` 3-tuples.
            rigid_contact_max: Sizes per-contact state. ``0`` disables
                contacts.
            max_contact_columns: Optional cap for per-column state.
                Contact columns are grouped by shape or body pair, so
                broad-phase pair budgets can be much tighter than
                ``rigid_contact_max`` in manifold-heavy scenes.
            step_layout: ``"multi_world"`` (per-world fast-tail; scales
                past ~256 worlds) or ``"single_world"`` (global JP
                colouring; wins for a few big worlds).
            threads_per_world: ``"auto"`` (default), 32, 16, or 8.
            multi_world_scheduler: Static multi-world solver scheduler.
                ``"auto"`` resolves once at construction from scene
                topology; ``"fast_tail"`` preserves the legacy path;
                ``"block_world"``/``"block_world_32"``/``"block_world_64"``
                /``"block_world_128"`` force one physical CUDA block per
                world. The resolved choice must be stable before graph
                capture.
            max_thread_blocks: Cap on the single-world PGS persistent
                grid; ``None`` auto-sizes. No effect on multi-world.
            mass_splitting: Enable Tonge mass splitting -- coloring caps
                at ``max_colored_partitions`` and the overflow bucket
                is solved Jacobi-style on per-partition copy states.
            max_colored_partitions: Regular-colour cap when mass
                splitting is on.
            mass_splitting_batch_size: Overflow batch size (B). Within
                a batch, constraints process sequentially (one thread);
                across batches, parallel. ``8`` matches C# PhoenX.
            partitioner_algorithm: ``"greedy"`` (default) uses
                :class:`IncrementalContactPartitioner`; ``"luby_fixed"``
                uses :class:`FixedIterationLubyPartitioner` (fixed
                launches, spill to overflow on saturation).
            device: Warp device. Defaults to ``bodies.position.device``.
        """
        if device is None:
            self.device = bodies.position.device
        else:
            self.device = wp.get_device(device)

        #: Opt-in per-column wall-clock profiler. When ``True``, PGS
        #: dispatches atomic-add their elapsed us into the column's
        #: ``time_us`` slot. Off by default to keep kernel cache stable.
        self.enable_column_timers: bool = bool(enable_column_timers)
        # Lazy 6-element scratch for per-type time_us reduction
        # (joints/cloth_tri/cloth_bend/soft_tet/contacts/soft_hex).
        # Slot ordering kept stable: contacts stays at index 4 so the
        # contact-side reduction kernel doesn't need rewiring.
        self._column_timer_totals: wp.array[wp.float32] | None = None

        self.bodies: BodyContainer = bodies
        self.constraints: ConstraintContainer = constraints

        self.num_bodies: int = int(bodies.position.shape[0])
        # Count kinematic bodies once so per-substep kinematic kernels
        # can short-circuit when no body is scripted. ``motion_type`` is
        # write-once at build time.
        if self.num_bodies > 0:
            mt = bodies.motion_type.numpy()
            self._num_kinematic_bodies: int = int((mt == int(MOTION_KINEMATIC)).sum())
        else:
            self._num_kinematic_bodies = 0
        self.rigid_contact_max: int = int(rigid_contact_max)
        if self.rigid_contact_max < 0:
            raise ValueError(f"rigid_contact_max must be >= 0 (got {self.rigid_contact_max})")
        if self.rigid_contact_max > 0:
            if max_contact_columns is None:
                self.max_contact_columns = max(1, self.rigid_contact_max)
            else:
                self.max_contact_columns = int(max_contact_columns)
                if self.max_contact_columns < 1:
                    raise ValueError(f"max_contact_columns must be >= 1 (got {self.max_contact_columns})")
        else:
            if max_contact_columns not in (None, 0):
                raise ValueError("max_contact_columns must be None or 0 when rigid_contact_max is 0")
            self.max_contact_columns = 0
        self.num_joints: int = int(num_joints)
        if self.num_joints < 0:
            raise ValueError(f"num_joints must be >= 0 (got {self.num_joints})")
        self.num_particles: int = int(num_particles)
        if self.num_particles < 0:
            raise ValueError(f"num_particles must be >= 0 (got {self.num_particles})")
        self.num_cloth_triangles: int = int(num_cloth_triangles)
        if self.num_cloth_triangles < 0:
            raise ValueError(f"num_cloth_triangles must be >= 0 (got {self.num_cloth_triangles})")
        self.num_cloth_bending: int = int(num_cloth_bending)
        if self.num_cloth_bending < 0:
            raise ValueError(f"num_cloth_bending must be >= 0 (got {self.num_cloth_bending})")
        self.num_soft_tetrahedra: int = int(num_soft_tetrahedra)
        if self.num_soft_tetrahedra < 0:
            raise ValueError(f"num_soft_tetrahedra must be >= 0 (got {self.num_soft_tetrahedra})")
        self.num_soft_hexahedra: int = int(num_soft_hexahedra)
        if self.num_soft_hexahedra < 0:
            raise ValueError(f"num_soft_hexahedra must be >= 0 (got {self.num_soft_hexahedra})")
        # Stamp the scene-wide ``has_position_level_writers`` flag on the
        # body container so :func:`body_set_access_mode` can warp-uniform
        # short-circuit in rigid-only scenes. Re-stamped by
        # :meth:`setup_cloth_collision_pipeline` and
        # :meth:`populate_soft_tetrahedra_from_model` whenever the cloth /
        # soft-tet counts change.
        self._refresh_has_position_level_writers()
        # Soft-tet constraint variant tracker. Stamped by
        # :meth:`populate_soft_tetrahedra_from_model`; dispatch uses it to
        # choose the ARAP or block Neo-Hookean soft-tet kernel statically.
        self._soft_tet_uses_neohookean: bool = False
        # Lazily allocate the particle store only when cloth is present;
        # rigid-only scenes pay zero memory for particles.
        self.particles: ParticleContainer | None = None
        if self.num_particles > 0:
            self.particles = particle_container_zeros(self.num_particles, device=self.device)
        # Scheduler-only particle world metadata. Keep it separate from
        # ParticleContainer so the hot particle SoA stays compact;
        # multi-world coloring uses it to resolve unified node ids
        # ``num_bodies + particle`` to a world.
        self._particle_world_id: wp.array[wp.int32] = wp.zeros(
            max(1, self.num_particles), dtype=wp.int32, device=self.device
        )
        # Length-1 sentinel :class:`ParticleContainer` for rigid-only
        # scenes -- kernels that take a particle parameter (cloth
        # element-emission, type-tag dispatch) need a non-empty SoA to
        # bind to even when the cloth branch is dead-eliminated.
        self._particle_sentinel: ParticleContainer | None = None
        # Cloth-aware collision pipeline (constructed by
        # :meth:`setup_cloth_collision_pipeline`). ``None`` means the
        # world is rigid-only or cloth-only-with-no-contacts; in either
        # case :meth:`collide` is a no-op.
        self._collision_pipeline = None
        self._cloth_shape_offset: int = 0
        self._cloth_gap: float = 0.0
        self._cloth_tri_indices = None
        # Per-shape endpoint table -- length S + T, populated alongside
        # the collision pipeline. Read by the contact-ingest kernel to
        # translate ``(shape_a, shape_b)`` into unified body-or-particle
        # nodes + kind tags for the contact column header.
        self._shape_endpoints: wp.array | None = None
        # Per-shape filter id for the contact-ingest same-body filter.
        # ``None`` means rigid-only behaviour (ingest falls back to
        # ``shape_body``); cloth-aware setups install a custom array
        # so distinct cloth tris don't collapse into one filter group.
        self._shape_filter_id: wp.array | None = None

        self.substeps = int(substeps)
        if self.substeps <= 0:
            raise ValueError(f"substeps must be >= 1 (got {self.substeps})")
        self.solver_iterations = int(solver_iterations)
        if self.solver_iterations < 1:
            raise ValueError(f"solver_iterations must be >= 1 (got {self.solver_iterations})")
        self.velocity_iterations = int(velocity_iterations)
        if self.velocity_iterations < 0:
            raise ValueError(f"velocity_iterations must be >= 0 (got {self.velocity_iterations})")
        sleeping_requested = float(sleeping_velocity_threshold) > 0.0
        cached_prepare_unsupported = (
            bool(mass_splitting)
            or sleeping_requested
            or self.num_particles > 0
            or self.num_cloth_triangles > 0
            or self.num_cloth_bending > 0
            or self.num_soft_tetrahedra > 0
            or self.num_soft_hexahedra > 0
        )
        contact_capacity_hint = int(max_contact_columns if max_contact_columns is not None else rigid_contact_max)
        self._prepare_refresh_stride_policy: str = "fixed"
        if isinstance(prepare_refresh_stride, str):
            if prepare_refresh_stride != "auto":
                raise ValueError(
                    f"prepare_refresh_stride must be an integer >= 1 or 'auto' (got {prepare_refresh_stride!r})"
                )
            self._prepare_refresh_stride_policy = "auto"
            self.prepare_refresh_stride = _choose_auto_prepare_refresh_stride(
                substeps=self.substeps,
                contact_capacity_hint=contact_capacity_hint,
                cached_prepare_unsupported=cached_prepare_unsupported,
            )
        else:
            self.prepare_refresh_stride = int(prepare_refresh_stride)
        if self.prepare_refresh_stride < 1:
            raise ValueError(f"prepare_refresh_stride must be >= 1 (got {self.prepare_refresh_stride})")
        if self.prepare_refresh_stride > 3 and contact_capacity_hint > 0:
            raise NotImplementedError(
                "prepare_refresh_stride > 3 currently supports joint-only rigid worlds; "
                "contact worlds should refresh at least every third substep"
            )
        self._multi_world_scheduler_policy: str = str(multi_world_scheduler)
        self._multi_world_scheduler: str = "fast_tail"
        self._multi_world_block_dim: int = 128

        # SOR boost (successive over-relaxation): every iterate
        # multiplies its computed delta lambda by this factor before
        # clamp/apply. 1.0 = vanilla PGS. Values in (1.0, 2.0) trade
        # per-iteration aggression for fewer iters to converge on
        # smooth modes; >= 2.0 diverges on most rigid-body PGS
        # problems. Applies to every constraint type (joints,
        # contacts, cloth, soft tet) via the kernel-level sor_boost
        # arg threaded into each iterate.
        self.sor_boost: float = float(sor_boost)
        if not (0.1 <= self.sor_boost <= 2.0):
            raise ValueError(f"sor_boost must be in [0.1, 2.0] (got {self.sor_boost})")

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
        self.articulation_dvi_host: bool = bool(articulation_dvi_host)
        if articulation_dvi_replaces_joint_pgs is None:
            articulation_dvi_replaces_joint_pgs = self.articulation_dvi_host
        self.articulation_dvi_replaces_joint_pgs: bool = bool(articulation_dvi_replaces_joint_pgs)
        if self.articulation_dvi_replaces_joint_pgs and not self.articulation_dvi_host:
            raise ValueError("articulation_dvi_replaces_joint_pgs requires articulation_dvi_host=True")
        self.articulation_dvi_host_solver: str = self._normalize_articulation_dvi_host_solver(
            articulation_dvi_host_solver
        )
        self.cache_articulation_topology: bool = bool(cache_articulation_topology)
        # Topology-only full-coordinate articulation system. Built at joint
        # initialization and reused by the DVI articulation path as it is
        # brought online.
        self.articulation_topology: ArticulationTopology | None = None
        self.articulation_system: PrefactorizedArticulationSystem | None = None
        self.articulation_device_system: ArticulationDeviceSystem | None = None
        self.articulation_dvi_joint_mask: np.ndarray | None = None
        self._joint_pgs_enabled: wp.array[wp.int32] = wp.ones(
            max(1, self.num_joints), dtype=wp.int32, device=self.device
        )

        self.num_worlds: int = int(num_worlds)
        if self.num_worlds <= 0:
            raise ValueError(f"num_worlds must be >= 1 (got {self.num_worlds})")
        if step_layout not in ("multi_world", "single_world"):
            raise ValueError(f"step_layout must be 'multi_world' or 'single_world' (got {step_layout!r})")
        self.step_layout: str = step_layout
        if self.prepare_refresh_stride != 1 and cached_prepare_unsupported:
            raise NotImplementedError(
                "prepare_refresh_stride > 1 currently supports rigid contact/joint worlds "
                "without deformables, mass splitting, or sleeping"
            )
        self._current_substep_index: int = 0
        # Threads-per-world. ``"auto"`` lets the GPU picker decide every
        # step from colour stats; an int forces a fixed value (validated
        # against the {8, 16, 32} set the kernels have been tuned for).
        # ``_tpw_choice`` is a 1-element GPU buffer that the fast-tail
        # kernels read at the top of every launch -- see
        # :func:`_pick_threads_per_world_kernel` for the heuristic.
        if isinstance(threads_per_world, str):
            if threads_per_world != "auto":
                raise ValueError(f"threads_per_world must be 'auto' or one of (8, 16, 32) (got {threads_per_world!r})")
            # Host-side fast path: pin obvious topology/occupancy cases so
            # graph capture does not need guarded no-op fast-tail variants.
            # Small fleets still prefer 32; saturated simple robot fleets can
            # profit from 16 or 8 lanes per world depending on topology.
            self._tpw_auto, initial_tpw = _choose_initial_threads_per_world(
                num_worlds=self.num_worlds,
                num_joints=self.num_joints,
                max_contact_columns=self.max_contact_columns,
                sm_count=getattr(self.device, "sm_count", 0) or 1,
            )
        else:
            tpw_int = int(threads_per_world)
            if tpw_int not in (8, 16, 32):
                raise ValueError(f"threads_per_world must be 'auto' or one of (8, 16, 32) (got {tpw_int})")
            self._tpw_auto = False
            initial_tpw = tpw_int
        self._tpw_launch_bound: int = _STRAGGLER_BLOCK_DIM if self._tpw_auto else int(initial_tpw)
        self._tpw_choice: wp.array[wp.int32] = wp.array([initial_tpw], dtype=wp.int32, device=self.device)
        # Scratch for the picker's parallel colour-count reduction.
        # Reset to 0 each step before the reduction kernel runs.
        self._tpw_total_colours: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.gravity: wp.array[wp.vec3f] = _build_gravity_array(gravity, self.num_worlds, self.device)
        self.default_friction = float(default_friction)

        # Global per-substep damping; lazy-alloc on opt-in. Mid-simulation
        # opt-in requires re-capturing any existing CUDA graph.
        self._global_damping: wp.array[wp.float32] | None = None
        self._global_damping_host: np.ndarray | None = None

        # ----- Step time bookkeeping -----
        self.step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # cid layout in the shared :class:`ConstraintContainer`:
        #   [0, num_joints):                                                                              joints
        #   [num_joints, +num_cloth_triangles):                                                           cloth tris
        #   [+num_cloth_triangles, +num_cloth_bending):                                                   cloth bending
        #   [+num_cloth_bending, +num_soft_tetrahedra):                                                   soft tets
        #   [+num_soft_tetrahedra, +num_soft_hexahedra):                                                  soft hexes
        #   [+num_soft_hexahedra, ...):                                                                   contacts
        self._cloth_bending_offset: int = self.num_joints + self.num_cloth_triangles
        self._soft_tet_offset: int = self._cloth_bending_offset + self.num_cloth_bending
        self._soft_hex_offset: int = self._soft_tet_offset + self.num_soft_tetrahedra
        self._contact_offset: int = self._soft_hex_offset + self.num_soft_hexahedra
        self._constraint_capacity: int = max(1, self._contact_offset + self.max_contact_columns)

        # Persistent grid fixed at construction (graph-capture stability).
        if max_thread_blocks is not None and int(max_thread_blocks) < 1:
            raise ValueError(f"max_thread_blocks must be >= 1 (got {max_thread_blocks})")
        self._max_thread_blocks: int | None = int(max_thread_blocks) if max_thread_blocks is not None else None
        self._singleworld_total_threads: int = _singleworld_total_threads(
            self._constraint_capacity,
            self.device,
            max_thread_blocks=self._max_thread_blocks,
        )

        # Head capture-while predicate. Persistent-grid sweep zeroes it on
        # hand-off to the fused tail kernel.
        self._head_active: wp.array[wp.int32] = wp.ones(1, dtype=wp.int32, device=self.device)
        self._fuse_threshold: int = int(FUSE_TAIL_MAX_COLOR_SIZE)
        self._fuse_tail_block_dim: int = int(FUSE_TAIL_BLOCK_DIM)

        # ----- Partitioner + per-world CSR buffers -----
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            self._constraint_capacity, dtype=ElementInteractionData, device=self.device
        )
        self._element_family: wp.array[wp.int32] = wp.zeros(
            self._constraint_capacity, dtype=wp.int32, device=self.device
        )
        # Joints + cloth tris + cloth bending + soft tets are the
        # only active cids until the first contact ingest. Bending +
        # tets are populated by their respective ``populate_*`` methods.
        self._num_active_constraints: wp.array[int] = wp.array(
            [self._contact_offset],
            dtype=wp.int32,
            device=self.device,
        )
        # Mass splitting config. When ``True``, the partitioner caps at
        # ``max_colored_partitions`` colours and the overflow bucket is
        # solved with copy states (C# PhoenX behaviour). The
        # :class:`IncrementalContactPartitioner` ctor validates the cap
        # against ``GREEDY_MAX_COLORS`` / ``MAX_COLORS``.
        self.mass_splitting_enabled: bool = bool(mass_splitting)
        if self.mass_splitting_enabled:
            # The multi-world fast-tail kernels run all colours at
            # ``parallel_id=0`` (they don't track overflow), so mass
            # splitting still requires the single-world step layout.
            # Joint and cloth-triangle constraints now route through the
            # slot-aware helpers, so they're free to coexist with mass
            # splitting in single-world mode.
            if step_layout != "single_world":
                raise NotImplementedError(
                    "mass_splitting=True currently requires step_layout='single_world' "
                    "(multi-world fast-tail kernels haven't been refactored yet)."
                )
        self.max_colored_partitions: int | None = int(max_colored_partitions) if self.mass_splitting_enabled else None
        if self.mass_splitting_enabled and int(mass_splitting_batch_size) < 1:
            raise ValueError(f"mass_splitting_batch_size must be >= 1 (got {mass_splitting_batch_size})")
        self.mass_splitting_batch_size: int = int(mass_splitting_batch_size) if self.mass_splitting_enabled else 1
        # Opt-in: drop both ``wp.capture_while`` loops on the mass-
        # splitting PGS hot path in favour of a host-side fixed
        # ``max_colored_partitions + 1`` unroll. Trade-off discussion in
        # :mod:`dispatch.single_world_mass_splitting_unrolled`.
        self.mass_splitting_unrolled: bool = bool(mass_splitting_unrolled) and self.mass_splitting_enabled
        self._configure_multi_world_scheduler(self._multi_world_scheduler_policy)
        # Unified body-or-particle node space for the partitioner:
        # ``[0, num_bodies)`` are rigid bodies; ``[num_bodies,
        # num_bodies + num_particles)`` are particles.
        self.partitioner_algorithm: str = str(partitioner_algorithm)
        # The cache-stir defaults are needed for contact-heavy dynamic
        # rigid stacks, where a perfectly locked colouring can bias the
        # PGS fixed point over many frames. Deformable-only worlds pay
        # the recolouring cost but do not have that rigid-stack drift
        # mode, so let them reuse the cached colouring unless a caller
        # explicitly opts into stirring. Mixed rigid/deformable scenes
        # keep the rigid-stack defaults.
        _has_dynamic_rigid_rows = self.num_joints > 0
        if self.num_bodies > 0:
            motion_type = bodies.motion_type.numpy()
            inverse_mass = bodies.inverse_mass.numpy()
            _has_dynamic_rigid_rows = _has_dynamic_rigid_rows or bool(
                np.any((motion_type == int(MOTION_DYNAMIC)) & (inverse_mass > 0.0))
            )
        # The grouped average pays tile collective work for every node,
        # including single-slot rigid nodes. Use it only for particle-
        # dominant mass-splitting worlds where the cooperative slot
        # broadcast wins; keep rigid-heavy scenes on the scalar path.
        self._mass_splitting_grouped_average = bool(
            self.mass_splitting_enabled and self.device.is_cuda and self.num_particles > self.num_bodies
        )
        # Rigid contacts and joints are velocity-level constraints. Mixed
        # deformable worlds can contain position-level cloth/soft rows, so
        # they must keep the generic access-mode synchronization path.
        self._mass_splitting_velocity_only_average = bool(
            self.mass_splitting_enabled
            and self.num_particles == 0
            and self.num_cloth_triangles == 0
            and self.num_cloth_bending == 0
            and self.num_soft_tetrahedra == 0
            and self.num_soft_hexahedra == 0
        )
        if warm_start_invalidate_period is None:
            warm_start_invalidate_period = 4 if _has_dynamic_rigid_rows else 0
        if warm_start_rotate_skip_color is None:
            warm_start_rotate_skip_color = _has_dynamic_rigid_rows

        # Warm-start coloring only feeds the single-world greedy build;
        # the per-world multi-world path uses a different kernel that
        # never reads the cache. Skip the allocation in that case.
        _warm_start_active: bool = bool(enable_warm_start_coloring) and step_layout == "single_world"
        if self.partitioner_algorithm == "greedy":
            self._partitioner = IncrementalContactPartitioner(
                max_num_interactions=self._constraint_capacity,
                max_num_nodes=max(1, self.num_bodies + self.num_particles),
                device=self.device,
                use_tile_scan=True,
                max_colored_partitions=self.max_colored_partitions,
                max_greedy_outer_iters=max_greedy_outer_iters,
                enable_warm_start=_warm_start_active,
            )
            self._partitioner.set_locality_family(self._element_family)
            self._partitioner.set_symmetric_sweep(bool(symmetric_color_sweep))
            self._partitioner.set_warm_start_invalidate_period(int(warm_start_invalidate_period))
            self._partitioner.set_warm_start_rotate_skip(
                bool(warm_start_rotate_skip_color),
                width=int(warm_start_rotate_skip_width),
            )
            self._partitioner.set_capture_while_greedy(bool(capture_while_greedy_coloring))
            self._partitioner.set_speculative_coloring(bool(speculative_coloring))
        elif self.partitioner_algorithm == "luby_fixed":
            if step_layout != "single_world":
                # Multi-world path reads ``_adjacency_section_end_indices``
                # etc. straight from the partitioner because
                # :class:`IncrementalContactPartitioner.reset` populates
                # them eagerly. The Luby variant builds adjacency lazily
                # in :meth:`build_csr` and doesn't expose a per-world
                # coloring kernel yet.
                raise NotImplementedError(
                    "partitioner_algorithm='luby_fixed' currently requires step_layout='single_world'."
                )
            self._partitioner = FixedIterationLubyPartitioner(
                max_num_interactions=self._constraint_capacity,
                max_num_nodes=max(1, self.num_bodies + self.num_particles),
                device=self.device,
                max_colored_partitions=self.max_colored_partitions,
            )
        else:
            raise ValueError(
                f"Unknown partitioner_algorithm '{self.partitioner_algorithm}'. "
                "Expected one of: 'greedy', 'luby_fixed'."
            )

        # Mass-splitting data plane. Always allocated (sentinel-sized
        # when disabled) so the constraint kernels can take
        # :class:`CopyStateContainer` as a parameter unconditionally.
        # When ``mass_splitting_enabled`` is False the broadcast /
        # average / writeback kernels short-circuit on
        # ``highest_index_in_use[0] == 0``.
        if self.mass_splitting_enabled:
            # Worst-case emitted ``(node, partition)`` pairs for this
            # scene. Rigid-only worlds only need two endpoints per
            # contact/joint; deformable contact worlds keep the full
            # ``MAX_BODIES`` bound because soft-tet/cloth contacts can
            # touch up to eight nodes. This directly sizes the radix-sort
            # scratch used by the mass-splitting interaction graph.
            ms_capacity = _mass_splitting_copy_capacity(
                num_joints=self.num_joints,
                num_cloth_triangles=self.num_cloth_triangles,
                num_cloth_bending=self.num_cloth_bending,
                num_soft_tetrahedra=self.num_soft_tetrahedra,
                num_soft_hexahedra=self.num_soft_hexahedra,
                num_particles=self.num_particles,
                max_contact_columns=self.max_contact_columns,
            )
            ms_nodes = max(1, self.num_bodies + self.num_particles)
        else:
            # Sentinel containers — kernels see highest_index_in_use==0
            # and short-circuit. Memory cost is negligible.
            ms_capacity = 1
            ms_nodes = 1
        self._copy_state: CopyStateContainer = copy_state_container_zeros(
            capacity=ms_capacity, num_nodes=ms_nodes, device=self.device
        )
        self._interaction_graph_scratch: InteractionGraphScratch = interaction_graph_scratch_zeros(
            capacity=ms_capacity, device=self.device
        )
        # Live single-world coloring choice. Flipped to False (round-based JP)
        # if a non-captured greedy build overflows the 64-color bitmask; never
        # flipped back (JP has no chromatic bound).
        self._use_greedy_coloring: bool = bool(PHOENX_USE_GREEDY_COLORING)

        cap = self._constraint_capacity
        nw = self.num_worlds
        self._world_element_ids_by_color: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._world_color_starts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_color_family_starts: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS) * int(_PER_WORLD_FAST_FAMILIES)),
            dtype=wp.int32,
            device=self.device,
        )
        self._world_csr_offsets: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        # Sized nw+1 so the inclusive scan output lands in world_csr_offsets.
        self._world_totals_shifted: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        self._world_num_colors: wp.array[wp.int32] = wp.zeros(nw, dtype=wp.int32, device=self.device)

        # Per-world JP scratch. _per_world_elements / _per_world_scatter_keys
        # are 2*cap (radix-sort ping-pong); _per_world_assigned is 1-based
        # colour (0 = unassigned).
        self._per_world_element_count: wp.array[wp.int32] = wp.zeros(nw, dtype=wp.int32, device=self.device)
        self._per_world_element_offsets: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        self._per_world_elements: wp.array[wp.int32] = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        self._per_world_scatter_keys: wp.array[wp.int32] = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        self._per_world_assigned: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self.device)
        # Greedy per-world scratch. overflow flag is surfaced via step_report
        # but not raised mid-step (fallback is to flip _use_greedy_coloring off).
        self._per_world_greedy_color_count: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS)), dtype=wp.int32, device=self.device
        )
        self._per_world_greedy_color_offsets: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS)), dtype=wp.int32, device=self.device
        )
        self._per_world_greedy_color_family_count: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS) * int(_PER_WORLD_FAST_FAMILIES)),
            dtype=wp.int32,
            device=self.device,
        )
        self._per_world_greedy_color_family_offsets: wp.array2d[wp.int32] = wp.zeros(
            (nw, int(GREEDY_MAX_COLORS) * int(_PER_WORLD_FAST_FAMILIES)),
            dtype=wp.int32,
            device=self.device,
        )
        self._per_world_greedy_overflow: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Contact state split across two narrow containers:
        # ContactContainer: keyed by contact index k (warm-start, prev lambdas).
        # ContactColumnContainer: keyed by local cid (column header).
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
            self._cid_of_contact_cur = wp.full(self.rigid_contact_max, -1, dtype=wp.int32, device=self.device)
            self._cid_of_contact_prev = wp.full(self.rigid_contact_max, -1, dtype=wp.int32, device=self.device)
            self._last_contact_generation = wp.full(1, -1, dtype=wp.int32, device=self.device)
            self._reuse_contact_indices = wp.zeros(1, dtype=wp.int32, device=self.device)
        else:
            self._contact_container = contact_container_zeros(1, device=self.device)
            self._contact_cols = contact_column_container_zeros(1, device=self.device)
            self._ingest_scratch = None
            self._cid_of_contact_cur = None
            self._cid_of_contact_prev = None
            self._last_contact_generation = wp.full(1, -1, dtype=wp.int32, device=self.device)
            self._reuse_contact_indices = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._enable_body_pair_grouping = False

        self._contact_views: ContactViews | None = None
        self._has_soft_contact_pd: bool = False
        self._contact_views_placeholder: ContactViews = self._make_placeholder_contact_views()

        # ----- Pairwise contact filter (packed int64 keys) -----
        self._collision_filter_keys: wp.array[wp.int64]
        self._set_collision_filter_pairs_impl(collision_filter_pairs or ())

        # ----- Optional material table -----
        self._shape_material: wp.array[wp.int32] | None = None
        self._materials: wp.array[MaterialData] | None = None
        # Optional shape_body installed via set_shape_body (used when callers
        # don't pass shape_body to step()).
        self._shape_body_internal: wp.array[wp.int32] | None = None
        # Lazy sentinel for optional per-contact stiffness/damping/friction.
        self._soft_contact_sentinel: wp.array[wp.float32] | None = None

        # Reference to the share-vertex / sleeping filter data, kept alive
        # so the Warp ABI sees a stable wp.struct binding across steps.
        self._share_vertex_filter_data: PhoenXClothShareVertexFilterData | None = None

        # Sleeping pipeline. Activated by ``sleeping_velocity_threshold > 0``;
        # zero leaves every helper at ``None`` so the per-step hot path skips
        # the entire block (no kernel launches, no scratch resets).
        sleeping_threshold = float(sleeping_velocity_threshold)
        if sleeping_threshold < 0.0:
            raise ValueError(f"sleeping_velocity_threshold must be >= 0 (got {sleeping_threshold})")
        sleeping_frames = int(sleeping_frames_required)
        if sleeping_frames < 0:
            raise ValueError(f"sleeping_frames_required must be >= 0 (got {sleeping_frames})")
        self._sleeping_velocity_threshold: float = sleeping_threshold
        self._sleeping_frames_required: int = sleeping_frames
        self._sleeping_enabled: bool = sleeping_threshold > 0.0
        self._island_builder: UnionFindIslandBuilder | None = None
        self._island_interaction_bodies: wp.array2d[wp.int32] | None = None
        self._island_max_velocity: wp.array[wp.float32] | None = None
        self._island_is_sleeping: wp.array[wp.int32] | None = None
        # Per-root flag arrays, indexed by body id (the persistent
        # ``island_root`` label). Both are zeroed at the start of each
        # pass and raised by per-body atomic_max.
        #   * _island_has_external_input -- raised by sleeping bodies
        #     carrying a user wrench, consumed pre-collide by
        #     :meth:`wake_on_external_input`.
        #   * _wake_flag -- raised in-step by the self-wake kernel for
        #     sleeping bodies whose own predicted score exceeds the
        #     sleep threshold (host-side velocity / force injection).
        self._island_has_external_input: wp.array[wp.int32] | None = None
        self._wake_flag: wp.array[wp.int32] | None = None
        # Per-compact-island scratch populated by
        # ``_phoenx_compute_island_root_per_compact_kernel`` (atomic_min
        # of awake body ids per compact id). The propagate kernel reads
        # this slot when a body's hysteresis counter saturates so the
        # body can stamp ``island_root`` with a stable body-id label.
        self._island_root_per_compact: wp.array[wp.int32] | None = None
        # Per-root active flag, indexed by body id. A sleeping island
        # is "active" this step iff at least one member shares a
        # constraint with an awake dynamic body. Active islands get
        # chain edges injected into the union-find input so the awake
        # body bridging them pulls the entire sleeping group into one
        # compact island -- propagate then wakes everyone atomically.
        # Inactive sleeping islands stay filtered for free.
        self._island_active: wp.array[wp.int32] | None = None
        # Separate union-find interaction counter so chain-edge
        # injection can ``atomic_add`` past ``_num_active_constraints``
        # without affecting other consumers (collapse kernel, partitioner,
        # narrow phase). Seeded from ``_num_active_constraints[0]`` at
        # the top of each sleeping pass.
        self._uf_num_interactions: wp.array[wp.int32] | None = None
        # Per-rigid-body bounding box, unioned across every attached
        # shape's world-frame AABB. Used by the sleeping pipeline to
        # estimate a body's swept volume so the spin term in the
        # velocity score (0.5 * diagonal * |omega|) accounts for
        # off-COM shapes.
        self._body_aabb_lower: wp.array2d[wp.float32] | None = None
        self._body_aabb_upper: wp.array2d[wp.float32] | None = None
        self._body_aabb_diagonal: wp.array[wp.float32] | None = None
        self._sleeping_num_bodies_device: wp.array[wp.int32] | None = None
        if self._sleeping_enabled:
            if self.num_bodies <= 0:
                raise ValueError("sleeping_velocity_threshold > 0 requires num_bodies > 0")
            self._island_builder = UnionFindIslandBuilder(
                num_bodies_capacity=self.num_bodies,
                device=self.device,
            )
            # Hold space for the regular per-element rows PLUS up to
            # ``num_bodies`` chain edges injected by the sleeping pass.
            self._island_interaction_bodies = wp.full(
                shape=(self._constraint_capacity + self.num_bodies, int(MAX_BODIES_PER_INTERACTION)),
                value=-1,
                dtype=wp.int32,
                device=self.device,
            )
            self._island_max_velocity = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
            self._island_is_sleeping = wp.zeros(self.num_bodies, dtype=wp.int32, device=self.device)
            self._island_has_external_input = wp.zeros(self.num_bodies, dtype=wp.int32, device=self.device)
            self._wake_flag = wp.zeros(self.num_bodies, dtype=wp.int32, device=self.device)
            self._island_active = wp.zeros(self.num_bodies, dtype=wp.int32, device=self.device)
            self._uf_num_interactions = wp.zeros(1, dtype=wp.int32, device=self.device)
            # Per-compact-island scratch: lowest awake body id per
            # compact id. Reset to num_bodies (a value past any real
            # body id) at the start of the sleeping pass so atomic_min
            # converges to the true minimum.
            self._island_root_per_compact = wp.full(
                self.num_bodies, value=self.num_bodies, dtype=wp.int32, device=self.device
            )
            self._body_aabb_lower = wp.zeros((self.num_bodies, 3), dtype=wp.float32, device=self.device)
            self._body_aabb_upper = wp.zeros((self.num_bodies, 3), dtype=wp.float32, device=self.device)
            self._body_aabb_diagonal = wp.zeros(self.num_bodies, dtype=wp.float32, device=self.device)
            self._sleeping_num_bodies_device = wp.array([self.num_bodies], dtype=wp.int32, device=self.device)

        # Cached references installed by :meth:`attach_collision_pipeline`.
        # When non-None, :meth:`step` falls back to them for the optional
        # ``shape_aabb_lower`` / ``shape_aabb_upper`` arguments (read only
        # when sleeping is enabled) so user code doesn't have to thread
        # the pipeline's narrow-phase AABBs through every step call.
        self._sleeping_shape_aabb_lower: wp.array[wp.vec3f] | None = None
        self._sleeping_shape_aabb_upper: wp.array[wp.vec3f] | None = None

        # Step-time dispatcher. Each (step_layout, mass_splitting)
        # combination has a dedicated class under :mod:`phoenx.dispatch`
        # so the hot path is straight-line with no capability checks.
        if self.step_layout == "single_world":
            if self.mass_splitting_enabled:
                if self.mass_splitting_unrolled:
                    self._dispatcher = SingleWorldMassSplittingUnrolledDispatcher(self)
                else:
                    self._dispatcher = SingleWorldMassSplittingDispatcher(self)
            else:
                self._dispatcher = SingleWorldDispatcher(self)
        else:
            # mass_splitting + multi_world is rejected by the earlier
            # validation in this ctor.
            self._dispatcher = MultiWorldDispatcher(self)

        self._assert_invariants()

        self._pre_compile_dispatch_kernels()

    def _fast_tail_fixed_tpw(self) -> int:
        """Static kernel axis for fixed-tpw launches; 0 keeps dynamic lookup."""
        return 0 if self._tpw_auto else int(self._tpw_launch_bound)

    def _fast_tail_auto_fixed_choices(self) -> tuple[int, ...]:
        """Static fast-tail variants captured for dynamic auto mode."""
        return (16, 32) if self._tpw_auto and self.step_layout != "single_world" else (self._fast_tail_fixed_tpw(),)

    def _pre_compile_dispatch_kernels(self) -> None:
        """Eagerly instantiate the factory-built PGS kernels for the current
        scene spec and parallel-compile their warp modules.

        Each ``module="unique"`` PGS dispatcher kernel (single-world
        head/fused prepare + iterate + relax, multi-world fast-tail
        prepare+iter + relax) is its own warp module and inlines the
        scene-specialised ``_dispatch_one_cid`` helper. On a cold cache
        each one costs ~10-15 s of NVRTC; loaded lazily on the first
        ``step()`` they serialise on a single Python thread.

        When ``wp.config.load_module_max_workers`` is set above 1 we
        instantiate every variant the active step layout will launch and
        hand the modules to :func:`wp.force_load`, which then drives
        codegen + NVRTC across a thread pool. The warp-side codegen
        phase is serialised on a module-level lock (see
        :func:`warp._src.context._codegen_lock`) so per-Adjoint state
        stays consistent; the heavy NVRTC step runs outside the lock.

        No-op when ``load_module_max_workers`` is unset, 0, or 1 — the
        kernels stay lazy and behaviour is bit-identical to the prior
        release.
        """
        max_workers = wp.config.load_module_max_workers
        if max_workers is None or max_workers <= 1:
            return

        kernels: list = []
        if self.step_layout == "single_world":
            # Six head/fused prepare + iterate + relax variants for the
            # current cloth / soft-tet / joint specialisation.
            kernels.extend(self._singleworld_kernels())
            if self.prepare_refresh_stride > 1:
                kernels.extend(self._singleworld_cached_prepare_kernels())
        else:
            include_cached_prepare = self.prepare_refresh_stride > 1
            if self._multi_world_scheduler == "block_world" and self._block_world_supported():
                prepare_kw = self._block_world_kernel_flags(self._multi_world_block_dim, cached_prepare=False)
                kernels.append(get_block_world_kernel(kind="prepare_plus_iterate", **prepare_kw))
                if include_cached_prepare:
                    cached_kw = self._block_world_kernel_flags(self._multi_world_block_dim, cached_prepare=True)
                    kernels.append(get_block_world_kernel(kind="prepare_plus_iterate", **cached_kw))
                relax_kw = self._block_world_kernel_flags(self._multi_world_block_dim)
                kernels.append(get_block_world_kernel(kind="relax", **relax_kw))
            else:
                for fixed_tpw in self._fast_tail_auto_fixed_choices():
                    prepare_kw = self._fast_tail_kernel_flags(fixed_tpw, cached_prepare=False)
                    kernels.append(get_fast_tail_kernel(kind="prepare_plus_iterate", **prepare_kw))
                    if include_cached_prepare:
                        cached_kw = self._fast_tail_kernel_flags(fixed_tpw, cached_prepare=True)
                        kernels.append(get_fast_tail_kernel(kind="prepare_plus_iterate", **cached_kw))
                    relax_kw = self._fast_tail_kernel_flags(fixed_tpw)
                    kernels.append(get_fast_tail_kernel(kind="relax", **relax_kw))

        # De-duplicate by module; ``functools.cache`` already collapses
        # identical (axes-tuple) factory calls but cheap to be defensive.
        modules = list({kernel.module for kernel in kernels if kernel is not None})
        if not modules:
            return

        wp.force_load(device=self.device, modules=modules, max_workers=max_workers)

    def _assert_invariants(self) -> None:
        """Validate per-step buffer shapes against the documented schema."""
        expected_constraint_dwords = self.required_constraint_dwords(
            self.num_joints,
            self.num_cloth_triangles,
            self.num_soft_tetrahedra,
            self.num_cloth_bending,
            self.num_soft_hexahedra,
        )
        expected_constraint_cols = max(
            1,
            int(self.num_joints)
            + int(self.num_cloth_triangles)
            + int(self.num_cloth_bending)
            + int(self.num_soft_tetrahedra)
            + int(self.num_soft_hexahedra),
        )
        actual_constraint_shape = self.constraints.data.shape
        assert actual_constraint_shape == (expected_constraint_dwords, expected_constraint_cols), (
            f"ConstraintContainer.data has shape {actual_constraint_shape}, expected "
            f"({expected_constraint_dwords}, {expected_constraint_cols}); use "
            f"PhoenXWorld.make_constraint_container() to build it"
        )
        actual_multiplier_shape = self.constraints.multipliers.shape
        assert actual_multiplier_shape == (CONSTRAINT_MULTIPLIER_DWORDS, expected_constraint_cols), (
            f"ConstraintContainer.multipliers has shape {actual_multiplier_shape}, expected "
            f"({CONSTRAINT_MULTIPLIER_DWORDS}, {expected_constraint_cols}); use "
            f"PhoenXWorld.make_constraint_container() to build it"
        )

        expected_col_cols = max(1, int(self.max_contact_columns))
        actual_col_shape = self._contact_cols.data.shape
        assert actual_col_shape == (CONTACT_DWORDS, expected_col_cols), (
            f"ContactColumnContainer.data has shape {actual_col_shape}, "
            f"expected ({CONTACT_DWORDS}, {expected_col_cols})"
        )

        expected_cc_cols = max(1, int(self.rigid_contact_max))
        for name, expected_rows in (
            ("impulses", CC_IMPULSE_DWORDS_PER_CONTACT),
            ("prev_impulses", CC_IMPULSE_DWORDS_PER_CONTACT),
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

    # Material system / collision filters / placeholder contact views.

    def set_materials(
        self,
        materials: wp.array | None,
        shape_material: wp.array | None,
    ) -> None:
        """Install per-shape friction materials."""
        self._materials = materials
        self._shape_material = shape_material

    def set_shape_body(self, shape_body: wp.array | None) -> None:
        """Install the shape->body map used by contact ingest. ``None`` clears."""
        self._shape_body_internal = shape_body

    @staticmethod
    def broad_phase_filter() -> tuple:
        """Return the ``(filter_func, filter_data_type)`` tuple to pass
        as ``broad_phase_filter`` when constructing the Newton
        :class:`~newton.CollisionPipeline` that feeds this world.

        Required only when sleeping is enabled or the scene contains
        cloth / soft-body deformables (the same filter handles both
        rigid-frozen-pair culling and deformable share-vertex skip)::

            cp = newton.CollisionPipeline(
                model,
                contact_matching="sticky",
                broad_phase_filter=PhoenXWorld.broad_phase_filter(),
            )

        Pair with :meth:`attach_collision_pipeline` after constructing
        the world; that method installs the per-step filter data and
        caches the pipeline's narrow-phase AABB arrays so
        :meth:`step` doesn't need them as args.
        """
        return (phoenx_cloth_share_vertex_filter, PhoenXClothShareVertexFilterData)

    def attach_collision_pipeline(
        self,
        collision_pipeline,
        *,
        num_rigid_shapes: int,
        shape_body: wp.array[wp.int32],
        phoenx_body_offset: int = 1,
    ) -> None:
        """Wire a rigid-only Newton :class:`~newton.CollisionPipeline`
        into PhoenX's sleeping pipeline.

        Builds the share-vertex filter data with sleeping fields
        populated, installs it on the pipeline, and caches the
        pipeline's per-shape AABB arrays + a PhoenX-offset shape_body
        map. After this call, the per-frame loop is the same regardless
        of whether sleeping is enabled::

            world.wake_on_external_input()  # no-op when disabled
            model.collide(state, contacts=contacts, collision_pipeline=cp)
            world.step(dt=dt, contacts=contacts)  # no shape_aabb args

        Pre-requisites:
          * ``collision_pipeline`` was constructed with
            ``broad_phase_filter=PhoenXWorld.broad_phase_filter()``.
          * Scene is rigid-only -- cloth / soft-tet scenes use
            :meth:`setup_cloth_collision_pipeline` which performs the
            equivalent wiring and adds the deformable suffix.

        Args:
            collision_pipeline: The Newton CollisionPipeline.
            num_rigid_shapes: Total rigid shape count (= ``model.shape_count``).
            shape_body: Raw ``model.shape_body`` (Newton-index per shape;
                ``-1`` for world-anchored). PhoenX shifts each entry by
                ``+phoenx_body_offset`` internally for the contact ingest
                path; the filter sees the raw Newton array.
            phoenx_body_offset: Newton-id -> PhoenX-slot offset (1 for
                the typical "slot 0 = world anchor" convention).
        """
        if self.num_cloth_triangles > 0 or self.num_soft_tetrahedra > 0 or self.num_soft_hexahedra > 0:
            raise RuntimeError(
                "attach_collision_pipeline is rigid-only; cloth / soft-tet / soft-hex scenes "
                "use setup_cloth_collision_pipeline()"
            )

        import numpy as _np  # noqa: PLC0415

        shape_body_np = shape_body.numpy() if isinstance(shape_body, wp.array) else _np.asarray(shape_body)
        shape_body_phx = _np.where(shape_body_np < 0, 0, shape_body_np + int(phoenx_body_offset))
        shape_body_phx_arr = wp.array(shape_body_phx.astype(_np.int32), dtype=wp.int32, device=self.device)
        self.set_shape_body(shape_body_phx_arr)

        tri_sentinel = wp.zeros((1, 3), dtype=wp.int32, device=self.device)
        tet_sentinel = wp.zeros((1, 4), dtype=wp.int32, device=self.device)
        filter_data = build_phoenx_share_vertex_filter_data(
            num_rigid_shapes=int(num_rigid_shapes),
            num_cloth_triangles=0,
            tri_indices=tri_sentinel,
            tet_indices=tet_sentinel,
            sleeping_enabled=self._sleeping_enabled,
            phoenx_body_offset=int(phoenx_body_offset),
            shape_body=shape_body if self._sleeping_enabled else None,
            body_island_root=self.bodies.island_root if self._sleeping_enabled else None,
            body_motion_type=self.bodies.motion_type if self._sleeping_enabled else None,
            device=self.device,
        )
        collision_pipeline.set_broad_phase_filter_data(filter_data)
        self._share_vertex_filter_data = filter_data

        if self._sleeping_enabled:
            nphase = collision_pipeline.narrow_phase
            self._sleeping_shape_aabb_lower = nphase.shape_aabb_lower
            self._sleeping_shape_aabb_upper = nphase.shape_aabb_upper

    # Kinematic pose scripting.

    def set_kinematic_pose(
        self,
        body: int,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """Script a kinematic body's end-of-next-step pose. The next step
        infers velocity and lerps/slerps across substeps. Use
        :meth:`set_kinematic_poses_batch` for many bodies."""
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
        """Batched :meth:`set_kinematic_pose`. Non-kinematic body_ids are
        silently ignored by the kernel; pre-filter on host for strict validation."""
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

    # Joint initialisation.

    @staticmethod
    def required_constraint_dwords(
        num_joints: int,
        num_cloth_triangles: int = 0,
        num_soft_tetrahedra: int = 0,
        num_cloth_bending: int = 0,
        num_soft_hexahedra: int = 0,
    ) -> int:
        """Dword width for joint/cloth/soft rows; contacts use separate storage."""
        widths = [1]
        if int(num_joints) > 0:
            widths.append(int(ADBS_DWORDS))
        if int(num_cloth_triangles) > 0:
            widths.append(int(CLOTH_TRIANGLE_DWORDS))
        if int(num_cloth_bending) > 0:
            widths.append(int(CLOTH_BENDING_DWORDS))
        if int(num_soft_tetrahedra) > 0:
            # Both ARAP and block Neo-Hookean variants share this cid
            # range; the choice is made at populate time. Reserve enough
            # dwords for either variant up-front so the container stays
            # static across runs.
            widths.append(int(SOFT_TET_DWORDS))
            widths.append(int(SOFT_TET_NEOHOOKEAN_DWORDS))
        if int(num_soft_hexahedra) > 0:
            widths.append(int(SOFT_HEX_DWORDS))
        return max(widths)

    @staticmethod
    def make_constraint_container(
        num_joints: int,
        device: wp.context.Devicelike = None,
        num_cloth_triangles: int = 0,
        num_soft_tetrahedra: int = 0,
        num_cloth_bending: int = 0,
        num_soft_hexahedra: int = 0,
    ) -> ConstraintContainer:
        """Factory for a correctly-sized :class:`ConstraintContainer`.

        cid layout:

        * ``[0, num_joints)`` -- joints
        * ``[num_joints, +num_cloth_triangles)`` -- cloth triangles
        * ``[..., +num_cloth_bending)`` -- cloth bending hinges
        * ``[..., +num_soft_tetrahedra)`` -- soft-body tets
        * ``[..., +num_soft_hexahedra)`` -- soft-body hexes
        """
        cap = max(
            1,
            int(num_joints)
            + int(num_cloth_triangles)
            + int(num_cloth_bending)
            + int(num_soft_tetrahedra)
            + int(num_soft_hexahedra),
        )
        return constraint_container_zeros(
            num_constraints=cap,
            num_dwords=PhoenXWorld.required_constraint_dwords(
                num_joints,
                num_cloth_triangles,
                num_soft_tetrahedra,
                num_cloth_bending,
                num_soft_hexahedra,
            ),
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
        friction_coefficient: wp.array | None = None,
        d6_limit_axis0: wp.array | None = None,
        d6_limit_axis1: wp.array | None = None,
        d6_limit_axis2: wp.array | None = None,
        d6_limit_lower: wp.array | None = None,
        d6_limit_upper: wp.array | None = None,
        d6_limit_count: wp.array | None = None,
        articulation_joint_mask: wp.array | None = None,
    ) -> None:
        """Pack ``num_joints`` actuated-DBS joint columns. Call once after
        :meth:`__init__`, before the first :meth:`step`. All input arrays must
        be length ``num_joints``.

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
            friction_coefficient: Per-joint Coulomb friction limit on
                the axial DoF [N*m for revolute, N for prismatic].
                ``None`` (default) zero-fills, disabling friction on
                every joint. Operates independently of the drive --
                total axial impulse is the sum of the clamped drive PD
                term and the clamped friction term, matching MuJoCo's
                ``dof_frictionloss + actuator`` decomposition.
            d6_limit_axis0, d6_limit_axis1, d6_limit_axis2: Optional
                world-frame angular limit axes for D6 BALL/UNIVERSAL.
            d6_limit_lower, d6_limit_upper: Optional per-axis limit
                windows [rad].
            d6_limit_count: Optional number of D6 angular limit axes.
            articulation_joint_mask: Optional per-column mask selecting
                joints owned by the DVI articulation solve. Use this to
                exclude loop-closure joints from the direct tree solve.
        """
        if self.num_joints <= 0:
            return
        if armature is None:
            armature = wp.zeros(self.num_joints, dtype=wp.float32, device=self.device)
        if friction_coefficient is None:
            friction_coefficient = wp.zeros(self.num_joints, dtype=wp.float32, device=self.device)
        if d6_limit_axis0 is None:
            d6_limit_axis0 = wp.zeros(self.num_joints, dtype=wp.vec3f, device=self.device)
        if d6_limit_axis1 is None:
            d6_limit_axis1 = wp.zeros(self.num_joints, dtype=wp.vec3f, device=self.device)
        if d6_limit_axis2 is None:
            d6_limit_axis2 = wp.zeros(self.num_joints, dtype=wp.vec3f, device=self.device)
        if d6_limit_lower is None:
            d6_limit_lower = wp.zeros(self.num_joints, dtype=wp.vec3f, device=self.device)
        if d6_limit_upper is None:
            d6_limit_upper = wp.zeros(self.num_joints, dtype=wp.vec3f, device=self.device)
        if d6_limit_count is None:
            d6_limit_count = wp.zeros(self.num_joints, dtype=wp.int32, device=self.device)
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
        if self.cache_articulation_topology:
            self._cache_prefactorized_articulation_topology(
                body1,
                body2,
                joint_mode,
                drive_mode=drive_mode,
                stiffness_drive=stiffness_drive,
                damping_drive=damping_drive,
                min_value=min_value,
                max_value=max_value,
                d6_limit_count=d6_limit_count,
                articulation_joint_mask=articulation_joint_mask,
            )
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
                friction_coefficient,
                d6_limit_axis0,
                d6_limit_axis1,
                d6_limit_axis2,
                d6_limit_lower,
                d6_limit_upper,
                d6_limit_count,
            ],
            device=self.device,
        )

    def _cache_prefactorized_articulation_topology(
        self,
        body1: wp.array,
        body2: wp.array,
        joint_mode: wp.array,
        *,
        drive_mode: wp.array | None = None,
        stiffness_drive: wp.array | None = None,
        damping_drive: wp.array | None = None,
        min_value: wp.array | None = None,
        max_value: wp.array | None = None,
        d6_limit_count: wp.array | None = None,
        articulation_joint_mask: wp.array | None = None,
    ) -> None:
        """Cache topology-only DVI articulation data from joint init arrays.

        The symbolic phase is topology-only and intentionally runs outside graph
        capture. Runtime DVI kernels will consume this metadata once the numeric
        path is connected.
        """
        self.articulation_topology = None
        self.articulation_system = None
        self.articulation_device_system = None
        self.articulation_dvi_joint_mask = None
        try:
            body1_np = body1.numpy()
            body2_np = body2.numpy()
            joint_mode_np = joint_mode.numpy()
            inverse_mass_np = self.bodies.inverse_mass.numpy()
            joint_mask_np = articulation_joint_mask.numpy() if articulation_joint_mask is not None else None
            drive_mode_np = drive_mode.numpy() if drive_mode is not None else None
            stiffness_drive_np = stiffness_drive.numpy() if stiffness_drive is not None else None
            damping_drive_np = damping_drive.numpy() if damping_drive is not None else None
            min_value_np = min_value.numpy() if min_value is not None else None
            max_value_np = max_value.numpy() if max_value is not None else None
            d6_limit_count_np = d6_limit_count.numpy() if d6_limit_count is not None else None
        except Exception:
            return

        if joint_mask_np is not None:
            joint_mask_np = np.asarray(joint_mask_np, dtype=bool)
            if joint_mask_np.shape != joint_mode_np.shape:
                raise ValueError(
                    f"articulation_joint_mask must have shape {joint_mode_np.shape}, got {joint_mask_np.shape}"
                )
            self.articulation_dvi_joint_mask = joint_mask_np.copy()

        if self.num_joints > 0:
            if self.articulation_dvi_replaces_joint_pgs and joint_mask_np is not None:
                pgs_enabled = np.where(joint_mask_np, 0, 1).astype(np.int32)
            elif self.articulation_dvi_replaces_joint_pgs:
                pgs_enabled = np.zeros(self.num_joints, dtype=np.int32)
            else:
                pgs_enabled = np.ones(self.num_joints, dtype=np.int32)
            self._joint_pgs_enabled.assign(pgs_enabled)

        static_body_indices = np.nonzero(inverse_mass_np <= 0.0)[0].astype(np.int32)
        if not self.articulation_dvi_replaces_joint_pgs:
            drive_mode_np = None
            stiffness_drive_np = None
            damping_drive_np = None
            min_value_np = None
            max_value_np = None
            d6_limit_count_np = None

        topology = ArticulationTopology.from_host(
            body1_np,
            body2_np,
            joint_mode_np,
            static_body_indices=static_body_indices,
            enabled_joint_mask=joint_mask_np,
            drive_mode=drive_mode_np,
            stiffness_drive=stiffness_drive_np,
            damping_drive=damping_drive_np,
            min_value=min_value_np,
            max_value=max_value_np,
            d6_limit_count=d6_limit_count_np,
        )
        self.articulation_topology = topology
        self.articulation_system = PrefactorizedArticulationSystem.from_topology(topology)
        self.articulation_device_system = ArticulationDeviceSystem.from_topology(
            topology, self.device, self.articulation_system.symbolic
        )

    @staticmethod
    def _normalize_articulation_dvi_host_solver(value: str) -> str:
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized in {"device_block_sparse", "device_sparse_ldl", "device_sparse_ldlt", "gpu_block_sparse"}:
            return "device_block_sparse"
        if normalized in {"block_sparse", "block_sparse_ldlt", "sparse_ldl", "sparse_ldlt"}:
            return "block_sparse"
        if normalized in {"dense", "dense_ldlt"}:
            return "dense"
        raise ValueError(
            f"articulation_dvi_host_solver must be 'device_block_sparse', 'block_sparse', or 'dense' (got {value!r})"
        )

    def solve_articulations_dvi_host(
        self,
        dt: float | None = None,
        *,
        alpha: float = 0.0,
        recovery_speed: float = -1.0,
        solver: str | None = None,
    ) -> bool:
        """Apply one DVI solve for cached articulation rows.

        This is the integration checkpoint for the full-coordinate articulation
        path: row population, matrix assembly, RHS construction, factorization,
        solve, and velocity application all run against PhoenX buffers.
        ``solver="device_block_sparse"`` stays on the device after row
        assembly; the host modes remain as validation fallbacks.
        """
        if (
            self.articulation_device_system is None
            or self.articulation_system is None
            or self.articulation_device_system.total_rows <= 0
        ):
            return False

        solve_dt = float(self.substep_dt if dt is None else dt)
        if solve_dt <= 0.0:
            raise ValueError(f"dt must be positive, got {solve_dt}")

        device_system = self.articulation_device_system
        solve_method = self._normalize_articulation_dvi_host_solver(
            self.articulation_dvi_host_solver if solver is None else solver
        )

        device_system.populate_from_adbs_constraints(self.constraints, self.bodies, dt=solve_dt, device=self.device)
        device_system.compute_residual(
            self.bodies,
            dt=solve_dt,
            alpha=float(alpha),
            recovery_speed=float(recovery_speed),
            device=self.device,
        )

        if solve_method == "device_block_sparse":
            device_system.assemble_block_sparse_matrix(
                self.bodies.inverse_mass,
                self.bodies.inverse_inertia_world,
                diagonal_regularization=self.articulation_system.diagonal_regularization,
                device=self.device,
            )
            device_system.solve_block_sparse_matrix(device=self.device)
        else:
            device_system.assemble_dense_matrix(
                self.bodies.inverse_mass, self.bodies.inverse_inertia_world, device=self.device
            )
            total_rows = device_system.total_rows
            matrix = device_system.matrix.numpy()[:total_rows, :total_rows]
            rhs = device_system.rhs.numpy()[:total_rows]
            self.articulation_system.factorize_dense_matrix(matrix)
            solution = self.articulation_system.solve_prefactorized(rhs, method=solve_method).astype(np.float32)
            device_system.solution.assign(solution)

        device_system.apply_solution(
            self.bodies,
            self.bodies.inverse_mass,
            self.bodies.inverse_inertia_world,
            device=self.device,
        )
        return True

    def _solve_articulations_dvi_host_for_step(self) -> None:
        if self.solve_articulations_dvi_host(dt=self.substep_dt):
            return
        if self.num_joints > 0:
            raise RuntimeError(
                "articulation_dvi_host=True requires initialized PhoenX ADBS joint topology before step()"
            )

    def set_collision_filter_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Replace the body-pair contact filter (canonical (min, max), deduped)."""
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

    def _particles_or_sentinel(self) -> ParticleContainer:
        """Return the particle SoA, allocating a length-1 sentinel for
        rigid-only scenes so kernels with a particle parameter still
        have something to bind."""
        if self.particles is not None:
            return self.particles
        if self._particle_sentinel is None:
            self._particle_sentinel = particle_container_zeros(1, device=self.device)
        return self._particle_sentinel

    def _copy_particle_world_ids_from_model(self, model) -> None:
        """Copy model particle world ids into the scheduler metadata buffer."""
        if self.num_particles == 0:
            return
        particle_world = getattr(model, "particle_world", None)
        if particle_world is None:
            if self.num_worlds > 1:
                raise RuntimeError("multi-world PhoenX deformable scheduling requires model.particle_world")
            self._particle_world_id.fill_(0)
            return
        if int(particle_world.shape[0]) != self.num_particles:
            raise ValueError(
                f"model.particle_world length ({particle_world.shape[0]}) != world.num_particles ({self.num_particles})"
            )
        wp.copy(self._particle_world_id, particle_world)

    def _copy_particle_world_ids_from_array(self, particle_world: wp.array[wp.int32] | None) -> None:
        """Copy caller-supplied particle world ids for array-backed soft rows."""
        if self.num_particles == 0:
            return
        if particle_world is None:
            if self.num_worlds > 1:
                raise RuntimeError(
                    "multi-world PhoenX deformable scheduling requires particle_world for array-backed particles"
                )
            self._particle_world_id.fill_(0)
            return
        if int(particle_world.shape[0]) != self.num_particles:
            raise ValueError(
                f"particle_world length ({particle_world.shape[0]}) != world.num_particles ({self.num_particles})"
            )
        wp.copy(self._particle_world_id, particle_world)

    def _refresh_has_position_level_writers(self) -> None:
        """Stamp :attr:`BodyContainer.has_position_level_writers` from
        the current cloth-triangle / cloth-bending / soft-tet counts.

        Only those constraint types ever write
        ``ACCESS_MODE_POSITION_LEVEL``, so the flag is 1 iff any of
        them is present. :func:`body_set_access_mode` reads this single
        element first; in rigid-only scenes the load is warp-uniform
        and the whole flip dead-code-eliminates.
        """
        flag = int(
            self.num_cloth_triangles > 0
            or self.num_cloth_bending > 0
            or self.num_soft_tetrahedra > 0
            or self.num_soft_hexahedra > 0
        )
        self.bodies.has_position_level_writers.fill_(flag)

    def populate_cloth_triangles_from_model(
        self,
        model,
        *,
        beta_lambda: float = 0.1,
        beta_mu: float = 0.1,
    ) -> None:
        """Stamp cloth-triangle constraint rows from a Newton :class:`Model`.

        Reads ``model.particle_q`` / ``particle_qd`` / ``particle_inv_mass``
        / ``tri_indices`` / ``tri_materials``. Triangle cids occupy
        ``[num_joints, num_joints + num_cloth_triangles)`` in the
        :class:`ConstraintContainer`. Also runs a host-side greedy
        graph coloring pass (one-shot at populate time) so the
        per-substep iterate kernel can launch parallel-safe kernels
        per color group.
        """
        if self.num_cloth_triangles == 0:
            return
        if self.particles is None:
            raise RuntimeError("populate_cloth_triangles_from_model requires num_particles > 0")
        model_particle_count = int(model.particle_count)
        model_tri_count = int(model.tri_count)
        if model_particle_count != self.num_particles:
            raise ValueError(
                f"populate_cloth_triangles_from_model: model.particle_count "
                f"({model_particle_count}) != world.num_particles ({self.num_particles})"
            )
        if model_tri_count != self.num_cloth_triangles:
            raise ValueError(
                f"populate_cloth_triangles_from_model: model.tri_count "
                f"({model_tri_count}) != world.num_cloth_triangles ({self.num_cloth_triangles})"
            )
        wp.copy(self.particles.position, model.particle_q)
        wp.copy(self.particles.velocity, model.particle_qd)
        wp.copy(self.particles.inverse_mass, model.particle_inv_mass)
        self._copy_particle_world_ids_from_model(model)
        wp.launch(
            cloth_init_triangle_rows_kernel,
            dim=self.num_cloth_triangles,
            inputs=[
                self.constraints,
                wp.int32(self.num_joints),  # cid_offset
                wp.int32(self.num_bodies),
                model.tri_indices,
                model.particle_q,
                model.tri_materials,
                wp.float32(beta_lambda),
                wp.float32(beta_mu),
            ],
            device=self.device,
        )
        # Bump the active-constraint count so the partitioner sees the
        # cloth-triangle suffix from the very first step.
        self._num_active_constraints.fill_(self._contact_offset)

    def populate_cloth_bending_from_model(
        self,
        model,
    ) -> None:
        """Stamp cloth-bending constraint rows from a Newton :class:`Model`.

        Reads ``model.particle_q`` / ``edge_indices`` /
        ``edge_bending_properties``. Cloth-bending cids occupy
        ``[num_joints + num_cloth_triangles, num_joints +
        num_cloth_triangles + num_cloth_bending)``.

        Newton's bending edges are stored as
        ``edge_indices[t] = (o0, o1, v1, v2)`` where ``v1, v2`` are the
        shared edge and ``o0, o1`` are the two opposite vertices.
        Boundary edges (``o0 < 0`` or ``o1 < 0``) are emitted as
        no-op constraints by the init kernel.
        """
        if self.num_cloth_bending == 0:
            return
        if self.particles is None:
            raise RuntimeError("populate_cloth_bending_from_model requires num_particles > 0")
        model_particle_count = int(model.particle_count)
        model_edge_count = int(model.edge_count)
        if model_particle_count != self.num_particles:
            raise ValueError(
                f"populate_cloth_bending_from_model: model.particle_count "
                f"({model_particle_count}) != world.num_particles ({self.num_particles})"
            )
        if model_edge_count != self.num_cloth_bending:
            raise ValueError(
                f"populate_cloth_bending_from_model: model.edge_count "
                f"({model_edge_count}) != world.num_cloth_bending ({self.num_cloth_bending})"
            )
        wp.copy(self.particles.position, model.particle_q)
        wp.copy(self.particles.velocity, model.particle_qd)
        wp.copy(self.particles.inverse_mass, model.particle_inv_mass)
        self._copy_particle_world_ids_from_model(model)
        wp.launch(
            cloth_bending_init_rows_kernel,
            dim=self.num_cloth_bending,
            inputs=[
                self.constraints,
                wp.int32(self._cloth_bending_offset),
                wp.int32(self.num_bodies),
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_bending_properties,
                wp.float32(1.0e-6),  # alpha floor
            ],
            device=self.device,
        )
        self._num_active_constraints.fill_(self._contact_offset)

    def populate_soft_tetrahedra_from_model(
        self,
        model,
        *,
        constraint_type: SoftBodyConstraintType = SoftBodyConstraintType.ARAP,
        beta_lambda: float = 0.1,
        beta_mu: float = 0.1,
    ) -> None:
        """Stamp soft-tetrahedron constraint rows from a Newton :class:`Model`.

        Reads ``model.particle_q`` / ``particle_qd`` / ``particle_inv_mass``
        / ``tet_indices`` / ``tet_poses`` / ``tet_materials``. Soft-tet cids
        occupy ``[num_joints + num_cloth_triangles, num_joints +
        num_cloth_triangles + num_soft_tetrahedra)`` in the
        :class:`ConstraintContainer`.

        ``tet_poses[t]`` is the pre-inverted rest-pose 3x3 stamped by
        :meth:`newton.ModelBuilder.add_tetrahedron`. ``tet_materials[t]``
        is ``(k_mu, k_lambda, k_damp)`` in SI Pa / Pa / dimensionless.

        Idempotent across multiple calls. May be combined with
        :meth:`populate_cloth_triangles_from_model` in the same scene;
        each populator stamps its own cid range and bumps
        ``_num_active_constraints`` to the joint + cloth + soft-tet sum.

        Args:
            constraint_type: Selects between the corotational ARAP shear
                row (:attr:`SoftBodyConstraintType.ARAP`) and the
                block-coupled stable Neo-Hookean variant
                (:attr:`SoftBodyConstraintType.BLOCK_NEOHOOKEAN`) from
                Ton-That, Kry & Andrews 2024. Both variants share the
                same row range; the per-row constraint type tag at dword
                0 routes the dispatcher to the matching kernel. Default
                preserves the prior single-row ARAP behaviour for
                backward compatibility.
            beta_mu: Macklin XPBD damping on the shear (ARAP) /
                deviatoric (Neo-Hookean) row [1/s]. Enters the lambda
                numerator as ``gamma_mu * grad . (x -
                position_prev_substep)`` with ``gamma_mu = beta_mu *
                substep_dt`` -- velocity-projected damping that does NOT
                damp at rest. ``0.0`` => bare XPBD; ``~0.1`` settles a
                free-falling soft cube within ~30 substeps.
            beta_lambda: Damping on the volume / hydrostatic row [1/s].
                Used by the block Neo-Hookean variant; reserved for the
                ARAP variant (volume row not implemented there).
        """
        if self.num_soft_tetrahedra == 0:
            return
        if self.particles is None:
            raise RuntimeError("populate_soft_tetrahedra_from_model requires num_particles > 0")
        model_particle_count = int(model.particle_count)
        model_tet_count = int(model.tet_count)
        if model_particle_count != self.num_particles:
            raise ValueError(
                f"populate_soft_tetrahedra_from_model: model.particle_count "
                f"({model_particle_count}) != world.num_particles ({self.num_particles})"
            )
        if model_tet_count != self.num_soft_tetrahedra:
            raise ValueError(
                f"populate_soft_tetrahedra_from_model: model.tet_count "
                f"({model_tet_count}) != world.num_soft_tetrahedra ({self.num_soft_tetrahedra})"
            )
        # Only re-copy particle SoA if cloth populate hasn't already done so;
        # both populators read the same model.particle_q so the result is
        # identical. Keeping the copy idempotent simplifies the call order.
        wp.copy(self.particles.position, model.particle_q)
        wp.copy(self.particles.velocity, model.particle_qd)
        wp.copy(self.particles.inverse_mass, model.particle_inv_mass)
        self._copy_particle_world_ids_from_model(model)
        if constraint_type == SoftBodyConstraintType.BLOCK_NEOHOOKEAN:
            wp.launch(
                soft_tet_neohookean_init_rows_kernel,
                dim=self.num_soft_tetrahedra,
                inputs=[
                    self.constraints,
                    wp.int32(self._soft_tet_offset),
                    wp.int32(self.num_bodies),
                    model.tet_indices,
                    model.particle_q,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(beta_lambda),  # hydrostatic-row damping
                    wp.float32(beta_mu),  # deviatoric-row damping
                ],
                device=self.device,
            )
            self._soft_tet_uses_neohookean = True
        else:
            wp.launch(
                soft_tet_init_rows_kernel,
                dim=self.num_soft_tetrahedra,
                inputs=[
                    self.constraints,
                    wp.int32(self._soft_tet_offset),  # cid_offset (after joints, cloth tris, cloth bending)
                    wp.int32(self.num_bodies),
                    model.tet_indices,
                    model.particle_q,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(beta_lambda),
                    wp.float32(beta_mu),
                ],
                device=self.device,
            )
            self._soft_tet_uses_neohookean = False
        self._num_active_constraints.fill_(self._contact_offset)

    def populate_soft_hexahedra_from_arrays(
        self,
        hex_indices: wp.array,  # wp.array2d[wp.int32], shape [num_hexahedra, 8]
        particle_q: wp.array,  # wp.array[wp.vec3f], rest particle positions
        hex_materials: wp.array,  # wp.array2d[wp.float32], shape [num_hexahedra, 4] = (k_mu, k_lambda, beta_h, beta_d)
        particle_qd: wp.array | None = None,  # wp.array[wp.vec3f], initial particle velocities (optional)
        particle_inv_mass: wp.array | None = None,  # wp.array[wp.float32], optional inverse masses
        particle_world: wp.array[wp.int32] | None = None,  # Optional particle world ids
        *,
        strain_model: str | int = "trace",
    ) -> None:
        """Stamp soft-hexahedron constraint rows from caller-supplied arrays.

        Hex constraints occupy ``[num_joints + num_cloth_triangles +
        num_cloth_bending + num_soft_tetrahedra, num_joints + ... +
        num_soft_hexahedra)`` in the :class:`ConstraintContainer`.
        Builds ``inv_rest = J^{-T}`` and ``rest_volume = 8 det(J)``
        from the rest corner positions of each hex; converts the per-
        element ``(k_mu, k_lambda)`` Lame pair into mixed strain/volume
        compliances ``alpha^H = 1/(k_lambda * V)``,
        ``alpha^D = 1/(k_mu * V)`` and the rest offset
        ``gamma = 1 + k_mu / k_lambda``. ``strain_model="trace"``
        uses the xpbd-fem-style integrated trace strain. ``"arap"``
        uses integrated per-Gauss-point ARAP strain coupled with the
        same center volume row.

        Caller is responsible for stamping ``hex_indices[h, 0..7]`` in
        canonical isoparametric 8-corner order (see
        :mod:`constraint_soft_hexahedron`).

        Args:
            hex_indices: ``[num_hexahedra, 8]`` int32 particle indices
                per hex, in canonical corner order.
            particle_q: ``[num_particles, 3]`` float32 rest positions.
                Copied into :attr:`particles.position`.
            hex_materials: ``[num_hexahedra, 4]`` float32
                ``(k_mu, k_lambda, beta_h, beta_d)`` per hex.
                ``k_mu`` / ``k_lambda`` are the second / first Lame
                parameters [Pa]; ``beta_h`` / ``beta_d`` are Macklin
                XPBD damping coefficients on the volume / strain rows
                [1/s] (``0`` => bare XPBD).
            particle_qd: Optional initial velocities. ``None`` leaves
                :attr:`particles.velocity` at its default (zeros from
                container construction).
            particle_inv_mass: Optional inverse-mass array. ``None``
                leaves :attr:`particles.inverse_mass` at its default.
                Pin a particle by setting its inv_mass to 0.
            particle_world: Optional particle world ids for multi-world
                scheduling. ``None`` maps particles to world 0 when
                ``num_worlds == 1``.
            strain_model: ``"trace"``/``"xpbd_fem"`` for the default
                integrated trace strain, or ``"arap"`` for integrated
                ARAP strain. Integer constants
                ``SOFT_HEX_STRAIN_MODEL_TRACE`` and
                ``SOFT_HEX_STRAIN_MODEL_ARAP`` are also accepted.

        Idempotent across repeated calls; safe to combine with
        :meth:`populate_soft_tetrahedra_from_model` in mixed scenes
        (each populator stamps its own cid range and
        ``_num_active_constraints`` is advanced to ``_contact_offset``
        so contacts append after).
        """
        if self.num_soft_hexahedra == 0:
            return
        if self.particles is None:
            raise RuntimeError("populate_soft_hexahedra_from_arrays requires num_particles > 0")
        if hex_indices.shape[0] != self.num_soft_hexahedra:
            raise ValueError(
                f"populate_soft_hexahedra_from_arrays: hex_indices.shape[0] "
                f"({hex_indices.shape[0]}) != world.num_soft_hexahedra ({self.num_soft_hexahedra})"
            )
        if hex_indices.shape[1] != 8:
            raise ValueError(
                f"populate_soft_hexahedra_from_arrays: hex_indices must be [N, 8] "
                f"(got shape {tuple(hex_indices.shape)})"
            )
        if hex_materials.shape[0] != self.num_soft_hexahedra or hex_materials.shape[1] != 4:
            raise ValueError(
                f"populate_soft_hexahedra_from_arrays: hex_materials must be "
                f"[N, 4] = (k_mu, k_lambda, beta_h, beta_d) (got shape {tuple(hex_materials.shape)})"
            )
        strain_model_value = _soft_hex_strain_model_value(strain_model)
        wp.copy(self.particles.position, particle_q)
        if particle_qd is not None:
            wp.copy(self.particles.velocity, particle_qd)
        if particle_inv_mass is not None:
            wp.copy(self.particles.inverse_mass, particle_inv_mass)
        self._copy_particle_world_ids_from_array(particle_world)
        wp.launch(
            soft_hex_init_rows_from_arrays_kernel,
            dim=self.num_soft_hexahedra,
            inputs=[
                self.constraints,
                wp.int32(self._soft_hex_offset),
                wp.int32(self.num_bodies),
                hex_indices,
                particle_q,
                hex_materials,
                wp.int32(strain_model_value),
            ],
            device=self.device,
        )
        self._num_active_constraints.fill_(self._contact_offset)

    def setup_cloth_collision_pipeline(
        self,
        model,
        *,
        cloth_thickness: float = 0.005,
        cloth_gap: float = 0.010,
        cloth_self_collision: bool = True,
        soft_body_thickness: float = 0.005,
        soft_body_gap: float = 0.010,
        broad_phase: str = "sap",
        contact_matching: str = "sticky",
        rigid_contact_max: int | None = None,
        shape_pairs_max: int | None = None,
        phoenx_body_offset: int = 1,
    ):
        """Construct (and stash on ``model._collision_pipeline``) a
        unified rigid + cloth-triangle + soft-tet :class:`CollisionPipeline`.

        Allocates ``extra_shape_count = num_cloth_triangles +
        num_soft_tetrahedra`` virtual shape slots, stamps static metadata
        for two suffixes:

        * Cloth-triangle suffix ``[S, S + T)`` -- ``shape_type=TRIANGLE``.
        * Soft-tet suffix ``[S + T, S + T + Tet)`` -- ``shape_type=TETRAHEDRON``.

        Per-step :meth:`update_cloth_shape_geometry` refreshes
        ``geom_transform`` / ``geom_data`` / ``shape_aabb_*`` / (tet only)
        ``shape_source`` from current particle positions.

        Narrow-phase reuses Newton's existing GeoType.TRIANGLE / TETRAHEDRON
        support-function dispatch unchanged.

        Args:
            model: Finalised :class:`~newton.Model` with ``model.shape_*``
                populated. Must have ``model.tri_count ==
                self.num_cloth_triangles`` and ``model.tet_count ==
                self.num_soft_tetrahedra``.
            cloth_thickness: Geometric Minkowski-skin half-thickness
                added to each cloth triangle [m]. Default 5 mm.
            cloth_gap: Speculative-contact enlargement on top of the
                thickness [m]. Default 10 mm. Total contact-detection
                radius is ``thickness + gap``.
            cloth_self_collision: Enable cloth triangle self-collision.
            soft_body_thickness: Per-tet skin half-thickness [m].
            soft_body_gap: Per-tet speculative-contact gap [m].
            broad_phase: ``"sap"`` (default), ``"nxn"``, or ``"explicit"``.
            contact_matching: PhoenX requires ``"sticky"`` (default) or
                ``"latest"`` so warm-starting works.
            rigid_contact_max: Override Newton's contact-buffer size.
            shape_pairs_max: Broad-phase candidate-pair budget override.

        Returns:
            The constructed :class:`CollisionPipeline`.
        """
        from newton._src.geometry.flags import ShapeFlags  # noqa: PLC0415
        from newton._src.geometry.types import GeoType  # noqa: PLC0415
        from newton._src.sim.collide import CollisionPipeline  # noqa: PLC0415

        def _soft_tet_collision_mask(tet_indices: np.ndarray) -> np.ndarray:
            tets = np.asarray(tet_indices, dtype=np.int32).reshape(-1, 4)
            if tets.shape[0] == 0:
                return np.zeros(0, dtype=bool)

            face_counts: dict[tuple[int, int, int], int] = {}
            for a, b, c, d in tets:
                for face in ((a, b, c), (a, d, b), (b, d, c), (a, c, d)):
                    key = tuple(sorted((int(face[0]), int(face[1]), int(face[2]))))
                    face_counts[key] = face_counts.get(key, 0) + 1

            surface_vertices: set[int] = set()
            for face, count in face_counts.items():
                if count == 1:
                    surface_vertices.update(face)
            if not surface_vertices:
                return np.ones(tets.shape[0], dtype=bool)
            surface = np.fromiter(surface_vertices, dtype=np.int32)
            return np.isin(tets, surface).any(axis=1)

        if self.num_cloth_triangles == 0 and self.num_soft_tetrahedra == 0:
            raise RuntimeError(
                "setup_cloth_collision_pipeline requires num_cloth_triangles > 0 "
                "or num_soft_tetrahedra > 0; rigid-only scenes use "
                "newton.CollisionPipeline directly"
            )
        S = int(model.shape_count)
        T = int(self.num_cloth_triangles)
        Tet = int(self.num_soft_tetrahedra)

        # Unified shape_world / shape_flags arrays of length S+T+Tet.
        # Rigid prefix mirrors model.shape_*; suffix lands in world 0
        # with default flags (= same flag value as a typical dynamic
        # rigid shape so the broad phase doesn't cull cloth tris).
        unified_shape_world = wp.zeros(S + T + Tet, dtype=wp.int32, device=self.device)
        if S > 0 and getattr(model, "shape_world", None) is not None:
            wp.copy(unified_shape_world, model.shape_world, count=S)
        unified_shape_flags = None
        if getattr(model, "shape_flags", None) is not None:
            shape_flags = model.shape_flags
            # Suffix flags = take the most permissive flag set we see
            # in the prefix so deformable shapes participate in broad phase.
            unified_shape_flags = wp.zeros(S + T + Tet, dtype=shape_flags.dtype, device=self.device)
            if S > 0:
                wp.copy(unified_shape_flags, shape_flags, count=S)
                seed_value = int(shape_flags.numpy()[0])
                if seed_value != 0:
                    arr = unified_shape_flags.numpy()
                    arr[S:] = seed_value
                    if Tet > 0:
                        tet_collides = _soft_tet_collision_mask(model.tet_indices.numpy())
                        collide_bit = int(ShapeFlags.COLLIDE_SHAPES)
                        tet_flags = arr[S + T : S + T + Tet]
                        tet_flags[~tet_collides] &= ~collide_bit
                    unified_shape_flags.assign(arr)

        # Length-1 sentinel for the unused mesh-indices argument in the
        # share-vertex filter when the matching deformable category is
        # absent (Warp arrays must be non-empty to be bound).
        if T == 0:
            tri_indices_for_filter = wp.zeros((1, 3), dtype=wp.int32, device=self.device)
        else:
            tri_indices_for_filter = model.tri_indices
        if Tet == 0:
            tet_indices_for_filter = wp.zeros((1, 4), dtype=wp.int32, device=self.device)
        else:
            tet_indices_for_filter = model.tet_indices

        pipeline = CollisionPipeline(
            model,
            broad_phase=broad_phase,
            contact_matching=contact_matching,
            rigid_contact_max=rigid_contact_max,
            shape_pairs_max=shape_pairs_max,
            extra_shape_count=T + Tet,
            unified_shape_world=unified_shape_world,
            unified_shape_flags=unified_shape_flags,
            broad_phase_filter=(
                phoenx_cloth_share_vertex_filter,
                PhoenXClothShareVertexFilterData,
            ),
        )

        # Bind the share-vertex filter's per-step data: tri/tet index
        # arrays + offsets, plus optional sleeping-aware fields. The
        # filter callback reads this at every broad-phase pair test to
        # drop pairs of deformables (cloth or soft-tet) that share at
        # least one particle, and (when sleeping is on) rigid-rigid
        # pairs where both bodies are flagged sleeping.
        share_vertex_data = build_phoenx_share_vertex_filter_data(
            num_rigid_shapes=S,
            num_cloth_triangles=T,
            tri_indices=tri_indices_for_filter,
            tet_indices=tet_indices_for_filter,
            sleeping_enabled=self._sleeping_enabled,
            phoenx_body_offset=int(phoenx_body_offset),
            shape_body=model.shape_body if self._sleeping_enabled else None,
            body_island_root=self.bodies.island_root if self._sleeping_enabled else None,
            body_motion_type=self.bodies.motion_type if self._sleeping_enabled else None,
            device=self.device,
        )
        pipeline.set_broad_phase_filter_data(share_vertex_data)
        self._share_vertex_filter_data = share_vertex_data
        if self._sleeping_enabled:
            self._sleeping_shape_aabb_lower = pipeline.narrow_phase.shape_aabb_lower
            self._sleeping_shape_aabb_upper = pipeline.narrow_phase.shape_aabb_upper

        # Stamp the static deformable-shape suffix metadata. Per-step
        # quantities (geom_xform, geom_data, AABB, shape_source for tets)
        # are written by :meth:`update_cloth_shape_geometry`.
        triangle_type = int(GeoType.TRIANGLE)
        tetrahedron_type = int(GeoType.TETRAHEDRON)

        def _fill_range_int(arr: wp.array, lo: int, hi: int, value: int) -> None:
            host = arr.numpy()
            host[lo:hi] = value
            arr.assign(host)

        def _fill_range_float(arr: wp.array, lo: int, hi: int, value: float) -> None:
            host = arr.numpy()
            host[lo:hi] = value
            arr.assign(host)

        if T > 0:
            _fill_range_int(pipeline.unified_shape_type, S, S + T, triangle_type)
            _fill_range_float(pipeline.unified_shape_margin, S, S + T, float(cloth_thickness))
            _fill_range_float(pipeline.unified_shape_gap, S, S + T, float(cloth_gap))
            _fill_range_float(pipeline.unified_shape_collision_radius, S, S + T, 0.0)
            cloth_collision_group = 1 if cloth_self_collision else -2
            _fill_range_int(pipeline.unified_shape_collision_group, S, S + T, cloth_collision_group)
        if Tet > 0:
            _fill_range_int(pipeline.unified_shape_type, S + T, S + T + Tet, tetrahedron_type)
            _fill_range_float(pipeline.unified_shape_margin, S + T, S + T + Tet, float(soft_body_thickness))
            _fill_range_float(pipeline.unified_shape_gap, S + T, S + T + Tet, float(soft_body_gap))
            _fill_range_float(pipeline.unified_shape_collision_radius, S + T, S + T + Tet, 0.0)
            _fill_range_int(pipeline.unified_shape_collision_group, S + T, S + T + Tet, 1)
        # ``unified_shape_body`` was already filled to -1 in both
        # suffixes by :meth:`CollisionPipeline._build_unified_shape_arrays`.
        # ``unified_shape_source_ptr`` defaults to 0; the per-step tet
        # geometry kernel writes the encoded 4th-vertex into it.

        # Stash for :meth:`update_cloth_shape_geometry` and downstream
        # collision dispatch.
        self._collision_pipeline = pipeline
        self._cloth_shape_offset: int = S
        self._soft_tet_shape_offset: int = S + T
        self._cloth_gap: float = float(cloth_gap)
        self._soft_body_gap: float = float(soft_body_gap)
        self._cloth_tri_indices = tri_indices_for_filter if T > 0 else None
        self._soft_tet_indices = tet_indices_for_filter if Tet > 0 else None

        # Per-shape filter id array. Length S + T + Tet. Rigid prefix
        # mirrors model.shape_body so the existing same-body collision
        # filter behaviour is preserved. Deformable suffixes get unique
        # negative ids ``-(2 + i)`` so distinct deformables (each
        # nominally anchored to the world via shape_body=-1) don't
        # collapse into a single filter group.
        S_int = int(S)
        T_int = int(T)
        Tet_int = int(Tet)
        filter_host = np.zeros(S_int + T_int + Tet_int, dtype=np.int32)
        if S_int > 0 and getattr(model, "shape_body", None) is not None:
            filter_host[:S_int] = model.shape_body.numpy()
        for i in range(T_int + Tet_int):
            filter_host[S_int + i] = -(2 + i)
        self._shape_filter_id = wp.array(filter_host, dtype=wp.int32, device=self.device)

        # Contact ingest operates on PhoenX body slots, not Newton body
        # ids. The collision pipeline's ``unified_shape_body`` must stay
        # in Newton indexing for narrow phase, so keep a separate map for
        # warm-start/contact-column ingest. Static and virtual deformable
        # shapes use the slot-0 world anchor until the endpoint overlay
        # replaces deformable sides with particle nodes.
        shape_body_phx = np.zeros(S_int + T_int + Tet_int, dtype=np.int32)
        if S_int > 0 and getattr(model, "shape_body", None) is not None:
            shape_body_raw = model.shape_body.numpy()
            shape_body_phx[:S_int] = np.where(shape_body_raw < 0, 0, shape_body_raw + int(phoenx_body_offset))
        self.set_shape_body(wp.array(shape_body_phx, dtype=wp.int32, device=self.device))

        # Per-shape endpoint table for cloth-aware contact ingest.
        # Allocated for the full unified shape range and populated once:
        # rigid prefix copies model.shape_body; cloth suffix decodes
        # tri_indices into 3-particle nodes; soft-tet suffix decodes
        # tet_indices into 4-particle nodes.
        self._shape_endpoints = shape_endpoints_zeros(S + T + Tet, device=self.device)
        wp.launch(
            _phoenx_populate_shape_endpoints_kernel,
            dim=S + T + Tet,
            inputs=[
                model.shape_body,
                tri_indices_for_filter,
                tet_indices_for_filter,
                wp.int32(S),
                wp.int32(T),
                wp.int32(S + T),
                wp.int32(Tet),
                wp.int32(self.num_bodies),
                wp.int32(phoenx_body_offset),
            ],
            outputs=[self._shape_endpoints],
            device=self.device,
        )

        # Wire as the model's default pipeline so model.collide(...)
        # picks it up. The user calls :meth:`collide` on this world
        # for the cloth-aware extended-AABB code path.
        model._collision_pipeline = pipeline
        return pipeline

    def update_cloth_shape_geometry(self) -> None:
        """Per-step refresh of the cloth-triangle + soft-tet shape suffixes.

        Reads current particle positions and re-canonicalises each
        deformable shape into its slot in the unified shape arrays.
        Cloth triangles get :func:`_phoenx_update_cloth_shape_geometry_kernel`;
        soft tets get :func:`_phoenx_update_soft_tet_shape_geometry_kernel`.
        Must be called once per step before
        :meth:`CollisionPipeline.collide_with_external_aabbs`.
        """
        if self._collision_pipeline is None:
            return
        pipeline = self._collision_pipeline
        if self.num_cloth_triangles > 0:
            wp.launch(
                _phoenx_update_cloth_shape_geometry_kernel,
                dim=self.num_cloth_triangles,
                inputs=[
                    self.particles,
                    self._cloth_tri_indices,
                    wp.int32(self._cloth_shape_offset),
                    pipeline.unified_shape_margin,
                    wp.float32(self._cloth_gap),
                ],
                outputs=[
                    pipeline.geom_transform,
                    pipeline.geom_data,
                    pipeline.narrow_phase.shape_aabb_lower,
                    pipeline.narrow_phase.shape_aabb_upper,
                ],
                device=self.device,
            )
        if self.num_soft_tetrahedra > 0:
            wp.launch(
                _phoenx_update_soft_tet_shape_geometry_kernel,
                dim=self.num_soft_tetrahedra,
                inputs=[
                    self.particles,
                    self._soft_tet_indices,
                    wp.int32(self._soft_tet_shape_offset),
                    pipeline.unified_shape_margin,
                    wp.float32(self._soft_body_gap),
                ],
                outputs=[
                    pipeline.geom_transform,
                    pipeline.geom_data,
                    pipeline.unified_shape_source_ptr,
                    pipeline.narrow_phase.shape_aabb_lower,
                    pipeline.narrow_phase.shape_aabb_upper,
                ],
                device=self.device,
            )

    def collide(self, state, contacts) -> None:
        """Run the unified rigid + cloth-triangle collision pipeline.

        Updates the cloth-triangle shape suffix from current particle
        positions, then dispatches to
        :meth:`CollisionPipeline.collide_with_external_aabbs`. Use this
        in place of :meth:`Model.collide` when the world has cloth
        triangles. CUDA-graph capture safe.
        """
        if self._collision_pipeline is None:
            raise RuntimeError(
                "PhoenXWorld.collide requires setup_cloth_collision_pipeline() to have been called first"
            )
        self.update_cloth_shape_geometry()
        self._collision_pipeline.collide_with_external_aabbs(state, contacts)

    def _make_placeholder_contact_views(self) -> ContactViews:
        """Size-1 dummy ContactViews for contact-free steps."""
        dummy_int = wp.zeros(1, dtype=wp.int32, device=self.device)
        dummy_vec3 = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        dummy_float = wp.zeros(1, dtype=wp.float32, device=self.device)
        # Length-0 soft-contact arrays; prepare kernel gates on shape per-contact.
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

    def _active_contact_views(self) -> ContactViews:
        """Current contacts or the graph-stable placeholder."""
        return self._contact_views if self._contact_views is not None else self._contact_views_placeholder

    def _refresh_prepare_this_substep(self) -> bool:
        """Return whether this substep should refresh cached row data."""
        return self.prepare_refresh_stride <= 1 or self._current_substep_index % self.prepare_refresh_stride == 0

    def _run_cached_prepare_bookkeeping(self, idt: wp.float32) -> None:
        """Apply cached contact warm-start when prepare data is reused."""
        if (
            self.mass_splitting_enabled
            or self._sleeping_enabled
            or self.num_particles > 0
            or self.num_cloth_triangles > 0
            or self.num_cloth_bending > 0
            or self.num_soft_tetrahedra > 0
            or self.num_soft_hexahedra > 0
        ):
            raise NotImplementedError(
                "cached prepare bookkeeping currently supports rigid contact/joint worlds "
                "without deformables, mass splitting, or sleeping"
            )
        cached_head, cached_fused = self._singleworld_cached_prepare_kernels()
        self._partitioner.begin_sweep()
        self._singleworld_head_plus_tail_sweep(cached_head, cached_fused, idt)

    def step(
        self,
        dt: float,
        contacts=None,
        shape_body=None,
        picking=None,
        vel_accum: wp.array[wp.vec3f] | None = None,
        omega_accum: wp.array[wp.vec3f] | None = None,
        shape_aabb_lower: wp.array[wp.vec3f] | None = None,
        shape_aabb_upper: wp.array[wp.vec3f] | None = None,
    ) -> None:
        """Advance the world by ``dt`` seconds.

        Phases: ingest contacts -> JP colouring -> substep loop (forces+gravity,
        main solve, integrate, relax) -> damping + inertia refresh -> clear forces.
        ``contacts`` requires non-disabled contact_matching ("sticky" recommended).

        ``shape_aabb_lower`` / ``shape_aabb_upper`` are read only when
        :attr:`sleeping_velocity_threshold` > 0. When ``None``, falls
        back to the arrays cached by :meth:`attach_collision_pipeline`
        (or :meth:`setup_cloth_collision_pipeline`); pass explicit
        arrays only if you need to point at a different pipeline.
        """
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        if dt < 1e-7:
            return
        if self._sleeping_enabled:
            if shape_aabb_lower is None:
                shape_aabb_lower = self._sleeping_shape_aabb_lower
            if shape_aabb_upper is None:
                shape_aabb_upper = self._sleeping_shape_aabb_upper

        self.step_dt = dt
        self.substep_dt = dt / self.substeps

        if self.enable_column_timers:
            self._zero_column_timers()

        self._ingest_and_warmstart_contacts(contacts, shape_body)
        if self._ingest_scratch is not None:
            # Contacts begin after the joint + cloth-tri + cloth-bending
            # + soft-tet blocks in the cid space. Contact cids are compacted
            # every step, so use the stable shape-pair key for contact
            # priority tie-breaks; otherwise lower-world contact count changes
            # perturb colouring and PGS order in later worlds.
            self._partitioner.set_costs_from_contact_pairs(
                self._contact_offset,
                self._ingest_scratch.num_contact_columns,
                self._contact_cols,
                self._ingest_scratch.pair_source_idx,
                self._ingest_scratch.pair_shape_a,
                self._ingest_scratch.pair_shape_b,
            )

        self._rebuild_elements()
        # Kinematic prepare BEFORE the sleeping pass: the per-island
        # score kernel reads ``bodies.velocity`` and must see the
        # pose-target-derived velocity for kinematic movers, otherwise a
        # kinematic body moving into a sleeping island can't lift the
        # score above threshold and the island never wakes.
        self._kinematic_prepare_step()
        if self._sleeping_enabled:
            self._run_sleeping_pass(shape_body, shape_aabb_lower, shape_aabb_upper)
        if self._constraint_capacity > 0:
            self._partitioner.reset(self._elements, self._num_active_constraints)
            if self.step_layout == "single_world":
                compute_family_starts = self._singleworld_needs_family_starts()
                if self.partitioner_algorithm == "greedy" and self._use_greedy_coloring:
                    # In-graph JP fallback if greedy's 64-colour bitmask overflows.
                    self._partitioner.build_csr_greedy_with_jp_fallback(compute_family_starts=compute_family_starts)
                else:
                    self._partitioner.build_csr(compute_family_starts=compute_family_starts)
            else:
                self._build_per_world_coloring()

            if self._tpw_auto and self.step_layout != "single_world":
                self._pick_tpw()

            # Per-step setup that depends on the just-built CSR. The
            # dispatcher rebuilds the mass-splitting interaction graph
            # here (no-op when mass splitting is disabled).
            self._dispatcher.begin_step()

        # Substep order: bias-on solve -> integrate -> bias-off relax. Reversing
        # would discard the positional bias's penetration recovery.
        inv_n = 1.0 / float(self.substeps)
        for k in range(self.substeps):
            self._current_substep_index = k
            if picking is not None:
                picking.apply_force()
            self._integrate_forces_and_gravity()
            # TGS-soft (Box2D-v3) substep order: solve-with-bias ->
            # integrate -> relax (bias=False). Reversing regresses
            # stacking / friction tests. Dispatcher owns:
            #   - solve(): pre-substep mass-splitting broadcast +
            #     prepare/iterate sweeps + post-solve writeback.
            #   - relax(): bias-off relax sweeps + post-relax writeback.
            # integrate_positions stays in step() -- identical for every
            # dispatcher.
            idt = wp.float32(1.0 / self.substep_dt)
            if self.articulation_dvi_host and self.articulation_dvi_replaces_joint_pgs:
                self._solve_articulations_dvi_host_for_step()
            self._dispatcher.solve(idt)
            if self.articulation_dvi_host and not self.articulation_dvi_replaces_joint_pgs:
                self._solve_articulations_dvi_host_for_step()
            self._integrate_positions()
            self._dispatcher.relax(idt)
            # Flip cloth particles' POSITION_LEVEL writes back to
            # VELOCITY_LEVEL. No-op for STATIC particles and rigid-only
            # scenes (num_particles == 0).
            self._recover_particle_velocities()
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

    def _ingest_and_warmstart_contacts(self, contacts, shape_body) -> None:
        """Translate Newton ``Contacts`` -> contact columns. Swap prev/current
        per-cid state, ingest -> warm-start -> forward-map stamp, fuse counts."""
        if contacts is None or self.max_contact_columns == 0 or self._ingest_scratch is None:
            self._num_active_constraints.fill_(self._contact_offset)
            self._contact_views = None
            self._has_soft_contact_pd = False
            return

        if getattr(contacts, "contact_matching", False) is False:
            raise ValueError('PhoenX requires Contacts with non-disabled contact_matching (use "sticky").')
        if shape_body is None:
            shape_body = self._shape_body_internal
        # When the cloth-aware pipeline is active, contact slots can
        # reference shape indices up to S + T. Use the PhoenX-indexed
        # full-length map stamped by setup_cloth_collision_pipeline.
        # Fall back to the pipeline map only for legacy callers; that
        # map remains Newton-indexed for narrow-phase contact generation.
        if (
            self._collision_pipeline is not None
            and getattr(self._collision_pipeline, "unified_shape_body", None) is not None
        ):
            if self._shape_body_internal is not None:
                shape_body = self._shape_body_internal
            else:
                shape_body = self._collision_pipeline.unified_shape_body
        if shape_body is None:
            raise ValueError(
                "step(contacts=...) requires shape_body. Pass model.shape_body or "
                "register shapes via WorldBuilder.add_shape_*."
            )

        # Soft-contact arrays are optional; length-0 sentinel short-circuits
        # the per-contact shape check in the kernels.
        if self._soft_contact_sentinel is None:
            self._soft_contact_sentinel = wp.zeros(0, dtype=wp.float32, device=self.device)
        contact_stiffness_src = getattr(contacts, "rigid_contact_stiffness", None)
        contact_damping_src = getattr(contacts, "rigid_contact_damping", None)
        self._has_soft_contact_pd = (contact_stiffness_src is not None and int(contact_stiffness_src.shape[0]) > 0) or (
            contact_damping_src is not None and int(contact_damping_src.shape[0]) > 0
        )
        contact_stiffness = contact_stiffness_src if contact_stiffness_src is not None else self._soft_contact_sentinel
        contact_damping = contact_damping_src if contact_damping_src is not None else self._soft_contact_sentinel
        contact_friction = (
            contacts.rigid_contact_friction
            if getattr(contacts, "rigid_contact_friction", None) is not None
            else self._soft_contact_sentinel
        )

        wp.launch(
            _update_contact_generation_reuse_kernel,
            dim=1,
            inputs=[contacts.contact_generation],
            outputs=[self._reuse_contact_indices, self._last_contact_generation],
            device=self.device,
        )

        # Keep history pointers stable so one-step CUDA graphs replay correctly.
        contact_container_copy_current_to_prev(self._contact_container, device=self.device)
        wp.copy(
            self._cid_of_contact_prev,
            self._cid_of_contact_cur,
            count=int(self.rigid_contact_max),
        )
        if self._enable_body_pair_grouping and self._ingest_scratch.inv_sort_perm is not None:
            wp.copy(
                self._ingest_scratch.prev_inv_sort_perm,
                self._ingest_scratch.inv_sort_perm,
                count=int(self.rigid_contact_max),
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
            filter_keys=self._collision_filter_keys,
            filter_count=self._collision_filter_count,
            shape_material=self._shape_material,
            materials=self._materials,
            enable_body_pair_grouping=self._enable_body_pair_grouping,
            shape_filter_id=self._shape_filter_id,
            cid_of_contact=self._cid_of_contact_cur,
            num_active_constraints=self._num_active_constraints,
            active_constraint_base=self._contact_offset,
        )

        # Compound grouping: views point at PhoenX's sorted scratch.
        # Otherwise: views point at Newton's narrow-phase arrays directly.
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

        # match_index is already in prev-sorted-k for compound, Newton-order otherwise.
        gather_match_index = (
            self._ingest_scratch.sorted_match_index
            if self._enable_body_pair_grouping
            else contacts.rigid_contact_match_index
        )
        gather_contact_warmstart(
            scratch=self._ingest_scratch,
            rigid_contact_match_index=gather_match_index,
            prev_cid_of_contact=self._cid_of_contact_prev,
            reuse_contact_indices=self._reuse_contact_indices,
            bodies=self.bodies,
            contacts=self._contact_views,
            cc=self._contact_container,
            device=self.device,
            # Cross-frame contact impulses can keep a nearly quiet stack above
            # the sleep threshold; substep-local warm-start still applies.
            carry_impulses=not self._sleeping_enabled,
        )

        # Cloth-aware overlay: when shape_endpoints is populated
        # (i.e. setup_cloth_collision_pipeline was called), re-stamp
        # the contact column header with unified-index nodes + kind
        # tags, and compute barycentric weights for any cloth-side
        # contacts. Rigid-only scenes skip this -- the existing
        # _contact_pack_columns_kernel above already wrote correct
        # rigid-rigid headers.
        if self._shape_endpoints is not None:
            wp.launch(
                _phoenx_pack_cloth_contact_endpoints_kernel,
                dim=max(1, self.max_contact_columns),
                inputs=[
                    self._ingest_scratch.pair_source_idx,
                    self._ingest_scratch.pair_shape_a,
                    self._ingest_scratch.pair_shape_b,
                    self._ingest_scratch.num_contact_columns,
                    self._shape_endpoints,
                ],
                outputs=[self._contact_cols],
                device=self.device,
            )
            wp.launch(
                _phoenx_pack_cloth_contact_barycentric_kernel,
                dim=max(1, self.rigid_contact_max),
                inputs=[
                    self._contact_views,
                    self._shape_endpoints,
                    self._particles_or_sentinel(),
                    wp.int32(self.num_bodies),
                ],
                outputs=[self._contact_container],
                device=self.device,
            )

        stamp_forward_contact_map(
            cid_base=self._contact_offset,
            scratch=self._ingest_scratch,
            cid_of_contact=self._cid_of_contact_cur,
            device=self.device,
        )

    def _rebuild_elements(self) -> None:
        """Project active constraints into the partitioner's element view."""
        if self._constraint_capacity == 0:
            return
        wp.launch(
            _constraints_to_elements_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self._contact_container,
                self.bodies,
                self._particles_or_sentinel(),
                self._num_active_constraints,
                wp.int32(self.num_joints),
                wp.int32(self.num_cloth_triangles),
                wp.int32(self.num_cloth_bending),
                wp.int32(self.num_soft_tetrahedra),
                wp.int32(self.num_soft_hexahedra),
                wp.int32(self.num_bodies),
                self._elements,
                self._element_family,
            ],
            device=self.device,
        )

    def wake_on_external_input(self) -> None:
        """Wake every island whose bodies carry a user-applied force or
        torque, *before* ``model.collide()`` runs the broad phase.

        The per-step sleeping pass inside :meth:`step` cannot drive
        broad-phase decisions on the wake frame: by the time it clears
        ``island_root`` for a body that picking just pushed, the
        sleep-aware broad-phase filter has already dropped that body's
        contact pairs and the substep solve sees an empty stack. Call
        this between ``picking.apply_force()`` and ``model.collide()``
        in the host loop so the wake decision arrives in time.

        Uses ``bodies.island_root`` -- a stable body-id label (the
        lowest body id in the island at the moment of sleep transition)
        -- to propagate the wake through the full original island, so
        picking a single plank wakes every plank that shared a contact
        island with it at sleep time. A no-op when the sleeping
        pipeline is disabled (no allocation, no kernel launches).
        """
        if not self._sleeping_enabled or self.num_bodies == 0 or self._island_builder is None:
            return
        if self._island_has_external_input is None:
            return
        self._island_has_external_input.zero_()
        wp.launch(
            _phoenx_island_fanin_external_input_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            outputs=[self._island_has_external_input],
            device=self.device,
        )
        wp.launch(
            _phoenx_apply_island_wake_kernel,
            dim=self.num_bodies,
            inputs=[
                self.bodies,
                self._island_has_external_input,
            ],
            device=self.device,
        )

    def _run_sleeping_pass(
        self,
        shape_body: wp.array[wp.int32] | None,
        shape_aabb_lower: wp.array[wp.vec3f] | None,
        shape_aabb_upper: wp.array[wp.vec3f] | None,
    ) -> None:
        """Detect, mark, and wake sleeping islands. Sequence:

        1. AABB fanin (per-body union of attached shape AABBs).
        2. Self-wake: sleeping bodies whose own predicted score is high
           (host-side velocity / force injection) flag their root, and
           the apply kernel clears ``island_root`` for the whole group.
        3. Detect *active* sleeping islands -- any sleeping body sharing
           a constraint with an awake dynamic body marks its island as
           active for this frame.
        4. Inject chain edges ``(body, island_root)`` for every body
           in an active island so the awake bridging body pulls the
           entire sleeping island back into the live union-find.
        5. Copy ``_elements`` -> 2D union-find buffer (inactive
           sleeping bodies filtered to -1; active sleeping bodies
           kept).
        6. Build islands (regular elements + chain edges).
        7. Per-compact-island lowest awake body id, for the stamp.
        8. Per-island max velocity, mark sleeping.
        9. Propagate: sleeping body in awake compact island -> wake
           (clear root); awake body whose island stayed below threshold
           long enough -> stamp ``island_root``.
        10. Collapse sleeping body slots to -1 in ``_elements`` so the
            partitioner drops them from coloring / overflow.
        """
        if self._constraint_capacity == 0 or self.num_bodies == 0:
            return
        if self._island_builder is None:
            return
        if shape_body is None or shape_aabb_lower is None or shape_aabb_upper is None:
            # No broad-phase AABBs available -- can't compute spin score.
            return

        self._island_max_velocity.zero_()
        if self._wake_flag is not None:
            self._wake_flag.zero_()
        if self._island_active is not None:
            self._island_active.zero_()
        if self._island_root_per_compact is not None:
            self._island_root_per_compact.fill_(self.num_bodies)

        # --- Per-body union AABB (for the spin term of the score). ---
        wp.launch(
            _phoenx_init_body_aabb_kernel,
            dim=self.num_bodies,
            outputs=[self._body_aabb_lower, self._body_aabb_upper],
            device=self.device,
        )
        # Cap to shape_body length: cloth setups pad with cloth-tri /
        # soft-tet suffix shapes not covered by the rigid-only array.
        num_shapes = min(int(shape_aabb_lower.shape[0]), int(shape_body.shape[0]))
        if num_shapes > 0:
            wp.launch(
                _phoenx_shape_aabb_fanin_kernel,
                dim=num_shapes,
                inputs=[shape_aabb_lower, shape_aabb_upper, shape_body],
                outputs=[self._body_aabb_lower, self._body_aabb_upper],
                device=self.device,
            )
        wp.launch(
            _phoenx_finalize_body_aabb_diagonal_kernel,
            dim=self.num_bodies,
            inputs=[self._body_aabb_lower, self._body_aabb_upper],
            outputs=[self._body_aabb_diagonal],
            device=self.device,
        )

        # --- Self-wake: velocity / force injection. ---
        if self._wake_flag is not None:
            wp.launch(
                _phoenx_self_wake_fanin_kernel,
                dim=self.num_bodies,
                inputs=[
                    self.bodies,
                    self._body_aabb_diagonal,
                    wp.float32(self._sleeping_velocity_threshold),
                    wp.float32(self.step_dt),
                ],
                outputs=[self._wake_flag],
                device=self.device,
            )
            wp.launch(
                _phoenx_apply_wake_flag_kernel,
                dim=self.num_bodies,
                inputs=[self.bodies, self._wake_flag],
                device=self.device,
            )

        # --- Detect active sleeping islands + inject chain edges. ---
        wp.launch(
            _phoenx_detect_active_islands_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.bodies,
                self._elements,
                self._num_active_constraints,
                wp.int32(self.num_bodies),
            ],
            outputs=[self._island_active],
            device=self.device,
        )

        # Seed the UF counter from the regular-element count so chain
        # edges atomically claim slots past it.
        wp.launch(
            _phoenx_seed_uf_num_interactions_kernel,
            dim=1,
            inputs=[self._num_active_constraints],
            outputs=[self._uf_num_interactions],
            device=self.device,
        )

        # --- Copy elements into the 2D UF input. ---
        # Note: the kernel writes -1 to slots past num_active_constraints.
        # Chain-edge injection runs *after* this, atomically claiming
        # slots past num_active_constraints and overwriting the -1s.
        wp.launch(
            _phoenx_copy_elements_to_int2d_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self._elements,
                self._num_active_constraints,
                wp.int32(self.num_bodies),
                self.bodies,
                self._island_active,
            ],
            outputs=[self._island_interaction_bodies],
            device=self.device,
        )

        wp.launch(
            _phoenx_inject_chain_edges_kernel,
            dim=self.num_bodies,
            inputs=[
                self.bodies,
                self._island_active,
                wp.int32(self._island_interaction_bodies.shape[0]),
            ],
            outputs=[self._island_interaction_bodies, self._uf_num_interactions],
            device=self.device,
        )

        # --- Build islands. Single UF pass over (elements + chain edges). ---
        self._island_builder.build_islands(
            self._island_interaction_bodies,
            self._uf_num_interactions,
            self._sleeping_num_bodies_device,
        )

        # --- Per-compact lowest awake body id (for the stamp). ---
        wp.launch(
            _phoenx_compute_island_root_per_compact_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self._island_builder.set_nr],
            outputs=[self._island_root_per_compact],
            device=self.device,
        )

        # --- Max velocity + sleeping mark + propagate. ---
        wp.launch(
            _phoenx_island_max_velocity_kernel,
            dim=self.num_bodies,
            inputs=[
                self.bodies,
                self._body_aabb_diagonal,
                self._island_builder.set_nr,
                wp.float32(self.step_dt),
                wp.float32(self._sleeping_velocity_threshold),
            ],
            outputs=[self._island_max_velocity],
            device=self.device,
        )
        wp.launch(
            _phoenx_mark_sleeping_islands_kernel,
            dim=self.num_bodies,
            inputs=[
                self._island_max_velocity,
                self._island_builder.num_sets,
                wp.float32(self._sleeping_velocity_threshold),
            ],
            outputs=[self._island_is_sleeping],
            device=self.device,
        )
        wp.launch(
            _phoenx_propagate_sleep_to_bodies_kernel,
            dim=self.num_bodies,
            inputs=[
                self.bodies,
                self._island_builder.set_nr,
                self._island_is_sleeping,
                self._island_root_per_compact,
                wp.int32(self._sleeping_frames_required),
            ],
            device=self.device,
        )

        # --- Collapse sleeping body slots in _elements for the partitioner. ---
        wp.launch(
            _phoenx_collapse_sleeping_elements_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.bodies,
                self._num_active_constraints,
                wp.int32(self.num_bodies),
            ],
            outputs=[self._elements],
            device=self.device,
        )

    def _mass_splitting_broadcast(self) -> None:
        """Fan body / particle state into every copy-state slot at substep
        start. Once-per-substep cost; no-op when ``highest_index_in_use[0]
        == 0`` (the kernel short-circuits)."""
        launch_broadcast_rigid_to_copy_states(
            self._copy_state,
            self.bodies,
            self._particles_or_sentinel(),
            num_bodies=self.num_bodies,
            dt=self.substep_dt,
        )

    def _mass_splitting_average_and_broadcast(self, inv_dt: float | None = None) -> None:
        """Merge divergent mass-splitting slots after one PGS sweep."""
        if inv_dt is None:
            inv_dt = 1.0 / self.substep_dt
        if self._mass_splitting_velocity_only_average:
            launch_average_and_broadcast_rigid_velocity(
                self._copy_state,
                self.bodies,
                self._particles_or_sentinel(),
                num_bodies=self.num_bodies,
                inv_dt=inv_dt,
            )
            return
        if self._mass_splitting_grouped_average:
            launch_average_and_broadcast_grouped(
                self._copy_state,
                self.bodies,
                self._particles_or_sentinel(),
                num_bodies=self.num_bodies,
                inv_dt=inv_dt,
            )
            return
        launch_average_and_broadcast(
            self._copy_state,
            self.bodies,
            self._particles_or_sentinel(),
            num_bodies=self.num_bodies,
            inv_dt=inv_dt,
        )

    def _mass_splitting_writeback(self, *, already_averaged: bool = False) -> None:
        """Write each body / particle's slot-0 velocity back to storage.

        Unless ``already_averaged`` is true, first merge divergent
        overflow-bucket slots with the mass-splitting average pass.
        Dispatchers pass ``already_averaged=True`` only when the
        immediately preceding operation was that same average pass. The
        writeback kernel still synchronizes slot 0 to velocity level, so
        single-slot position-level writes are preserved.
        """
        inv_dt = 1.0 / self.substep_dt
        if not already_averaged:
            self._mass_splitting_average_and_broadcast(inv_dt)
        launch_copy_state_into_rigids(
            self._copy_state,
            self.bodies,
            self._particles_or_sentinel(),
            num_bodies=self.num_bodies,
            inv_dt=inv_dt,
        )

    def _rebuild_mass_splitting_graph(self) -> None:
        """Per-step emit + build of the (body, partition_key) interaction
        graph used by the copy-state read/write helpers.

        Reads the partitioner's current CSR layout
        (``element_ids_by_color`` / ``color_starts`` /
        ``interaction_id_to_partition``) and writes
        :attr:`_copy_state`'s ``section_end`` / ``partition_list`` /
        ``highest_index_in_use`` arrays.

        Single-world only. Graph-capture safe.
        """
        # The emit kernel atomically appends one entry per non-static
        # endpoint; the previous step's build call has already left
        # ``scratch.num_pairs`` at 0 so the launch picks up clean.
        wp.launch(
            record_all_interactions_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self._elements,
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._num_active_constraints,
                self._partitioner.interaction_id_to_partition,
                wp.int32(int(self.max_colored_partitions)),
                wp.int32(int(self.mass_splitting_batch_size)),
                self._interaction_graph_scratch,
            ],
            device=self.device,
        )
        # Sort + dedup + sections; writes back into self._copy_state.
        build_interaction_graph(self._interaction_graph_scratch, self._copy_state)
        # Per-cid slot / count cache for the iterate hot path. Reads
        # ``copy_state.section_end`` / ``partition_list`` (just stamped)
        # plus the partitioner's CSR; writes ``constraints.slot_cache``
        # and ``constraints.count_cache``. Constraint iterates then
        # bypass :func:`get_state_index` on the inner loop.
        build_constraint_slot_cache(
            self._partitioner.element_ids_by_color,
            self._partitioner.color_starts,
            self._num_active_constraints,
            self._copy_state,
            self.constraints,
            self._contact_cols,
            self._contact_offset,
            max_colored_partitions=int(self.max_colored_partitions),
            ms_batch_size=int(self.mass_splitting_batch_size),
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
                self._partitioner._packed_priorities,
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
        """Parallel per-world JP coloring: count -> scan -> stable bucket sort
        -> one block per world runs JP MIS. Reuses partitioner adjacency CSR
        (worlds are disjoint after static-null-out)."""
        nw = self.num_worlds
        cap = self._constraint_capacity

        self._per_world_element_count.zero_()
        self._world_totals_shifted.zero_()
        wp.launch(
            _count_elements_per_world_kernel,
            dim=cap,
            inputs=[
                self._elements,
                self._num_active_constraints,
                self.bodies,
                self._particle_world_id,
                wp.int32(self.num_bodies),
            ],
            outputs=[self._per_world_element_count, self._world_totals_shifted],
            device=self.device,
        )

        wp.utils.array_scan(self._world_totals_shifted, self._per_world_element_offsets, inclusive=True)

        # Stable sort scatter (deterministic; replaces atomic-cursor scatter).
        wp.launch(
            _build_scatter_keys_kernel,
            dim=cap,
            inputs=[
                self._elements,
                self._num_active_constraints,
                self.bodies,
                self._particle_world_id,
                wp.int32(self.num_bodies),
                wp.int32(cap),
            ],
            outputs=[self._per_world_scatter_keys, self._per_world_elements],
            device=self.device,
        )
        sort_variable_length_int(
            self._per_world_scatter_keys,
            self._per_world_elements,
            self._num_active_constraints,
        )

        # Per-world JP/greedy clears assigned flags for each active world.
        wp.copy(self._world_csr_offsets, self._per_world_element_offsets)
        if self._use_greedy_coloring:
            self._per_world_greedy_overflow.zero_()
            wp.launch_tiled(
                _per_world_greedy_coloring_kernel,
                dim=[nw],
                inputs=[
                    self._per_world_element_offsets,
                    self._per_world_element_count,
                    self._per_world_elements,
                    self._elements,
                    self._element_family,
                    self._partitioner._adjacency_section_end_indices,
                    self._partitioner._vertex_to_adjacent_elements,
                    self._partitioner._packed_priorities,
                    int(GREEDY_MAX_COLORS),
                ],
                outputs=[
                    self._per_world_assigned,
                    self._per_world_greedy_color_count,
                    self._per_world_greedy_color_offsets,
                    self._per_world_greedy_color_family_count,
                    self._per_world_greedy_color_family_offsets,
                    self._world_element_ids_by_color,
                    self._world_color_starts,
                    self._world_color_family_starts,
                    self._world_num_colors,
                    self._per_world_greedy_overflow,
                ],
                block_dim=_PER_WORLD_COLORING_BLOCK_DIM,
                device=self.device,
            )
            self._maybe_fallback_from_per_world_greedy_overflow(nw)
        else:
            # Round-equals-colour JP. Cost-biased priorities: contacts use
            # contact_count, joints stay at cost 0.
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
                    self._partitioner._packed_priorities,
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
        """Apply force/torque + gravity in one per-substep launch.

        Bodies and particles share the same substep-entry semantics:
        snapshot pose into ``*_prev_substep``, set ``access_mode``,
        apply gravity. The two launches are independent and can fuse
        in CUDA-graph capture.
        """
        if self.num_bodies > 0:
            wp.launch(
                _phoenx_apply_forces_and_gravity_kernel,
                dim=self.num_bodies,
                inputs=[self.bodies, self.gravity, wp.float32(self.substep_dt)],
                device=self.device,
            )
        if self.num_particles > 0 and self.particles is not None:
            wp.launch(
                cloth_predict_kernel,
                dim=self.num_particles,
                inputs=[self.particles, self._particle_world_id, self.gravity, wp.float32(self.substep_dt)],
                device=self.device,
            )

    def _recover_particle_velocities(self) -> None:
        """Substep exit: flip every particle to ``VELOCITY_LEVEL``,
        which finite-diffs the position delta into velocity."""
        if self.num_particles == 0 or self.particles is None:
            return
        wp.launch(
            cloth_recover_kernel,
            dim=self.num_particles,
            inputs=[self.particles, wp.float32(1.0 / self.substep_dt)],
            device=self.device,
        )

    def _solve_main(self) -> None:
        """Per-substep main PGS solve: prepare + solver_iterations iterate sweeps
        with bias=True (fused into one launch)."""
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._active_contact_views()
        cached_prepare = not self._refresh_prepare_this_substep()
        for fixed_tpw in self._fast_tail_auto_fixed_choices():
            kernel = get_fast_tail_kernel(
                kind="prepare_plus_iterate",
                **self._fast_tail_kernel_flags(fixed_tpw, cached_prepare=bool(cached_prepare)),
            )
            wp.launch(
                kernel,
                dim=self._fast_tail_launch_dim_for(fixed_tpw if fixed_tpw > 0 else self._tpw_launch_bound),
                block_dim=self._fast_tail_block_dim(),
                inputs=[
                    self.constraints,
                    self._contact_cols,
                    self.bodies,
                    self._particles_or_sentinel(),
                    idt,
                    wp.float32(self.sor_boost),
                    self._world_element_ids_by_color,
                    self._world_color_starts,
                    self._world_color_family_starts,
                    self._world_csr_offsets,
                    self._world_num_colors,
                    self._contact_container,
                    contact_views,
                    wp.int32(self.solver_iterations),
                    wp.int32(self.num_worlds),
                    wp.int32(self.num_joints),
                    self._joint_pgs_enabled,
                    wp.int32(self.num_cloth_triangles),
                    wp.int32(self.num_cloth_bending),
                    wp.int32(self.num_soft_tetrahedra),
                    wp.int32(self.num_soft_hexahedra),
                    wp.int32(self.num_bodies),
                    self._tpw_choice,
                    self._copy_state,
                ],
                device=self.device,
            )

    def _relax_velocities(self) -> None:
        """TGS-soft relax (bias=False) — removes drift velocity from main bias."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._active_contact_views()
        for fixed_tpw in self._fast_tail_auto_fixed_choices():
            kernel = get_fast_tail_kernel(
                kind="relax",
                **self._fast_tail_kernel_flags(fixed_tpw),
            )
            self._launch_fast_iter(
                kernel,
                self.velocity_iterations,
                idt,
                contact_views,
                launch_tpw_bound=fixed_tpw if fixed_tpw > 0 else self._tpw_launch_bound,
            )

    def _parse_multi_world_scheduler_policy(self, policy: str) -> tuple[str, int]:
        """Normalize a construction-time multi-world scheduler policy."""
        normalized = str(policy).strip().lower().replace("-", "_")
        if normalized in ("auto", "fast_tail"):
            return normalized, 128
        if normalized == "fasttail":
            return "fast_tail", 128
        if normalized == "block_world":
            return "block_world", 128

        prefix = "block_world_"
        if normalized.startswith(prefix):
            block_dim = self._validate_block_world_dim(int(normalized[len(prefix) :]))
            return "block_world", block_dim

        raise ValueError(
            "multi_world_scheduler must be 'auto', 'fast_tail', 'block_world', "
            "'block_world_32', 'block_world_64', or 'block_world_128' "
            f"(got {policy!r})"
        )

    def _auto_multi_world_scheduler(self) -> tuple[str, int]:
        """Choose a fixed scheduler from construction-time topology."""
        return _choose_multi_world_scheduler(
            block_world_supported=self._block_world_supported(),
            num_worlds=self.num_worlds,
            num_joints=self.num_joints,
            max_contact_columns=self.max_contact_columns,
        )

    def _configure_multi_world_scheduler(self, policy: str) -> None:
        """Resolve the multi-world scheduler once before any graph capture."""
        kind, block_dim = self._parse_multi_world_scheduler_policy(policy)
        if kind == "auto":
            kind, block_dim = self._auto_multi_world_scheduler()
        elif kind == "block_world" and not self._block_world_supported():
            raise NotImplementedError("block-world scheduler currently supports rigid multi-world scenes only")

        self._multi_world_scheduler_policy = str(policy)
        self._multi_world_scheduler = kind
        self._multi_world_block_dim = self._validate_block_world_dim(block_dim)

    def _block_world_supported(self) -> bool:
        """Return whether the private block-world scheduler can run this scene."""
        return bool(
            self.step_layout != "single_world"
            and not self.mass_splitting_enabled
            and self.num_particles == 0
            and self.num_cloth_triangles == 0
            and self.num_cloth_bending == 0
            and self.num_soft_tetrahedra == 0
            and self.num_soft_hexahedra == 0
            and self._contact_offset == self.num_joints
        )

    def _block_world_launch_dim(self, block_dim: int) -> int:
        """Launch one physical block per world for the block-world scheduler."""
        return max(1, int(self.num_worlds) * int(block_dim))

    def _validate_block_world_dim(self, block_dim: int) -> int:
        block_dim = int(block_dim)
        if block_dim not in (32, 64, 128):
            raise ValueError(f"block_world block_dim must be one of (32, 64, 128), got {block_dim}")
        return block_dim

    def _solve_main_block_world(self, block_dim: int | None = None) -> None:
        """Private multi-world PGS solve using one physical block per world."""
        if self._constraint_capacity == 0:
            return
        if not self._block_world_supported():
            raise NotImplementedError("block-world scheduler currently supports rigid multi-world scenes only")
        block_dim = self._validate_block_world_dim(self._multi_world_block_dim if block_dim is None else block_dim)
        contact_views = self._active_contact_views()
        cached_prepare = not self._refresh_prepare_this_substep()
        kernel = get_block_world_kernel(
            kind="prepare_plus_iterate",
            **self._block_world_kernel_flags(block_dim, cached_prepare=bool(cached_prepare)),
        )
        wp.launch(
            kernel,
            dim=self._block_world_launch_dim(block_dim),
            block_dim=block_dim,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._particles_or_sentinel(),
                wp.float32(1.0 / self.substep_dt),
                wp.float32(self.sor_boost),
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(self.solver_iterations),
                wp.int32(self.num_worlds),
                wp.int32(self.num_joints),
                self._joint_pgs_enabled,
                wp.int32(self.num_bodies),
                self._copy_state,
            ],
            device=self.device,
        )

    def _relax_velocities_block_world(self, block_dim: int | None = None) -> None:
        """Private multi-world TGS-soft relax using one physical block per world."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        if not self._block_world_supported():
            raise NotImplementedError("block-world scheduler currently supports rigid multi-world scenes only")
        block_dim = self._validate_block_world_dim(self._multi_world_block_dim if block_dim is None else block_dim)
        contact_views = self._active_contact_views()
        kernel = get_block_world_kernel(
            kind="relax",
            **self._block_world_kernel_flags(block_dim),
        )
        wp.launch(
            kernel,
            dim=self._block_world_launch_dim(block_dim),
            block_dim=block_dim,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._particles_or_sentinel(),
                wp.int32(self.num_bodies),
                wp.float32(1.0 / self.substep_dt),
                wp.float32(self.sor_boost),
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(self.velocity_iterations),
                wp.int32(self.num_worlds),
                wp.int32(self.num_joints),
                self._joint_pgs_enabled,
                self._copy_state,
            ],
            device=self.device,
        )

    # Single-world dispatch (wp.capture_while over the global colour CSR).

    def _launch_singleworld_head(
        self,
        kernel,
        idt: wp.float32,
        fuse_threshold: wp.int32,
    ) -> None:
        """Launch the persistent-grid single-world head kernel."""
        contact_views = self._active_contact_views()
        ms_cap = wp.int32(-1 if self.max_colored_partitions is None else int(self.max_colored_partitions))
        ms_batch = wp.int32(int(self.mass_splitting_batch_size))
        wp.launch(
            kernel,
            dim=self._singleworld_total_threads,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._particles_or_sentinel(),
                idt,
                wp.float32(self.sor_boost),
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.color_family_starts,
                self._partitioner.num_colors,
                self._partitioner.color_cursor,
                self._contact_container,
                contact_views,
                wp.int32(self.num_joints),
                self._joint_pgs_enabled,
                wp.int32(self.num_cloth_triangles),
                wp.int32(self.num_cloth_bending),
                wp.int32(self.num_soft_tetrahedra),
                wp.int32(self.num_soft_hexahedra),
                wp.int32(self.num_bodies),
                wp.int32(self._singleworld_total_threads),
                fuse_threshold,
                self._head_active,
                self._copy_state,
                ms_cap,
                ms_batch,
                self._partitioner.sweep_direction,
            ],
            block_dim=_SINGLEWORLD_BLOCK_DIM,
            device=self.device,
        )

    def _capture_singleworld_sweep(self, kernel, **kw) -> None:
        """capture_while body: head-path sweep on the persistent grid, unrolled
        NUM_INNER_WHILE_ITERATIONS times. Tail launches no-op once head_active
        clears within the same outer iter."""
        idt = kw.get("idt", wp.float32(0.0))
        fuse_threshold = wp.int32(self._fuse_threshold)
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            self._launch_singleworld_head(kernel, idt, fuse_threshold)

    def _capture_singleworld_tail_sweep(self, kernel, **kw) -> None:
        """capture_while body: drain remaining small colours in one block via
        wp.launch_tiled. Hands back to the head path on a colour > fuse_threshold."""
        contact_views = self._active_contact_views()
        idt = kw.get("idt", wp.float32(0.0))
        ms_cap = wp.int32(-1 if self.max_colored_partitions is None else int(self.max_colored_partitions))
        ms_batch = wp.int32(int(self.mass_splitting_batch_size))
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._particles_or_sentinel(),
                idt,
                wp.float32(self.sor_boost),
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.color_family_starts,
                self._partitioner.num_colors,
                self._partitioner.color_cursor,
                self._contact_container,
                contact_views,
                wp.int32(self.num_joints),
                self._joint_pgs_enabled,
                wp.int32(self.num_cloth_triangles),
                wp.int32(self.num_cloth_bending),
                wp.int32(self.num_soft_tetrahedra),
                wp.int32(self.num_soft_hexahedra),
                wp.int32(self.num_bodies),
                wp.int32(self._fuse_threshold),
                self._copy_state,
                ms_cap,
                ms_batch,
                self._partitioner.sweep_direction,
                self._head_active,
            ],
            block_dim=self._fuse_tail_block_dim,
            device=self.device,
        )

    def _singleworld_head_plus_tail_sweep(self, head_kernel, tail_kernel, idt: wp.float32) -> None:
        """Persistent-grid head + fused single-block tail.

        Outer ``wp.capture_while`` on ``color_cursor`` so head and tail
        alternate until every colour is drained. Each round: re-arm
        ``head_active``, run the head sweep (drains large colours;
        bails when ``count <= fuse_threshold``), then run a single
        tail launch (the tail kernel's internal ``while cursor > 0``
        walks every remaining small colour). The tail bails back when
        a colour grows past ``fuse_threshold`` (a sweep-time stack
        change can flip the size of an upcoming colour), so the outer
        loop re-arms ``head_active`` and lets head pick that colour
        up; with the fused-tail overflow fix the tail decrements past
        the mass-splitting overflow column the same way it decrements
        any small column, so neither side can leave ``color_cursor``
        stuck without progress.

        The previous version used two sequential ``wp.capture_while``
        (head on ``head_active``, then tail on ``color_cursor``) and
        deadlocked whenever the tail bailed without decrementing the
        cursor: capture re-launched the tail, tail bailed again,
        ping-pong forever. Reproducer:
        ``example_cloth_hanging`` with mass splitting enabled, on the
        first frame where the cube contacts the cloth.
        """

        # ``head_active`` is re-armed by the previous round's tail kernel
        # (see :func:`_make_singleworld_fused_kernel`), so the per-round
        # ``_reset_head_active_kernel`` launch is gone. Initial state is
        # set to 1 once at solver setup; subsequent rounds inherit
        # ``head_active[0] = 1`` from the tail's lane-0 writeback.
        def _round() -> None:
            wp.capture_while(
                self._head_active,
                self._capture_singleworld_sweep,
                kernel=head_kernel,
                idt=idt,
            )
            self._capture_singleworld_tail_sweep(kernel=tail_kernel, idt=idt)

        wp.capture_while(self._partitioner.color_cursor, _round)

    def _solve_main_singleworld(self) -> None:
        """Single-world prepare + main PGS iterate. Each sweep is head (large
        colours) then fused tail (small colours).

        When mass splitting is enabled, an
        :func:`launch_average_and_broadcast` runs after every full
        color sweep (prepare and each iterate iteration). This is the
        C# ``MassSplitting.RunMethodParallelIterate`` +
        ``AverageAndBroadcast`` pattern: every iteration leaves
        divergent slots merged back to their mean, so the next
        iteration starts from a consistent state and Jacobi-block
        convergence on the overflow bucket actually converges.
        """
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)

        prepare_head, prepare_fused, iterate_head, iterate_fused, _, _ = self._singleworld_kernels()

        if self._refresh_prepare_this_substep():
            self._partitioner.begin_sweep()
            self._singleworld_head_plus_tail_sweep(prepare_head, prepare_fused, idt)
            if self.mass_splitting_enabled:
                # Prepare applies the warm-start impulse to each body's
                # slots. Average it so the iterate phase starts from
                # converged slot values.
                self._mass_splitting_average_and_broadcast(1.0 / self.substep_dt)
        else:
            self._run_cached_prepare_bookkeeping(idt)

        for _ in range(self.solver_iterations):
            self._partitioner.begin_sweep()
            self._singleworld_head_plus_tail_sweep(iterate_head, iterate_fused, idt)
            if self.mass_splitting_enabled:
                self._mass_splitting_average_and_broadcast(1.0 / self.substep_dt)

    def _relax_velocities_singleworld(self) -> None:
        """Single-world TGS-soft relax sweeps (bias OFF)."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        _, _, _, _, relax_head, relax_fused = self._singleworld_kernels()
        for _ in range(self.velocity_iterations):
            self._partitioner.begin_sweep()
            self._singleworld_head_plus_tail_sweep(relax_head, relax_fused, idt)
            if self.mass_splitting_enabled:
                self._mass_splitting_average_and_broadcast(1.0 / self.substep_dt)

    def _selective_joint_pgs_enabled(self) -> bool:
        """Return whether PGS should skip only DVI-owned joint columns."""
        if not (self.articulation_dvi_host and self.articulation_dvi_replaces_joint_pgs):
            return False
        if self.articulation_dvi_joint_mask is None:
            return False
        return self.num_joints > 0 and not bool(self.articulation_dvi_joint_mask.all())

    def _skip_all_joint_pgs(self) -> bool:
        """Return whether every joint column is owned by DVI."""
        return bool(
            self.articulation_dvi_host
            and self.articulation_dvi_replaces_joint_pgs
            and not self._selective_joint_pgs_enabled()
        )

    def _dispatch_specialization_flags(self) -> dict[str, bool]:
        """Static dispatch axes shared by single-world and fast-tail kernels."""
        cloth_on = (
            self.num_cloth_triangles > 0
            or self.num_soft_tetrahedra > 0
            or self.num_cloth_bending > 0
            or self.num_soft_hexahedra > 0
        )
        return {
            "revolute_only": bool(self._use_revolute_specialization),
            "cloth_support": cloth_on,
            "soft_tet_neohookean": bool(self._soft_tet_uses_neohookean),
            "enable_column_timers": self.enable_column_timers,
            "has_joints": self.num_joints > 0,
            "skip_joint_pgs": self._skip_all_joint_pgs(),
            "selective_joint_pgs": self._selective_joint_pgs_enabled(),
            "has_sleeping": self._sleeping_enabled,
            "has_soft_contact_pd": bool(self._has_soft_contact_pd),
        }

    def _fast_tail_kernel_flags(self, fixed_tpw: int, *, cached_prepare: bool | None = None) -> dict[str, object]:
        """Factory flags for multi-world fast-tail kernels."""
        joint_sweeps, contact_sweeps, outer_chunk = _choose_fast_tail_solve_schedule(substeps=self.substeps)
        kw: dict[str, object] = {
            **self._dispatch_specialization_flags(),
            "has_contacts": self.max_contact_columns > 0,
            "fixed_tpw": int(fixed_tpw),
            "guard_tpw": self._tpw_auto,
            "family_split": self._fast_tail_family_split(),
            "solve_joint_inner_sweeps": joint_sweeps,
            "solve_contact_inner_sweeps": contact_sweeps,
            "solve_outer_iteration_chunk": outer_chunk,
        }
        if cached_prepare is not None:
            kw["cached_prepare"] = bool(cached_prepare)
        return kw

    def _block_world_kernel_flags(self, block_dim: int, *, cached_prepare: bool | None = None) -> dict[str, object]:
        """Factory flags for rigid block-world kernels."""
        dispatch_kw = self._dispatch_specialization_flags()
        kw: dict[str, object] = {
            "revolute_only": dispatch_kw["revolute_only"],
            "has_joints": dispatch_kw["has_joints"],
            "has_contacts": self.max_contact_columns > 0,
            "skip_joint_pgs": dispatch_kw["skip_joint_pgs"],
            "selective_joint_pgs": dispatch_kw["selective_joint_pgs"],
            "has_sleeping": dispatch_kw["has_sleeping"],
            "has_soft_contact_pd": dispatch_kw["has_soft_contact_pd"],
            "enable_column_timers": dispatch_kw["enable_column_timers"],
            "block_dim": int(block_dim),
        }
        if cached_prepare is not None:
            kw["cached_prepare"] = bool(cached_prepare)
        return kw

    def _singleworld_kernels(self):
        """Return ``(prepare_head, prepare_fused, iterate_head,
        iterate_fused, relax_head, relax_fused)``. Specialised via
        compile-time ``revolute_only``, ``cloth_support``, and the
        scene-wide soft-tet variant."""
        kw = {
            **self._dispatch_specialization_flags(),
            "has_contacts": self.max_contact_columns > 0,
            "has_mass_splitting": self.mass_splitting_enabled,
            "rigid_direct": self._singleworld_rigid_direct(),
        }
        return (
            get_singleworld_kernel(phase="prepare", fused=False, **kw),
            get_singleworld_kernel(phase="prepare", fused=True, **kw),
            get_singleworld_kernel(phase="iterate", fused=False, **kw),
            get_singleworld_kernel(phase="iterate", fused=True, **kw),
            get_singleworld_kernel(phase="relax", fused=False, **kw),
            get_singleworld_kernel(phase="relax", fused=True, **kw),
        )

    def _singleworld_cached_prepare_kernels(self):
        """Return cached-contact-warm-start head/fused kernels."""
        return (
            get_singleworld_kernel(
                phase="cached_prepare",
                fused=False,
                revolute_only=True,
                cloth_support=False,
                enable_column_timers=self.enable_column_timers,
                soft_tet_neohookean=False,
                has_joints=self.num_joints > 0,
                has_contacts=self.max_contact_columns > 0,
                skip_joint_pgs=self._skip_all_joint_pgs(),
                selective_joint_pgs=self._selective_joint_pgs_enabled(),
                has_mass_splitting=False,
                has_sleeping=False,
                has_soft_contact_pd=False,
                rigid_direct=self._singleworld_rigid_direct(),
            ),
            get_singleworld_kernel(
                phase="cached_prepare",
                fused=True,
                revolute_only=True,
                cloth_support=False,
                enable_column_timers=self.enable_column_timers,
                soft_tet_neohookean=False,
                has_joints=self.num_joints > 0,
                has_contacts=self.max_contact_columns > 0,
                skip_joint_pgs=self._skip_all_joint_pgs(),
                selective_joint_pgs=self._selective_joint_pgs_enabled(),
                has_mass_splitting=False,
                has_sleeping=False,
                has_soft_contact_pd=False,
                rigid_direct=self._singleworld_rigid_direct(),
            ),
        )

    def _singleworld_rigid_direct(self) -> bool:
        """Use typed rigid loops in single-world PGS kernels."""
        if self.step_layout != "single_world" or self.mass_splitting_enabled:
            return False
        if (
            self.num_cloth_triangles > 0
            or self.num_cloth_bending > 0
            or self.num_soft_tetrahedra > 0
            or self.num_soft_hexahedra > 0
        ):
            return False
        return self.num_joints > 0 or self.max_contact_columns > 0

    def _singleworld_needs_family_starts(self) -> bool:
        """Mixed rigid scenes need joint/contact subranges."""
        return self._singleworld_rigid_direct() and self.num_joints > 0 and self.max_contact_columns > 0

    def _fast_tail_family_split(self) -> bool:
        """Use solver-family subranges in multi-world fast-tail."""
        return _choose_fast_tail_family_split_for_scene(
            step_layout=self.step_layout,
            use_greedy_coloring=self._use_greedy_coloring,
            num_worlds=self.num_worlds,
            num_joints=self.num_joints,
            max_contact_columns=self.max_contact_columns,
            num_cloth_triangles=self.num_cloth_triangles,
            num_cloth_bending=self.num_cloth_bending,
            num_soft_tetrahedra=self.num_soft_tetrahedra,
            num_soft_hexahedra=self.num_soft_hexahedra,
        )

    def _fast_tail_worlds_per_block(self) -> int:
        """Return the selected fast-tail block packing for this world."""
        return _choose_fast_tail_worlds_per_block_for_scene(
            num_worlds=self.num_worlds,
            num_joints=self.num_joints,
            max_contact_columns=self.max_contact_columns,
            step_layout=self.step_layout,
            tpw_launch_bound=self._tpw_launch_bound,
        )

    def _fast_tail_block_dim(self) -> int:
        """``_STRAGGLER_BLOCK_DIM * worlds_per_block`` (integer warps for __syncwarp)."""
        return _STRAGGLER_BLOCK_DIM * self._fast_tail_worlds_per_block()

    def _fast_tail_launch_dim_for(self, tpw_bound: int) -> int:
        """Padded launch dim for a fast-tail tpw upper bound."""
        block_dim = self._fast_tail_block_dim()
        raw = self.num_worlds * int(tpw_bound)
        return ((raw + block_dim - 1) // block_dim) * block_dim

    def _fast_tail_launch_dim(self) -> int:
        """Padded launch dim for the current fast-tail tpw upper bound."""
        return self._fast_tail_launch_dim_for(self._tpw_launch_bound)

    def _pick_tpw(self) -> None:
        """Per-step GPU tpw picker: parallel reduction over _world_num_colors,
        then a 1-thread kernel writes _tpw_choice[0]. Captured-graph safe."""
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
        launch_tpw_bound: int | None = None,
    ) -> None:
        """Launch an iterate/relax kernel running ``num_iterations`` sweeps internally."""
        wp.launch(
            kernel,
            dim=self._fast_tail_launch_dim_for(launch_tpw_bound or self._tpw_launch_bound),
            block_dim=self._fast_tail_block_dim(),
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._particles_or_sentinel(),
                wp.int32(self.num_bodies),
                idt,
                wp.float32(self.sor_boost),
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_color_family_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(num_iterations),
                wp.int32(self.num_worlds),
                wp.int32(self.num_joints),
                self._joint_pgs_enabled,
                wp.int32(self.num_cloth_triangles),
                wp.int32(self.num_cloth_bending),
                wp.int32(self.num_soft_tetrahedra),
                wp.int32(self.num_soft_hexahedra),
                self._tpw_choice,
                self._copy_state,
            ],
            device=self.device,
        )

    def _integrate_positions(self) -> None:
        """x += v*dt; q = dq(w*dt) * q for dynamic bodies. Kinematic poses
        advance via :meth:`_kinematic_interpolate_substep`."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _integrate_velocities_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _kinematic_prepare_step(self) -> None:
        """Per-step kinematic prepare: snapshot prev, resolve target, infer
        velocity. Skipped when no kinematic bodies (saves ~4us)."""
        if self.num_bodies == 0 or self._num_kinematic_bodies == 0:
            return
        wp.launch(
            _kinematic_prepare_step_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.step_dt)],
            device=self.device,
        )

    def _kinematic_interpolate_substep(self, alpha: float) -> None:
        """Per-substep kinematic pose lerp/slerp. Skipped when no kinematic bodies."""
        if self.num_bodies == 0 or self._num_kinematic_bodies == 0:
            return
        wp.launch(
            _kinematic_interpolate_substep_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(alpha)],
            device=self.device,
        )

    def _refresh_world_inertia(self) -> None:
        """Per-substep refresh of inverse_inertia_world (R * I^-1 * R^T) after
        :meth:`_integrate_positions`."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_refresh_world_inertia_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    def _gather_column_timers(self, num_contact_columns: int) -> dict[str, float]:
        """Sum every per-column ``time_us`` slot into per-type totals.

        Reduces on-device into a 5-element scratch buffer
        (joints / cloth_tri / cloth_bend / soft_tet / contacts) and
        copies only that 20-byte payload back to host. The previous
        full-row ``.numpy()`` copy was ``rigid_contact_max * 16 * 4``
        bytes per call -- megabytes on dense scenes -- which serialised
        the eager-mode viewer loop.
        """
        if self._column_timer_totals is None:
            self._column_timer_totals = wp.zeros(6, dtype=wp.float32, device=self.device)
        else:
            self._column_timer_totals.zero_()
        if self._contact_offset > 0:
            # Per-schema time_us dword offset differs between the ARAP
            # and block Neo-Hookean variants; pick the one the scene
            # actually populated. Mixed variants in one container would
            # need a per-cid type-tag read here; not currently supported.
            soft_tet_time_off = (
                int(SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET)
                if self._soft_tet_uses_neohookean
                else int(SOFT_TET_TIME_US_OFFSET)
            )
            wp.launch(
                _reduce_constraint_time_us_kernel,
                dim=self._contact_offset,
                inputs=[
                    self.constraints,
                    wp.int32(ADBS_TIME_US_OFFSET),
                    wp.int32(CLOTH_TRIANGLE_TIME_US_OFFSET),
                    wp.int32(CLOTH_BENDING_TIME_US_OFFSET),
                    wp.int32(soft_tet_time_off),
                    wp.int32(SOFT_HEX_TIME_US_OFFSET),
                    wp.int32(self.num_joints),
                    wp.int32(self.num_cloth_triangles),
                    wp.int32(self.num_cloth_bending),
                    wp.int32(self.num_soft_tetrahedra),
                    wp.int32(self.num_soft_hexahedra),
                    self._column_timer_totals,
                ],
                device=self.device,
            )
        if num_contact_columns > 0 and self.max_contact_columns > 0:
            wp.launch(
                _reduce_contact_time_us_kernel,
                dim=num_contact_columns,
                inputs=[
                    self._contact_cols,
                    wp.int32(num_contact_columns),
                    wp.int32(CONTACT_TIME_US_OFFSET),
                    self._column_timer_totals,
                ],
                device=self.device,
            )
        totals = self._column_timer_totals.numpy()
        return {
            "time_us_total_joints": float(totals[0]),
            "time_us_total_cloth_triangles": float(totals[1]),
            "time_us_total_cloth_bending": float(totals[2]),
            "time_us_total_soft_tetrahedra": float(totals[3]),
            "time_us_total_contacts": float(totals[4]),
            "time_us_total_soft_hexahedra": float(totals[5]),
        }

    def _zero_column_timers(self) -> None:
        """Zero every per-column ``time_us`` slot. Called at step start
        when :attr:`enable_column_timers` is set."""
        if self._contact_offset > 0:
            soft_tet_time_off = (
                int(SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET)
                if self._soft_tet_uses_neohookean
                else int(SOFT_TET_TIME_US_OFFSET)
            )
            wp.launch(
                _zero_constraint_time_us_kernel,
                dim=self._contact_offset,
                inputs=[
                    self.constraints,
                    self._num_active_constraints,
                    wp.int32(ADBS_TIME_US_OFFSET),
                    wp.int32(CLOTH_TRIANGLE_TIME_US_OFFSET),
                    wp.int32(CLOTH_BENDING_TIME_US_OFFSET),
                    wp.int32(soft_tet_time_off),
                    wp.int32(SOFT_HEX_TIME_US_OFFSET),
                    wp.int32(self.num_joints),
                    wp.int32(self.num_cloth_triangles),
                    wp.int32(self.num_cloth_bending),
                    wp.int32(self.num_soft_tetrahedra),
                    wp.int32(self.num_soft_hexahedra),
                ],
                device=self.device,
            )
        if self.max_contact_columns > 0:
            wp.launch(
                _zero_contact_time_us_kernel,
                dim=self.max_contact_columns,
                inputs=[self._contact_cols, wp.int32(self.max_contact_columns), wp.int32(CONTACT_TIME_US_OFFSET)],
                device=self.device,
            )

    def _update_inertia_and_clear_forces(self) -> None:
        """End-of-step: damping + inertia rebuild + force/torque zeroing (fused)."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_update_inertia_and_clear_forces_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    def _apply_global_damping(self) -> None:
        """Per-substep global damping. No-op until user opts in."""
        if self._global_damping is None or self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_apply_global_damping_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self._global_damping],
            device=self.device,
        )

    # Global damping API.

    def _ensure_global_damping_allocated(self) -> None:
        """Lazy-allocate the device array on first opt-in."""
        if self._global_damping is None:
            self._global_damping = wp.zeros(2, dtype=wp.float32, device=self.device)
            self._global_damping_host = np.zeros(2, dtype=np.float32)

    def set_global_linear_damping(self, value: float) -> None:
        """v *= 1 - value at the end of every substep. Value in [0, 1].
        First opt-in mid-simulation invalidates already-captured graphs;
        call ``set_global_*_damping(0.0)`` before capture to lock the kernel in."""
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"global_linear_damping must be in [0, 1] (got {v})")
        self._ensure_global_damping_allocated()
        self._global_damping_host[0] = v
        self._global_damping.assign(self._global_damping_host)

    def set_global_angular_damping(self, value: float) -> None:
        """w *= 1 - value per substep. See :meth:`set_global_linear_damping`."""
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"global_angular_damping must be in [0, 1] (got {v})")
        self._ensure_global_damping_allocated()
        self._global_damping_host[1] = v
        self._global_damping.assign(self._global_damping_host)

    def get_global_linear_damping(self) -> float:
        """Current factor; 0.0 if never opted in."""
        if self._global_damping_host is None:
            return 0.0
        return float(self._global_damping_host[0])

    def get_global_angular_damping(self) -> float:
        """Current factor; 0.0 if never opted in."""
        if self._global_damping_host is None:
            return 0.0
        return float(self._global_damping_host[1])

    # Diagnostics.

    @property
    def num_constraints(self) -> int:
        """Total allocated cid capacity (joints + contact columns)."""
        return self._constraint_capacity

    def gather_constraint_wrenches(self, out: wp.array) -> None:
        """Per-cid world-frame wrench on body2 (last-substep average)."""
        if self._constraint_capacity == 0:
            return
        out.zero_()
        if self.substep_dt <= 0.0:
            return
        contact_views = self._active_contact_views()
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            _constraint_gather_wrenches_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                wp.int32(self._constraint_capacity),
                wp.int32(self.num_joints),
                idt,
                self._contact_container,
                contact_views,
            ],
            outputs=[out],
            device=self.device,
        )

    def gather_constraint_errors(self, out: wp.array) -> None:
        """Per-cid position-level residual."""
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
                wp.int32(self.num_joints),
            ],
            outputs=[out],
            device=self.device,
        )

    def num_colors_used(self) -> int:
        """Number of graph colours from the last PGS. Triggers D2H copy."""
        if self.step_layout == "single_world":
            return int(self._partitioner.num_colors.numpy()[0])
        return int(self._world_num_colors.numpy().max(initial=0))

    def step_report(self) -> PhoenXWorld.StepReport:
        """Diagnostic snapshot of the last step. Triggers D2H copies."""
        num_contact_columns = (
            int(self._ingest_scratch.num_contact_columns.numpy()[0])
            if self._contact_views is not None and self._ingest_scratch is not None
            else 0
        )
        num_active = (
            int(self._num_active_constraints.numpy()[0])
            if self._num_active_constraints is not None
            else self._contact_offset + num_contact_columns
        )

        if self.enable_column_timers:
            timer_kwargs = self._gather_column_timers(num_contact_columns)
        else:
            timer_kwargs = {}

        # Unified-node degree from the partitioner's adjacency CSR end array.
        # Rigid bodies occupy [0, num_bodies); particles follow after that.
        num_nodes = self.num_bodies + self.num_particles
        if num_active > 0 and num_nodes > 0:
            ends = self._partitioner._adjacency_section_end_indices.numpy()
            n_nodes = min(int(num_nodes), int(ends.shape[0]))
            if n_nodes > 0:
                degrees = ends[:n_nodes].astype(np.int64, copy=False)
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
                **timer_kwargs,
            )

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
            **timer_kwargs,
        )

    def gather_contact_wrenches(self, out: wp.array) -> None:
        """Per-contact wrench (force + torque) from the last substep."""
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
