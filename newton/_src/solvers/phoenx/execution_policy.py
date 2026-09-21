# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Execution scheduling and capacity policies for PhoenX worlds."""

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import MAX_BODIES
from newton._src.solvers.phoenx.simulation_kernels import (
    _STRAGGLER_BLOCK_DIM,
    _choose_fast_tail_worlds_per_block,
)


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
        sparse_joint_contact_world = 0.0 < joints_per_world <= 32.0 and contacts_capacity_per_world <= 96.0
        small_joint_world = 0.0 < joints_per_world <= 64.0 and contacts_capacity_per_world <= 512.0
        dense_joint_world = joints_per_world > 64.0
        dense_contact_only_world = num_joints == 0 and contacts_capacity_per_world > 256.0
        simple_saturated_joint_contact_world = (
            worlds >= 16 * sm
            and 0.0 < joints_per_world <= 32.0
            and (
                128.0 <= contacts_capacity_per_world <= 384.0
                or (joints_per_world <= 20.0 and contacts_capacity_per_world <= 512.0)
            )
        )
        if sparse_joint_contact_world:
            return False, 8
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

    # Sparse worlds are better packed by fast-tail (many worlds per block
    # keeps lanes busy); one physical block per world only pays off once
    # each world carries enough rows to fill a CTA's scheduling.
    if rows_per_world < 16.0:
        return "fast_tail", 128

    # Contact-only fleets: a dedicated block per world pays off only when the
    # per-world contact graph is dense enough to fill it; otherwise fast-tail's
    # cross-world packing keeps more lanes busy.
    if joints_per_world == 0.0:
        if contacts_per_world >= 512.0:
            return "block_world", 128
        return "fast_tail", 128

    # Small contact-light robots do not carry enough bounded row work to fill
    # one CTA per world. Mini subwarp and full-solver brackets agree that
    # packing these worlds wins until the articulation becomes dense.
    if joints_per_world <= 32.0 and contacts_per_world <= 96.0:
        return "fast_tail", 128

    # At full RL-fleet occupancy, compact articulations keep fast-tail's
    # subwarp packing busy; dedicating a CTA to every world wastes lanes.
    if num_worlds >= 4096 and joints_per_world <= 20.0:
        return "fast_tail", 128

    # Wider articulations underfill fast-tail lane groups. A 32-thread block
    # per world restores lane utilization while preserving PGS row order.
    if num_worlds >= 512:
        return "block_world", 32

    return "fast_tail", 128


def _choose_fast_tail_solve_schedule(*, substeps: int) -> tuple[int, int, int]:
    """Return the full-color PGS schedule used by the fast-tail solver."""
    _ = substeps
    return 1, 1, 1


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
    cuda_blocks_per_sm: int = 4,
) -> int:
    """Size the fixed single-world persistent grid from capacity and SM count.

    ``max_thread_blocks`` overrides the automatic ``cuda_blocks_per_sm`` cap.
    """
    block_dim = _SINGLEWORLD_BLOCK_DIM
    capacity_blocks = (max(1, int(constraint_capacity)) + block_dim - 1) // block_dim
    if max_thread_blocks is not None:
        if int(max_thread_blocks) < 1:
            raise ValueError(f"max_thread_blocks must be >= 1 (got {max_thread_blocks})")
        num_blocks = max(1, min(capacity_blocks, int(max_thread_blocks)))
        return block_dim * num_blocks
    device_obj = wp.get_device(device)
    if device_obj.is_cuda:
        max_blocks_limit = device_obj.sm_count * int(cuda_blocks_per_sm)
    else:
        max_blocks_limit = 256
    min_blocks = 32
    num_blocks = max(min_blocks, min(capacity_blocks, max_blocks_limit))
    return block_dim * num_blocks
