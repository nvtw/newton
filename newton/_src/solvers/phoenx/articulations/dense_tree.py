# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tree-sparse dense mass factors for small reduced-coordinate trees.

The dense backend of :class:`ReducedPhoenXArticulation` replaces the
articulated-body (ABA) recurrences by an explicit factor of each
articulation's generalized mass. One warp per articulation assembles ``M``
about the articulation origin with the composite-rigid-body algorithm (CRBA),
adds the implicit joint diagonal (armature, ``dt kd``, ``dt^2 ke``), and
factors ``M = L^T L``. Eliminating leaves first, ``L`` inherits the ancestor
sparsity of ``M`` without fill-in (Featherstone, *Rigid Body Dynamics
Algorithms*, Sec. 6.5): row ``k`` is nonzero only at the ancestor DOFs of
``k`` and at ``k``. Rows are stored in path layout, their entries following
the DOF's root-to-self path, so the entry of ancestor ``j`` sits at the depth
of ``j``.

All passes are written in pull form, so every inner loop only reads:

* row ``k`` of the factor is ``M_kj - sum_i L_ik L_ij`` over the descendants
  ``i`` of ``k``, and DOFs of equal height are independent;
* ``L^T z = b`` gives ``z_k = (b_k - sum_i L_ik z_i) / L_kk`` over the same
  descendants, leaves first;
* ``L x = z`` gives ``x_i = (z_i - sum_j L_ij x_j) / L_ii`` over the ancestors
  ``j`` of ``i``, root first.

The velocity advance applies ``M^-1 = L^-1 L^-T`` on each articulation's lane
group, and the contact rows store ``J`` and ``W = M^-1 J^T`` for the projected
Gauss-Seidel contact solve, with ``J M^-1 J^T = |L^-T J^T|^2``.

Topology always comes from arrays: identical articulation topologies share one
template of path tables. Only the shared-memory capacities are compile-time
buckets.
"""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _CACHED_PAGE_COUNT,
    _MAX_ROWS,
    _store_point_row_effective_mass,
    _sync_contact_group,
    _sync_contact_warp,
)
from newton._src.solvers.phoenx.body import BodyContainer, DenseTreeData
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer

DENSE_DOF_BUCKETS = (16, 32, 48, 64)
"""Compile-time DOF capacities of the dense backend; larger trees keep the articulated-body path."""

_LANES = 32
_SYM66 = wp.types.vector(length=21, dtype=wp.float32)
# Rounds of 32 entries that one staging chunk keeps in flight.
_STAGE_ROUNDS = 8
_INERTIA_STRIDE = 24
_MOTION_STRIDE = 8
_vec24 = wp.types.vector(length=_INERTIA_STRIDE, dtype=wp.float32)
# Relative pivot floor of the float32 factorization: a few ulps of the assembled
# diagonal keep a round-off-cancelled pivot positive without changing
# well-conditioned trees.
_PIVOT_FLOOR = wp.constant(wp.float32(2.0**-20))


def dense_dof_bucket(dof_count: int) -> int:
    """Return the smallest DOF bucket that holds ``dof_count`` DOFs, or 0."""
    for bucket in DENSE_DOF_BUCKETS:
        if dof_count <= bucket:
            return bucket
    return 0


@functools.cache
def _make_shared_base(length: int):
    """Return the address of one block's static shared workspace of ``length`` words.

    Warp tiles synchronize the block after every element write; the dense
    kernels order their phases with ``__syncwarp`` instead and address raw,
    16-byte aligned shared storage.
    """
    snippet = f"""
#if defined(__CUDA_ARCH__)
    __shared__ __align__(16) float dense_shared[{int(length)}];
    return (uint64_t)dense_shared;
#else
    return 0;
#endif
"""

    @wp.func_native(snippet)
    def shared_base() -> wp.uint64: ...

    return shared_base


@wp.func_native("return ((const float*)base)[index];")
def _load(base: wp.uint64, index: wp.int32) -> wp.float32: ...


@wp.func_native("((float*)base)[index] = value;")
def _store(base: wp.uint64, index: wp.int32, value: wp.float32): ...


@wp.func_native("return ((const int*)base)[index];")
def _load_int(base: wp.uint64, index: wp.int32) -> wp.int32: ...


@wp.func_native("((int*)base)[index] = value;")
def _store_int(base: wp.uint64, index: wp.int32, value: wp.int32): ...


@wp.func_native(
    """
    const float4* source = (const float4*)((const float*)base + index);
    wp::vec_t<24, float> result;
    for (int word = 0; word < 6; ++word) {
        const float4 value = source[word];
        result[4 * word] = value.x;
        result[4 * word + 1] = value.y;
        result[4 * word + 2] = value.z;
        result[4 * word + 3] = value.w;
    }
    return result;
"""
)
def _load24(base: wp.uint64, index: wp.int32) -> _vec24:
    """Load a padded spatial inertia from 16-byte aligned shared storage."""
    ...


@wp.func_native(
    """
    const float4* source = (const float4*)((const float*)base + index);
    const float4 low = source[0];
    const float4 high = source[1];
    wp::spatial_vector result;
    result[0] = low.x;
    result[1] = low.y;
    result[2] = low.z;
    result[3] = low.w;
    result[4] = high.x;
    result[5] = high.y;
    return result;
"""
)
def _load_motion(base: wp.uint64, index: wp.int32) -> wp.spatial_vector:
    """Load a padded motion subspace from 16-byte aligned shared storage."""
    ...


def _unrolled_module(name: str, max_unroll: int):
    """Return a module whose fixed-count loops up to ``max_unroll`` unroll into registers."""
    module = wp.get_module(name)
    module.options["max_unroll"] = max(int(module.options.get("max_unroll", 16)), int(max_unroll))
    return module


@wp.func
def _descendant_products(
    shared: wp.uint64,
    desc_row_base: wp.int32,
    begin: wp.int32,
    end: wp.int32,
    factor_base: wp.int32,
    depth: wp.int32,
    column: wp.int32,
) -> wp.vec2:
    """Return ``sum_i L_ik L_i,column`` and ``sum_i L_ik^2`` over the descendants ``i`` of ``k``.

    ``L_ik`` sits at ``depth`` in row ``i``, whose shared start ``desc_row``
    lists. Four independent partial sums overlap the shared-memory latencies.
    """
    products = wp.vec4()
    squares = wp.vec4()
    index = begin
    while index + wp.int32(3) < end:
        row0 = factor_base + _load_int(shared, desc_row_base + index)
        row1 = factor_base + _load_int(shared, desc_row_base + index + wp.int32(1))
        row2 = factor_base + _load_int(shared, desc_row_base + index + wp.int32(2))
        row3 = factor_base + _load_int(shared, desc_row_base + index + wp.int32(3))
        l0 = _load(shared, row0 + depth)
        l1 = _load(shared, row1 + depth)
        l2 = _load(shared, row2 + depth)
        l3 = _load(shared, row3 + depth)
        products += wp.vec4(
            l0 * _load(shared, row0 + column),
            l1 * _load(shared, row1 + column),
            l2 * _load(shared, row2 + column),
            l3 * _load(shared, row3 + column),
        )
        squares += wp.vec4(l0 * l0, l1 * l1, l2 * l2, l3 * l3)
        index += wp.int32(4)
    while index < end:
        row0 = factor_base + _load_int(shared, desc_row_base + index)
        l0 = _load(shared, row0 + depth)
        products[0] += l0 * _load(shared, row0 + column)
        squares[0] += l0 * l0
        index += wp.int32(1)
    return wp.vec2(
        (products[0] + products[1]) + (products[2] + products[3]),
        (squares[0] + squares[1]) + (squares[2] + squares[3]),
    )


@wp.func
def _descendant_solve_sum(
    shared: wp.uint64,
    desc_row_base: wp.int32,
    desc_dof_base: wp.int32,
    begin: wp.int32,
    end: wp.int32,
    factor_offset: wp.int32,
    lane: wp.int32,
) -> wp.float32:
    """Return ``sum_i L_ik z_i`` over the descendants ``i`` of ``k`` for a lane's column ``z``."""
    sums = wp.vec4()
    index = begin
    while index + wp.int32(3) < end:
        sums += wp.vec4(
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index) + lane),
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index + wp.int32(1)))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index + wp.int32(1)) + lane),
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index + wp.int32(2)))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index + wp.int32(2)) + lane),
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index + wp.int32(3)))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index + wp.int32(3)) + lane),
        )
        index += wp.int32(4)
    while index < end:
        sums[0] += _load(shared, factor_offset + _load_int(shared, desc_row_base + index)) * _load(
            shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index) + lane
        )
        index += wp.int32(1)
    return (sums[0] + sums[1]) + (sums[2] + sums[3])


@wp.func
def _ancestor_solve_sum(
    shared: wp.uint64,
    path_base: wp.int32,
    factor_base: wp.int32,
    begin: wp.int32,
    end: wp.int32,
    lane: wp.int32,
) -> wp.float32:
    """Return ``sum_j L_ij x_j`` over the ancestor entries ``begin:end`` of row ``i`` for a lane's column."""
    sums = wp.vec4()
    entry = begin
    while entry + wp.int32(3) < end:
        sums += wp.vec4(
            _load(shared, factor_base + entry)
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry) + lane),
            _load(shared, factor_base + entry + wp.int32(1))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry + wp.int32(1)) + lane),
            _load(shared, factor_base + entry + wp.int32(2))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry + wp.int32(2)) + lane),
            _load(shared, factor_base + entry + wp.int32(3))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry + wp.int32(3)) + lane),
        )
        entry += wp.int32(4)
    while entry < end:
        sums[0] += _load(shared, factor_base + entry) * _load(
            shared, wp.int32(_LANES) * _load_int(shared, path_base + entry) + lane
        )
        entry += wp.int32(1)
    return (sums[0] + sums[1]) + (sums[2] + sums[3])


@wp.func
def _sym66_times(packed: _vec24, vector: wp.spatial_vector) -> wp.spatial_vector:
    """Multiply a row-major upper-triangle packed symmetric 6x6 matrix by a vector."""
    result = wp.spatial_vector()
    index = wp.int32(0)
    for row in range(6):
        for column in range(row, 6):
            value = packed[index]
            result[row] += value * vector[column]
            if column != row:
                result[column] += value * vector[row]
            index += wp.int32(1)
    return result


@wp.func
def dense_tree_solve(
    dense: DenseTreeData,
    articulation: wp.int32,
    dof_start: wp.int32,
    lane: wp.int32,
    width: wp.int32,
    mask: wp.uint32,
    rhs: wp.array[wp.float32],
    solution: wp.array[wp.float32],
):
    """Write ``M^-1 rhs`` of one articulation into ``solution`` on its lane group.

    ``rhs`` must be complete for the group on entry; it receives ``L^-T rhs``.
    Lanes own the DOFs of one height level (``L^T z = rhs``) or depth level
    (``L x = z``) at a time.
    """
    base = dense.factor_start[articulation]
    template = dense.articulation_template[articulation]
    dof_base = dense.template_dof_start[template]
    for level in range(
        dense.template_height_start[template], dense.template_height_start[template + wp.int32(1)] - wp.int32(1)
    ):
        index = dense.height_start[level] + lane
        while index < dense.height_start[level + wp.int32(1)]:
            k = dense.height_dof[index]
            depth = dense.dof_entry[dof_base + k + wp.int32(1)] - wp.int32(1) - dense.dof_entry[dof_base + k]
            value = rhs[dof_start + k]
            for descendant in range(dense.desc_start[dof_base + k], dense.desc_start[dof_base + k + wp.int32(1)]):
                value -= (
                    dense.factor[base + dense.desc_row[descendant] + depth]
                    * rhs[dof_start + dense.desc_dof[descendant]]
                )
            rhs[dof_start + k] = value / dense.factor[base + dense.dof_entry[dof_base + k + wp.int32(1)] - wp.int32(1)]
            index += width
        _sync_contact_group(mask)
    path_base = dense.template_path_start[template]
    for level in range(
        dense.template_depth_start[template], dense.template_depth_start[template + wp.int32(1)] - wp.int32(1)
    ):
        index = dense.depth_start[level] + lane
        while index < dense.depth_start[level + wp.int32(1)]:
            i = dense.depth_dof[index]
            entry = dense.dof_entry[dof_base + i]
            diagonal_entry = dense.dof_entry[dof_base + i + wp.int32(1)] - wp.int32(1)
            value = rhs[dof_start + i]
            while entry < diagonal_entry:
                value -= dense.factor[base + entry] * solution[dof_start + dense.path[path_base + entry]]
                entry += wp.int32(1)
            solution[dof_start + i] = value / dense.factor[base + diagonal_entry]
            index += width
        _sync_contact_group(mask)


@wp.kernel(enable_backward=False)
def _dense_solve_kernel(
    articulation_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    dense: DenseTreeData,
    rhs: wp.array[wp.float32],
    solution: wp.array[wp.float32],
):
    thread = wp.tid()
    articulation = thread // wp.int32(_LANES)
    lane = thread - articulation * wp.int32(_LANES)
    dense_tree_solve(
        dense,
        articulation,
        joint_qd_start[articulation_start[articulation]],
        lane,
        wp.int32(_LANES),
        wp.uint32(0xFFFFFFFF),
        rhs,
        solution,
    )


@wp.kernel(enable_backward=False)
def _mask_kernel(values: wp.array[wp.int32], mask: wp.array[wp.int32], masked: wp.array[wp.int32]):
    index = wp.tid()
    masked[index] = values[index] * mask[index]


@functools.cache
def _make_dense_factor_kernel(max_dofs: int, joint_capacity: int, entry_capacity: int):
    """CRBA plus leaves-first ``L^T L`` factorization of each articulation on one warp.

    Per-articulation inputs are staged into shared memory by fixed-count,
    unrolled loads, so their global latencies overlap instead of chaining.
    Shared spatial inertias and motion subspaces are padded to whole
    ``float4`` words.
    """
    inertia_base = entry_capacity
    motion_base = inertia_base + _INERTIA_STRIDE * joint_capacity
    diagonal_base = motion_base + _MOTION_STRIDE * max_dofs
    dof_entry_base = diagonal_base + max_dofs
    desc_start_base = dof_entry_base + max_dofs + 1
    desc_row_base = desc_start_base + max_dofs + 1
    shared_base = _make_shared_base(desc_row_base + entry_capacity)
    entry_chunks = (entry_capacity // _LANES + _STAGE_ROUNDS - 1) // _STAGE_ROUNDS
    chunk_indices = wp.types.vector(length=_STAGE_ROUNDS, dtype=wp.int32)
    joint_rounds = (joint_capacity + _LANES - 1) // _LANES
    dof_rounds = (max_dofs + _LANES) // _LANES
    motion_rounds = (6 * max_dofs + _LANES - 1) // _LANES
    child_vector = wp.types.vector(length=joint_rounds, dtype=wp.int32)
    inertia_vector = wp.types.vector(length=21 * joint_rounds, dtype=wp.float32)
    motion_vector = wp.types.vector(length=motion_rounds, dtype=wp.float32)
    dof_values = wp.types.vector(length=dof_rounds, dtype=wp.float32)
    dof_indices = wp.types.vector(length=dof_rounds, dtype=wp.int32)
    module = _unrolled_module(f"dense_tree_factor_{max_dofs}_{joint_capacity}_{entry_capacity}", 21 * joint_rounds)

    @wp.kernel(enable_backward=False, module=module)
    def dense_factor_kernel(
        bucket_articulation: wp.array[wp.int32],
        articulation_start: wp.array[wp.int32],
        articulation_end: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        motion: wp.array[wp.float32],
        body_inertia: wp.array[wp.float32],
        factor_diagonal: wp.array[wp.float32],
        template_joint_start: wp.array[wp.int32],
        subtree_start: wp.array[wp.int32],
        subtree_joint: wp.array[wp.int32],
        dof_joint: wp.array[wp.int32],
        level_pair_start: wp.array[wp.int32],
        level_pair: wp.array[wp.int32],
        dense: DenseTreeData,
    ):
        thread = wp.tid()
        block = thread // wp.int32(_LANES)
        lane = thread - block * wp.int32(_LANES)
        articulation = bucket_articulation[block]
        joint_start = articulation_start[articulation]
        joint_count = articulation_end[articulation] - joint_start
        dof_start = joint_qd_start[joint_start]
        n = joint_qd_start[articulation_end[articulation]] - dof_start
        template = dense.articulation_template[articulation]
        dof_base = dense.template_dof_start[template]
        path_base = dense.template_path_start[template]
        joint_base = template_joint_start[template]
        shared = shared_base()

        # Stage body inertias, motion subspaces, the implicit diagonal, and the row starts.
        child = child_vector()
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                child[r] = joint_child[joint_start + joint]
        inertia = inertia_vector()
        for r in range(joint_rounds):
            if lane + wp.int32(r * _LANES) < joint_count:
                for component in range(21):
                    inertia[r * 21 + component] = body_inertia[wp.int32(21) * child[r] + wp.int32(component)]
        motion_values = motion_vector()
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                motion_values[r] = motion[wp.int32(6) * dof_start + index]
        diagonal = dof_values()
        dof_entry = dof_indices()
        desc_start = dof_indices()
        desc_first = dense.desc_start[dof_base]
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof < n:
                diagonal[r] = factor_diagonal[dof_start + dof]
            if dof <= n:
                dof_entry[r] = dense.dof_entry[dof_base + dof]
                desc_start[r] = dense.desc_start[dof_base + dof] - desc_first
        desc_count = dense.desc_start[dof_base + n] - desc_first
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                for component in range(21):
                    _store(
                        shared,
                        wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * joint + wp.int32(component),
                        inertia[r * 21 + component],
                    )
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                dof = index // wp.int32(6)
                _store(
                    shared,
                    wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * dof + index - wp.int32(6) * dof,
                    motion_values[r],
                )
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof < n:
                _store(shared, wp.int32(diagonal_base) + dof, diagonal[r])
            if dof <= n:
                _store_int(shared, wp.int32(dof_entry_base) + dof, dof_entry[r])
                _store_int(shared, wp.int32(desc_start_base) + dof, desc_start[r])
        # Fixed-size chunks bound the registers that hold loads in flight.
        for chunk in range(entry_chunks):
            desc_rows = chunk_indices()
            for r in range(_STAGE_ROUNDS):
                index = lane + (chunk * wp.int32(_STAGE_ROUNDS) + wp.int32(r)) * wp.int32(_LANES)
                if index < desc_count:
                    desc_rows[r] = dense.desc_row[desc_first + index]
            for r in range(_STAGE_ROUNDS):
                index = lane + (chunk * wp.int32(_STAGE_ROUNDS) + wp.int32(r)) * wp.int32(_LANES)
                if index < desc_count:
                    _store_int(shared, wp.int32(desc_row_base) + index, desc_rows[r])
        _sync_contact_warp()

        # Composite spatial inertias: each lane sums its joints' subtrees in registers.
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                composite = _vec24()
                for index in range(subtree_start[joint_base + joint], subtree_start[joint_base + joint + wp.int32(1)]):
                    composite += _load24(
                        shared, wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * subtree_joint[index]
                    )
                for component in range(21):
                    inertia[r * 21 + component] = composite[component]
        _sync_contact_warp()
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                for component in range(21):
                    _store(
                        shared,
                        wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * joint + wp.int32(component),
                        inertia[r * 21 + component],
                    )
        _sync_contact_warp()

        # CRBA in path layout: M[i, k] = S_k . (I_c(joint(i)) S_i) for the ancestors k of i.
        row = lane
        while row < n:
            composite = _load24(shared, wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * dof_joint[dof_base + row])
            momentum = _sym66_times(
                composite, _load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * row)
            )
            entry = _load_int(shared, wp.int32(dof_entry_base) + row)
            diagonal_entry = _load_int(shared, wp.int32(dof_entry_base) + row + wp.int32(1)) - wp.int32(1)
            while entry < diagonal_entry:
                column = dense.path[path_base + entry]
                _store(
                    shared,
                    entry,
                    wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * column), momentum),
                )
                entry += wp.int32(1)
            # Keep the assembled diagonal for the relative pivot floor.
            value = wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * row), momentum)
            value += _load(shared, wp.int32(diagonal_base) + row)
            _store(shared, wp.int32(diagonal_base) + row, value)
            _store(shared, diagonal_entry, value)
            row += wp.int32(_LANES)
        _sync_contact_warp()

        # M = L^T L by height levels: lane (k, t) forms entry t of row k from the scaled
        # rows of k's descendants, H_kt - sum_i L_ik L_it, and scales it by L_kk, which it
        # forms alike. Rows of equal height never read each other.
        for level in range(
            dense.template_height_start[template],
            dense.template_height_start[template + wp.int32(1)] - wp.int32(1),
        ):
            pair = level_pair_start[level] + lane
            while pair < level_pair_start[level + wp.int32(1)]:
                packed = level_pair[pair]
                k = packed >> wp.int32(16)
                t = packed & wp.int32(0xFFFF)
                entry = _load_int(shared, wp.int32(dof_entry_base) + k)
                depth = _load_int(shared, wp.int32(dof_entry_base) + k + wp.int32(1)) - wp.int32(1) - entry
                # The assembled diagonal, as the diagonal lane may already store L_kk.
                assembled = _load(shared, wp.int32(diagonal_base) + k)
                sums = _descendant_products(
                    shared,
                    wp.int32(desc_row_base),
                    _load_int(shared, wp.int32(desc_start_base) + k),
                    _load_int(shared, wp.int32(desc_start_base) + k + wp.int32(1)),
                    wp.int32(0),
                    depth,
                    t,
                )
                value = _load(shared, entry + t) - sums[0]
                pivot = wp.sqrt(wp.max(assembled - sums[1], _PIVOT_FLOOR * assembled))
                if t == depth:
                    value = pivot
                else:
                    value = value / pivot
                _store(shared, entry + t, value)
                pair += wp.int32(_LANES)
            _sync_contact_warp()

        base = dense.factor_start[articulation]
        entry_count = _load_int(shared, wp.int32(dof_entry_base) + n)
        entry = lane
        while entry < entry_count:
            dense.factor[base + entry] = _load(shared, entry)
            entry += wp.int32(_LANES)

    return dense_factor_kernel


@functools.cache
def _make_dense_contact_rows_kernel(max_dofs: int, entry_capacity: int, patch_rows: bool):
    """Contact rows ``J`` and responses ``W = M^-1 J^T`` from the dense factor.

    The factor and its tables are staged into shared memory. Each lane owns one
    row and solves it serially in its own shared-memory column, so all lanes
    walk the factor in lockstep and share its loads.
    """
    factor_base = _LANES * max_dofs
    path_base_shared = factor_base + entry_capacity
    desc_row_base = path_base_shared + entry_capacity
    desc_dof_base = desc_row_base + entry_capacity
    dof_entry_base = desc_dof_base + entry_capacity
    desc_start_base = dof_entry_base + max_dofs + 1
    motion_base = 4 * ((desc_start_base + max_dofs + 4) // 4)
    shared_base = _make_shared_base(motion_base + _MOTION_STRIDE * max_dofs)
    entry_chunks = (entry_capacity // _LANES + _STAGE_ROUNDS - 1) // _STAGE_ROUNDS
    dof_rounds = (max_dofs + _LANES) // _LANES
    motion_rounds = (6 * max_dofs + _LANES - 1) // _LANES
    chunk_values = wp.types.vector(length=_STAGE_ROUNDS, dtype=wp.float32)
    chunk_indices = wp.types.vector(length=_STAGE_ROUNDS, dtype=wp.int32)
    dof_indices = wp.types.vector(length=dof_rounds, dtype=wp.int32)
    motion_vector = wp.types.vector(length=motion_rounds, dtype=wp.float32)
    module = _unrolled_module(
        f"dense_tree_contact_rows_{max_dofs}_{entry_capacity}_{int(patch_rows)}", max(_STAGE_ROUNDS, motion_rounds)
    )

    @wp.func
    def add_path(
        bodies: BodyContainer,
        shared: wp.uint64,
        body: wp.int32,
        wrench: wp.spatial_vector,
        joint_start: wp.int32,
        joint_last_dof: wp.array[wp.int32],
        joint_base: wp.int32,
        lane: wp.int32,
    ) -> wp.int32:
        """Add ``S_k . wrench`` along the root path of ``body`` to the lane's column.

        Returns the last DOF of that path, or -1 when ``body`` moves with no DOF.
        """
        dof = joint_last_dof[joint_base + bodies.reduced.body_joint[body] - joint_start]
        if dof >= wp.int32(0):
            entry_end = _load_int(shared, wp.int32(dof_entry_base) + dof + wp.int32(1))
            for entry in range(_load_int(shared, wp.int32(dof_entry_base) + dof), entry_end):
                column = _load_int(shared, wp.int32(path_base_shared) + entry)
                slot = wp.int32(_LANES) * column + lane
                value = wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * column), wrench)
                _store(shared, slot, _load(shared, slot) + value)
        return dof

    @wp.kernel(enable_backward=False, module=module)
    def dense_contact_rows_kernel(
        bucket_articulation: wp.array[wp.int32],
        bodies: BodyContainer,
        enabled: wp.array[wp.int32],
        point_count: wp.array[wp.int32],
        row_count_array: wp.array[wp.int32],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_body_pair: wp.array2d[wp.int32],
        row_wrench_pair: wp.array2d[wp.spatial_vector],
        point_contact: wp.array2d[wp.int32],
        cc: ContactContainer,
        max_page_count: wp.array[wp.int32],
        page_index: wp.array[wp.int32],
        prepare: wp.bool,
        joint_last_dof: wp.array[wp.int32],
        template_joint_start: wp.array[wp.int32],
        motion: wp.array[wp.float32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
    ):
        thread = wp.tid()
        block = thread // wp.int32(_LANES)
        lane = thread - block * wp.int32(_LANES)
        articulation = bucket_articulation[block]
        if enabled[articulation] == wp.int32(0):
            return
        page = page_index[0]
        if not prepare and (
            page == wp.int32(0) or (page == wp.int32(1) and max_page_count[0] <= wp.int32(_CACHED_PAGE_COUNT))
        ):
            return
        data = bodies.reduced
        dense = data.dense
        joint_start = data.articulation_start[articulation]
        dof_start = data.joint_qd_start[joint_start]
        n = data.joint_qd_start[data.articulation_end[articulation]] - dof_start
        base = dense.factor_start[articulation]
        template = dense.articulation_template[articulation]
        dof_base = dense.template_dof_start[template]
        path_base = dense.template_path_start[template]
        joint_base = template_joint_start[template]
        storage_page = wp.min(page, wp.int32(_CACHED_PAGE_COUNT - 1))
        packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
        row_count = row_count_array[packed_articulation]
        shared = shared_base()

        # Stage the factor, its paths, the descendant lists, and the row starts.
        entry_count = dense.dof_entry[dof_base + n]
        desc_first = dense.desc_start[dof_base]
        desc_count = dense.desc_start[dof_base + n] - desc_first
        # Fixed-size chunks bound the registers that hold loads in flight.
        for chunk in range(entry_chunks):
            factor_values = chunk_values()
            path_values = chunk_indices()
            desc_rows = chunk_indices()
            desc_dofs = chunk_indices()
            for r in range(_STAGE_ROUNDS):
                entry = lane + (chunk * wp.int32(_STAGE_ROUNDS) + wp.int32(r)) * wp.int32(_LANES)
                if entry < entry_count:
                    factor_values[r] = dense.factor[base + entry]
                    path_values[r] = dense.path[path_base + entry]
                if entry < desc_count:
                    desc_rows[r] = dense.desc_row[desc_first + entry]
                    desc_dofs[r] = dense.desc_dof[desc_first + entry]
            for r in range(_STAGE_ROUNDS):
                entry = lane + (chunk * wp.int32(_STAGE_ROUNDS) + wp.int32(r)) * wp.int32(_LANES)
                if entry < entry_count:
                    _store(shared, wp.int32(factor_base) + entry, factor_values[r])
                    _store_int(shared, wp.int32(path_base_shared) + entry, path_values[r])
                if entry < desc_count:
                    _store_int(shared, wp.int32(desc_row_base) + entry, desc_rows[r])
                    _store_int(shared, wp.int32(desc_dof_base) + entry, desc_dofs[r])
        dof_entry = dof_indices()
        desc_start = dof_indices()
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof <= n:
                dof_entry[r] = dense.dof_entry[dof_base + dof]
                desc_start[r] = dense.desc_start[dof_base + dof] - desc_first
        motion_values = motion_vector()
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                motion_values[r] = motion[wp.int32(6) * dof_start + index]
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                dof = index // wp.int32(6)
                _store(
                    shared,
                    wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * dof + index - wp.int32(6) * dof,
                    motion_values[r],
                )
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof <= n:
                _store_int(shared, wp.int32(dof_entry_base) + dof, dof_entry[r])
                _store_int(shared, wp.int32(desc_start_base) + dof, desc_start[r])
        _sync_contact_warp()

        row = lane
        while row < row_count:
            packed_row = storage_page * wp.int32(_MAX_ROWS) + row
            for dof in range(n):
                _store(shared, wp.int32(_LANES) * dof + lane, 0.0)
            body = row_body[packed_articulation, row]
            wrench = row_wrench[packed_articulation, row]
            path_dof = add_path(bodies, shared, body, wrench, joint_start, joint_last_dof, joint_base, lane)
            body_pair = wp.int32(-1)
            wrench_pair = wp.spatial_vector()
            if wp.static(not patch_rows):
                body_pair = row_body_pair[packed_articulation, row]
                if body_pair >= wp.int32(0):
                    wrench_pair = row_wrench_pair[packed_articulation, row]
                    add_path(bodies, shared, body_pair, wrench_pair, joint_start, joint_last_dof, joint_base, lane)
            for dof in range(n):
                packed_jacobian[packed_row, dof_start + dof] = _load(shared, wp.int32(_LANES) * dof + lane)

            # z = L^-T J^T, leaves first; J M^-1 J^T = |z|^2. A one-body row is
            # supported on the root path of its body only, and so is z.
            inverse_mass = wp.float32(0.0)
            path_entry = wp.int32(0)
            path_length = wp.int32(0)
            if path_dof >= wp.int32(0):
                path_entry = _load_int(shared, wp.int32(dof_entry_base) + path_dof)
                path_length = _load_int(shared, wp.int32(dof_entry_base) + path_dof + wp.int32(1)) - path_entry
            if body_pair >= wp.int32(0):
                path_length = wp.int32(0)
            for reverse in range(path_length):
                position = path_length - wp.int32(1) - reverse
                k = _load_int(shared, wp.int32(path_base_shared) + path_entry + position)
                slot = wp.int32(_LANES) * k + lane
                value = _load(shared, slot)
                for later in range(position + wp.int32(1), path_length):
                    i = _load_int(shared, wp.int32(path_base_shared) + path_entry + later)
                    value -= _load(
                        shared, wp.int32(factor_base) + _load_int(shared, wp.int32(dof_entry_base) + i) + position
                    ) * _load(shared, wp.int32(_LANES) * i + lane)
                value /= _load(
                    shared, wp.int32(factor_base) + _load_int(shared, wp.int32(dof_entry_base) + k + wp.int32(1)) - 1
                )
                _store(shared, slot, value)
                inverse_mass += value * value
            pair_count = n
            if body_pair < wp.int32(0):
                pair_count = wp.int32(0)
            for reverse in range(pair_count):
                k = n - wp.int32(1) - reverse
                entry = _load_int(shared, wp.int32(dof_entry_base) + k)
                diagonal_entry = _load_int(shared, wp.int32(dof_entry_base) + k + wp.int32(1)) - wp.int32(1)
                depth = diagonal_entry - entry
                slot = wp.int32(_LANES) * k + lane
                value = _load(shared, slot) - _descendant_solve_sum(
                    shared,
                    wp.int32(desc_row_base),
                    wp.int32(desc_dof_base),
                    _load_int(shared, wp.int32(desc_start_base) + k),
                    _load_int(shared, wp.int32(desc_start_base) + k + wp.int32(1)),
                    wp.int32(factor_base) + depth,
                    lane,
                )
                value /= _load(shared, wp.int32(factor_base) + diagonal_entry)
                _store(shared, slot, value)
                inverse_mass += value * value
            # W = L^-1 z, root first.
            for i in range(n):
                entry = _load_int(shared, wp.int32(dof_entry_base) + i)
                diagonal_entry = _load_int(shared, wp.int32(dof_entry_base) + i + wp.int32(1)) - wp.int32(1)
                slot = wp.int32(_LANES) * i + lane
                value = _load(shared, slot) - _ancestor_solve_sum(
                    shared, wp.int32(path_base_shared), wp.int32(factor_base), entry, diagonal_entry, lane
                )
                value /= _load(shared, wp.int32(factor_base) + diagonal_entry)
                _store(shared, slot, value)
                packed_response[packed_row, dof_start + i] = value
            if wp.static(not patch_rows):
                _store_point_row_effective_mass(
                    bodies,
                    cc,
                    point_contact,
                    packed_articulation,
                    row,
                    body,
                    wrench,
                    body_pair,
                    wrench_pair,
                    inverse_mass,
                    prepare,
                )
            row += wp.int32(_LANES)

    return dense_contact_rows_kernel


def _articulation_template(joint_parent_local: list[int], joint_dof_count: list[int]) -> dict:
    """Build the path, descendant, and level tables of one local articulation topology."""
    joint_count = len(joint_parent_local)
    joint_dof_start = np.concatenate(([0], np.cumsum(joint_dof_count))).astype(int)
    dof_count = int(joint_dof_start[-1])
    # DOFs of one joint form a chain below the last DOF of the nearest ancestor joint with DOFs.
    joint_last_dof = [-1] * joint_count
    dof_parent = [-1] * dof_count
    dof_joint = [0] * dof_count
    for joint in range(joint_count):
        parent = joint_parent_local[joint]
        inherited = joint_last_dof[parent] if parent >= 0 else -1
        for dof in range(joint_dof_start[joint], joint_dof_start[joint + 1]):
            dof_parent[dof] = inherited
            dof_joint[dof] = joint
            inherited = dof
        joint_last_dof[joint] = inherited
    paths: list[list[int]] = []
    for dof in range(dof_count):
        parent = dof_parent[dof]
        paths.append([*(paths[parent] if parent >= 0 else []), dof])
    dof_entry = np.concatenate(([0], np.cumsum([len(path) for path in paths]))).astype(np.int32)
    descendants = [[i for i in range(dof_count) if i != k and k in paths[i]] for k in range(dof_count)]
    height = [0] * dof_count
    for dof in reversed(range(dof_count)):
        if dof_parent[dof] >= 0:
            height[dof_parent[dof]] = max(height[dof_parent[dof]], height[dof] + 1)
    height_levels = [[k for k in range(dof_count) if height[k] == h] for h in range(max(height, default=-1) + 1)]
    depth_levels = [
        [k for k in range(dof_count) if len(paths[k]) == d + 1] for d in range(max(map(len, paths), default=0))
    ]
    # Factor pairs (k, t) of each height level, packed as k << 16 | t.
    level_pairs = [[k << 16 | t for k in level for t in range(len(paths[k]))] for level in height_levels]
    subtrees: list[list[int]] = [[joint] for joint in range(joint_count)]
    for joint in reversed(range(joint_count)):
        parent = joint_parent_local[joint]
        if parent >= 0:
            subtrees[parent].extend(subtrees[joint])

    def starts(lists) -> np.ndarray:
        return np.concatenate(([0], np.cumsum([len(item) for item in lists]))).astype(np.int32)

    def flat(lists) -> np.ndarray:
        return np.asarray([value for item in lists for value in item], dtype=np.int32)

    return {
        "dof_entry": dof_entry,
        "path": flat(paths),
        "dof_joint": np.asarray([*dof_joint, 0], dtype=np.int32),
        "desc_start": starts(descendants),
        "desc_row": np.asarray([dof_entry[i] for item in descendants for i in item], dtype=np.int32),
        "desc_dof": flat(descendants),
        "height_start": starts(height_levels),
        "height_dof": flat(height_levels),
        "depth_start": starts(depth_levels),
        "depth_dof": flat(depth_levels),
        "level_pair_start": starts(level_pairs),
        "level_pair": flat(level_pairs),
        "subtree_start": starts(subtrees),
        "subtree_joint": flat(subtrees),
        "joint_last_dof": np.asarray([*joint_last_dof, -1], dtype=np.int32),
    }


class DenseTreeOperator:
    """Bucketed tree-sparse mass factors for the articulations of a reduced system.

    Args:
        model: Model whose articulations the reduced system owns.
        system: The :class:`ReducedArticulationSystem` providing per-pose
            motion subspaces, spatial inertias, and the implicit joint diagonal.
    """

    def __init__(self, model, system):
        self.model = model
        self.system = system
        self.device = model.device
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        articulation_count = int(model.articulation_count)
        body_joint = np.full(int(model.body_count), -1, dtype=np.int32)
        for joint, child in enumerate(joint_child):
            body_joint[int(child)] = joint

        # Trees beyond the buckets fall back to the articulated-body recurrences.
        self.fallback_reasons: list[str] = []
        template_index: dict[tuple, int] = {}
        templates: list[dict] = []
        articulation_template = np.zeros(articulation_count, dtype=np.int32)
        factor_start = np.full(articulation_count, -1, dtype=np.int32)
        bucket_lists: dict[int, list[int]] = {}
        factor_size = 0
        for articulation in range(articulation_count):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            parents: list[int] = []
            for joint in range(start, end):
                parent = int(joint_parent[joint])
                parent_joint = int(body_joint[parent]) if parent >= 0 else -1
                # A root attached to a body outside the tree moves with no tree DOF.
                parents.append(parent_joint - start if start <= parent_joint < end else -1)
            # The path tables assume a parent precedes its children.
            if any(parent >= index for index, parent in enumerate(parents)):
                self.fallback_reasons.append(f"articulation {articulation} is not parent-first")
                continue
            dof_counts = [int(joint_qd_start[joint + 1] - joint_qd_start[joint]) for joint in range(start, end)]
            bucket = dense_dof_bucket(max(sum(dof_counts), end - start))
            if bucket == 0:
                self.fallback_reasons.append(
                    f"articulation {articulation} has more than {DENSE_DOF_BUCKETS[-1]} DOFs or joints"
                )
                continue
            key = (tuple(parents), tuple(dof_counts))
            if key not in template_index:
                template_index[key] = len(templates)
                templates.append(_articulation_template(parents, dof_counts))
            template = templates[template_index[key]]
            articulation_template[articulation] = template_index[key]
            bucket_lists.setdefault(bucket, []).append(articulation)
            factor_start[articulation] = factor_size
            factor_size += int(template["dof_entry"][-1])
        self.has_fallback = bool(self.fallback_reasons)

        def capacity(values) -> int:
            return _LANES * ((max(values) + _LANES - 1) // _LANES)

        # Each bucket's shared workspace holds its largest joint count and factor size.
        self.buckets = [
            (
                bucket,
                capacity(int(articulation_end[a] - articulation_start[a]) for a in articulations),
                capacity(int(templates[articulation_template[a]]["dof_entry"][-1]) for a in articulations),
                wp.array(np.asarray(articulations, dtype=np.int32), device=self.device),
            )
            for bucket, articulations in sorted(bucket_lists.items())
        ]

        def offsets(name: str) -> np.ndarray:
            return np.concatenate(([0], np.cumsum([len(template[name]) for template in templates]))).astype(np.int32)

        def concatenated(name: str, rebase: str | None = None) -> wp.array:
            # Start tables of nested lists are rebased onto their template's slice of ``rebase``.
            base = offsets(rebase) if rebase is not None else np.zeros(len(templates) + 1, dtype=np.int32)
            values = [template[name] + base[index] for index, template in enumerate(templates)]
            return wp.array(
                np.concatenate(values or [np.zeros(1, dtype=np.int32)]).astype(np.int32),
                dtype=wp.int32,
                device=self.device,
            )

        def level_offsets(name: str) -> wp.array:
            # A template's levels run from its first start slot to the slot before the next
            # template's: each start table holds one more slot than levels.
            return wp.array(offsets(name), dtype=wp.int32, device=self.device)

        data = DenseTreeData()
        data.factor = wp.zeros(max(1, factor_size), dtype=wp.float32, device=self.device)
        data.factor_start = wp.array(factor_start, dtype=wp.int32, device=self.device)
        data.articulation_template = wp.array(articulation_template, dtype=wp.int32, device=self.device)
        data.template_dof_start = wp.array(offsets("dof_entry"), dtype=wp.int32, device=self.device)
        data.template_path_start = wp.array(offsets("path"), dtype=wp.int32, device=self.device)
        data.dof_entry = concatenated("dof_entry")
        data.path = concatenated("path")
        data.desc_start = concatenated("desc_start", "desc_row")
        data.desc_row = concatenated("desc_row")
        data.desc_dof = concatenated("desc_dof")
        data.template_height_start = level_offsets("height_start")
        data.height_start = concatenated("height_start", "height_dof")
        data.height_dof = concatenated("height_dof")
        data.template_depth_start = level_offsets("depth_start")
        data.depth_start = concatenated("depth_start", "depth_dof")
        data.depth_dof = concatenated("depth_dof")
        self.data = data
        self.template_joint_start = wp.array(offsets("subtree_start"), dtype=wp.int32, device=self.device)
        self.subtree_start = concatenated("subtree_start", "subtree_joint")
        self.subtree_joint = concatenated("subtree_joint")
        self.dof_joint = concatenated("dof_joint")
        self.joint_last_dof = concatenated("joint_last_dof")
        self.level_pair_start = concatenated("level_pair_start", "level_pair")
        self.level_pair = concatenated("level_pair")
        # Flat float views of the per-pose motion subspaces and packed spatial inertias.
        self.motion = wp.array(
            ptr=system.joint_s.ptr, dtype=wp.float32, shape=(6 * system.joint_s.shape[0],), device=self.device
        )
        self.body_inertia = wp.array(
            ptr=system.body_i_s.ptr, dtype=wp.float32, shape=(21 * system.body_i_s.shape[0],), device=self.device
        )
        fallback_mask = np.zeros(articulation_count, dtype=np.int32)
        fallback_mask[factor_start < 0] = 1
        self.fallback_mask = wp.array(fallback_mask, dtype=wp.int32, device=self.device)
        self._fallback_enabled = wp.zeros(max(1, articulation_count), dtype=wp.int32, device=self.device)
        # Set by the owner when ABA-only consumers (coupled contacts, loops) also need the
        # articulated-body factor.
        self.requires_aba = False
        self._solution: wp.array[wp.float32] | None = None

    def factor_mass(self) -> None:
        """Assemble and factor every articulation's generalized mass at the current pose."""
        system = self.system
        model = self.model
        for bucket, joint_capacity, entry_capacity, articulations in self.buckets:
            wp.launch(
                _make_dense_factor_kernel(bucket, joint_capacity, entry_capacity),
                dim=articulations.shape[0] * _LANES,
                block_dim=_LANES,
                inputs=[
                    articulations,
                    model.articulation_start,
                    model.articulation_end,
                    model.joint_child,
                    model.joint_qd_start,
                    self.motion,
                    self.body_inertia,
                    system.joint_factor_diagonal,
                    self.template_joint_start,
                    self.subtree_start,
                    self.subtree_joint,
                    self.dof_joint,
                    self.level_pair_start,
                    self.level_pair,
                    self.data,
                ],
                device=self.device,
            )

    def solve(self, rhs: wp.array[wp.float32]) -> wp.array[wp.float32]:
        """Return ``M^-1 rhs`` for the last factored mass; ``rhs`` is overwritten."""
        if self._solution is None:
            self._solution = wp.zeros_like(rhs)
        wp.launch(
            _dense_solve_kernel,
            dim=int(self.model.articulation_count) * _LANES,
            block_dim=_LANES,
            inputs=[self.model.articulation_start, self.model.joint_qd_start, self.data, rhs],
            outputs=[self._solution],
            device=self.device,
        )
        return self._solution

    def fallback_enabled(self, enabled: wp.array[wp.int32]) -> wp.array[wp.int32]:
        """Return ``enabled`` restricted to the trees on the articulated-body path."""
        wp.launch(
            _mask_kernel,
            dim=self.fallback_mask.shape[0],
            inputs=[enabled, self.fallback_mask],
            outputs=[self._fallback_enabled],
            device=self.device,
        )
        return self._fallback_enabled

    def build_contact_rows(self, contact, bodies: BodyContainer, cc: ContactContainer, prepare: bool) -> None:
        """Build the packed rows and responses of a :class:`ReducedContactBlockSystem` page."""
        for bucket, _joint_capacity, entry_capacity, articulations in self.buckets:
            wp.launch(
                _make_dense_contact_rows_kernel(bucket, entry_capacity, contact.patch_rows),
                dim=articulations.shape[0] * _LANES,
                block_dim=_LANES,
                inputs=[
                    articulations,
                    bodies,
                    contact.enabled,
                    contact.point_count,
                    contact.row_count,
                    contact.row_body,
                    contact.row_wrench,
                    contact.row_body_pair,
                    contact.row_wrench_pair,
                    contact.point_contact,
                    cc,
                    contact.max_page_count,
                    contact.page_index,
                    wp.bool(prepare),
                    self.joint_last_dof,
                    self.template_joint_start,
                    self.motion,
                ],
                outputs=[contact.packed_jacobian, contact.packed_response],
                device=self.device,
            )


__all__ = ["DENSE_DOF_BUCKETS", "DenseTreeOperator", "dense_dof_bucket", "dense_tree_solve"]
