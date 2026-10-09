# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact joint-block Cholesky for homogeneous, chordal bilateral systems.

Each joint contributes at most six bilateral rows. A simplicial elimination
order preserves block sparsity; identity dummy rows complete short blocks.
Response tiles omit those dummy rows, preserving the compact Schur layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from ...linalg.factorize.llt_blocked_rcm import _sync_threads, get_float32_array_offset_ptr
from .kernels import BILATERAL_DIAGONAL_FLOOR, _compact_schur_fits

if TYPE_CHECKING:
    from ...dynamics.dual import DualProblem
    from .sparse import SparseDVIPath

wp.set_module_options({"enable_backward": False, "enable_mathdx_solver": False, "enable_mathdx_gemm": False})


def _build_topology(path: SparseDVIPath, pair_world: wp.array, pair_row: wp.array, pair_col: wp.array):
    """Build a zero-fill schedule, rejecting unsupported or heterogeneous rows."""
    joints = path.model.joints
    world = joints.wid.numpy()
    dynamic_counts = joints.num_dynamic_cts.numpy()
    kinematic_counts = joints.num_kinematic_cts.numpy()
    dynamic_offsets = joints.dynamic_cts_offset_total_cts.numpy()
    kinematic_offsets = joints.kinematic_cts_offset_total_cts.numpy()
    origin = path.model.info.total_cts_offset.numpy()
    dimensions = path.data.bilateral_dim.numpy()
    worlds = path.size.num_worlds
    groups_by_world = [[] for _ in range(worlds)]
    for joint, wid in enumerate(world):
        rows = list(
            range(
                int(dynamic_offsets[joint] - origin[wid]),
                int(dynamic_offsets[joint] - origin[wid] + dynamic_counts[joint]),
            )
        )
        rows += list(
            range(
                int(kinematic_offsets[joint] - origin[wid]),
                int(kinematic_offsets[joint] - origin[wid] + kinematic_counts[joint]),
            )
        )
        if rows:
            if len(rows) > 6:
                return None, "joint group exceeds six bilateral rows"
            groups_by_world[wid].append(tuple(rows))
    groups = groups_by_world[0]
    n = int(dimensions[0])
    if n == 0 or any(g != groups for g in groups_by_world) or not np.all(dimensions == n):
        return None, "empty or heterogeneous joint-row topology"
    if sorted(row for group in groups for row in group) != list(range(n)):
        return None, "joint groups do not partition the bilateral rows exactly"
    group_of = np.empty(n, dtype=np.int32)
    for i, group in enumerate(groups):
        group_of[list(group)] = i
    pw, pr, pc = pair_world.numpy(), pair_row.numpy(), pair_col.numpy()
    if np.any(pr < 0) or np.any(pc < 0) or np.any(pr >= n) or np.any(pc >= n):
        return None, "pair row outside static bilateral dimensions"
    ng = len(groups)
    lo = np.minimum(group_of[pr], group_of[pc])
    hi = np.maximum(group_of[pr], group_of[pc])
    # Unique block edges per world verify all replicated worlds share the graph.
    edges = np.unique(pw.astype(np.int64) * ng * ng + lo.astype(np.int64) * ng + hi)
    counts = np.bincount(edges // (ng * ng), minlength=worlds)
    if not np.all(counts == counts[0]):
        return None, "heterogeneous block graphs"
    edge_sets = edges.reshape(worlds, int(counts[0])) % (ng * ng)
    if not np.all(edge_sets == edge_sets[0]):
        return None, "heterogeneous block graphs"
    graph = [set() for _ in groups]
    for encoded in edge_sets[0]:
        a, b = divmod(int(encoded), ng)
        if a != b:
            graph[a].add(b)
            graph[b].add(a)
    remaining = set(range(ng))
    order = []
    while remaining:
        choices = []
        for node in remaining:
            adjacent = graph[node] & remaining
            if all(b in graph[a] for a in adjacent for b in adjacent if a != b):
                choices.append((len(adjacent), node))
        if not choices:
            return None, "nonchordal graph: zero-fill block elimination unavailable"
        node = min(choices)[1]
        order.append(node)
        remaining.remove(node)
    positions = {node: i for i, node in enumerate(order)}
    pairs = [(i, i) for i in range(ng)]
    for node in order:
        for other in graph[node]:
            if positions[other] > positions[node]:
                pairs.append((positions[other], positions[node]))
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    index = {pair: i for i, pair in enumerate(pairs)}
    starts, left, right = [0], [], []
    for i, j in pairs:
        for k in range(j):
            if (i, k) in index and (j, k) in index:
                left.append(index[i, k])
                right.append(index[j, k])
        starts.append(len(left))
    previous_start, previous, following_start, following = [0], [], [0], []
    for i in range(ng):
        previous.extend(index[i, k] for k in range(i) if (i, k) in index)
        following.extend(index[k, i] for k in range(i + 1, ng) if (k, i) in index)
        previous_start.append(len(previous))
        following_start.append(len(following))
    lengths = [len(groups[node]) for node in order]
    scalar_order = [row for node in order for row in (*groups[node], *([-1] * (6 - len(groups[node]))))]
    row_offsets = np.cumsum([0, *lengths]).tolist()
    rows, cols = zip(*pairs, strict=False)
    return {
        "n": n,
        "ng": ng,
        "padded_n": ng * 6,
        "nb": len(pairs),
        "order": scalar_order,
        "rows": rows,
        "cols": cols,
        "diagonal": [index[i, i] for i in range(ng)],
        "starts": starts,
        "left": left,
        "right": right,
        "pstart": previous_start,
        "previous": previous,
        "fstart": following_start,
        "following": following,
        "lengths": lengths,
        "row_offsets": row_offsets,
    }, "eligible homogeneous chordal joint-block graph"


@wp.kernel
def _factor(
    blocks: wp.array[wp.float32],
    factors: wp.array[wp.float32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    diagonal: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    lefts: wp.array[wp.int32],
    rights: wp.array[wp.int32],
    nb: int,
):
    w, _lane = wp.tid()
    aa = wp.array(ptr=get_float32_array_offset_ptr(blocks, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    ll = wp.array(ptr=get_float32_array_offset_ptr(factors, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    for b in range(nb):
        value = wp.tile_load(aa, shape=(6, 6), offset=(b * 6, 0), storage="shared")
        for update in range(starts[b], starts[b + 1]):
            left = wp.tile_load(ll, shape=(6, 6), offset=(lefts[update] * 6, 0))
            right = wp.tile_load(ll, shape=(6, 6), offset=(rights[update] * 6, 0))
            wp.tile_matmul(left, wp.tile_transpose(right), value, alpha=-1.0)
        if rows[b] == cols[b]:
            wp.tile_cholesky_inplace(value, fill_mode="upper")
            wp.tile_store(ll, wp.tile_transpose(value), offset=(b * 6, 0))
        else:
            d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[cols[b]] * 6, 0))
            transposed = wp.tile_transpose(value)
            wp.tile_lower_solve_inplace(d, transposed)
            wp.tile_store(ll, wp.tile_transpose(transposed), offset=(b * 6, 0))


@wp.kernel
def _initialize_dummy_rows(
    order: wp.array[wp.int32], diagonal: wp.array[wp.int32], matrix: wp.array[wp.float32], num_blocks: int
):
    world, row = wp.tid()
    if order[row] < 0:
        local = row % 6
        matrix[world * num_blocks * 36 + diagonal[row // 6] * 36 + local * 7] = 1.0


@wp.kernel
def _assemble_diagonal(
    njc: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    bilateral_vio: wp.array[wp.int32],
    problem_diag: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    row_diagonal: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    num_blocks: int,
):
    world, row = wp.tid()
    if row >= njc[world]:
        return
    diag = wp.abs(problem_diag[problem_vio[world] + row])
    p = wp.sqrt(1.0 / (diag + FLOAT32_EPS))
    scale[bilateral_vio[world] + row] = p
    matrix[world * num_blocks * 36 + row_diagonal[row]] = p * diag * p + wp.float32(BILATERAL_DIAGONAL_FLOOR)


@wp.kernel
def _assemble_entries(
    inv_mass: wp.array[wp.float32],
    inv_inertia: wp.array[wp.mat33f],
    entry_starts: wp.array[wp.int32],
    pair_world: wp.array[wp.int32],
    pair_row: wp.array[wp.int32],
    pair_col: wp.array[wp.int32],
    pair_body: wp.array[wp.int32],
    pair_i: wp.array[wp.int32],
    pair_j: wp.array[wp.int32],
    jacobian: wp.array[vec6f],
    bilateral_vio: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    entry_target: wp.array[wp.int32],
    block_rows: wp.array[wp.int32],
    block_cols: wp.array[wp.int32],
    num_blocks: int,
    matrix: wp.array[wp.float32],
):
    entry = wp.tid()
    first = entry_starts[entry]
    value = wp.float32(0.0)
    # Keep body accumulation and normalization in the dense assembly's order.
    for pair in range(first, entry_starts[entry + 1]):
        block_i = jacobian[pair_i[pair]]
        block_j = jacobian[pair_j[pair]]
        Jv_i = wp.vec3f(block_i[0], block_i[1], block_i[2])
        Jv_j = wp.vec3f(block_j[0], block_j[1], block_j[2])
        Jw_i = wp.vec3f(block_i[3], block_i[4], block_i[5])
        Jw_j = wp.vec3f(block_j[3], block_j[4], block_j[5])
        body = pair_body[pair]
        value += inv_mass[body] * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_inertia[body] @ Jw_j)
    bvio = bilateral_vio[pair_world[first]]
    value = scale[bvio + pair_row[first]] * value * scale[bvio + pair_col[first]]
    target = entry_target[entry]
    matrix[target] = value
    local = target % (num_blocks * 36)
    block = local // 36
    if block_rows[block] == block_cols[block]:
        # Diagonal tiles need both triangles; off-diagonal tiles have one owner.
        cell = local % 36
        matrix[target - cell + (cell % 6) * 6 + cell // 6] = value


@wp.kernel
def _check_factor(
    factors: wp.array[wp.float32], diagonal: wp.array[wp.int32], failure: wp.array[wp.int32], nb: int, ng: int
):
    w = wp.tid()
    failed = int(0)
    for e in range(nb * 36):
        if not wp.isfinite(factors[w * nb * 36 + e]):
            failed = 1
    for row in range(ng):
        for i in range(6):
            if factors[w * nb * 36 + diagonal[row] * 36 + i * 6 + i] <= 0.0:
                failed = 1
    wp.atomic_max(failure, 0, failed)


@wp.kernel
def _check_dimensions(
    dim: wp.array[wp.int32], njc: wp.array[wp.int32], stride: wp.array[wp.int32], n: int, failure: wp.array[wp.int32]
):
    w = wp.tid()
    nu = dim[w] - njc[w]
    if njc[w] != n or nu < 0 or nu > n or not _compact_schur_fits(njc[w], nu, stride[w]):
        wp.atomic_max(failure, 0, 1)


@wp.kernel
def _pack_rhs(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    order: wp.array[wp.int32],
    rhs: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    padded: int,
):
    w, row = wp.tid()
    if dim[w] == 0:
        return
    value = wp.float32(0)
    if order[row] >= 0:
        value = rhs[vio[w] + order[row]]
    scratch[w * padded + row] = value


@wp.kernel
def _solve(
    factors: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    active: wp.array[wp.int32],
    diagonal: wp.array[wp.int32],
    previous_start: wp.array[wp.int32],
    previous: wp.array[wp.int32],
    following_start: wp.array[wp.int32],
    following: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    nb: int,
    ng: int,
    padded: int,
):
    w, _lane = wp.tid()
    if active[w] == 0:
        return
    ll = wp.array(ptr=get_float32_array_offset_ptr(factors, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    yy = wp.array(ptr=get_float32_array_offset_ptr(scratch, w * padded), shape=(padded, 1), dtype=wp.float32)
    for row in range(ng):
        value = wp.tile_load(yy, shape=(6, 1), offset=(row * 6, 0), storage="shared")
        for entry in range(previous_start[row], previous_start[row + 1]):
            block = previous[entry]
            left = wp.tile_load(ll, shape=(6, 6), offset=(block * 6, 0))
            right = wp.tile_load(yy, shape=(6, 1), offset=(cols[block] * 6, 0))
            wp.tile_matmul(left, right, value, alpha=-1.0)
        d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[row] * 6, 0))
        wp.tile_lower_solve_inplace(d, value)
        wp.tile_store(yy, value, offset=(row * 6, 0))
    for row in range(ng - 1, -1, -1):
        value = wp.tile_load(yy, shape=(6, 1), offset=(row * 6, 0), storage="shared")
        for entry in range(following_start[row], following_start[row + 1]):
            block = following[entry]
            left = wp.tile_load(ll, shape=(6, 6), offset=(block * 6, 0))
            right = wp.tile_load(yy, shape=(6, 1), offset=(rows[block] * 6, 0))
            wp.tile_matmul(wp.tile_transpose(left), right, value, alpha=-1.0)
        d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[row] * 6, 0))
        wp.tile_upper_solve_inplace(wp.tile_transpose(d), value)
        wp.tile_store(yy, value, offset=(row * 6, 0))


@wp.kernel
def _scatter(
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    order: wp.array[wp.int32],
    scratch: wp.array[wp.float32],
    out: wp.array[wp.float32],
    padded: int,
):
    w, row = wp.tid()
    if dim[w] != 0 and order[row] >= 0:
        out[vio[w] + order[row]] = scratch[w * padded + row]


@wp.kernel
def _response(
    factors: wp.array[wp.float32],
    diagonal: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    previous: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    nb: int,
    ng: int,
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    rio: wp.array[wp.int32],
    order: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    coupling: wp.array[wp.float32],
    response: wp.array[wp.float32],
    lengths: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
):
    w, group, lane = wp.tid()
    nu = dim[w] - njc[w]
    if group * 16 >= nu:
        return
    ll = wp.array(ptr=get_float32_array_offset_ptr(factors, w * nb * 36), shape=(nb * 6, 6), dtype=wp.float32)
    for row in range(ng):
        value = wp.tile_zeros(shape=(6, 16), dtype=wp.float32, storage="shared")
        for entry in range(lane, 96, 32):
            local = entry // 16
            col = group * 16 + entry % 16
            original = order[row * 6 + local]
            v = wp.float32(0)
            if original >= 0 and col < nu:
                v = scale[vio[w] + original] * coupling[rio[w] + original * nu + col]
            value[local, entry % 16] = v
        _sync_threads()
        for entry in range(starts[row], starts[row + 1]):
            block = previous[entry]
            left = wp.tile_load(ll, shape=(6, 6), offset=(block * 6, 0))
            previous_row = cols[block]
            source = wp.array(
                ptr=get_float32_array_offset_ptr(response, rio[w] + offsets[previous_row] * nu),
                shape=(lengths[previous_row], nu),
                dtype=wp.float32,
            )
            right = wp.tile_load(source, shape=(6, 16), offset=(0, group * 16))
            wp.tile_matmul(left, right, value, alpha=-1.0)
        d = wp.tile_load(ll, shape=(6, 6), offset=(diagonal[row] * 6, 0))
        wp.tile_lower_solve_inplace(d, value)
        target = wp.array(
            ptr=get_float32_array_offset_ptr(response, rio[w] + offsets[row] * nu),
            shape=(lengths[row], nu),
            dtype=wp.float32,
        )
        wp.tile_store(target, value, offset=(0, group * 16))


class JointBlockSolver:
    """Own a fixed block schedule and scratch without modifying the RCM solver.

    ``failure`` is a device scalar. After ``prepare``, the caller must select
    the original complete solve whenever it is nonzero, before changing impulses.
    """

    @classmethod
    def create(cls, path: SparseDVIPath) -> JointBlockSolver | None:
        """Allocate a schedule only for supported homogeneous joint topologies."""
        if path.bilateral_nzb_pairs is None or not path.device.is_cuda:
            return None
        metadata, _ = _build_topology(path, *path.bilateral_nzb_pairs[:3])
        if metadata is None:
            return None
        return cls(path, metadata)

    def __init__(self, path: SparseDVIPath, metadata: dict):
        self._device = path.device
        self._info = path.data.bilateral_operator.info
        self._worlds = path.size.num_worlds
        self._n = metadata["n"]
        self._groups = metadata["ng"]
        self._padded = metadata["padded_n"]
        self._blocks = metadata["nb"]
        self._schedule = {
            name: wp.array(np.asarray(value, dtype=np.int32), dtype=wp.int32, device=self._device)
            for name, value in metadata.items()
            if name not in ("n", "ng", "padded_n", "nb")
        }
        self._matrix = wp.zeros(self._worlds * self._blocks * 36, dtype=wp.float32, device=self._device)
        self._factor = wp.empty_like(self._matrix)
        self._vector = wp.empty(self._worlds * self._padded, dtype=wp.float32, device=self._device)
        self.failure = wp.zeros(1, dtype=wp.int32, device=self._device)

        order = np.asarray(metadata["order"], dtype=np.int32)
        inverse = np.empty(self._n, dtype=np.int32)
        inverse[order[order >= 0]] = np.flatnonzero(order >= 0)
        diagonal = np.asarray(metadata["diagonal"], dtype=np.int32)
        self._row_diagonal = wp.array(
            diagonal[inverse // 6] * 36 + (inverse % 6) * 7, dtype=wp.int32, device=self._device
        )
        lookup = np.full((self._groups, self._groups), -1, dtype=np.int32)
        lookup[np.asarray(metadata["rows"]), np.asarray(metadata["cols"])] = np.arange(self._blocks, dtype=np.int32)
        first = path.bilateral_entry_starts.numpy()[:-1]
        worlds = path.bilateral_nzb_pairs[0].numpy()[first]
        i = inverse[path.bilateral_nzb_pairs[1].numpy()[first]]
        j = inverse[path.bilateral_nzb_pairs[2].numpy()[first]]
        swap = i // 6 < j // 6
        rows, cols = np.where(swap, j, i), np.where(swap, i, j)
        blocks = lookup[rows // 6, cols // 6]
        offsets = worlds * (self._blocks * 36) + blocks * 36 + (rows % 6) * 6 + cols % 6
        self._entry_target = wp.array(offsets, dtype=wp.int32, device=self._device)
        # Numeric factorization has separate storage, so structural zeros and
        # dummy identities need initialization only when the topology is built.
        wp.launch(
            _initialize_dummy_rows,
            dim=(self._worlds, self._padded),
            inputs=[self._schedule["order"], self._schedule["diagonal"], self._matrix, self._blocks],
            device=self._device,
        )

    def assemble(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Assemble normalized joint blocks directly from their body contributions."""
        state = path.data.state
        state.bilateral_preconditioner.zero_()
        problem.delassus.diagonal(state.scratch)
        wp.launch(
            _assemble_diagonal,
            dim=(self._worlds, self._n),
            inputs=[
                problem.data.njc,
                problem.data.vio,
                self._info.vio,
                state.scratch,
                state.bilateral_preconditioner,
                self._row_diagonal,
                self._matrix,
                self._blocks,
            ],
            device=self._device,
        )
        if self._entry_target.size:
            wp.launch(
                _assemble_entries,
                dim=self._entry_target.size,
                inputs=[
                    path.model.bodies.inv_m_i,
                    path.model_data.bodies.inv_I_i,
                    path.bilateral_entry_starts,
                    *path.bilateral_nzb_pairs,
                    problem.delassus.constraint_jacobian.nzb_values,
                    self._info.vio,
                    state.bilateral_preconditioner,
                    self._entry_target,
                    self._schedule["rows"],
                    self._schedule["cols"],
                    self._blocks,
                    self._matrix,
                ],
                device=self._device,
            )

    def prepare(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Factor the already assembled, normalized joint blocks.

        Preserve its diagonal and regularization exactly. Unsupported dynamic
        dimensions or failed factors request the caller's original solve.
        """
        self.failure.zero_()
        schedule = self._schedule
        wp.launch(
            _check_dimensions,
            dim=self._worlds,
            inputs=[
                problem.data.dim,
                problem.data.njc,
                path.data.state.bilateral_response_stride,
                self._n,
                self.failure,
            ],
            device=self._device,
        )
        wp.launch_tiled(
            _factor,
            dim=self._worlds,
            inputs=[
                self._matrix,
                self._factor,
                schedule["rows"],
                schedule["cols"],
                schedule["diagonal"],
                schedule["starts"],
                schedule["left"],
                schedule["right"],
                self._blocks,
            ],
            block_dim=32,
            device=self._device,
        )
        wp.launch(
            _check_factor,
            dim=self._worlds,
            inputs=[self._factor, schedule["diagonal"], self.failure, self._blocks, self._groups],
            device=self._device,
        )

    def solve(
        self,
        rhs: wp.array[wp.float32],
        x: wp.array[wp.float32],
        active_dim: wp.array[wp.int32] | None = None,
    ) -> None:
        """Solve in original row order; inactive worlds retain their outputs."""
        info, schedule = self._info, self._schedule
        dimensions = info.dim if active_dim is None else active_dim
        wp.launch(
            _pack_rhs,
            dim=(self._worlds, self._padded),
            inputs=[dimensions, info.vio, schedule["order"], rhs, self._vector, self._padded],
            device=self._device,
        )
        wp.launch_tiled(
            _solve,
            dim=self._worlds,
            inputs=[
                self._factor,
                self._vector,
                dimensions,
                schedule["diagonal"],
                schedule["pstart"],
                schedule["previous"],
                schedule["fstart"],
                schedule["following"],
                schedule["rows"],
                schedule["cols"],
                self._blocks,
                self._groups,
                self._padded,
            ],
            block_dim=32,
            device=self._device,
        )
        wp.launch(
            _scatter,
            dim=(self._worlds, self._padded),
            inputs=[dimensions, info.vio, schedule["order"], self._vector, x, self._padded],
            device=self._device,
        )

    def response(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Whiten into compact response storage, omitting identity dummy rows."""
        state, schedule = path.data.state, self._schedule
        wp.launch_tiled(
            _response,
            dim=(self._worlds, (self._n + 15) // 16),
            inputs=[
                self._factor,
                schedule["diagonal"],
                schedule["pstart"],
                schedule["previous"],
                schedule["cols"],
                self._blocks,
                self._groups,
                problem.data.dim,
                problem.data.njc,
                self._info.vio,
                state.bilateral_response_mio,
                schedule["order"],
                state.bilateral_preconditioner,
                state.bilateral_coupling,
                state.bilateral_response,
                schedule["lengths"],
                schedule["row_offsets"],
            ],
            block_dim=32,
            device=self._device,
        )
