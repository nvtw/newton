# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental batched sparse PCG with a multilevel additive Schwarz preconditioner.

The global matrix is stored as row-contiguous 3x3 BSR. Dense matrices only
exist inside the small Schwarz subdomains. The hierarchy uses injection
restriction: every 16 nodes become one node on the next level, and corrections
from all levels are added during prolongation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kamino._src.linalg.conjugate import BatchedLinearOperator, CGSolver

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_CLUSTER_NODE_COUNT = 16
_CLUSTER_DOF_COUNT = 3 * _CLUSTER_NODE_COUNT
_MAX_LEVEL_COUNT = 8


@wp.func_native(
    r"""
#if defined(__CUDA_ARCH__)
    constexpr int ROWS = 48;
    constexpr int PADDED_LD = 52;
    __shared__ __align__(16) float shared_data[ROWS * PADDED_LD + ROWS];
    float* shared_matrix = shared_data;
    float* shared_residual = shared_data + ROWS * PADDED_LD;
    const int base = cluster * ROWS;

    #if __CUDA_ARCH__ >= 800
    for (int copy = thread; copy < ROWS * 12; copy += blockDim.x) {
        const int row = copy / 12;
        const int segment = copy - row * 12;
        float* dst = shared_matrix + row * PADDED_LD + segment * 4;
        const float* src = factors.data + (base + row) * ROWS + segment * 4;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    #else
    for (int index = thread; index < ROWS * ROWS; index += blockDim.x) {
        const int row = index / ROWS;
        const int col = index - row * ROWS;
        shared_matrix[row * PADDED_LD + col] = factors.data[base * ROWS + index];
    }
    #endif

    if (thread < ROWS) {
        const int slot = cluster * 16 + thread / 3;
        const int component = thread % 3;
        const int begin = slot_fine_begin.data[slot];
        const int count = slot_fine_count.data[slot];
        float total = 0.0f;
        for (int index = 0; index < count; ++index)
            total += residual.data[3 * (begin + index) + component];
        shared_residual[thread] = total;
    }
    #if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    #endif
    __syncthreads();

    float energy_value = 0.0f;
    if (thread < ROWS) {
        float total = 0.0f;
        #pragma unroll
        for (int col = 0; col < ROWS; ++col)
            total = fmaf(shared_matrix[thread * PADDED_LD + col], shared_residual[col], total);
        result.data[base + thread] = total;
        energy_value = shared_residual[thread] * total;
    }
    if (compute_energy) {
        __syncthreads();
        shared_matrix[thread] = energy_value;
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (thread < offset)
                shared_matrix[thread] += shared_matrix[thread + offset];
            __syncthreads();
        }
        if (thread == 0)
            energy.data[cluster] = shared_matrix[0];
    }
#endif
"""
)
def _fp32_async_cluster_gemv(
    factors: wp.array2d[wp.float32],
    residual: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    result: wp.array2d[wp.float32],
    energy: wp.array[wp.float32],
    compute_energy: wp.bool,
    cluster: int,
    thread: int,
): ...


@wp.kernel
def _insert_bsr_blocks_kernel(
    input_row: wp.array[wp.int32],
    input_col: wp.array[wp.int32],
    input_value: wp.array[wp.mat33f],
    input_count: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    overflow: wp.array[wp.int32],
):
    block = wp.tid()
    if block >= input_count[0]:
        return
    row = input_row[block]
    slot = wp.atomic_add(row_nnz, row, 1)
    capacity = row_offsets[row + 1] - row_offsets[row]
    if slot >= capacity:
        wp.atomic_sub(row_nnz, row, 1)
        overflow[0] = 1
        return
    output = row_offsets[row] + slot
    col_indices[output] = input_col[block]
    values[output] = input_value[block]


@wp.kernel
def _bsr_gemv_kernel(
    row_offsets: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    world_row_offsets: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    alpha: wp.float32,
    beta: wp.float32,
):
    world, local_row = wp.tid()
    if local_row >= world_row_count[world] or not world_active[world]:
        return

    row = world_row_offsets[world] + local_row
    result = wp.vec3f(0.0)
    row_begin = row_offsets[row]
    for local_block in range(row_nnz[row]):
        block = row_begin + local_block
        col = col_indices[block]
        x_col = wp.vec3f(x[3 * col], x[3 * col + 1], x[3 * col + 2])
        result += values[block] * x_col

    base = 3 * row
    y[base] = alpha * result[0] + beta * y[base]
    y[base + 1] = alpha * result[1] + beta * y[base + 1]
    y[base + 2] = alpha * result[2] + beta * y[base + 2]


@wp.kernel
def _bsr_spmv_dot_kernel(
    row_offsets: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    world_row_offsets: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    p: wp.array[wp.float32],
    ap: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
):
    world, chunk, thread = wp.tid()
    local_row = chunk * wp.block_dim() + thread
    contribution = wp.float32(0.0)
    if local_row < world_row_count[world] and world_active[world]:
        row = world_row_offsets[world] + local_row
        result = wp.vec3f(0.0)
        row_begin = row_offsets[row]
        for local_block in range(row_nnz[row]):
            block = row_begin + local_block
            col = col_indices[block]
            p_col = wp.vec3f(p[3 * col], p[3 * col + 1], p[3 * col + 2])
            result += values[block] * p_col
        base = 3 * row
        ap[base] = result[0]
        ap[base + 1] = result[1]
        ap[base + 2] = result[2]
        contribution = p[base] * result[0] + p[base + 1] * result[1] + p[base + 2] * result[2]
    chunk_sum = wp.tile_sum(wp.tile(contribution))[0]
    if thread == 0:
        wp.atomic_add(p_ap, world, chunk_sum)


@wp.kernel
def _cg_update_xr_save_kernel(
    tolerance: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    rz_old: wp.array[wp.float32],
    rz_new: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
    p: wp.array[wp.float32],
    ap: wp.array[wp.float32],
    x: wp.array[wp.float32],
    r: wp.array[wp.float32],
    vector_offsets: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
):
    world, local_dof = wp.tid()
    if local_dof >= dimensions[world]:
        return
    current_rz = rz_new[world]
    alpha = wp.where(
        residual[world] > tolerance[world] and p_ap[world] > 0.0,
        current_rz / p_ap[world],
        wp.float32(0.0),
    )
    dof = vector_offsets[world] + local_dof
    x[dof] += alpha * p[dof]
    r[dof] -= alpha * ap[dof]
    if local_dof == 0:
        rz_old[world] = current_rz


@wp.kernel
def _cg_update_xr_rr_kernel(
    tolerance: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    rz_old: wp.array[wp.float32],
    rz_new: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
    p: wp.array[wp.float32],
    ap: wp.array[wp.float32],
    x: wp.array[wp.float32],
    r: wp.array[wp.float32],
    vector_offsets: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    rr_partials: wp.array2d[wp.float32],
):
    world, chunk, thread = wp.tid()
    local_dof = chunk * wp.block_dim() + thread
    contribution = wp.float32(0.0)
    current_rz = rz_new[world]
    if local_dof < dimensions[world]:
        alpha = wp.where(
            residual[world] > tolerance[world] and p_ap[world] > 0.0,
            current_rz / p_ap[world],
            wp.float32(0.0),
        )
        dof = vector_offsets[world] + local_dof
        x[dof] += alpha * p[dof]
        value = r[dof] - alpha * ap[dof]
        r[dof] = value
        contribution = value * value
    chunk_sum = wp.tile_sum(wp.tile(contribution))[0]
    if thread == 0:
        rr_partials[world, chunk] = chunk_sum
        if chunk == 0:
            rz_old[world] = current_rz


@wp.kernel
def _finalize_rr_rz_kernel(
    dimensions: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_cluster_offsets: wp.array[wp.int32],
    world_cluster_count: wp.array[wp.int32],
    rr_partials: wp.array2d[wp.float32],
    cluster_energy: wp.array[wp.float32],
    rr: wp.array[wp.float32],
    rz: wp.array[wp.float32],
):
    world, thread = wp.tid()
    rr_partial = wp.float32(0.0)
    rz_partial = wp.float32(0.0)
    if world_active[world]:
        chunk_count = (dimensions[world] + 255) // 256
        chunk = thread
        while chunk < chunk_count:
            rr_partial += rr_partials[world, chunk]
            chunk += wp.block_dim()
        cluster_begin = world_cluster_offsets[world]
        cluster = thread
        while cluster < world_cluster_count[world]:
            rz_partial += cluster_energy[cluster_begin + cluster]
            cluster += wp.block_dim()
    rr_sum = wp.tile_sum(wp.tile(rr_partial))[0]
    rz_sum = wp.tile_sum(wp.tile(rz_partial))[0]
    if thread == 0:
        rr[world] = rr_sum
        rz[world] = rz_sum


@wp.kernel
def _cg_update_p_reset_kernel(
    tolerance: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    rz_old: wp.array[wp.float32],
    rz_new: wp.array[wp.float32],
    z: wp.array[wp.float32],
    p: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
    vector_offsets: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
):
    world, local_dof = wp.tid()
    if local_dof >= dimensions[world]:
        return
    beta = wp.where(
        residual[world] > tolerance[world] and rz_old[world] > 0.0,
        rz_new[world] / rz_old[world],
        wp.float32(0.0),
    )
    dof = vector_offsets[world] + local_dof
    p[dof] = z[dof] + beta * p[dof]
    if local_dof == 0:
        p_ap[world] = 0.0


@wp.kernel
def _accumulate_iterations_kernel(
    pass_iterations: wp.array[wp.int32],
    total_iterations: wp.array[wp.int32],
):
    world = wp.tid()
    total_iterations[world] += pass_iterations[world]


@wp.kernel
def _scatter_hierarchy_kernel(
    block_rows: wp.array[wp.int32],
    block_slots: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    ancestors: wp.array2d[wp.int32],
    factors: wp.array2d[wp.float32],
):
    block = wp.tid()
    row = block_rows[block]
    if block_slots[block] >= row_nnz[row]:
        return
    col = col_indices[block]
    value = values[block]

    for level in range(_MAX_LEVEL_COUNT):
        row_node = ancestors[row, level]
        col_node = ancestors[col, level]
        if row_node < 0 or col_node < 0:
            break
        cluster = row_node // _CLUSTER_NODE_COUNT
        if cluster != col_node // _CLUSTER_NODE_COUNT:
            continue
        local_row = 3 * (row_node % _CLUSTER_NODE_COUNT)
        local_col = 3 * (col_node % _CLUSTER_NODE_COUNT)
        matrix_row = cluster * _CLUSTER_DOF_COUNT + local_row
        for i in range(3):
            for j in range(3):
                wp.atomic_add(factors, matrix_row + i, local_col + j, value[i, j])


@wp.kernel
def _initialize_cluster_kernel(
    cluster_dof_count: wp.array[wp.int32],
    regularization: wp.float32,
    factors: wp.array2d[wp.float32],
):
    cluster, row, col = wp.tid()
    value = wp.float32(0.0)
    if row == col:
        value = wp.where(row < cluster_dof_count[cluster], regularization, wp.float32(1.0))
    factors[cluster * _CLUSTER_DOF_COUNT + row, col] = value


@wp.kernel
def _factorize_cluster_kernel(factors: wp.array2d[wp.float32]):
    cluster, thread = wp.tid()
    factor = wp.tile_load(
        factors,
        shape=(_CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
        offset=(cluster * _CLUSTER_DOF_COUNT, 0),
        storage="shared",
    )
    block_dim = wp.block_dim()
    element_count = _CLUSTER_DOF_COUNT * _CLUSTER_DOF_COUNT
    wp.tile_cholesky_inplace(factor)
    inverse = wp.tile_zeros(
        shape=(_CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
        dtype=wp.float32,
        storage="shared",
    )
    for iteration in range((element_count + 127) // 128):
        index = thread + iteration * block_dim
        if index < element_count:
            row = index // _CLUSTER_DOF_COUNT
            col = index % _CLUSTER_DOF_COUNT
            inverse[row, col] = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
    wp.tile_lower_solve_inplace(factor, inverse)
    wp.tile_upper_solve_inplace(wp.tile_transpose(factor), inverse)
    wp.tile_store(factors, inverse, offset=(cluster * _CLUSTER_DOF_COUNT, 0))


@wp.kernel(enable_backward=False)
def _apply_cluster_inverse_fp32_async_kernel(
    factors: wp.array2d[wp.float32],
    cluster_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    residual: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    hierarchy_z: wp.array2d[wp.float32],
    cluster_energy: wp.array[wp.float32],
    compute_energy: wp.bool,
):
    cluster, thread = wp.tid()
    if world_active[cluster_world[cluster]]:
        _fp32_async_cluster_gemv(
            factors,
            residual,
            slot_fine_begin,
            slot_fine_count,
            hierarchy_z,
            cluster_energy,
            compute_energy,
            cluster,
            thread,
        )
    elif compute_energy and thread == 0:
        cluster_energy[cluster] = 0.0


@wp.kernel
def _prolongate_kernel(
    ancestors: wp.array2d[wp.int32],
    fine_node_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_z: wp.array2d[wp.float32],
    z: wp.array[wp.float32],
):
    fine_node, component = wp.tid()
    if not world_active[fine_node_world[fine_node]]:
        z[3 * fine_node + component] = 0.0
        return
    value = wp.float32(0.0)
    for level in range(_MAX_LEVEL_COUNT):
        node = ancestors[fine_node, level]
        if node < 0:
            break
        value += hierarchy_z[3 * node + component, 0]
    z[3 * fine_node + component] = value


@dataclass(frozen=True)
class BatchedBSRMatrix:
    """Batched square matrices in row-contiguous 3x3 BSR storage."""

    row_offsets: wp.array
    row_nnz: wp.array
    col_indices: wp.array
    values: wp.array
    block_rows: wp.array
    block_slots: wp.array
    world_row_offsets: wp.array
    world_row_count: wp.array
    world_scalar_offsets: wp.array
    active_dims: wp.array
    fine_node_world: wp.array
    overflow: wp.array
    world_count: int
    total_row_count: int
    total_scalar_count: int
    max_row_count: int
    max_nnz_count: int
    device: wp.Device

    @classmethod
    def from_host(
        cls,
        row_offsets: Sequence[np.ndarray],
        col_indices: Sequence[np.ndarray],
        values: Sequence[np.ndarray],
        *,
        row_capacities: Sequence[np.ndarray] | None = None,
        device: wp.DeviceLike = None,
    ) -> BatchedBSRMatrix:
        """Upload independent local-indexed 3x3 BSR matrices."""
        if not (len(row_offsets) == len(col_indices) == len(values)) or not row_offsets:
            raise ValueError("row_offsets, col_indices, and values must have equal nonzero lengths")

        row_counts: list[int] = []
        global_rows = [0]
        packed_row_offsets = [0]
        packed_row_nnz: list[np.ndarray] = []
        packed_cols: list[int] = []
        packed_values: list[np.ndarray] = []
        packed_block_rows: list[int] = []
        packed_block_slots: list[int] = []
        node_world: list[np.ndarray] = []

        if row_capacities is not None and len(row_capacities) != len(row_offsets):
            raise ValueError("row_capacities must have one array per world")

        for world, (row_data, col_data, block_data) in enumerate(zip(row_offsets, col_indices, values, strict=True)):
            rows = np.asarray(row_data, dtype=np.int32)
            cols = np.asarray(col_data, dtype=np.int32)
            blocks = np.asarray(block_data, dtype=np.float32)
            row_count = rows.size - 1
            if row_count <= 0 or rows[0] != 0 or rows[-1] != cols.size:
                raise ValueError(f"invalid BSR row offsets for world {world}")
            if blocks.shape != (cols.size, 3, 3):
                raise ValueError(f"values for world {world} must have shape ({cols.size}, 3, 3)")
            if np.any(rows[1:] < rows[:-1]) or np.any(cols < 0) or np.any(cols >= row_count):
                raise ValueError(f"invalid BSR indices for world {world}")

            row_nnz = np.diff(rows).astype(np.int32)
            capacities = row_nnz if row_capacities is None else np.asarray(row_capacities[world], dtype=np.int32)
            if capacities.shape != (row_count,) or np.any(capacities < row_nnz):
                raise ValueError(f"row capacities for world {world} must cover every active row entry")

            row_base = global_rows[-1]
            row_counts.append(row_count)
            global_rows.append(row_base + row_count)
            packed_row_nnz.append(row_nnz)
            for local_row in range(row_count):
                begin = int(rows[local_row])
                count = int(row_nnz[local_row])
                capacity = int(capacities[local_row])
                for slot in range(capacity):
                    packed_block_rows.append(row_base + local_row)
                    packed_block_slots.append(slot)
                    if slot < count:
                        packed_cols.append(row_base + int(cols[begin + slot]))
                        packed_values.append(blocks[begin + slot])
                    else:
                        packed_cols.append(row_base + local_row)
                        packed_values.append(np.zeros((3, 3), dtype=np.float32))
                packed_row_offsets.append(packed_row_offsets[-1] + capacity)
            node_world.append(np.full(row_count, world, dtype=np.int32))

        device = wp.get_device(device)
        row_counts_np = np.asarray(row_counts, dtype=np.int32)
        global_rows_np = np.asarray(global_rows, dtype=np.int32)
        return cls(
            row_offsets=wp.array(np.asarray(packed_row_offsets, dtype=np.int32), device=device),
            row_nnz=wp.array(np.concatenate(packed_row_nnz), device=device),
            col_indices=wp.array(np.asarray(packed_cols, dtype=np.int32), device=device),
            values=wp.array(np.asarray(packed_values, dtype=np.float32), dtype=wp.mat33f, device=device),
            block_rows=wp.array(np.asarray(packed_block_rows, dtype=np.int32), device=device),
            block_slots=wp.array(np.asarray(packed_block_slots, dtype=np.int32), device=device),
            world_row_offsets=wp.array(global_rows_np[:-1], device=device),
            world_row_count=wp.array(row_counts_np, device=device),
            world_scalar_offsets=wp.array(3 * global_rows_np[:-1], device=device),
            active_dims=wp.array(3 * row_counts_np, device=device),
            fine_node_world=wp.array(np.concatenate(node_world), device=device),
            overflow=wp.zeros(1, dtype=wp.int32, device=device),
            world_count=len(row_counts),
            total_row_count=int(global_rows_np[-1]),
            total_scalar_count=3 * int(global_rows_np[-1]),
            max_row_count=int(row_counts_np.max()),
            max_nnz_count=len(packed_cols),
            device=device,
        )

    @property
    def storage_bytes(self) -> int:
        """Return GPU bytes used by the sparse matrix data and metadata."""
        arrays = (
            self.row_offsets,
            self.row_nnz,
            self.col_indices,
            self.values,
            self.block_rows,
            self.block_slots,
            self.world_row_offsets,
            self.world_row_count,
            self.world_scalar_offsets,
            self.active_dims,
            self.fine_node_world,
            self.overflow,
        )
        return sum(array.size * wp.types.type_size_in_bytes(array.dtype) for array in arrays)

    def begin_assembly(self) -> None:
        """Clear active row lengths before a device-side pattern rebuild."""
        self.row_nnz.zero_()
        self.overflow.zero_()

    def insert_blocks(
        self,
        rows: wp.array,
        cols: wp.array,
        values: wp.array,
        count: wp.array,
    ) -> None:
        """Insert global-indexed BSR blocks into fixed row-capacity storage.

        Duplicate entries are legal and are accumulated naturally by SpMV and
        preconditioner assembly. ``count`` is a one-element device array, so
        the active contribution count may change inside a captured graph.
        """
        if rows.shape != cols.shape or rows.shape[0] != values.shape[0]:
            raise ValueError("rows, cols, and values must have equal capacities")
        wp.launch(
            _insert_bsr_blocks_kernel,
            dim=rows.shape[0],
            inputs=[
                rows,
                cols,
                values,
                count,
                self.row_offsets,
                self.row_nnz,
                self.col_indices,
                self.values,
                self.overflow,
            ],
            device=self.device,
        )

    def gemv(
        self,
        x: wp.array,
        y: wp.array,
        world_active: wp.array,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Compute ``y = alpha * A * x + beta * y``."""
        wp.launch(
            _bsr_gemv_kernel,
            dim=(self.world_count, self.max_row_count),
            inputs=[
                self.row_offsets,
                self.row_nnz,
                self.col_indices,
                self.values,
                self.world_row_offsets,
                self.world_row_count,
                world_active,
                x,
                y,
                wp.float32(alpha),
                wp.float32(beta),
            ],
            device=self.device,
        )


class MASPreconditioner:
    """Multilevel additive Schwarz preconditioner for :class:`BatchedBSRMatrix`."""

    def __init__(
        self,
        matrix: BatchedBSRMatrix,
        *,
        regularization: float = 1.0e-6,
    ):
        if not matrix.device.is_cuda:
            raise ValueError("MASPreconditioner currently requires a CUDA device")
        self.matrix = matrix
        self.device = matrix.device
        self.regularization = float(regularization)
        self.compute_cluster_energy = False
        metadata = self._build_hierarchy_metadata(matrix)
        self.level_count = metadata["level_count"]
        self.cluster_count = metadata["cluster_count"]

        self.ancestors = wp.array(metadata["ancestors"], device=self.device)
        self.cluster_dof_count = wp.array(metadata["cluster_dof_count"], device=self.device)
        self.cluster_world = wp.array(metadata["cluster_world"], device=self.device)
        self.world_cluster_offsets = wp.array(metadata["world_cluster_offsets"], device=self.device)
        self.world_cluster_count = wp.array(metadata["world_cluster_count"], device=self.device)
        self.slot_fine_begin = wp.array(metadata["slot_fine_begin"], device=self.device)
        self.slot_fine_count = wp.array(metadata["slot_fine_count"], device=self.device)
        self.factors = wp.zeros(
            (self.cluster_count * _CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
            dtype=wp.float32,
            device=self.device,
        )
        hierarchy_scalar_count = self.cluster_count * _CLUSTER_DOF_COUNT
        self.hierarchy_z = wp.zeros((hierarchy_scalar_count, 1), dtype=wp.float32, device=self.device)
        self.cluster_energy = wp.zeros(self.cluster_count, dtype=wp.float32, device=self.device)
        self.refit()

    @staticmethod
    def _build_hierarchy_metadata(matrix: BatchedBSRMatrix) -> dict[str, np.ndarray | int]:
        row_counts = matrix.world_row_count.numpy()
        row_bases = np.concatenate(([0], np.cumsum(row_counts[:-1], dtype=np.int64))).astype(np.int32)
        ancestors = np.full((matrix.total_row_count, _MAX_LEVEL_COUNT), -1, dtype=np.int32)
        cluster_dof_count: list[int] = []
        cluster_world: list[int] = []
        slot_fine_begin: list[int] = []
        slot_fine_count: list[int] = []
        world_cluster_offsets: list[int] = []
        world_cluster_count: list[int] = []
        cluster_count = 0
        max_level_count = 0

        for world, (fine_base, fine_count) in enumerate(zip(row_bases, row_counts, strict=True)):
            world_cluster_begin = cluster_count
            world_cluster_offsets.append(world_cluster_begin)
            level_size = int(fine_count)
            level = 0
            fine_indices = np.arange(int(fine_count), dtype=np.int32)
            while True:
                max_level_count = max(max_level_count, level + 1)
                cluster_count_level = (level_size + _CLUSTER_NODE_COUNT - 1) // _CLUSTER_NODE_COUNT
                level_cluster_begin = cluster_count
                fine_span = _CLUSTER_NODE_COUNT**level
                for cluster_local in range(cluster_count_level):
                    node_begin = cluster_local * _CLUSTER_NODE_COUNT
                    node_count = min(_CLUSTER_NODE_COUNT, level_size - node_begin)
                    cluster_dof_count.append(3 * node_count)
                    cluster_world.append(world)
                    for local in range(_CLUSTER_NODE_COUNT):
                        level_node = node_begin + local
                        begin = int(fine_base) + level_node * fine_span
                        count = max(0, min(fine_span, int(fine_base + fine_count) - begin))
                        slot_fine_begin.append(begin if count else 0)
                        slot_fine_count.append(count)
                level_nodes = fine_indices // fine_span
                ancestors[fine_base : fine_base + fine_count, level] = (
                    level_cluster_begin + level_nodes // _CLUSTER_NODE_COUNT
                ) * _CLUSTER_NODE_COUNT + level_nodes % _CLUSTER_NODE_COUNT
                cluster_count += cluster_count_level
                if level_size <= _CLUSTER_NODE_COUNT or level + 1 >= _MAX_LEVEL_COUNT:
                    break
                level_size = cluster_count_level
                level += 1
            world_cluster_count.append(cluster_count - world_cluster_begin)

        return {
            "ancestors": ancestors,
            "cluster_dof_count": np.asarray(cluster_dof_count, dtype=np.int32),
            "cluster_world": np.asarray(cluster_world, dtype=np.int32),
            "slot_fine_begin": np.asarray(slot_fine_begin, dtype=np.int32),
            "slot_fine_count": np.asarray(slot_fine_count, dtype=np.int32),
            "world_cluster_offsets": np.asarray(world_cluster_offsets, dtype=np.int32),
            "world_cluster_count": np.asarray(world_cluster_count, dtype=np.int32),
            "level_count": max_level_count,
            "cluster_count": cluster_count,
        }

    @property
    def storage_bytes(self) -> int:
        """Return GPU bytes used by the hierarchy, factors, and work vectors."""
        arrays = [
            self.ancestors,
            self.cluster_dof_count,
            self.cluster_world,
            self.world_cluster_offsets,
            self.world_cluster_count,
            self.slot_fine_begin,
            self.slot_fine_count,
            self.factors,
            self.hierarchy_z,
            self.cluster_energy,
        ]
        return sum(array.size * wp.types.type_size_in_bytes(array.dtype) for array in arrays)

    def refit(self) -> None:
        """Reassemble and refactor numeric blocks after sparse values or pattern change."""
        wp.launch(
            _initialize_cluster_kernel,
            dim=(self.cluster_count, _CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
            inputs=[self.cluster_dof_count, wp.float32(self.regularization), self.factors],
            device=self.device,
        )
        wp.launch(
            _scatter_hierarchy_kernel,
            dim=self.matrix.max_nnz_count,
            inputs=[
                self.matrix.block_rows,
                self.matrix.block_slots,
                self.matrix.row_nnz,
                self.matrix.col_indices,
                self.matrix.values,
                self.ancestors,
                self.factors,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            _factorize_cluster_kernel,
            dim=self.cluster_count,
            inputs=[self.factors],
            block_dim=128,
            device=self.device,
        )

    def apply(self, r: wp.array, z: wp.array, world_active: wp.array) -> None:
        """Apply the preconditioner, ``z = M^-1 r``."""
        wp.launch_tiled(
            _apply_cluster_inverse_fp32_async_kernel,
            dim=self.cluster_count,
            inputs=[
                self.factors,
                self.cluster_world,
                world_active,
                r,
                self.slot_fine_begin,
                self.slot_fine_count,
                self.hierarchy_z,
                self.cluster_energy,
                self.compute_cluster_energy,
            ],
            block_dim=256,
            device=self.device,
        )
        wp.launch(
            _prolongate_kernel,
            dim=(self.matrix.total_row_count, 3),
            inputs=[self.ancestors, self.matrix.fine_node_world, world_active, self.hierarchy_z, z],
            device=self.device,
        )


class _BSRFastCGSolver(CGSolver):
    """CG specialization that fuses BSR SpMV with its dot reduction."""

    def __init__(
        self,
        matrix: BatchedBSRMatrix,
        preconditioner: MASPreconditioner,
        *args,
        use_energy_dot: bool,
        **kwargs,
    ):
        self.bsr_matrix = matrix
        self.mas_preconditioner = preconditioner
        self.use_energy_dot = use_energy_dot
        preconditioner.compute_cluster_energy = self.use_energy_dot
        super().__init__(*args, **kwargs)

    def _allocate(self):
        super()._allocate()
        self.p_ap = wp.zeros(self.n_worlds, dtype=wp.float32, device=self.device)
        self.scalar_chunk_count = (self.maxdims + 255) // 256
        self.rr_partials = wp.zeros(
            (self.n_worlds, self.scalar_chunk_count),
            dtype=wp.float32,
            device=self.device,
        )

    def solve(self, *args, **kwargs):
        self.p_ap.zero_()
        return super().solve(*args, **kwargs)

    def do_iteration(self, p, Ap, rz_old, rz_new, z, x, r, r_norm_sq, active_dims, world_active):
        chunks = (self.bsr_matrix.max_row_count + 255) // 256
        wp.launch_tiled(
            _bsr_spmv_dot_kernel,
            dim=(self.n_worlds, chunks),
            inputs=[
                self.bsr_matrix.row_offsets,
                self.bsr_matrix.row_nnz,
                self.bsr_matrix.col_indices,
                self.bsr_matrix.values,
                self.bsr_matrix.world_row_offsets,
                self.bsr_matrix.world_row_count,
                world_active,
                p,
                Ap,
                self.p_ap,
            ],
            block_dim=256,
            device=self.device,
        )
        if self.use_energy_dot:
            wp.launch_tiled(
                _cg_update_xr_rr_kernel,
                dim=(self.n_worlds, self.scalar_chunk_count),
                inputs=[
                    self.atol_sq,
                    r_norm_sq,
                    rz_old,
                    rz_new,
                    self.p_ap,
                    p,
                    Ap,
                    x,
                    r,
                    self.vio,
                    active_dims,
                    self.rr_partials,
                ],
                block_dim=256,
                device=self.device,
            )
            self.Mi.matvec(r, z, world_active)
            wp.launch_tiled(
                _finalize_rr_rz_kernel,
                dim=self.n_worlds,
                inputs=[
                    active_dims,
                    world_active,
                    self.mas_preconditioner.world_cluster_offsets,
                    self.mas_preconditioner.world_cluster_count,
                    self.rr_partials,
                    self.mas_preconditioner.cluster_energy,
                    r_norm_sq,
                    rz_new,
                ],
                block_dim=256,
                device=self.device,
            )
        else:
            wp.launch(
                _cg_update_xr_save_kernel,
                dim=(self.n_worlds, self.maxdims),
                inputs=[
                    self.atol_sq,
                    r_norm_sq,
                    rz_old,
                    rz_new,
                    self.p_ap,
                    p,
                    Ap,
                    x,
                    r,
                    self.vio,
                    active_dims,
                ],
                device=self.device,
            )
            self.update_rr_rz(r, z, self.r_repeated, active_dims, world_active)
        wp.launch(
            _cg_update_p_reset_kernel,
            dim=(self.n_worlds, self.maxdims),
            inputs=[
                self.atol_sq,
                r_norm_sq,
                rz_old,
                rz_new,
                z,
                p,
                self.p_ap,
                self.vio,
                active_dims,
            ],
            device=self.device,
        )


class BatchedMASPCG:
    """Float32 batched PCG using sparse BSR and multilevel additive Schwarz."""

    def __init__(
        self,
        matrix: BatchedBSRMatrix,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-5,
        max_iterations: int | None = None,
        use_cuda_graph: bool = True,
        loop_granularity: int = 1,
        regularization: float = 1.0e-6,
        refinement_passes: int = 1,
    ):
        self.matrix = matrix
        if refinement_passes < 1:
            raise ValueError("refinement_passes must be at least one")
        self.refinement_passes = int(refinement_passes)
        self.preconditioner = MASPreconditioner(
            matrix,
            regularization=regularization,
        )
        self.world_active = wp.full(matrix.world_count, True, dtype=wp.bool, device=matrix.device)
        max_iterations = max_iterations or max(16, 3 * matrix.max_row_count)
        maxiter = wp.full(matrix.world_count, max_iterations, dtype=wp.int32, device=matrix.device)

        def gemv(x, y, active, alpha, beta):
            matrix.gemv(x, y, active, alpha, beta)

        def matvec(x, y, active):
            matrix.gemv(x, y, active)

        operator = BatchedLinearOperator(
            gemv,
            matrix.world_count,
            3 * matrix.max_row_count,
            matrix.active_dims,
            matrix.device,
            wp.float32,
            matvec_fn=matvec,
            vio=matrix.world_scalar_offsets,
            total_vec_size=matrix.total_scalar_count,
        )

        def precondition(r, z, active):
            self.preconditioner.apply(r, z, active)

        inverse = BatchedLinearOperator(
            precondition,
            matrix.world_count,
            3 * matrix.max_row_count,
            matrix.active_dims,
            matrix.device,
            wp.float32,
            matvec_fn=precondition,
            vio=matrix.world_scalar_offsets,
            total_vec_size=matrix.total_scalar_count,
        )
        self._solver = _BSRFastCGSolver(
            matrix,
            self.preconditioner,
            operator,
            use_energy_dot=matrix.world_count == 1 and matrix.max_row_count >= 2048,
            world_active=self.world_active,
            atol=float(atol),
            rtol=float(rtol),
            maxiter=maxiter,
            Mi=inverse,
            use_graph=use_cuda_graph,
            loop_granularity=loop_granularity,
        )
        self.iterations = self._solver.cur_iter
        self.residual = self._solver.residual
        self._total_iterations = None
        if self.refinement_passes > 1:
            self._total_iterations = wp.zeros(matrix.world_count, dtype=wp.int32, device=matrix.device)
            self.iterations = self._total_iterations

    @property
    def storage_bytes(self) -> int:
        """Return matrix and preconditioner storage, excluding PCG work vectors."""
        return self.matrix.storage_bytes + self.preconditioner.storage_bytes

    def solve(
        self,
        b: wp.array,
        x: wp.array,
        *,
        world_active: wp.array | None = None,
        refit: bool = True,
    ):
        """Solve ``A x = b`` and return per-world iteration and residual arrays."""
        active = self.world_active if world_active is None else world_active
        if refit:
            self.preconditioner.refit()
        if self.refinement_passes == 1:
            return self._solver.solve(b, x, world_active=active)
        self._total_iterations.zero_()
        for _pass in range(self.refinement_passes):
            self._solver.solve(b, x, world_active=active)
            wp.launch(
                _accumulate_iterations_kernel,
                dim=self.matrix.world_count,
                inputs=[self._solver.cur_iter, self._total_iterations],
                device=self.matrix.device,
            )
        return self.iterations, self.residual

    def capture(
        self,
        b: wp.array,
        x: wp.array,
        *,
        refit: bool = True,
        zero_initial_guess: bool = True,
    ) -> wp.Graph:
        """Capture numeric refit and the device-conditional PCG solve as one graph."""
        with wp.ScopedCapture(self.matrix.device) as capture:
            if zero_initial_guess:
                x.zero_()
            self.solve(b, x, refit=refit)
        return capture.graph
