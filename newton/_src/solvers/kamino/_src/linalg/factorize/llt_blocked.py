# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: Blocked LLT (i.e. Cholesky) factorization using Warp's Tile API.

Performance notes
-----------------
Compared to the original implementation, this version:
  * Uses the 3-operand accumulator form of ``wp.tile_matmul(A, B, C, alpha=-1.0)``
    which fuses the trailing/rhs update into a single FMA instead of computing
    a temporary tile and doing a separate in-place subtract.
  * Uses ``wp.tile_lower_solve_inplace`` / ``wp.tile_upper_solve_inplace`` and
    ``wp.tile_cholesky_inplace`` so the factorize / triangular solves write
    back into their input tile (no extra register / shared tile round-trip).

Identity padding is retained in both factorize and solve kernels: in Kamino
each block's flat storage region may be strictly larger than the active
``n_i`` (because ``maxdim[i]`` can exceed ``dim[i]``), so the padded rows of
``L_kk`` / ``b_i`` / ``y_i`` / ``x_i`` contain arbitrary values from
neighboring (possibly active) blocks and must be masked before the
triangular solves.
"""

from ctypes import sizeof
from functools import cache

import warp as wp

from ...core.types import float32, int32

###
# Module interface
###

__all__ = [
    "llt_blocked_factorize",
    "llt_blocked_solve",
    "llt_blocked_solve_inplace",
    "make_llt_blocked_factorize_kernel",
    "make_llt_blocked_solve_inplace_kernel",
    "make_llt_blocked_solve_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###

get_array_ptr_cpp = """return (uint64_t)arr.data;"""
"""A native C++ function to get the raw pointer of a warp array."""


def make_get_array_offset_ptr_func(dtype):
    """Creates a function to get the offset pointer of a warp array."""

    # Define a Warp wrapper around a native C++ function to get the raw pointer of a warp array
    @wp.func_native(get_array_ptr_cpp)
    def get_dtype_array_ptr(arr: wp.array(dtype=dtype)) -> wp.uint64: ...

    # Define a Warp function to get the raw pointer of a warp array with an offset
    @wp.func
    def get_dtype_array_offset_ptr(arr: wp.array(dtype=dtype), start_index: int) -> wp.uint64:
        return get_dtype_array_ptr(arr) + wp.uint64(start_index * wp.static(sizeof(dtype._type_)))

    return get_dtype_array_offset_ptr


get_int32_array_offset_ptr = make_get_array_offset_ptr_func(wp.int32)
"""A Warp function to get the offset pointer of a int32 warp array."""

get_float32_array_offset_ptr = make_get_array_offset_ptr_func(wp.float32)
"""A Warp function to get the offset pointer of a float32 warp array."""


###
# Kernels
###


@cache
def make_llt_blocked_factorize_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_factorize_kernel(
        # Inputs:
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        A: wp.array(dtype=float32),
        # Outputs:
        L: wp.array(dtype=float32),
    ):
        # Retrieve the thread index and thread-block configuration
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        # Retrieve the matrix block dimensions and size
        n_i = dim[tid]
        A_i_start = mio[tid]

        # Retrieve a pointer to the start of the i-th matrix in A
        A_i_ptr = get_float32_array_offset_ptr(A, A_i_start)
        L_i_ptr = get_float32_array_offset_ptr(L, A_i_start)

        # Create a temporary warp array pointing to the i-th matrix
        A_i = wp.array(ptr=A_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)

        # Round up the active dimension to the next multiple of block_size
        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n_i_padded, block_size):
            end = k + block_size

            # Load current diagonal block A[k:end, k:end]
            A_kk_tile = wp.tile_load(A_i, shape=(block_size, block_size), offset=(k, k), storage="shared")

            # Identity-pad the tail of the last diagonal block so that the
            # subsequent factorize / triangular solve operates on a valid
            # square tile even when n_i is not a multiple of block_size.
            if k + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for i in range(num_iterations):
                    linear_index = tid_block + i * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= n_i or k + col >= n_i:
                        value = wp.where(row == col, float32(1), float32(0))
                    A_kk_tile[row, col] = value

            # Trailing update: A_kk -= sum_{j<k} L_kj L_kj^T  (fused FMA).
            for j in range(0, k, block_size):
                L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(k, j))
                L_block_T = wp.tile_transpose(L_block)
                wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            # In-place Cholesky factorization of the current diagonal block
            wp.tile_cholesky_inplace(A_kk_tile)
            wp.tile_store(L_i, A_kk_tile, offset=(k, k))

            # Process the sub-diagonal blocks below the current diagonal block
            for i in range(end, n_i_padded, block_size):
                A_ik_tile = wp.tile_load(A_i, shape=(block_size, block_size), offset=(i, k), storage="shared")

                # Identity-pad the tail of the last sub-diagonal block.
                if i + block_size > n_i or k + block_size > n_i:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                    for ii in range(num_iterations):
                        linear_index = tid_block + ii * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= n_i or k + col >= n_i:
                            value = wp.where(i + row == k + col, float32(1), float32(0))
                        A_ik_tile[row, col] = value

                # Trailing update: A_ik -= sum_{j<k} L_ij L_kj^T  (fused FMA).
                for j in range(0, k, block_size):
                    L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                    L_2_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(k, j))
                    L_T_tile = wp.tile_transpose(L_2_tile)
                    wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                # Solve L_kk * X^T = A_ik^T  =>  L_ik = X.
                # (In-place triangular solve on the transposed tile.)
                A_ik_T_tile = wp.tile_transpose(A_ik_tile)
                wp.tile_lower_solve_inplace(A_kk_tile, A_ik_T_tile)
                L_ik_tile = wp.tile_transpose(A_ik_T_tile)
                wp.tile_store(L_i, L_ik_tile, offset=(i, k))

    return llt_blocked_factorize_kernel


@cache
def make_llt_blocked_solve_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_solve_kernel(
        # Inputs:
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        vio: wp.array(dtype=int32),
        L: wp.array(dtype=float32),
        b: wp.array(dtype=float32),
        # Outputs:
        y: wp.array(dtype=float32),
        x: wp.array(dtype=float32),
    ):
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_i = dim[tid]
        L_i_start = mio[tid]
        v_i_start = vio[tid]

        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        b_i_ptr = get_float32_array_offset_ptr(b, v_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)

        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        b_i = wp.array(ptr=b_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L * y = b
        for i in range(0, n_i_padded, block_size):
            rhs_tile = wp.tile_load(b_i, shape=(block_size, 1), offset=(i, 0))

            # Mask out any garbage values loaded past the active dimension
            # (the flat rhs buffer may be larger than n_i).
            if i + block_size > n_i:
                num_tile_elements = block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    row = (tid_block + ii * num_threads_per_block) % num_tile_elements
                    if i + row >= n_i:
                        rhs_tile[row, 0] = float32(0)

            for j in range(0, i, block_size):
                L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            # Identity-pad the tail of the last diagonal block so the
            # triangular solve produces 0 for the inactive rows.
            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = (tid_block + ii * num_threads_per_block) % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_tile[row, col]
                    if i + row >= n_i:
                        value = wp.where(row == col, float32(1), float32(0))
                    L_tile[row, col] = value

            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y_i, rhs_tile, offset=(i, 0))

        # Backward substitution: solve L^T * x = y
        for i in range(n_i_padded - block_size, -1, -block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(y_i, shape=(block_size, 1), offset=(i, 0))
            # Note: y was freshly written by the forward pass so the tail rows
            # of the last y-tile are already zero (see masking above). No
            # additional rhs masking needed here.

            for j in range(i_end, n_i_padded, block_size):
                L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                L_T_tile = wp.tile_transpose(L_tile)
                x_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(j, 0))
                wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            # Identity-pad the tail of the last diagonal block.
            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = (tid_block + ii * num_threads_per_block) % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_tile[row, col]
                    if i + row >= n_i:
                        value = wp.where(row == col, float32(1), float32(0))
                    L_tile[row, col] = value

            L_T_tile = wp.tile_transpose(L_tile)
            wp.tile_upper_solve_inplace(L_T_tile, rhs_tile)
            wp.tile_store(x_i, rhs_tile, offset=(i, 0))

    return llt_blocked_solve_kernel


@cache
def make_llt_blocked_solve_inplace_kernel(block_size: int):
    @wp.kernel
    def llt_blocked_solve_inplace_kernel(
        # Inputs:
        dim: wp.array(dtype=int32),
        mio: wp.array(dtype=int32),
        vio: wp.array(dtype=int32),
        L: wp.array(dtype=float32),
        # Outputs:
        y: wp.array(dtype=float32),
        x: wp.array(dtype=float32),
    ):
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        n_i = dim[tid]
        L_i_start = mio[tid]
        v_i_start = vio[tid]

        L_i_ptr = get_float32_array_offset_ptr(L, L_i_start)
        y_i_ptr = get_float32_array_offset_ptr(y, v_i_start)
        x_i_ptr = get_float32_array_offset_ptr(x, v_i_start)

        L_i = wp.array(ptr=L_i_ptr, shape=(n_i, n_i), dtype=wp.float32)
        y_i = wp.array(ptr=y_i_ptr, shape=(n_i, 1), dtype=wp.float32)
        x_i = wp.array(ptr=x_i_ptr, shape=(n_i, 1), dtype=wp.float32)

        n_i_padded = ((n_i + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L * y = b (b is initially stored in x).
        for i in range(0, n_i_padded, block_size):
            rhs_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(i, 0))

            # Mask out any garbage rhs values in the tail.
            if i + block_size > n_i:
                num_tile_elements = block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    row = (tid_block + ii * num_threads_per_block) % num_tile_elements
                    if i + row >= n_i:
                        rhs_tile[row, 0] = float32(0)

            for j in range(0, i, block_size):
                L_block = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, j))
                y_block = wp.tile_load(y_i, shape=(block_size, 1), offset=(j, 0))
                wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            # Identity-pad the tail of the last diagonal block.
            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = (tid_block + ii * num_threads_per_block) % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_tile[row, col]
                    if i + row >= n_i:
                        value = wp.where(row == col, float32(1), float32(0))
                    L_tile[row, col] = value

            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y_i, rhs_tile, offset=(i, 0))

        # Backward substitution: solve L^T * x = y
        for i in range(n_i_padded - block_size, -1, -block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(y_i, shape=(block_size, 1), offset=(i, 0))

            for j in range(i_end, n_i_padded, block_size):
                L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(j, i))
                L_T_tile = wp.tile_transpose(L_tile)
                x_tile = wp.tile_load(x_i, shape=(block_size, 1), offset=(j, 0))
                wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_i, shape=(block_size, block_size), offset=(i, i))
            # Identity-pad the tail of the last diagonal block.
            if i + block_size > n_i:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for ii in range(num_iterations):
                    linear_index = (tid_block + ii * num_threads_per_block) % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = L_tile[row, col]
                    if i + row >= n_i:
                        value = wp.where(row == col, float32(1), float32(0))
                    L_tile[row, col] = value

            L_T_tile = wp.tile_transpose(L_tile)
            wp.tile_upper_solve_inplace(L_T_tile, rhs_tile)
            wp.tile_store(x_i, rhs_tile, offset=(i, 0))

    return llt_blocked_solve_inplace_kernel


###
# Launchers
###


def llt_blocked_factorize(
    kernel,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    A: wp.array(dtype=float32),
    L: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 128,  # TODO: Rename this to be clearer that this is the number of threads per TILE block and not matrix block
    device: wp.DeviceLike = None,
):
    """
    Launches the blocked Cholesky factorization kernel for a block partitioned matrix.

    Args:
        kernel: The kernel function to use for the blocked factorization.
        num_blocks (int): The number of matrix blocks to process.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
        dim (wp.array): An array of shape `(num_blocks,)` containing the active dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offset (mio) of each matrix block.
        A (wp.array): The flat input array containing the input matrix blocks to be factorized.
        L (wp.array): The flat output array containing the factorization of each matrix block.
    """
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, mio, A, L], block_dim=block_dim, device=device)


def llt_blocked_solve(
    kernel,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 64,
    device: wp.DeviceLike = None,
):
    """
    Launches the blocked Cholesky solve kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offsets of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the vector index offsets of each vector block.
        L (wp.array2d): The flat input array containing the Cholesky factorization of each matrix block.
        b (wp.array): The flat input array containing the stacked right-hand side vectors.
        y (wp.array): The output array where the intermediate result will be stored.
        x (wp.array): The output array where the solution to the linear system `A @ x = b` will be stored.
        kernel: The kernel function to use for the blocked solve.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(
        kernel=kernel, dim=num_blocks, inputs=[dim, mio, vio, L, b, y, x], block_dim=block_dim, device=device
    )


def llt_blocked_solve_inplace(
    kernel,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    num_blocks: int = 1,
    block_dim: int = 64,
    device: wp.DeviceLike = None,
):
    """
    Launches the blocked Cholesky in-place solve kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the matrix index offsets of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the vector index offsets of each vector block.
        L (wp.array2d): The flat input array containing the Cholesky factorization of each matrix block.
        x (wp.array): The input/output array where the solution to the linear system `A @ x = b` will be stored in-place.
        kernel: The kernel function to use for the blocked in-place solve.
        block_dim (int): The dimension of the thread block to use for the kernel launch.
    """
    wp.launch_tiled(kernel=kernel, dim=num_blocks, inputs=[dim, mio, vio, L, y, x], block_dim=block_dim, device=device)
