# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: ND-reordered semi-sparse Blocked LLT solver.

Mirrors :class:`LLTBlockedSolver` from :mod:`linalg.linear` but transparently
computes and applies a per-block fill-reducing nested-dissection permutation,
and uses a per-block tile-granularity zero-block mask ("semi-sparse") to skip
work on guaranteed-zero tiles during factorize and solve.

The caller-visible API is identical to :class:`LLTBlockedSolver`:

.. code-block:: python

    solver = LLTBlockedNDSolver(operator=operator, block_size=32, dtype=float32)
    solver.compute(A)          # factorizes; reordering is internal
    solver.solve(b, x)         # or: solver.solve_inplace(x)

The reordering ``P`` and its inverse ``inv_P`` are stored on the solver next
to the factorization buffer ``L``. They are exposed as read-only properties
for debugging/introspection.
"""

from typing import Any

import warp as wp

from ...core.types import FloatType, float32, int32, override
from ..core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from ..linear import DirectSolver
from . import nested_dissection as _nd
from . import rcm as _rcm
from . import rcm_batch as _rcm_batch
from .llt_blocked_nd import (
    llt_blocked_nd_build_inv_P,
    llt_blocked_nd_build_tile_pattern,
    llt_blocked_nd_factorize,
    llt_blocked_nd_permute_matrix,
    llt_blocked_nd_permute_vector,
    llt_blocked_nd_solve,
    llt_blocked_nd_solve_inplace,
    llt_blocked_nd_symbolic_fill_in,
    make_llt_blocked_nd_build_inv_P_kernel,
    make_llt_blocked_nd_build_tile_pattern_kernel,
    make_llt_blocked_nd_factorize_kernel,
    make_llt_blocked_nd_permute_matrix_kernel,
    make_llt_blocked_nd_permute_vector_kernel,
    make_llt_blocked_nd_solve_inplace_kernel,
    make_llt_blocked_nd_solve_kernel,
    make_llt_blocked_nd_symbolic_fill_in_kernel,
)


###
# Module interface
###

__all__ = ["LLTBlockedNDSolver"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


class LLTBlockedNDSolver(DirectSolver):
    """ND-reordered, semi-sparse Blocked LLT (Cholesky) solver.

    Same public API as :class:`newton._src.solvers.kamino._src.linalg.linear.LLTBlockedSolver`.
    Internally:

    1. ``compute(A)`` / ``_factorize_impl``:

       a. Runs one nested-dissection launch per block to compute per-block
          permutations ``P_i`` (stored concatenated in ``self._P``).
       b. Builds ``inv_P`` from ``P``.
       c. Permutes ``A -> A_hat`` (``A_hat_i = P_i A_i P_i^T``).
       d. Builds a tile-level sparsity mask for each block by thresholding
          ``|A_hat_i|`` and then inflates it by a classical block symbolic
          Cholesky fill-in step. Both steps run on the GPU with fixed launch
          shapes and are CUDA-graph capturable.
       e. Numerically factorizes ``A_hat = L L^T`` with the tile-pattern-aware
          kernel that skips guaranteed-zero tiles.

    2. ``solve(b, x)`` / ``_solve_impl``:

       a. Permutes ``b -> b_hat``.
       b. Runs the tile-pattern-aware blocked forward/backward solve.
       c. Un-permutes ``x_hat -> x``.

    All launches use fixed dimensions known at ``finalize()`` time, so the
    entire ``compute`` and ``solve`` flow can be captured into a CUDA graph
    by the caller (same pattern as :class:`LLTBlockedSolver`).
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        block_size: int = 32,
        solve_block_dim: int = 128,
        factortize_block_dim: int = 128,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        # ND-specific options
        nd_tol: float = 0.0,
        nd_power_iters=None,
        # Reordering backend.
        # - 'rcm_batch' (default): kernel-based Reverse Cuthill-McKee with all
        #   blocks processed in a single batched launch per stage. Much lower
        #   per-call overhead than 'spectral' and wins on bandwidth-structured
        #   matrices (up to ~2x vs LLTBlockedSolver), ties elsewhere.
        # - 'rcm'      : per-block RCM (see :mod:`rcm`); same algorithm, one
        #   recorded launch set per block. Kept for diagnostics / single-block
        #   use; prefer 'rcm_batch' for multi-block workloads.
        # - 'spectral' : nested_dissection (spectral bisection via power
        #   iteration). Fill-reducing but high launch-count per block. Kept
        #   as an opt-in for research / dense-matrix experiments.
        reorder_algorithm: str = "rcm_batch",
        # For 'rcm' only: cap on BFS steps per block. None => auto
        # (``2*ceil(sqrt(n)) + 4``).
        rcm_max_bfs_iters: int | None = None,
        dtype: FloatType = float32,
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        """
        Args:
            operator: optional operator; if provided, :meth:`finalize` is called during init.
            block_size: tile block size passed to the kernel factories.
            solve_block_dim: thread-block size for solve kernels.
            factortize_block_dim: thread-block size for factorize kernel.
            nd_tol: threshold below which an off-diagonal entry is treated as zero
                by the ND partitioner and by the tile-pattern builder.
            nd_power_iters: number of power iterations per ND level. ``None``
                forwards to the ND module's default ('auto').
        """
        # LLT-specific internal data
        self._L: wp.array | None = None
        self._y: wp.array | None = None
        # Reordering + semi-sparse state
        self._A_hat: wp.array | None = None
        self._b_hat: wp.array | None = None
        self._x_hat: wp.array | None = None
        self._P: wp.array | None = None
        self._inv_P: wp.array | None = None
        self._tile_pattern: wp.array | None = None
        self._tpo: wp.array | None = None
        self._max_dim: int = 0
        # Per-block ND launch callbacks (one per block). Each is a closure
        # that calls a recorded ND launch writing a slice of self._P.
        self._nd_callbacks: list = []
        # ND scratch must be kept alive to keep the recorded launches valid.
        self._nd_scratches: list = []
        # Views into flat A used by the ND launches. These wp.array views are
        # bound at finalize() time and must outlive the recorded launches.
        self._A_views: list = []
        self._A_views_attached_to: wp.array | None = None

        # Cache the fixed block/tile dimensions
        self._block_size: int = block_size
        self._solve_block_dim: int = solve_block_dim
        self._factortize_block_dim: int = factortize_block_dim

        # ND options
        self._nd_tol: float = nd_tol
        self._nd_power_iters = nd_power_iters if nd_power_iters is not None else 'auto'
        if reorder_algorithm not in ("spectral", "rcm", "rcm_batch"):
            raise ValueError(
                "reorder_algorithm must be 'spectral', 'rcm', or 'rcm_batch'; "
                f"got {reorder_algorithm!r}"
            )
        self._reorder_algorithm: str = reorder_algorithm
        self._rcm_max_bfs_iters = rcm_max_bfs_iters

        # Build kernels (cached by block_size / max_dim at allocate time).
        self._factorize_kernel = make_llt_blocked_nd_factorize_kernel(block_size)
        self._solve_kernel = make_llt_blocked_nd_solve_kernel(block_size)
        self._solve_inplace_kernel = make_llt_blocked_nd_solve_inplace_kernel(block_size)
        # Auxiliary kernels resolved in _allocate_impl once we know max_dim.
        self._permute_matrix_kernel = None
        self._permute_vector_kernel = None
        self._build_inv_P_kernel = None
        self._build_tile_pattern_kernel = None
        self._symbolic_fill_in_kernel = None

        # Initialize base class members
        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Properties
    ###

    @property
    def L(self) -> wp.array:
        if self._L is None:
            raise ValueError("The factorization array has not been allocated!")
        return self._L

    @property
    def y(self) -> wp.array:
        if self._y is None:
            raise ValueError("The intermediate result array has not been allocated!")
        return self._y

    @property
    def P(self) -> wp.array:
        """Concatenated per-block ND permutation (int32[total_vec_size])."""
        if self._P is None:
            raise ValueError("Permutation array has not been allocated!")
        return self._P

    @property
    def inv_P(self) -> wp.array:
        """Concatenated per-block inverse ND permutation (int32[total_vec_size])."""
        if self._inv_P is None:
            raise ValueError("Inverse permutation array has not been allocated!")
        return self._inv_P

    @property
    def tile_pattern(self) -> wp.array:
        """Concatenated per-block tile-sparsity mask (int32, lower-tri inflated by fill-in)."""
        if self._tile_pattern is None:
            raise ValueError("Tile pattern array has not been allocated!")
        return self._tile_pattern

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        info = self._operator.info
        self._max_dim = int(info.max_dimension)

        # Resolve auxiliary kernels now that max_dim is known.
        self._permute_matrix_kernel = make_llt_blocked_nd_permute_matrix_kernel(self._max_dim)
        self._permute_vector_kernel = make_llt_blocked_nd_permute_vector_kernel(self._max_dim)
        self._build_inv_P_kernel = make_llt_blocked_nd_build_inv_P_kernel(self._max_dim)
        self._build_tile_pattern_kernel = make_llt_blocked_nd_build_tile_pattern_kernel(self._block_size, self._max_dim)
        max_n_tiles = (self._max_dim + self._block_size - 1) // self._block_size
        self._symbolic_fill_in_kernel = make_llt_blocked_nd_symbolic_fill_in_kernel(max_n_tiles)

        # Per-block tile-pattern layout: n_tiles_i^2 entries per block.
        # Computed on host once from info.dimensions (a cheap list).
        dims = list(info.dimensions)
        bs = self._block_size
        tp_sizes = [((d + bs - 1) // bs) ** 2 for d in dims]
        tp_offsets = [0]
        for s in tp_sizes:
            tp_offsets.append(tp_offsets[-1] + s)
        total_tp_size = tp_offsets[-1]

        with wp.ScopedDevice(self._device):
            # Factorization + intermediate buffers.
            self._L = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)

            # Reordering scratch.
            self._A_hat = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._b_hat = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)
            self._x_hat = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)

            # Permutations (indexed by vio, length dim per block).
            self._P = wp.zeros(shape=(info.total_vec_size,), dtype=int32)
            self._inv_P = wp.zeros(shape=(info.total_vec_size,), dtype=int32)

            # Tile-pattern flat storage + offsets.
            self._tile_pattern = wp.zeros(shape=(total_tp_size,), dtype=int32)
            self._tpo = wp.array(tp_offsets[:-1], dtype=int32)

        # Pre-build per-block ND recorded launches. Each ND launch receives
        # 2-D views into self._A_hat's per-block slice (we compute P on
        # A_hat? no - we compute P on the *input* A to get a fill-reducing
        # ordering for A; then we permute A -> A_hat using P and factorize
        # A_hat). So the ND views live over the user's A buffer which is
        # supplied at factorize time. We therefore (re)build these closures
        # lazily in _factorize_impl on first call, keyed off the A ptr
        # identity; if the user hands us a different A buffer we rebuild.
        self._nd_callbacks = []
        self._nd_scratches = []
        self._A_views = []
        self._A_views_attached_to = None

        self._has_factors = False

    @override
    def _reset_impl(self) -> None:
        self._L.zero_()
        self._y.zero_()
        self._A_hat.zero_()
        self._b_hat.zero_()
        self._x_hat.zero_()
        self._P.zero_()
        self._inv_P.zero_()
        self._tile_pattern.zero_()
        self._has_factors = False

    def _ensure_nd_launches_bound(self, A: wp.array) -> None:
        """Build the per-block ND recorded launches, bound to the current A buffer.

        Because each ND launch captures a ``wp.array2d`` *view* of A over a
        single block, the view (and the ND scratch it was created against)
        must be alive for as long as the launches are reused. We rebind only
        when A's underlying pointer changes (rare - typically finalize() /
        factorize() pass the same array).
        """
        if self._A_views_attached_to is A:
            return

        info = self._operator.info
        dims = list(info.dimensions)
        # mio, vio are on device; read once to host for per-block view setup.
        mio_host = info.mio.numpy()
        vio_host = info.vio.numpy()

        self._nd_callbacks = []
        self._nd_scratches = []
        self._A_views = []

        # Resolve Warp dtype for ND (matches A's dtype).
        nd_dtype = self._dtype

        # Fast path for 'rcm_batch': a single callback covers all blocks.
        if self._reorder_algorithm == "rcm_batch":
            with wp.ScopedDevice(self._device):
                cb = _rcm_batch.create_rcm_batch_launch(
                    A_flat=A,
                    perm_flat=self._P,
                    dims=info.dim,
                    mio=info.mio,
                    vio=info.vio,
                    num_blocks=info.num_blocks,
                    max_dim=int(info.max_dimension),
                    tol=self._nd_tol,
                    max_bfs_iters=self._rcm_max_bfs_iters,
                    use_cuda_graph=False,
                    device=self._device,
                )
            self._nd_callbacks = [cb]
            self._nd_scratches = [None]
            self._A_views = [(A, self._P)]  # keep references alive
            self._A_views_attached_to = A
            return

        # The ND module's get_array_ptr equivalent is only available through
        # creating a wp.array view with an explicit ptr. We obtain A's device
        # pointer via the Warp array's ``ptr`` attribute.
        A_base_ptr = A.ptr
        elem_size = wp.types.type_size_in_bytes(A.dtype)

        with wp.ScopedDevice(self._device):
            for b in range(info.num_blocks):
                n_i = int(dims[b])
                mat_off = int(mio_host[b])
                vec_off = int(vio_host[b])

                P_view = wp.array(
                    ptr=self._P.ptr + vec_off * wp.types.type_size_in_bytes(self._P.dtype),
                    shape=(n_i,),
                    dtype=int32,
                    device=self._device,
                )

                if self._reorder_algorithm == "spectral":
                    A_view = wp.array(
                        ptr=A_base_ptr + mat_off * elem_size,
                        shape=(n_i, n_i),
                        dtype=nd_dtype,
                        device=self._device,
                    )
                    scratch = _nd.allocate_nd_scratch(n_i, device=self._device, dtype=nd_dtype)
                    cb = _nd.create_nested_dissection_launch(
                        A_view,
                        P_view,
                        tol=self._nd_tol,
                        power_iters=self._nd_power_iters,
                        dtype=nd_dtype,
                        device=self._device,
                        use_cuda_graph=False,  # captured as part of the outer graph if any
                        scratch=scratch,
                    )
                    # 'scratch' holds the live ND buffers; keep a ref.
                    self._A_views.append((A_view, P_view))
                    self._nd_scratches.append(scratch)
                    self._nd_callbacks.append(cb)
                else:  # 'rcm'
                    # RCM wants a flat (n*n,) array view.
                    A_view = wp.array(
                        ptr=A_base_ptr + mat_off * elem_size,
                        shape=(n_i * n_i,),
                        dtype=nd_dtype,
                        device=self._device,
                    )
                    cb = _rcm.create_rcm_launch(
                        A_view,
                        P_view,
                        tol=self._nd_tol,
                        max_bfs_iters=self._rcm_max_bfs_iters,
                        use_cuda_graph=False,  # captured under outer graph
                        device=self._device,
                    )
                    # RCM allocates its own scratch inside create_rcm_launch;
                    # the closure keeps it alive via the captured launches.
                    self._A_views.append((A_view, P_view))
                    self._nd_scratches.append(None)
                    self._nd_callbacks.append(cb)

        self._A_views_attached_to = A

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Bind / rebind ND views to the current A buffer.
        self._ensure_nd_launches_bound(A)

        # 1. Compute per-block P by running each ND callback. These are
        #    recorded Warp launches internally and are safe to replay under
        #    CUDA graph capture initiated by the caller.
        for cb in self._nd_callbacks:
            cb()

        # 2. Build inv_P.
        llt_blocked_nd_build_inv_P(
            kernel=self._build_inv_P_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._P,
            inv_P=self._inv_P,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # 3. Permute A -> A_hat (symmetric).
        llt_blocked_nd_permute_matrix(
            kernel=self._permute_matrix_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            P=self._P,
            A=A,
            A_hat=self._A_hat,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # 4. Build tile-pattern from |A_hat| > nd_tol and inflate by symbolic fill-in.
        self._tile_pattern.zero_()
        llt_blocked_nd_build_tile_pattern(
            kernel=self._build_tile_pattern_kernel,
            dim=info.dim,
            mio=info.mio,
            tpo=self._tpo,
            tol=self._nd_tol,
            A_hat=self._A_hat,
            tile_pattern=self._tile_pattern,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )
        llt_blocked_nd_symbolic_fill_in(
            kernel=self._symbolic_fill_in_kernel,
            dim=info.dim,
            tpo=self._tpo,
            block_size=self._block_size,
            tile_pattern=self._tile_pattern,
            num_blocks=num_blocks,
            device=self._device,
        )

        # 5. Numeric factorization with tile-pattern skips.
        llt_blocked_nd_factorize(
            kernel=self._factorize_kernel,
            dim=info.dim,
            mio=info.mio,
            tpo=self._tpo,
            A=self._A_hat,
            tile_pattern=self._tile_pattern,
            L=self._L,
            num_blocks=num_blocks,
            block_dim=self._factortize_block_dim,
            device=self._device,
        )

    @override
    def _reconstruct_impl(self, A: wp.array) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array, x: wp.array) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Permute b -> b_hat.
        llt_blocked_nd_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._P,
            src=b,
            dst=self._b_hat,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # Solve L L^T x_hat = b_hat.
        llt_blocked_nd_solve(
            kernel=self._solve_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            L=self._L,
            tile_pattern=self._tile_pattern,
            b=self._b_hat,
            y=self._y,
            x=self._x_hat,
            num_blocks=num_blocks,
            block_dim=self._solve_block_dim,
            device=self._device,
        )

        # Un-permute: x[P[r]] = x_hat[r]  =>  x[r] = x_hat[inv_P[r]].
        # Our permute_vector kernel computes ``dst[r] = src[P[r]]``, so using
        # inv_P as P and (x_hat, x) as (src, dst) yields the desired inverse
        # permutation on x.
        llt_blocked_nd_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._inv_P,
            src=self._x_hat,
            dst=x,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Permute x -> x_hat (x is the RHS here).
        llt_blocked_nd_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._P,
            src=x,
            dst=self._x_hat,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # In-place solve on x_hat (x_hat starts as the permuted RHS; y is scratch).
        llt_blocked_nd_solve_inplace(
            kernel=self._solve_inplace_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            L=self._L,
            tile_pattern=self._tile_pattern,
            y=self._y,
            x=self._x_hat,
            num_blocks=num_blocks,
            block_dim=self._solve_block_dim,
            device=self._device,
        )

        # Un-permute x_hat -> x.
        llt_blocked_nd_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._inv_P,
            src=self._x_hat,
            dst=x,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )
