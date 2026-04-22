# Vendored from socu/nested_dissection.py. Do not edit without syncing.
#
# Upstream: C:\\git3\\socu\\socu\\nested_dissection.py
# The only local change vs. upstream is inlining socu.utils.create_cuda_graph_callback
# (see bottom of this file) so the newton tree has zero runtime dependency on the
# external ``socu`` package.

"""
Generalized nested dissection on the GPU for dense SPD matrices.

This module produces a fill-reducing permutation of a dense SPD matrix whose
sparsity pattern is discovered internally by thresholding the off-diagonal
entries. The partitioning method is fixed-topology spectral bisection (power
iteration on the graph Laplacian of each subgraph), executed end-to-end on the
GPU via NVIDIA Warp and fully CUDA-graph-capturable.

The design mirrors the ``create_*_launch`` callback pattern in
``socu.block_tridiag_solver``: an expensive creator function allocates all
scratch, constructs recorded kernel launches with fixed launch dimensions, and
returns a zero-argument callback that writes the permutation in-place.

Example
-------

.. code-block:: python

    import warp as wp
    from socu.nested_dissection import create_nested_dissection_launch

    A = wp.from_numpy(A_np, dtype=wp.float64, device='cuda')
    perm = wp.empty(n, dtype=wp.int32, device='cuda')

    nd_launch = create_nested_dissection_launch(
        A, perm, tol=0.0, use_cuda_graph=True,  # power_iters defaults to 'auto'
    )
    nd_launch()  # fills perm on GPU; fully graph-captured

Notes
-----
Quality is below METIS; this is the price paid for CUDA-graph compatibility.
Per-call cost is dominated by ``D * K`` segmented dense matvecs, where
``D = ceil(log2(n))`` and ``K`` is ``power_iters``.
"""

import ctypes
from functools import lru_cache
from typing import Callable

import warp as wp


def create_cuda_graph_callback(callback, device=None, stream=None):
    """Inlined from ``socu.utils.create_cuda_graph_callback`` (unchanged).

    Captures ``callback``'s Warp launches into a CUDA graph and returns a
    zero-argument function that replays the captured graph.
    """
    with wp.ScopedCapture(device=device, stream=stream) as capture:
        callback()

    graph = capture.graph

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")
        device = stream.device
    else:
        device = graph.device
        stream = device.stream

    # populate graph executable
    if graph.graph_exec is None:
        g = ctypes.c_void_p()
        result = wp._src.context.runtime.core.wp_cuda_graph_create_exec(
            graph.device.context, stream.cuda_stream, graph.graph, ctypes.byref(g)
        )
        if not result:
            raise RuntimeError(f"Graph creation error: {wp.context.runtime.get_error_string()}")
        graph.graph_exec = g

    def graph_callback():
        wp.capture_launch(graph)

    return graph_callback


def calculate_nd_iterations(n: int) -> int:
    """Number of nested-dissection recursion levels.

    Equivalent to ``ceil(log2(n))`` for ``n >= 1``. Mirrors the
    ``calculate_recursive_iterations`` helper used elsewhere in socu.
    """
    assert n >= 1
    if n == 1:
        return 0
    return (n - 1).bit_length()


def allocate_nd_scratch(n: int, device, dtype=wp.float64) -> dict:
    """Preallocate all device-side scratch used by the nested-dissection launch.

    Returned dict keys:

    - ``labels``        : int32[n]   current subgraph id; -1 means retired.
    - ``side``          : int32[n]   per-level +1 / -1 / 0 classification.
    - ``fiedler``       : dtype[n]   current Fiedler approximation.
    - ``y``             : dtype[n]   scratch for Laplacian matvec.
    - ``seg_sum``       : dtype[S]   segmented reduction scratch (dtype).
    - ``seg_count``     : int32[S]   segmented reduction scratch (count).
    - ``seg_norm2``     : dtype[S]   segmented L2 norm squared.
    - ``head_sep``      : int32[1]   back-to-front write cursor in perm.
    - ``head_leaf``     : int32[1]   front-to-back write cursor in perm.
    - ``level_offsets`` : int32[D+2] recorded separator block boundaries.
    """
    D = calculate_nd_iterations(n)
    S_max = 1 << max(D, 1)  # upper bound on number of subgraphs at any level

    return {
        'labels':        wp.empty(n, dtype=wp.int32, device=device),
        'side':          wp.empty(n, dtype=wp.int32, device=device),
        'fiedler':       wp.empty(n, dtype=dtype,    device=device),
        'y':             wp.empty(n, dtype=dtype,    device=device),
        'seg_sum':       wp.empty(S_max, dtype=dtype, device=device),
        'seg_count':     wp.empty(S_max, dtype=wp.int32, device=device),
        'seg_norm2':     wp.empty(S_max, dtype=dtype, device=device),
        'head_sep':      wp.empty(1, dtype=wp.int32, device=device),
        'head_leaf':     wp.empty(1, dtype=wp.int32, device=device),
        'level_offsets': wp.empty(D + 2, dtype=wp.int32, device=device),
    }


# ---------------------------------------------------------------------------
# Kernels
#
# Every kernel is per-vertex (launch dim == n) or per-subgraph (launch dim ==
# S_max), so launch shapes depend only on quantities known at creator time.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_kernels(dtype):
    """Build and cache the Warp kernels for a given floating-point dtype.

    All kernels live in a fresh module with backward and unrolling disabled,
    matching the convention used across ``socu.block_tridiag_solver``.
    """

    module = wp.Module(f'nested_dissection_kernels_{dtype.__name__}', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def init_state_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        head_sep: wp.array(dtype=wp.int32),        # type: ignore
        head_leaf: wp.array(dtype=wp.int32),       # type: ignore
        level_offsets: wp.array(dtype=wp.int32),   # type: ignore
        perm: wp.array(dtype=wp.int32),            # type: ignore
        D_plus_2: int,
    ):
        i = wp.tid()
        # labels[i] = 0 (single root subgraph)
        if i < n:
            labels[i] = int(0)
            perm[i] = int(-1)
        # one thread resets cursors and level_offsets
        if i == 0:
            head_sep[0] = n                # grows downward via atomic_sub
            head_leaf[0] = int(0)          # grows upward via atomic_add
        if i < D_plus_2:
            level_offsets[i] = n

    @wp.kernel(module=module)
    def fiedler_init_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        if labels[i] < 0:
            fiedler[i] = dtype(0.0)
            return
        # Deterministic, reproducible initial vector that is NOT in the
        # constant nullspace. Using a fractional of a prime-like step avoids
        # pathological zeros.
        x = dtype(float(((i * int(2654435761)) & int(2147483647)))) / dtype(2147483647.0)
        fiedler[i] = x - dtype(0.5)

    @wp.kernel(module=module)
    def seg_zero_dtype_kernel(
        S_max: int,
        seg: wp.array(dtype=dtype),                # type: ignore
    ):
        s = wp.tid()
        if s < S_max:
            seg[s] = dtype(0.0)

    @wp.kernel(module=module)
    def seg_zero_int_kernel(
        S_max: int,
        seg: wp.array(dtype=wp.int32),             # type: ignore
    ):
        s = wp.tid()
        if s < S_max:
            seg[s] = int(0)

    @wp.kernel(module=module)
    def seg_accumulate_mean_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
        seg_sum: wp.array(dtype=dtype),            # type: ignore
        seg_count: wp.array(dtype=wp.int32),       # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        lbl = labels[i]
        if lbl < 0:
            return
        wp.atomic_add(seg_sum, lbl, fiedler[i])
        wp.atomic_add(seg_count, lbl, int(1))

    @wp.kernel(module=module)
    def seg_apply_mean_subtract_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
        seg_sum: wp.array(dtype=dtype),            # type: ignore
        seg_count: wp.array(dtype=wp.int32),       # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        lbl = labels[i]
        if lbl < 0:
            return
        c = seg_count[lbl]
        if c > int(0):
            fiedler[i] = fiedler[i] - seg_sum[lbl] / dtype(float(c))

    @wp.kernel(module=module)
    def laplacian_matvec_kernel(
        n: int,
        tol: dtype,
        A: wp.array2d(dtype=dtype),                # type: ignore
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
        y: wp.array(dtype=dtype),                  # type: ignore
    ):
        """y = L_sub * fiedler where L_sub is the graph Laplacian restricted
        to edges whose endpoints share the current subgraph id."""
        i = wp.tid()
        if i >= n:
            return
        lbl_i = labels[i]
        if lbl_i < 0:
            y[i] = dtype(0.0)
            return
        # Compute D_i * f_i - sum_{j in N(i) cap same subgraph} f_j.
        deg = dtype(0.0)
        acc = dtype(0.0)
        f_i = fiedler[i]
        for j in range(n):
            if j == i:
                continue
            if labels[j] != lbl_i:
                continue
            a = A[i, j]
            # Treat any |A_ij| > tol as an edge. Using symmetric mask so that
            # matvec stays symmetric numerically regardless of small asymmetry.
            aij = wp.abs(a)
            if aij > tol:
                deg = deg + dtype(1.0)
                acc = acc + fiedler[j]
        y[i] = deg * f_i - acc

    @wp.kernel(module=module)
    def seg_accumulate_norm2_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
        seg_norm2: wp.array(dtype=dtype),          # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        lbl = labels[i]
        if lbl < 0:
            return
        v = fiedler[i]
        wp.atomic_add(seg_norm2, lbl, v * v)

    @wp.kernel(module=module)
    def seg_apply_normalize_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
        seg_norm2: wp.array(dtype=dtype),          # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        lbl = labels[i]
        if lbl < 0:
            return
        s2 = seg_norm2[lbl]
        if s2 > dtype(1e-30):
            fiedler[i] = fiedler[i] / wp.sqrt(s2)
        else:
            fiedler[i] = dtype(0.0)

    @wp.kernel(module=module)
    def copy_y_to_fiedler_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        y: wp.array(dtype=dtype),                  # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        if labels[i] < 0:
            return
        fiedler[i] = y[i]

    @wp.kernel(module=module)
    def classify_kernel(
        n: int,
        fiedler_eps: dtype,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        fiedler: wp.array(dtype=dtype),            # type: ignore
        side: wp.array(dtype=wp.int32),            # type: ignore
    ):
        """Assign +1 / -1 to every active vertex. Uses Fiedler sign with a
        deterministic fallback based on (subgraph-local) vertex index when the
        Fiedler value is below ``fiedler_eps`` - this keeps launch shapes fixed
        even on degenerate (disconnected / near-zero-gap) subgraphs."""
        i = wp.tid()
        if i >= n:
            return
        if labels[i] < 0:
            side[i] = int(0)
            return
        v = fiedler[i]
        if v > fiedler_eps:
            side[i] = int(1)
        elif v < -fiedler_eps:
            side[i] = int(-1)
        else:
            # Deterministic tie-break: even index -> +, odd index -> -. This
            # cannot starve either side because it partitions by parity.
            if (i & int(1)) == int(0):
                side[i] = int(1)
            else:
                side[i] = int(-1)

    @wp.kernel(module=module)
    def emit_separators_kernel(
        n: int,
        tol: dtype,
        A: wp.array2d(dtype=dtype),                # type: ignore
        labels: wp.array(dtype=wp.int32),          # type: ignore
        side: wp.array(dtype=wp.int32),            # type: ignore
        head_sep: wp.array(dtype=wp.int32),        # type: ignore
        perm: wp.array(dtype=wp.int32),            # type: ignore
    ):
        """A vertex on the +1 side that has any neighbor in the same subgraph
        on the -1 side becomes a separator vertex. Emission uses an atomic
        decrement on the back-cursor so separators land at the *end* of perm,
        higher levels to the right of lower levels."""
        i = wp.tid()
        if i >= n:
            return
        if labels[i] < 0:
            return
        if side[i] != int(1):
            return
        lbl_i = labels[i]
        is_sep = int(0)
        for j in range(n):
            if j == i:
                continue
            if labels[j] != lbl_i:
                continue
            if side[j] != int(-1):
                continue
            aij = wp.abs(A[i, j])
            if aij > tol:
                is_sep = int(1)
        if is_sep == int(1):
            slot = wp.atomic_sub(head_sep, 0, int(1)) - int(1)
            perm[slot] = i
            labels[i] = int(-1)

    @wp.kernel(module=module)
    def record_level_offset_kernel(
        level: int,
        head_sep: wp.array(dtype=wp.int32),        # type: ignore
        level_offsets: wp.array(dtype=wp.int32),   # type: ignore
    ):
        # Single-thread launch (dim=1). Captures head_sep into level_offsets[level].
        if wp.tid() == 0:
            level_offsets[level] = head_sep[0]

    @wp.kernel(module=module)
    def relabel_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        side: wp.array(dtype=wp.int32),            # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        lbl = labels[i]
        if lbl < 0:
            return
        s = side[i]
        bit = int(0)
        if s > int(0):
            bit = int(1)
        labels[i] = int(2) * lbl + bit

    @wp.kernel(module=module)
    def emit_leaves_kernel(
        n: int,
        labels: wp.array(dtype=wp.int32),          # type: ignore
        head_leaf: wp.array(dtype=wp.int32),       # type: ignore
        perm: wp.array(dtype=wp.int32),            # type: ignore
    ):
        i = wp.tid()
        if i >= n:
            return
        if labels[i] < 0:
            return
        slot = wp.atomic_add(head_leaf, 0, int(1))
        perm[slot] = i
        labels[i] = int(-1)

    return {
        'init_state':               init_state_kernel,
        'fiedler_init':             fiedler_init_kernel,
        'seg_zero_dtype':           seg_zero_dtype_kernel,
        'seg_zero_int':             seg_zero_int_kernel,
        'seg_accumulate_mean':      seg_accumulate_mean_kernel,
        'seg_apply_mean_subtract':  seg_apply_mean_subtract_kernel,
        'laplacian_matvec':         laplacian_matvec_kernel,
        'seg_accumulate_norm2':     seg_accumulate_norm2_kernel,
        'seg_apply_normalize':      seg_apply_normalize_kernel,
        'copy_y_to_fiedler':        copy_y_to_fiedler_kernel,
        'classify':                 classify_kernel,
        'emit_separators':          emit_separators_kernel,
        'record_level_offset':      record_level_offset_kernel,
        'relabel':                  relabel_kernel,
        'emit_leaves':              emit_leaves_kernel,
    }


# ---------------------------------------------------------------------------
# Public launch creator
# ---------------------------------------------------------------------------

def create_nested_dissection_launch(
    A: wp.array,
    perm: wp.array,
    *,
    tol: float = 0.0,
    power_iters='auto',
    fiedler_eps: float = 1e-9,
    dtype=wp.float64,
    device=None,
    stream=None,
    use_cuda_graph: bool = False,
    scratch: dict = None,
) -> Callable[[], None]:
    """Create a zero-argument callback that fills ``perm`` with a fill-reducing
    nested-dissection permutation of the dense SPD matrix ``A``.

    Parameters
    ----------
    A : wp.array2d
        Dense ``n x n`` SPD matrix on the target device. Only the sparsity
        pattern induced by ``|A_ij| > tol`` is used; magnitudes otherwise
        ignored by the partitioner.
    perm : wp.array
        Preallocated ``int32[n]`` device array. The callback writes the
        permutation into this array in-place.
    tol : float
        Threshold for deciding which off-diagonal entries represent graph
        edges. Use ``0.0`` for strictly structural sparsity, or a small
        positive value to ignore numerical noise.
    power_iters : int | 'auto'
        Fixed number of power-iteration steps per level. Larger values yield a
        better Fiedler approximation at the cost of per-call time. Pass
        ``'auto'`` (the default) to size K adaptively as
        ``max(25, ceil(2.5 * log2(n)))``. The schedule was chosen under the
        benchmark-gated acceptance criterion in ``benchmarks/`` to reduce
        typical replay time by ~14% vs the legacy fixed K=30 while keeping
        fill_ratio regression within the 10% per-case budget across the
        bench suite (2D grids, 3D grids, random sparse, banded, disconnected).
    fiedler_eps : float
        Magnitude below which the Fiedler value is considered numerically zero
        and a deterministic parity tie-break is used instead. Keeps every
        subgraph non-empty on both sides.
    dtype :
        Warp floating-point dtype of ``A`` (``wp.float32`` or ``wp.float64``).
    device :
        Warp device; defaults to the device of ``A``.
    stream :
        Warp stream for launches; defaults to the device's default stream.
    use_cuda_graph : bool
        If ``True``, the returned callback replays a captured CUDA graph,
        eliminating per-launch overhead. All kernel launches below are
        capture-safe.
    scratch : dict, optional
        Preallocated scratch (from :func:`allocate_nd_scratch`). If ``None``,
        fresh scratch is allocated internally.
    """

    if A.ndim != 2:
        raise ValueError(f"A must be 2-dimensional, got shape {A.shape}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    n = A.shape[0]

    if perm.ndim != 1 or perm.shape[0] != n:
        raise ValueError(
            f"perm must be a 1-D int32 array of length {n}, got shape {perm.shape}"
        )
    if perm.dtype != wp.int32:
        raise ValueError(f"perm must have dtype wp.int32, got {perm.dtype}")

    if device is None:
        device = A.device

    D = calculate_nd_iterations(n)
    S_max = 1 << max(D, 1)

    # Resolve adaptive power_iters. 2.5 * log2(n) with a floor of 25 gives
    # K=25 for n in [~500, ~1400] - a 17% reduction from the legacy default
    # K=30 at our target scale. More aggressive reductions (K<=20) were
    # benchmarked and rejected: they regressed fill on grid3d_8x8x8 by
    # 13-22% because tight 3D-grid clusters need more power iterations for
    # the Fiedler vector to separate. See ``benchmarks/README.md``.
    if isinstance(power_iters, str):
        if power_iters != 'auto':
            raise ValueError(
                f"power_iters must be an int or 'auto', got {power_iters!r}"
            )
        import math as _math
        power_iters = max(25, int(_math.ceil(2.5 * _math.log2(max(n, 2)))))
    elif not isinstance(power_iters, int):
        raise TypeError(
            f"power_iters must be int or 'auto', got {type(power_iters)}"
        )

    if scratch is None:
        scratch = allocate_nd_scratch(n, device=device, dtype=dtype)

    labels        = scratch['labels']
    side          = scratch['side']
    fiedler       = scratch['fiedler']
    y             = scratch['y']
    seg_sum       = scratch['seg_sum']
    seg_count     = scratch['seg_count']
    seg_norm2     = scratch['seg_norm2']
    head_sep      = scratch['head_sep']
    head_leaf     = scratch['head_leaf']
    level_offsets = scratch['level_offsets']

    K = _get_kernels(dtype)

    # Warp unpacks Python floats into the scalar's native type, matching the
    # ``dtype`` annotation on each kernel argument.
    tol_scalar = float(tol)
    fiedler_eps_scalar = float(fiedler_eps)

    # ------------------------------------------------------------------
    # Record every launch once. Inputs are bound up-front; only kernels
    # that need per-level parameters are re-parameterized via set_params.
    # All launch dims are fixed at creator time, which is what CUDA graph
    # capture requires.
    # ------------------------------------------------------------------

    init_launch = wp.launch(
        K['init_state'],
        dim=max(n, D + 2),
        inputs=[n, labels, head_sep, head_leaf, level_offsets, perm, D + 2],
        device=device, stream=stream, record_cmd=True,
    )

    fiedler_init_launch = wp.launch(
        K['fiedler_init'],
        dim=n,
        inputs=[n, labels, fiedler],
        device=device, stream=stream, record_cmd=True,
    )

    seg_zero_sum_launch = wp.launch(
        K['seg_zero_dtype'],
        dim=S_max,
        inputs=[S_max, seg_sum],
        device=device, stream=stream, record_cmd=True,
    )

    seg_zero_count_launch = wp.launch(
        K['seg_zero_int'],
        dim=S_max,
        inputs=[S_max, seg_count],
        device=device, stream=stream, record_cmd=True,
    )

    seg_zero_norm2_launch = wp.launch(
        K['seg_zero_dtype'],
        dim=S_max,
        inputs=[S_max, seg_norm2],
        device=device, stream=stream, record_cmd=True,
    )

    seg_acc_mean_launch = wp.launch(
        K['seg_accumulate_mean'],
        dim=n,
        inputs=[n, labels, fiedler, seg_sum, seg_count],
        device=device, stream=stream, record_cmd=True,
    )

    seg_apply_mean_launch = wp.launch(
        K['seg_apply_mean_subtract'],
        dim=n,
        inputs=[n, labels, fiedler, seg_sum, seg_count],
        device=device, stream=stream, record_cmd=True,
    )

    matvec_launch = wp.launch(
        K['laplacian_matvec'],
        dim=n,
        inputs=[n, tol_scalar, A, labels, fiedler, y],
        device=device, stream=stream, record_cmd=True,
    )

    seg_acc_norm2_launch = wp.launch(
        K['seg_accumulate_norm2'],
        dim=n,
        inputs=[n, labels, y, seg_norm2],
        device=device, stream=stream, record_cmd=True,
    )

    seg_apply_normalize_launch = wp.launch(
        K['seg_apply_normalize'],
        dim=n,
        inputs=[n, labels, y, seg_norm2],
        device=device, stream=stream, record_cmd=True,
    )

    copy_y_to_fiedler_launch = wp.launch(
        K['copy_y_to_fiedler'],
        dim=n,
        inputs=[n, labels, y, fiedler],
        device=device, stream=stream, record_cmd=True,
    )

    classify_launch = wp.launch(
        K['classify'],
        dim=n,
        inputs=[n, fiedler_eps_scalar, labels, fiedler, side],
        device=device, stream=stream, record_cmd=True,
    )

    emit_sep_launch = wp.launch(
        K['emit_separators'],
        dim=n,
        inputs=[n, tol_scalar, A, labels, side, head_sep, perm],
        device=device, stream=stream, record_cmd=True,
    )

    # One recorded launch per level since it captures ``level`` as a parameter.
    record_offset_launches = []
    for l in range(D + 1):
        rec = wp.launch(
            K['record_level_offset'],
            dim=1,
            inputs=[l, head_sep, level_offsets],
            device=device, stream=stream, record_cmd=True,
        )
        record_offset_launches.append(rec)

    relabel_launch = wp.launch(
        K['relabel'],
        dim=n,
        inputs=[n, labels, side],
        device=device, stream=stream, record_cmd=True,
    )

    emit_leaves_launch = wp.launch(
        K['emit_leaves'],
        dim=n,
        inputs=[n, labels, head_leaf, perm],
        device=device, stream=stream, record_cmd=True,
    )

    def callback():
        # Reset state: labels all zero, perm cleared, cursors reset.
        init_launch.launch()

        # Level 0..D-1: bisect every current subgraph.
        for _level in range(D):
            # Initialize Fiedler guess (deterministic, non-constant).
            fiedler_init_launch.launch()

            for _k in range(power_iters):
                # Project fiedler onto orthogonal complement of per-subgraph
                # constant vector (mean subtraction).
                seg_zero_sum_launch.launch()
                seg_zero_count_launch.launch()
                seg_acc_mean_launch.launch()
                seg_apply_mean_launch.launch()

                # y = L_sub * fiedler
                matvec_launch.launch()

                # Normalize y per subgraph.
                seg_zero_norm2_launch.launch()
                seg_acc_norm2_launch.launch()
                seg_apply_normalize_launch.launch()

                # fiedler <- y
                copy_y_to_fiedler_launch.launch()

            # Classify +/- sides and emit the vertex separator to perm.
            classify_launch.launch()
            emit_sep_launch.launch()

            # Record current head_sep as the end of this level's separator
            # block (between separators of adjacent levels).
            record_offset_launches[_level].launch()

            # Relabel surviving vertices into their child subgraph.
            relabel_launch.launch()

        # Final level_offsets entry marks the boundary between separator
        # region (right) and leaf region (left) of perm.
        record_offset_launches[D].launch()

        # Dump everything that survived all bisections into the front of perm.
        emit_leaves_launch.launch()

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device=device, stream=stream)

    return callback
