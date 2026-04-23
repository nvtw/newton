# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU Reverse Cuthill-McKee (RCM) reordering for dense SPD matrices.

Produces a bandwidth-reducing permutation that, for sparse SPD matrices,
tends to produce very good tile-granularity skip patterns for
:class:`LLTBlockedNDSolver`.

Design goals
------------

- Pure GPU, end-to-end. Single zero-argument callback, CUDA-graph-capturable.
- Fixed topology: all launch dimensions are known at creation time. The BFS
  depth upper bound is ``n``, so we launch exactly ``n`` BFS-step kernels.
  After the BFS has finished, subsequent BFS-step kernels are cheap no-ops.
- Input is a dense ``(n, n)`` SPD matrix. Adjacency is discovered implicitly
  by thresholding ``|A_ij| > tol``.
- Output is a permutation ``perm`` of ``[0..n)`` such that the bandwidth of
  ``A[perm][:, perm]`` is typically much smaller than the bandwidth of ``A``.

API
---

.. code-block:: python

    import warp as wp
    from newton._src.solvers.kamino._src.linalg.factorize.rcm import create_rcm_launch

    A = wp.from_numpy(A_np, dtype=wp.float32, device='cuda')
    perm = wp.empty(n, dtype=wp.int32, device='cuda')

    rcm = create_rcm_launch(A, perm, tol=0.0, use_cuda_graph=True)
    rcm()  # fills perm on GPU; fully graph-captured

Algorithm
---------

1. Compute per-vertex degree ``deg[i] = |{j != i : |A_ij| > tol}|``.
2. Pick the root as ``argmin(deg)`` (simple pseudo-peripheral heuristic).
3. Plain level-set BFS from the root. Within a level we do NOT sort by
   degree (classical RCM does; this is the "CM" variant). Skipping the
   intra-level sort loses little quality on typical sparse SPD matrices
   and avoids an expensive per-level sort kernel.
4. Write the BFS order into a scratch buffer, then reverse it into ``perm``.

For disconnected graphs: any vertex that has not been visited after the
BFS saturates is appended to ``perm`` in original index order (before the
final reversal). This keeps ``perm`` a valid permutation of ``[0..n)``.

Per-call kernel launches
------------------------

- 1 kernel     : degree + root selection (fused)
- 1 kernel     : initial frontier (set root as visited, level 0)
- ``n`` kernels: BFS neighbor-expansion step. Most of these are no-ops
  after BFS saturates; we keep them for fixed-topology graph capture.
- 1 kernel     : append unreached vertices in-order
- 1 kernel     : reverse the order into ``perm``

Total: ``n + 4`` launches. For ``n = 256`` that's 260 launches, vs the
spectral scheme's ~1650 at ``power_iters=25`` or ~320 at ``power_iters=5``.
"""

import ctypes
from functools import lru_cache
from typing import Callable

import warp as wp


def create_cuda_graph_callback(callback: Callable[[], None], device=None, stream=None) -> Callable[[], None]:
    """Capture ``callback`` into a CUDA graph and return a zero-arg replay fn."""
    with wp.ScopedCapture(device=device, stream=stream) as capture:
        callback()

    graph = capture.graph

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(
                f"Cannot launch graph from device {graph.device} on stream from device {stream.device}"
            )
        device = stream.device
    else:
        device = graph.device
        stream = device.stream

    if graph.graph_exec is None:
        g = ctypes.c_void_p()
        result = wp._src.context.runtime.core.wp_cuda_graph_create_exec(
            graph.device.context, stream.cuda_stream, graph.graph, ctypes.byref(g)
        )
        if not result:
            raise RuntimeError(f"Graph creation error: {wp._src.context.runtime.get_error_string()}")
        graph.graph_exec = g

    def graph_callback():
        wp.capture_launch(graph)

    return graph_callback


# ---------------------------------------------------------------------------
# Scratch allocation
# ---------------------------------------------------------------------------


def allocate_rcm_scratch(n: int, device) -> dict:
    """Preallocate all device-side scratch used by the RCM launch.

    Keys:

    - ``degree``       : int32[n]   per-vertex off-diagonal degree.
    - ``level``        : int32[n]   BFS level (``-1`` == not yet visited).
    - ``order_buf``    : int32[n]   BFS order (pre-reverse).
    - ``head``         : int32[1]   atomic write cursor into ``order_buf``.
    - ``active``       : int32[1]   per-step flag; non-zero => BFS still alive.
    - ``root``         : int32[1]   chosen root vertex.
    - ``current_level``: int32[1]   BFS step counter.
    """
    return {
        "degree":        wp.empty(n, dtype=wp.int32, device=device),
        "level":         wp.empty(n, dtype=wp.int32, device=device),
        "order_buf":     wp.empty(n, dtype=wp.int32, device=device),
        "head":          wp.empty(1, dtype=wp.int32, device=device),
        # BFS liveness flag: 1 => keep expanding, 0 => saturated.
        "alive":         wp.empty(1, dtype=wp.int32, device=device),
        # Per-step discovery flag: set to 1 inside bfs_step_kernel whenever a
        # new vertex is visited. Reset to 0 by pre_step_kernel each iteration.
        "discovered":    wp.empty(1, dtype=wp.int32, device=device),
        "root":          wp.empty(1, dtype=wp.int32, device=device),
        "current_level": wp.empty(1, dtype=wp.int32, device=device),
    }


# ---------------------------------------------------------------------------
# Kernel factory (one per (dtype, n) combination)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _make_rcm_kernels(dtype, n: int):
    """Build and cache all RCM kernels for a given ``(dtype, n)``."""
    module_name = f"rcm_kernels_{getattr(dtype, '__name__', str(dtype))}_n{n}"
    module = wp.get_module(module_name)

    @wp.kernel(module=module)
    def init_and_degree_kernel(
        n_in: int,
        tol: dtype,  # type: ignore[valid-type]
        A: wp.array(dtype=dtype),                   # type: ignore[valid-type]
        degree: wp.array(dtype=wp.int32),           # type: ignore[valid-type]
        level: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        head: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
        alive: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        discovered: wp.array(dtype=wp.int32),       # type: ignore[valid-type]
        root: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
        current_level: wp.array(dtype=wp.int32),    # type: ignore[valid-type]
    ):
        i = wp.tid()
        if i >= n_in:
            return

        # Per-vertex init.
        level[i] = int(-1)

        # Compute degree by scanning the row. Abs-threshold `tol` matches the
        # convention used elsewhere (build_tile_pattern).
        d = int(0)
        base = i * n_in
        for j in range(n_in):
            if j == i:
                continue
            av = wp.abs(A[base + j])
            if av > tol:
                d += int(1)
        degree[i] = d

        # One thread resets the global cursors/flags. Note this happens
        # before ``init_frontier`` which pushes the root into order_buf[0]
        # (hence ``head = 0``) and before the first BFS step which reads
        # ``alive = 1`` and may set ``discovered = 1``.
        if i == 0:
            head[0] = int(0)
            alive[0] = int(1)
            discovered[0] = int(0)
            root[0] = int(0)
            current_level[0] = int(0)

    @wp.kernel(module=module)
    def select_root_kernel(
        n_in: int,
        degree: wp.array(dtype=wp.int32),           # type: ignore[valid-type]
        root: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
    ):
        """Pick a pseudo-peripheral root: lowest-degree vertex (ties broken by lowest index).

        Implemented as a serialized scan in one thread (n <= ~1000 in our use).
        This removes the need for a two-phase parallel reduction kernel.
        """
        tid = wp.tid()
        if tid != 0:
            return
        best_deg = int(2147483647)
        best_idx = int(0)
        for i in range(n_in):
            d = degree[i]
            if d < best_deg:
                best_deg = d
                best_idx = i
        root[0] = best_idx

    @wp.kernel(module=module)
    def init_frontier_kernel(
        root: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
        level: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        order_buf: wp.array(dtype=wp.int32),        # type: ignore[valid-type]
        head: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
    ):
        """Seed the BFS: level[root] = 0, emit root to order_buf[0]."""
        tid = wp.tid()
        if tid != 0:
            return
        r = root[0]
        level[r] = int(0)
        # Atomically claim the next slot (0) in order_buf and write the root.
        slot = wp.atomic_add(head, 0, int(1))
        order_buf[slot] = r

    @wp.kernel(module=module)
    def bfs_step_kernel(
        n_in: int,
        tol: dtype,                                 # type: ignore[valid-type]
        A: wp.array(dtype=dtype),                   # type: ignore[valid-type]
        level: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        order_buf: wp.array(dtype=wp.int32),        # type: ignore[valid-type]
        head: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
        alive: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        discovered: wp.array(dtype=wp.int32),       # type: ignore[valid-type]
        current_level: wp.array(dtype=wp.int32),    # type: ignore[valid-type]
    ):
        """Expand all frontier vertices at ``current_level[0]``.

        Semantics:
        - If ``alive[0] == 0`` this kernel is a no-op (BFS saturated). This
          is what makes the fixed ``n``-step iteration safe for graph capture.
        - For each frontier vertex ``i`` (``level[i] == current_level[0]``),
          atomically claim each unvisited neighbor ``j`` via atomic-CAS on
          ``level[j]`` and emit it to ``order_buf`` at an atomic-add slot.
        - On ANY discovery, set ``discovered[0] = 1`` via ``atomic_max``.
        """
        i = wp.tid()
        if i >= n_in:
            return
        if alive[0] == int(0):
            return

        cur = current_level[0]
        if level[i] != cur:
            return

        base = i * n_in
        next_lvl = cur + int(1)
        for j in range(n_in):
            if j == i:
                continue
            av = wp.abs(A[base + j])
            if av > tol:
                if level[j] == int(-1):
                    old = wp.atomic_cas(level, j, int(-1), next_lvl)
                    if old == int(-1):
                        wp.atomic_max(discovered, 0, int(1))
                        slot = wp.atomic_add(head, 0, int(1))
                        order_buf[slot] = j


    @wp.kernel(module=module)
    def post_step_kernel(
        alive: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        discovered: wp.array(dtype=wp.int32),       # type: ignore[valid-type]
        current_level: wp.array(dtype=wp.int32),    # type: ignore[valid-type]
    ):
        """Between BFS steps: (1) if nothing was discovered this step, mark
        BFS as saturated (``alive = 0``); (2) reset ``discovered = 0`` for
        the next step; (3) advance ``current_level``.

        Combining the reset and the liveness check into one single-thread
        kernel halves the per-step overhead. Single-thread launch.
        """
        tid = wp.tid()
        if tid != 0:
            return
        if discovered[0] == int(0):
            alive[0] = int(0)
        discovered[0] = int(0)
        current_level[0] = current_level[0] + int(1)

    @wp.kernel(module=module)
    def append_unreached_kernel(
        n_in: int,
        level: wp.array(dtype=wp.int32),            # type: ignore[valid-type]
        order_buf: wp.array(dtype=wp.int32),        # type: ignore[valid-type]
        head: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
    ):
        """For disconnected graphs: append any vertex with ``level == -1``
        to the BFS order in ascending index order.

        Run as a single-thread serialized scan for determinism and simplicity.
        """
        tid = wp.tid()
        if tid != 0:
            return
        pos = head[0]
        for i in range(n_in):
            if level[i] == int(-1):
                order_buf[pos] = i
                pos += int(1)
        head[0] = pos

    @wp.kernel(module=module)
    def reverse_into_perm_kernel(
        n_in: int,
        order_buf: wp.array(dtype=wp.int32),        # type: ignore[valid-type]
        perm: wp.array(dtype=wp.int32),             # type: ignore[valid-type]
    ):
        """``perm[i] = order_buf[n - 1 - i]`` -- the "reverse" in RCM."""
        i = wp.tid()
        if i >= n_in:
            return
        perm[i] = order_buf[n_in - int(1) - i]

    return {
        "init_and_degree": init_and_degree_kernel,
        "select_root": select_root_kernel,
        "init_frontier": init_frontier_kernel,
        "bfs_step": bfs_step_kernel,
        "post_step": post_step_kernel,
        "append_unreached": append_unreached_kernel,
        "reverse_into_perm": reverse_into_perm_kernel,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_bfs_iters(n: int) -> int:
    """Upper bound on BFS depth we'll actually execute.

    Real-world BFS depths on structured sparse graphs are typically O(sqrt(n))
    or smaller. We use ``2*ceil(sqrt(n)) + 4`` as a safe upper bound that
    drastically reduces kernel-launch count compared to the conservative n.

    If the graph's actual BFS depth exceeds this cap, any unreached vertices
    are caught by :func:`append_unreached_kernel` at the end and appended in
    ascending index order (i.e., treated as additional disconnected components).

    Tunable via the ``max_bfs_iters`` parameter on :func:`create_rcm_launch`.
    """
    import math
    return 2 * int(math.ceil(math.sqrt(n))) + 4


def create_rcm_launch(
    A: wp.array,
    perm: wp.array,
    tol: float = 0.0,
    max_bfs_iters: int | None = None,
    use_cuda_graph: bool = True,
    device=None,
    stream=None,
) -> Callable[[], None]:
    """Build a zero-argument callback that writes an RCM permutation into ``perm``.

    Parameters
    ----------
    A : wp.array
        Dense ``(n, n)`` SPD matrix, row-major flat (size ``n*n``). dtype must
        be a Warp float type. Only ``|A_ij| > tol`` is consulted, so the
        numerical values only affect the sparsity pattern.
    perm : wp.array
        Output permutation, shape ``(n,)``, dtype ``int32``.
    tol : float
        Threshold used when discovering the sparsity pattern. ``0.0`` (default)
        treats any nonzero off-diagonal entry as an edge.
    use_cuda_graph : bool
        If True, the returned callable replays a prebuilt CUDA graph. If
        False, launches are issued eagerly (useful for debugging).
    device, stream:
        Forwarded to the capture helper.
    """
    n = perm.shape[0]
    if A.size != n * n:
        raise ValueError(f"A must have {n*n} elements (n*n); got {A.size}")
    if perm.dtype != wp.int32:
        raise TypeError(f"perm must be int32; got {perm.dtype}")
    dtype = A.dtype

    if device is None:
        device = A.device
    if max_bfs_iters is None:
        max_bfs_iters = _default_bfs_iters(n)
    # Clamp to n to keep the safety invariant that BFS can always complete.
    max_bfs_iters = min(max_bfs_iters, n)
    scratch = allocate_rcm_scratch(n, device=device)

    K = _make_rcm_kernels(dtype, n)

    # Pre-record all launches (fixed-topology).
    init_and_degree_launch = wp.launch(
        K["init_and_degree"],
        dim=n,
        inputs=[
            n, float(tol), A,
            scratch["degree"], scratch["level"], scratch["head"],
            scratch["alive"], scratch["discovered"],
            scratch["root"], scratch["current_level"],
        ],
        device=device, stream=stream, record_cmd=True,
    )
    select_root_launch = wp.launch(
        K["select_root"],
        dim=1, inputs=[n, scratch["degree"], scratch["root"]],
        device=device, stream=stream, record_cmd=True,
    )
    init_frontier_launch = wp.launch(
        K["init_frontier"],
        dim=1,
        inputs=[scratch["root"], scratch["level"], scratch["order_buf"], scratch["head"]],
        device=device, stream=stream, record_cmd=True,
    )
    bfs_step_launch = wp.launch(
        K["bfs_step"],
        dim=n,
        inputs=[
            n, float(tol), A,
            scratch["level"], scratch["order_buf"], scratch["head"],
            scratch["alive"], scratch["discovered"], scratch["current_level"],
        ],
        device=device, stream=stream, record_cmd=True,
    )
    post_step_launch = wp.launch(
        K["post_step"],
        dim=1,
        inputs=[scratch["alive"], scratch["discovered"], scratch["current_level"]],
        device=device, stream=stream, record_cmd=True,
    )
    append_unreached_launch = wp.launch(
        K["append_unreached"],
        dim=1,
        inputs=[n, scratch["level"], scratch["order_buf"], scratch["head"]],
        device=device, stream=stream, record_cmd=True,
    )
    reverse_launch = wp.launch(
        K["reverse_into_perm"],
        dim=n,
        inputs=[n, scratch["order_buf"], perm],
        device=device, stream=stream, record_cmd=True,
    )

    def callback():
        init_and_degree_launch.launch()
        select_root_launch.launch()
        init_frontier_launch.launch()

        # BFS up to max_bfs_iters steps. After saturation (``alive=0``) each
        # bfs_step is a no-op, but we still launch to keep the graph topology
        # fixed. Any vertices not reached by then are picked up by
        # append_unreached_kernel (treated as additional components).
        # Each step = 2 launches: bfs_step + post_step (fused reset+liveness).
        for _ in range(max_bfs_iters):
            bfs_step_launch.launch()
            post_step_launch.launch()

        append_unreached_launch.launch()
        reverse_launch.launch()

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device=device, stream=stream)
    return callback
