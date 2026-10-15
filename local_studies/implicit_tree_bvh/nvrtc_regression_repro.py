# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro: BVH ray traversal runs up to 1.6x slower with Warp 1.18 (NVRTC 13.4) than 1.17 (NVRTC 12.9).

The kernel walks an implicit (pointer-free) binary BVH over random boxes with a
stackless, front-to-back traversal and counts the boxes each ray enters. The
generated CUDA, compile options and results are identical across the two Warp
versions; only the bundled NVRTC differs.

Run the same file against both versions::

    uv run --no-project --with warp-lang==1.17.0 --with numpy python nvrtc_regression_repro.py
    uv run --no-project --with warp-lang==1.18.0 --with numpy python nvrtc_regression_repro.py

Each run prints the Warp and toolkit versions and, for two variants of the node
load (plain array reads, and ``__ldg`` float4 reads through ``wp.func_native``),
the best time per launch and a checksum of the hit counts (identical across
versions). Measured on an RTX PRO 6000 Blackwell (sm_120), driver 13.4:

==========================  ===================  ===================
node loads                  Warp 1.17            Warp 1.18
                            (NVRTC 12.9)         (NVRTC 13.4)
==========================  ===================  ===================
plain array loads           0.889 ms             1.034 ms (1.16x)
``__ldg`` float4 loads      0.535 ms             0.874 ms (1.63x)
==========================  ===================  ===================

In the application this was reduced from (Newton's implicit BVH), loading the
PTX that Warp 1.17 compiled into Warp 1.18's kernel cache (both versions give
the module the same hash) restored the 1.17 timing with identical source and
compile options, isolating NVRTC 13.4 code generation as the cause.
"""

import numpy as np
import warp as wp

NUM_BOXES = 10_000
NUM_RAYS = 1 << 18
REPLAYS = 50
TRIALS = 7


# ---------------------------------------------------------------------------
# Tree: complete binary heap, node 1 = root, children 2n and 2n + 1, leaves
# [leaf_base, 2 leaf_base). Each node is two vec4s: (lower, unused) and
# (upper, last_leaf as float; -1 marks an empty subtree).
# ---------------------------------------------------------------------------


def build_tree(lowers, uppers):
    n = len(lowers)
    centers = 0.5 * (lowers + uppers)
    grid = ((centers - centers.min(0)) / (np.ptp(centers, 0).max() + 1e-9) * 1023).astype(np.uint64)

    def spread(v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        return (v | (v << 2)) & 0x09249249

    order = np.argsort(spread(grid[:, 0]) | (spread(grid[:, 1]) << 1) | (spread(grid[:, 2]) << 2), kind="stable")
    leaf_base = 1 << int(np.ceil(np.log2(max(n, 2))))
    lo = np.full((2 * leaf_base, 3), 1.0e30, np.float32)
    hi = np.full((2 * leaf_base, 3), -1.0e30, np.float32)
    last = np.full(2 * leaf_base, -1.0, np.float32)
    lo[leaf_base : leaf_base + n] = lowers[order]
    hi[leaf_base : leaf_base + n] = uppers[order]
    last[leaf_base : leaf_base + n] = np.arange(n)
    for node in range(leaf_base - 1, 0, -1):
        lo[node] = np.minimum(lo[2 * node], lo[2 * node + 1])
        hi[node] = np.maximum(hi[2 * node], hi[2 * node + 1])
        last[node] = max(last[2 * node], last[2 * node + 1])
    nodes = np.zeros((2 * leaf_base, 2, 4), np.float32)
    nodes[:, 0, :3] = lo
    nodes[:, 1, :3] = hi
    nodes[:, 1, 3] = last
    return nodes.reshape(-1, 4), leaf_base


@wp.func_native("""
const float4 v = __ldg(reinterpret_cast<const float4*>(ptr) + index);
return wp::vec4(v.x, v.y, v.z, v.w);
""")
def load_float4(ptr: wp.uint64, index: wp.int32) -> wp.vec4: ...


def make_kernel(use_ldg: bool):
    """Build the traversal kernel; node loads use ``__ldg`` (func_native) or plain array reads."""

    @wp.func
    def node_entry(
        nodes: wp.array[wp.vec4],
        nodes_ptr: wp.uint64,
        node: wp.int32,
        origin: wp.vec3,
        inv_dir: wp.vec3,
        max_t: wp.float32,
    ) -> wp.float32:
        """Ray entry parameter into ``node``'s box within ``max_t``, or -1."""
        a = wp.vec4()
        b = wp.vec4()
        if wp.static(use_ldg):
            a = load_float4(nodes_ptr, 2 * node)
            b = load_float4(nodes_ptr, 2 * node + 1)
        else:
            a = nodes[2 * node]
            b = nodes[2 * node + 1]
        if b[3] < 0.0:
            return -1.0
        t1 = wp.cw_mul(wp.vec3(a[0], a[1], a[2]) - origin, inv_dir)
        t2 = wp.cw_mul(wp.vec3(b[0], b[1], b[2]) - origin, inv_dir)
        t_near = wp.min(t1, t2)
        t_far = wp.max(t1, t2)
        t_enter = wp.max(wp.max(t_near[0], t_near[1]), wp.max(t_near[2], 0.0))
        t_exit = wp.min(wp.min(t_far[0], t_far[1]), wp.min(t_far[2], max_t))
        if t_enter <= t_exit:
            return t_enter
        return -1.0

    @wp.func
    def count_hits(
        nodes: wp.array[wp.vec4],
        nodes_ptr: wp.uint64,
        leaf_base: wp.int32,
        origin: wp.vec3,
        direction: wp.vec3,
        max_t: wp.float32,
    ) -> wp.int32:
        """Stackless front-to-back traversal; one trail bit per level records a pending sibling."""
        inv_dir = wp.vec3(1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2])
        hits = wp.int32(0)
        node = wp.int32(1)
        level = wp.int32(-1)
        trail = wp.uint32(0)
        expand = bool(False)
        t = node_entry(nodes, nodes_ptr, 1, origin, inv_dir, max_t)
        pending_t = wp.float32(-1.0)
        if t >= 0.0:
            level = 0
            expand = True
        while level >= 0:
            if expand:
                if node >= leaf_base:
                    if t <= max_t:
                        hits += 1
                    expand = False
                else:
                    left = node * 2
                    t_left = node_entry(nodes, nodes_ptr, left, origin, inv_dir, max_t)
                    t_right = node_entry(nodes, nodes_ptr, left + 1, origin, inv_dir, max_t)
                    if t_left >= 0.0 or t_right >= 0.0:
                        level += 1
                        bit = wp.uint32(1) << wp.uint32(level)
                        trail = trail | bit
                        if t_left >= 0.0 and t_right >= 0.0:
                            trail = trail & ~bit
                        if t_left >= 0.0 and (t_right < 0.0 or t_left <= t_right):
                            node = left
                            t = t_left
                            pending_t = t_right
                        else:
                            node = left + 1
                            t = t_right
                            pending_t = t_left
                    else:
                        expand = False
            elif level == 0:
                level = -1
            else:
                bit = wp.uint32(1) << wp.uint32(level)
                if (trail & bit) == wp.uint32(0):
                    trail = trail | bit
                    node = node ^ 1
                    if node >= leaf_base:
                        t = pending_t
                    else:
                        t = 0.0
                    expand = t >= 0.0
                else:
                    node = node >> 1
                    level -= 1
        return hits

    @wp.kernel(module="unique")
    def ray_hits_kernel(
        nodes: wp.array[wp.vec4],
        nodes_ptr: wp.uint64,
        leaf_base: wp.int32,
        origins: wp.array[wp.vec3],
        directions: wp.array[wp.vec3],
        counts: wp.array[wp.int32],
    ):
        i = wp.tid()
        counts[i] = count_hits(nodes, nodes_ptr, leaf_base, origins[i], directions[i], 1.0e30)

    return ray_hits_kernel


def main():
    wp.init()
    device = wp.get_device("cuda:0")
    rng = np.random.default_rng(1234)
    side = NUM_BOXES ** (1.0 / 3.0)
    centers = rng.uniform(0.0, side, (NUM_BOXES, 3)).astype(np.float32)
    half = rng.uniform(0.1, 0.5, (NUM_BOXES, 3)).astype(np.float32)
    nodes_np, leaf_base = build_tree(centers - half, centers + half)
    directions_np = rng.standard_normal((NUM_RAYS, 3)).astype(np.float32)
    directions_np /= np.linalg.norm(directions_np, axis=1, keepdims=True)

    nodes = wp.array(nodes_np, dtype=wp.vec4, device=device)
    origins = wp.array(rng.uniform(0.0, side, (NUM_RAYS, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    directions = wp.array(directions_np, dtype=wp.vec3, device=device)
    counts = wp.zeros(NUM_RAYS, dtype=wp.int32, device=device)
    inputs = [nodes, nodes.ptr, leaf_base, origins, directions, counts]
    toolkit = getattr(wp._src.context.runtime, "toolkit_version", None)
    print(f"warp {wp.__version__}  toolkit {toolkit}  device {device.name}")
    for name, use_ldg in (("plain array loads", False), ("__ldg float4 loads", True)):
        kernel = make_kernel(use_ldg)
        wp.launch(kernel, dim=NUM_RAYS, inputs=inputs, device=device)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(REPLAYS):
                wp.launch(kernel, dim=NUM_RAYS, inputs=inputs, device=device)
        best = float("inf")
        for _ in range(TRIALS):
            start = wp.Event(device, enable_timing=True)
            end = wp.Event(device, enable_timing=True)
            wp.record_event(start)
            wp.capture_launch(capture.graph)
            wp.record_event(end)
            wp.synchronize_device(device)
            best = min(best, wp.get_event_elapsed_time(start, end) / REPLAYS)
        hits = counts.numpy()
        print(f"  {name:20s} {best:.3f} ms per launch  (hits/ray {hits.mean():.3f}, checksum {int(hits.sum())})")


if __name__ == "__main__":
    main()
