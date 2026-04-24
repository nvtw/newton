# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private test-only helpers for the jitter solver unit tests.

This module is *not* part of the public API; it lives under
``newton._src`` and is imported exclusively from ``test_*.py``
siblings. It consolidates the hot inner-loop patterns that every
jitter test file used to open-code so bug fixes (graph-capture
correctness, warm-up semantics, CPU fallback) land in one place.

Currently only :func:`run_settle_loop` lives here. Add more shared
helpers as the tests grow.
"""

import warp as wp

__all__ = ["GRAPH_CAPTURE_FRAME_THRESHOLD", "run_settle_loop"]


# Minimum ``frames`` count at which CUDA graph capture pays for itself.
#
# Each ``world.step(dt)`` call on CUDA pays a fixed Python + Warp
# launch overhead of ~0.5-2 ms (hundreds of tiny kernel launches per
# substep); a CUDA graph replay collapses that to ~10 us. Below this
# threshold the capture setup (warm-up step + ScopedCapture) is a net
# loss, so we fall back to eager stepping. 40 frames was chosen
# empirically: at FPS=60 that's ~0.67 s of simulated time, i.e. any
# non-trivial settle loop amortises the capture.
GRAPH_CAPTURE_FRAME_THRESHOLD = 40


def run_settle_loop(world, frames: int, dt: float) -> None:
    """Advance ``world`` by ``frames`` steps of size ``dt`` seconds each.

    On CUDA and when ``frames`` exceeds :data:`GRAPH_CAPTURE_FRAME_THRESHOLD`,
    records a single ``world.step(dt)`` into a CUDA graph after a warm-up
    step and replays it for the remaining iterations. Warp graph
    semantics: the graph captures the exact kernel launches made by
    ``world.step(dt)``; because tests use a fixed ``dt`` and a fixed
    solver configuration, replays are bit-identical to eager mode.

    Falls back to plain eager stepping on CPU (where ``wp.ScopedCapture``
    is a no-op) and for small frame counts (where capture overhead
    dominates).

    Args:
        world: A :class:`newton._src.solvers.phoenx.World` built via
            :class:`WorldBuilder`. The world must be fully finalised
            (``b.finalize(...)`` has already been called).
        frames: Number of ``step(dt)`` iterations to advance.
        dt: Timestep per iteration [s].
    """
    device = wp.get_device()
    use_graph = device.is_cuda and frames >= GRAPH_CAPTURE_FRAME_THRESHOLD

    if not use_graph:
        for _ in range(frames):
            world.step(dt)
        return

    # Warm-up step outside the capture: ensures all Warp modules
    # referenced by ``step(dt)`` are JIT-compiled and loaded, and
    # absorbs the first-call allocation of any lazily-created scratch
    # buffers. Capturing those would fail (or pin the wrong
    # allocation into the graph).
    world.step(dt)

    with wp.ScopedCapture(device=device) as capture:
        world.step(dt)
    graph = capture.graph

    # Already advanced two steps (one warm-up + one capture). Replay
    # the remainder through the graph.
    for _ in range(frames - 2):
        wp.capture_launch(graph)
