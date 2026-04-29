# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private test-only helpers for the jitter solver unit tests.

This module is *not* part of the public API; it lives under
``newton._src`` and is imported exclusively from ``test_*.py``
siblings. It consolidates the hot inner-loop patterns that every
jitter test file used to open-code so bug fixes (graph-capture
correctness, warm-up semantics, CPU fallback) land in one place.
"""

import warp as wp

__all__ = [
    "GRAPH_CAPTURE_FRAME_THRESHOLD",
    "STEP_LAYOUTS",
    "make_graph_stepper",
    "run_settle_loop",
]


# Layouts every behavioural test should exercise. ``"multi_world"`` is
# the default per-world fast-tail dispatch; ``"single_world"`` drives
# the global JP colouring via ``wp.capture_while``. Both must produce
# the same observable physics on any scene -- subTest the assertion
# block over both so a regression in either path fails CI.
STEP_LAYOUTS = ("multi_world", "single_world")


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


def make_graph_stepper(world, dt: float):
    """Return a ``step(n_frames: int) -> None`` callable that advances
    ``world`` ``n_frames`` times at fixed ``dt`` seconds per step,
    using CUDA graph capture under the hood.

    Unlike :func:`run_settle_loop` (which captures a fresh graph each
    call), this returns a *persistent* graph + replay closure -- ideal
    for tests that step a single frame at a time inside a Python loop
    (e.g. recording the pose trajectory) where the per-call capture
    overhead of :func:`run_settle_loop` would defeat the purpose of
    graph capture.

    The graph is recorded once on first use after a warm-up step;
    subsequent invocations replay it via ``wp.capture_launch``. Because
    the warm-up + capture themselves advance the world by 2 frames,
    those frames are carried as a "head start" -- the returned
    closure honours the requested ``n_frames`` budget exactly.

    On CPU (``device.is_cuda is False``) the returned closure falls
    back to plain eager stepping.

    Args:
        world: A finalised :class:`PhoenXWorld`.
        dt: Fixed substep [s] used for every replay. Recording at one
            ``dt`` and replaying at another would silently skew the
            integrator -- always re-create the stepper if ``dt``
            changes.

    Returns:
        A callable ``step(n_frames: int) -> None``.
    """
    device = wp.get_device()
    if not device.is_cuda:
        def _step_eager(n_frames: int) -> None:
            for _ in range(int(n_frames)):
                world.step(dt)
        return _step_eager

    state = {"graph": None, "head_start": 0}

    def _step_cuda(n_frames: int) -> None:
        n = int(n_frames)
        if n <= 0:
            return
        if state["graph"] is None:
            # First call: warm-up + capture. The two frames spent here
            # become a head start that absorbs the next ``n`` request.
            world.step(dt)
            with wp.ScopedCapture(device=device) as capture:
                world.step(dt)
            state["graph"] = capture.graph
            state["head_start"] = 2
        # Honour the caller's frame budget exactly: deduct the head
        # start before replaying.
        consumed = min(state["head_start"], n)
        state["head_start"] -= consumed
        remaining = n - consumed
        for _ in range(remaining):
            wp.capture_launch(state["graph"])

    return _step_cuda
