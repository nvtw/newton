# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private helpers for PhoenX unit tests."""

import unittest

import warp as wp

__all__ = [
    "STEP_LAYOUTS",
    "make_graph_stepper",
    "make_solver_graph_stepper",
    "require_cuda_graph_capture",
    "run_settle_loop",
    "run_solver_capture_loop",
]

STEP_LAYOUTS = ("multi_world", "single_world")


def require_cuda_graph_capture(label: str = "PhoenX tests"):
    device = wp.get_preferred_device()
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise unittest.SkipTest(
            f"{label} require CUDA graph capture with Warp mempool enabled (device: {device.name!r})."
        )
    return device


def run_settle_loop(world, frames: int, dt: float) -> None:
    """Advance a PhoenX world with one captured graph."""
    if frames < 1:
        return

    device = require_cuda_graph_capture("PhoenX settle tests")
    world.step(dt)
    if frames == 1:
        return

    with wp.ScopedCapture(device=device) as capture:
        world.step(dt)

    for _ in range(frames - 2):
        wp.capture_launch(capture.graph)


def make_graph_stepper(world, dt: float):
    """Return a persistent graph-backed ``world.step`` loop."""
    device = require_cuda_graph_capture("PhoenX graph tests")
    graph = None
    head_start = 0

    def step(n_frames: int) -> None:
        nonlocal graph, head_start
        n = int(n_frames)
        if n <= 0:
            return
        if graph is None:
            world.step(dt)
            with wp.ScopedCapture(device=device) as capture:
                world.step(dt)
            graph = capture.graph
            head_start = 2

        consumed = min(head_start, n)
        head_start -= consumed
        for _ in range(n - consumed):
            wp.capture_launch(graph)

    return step


def make_solver_graph_stepper(
    solver,
    state_0,
    state_1,
    control,
    contacts,
    model,
    dt: float,
    *,
    collide: bool = True,
    clear_forces: bool = True,
):
    """Return a graph-backed stepper for SolverPhoenX ping-pong states."""
    device = require_cuda_graph_capture("PhoenX solver tests")
    graph = None
    head_start = 0

    def step_pair() -> None:
        nonlocal state_0, state_1
        for _ in range(2):
            if clear_forces:
                state_0.clear_forces()
            if collide and contacts is not None:
                model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    def step(frame_count: int):
        nonlocal graph, head_start
        n = int(frame_count)
        if n <= 0:
            return state_0, state_1
        if graph is None:
            if n < 4:
                raise ValueError("first captured SolverPhoenX step request must be at least four frames")
            step_pair()
            with wp.ScopedCapture(device=device) as capture:
                step_pair()
            graph = capture.graph
            head_start = 4

        consumed = min(head_start, n)
        head_start -= consumed
        remaining = n - consumed
        if remaining % 2:
            raise ValueError("captured SolverPhoenX stepper advances two frames per replay")

        for _ in range(remaining // 2):
            wp.capture_launch(graph)

        return state_0, state_1

    return step


def run_solver_capture_loop(
    solver,
    state_0,
    state_1,
    control,
    contacts,
    model,
    frames: int,
    dt: float,
    *,
    collide: bool = True,
    clear_forces: bool = True,
):
    """Advance SolverPhoenX with a captured two-step graph."""
    step = make_solver_graph_stepper(
        solver,
        state_0,
        state_1,
        control,
        contacts,
        model,
        dt,
        collide=collide,
        clear_forces=clear_forces,
    )
    return step(frames)
