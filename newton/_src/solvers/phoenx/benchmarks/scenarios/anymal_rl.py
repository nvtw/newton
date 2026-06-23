# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Anymal C PhoenX RL environment benchmark scenario.

This scenario reuses the PhoenX Anymal RL environment so scheduler sweeps
measure the same robot model, contact setup, substep cadence, and solver defaults used by
the Warp-only Anymal PPO examples. The benchmark step uses zero actions and
disables auto-reset to keep CUDA graph capture stable.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.benchmarks.runner import SceneHandle, _gpu_used_bytes
from newton._src.solvers.phoenx.rl_training.anymal import ConfigEnvAnymalPhoenX, EnvAnymalPhoenX


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int = 1,
    *,
    step_layout: str = "multi_world",
    prepare_refresh_stride: int | str = "auto",
) -> SceneHandle:
    """Build the Anymal C RL training environment for scheduler benchmarks."""

    if solver_name != "phoenx":
        raise ValueError(f"Anymal RL benchmark supports only SolverPhoenX, got {solver_name!r}")
    if step_layout != "multi_world":
        raise ValueError("Anymal RL benchmark currently expects step_layout='multi_world'")
    if prepare_refresh_stride != "auto":
        raise ValueError("Anymal RL benchmark currently uses the environment's auto prepare-refresh policy")

    mem_before = _gpu_used_bytes()
    env = EnvAnymalPhoenX(
        ConfigEnvAnymalPhoenX(
            world_count=int(num_worlds),
            sim_substeps=int(substeps),
            solver_iterations=int(solver_iterations),
            velocity_iterations=int(velocity_iterations),
            reward_mode="dense_command",
            command=(1.0, 0.0, 0.0, 0.0),
            auto_reset=False,
        ),
        device=wp.get_device(),
    )
    actions = wp.zeros((int(num_worlds), env.action_dim), dtype=wp.float32, device=env.device)
    solver = env.solver

    def simulate_one_frame() -> None:
        # Keep solver in the closure so scheduler benchmarks can extract it.
        _ = solver.world.num_worlds
        env.step(actions)

    wp.synchronize_device()
    setup_bytes = max(0, _gpu_used_bytes() - mem_before)
    return SceneHandle(
        name="anymal_rl",
        solver_name=solver_name,
        num_worlds=int(num_worlds),
        substeps=int(substeps),
        solver_iterations=int(solver_iterations),
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
