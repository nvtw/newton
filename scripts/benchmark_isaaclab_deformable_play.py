# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Measure IsaacLab RSL-RL checkpoint play in synchronized 24-step batches.

Run from an IsaacLab 09b1d544dd checkout with this Newton revision installed:

    uv run --no-sync python C:/git3/newton/scripts/benchmark_isaaclab_deformable_play.py \
        --task Isaac-Lift-Cloth-Franka --num_envs 16 --viz none \
        physics=newton_mjwarp_vbd_proxy
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import warp as wp

wp.config.enable_backward = False


def main(argv: list[str] | None = None) -> int:
    import isaaclab_tasks  # noqa: F401
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils import to_dict
    from isaaclab_rl.entrypoints import common
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("Isaac-Lift-Soft-Franka", "Isaac-Lift-Cloth-Franka"), required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--batch_steps", type=int, default=24)
    parser.add_argument("--batches", type=int, default=20)
    add_launcher_args(parser)
    common.add_frontend_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:] if argv is None else argv)
    sys.argv = [sys.argv[0], *remaining]

    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=True)
    import importlib.metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed

    with launch_simulation(env_cfg, args):
        checkpoint = args.checkpoint or common.resolve_published_checkpoint("rsl_rl", args.task, env_cfg)
        if checkpoint is None:
            raise FileNotFoundError(f"No published checkpoint for {args.task}; pass --checkpoint")
        env = common.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=True)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner_type = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
            runner = runner_type(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            import torch

            def step():
                nonlocal obs
                with torch.inference_mode():
                    result = env.step(policy(obs))
                if len(result) == 5:
                    obs, _, terminated, truncated, _ = result
                    dones = terminated | truncated
                else:
                    obs, _, dones, _ = result
                if hasattr(policy, "reset"):
                    policy.reset(dones)

            for _ in range(args.warmup_steps):
                step()
            torch.cuda.synchronize()

            batch_rates = []
            for _ in range(args.batches):
                start = time.perf_counter()
                for _ in range(args.batch_steps):
                    step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                batch_rates.append(args.num_envs * args.batch_steps / elapsed)

            import newton

            print(
                "BENCHMARK_RESULT="
                + json.dumps(
                    {
                        "task": args.task,
                        "num_envs": args.num_envs,
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "newton_path": newton.__file__,
                        "batch_rates": batch_rates,
                        "mean_environment_steps_per_s": statistics.mean(batch_rates),
                    }
                ),
                flush=True,
            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
