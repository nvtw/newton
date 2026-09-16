"""Exercise the existing reduced G1 production recipe without resetting failures."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.g1 import EnvG1PhoenX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--world-count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_g1_reduced_current.json"))
    parser.add_argument("--benchmark-frames", type=int, default=0)
    args = parser.parse_args()
    wp.init()
    config = g1_recipe.default_g1_env_config(
        world_count=args.world_count,
        max_episode_steps=0,
        auto_reset=False,
        randomize_commands_on_reset=False,
        reset_noise=0.0,
    )
    env = EnvG1PhoenX(config, device="cuda:0")
    env.reset()
    actions = wp.zeros((args.world_count, env.action_dim), dtype=wp.float32, device=env.device)
    history = []
    pose_history = []
    velocity_history = []
    failure = None
    for frame in range(args.frames):
        obs, rewards, dones = env.step(actions)
        q = env.state_0.joint_q.numpy().reshape(args.world_count, env.coord_stride)
        qd = env.state_0.joint_qd.numpy().reshape(args.world_count, env.dof_stride)
        body_q = env.state_0.body_q.numpy()
        pose_history.append(q.copy())
        velocity_history.append(qd.copy())
        count = int(env.contacts.rigid_contact_count.numpy()[0])
        finite = all(np.isfinite(a).all() for a in (q, qd, body_q, obs.numpy(), rewards.numpy()))
        error = np.abs(q[:, 7 : 7 + env.action_dim] - env.default_joint_pos.numpy())
        history.append(
            {
                "frame": frame,
                "finite": bool(finite),
                "min_base_height_m": float(q[:, 2].min()),
                "max_joint_speed_rad_s": float(np.abs(qd[:, 6:]).max()),
                "mean_tracking_error_rad": float(error.mean()),
                "max_tracking_error_rad": float(error.max()),
                "contacts": count,
                "terminated": int(np.count_nonzero(dones.numpy())),
            }
        )
        if not finite or count >= config.rigid_contact_max_per_world * args.world_count:
            failure = f"Nonfinite state or contact capacity reached at frame {frame}"
            break
    report = {
        "status": "failed" if failure else "completed",
        "failure": failure,
        "purpose": "Physical smoke with fixed zero actions; not trained-policy balance acceptance or timing",
        "world_count": args.world_count,
        "frame_dt": config.frame_dt,
        "sim_substeps": config.sim_substeps,
        "solver_iterations": config.solver_iterations,
        "articulation_mode": config.articulation_mode,
        "reduced_articulation_path": config.reduced_articulation_path,
        "actuation_model": config.actuation_model,
        "contact_friction_model": config.contact_friction_model,
        "history": history,
    }
    if args.benchmark_frames:
        if args.benchmark_frames % 2:
            raise ValueError("Benchmark frames must be even for the alternating state buffers")
        with wp.ScopedCapture(device=env.device) as capture:
            env.step(actions)
            env.step(actions)
        for _ in range(5):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(env.device)
        start = time.perf_counter()
        for _ in range(args.benchmark_frames // 2):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(env.device)
        elapsed = time.perf_counter() - start
        report["timing"] = {
            "frames": args.benchmark_frames,
            "seconds": elapsed,
            "ms_per_policy_step": elapsed * 1000 / args.benchmark_frames,
            "world_steps_per_second": args.world_count * args.benchmark_frames / elapsed,
            "scope": "Full EnvG1PhoenX.step CUDA graph; includes observations/rewards, excludes checks",
        }
        assert np.isfinite(env.state_0.body_q.numpy()).all()
    np.savez_compressed(
        args.output.with_suffix(".npz"), joint_q=np.asarray(pose_history), joint_qd=np.asarray(velocity_history)
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}))
    print("FINAL", history[-1])
    if failure:
        raise AssertionError(failure)


if __name__ == "__main__":
    main()
