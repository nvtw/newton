#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Train ANYmal C walking policy from scratch.

Usage:
    uv run python train_anymal.py                    # 200M steps, 4096 envs
    uv run python train_anymal.py --steps 50000000   # shorter run
    uv run python train_anymal.py --resume            # resume from checkpoint
"""
import argparse
import os
import resource
import subprocess
import sys
import threading

# Memory watchdog
_RSS_LIMIT = 15000
_stop = threading.Event()


def _watchdog():
    while not _stop.is_set():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024 > _RSS_LIMIT:
            print(f"\nWATCHDOG: RSS exceeded {_RSS_LIMIT} MB", flush=True)
            os._exit(1)
        _stop.wait(2.0)


threading.Thread(target=_watchdog, daemon=True).start()

import warp as wp  # noqa: E402

wp.init()

import numpy as np  # noqa: E402

from newton._src.pufferlib.envs.anymal import AnymalEnv  # noqa: E402
from newton._src.pufferlib.onnx_export import export_policy_to_onnx, save_checkpoint, load_checkpoint  # noqa: E402
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer  # noqa: E402


def gpu_power():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True,
        text=True,
    )
    return float(r.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N = args.envs
    best = [-1e9]
    save_count = [0]

    def ckpt(policy, iteration, total_steps, best_return, obs_normalizer):
        if best_return > best[0]:
            best[0] = best_return
            save_checkpoint(policy, "anymal_checkpoint.npz")
            om, ov = (None, None) if obs_normalizer is None else obs_normalizer.get_mean_var()
            export_policy_to_onnx(policy, 48, 12, "anymal_best.onnx", obs_mean=om, obs_var=ov)
            save_count[0] += 1
            print(f"  -> Checkpoint #{save_count[0]} at return {best_return:.1f}", flush=True)

    def log_fn(avg_ret, best_ret, loss_np, sps):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        watts = gpu_power()
        kl = loss_np[5] if len(loss_np) > 5 else 0
        return (
            f"Ret {avg_ret:8.1f} | Best {best_ret:8.1f} | "
            f"Ent {loss_np[3]:7.4f} | KL {kl:8.5f} | "
            f"SPS {sps:>9,.0f} | GPU {watts:.0f}W | RSS {rss}MB"
        )

    config = PPOConfig(
        num_envs=N,
        horizon=24,
        obs_dim=48,
        num_actions=12,
        hidden=128,  # 3×128 with biases = IsaacLab architecture
        total_timesteps=args.steps,
        seed=args.seed,
        # Optimizer: AdamW matching RSL-RL defaults + adaptive LR
        optimizer="adamw",
        lr=1e-3,
        anneal_lr=False,
        desired_kl=0.008,  # Tighter than RSL-RL default to prevent entropy drift
        activation="elu",
        num_hidden_layers=3,
        use_bias=True,
        # PPO hyperparams (RSL-RL defaults)
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=1.0,
        vf_clip_coef=0.2,
        ent_coef=0.003,
        max_grad_norm=1.0,
        momentum=0.9,
        # 5 epochs of 4 minibatches (RSL-RL default)
        replay_ratio=5.0,
        minibatch_size=max(N * 24 // 4, 4096),
        # Continuous action space
        continuous=True,
        init_logstd=-1.2,  # → init_std≈0.3 (gentle for MuJoCo: 3σ=0.9rad offset)
        # Observation normalization (critical for locomotion)
        normalize_obs=False,  # IsaacLab pre-trained policy uses raw obs
        reward_clamp=10.0,  # only_positive clips to [0, ~1.5], clamp at 10 for safety
        # Display
        env_name="ANYmal C Walk",
        best_return_init=-1e9,
        return_format="8.1f",
        step_width=11,
        log_interval=10,
        checkpoint_fn=ckpt,
        format_return_fn=log_fn,
        device="cuda:0",
    )

    def make_env(device):
        return AnymalEnv(num_envs=N, device=device, seed=args.seed, use_mujoco_contacts=True)

    print(f"Training ANYmal C walking policy: {N} envs, {args.steps:,} steps", flush=True)
    PPOTrainer(config, make_env).train()

    # Auto-test the best policy
    _run_test()
    _stop.set()


def _run_test():
    """Test the trained policy headless for 500 frames."""
    print("\nTesting best policy...", flush=True)
    from newton._src.onnx_runtime import OnnxRuntime  # noqa: PLC0415
    from newton import GeoType  # noqa: PLC0415
    import newton  # noqa: PLC0415
    import newton.utils  # noqa: PLC0415

    if not os.path.exists("anymal_best.onnx"):
        print("No checkpoint found, skipping test.", flush=True)
        return

    _LAB = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=np.intp)
    _MJ = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11], dtype=np.intp)

    def qri(q, v):
        qw = q[..., 3:4]
        qv = q[..., :3]
        return v * (2 * qw * qw - 1) - np.cross(qv, v, axis=-1) * qw * 2 + qv * np.sum(qv * v, axis=-1, keepdims=True) * 2

    bb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(bb)
    bb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1e3, limit_kd=1e1)
    bb.default_shape_cfg.ke = 5e4
    bb.default_shape_cfg.kd = 5e2
    bb.default_shape_cfg.kf = 1e3
    bb.default_shape_cfg.mu = 0.75
    pa = newton.utils.download_asset("anybotics_anymal_c")
    bb.add_urdf(str(pa / "urdf" / "anymal.urdf"), floating=True, enable_self_collisions=False,
                collapse_fixed_joints=True, ignore_inertial_definitions=False)
    for i in range(len(bb.shape_type)):
        if bb.shape_type[i] == GeoType.SPHERE:
            r = bb.shape_scale[i][0]
            bb.shape_scale[i] = (r * 2, 0, 0)
    for n2, v2 in {"LF_HAA": 0, "LF_HFE": 0.4, "LF_KFE": -0.8, "RF_HAA": 0, "RF_HFE": 0.4, "RF_KFE": -0.8,
                    "LH_HAA": 0, "LH_HFE": -0.4, "LH_KFE": 0.8, "RH_HAA": 0, "RH_HFE": -0.4, "RH_KFE": 0.8}.items():
        idx = next(i for i, l in enumerate(bb.joint_label) if l.endswith(f"/{n2}"))
        bb.joint_q[idx + 6] = v2
    for i in range(len(bb.joint_target_ke)):
        bb.joint_target_ke[i] = 150
        bb.joint_target_kd[i] = 5
    bb.add_ground_plane()
    mm = bb.finalize()
    sol2 = newton.solvers.SolverMuJoCo(mm, use_mujoco_contacts=False, solver="newton",
                                        ls_parallel=False, ls_iterations=50, njmax=50, nconmax=100)
    s0 = mm.state()
    s1 = mm.state()
    cc = mm.control()
    ct = mm.contacts()
    newton.eval_fk(mm, s0.joint_q, s0.joint_qd, s0)
    pol = OnnxRuntime("anymal_best.onnx", device=str(wp.get_device()))
    ji = s0.joint_q.numpy()[7:].reshape(1, 12).astype(np.float32)
    ar = np.zeros((1, 12), dtype=np.float32)
    cmd = np.array([[1, 0, 0]], dtype=np.float32)
    for f in range(500):
        q = s0.joint_q.numpy()
        qd = s0.joint_qd.numpy()
        rq = q[3:7].reshape(1, 4)
        obs = np.concatenate([
            qri(rq, qd[:3].reshape(1, 3)), qri(rq, qd[3:6].reshape(1, 3)),
            qri(rq, np.array([[0, 0, -1]], dtype=np.float32)), cmd,
            (q[7:].reshape(1, 12) - ji)[:, _LAB], qd[6:].reshape(1, 12)[:, _LAB], ar,
        ], axis=1).astype(np.float32)
        aw = pol({"observation": wp.array(obs, dtype=wp.float32, device=wp.get_device())})["action"]
        ar = aw.numpy()
        tp = np.zeros(18, dtype=np.float32)
        tp[6:] = (ji + 0.5 * ar[:, _MJ]).squeeze(0)
        wp.copy(cc.joint_target_pos, wp.array(tp, dtype=wp.float32, device=wp.get_device()))
        for _ in range(4):
            s0.clear_forces()
            mm.collide(s0, ct)
            sol2.step(s0, s1, cc, ct, 0.005)
            s0, s1 = s1, s0
        if f % 100 == 0:
            print(f"  F{f}: pos=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}]", flush=True)
    q = s0.joint_q.numpy()
    dist = np.linalg.norm(q[:2])
    print(f"  Final: pos=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}] dist={dist:.2f} h={q[2]:.2f}", flush=True)
    print(f"  WALKING: {q[2] > 0.3 and dist > 3.0}", flush=True)


if __name__ == "__main__":
    main()
