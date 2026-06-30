# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train a Unitree G1 walking policy with the nanoG1 recipe in PhoenX.

A direct port of the frozen nanoG1 v3 task to PhoenX. It matches the
randomized velocity-command distribution, 98-value observation layout, leg-only
action mask, phase-based gait shaping, reward weights, Muon optimizer, replay
schedule, and mirror-symmetry loss from the pinned nanoG1 PufferLib build.

Run:

    uv run --extra dev -m newton._src.solvers.phoenx.rl_training.examples.train_g1_nanog1 \\
        --device cuda:0 --output-dir /tmp/g1_nanog1

Checkpoints are written to --output-dir every --checkpoint-interval iterations
and after the final iteration. Pass --no-randomize-commands to run the easier
fixed forward-command ablation instead of the matched nanoG1 task.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

# nanoG1 walk-gate threshold: tracking_perf >= 0.3 sustained at 0.8 m/s.
_WALK_GATE_TRACKING_PERF = 0.30
_WALK_GATE_SAMPLES = 75_000_000  # nanoG1 passes here on average


def _nanog1_env_config(
    *,
    world_count: int,
    command_x: float,
    sim_substeps: int,
    solver_iterations: int,
    articulation_mode: str,
    ground_friction: float,
    randomize_commands: bool,
    command_x_range: tuple[float, float],
    command_y_range: tuple[float, float],
    command_yaw_range: tuple[float, float],
    command_zero_probability: float,
) -> rl.ConfigEnvG1PhoenX:
    return g1_recipe.default_g1_env_config(
        world_count=world_count,
        command=(float(command_x), 0.0, 0.0),
        sim_substeps=sim_substeps,
        solver_iterations=solver_iterations,
        articulation_mode=articulation_mode,
        ground_friction=ground_friction,
        # nanoG1 reward weights
        reward_mode="nanog1_dense",
        w_alive=g1_recipe.W_ALIVE,
        w_termination=g1_recipe.W_TERMINATION,
        w_track_lin=g1_recipe.W_TRACK_LIN,
        w_track_ang=g1_recipe.W_TRACK_ANG,
        w_lin_vel_z=g1_recipe.W_LIN_VEL_Z,
        w_ang_vel_xy=g1_recipe.W_ANG_VEL_XY,
        w_orientation=g1_recipe.W_ORIENTATION,
        w_torque=g1_recipe.W_TORQUE,
        w_action_rate=g1_recipe.W_ACTION_RATE,
        # nanoG1 v3 gait shaping compiled by G1_TASK_V3
        w_command_progress=0.0,
        w_gait_contact=g1_recipe.W_GAIT_CONTACT,
        w_gait_swing=g1_recipe.W_GAIT_SWING,
        w_gait_swing_contact=g1_recipe.W_GAIT_SWING_CONTACT,
        w_gait_hip=g1_recipe.W_GAIT_HIP,
        w_base_height=g1_recipe.W_BASE_HEIGHT,
        w_feet_air_time=0.0,
        w_feet_slide=0.0,
        w_joint_deviation_hip=0.0,
        w_joint_deviation_waist=0.0,
        w_joint_deviation_upper=0.0,
        w_joint_acc_legs=0.0,
        w_joint_pos_limit_ankle=0.0,
        w_mechanical_power=0.0,
        # Physics — same as recipe defaults, kept explicit for clarity
        actuation_model=g1_recipe.ACTUATION_MODEL,
        action_scale=g1_recipe.ACTION_SCALE,
        controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
        contact_geometry=g1_recipe.CONTACT_GEOMETRY,
        joint_friction_model=g1_recipe.JOINT_FRICTION_MODEL,
        # Command randomization
        randomize_commands_on_reset=randomize_commands,
        command_x_range=command_x_range,
        command_y_range=command_y_range,
        command_yaw_range=command_yaw_range,
        command_zero_probability=command_zero_probability,
    )


def _nanog1_ppo_config() -> rl.ConfigPPO:
    return g1_recipe.default_g1_ppo_config(
        optimizer=g1_recipe.OPTIMIZER,
        actor_lr=g1_recipe.ACTOR_LR,
        critic_lr=g1_recipe.CRITIC_LR,
        muon_momentum=g1_recipe.MUON_MOMENTUM,
        anneal_lr=g1_recipe.ANNEAL_LR,
        lr_anneal_timesteps=g1_recipe.LR_ANNEAL_TIMESTEPS,
        min_lr_ratio=g1_recipe.MIN_LR_RATIO,
        gamma=g1_recipe.GAMMA,
        gae_lambda=g1_recipe.GAE_LAMBDA,
        clip_ratio=g1_recipe.CLIP_RATIO,
        entropy_coeff=g1_recipe.ENTROPY_COEFF,
        max_grad_norm=g1_recipe.MAX_GRAD_NORM,
        reward_clip=g1_recipe.REWARD_CLIP,
        minibatch_size=g1_recipe.MINIBATCH_SIZE,
        replay_ratio=g1_recipe.REPLAY_RATIO,
        priority_alpha=g1_recipe.PRIORITY_ALPHA,
        priority_beta=g1_recipe.PRIORITY_BETA,
        vtrace_rho_clip=g1_recipe.VTRACE_RHO_CLIP,
        vtrace_c_clip=g1_recipe.VTRACE_C_CLIP,
        mirror_loss_coeff=g1_recipe.MIRROR_LOSS_COEFF,
        value_loss_coeff=g1_recipe.VALUE_LOSS_COEFF,
        value_clip_range=g1_recipe.VALUE_CLIP_RANGE,
        manual_actor_backward=g1_recipe.MANUAL_ACTOR_BACKWARD,
        manual_critic_backward=g1_recipe.MANUAL_CRITIC_BACKWARD,
        manual_mlp_weight_grad_dtype=g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE,
        manual_mlp_forward_dtype=g1_recipe.MANUAL_MLP_FORWARD_DTYPE,
        policy_network=g1_recipe.POLICY_NETWORK,
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Warp device, e.g. 'cuda:0'")
    parser.add_argument("--output-dir", default="/tmp/g1_nanog1", help="Checkpoint output directory")
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument(
        "--world-count",
        type=int,
        default=g1_recipe.WORLD_COUNT,
        help="Number of parallel simulation worlds (default: %(default)s)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=g1_recipe.ROLLOUT_STEPS,
        help="Policy steps per PPO rollout (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=573,
        help=(
            "Training iterations.  Default 573 ≈ 150M samples at "
            f"{g1_recipe.WORLD_COUNT} worlds x {g1_recipe.ROLLOUT_STEPS} steps."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print diagnostics every N iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N iterations; 0 = only at end (default: %(default)s)",
    )
    parser.add_argument("--resume-checkpoint", default=None, help="Resume from this .npz checkpoint")
    parser.add_argument(
        "--resume-policy-only",
        action="store_true",
        help="Load network weights but initialize fresh optimizer state",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph_leapfrog"),
        default="graph_leapfrog",
        help="CUDA execution mode — graph_leapfrog gives ~2-4x throughput (default: %(default)s)",
    )
    parser.add_argument(
        "--command-x",
        type=float,
        default=0.8,
        help="Fixed forward velocity command [m/s] (default: %(default)s, same as nanoG1)",
    )
    parser.add_argument(
        "--randomize-commands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample nanoG1 commands per episode; pass --no-randomize-commands to train only on the fixed --command-x."
        ),
    )
    parser.add_argument("--command-x-min", type=float, default=g1_recipe.COMMAND_X_RANGE[0])
    parser.add_argument("--command-x-max", type=float, default=g1_recipe.COMMAND_X_RANGE[1])
    parser.add_argument("--command-y-min", type=float, default=g1_recipe.COMMAND_Y_RANGE[0])
    parser.add_argument("--command-y-max", type=float, default=g1_recipe.COMMAND_Y_RANGE[1])
    parser.add_argument("--command-yaw-min", type=float, default=g1_recipe.COMMAND_YAW_RANGE[0])
    parser.add_argument("--command-yaw-max", type=float, default=g1_recipe.COMMAND_YAW_RANGE[1])
    parser.add_argument(
        "--command-zero-probability",
        type=float,
        default=g1_recipe.COMMAND_ZERO_PROBABILITY,
        help="Fraction of episodes assigned zero command when randomizing (default: %(default)s)",
    )
    parser.add_argument(
        "--command-curriculum-start",
        type=float,
        default=1.0,
        help="Initial multiplier for randomized command ranges (default: %(default)s)",
    )
    parser.add_argument(
        "--command-curriculum-samples",
        type=int,
        default=0,
        help="Samples over which command ranges grow to full scale; 0 disables it (default: %(default)s)",
    )
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument(
        "--articulation-mode",
        choices=("maximal", "hybrid", "reduced"),
        default="reduced",
        help="PhoenX articulation dynamics mode (default: %(default)s)",
    )
    parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=g1_recipe.LOG_STD_INIT,
        help="Initial policy log-standard-deviation (default: %(default)s)",
    )
    parser.add_argument(
        "--mirror-loss-coeff",
        type=float,
        default=g1_recipe.MIRROR_LOSS_COEFF,
        help="Left/right policy symmetry coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--no-readback-diagnostics",
        action="store_true",
        help="Skip per-iteration CPU readback for action/log-std diagnostics (faster)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_per_iter = int(args.rollout_steps) * int(args.world_count)
    total_samples = int(args.iterations) * samples_per_iter
    walk_gate_iter = max(1, _WALK_GATE_SAMPLES // samples_per_iter)

    print(
        f"G1 nanoG1 training\n"
        f"  worlds={args.world_count}  rollout={args.rollout_steps}  "
        f"samples_per_iter={samples_per_iter:,}\n"
        f"  iterations={args.iterations}  total_samples={total_samples / 1e6:.1f}M\n"
        f"  walk_gate_at_iter≈{walk_gate_iter}  "
        f"({_WALK_GATE_SAMPLES / 1e6:.0f}M samples)\n"
        f"  execution={args.execution_mode}  "
        f"articulation={args.articulation_mode}  "
        f"command_x={args.command_x:.2f}  "
        f"randomize_commands={args.randomize_commands}\n"
        f"  output_dir={output_dir}"
    )

    env_config = _nanog1_env_config(
        world_count=int(args.world_count),
        command_x=float(args.command_x),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        articulation_mode=str(args.articulation_mode),
        ground_friction=float(args.ground_friction),
        randomize_commands=bool(args.randomize_commands),
        command_x_range=(float(args.command_x_min), float(args.command_x_max)),
        command_y_range=(float(args.command_y_min), float(args.command_y_max)),
        command_yaw_range=(float(args.command_yaw_min), float(args.command_yaw_max)),
        command_zero_probability=float(args.command_zero_probability),
    )
    ppo_config = _nanog1_ppo_config()
    ppo_config.mirror_loss_coeff = float(args.mirror_loss_coeff)
    checkpoint_pattern = str(output_dir / "g1_nanog1_{iteration:06d}.npz")

    t_start = time.perf_counter()
    result = rl.train_g1_ppo(
        rl.ConfigTrainG1PPO(
            iterations=int(args.iterations),
            rollout_steps=int(args.rollout_steps),
            log_std_init=float(args.log_std_init),
            env_config=env_config,
            ppo_config=ppo_config,
            device=args.device,
            seed=int(args.seed),
            log_interval=int(args.log_interval),
            randomize_commands=bool(args.randomize_commands),
            command_sampling="episode",
            command_x_range=(float(args.command_x_min), float(args.command_x_max)),
            command_y_range=(float(args.command_y_min), float(args.command_y_max)),
            command_yaw_range=(float(args.command_yaw_min), float(args.command_yaw_max)),
            command_zero_probability=float(args.command_zero_probability),
            command_curriculum_start=float(args.command_curriculum_start),
            command_curriculum_samples=int(args.command_curriculum_samples),
            reset_recurrent_state_on_rollout_start=g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
            resume_checkpoint=args.resume_checkpoint,
            resume_policy_only=bool(args.resume_policy_only),
            checkpoint_path=checkpoint_pattern,
            checkpoint_interval=int(args.checkpoint_interval),
            readback_diagnostics=not bool(args.no_readback_diagnostics),
            execution_mode=str(args.execution_mode),
        )
    )
    elapsed = time.perf_counter() - t_start

    final_iter = int(result.trainer.iteration)
    final_samples = final_iter * samples_per_iter
    avg_sps = final_samples / max(elapsed, 1.0e-9)
    t_walk_estimate = _WALK_GATE_SAMPLES / avg_sps if avg_sps > 0 else float("inf")

    final_stats = result.history[-1] if result.history else None
    print("\n=== G1 nanoG1 RESULT ===")
    print(
        f"  elapsed={elapsed:.1f}s  "
        f"iterations={final_iter}  samples={final_samples / 1e6:.1f}M\n"
        f"  avg_sps={avg_sps:.0f}  est_T_walk={t_walk_estimate:.1f}s"
    )
    if final_stats is not None:
        print(
            f"  final_tracking_perf={final_stats.mean_tracking_perf:.3f}  "
            f"final_reward_step={final_stats.mean_reward:.4f}  "
            f"final_done={final_stats.mean_done:.4f}"
        )
        walk_fraction = final_samples / _WALK_GATE_SAMPLES
        status = (
            "tracking threshold reached"
            if final_stats.mean_tracking_perf >= _WALK_GATE_TRACKING_PERF
            else "tracking threshold not reached"
        )
        print(f"  training_signal={status}  samples_vs_75M={walk_fraction:.2f}x")
        print("  Run the deterministic quality gate before claiming the policy walks.")

    final_path = str(output_dir / f"g1_nanog1_{final_iter:06d}.npz")
    result.trainer.save_checkpoint(final_path, iteration=final_iter)
    print(f"  final_checkpoint={final_path}")
    print("=== END ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
