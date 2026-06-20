# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure whether G1 rollout and PPO update workloads overlap on CUDA streams.

This is a feasibility benchmark, not a production async PPO trainer. It runs a
G1 rollout workload and a PPO update workload on independent buffers/trainers so
we can measure the GPU overlap available from CUDA streams and CUDA graph replay
without changing PPO semantics in ``train_g1_ppo``.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_rollout_update_overlap
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_rollout_update_overlap --no-graphs
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

Workload = Callable[[int], None]


def _parse_hidden_layers(text: str) -> tuple[int, ...]:
    widths = tuple(int(item) for item in text.split(",") if item)
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("hidden layers must be a comma-separated list of positive widths")
    return widths


def _g1_ppo_config(args: argparse.Namespace) -> rl.ConfigPPO:
    return g1_recipe.default_g1_ppo_config(
        train_epochs=int(args.train_epochs),
        mirror_loss_coeff=float(args.mirror_loss_coeff),
        minibatch_size=int(args.minibatch_size),
        replay_ratio=float(args.replay_ratio),
        priority_alpha=float(args.priority_alpha),
        priority_beta=float(args.priority_beta),
        manual_actor_backward=not bool(args.no_manual_actor_backward),
        manual_critic_backward=not bool(args.no_manual_critic_backward),
        manual_mlp_weight_grad_dtype=str(args.manual_mlp_weight_grad_dtype),
        manual_mlp_forward_dtype=str(args.manual_mlp_forward_dtype),
        vtrace_rho_clip=float(args.vtrace_rho_clip),
        vtrace_c_clip=float(args.vtrace_c_clip),
        reward_clip=float(args.reward_clip),
        max_grad_norm=float(args.max_grad_norm),
    )


def _g1_env_config(args: argparse.Namespace) -> rl.ConfigEnvG1PhoenX:
    return rl.ConfigEnvG1PhoenX(
        world_count=int(args.world_count),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        controlled_action_count=int(args.controlled_action_count),
        parse_meshes=bool(args.parse_meshes),
        rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
        threads_per_world=args.threads_per_world,
        multi_world_scheduler=str(args.multi_world_scheduler),
        prepare_refresh_stride=args.prepare_refresh_stride,
    )


def _make_trainer(
    env: rl.EnvG1PhoenX,
    ppo_config: rl.ConfigPPO,
    args: argparse.Namespace,
    *,
    seed: int,
) -> rl.TrainerPPO:
    return rl.TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=tuple(args.hidden_layers),
        config=ppo_config,
        device=env.device,
        seed=int(seed),
        squash_actions=True,
        activation=str(args.activation),
        log_std_init=float(args.log_std_init),
        mirror_map=rl.g1_mirror_map_ppo() if ppo_config.mirror_loss_coeff > 0.0 else None,
    )


def _randomize_commands(env: rl.EnvG1PhoenX, args: argparse.Namespace, seed: int) -> None:
    if args.no_command_randomization:
        return
    env.randomize_commands(
        seed=int(seed),
        command_x_range=tuple(args.command_x_range),
        command_y_range=tuple(args.command_y_range),
        command_yaw_range=tuple(args.command_yaw_range),
    )


def _make_workloads(args: argparse.Namespace, device: wp.context.Device) -> tuple[Workload, Workload, int]:
    env_config = _g1_env_config(args)
    ppo_config = _g1_ppo_config(args)

    env_rollout = rl.EnvG1PhoenX(env_config, device=device)
    trainer_rollout = _make_trainer(env_rollout, ppo_config, args, seed=int(args.seed))
    buffer_rollout = rl.BufferRollout(
        num_steps=int(args.rollout_steps),
        num_envs=env_rollout.world_count,
        obs_dim=env_rollout.obs_dim,
        action_dim=env_rollout.action_dim,
        device=device,
    )
    trainer_rollout.reserve_update_buffers(buffer_rollout)

    env_update = rl.EnvG1PhoenX(env_config, device=device)
    trainer_update = _make_trainer(env_update, ppo_config, args, seed=int(args.seed) + 17)
    buffer_update = rl.BufferRollout(
        num_steps=int(args.rollout_steps),
        num_envs=env_update.world_count,
        obs_dim=env_update.obs_dim,
        action_dim=env_update.action_dim,
        device=device,
    )
    trainer_update.reserve_update_buffers(buffer_update)

    _randomize_commands(env_rollout, args, int(args.seed) + 1_000)
    env_rollout.collect_ppo_rollout(trainer_rollout, buffer_rollout, seed=int(args.seed) + 2_000)
    _randomize_commands(env_update, args, int(args.seed) + 3_000)
    env_update.collect_ppo_rollout(trainer_update, buffer_update, seed=int(args.seed) + 4_000)
    trainer_update.update(buffer_update, read_stats=False)
    wp.synchronize_device(device)

    def rollout_workload(index: int) -> None:
        seed = int(args.seed) + 10_000 + int(index)
        _randomize_commands(env_rollout, args, seed)
        env_rollout.collect_ppo_rollout(trainer_rollout, buffer_rollout, seed=seed + 1_000_000)

    def update_workload(index: int) -> None:
        del index
        trainer_update.update(buffer_update, read_stats=False)

    return rollout_workload, update_workload, buffer_rollout.num_samples


def _run_on_stream(stream: wp.Stream, workload: Workload, index: int) -> None:
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        workload(index)


def _measure_single_stream(
    workload_a: Workload,
    workload_b: Workload | None,
    *,
    stream: wp.Stream,
    device: wp.context.Device,
    repeats: int,
    start_index: int,
) -> float:
    main_stream = wp.get_stream(device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(main_stream)
    t0 = time.perf_counter()
    for i in range(int(repeats)):
        index = int(start_index) + i
        _run_on_stream(stream, workload_a, index)
        if workload_b is not None:
            _run_on_stream(stream, workload_b, index)
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12)


def _measure_two_streams(
    rollout_workload: Workload,
    update_workload: Workload,
    *,
    rollout_stream: wp.Stream,
    update_stream: wp.Stream,
    device: wp.context.Device,
    repeats: int,
    start_index: int,
) -> float:
    main_stream = wp.get_stream(device)
    for stream in (rollout_stream, update_stream):
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(main_stream)
    t0 = time.perf_counter()
    for i in range(int(repeats)):
        index = int(start_index) + i
        _run_on_stream(rollout_stream, rollout_workload, index)
        _run_on_stream(update_stream, update_workload, index)
    wp.wait_stream(rollout_stream)
    wp.wait_stream(update_stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12)


def _capture_workload_graph(
    workload: Workload,
    *,
    stream: wp.Stream,
    device: wp.context.Device,
    index: int,
):
    main_stream = wp.get_stream(device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(main_stream)
        with wp.ScopedCapture(device=device, stream=stream) as capture:
            workload(int(index))
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return capture.graph


def _measure_single_stream_graphs(
    graph_a,
    graph_b,
    *,
    stream: wp.Stream,
    device: wp.context.Device,
    repeats: int,
) -> float:
    main_stream = wp.get_stream(device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(main_stream)
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        wp.capture_launch(graph_a, stream=stream)
        if graph_b is not None:
            wp.capture_launch(graph_b, stream=stream)
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12)


def _measure_two_stream_graphs(
    rollout_graph,
    update_graph,
    *,
    rollout_stream: wp.Stream,
    update_stream: wp.Stream,
    device: wp.context.Device,
    repeats: int,
) -> float:
    main_stream = wp.get_stream(device)
    for stream in (rollout_stream, update_stream):
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(main_stream)
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        wp.capture_launch(rollout_graph, stream=rollout_stream)
        wp.capture_launch(update_graph, stream=update_stream)
    wp.wait_stream(rollout_stream)
    wp.wait_stream(update_stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12)


def _metrics(samples_per_pair: int, repeats: int, rollout_s: float, update_s: float, seq_s: float, overlap_s: float):
    env_samples = float(samples_per_pair) * float(repeats)
    seq_sps = env_samples / seq_s
    overlap_sps = env_samples / overlap_s
    hidden = max((rollout_s + update_s) - overlap_s, 0.0)
    return {
        "rollout_seconds": rollout_s,
        "update_seconds": update_s,
        "sequential_seconds": seq_s,
        "overlap_seconds": overlap_s,
        "sequential_env_samples_per_s": seq_sps,
        "overlap_env_samples_per_s": overlap_sps,
        "overlap_speedup": seq_s / overlap_s,
        "hidden_seconds": hidden,
        "hidden_fraction_of_update": min(hidden / max(update_s, 1.0e-12), 1.0),
        "rollout_plus_update_seconds": rollout_s + update_s,
    }


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("G1 rollout/update overlap benchmark requires CUDA with Warp mempool enabled")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative")

    rollout_workload, update_workload, samples_per_pair = _make_workloads(args, device)

    warmup_rollout = wp.Stream(device)
    warmup_update = wp.Stream(device)
    if args.warmup_repeats > 0:
        _measure_two_streams(
            rollout_workload,
            update_workload,
            rollout_stream=warmup_rollout,
            update_stream=warmup_update,
            device=device,
            repeats=int(args.warmup_repeats),
            start_index=100_000,
        )

    rollout_stream = wp.Stream(device)
    update_stream = wp.Stream(device)
    single_stream = wp.Stream(device)
    rollout_only_stream = wp.Stream(device)
    update_only_stream = wp.Stream(device)

    rollout_s = _measure_single_stream(
        rollout_workload,
        None,
        stream=rollout_only_stream,
        device=device,
        repeats=int(args.repeats),
        start_index=200_000,
    )
    update_s = _measure_single_stream(
        update_workload,
        None,
        stream=update_only_stream,
        device=device,
        repeats=int(args.repeats),
        start_index=300_000,
    )
    seq_s = _measure_single_stream(
        rollout_workload,
        update_workload,
        stream=single_stream,
        device=device,
        repeats=int(args.repeats),
        start_index=400_000,
    )
    overlap_s = _measure_two_streams(
        rollout_workload,
        update_workload,
        rollout_stream=rollout_stream,
        update_stream=update_stream,
        device=device,
        repeats=int(args.repeats),
        start_index=500_000,
    )

    result: dict[str, object] = {
        "engine": "phoenx_g1_rollout_update_overlap",
        "metric": "independent rollout/update workload overlap, not training quality",
        "device": device.name,
        "mode": "eager_streams",
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "samples_per_pair": int(samples_per_pair),
        "repeats": int(args.repeats),
        "warmup_repeats": int(args.warmup_repeats),
        "sim_substeps": int(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "train_epochs": int(args.train_epochs),
        "minibatch_size": int(args.minibatch_size),
        "replay_ratio": float(args.replay_ratio),
        "manual_mlp_weight_grad_dtype": str(args.manual_mlp_weight_grad_dtype),
        "manual_mlp_forward_dtype": str(args.manual_mlp_forward_dtype),
        "command_randomization": not bool(args.no_command_randomization),
        "eager": _metrics(samples_per_pair, int(args.repeats), rollout_s, update_s, seq_s, overlap_s),
    }

    if not args.no_graphs:
        graph_rollout_stream = wp.Stream(device)
        graph_update_stream = wp.Stream(device)
        rollout_graph = _capture_workload_graph(
            rollout_workload,
            stream=graph_rollout_stream,
            device=device,
            index=600_000,
        )
        update_graph = _capture_workload_graph(
            update_workload,
            stream=graph_update_stream,
            device=device,
            index=700_000,
        )
        graph_single_stream = wp.Stream(device)
        graph_rollout_only_stream = wp.Stream(device)
        graph_update_only_stream = wp.Stream(device)
        graph_overlap_rollout_stream = wp.Stream(device)
        graph_overlap_update_stream = wp.Stream(device)

        graph_rollout_s = _measure_single_stream_graphs(
            rollout_graph,
            None,
            stream=graph_rollout_only_stream,
            device=device,
            repeats=int(args.repeats),
        )
        graph_update_s = _measure_single_stream_graphs(
            update_graph,
            None,
            stream=graph_update_only_stream,
            device=device,
            repeats=int(args.repeats),
        )
        graph_seq_s = _measure_single_stream_graphs(
            rollout_graph,
            update_graph,
            stream=graph_single_stream,
            device=device,
            repeats=int(args.repeats),
        )
        graph_overlap_s = _measure_two_stream_graphs(
            rollout_graph,
            update_graph,
            rollout_stream=graph_overlap_rollout_stream,
            update_stream=graph_overlap_update_stream,
            device=device,
            repeats=int(args.repeats),
        )
        result["graphs"] = _metrics(
            samples_per_pair,
            int(args.repeats),
            graph_rollout_s,
            graph_update_s,
            graph_seq_s,
            graph_overlap_s,
        )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--hidden-layers", type=_parse_hidden_layers, default=g1_recipe.HIDDEN_LAYERS)
    parser.add_argument("--activation", default=g1_recipe.ACTIVATION)
    parser.add_argument("--log-std-init", type=float, default=g1_recipe.LOG_STD_INIT)
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    parser.add_argument("--parse-meshes", action="store_true", default=g1_recipe.PARSE_MESHES)
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--threads-per-world", default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", default=g1_recipe.PREPARE_REFRESH_STRIDE)
    parser.add_argument("--train-epochs", type=int, default=g1_recipe.TRAIN_EPOCHS)
    parser.add_argument("--mirror-loss-coeff", type=float, default=g1_recipe.MIRROR_LOSS_COEFF)
    parser.add_argument("--minibatch-size", type=int, default=g1_recipe.MINIBATCH_SIZE)
    parser.add_argument("--replay-ratio", type=float, default=g1_recipe.REPLAY_RATIO)
    parser.add_argument("--priority-alpha", type=float, default=g1_recipe.PRIORITY_ALPHA)
    parser.add_argument("--priority-beta", type=float, default=g1_recipe.PRIORITY_BETA)
    parser.add_argument("--no-manual-actor-backward", action="store_true")
    parser.add_argument("--no-manual-critic-backward", action="store_true")
    parser.add_argument("--manual-mlp-weight-grad-dtype", default=g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE)
    parser.add_argument("--manual-mlp-forward-dtype", default=g1_recipe.MANUAL_MLP_FORWARD_DTYPE)
    parser.add_argument("--vtrace-rho-clip", type=float, default=g1_recipe.VTRACE_RHO_CLIP)
    parser.add_argument("--vtrace-c-clip", type=float, default=g1_recipe.VTRACE_C_CLIP)
    parser.add_argument("--reward-clip", type=float, default=g1_recipe.REWARD_CLIP)
    parser.add_argument("--max-grad-norm", type=float, default=g1_recipe.MAX_GRAD_NORM)
    parser.add_argument("--no-command-randomization", action="store_true")
    parser.add_argument("--command-x-range", type=float, nargs=2, default=g1_recipe.COMMAND_X_RANGE)
    parser.add_argument("--command-y-range", type=float, nargs=2, default=g1_recipe.COMMAND_Y_RANGE)
    parser.add_argument("--command-yaw-range", type=float, nargs=2, default=g1_recipe.COMMAND_YAW_RANGE)
    parser.add_argument("--no-graphs", action="store_true")
    return parser


def main() -> None:
    wp.init()
    args = build_arg_parser().parse_args()
    result = benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
