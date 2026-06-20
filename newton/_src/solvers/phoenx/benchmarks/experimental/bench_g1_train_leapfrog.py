# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark an experimental leapfrog G1 PPO schedule.

The leapfrog schedule updates a master PPO trainer from rollout N while a frozen
rollout trainer collects rollout N+1 on another CUDA stream. After both streams
finish, master weights are copied to the frozen trainer for the next iteration.
This measures the real scheduling pattern needed by an async PPO/V-trace mode,
including the frozen-policy device copy. It does not claim training quality.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog
"""

from __future__ import annotations

import argparse
import json
import time

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_rollout_update_overlap import (
    _g1_env_config,
    _g1_ppo_config,
    _make_trainer,
    _parse_hidden_layers,
    _randomize_commands,
)
from newton._src.solvers.phoenx.rl_training import g1_recipe


def _copy_trainer_policy(dst: rl.TrainerPPO, src: rl.TrainerPPO) -> None:
    dst.actor.copy_from(src.actor)
    if dst.critic is not None or src.critic is not None:
        if dst.critic is None or src.critic is None:
            raise RuntimeError("trainer critic layouts do not match")
        dst.critic.copy_from(src.critic)


def _make_buffer(env: rl.EnvG1PhoenX, steps: int) -> rl.BufferRollout:
    return rl.BufferRollout(
        num_steps=int(steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=env.device,
    )


class _LeapfrogFixture:
    def __init__(self, args: argparse.Namespace, device: wp.context.Device):
        self.args = args
        self.device = device
        env_config = _g1_env_config(args)
        ppo_config = _g1_ppo_config(args)
        self.env = rl.EnvG1PhoenX(env_config, device=device)
        self.master = _make_trainer(self.env, ppo_config, args, seed=int(args.seed))
        self.rollout = _make_trainer(self.env, ppo_config, args, seed=int(args.seed) + 17)
        self.buffers = (
            _make_buffer(self.env, int(args.rollout_steps)),
            _make_buffer(self.env, int(args.rollout_steps)),
        )
        for trainer in (self.master, self.rollout):
            for buffer in self.buffers:
                trainer.reserve_update_buffers(buffer)
        _copy_trainer_policy(self.rollout, self.master)
        self.iteration = 0

    def collect(self, buffer: rl.BufferRollout) -> None:
        seed = int(self.args.seed) + 100_003 * self.iteration
        _randomize_commands(self.env, self.args, seed)
        self.env.collect_ppo_rollout(self.rollout, buffer, seed=seed + 10_000_000)
        self.iteration += 1

    def update(self, buffer: rl.BufferRollout) -> None:
        self.master.update(buffer, read_stats=False)
        self.master.iteration += 1

    @property
    def num_samples(self) -> int:
        return self.buffers[0].num_samples


def _run_sync_pair(
    fixture: _LeapfrogFixture, prev: int, nxt: int, copy_stream: wp.Stream | None = None
) -> tuple[int, int]:
    fixture.collect(fixture.buffers[nxt])
    fixture.update(fixture.buffers[prev])
    if copy_stream is None:
        _copy_trainer_policy(fixture.rollout, fixture.master)
    else:
        with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
            _copy_trainer_policy(fixture.rollout, fixture.master)
        wp.wait_stream(copy_stream)
    return nxt, prev


def _run_leapfrog_pair(
    fixture: _LeapfrogFixture,
    prev: int,
    nxt: int,
    *,
    rollout_stream: wp.Stream,
    update_stream: wp.Stream,
    copy_stream: wp.Stream,
) -> tuple[int, int]:
    main_stream = wp.get_stream(fixture.device)
    for stream in (rollout_stream, update_stream):
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(main_stream)

    with wp.ScopedStream(rollout_stream, sync_enter=False, sync_exit=False):
        fixture.collect(fixture.buffers[nxt])
    with wp.ScopedStream(update_stream, sync_enter=False, sync_exit=False):
        fixture.update(fixture.buffers[prev])

    with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(rollout_stream)
        wp.wait_stream(update_stream)
        _copy_trainer_policy(fixture.rollout, fixture.master)
    wp.wait_stream(copy_stream)
    return nxt, prev


def _prime_fixture(fixture: _LeapfrogFixture) -> tuple[int, int]:
    fixture.collect(fixture.buffers[0])
    fixture.update(fixture.buffers[0])
    _copy_trainer_policy(fixture.rollout, fixture.master)
    fixture.collect(fixture.buffers[1])
    wp.synchronize_device(fixture.device)
    return 1, 0


def _measure_sync(args: argparse.Namespace, device: wp.context.Device) -> tuple[float, int]:
    fixture = _LeapfrogFixture(args, device)
    prev, nxt = _prime_fixture(fixture)
    copy_stream = wp.Stream(device)
    for _ in range(int(args.warmup_iterations)):
        prev, nxt = _run_sync_pair(fixture, prev, nxt, copy_stream)
    wp.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(int(args.iterations)):
        prev, nxt = _run_sync_pair(fixture, prev, nxt, copy_stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12), fixture.num_samples


def _measure_leapfrog(args: argparse.Namespace, device: wp.context.Device) -> tuple[float, int]:
    fixture = _LeapfrogFixture(args, device)
    prev, nxt = _prime_fixture(fixture)
    rollout_stream = wp.Stream(device)
    update_stream = wp.Stream(device)
    copy_stream = wp.Stream(device)
    for _ in range(int(args.warmup_iterations)):
        prev, nxt = _run_leapfrog_pair(
            fixture,
            prev,
            nxt,
            rollout_stream=rollout_stream,
            update_stream=update_stream,
            copy_stream=copy_stream,
        )
    wp.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(int(args.iterations)):
        prev, nxt = _run_leapfrog_pair(
            fixture,
            prev,
            nxt,
            rollout_stream=rollout_stream,
            update_stream=update_stream,
            copy_stream=copy_stream,
        )
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12), fixture.num_samples


def _capture_graph(stream: wp.Stream, device: wp.context.Device, workload) -> object:
    main_stream = wp.get_stream(device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(main_stream)
        with wp.ScopedCapture(device=device, stream=stream) as capture:
            workload()
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return capture.graph


def _measure_sync_graphs(args: argparse.Namespace, device: wp.context.Device) -> tuple[float, int]:
    fixture = _LeapfrogFixture(args, device)
    prev, nxt = _prime_fixture(fixture)
    copy_stream = wp.Stream(device)
    for _ in range(int(args.warmup_iterations)):
        prev, nxt = _run_sync_pair(fixture, prev, nxt, copy_stream)
    wp.synchronize_device(device)

    stream = wp.Stream(device)
    rollout_graph = _capture_graph(stream, device, lambda: fixture.collect(fixture.buffers[nxt]))
    update_graph = _capture_graph(stream, device, lambda: fixture.update(fixture.buffers[prev]))
    copy_graph = _capture_graph(stream, device, lambda: _copy_trainer_policy(fixture.rollout, fixture.master))
    wp.capture_launch(rollout_graph, stream=stream)
    wp.capture_launch(update_graph, stream=stream)
    wp.capture_launch(copy_graph, stream=stream)
    wp.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(int(args.iterations)):
        wp.capture_launch(rollout_graph, stream=stream)
        wp.capture_launch(update_graph, stream=stream)
        wp.capture_launch(copy_graph, stream=stream)
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12), fixture.num_samples


class _GraphPhase:
    def __init__(self, rollout_graph, update_graph, prev: int, nxt: int):
        self.rollout_graph = rollout_graph
        self.update_graph = update_graph
        self.prev = int(prev)
        self.nxt = int(nxt)


def _capture_graph_phase(
    fixture: _LeapfrogFixture,
    *,
    prev: int,
    nxt: int,
    rollout_stream: wp.Stream,
    update_stream: wp.Stream,
    device: wp.context.Device,
) -> _GraphPhase:
    rollout_graph = _capture_graph(rollout_stream, device, lambda: fixture.collect(fixture.buffers[nxt]))
    update_graph = _capture_graph(update_stream, device, lambda: fixture.update(fixture.buffers[prev]))
    return _GraphPhase(rollout_graph, update_graph, prev, nxt)


def _measure_leapfrog_graphs(args: argparse.Namespace, device: wp.context.Device) -> tuple[float, int]:
    fixture = _LeapfrogFixture(args, device)
    prev, nxt = _prime_fixture(fixture)
    rollout_stream = wp.Stream(device)
    update_stream = wp.Stream(device)
    copy_stream = wp.Stream(device)
    for _ in range(int(args.warmup_iterations)):
        prev, nxt = _run_leapfrog_pair(
            fixture,
            prev,
            nxt,
            rollout_stream=rollout_stream,
            update_stream=update_stream,
            copy_stream=copy_stream,
        )
    wp.synchronize_device(device)

    phase_a = _capture_graph_phase(
        fixture,
        prev=prev,
        nxt=nxt,
        rollout_stream=rollout_stream,
        update_stream=update_stream,
        device=device,
    )
    phase_b = _capture_graph_phase(
        fixture,
        prev=nxt,
        nxt=prev,
        rollout_stream=rollout_stream,
        update_stream=update_stream,
        device=device,
    )
    copy_graph = _capture_graph(copy_stream, device, lambda: _copy_trainer_policy(fixture.rollout, fixture.master))

    for phase in (phase_a, phase_b):
        wp.capture_launch(phase.rollout_graph, stream=rollout_stream)
        wp.capture_launch(phase.update_graph, stream=update_stream)
        with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(rollout_stream)
            wp.wait_stream(update_stream)
        wp.capture_launch(copy_graph, stream=copy_stream)
    wp.synchronize_device(device)

    t0 = time.perf_counter()
    for i in range(int(args.iterations)):
        phase = phase_a if i % 2 == 0 else phase_b
        wp.capture_launch(phase.rollout_graph, stream=rollout_stream)
        wp.capture_launch(phase.update_graph, stream=update_stream)
        with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(rollout_stream)
            wp.wait_stream(update_stream)
        wp.capture_launch(copy_graph, stream=copy_stream)
    wp.wait_stream(copy_stream)
    wp.synchronize_device(device)
    return max(time.perf_counter() - t0, 1.0e-12), fixture.num_samples


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("G1 leapfrog benchmark requires CUDA with Warp mempool enabled")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    sync_seconds, samples_per_iter = _measure_sync(args, device)
    leapfrog_seconds, _ = _measure_leapfrog(args, device)
    graph_sync_seconds = 0.0
    graph_leapfrog_seconds = 0.0
    graph_error = ""
    if not args.no_graphs:
        try:
            graph_sync_seconds, _ = _measure_sync_graphs(args, device)
            graph_leapfrog_seconds, _ = _measure_leapfrog_graphs(args, device)
        except RuntimeError as exc:
            graph_error = str(exc)
    total_samples = float(samples_per_iter) * float(args.iterations)
    sync_sps = total_samples / sync_seconds
    leapfrog_sps = total_samples / leapfrog_seconds
    graph_sync_sps = total_samples / graph_sync_seconds if graph_sync_seconds > 0.0 else 0.0
    graph_leapfrog_sps = total_samples / graph_leapfrog_seconds if graph_leapfrog_seconds > 0.0 else 0.0
    return {
        "engine": "phoenx_g1_leapfrog_ppo_experimental",
        "metric": "real frozen-policy rollout/update stream overlap, not training quality",
        "device": device.name,
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "iterations": int(args.iterations),
        "warmup_iterations": int(args.warmup_iterations),
        "samples_per_iteration": int(samples_per_iter),
        "sim_substeps": int(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "train_epochs": int(args.train_epochs),
        "minibatch_size": int(args.minibatch_size),
        "replay_ratio": float(args.replay_ratio),
        "manual_mlp_weight_grad_dtype": str(args.manual_mlp_weight_grad_dtype),
        "manual_mlp_forward_dtype": str(args.manual_mlp_forward_dtype),
        "sync_seconds": sync_seconds,
        "leapfrog_seconds": leapfrog_seconds,
        "sync_env_samples_per_s": sync_sps,
        "leapfrog_env_samples_per_s": leapfrog_sps,
        "leapfrog_speedup": sync_seconds / leapfrog_seconds,
        "graph_sync_seconds": graph_sync_seconds,
        "graph_leapfrog_seconds": graph_leapfrog_seconds,
        "graph_sync_env_samples_per_s": graph_sync_sps,
        "graph_leapfrog_env_samples_per_s": graph_leapfrog_sps,
        "graph_leapfrog_speedup": graph_sync_seconds / graph_leapfrog_seconds if graph_leapfrog_seconds > 0.0 else 0.0,
        "graph_fixed_seed_note": "separate stream graphs replay fixed Python scalar seeds; use only as a throughput upper bound",
        "graph_error": graph_error,
        "command_randomization": not bool(args.no_command_randomization),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=1)
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
    parser.add_argument("--graphs", dest="no_graphs", action="store_false")
    parser.set_defaults(no_graphs=True)
    return parser


def main() -> None:
    wp.init()
    args = build_arg_parser().parse_args()
    result = benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
