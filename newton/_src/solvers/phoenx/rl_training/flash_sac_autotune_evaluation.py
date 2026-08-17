# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Captured paired held-out evaluation for FlashSAC LR autotuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .flash_sac import EnvFlashSAC, TrainerFlashSAC

if TYPE_CHECKING:
    from .flash_sac_autotune import (
        ControllerFlashSACLRAutotune,
        GraphFlashSACLRAutotune,
        ResultFlashSACLRAutotune,
    )


@wp.kernel
def _reset_evaluation_kernel(
    seed: int,
    seed_counters: wp.array[wp.int32],
    scores: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    safe: wp.array[wp.int32],
    terminated: wp.array[wp.int32],
):
    world = wp.tid()
    if world == 0:
        seed_counters[0] = seed
    scores[world] = 0.0
    alive[world] = 1
    safe[world] = 1
    terminated[world] = 0


@wp.kernel
def _accumulate_evaluation_kernel(
    actions: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    terminateds: wp.array[wp.float32],
    scores: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    safe: wp.array[wp.int32],
    terminated: wp.array[wp.int32],
):
    world = wp.tid()
    if alive[world] == 0:
        return
    finite = wp.isfinite(rewards[world])
    for action in range(actions.shape[1]):
        finite = finite and wp.isfinite(actions[world, action])
    if not finite:
        alive[world] = 0
        safe[world] = 0
    else:
        scores[world] = scores[world] + rewards[world]
        if terminateds[world] != 0.0:
            alive[world] = 0
            terminated[world] = 1


def _clone_evaluation_trainer(source: TrainerFlashSAC, world_count: int) -> TrainerFlashSAC:
    trainer = TrainerFlashSAC(
        obs_dim=source.obs_dim,
        action_dim=source.action_dim,
        config=source.config,
        device=source.device,
        seed=source.seed,
    )
    trainer.copy_training_state_from(source)
    trainer.reserve_buffers(world_count)
    return trainer


class EvaluatorPairedFlashSAC:
    """Evaluate two policies on isolated, identically seeded environments.

    Args:
        trainers: Initial champion and challenger trainers.
        envs: Isolated champion and challenger evaluation environments.
        horizon_steps: Fixed evaluation horizon.
        seed: Shared deterministic evaluation seed.
        metric_source: Optional callback returning one device metric per world.
            Environment rewards are accumulated when omitted.
    """

    @dataclass(frozen=True)
    class Result:
        """Store named outputs from one paired evaluation."""

        champion_scores: np.ndarray
        challenger_scores: np.ndarray
        champion_finite: bool
        challenger_finite: bool
        champion_termination_rate: float
        challenger_termination_rate: float

    def __init__(
        self,
        trainers: tuple[TrainerFlashSAC, TrainerFlashSAC],
        envs: tuple[EnvFlashSAC, EnvFlashSAC],
        *,
        horizon_steps: int,
        seed: int,
        metric_source: Callable[[EnvFlashSAC, wp.array[wp.float32]], wp.array[wp.float32]] | None = None,
    ):
        if len(trainers) != 2 or len(envs) != 2:
            raise ValueError("paired evaluation requires exactly two trainers and environments")
        if int(horizon_steps) <= 0:
            raise ValueError("evaluation horizon must be positive")
        first = trainers[0]
        worlds = int(envs[0].world_count)
        if worlds <= 0 or int(envs[1].world_count) != worlds:
            raise ValueError("paired evaluation environments must have equal positive world counts")
        for trainer, env in zip(trainers, envs, strict=True):
            if trainer.device != first.device or env.device != first.device:
                raise ValueError("paired evaluation must use one device")
            if trainer.obs_dim != env.obs_dim or trainer.action_dim != int(
                getattr(env, "policy_action_dim", env.action_dim)
            ):
                raise ValueError("evaluation trainer and environment dimensions do not match")
        self.device = first.device
        self.world_count = worlds
        self.horizon_steps = int(horizon_steps)
        self.seed = int(seed)
        self.metric_source = metric_source
        self.envs = envs
        self.trainers = tuple(_clone_evaluation_trainer(trainer, worlds) for trainer in trainers)
        self._seed_counters = tuple(wp.array([self.seed], dtype=wp.int32, device=self.device) for _ in range(2))
        self._scores = tuple(wp.zeros(worlds, dtype=wp.float32, device=self.device) for _ in range(2))
        self._alive = tuple(wp.ones(worlds, dtype=wp.int32, device=self.device) for _ in range(2))
        self._safe = tuple(wp.ones(worlds, dtype=wp.int32, device=self.device) for _ in range(2))
        self._terminated = tuple(wp.zeros(worlds, dtype=wp.int32, device=self.device) for _ in range(2))
        for env, counter in zip(self.envs, self._seed_counters, strict=True):
            if hasattr(env, "use_reset_seed_counter"):
                env.use_reset_seed_counter(counter)
            if hasattr(env, "use_command_seed_counter"):
                env.use_command_seed_counter(counter)
        sim_times = tuple(getattr(env, "sim_time", None) for env in self.envs)
        with wp.ScopedCapture(device=self.device) as capture:
            for member in range(2):
                trainer = self.trainers[member]
                env = self.envs[member]
                counter = self._seed_counters[member]
                wp.launch(
                    _reset_evaluation_kernel,
                    dim=worlds,
                    inputs=[self.seed],
                    outputs=[
                        counter,
                        self._scores[member],
                        self._alive[member],
                        self._safe[member],
                        self._terminated[member],
                    ],
                    device=self.device,
                )
                obs = env.reset()
                for step in range(self.horizon_steps):
                    actions, _log_probs = trainer.act_reuse_seed_counter(
                        obs,
                        seed_counter=counter,
                        seed_offset=step,
                        deterministic=True,
                    )
                    next_obs, rewards, dones = env.step(actions)
                    metrics = rewards if self.metric_source is None else self.metric_source(env, rewards)
                    if metrics.shape != rewards.shape:
                        raise ValueError("evaluation metric source must return one value per world")
                    wp.launch(
                        _accumulate_evaluation_kernel,
                        dim=worlds,
                        inputs=[
                            actions,
                            metrics,
                            getattr(env, "step_terminateds", dones),
                        ],
                        outputs=[
                            self._scores[member],
                            self._alive[member],
                            self._safe[member],
                            self._terminated[member],
                        ],
                        device=self.device,
                    )
                    obs = next_obs
        self.graph = capture.graph
        for env, sim_time in zip(self.envs, sim_times, strict=True):
            if sim_time is not None:
                env.sim_time = sim_time

    def evaluate(self, sources: tuple[TrainerFlashSAC, TrainerFlashSAC]) -> EvaluatorPairedFlashSAC.Result:
        """Copy policies into isolated trainers and replay one paired evaluation."""

        for destination, source in zip(self.trainers, sources, strict=True):
            destination.copy_training_state_from(source)
        wp.capture_launch(self.graph)
        champion_scores = self._scores[0].numpy() / np.float32(self.horizon_steps)
        challenger_scores = self._scores[1].numpy() / np.float32(self.horizon_steps)
        challenger_finite = bool(np.all(self._safe[1].numpy() != 0))
        champion_finite = bool(np.all(self._safe[0].numpy() != 0))
        champion_termination_rate = float(np.mean(self._terminated[0].numpy()))
        challenger_termination_rate = float(np.mean(self._terminated[1].numpy()))
        return self.Result(
            champion_scores=champion_scores,
            challenger_scores=challenger_scores,
            champion_finite=champion_finite,
            challenger_finite=challenger_finite,
            champion_termination_rate=champion_termination_rate,
            challenger_termination_rate=challenger_termination_rate,
        )


@dataclass
class CadenceFlashSACLRAutotune:
    """Run captured training and coarse deterministic paired evaluation.

    Args:
        controller: Bounded learning-rate search controller.
        training_graph: Captured backend-neutral training graph.
        evaluator: Isolated paired evaluator.
        evaluation_interval: Training launches between paired evaluations.
        launch_count: Initial launch counter.
    """

    controller: ControllerFlashSACLRAutotune
    training_graph: GraphFlashSACLRAutotune
    evaluator: EvaluatorPairedFlashSAC
    evaluation_interval: int
    launch_count: int = 0

    def __post_init__(self) -> None:
        if int(self.evaluation_interval) <= 0:
            raise ValueError("evaluation_interval must be positive")

    def launch(self) -> ResultFlashSACLRAutotune | None:
        """Launch training once and evaluate only at the configured coarse interval."""

        self.training_graph.launch()
        self.launch_count += 1
        if self.launch_count % int(self.evaluation_interval) != 0 or self.controller.converged:
            return None
        self.training_graph.synchronize()
        sources = getattr(self.training_graph, "trainers", self.controller.trainers)
        scores = self.evaluator.evaluate(sources)
        if hasattr(self.training_graph, "evaluate_paired"):
            return self.training_graph.evaluate_paired(
                scores.champion_scores,
                scores.challenger_scores,
                challenger_safe=scores.champion_finite and scores.challenger_finite,
                champion_termination_rate=scores.champion_termination_rate,
                challenger_termination_rate=scores.challenger_termination_rate,
            )
        return self.controller.evaluate_paired(
            scores.champion_scores,
            scores.challenger_scores,
            challenger_safe=scores.champion_finite and scores.challenger_finite,
            champion_termination_rate=scores.champion_termination_rate,
            challenger_termination_rate=scores.challenger_termination_rate,
        )
