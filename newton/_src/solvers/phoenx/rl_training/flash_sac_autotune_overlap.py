# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Captured overlap lifecycle for internal FlashSAC LR autotuning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .flash_sac import (
    BufferReplayFlashSAC,
    EnvFlashSAC,
    _allocate_flash_sac_batch,
    _capture_flash_stream_graph,
    _copy_flash_sac_batch,
)
from .flash_sac_networks import NetworkFlashSAC
from .sac import BatchSAC

if TYPE_CHECKING:
    from .flash_sac_autotune import ControllerFlashSACLRAutotune


def _make_rollout_actor(
    controller: ControllerFlashSACLRAutotune,
    member: int,
    *,
    seed: int,
):
    trainer = controller.trainers[member]
    source = trainer.actor.net
    actor = copy.copy(trainer.actor)
    actor.net = NetworkFlashSAC(
        input_dim=source.input_dim,
        hidden_dim=source.hidden_dim,
        num_blocks=source.num_blocks,
        output_dim=source.output_dim,
        actor_heads=True,
        device=controller.device,
        seed=int(seed) + member,
        contraction_dtype="float32",
    )
    actor.log_std = wp.clone(trainer.actor.log_std)
    actor._sample_reuse_capacity = 0
    actor._sample_reuse_actions = None
    actor._sample_reuse_log_probs = None
    actor._sample_reuse_eps = None
    actor.copy_from(trainer.actor)
    actor.reserve_reuse_buffers(controller.rollout_world_count)
    return actor


def capture_lr_autotune_overlap(
    controller: ControllerFlashSACLRAutotune,
    env: EnvFlashSAC,
    replay: BufferReplayFlashSAC,
    *,
    updates_per_step: int,
    interactions_per_launch: int,
    seed: int,
) -> GraphFlashSACLRAutotuneOverlap:
    """Capture split rollout and identical-batch population learning."""

    updates = int(updates_per_step)
    interactions = int(interactions_per_launch)
    if env.device != controller.device or replay.device != controller.device:
        raise ValueError("environment, replay, and LR autotuner must use the same device")
    if int(env.world_count) != controller.rollout_world_count:
        raise ValueError("environment world count does not match LR autotuner setup")
    if updates <= 0 or interactions <= 0:
        raise ValueError("overlap cadence dimensions must be positive")
    if interactions % 2 != 0:
        raise ValueError("interactions_per_launch must be even for environment state-buffer parity")
    if not replay.can_sample():
        raise RuntimeError("warm shared replay before capturing LR autotuning overlap")

    total_updates = updates * interactions
    policy_frequency = int(controller.trainers[0].config.policy_frequency)
    if total_updates % policy_frequency != 0:
        raise ValueError("overlap updates must span a complete policy-frequency cadence")
    start_gradient_update = controller.trainers[0]._gradient_update_count
    replay.reserve_graph_buffers(env.world_count)
    for trainer in controller.trainers:
        trainer.reserve_buffers(env.world_count)
    controller.single_trainer.reserve_buffers(env.world_count)
    rollout_actors = tuple(_make_rollout_actor(controller, member, seed=seed) for member in range(2))
    phase_batches = tuple(
        tuple(_allocate_flash_sac_batch(controller.trainers[0]) for _ in range(total_updates)) for _phase in range(2)
    )
    pre_step_obs = wp.empty((env.world_count, env.obs_dim), dtype=wp.float32, device=controller.device)
    zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=controller.device)
    env_seed_counter = wp.array([int(seed)], dtype=wp.int32, device=controller.device)
    if hasattr(env, "use_reset_seed_counter"):
        env.use_reset_seed_counter(env_seed_counter)
    if hasattr(env, "use_command_seed_counter"):
        env.use_command_seed_counter(env_seed_counter)

    def collect() -> None:
        for _interaction in range(interactions):
            wp.copy(pre_step_obs, env.obs)
            member_actions = []
            for trainer, actor in zip(controller.trainers, rollout_actors, strict=True):
                exploration_seed = trainer.prepare_graph_exploration_seed()
                actions, _log_probs, _policy_out = actor.sample_reuse_seed_counter(
                    pre_step_obs,
                    seed_counter=exploration_seed,
                )
                member_actions.append(actions)
            actions = controller.route_split_actions(member_actions[0], member_actions[1])
            next_obs, rewards, dones = env.step(actions)
            replay.add_batch_graph(
                pre_step_obs,
                actions,
                rewards,
                getattr(env, "step_terminateds", dones),
                getattr(env, "step_next_obs", next_obs),
                truncateds=getattr(env, "step_truncateds", zero_truncateds),
            )

    def prepare(phase: int, *, single: bool) -> None:
        if single:
            rollout_actors[0].copy_from(controller.single_trainer.actor)
            rollout_actors[1].copy_from(controller.single_trainer.actor)
        else:
            for actor, trainer in zip(rollout_actors, controller.trainers, strict=True):
                actor.copy_from(trainer.actor)
        for update_index, batch in enumerate(phase_batches[phase]):
            sampled = replay.sample_graph_seed_counter(
                controller.trainers[0]._device_update_count,
                seed_offset=update_index + 101,
            )
            _copy_flash_sac_batch(batch, sampled)

    controller.population._seed_base.assign(np.asarray([int(seed), int(seed) + 1], dtype=np.int32))

    def update_population(phase: int) -> None:
        for update_index, batch in enumerate(phase_batches[phase]):
            if (start_gradient_update + update_index) % policy_frequency == 0:
                controller.population._fused_complete_update_operations(batch, read_stats=False)
            else:
                controller.population._fused_critic_only_update_operations(batch, read_stats=False)

    rollout_stream = wp.Stream(controller.device, priority=-1)
    update_stream = wp.Stream(controller.device)
    prepare_stream = wp.Stream(controller.device)
    rollout_graph = _capture_flash_stream_graph(rollout_stream, controller.device, collect)
    population_update_graphs = tuple(
        _capture_flash_stream_graph(
            update_stream,
            controller.device,
            lambda phase=phase: update_population(phase),
        )
        for phase in range(2)
    )

    def update_single(phase: int) -> None:
        for update_index, batch in enumerate(phase_batches[phase]):
            include_actor = (start_gradient_update + update_index) % policy_frequency == 0
            controller.single_trainer._graph_update_operations(
                batch,
                include_actor=include_actor,
                seed_base=int(seed),
            )

    single_update_graphs = tuple(
        _capture_flash_stream_graph(update_stream, controller.device, lambda phase=phase: update_single(phase))
        for phase in range(2)
    )
    population_prepare_graphs = tuple(
        _capture_flash_stream_graph(
            prepare_stream,
            controller.device,
            lambda phase=phase: prepare(phase, single=False),
        )
        for phase in range(2)
    )
    single_prepare_graphs = tuple(
        _capture_flash_stream_graph(
            prepare_stream,
            controller.device,
            lambda phase=phase: prepare(phase, single=True),
        )
        for phase in range(2)
    )
    wp.capture_launch(population_prepare_graphs[0], stream=prepare_stream)
    wp.synchronize_device(controller.device)
    return GraphFlashSACLRAutotuneOverlap(
        controller=controller,
        replay=replay,
        env=env,
        rollout_graph=rollout_graph,
        population_update_graphs=population_update_graphs,
        single_update_graphs=single_update_graphs,
        population_prepare_graphs=population_prepare_graphs,
        single_prepare_graphs=single_prepare_graphs,
        rollout_stream=rollout_stream,
        update_stream=update_stream,
        prepare_stream=prepare_stream,
        phase_batches=phase_batches,
        rollout_actors=rollout_actors,
        retained_arrays=(pre_step_obs, zero_truncateds, env_seed_counter),
        interactions_per_launch=interactions,
        updates_per_launch=total_updates,
    )


@dataclass
class GraphFlashSACLRAutotuneOverlap:
    """Captured split-rollout and shared-batch LR-search cadence."""

    controller: ControllerFlashSACLRAutotune
    replay: BufferReplayFlashSAC
    env: EnvFlashSAC
    rollout_graph: object | None
    population_update_graphs: tuple[object, object] | None
    single_update_graphs: tuple[object, object] | None
    population_prepare_graphs: tuple[object, object] | None
    single_prepare_graphs: tuple[object, object] | None
    rollout_stream: wp.Stream | None
    update_stream: wp.Stream | None
    prepare_stream: wp.Stream | None
    phase_batches: tuple[tuple[BatchSAC, ...], tuple[BatchSAC, ...]] | None
    rollout_actors: tuple[Any, Any] | None
    retained_arrays: tuple[wp.array[Any], ...]
    interactions_per_launch: int
    updates_per_launch: int
    phase: int = 0

    def synchronize(self) -> None:
        """Wait for rollout, learner, and preparation streams."""

        streams = (self.rollout_stream, self.update_stream, self.prepare_stream)
        if all(stream is not None for stream in streams):
            main_stream = wp.get_stream(self.controller.device)
            with wp.ScopedStream(main_stream, sync_enter=False, sync_exit=False):
                for stream in streams:
                    assert stream is not None
                    wp.wait_stream(stream)
        wp.synchronize_device(self.controller.device)

    def launch(self) -> None:
        """Overlap one rollout with P2 or converged-P1 learner work."""

        if (
            self.rollout_graph is None
            or self.population_update_graphs is None
            or self.single_update_graphs is None
            or self.population_prepare_graphs is None
            or self.single_prepare_graphs is None
            or self.rollout_stream is None
            or self.update_stream is None
            or self.prepare_stream is None
        ):
            raise RuntimeError("FlashSAC LR autotuning overlap graph is closed")
        phase = self.phase
        learner_stream = self.update_stream
        with wp.ScopedStream(learner_stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(self.prepare_stream)
        with wp.ScopedStream(self.rollout_stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(self.prepare_stream)
        if self.controller.converged:
            wp.capture_launch(self.single_update_graphs[phase], stream=learner_stream)
            self.controller.single_trainer._gradient_update_count += self.updates_per_launch
            self.controller.single_trainer._update_count += self.updates_per_launch
        else:
            wp.capture_launch(self.population_update_graphs[phase], stream=learner_stream)
            for trainer in self.controller.trainers:
                trainer._gradient_update_count += self.updates_per_launch
                trainer._update_count += self.updates_per_launch
        wp.capture_launch(self.rollout_graph, stream=self.rollout_stream)
        next_phase = 1 - phase
        with wp.ScopedStream(self.prepare_stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(learner_stream)
            wp.wait_stream(self.rollout_stream)
        prepare_graphs = self.single_prepare_graphs if self.controller.converged else self.population_prepare_graphs
        wp.capture_launch(prepare_graphs[next_phase], stream=self.prepare_stream)
        self.phase = next_phase
        self.replay.advance_graph_host_state(self.interactions_per_launch)
        if hasattr(self.env, "sim_time") and hasattr(self.env, "config"):
            self.env.sim_time += self.interactions_per_launch * float(self.env.config.frame_dt)

    def __enter__(self) -> GraphFlashSACLRAutotuneOverlap:
        """Return this captured overlap handle."""

        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        """Drain and close this captured overlap handle."""

        self.close()

    def close(self) -> None:
        """Drain streams and release captured graphs and retained arrays."""

        if self.rollout_graph is None:
            return
        self.synchronize()
        self.rollout_graph = None
        self.population_update_graphs = None
        self.single_update_graphs = None
        self.population_prepare_graphs = None
        self.single_prepare_graphs = None
        self.rollout_stream = None
        self.update_stream = None
        self.prepare_stream = None
        self.phase_batches = None
        self.rollout_actors = None
        self.retained_arrays = ()

    def __del__(self) -> None:
        self.close()
