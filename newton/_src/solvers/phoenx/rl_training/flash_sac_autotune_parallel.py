# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent-trainer parallel overlap experiment for FlashSAC LR tuning."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .cublas import release_cublas_workspace
from .flash_sac import (
    BufferReplayFlashSAC,
    EnvFlashSAC,
    TrainerFlashSAC,
    _allocate_flash_sac_batch,
    _capture_flash_stream_graph,
    _copy_flash_sac_batch,
)
from .flash_sac_networks import NetworkFlashSAC
from .sac import BatchSAC

if TYPE_CHECKING:
    from .flash_sac_autotune import ControllerFlashSACLRAutotune, ResultFlashSACLRAutotune


@wp.kernel
def _route_partitioned_actions_kernel(
    champion_actions: wp.array2d[wp.float32],
    challenger_actions: wp.array2d[wp.float32],
    champion_worlds: int,
    actions: wp.array2d[wp.float32],
):
    world, action = wp.tid()
    if world < champion_worlds:
        actions[world, action] = champion_actions[world, action]
    else:
        actions[world, action] = challenger_actions[world - champion_worlds, action]


@wp.func
def _challenger_action_requires_fallback(
    squared_difference: float,
    maximum_difference: float,
    action_dim: int,
    finite: wp.int32,
    rms_limit: float,
    max_limit: float,
):
    """Reject diffuse RMS drift and isolated unsafe action spikes."""

    rms_difference = wp.sqrt(squared_difference / float(action_dim))
    return finite == wp.int32(0) or rms_difference > rms_limit or maximum_difference > max_limit


@wp.kernel
def _guard_challenger_actions_kernel(
    champion_actions: wp.array2d[wp.float32],
    challenger_actions: wp.array2d[wp.float32],
    rms_limit: float,
    max_limit: float,
    interaction: int,
    guarded_actions: wp.array2d[wp.float32],
    fallbacks: wp.array2d[wp.int32],
):
    world = wp.tid()
    squared_difference = float(0.0)
    maximum_difference = float(0.0)
    finite = wp.int32(1)
    action_dim = challenger_actions.shape[1]
    for action in range(action_dim):
        champion = champion_actions[world, action]
        challenger = challenger_actions[world, action]
        if not wp.isfinite(champion) or not wp.isfinite(challenger):
            finite = wp.int32(0)
        difference = wp.abs(challenger - champion)
        squared_difference += difference * difference
        maximum_difference = wp.max(maximum_difference, difference)
    fallback = _challenger_action_requires_fallback(
        squared_difference,
        maximum_difference,
        action_dim,
        finite,
        rms_limit,
        max_limit,
    )
    fallbacks[interaction, world] = wp.int32(fallback)
    for action in range(action_dim):
        if fallback:
            guarded_actions[world, action] = champion_actions[world, action]
        else:
            guarded_actions[world, action] = challenger_actions[world, action]


def _clone_trainer(source: TrainerFlashSAC) -> TrainerFlashSAC:
    trainer = TrainerFlashSAC(
        obs_dim=source.obs_dim,
        action_dim=source.action_dim,
        config=copy.deepcopy(source.config),
        device=source.device,
        seed=source.seed,
    )
    trainer.copy_training_state_from(source)
    return trainer


def _make_rollout_actor(trainer: TrainerFlashSAC, world_count: int):
    source = trainer.actor.net
    actor = copy.copy(trainer.actor)
    actor.net = NetworkFlashSAC(
        input_dim=source.input_dim,
        hidden_dim=source.hidden_dim,
        num_blocks=source.num_blocks,
        output_dim=source.output_dim,
        actor_heads=True,
        device=trainer.device,
        seed=trainer.seed,
        contraction_dtype="float32",
    )
    actor.log_std = wp.clone(trainer.actor.log_std)
    actor._sample_reuse_capacity = 0
    actor._sample_reuse_actions = None
    actor._sample_reuse_log_probs = None
    actor._sample_reuse_eps = None
    actor.copy_from(trainer.actor)
    actor.reserve_reuse_buffers(world_count)
    return actor


def capture_lr_autotune_parallel_overlap(
    controller: ControllerFlashSACLRAutotune,
    env: EnvFlashSAC,
    replay: BufferReplayFlashSAC,
    *,
    updates_per_step: int,
    interactions_per_launch: int,
    seed: int,
) -> GraphFlashSACLRAutotuneParallelOverlap:
    """Capture two exact scalar learners concurrently on one shared batch stream."""

    updates = int(updates_per_step)
    interactions = int(interactions_per_launch)
    if env.device != controller.device or replay.device != controller.device:
        raise ValueError("environment, replay, and LR autotuner must use the same device")
    if int(env.world_count) != controller.rollout_world_count:
        raise ValueError("environment world count does not match LR autotuner setup")
    if updates <= 0 or interactions <= 0 or interactions % 2 != 0:
        raise ValueError("parallel overlap requires positive updates and even interactions")
    if not replay.can_sample():
        raise RuntimeError("warm shared replay before capturing LR autotuning overlap")
    total_updates = updates * interactions
    controller.configure_policy_frequency_family(total_updates, allow_search=True)
    policy_frequencies = controller.policy_frequency_choices

    trainers = tuple(_clone_trainer(trainer) for trainer in controller.trainers)
    for trainer in trainers:
        trainer.reserve_buffers(env.world_count)
        trainer.reserve_update_buffers(replay.batch_size)
    single_trainer = _clone_trainer(controller.single_trainer)
    single_trainer.reserve_buffers(env.world_count)
    single_trainer.reserve_update_buffers(replay.batch_size)
    for member, trainer in enumerate(trainers):
        trainer._device_interaction_seed.assign(np.asarray([int(seed) + member], dtype=np.int32))
    single_trainer._device_interaction_seed.assign(np.asarray([int(seed)], dtype=np.int32))
    challenger_worlds = int(controller.challenger_worlds)
    if challenger_worlds == 0:
        rollout_actors = (_make_rollout_actor(trainers[0], env.world_count),)
        guard_actor = None
    else:
        rollout_rows = (controller.champion_worlds, challenger_worlds)
        rollout_actors = tuple(
            _make_rollout_actor(trainer, rows) for trainer, rows in zip(trainers, rollout_rows, strict=True)
        )
        guard_actor = _make_rollout_actor(trainers[0], challenger_worlds)
    phase_batches = tuple(
        tuple(_allocate_flash_sac_batch(trainers[0]) for _ in range(total_updates)) for _phase in range(2)
    )
    pre_step_obs = wp.empty((env.world_count, env.obs_dim), dtype=wp.float32, device=controller.device)
    champion_obs = pre_step_obs[: controller.champion_worlds]
    challenger_obs = pre_step_obs[controller.champion_worlds :] if challenger_worlds else None
    zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=controller.device)
    env_seed_counter = wp.array([int(seed)], dtype=wp.int32, device=controller.device)
    guard_rows = max(1, challenger_worlds)
    guarded_challenger_actions = wp.empty((guard_rows, env.action_dim), dtype=wp.float32, device=controller.device)
    challenger_fallbacks = wp.zeros((interactions, guard_rows), dtype=wp.int32, device=controller.device)
    guard_rms_limit = float(controller.config.challenger_action_rms_limit)
    guard_max_limit = float(controller.config.challenger_action_max_limit)
    if hasattr(env, "use_reset_seed_counter"):
        env.use_reset_seed_counter(env_seed_counter)
    if hasattr(env, "use_command_seed_counter"):
        env.use_command_seed_counter(env_seed_counter)

    single_rollout_actor = _make_rollout_actor(single_trainer, env.world_count)
    single_rollout_condition = wp.zeros(1, dtype=wp.int32, device=controller.device)
    routed_actions = controller._routed_actions

    def collect() -> None:
        for _interaction in range(interactions):
            wp.copy(pre_step_obs, env.obs)

            def sample_single() -> None:
                exploration_seed = single_trainer.prepare_graph_exploration_seed()
                single_actions, _log_probs, _policy_out = single_rollout_actor.sample_reuse_seed_counter(
                    pre_step_obs, seed_counter=exploration_seed
                )
                wp.copy(routed_actions, single_actions)

            def sample_population(interaction: int = _interaction) -> None:
                champion_seed = trainers[0].prepare_graph_exploration_seed()
                champion_actions, _log_probs, _policy_out = rollout_actors[0].sample_reuse_seed_counter(
                    champion_obs, seed_counter=champion_seed
                )
                if challenger_worlds:
                    assert challenger_obs is not None and guard_actor is not None
                    challenger_seed = trainers[1].prepare_graph_exploration_seed()
                    challenger_actions, _log_probs, _policy_out = rollout_actors[1].sample_reuse_seed_counter(
                        challenger_obs, seed_counter=challenger_seed
                    )
                    guard_actions, _log_probs, _policy_out = guard_actor.sample_reuse_seed_counter(
                        challenger_obs, seed_counter=challenger_seed
                    )
                    wp.launch(
                        _guard_challenger_actions_kernel,
                        dim=challenger_worlds,
                        inputs=[
                            guard_actions,
                            challenger_actions,
                            guard_rms_limit,
                            guard_max_limit,
                            interaction,
                        ],
                        outputs=[guarded_challenger_actions, challenger_fallbacks],
                        device=controller.device,
                    )
                    wp.launch(
                        _route_partitioned_actions_kernel,
                        dim=routed_actions.shape,
                        inputs=[champion_actions, guarded_challenger_actions, controller.champion_worlds],
                        outputs=[routed_actions],
                        device=controller.device,
                    )
                else:
                    wp.copy(routed_actions, champion_actions)

            wp.capture_if(
                single_rollout_condition,
                on_true=sample_single,
                on_false=sample_population,
            )
            next_obs, rewards, dones = env.step(routed_actions)
            replay.add_batch_graph(
                pre_step_obs,
                routed_actions,
                rewards,
                getattr(env, "step_terminateds", dones),
                getattr(env, "step_next_obs", next_obs),
                truncateds=getattr(env, "step_truncateds", zero_truncateds),
            )

    def prepare(phase: int, *, single: bool) -> None:
        if single:
            single_rollout_actor.copy_from(single_trainer.actor)
            sample_counter = single_trainer._device_update_count
        else:
            for actor, trainer in zip(rollout_actors, trainers[: len(rollout_actors)], strict=True):
                actor.copy_from(trainer.actor)
            if guard_actor is not None:
                guard_actor.copy_from(trainers[0].actor)
            sample_counter = trainers[0]._device_update_count
        for update_index, batch in enumerate(phase_batches[phase]):
            sampled = replay.sample_graph_seed_counter(sample_counter, seed_offset=update_index + 101)
            _copy_flash_sac_batch(batch, sampled)

    rollout_stream = wp.Stream(controller.device, priority=-1)
    learner_streams = (wp.Stream(controller.device), wp.Stream(controller.device))
    single_stream = wp.Stream(controller.device)
    prepare_stream = wp.Stream(controller.device)
    rollout_graph = _capture_flash_stream_graph(rollout_stream, controller.device, collect)
    if len(policy_frequencies) > 1 and not wp.is_conditional_graph_supported():
        raise RuntimeError("policy-frequency autotuning requires CUDA graph conditional-node support")
    for trainer, frequency in zip(trainers, controller.member_policy_frequencies, strict=True):
        trainer.set_pbt_policy_frequency(int(frequency))
    single_trainer.set_pbt_policy_frequency(controller.best_policy_frequency)

    def update_member(member: int, phase: int) -> None:
        trainer = trainers[member]
        for batch in phase_batches[phase]:
            trainer._graph_update_operations(
                batch,
                include_actor=False,
                seed_base=int(seed) + member,
                conditional_actor=True,
            )

    def update_single(phase: int) -> None:
        for batch in phase_batches[phase]:
            single_trainer._graph_update_operations(
                batch,
                include_actor=False,
                seed_base=int(seed),
                conditional_actor=True,
            )

    update_graphs = tuple(
        tuple(
            _capture_flash_stream_graph(
                learner_streams[member],
                controller.device,
                lambda member=member, phase=phase: update_member(member, phase),
            )
            for phase in range(2)
        )
        for member in range(2)
    )
    single_graphs = tuple(
        _capture_flash_stream_graph(single_stream, controller.device, lambda phase=phase: update_single(phase))
        for phase in range(2)
    )
    population_prepare_graphs = tuple(
        _capture_flash_stream_graph(prepare_stream, controller.device, lambda phase=phase: prepare(phase, single=False))
        for phase in range(2)
    )
    single_prepare_graphs = tuple(
        _capture_flash_stream_graph(prepare_stream, controller.device, lambda phase=phase: prepare(phase, single=True))
        for phase in range(2)
    )
    wp.capture_launch(population_prepare_graphs[0], stream=prepare_stream)
    wp.synchronize_device(controller.device)
    return GraphFlashSACLRAutotuneParallelOverlap(
        controller=controller,
        trainers=trainers,
        single_trainer=single_trainer,
        replay=replay,
        env=env,
        rollout_graph=rollout_graph,
        update_graphs=update_graphs,
        single_graphs=single_graphs,
        population_prepare_graphs=population_prepare_graphs,
        single_prepare_graphs=single_prepare_graphs,
        rollout_stream=rollout_stream,
        learner_streams=learner_streams,
        single_stream=single_stream,
        prepare_stream=prepare_stream,
        phase_batches=phase_batches,
        rollout_actors=rollout_actors + ((guard_actor,) if guard_actor is not None else ()),
        single_rollout_actor=single_rollout_actor,
        single_rollout_condition=single_rollout_condition,
        challenger_fallbacks=challenger_fallbacks,
        challenger_world_count=challenger_worlds,
        retained_arrays=(pre_step_obs, zero_truncateds, env_seed_counter, guarded_challenger_actions),
        interactions_per_launch=interactions,
        updates_per_launch=total_updates,
    )


@dataclass
class GraphFlashSACLRAutotuneParallelOverlap:
    """Captured exact scalar P2 learner overlap with shared phase batches."""

    controller: ControllerFlashSACLRAutotune
    trainers: tuple[TrainerFlashSAC, TrainerFlashSAC]
    single_trainer: TrainerFlashSAC
    replay: BufferReplayFlashSAC
    env: EnvFlashSAC
    rollout_graph: object | None
    update_graphs: tuple[tuple[object, object], tuple[object, object]] | None
    single_graphs: tuple[object, object] | None
    population_prepare_graphs: tuple[object, object] | None
    single_prepare_graphs: tuple[object, object] | None
    rollout_stream: wp.Stream | None
    learner_streams: tuple[wp.Stream, wp.Stream] | None
    single_stream: wp.Stream | None
    prepare_stream: wp.Stream | None
    phase_batches: tuple[tuple[BatchSAC, ...], tuple[BatchSAC, ...]] | None
    rollout_actors: tuple[Any, ...] | None
    single_rollout_actor: Any | None
    single_rollout_condition: wp.array[wp.int32] | None
    challenger_fallbacks: wp.array2d[wp.int32]
    challenger_world_count: int
    retained_arrays: tuple[wp.array[Any], ...]
    interactions_per_launch: int
    updates_per_launch: int
    phase: int = 0

    def synchronize(self) -> None:
        """Drain every learner, rollout, and preparation stream."""

        streams = (self.rollout_stream, self.single_stream, self.prepare_stream, *(self.learner_streams or ()))
        if all(stream is not None for stream in streams):
            main_stream = wp.get_stream(self.controller.device)
            with wp.ScopedStream(main_stream, sync_enter=False, sync_exit=False):
                for stream in streams:
                    assert stream is not None
                    wp.wait_stream(stream)
        wp.synchronize_device(self.controller.device)

    def challenger_fallback_fraction(self) -> float:
        """Return the fraction of challenger actions replaced by champion actions."""

        if self.challenger_world_count == 0:
            return 0.0
        self.synchronize()
        return float(self.challenger_fallbacks.numpy()[:, : self.challenger_world_count].mean())

    def sync_controller_state(self) -> None:
        """Copy independent learner state into the controller without rebinding arrays."""

        self.synchronize()
        if self.controller.converged:
            self.controller.single_trainer.copy_training_state_from(self.single_trainer)
        else:
            for destination, source in zip(self.controller.trainers, self.trainers, strict=True):
                destination.copy_training_state_from(source)

    def evaluation_trainers(self) -> tuple[TrainerFlashSAC, TrainerFlashSAC]:
        """Return paired learners or the converged learner twice."""

        if self.controller.converged:
            return (self.single_trainer, self.single_trainer)
        return self.trainers

    def reopen_search(self) -> None:
        """Restart paired graphs from the current converged learner."""

        if not self.controller.converged:
            raise RuntimeError("FlashSAC LR search is already active")
        self.sync_controller_state()
        self.controller.reopen_search()
        self.sync_from_controller_state()

    def start_single_policy_bootstrap(self) -> None:
        """Switch to the pre-captured champion-only training path."""

        self.sync_controller_state()
        self.controller.start_single_policy_bootstrap()
        self.sync_from_controller_state()

    def evaluate_paired(self, *args: Any, **kwargs: Any) -> ResultFlashSACLRAutotune:
        """Join learners, evaluate through the controller, and mirror the decision back."""

        self.sync_controller_state()
        result = self.controller.evaluate_paired(*args, **kwargs)
        self.sync_from_controller_state()
        return result

    def sync_from_controller_state(self) -> None:
        """Copy controller decisions into graph-owned fixed-address learners."""

        self.synchronize()
        for destination, source in zip(self.trainers, self.controller.trainers, strict=True):
            destination.copy_training_state_from(source)
        self.single_trainer.copy_training_state_from(self.controller.single_trainer)
        if self.single_rollout_condition is None:
            raise RuntimeError("parallel FlashSAC LR autotuning graph is closed")
        self.single_rollout_condition.assign(np.asarray([int(self.controller.converged)], dtype=np.int32))

    def launch(self) -> None:
        """Overlap rollout with two independent learners or converged P1."""

        if (
            self.rollout_graph is None
            or self.single_rollout_actor is None
            or self.single_rollout_condition is None
            or self.update_graphs is None
            or self.single_graphs is None
            or self.population_prepare_graphs is None
            or self.single_prepare_graphs is None
            or self.rollout_stream is None
            or self.learner_streams is None
            or self.single_stream is None
            or self.prepare_stream is None
        ):
            raise RuntimeError("parallel FlashSAC LR autotuning graph is closed")
        phase = self.phase
        active_streams = (self.single_stream,) if self.controller.converged else self.learner_streams
        for stream in active_streams:
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(self.prepare_stream)
        with wp.ScopedStream(self.rollout_stream, sync_enter=False, sync_exit=False):
            wp.wait_stream(self.prepare_stream)
        if self.controller.converged:
            wp.capture_launch(self.single_graphs[phase], stream=self.single_stream)
            self.single_trainer._gradient_update_count += self.updates_per_launch
            self.single_trainer._update_count += self.updates_per_launch
        else:
            for member, stream in enumerate(self.learner_streams):
                wp.capture_launch(self.update_graphs[member][phase], stream=stream)
            for trainer in self.trainers:
                trainer._gradient_update_count += self.updates_per_launch
                trainer._update_count += self.updates_per_launch
        wp.capture_launch(self.rollout_graph, stream=self.rollout_stream)
        next_phase = 1 - phase
        with wp.ScopedStream(self.prepare_stream, sync_enter=False, sync_exit=False):
            for stream in active_streams:
                wp.wait_stream(stream)
            wp.wait_stream(self.rollout_stream)
        prepare_graphs = self.single_prepare_graphs if self.controller.converged else self.population_prepare_graphs
        wp.capture_launch(prepare_graphs[next_phase], stream=self.prepare_stream)
        self.phase = next_phase
        self.replay.advance_graph_host_state(self.interactions_per_launch)
        if hasattr(self.env, "sim_time") and hasattr(self.env, "config"):
            self.env.sim_time += self.interactions_per_launch * float(self.env.config.frame_dt)

    def close(self) -> None:
        """Drain work and release graphs before their fixed-address owners."""

        if self.rollout_graph is None:
            return
        self.synchronize()
        streams = (self.rollout_stream, self.single_stream, self.prepare_stream, *(self.learner_streams or ()))
        self.rollout_graph = None
        self.update_graphs = None
        self.single_graphs = None
        self.population_prepare_graphs = None
        self.single_prepare_graphs = None
        self.rollout_stream = None
        for stream in streams:
            if stream is not None:
                release_cublas_workspace(self.controller.device, stream)
        self.learner_streams = None
        self.single_stream = None
        self.prepare_stream = None
        self.phase_batches = None
        self.rollout_actors = None
        self.single_rollout_actor = None
        self.single_rollout_condition = None
        self.retained_arrays = ()

    def __del__(self) -> None:
        self.close()
