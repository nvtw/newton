# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .flash_sac_networks import EnsembleNetworkFlashSAC, NetworkFlashSAC
from .kernels import (
    TANH_EPS,
    _clip,
    _normal_log_prob,
    population_copy_float_1d_kernel,
    population_copy_float_2d_kernel,
    population_copy_int_1d_kernel,
    population_copy_int_2d_kernel,
    population_pair_indices_kernel,
    population_repeat_pair_kernel,
    soft_update_2d_kernel,
    soft_update_3d_kernel,
)
from .optim import AdamPopulation, AMPStatePopulation
from .sac import BatchSAC, StatsSACUpdate

if TYPE_CHECKING:
    from .flash_sac import TrainerFlashSAC


@wp.kernel
def _population_prepare_update_kernel(
    gradient_update_count: wp.array2d[wp.int32],
    update_count: wp.array2d[wp.int32],
    seed_base: wp.array[wp.int32],
    warmup_steps: wp.int32,
    decay_steps: wp.int32,
    peak_lr: wp.float32,
    end_lr: wp.float32,
    actor_base_lr: wp.float32,
    critic_base_lr: wp.float32,
    alpha_base_lr: wp.float32,
    update_seed: wp.array2d[wp.int32],
    actor_lr_scale: wp.array[wp.float32],
    critic_lr_scale: wp.array[wp.float32],
    alpha_lr_scale: wp.array[wp.float32],
):
    member = wp.tid()
    step = gradient_update_count[member, 0]
    seed64 = wp.int64(seed_base[member]) + wp.int64(update_count[member, 0]) * wp.int64(9973)
    update_seed[member, 0] = wp.int32(seed64 % wp.int64(2147483647))
    lr = peak_lr
    if warmup_steps > 0 and step < warmup_steps:
        lr = peak_lr * wp.float32(step + wp.int32(1)) / wp.float32(warmup_steps)
    else:
        decay_step = wp.min(wp.max(step - warmup_steps, wp.int32(0)), decay_steps)
        progress = wp.float32(decay_step) / wp.float32(decay_steps)
        cosine = wp.float32(0.5) * (wp.float32(1.0) + wp.cos(wp.pi * progress))
        lr = end_lr + (peak_lr - end_lr) * cosine
    actor_lr_scale[member] = lr / actor_base_lr
    critic_lr_scale[member * wp.int32(2)] = lr / critic_base_lr
    critic_lr_scale[member * wp.int32(2) + wp.int32(1)] = lr / critic_base_lr
    alpha_lr_scale[member] = lr / alpha_base_lr


@wp.kernel
def _population_increment_counters_kernel(
    gradient_update_count: wp.array2d[wp.int32],
    update_count: wp.array2d[wp.int32],
):
    member = wp.tid()
    gradient_update_count[member, 0] += wp.int32(1)
    update_count[member, 0] += wp.int32(1)


@wp.kernel
def _concat_population_actor_observations_kernel(
    obs: wp.array2d[wp.float32],
    next_obs: wp.array2d[wp.float32],
    batch_size: wp.int32,
    out: wp.array2d[wp.float32],
):
    row, column = wp.tid()
    if row < batch_size:
        out[row, column] = obs[row, column]
    else:
        out[row, column] = next_obs[row - batch_size, column]


@wp.kernel
def _population_fill_eps_kernel(
    seed: wp.array2d[wp.int32],
    seed_offset: wp.int32,
    eps: wp.array3d[wp.float32],
):
    member, row, column = wp.tid()
    flat = row * eps.shape[2] + column
    member_seed = wp.int32((wp.int64(seed[member, 0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    rng = wp.rand_init(member_seed, flat)
    eps[member, row, column] = wp.randn(rng)


@wp.kernel
def _population_sample_actions_kernel(
    policy_out: wp.array3d[wp.float32],
    row_offset: wp.int32,
    eps: wp.array3d[wp.float32],
    action_dim: wp.int32,
    log_std_min: wp.float32,
    log_std_max: wp.float32,
    actions: wp.array3d[wp.float32],
    log_probs: wp.array2d[wp.float32],
):
    member, row = wp.tid()
    total = wp.float32(0.0)
    for action in range(action_dim):
        mean = policy_out[member, row + row_offset, action]
        raw_log_std = policy_out[member, row + row_offset, action_dim + action]
        log_std = _clip(raw_log_std, log_std_min, log_std_max)
        pre_tanh = mean + wp.exp(log_std) * eps[member, row, action]
        value = wp.tanh(pre_tanh)
        total += _normal_log_prob(pre_tanh, mean, log_std)
        total -= wp.log(wp.float32(1.0) - value * value + wp.float32(TANH_EPS))
        actions[member, row, action] = value
    log_probs[member, row] = total


@wp.kernel
def _population_actor_q_input_kernel(
    obs: wp.array2d[wp.float32],
    actions: wp.array3d[wp.float32],
    obs_dim: wp.int32,
    batch_size: wp.int32,
    out: wp.array3d[wp.float32],
):
    critic, row, column = wp.tid()
    member = critic / wp.int32(2)
    source_row = row % batch_size
    if column < obs_dim:
        out[critic, row, column] = obs[source_row, column]
    else:
        out[critic, row, column] = actions[member, source_row, column - obs_dim]


@wp.kernel
def _population_training_q_input_kernel(
    obs: wp.array2d[wp.float32],
    replay_actions: wp.array2d[wp.float32],
    next_obs: wp.array2d[wp.float32],
    next_actions: wp.array3d[wp.float32],
    obs_dim: wp.int32,
    batch_size: wp.int32,
    out: wp.array3d[wp.float32],
):
    critic, row, column = wp.tid()
    member = critic / wp.int32(2)
    if row < batch_size:
        if column < obs_dim:
            out[critic, row, column] = obs[row, column]
        else:
            out[critic, row, column] = replay_actions[row, column - obs_dim]
    else:
        next_row = row - batch_size
        if column < obs_dim:
            out[critic, row, column] = next_obs[next_row, column]
        else:
            out[critic, row, column] = next_actions[member, next_row, column - obs_dim]


@wp.kernel
def _population_distributional_q_value_kernel(
    logits: wp.array3d[wp.float32],
    batch_size: wp.int32,
    num_atoms: wp.int32,
    v_min: wp.float32,
    v_max: wp.float32,
    values: wp.array3d[wp.float32],
):
    critic, row = wp.tid()
    max_logit = logits[critic, row, 0]
    for atom in range(1, num_atoms):
        max_logit = wp.max(max_logit, logits[critic, row, atom])
    normalizer = wp.float32(0.0)
    weighted_support = wp.float32(0.0)
    delta = (v_max - v_min) / wp.float32(num_atoms - wp.int32(1))
    for atom in range(num_atoms):
        probability = wp.exp(logits[critic, row, atom] - max_logit)
        normalizer += probability
        weighted_support += probability * (v_min + wp.float32(atom) * delta)
    values[critic, row, 0] = weighted_support / normalizer


@wp.kernel
def _population_actor_q_backward_kernel(
    logits: wp.array3d[wp.float32],
    q_values: wp.array3d[wp.float32],
    loss_scale: wp.array[wp.float32],
    batch_size: wp.int32,
    num_atoms: wp.int32,
    v_min: wp.float32,
    v_max: wp.float32,
    average_critics: wp.bool,
    logits_grad: wp.array3d[wp.float32],
):
    critic, row, atom = wp.tid()
    if row >= batch_size:
        logits_grad[critic, row, atom] = wp.float32(0.0)
        return
    member = critic / wp.int32(2)
    first = member * wp.int32(2)
    second = first + wp.int32(1)
    max_logit = logits[critic, row, 0]
    for source_atom in range(1, num_atoms):
        max_logit = wp.max(max_logit, logits[critic, row, source_atom])
    normalizer = wp.float32(0.0)
    for source_atom in range(num_atoms):
        normalizer += wp.exp(logits[critic, row, source_atom] - max_logit)
    probability = wp.exp(logits[critic, row, atom] - max_logit) / normalizer
    delta = (v_max - v_min) / wp.float32(num_atoms - wp.int32(1))
    support = v_min + wp.float32(atom) * delta
    multiplier = wp.float32(0.0)
    if average_critics:
        multiplier = -wp.float32(0.5)
    elif q_values[first, row, 0] <= q_values[second, row, 0]:
        if critic == first:
            multiplier = -wp.float32(1.0)
    elif critic == second:
        multiplier = -wp.float32(1.0)
    logits_grad[critic, row, atom] = (
        loss_scale[member] * multiplier * probability * (support - q_values[critic, row, 0]) / wp.float32(batch_size)
    )


@wp.kernel
def _population_actor_policy_backward_kernel(
    policy_out: wp.array3d[wp.float32],
    eps: wp.array3d[wp.float32],
    q_input_grad: wp.array3d[wp.float32],
    loss_scale: wp.array[wp.float32],
    alpha: wp.array2d[wp.float32],
    obs_dim: wp.int32,
    action_dim: wp.int32,
    batch_size: wp.int32,
    log_std_min: wp.float32,
    log_std_max: wp.float32,
    policy_out_grad: wp.array3d[wp.float32],
):
    member, row = wp.tid()
    if row >= batch_size:
        for output in range(action_dim * wp.int32(2)):
            policy_out_grad[member, row, output] = wp.float32(0.0)
        return
    first = member * wp.int32(2)
    second = first + wp.int32(1)
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    for action in range(action_dim):
        mean = policy_out[member, row, action]
        raw_log_std = policy_out[member, row, action_dim + action]
        log_std = _clip(raw_log_std, log_std_min, log_std_max)
        std_eps = wp.exp(log_std) * eps[member, row, action]
        value = wp.tanh(mean + std_eps)
        action_grad = q_input_grad[first, row, obs_dim + action] + q_input_grad[second, row, obs_dim + action]
        correction_grad = (
            wp.float32(2.0)
            * value
            * (wp.float32(1.0) - value * value)
            / (wp.float32(1.0) - value * value + wp.float32(TANH_EPS))
        )
        scaled_alpha = loss_scale[member] * alpha[member, 0]
        pre_grad = action_grad * (wp.float32(1.0) - value * value)
        pre_grad += scaled_alpha * inv_batch * correction_grad
        policy_out_grad[member, row, action] = pre_grad
        log_std_grad = wp.float32(0.0)
        if raw_log_std >= log_std_min and raw_log_std <= log_std_max:
            log_std_grad = pre_grad * std_eps - scaled_alpha * inv_batch
        policy_out_grad[member, row, action_dim + action] = log_std_grad


@wp.kernel
def _population_actor_loss_kernel(
    q_values: wp.array3d[wp.float32],
    log_probs: wp.array2d[wp.float32],
    alpha: wp.array2d[wp.float32],
    batch_size: wp.int32,
    average_critics: wp.bool,
    loss: wp.array2d[wp.float32],
):
    member = wp.tid()
    total = wp.float32(0.0)
    first = member * wp.int32(2)
    second = first + wp.int32(1)
    for row in range(batch_size):
        q = wp.min(q_values[first, row, 0], q_values[second, row, 0])
        if average_critics:
            q = wp.float32(0.5) * (q_values[first, row, 0] + q_values[second, row, 0])
        total += alpha[member, 0] * log_probs[member, row] - q
    loss[member, 0] = total / wp.float32(batch_size)


@wp.kernel
def _population_alpha_loss_kernel(
    log_probs: wp.array2d[wp.float32],
    log_alpha: wp.array2d[wp.float32],
    batch_size: wp.int32,
    target_entropy: wp.float32,
    loss: wp.array2d[wp.float32],
    log_alpha_grad: wp.array2d[wp.float32],
):
    member = wp.tid()
    entropy_sum = wp.float32(0.0)
    for row in range(batch_size):
        entropy_sum -= log_probs[member, row]
    alpha = wp.exp(log_alpha[member, 0])
    value = alpha * (entropy_sum / wp.float32(batch_size) - target_entropy)
    loss[member, 0] = value
    log_alpha_grad[member, 0] = value


@wp.kernel
def _population_refresh_alpha_kernel(log_alpha: wp.array2d[wp.float32], alpha: wp.array2d[wp.float32]):
    member = wp.tid()
    alpha[member, 0] = wp.exp(log_alpha[member, 0])


@wp.kernel
def _population_distributional_projection_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    target_logits: wp.array3d[wp.float32],
    next_log_probs: wp.array2d[wp.float32],
    alpha: wp.array2d[wp.float32],
    gamma: wp.float32,
    batch_size: wp.int32,
    num_atoms: wp.int32,
    v_min: wp.float32,
    v_max: wp.float32,
    min_target: wp.bool,
    targets: wp.array3d[wp.float32],
):
    member, row, destination_atom = wp.tid()
    first = member * wp.int32(2)
    second = first + wp.int32(1)
    next_row = row + batch_size
    max1 = target_logits[first, next_row, 0]
    max2 = target_logits[second, next_row, 0]
    for atom in range(1, num_atoms):
        max1 = wp.max(max1, target_logits[first, next_row, atom])
        max2 = wp.max(max2, target_logits[second, next_row, atom])
    normalizer1 = wp.float32(0.0)
    normalizer2 = wp.float32(0.0)
    q1 = wp.float32(0.0)
    q2 = wp.float32(0.0)
    delta = (v_max - v_min) / wp.float32(num_atoms - wp.int32(1))
    for atom in range(num_atoms):
        probability1 = wp.exp(target_logits[first, next_row, atom] - max1)
        probability2 = wp.exp(target_logits[second, next_row, atom] - max2)
        support = v_min + wp.float32(atom) * delta
        normalizer1 += probability1
        normalizer2 += probability2
        q1 += probability1 * support
        q2 += probability2 * support
    use_first = q1 / normalizer1 <= q2 / normalizer2
    projected1 = wp.float32(0.0)
    projected2 = wp.float32(0.0)
    for source_atom in range(num_atoms):
        support = v_min + wp.float32(source_atom) * delta
        target_value = rewards[row] + gamma * (wp.float32(1.0) - dones[row]) * (
            support - alpha[member, 0] * next_log_probs[member, row]
        )
        target_value = wp.min(wp.max(target_value, v_min), v_max)
        position = (target_value - v_min) / delta
        lower = wp.int32(wp.floor(position))
        upper = wp.min(lower + wp.int32(1), num_atoms - wp.int32(1))
        upper_weight = position - wp.float32(lower)
        weight = wp.float32(0.0)
        if destination_atom == lower:
            weight += wp.float32(1.0) - upper_weight
        if destination_atom == upper and upper != lower:
            weight += upper_weight
        probability1 = wp.exp(target_logits[first, next_row, source_atom] - max1) / normalizer1
        probability2 = wp.exp(target_logits[second, next_row, source_atom] - max2) / normalizer2
        if min_target:
            probability = probability1 if use_first else probability2
            projected1 += probability * weight
            projected2 += probability * weight
        else:
            projected1 += probability1 * weight
            projected2 += probability2 * weight
    targets[first, row, destination_atom] = projected1
    targets[second, row, destination_atom] = projected2


@wp.kernel
def _population_critic_loss_backward_kernel(
    logits: wp.array3d[wp.float32],
    targets: wp.array3d[wp.float32],
    loss_scale: wp.array[wp.float32],
    batch_size: wp.int32,
    num_atoms: wp.int32,
    logits_grad: wp.array3d[wp.float32],
):
    critic, row, atom = wp.tid()
    if row >= batch_size:
        logits_grad[critic, row, atom] = wp.float32(0.0)
        return
    max_logit = logits[critic, row, 0]
    for source_atom in range(1, num_atoms):
        max_logit = wp.max(max_logit, logits[critic, row, source_atom])
    normalizer = wp.float32(0.0)
    for source_atom in range(num_atoms):
        normalizer += wp.exp(logits[critic, row, source_atom] - max_logit)
    probability = wp.exp(logits[critic, row, atom] - max_logit) / normalizer
    member = critic / wp.int32(2)
    logits_grad[critic, row, atom] = (
        loss_scale[member] * (probability - targets[critic, row, atom]) / wp.float32(batch_size)
    )


@wp.kernel
def _population_critic_loss_kernel(
    logits: wp.array3d[wp.float32],
    targets: wp.array3d[wp.float32],
    batch_size: wp.int32,
    num_atoms: wp.int32,
    loss: wp.array2d[wp.float32],
):
    member = wp.tid()
    total = wp.float32(0.0)
    for critic_offset in range(2):
        critic = member * wp.int32(2) + critic_offset
        for row in range(batch_size):
            max_logit = logits[critic, row, 0]
            for atom in range(1, num_atoms):
                max_logit = wp.max(max_logit, logits[critic, row, atom])
            normalizer = wp.float32(0.0)
            for atom in range(num_atoms):
                normalizer += wp.exp(logits[critic, row, atom] - max_logit)
            log_normalizer = wp.log(normalizer)
            for atom in range(num_atoms):
                total -= targets[critic, row, atom] * (logits[critic, row, atom] - max_logit - log_normalizer)
    loss[member, 0] = total / wp.float32(batch_size)


@wp.kernel
def _population_unscale_2d_kernel(
    values: wp.array2d[wp.float32],
    scale: wp.array[wp.float32],
    members_per_scale: wp.int32,
    found_inf: wp.array[wp.int32],
):
    member, column = wp.tid()
    scale_member = member / members_per_scale
    value = values[member, column] / scale[scale_member]
    if not wp.isfinite(value):
        wp.atomic_max(found_inf, scale_member, wp.int32(1))
    values[member, column] = value


@wp.kernel
def _population_unscale_3d_kernel(
    values: wp.array3d[wp.float32],
    scale: wp.array[wp.float32],
    members_per_scale: wp.int32,
    found_inf: wp.array[wp.int32],
):
    member, row, column = wp.tid()
    scale_member = member / members_per_scale
    value = values[member, row, column] / scale[scale_member]
    if not wp.isfinite(value):
        wp.atomic_max(found_inf, scale_member, wp.int32(1))
    values[member, row, column] = value


@wp.kernel
def _population_reduce_found_inf_kernel(
    flags: wp.array2d[wp.int32],
    found_inf: wp.array[wp.int32],
):
    member = wp.tid()
    value = wp.int32(0)
    for group in range(flags.shape[0]):
        value = wp.max(value, flags[group, member])
    found_inf[member] = value


class StateFlashSACPopulation:
    """Own fixed-address network, optimizer, scaler, and counter state for a FlashSAC population."""

    _SCALAR_STATE_NAMES = (
        "_alpha",
        "_obs_count",
        "_device_update_count",
        "_device_gradient_update_count",
        "_device_update_seed",
        "_device_noise_repeat_count",
        "_device_noise_repeat_steps",
        "_device_exploration_seed",
        "_device_interaction_seed",
        "_device_actor_condition",
        "_device_actor_skip_condition",
    )

    def __init__(self, trainers: tuple[TrainerFlashSAC, ...]):
        if not trainers:
            raise ValueError("FlashSAC population requires at least one trainer")
        first = trainers[0]
        if not isinstance(first.actor.net, NetworkFlashSAC):
            raise ValueError("FlashSAC population requires the reference network backbone")
        tunable = {"actor_lr", "critic_lr", "alpha_lr"}
        for trainer in trainers[1:]:
            if (
                trainer.device != first.device
                or trainer.obs_dim != first.obs_dim
                or trainer.action_dim != first.action_dim
            ):
                raise ValueError("FlashSAC population trainer devices and dimensions must match")
            for field in fields(first.config):
                if field.name not in tunable and getattr(trainer.config, field.name) != getattr(
                    first.config, field.name
                ):
                    raise ValueError(f"FlashSAC config field '{field.name}' is incompatible")
        self.trainers = trainers
        self.population_count = len(trainers)
        self.device = first.device
        self.actors = EnsembleNetworkFlashSAC(*(trainer.actor.net for trainer in trainers))
        critics = tuple(network for trainer in trainers for network in (trainer.critic1, trainer.critic2))
        targets = tuple(network for trainer in trainers for network in (trainer.target_critic1, trainer.target_critic2))
        self.critics = EnsembleNetworkFlashSAC(*critics)
        self.target_critics = EnsembleNetworkFlashSAC(*targets)

        self.log_alpha = self._stack_and_bind(trainers, "log_alpha", requires_grad=True)
        self.actor_log_std = self._stack_nested_and_bind(trainers, "actor", "log_std")
        self.obs_mean = self._stack_and_bind(trainers, "_obs_mean")
        self.obs_m2 = self._stack_and_bind(trainers, "_obs_m2")
        self.scalar_state = {name: self._stack_and_bind(trainers, name) for name in self._SCALAR_STATE_NAMES}

        self.actor_loss = self._stack_and_bind(trainers, "_actor_loss", requires_grad=True)
        self.critic_loss = self._stack_and_bind(trainers, "_critic_loss", requires_grad=True)
        self.alpha_loss = self._stack_and_bind(trainers, "_alpha_loss", requires_grad=True)
        self.scaler = AMPStatePopulation(self.population_count, self.device)
        self.scaler.scale.assign(np.concatenate(tuple(trainer._amp_scale.numpy() for trainer in trainers)))
        self.scaler.growth_tracker.assign(
            np.concatenate(tuple(trainer._amp_growth_tracker.numpy() for trainer in trainers))
        )
        self.scaler.found_inf.assign(np.concatenate(tuple(trainer._amp_found_inf.numpy() for trainer in trainers)))
        self.scaler.step_condition.assign(
            np.concatenate(tuple(trainer._amp_step_condition.numpy() for trainer in trainers))
        )
        for member, trainer in enumerate(trainers):
            trainer._amp_scale = self.scaler.scale[member : member + 1]
            trainer._loss_scale = trainer._amp_scale
            trainer._amp_growth_tracker = self.scaler.growth_tracker[member : member + 1]
            trainer._amp_found_inf = self.scaler.found_inf[member : member + 1]
            trainer._amp_step_condition = self.scaler.step_condition[member : member + 1]

        self.actor_optimizer = AdamPopulation(self.actors.population_parameters(), lr=first.config.actor_lr)
        self.critic_optimizer = AdamPopulation(self.critics.population_parameters(), lr=first.config.critic_lr)
        self.alpha_optimizer = AdamPopulation([self.log_alpha], lr=first.config.alpha_lr)
        self._initialize_optimizer(self.actor_optimizer, tuple(trainer.actor_optimizer for trainer in trainers))
        self._initialize_optimizer(
            self.critic_optimizer,
            tuple(
                optimizer
                for trainer in trainers
                for optimizer in (trainer.critic1_optimizer, trainer.critic2_optimizer)
            ),
        )
        self._initialize_optimizer(self.alpha_optimizer, tuple(trainer.alpha_optimizer for trainer in trainers))
        self.actor_optimizer.step_condition = self.scaler.step_condition
        self.critic_step_condition = wp.ones(self.population_count * 2, dtype=wp.int32, device=self.device)
        self.critic_optimizer.step_condition = self.critic_step_condition

        self._source_first = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._source_second = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._destination_first = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._destination_second = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.sync_critic_step_condition()
        self._bind_scalar_update_views()

        self._reserve_update_buffers()

    def _bind_scalar_optimizer(self, scalar, population: AdamPopulation, member: int) -> None:
        scalar.params = [value[member] for value in population.params]
        scalar.m = [value[member] for value in population.m]
        scalar.v = [value[member] for value in population.v]
        scalar._step_count = population._step_count[member : member + 1]
        scalar._step_corrections = population._step_corrections[member]
        scalar._grad_sumsq = population._grad_sumsq[member : member + 1]
        scalar.lr_scale = population.lr_scale[member : member + 1]
        scalar.pbt_lr_scale = population.pbt_lr_scale[member : member + 1]
        scalar.step_condition = population.step_condition[member : member + 1]

    def _bind_scalar_update_views(self) -> None:
        for member, trainer in enumerate(self.trainers):
            self._bind_scalar_optimizer(trainer.actor_optimizer, self.actor_optimizer, member)
            self._bind_scalar_optimizer(trainer.critic1_optimizer, self.critic_optimizer, member * 2)
            self._bind_scalar_optimizer(trainer.critic2_optimizer, self.critic_optimizer, member * 2 + 1)
            self._bind_scalar_optimizer(trainer.alpha_optimizer, self.alpha_optimizer, member)
            trainer.critic1_optimizer.step_condition = trainer._amp_step_condition
            trainer.critic2_optimizer.step_condition = trainer._amp_step_condition
            trainer._critic_ensemble = self.critics if self.population_count == 1 else None
            trainer._target_critic_ensemble = self.target_critics if self.population_count == 1 else None
            for network in (
                trainer.actor.net,
                trainer.critic1,
                trainer.critic2,
                trainer.target_critic1,
                trainer.target_critic2,
            ):
                network.refresh_contraction_weights()

    def _reserve_update_buffers(self) -> None:
        """Allocate every complete-update buffer before graph capture."""

        first = self.trainers[0]
        population = self.population_count
        critic_count = population * 2
        batch_size = int(first.config.sample_batch_size)
        rows = batch_size * 2
        obs_dim = first.obs_dim
        action_dim = first.action_dim
        q_dim = obs_dim + action_dim
        atoms = int(first.config.distributional_atoms)
        self._update_batch_size = batch_size
        self._seed_base = wp.zeros(population, dtype=wp.int32, device=self.device)
        self._actor_found_inf = wp.zeros(population, dtype=wp.int32, device=self.device)
        self._actor_parameters = tuple(self.actors.population_parameters())
        self._critic_parameters = tuple(self.critics.population_parameters())
        self._target_critic_parameters = tuple(self.target_critics.population_parameters())
        self._actor_parameter_grads = tuple(parameter.grad for parameter in self._actor_parameters)
        self._critic_parameter_grads = tuple(parameter.grad for parameter in self._critic_parameters)
        self._log_alpha_grad = self.log_alpha.grad
        self._actor_step_condition = wp.ones(population, dtype=wp.int32, device=self.device)
        self._actor_param_found_inf = wp.zeros(
            (len(self._actor_parameters), population), dtype=wp.int32, device=self.device
        )
        self._critic_param_found_inf = wp.zeros(
            (len(self._critic_parameters), population), dtype=wp.int32, device=self.device
        )
        self._actor_param_found_inf_views = tuple(
            self._actor_param_found_inf[group] for group in range(len(self._actor_parameters))
        )
        self._critic_param_found_inf_views = tuple(
            self._critic_param_found_inf[group] for group in range(len(self._critic_parameters))
        )
        self._actor_observations = wp.empty((rows, obs_dim), dtype=wp.float32, device=self.device)
        self._actor_eps = wp.empty((population, batch_size, action_dim), dtype=wp.float32, device=self.device)
        self._actor_actions = wp.empty_like(self._actor_eps)
        self._actor_log_probs = wp.empty((population, batch_size), dtype=wp.float32, device=self.device)
        self._next_eps = wp.empty_like(self._actor_eps)
        self._next_actions = wp.empty_like(self._actor_eps)
        self._next_log_probs = wp.empty_like(self._actor_log_probs)
        self._actor_q_inputs = wp.empty((critic_count, batch_size, q_dim), dtype=wp.float32, device=self.device)
        self._actor_q_output_grads = wp.empty((critic_count, batch_size, atoms), dtype=wp.float32, device=self.device)
        self._actor_q_input_grads = wp.empty_like(self._actor_q_inputs)
        self._q_inputs = wp.empty((critic_count, rows, q_dim), dtype=wp.float32, device=self.device)
        self._q_values = wp.empty((critic_count, batch_size, 1), dtype=wp.float32, device=self.device)
        self._q_output_grads = wp.empty((critic_count, rows, atoms), dtype=wp.float32, device=self.device)
        self._policy_output_grads = wp.empty((population, rows, action_dim * 2), dtype=wp.float32, device=self.device)
        self._target_distributions = wp.empty((critic_count, batch_size, atoms), dtype=wp.float32, device=self.device)
        self._actor_q_input_grad_views = tuple(self._actor_q_input_grads[index] for index in range(critic_count))
        self.actors.reserve_buffers(rows)
        self.critics.reserve_buffers(batch_size)
        self._actor_critic_workspace = self.critics._workspace
        self._reserve_backward_contraction_mirrors(self.critics)
        self.critics.reserve_buffers(rows)
        self._training_critic_workspace = self.critics._workspace
        self._reserve_backward_contraction_mirrors(self.critics)
        self.target_critics.reserve_buffers(rows)
        self._reserve_backward_contraction_mirrors(self.actors)
        self._workspace_arrays = self._collect_update_arrays(
            self.actors._workspace,
            self._actor_critic_workspace,
            self._training_critic_workspace,
            self.target_critics._workspace,
            self.actors._fp16_inputs_2d,
            self.actors._fp16_inputs_3d,
            self.critics._fp16_inputs_2d,
            self.critics._fp16_inputs_3d,
            self.target_critics._fp16_inputs_2d,
            self.target_critics._fp16_inputs_3d,
        )

    @staticmethod
    def _collect_update_arrays(*roots: object) -> tuple[wp.array[Any], ...]:
        arrays: list[wp.array[Any]] = []
        seen: set[int] = set()

        def visit(value: object) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list | tuple):
                for child in value:
                    visit(child)
            elif hasattr(value, "ptr") and int(value.ptr) not in seen:
                seen.add(int(value.ptr))
                arrays.append(value)

        visit(roots)
        return tuple(arrays)

    @staticmethod
    def _reserve_backward_contraction_mirrors(ensemble: EnsembleNetworkFlashSAC) -> None:
        if ensemble.contraction_dtype != "float16":
            return
        backward = ensemble._workspace["backward"]
        values = [backward["output_grad"], backward["rms_grad"]]
        for block in backward["blocks"]:
            values.extend((block[1], block[4]))
        for value in values:
            ensemble._contraction_input(value, refresh=False)

    def update_buffer_arrays(self) -> tuple[wp.array[Any], ...]:
        """Return fixed-address complete-update buffers for pointer audits."""

        return (
            self._seed_base,
            self._actor_found_inf,
            self._actor_step_condition,
            self._actor_param_found_inf,
            self._critic_param_found_inf,
            self._actor_observations,
            self._actor_eps,
            self._actor_actions,
            self._actor_log_probs,
            self._next_eps,
            self._next_actions,
            self._next_log_probs,
            self._q_inputs,
            self._actor_q_inputs,
            self._actor_q_output_grads,
            self._actor_q_input_grads,
            self._q_values,
            self._q_output_grads,
            self._policy_output_grads,
            self._target_distributions,
            self.actor_loss,
            self.critic_loss,
            self.alpha_loss,
            *self._workspace_arrays,
        )

    def _validate_fused_batch(self, batch: BatchSAC) -> None:
        first = self.trainers[0]
        if self.population_count == 1:
            return
        if int(first.config.update_steps) != 1:
            raise ValueError("Population-fused FlashSAC requires update_steps=1")
        if not first.config.distributional_critic:
            raise ValueError("Population-fused FlashSAC currently requires distributional critics")
        if int(batch.batch_size) != self._update_batch_size:
            raise ValueError("Population-fused batch size must match setup sample_batch_size")
        if int(batch.obs.shape[1]) != first.obs_dim or int(batch.next_obs.shape[1]) != first.obs_dim:
            raise ValueError("Batch observation dimensions do not match population")
        if int(batch.actions.shape[1]) != first.action_dim:
            raise ValueError("Batch action dimensions do not match population")

    def _unscale_population_parameters(
        self,
        gradients: tuple[wp.array[Any], ...],
        *,
        members_per_scale: int,
        parameter_flags: wp.array2d[wp.int32],
        parameter_flag_views: tuple[wp.array[wp.int32], ...],
    ) -> None:
        parameter_flags.zero_()
        for gradient, found_inf in zip(gradients, parameter_flag_views, strict=True):
            kernel = _population_unscale_2d_kernel if gradient.ndim == 2 else _population_unscale_3d_kernel
            wp.launch(
                kernel,
                dim=gradient.shape,
                inputs=[gradient, self.scaler.scale, int(members_per_scale)],
                outputs=[found_inf],
                device=self.device,
            )
        wp.launch(
            _population_reduce_found_inf_kernel,
            dim=self.population_count,
            inputs=[parameter_flags],
            outputs=[self.scaler.found_inf],
            device=self.device,
        )

    def _normalize_population_networks(self, networks: tuple[NetworkFlashSAC, ...]) -> None:
        if not self.trainers[0].config.normalize_weights:
            return
        for network in networks:
            network.normalize_weights()

    def _fused_complete_update_operations(self, batch: BatchSAC, *, read_stats: bool) -> None:
        first = self.trainers[0]
        config = first.config
        population = self.population_count
        critic_count = population * 2
        batch_size = self._update_batch_size
        rows = batch_size * 2
        atoms = int(config.distributional_atoms)
        wp.launch(
            _population_prepare_update_kernel,
            dim=population,
            inputs=[
                self.scalar_state["_device_gradient_update_count"],
                self.scalar_state["_device_update_count"],
                self._seed_base,
                int(config.learning_rate_warmup_steps),
                int(config.learning_rate_decay_steps),
                float(config.actor_lr),
                float(config.learning_rate_end),
                float(self.actor_optimizer.lr),
                float(self.critic_optimizer.lr),
                float(self.alpha_optimizer.lr),
            ],
            outputs=[
                self.scalar_state["_device_update_seed"],
                self.actor_optimizer.lr_scale,
                self.critic_optimizer.lr_scale,
                self.alpha_optimizer.lr_scale,
            ],
            device=self.device,
        )
        wp.launch(
            _concat_population_actor_observations_kernel,
            dim=self._actor_observations.shape,
            inputs=[batch.obs, batch.next_obs, batch_size],
            outputs=[self._actor_observations],
            device=self.device,
        )

        policy_out = self.actors.forward_all_manual(self._actor_observations, training=True)
        wp.launch(
            _population_fill_eps_kernel,
            dim=self._actor_eps.shape,
            inputs=[self.scalar_state["_device_update_seed"], 0],
            outputs=[self._actor_eps],
            device=self.device,
        )
        wp.launch(
            _population_sample_actions_kernel,
            dim=(population, batch_size),
            inputs=[
                policy_out,
                0,
                self._actor_eps,
                first.action_dim,
                first.actor.log_std_min,
                first.actor.log_std_max,
            ],
            outputs=[self._actor_actions, self._actor_log_probs],
            device=self.device,
        )
        self.critics._workspace = self._actor_critic_workspace
        self.critics._workspace_rows = batch_size
        wp.launch(
            _population_actor_q_input_kernel,
            dim=self._actor_q_inputs.shape,
            inputs=[batch.obs, self._actor_actions, first.obs_dim, batch_size],
            outputs=[self._actor_q_inputs],
            device=self.device,
        )
        actor_q_logits = self.critics.forward_all_manual(self._actor_q_inputs, training=False)
        wp.launch(
            _population_distributional_q_value_kernel,
            dim=(critic_count, batch_size),
            inputs=[
                actor_q_logits,
                batch_size,
                atoms,
                float(config.distributional_v_min),
                float(config.distributional_v_max),
            ],
            outputs=[self._q_values],
            device=self.device,
        )
        wp.launch(
            _population_actor_q_backward_kernel,
            dim=self._actor_q_output_grads.shape,
            inputs=[
                actor_q_logits,
                self._q_values,
                self.scaler.scale,
                batch_size,
                atoms,
                float(config.distributional_v_min),
                float(config.distributional_v_max),
                bool(config.average_critics),
            ],
            outputs=[self._actor_q_output_grads],
            device=self.device,
        )
        self.critics.backward_all_manual(self._actor_q_output_grads, input_grads=self._actor_q_input_grad_views)
        self.critics._workspace = self._training_critic_workspace
        self.critics._workspace_rows = rows
        for gradient in self._critic_parameter_grads:
            gradient.zero_()
        wp.launch(
            _population_actor_policy_backward_kernel,
            dim=(population, rows),
            inputs=[
                policy_out,
                self._actor_eps,
                self._actor_q_input_grads,
                self.scaler.scale,
                self.scalar_state["_alpha"],
                first.obs_dim,
                first.action_dim,
                batch_size,
                first.actor.log_std_min,
                first.actor.log_std_max,
            ],
            outputs=[self._policy_output_grads],
            device=self.device,
        )
        if read_stats:
            wp.launch(
                _population_actor_loss_kernel,
                dim=population,
                inputs=[
                    self._q_values,
                    self._actor_log_probs,
                    self.scalar_state["_alpha"],
                    batch_size,
                    bool(config.average_critics),
                ],
                outputs=[self.actor_loss],
                device=self.device,
            )
        self.scaler.reset_found_inf()
        self.actors.backward_all_manual(self._policy_output_grads)
        self._unscale_population_parameters(
            self._actor_parameter_grads,
            members_per_scale=1,
            parameter_flags=self._actor_param_found_inf,
            parameter_flag_views=self._actor_param_found_inf_views,
        )
        self.scaler.update()
        wp.copy(self._actor_found_inf, self.scaler.found_inf)
        wp.copy(self._actor_step_condition, self.scaler.step_condition)
        self.actor_optimizer.step()
        self._normalize_population_networks(self.actors.networks)
        self.actors.refresh_contraction_weights()

        if config.auto_alpha:
            wp.launch(
                _population_alpha_loss_kernel,
                dim=population,
                inputs=[
                    self._actor_log_probs,
                    self.log_alpha,
                    batch_size,
                    float(first.target_entropy),
                ],
                outputs=[self.alpha_loss, self._log_alpha_grad],
                device=self.device,
            )
            self.alpha_optimizer.step()
            wp.launch(
                _population_refresh_alpha_kernel,
                dim=population,
                inputs=[self.log_alpha],
                outputs=[self.scalar_state["_alpha"]],
                device=self.device,
            )

        policy_out = self.actors.forward_all(self._actor_observations, training=False)
        wp.launch(
            _population_fill_eps_kernel,
            dim=self._next_eps.shape,
            inputs=[self.scalar_state["_device_update_seed"], 2],
            outputs=[self._next_eps],
            device=self.device,
        )
        wp.launch(
            _population_sample_actions_kernel,
            dim=(population, batch_size),
            inputs=[
                policy_out,
                batch_size,
                self._next_eps,
                first.action_dim,
                first.actor.log_std_min,
                first.actor.log_std_max,
            ],
            outputs=[self._next_actions, self._next_log_probs],
            device=self.device,
        )
        wp.launch(
            _population_training_q_input_kernel,
            dim=self._q_inputs.shape,
            inputs=[
                batch.obs,
                batch.actions,
                batch.next_obs,
                self._next_actions,
                first.obs_dim,
                batch_size,
            ],
            outputs=[self._q_inputs],
            device=self.device,
        )
        target_logits = self.target_critics.forward_all(self._q_inputs, training=True)
        critic_logits = self.critics.forward_all_manual(self._q_inputs, training=True)
        wp.launch(
            _population_distributional_projection_kernel,
            dim=(population, batch_size, atoms),
            inputs=[
                batch.rewards,
                batch.dones,
                target_logits,
                self._next_log_probs,
                self.scalar_state["_alpha"],
                float(config.gamma**config.n_step),
                batch_size,
                atoms,
                float(config.distributional_v_min),
                float(config.distributional_v_max),
                bool(config.distributional_min_target),
            ],
            outputs=[self._target_distributions],
            device=self.device,
        )
        wp.launch(
            _population_critic_loss_backward_kernel,
            dim=self._q_output_grads.shape,
            inputs=[
                critic_logits,
                self._target_distributions,
                self.scaler.scale,
                batch_size,
                atoms,
            ],
            outputs=[self._q_output_grads],
            device=self.device,
        )
        if read_stats:
            wp.launch(
                _population_critic_loss_kernel,
                dim=population,
                inputs=[critic_logits, self._target_distributions, batch_size, atoms],
                outputs=[self.critic_loss],
                device=self.device,
            )
        self.scaler.reset_found_inf()
        self.critics.backward_all_manual(self._q_output_grads)
        self._unscale_population_parameters(
            self._critic_parameter_grads,
            members_per_scale=2,
            parameter_flags=self._critic_param_found_inf,
            parameter_flag_views=self._critic_param_found_inf_views,
        )
        self.scaler.update()
        self.sync_critic_step_condition()
        self.critic_optimizer.step()
        self._normalize_population_networks(self.critics.networks)
        self.critics.refresh_contraction_weights()
        self.soft_update_targets(float(config.tau))
        wp.launch(
            _population_increment_counters_kernel,
            dim=population,
            inputs=[
                self.scalar_state["_device_gradient_update_count"],
                self.scalar_state["_device_update_count"],
            ],
            device=self.device,
        )

    def _fused_critic_only_update_operations(self, batch: BatchSAC, *, read_stats: bool) -> None:
        """Record one population critic-only update for policy-frequency cadence."""

        first = self.trainers[0]
        config = first.config
        population = self.population_count
        batch_size = self._update_batch_size
        atoms = int(config.distributional_atoms)
        wp.launch(
            _population_prepare_update_kernel,
            dim=population,
            inputs=[
                self.scalar_state["_device_gradient_update_count"],
                self.scalar_state["_device_update_count"],
                self._seed_base,
                int(config.learning_rate_warmup_steps),
                int(config.learning_rate_decay_steps),
                float(config.actor_lr),
                float(config.learning_rate_end),
                float(self.actor_optimizer.lr),
                float(self.critic_optimizer.lr),
                float(self.alpha_optimizer.lr),
            ],
            outputs=[
                self.scalar_state["_device_update_seed"],
                self.actor_optimizer.lr_scale,
                self.critic_optimizer.lr_scale,
                self.alpha_optimizer.lr_scale,
            ],
            device=self.device,
        )
        wp.launch(
            _concat_population_actor_observations_kernel,
            dim=self._actor_observations.shape,
            inputs=[batch.obs, batch.next_obs, batch_size],
            outputs=[self._actor_observations],
            device=self.device,
        )
        policy_out = self.actors.forward_all(self._actor_observations, training=False)
        wp.launch(
            _population_fill_eps_kernel,
            dim=self._next_eps.shape,
            inputs=[self.scalar_state["_device_update_seed"], 2],
            outputs=[self._next_eps],
            device=self.device,
        )
        wp.launch(
            _population_sample_actions_kernel,
            dim=(population, batch_size),
            inputs=[
                policy_out,
                batch_size,
                self._next_eps,
                first.action_dim,
                first.actor.log_std_min,
                first.actor.log_std_max,
            ],
            outputs=[self._next_actions, self._next_log_probs],
            device=self.device,
        )
        wp.launch(
            _population_training_q_input_kernel,
            dim=self._q_inputs.shape,
            inputs=[
                batch.obs,
                batch.actions,
                batch.next_obs,
                self._next_actions,
                first.obs_dim,
                batch_size,
            ],
            outputs=[self._q_inputs],
            device=self.device,
        )
        target_logits = self.target_critics.forward_all(self._q_inputs, training=True)
        critic_logits = self.critics.forward_all_manual(self._q_inputs, training=True)
        wp.launch(
            _population_distributional_projection_kernel,
            dim=(population, batch_size, atoms),
            inputs=[
                batch.rewards,
                batch.dones,
                target_logits,
                self._next_log_probs,
                self.scalar_state["_alpha"],
                float(config.gamma**config.n_step),
                batch_size,
                atoms,
                float(config.distributional_v_min),
                float(config.distributional_v_max),
                bool(config.distributional_min_target),
            ],
            outputs=[self._target_distributions],
            device=self.device,
        )
        wp.launch(
            _population_critic_loss_backward_kernel,
            dim=self._q_output_grads.shape,
            inputs=[
                critic_logits,
                self._target_distributions,
                self.scaler.scale,
                batch_size,
                atoms,
            ],
            outputs=[self._q_output_grads],
            device=self.device,
        )
        if read_stats:
            wp.launch(
                _population_critic_loss_kernel,
                dim=population,
                inputs=[critic_logits, self._target_distributions, batch_size, atoms],
                outputs=[self.critic_loss],
                device=self.device,
            )
        self.scaler.reset_found_inf()
        self.critics.backward_all_manual(self._q_output_grads)
        self._unscale_population_parameters(
            self._critic_parameter_grads,
            members_per_scale=2,
            parameter_flags=self._critic_param_found_inf,
            parameter_flag_views=self._critic_param_found_inf_views,
        )
        self.scaler.update()
        self.sync_critic_step_condition()
        self.critic_optimizer.step()
        self._normalize_population_networks(self.critics.networks)
        self.critics.refresh_contraction_weights()
        self.soft_update_targets(float(config.tau))
        wp.launch(
            _population_increment_counters_kernel,
            dim=population,
            inputs=[
                self.scalar_state["_device_gradient_update_count"],
                self.scalar_state["_device_update_count"],
            ],
            device=self.device,
        )

    def capture_fused_critic_update(self, batch: BatchSAC, *, seed: int) -> object:
        """Capture one fixed-address critic-only population update."""

        self._validate_fused_batch(batch)
        self._seed_base.assign(
            np.asarray([int(seed) + member for member in range(self.population_count)], dtype=np.int32)
        )
        pointers = tuple(array.ptr for array in (*self.state_arrays(), *self.update_buffer_arrays()))
        with wp.ScopedCapture(device=self.device) as capture:
            self._fused_critic_only_update_operations(batch, read_stats=False)
        if pointers != tuple(array.ptr for array in (*self.state_arrays(), *self.update_buffer_arrays())):
            raise RuntimeError("Population critic capture changed a setup-owned array address")
        return capture.graph

    def update_all_fused(self, batch: BatchSAC, *, seed: int, read_stats: bool = True) -> tuple[StatsSACUpdate, ...]:
        """Run one allocation-stable population update, specializing P=1 to the scalar path."""

        if self.population_count == 1:
            return (self.update_p1(batch, seed=seed, read_stats=read_stats),)
        self._validate_fused_batch(batch)
        seed_bases = np.asarray(
            [int(seed) + member - int(trainer._update_count) * 9973 for member, trainer in enumerate(self.trainers)],
            dtype=np.int32,
        )
        self._seed_base.assign(seed_bases)
        self._fused_complete_update_operations(batch, read_stats=read_stats)
        for trainer in self.trainers:
            trainer._gradient_update_count += 1
            trainer._update_count += 1
        if not read_stats:
            return tuple(StatsSACUpdate(0.0, 0.0, 0.0, 0.0) for _ in self.trainers)
        actor_loss = self.actor_loss.numpy()[:, 0]
        critic_loss = self.critic_loss.numpy()[:, 0]
        alpha_loss = self.alpha_loss.numpy()[:, 0]
        alpha = self.scalar_state["_alpha"].numpy()[:, 0]
        return tuple(
            StatsSACUpdate(
                actor_loss=float(actor_loss[member]),
                critic_loss=float(critic_loss[member]),
                alpha_loss=float(alpha_loss[member]),
                alpha=float(alpha[member]),
            )
            for member in range(self.population_count)
        )

    update_fused = update_all_fused

    def capture_fused_update(self, batch: BatchSAC, *, seed: int) -> object:
        """Capture the no-stat population update with fixed batch and state addresses."""

        self._validate_fused_batch(batch)
        if self.population_count == 1:
            return self.trainers[0].capture_update_graph(batch, seed=seed)
        if not self.device.is_cuda or not self.device.is_mempool_enabled:
            raise RuntimeError("Population FlashSAC update graphs require CUDA memory pools")
        self._seed_base.assign(
            np.asarray([int(seed) + member for member in range(self.population_count)], dtype=np.int32)
        )
        pointers = tuple(array.ptr for array in (*self.state_arrays(), *self.update_buffer_arrays()))
        with wp.ScopedCapture(device=self.device) as capture:
            self._fused_complete_update_operations(batch, read_stats=False)
        if pointers != tuple(array.ptr for array in (*self.state_arrays(), *self.update_buffer_arrays())):
            raise RuntimeError("Population update capture changed a setup-owned array address")
        return capture.graph

    def update_p1(self, batch, *, seed: int | None = None, read_stats: bool = True):
        """Run one exact scalar-specialized update through population-owned state."""

        if self.population_count != 1:
            raise RuntimeError("update_p1 requires a one-member population")
        return self.trainers[0].update(batch, seed=seed, read_stats=read_stats)

    def update_all(self, batch, *, seed: int, read_stats: bool = True):
        """Run exact independent member updates over shared population-owned storage."""
        stats = tuple(
            trainer.update(batch, seed=int(seed) + member, read_stats=read_stats)
            for member, trainer in enumerate(self.trainers)
        )
        self.sync_critic_step_condition()
        self.actors.refresh_contraction_weights()
        self.critics.refresh_contraction_weights()
        self.target_critics.refresh_contraction_weights()
        return stats

    def _stack_and_bind(self, trainers: tuple[TrainerFlashSAC, ...], name: str, *, requires_grad: bool = False):
        source = tuple(getattr(trainer, name) for trainer in trainers)
        stacked = wp.array(
            np.stack(tuple(value.numpy() for value in source)),
            dtype=source[0].dtype,
            device=self.device,
            requires_grad=requires_grad,
        )
        for member, trainer in enumerate(trainers):
            setattr(trainer, name, stacked[member])
        return stacked

    def _stack_nested_and_bind(self, trainers: tuple[TrainerFlashSAC, ...], owner: str, name: str):
        source = tuple(getattr(getattr(trainer, owner), name) for trainer in trainers)
        stacked = wp.array(
            np.stack(tuple(value.numpy() for value in source)), dtype=source[0].dtype, device=self.device
        )
        for member, trainer in enumerate(trainers):
            setattr(getattr(trainer, owner), name, stacked[member])
        return stacked

    @staticmethod
    def _initialize_optimizer(population: AdamPopulation, optimizers: tuple) -> None:
        population._step_count.assign(np.asarray([optimizer.step_count for optimizer in optimizers], dtype=np.int32))
        population._step_corrections.assign(
            np.stack(tuple(optimizer._step_corrections.numpy() for optimizer in optimizers))
        )
        population._grad_sumsq.assign(np.concatenate(tuple(optimizer._grad_sumsq.numpy() for optimizer in optimizers)))
        population.lr_scale.assign(np.concatenate(tuple(optimizer.lr_scale.numpy() for optimizer in optimizers)))
        population.pbt_lr_scale.assign(
            np.concatenate(tuple(optimizer.pbt_lr_scale.numpy() for optimizer in optimizers))
        )
        population.step_condition.assign(
            np.concatenate(tuple(optimizer.step_condition.numpy() for optimizer in optimizers))
        )
        for group, (m, v) in enumerate(zip(population.m, population.v, strict=True)):
            m.assign(np.stack(tuple(optimizer.m[group].numpy() for optimizer in optimizers)))
            v.assign(np.stack(tuple(optimizer.v[group].numpy() for optimizer in optimizers)))

    def sync_critic_step_condition(self) -> None:
        """Repeat each global trainer overflow condition for its two critics."""

        wp.launch(
            population_repeat_pair_kernel,
            dim=(self.population_count, 2),
            inputs=[self.scaler.step_condition],
            outputs=[self.critic_step_condition],
            device=self.device,
        )

    def soft_update_targets(self, tau: float) -> None:
        """EMA online critic parameters into targets without touching BN running buffers."""

        for target, online in zip(self._target_critic_parameters, self._critic_parameters, strict=True):
            kernel = soft_update_2d_kernel if target.ndim == 2 else soft_update_3d_kernel
            wp.launch(kernel, dim=target.shape, inputs=[online, float(tau)], outputs=[target], device=self.device)
        self.target_critics.refresh_contraction_weights()

    def copy_member(self, source_index: wp.array[wp.int32], destination_index: wp.array[wp.int32]) -> None:
        """Promote one complete learner member without changing any allocation address."""

        wp.launch(
            population_pair_indices_kernel,
            dim=1,
            inputs=[source_index, destination_index],
            outputs=[self._source_first, self._source_second, self._destination_first, self._destination_second],
            device=self.device,
        )
        self.actors.copy_population_member(source_index, destination_index)
        self.actor_optimizer.copy_member(source_index, destination_index)
        self.scaler.copy_member(source_index, destination_index)
        self.alpha_optimizer.copy_member(source_index, destination_index)
        for values in (self.log_alpha, self.actor_log_std, self.obs_mean, self.obs_m2, *self.scalar_state.values()):
            if values.ndim == 1:
                kernel = population_copy_int_1d_kernel if values.dtype == wp.int32 else population_copy_float_1d_kernel
            else:
                kernel = population_copy_int_2d_kernel if values.dtype == wp.int32 else population_copy_float_2d_kernel
            wp.launch(
                kernel,
                dim=1 if values.ndim == 1 else values.shape[1],
                inputs=[values, source_index, destination_index],
                device=self.device,
            )
        for source, destination in (
            (self._source_first, self._destination_first),
            (self._source_second, self._destination_second),
        ):
            self.critics.copy_population_member(source, destination)
            self.target_critics.copy_population_member(source, destination)
            self.critic_optimizer.copy_member(source, destination)
        self.sync_critic_step_condition()

    def state_arrays(self) -> tuple[wp.array[Any], ...]:
        """Return all fixed-address arrays owned by the population state."""

        return (
            *self.actors.population_state_arrays(),
            *self.critics.population_state_arrays(),
            *self.target_critics.population_state_arrays(),
            *self.actors._fp16_weights.values(),
            *self.critics._fp16_weights.values(),
            *self.target_critics._fp16_weights.values(),
            self.log_alpha,
            self.actor_log_std,
            self.obs_mean,
            self.obs_m2,
            *self.scalar_state.values(),
            *self.actor_optimizer.state_arrays(),
            *self.critic_optimizer.state_arrays(),
            *self.alpha_optimizer.state_arrays(),
            *self.scaler.state_arrays(),
            self.critic_step_condition,
            self._source_first,
            self._source_second,
            self._destination_first,
            self._destination_second,
        )
