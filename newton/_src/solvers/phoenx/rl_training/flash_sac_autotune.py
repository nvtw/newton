# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal champion/challenger hyperparameter search for FlashSAC."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import warp as wp

from .flash_sac import (
    BufferReplayFlashSAC,
    EnvFlashSAC,
    TrainerFlashSAC,
    _allocate_flash_sac_batch,
)
from .flash_sac_autotune_overlap import capture_lr_autotune_overlap
from .flash_sac_autotune_parallel import capture_lr_autotune_parallel_overlap
from .flash_sac_population import StateFlashSACPopulation
from .sac import BatchSAC, StatsSACUpdate


def _proposal_direction(search_round: int, coordinate_count: int) -> float:
    """Probe faster learning first, then bracket below the current value."""

    return 1.0 if (int(search_round) // int(coordinate_count)) % 2 == 0 else -1.0


@wp.kernel
def _route_split_actions_kernel(
    champion_actions: wp.array2d[wp.float32],
    challenger_actions: wp.array2d[wp.float32],
    champion_worlds: wp.int32,
    challenger_enabled: wp.array[wp.int32],
    actions: wp.array2d[wp.float32],
):
    world, action = wp.tid()
    if challenger_enabled[0] != wp.int32(0) and world >= champion_worlds:
        actions[world, action] = challenger_actions[world, action]
    else:
        actions[world, action] = champion_actions[world, action]


@wp.kernel
def _paired_evaluation_kernel(
    champion_scores: wp.array[wp.float32],
    challenger_scores: wp.array[wp.float32],
    challenger_safe: wp.array[wp.int32],
    delta: wp.array[wp.float32],
    valid: wp.array[wp.int32],
):
    total = wp.float32(0.0)
    is_valid = challenger_safe[0]
    for episode in range(champion_scores.shape[0]):
        champion = champion_scores[episode]
        challenger = challenger_scores[episode]
        if not wp.isfinite(champion) or not wp.isfinite(challenger):
            is_valid = wp.int32(0)
        total += challenger - champion
    delta[0] = total / wp.float32(champion_scores.shape[0])
    valid[0] = is_valid


@dataclass(frozen=True)
class ConfigFlashSACLRAutotune:
    """Configure bounded paired FlashSAC hyperparameter search.

    Safe low-signal candidates receive a minimum resource rung before paired
    decisions begin.
    """

    evaluation_episodes: int = 8
    challenger_fraction: float = 0.0
    initial_perturbation_factor: float = 2.0
    challenger_action_rms_limit: float = 0.35
    challenger_action_max_limit: float = 0.75
    minimum_perturbation_factor: float = 1.02
    multiplier_bounds: tuple[float, float] = (0.5, 2.0)
    target_update_rate_multiplier_bounds: tuple[float, float] = (0.5, 2.0)
    minimum_evidence_windows: int = 2
    informative_score_threshold: float = 0.05
    improvement_margin: float = 0.01
    policy_frequency_choices: tuple[int, ...] = (1, 2, 4)
    relative_improvement_margin: float = 0.10
    minimum_effect_delta: float = 1.0e-4
    termination_rate_margin: float = 0.05
    regression_margin: float = 0.05
    promotion_windows: int = 2
    exploit_after_candidate: bool = True
    convergence_windows: int = 12
    minimum_search_windows: int = 12
    reopen_stagnation_windows: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        if not 0.0 <= self.challenger_fraction < 0.5:
            raise ValueError("challenger_fraction must be in [0, 0.5)")
        if self.minimum_perturbation_factor <= 1.0:
            raise ValueError("minimum_perturbation_factor must exceed one")
        if self.challenger_action_rms_limit <= 0.0 or self.challenger_action_max_limit <= 0.0:
            raise ValueError("challenger action divergence limits must be positive")
        if self.initial_perturbation_factor < self.minimum_perturbation_factor:
            raise ValueError("initial_perturbation_factor must not be smaller than its minimum")
        if self.multiplier_bounds[0] <= 0.0 or self.multiplier_bounds[1] < 1.0:
            raise ValueError("multiplier_bounds must be positive and contain one")
        if self.multiplier_bounds[0] > 1.0 or self.multiplier_bounds[0] >= self.multiplier_bounds[1]:
            raise ValueError("multiplier_bounds must straddle one")
        target_bounds = self.target_update_rate_multiplier_bounds
        if target_bounds[0] <= 0.0 or target_bounds[1] < 1.0:
            raise ValueError("target_update_rate_multiplier_bounds must be positive and contain one")
        if target_bounds[0] > 1.0 or target_bounds[0] >= target_bounds[1]:
            raise ValueError("target_update_rate_multiplier_bounds must straddle one")
        if self.improvement_margin < 0.0:
            raise ValueError("improvement_margin must be non-negative")
        if self.minimum_evidence_windows <= 0:
            raise ValueError("minimum_evidence_windows must be positive")
        if self.informative_score_threshold < 0.0:
            raise ValueError("informative_score_threshold must be non-negative")
        if self.relative_improvement_margin < 0.0:
            raise ValueError("relative_improvement_margin must be non-negative")
        if self.minimum_effect_delta < 0.0:
            raise ValueError("minimum_effect_delta must be non-negative")
        if (
            not self.policy_frequency_choices
            or any(value <= 0 for value in self.policy_frequency_choices)
            or tuple(sorted(set(self.policy_frequency_choices))) != self.policy_frequency_choices
        ):
            raise ValueError("policy_frequency_choices must be positive, unique, and sorted")
        if self.termination_rate_margin < 0.0:
            raise ValueError("termination_rate_margin must be non-negative")
        if self.regression_margin < 0.0:
            raise ValueError("regression_margin must be non-negative")
        if (
            self.promotion_windows <= 0
            or self.convergence_windows <= 0
            or self.minimum_search_windows <= 0
            or self.reopen_stagnation_windows <= 0
        ):
            raise ValueError("window counts must be positive")


@dataclass(frozen=True)
class ResultFlashSACLRAutotune:
    """Describe one paired evaluation decision."""

    paired_delta: float
    action: str
    consecutive_wins: int
    converged: bool


class GraphFlashSACLRAutotune(Protocol):
    """Represent a backend-neutral captured FlashSAC autotuning graph."""

    trainers: tuple[TrainerFlashSAC, TrainerFlashSAC]

    def launch(self) -> None:
        """Launch one captured training cadence."""

    def synchronize(self) -> None:
        """Wait for all captured training work to complete."""

    def sync_controller_state(self) -> None:
        """Copy backend learner state into its controller."""

    def evaluation_trainers(self) -> tuple[TrainerFlashSAC, TrainerFlashSAC]:
        """Return the fixed-address policies to evaluate."""

    def reopen_search(self) -> None:
        """Restart paired search from the current converged learner."""

    def evaluate_paired(self, *args: Any, **kwargs: Any) -> ResultFlashSACLRAutotune:
        """Evaluate paired scores and apply the controller decision."""

    def challenger_fallback_fraction(self) -> float:
        """Return the most recent guarded-action fallback fraction."""

    def close(self) -> None:
        """Drain work and release captured graph resources."""


class ControllerFlashSACLRAutotune:
    """Run a bounded two-member FlashSAC hyperparameter search."""

    _RATE_NAMES = ("actor", "critic", "alpha")

    @classmethod
    def from_trainer(
        cls,
        trainer: TrainerFlashSAC,
        *,
        rollout_world_count: int,
        config: ConfigFlashSACLRAutotune | None = None,
    ) -> ControllerFlashSACLRAutotune:
        """Create a paired search controller from one configured trainer.

        The supplied trainer becomes the owned champion. A compatible challenger,
        fixed learner batch, and confirmed-best snapshot are allocated during setup.

        Args:
            trainer: Configured champion trainer to own.
            rollout_world_count: Number of training rollout worlds.
            config: Optional bounded-search configuration.

        Returns:
            A setup-complete paired learning-rate controller.
        """

        challenger = TrainerFlashSAC(
            obs_dim=trainer.obs_dim,
            action_dim=trainer.action_dim,
            config=trainer.config,
            device=trainer.device,
            seed=trainer.seed + 1,
        )
        challenger.copy_training_state_from(trainer)
        batch = _allocate_flash_sac_batch(trainer)
        return cls((trainer, challenger), batch, rollout_world_count=rollout_world_count, config=config)

    def __init__(
        self,
        trainers: tuple[TrainerFlashSAC, TrainerFlashSAC],
        batch: BatchSAC,
        *,
        rollout_world_count: int,
        config: ConfigFlashSACLRAutotune | None = None,
        single_trainer: TrainerFlashSAC | None = None,
    ):
        if len(trainers) != 2:
            raise ValueError("FlashSAC LR autotuning requires exactly two trainers")
        self.config = config or ConfigFlashSACLRAutotune()
        self.batch = batch
        self.population = StateFlashSACPopulation(trainers)
        self.trainers = trainers
        self.device = self.population.device
        worlds = int(rollout_world_count)
        if worlds <= 1:
            raise ValueError("rollout_world_count must exceed one")
        self.rollout_world_count = worlds
        if self.config.challenger_fraction == 0.0:
            self.champion_worlds = worlds
        else:
            self.champion_worlds = max(
                1,
                min(worlds - 1, round(worlds * (1.0 - self.config.challenger_fraction))),
            )
        self.challenger_worlds = worlds - self.champion_worlds
        self._routed_actions = wp.empty((worlds, trainers[0].action_dim), dtype=wp.float32, device=self.device)
        self._challenger_enabled = wp.ones(1, dtype=wp.int32, device=self.device)
        episodes = int(self.config.evaluation_episodes)
        self._champion_scores = wp.empty(episodes, dtype=wp.float32, device=self.device)
        self._challenger_scores = wp.empty(episodes, dtype=wp.float32, device=self.device)
        self._challenger_safe = wp.ones(1, dtype=wp.int32, device=self.device)
        self._paired_delta = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._evaluation_valid = wp.ones(1, dtype=wp.int32, device=self.device)
        self._champion_index = wp.array([0], dtype=wp.int32, device=self.device)
        self._challenger_index = wp.array([1], dtype=wp.int32, device=self.device)

        first = trainers[0]
        self.default_rates = np.asarray(
            [first.config.actor_lr, first.config.critic_lr, first.config.alpha_lr], dtype=np.float64
        )
        if not np.all(np.isfinite(self.default_rates)) or np.any(self.default_rates <= 0.0):
            raise ValueError("configured FlashSAC learning rates must be positive and finite")
        self.member_rates = np.stack((self.default_rates, self.default_rates))
        self.default_target_update_rate = float(first.config.tau)
        if not math.isfinite(self.default_target_update_rate) or not 0.0 < self.default_target_update_rate <= 1.0:
            raise ValueError("configured FlashSAC target update rate must be finite and in (0, 1]")
        self.member_target_update_rates = np.full(2, self.default_target_update_rate, dtype=np.float64)
        self.perturbation_factor = float(self.config.initial_perturbation_factor)
        self.default_policy_frequency = int(first.config.policy_frequency)
        if self.default_policy_frequency <= 0:
            raise ValueError("configured FlashSAC policy frequency must be positive")
        self.policy_frequency_choices = (self.default_policy_frequency,)
        self.member_policy_frequencies = np.full(2, self.default_policy_frequency, dtype=np.int64)
        self.search_round = 0
        self._proposal_rejections = 0
        self._candidate_evidence_windows = 0
        self._candidate_score_sums = np.zeros(2, dtype=np.float64)
        self._candidate_termination_rate_sums = np.zeros(2, dtype=np.float64)
        self._candidate_decision_windows = 0
        self.consecutive_wins = 0
        self.stagnant_windows = 0
        self.evaluation_count = 0
        self.converged = False
        self.reopen_count = 0
        self._captured_population_graph: object | None = None
        self.best_valid = False
        self.best_score = -math.inf
        self.best_termination_rate = math.inf
        self.best_rates = self.member_rates[0].copy()
        self.best_target_update_rate = float(self.member_target_update_rates[0])
        self.best_member = 0
        self.best_policy_frequency = int(self.member_policy_frequencies[0])
        self._best_candidate_scores = np.full(2, -math.inf, dtype=np.float64)
        self._best_candidate_termination_rates = np.full(2, math.inf, dtype=np.float64)
        self._best_candidate_windows = np.zeros(2, dtype=np.int64)
        self._captured_population_critic_graph: object | None = None
        self._captured_single_graph: object | None = None

        if single_trainer is None:
            single_trainer = TrainerFlashSAC(
                obs_dim=first.obs_dim,
                action_dim=first.action_dim,
                config=first.config,
                device=self.device,
                seed=first.seed,
            )
            single_trainer.copy_training_state_from(first)
        self.single_trainer = single_trainer
        self._propose_challenger()

    @property
    def population_active(self) -> bool:
        """Return whether champion and challenger updates are still active."""

        return not self.converged

    @property
    def challenger_enabled(self) -> bool:
        """Return whether rollout worlds currently use the challenger."""

        return bool(self._challenger_enabled.numpy()[0])

    def configure_policy_frequency_family(self, update_span: int, *, allow_search: bool) -> None:
        """Configure policy cadences represented by pre-captured update graphs."""

        span = int(update_span)
        if span <= 0:
            raise ValueError("policy-frequency update span must be positive")
        if span % self.default_policy_frequency != 0:
            raise ValueError("captured update span must divide the configured policy frequency")
        choices = tuple(value for value in self.config.policy_frequency_choices if span % value == 0)
        if self.default_policy_frequency not in choices:
            choices = tuple(sorted((*choices, self.default_policy_frequency)))
        if not allow_search:
            choices = (self.default_policy_frequency,)
        self.policy_frequency_choices = choices
        for member in range(2):
            if int(self.member_policy_frequencies[member]) not in choices:
                self.member_policy_frequencies[member] = self.default_policy_frequency
        if self.best_policy_frequency not in choices:
            self.best_policy_frequency = self.default_policy_frequency

    def _set_member_rates(self) -> None:
        rates = self.member_rates.astype(np.float32)
        self.population.actor_optimizer.set_pbt_lrs(rates[:, 0])
        self.population.critic_optimizer.set_pbt_lrs(np.repeat(rates[:, 1], 2))
        self.population.alpha_optimizer.set_pbt_lrs(rates[:, 2])
        self.population.scalar_state["_device_target_update_rate"].assign(
            self.member_target_update_rates.astype(np.float32).reshape(2, 1)
        )
        for trainer, frequency in zip(self.trainers, self.member_policy_frequencies, strict=True):
            trainer.set_pbt_policy_frequency(int(frequency))

    def _clear_candidate_comparison(self) -> None:
        """Reset host evidence for the next fixed-address challenger."""

        self._candidate_evidence_windows = 0
        self._candidate_score_sums.fill(0.0)
        self._candidate_termination_rate_sums.fill(0.0)
        self._candidate_decision_windows = 0

    def _propose_challenger(self) -> None:
        lower = self.default_rates * float(self.config.multiplier_bounds[0])
        upper = self.default_rates * float(self.config.multiplier_bounds[1])
        target_bounds = self.config.target_update_rate_multiplier_bounds
        target_lower = self.default_target_update_rate * float(target_bounds[0])
        target_upper = self.default_target_update_rate * float(target_bounds[1])
        coordinate_count = 6
        for _attempt in range(coordinate_count * 2):
            proposal = self.member_rates[0].copy()
            proposal_target_update_rate = float(self.member_target_update_rates[0])
            proposal_policy_frequency = int(self.member_policy_frequencies[0])
            phase = self.search_round % coordinate_count
            direction = _proposal_direction(self.search_round, coordinate_count)
            factor = math.exp(direction * math.log(self.perturbation_factor))
            if phase < 4:
                selected = range(3) if phase == 0 else (phase - 1,)
                for rate_index in selected:
                    proposal[rate_index] = np.clip(proposal[rate_index] * factor, lower[rate_index], upper[rate_index])
            elif phase == 4:
                proposal_target_update_rate = float(
                    np.clip(proposal_target_update_rate * factor, target_lower, target_upper)
                )
            else:
                current_index = self.policy_frequency_choices.index(proposal_policy_frequency)
                offset = 1 if direction > 0.0 else -1
                proposal_index = min(max(current_index + offset, 0), len(self.policy_frequency_choices) - 1)
                proposal_policy_frequency = self.policy_frequency_choices[proposal_index]
            self.search_round += 1
            if (
                not np.array_equal(proposal, self.member_rates[0])
                or proposal_target_update_rate != self.member_target_update_rates[0]
                or proposal_policy_frequency != self.member_policy_frequencies[0]
            ):
                break
        else:
            raise RuntimeError("bounded hyperparameter search could not produce a distinct challenger")
        self.member_rates[1] = proposal
        self.member_target_update_rates[1] = proposal_target_update_rate
        self.member_policy_frequencies[1] = proposal_policy_frequency
        self._clear_candidate_comparison()
        self._best_candidate_windows[1] = 0
        self._set_member_rates()

    def begin_search_window(self) -> None:
        """Re-enable the reset challenger for a new rollout window."""

        if not self.converged:
            self._challenger_enabled.assign(np.asarray([1], dtype=np.int32))

    def reopen_search(self) -> None:
        """Restart paired search from the setup-owned converged learner."""

        if not self.converged:
            raise RuntimeError("FlashSAC LR search is already active")
        for trainer in self.trainers:
            trainer.copy_training_state_from(self.single_trainer)
        self.member_rates[:] = self.best_rates
        self.member_target_update_rates[:] = self.best_target_update_rate
        self.member_policy_frequencies[:] = self.best_policy_frequency
        self.consecutive_wins = 0
        self.stagnant_windows = 0
        self._clear_candidate_comparison()
        self._best_candidate_windows[:] = 0
        self.converged = False
        self.reopen_count += 1
        self._propose_challenger()
        self._challenger_enabled.assign(np.asarray([1], dtype=np.int32))

    def route_split_actions(
        self,
        champion_actions: wp.array2d[wp.float32],
        challenger_actions: wp.array2d[wp.float32],
    ) -> wp.array2d[wp.float32]:
        """Route disjoint rollout worlds through champion or challenger actions."""

        expected = self._routed_actions.shape
        if champion_actions.shape != expected or challenger_actions.shape != expected:
            raise ValueError("split rollout action arrays do not match setup shape")
        wp.launch(
            _route_split_actions_kernel,
            dim=expected,
            inputs=[champion_actions, challenger_actions, self.champion_worlds, self._challenger_enabled],
            outputs=[self._routed_actions],
            device=self.device,
        )
        return self._routed_actions

    def capture(self, *, seed: int) -> None:
        """Capture fixed-address population and convergence learner graphs."""
        self.configure_policy_frequency_family(self.default_policy_frequency, allow_search=False)

        if self._captured_population_graph is not None:
            return
        self._captured_population_graph = self.population.capture_fused_update(self.batch, seed=int(seed))
        self._captured_population_critic_graph = self.population.capture_fused_critic_update(self.batch, seed=int(seed))
        self._captured_single_graph = self.single_trainer.capture_update_graph(self.batch, seed=int(seed))

    def capture_overlap(
        self,
        env: EnvFlashSAC,
        replay: BufferReplayFlashSAC,
        *,
        updates_per_step: int = 2,
        interactions_per_launch: int = 2,
        seed: int = 0,
        population_backend: str = "parallel",
    ) -> GraphFlashSACLRAutotune:
        """Capture split rollout and identical-batch learner overlap."""

        backend = str(population_backend)
        if backend == "parallel":
            return capture_lr_autotune_parallel_overlap(
                self,
                env,
                replay,
                updates_per_step=int(updates_per_step),
                interactions_per_launch=int(interactions_per_launch),
                seed=int(seed),
            )
        if backend != "fused":
            raise ValueError("population_backend must be 'fused' or 'parallel'")
        return capture_lr_autotune_overlap(
            self,
            env,
            replay,
            updates_per_step=int(updates_per_step),
            interactions_per_launch=int(interactions_per_launch),
            seed=int(seed),
        )

    def launch(self) -> StatsSACUpdate | tuple[StatsSACUpdate, StatsSACUpdate]:
        """Replay one captured learner update in population or converged mode."""

        if (
            self._captured_population_graph is None
            or self._captured_population_critic_graph is None
            or self._captured_single_graph is None
        ):
            raise RuntimeError("capture the FlashSAC LR autotuner before launch")
        if self.converged:
            return self._captured_single_graph.launch(read_stats=False)
        include_actor = self.trainers[0]._gradient_update_count % int(self.trainers[0].config.policy_frequency) == 0
        graph = self._captured_population_graph if include_actor else self._captured_population_critic_graph
        wp.capture_launch(graph)
        for trainer in self.trainers:
            trainer._gradient_update_count += 1
            trainer._update_count += 1
        return (StatsSACUpdate(0.0, 0.0, 0.0, 0.0), StatsSACUpdate(0.0, 0.0, 0.0, 0.0))

    def _reset_challenger(self) -> None:
        self.population.copy_member(self._champion_index, self._challenger_index)
        self.member_rates[1] = self.member_rates[0]
        self.member_target_update_rates[1] = self.member_target_update_rates[0]
        self.member_policy_frequencies[1] = self.member_policy_frequencies[0]
        self.consecutive_wins = 0
        last_round = max(0, self.search_round - 1)
        self._proposal_rejections += 1
        distance = self.perturbation_factor - 1.0
        self.perturbation_factor = max(float(self.config.minimum_perturbation_factor), 1.0 + distance * 0.75)
        if self.config.exploit_after_candidate:
            self.search_round = last_round + 1
            self._proposal_rejections = 0
        elif self._proposal_rejections == 1:
            self.search_round = last_round + 6
        else:
            self.search_round = last_round + 1
            self._proposal_rejections = 0
        self._propose_challenger()

    def _promote_challenger(self) -> None:
        self.population.copy_member(self._challenger_index, self._champion_index)
        self.member_rates[0] = self.member_rates[1]
        self.member_target_update_rates[0] = self.member_target_update_rates[1]
        self.member_policy_frequencies[0] = self.member_policy_frequencies[1]
        self.population.copy_member(self._champion_index, self._challenger_index)
        self.member_rates[1] = self.member_rates[0]
        self.member_target_update_rates[1] = self.member_target_update_rates[0]
        self.member_policy_frequencies[1] = self.member_policy_frequencies[0]
        self.consecutive_wins = 0
        self.stagnant_windows = 0
        self.search_round = max(0, self.search_round - 1)
        self._proposal_rejections = 0
        self._best_candidate_windows[:] = 0
        self._propose_challenger()

    def _quality_better(
        self, score: float, termination_rate: float, reference_score: float, reference_rate: float
    ) -> bool:
        if termination_rate < reference_rate - float(self.config.termination_rate_margin):
            return True
        return termination_rate <= reference_rate + float(
            self.config.termination_rate_margin
        ) and score >= reference_score + float(self.config.improvement_margin)

    def _challenger_better(
        self,
        champion_score: float,
        challenger_score: float,
        champion_termination_rate: float,
        challenger_termination_rate: float,
    ) -> bool:
        """Compare paired quality using safety, absolute, and relative margins."""

        termination_margin = float(self.config.termination_rate_margin)
        if challenger_termination_rate < champion_termination_rate - termination_margin:
            return True
        if challenger_termination_rate > champion_termination_rate + termination_margin:
            return False
        delta = challenger_score - champion_score
        if delta >= float(self.config.improvement_margin):
            return True
        scale = max(abs(champion_score), float(self.config.minimum_effect_delta))
        return delta >= float(self.config.minimum_effect_delta) and (
            delta / scale >= float(self.config.relative_improvement_margin)
        )

    def _consider_best(self, member: int, score: float, termination_rate: float) -> None:
        """Confirm each member's best snapshot across independent windows."""

        if not math.isfinite(score) or not math.isfinite(termination_rate):
            self._best_candidate_windows[member] = 0
            return
        if self.best_valid and not self._quality_better(
            score, termination_rate, self.best_score, self.best_termination_rate
        ):
            self._best_candidate_windows[member] = 0
            return
        if self._best_candidate_windows[member] == 0:
            self._best_candidate_scores[member] = score
            self._best_candidate_termination_rates[member] = termination_rate
            self._best_candidate_windows[member] = 1
        elif termination_rate <= self._best_candidate_termination_rates[member] + float(
            self.config.termination_rate_margin
        ) and score >= self._best_candidate_scores[member] - float(self.config.regression_margin):
            self._best_candidate_scores[member] = score
            self._best_candidate_termination_rates[member] = termination_rate
            self._best_candidate_windows[member] += 1
        else:
            self._best_candidate_scores[member] = score
            self._best_candidate_termination_rates[member] = termination_rate
            self._best_candidate_windows[member] = 1
        if self._best_candidate_windows[member] >= int(self.config.promotion_windows):
            self._snapshot_member_as_best(member, score, termination_rate)
            self._best_candidate_windows[member] = 0

    def _snapshot_member_as_best(self, member: int, score: float, termination_rate: float) -> None:
        """Preserve one live member in the preallocated best-state trainer."""

        self.single_trainer.copy_training_state_from(self.trainers[member])
        self.best_valid = True
        self.best_member = member
        self.best_score = score
        self.best_termination_rate = termination_rate
        self.best_rates[:] = self.member_rates[member]
        self.best_target_update_rate = float(self.member_target_update_rates[member])
        self.best_policy_frequency = int(self.member_policy_frequencies[member])

    def _best_regressed(self, score: float, termination_rate: float) -> bool:
        if not self.best_valid:
            return False
        if termination_rate > self.best_termination_rate + float(self.config.termination_rate_margin):
            return True
        return termination_rate <= self.best_termination_rate + float(
            self.config.termination_rate_margin
        ) and score < self.best_score - float(self.config.regression_margin)

    def _restore_best(self) -> None:
        for trainer in self.trainers:
            trainer.copy_training_state_from(self.single_trainer)
        self.member_rates[0] = self.best_rates
        self.member_rates[1] = self.best_rates
        self.member_target_update_rates[0] = self.best_target_update_rate
        self.member_target_update_rates[1] = self.best_target_update_rate
        self.member_policy_frequencies[0] = self.best_policy_frequency
        self.member_policy_frequencies[1] = self.best_policy_frequency
        self.consecutive_wins = 0
        self._best_candidate_windows[:] = 0
        self._propose_challenger()

    def _converge_to_single(self) -> None:
        if not self.best_valid:
            self.single_trainer.copy_training_state_from(self.trainers[0])
            self.best_rates[:] = self.member_rates[0]
            self.best_target_update_rate = float(self.member_target_update_rates[0])
            self.best_policy_frequency = int(self.member_policy_frequencies[0])
        self.single_trainer.set_pbt_learning_rates(*self.best_rates)
        self.converged = True
        self.single_trainer.set_pbt_target_update_rate(self.best_target_update_rate)
        self.single_trainer.set_pbt_policy_frequency(self.best_policy_frequency)
        self._challenger_enabled.assign(np.asarray([0], dtype=np.int32))

    def _converge_from_live_champion(self, score: float, termination_rate: float) -> None:
        """Keep safe champion progress when ending an ordinary probe."""

        if math.isfinite(score) and math.isfinite(termination_rate):
            self._snapshot_member_as_best(0, score, termination_rate)
        self._converge_to_single()

    def evaluate_paired(
        self,
        champion_scores: np.ndarray,
        challenger_scores: np.ndarray,
        *,
        challenger_safe: bool = True,
        champion_termination_rate: float = 0.0,
        challenger_termination_rate: float = 0.0,
    ) -> ResultFlashSACLRAutotune:
        """Apply deterministic paired held-out evaluation and hysteresis."""

        expected = (int(self.config.evaluation_episodes),)
        champion = np.asarray(champion_scores, dtype=np.float32)
        challenger = np.asarray(challenger_scores, dtype=np.float32)
        if champion.shape != expected or challenger.shape != expected:
            raise ValueError("paired evaluation arrays do not match evaluation_episodes")
        self._champion_scores.assign(champion)
        self._challenger_scores.assign(challenger)
        relative_safe = (
            math.isfinite(champion_termination_rate)
            and math.isfinite(challenger_termination_rate)
            and challenger_termination_rate <= champion_termination_rate + float(self.config.termination_rate_margin)
        )
        self._challenger_safe.assign(np.asarray([int(challenger_safe and relative_safe)], dtype=np.int32))
        wp.launch(
            _paired_evaluation_kernel,
            dim=1,
            inputs=[self._champion_scores, self._challenger_scores, self._challenger_safe],
            outputs=[self._paired_delta, self._evaluation_valid],
            device=self.device,
        )
        delta = float(self._paired_delta.numpy()[0])
        valid = bool(self._evaluation_valid.numpy()[0])
        self.evaluation_count += 1
        champion_score = float(np.mean(champion))
        challenger_score = float(np.mean(challenger))
        self._consider_best(0, champion_score, float(champion_termination_rate))
        challenger_has_evidence = max(champion_score, challenger_score) >= float(
            self.config.informative_score_threshold
        ) or self._candidate_evidence_windows + 1 >= int(self.config.minimum_evidence_windows)
        if challenger_safe and relative_safe and challenger_has_evidence:
            self._consider_best(1, challenger_score, float(challenger_termination_rate))
        else:
            self._best_candidate_windows[1] = 0
        if self._best_regressed(champion_score, float(champion_termination_rate)):
            self.stagnant_windows += 1
            self._restore_best()
            if self.config.exploit_after_candidate:
                self._converge_to_single()
            return ResultFlashSACLRAutotune(delta, "rollback", self.consecutive_wins, self.converged)

        action = "continue"
        if not valid:
            self._challenger_enabled.assign(np.asarray([0], dtype=np.int32))
            self.stagnant_windows += 1
            self._reset_challenger()
            action = "safety_fallback"
        elif max(champion_score, challenger_score) < float(
            self.config.informative_score_threshold
        ) and self._candidate_evidence_windows + 1 < int(self.config.minimum_evidence_windows):
            self._candidate_evidence_windows += 1
            self._candidate_score_sums += (champion_score, challenger_score)
            self._candidate_termination_rate_sums += (
                float(champion_termination_rate),
                float(challenger_termination_rate),
            )
            self.consecutive_wins = 0
            action = "gather_evidence"
        elif self._candidate_evidence_windows > 0 and self._candidate_decision_windows == 0:
            self._candidate_evidence_windows += 1
            self._candidate_score_sums += (champion_score, challenger_score)
            self._candidate_termination_rate_sums += (
                float(champion_termination_rate),
                float(challenger_termination_rate),
            )
            evidence_count = float(self._candidate_evidence_windows)
            mean_scores = self._candidate_score_sums / evidence_count
            mean_termination_rates = self._candidate_termination_rate_sums / evidence_count
            current_stable = challenger_score >= champion_score - float(
                self.config.regression_margin
            ) and challenger_termination_rate <= champion_termination_rate + float(self.config.termination_rate_margin)
            if current_stable and self._challenger_better(*mean_scores, *mean_termination_rates):
                self._candidate_score_sums[:] = mean_scores
                self._candidate_termination_rate_sums[:] = mean_termination_rates
                self._candidate_decision_windows = 1
                self.consecutive_wins = 1
                action = "continue"
            else:
                self.stagnant_windows += 1
                self._reset_challenger()
                action = "reject"
        else:
            current_better = self._challenger_better(
                champion_score,
                challenger_score,
                float(champion_termination_rate),
                float(challenger_termination_rate),
            )
            if current_better or self._candidate_decision_windows > 0:
                self._candidate_evidence_windows += 1
                self._candidate_score_sums += (champion_score, challenger_score)
                self._candidate_termination_rate_sums += (
                    float(champion_termination_rate),
                    float(challenger_termination_rate),
                )
                self._candidate_decision_windows += 1
                if current_better:
                    self.consecutive_wins += 1
                if self._candidate_decision_windows >= int(self.config.promotion_windows):
                    window_count = float(self._candidate_decision_windows)
                    mean_scores = self._candidate_score_sums / window_count
                    mean_termination_rates = self._candidate_termination_rate_sums / window_count
                    current_stable = challenger_score >= champion_score - float(
                        self.config.regression_margin
                    ) and challenger_termination_rate <= champion_termination_rate + float(
                        self.config.termination_rate_margin
                    )
                    if current_stable and self._challenger_better(*mean_scores, *mean_termination_rates):
                        self._promote_challenger()
                        if self.config.exploit_after_candidate:
                            # Repeated paired evidence justifies switching to the preallocated P1 path.
                            self.single_trainer.copy_training_state_from(self.trainers[0])
                            self.best_valid = True
                            self.best_member = 0
                            self.best_score = challenger_score
                            self.best_termination_rate = float(challenger_termination_rate)
                            self.best_rates[:] = self.member_rates[0]
                            self.best_target_update_rate = float(self.member_target_update_rates[0])
                            self.best_policy_frequency = int(self.member_policy_frequencies[0])
                            self._converge_to_single()
                        action = "promote"
                    else:
                        self.stagnant_windows += 1
                        self._reset_challenger()
                        action = "reject"
            else:
                self._candidate_evidence_windows += 1
                self.stagnant_windows += 1
                self._reset_challenger()
                action = "reject"
        if self.config.exploit_after_candidate and action in ("reject", "safety_fallback"):
            self._converge_from_live_champion(champion_score, float(champion_termination_rate))
        if self.evaluation_count >= int(self.config.minimum_search_windows) and self.stagnant_windows >= int(
            self.config.convergence_windows
        ):
            self._converge_to_single()
            action = "converge"
        return ResultFlashSACLRAutotune(delta, action, self.consecutive_wins, self.converged)

    def finalize_best(
        self,
        *,
        policy: str = "best_confirmed",
        live_score: float | None = None,
        live_termination_rate: float | None = None,
    ) -> None:
        """Finalize search into the setup-owned single learner."""

        if policy == "none":
            return
        if policy == "live":
            if live_score is None or live_termination_rate is None:
                raise ValueError("live finalization requires score and termination rate")
            self._snapshot_member_as_best(0, float(live_score), float(live_termination_rate))
        elif policy == "best_confirmed":
            if not self.best_valid:
                raise RuntimeError("no repeatedly confirmed best FlashSAC policy is available")
        else:
            raise ValueError("finalization policy must be 'best_confirmed', 'live', or 'none'")
        for trainer in self.trainers:
            trainer.copy_training_state_from(self.single_trainer)
        self.member_rates[:] = self.best_rates
        self.member_target_update_rates[:] = self.best_target_update_rate
        self.member_policy_frequencies[:] = self.best_policy_frequency
        self._converge_to_single()

    def state_arrays(self) -> tuple[wp.array[Any], ...]:
        """Return setup-owned controller and population arrays for pointer audits."""

        return (
            *self.population.state_arrays(),
            *self.population.update_buffer_arrays(),
            self._routed_actions,
            self._challenger_enabled,
            self._champion_scores,
            self._challenger_scores,
            self._challenger_safe,
            self._paired_delta,
            self._evaluation_valid,
            self._champion_index,
            self._challenger_index,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save learner companions and deterministic LR-search state."""

        wp.synchronize_device(self.device)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        champion_path = output.with_name(f"{output.stem}.champion.npz")
        challenger_path = output.with_name(f"{output.stem}.challenger.npz")
        single_path = output.with_name(f"{output.stem}.single.npz")
        self.trainers[0].save_checkpoint(champion_path)
        self.trainers[1].save_checkpoint(challenger_path)
        self.single_trainer.save_checkpoint(single_path)
        data: dict[str, np.ndarray] = {
            "champion_path": np.asarray(champion_path.name),
            "challenger_path": np.asarray(challenger_path.name),
            "single_path": np.asarray(single_path.name),
            "member_rates": self.member_rates,
            "default_rates": self.default_rates,
            "perturbation_factor": np.asarray(self.perturbation_factor),
            "member_target_update_rates": self.member_target_update_rates,
            "default_target_update_rate": np.asarray(self.default_target_update_rate),
            "best_target_update_rate": np.asarray(self.best_target_update_rate),
            "member_policy_frequencies": self.member_policy_frequencies,
            "default_policy_frequency": np.asarray(self.default_policy_frequency, dtype=np.int64),
            "policy_frequency_choices": np.asarray(self.policy_frequency_choices, dtype=np.int64),
            "best_policy_frequency": np.asarray(self.best_policy_frequency, dtype=np.int64),
            "search_round": np.asarray(self.search_round, dtype=np.int64),
            "consecutive_wins": np.asarray(self.consecutive_wins, dtype=np.int64),
            "stagnant_windows": np.asarray(self.stagnant_windows, dtype=np.int64),
            "best_valid": np.asarray(self.best_valid),
            "best_score": np.asarray(self.best_score),
            "best_termination_rate": np.asarray(self.best_termination_rate),
            "best_rates": self.best_rates,
            "best_member": np.asarray(self.best_member, dtype=np.int64),
            "proposal_rejections": np.asarray(self._proposal_rejections, dtype=np.int64),
            "candidate_evidence_windows": np.asarray(self._candidate_evidence_windows, dtype=np.int64),
            "candidate_score_sums": self._candidate_score_sums,
            "candidate_termination_rate_sums": self._candidate_termination_rate_sums,
            "candidate_decision_windows": np.asarray(self._candidate_decision_windows, dtype=np.int64),
            "best_candidate_scores": self._best_candidate_scores,
            "best_candidate_termination_rates": self._best_candidate_termination_rates,
            "best_candidate_windows": self._best_candidate_windows,
            "evaluation_count": np.asarray(self.evaluation_count, dtype=np.int64),
            "converged": np.asarray(self.converged),
            "reopen_count": np.asarray(self.reopen_count, dtype=np.int64),
            "challenger_enabled": self._challenger_enabled.numpy(),
            "rollout_world_count": np.asarray(self.rollout_world_count, dtype=np.int64),
        }
        for name, value in asdict(self.config).items():
            data[f"config_{name}"] = np.asarray(value)
        np.savez(output, **data)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        batch: BatchSAC,
        *,
        device: wp.context.Devicelike = None,
    ) -> ControllerFlashSACLRAutotune:
        """Restore learner and search state into newly allocated setup storage."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            config = ConfigFlashSACLRAutotune(
                evaluation_episodes=int(data["config_evaluation_episodes"]),
                challenger_fraction=float(data["config_challenger_fraction"]),
                challenger_action_rms_limit=float(data["config_challenger_action_rms_limit"]),
                challenger_action_max_limit=float(data["config_challenger_action_max_limit"]),
                initial_perturbation_factor=float(data["config_initial_perturbation_factor"]),
                minimum_perturbation_factor=float(data["config_minimum_perturbation_factor"]),
                multiplier_bounds=tuple(float(value) for value in data["config_multiplier_bounds"]),
                improvement_margin=float(data["config_improvement_margin"]),
                minimum_evidence_windows=int(data["config_minimum_evidence_windows"])
                if "config_minimum_evidence_windows" in data
                else 1,
                informative_score_threshold=float(data["config_informative_score_threshold"])
                if "config_informative_score_threshold" in data
                else 0.0,
                relative_improvement_margin=float(data["config_relative_improvement_margin"])
                if "config_relative_improvement_margin" in data
                else 0.0,
                minimum_effect_delta=float(data["config_minimum_effect_delta"])
                if "config_minimum_effect_delta" in data
                else 0.0,
                target_update_rate_multiplier_bounds=tuple(
                    float(value) for value in data["config_target_update_rate_multiplier_bounds"]
                )
                if "config_target_update_rate_multiplier_bounds" in data
                else (0.5, 2.0),
                policy_frequency_choices=tuple(int(value) for value in data["config_policy_frequency_choices"])
                if "config_policy_frequency_choices" in data
                else (1, 2, 4),
                promotion_windows=int(data["config_promotion_windows"]),
                exploit_after_candidate=bool(data["config_exploit_after_candidate"])
                if "config_exploit_after_candidate" in data
                else bool(data["config_exploit_after_promotion"])
                if "config_exploit_after_promotion" in data
                else False,
                regression_margin=float(data["config_regression_margin"]),
                convergence_windows=int(data["config_convergence_windows"]),
                termination_rate_margin=float(data["config_termination_rate_margin"]),
                minimum_search_windows=int(data["config_minimum_search_windows"]),
                reopen_stagnation_windows=int(data["config_reopen_stagnation_windows"])
                if "config_reopen_stagnation_windows" in data
                else 4,
                seed=int(data["config_seed"]),
            )
            champion = TrainerFlashSAC.load_checkpoint(source.parent / str(data["champion_path"].item()), device=device)
            challenger = TrainerFlashSAC.load_checkpoint(
                source.parent / str(data["challenger_path"].item()), device=device
            )
            single = TrainerFlashSAC.load_checkpoint(source.parent / str(data["single_path"].item()), device=device)
            controller = cls(
                (champion, challenger),
                batch,
                rollout_world_count=int(data["rollout_world_count"]),
                config=config,
                single_trainer=single,
            )
            controller.member_rates[:] = data["member_rates"]
            controller.default_rates[:] = data["default_rates"]
            controller.perturbation_factor = float(data["perturbation_factor"])
            if "member_target_update_rates" in data:
                controller.member_target_update_rates[:] = data["member_target_update_rates"]
                controller.default_target_update_rate = float(data["default_target_update_rate"])
            else:
                controller.member_target_update_rates[:] = controller.default_target_update_rate
            controller.search_round = int(data["search_round"])
            controller.consecutive_wins = int(data["consecutive_wins"])
            controller.stagnant_windows = int(data["stagnant_windows"])
            controller.evaluation_count = int(data["evaluation_count"])
            controller.converged = bool(data["converged"])
            controller._challenger_enabled.assign(data["challenger_enabled"])
            controller.reopen_count = int(data["reopen_count"]) if "reopen_count" in data else 0
            controller._set_member_rates()
            controller._proposal_rejections = int(data["proposal_rejections"])
            controller._candidate_evidence_windows = (
                int(data["candidate_evidence_windows"]) if "candidate_evidence_windows" in data else 0
            )
            if "candidate_score_sums" in data:
                controller._candidate_score_sums[:] = data["candidate_score_sums"]
                controller._candidate_termination_rate_sums[:] = data["candidate_termination_rate_sums"]
                controller._candidate_decision_windows = int(data["candidate_decision_windows"])
            else:
                controller._candidate_score_sums.fill(0.0)
                controller._candidate_termination_rate_sums.fill(0.0)
                controller._candidate_decision_windows = 0
            controller.best_valid = bool(data["best_valid"])
            controller.best_score = float(data["best_score"])
            controller.best_termination_rate = float(data["best_termination_rate"])
            if "member_policy_frequencies" in data:
                controller.member_policy_frequencies[:] = data["member_policy_frequencies"]
                controller.default_policy_frequency = int(data["default_policy_frequency"])
                controller.policy_frequency_choices = tuple(int(value) for value in data["policy_frequency_choices"])
            else:
                controller.member_policy_frequencies[:] = controller.default_policy_frequency
                controller.policy_frequency_choices = (controller.default_policy_frequency,)
            controller.best_policy_frequency = (
                int(data["best_policy_frequency"])
                if "best_policy_frequency" in data
                else controller.default_policy_frequency
            )

            controller.best_rates[:] = data["best_rates"]
            controller.best_member = int(data["best_member"])
            controller.best_target_update_rate = (
                float(data["best_target_update_rate"])
                if "best_target_update_rate" in data
                else controller.default_target_update_rate
            )
            if "best_candidate_scores" in data:
                controller._best_candidate_scores[:] = data["best_candidate_scores"]
                controller._best_candidate_termination_rates[:] = data["best_candidate_termination_rates"]
                controller._best_candidate_windows[:] = data["best_candidate_windows"]
            else:
                member = int(data["best_candidate_member"])
                if member >= 0:
                    controller._best_candidate_scores[member] = float(data["best_candidate_score"])
                    controller._best_candidate_termination_rates[member] = float(
                        data["best_candidate_termination_rate"]
                    )
                    controller._best_candidate_windows[member] = int(data["best_candidate_windows"])
            return controller
