# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal champion/challenger learning-rate search for FlashSAC."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .flash_sac import BufferReplayFlashSAC, EnvFlashSAC, TrainerFlashSAC
from .flash_sac_population import StateFlashSACPopulation
from .sac import BatchSAC, StatsSACUpdate

if TYPE_CHECKING:
    from .flash_sac_autotune_overlap import GraphFlashSACLRAutotuneOverlap
    from .flash_sac_autotune_parallel import GraphFlashSACLRAutotuneParallelOverlap


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
    """Configure the internal bounded FlashSAC learning-rate search."""

    evaluation_episodes: int = 8
    challenger_fraction: float = 0.10
    initial_perturbation_factor: float = 1.50
    challenger_action_rms_limit: float = 0.35
    challenger_action_max_limit: float = 0.75
    minimum_perturbation_factor: float = 1.02
    multiplier_bounds: tuple[float, float] = (0.5, 2.0)
    improvement_margin: float = 0.01
    termination_rate_margin: float = 0.05
    regression_margin: float = 0.05
    promotion_windows: int = 2
    convergence_windows: int = 12
    minimum_search_windows: int = 12
    seed: int = 0

    def __post_init__(self) -> None:
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        if not 0.0 < self.challenger_fraction < 0.5:
            raise ValueError("challenger_fraction must be between zero and one half")
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
        if self.improvement_margin < 0.0:
            raise ValueError("improvement_margin must be non-negative")
        if self.termination_rate_margin < 0.0:
            raise ValueError("termination_rate_margin must be non-negative")
        if self.regression_margin < 0.0:
            raise ValueError("regression_margin must be non-negative")
        if self.promotion_windows <= 0 or self.convergence_windows <= 0 or self.minimum_search_windows <= 0:
            raise ValueError("window counts must be positive")


@dataclass(frozen=True)
class ResultFlashSACLRAutotune:
    """Describe one paired evaluation decision."""

    paired_delta: float
    action: str
    consecutive_wins: int
    converged: bool


class ControllerFlashSACLRAutotune:
    """Run a bounded two-member FlashSAC learning-rate search."""

    _RATE_NAMES = ("actor", "critic", "alpha")

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
        self.champion_worlds = max(1, min(worlds - 1, round(worlds * (1.0 - self.config.challenger_fraction))))
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
        self.perturbation_factor = float(self.config.initial_perturbation_factor)
        self.search_round = 0
        self._proposal_rejections = 0
        self.consecutive_wins = 0
        self.stagnant_windows = 0
        self.evaluation_count = 0
        self.converged = False
        self._captured_population_graph: object | None = None
        self.best_valid = False
        self.best_score = -math.inf
        self.best_termination_rate = math.inf
        self.best_rates = self.member_rates[0].copy()
        self.best_member = 0
        self._best_candidate_score = -math.inf
        self._best_candidate_termination_rate = math.inf
        self._best_candidate_windows = 0
        self._best_candidate_member = -1
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

    def _set_member_rates(self) -> None:
        rates = self.member_rates.astype(np.float32)
        self.population.actor_optimizer.set_pbt_lrs(rates[:, 0])
        self.population.critic_optimizer.set_pbt_lrs(np.repeat(rates[:, 1], 2))
        self.population.alpha_optimizer.set_pbt_lrs(rates[:, 2])

    def _propose_challenger(self) -> None:
        lower = self.default_rates * float(self.config.multiplier_bounds[0])
        upper = self.default_rates * float(self.config.multiplier_bounds[1])
        for _attempt in range(8):
            proposal = self.member_rates[0].copy()
            phase = self.search_round % 4
            selected = range(3) if phase == 0 else (phase - 1,)
            direction = 1.0 if ((self.search_round // 4 + int(self.config.seed)) % 2 == 0) else -1.0
            factor = math.exp(direction * math.log(self.perturbation_factor))
            for rate_index in selected:
                proposal[rate_index] = np.clip(proposal[rate_index] * factor, lower[rate_index], upper[rate_index])
            self.search_round += 1
            if not np.array_equal(proposal, self.member_rates[0]):
                break
        else:
            raise RuntimeError("bounded LR search could not produce a distinct challenger")
        self.member_rates[1] = proposal
        self._set_member_rates()

    def begin_search_window(self) -> None:
        """Re-enable the reset challenger for a new rollout window."""

        if not self.converged:
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
    ) -> GraphFlashSACLRAutotuneOverlap | GraphFlashSACLRAutotuneParallelOverlap:
        """Capture split rollout and identical-batch learner overlap."""

        backend = str(population_backend)
        if backend == "parallel":
            from .flash_sac_autotune_parallel import capture_lr_autotune_parallel_overlap  # noqa: PLC0415

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
        from .flash_sac_autotune_overlap import capture_lr_autotune_overlap  # noqa: PLC0415

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
        self.consecutive_wins = 0
        last_round = max(0, self.search_round - 1)
        self._proposal_rejections += 1
        distance = self.perturbation_factor - 1.0
        self.perturbation_factor = max(float(self.config.minimum_perturbation_factor), 1.0 + distance * 0.75)
        if self._proposal_rejections == 1:
            self.search_round = last_round + 4
        else:
            self.search_round = last_round + 1
            self._proposal_rejections = 0
        self._propose_challenger()

    def _promote_challenger(self) -> None:
        self.population.copy_member(self._challenger_index, self._champion_index)
        self.member_rates[0] = self.member_rates[1]
        self.population.copy_member(self._champion_index, self._challenger_index)
        self.member_rates[1] = self.member_rates[0]
        self.consecutive_wins = 0
        self.stagnant_windows = 0
        self.search_round = max(0, self.search_round - 1)
        self._proposal_rejections = 0
        self._propose_challenger()

    def _quality_better(
        self, score: float, termination_rate: float, reference_score: float, reference_rate: float
    ) -> bool:
        if termination_rate < reference_rate - float(self.config.termination_rate_margin):
            return True
        return termination_rate <= reference_rate + float(
            self.config.termination_rate_margin
        ) and score >= reference_score + float(self.config.improvement_margin)

    def _consider_best(self, member: int, score: float, termination_rate: float) -> None:
        if not math.isfinite(score) or not math.isfinite(termination_rate):
            self._best_candidate_windows = 0
            self._best_candidate_member = -1
            return
        if self.best_valid and not self._quality_better(
            score, termination_rate, self.best_score, self.best_termination_rate
        ):
            self._best_candidate_windows = 0
            self._best_candidate_member = -1
            return
        if self._best_candidate_windows == 0 or self._best_candidate_member != member:
            self._best_candidate_score = score
            self._best_candidate_termination_rate = termination_rate
            self._best_candidate_windows = 1
            self._best_candidate_member = member
        elif termination_rate <= self._best_candidate_termination_rate + float(
            self.config.termination_rate_margin
        ) and score >= self._best_candidate_score - float(self.config.improvement_margin):
            self._best_candidate_score = max(score, self._best_candidate_score)
            self._best_candidate_termination_rate = min(termination_rate, self._best_candidate_termination_rate)
            self._best_candidate_windows += 1
        else:
            self._best_candidate_score = score
            self._best_candidate_termination_rate = termination_rate
            self._best_candidate_windows = 1
        if self._best_candidate_windows >= int(self.config.promotion_windows):
            self.single_trainer.copy_training_state_from(self.trainers[member])
            self.best_valid = True
            self.best_member = member
            self.best_score = self._best_candidate_score
            self.best_termination_rate = self._best_candidate_termination_rate
            self.best_rates[:] = self.member_rates[member]
            self._best_candidate_windows = 0
            self._best_candidate_member = -1

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
        self.consecutive_wins = 0
        self._propose_challenger()

    def _converge_to_single(self) -> None:
        if not self.best_valid:
            self.single_trainer.copy_training_state_from(self.trainers[0])
            self.best_rates[:] = self.member_rates[0]
        self.single_trainer.set_pbt_learning_rates(*self.best_rates)
        self.converged = True
        self._challenger_enabled.assign(np.asarray([0], dtype=np.int32))

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
        candidate_member = 0
        candidate_score = champion_score
        candidate_termination_rate = float(champion_termination_rate)
        if (
            challenger_safe
            and relative_safe
            and math.isfinite(challenger_score)
            and (
                challenger_termination_rate < champion_termination_rate
                or (challenger_termination_rate == champion_termination_rate and challenger_score > champion_score)
            )
        ):
            candidate_member = 1
            candidate_score = challenger_score
            candidate_termination_rate = float(challenger_termination_rate)
        self._consider_best(candidate_member, candidate_score, candidate_termination_rate)
        if self._best_regressed(champion_score, float(champion_termination_rate)):
            self.stagnant_windows += 1
            self._restore_best()
            return ResultFlashSACLRAutotune(delta, "rollback", self.consecutive_wins, self.converged)

        action = "continue"
        if not valid:
            self._challenger_enabled.assign(np.asarray([0], dtype=np.int32))
            self.stagnant_windows += 1
            self._reset_challenger()
            action = "safety_fallback"
        elif delta >= float(self.config.improvement_margin):
            self.consecutive_wins += 1
            if self.consecutive_wins >= int(self.config.promotion_windows):
                self._promote_challenger()
                action = "promote"
        else:
            self.stagnant_windows += 1
            self._reset_challenger()
            action = "reject"
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
            self.single_trainer.copy_training_state_from(self.trainers[0])
            self.best_valid = True
            self.best_member = 0
            self.best_score = float(live_score)
            self.best_termination_rate = float(live_termination_rate)
            self.best_rates[:] = self.member_rates[0]
        elif policy == "best_confirmed":
            if not self.best_valid:
                raise RuntimeError("no repeatedly confirmed best FlashSAC policy is available")
        else:
            raise ValueError("finalization policy must be 'best_confirmed', 'live', or 'none'")
        for trainer in self.trainers:
            trainer.copy_training_state_from(self.single_trainer)
        self.member_rates[:] = self.best_rates
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
            "search_round": np.asarray(self.search_round, dtype=np.int64),
            "consecutive_wins": np.asarray(self.consecutive_wins, dtype=np.int64),
            "stagnant_windows": np.asarray(self.stagnant_windows, dtype=np.int64),
            "best_valid": np.asarray(self.best_valid),
            "best_score": np.asarray(self.best_score),
            "best_termination_rate": np.asarray(self.best_termination_rate),
            "best_rates": self.best_rates,
            "best_member": np.asarray(self.best_member, dtype=np.int64),
            "proposal_rejections": np.asarray(self._proposal_rejections, dtype=np.int64),
            "best_candidate_score": np.asarray(self._best_candidate_score),
            "best_candidate_termination_rate": np.asarray(self._best_candidate_termination_rate),
            "best_candidate_windows": np.asarray(self._best_candidate_windows, dtype=np.int64),
            "best_candidate_member": np.asarray(self._best_candidate_member, dtype=np.int64),
            "evaluation_count": np.asarray(self.evaluation_count, dtype=np.int64),
            "converged": np.asarray(self.converged),
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
                promotion_windows=int(data["config_promotion_windows"]),
                regression_margin=float(data["config_regression_margin"]),
                convergence_windows=int(data["config_convergence_windows"]),
                termination_rate_margin=float(data["config_termination_rate_margin"]),
                minimum_search_windows=int(data["config_minimum_search_windows"]),
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
            controller.search_round = int(data["search_round"])
            controller.consecutive_wins = int(data["consecutive_wins"])
            controller.stagnant_windows = int(data["stagnant_windows"])
            controller.evaluation_count = int(data["evaluation_count"])
            controller.converged = bool(data["converged"])
            controller._challenger_enabled.assign(data["challenger_enabled"])
            controller._set_member_rates()
            controller._proposal_rejections = int(data["proposal_rejections"])
            controller.best_valid = bool(data["best_valid"])
            controller.best_score = float(data["best_score"])
            controller.best_termination_rate = float(data["best_termination_rate"])
            controller.best_rates[:] = data["best_rates"]
            controller.best_member = int(data["best_member"])
            controller._best_candidate_score = float(data["best_candidate_score"])
            controller._best_candidate_termination_rate = float(data["best_candidate_termination_rate"])
            controller._best_candidate_windows = int(data["best_candidate_windows"])
            controller._best_candidate_member = int(data["best_candidate_member"])
            return controller
