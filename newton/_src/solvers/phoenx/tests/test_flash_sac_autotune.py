# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for internal FlashSAC learning-rate autotuning."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton.rl as public_rl
from newton._src.solvers.phoenx.benchmarks.run_flash_sac_autotune_g1 import _root_command_tracking_kernel
from newton._src.solvers.phoenx.rl_training.flash_sac import ConfigFlashSAC, TrainerFlashSAC
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune import (
    ConfigFlashSACLRAutotune,
    ControllerFlashSACLRAutotune,
    GraphFlashSACLRAutotune,
    _proposal_direction,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune_evaluation import (
    CadenceFlashSACLRAutotune,
    EvaluatorPairedFlashSAC,
    _bootstrap_ready,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune_parallel import _guard_challenger_actions_kernel
from newton._src.solvers.phoenx.rl_training.sac import BatchSAC
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class _AutotuneSmokeEnv:
    def __init__(self, obs: wp.array2d[wp.float32], next_obs: wp.array2d[wp.float32]):
        self.world_count = int(obs.shape[0])
        self.obs_dim = int(obs.shape[1])
        self.action_dim = 2
        self.policy_action_dim = 2
        self.device = obs.device
        self.obs = obs
        self._initial_obs = obs
        self.step_next_obs = next_obs
        self.step_terminateds = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_truncateds = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._rewards = wp.ones(self.world_count, dtype=wp.float32, device=self.device)
        self._dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.last_actions = wp.empty((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.capture_step_calls = 0

        self.sim_time = 0.0
        self.config = SimpleNamespace(frame_dt=0.01)

    def reset(self):
        self.obs = self._initial_obs
        return self.obs

    def step(self, actions: wp.array2d[wp.float32]):
        self.capture_step_calls += 1
        wp.copy(self.last_actions, actions)
        self.obs = self.step_next_obs
        return self.obs, self._rewards, self._dones


class TestFlashSACLRAutotune(unittest.TestCase):
    """Validate deterministic, allocation-stable LR search control."""

    def test_public_facade_builds_controller_from_trainer(self) -> None:
        """Build the public paired controller from one owned champion trainer."""

        device = require_cuda_graph_capture("FlashSAC LR autotune public facade")
        trainer = public_rl.TrainerFlashSAC(
            obs_dim=3,
            action_dim=2,
            config=public_rl.ConfigFlashSAC(
                actor_hidden_dim=4,
                actor_num_blocks=1,
                critic_hidden_dim=4,
                critic_num_blocks=1,
                distributional_atoms=5,
                sample_batch_size=4,
                buffer_min_length=8,
                buffer_max_length=64,
            ),
            device=device,
            seed=127,
        )
        controller = public_rl.ControllerFlashSACLRAutotune.from_trainer(
            trainer,
            rollout_world_count=8,
            config=public_rl.ConfigFlashSACLRAutotune(evaluation_episodes=4),
        )
        self.assertIs(controller.trainers[0], trainer)
        self.assertIsNot(controller.trainers[1], trainer)
        self.assertEqual(controller.batch.obs.shape, (4, 3))
        self.assertEqual(controller.rollout_world_count, 8)
        self.assertEqual(controller.champion_worlds, 8)
        self.assertEqual(controller.challenger_worlds, 0)
        for champion_state, challenger_state in zip(
            controller.trainers[0].actor.net.state_arrays(),
            controller.trainers[1].actor.net.state_arrays(),
            strict=True,
        ):
            np.testing.assert_array_equal(champion_state.numpy(), challenger_state.numpy())
        self.assertIs(public_rl.GraphFlashSACLRAutotune, GraphFlashSACLRAutotune)

    def test_bootstrap_waits_for_survivable_learning_signal(self) -> None:
        """Start paired search only after champion rollouts become informative."""

        self.assertFalse(_bootstrap_ready(0.04, 0.0, True, 0.05, 0.5))
        self.assertFalse(_bootstrap_ready(0.20, 0.75, True, 0.05, 0.5))
        self.assertFalse(_bootstrap_ready(0.20, 0.0, False, 0.05, 0.5))
        self.assertTrue(_bootstrap_ready(0.20, 0.0, True, 0.05, 0.5))

    def test_adaptive_confirmation_cadence(self) -> None:
        """Confirm promising evidence early and preserve full initial resource rungs."""

        cadence = CadenceFlashSACLRAutotune(
            controller=SimpleNamespace(),
            training_graph=SimpleNamespace(),
            evaluator=SimpleNamespace(),
            evaluation_interval=400,
            confirmation_interval=100,
            launch_count=25,
        )
        self.assertEqual(cadence._next_evaluation_launch, 425)

        cadence.launch_count = 425
        cadence._schedule_after(SimpleNamespace(action="continue"))
        self.assertEqual(cadence._next_evaluation_launch, 525)

        cadence.launch_count = 525
        cadence._schedule_after(SimpleNamespace(action="reject"))
        self.assertEqual(cadence._next_evaluation_launch, 925)

    def test_proposal_direction_probes_faster_learning_first(self) -> None:
        """Probe upward before reversing each bounded coordinate bracket."""

        self.assertEqual(_proposal_direction(0, 6), 1.0)
        self.assertEqual(_proposal_direction(5, 6), 1.0)
        self.assertEqual(_proposal_direction(6, 6), -1.0)

    def test_sparse_plateau_reopens_preallocated_search(self) -> None:
        """Reopen paired search only after sustained converged-policy stagnation."""

        controller = SimpleNamespace(
            converged=True,
            config=SimpleNamespace(
                improvement_margin=0.01,
                termination_rate_margin=0.05,
                reopen_stagnation_windows=2,
            ),
        )
        graph = SimpleNamespace(reopen_count=0)

        def reopen_search() -> None:
            graph.reopen_count += 1
            controller.converged = False

        graph.reopen_search = reopen_search
        cadence = CadenceFlashSACLRAutotune(
            controller=controller,
            training_graph=graph,
            evaluator=SimpleNamespace(),
            evaluation_interval=100,
            monitor_interval=400,
        )
        scores = EvaluatorPairedFlashSAC.Result(
            champion_scores=np.full(4, 0.5, dtype=np.float32),
            challenger_scores=np.full(4, 0.5, dtype=np.float32),
            champion_finite=True,
            challenger_finite=True,
            champion_termination_rate=0.0,
            challenger_termination_rate=0.0,
        )
        self.assertEqual(cadence._monitor_converged(scores).action, "monitor")
        self.assertEqual(cadence._monitor_converged(scores).action, "monitor")
        state = cadence.state()
        restored = CadenceFlashSACLRAutotune(
            controller=controller,
            training_graph=graph,
            evaluator=SimpleNamespace(),
            evaluation_interval=100,
            monitor_interval=400,
        )
        restored.restore_state(state)
        self.assertEqual(restored.state(), state)
        result = restored._monitor_converged(scores)
        self.assertEqual(result.action, "reopen")
        self.assertEqual(graph.reopen_count, 1)
        self.assertFalse(result.converged)

    def test_challenger_action_guard(self) -> None:
        """Route only finite challenger actions within both divergence limits."""

        device = require_cuda_graph_capture("FlashSAC challenger action guard")
        champion_np = np.asarray([[0.0, 0.0], [0.1, -0.1], [0.2, 0.3], [-0.4, 0.2]], dtype=np.float32)
        challenger_np = np.asarray([[0.1, -0.1], [0.3, -0.2], [1.2, 0.3], [np.nan, 0.2]], dtype=np.float32)
        champion = wp.array(champion_np, dtype=wp.float32, device=device)
        challenger = wp.array(challenger_np, dtype=wp.float32, device=device)
        guarded = wp.empty_like(challenger)
        fallbacks = wp.empty((1, 4), dtype=wp.int32, device=device)
        wp.launch(
            _guard_challenger_actions_kernel,
            dim=4,
            inputs=[champion, challenger, 0.25, 0.50, 0],
            outputs=[guarded, fallbacks],
            device=device,
        )
        expected = np.asarray(
            [
                challenger_np[0],
                challenger_np[1],
                champion_np[2],
                champion_np[3],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(guarded.numpy(), expected)
        np.testing.assert_array_equal(fallbacks.numpy(), np.asarray([[0, 0, 1, 1]], dtype=np.int32))
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _guard_challenger_actions_kernel,
                dim=4,
                inputs=[champion, challenger, 0.25, 0.50, 0],
                outputs=[guarded, fallbacks],
                device=device,
            )
        wp.capture_launch(capture.graph)

    @staticmethod
    def _make_controller(
        device: wp.Device,
        *,
        autotune: ConfigFlashSACLRAutotune | None = None,
    ) -> ControllerFlashSACLRAutotune:
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=5,
            normalize_rewards=False,
            sample_batch_size=4,
            use_amp=True,
            actor_lr=6.0e-4,
            critic_lr=6.0e-4,
            alpha_lr=6.0e-4,
            buffer_min_length=8,
            buffer_max_length=64,
        )
        trainers = tuple(
            TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=431 + member)
            for member in range(2)
        )
        for trainer in trainers:
            trainer._amp_scale.assign(np.asarray([4096.0], dtype=np.float32))
        rng = np.random.default_rng(431)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(4, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(4, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=4).astype(np.float32), device=device),
            dones=wp.array(np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32), device=device),
            next_obs=wp.array(rng.normal(size=(4, 3)).astype(np.float32), device=device),
        )
        return ControllerFlashSACLRAutotune(
            trainers,
            batch,
            rollout_world_count=8,
            config=autotune or ConfigFlashSACLRAutotune(evaluation_episodes=4),
        )

    @staticmethod
    def _make_scalar_oracles(controller: ControllerFlashSACLRAutotune) -> tuple[TrainerFlashSAC, TrainerFlashSAC]:
        oracles = tuple(
            TrainerFlashSAC(
                obs_dim=trainer.obs_dim,
                action_dim=trainer.action_dim,
                config=copy.deepcopy(trainer.config),
                device=controller.device,
                seed=trainer.seed,
            )
            for trainer in controller.trainers
        )
        for member, oracle in enumerate(oracles):
            oracle.config.policy_frequency = int(controller.member_policy_frequencies[member])
            oracle._amp_scale.assign(np.asarray([4096.0], dtype=np.float32))
            oracle.set_pbt_learning_rates(*controller.member_rates[member])
            oracle.set_pbt_target_update_rate(controller.member_target_update_rates[member])
        return oracles

    def _assert_population_matches_oracles(
        self,
        controller: ControllerFlashSACLRAutotune,
        oracles: tuple[TrainerFlashSAC, TrainerFlashSAC],
    ) -> None:
        for member, (actual, expected) in enumerate(zip(controller.trainers, oracles, strict=True)):
            for network_name, actual_network, expected_network in (
                ("actor", actual.actor.net, expected.actor.net),
                ("critic1", actual.critic1, expected.critic1),
                ("critic2", actual.critic2, expected.critic2),
                ("target1", actual.target_critic1, expected.target_critic1),
                ("target2", actual.target_critic2, expected.target_critic2),
            ):
                for state_index, (actual_state, expected_state) in enumerate(
                    zip(actual_network.state_arrays(), expected_network.state_arrays(), strict=True)
                ):
                    np.testing.assert_allclose(
                        actual_state.numpy(),
                        expected_state.numpy(),
                        rtol=1.0e-2,
                        # FP16 population kernels use a batched reduction layout; the observed
                        # four-step worst case is a 6.72e-5 BatchNorm bias delta.
                        atol=1.0e-5 if network_name == "actor" else 1.0e-4,
                        err_msg=f"member={member} network={network_name} state={state_index}",
                    )
            np.testing.assert_allclose(actual.log_alpha.numpy(), expected.log_alpha.numpy(), rtol=1.0e-2, atol=1.0e-6)
            np.testing.assert_array_equal(actual._amp_scale.numpy(), expected._amp_scale.numpy())
            np.testing.assert_array_equal(actual._amp_growth_tracker.numpy(), expected._amp_growth_tracker.numpy())
            self.assertEqual(actual._gradient_update_count, expected._gradient_update_count)
            self.assertEqual(actual._update_count, expected._update_count)
            for optimizer_index, (actual_optimizer, expected_optimizer) in enumerate(
                (
                    (actual.actor_optimizer, expected.actor_optimizer),
                    (actual.critic1_optimizer, expected.critic1_optimizer),
                    (actual.critic2_optimizer, expected.critic2_optimizer),
                    (actual.alpha_optimizer, expected.alpha_optimizer),
                )
            ):
                self.assertEqual(actual_optimizer.step_count, expected_optimizer.step_count)
                np.testing.assert_allclose(actual_optimizer.lr_scale.numpy(), expected_optimizer.lr_scale.numpy())
                np.testing.assert_allclose(
                    actual_optimizer.pbt_lr_scale.numpy(), expected_optimizer.pbt_lr_scale.numpy()
                )
                for actual_moment, expected_moment in zip(
                    actual_optimizer.m + actual_optimizer.v,
                    expected_optimizer.m + expected_optimizer.v,
                    strict=True,
                ):
                    actual_values = actual_moment.numpy()
                    expected_values = expected_moment.numpy()
                    significant = np.abs(expected_values) > 2.0e-4
                    self.assertTrue(
                        np.all(actual_values[significant] * expected_values[significant] >= 0.0),
                        msg=f"member={member} optimizer={optimizer_index} moment sign",
                    )
                    np.testing.assert_allclose(
                        actual_values,
                        expected_values,
                        rtol=3.5e-1,
                        # Near-zero actor moments expose the FP16 batched-reduction layout;
                        # the sign guard above rejects the Adam sign-flip failure mode.
                        atol=2.0e-4 if optimizer_index == 0 else 5.0e-5,
                        err_msg=f"member={member} optimizer={optimizer_index}",
                    )

    @staticmethod
    def _actor_state(trainer: TrainerFlashSAC) -> tuple[np.ndarray, ...]:
        return tuple(
            value.numpy().copy()
            for value in (
                *trainer.actor.net.state_arrays(),
                trainer.log_alpha,
                *trainer.actor_optimizer.m,
                trainer.actor.log_std,
                trainer._alpha,
                *trainer.actor_optimizer.v,
                trainer.actor_optimizer._step_count,
                *trainer.alpha_optimizer.m,
                *trainer.alpha_optimizer.v,
                trainer.alpha_optimizer._step_count,
                trainer._amp_scale,
            )
        )

    @staticmethod
    def _critic_state(trainer: TrainerFlashSAC) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        online = tuple(
            value.numpy().copy() for network in (trainer.critic1, trainer.critic2) for value in network.state_arrays()
        )
        targets = tuple(
            value.numpy().copy()
            for network in (trainer.target_critic1, trainer.target_critic2)
            for value in network.state_arrays()
        )
        return online, targets

    def test_captured_population_respects_policy_frequency(self) -> None:
        """Match four alternating P2 graph updates to two scalar trainers."""

        device = require_cuda_graph_capture("FlashSAC LR autotune policy cadence")
        controller = self._make_controller(device)
        controller.member_target_update_rates[:] = (0.02, 0.005)
        controller._set_member_rates()
        oracles = self._make_scalar_oracles(controller)
        seed = 953
        controller.capture(seed=seed)
        previous_actor_state: tuple[tuple[np.ndarray, ...], ...] | None = None
        previous_critic_state = tuple(self._critic_state(trainer) for trainer in controller.trainers)
        previous_critic_steps = tuple(oracle.critic1_optimizer.step_count for oracle in oracles)
        for step in range(4):
            for member, oracle in enumerate(oracles):
                oracle.update(controller.batch, seed=seed + member + step * 9973, read_stats=False)
            controller.launch()
            with self.subTest(step=step):
                self._assert_population_matches_oracles(controller, oracles)
            np.testing.assert_array_equal(
                controller.population.scalar_state["_device_update_seed"].numpy()[:, 0],
                np.asarray([seed + step * 9973, seed + 1 + step * 9973], dtype=np.int32),
            )
            actor_state = tuple(self._actor_state(trainer) for trainer in controller.trainers)
            critic_steps = tuple(trainer.critic1_optimizer.step_count for trainer in controller.trainers)
            critic_state = tuple(self._critic_state(trainer) for trainer in controller.trainers)
            for (online, targets), (previous_online, previous_targets) in zip(
                critic_state, previous_critic_state, strict=True
            ):
                self.assertTrue(
                    any(
                        not np.array_equal(actual, previous)
                        for actual, previous in zip(online, previous_online, strict=False)
                    )
                )
                self.assertTrue(
                    any(
                        not np.array_equal(actual, previous)
                        for actual, previous in zip(targets, previous_targets, strict=False)
                    )
                )
            self.assertEqual(critic_steps, tuple(value + 1 for value in previous_critic_steps))
            if step % 2 == 1:
                assert previous_actor_state is not None
                for actual_member, previous_member in zip(actor_state, previous_actor_state, strict=True):
                    for actual, previous in zip(actual_member, previous_member, strict=True):
                        np.testing.assert_array_equal(actual, previous)
            previous_actor_state = actor_state
            previous_critic_steps = critic_steps
            previous_critic_state = critic_state

    def test_round_robin_proposals_promote_with_hysteresis(self) -> None:
        """Promote after repeated wins and advance when a coordinate reaches its bound."""

        device = require_cuda_graph_capture("FlashSAC LR autotune hysteresis")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                promotion_windows=2,
                improvement_margin=0.01,
                exploit_after_candidate=False,
            ),
        )
        np.testing.assert_allclose(controller.member_rates[1], controller.default_rates * 2.0)
        actor_state = controller.population.actors.population_state_arrays()[0]
        values = actor_state.numpy()
        values[1] = values[1] + np.float32(0.125)
        actor_state.assign(values)
        champion = np.asarray([1.0, 1.1, 0.9, 1.0], dtype=np.float32)
        challenger = champion + np.float32(0.02)
        first = controller.evaluate_paired(champion, challenger)
        self.assertEqual(first.action, "continue")
        self.assertEqual(first.consecutive_wins, 1)
        second = controller.evaluate_paired(champion, challenger)
        self.assertEqual(second.action, "promote")
        np.testing.assert_array_equal(actor_state.numpy()[0], actor_state.numpy()[1])
        np.testing.assert_allclose(controller.member_rates[1], controller.member_rates[0])
        self.assertAlmostEqual(
            controller.member_target_update_rates[1],
            controller.default_target_update_rate * 2.0,
        )
        self.assertTrue(controller.best_valid)
        self.assertEqual(controller.best_member, 1)
        self.assertAlmostEqual(controller.best_score, float(np.mean(challenger)), places=6)
        np.testing.assert_allclose(controller.best_rates, controller.default_rates * 2.0)

        promoted = actor_state.numpy()
        promoted[0] += np.float32(0.25)
        actor_state.assign(promoted)
        improved = champion + np.float32(0.30)
        inferior = champion
        controller.evaluate_paired(improved, inferior)
        controller.evaluate_paired(improved, inferior)
        self.assertEqual(controller.best_member, 0)
        self.assertAlmostEqual(controller.best_score, float(np.mean(improved)), places=6)
        confirmed_actor = tuple(
            value.numpy()[0].copy() for value in controller.population.actors.population_state_arrays()
        )

        degraded = actor_state.numpy()
        degraded[0] += np.float32(0.50)
        actor_state.assign(degraded)
        controller.finalize_best()
        self.assertTrue(controller.converged)
        for population_state, expected in zip(
            controller.population.actors.population_state_arrays(), confirmed_actor, strict=True
        ):
            np.testing.assert_array_equal(population_state.numpy()[0], expected)
            np.testing.assert_array_equal(population_state.numpy()[1], expected)

    def test_rejected_proposal_reverses_then_advances_coordinate(self) -> None:
        """Reverse and shrink a rejected bracket before advancing its coordinate."""

        device = require_cuda_graph_capture("FlashSAC LR autotune bracket")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                exploit_after_candidate=False,
            ),
        )
        scores = np.ones(4, dtype=np.float32)
        self.assertEqual(controller.evaluate_paired(scores, scores).action, "reject")
        np.testing.assert_allclose(
            controller.member_rates[1], controller.default_rates / controller.perturbation_factor
        )
        self.assertEqual(controller.evaluate_paired(scores, scores).action, "reject")
        changed = controller.member_rates[1] != controller.member_rates[0]
        self.assertEqual(int(np.count_nonzero(changed)), 1)
        self.assertTrue(changed[0])

    def test_low_signal_candidate_receives_bounded_evidence_rung(self) -> None:
        """Retain a safe low-signal challenger for a bounded resource rung."""

        device = require_cuda_graph_capture("FlashSAC LR autotune evidence rung")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                minimum_evidence_windows=3,
                informative_score_threshold=0.05,
            ),
        )
        scores = np.full(4, 0.01, dtype=np.float32)
        challenger_rates = controller.member_rates[1].copy()
        search_round = controller.search_round

        first = controller.evaluate_paired(scores, scores)
        second = controller.evaluate_paired(scores, scores)

        self.assertEqual(first.action, "gather_evidence")
        self.assertEqual(second.action, "gather_evidence")
        self.assertEqual(controller._candidate_evidence_windows, 2)
        self.assertEqual(controller.search_round, search_round)
        np.testing.assert_array_equal(controller.member_rates[1], challenger_rates)

        third = controller.evaluate_paired(scores, scores)
        self.assertEqual(third.action, "reject")
        self.assertEqual(controller._candidate_evidence_windows, 0)
        self.assertNotEqual(controller.search_round, search_round)

        relative = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                minimum_evidence_windows=3,
                informative_score_threshold=0.05,
                relative_improvement_margin=0.10,
                minimum_effect_delta=1.0e-4,
            ),
        )
        champion = np.full(4, 0.003, dtype=np.float32)
        challenger = np.full(4, 0.0042, dtype=np.float32)
        self.assertEqual(relative.evaluate_paired(champion, challenger).action, "gather_evidence")
        self.assertEqual(relative.evaluate_paired(champion, challenger).action, "gather_evidence")
        result = relative.evaluate_paired(champion, challenger)
        self.assertEqual(result.action, "continue")
        self.assertEqual(result.consecutive_wins, 1)

        unsafe = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                minimum_evidence_windows=3,
                informative_score_threshold=0.05,
            ),
        )
        result = unsafe.evaluate_paired(scores, scores, challenger_safe=False)
        self.assertEqual(result.action, "safety_fallback")
        self.assertEqual(unsafe._candidate_evidence_windows, 0)

    def test_rejection_converges_from_live_champion(self) -> None:
        """Preserve safe champion learning when ending a rejected search."""

        device = require_cuda_graph_capture("FlashSAC live champion convergence")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                minimum_evidence_windows=2,
                promotion_windows=2,
                exploit_after_candidate=True,
            ),
        )
        low = np.full(4, 0.01, dtype=np.float32)
        self.assertEqual(controller.evaluate_paired(low, low).action, "gather_evidence")
        champion = np.full(4, 0.08, dtype=np.float32)
        challenger = np.full(4, 0.10, dtype=np.float32)
        self.assertEqual(controller.evaluate_paired(champion, challenger).action, "continue")
        self.assertTrue(controller.best_valid)

        population_actor = controller.population.actors.population_state_arrays()[0]
        live = population_actor.numpy()
        live[0] += np.float32(0.25)
        population_actor.assign(live)
        live_champion = tuple(
            value.numpy()[0].copy() for value in controller.population.actors.population_state_arrays()
        )

        result = controller.evaluate_paired(
            np.full(4, 0.09, dtype=np.float32),
            np.full(4, 0.08, dtype=np.float32),
        )
        self.assertEqual(result.action, "reject")
        self.assertTrue(controller.converged)
        for actual, expected in zip(controller.single_trainer.actor.net.state_arrays(), live_champion, strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_paired_window_mean_retains_a_strong_candidate(self) -> None:
        """Promote from repeated paired evidence despite a narrow second window."""

        device = require_cuda_graph_capture("FlashSAC paired window evidence")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                promotion_windows=2,
                informative_score_threshold=0.05,
            ),
        )
        first_champion = np.full(4, 0.25, dtype=np.float32)
        first_challenger = np.full(4, 0.35, dtype=np.float32)
        second_champion = np.full(4, 0.80, dtype=np.float32)
        second_challenger = np.full(4, 0.806, dtype=np.float32)

        first = controller.evaluate_paired(first_champion, first_challenger)
        second = controller.evaluate_paired(second_champion, second_challenger)

        self.assertEqual(first.action, "continue")
        self.assertEqual(second.action, "promote")
        np.testing.assert_allclose(controller.member_rates[0], controller.default_rates * 2.0)
        self.assertTrue(second.converged)
        self.assertTrue(controller.converged)
        for expected, actual in zip(
            controller.trainers[0].actor.net.state_arrays(),
            controller.single_trainer.actor.net.state_arrays(),
            strict=True,
        ):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        regressed = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                promotion_windows=2,
                informative_score_threshold=0.05,
            ),
        )
        self.assertEqual(
            regressed.evaluate_paired(first_champion, np.full(4, 0.45, dtype=np.float32)).action,
            "continue",
        )
        collapsed = regressed.evaluate_paired(second_champion, np.full(4, 0.72, dtype=np.float32))
        self.assertEqual(collapsed.action, "reject")

    def test_target_update_rate_is_a_bounded_search_coordinate(self) -> None:
        """Propose a graph-safe target update rate without changing learning rates."""

        device = require_cuda_graph_capture("FlashSAC target update rate autotuning")
        controller = self._make_controller(device)
        controller.search_round = 4
        controller.member_rates[1] = controller.member_rates[0]
        controller.member_target_update_rates[1] = controller.member_target_update_rates[0]

        controller._propose_challenger()

        np.testing.assert_array_equal(controller.member_rates[1], controller.member_rates[0])
        self.assertAlmostEqual(
            controller.member_target_update_rates[1],
            controller.default_target_update_rate * controller.config.initial_perturbation_factor,
        )
        np.testing.assert_allclose(
            controller.trainers[1]._device_target_update_rate.numpy(),
            np.asarray(
                [controller.default_target_update_rate * controller.config.initial_perturbation_factor],
                dtype=np.float32,
            ),
        )

    def test_policy_frequency_is_a_precaptured_structural_coordinate(self) -> None:
        """Propose only policy cadences that exactly divide the captured update span."""

        device = require_cuda_graph_capture("FlashSAC policy-frequency autotuning")
        controller = self._make_controller(device)
        controller.configure_policy_frequency_family(4, allow_search=True)
        controller.search_round = 5
        controller.member_rates[1] = controller.member_rates[0]
        controller.member_target_update_rates[1] = controller.member_target_update_rates[0]
        controller.member_policy_frequencies[1] = controller.member_policy_frequencies[0]

        controller._propose_challenger()

        np.testing.assert_array_equal(controller.member_rates[1], controller.member_rates[0])
        self.assertEqual(controller.member_target_update_rates[1], controller.member_target_update_rates[0])
        self.assertEqual(controller.policy_frequency_choices, (1, 2, 4))
        self.assertEqual(controller.member_policy_frequencies.tolist(), [2, 4])

    def test_best_evidence_tracks_both_members_independently(self) -> None:
        """Confirm an improving champion even when the paired winner changes."""

        device = require_cuda_graph_capture("FlashSAC independent best evidence")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4, promotion_windows=2, exploit_after_candidate=False
            ),
        )
        first_champion = np.full(4, 0.60, dtype=np.float32)
        first_challenger = np.full(4, 0.70, dtype=np.float32)
        second_champion = np.full(4, 0.80, dtype=np.float32)
        second_challenger = np.full(4, 0.76, dtype=np.float32)

        first = controller.evaluate_paired(first_champion, first_challenger)
        second = controller.evaluate_paired(second_champion, second_challenger)

        self.assertEqual(first.action, "continue")
        self.assertEqual(second.action, "promote")
        self.assertTrue(controller.best_valid)
        self.assertEqual(controller.best_member, 0)
        self.assertAlmostEqual(controller.best_score, 0.80, places=6)
        np.testing.assert_array_equal(controller.best_rates, controller.default_rates)

    def test_best_snapshot_tolerates_safe_learning_drift(self) -> None:
        """Confirm a much better policy despite modest drift between evidence windows."""

        device = require_cuda_graph_capture("FlashSAC best evidence drift")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4, promotion_windows=2, exploit_after_candidate=False
            ),
        )
        baseline = np.full(4, 0.30, dtype=np.float32)
        controller.evaluate_paired(baseline, baseline)
        controller.evaluate_paired(baseline, baseline)
        self.assertTrue(controller.best_valid)

        controller.evaluate_paired(
            np.full(4, 0.70, dtype=np.float32),
            np.full(4, 0.76, dtype=np.float32),
        )
        controller.evaluate_paired(
            np.full(4, 0.71, dtype=np.float32),
            np.full(4, 0.73, dtype=np.float32),
        )

        self.assertEqual(controller.best_member, 1)
        self.assertAlmostEqual(controller.best_score, 0.73, places=6)

    def test_best_snapshot_rolls_back_transient_quality_collapse(self) -> None:
        """Restore the repeated best policy after live training regresses."""

        device = require_cuda_graph_capture("FlashSAC LR autotune best rollback")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(evaluation_episodes=4, promotion_windows=2),
        )
        stable = np.full(4, 0.5, dtype=np.float32)
        controller.evaluate_paired(stable, stable, champion_termination_rate=0.0, challenger_termination_rate=0.0)
        controller.evaluate_paired(stable, stable, champion_termination_rate=0.0, challenger_termination_rate=0.0)
        self.assertTrue(controller.best_valid)
        best_actor = tuple(value.numpy().copy() for value in controller.single_trainer.actor.net.state_arrays())
        champion_actor = controller.population.actors.population_state_arrays()[0]
        changed = champion_actor.numpy()
        changed[0] += np.float32(0.25)
        champion_actor.assign(changed)
        collapsed = np.full(4, 0.1, dtype=np.float32)
        result = controller.evaluate_paired(
            collapsed,
            collapsed,
            champion_termination_rate=1.0,
            challenger_termination_rate=1.0,
        )
        self.assertEqual(result.action, "rollback")
        for population_state, expected in zip(
            controller.population.actors.population_state_arrays(), best_actor, strict=True
        ):
            np.testing.assert_array_equal(population_state.numpy()[0], expected)
        np.testing.assert_array_equal(controller.member_rates[0], controller.best_rates)
        self.assertEqual(controller.member_target_update_rates[0], controller.best_target_update_rate)

    def test_split_rollout_routes_safely(self) -> None:
        """Route challenger worlds and immediately fall back after an unsafe gate."""

        device = require_cuda_graph_capture("FlashSAC LR autotune split rollout")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                challenger_fraction=0.25,
                exploit_after_candidate=False,
            ),
        )
        champion = wp.zeros((8, 2), dtype=wp.float32, device=device)
        challenger = wp.ones((8, 2), dtype=wp.float32, device=device)
        routed = controller.route_split_actions(champion, challenger).numpy()
        np.testing.assert_array_equal(routed[: controller.champion_worlds], 0.0)
        np.testing.assert_array_equal(routed[controller.champion_worlds :], 1.0)
        scores = np.ones(4, dtype=np.float32)
        result = controller.evaluate_paired(scores, scores, challenger_safe=False)
        self.assertEqual(result.action, "safety_fallback")
        np.testing.assert_array_equal(controller.route_split_actions(champion, challenger).numpy(), 0.0)
        controller.begin_search_window()
        np.testing.assert_array_equal(
            controller.route_split_actions(champion, challenger).numpy()[controller.champion_worlds :], 1.0
        )

    def test_relative_termination_safety_allows_equal_early_failures(self) -> None:
        """Compare challenger falls to the paired champion instead of an absolute gate."""

        device = require_cuda_graph_capture("FlashSAC LR autotune relative safety")
        scores = np.ones(4, dtype=np.float32)
        equal = self._make_controller(device)
        result = equal.evaluate_paired(
            scores,
            scores,
            champion_termination_rate=0.75,
            challenger_termination_rate=0.75,
        )
        self.assertEqual(result.action, "reject")
        worse = self._make_controller(device)
        result = worse.evaluate_paired(
            scores,
            scores,
            champion_termination_rate=0.25,
            challenger_termination_rate=0.50,
        )
        self.assertEqual(result.action, "safety_fallback")

    def test_captured_population_converges_to_preallocated_single(self) -> None:
        """Replay fixed graphs, preserve pointers, and switch to the setup-owned P1 trainer."""

        device = require_cuda_graph_capture("FlashSAC LR autotune capture")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(
                evaluation_episodes=4,
                convergence_windows=2,
                promotion_windows=2,
                minimum_search_windows=2,
            ),
        )
        pointers = tuple(int(value.ptr) for value in controller.state_arrays())
        controller.capture(seed=719)
        controller.launch()
        controller.launch()
        scores = np.ones(4, dtype=np.float32)
        self.assertEqual(controller.evaluate_paired(scores, scores - np.float32(0.1)).action, "reject")
        self.assertEqual(controller.evaluate_paired(scores, scores - np.float32(0.1)).action, "converge")
        self.assertFalse(controller.population_active)
        controller.launch()
        controller.launch()
        self.assertEqual(pointers, tuple(int(value.ptr) for value in controller.state_arrays()))
        self.assertEqual(controller.single_trainer._update_count, controller.trainers[0]._update_count + 2)

    def test_checkpoint_restores_search_and_learner_state(self) -> None:
        """Restore controller, population members, and the preallocated P1 state."""

        device = require_cuda_graph_capture("FlashSAC LR autotune checkpoint")
        controller = self._make_controller(device)
        scores = np.ones(4, dtype=np.float32)
        controller.evaluate_paired(scores, scores + np.float32(0.02))
        controller.configure_policy_frequency_family(4, allow_search=True)
        controller.member_policy_frequencies[:] = (2, 4)
        controller.best_policy_frequency = 4
        controller.reopen_count = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.npz"
            controller.save_checkpoint(path)
            restored = ControllerFlashSACLRAutotune.load_checkpoint(path, controller.batch, device=device)
        np.testing.assert_array_equal(restored.member_rates, controller.member_rates)
        np.testing.assert_array_equal(restored.member_target_update_rates, controller.member_target_update_rates)
        np.testing.assert_array_equal(restored.member_policy_frequencies, controller.member_policy_frequencies)
        self.assertEqual(restored.policy_frequency_choices, controller.policy_frequency_choices)
        self.assertEqual(restored.best_policy_frequency, controller.best_policy_frequency)
        self.assertEqual(restored.search_round, controller.search_round)
        self.assertEqual(restored.consecutive_wins, controller.consecutive_wins)
        self.assertEqual(restored._candidate_evidence_windows, controller._candidate_evidence_windows)
        np.testing.assert_array_equal(restored._candidate_score_sums, controller._candidate_score_sums)
        np.testing.assert_array_equal(
            restored._candidate_termination_rate_sums, controller._candidate_termination_rate_sums
        )
        self.assertEqual(restored._candidate_decision_windows, controller._candidate_decision_windows)
        self.assertEqual(restored.evaluation_count, controller.evaluation_count)
        self.assertEqual(restored.reopen_count, controller.reopen_count)
        self.assertEqual(restored.config.reopen_stagnation_windows, controller.config.reopen_stagnation_windows)
        self.assertEqual(restored.best_member, controller.best_member)
        np.testing.assert_array_equal(restored._best_candidate_scores, controller._best_candidate_scores)
        np.testing.assert_array_equal(
            restored._best_candidate_termination_rates, controller._best_candidate_termination_rates
        )
        np.testing.assert_array_equal(restored._best_candidate_windows, controller._best_candidate_windows)
        for actual, expected in zip(
            restored.population.state_arrays(), controller.population.state_arrays(), strict=True
        ):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_overlap_uses_shared_batches_and_switches_to_p1(self) -> None:
        """Overlap split rollout with shared-batch P2 updates and converged P1 updates."""

        device = require_cuda_graph_capture("FlashSAC LR autotune overlap")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(evaluation_episodes=4, convergence_windows=2, minimum_search_windows=2),
        )
        rng = np.random.default_rng(811)
        oracles = self._make_scalar_oracles(controller)
        controller.population.scaler.scale.assign(np.asarray([16.0, 16.0], dtype=np.float32))
        for oracle in oracles:
            oracle._amp_scale.assign(np.asarray([16.0], dtype=np.float32))
        obs = wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device)
        next_obs = wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device)
        env = _AutotuneSmokeEnv(obs, next_obs)
        replay = controller.trainers[0].initialize_replay_buffer()
        replay.reserve_graph_buffers(env.world_count)
        replay.add_batch_graph(
            obs,
            wp.zeros((8, 2), dtype=wp.float32, device=device),
            wp.ones(8, dtype=wp.float32, device=device),
            wp.zeros(8, dtype=wp.float32, device=device),
            next_obs,
            truncateds=wp.zeros(8, dtype=wp.float32, device=device),
        )
        replay.advance_graph_host_state()
        wp.synchronize_device(device)
        pointers = tuple(int(value.ptr) for value in controller.state_arrays())
        initial_replay_size = replay.size
        graph = controller.capture_overlap(
            env,
            replay,
            updates_per_step=1,
            interactions_per_launch=2,
            seed=811,
            population_backend="fused",
        )
        self.addCleanup(graph.close)
        self.assertIsNotNone(graph.phase_batches)
        self.assertEqual(len(graph.rollout_actors or ()), 1)
        self.assertEqual(graph.challenger_fallback_fraction(), 0.0)
        for launch_index in range(2):
            assert graph.phase_batches is not None
            for update_index, batch in enumerate(graph.phase_batches[graph.phase]):
                batch_arrays = (batch.obs, batch.actions, batch.rewards, batch.dones, batch.next_obs)
                batch_pointers = tuple(int(value.ptr) for value in batch_arrays)
                batch_values = tuple(value.numpy().copy() for value in batch_arrays)
                step = launch_index * graph.updates_per_launch + update_index
                for member, oracle in enumerate(oracles):
                    actor_step_before = oracle.actor_optimizer.step_count
                    oracle.update(batch, seed=811 + member + step * 9973, read_stats=False)
                    expected_delta = int(step % int(oracle.config.policy_frequency) == 0)
                    self.assertEqual(oracle.actor_optimizer.step_count, actor_step_before + expected_delta)
                self.assertEqual(batch_pointers, tuple(int(value.ptr) for value in batch_arrays))
                for before, value in zip(batch_values, batch_arrays, strict=True):
                    np.testing.assert_array_equal(value.numpy(), before)

            wp.synchronize_device(device)
            graph.launch()
            graph.synchronize()
            np.testing.assert_array_equal(controller.population._actor_found_inf.numpy(), 0)
            np.testing.assert_array_equal(
                controller.population.scalar_state["_device_update_count"].numpy()[:, 0],
                np.asarray([2 * (launch_index + 1), 2 * (launch_index + 1)], dtype=np.int32),
            )
            np.testing.assert_array_equal(controller.population._actor_step_condition.numpy(), 1)
            self.assertEqual(
                tuple(trainer.actor_optimizer.step_count for trainer in controller.trainers),
                (launch_index + 1, launch_index + 1),
            )
            expected_size = min(initial_replay_size + env.world_count * 2 * (launch_index + 1), replay.capacity)
            self.assertEqual(replay.size, expected_size)
            self.assertAlmostEqual(env.sim_time, 0.02 * (launch_index + 1))
            with self.subTest(launch_index=launch_index):
                self._assert_population_matches_oracles(controller, oracles)
        self.assertEqual(controller.trainers[0]._update_count, 4)
        self.assertEqual(controller.trainers[1]._update_count, 4)
        scores = np.ones(4, dtype=np.float32)
        controller.evaluate_paired(scores, scores - np.float32(0.1))
        controller.evaluate_paired(scores, scores - np.float32(0.1))
        single_before = controller.single_trainer._update_count
        graph.launch()
        graph.launch()
        graph.synchronize()
        self.assertEqual(controller.single_trainer._update_count, single_before + 4)
        self.assertEqual(pointers, tuple(int(value.ptr) for value in controller.state_arrays()))
        graph.reopen_search()
        self.assertTrue(controller.population_active)
        population_before = controller.trainers[0]._update_count
        graph.launch()
        graph.synchronize()
        self.assertEqual(controller.trainers[0]._update_count, population_before + 2)
        self.assertEqual(controller.reopen_count, 1)
        self.assertEqual(pointers, tuple(int(value.ptr) for value in controller.state_arrays()))
        graph.close()

    def test_parallel_overlap_matches_two_exact_scalar_trainers(self) -> None:
        """Match parallel member graphs to exact scalar learners on shared batches."""

        device = require_cuda_graph_capture("FlashSAC LR autotune parallel overlap")
        controller = self._make_controller(device)
        controller.configure_policy_frequency_family(2, allow_search=True)
        controller.member_policy_frequencies[:] = (1, 2)
        oracles = self._make_scalar_oracles(controller)
        obs = wp.zeros((8, 3), dtype=wp.float32, device=device)
        next_obs = wp.ones((8, 3), dtype=wp.float32, device=device)
        env = _AutotuneSmokeEnv(obs, next_obs)
        replay = controller.trainers[0].initialize_replay_buffer()
        replay.reserve_graph_buffers(env.world_count)
        replay.add_batch_graph(
            obs,
            wp.zeros((8, 2), dtype=wp.float32, device=device),
            wp.ones(8, dtype=wp.float32, device=device),
            wp.zeros(8, dtype=wp.float32, device=device),
            next_obs,
            truncateds=wp.zeros(8, dtype=wp.float32, device=device),
        )
        replay.advance_graph_host_state()
        graph = controller.capture_overlap(
            env,
            replay,
            updates_per_step=1,
            interactions_per_launch=2,
            seed=977,
            population_backend="parallel",
        )
        self.addCleanup(graph.close)
        np.testing.assert_array_equal(graph.single_trainer._device_interaction_seed.numpy(), [977])
        np.testing.assert_array_equal(graph.trainers[0]._device_interaction_seed.numpy(), [977])
        np.testing.assert_array_equal(graph.trainers[1]._device_interaction_seed.numpy(), [978])
        self.assertEqual(graph.challenger_world_count, 0)
        self.assertEqual(graph.challenger_fallback_fraction(), 0.0)
        self.assertEqual(len(graph.rollout_actors or ()), 1)
        pointers = tuple(
            int(value.ptr)
            for trainer in graph.trainers
            for network in (trainer.actor.net, trainer.critic1, trainer.critic2)
            for value in network.state_arrays()
        )
        for launch_index in range(2):
            assert graph.phase_batches is not None
            for update_index, batch in enumerate(graph.phase_batches[graph.phase]):
                step = launch_index * graph.updates_per_launch + update_index
                batch_arrays = (batch.obs, batch.actions, batch.rewards, batch.dones, batch.next_obs)
                batch_values = tuple(value.numpy().copy() for value in batch_arrays)
                for member, oracle in enumerate(oracles):
                    oracle.update(batch, seed=977 + member + step * 9973, read_stats=False)
                for before, value in zip(batch_values, batch_arrays, strict=True):
                    np.testing.assert_array_equal(value.numpy(), before)
            graph.launch()
            graph.synchronize()
            assert graph.rollout_actors is not None
            rollout_actions = graph.rollout_actors[0]._sample_reuse_actions
            assert rollout_actions is not None
            np.testing.assert_array_equal(env.last_actions.numpy(), rollout_actions.numpy())
            for actual, expected in zip(graph.trainers, oracles, strict=True):
                for actual_network, expected_network in (
                    (actual.actor.net, expected.actor.net),
                    (actual.critic1, expected.critic1),
                    (actual.critic2, expected.critic2),
                    (actual.target_critic1, expected.target_critic1),
                    (actual.target_critic2, expected.target_critic2),
                ):
                    for actual_state, expected_state in zip(
                        actual_network.state_arrays(), expected_network.state_arrays(), strict=True
                    ):
                        np.testing.assert_allclose(
                            actual_state.numpy(), expected_state.numpy(), rtol=1.0e-6, atol=1.0e-7
                        )
                self.assertEqual(actual.actor_optimizer.step_count, expected.actor_optimizer.step_count)
                self.assertEqual(actual.critic1_optimizer.step_count, expected.critic1_optimizer.step_count)
                np.testing.assert_array_equal(actual._amp_scale.numpy(), expected._amp_scale.numpy())
        self.assertEqual(
            pointers,
            tuple(
                int(value.ptr)
                for trainer in graph.trainers
                for network in (trainer.actor.net, trainer.critic1, trainer.critic2)
                for value in network.state_arrays()
            ),
        )
        scores = np.ones(4, dtype=np.float32)
        result = graph.evaluate_paired(scores, scores)
        self.assertEqual(result.action, "reject")
        self.assertAlmostEqual(env.sim_time, 0.04)
        graph.sync_controller_state()
        controller._converge_to_single()
        graph.sync_from_controller_state()
        graph.launch()
        graph.synchronize()
        single_updates = graph.single_trainer._update_count
        graph.reopen_search()
        self.assertFalse(controller.converged)
        graph.launch()
        graph.synchronize()
        self.assertEqual(graph.single_trainer._update_count, single_updates)
        self.assertEqual(controller.reopen_count, 1)
        self.assertEqual(graph.trainers[0]._update_count, single_updates + graph.updates_per_launch)
        self.assertEqual(
            pointers,
            tuple(
                int(value.ptr)
                for trainer in graph.trainers
                for network in (trainer.actor.net, trainer.critic1, trainer.critic2)
                for value in network.state_arrays()
            ),
        )

    def test_parallel_bootstrap_matches_ordinary_overlap(self) -> None:
        """Match P1 bootstrap to ordinary overlap on identical graph inputs."""

        device = require_cuda_graph_capture("FlashSAC LR autotune bootstrap parity")
        controller = self._make_controller(
            device,
            autotune=ConfigFlashSACLRAutotune(evaluation_episodes=4, challenger_fraction=0.25),
        )
        source = controller.trainers[0]
        fixed = TrainerFlashSAC(
            obs_dim=source.obs_dim,
            action_dim=source.action_dim,
            config=source.config,
            device=device,
            seed=source.seed,
        )
        fixed.copy_training_state_from(source)
        fixed_obs = wp.zeros((8, 3), dtype=wp.float32, device=device)
        auto_obs = wp.zeros((8, 3), dtype=wp.float32, device=device)
        fixed_next_obs = wp.ones((8, 3), dtype=wp.float32, device=device)
        auto_next_obs = wp.ones((8, 3), dtype=wp.float32, device=device)
        fixed_env = _AutotuneSmokeEnv(fixed_obs, fixed_next_obs)
        auto_env = _AutotuneSmokeEnv(auto_obs, auto_next_obs)

        def warm_replay(
            trainer: TrainerFlashSAC, env: _AutotuneSmokeEnv, next_obs: wp.array2d[wp.float32]
        ) -> BufferReplayFlashSAC:
            replay = trainer.initialize_replay_buffer()
            replay.reserve_graph_buffers(env.world_count)
            replay.add_batch_graph(
                env.obs,
                wp.zeros((8, 2), dtype=wp.float32, device=device),
                wp.ones(8, dtype=wp.float32, device=device),
                wp.zeros(8, dtype=wp.float32, device=device),
                next_obs,
                truncateds=wp.zeros(8, dtype=wp.float32, device=device),
            )
            replay.advance_graph_host_state()
            return replay

        fixed_graph = fixed.capture_training_graph(
            fixed_env,
            warm_replay(fixed, fixed_env, fixed_next_obs),
            updates_per_step=1,
            interactions_per_graph=2,
            seed=997,
            overlap=True,
        )
        auto_graph = controller.capture_overlap(
            auto_env,
            warm_replay(controller.trainers[0], auto_env, auto_next_obs),
            updates_per_step=1,
            interactions_per_launch=2,
            seed=997,
            population_backend="parallel",
        )
        self.addCleanup(fixed_graph.close)
        self.addCleanup(auto_graph.close)
        auto_graph.start_single_policy_bootstrap()

        for _launch in range(20):
            fixed_graph.launch()
            auto_graph.launch()
            fixed_graph.synchronize()
            auto_graph.synchronize()
            np.testing.assert_array_equal(fixed_env.last_actions.numpy(), auto_env.last_actions.numpy())
            for actual, expected in zip(
                self._actor_state(auto_graph.single_trainer), self._actor_state(fixed), strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
            actual_critics = self._critic_state(auto_graph.single_trainer)
            expected_critics = self._critic_state(fixed)
            for actual_group, expected_group in zip(actual_critics, expected_critics, strict=True):
                for actual, expected in zip(actual_group, expected_group, strict=True):
                    np.testing.assert_array_equal(actual, expected)

    def test_captured_paired_evaluation_is_isolated_and_deterministic(self) -> None:
        """Evaluate identical policies with paired seeds without mutating training state."""

        device = require_cuda_graph_capture("FlashSAC LR autotune paired evaluation")
        controller = self._make_controller(device)
        obs = wp.zeros((4, 3), dtype=wp.float32, device=device)
        next_obs = wp.ones((4, 3), dtype=wp.float32, device=device)
        envs = (_AutotuneSmokeEnv(obs, next_obs), _AutotuneSmokeEnv(obs, next_obs))
        evaluator = EvaluatorPairedFlashSAC(controller.trainers, envs, horizon_steps=3, seed=991)
        self.assertEqual(tuple(env.capture_step_calls for env in envs), (2, 2))
        before = tuple(
            value.numpy().copy()
            for trainer in controller.trainers
            for network in (trainer.actor.net, trainer.critic1, trainer.critic2)
            for value in network.state_arrays()
        )
        result = evaluator.evaluate(controller.trainers)
        np.testing.assert_array_equal(result.champion_scores, result.challenger_scores)
        self.assertTrue(result.champion_finite)
        self.assertTrue(result.challenger_finite)
        self.assertEqual(result.champion_termination_rate, 0.0)
        self.assertEqual(result.challenger_termination_rate, 0.0)
        after = tuple(
            value.numpy()
            for trainer in controller.trainers
            for network in (trainer.actor.net, trainer.critic1, trainer.critic2)
            for value in network.state_arrays()
        )
        for expected, actual in zip(before, after, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_g1_command_tracking_gives_standstill_zero_credit(self) -> None:
        """Give zero score at standstill and full score at the commanded speed."""

        device = require_cuda_graph_capture("FlashSAC G1 command tracking")
        joint_q = np.zeros((2, 7), dtype=np.float32)
        joint_q[:, 6] = 1.0
        joint_qd = np.zeros((2, 6), dtype=np.float32)
        joint_qd[1, 0] = 0.8
        scores = wp.empty(2, dtype=wp.float32, device=device)
        wp.launch(
            _root_command_tracking_kernel,
            dim=2,
            inputs=[
                wp.array(joint_q.reshape(-1), device=device),
                wp.array(joint_qd.reshape(-1), device=device),
                wp.ones(2, dtype=wp.float32, device=device),
                7,
                6,
                0.8,
            ],
            outputs=[scores],
            device=device,
        )
        np.testing.assert_allclose(scores.numpy(), [0.0, 1.0], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
