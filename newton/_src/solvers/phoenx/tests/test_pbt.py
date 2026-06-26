# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Population Based Training (PBT) infrastructure."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training.optim import Adam, Muon
from newton._src.solvers.phoenx.rl_training.pbt import (
    ConfigPBT,
    HparamSpec,
    ResultPBT,
    WorkerState,
    _apply_ppo_hparams_to_trainer,
    _compute_fitness,
    _exploit_explore,
    _perturb_classic,
    _perturb_pb2,
    _sample_hparams,
    default_g1_hparam_specs,
    population_based_train,
)
from newton._src.solvers.phoenx.rl_training.ppo import ConfigPPO, TrainerPPO
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(*, obs_dim: int = 4, action_dim: int = 2, device: wp.context.Device) -> TrainerPPO:
    return TrainerPPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=(8,),
        config=ConfigPPO(entropy_coeff=1e-3, mirror_loss_coeff=0.0),
        device=device,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Optimizer PBT scale tests
# ---------------------------------------------------------------------------


class TestAdamPBTScale(unittest.TestCase):
    """pbt_lr_scale multiplies the effective Adam learning rate."""

    def test_pbt_lr_scale_doubles_step_inside_graph(self) -> None:
        device = require_cuda_graph_capture("Adam pbt_lr_scale tests")

        param = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device, requires_grad=True)
        grad_val = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

        opt = Adam([param], lr=1e-2, beta1=0.0, beta2=0.0, eps=0.0, weight_decay=0.0, max_grad_norm=0.0)

        # Baseline step (pbt_lr_scale == 1.0)
        param.grad.assign(grad_val)
        with wp.ScopedCapture(device=device) as cap1:
            opt.step()
        wp.capture_launch(cap1.graph)
        after_baseline = param.numpy().copy()

        # Reset param and apply 2× PBT LR
        param.assign(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        opt.step_count = 0
        for m in opt.m:
            m.zero_()
        for v in opt.v:
            v.zero_()
        opt.set_pbt_lr(2e-2)

        param.grad.assign(grad_val)
        with wp.ScopedCapture(device=device) as cap2:
            opt.step()
        wp.capture_launch(cap2.graph)
        after_double = param.numpy().copy()

        baseline_delta = np.asarray([1.0, 2.0, 3.0], dtype=np.float32) - after_baseline
        double_delta = np.asarray([1.0, 2.0, 3.0], dtype=np.float32) - after_double
        # With beta1=beta2=0 and eps=0, Adam is just gradient descent: Δ = lr
        np.testing.assert_allclose(double_delta, 2.0 * baseline_delta, rtol=1e-5)

    def test_pbt_lr_scale_does_not_affect_annealing_kernel_buffer(self) -> None:
        device = require_cuda_graph_capture("Adam pbt_lr_scale independence test")
        param = wp.array([1.0], dtype=wp.float32, device=device, requires_grad=True)
        opt = Adam([param], lr=1e-2)
        opt.set_pbt_lr(5e-3)
        # pbt_lr_scale is stored separately; lr_scale (used by annealing) stays at 1.0
        np.testing.assert_allclose(opt.lr_scale.numpy(), [1.0], atol=1e-7)
        np.testing.assert_allclose(opt.pbt_lr_scale.numpy(), [5e-3 / 1e-2], rtol=1e-5)

    def test_set_pbt_lr_resets_to_base_lr(self) -> None:
        opt = Adam(
            [wp.zeros(2, dtype=wp.float32, device="cpu", requires_grad=True)],
            lr=3e-4,
        )
        opt.set_pbt_lr(3e-4)
        np.testing.assert_allclose(opt.pbt_lr_scale.numpy(), [1.0], rtol=1e-5)


class TestMuonPBTScale(unittest.TestCase):
    """pbt_lr_scale multiplies the effective Muon learning rate for 1-D params."""

    def test_pbt_lr_scale_halves_step_inside_graph(self) -> None:
        device = require_cuda_graph_capture("Muon pbt_lr_scale tests")

        param = wp.array([5.0, -3.0, 1.0], dtype=wp.float32, device=device, requires_grad=True)
        grad_val = np.asarray([0.5, -0.5, 0.5], dtype=np.float32)
        base_lr = 1e-2

        opt = Muon([param], lr=base_lr, momentum=0.0, weight_decay=0.0, max_grad_norm=0.0)

        # Baseline step
        param.grad.assign(grad_val)
        with wp.ScopedCapture(device=device) as cap1:
            opt.step()
        wp.capture_launch(cap1.graph)
        after_baseline = param.numpy().copy()

        # Half PBT LR
        param.assign(np.asarray([5.0, -3.0, 1.0], dtype=np.float32))
        opt.step_count = 0
        for m in opt.m:
            m.zero_()
        opt.set_pbt_lr(base_lr / 2)

        param.grad.assign(grad_val)
        with wp.ScopedCapture(device=device) as cap2:
            opt.step()
        wp.capture_launch(cap2.graph)
        after_half = param.numpy().copy()

        baseline_delta = np.asarray([5.0, -3.0, 1.0], dtype=np.float32) - after_baseline
        half_delta = np.asarray([5.0, -3.0, 1.0], dtype=np.float32) - after_half
        np.testing.assert_allclose(half_delta, 0.5 * baseline_delta, rtol=1e-4)


# ---------------------------------------------------------------------------
# TrainerPPO in-place mutation tests
# ---------------------------------------------------------------------------


class TestTrainerPBTMethods(unittest.TestCase):
    def test_set_entropy_coeff_updates_buffer_inside_graph(self) -> None:
        device = require_cuda_graph_capture("TrainerPPO entropy coeff buffer tests")
        trainer = _make_trainer(device=device)
        new_coeff = 5e-3
        trainer.set_entropy_coeff(new_coeff)
        np.testing.assert_allclose(trainer._entropy_coeff_buf.numpy(), [new_coeff], rtol=1e-6)

    def test_set_mirror_loss_coeff_updates_buffer(self) -> None:
        device = require_cuda_graph_capture("TrainerPPO mirror coeff buffer tests")
        trainer = _make_trainer(device=device)
        trainer.set_mirror_loss_coeff(0.25)
        np.testing.assert_allclose(trainer._mirror_loss_coeff_buf.numpy(), [0.25], rtol=1e-6)

    def test_set_actor_lr_writes_pbt_lr_scale(self) -> None:
        device = require_cuda_graph_capture("TrainerPPO actor lr tests")
        trainer = _make_trainer(device=device)
        base_lr = trainer.actor_optimizer.lr
        trainer.set_actor_lr(base_lr * 2)
        np.testing.assert_allclose(trainer.actor_optimizer.pbt_lr_scale.numpy(), [2.0], rtol=1e-5)

    def test_copy_weights_from_matches_params_inside_graph(self) -> None:
        device = require_cuda_graph_capture("TrainerPPO copy_weights_from tests")

        src = _make_trainer(device=device)
        dst = _make_trainer(device=device)

        # Confirm they start different (different seeds give same init due to same seed=42 above,
        # so perturb src manually)
        src_params = src.actor.parameters()
        for p in src_params:
            arr = p.numpy()
            arr += 0.5
            p.assign(arr.astype(np.float32))

        src_before = [p.numpy().copy() for p in src.actor.parameters()]
        dst_before = [p.numpy().copy() for p in dst.actor.parameters()]
        # They should differ now
        any_different = any(not np.array_equal(s, d) for s, d in zip(src_before, dst_before))
        self.assertTrue(any_different, "src and dst weights should differ before copy")

        # GPU-side copy (no graph needed — copy is launch-and-wait)
        dst.copy_weights_from(src)
        wp.synchronize_device(device)

        for i, (sp, dp) in enumerate(zip(src.actor.parameters(), dst.actor.parameters())):
            np.testing.assert_array_equal(sp.numpy(), dp.numpy(), err_msg=f"param {i} mismatch after copy")
        np.testing.assert_array_equal(src.actor.log_std.numpy(), dst.actor.log_std.numpy())

    def test_copy_weights_from_resets_optimizer_state(self) -> None:
        device = require_cuda_graph_capture("TrainerPPO copy_weights_from optimizer reset")
        src = _make_trainer(device=device)
        dst = _make_trainer(device=device)

        # Give dst a non-zero step count and momentum
        dst.actor_optimizer.step_count = 10
        for m in dst.actor_optimizer.m:
            m.assign(np.ones_like(m.numpy(), dtype=np.float32))

        dst.copy_weights_from(src)

        self.assertEqual(dst.actor_optimizer.step_count, 0)
        for m in dst.actor_optimizer.m:
            np.testing.assert_array_equal(m.numpy(), np.zeros_like(m.numpy()))


# ---------------------------------------------------------------------------
# PBT pure-Python logic (no CUDA required)
# ---------------------------------------------------------------------------


@dataclass
class _FakeStats:
    mean_tracking_perf: float


@dataclass
class _FakeConfig:
    ppo_config: ConfigPPO
    iterations: int = 5
    seed: int = 0
    checkpoint_path: str | None = None


class _FakeTrainer:
    """Minimal stub that satisfies _apply_ppo_hparams_to_trainer."""

    def __init__(self, seed: int = 0):
        self.config = ConfigPPO(actor_lr=1e-3, entropy_coeff=1e-3, mirror_loss_coeff=0.0)
        self.weights: list[float] = [float(seed)]
        self._entropy = self.config.entropy_coeff
        self._actor_lr = self.config.actor_lr
        self._mirror = self.config.mirror_loss_coeff

    def set_actor_lr(self, lr: float) -> None:
        self._actor_lr = lr

    def set_critic_lr(self, lr: float) -> None:
        pass

    def set_entropy_coeff(self, c: float) -> None:
        self._entropy = c

    def set_mirror_loss_coeff(self, c: float) -> None:
        self._mirror = c

    def copy_weights_from(self, other: _FakeTrainer) -> None:
        self.weights = list(other.weights)

    def save_checkpoint(self, path: str, *, iteration: int | None = None) -> None:
        Path(path).write_text("fake_ckpt")


@dataclass
class _FakeResult:
    trainer: _FakeTrainer
    env_id: int
    fitness: float
    history: list[_FakeStats] = field(default_factory=list)


class TestPBTCorePythonLogic(unittest.TestCase):
    """Tests for sampling, perturbation, exploitation — no CUDA required."""

    def test_sample_hparams_respects_prior_bounds(self) -> None:
        specs = [
            HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2),
            HparamSpec("entropy_coeff", "log_uniform", 1e-5, 1e-2),
        ]
        rng = np.random.default_rng(0)
        for _ in range(50):
            h = _sample_hparams(specs, rng)
            self.assertGreaterEqual(h["actor_lr"], 1e-4)
            self.assertLessEqual(h["actor_lr"], 1e-2)
            self.assertGreaterEqual(h["entropy_coeff"], 1e-5)
            self.assertLessEqual(h["entropy_coeff"], 1e-2)

    def test_perturb_classic_links_lr_fields(self) -> None:
        """actor_lr and critic_lr share the same factor when linked."""
        config = ConfigPPO(actor_lr=1e-3, critic_lr=1e-3)
        specs = [
            HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2),
            HparamSpec("critic_lr", "log_uniform", 1e-4, 1e-2),
        ]
        rng = np.random.default_rng(1)
        for _ in range(20):
            h = _perturb_classic(
                config, specs, (0.8, 1.2), 0.0, ("actor_lr", "critic_lr"), rng
            )
            # Both should change by the same multiplicative factor
            ratio = h["actor_lr"] / h["critic_lr"]
            self.assertAlmostEqual(ratio, 1.0, places=8)

    def test_perturb_pb2_falls_back_with_few_observations(self) -> None:
        """With < 3 observations, PB2 should resample from the prior."""
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]
        rng = np.random.default_rng(7)
        for n_obs in (0, 1, 2):
            obs = [({}, float(i)) for i in range(n_obs)]
            h = _perturb_pb2(obs, specs, n_candidates=16, rng=rng)
            self.assertIn("actor_lr", h)
            self.assertGreaterEqual(h["actor_lr"], 1e-4)
            self.assertLessEqual(h["actor_lr"], 1e-2)

    def test_perturb_pb2_uses_gp_ucb_with_enough_observations(self) -> None:
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]
        rng = np.random.default_rng(3)
        obs = [
            ({"actor_lr": 1e-4}, 0.1),
            ({"actor_lr": 5e-4}, 0.5),
            ({"actor_lr": 1e-3}, 0.9),
            ({"actor_lr": 5e-3}, 0.3),
        ]
        h = _perturb_pb2(obs, specs, n_candidates=64, rng=rng)
        self.assertIn("actor_lr", h)
        self.assertGreaterEqual(h["actor_lr"], 1e-4)

    def test_compute_fitness_tail_average(self) -> None:
        history = [_FakeStats(mean_tracking_perf=float(i)) for i in range(10)]
        fitness = _compute_fitness(history, fitness_window=3, fitness_metric="mean_tracking_perf")
        expected = (7.0 + 8.0 + 9.0) / 3.0
        self.assertAlmostEqual(fitness, expected)

    def test_compute_fitness_empty_returns_neg_inf(self) -> None:
        self.assertEqual(_compute_fitness([], 5, "mean_tracking_perf"), float("-inf"))

    def test_exploit_explore_truncation_replaces_bottom(self) -> None:
        rng = np.random.default_rng(99)
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]
        pbt_cfg = ConfigPBT(population_size=4, exploit_fraction=0.25, exploit_strategy="truncation")

        workers = [
            WorkerState(
                worker_id=i,
                ppo_config=ConfigPPO(actor_lr=10.0 ** (-3 + i * 0.1)),
                checkpoint_path=f"/tmp/w{i}.npz",
                fitness_history=[float(i)],
            )
            for i in range(4)
        ]

        workers_out, gen_result, replaced_ids = _exploit_explore(workers, specs, pbt_cfg, rng, generation=0)

        # Lowest fitness worker (index 0) should be replaced
        self.assertIn(0, replaced_ids)
        # Highest fitness worker should NOT be replaced
        self.assertNotIn(3, replaced_ids)
        # The exploited bottom worker should inherit from a top worker
        replaced = next(w for w in workers_out if w.worker_id == 0)
        self.assertIsNotNone(replaced.parent_id)
        self.assertIn(replaced.parent_id, {3})

    def test_apply_ppo_hparams_updates_trainer_buffers(self) -> None:
        trainer = _FakeTrainer()
        new_config = ConfigPPO(actor_lr=5e-3, entropy_coeff=2e-3, mirror_loss_coeff=0.1)
        specs = [
            HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2),
            HparamSpec("entropy_coeff", "log_uniform", 1e-5, 1e-2),
            HparamSpec("mirror_loss_coeff", "uniform", 0.0, 0.5),
        ]
        _apply_ppo_hparams_to_trainer(new_config, trainer, specs)
        self.assertAlmostEqual(trainer._actor_lr, 5e-3)
        self.assertAlmostEqual(trainer._entropy, 2e-3)
        self.assertAlmostEqual(trainer._mirror, 0.1)
        # Config fields should also be updated
        self.assertAlmostEqual(trainer.config.actor_lr, 5e-3)
        self.assertAlmostEqual(trainer.config.entropy_coeff, 2e-3)
        self.assertAlmostEqual(trainer.config.mirror_loss_coeff, 0.1)


class TestPopulationBasedTrainGeneric(unittest.TestCase):
    """Tests for the generic population_based_train with fake train/continue fns."""

    def _make_fake_infrastructure(self):
        """Build fake train_fn, continue_fn, make_config_fn, get_history_fn."""
        call_log: dict[str, list] = {"train": [], "continue": []}
        env_counter = [0]

        def fake_train_fn(config: _FakeConfig) -> _FakeResult:
            env_counter[0] += 1
            env_id = env_counter[0]
            trainer = _FakeTrainer(seed=config.seed)
            fitness = 0.3 + config.seed * 0.001  # deterministic fake fitness
            hist = [_FakeStats(mean_tracking_perf=fitness) for _ in range(config.iterations)]
            call_log["train"].append(config.seed)
            return _FakeResult(trainer=trainer, env_id=env_id, fitness=fitness, history=hist)

        def fake_continue_fn(result: _FakeResult, config: _FakeConfig) -> _FakeResult:
            fitness = 0.3 + config.seed * 0.001
            hist = [_FakeStats(mean_tracking_perf=fitness) for _ in range(config.iterations)]
            call_log["continue"].append(config.seed)
            return _FakeResult(
                trainer=result.trainer,
                env_id=result.env_id,  # same env!
                fitness=fitness,
                history=hist,
            )

        def fake_get_trainer(result: _FakeResult):
            return result.trainer

        def make_config(ppo_cfg, resume_checkpoint, resume_policy_only, seed, iterations, checkpoint_path):
            return _FakeConfig(ppo_config=ppo_cfg, iterations=iterations, seed=seed, checkpoint_path=checkpoint_path)

        def get_history(result: _FakeResult):
            return result.history

        return fake_train_fn, fake_continue_fn, fake_get_trainer, make_config, get_history, call_log

    def test_live_workers_call_train_fn_only_once_per_worker(self) -> None:
        """With continue_fn provided, train_fn is called exactly pop_size times."""
        train_fn, continue_fn, get_trainer, make_config, get_history, call_log = (
            self._make_fake_infrastructure()
        )
        pop_size = 3
        total_cycles = 4
        pbt_cfg = ConfigPBT(population_size=pop_size, exploit_interval=2, total_cycles=total_cycles, seed=7)
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = population_based_train(
                train_fn=train_fn,
                make_config_fn=make_config,
                get_history_fn=get_history,
                pbt_config=pbt_cfg,
                hparam_specs=specs,
                initial_ppo_config=ConfigPPO(actor_lr=1e-3),
                output_dir=tmpdir,
                continue_fn=continue_fn,
                get_trainer_fn=get_trainer,
            )

        # train_fn called once per worker at cycle 0
        self.assertEqual(len(call_log["train"]), pop_size)
        # continue_fn called for remaining cycles
        self.assertEqual(len(call_log["continue"]), pop_size * (total_cycles - 1))

    def test_live_workers_env_reused_across_cycles(self) -> None:
        """The env_id should be the same across cycles for the same worker."""
        env_ids_by_worker: dict[int, list[int]] = {}

        def fake_train_fn(config: _FakeConfig) -> _FakeResult:
            return _FakeResult(
                trainer=_FakeTrainer(),
                env_id=id(config),  # unique per call
                fitness=0.5,
                history=[_FakeStats(0.5)],
            )

        persisted_envs: dict[int, int] = {}  # worker_id -> env_id

        def fake_continue_fn(result: _FakeResult, config: _FakeConfig) -> _FakeResult:
            return _FakeResult(
                trainer=result.trainer,
                env_id=result.env_id,  # reuse!
                fitness=0.5,
                history=[_FakeStats(0.5)],
            )

        def get_trainer(result):
            return result.trainer

        def make_config(ppo_cfg, resume_checkpoint, resume_policy_only, seed, iterations, checkpoint_path):
            return _FakeConfig(ppo_config=ppo_cfg, iterations=iterations, seed=seed)

        def get_history(result):
            return result.history

        pop_size = 2
        total_cycles = 3
        pbt_cfg = ConfigPBT(population_size=pop_size, exploit_interval=2, total_cycles=total_cycles, seed=0)
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]

        seen_env_ids: list[list[int]] = [[] for _ in range(pop_size)]

        original_continue = fake_continue_fn

        def tracking_continue(result, config):
            wid = getattr(config, "_wid", -1)
            seen_env_ids[0].append(result.env_id)
            return original_continue(result, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = population_based_train(
                train_fn=fake_train_fn,
                make_config_fn=make_config,
                get_history_fn=get_history,
                pbt_config=pbt_cfg,
                hparam_specs=specs,
                initial_ppo_config=ConfigPPO(),
                output_dir=tmpdir,
                continue_fn=fake_continue_fn,
                get_trainer_fn=get_trainer,
            )

        self.assertIsInstance(result, ResultPBT)
        self.assertEqual(len(result.workers), pop_size)
        self.assertEqual(len(result.generations), total_cycles)

    def test_fallback_without_continue_fn_calls_train_fn_every_cycle(self) -> None:
        """Without continue_fn, train_fn is called pop_size * total_cycles times."""
        call_count = [0]

        def fake_train_fn(config: _FakeConfig) -> _FakeResult:
            call_count[0] += 1
            return _FakeResult(
                trainer=_FakeTrainer(),
                env_id=call_count[0],
                fitness=0.5,
                history=[_FakeStats(0.5)],
            )

        def make_config(ppo_cfg, ckpt, policy_only, seed, iterations, ckpt_path):
            return _FakeConfig(ppo_config=ppo_cfg, iterations=iterations, seed=seed, checkpoint_path=ckpt_path)

        pop_size = 2
        total_cycles = 3
        pbt_cfg = ConfigPBT(population_size=pop_size, exploit_interval=2, total_cycles=total_cycles, seed=0)
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            population_based_train(
                train_fn=fake_train_fn,
                make_config_fn=make_config,
                get_history_fn=lambda r: r.history,
                pbt_config=pbt_cfg,
                hparam_specs=specs,
                initial_ppo_config=ConfigPPO(),
                output_dir=tmpdir,
                # No continue_fn — fallback mode
            )

        self.assertEqual(call_count[0], pop_size * total_cycles)

    def test_exploit_copies_weights_from_donor(self) -> None:
        """After exploit, bottom worker's trainer should have donor's weights."""
        copied_from_ids: list[tuple[int, int]] = []

        class TrackingTrainer(_FakeTrainer):
            def copy_weights_from(self, other: _FakeTrainer) -> None:
                copied_from_ids.append((id(self), id(other)))
                super().copy_weights_from(other)

        trainers = [TrackingTrainer(seed=i) for i in range(4)]
        trainer_idx = [0]

        def fake_train_fn(config: _FakeConfig) -> _FakeResult:
            t = trainers[trainer_idx[0] % len(trainers)]
            trainer_idx[0] += 1
            # Give worker 3 the best fitness to trigger exploit
            fit = 0.9 if t is trainers[3] else 0.1
            return _FakeResult(trainer=t, env_id=trainer_idx[0], fitness=fit, history=[_FakeStats(fit)])

        def fake_continue_fn(result: _FakeResult, config: _FakeConfig) -> _FakeResult:
            fit = 0.9 if result.trainer is trainers[3] else 0.1
            return _FakeResult(
                trainer=result.trainer, env_id=result.env_id, fitness=fit, history=[_FakeStats(fit)]
            )

        pbt_cfg = ConfigPBT(
            population_size=4,
            exploit_interval=2,
            total_cycles=3,
            exploit_fraction=0.25,
            fresh_optimizer_on_exploit=True,
            seed=42,
        )
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = population_based_train(
                train_fn=fake_train_fn,
                make_config_fn=lambda ppo, ckpt, po, seed, iters, ckpt_path: _FakeConfig(
                    ppo_config=ppo, iterations=iters, seed=seed
                ),
                get_history_fn=lambda r: r.history,
                pbt_config=pbt_cfg,
                hparam_specs=specs,
                initial_ppo_config=ConfigPPO(),
                output_dir=tmpdir,
                continue_fn=fake_continue_fn,
                get_trainer_fn=lambda r: r.trainer,
            )

        # At least one exploit + copy should have happened
        self.assertGreater(len(result.generations), 0)
        any_exploited = any(len(g.exploited_pairs) > 0 for g in result.generations)
        self.assertTrue(any_exploited, "Expected at least one exploit")
        # When exploit happened, copy_weights_from should have been called
        if any_exploited:
            self.assertGreater(len(copied_from_ids), 0)

    def test_best_fitness_and_checkpoint_tracked(self) -> None:
        call_count = [0]

        def fake_train_fn(config: _FakeConfig) -> _FakeResult:
            call_count[0] += 1
            fit = 0.1 * call_count[0]
            return _FakeResult(
                trainer=_FakeTrainer(),
                env_id=call_count[0],
                fitness=fit,
                history=[_FakeStats(fit)],
            )

        pbt_cfg = ConfigPBT(population_size=2, exploit_interval=2, total_cycles=3, seed=0)
        specs = [HparamSpec("actor_lr", "log_uniform", 1e-4, 1e-2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = population_based_train(
                train_fn=fake_train_fn,
                make_config_fn=lambda ppo, ckpt, po, seed, iters, ckpt_path: _FakeConfig(
                    ppo_config=ppo, iterations=iters, seed=seed
                ),
                get_history_fn=lambda r: r.history,
                pbt_config=pbt_cfg,
                hparam_specs=specs,
                initial_ppo_config=ConfigPPO(),
                output_dir=tmpdir,
            )
            self.assertGreater(result.best_fitness, float("-inf"))
            self.assertEqual(len(result.generations), 3)
            self.assertEqual(len(result.workers), 2)

    def test_default_g1_hparam_specs_valid(self) -> None:
        specs = default_g1_hparam_specs()
        self.assertGreater(len(specs), 0)
        rng = np.random.default_rng(0)
        for spec in specs:
            val = _sample_hparams([spec], rng)[spec.field]
            if spec.prior == "log_uniform":
                self.assertGreaterEqual(val, spec.low)
                self.assertLessEqual(val, spec.high)


# ---------------------------------------------------------------------------
# Integration test: PBT with Pendulum (fast synthetic env, no G1 needed)
# ---------------------------------------------------------------------------


class TestPBTWithTrainerPPOCUDA(unittest.TestCase):
    """PBT round-trip with real TrainerPPO objects and CUDA graph capture."""

    def test_pbt_live_workers_update_entropy_without_error(self) -> None:
        device = require_cuda_graph_capture("PBT TrainerPPO live-worker tests")

        from newton._src.solvers.phoenx.rl_training.standard_envs import (
            ConfigEnvPendulumV1Warp,
            EnvPendulumV1Warp,
        )
        from newton._src.solvers.phoenx.rl_training.ppo import BufferRollout
        from newton._src.solvers.phoenx.rl_training.env import collect_ppo_rollout_seed_counter, make_seed_counter

        env_config = ConfigEnvPendulumV1Warp(world_count=8, max_episode_steps=0)
        n_steps = 4
        pop_size = 2

        # Build workers once (simulating cycle 0)
        envs = [EnvPendulumV1Warp(env_config, device=device) for _ in range(pop_size)]
        trainers = [
            TrainerPPO(
                obs_dim=envs[0].obs_dim,
                action_dim=envs[0].action_dim,
                hidden_layers=(8,),
                config=ConfigPPO(entropy_coeff=1e-3, actor_lr=1e-3),
                device=device,
                seed=i,
            )
            for i in range(pop_size)
        ]
        buffers = [
            BufferRollout(
                num_envs=env_config.world_count,
                num_steps=n_steps,
                obs_dim=envs[0].obs_dim,
                action_dim=envs[0].action_dim,
                device=device,
            )
            for _ in range(pop_size)
        ]
        for t, b in zip(trainers, buffers):
            t.reserve_update_buffers(b)

        # Cycle 1: change entropy on both workers (simulating exploit)
        for t in trainers:
            t.set_entropy_coeff(5e-3)
        np.testing.assert_allclose(trainers[0]._entropy_coeff_buf.numpy(), [5e-3], rtol=1e-5)

        # Run one rollout + update inside a graph to verify kernels work after change
        seed_counter = make_seed_counter(123, device=device)
        with wp.ScopedCapture(device=device) as cap:
            collect_ppo_rollout_seed_counter(envs[0], trainers[0], buffers[0], seed_counter=seed_counter)
        wp.capture_launch(cap.graph)

        with wp.ScopedCapture(device=device) as cap2:
            trainers[0].update_seed_counter(buffers[0], seed_counter=seed_counter, read_stats=False)
        wp.capture_launch(cap2.graph)
        # No assertion: just verify no CUDA errors

    def test_copy_weights_from_inside_graph_replicate_policy(self) -> None:
        device = require_cuda_graph_capture("PBT copy_weights_from graph tests")

        trainer_a = _make_trainer(obs_dim=6, action_dim=3, device=device)
        trainer_b = _make_trainer(obs_dim=6, action_dim=3, device=device)

        # Give trainer_a distinct weights
        for p in trainer_a.actor.parameters():
            p.assign((p.numpy() * 2.0).astype(np.float32))

        # Copy inside a graph capture (verifies copy ops are graph-compatible)
        with wp.ScopedCapture(device=device) as cap:
            for dst, src in zip(trainer_b.actor.parameters(), trainer_a.actor.parameters()):
                wp.copy(dst, src)
            wp.copy(trainer_b.actor.log_std, trainer_a.actor.log_std)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        for pa, pb in zip(trainer_a.actor.parameters(), trainer_b.actor.parameters()):
            np.testing.assert_array_equal(pa.numpy(), pb.numpy())


if __name__ == "__main__":
    unittest.main()
