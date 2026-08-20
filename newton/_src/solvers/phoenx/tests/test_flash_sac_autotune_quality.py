# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-task quality tests for FlashSAC automatic hyperparameter tuning."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training.flash_sac import (
    BufferReplayFlashSAC,
    ConfigFlashSAC,
    TrainerFlashSAC,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune import (
    ConfigFlashSACLRAutotune,
    ControllerFlashSACLRAutotune,
)
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


def _target_actions(obs: np.ndarray) -> np.ndarray:
    """Return the contextual-bandit optimum."""

    return np.tanh(1.2 * obs[:, 0] - 0.7 * obs[:, 1])


def _policy_mse(trainer: TrainerFlashSAC, obs: wp.array2d[wp.float32], target: np.ndarray) -> float:
    """Measure deterministic policy error."""

    actions = trainer.act(obs, seed=0, deterministic=True)[0].numpy()[:, 0]
    return float(np.mean((actions - target) ** 2))


class TestFlashSACLRAutotuneQuality(unittest.TestCase):
    """Validate automatic discovery beyond the G1 locomotion task."""

    def test_discovers_faster_continuous_control_rates_across_seeds(self) -> None:
        """Discover a faster linked rate on three continuous-control seeds."""

        device = require_cuda_graph_capture("FlashSAC LR autotune continuous quality")
        for seed in (3, 7, 11):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                config = ConfigFlashSAC(
                    gamma=0.0,
                    initial_alpha=0.01,
                    target_entropy=0.0,
                    actor_lr=5.0e-4,
                    critic_lr=5.0e-4,
                    alpha_lr=5.0e-4,
                    learning_rate_end=5.0e-4,
                    learning_rate_decay_steps=1000,
                    actor_hidden_dim=32,
                    critic_hidden_dim=32,
                    actor_num_blocks=1,
                    critic_num_blocks=1,
                    distributional_atoms=51,
                    normalize_rewards=False,
                    buffer_min_length=1,
                    buffer_max_length=65536,
                    sample_batch_size=1024,
                )
                champion = TrainerFlashSAC(obs_dim=2, action_dim=1, config=config, device=device, seed=seed)
                control = TrainerFlashSAC(obs_dim=2, action_dim=1, config=config, device=device, seed=seed)
                controller = ControllerFlashSACLRAutotune.from_trainer(
                    champion,
                    rollout_world_count=512,
                    config=ConfigFlashSACLRAutotune(
                        evaluation_episodes=8,
                        minimum_evidence_windows=6,
                        informative_score_threshold=0.0,
                        promotion_windows=2,
                        convergence_windows=20,
                        minimum_search_windows=4,
                        policy_frequency_choices=(2,),
                        seed=seed,
                    ),
                )
                replay = BufferReplayFlashSAC(
                    minimum_size=1,
                    capacity=65536,
                    obs_dim=2,
                    action_dim=1,
                    batch_size=1024,
                    normalize_rewards=False,
                    device=device,
                )
                eval_obs_np = rng.uniform(-1.0, 1.0, (1024, 2)).astype(np.float32)
                eval_target = _target_actions(eval_obs_np)
                eval_obs = wp.array(eval_obs_np, device=device)

                for update in range(175):
                    obs_np = rng.uniform(-1.0, 1.0, (512, 2)).astype(np.float32)
                    obs = wp.array(obs_np, device=device)
                    if update < 4:
                        actions_np = rng.uniform(-1.0, 1.0, (512, 1)).astype(np.float32)
                    else:
                        champion_actions = controller.trainers[0].act(obs[:256], seed=seed * 1000 + update)[0].numpy()
                        challenger_actions = (
                            controller.trainers[1].act(obs[256:], seed=seed * 1000 + update + 1)[0].numpy()
                        )
                        actions_np = np.concatenate((champion_actions, challenger_actions))
                    target = _target_actions(obs_np)
                    rewards = -((actions_np[:, 0] - target) ** 2).astype(np.float32)
                    replay.add_batch(
                        obs,
                        wp.array(actions_np, device=device),
                        wp.array(rewards, device=device),
                        wp.ones(512, dtype=wp.float32, device=device),
                        obs,
                    )
                    batch = replay.sample(seed=update)
                    controller.population.update_all_fused(batch, seed=10000 + update, read_stats=False)
                    control.update(batch, seed=10000 + update, read_stats=False)
                    if (update + 1) % 25 == 0:
                        errors = [_policy_mse(trainer, eval_obs, eval_target) for trainer in controller.trainers]
                        controller.evaluate_paired(
                            np.full(8, -errors[0], dtype=np.float32),
                            np.full(8, -errors[1], dtype=np.float32),
                        )

                tuned_error = _policy_mse(controller.trainers[0], eval_obs, eval_target)
                control_error = _policy_mse(control, eval_obs, eval_target)
                np.testing.assert_allclose(controller.member_rates[0], 1.0e-3, rtol=0.0, atol=0.0)
                self.assertLess(tuned_error, 0.1)
                self.assertLess(tuned_error, 0.4 * control_error)


if __name__ == "__main__":
    unittest.main()
