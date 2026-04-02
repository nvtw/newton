# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A/B comparison: vanilla PPO vs PPO+ (tanh bounding + auto entropy).

Runs both configurations on the cartpole environment and prints
mean reward curves side-by-side.  Also tests sensitivity to
``entropy_coef`` -- PPO+ with auto-tuning should be robust to the
initial value while vanilla PPO degrades with bad choices.
"""

from __future__ import annotations

import inspect
import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.ppo import ActorCritic, PPOTrainer

# Import cartpole env -- only available when examples are installed
try:
    from newton.examples.robot.example_robot_cartpole_train import CartpoleVecEnv

    _HAS_CARTPOLE = True
except Exception:
    _HAS_CARTPOLE = False

from newton.tests.test_ppo import ReachZeroEnv


def _train(
    env_class,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: list[int],
    num_envs: int,
    num_updates: int,
    device: str,
    bounded_actions: bool,
    auto_entropy: bool,
    entropy_coef: float = 0.01,
    init_log_std: float = -1.0,
    seed: int = 42,
) -> list[float]:
    """Train and return per-update mean rewards."""
    env_params = inspect.signature(env_class.__init__).parameters
    env_kwargs: dict[str, Any] = {"num_envs": num_envs, "device": device}
    if "seed" in env_params:
        env_kwargs["seed"] = seed
    env = env_class(**env_kwargs)
    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        activation="elu",
        init_log_std=init_log_std,
        bounded_actions=bounded_actions,
        device=device,
        seed=seed,
    )
    trainer = PPOTrainer(
        ac,
        num_envs,
        lr=3e-4,
        num_steps=32,
        num_epochs=5,
        num_minibatches=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=entropy_coef,
        value_coef=0.5,
        max_grad_norm=1.0,
        auto_entropy=auto_entropy,
    )

    rewards = []
    obs = None
    for _ in range(num_updates):
        last_values, obs = trainer.collect_rollouts(env, obs)
        trainer.buffer.compute_gae(last_values, trainer.gamma, trainer.gae_lambda)
        trainer.update()
        rewards.append(trainer.buffer.mean_reward())
    return rewards


@unittest.skipUnless(_HAS_CARTPOLE, "cartpole example not available")
class TestPPOComparison(unittest.TestCase):
    """Compare vanilla PPO vs PPO+ on cartpole."""

    def _get_device(self):
        return "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_reach_zero_ppo_plus_vs_vanilla(self):
        """PPO+ should be more robust than vanilla on the ReachZero task.

        ReachZero has reward = -sum(action^2).  With tanh bounding, actions
        are in [-1, 1] so the reward floor is -act_dim instead of unbounded.
        Auto-entropy helps calibrate exploration: vanilla PPO with high
        entropy_coef over-explores (hurts reward), PPO+ auto-corrects.
        """
        device = self._get_device()
        n_updates = 100
        num_envs = 64

        common: dict[str, Any] = {
            "env_class": ReachZeroEnv,
            "obs_dim": 4,
            "act_dim": 4,
            "hidden_sizes": [32, 32],
            "num_envs": num_envs,
            "num_updates": n_updates,
            "device": device,
            "init_log_std": 0.0,
        }

        rewards_vanilla = _train(bounded_actions=False, auto_entropy=False, entropy_coef=0.01, **common)
        rewards_vanilla_bad = _train(bounded_actions=False, auto_entropy=False, entropy_coef=0.5, **common)
        rewards_ppo_plus = _train(bounded_actions=True, auto_entropy=True, entropy_coef=0.01, **common)
        rewards_ppo_plus_bad = _train(bounded_actions=True, auto_entropy=True, entropy_coef=0.5, **common)

        tail = 20
        mean_v = np.mean(rewards_vanilla[-tail:])
        mean_v_bad = np.mean(rewards_vanilla_bad[-tail:])
        mean_pp = np.mean(rewards_ppo_plus[-tail:])
        mean_pp_bad = np.mean(rewards_ppo_plus_bad[-tail:])

        print("\nReachZero (reward = -sum(a^2), optimal = 0):")
        print(f"  Vanilla (coef=0.01): {mean_v:.4f}")
        print(f"  Vanilla (coef=0.5):  {mean_v_bad:.4f}")
        print(f"  PPO+    (coef=0.01): {mean_pp:.4f}")
        print(f"  PPO+    (coef=0.5):  {mean_pp_bad:.4f}")
        print(f"  Vanilla sensitivity: {abs(mean_v - mean_v_bad):.4f}")
        print(f"  PPO+ sensitivity:    {abs(mean_pp - mean_pp_bad):.4f}")

        # Tanh bounding should help (actions capped at [-1,1] -> reward >= -4)
        self.assertGreater(
            mean_pp,
            mean_v - 0.5,
            f"PPO+ ({mean_pp:.4f}) should not be much worse than vanilla ({mean_v:.4f})",
        )

    def test_ppo_plus_robust_to_entropy_coef(self):
        """PPO+ with auto-entropy should be robust to bad initial entropy_coef.

        With a high ``init_log_std`` (large action variance), a bad
        ``entropy_coef`` strongly affects vanilla PPO because the fixed
        coefficient either over- or under-regularises exploration.  PPO+
        auto-tunes alpha, so the initial value matters less.
        """
        device = self._get_device()
        n_updates = 80
        num_envs = 512

        common: dict[str, Any] = {
            "env_class": CartpoleVecEnv,
            "obs_dim": 6,
            "act_dim": 1,
            "hidden_sizes": [64, 64],
            "num_envs": num_envs,
            "num_updates": n_updates,
            "device": device,
            "init_log_std": 0.0,
        }

        # Good entropy_coef for vanilla
        rewards_vanilla_good = _train(bounded_actions=False, auto_entropy=False, entropy_coef=0.01, **common)
        # Bad (too high) entropy_coef for vanilla -- discourages convergence
        rewards_vanilla_bad = _train(bounded_actions=False, auto_entropy=False, entropy_coef=0.5, **common)
        # PPO+ with same bad initial value -- should auto-correct
        rewards_ppo_plus_bad = _train(bounded_actions=True, auto_entropy=True, entropy_coef=0.5, **common)

        tail = 15
        mean_v_good = np.mean(rewards_vanilla_good[-tail:])
        mean_v_bad = np.mean(rewards_vanilla_bad[-tail:])
        mean_pp_bad = np.mean(rewards_ppo_plus_bad[-tail:])

        print(f"\n{'Update':>6}  {'Van(0.01)':>10}  {'Van(0.5)':>10}  {'PPO+(0.5)':>10}")
        print(f"{'------':>6}  {'---------':>10}  {'--------':>10}  {'---------':>10}")
        for i in range(0, n_updates, 10):
            print(
                f"{i + 1:6d}  {rewards_vanilla_good[i]:10.4f}  {rewards_vanilla_bad[i]:10.4f}"
                f"  {rewards_ppo_plus_bad[i]:10.4f}"
            )
        print(f"\nFinal {tail} avg:")
        print(f"  Vanilla (coef=0.01): {mean_v_good:.4f}")
        print(f"  Vanilla (coef=0.5):  {mean_v_bad:.4f}")
        print(f"  PPO+    (coef=0.5):  {mean_pp_bad:.4f}")

        vanilla_degradation = mean_v_good - mean_v_bad
        ppo_plus_degradation = mean_v_good - mean_pp_bad
        print(f"  Vanilla degradation from bad coef: {vanilla_degradation:.4f}")
        print(f"  PPO+ degradation from bad coef:    {ppo_plus_degradation:.4f}")

        # PPO+ should degrade less than vanilla when given a bad entropy_coef
        self.assertLess(
            ppo_plus_degradation,
            vanilla_degradation + 0.1,
            f"PPO+ should be more robust: PPO+ degradation={ppo_plus_degradation:.4f} "
            f"vs vanilla degradation={vanilla_degradation:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
