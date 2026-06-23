# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from newton._src.solvers.phoenx.experimental.train_g1_command_curriculum import (
    _make_parser,
    build_command_curriculum,
    build_train_config,
    check_phase_gate,
    select_phases,
)


class TestG1CommandCurriculum(unittest.TestCase):
    def test_advanced_command_curriculum_has_staged_gates(self):
        phases = build_command_curriculum("advanced-command")

        self.assertEqual([phase.name for phase in phases], ["slow_forward", "forward_walk", "omnidirectional_walk"])
        self.assertEqual(phases[0].gate_commands, ((0.3, 0.0, 0.0),))
        self.assertIn((0.6, 0.0, 0.0), phases[1].gate_commands)

    def test_select_phases_supports_resumed_runs(self):
        phases = build_command_curriculum("advanced-command")

        selected = select_phases(phases, start_phase=1, phase_count=1)

        self.assertEqual([(index, phase.name) for index, phase in selected], [(1, "forward_walk")])

    def test_train_config_uses_rollout_command_randomization(self):
        args = _make_parser().parse_args(["--dry-run"])
        phase = build_command_curriculum("simple-forward")[0]

        config = build_train_config(
            phase,
            args,
            seed=123,
            resume_checkpoint=None,
            checkpoint_path=Path("/tmp/phoenx_g1_command_test_{iteration}.npz"),
        )

        self.assertEqual(config.env_config.reward_mode, "nanog1_dense")
        self.assertTrue(config.randomize_commands)
        self.assertEqual(config.command_sampling, "rollout")
        self.assertEqual(config.command_x_range, phase.command_x_range)
        self.assertGreater(config.ppo_config.mirror_loss_coeff, 0.0)
        self.assertEqual(config.ppo_config.reward_clip, 4.0)

    def test_reward_clip_override_supports_survival_ablation(self):
        args = _make_parser().parse_args(["--dry-run", "--reward-clip", "0.0"])
        phase = build_command_curriculum("simple-forward")[0]

        config = build_train_config(
            phase,
            args,
            seed=123,
            resume_checkpoint=None,
            checkpoint_path=Path("/tmp/phoenx_g1_command_test_{iteration}.npz"),
        )

        self.assertEqual(config.ppo_config.reward_clip, 0.0)

    def test_phase_gate_checks_command_metrics(self):
        phase = build_command_curriculum("simple-forward")[0]
        stats = [
            {
                "command": (0.3, 0.0, 0.0),
                "steps": 700,
                "fall_fraction": 0.0,
                "mean_survival_steps": 700.0,
                "mean_tracking_perf": 0.5,
            }
        ]

        passed, failures = check_phase_gate(phase, stats)

        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_phase_gate_rejects_falling_commands(self):
        phase = build_command_curriculum("simple-forward")[0]
        stats = [
            {
                "command": (0.3, 0.0, 0.0),
                "steps": 700,
                "fall_fraction": 1.0,
                "mean_survival_steps": 100.0,
                "mean_tracking_perf": 0.0,
            }
        ]

        passed, failures = check_phase_gate(phase, stats)

        self.assertFalse(passed)
        self.assertTrue(any("fall_fraction" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
