# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from newton._src.solvers.phoenx.rl_training.examples.train_g1_curriculum import (
    _make_parser,
    build_curriculum,
    build_train_config,
    check_phase_gate,
    select_curriculum_phases,
)


class TestG1TrainCurriculum(unittest.TestCase):
    def test_target_curricula_start_outside_success_radius(self):
        for recipe in ("simple-target", "advanced-target"):
            with self.subTest(recipe=recipe):
                phases = build_curriculum(recipe)
                self.assertGreater(len(phases), 0)
                for phase in phases:
                    self.assertGreater(phase.target_distance_start, phase.sparse_target_radius)
                    self.assertGreaterEqual(phase.target_distance_end, phase.target_distance_start)
                    self.assertLessEqual(phase.target_angle_min, phase.target_angle_max)

    def test_iteration_scale_keeps_nonzero_phase_lengths(self):
        phases = build_curriculum("advanced-target", iteration_scale=0.01)

        self.assertEqual(len(phases), 5)
        self.assertTrue(all(phase.iterations >= 1 for phase in phases))

    def test_select_curriculum_phases_supports_resumed_runs(self):
        phases = build_curriculum("advanced-target")

        selected = select_curriculum_phases(phases, start_phase=3, phase_count=1)

        self.assertEqual([(index, phase.name) for index, phase in selected], [(3, "long_forward_cone")])

    def test_phase_gate_checks_required_targets(self):
        phase = build_curriculum("advanced-target")[1]
        good_stats = [
            {
                "target_position": (0.6, 0.0),
                "strict_success_fraction": 1.0,
                "fall_fraction": 0.0,
                "tilt_violation_fraction": 0.0,
            },
            {
                "target_position": (1.0, 0.0),
                "strict_success_fraction": 0.95,
                "fall_fraction": 0.01,
                "tilt_violation_fraction": 0.0,
            },
        ]

        passed, failures = check_phase_gate(phase, good_stats)

        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_phase_gate_rejects_failed_or_missing_targets(self):
        phase = build_curriculum("advanced-target")[1]
        bad_stats = [
            {
                "target_position": (0.6, 0.0),
                "strict_success_fraction": 1.0,
                "fall_fraction": 0.0,
                "tilt_violation_fraction": 0.0,
            },
        ]

        passed, failures = check_phase_gate(phase, bad_stats)

        self.assertFalse(passed)
        self.assertTrue(any("missing eval target x=1" in failure for failure in failures))

    def test_train_config_uses_dense_target_without_command_randomization(self):
        args = _make_parser().parse_args(["--dry-run"])
        phase = build_curriculum("simple-target")[0]

        config = build_train_config(
            phase,
            args,
            seed=123,
            resume_checkpoint=None,
            checkpoint_path=Path("/tmp/phoenx_g1_test_{iteration}.npz"),
        )

        self.assertEqual(config.env_config.reward_mode, "dense_target")
        self.assertFalse(config.randomize_commands)
        self.assertTrue(config.randomize_target_positions)
        self.assertGreater(config.target_distance_start, config.env_config.sparse_target_radius)
        self.assertEqual(config.env_config.command, (0.0, 0.0, 0.0))
        self.assertEqual(config.env_config.ground_friction, 0.4)
        self.assertEqual(config.target_curriculum_start_samples, 0)
        self.assertTrue(config.randomize_target_distance_range)
        self.assertGreater(config.ppo_config.mirror_loss_coeff, 0.0)


if __name__ == "__main__":
    unittest.main()
