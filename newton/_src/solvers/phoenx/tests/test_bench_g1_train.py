# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from types import SimpleNamespace

from newton._src.solvers.phoenx.benchmarks.bench_g1_train import _summarize_measured_history
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog import (
    _g1_env_config,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog import (
    build_arg_parser as build_leapfrog_arg_parser,
)
from newton._src.solvers.phoenx.rl_training import g1_recipe


def _stats(elapsed: float) -> SimpleNamespace:
    return SimpleNamespace(rollout_seconds=elapsed, update_seconds=0.0)


class TestBenchG1Train(unittest.TestCase):
    def test_graph_leapfrog_excludes_final_drain(self):
        measured, env_sps, excluded_drain = _summarize_measured_history(
            [_stats(4.0), _stats(2.0), _stats(0.1)],
            warmup_iterations=1,
            execution_mode="graph_leapfrog",
            samples_per_interval=100,
        )

        self.assertEqual(measured, [_stats(2.0)])
        self.assertEqual(env_sps, 50.0)
        self.assertTrue(excluded_drain)

    def test_rejects_graph_run_with_only_drain_after_warmup(self):
        with self.assertRaisesRegex(ValueError, "no complete measured training intervals"):
            _summarize_measured_history(
                [_stats(4.0), _stats(0.1)],
                warmup_iterations=1,
                execution_mode="graph_leapfrog",
                samples_per_interval=100,
            )

    def test_uses_aggregate_elapsed_time(self):
        _, env_sps, excluded_drain = _summarize_measured_history(
            [_stats(1.0), _stats(3.0)],
            warmup_iterations=0,
            execution_mode="eager",
            samples_per_interval=100,
        )

        self.assertEqual(env_sps, 50.0)
        self.assertFalse(excluded_drain)

    def test_leapfrog_benchmark_inherits_production_environment(self):
        args = build_leapfrog_arg_parser().parse_args([])
        config = _g1_env_config(args)

        self.assertEqual(config.articulation_mode, g1_recipe.ARTICULATION_MODE)
        self.assertEqual(config.reduced_articulation_path, g1_recipe.REDUCED_ARTICULATION_PATH)
        self.assertEqual(config.contact_geometry, g1_recipe.CONTACT_GEOMETRY)
        self.assertEqual(config.contact_friction_model, g1_recipe.CONTACT_FRICTION_MODEL)
        self.assertEqual(config.actuation_model, g1_recipe.ACTUATION_MODEL)
        self.assertEqual(config.observation_mode, g1_recipe.OBSERVATION_MODE)


if __name__ == "__main__":
    unittest.main()
