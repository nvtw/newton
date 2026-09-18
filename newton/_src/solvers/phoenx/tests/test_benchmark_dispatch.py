# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Benchmarks must measure the dispatcher used by simulation steps."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from newton._src.solvers.phoenx.benchmarks import bench_reorder_coalescing, bench_singleworld_tail_fuse


class TestBenchmarkDispatch(unittest.TestCase):
    def test_singleworld_coalescing_uses_live_dispatcher(self):
        solve = Mock()
        world = SimpleNamespace(step_layout="single_world", substep_dt=0.01, _dispatcher=SimpleNamespace(solve=solve))
        with (
            patch.object(bench_reorder_coalescing.wp, "get_device", return_value=SimpleNamespace(is_cuda=False)),
            patch.object(bench_reorder_coalescing.wp, "synchronize_device"),
        ):
            bench_reorder_coalescing._bench_solve_main(world, n_runs=4, warmup=2, trials=3)
        self.assertEqual(solve.call_count, 14)
        for call in solve.call_args_list:
            self.assertEqual(float(call.args[0]), 100.0)

    def test_tail_benchmark_captures_live_dispatcher(self):
        solve = Mock()
        world = SimpleNamespace(device="cuda:0", substep_dt=0.01, _dispatcher=SimpleNamespace(solve=solve))
        with (
            patch.object(bench_singleworld_tail_fuse.wp, "ScopedCapture"),
            patch.object(bench_singleworld_tail_fuse.wp, "capture_launch") as launch,
            patch.object(bench_singleworld_tail_fuse.wp, "synchronize_device"),
        ):
            bench_singleworld_tail_fuse._bench_solve(world, n_runs=4, warmup=2, trials=3)
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(launch.call_count, 12)
        for call in solve.call_args_list:
            self.assertEqual(float(call.args[0]), 100.0)


if __name__ == "__main__":
    unittest.main()
