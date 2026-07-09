# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CI smoke coverage for the experimental Kamino DVI solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solvers_dvi import TestDVISolver


_DVI_SMOKE_TESTS = (
    "test_00_config_selection",
    "test_01_dvi_solve_dense_dual_problem",
    "test_02_public_solver_step_with_dvi",
    "test_03_dvi_solve_single_contact",
    "test_03a_sparse_dvi_filtered_matvec_matches_full_rows",
    "test_04_dvi_solve_active_joint_limit",
    "test_06_dvi_warmstart_modes",
    "test_07_dvi_singular_limit_rows_remain_finite",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load a focused cross-platform subset into the main test suite."""
    del loader, tests, pattern
    return unittest.TestSuite(TestDVISolver(name) for name in _DVI_SMOKE_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
