# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run unchanged analytical joint assertions through the public block policy."""

import unittest
from unittest.mock import patch

import newton
from local_studies.colibri.test_fused_bilateral import adapted_suite

if __name__ == "__main__":
    original = newton.solvers.SolverPhoenX.__init__

    def initialize(self, *args, **kwargs):
        kwargs["step_layout"] = "single_world"
        kwargs["joint_solver"] = "block_pgs"
        original(self, *args, **kwargs)

    with patch.object(newton.solvers.SolverPhoenX, "__init__", initialize):
        result = unittest.TextTestRunner(verbosity=2).run(adapted_suite())
    raise SystemExit(not result.wasSuccessful())
