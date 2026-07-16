# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility aggregate for Kamino DVI solver tests."""

from newton._src.solvers.kamino.tests.test_solvers_dvi_config import TestDVIConfig
from newton._src.solvers.kamino.tests.test_solvers_dvi_contacts import TestDVIContacts
from newton._src.solvers.kamino.tests.test_solvers_dvi_dense import TestDVIDense
from newton._src.solvers.kamino.tests.test_solvers_dvi_dr_legs import TestDVIDrLegs
from newton._src.solvers.kamino.tests.test_solvers_dvi_metrics import TestDVIMetrics
from newton._src.solvers.kamino.tests.test_solvers_dvi_sparse import TestDVISparse


class TestDVISolver(
    TestDVIConfig,
    TestDVIDense,
    TestDVISparse,
    TestDVIContacts,
    TestDVIMetrics,
    TestDVIDrLegs,
):
    """Aggregate focused DVI tests for existing quality gates."""


if __name__ == "__main__":
    import unittest

    unittest.main()
