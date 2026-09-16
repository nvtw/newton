"""Check the experimental temporal schedule against conservation fixtures."""
import unittest
from unittest.mock import patch
import newton
from local_studies.colibri.temporal_schedule import install_temporal_schedule
from newton._src.solvers.phoenx.tests.test_maximal_contact_conservation import TestMaximalContactConservation

if __name__ == "__main__":
    install_temporal_schedule()
    original = newton.solvers.SolverPhoenX.__init__

    def initialize(self, *args, **kwargs):
        kwargs.update(substeps=30, solver_iterations=1, prepare_refresh_stride=1)
        original(self, *args, **kwargs)

    with patch.object(newton.solvers.SolverPhoenX, "__init__", initialize):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMaximalContactConservation)
        result = unittest.TextTestRunner().run(suite)
    raise SystemExit(not result.wasSuccessful())
