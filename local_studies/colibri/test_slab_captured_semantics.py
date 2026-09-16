"""Run existing drive/conservation assertions with forced Jacobi body copies."""

import unittest
from unittest.mock import patch

import newton
from local_studies.colibri.bilateral_pgs import install_fused
from local_studies.colibri.contact_chunks import install as install_chunks
from local_studies.colibri.contact_chunks import reserve_capacity
from local_studies.colibri.slab_adapter import install as install_slabs
from local_studies.colibri.test_fused_bilateral import adapted_suite

reserve_capacity()

original = newton.solvers.SolverPhoenX.__init__


def initialize(self, *args, **kwargs):
    kwargs["step_layout"] = "single_world"
    kwargs["mass_splitting"] = True
    kwargs["max_colored_partitions"] = 8
    kwargs["mass_splitting_batch_size"] = 1
    original(self, *args, **kwargs)
    install_fused(self, mass_splitting=True)
    install_chunks(self, chunk_size=6)
    install_slabs(self, colors_per_slab=8, gpu=True)


with patch.object(newton.solvers.SolverPhoenX, "__init__", initialize):
    result = unittest.TextTestRunner().run(adapted_suite())
raise SystemExit(not result.wasSuccessful())
