"""Run the actual G1 harness with only the pre-cache packed kernel substituted."""

import importlib.util
import runpy

from newton._src.solvers.phoenx.articulations import reduced_contact_block

spec = importlib.util.spec_from_file_location("g1_mobility_reference_factory", "/tmp/g1_mobility_reference_factory.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
for width in reduced_contact_block._SOLVE_GENERALIZED_CONTACT_TILE_OPS:
    operations = reference._make_reference_ops(width)
    reduced_contact_block._SOLVE_GENERALIZED_CONTACT_TILE_OPS[width] = operations
    reduced_contact_block._SOLVE_GENERALIZED_CONTACT_TILE_DEVICES[width] = operations[0]
    reduced_contact_block._SOLVE_GENERALIZED_CONTACT_TILE_KERNELS[width] = operations[1]
runpy.run_module("local_studies.colibri.check_g1_policy", run_name="__main__")
