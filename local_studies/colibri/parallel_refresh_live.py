"""Live exact-arithmetic control for parallel contact mobility preparation."""

import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri.parallel_mobility_refresh import make_kernel
from newton._src.solvers.phoenx import solver_phoenx
from newton._src.solvers.phoenx.articulations import maximal_contact_gs


def main():
    """Replace only the actual refresh launch binding and retain source evidence."""
    candidate, _, provenance = make_kernel()
    original = solver_phoenx.refresh_maximal_contact_mobility_kernel
    assert original is maximal_contact_gs.refresh_maximal_contact_mobility_kernel
    solver_phoenx.refresh_maximal_contact_mobility_kernel = candidate
    maximal_contact_gs.refresh_maximal_contact_mobility_kernel = candidate
    assert solver_phoenx.refresh_maximal_contact_mobility_kernel is candidate
    assert maximal_contact_gs.refresh_maximal_contact_mobility_kernel is candidate
    provenance["launch_binding"] = "solver_phoenx.refresh_maximal_contact_mobility_kernel"
    provenance["arithmetic_changed"] = False
    provenance["scope"] = "Parallel per-contact ownership only; original conditioned solver and contact law"
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    print("PARALLEL_REFRESH_BINDING", json.dumps(provenance), flush=True)
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        assert solver_phoenx.refresh_maximal_contact_mobility_kernel is candidate
        output.with_suffix(".parallel_refresh.json").write_text(json.dumps(provenance, indent=2))
        solver_phoenx.refresh_maximal_contact_mobility_kernel = original
        maximal_contact_gs.refresh_maximal_contact_mobility_kernel = original


if __name__ == "__main__":
    main()
