"""Isolate tangent correction relaxation without pooling friction capacity.

Use through direct_relax_no_skip with COLIBRI_STAGE_RUNNER pointing here.
COLIBRI_FRICTION_RELAXATION defaults to 1.0 (exact source control).
This is a diagnostic, not a proposed production default.
"""

import ast
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

from newton._src.solvers.phoenx import solver_phoenx
from newton._src.solvers.phoenx.articulations import maximal_contact_gs as gs


def make_kernel(factor):
    """Keep projection, force limits, normal updates and impulse scatter intact."""
    if not 0.0 < factor <= 1.0:
        raise ValueError("Friction correction factor must be in (0, 1]")
    source = Path(gs.__file__).read_text()
    name = "iterate_maximal_contact_runs_kernel"
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    original = "\n".join(source.splitlines()[node.decorator_list[0].lineno - 1 : node.end_lineno])
    modified = original
    for axis in (0, 1):
        marker = f"sor_boost * rhs{axis},"
        assert modified.count(marker) == 1, "Unexpected native friction projection"
        if factor != 1.0:
            modified = modified.replace(marker, f"wp.float32({factor!r}) * sor_boost * rhs{axis},")
    path = Path(tempfile.mkdtemp(prefix="colibri_friction_relaxation_")) / "kernel.py"
    path.write_text(modified + "\n")
    module_name = "colibri_friction_relaxation_generated"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Resolve the extracted kernel against the actual runtime-owned native math.
    module.__dict__.update({key: value for key, value in vars(gs).items() if not key.startswith("__")})
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, name), {
        "factor": factor,
        "native_path": gs.__file__,
        "native_kernel_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "candidate_kernel_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "generated_path": str(path),
        "unit_factor_source_identical": original == modified,
        "scope": "Tangent residual correction only; original normal solve, per-point cones and paired impulses",
    }


def main():
    """Bind the actual contact iteration launch and save the exact variant."""
    factor = float(os.environ.get("COLIBRI_FRICTION_RELAXATION", "1.0"))
    candidate, provenance = make_kernel(factor)
    name = "iterate_maximal_contact_runs_kernel"
    original = getattr(solver_phoenx, name)
    assert original is getattr(gs, name)
    setattr(solver_phoenx, name, candidate)
    setattr(gs, name, candidate)
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    print("FRICTION_RELAXATION_BINDING", json.dumps(provenance), flush=True)
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        assert getattr(solver_phoenx, name) is candidate
        output.with_suffix(".friction_relaxation.json").write_text(json.dumps(provenance, indent=2))
        setattr(solver_phoenx, name, original)
        setattr(gs, name, original)


if __name__ == "__main__":
    main()
