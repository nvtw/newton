"""Local stabilized Coulomb law using the actual accumulated normal impulse."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import warp as wp

from newton._src.solvers.phoenx.constraints import contact_projection


@wp.func
def actual_normal_load(
    lambda_n: wp.float32, eff_n: wp.float32, bias_n: wp.float32, mass_coeff_n: wp.float32, sor_boost: wp.float32
) -> wp.float32:
    return wp.max(lambda_n, wp.float32(0.0))


original = contact_projection._friction_normal_lambda
contact_projection._friction_normal_lambda = actual_normal_load
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
paths = [Path(__file__).resolve(), *sorted(Path("newton/_src").rglob("*.py"))]
paths = [p for p in paths if "tests" not in p.parts]
paths += [
    Path("newton/examples/phoenx/example_phoenx_colibri.py"),
    Path("newton/examples/kamino/example_kamino_colibri.py"),
]
paths += sorted(p for p in Path("newton/examples/assets/colibri").rglob("*") if p.is_file())


def fingerprints():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


source_before = fingerprints()
arguments = list(sys.argv)
try:
    runpy.run_module(module, run_name="__main__")
finally:
    contact_projection._friction_normal_lambda = original

    unchanged = fingerprints() == source_before
    if "--output" in arguments:
        output = Path(arguments[arguments.index("--output") + 1])
        output.with_suffix(".candidate.json").write_text(
            json.dumps(
                {
                    "source_sha256": source_before,
                    "source_unchanged": unchanged,
                    "arguments": arguments,
                    "law": "Single physical velocity stabilized Coulomb cap uses actual accumulated normal impulse",
                    "production_edits": False,
                },
                indent=2,
            )
        )
    assert unchanged, "Source or scene assets changed during local candidate run"
