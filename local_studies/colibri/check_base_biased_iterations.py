"""Measure base drift while changing only biased solver iterations."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx.solver import SolverPhoenX


def main():
    """Retain the dynamic base, friction, temporal budget, and final relaxation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "newton").rglob("*.py"))
    paths += sorted(p for p in (root / "newton/examples/assets/colibri").rglob("*") if p.is_file())

    def hashes():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    report = {
        "source_asset_sha256": hashes(),
        "biased_iterations": args.iterations,
        "substeps_per_120hz_update": 30,
        "velocity_iterations": 1,
        "scope": "Diagnostic increased solve work; not a matched PhysX performance claim",
    }
    original = SolverPhoenX.__init__

    def initialize(self, *positional, **kwargs):
        kwargs["solver_iterations"] = args.iterations
        original(self, *positional, **kwargs)
        assert self.world.substeps == 30
        assert self.world.velocity_iterations == 1
        assert self.world.velocity_relaxation == "final_substep"

    SolverPhoenX.__init__ = initialize
    sys.argv = [
        sys.argv[0],
        "--frames",
        str(args.frames),
        "--substeps",
        "30",
        "--save-history",
        "--output",
        str(args.output) + ".json",
    ]
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
        with np.load(str(args.output) + ".npz") as data:
            index = data["labels"].astype(str).tolist().index("FrameGround")
            q = data["q_history"][:, index].astype(float)
            initial = data["initial_q"][index].astype(float)
            delta = q[:, :3] - initial[:3]
            rotations = q[:, 3:] / np.linalg.norm(q[:, 3:], axis=1)[:, None]
            initial_rotation = initial[3:] / np.linalg.norm(initial[3:])
            angle = 2 * np.arccos(np.clip(np.abs(rotations @ initial_rotation), 0, 1))
            report["final_base_displacement_m"] = delta[-1].tolist()
            report["max_base_horizontal_drift_m"] = float(np.linalg.norm(delta[:, :2], axis=1).max())
            report["max_base_rotation_rad"] = float(angle.max())
            report["final_base_rotation_rad"] = float(angle[-1])
    finally:
        SolverPhoenX.__init__ = original
        report["source_assets_unchanged"] = hashes() == report["source_asset_sha256"]
        Path(str(args.output) + ".base.json").write_text(json.dumps(report, indent=2))
        if not report["source_assets_unchanged"]:
            raise RuntimeError("Sources or assets changed during diagnostic")


if __name__ == "__main__":
    main()
