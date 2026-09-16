"""Gate added native final relaxation work with clean public physics timing."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import newton.examples
from local_studies.colibri.validate_spatial_priority import compare_arrays
from newton._src.solvers.phoenx.solver import SolverPhoenX


def main():
    """Keep every public scene option except the explicit velocity count unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity-iterations", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--frames", type=int, default=18000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-trajectory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "newton").rglob("*.py"))
    paths += sorted((Path(newton.examples.get_asset_directory()) / "colibri").rglob("*"))
    paths = [p for p in paths if p.is_file()]

    def fingerprints():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    hashes = fingerprints()
    metadata = {
        "velocity_iterations": args.velocity_iterations,
        "substeps_per_collision_update": 30,
        "collision_hz": 120,
        "additional_relaxation_sweeps_per_frame": 2 * (args.velocity_iterations - 1),
        "scope": (
            "Public bounds and clean step timing; unchanged default final velocity work"
            if args.velocity_iterations == 1
            else "Public bounds and clean step timing; increased velocity work, not a matched PhysX budget"
        ),
        "source_asset_sha256": hashes,
        "source_scope": "All Newton Python sources and Colibri asset files",
        "phase_capture_hooks": False,
        "frames_requested": args.frames,
    }
    if args.reference_trajectory is not None:
        metadata["reference_trajectory"] = str(args.reference_trajectory.resolve())
        metadata["reference_sha256"] = hashlib.sha256(args.reference_trajectory.read_bytes()).hexdigest()
    meta_path = Path(args.output + ".source.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    old = SolverPhoenX.__init__

    def initialize(self, *positional, **kwargs):
        kwargs["velocity_iterations"] = args.velocity_iterations
        old(self, *positional, **kwargs)
        assert self.world.velocity_relaxation == "final_substep"
        assert self.world.substeps == 30
        assert self.world.solver_iterations == 1

    SolverPhoenX.__init__ = initialize
    sys.argv = [
        sys.argv[0],
        "--frames",
        str(args.frames),
        "--substeps",
        "30",
        "--save-history",
        "--output",
        args.output + ".json",
    ]
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
        if args.reference_trajectory is not None:
            equal = compare_arrays(args.reference_trajectory, args.output + ".npz")
            metadata["reference_arrays_byte_equal"] = equal
            metadata["reference_unchanged"] = (
                hashlib.sha256(args.reference_trajectory.read_bytes()).hexdigest() == metadata["reference_sha256"]
            )
            if not equal or not all(equal.values()) or not metadata["reference_unchanged"]:
                raise AssertionError("Saved trajectory differs from the reference; see source metadata")
    finally:
        SolverPhoenX.__init__ = old
        metadata["source_assets_unchanged"] = fingerprints() == hashes
        meta_path.write_text(json.dumps(metadata, indent=2))
        if not metadata["source_assets_unchanged"]:
            raise RuntimeError("Canonical solver, collision, scene or assets changed during validation")


if __name__ == "__main__":
    main()
