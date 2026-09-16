"""Run public Colibri with only the native velocity iteration count changed."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri.check_native_velocity_iterations import residuals
from newton._src.solvers.phoenx.solver import SolverPhoenX


def main():
    """Set the existing constructor option before the public scene captures its graph."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity-iterations", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old = SolverPhoenX.__init__

    def initialize(self, *positional, **kwargs):
        kwargs["velocity_iterations"] = args.velocity_iterations
        old(self, *positional, **kwargs)
        assert self.world.velocity_relaxation == "final_substep"

    SolverPhoenX.__init__ = initialize
    sys.argv = [
        sys.argv[0],
        "--frames",
        "330",
        "--save-history",
        "--snapshot",
        args.output + "_snapshot.npz",
        "--output",
        args.output + "_trajectory.json",
    ]
    try:
        runpy.run_module("local_studies.colibri.capture_support_relax", run_name="__main__")
    finally:
        SolverPhoenX.__init__ = old
    s = np.load(args.output + "_snapshot.npz")
    report = residuals(
        s, s["after_velocity"], s["after_angular_velocity"], s["after_impulses"], s["after_joint_accumulated"]
    )
    report.update(
        velocity_iterations=args.velocity_iterations,
        collision_hz=120,
        substeps=30,
        scope="Live330; more native final relaxation work; changed trajectory, not matched velocity budget",
    )
    Path(args.output + ".json").write_text(json.dumps(report, indent=2))
    print("LIVE_NATIVE_RELAX", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
