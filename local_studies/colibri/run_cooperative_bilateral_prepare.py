"""Local public-scene byte-equivalence run for cooperative preparation."""

import argparse
import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri.check_native_velocity_iterations import validate_reference
from local_studies.colibri.cooperative_bilateral_prepare import launch
from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--block-dim", type=int, default=32)
    parser.add_argument("--reference", default="/tmp/colibri_spatial_priority_validation_velocity1_first.npz")
    options, remaining = parser.parse_known_args()
    if "--output" not in remaining:
        raise ValueError("Explicit independent output path required")
    output = Path(remaining[remaining.index("--output") + 1])
    original = BlockJointSystem.prepare_and_factor

    def prepare(self, idt):
        if self.enabled:
            w = self._block_world
            launch(w.constraints, w.bodies, w._copy_state, w.num_joints, self.model.device, options.block_dim)

    BlockJointSystem.prepare_and_factor = prepare
    sys.argv = [sys.argv[0], *remaining]
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        BlockJointSystem.prepare_and_factor = original
    validation = validate_reference(output.with_suffix(".npz"), options.reference, "trajectory")
    output.with_suffix(".equivalence.json").write_text(json.dumps(validation, indent=2))
    print("PREPARE_EQUIVALENCE", validation)


if __name__ == "__main__":
    main()
