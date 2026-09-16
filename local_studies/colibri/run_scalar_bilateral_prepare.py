"""Process-local pre-optimization preparation control, all other source current."""

import runpy

import warp as wp

from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem
from newton._src.solvers.phoenx.constraints.bilateral_joint import prepare_bilateral_joint_blocks


def main():
    original = BlockJointSystem.prepare_and_factor

    def prepare(self, idt):
        if self.enabled:
            w = self._block_world
            wp.launch(
                prepare_bilateral_joint_blocks,
                w.num_joints,
                [w.constraints, w.bodies, w._copy_state],
                device=self.model.device,
            )

    BlockJointSystem.prepare_and_factor = prepare
    try:
        runpy.run_module("local_studies.colibri.check_clean_velocity_iterations", run_name="__main__")
    finally:
        BlockJointSystem.prepare_and_factor = original


if __name__ == "__main__":
    main()
