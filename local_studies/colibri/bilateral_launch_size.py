# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local launch-only experiment: distribute small joint batches across blocks."""

import runpy

import warp as wp

from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem
from newton._src.solvers.phoenx.constraints.bilateral_joint import prepare_bilateral_joint_blocks


def prepare_and_factor(self, idt):
    if self.enabled:
        world = self._block_world
        wp.launch(
            prepare_bilateral_joint_blocks,
            dim=world.num_joints,
            inputs=[world.constraints, world.bodies, world._copy_state],
            device=self.model.device,
            block_dim=32,
        )


if __name__ == "__main__":
    BlockJointSystem.prepare_and_factor = prepare_and_factor
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
