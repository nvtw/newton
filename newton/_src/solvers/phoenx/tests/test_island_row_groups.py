# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Islands of one world solved in separate blocks.

One world with reduced-coordinate robots spreads the islands of its free bodies over several
solver blocks. Islands share no body that moves, so the result must not depend on how many
blocks there are.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _run(
    groups: int | None, *, sleeping_threshold: float, frames: int = 25
) -> tuple[np.ndarray, newton.solvers.SolverPhoenX]:
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    # A robot keeps the scene on the reduced path; the stacks beside it are free bodies.
    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.2, hy=0.2, hz=0.2)
    arm = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity()))
    builder.add_shape_box(arm, hx=0.05, hy=0.05, hz=0.2)
    root = builder.add_joint_free(base)
    hinge = builder.add_joint_revolute(
        base,
        arm,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
        target_ke=200.0,
        target_kd=20.0,
    )
    builder.add_articulation([root, hinge])
    # Stacks of different heights, some leaning so their boxes slide and turn.
    for stack in range(9):
        x = 1.5 + 0.8 * (stack % 3)
        y = -0.8 + 0.8 * (stack // 3)
        for level in range(1 + stack % 4):
            offset = 0.03 * level * (stack % 2)
            body = builder.add_body(xform=wp.transform(wp.vec3(x + offset, y, 0.11 + 0.21 * level), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize()
    solver = newton.solvers.SolverPhoenX(
        model, joint_mode="reduced", substeps=4, solver_iterations=4, sleeping_velocity_threshold=sleeping_threshold
    )
    if groups is not None:
        solver.world._allocate_partition(groups)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    contacts = model._collision_pipeline.contacts()
    with wp.ScopedCapture() as capture:
        solver.collide(state, contacts)
        solver.step(state, state, control, contacts, 1.0 / 60.0)
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    return state.body_q.numpy(), solver


@unittest.skipUnless(wp.is_cuda_available(), "island row groups run in the captured CUDA step")
class TestIslandRowGroups(unittest.TestCase):
    def test_islands_in_separate_blocks_move_as_in_one_block(self):
        """Preserve stack motion across group counts with and without sleeping."""
        for sleeping_threshold in (0.0, 0.05):
            with self.subTest(sleeping_threshold=sleeping_threshold):
                grouped, solver = _run(None, sleeping_threshold=sleeping_threshold)
                world = solver.world
                self.assertEqual(world._multi_world_scheduler, "block_world")
                self.assertGreaterEqual(world._partition_count, world.device.sm_count)
                # Rows really went to several blocks.
                self.assertGreater(int(np.count_nonzero(world._per_world_element_count.numpy())), 1)
                single, _ = _run(1, sleeping_threshold=sleeping_threshold)
                np.testing.assert_array_equal(grouped, single)
                # Fewer blocks than islands share blocks; the result still does not change.
                shared, _ = _run(3, sleeping_threshold=sleeping_threshold)
                np.testing.assert_array_equal(shared, single)
                self.assertTrue(np.isfinite(grouped).all())
                self.assertGreater(float(grouped[2:, 2].min()), 0.05)  # no box sank into the ground


if __name__ == "__main__":
    unittest.main()
