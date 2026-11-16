# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check transactional case sealing without private assets or wrapping logic."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_control import _seal_cases
from newton.solvers import SolverPhoenX, phoenx


@wp.kernel
def _retry_tape_allocations(dynamics: phoenx.DynamicsData, joints: wp.array[int]):
    i = wp.tid()
    joints[i] = phoenx.add_joint(
        dynamics, phoenx.joint_desc(0, i + 1, wp.transform_identity(), wp.transform_identity())
    )


class TestBottlingSealing(unittest.TestCase):
    def test_tape_replaces_hinges_only_after_complete_allocation(self):
        """Failed tape batches leave folding hinges active and release their slots."""
        with wp.ScopedDevice("cpu"):
            for capacity in (3, 4):
                with self.subTest(capacity=capacity):
                    builder = newton.ModelBuilder(gravity=(0, 0, 0))
                    root = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
                    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1, cfg=builder.ShapeConfig(density=0))
                    authored = [builder.add_joint_fixed(-1, root)]
                    for _ in range(4):
                        body = builder.add_link(mass=0.1, inertia=wp.mat33(np.eye(3) * 0.01))
                        authored.append(builder.add_joint_revolute(root, body, axis=newton.Axis.Z))
                    builder.add_articulation(authored)
                    builder.reserve_dynamic_capacity(joints=capacity)
                    model = builder.finalize()
                    solver = SolverPhoenX(model, joint_mode="maximal_pgs", step_layout="single_world", substeps=1)
                    state, control = model.state(), model.control()
                    bodies = wp.array([[0, 1, 2, 3, 4]], dtype=int)
                    hinges = wp.array([authored[1:]], dtype=int)
                    requests, sealed = wp.ones(1, dtype=int), wp.zeros(1, dtype=int)
                    tape = wp.full((1, 4), -1, dtype=int)
                    pose, velocity = state.body_q.numpy().copy(), state.body_qd.numpy().copy()
                    mass, inertia = model.body_mass.numpy().copy(), model.body_inertia.numpy().copy()
                    inputs = [
                        solver.dynamics.data,
                        state.body_q,
                        state.body_qd,
                        bodies,
                        requests,
                        sealed,
                        tape,
                        hinges,
                        model.joint_enabled,
                    ]
                    wp.launch(_seal_cases, dim=1, inputs=inputs)
                    np.testing.assert_array_equal(state.body_q.numpy(), pose)
                    np.testing.assert_array_equal(state.body_qd.numpy(), velocity)
                    np.testing.assert_array_equal(model.body_mass.numpy(), mass)
                    np.testing.assert_array_equal(model.body_inertia.numpy(), inertia)
                    success = capacity == 4
                    self.assertEqual(int(sealed.numpy()[0]), int(success))
                    np.testing.assert_array_equal(model.joint_enabled.numpy()[authored[1:]], not success)
                    solver.step(state, state, control, None, 1 / 240)
                    self.assertEqual(int(solver.dynamics.joints.active.numpy().sum()), 4 if success else 0)
                    if not success:
                        np.testing.assert_array_equal(tape.numpy(), -1)
                        # Cancelled allocations spend one commit in quarantine.
                        solver.step(state, state, control, None, 1 / 240)
                        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
                        retry_ids = wp.full(3, -1, dtype=int)
                        wp.launch(_retry_tape_allocations, dim=3, inputs=[solver.dynamics.data, retry_ids])
                        self.assertTrue((retry_ids.numpy() >= 0).all())
                        solver.step(state, state, control, None, 1 / 240)
                        self.assertEqual(int(solver.dynamics.joints.active.numpy().sum()), 3)
                    else:
                        ids = tape.numpy().copy()
                        wp.launch(_seal_cases, dim=1, inputs=inputs)
                        np.testing.assert_array_equal(tape.numpy(), ids)


if __name__ == "__main__":
    unittest.main()
