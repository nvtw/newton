# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify temporal integration ownership across runtime weld components."""

import unittest

import numpy as np
import warp as wp

import newton


class TestTemporalWeldOwnership(unittest.TestCase):
    def _check_chain(self, device, *, capture):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            bodies = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(9)]
            joints = [builder.add_joint_revolute(-1, bodies[0], axis=newton.Axis.Z)]
            joints.extend(builder.add_joint_fixed(bodies[i - 1], bodies[i]) for i in range(1, 9))
            builder.add_articulation(joints)
            model = builder.finalize()
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                temporal_iterations=True,
                solver_iterations=4,
                substeps=1,
            )
            world = solver.world
            names = ("joint_type", "joint_parent", "joint_child", "joint_dof_dim", "joint_enabled")
            original = {name: getattr(model, name).numpy().copy() for name in names}
            # Only classification is exercised after reordering these joint rows;
            # joint coordinate offsets and articulation topology are not consumed.
            for reverse in (False, True):
                with self.subTest(device=str(device), reverse=reverse, capture=capture):
                    order = np.arange(model.joint_count)
                    if reverse:
                        order = order[::-1]
                    for name in names:
                        getattr(model, name).assign(original[name][order].copy())
                    world.update_temporal_jointed_bodies(model)
                    graph = None
                    if capture:
                        with wp.ScopedCapture() as captured:
                            world.update_temporal_jointed_bodies(model)
                        graph = captured.graph
                    for disabled in (False, True, False):
                        enabled = np.ones(model.joint_count, dtype=bool)
                        if disabled:
                            enabled[order == 4] = False
                        model.joint_enabled.assign(enabled)
                        if graph is None:
                            world.update_temporal_jointed_bodies(model)
                        else:
                            wp.capture_launch(graph)
                        expected = np.ones(10, dtype=np.int32)
                        # PhoenX reserves body zero for the static world.
                        expected[0] = 0
                        if disabled:
                            expected[5:] = 0
                        np.testing.assert_array_equal(
                            world._temporal_jointed.numpy(),
                            expected,
                            err_msg=f"reverse={reverse}, disabled={disabled}, capture={capture}",
                        )

    def test_weld_chain_cpu(self):
        """Propagate ownership through either joint order and runtime weld toggles."""
        self._check_chain("cpu", capture=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_weld_chain_cuda_graph(self):
        """Replay captured ownership updates after disabling and re-enabling a weld."""
        self._check_chain(wp.get_cuda_device(), capture=True)


if __name__ == "__main__":
    unittest.main()
