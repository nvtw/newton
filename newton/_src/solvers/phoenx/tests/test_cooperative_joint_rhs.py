# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise grouped CUDA joint RHS policy and scalar equivalence."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx import solver_phoenx as world_module
from newton._src.solvers.phoenx.dispatch.color_groups import (
    get_sweep_block_dim,
    get_sweep_kernel,
    use_cooperative_joint_rhs,
)
from newton._src.solvers.phoenx.tests import test_inactive_joint_prepare as transitions
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@unittest.skipUnless(_cuda_with_graph_capture(), "Cooperative joint RHS requires CUDA")
class TestCooperativeJointRHS(unittest.TestCase):
    def test_grouped_block_joint_launch_uses_cooperative_lanes(self):
        """Select eight lanes per joint while leaving preparation scalar."""
        model = make_model(100.0)
        solver = make_solver(model, mass_splitting=True, mass_splitting_color_group_size=4)
        state = model.state()
        original = wp.launch
        widths = []

        def launch(kernel, *args, **kwargs):
            if kernel.key == "get_sweep_kernel__locals__sweep":
                widths.append(kwargs["block_dim"])
            return original(kernel, *args, **kwargs)

        with patch.object(wp, "launch", launch):
            solver.step(state, state, model.control(), None, 0.01)
        self.assertIn(32, widths)
        self.assertIn(256, widths)

    def test_row_counts_disabled_rows_and_partial_tiles(self):
        """Match scalar outputs for zero to six rows and a partial final warp."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for pair in range(39):
            parent = builder.add_link(
                xform=wp.transform(wp.vec3(3.0 * pair, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
            )
            child = builder.add_link(
                xform=wp.transform(wp.vec3(3.0 * pair + 1.0, 0.0, 0.0), wp.quat_identity()),
                mass=2.0,
                inertia=wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
            )
            root = builder.add_joint_free(parent)
            fixed = builder.add_joint_fixed(
                parent,
                child,
                parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            )
            builder.add_articulation([root, fixed])
        model = builder.finalize(device="cuda:0")
        solver = make_solver(model, mass_splitting=True, mass_splitting_color_group_size=4)
        state = model.state()
        solver.step(state, state, model.control(), None, 0.001)
        world = solver.world
        data = world.constraints.bilateral
        counts = data.row_count.numpy()
        owned = np.flatnonzero(counts)
        self.assertEqual(len(owned), 39)
        self.assertTrue(np.all(counts[owned] == 6))
        counts[owned] = np.arange(len(owned)) % 7
        data.row_count.assign(counts)
        valid = data.valid.numpy()
        valid[owned[8]] = 0
        data.valid.assign(valid)
        enabled = world._joint_pgs_enabled.numpy()
        enabled[owned[12]] = 0
        enabled[owned[13]] = 2
        world._joint_pgs_enabled.assign(enabled)
        arrays = [
            world.bodies.velocity,
            world.bodies.angular_velocity,
            world._copy_state.velocity,
            world._copy_state.angular_velocity,
            world.constraints.multipliers,
            data.accumulated,
        ]
        rng = np.random.default_rng(321)
        for array in arrays[:4]:
            array.assign(rng.normal(0.0, 0.1, array.numpy().shape).astype(np.float32))
        saved = [wp.clone(array) for array in arrays]
        topology = world._color_group_data
        inputs = [
            world.constraints,
            world._contact_cols,
            world.bodies,
            world._particles_or_sentinel(),
            world._contact_container,
            world._active_contact_views(),
            world._copy_state,
            world.num_joints,
            world._joint_pgs_enabled,
            world.num_bodies,
            wp.float32(1000.0),
            topology["ids"],
            topology["starts"],
            topology["num_colors"],
            world.mass_splitting_color_group_size,
        ]
        for globally_enabled in (1, 0):
            data.enabled = globally_enabled
            world.constraints.bilateral = data
            for phase in ("iterate", "relax"):
                expected = None
                for cooperative in (False, True):
                    for destination, source in zip(arrays, saved, strict=True):
                        wp.copy(destination, source)
                    block_dim = get_sweep_block_dim(cooperative)
                    wp.launch(
                        get_sweep_kernel(phase, False, cooperative_joints=cooperative),
                        (64, block_dim),
                        inputs,
                        block_dim=block_dim,
                        device=world.device,
                    )
                    actual = [array.numpy().copy() for array in arrays]
                    if expected is None:
                        expected = actual
                    else:
                        for a, b in zip(expected, actual, strict=True):
                            self.assertEqual(a.tobytes(), b.tobytes())

    def test_live_geometry_and_property_transitions(self):
        """Keep finite limits, stale releases, drives and row rebinding bit-exact."""
        original = transitions.make_solver

        def grouped(model, **kwargs):
            return original(model, mass_splitting_color_group_size=2, **kwargs)

        with patch.object(transitions, "make_solver", grouped):
            for prismatic in (False, True):
                with patch.object(world_module, "use_cooperative_joint_rhs", return_value=False):
                    expected, _ = transitions.run(True, prismatic, True)
                actual, _ = transitions.run(True, prismatic, True)
                for before, after in zip(expected, actual, strict=True):
                    for a, b in zip(before, after, strict=True):
                        self.assertEqual(a.tobytes(), b.tobytes())


class TestCooperativeJointPolicy(unittest.TestCase):
    def test_scalar_fallback_selection(self):
        """Keep CPU, contact-only scenes and preparation on the scalar mapping."""
        for phase in ("prepare", "cached_prepare", "iterate", "relax"):
            for is_cuda in (False, True):
                for has_blocks in (False, True):
                    expected = is_cuda and has_blocks and phase in ("iterate", "relax")
                    actual = use_cooperative_joint_rhs(
                        phase=phase,
                        is_cuda=is_cuda,
                        has_bilateral_blocks=has_blocks,
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(get_sweep_block_dim(actual), 256 if expected else 32)


if __name__ == "__main__":
    unittest.main()
