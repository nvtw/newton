# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check ordered four-lane contact solving against the scalar color path."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx import simulation_kernels as kernels
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as contact
from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_CONTACT_COUNT
from newton._src.solvers.phoenx.constraints.constraint_container import constraint_container_zeros
from newton._src.solvers.phoenx.examples.example_kapla_hoberman import Example
from newton._src.solvers.phoenx.simulation import PhoenXWorld


class _NullViewer:
    supports_simulation_render_overlap = False

    def set_model(self, model):
        pass

    def set_camera(self, **kwargs):
        pass

    def apply_forces(self, state):
        pass


class TestCooperativeContactDevice(unittest.TestCase):
    def test_opt_in_rejects_cpu(self):
        bodies = body_container_zeros(1, device="cpu")
        constraints = constraint_container_zeros(0, 1, device="cpu")
        with patch.dict(os.environ, {"PHOENX_EXPERIMENTAL_COOPERATIVE_CONTACTS": "1"}):
            with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
                PhoenXWorld(bodies, constraints, device="cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required for subgroup shuffles")
class TestCooperativeContactParity(unittest.TestCase):
    def test_regular_color_preserves_ordered_contact_and_joint_state(self):
        self.assertTrue(hasattr(contact, "contact_iterate_lean_no_sleep_no_soft_pd_cooperative"))
        example = Example(_NullViewer(), None)
        for _ in range(2):
            example.step()
        world = example.solver.world
        color_starts = world._partitioner.color_starts.numpy()
        family_starts = world._partitioner.color_family_starts.numpy()
        begin = int(family_starts[1])
        end = int(color_starts[1])
        self.assertGreater(end - begin, 0)
        packed = world._contact_cols_packed.data.numpy().view(np.int32)
        self.assertTrue(np.any(packed[int(_OFF_CONTACT_COUNT), begin:end] > 1))

        cc = world._contact_container_solve
        arrays = (
            world.bodies.velocity,
            world.bodies.angular_velocity,
            world._copy_state.velocity,
            world._copy_state.angular_velocity,
            cc.impulses,
            world.constraints.d6.lower_impulse,
            world.constraints.d6.upper_impulse,
            world.constraints.d6.friction_impulse,
            world.constraints.bilateral.accumulated,
            world.constraints.multipliers,
        )
        snapshots = tuple(wp.clone(array) for array in arrays)
        previous_flag = kernels._EXPERIMENTAL_COOPERATIVE_CONTACTS

        def build_kernel(enabled):
            kernels._EXPERIMENTAL_COOPERATIVE_CONTACTS = enabled
            kernels._make_singleworld_rigid_direct_color_func.cache_clear()
            kernels._make_singleworld_persistent_kernel.cache_clear()
            return world._singleworld_kernels()[2]

        try:
            scalar = build_kernel(False)
            cooperative = build_kernel(True)
            self.assertIsNot(scalar, cooperative)

            def capture(kernel):
                wp.capture_begin(device=example.device)
                world._launch_singleworld_head(kernel, wp.float32(1.0 / world.substep_dt), wp.int32(-1), cc, 0)
                return wp.capture_end(device=example.device)

            scalar_graph = capture(scalar)
            cooperative_graph = capture(cooperative)

            def solve(graph):
                for target, snapshot in zip(arrays, snapshots, strict=True):
                    wp.copy(target, snapshot)
                world._partitioner.sweep_direction.assign([0])
                wp.synchronize_device(example.device)
                wp.capture_launch(graph)
                return tuple(array.numpy().copy() for array in arrays)

            expected = solve(scalar_graph)
            repeated = solve(scalar_graph)
            for reference, repeat in zip(expected, repeated, strict=True):
                np.testing.assert_array_equal(repeat, reference)
            actual = solve(cooperative_graph)
            for reference, candidate in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(candidate, reference)
        finally:
            kernels._EXPERIMENTAL_COOPERATIVE_CONTACTS = previous_flag
            kernels._make_singleworld_rigid_direct_color_func.cache_clear()
            kernels._make_singleworld_persistent_kernel.cache_clear()


if __name__ == "__main__":
    unittest.main()
