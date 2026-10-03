# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check ordered four-lane contact solving against the scalar color path."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx import simulation as phoenx_simulation
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as contact
from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_CONTACT_COUNT
from newton._src.solvers.phoenx.examples.example_kapla_hoberman import Example


class _NullViewer:
    supports_simulation_render_overlap = False

    def set_model(self, model):
        pass

    def set_camera(self, **kwargs):
        pass

    def apply_forces(self, state):
        pass


class TestCooperativeContactDevice(unittest.TestCase):
    def test_cpu_keeps_scalar_contacts(self):
        """Select subgroup shuffles only for CUDA single-world kernels."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        world = newton.solvers.SolverPhoenX(model, step_layout="single_world").world
        requested = []
        original = phoenx_simulation.get_singleworld_kernel

        def get_kernel(*args, **kwargs):
            requested.append(kwargs.get("cooperative_contacts", False))
            return original(*args, **kwargs)

        with patch.object(phoenx_simulation, "get_singleworld_kernel", get_kernel):
            world._singleworld_kernels()
        self.assertTrue(requested)
        self.assertFalse(any(requested))


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
        original = phoenx_simulation.get_singleworld_kernel

        def build_kernel(enabled):
            def get_kernel(*args, **kwargs):
                if "cooperative_contacts" in kwargs:
                    kwargs["cooperative_contacts"] = enabled
                return original(*args, **kwargs)

            with patch.object(phoenx_simulation, "get_singleworld_kernel", get_kernel):
                return world._singleworld_kernels()[2]

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


if __name__ == "__main__":
    unittest.main()
