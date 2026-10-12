# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify adaptive contact lanes with mixed D6 joints and overflow."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx import simulation as phoenx_simulation
from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_CONTACT_COUNT
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING


def make_scene():
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    box_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, mu=0.7)
    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, mu=0.7)
    # Each box has several ground/box contact points; shared endpoints create
    # enough graph pressure to exercise the overflow color at a cap of one.
    for row in range(4):
        for col in range(4):
            body = builder.add_body(xform=wp.transform(wp.vec3(col * 0.38, row * 0.38, 0.24), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.24, hy=0.24, hz=0.24, cfg=box_cfg)
    spheres = []
    for i in range(7):
        body = builder.add_body(xform=wp.transform(wp.vec3(4.0 + i * 0.5, 0.0, 0.19), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.2, cfg=sphere_cfg)
        spheres.append(body)
    # A colored D6 block is independent of most ground contacts and can run
    # in a mixed joint/contact color.
    builder.add_joint_d6(
        parent=spheres[0],
        child=spheres[1],
        linear_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X)],
    )
    builder.color()
    model = builder.finalize(device=wp.get_preferred_device())
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching=PHOENX_CONTACT_MATCHING,
        broad_phase="sap",
        shape_pairs_max=2048,
        rigid_contact_max=4096,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        substeps=1,
        solver_iterations=2,
        velocity_iterations=1,
        step_layout="single_world",
        mass_splitting=True,
        mass_splitting_overflow_only=True,
        max_colored_partitions=1,
        mass_splitting_batch_size=1,
        mass_splitting_unrolled=True,
        partitioner_algorithm="endpoint_owner",
        colored_contact_headers=True,
        colored_contact_rows=True,
        joint_mode="maximal_pgs",
        sleeping_velocity_threshold=0.0,
    )
    state = model.state()
    state.body_q.assign(model.body_q)
    control = model.control()
    pipeline.collide(state, contacts)
    solver.step(state, state, control, contacts, 1.0 / 120.0)
    return solver.world


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestSingletonFullKernel(unittest.TestCase):
    def test_mixed_color_overflow_and_tail_bitwise_parity(self):
        world = make_scene()
        self.assertTrue(world._colored_singleton_contact_lanes, "eligible scene should select singleton lanes")
        partitioner = world._partitioner
        starts = partitioner.color_starts.numpy()
        family = partitioner.color_family_starts.numpy()
        singleton_end = partitioner.color_singleton_end.numpy()
        colors = int(partitioner.num_colors.numpy()[0])
        self.assertGreater(colors, 1, "scene must exercise overflow")
        self.assertGreater(int(starts[2] - starts[1]), 0, "overflow must contain work")

        packed = world._contact_cols_packed.data.numpy().view(np.int32)
        count1 = count_many = 0
        mixed = partial_tail = False
        for color in range(world.max_colored_partitions):
            begin = int(family[color * 6 + 1])
            end = int(family[color * 6 + 2])
            one_end = int(singleton_end[color])
            counts = packed[int(_OFF_CONTACT_COUNT), begin:end]
            count1 += int(np.count_nonzero(counts == 1))
            count_many += int(np.count_nonzero(counts > 1))
            self.assertEqual(one_end - begin, int(np.count_nonzero(counts <= 1)))
            mixed |= int(family[color * 6]) < begin < end
            partial_tail |= (end - begin) % 32 != 0
        self.assertGreaterEqual(count1, 3, "one-point lane path needs meaningful coverage")
        self.assertGreater(count_many, 0, "four-lane path must also run")
        self.assertTrue(mixed, "test requires a color with D6 and contacts")
        self.assertTrue(partial_tail, "test requires a partial contact warp")

        cc = world._contact_container_solve
        arrays = (
            world.bodies.position,
            world.bodies.orientation,
            world.bodies.velocity,
            world.bodies.angular_velocity,
            world._copy_state.position,
            world._copy_state.orientation,
            world._copy_state.velocity,
            world._copy_state.angular_velocity,
            cc.impulses,
            world.constraints.d6.lower_impulse,
            world.constraints.d6.upper_impulse,
            world.constraints.d6.friction_impulse,
            world.constraints.bilateral.accumulated,
            world.constraints.multipliers,
        )
        initial = tuple(wp.clone(array) for array in arrays)
        original = phoenx_simulation.get_singleworld_kernel

        def build_kernel(adaptive):
            def get_kernel(*args, **kwargs):
                if kwargs.get("phase") == "iterate" and not kwargs.get("fused", False):
                    kwargs["adaptive_singleton_contacts"] = adaptive
                return original(*args, **kwargs)

            with patch.object(phoenx_simulation, "get_singleworld_kernel", get_kernel):
                return world._singleworld_kernels()[2]

        def capture(kernel):
            wp.capture_begin(device=world.device)
            world._launch_singleworld_head(
                kernel,
                wp.float32(1.0 / world.substep_dt),
                wp.int32(-1),
                cc,
                0,
            )
            return wp.capture_end(device=world.device)

        reference = capture(build_kernel(False))
        adaptive = capture(build_kernel(True))

        def solve(graph):
            for target, value in zip(arrays, initial, strict=True):
                wp.copy(target, value)
            partitioner.sweep_direction.assign([0])
            wp.synchronize_device(world.device)
            wp.capture_launch(graph)
            return tuple(array.numpy().copy() for array in arrays)

        baseline = solve(reference)
        repeat = solve(reference)
        accelerated = solve(adaptive)
        repeat_accelerated = solve(adaptive)
        for lhs, rhs in zip(baseline, repeat, strict=True):
            np.testing.assert_array_equal(lhs, rhs)
        for lhs, rhs in zip(baseline, accelerated, strict=True):
            np.testing.assert_array_equal(lhs, rhs)
        for lhs, rhs in zip(accelerated, repeat_accelerated, strict=True):
            np.testing.assert_array_equal(lhs, rhs)


if __name__ == "__main__":
    unittest.main()
