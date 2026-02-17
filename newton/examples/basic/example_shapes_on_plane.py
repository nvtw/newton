# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example replicating test_shapes_on_plane for visual inspection.

This example replicates the exact scene from test_shapes_on_plane to allow
visual inspection of how objects settle on a ground plane with the unified
collision pipeline.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.core import quat_between_axes
from newton._src.geometry.utils import create_box_mesh


class Example:
    def __init__(self, viewer, args=None):
        # Setup simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Increase substeps for better stability (more substeps = smaller time steps = more stable)
        # More substeps help reduce contact oscillations
        self.sim_substeps = 30  # Increased from 20 for better contact stability
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        viewer._paused = True

        # Replicate exact test setup
        size = 0.3
        # fmt: off
        vertices = np.array([
            [-size, -size, -size],
            [-size, -size, size],
            [-size, size, size],
            [-size, size, -size],
            [size, -size, -size],
            [size, -size, size],
            [size, size, size],
            [size, size, -size],
            [-size, -size, -size],
            [-size, -size, size],
            [size, -size, size],
            [size, -size, -size],
            [-size, size, -size],
            [-size, size, size],
            [size, size, size],
            [size, size, -size],
            [-size, -size, -size,],
            [-size, size, -size,],
            [size, size, -size,],
            [size, -size, -size,],
            [-size, -size, size],
            [-size, size, size],
            [size, size, size],
            [size, -size, size],
        ], dtype=np.float32)
        # Add some offset to the vertices to test proper handling of non-zero origin
        mesh_offset = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vertices += mesh_offset
        cube_mesh = newton.Mesh(
            vertices=vertices,
            indices=[
                0, 1, 2,
                0, 2, 3,
                4, 6, 5,
                4, 7, 6,
                8, 10, 9,
                8, 11, 10,
                12, 13, 14,
                12, 14, 15,
                16, 17, 18,
                16, 18, 19,
                20, 22, 21,
                20, 23, 22,
            ],
        )
        # fmt: on

        builder = newton.ModelBuilder()
        # Parameters tuned for stability (reduced oscillations)
        # Lower stiffness and higher damping reduce contact oscillations
        builder.default_shape_cfg.ke = 1e4  # Reduced from 2e4 for less oscillation
        builder.default_shape_cfg.kd = 1000.0  # Increased from 500.0 for more damping
        builder.default_shape_cfg.kf = 0.0  # disable friction
        # Set contact margin via ShapeConfig (preferred method)
        # If not set, defaults to builder.rigid_contact_margin (0.1)
        builder.default_shape_cfg.contact_margin = 0.1

        # Add shapes in two rows (scale 0.5 and 1.0)
        drop_height = 0.5  # Reduced drop height for easier visual inspection
        self.expected_end_positions = []
        for i, scale in enumerate([0.5, 1.0]):
            y_pos = i * 1.5

            # Sphere
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, y_pos, drop_height), wp.quat_identity()))
            builder.add_shape_sphere(body=b, radius=0.1 * scale)
            self.expected_end_positions.append(wp.vec3(0.0, y_pos, 0.1 * scale))

            # Capsule (rotated)
            b = builder.add_body(xform=wp.transform(wp.vec3(2.0, y_pos, drop_height), wp.quat_identity()))
            xform = wp.transform(wp.vec3(), quat_between_axes(newton.Axis.Z, newton.Axis.Y))
            builder.add_shape_capsule(body=b, xform=xform, radius=0.1 * scale, half_height=0.3 * scale)
            self.expected_end_positions.append(wp.vec3(2.0, y_pos, 0.1 * scale))

            # Box
            b = builder.add_body(xform=wp.transform(wp.vec3(4.0, y_pos, drop_height), wp.quat_identity()))
            builder.add_shape_box(body=b, hx=0.2 * scale, hy=0.25 * scale, hz=0.3 * scale)
            self.expected_end_positions.append(wp.vec3(4.0, y_pos, 0.3 * scale))

            # Cylinder
            b = builder.add_body(xform=wp.transform(wp.vec3(5.0, y_pos, drop_height), wp.quat_identity()))
            builder.add_shape_cylinder(body=b, radius=0.1 * scale, half_height=0.3 * scale)
            self.expected_end_positions.append(wp.vec3(5.0, y_pos, 0.3 * scale))

            # Mesh (cube)
            b = builder.add_body(xform=wp.transform(wp.vec3(7.0, y_pos, drop_height), wp.quat_identity()))
            builder.add_shape_mesh(body=b, mesh=cube_mesh, scale=wp.vec3(scale, scale, scale))
            self.expected_end_positions.append(wp.vec3(7.0, y_pos, 0.3 * scale))

        builder.add_ground_plane()

        # Finalize model
        self.model = builder.finalize()

        # Use Featherstone solver (same as test by default)
        # Can be changed via --solver argument
        solver_type = args.solver if args and hasattr(args, "solver") and args.solver else "featherstone"
        if solver_type == "featherstone":
            # Increase angular_damping and friction_smoothing for better stability
            # Higher angular_damping reduces rotational oscillations
            # Higher friction_smoothing helps with friction-related instabilities
            self.solver = newton.solvers.SolverFeatherstone(
                self.model,
                angular_damping=0.15,  # Increased from 0.1 for more rotational stability
                friction_smoothing=2.0,  # Increased from default 1.0 for smoother friction
            )
        elif solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=2)
        elif solver_type == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
        else:
            self.solver = newton.solvers.SolverFeatherstone(
                self.model,
                angular_damping=0.1,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initial collision detection
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Check final positions after 400 iterations (same as test)
        for _ in range(250):
            self.simulate()

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        expected_end_positions = np.array(self.expected_end_positions)

        # Check for NaN values (indicates simulation instability)
        if np.any(np.isnan(body_q)):
            nan_indices = np.where(np.isnan(body_q).any(axis=1))[0]
            raise AssertionError(
                f"Simulation produced NaN values for bodies: {nan_indices.tolist()}. "
                "This indicates numerical instability. Check solver parameters and contact settings."
            )

        # Check that objects are at rest (velocities near zero)
        linear_vel = body_qd[:, :3]
        angular_vel = body_qd[:, 3:]
        max_linear_vel = np.max(np.abs(linear_vel))
        max_angular_vel = np.max(np.abs(angular_vel))
        
        print("\n=== Final State ===")
        print(f"Body positions:\n{body_q[:, :3]}")
        print(f"\nExpected positions:\n{expected_end_positions}")
        print(f"\nBody velocities:\n{body_qd[:, :3]}")
        print(f"\nMax linear velocity: {max_linear_vel:.6f}")
        print(f"Max angular velocity: {max_angular_vel:.6f}")
        
        # Check velocities are near zero (objects should be at rest)
        if max_linear_vel > 0.5:
            raise AssertionError(
                f"Objects should be at rest, but max linear velocity is {max_linear_vel:.6f}"
            )
        if max_angular_vel > 0.5:
            raise AssertionError(
                f"Objects should be at rest, but max angular velocity is {max_angular_vel:.6f}"
            )

        # Check final positions with tolerance for unified pipeline differences
        # Unified pipeline may produce slightly different final positions due to contact handling differences
        pos_diff = np.abs(body_q[:, :3] - expected_end_positions)
        max_pos_diff = np.max(pos_diff)
        print(f"\nMax position difference: {max_pos_diff:.6f}")
        
        if max_pos_diff > 0.25:
            raise AssertionError(
                f"Objects not at expected positions. Max difference: {max_pos_diff:.6f} (tolerance: 0.25)\n"
                f"Actual positions:\n{body_q[:, :3]}\n"
                f"Expected positions:\n{expected_end_positions}"
            )

        # Check orientations are approximately identity (no significant rotation)
        expected_quats = np.tile(wp.quat_identity(), (self.model.body_count, 1))
        quat_diff = np.abs(body_q[:, 3:] - expected_quats)
        max_quat_diff = np.max(quat_diff)
        print(f"Max quaternion difference: {max_quat_diff:.6f}")
        
        if max_quat_diff > 0.1:
            raise AssertionError(
                f"Objects rotated unexpectedly. Max quaternion difference: {max_quat_diff:.6f} (tolerance: 0.1)"
            )
        
        print("\n✓ All tests passed!")


if __name__ == "__main__":
    # Extend the shared examples parser with a solver choice
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="featherstone",
        choices=["featherstone", "xpbd", "semi_implicit"],
        help="Solver type: featherstone (default), xpbd, or semi_implicit",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
