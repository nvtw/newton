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

###########################################################################
# Example Heightfield
#
# Shows how to use a heightfield terrain instead of a ground plane.
# Heightfields are efficient for large terrain areas because they exploit
# the regular grid structure for O(1) cell lookup during collision detection.
#
# Command: python -m newton.examples heightfield
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # Create heightfield terrain instead of ground plane
        size_x, size_y = 20.0, 20.0
        resolution = 50
        amplitude = 1.0
        frequency = 0.3

        x = np.linspace(0, size_x, resolution)
        y = np.linspace(0, size_y, resolution)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Generate rolling hills terrain
        heights = (
            amplitude * np.sin(X * frequency) * np.cos(Y * frequency)
            + amplitude * 0.5 * np.sin(X * frequency * 2.0 + 1.0)
            + amplitude * 0.3 * np.cos(Y * frequency * 1.5 + 0.5)
        ).astype(np.float32)

        # Add heightfield (mesh is already centered at origin by _create_heightfield_mesh)
        builder.add_shape_heightfield(
            heights=heights,
            extent_x=size_x,
            extent_y=size_y,
        )

        # z height to drop shapes from
        drop_z = 5.0

        # SPHERE
        body_sphere = builder.add_body(xform=wp.transform(p=(0.0, -6.0, drop_z), q=wp.quat_identity()), key="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # ELLIPSOID (flat disk shape: a=b > c for stability when resting)
        body_ellipsoid = builder.add_body(
            xform=wp.transform(p=(0.0, -4.0, drop_z), q=wp.quat_identity()), key="ellipsoid"
        )
        builder.add_shape_ellipsoid(body_ellipsoid, a=0.5, b=0.5, c=0.25)

        # CAPSULE
        body_capsule = builder.add_body(xform=wp.transform(p=(0.0, -2.0, drop_z), q=wp.quat_identity()), key="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # CYLINDER
        body_cylinder = builder.add_body(xform=wp.transform(p=(0.0, 0.0, drop_z), q=wp.quat_identity()), key="cylinder")
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # BOX
        body_box = builder.add_body(xform=wp.transform(p=(0.0, 2.0, drop_z), q=wp.quat_identity()), key="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # CONE
        body_cone = builder.add_body(xform=wp.transform(p=(0.0, 4.0, drop_z), q=wp.quat_identity()), key="cone")
        builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline from command-line args
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            rigid_contact_max_per_pair=100,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        # Test that objects didn't fall through the terrain
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "objects above terrain",
            lambda q, qd: q[2] > -2.0,  # z should be above minimum terrain height
            list(range(self.model.body_count)),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
