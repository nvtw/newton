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
# Example Basic Shapes 2 - Sliding Cubes on Ramp
#
# This example demonstrates two cubes sliding down a tilted ramp with
# an end wall at the bottom. Uses CollisionPipeline2 with dynamic broad phase.
#
# Command: python -m newton.examples basic_shapes2
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.core import quat_between_axes
from newton._src.sim.collide2 import CollisionPipeline2, BroadPhaseMode

# Solver Selection
SOLVER_TYPE = "XPBD"  # Options: "XPBD", "FEATHERSTONE"

# CUDA Graph Capture
USE_CUDA_GRAPH = True  # Set to True to enable CUDA graph capture

# Broad Phase Mode
BROAD_PHASE_MODE = BroadPhaseMode.NXN

# Scene Configuration

RAMP_LENGTH = 10.0  # Length of the ramp
RAMP_THICKNESS = 0.5  # Thickness of the ramp
RAMP_ANGLE = np.radians(30.0)  # Tilt angle (30 degrees)
WALL_HEIGHT = 2.0  # Height of the end wall
CUBE_SIZE = 1.0*0.99  # Size of the sliding cubes
RAMP_WIDTH = CUBE_SIZE*2.01  # Width of the ramp and wall

class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # Start in paused mode
        viewer._paused = True

        print("Example Basic Shapes 2 - Sliding Cubes on Ramp")
        print("Starting in PAUSED mode - Press SPACE to start simulation")

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 2e4
        builder.default_shape_cfg.kd = 500.0
        builder.default_shape_cfg.kf = 0.5  # Add some friction

        # Calculate ramp geometry
        ramp_center_y = RAMP_LENGTH / 2 * np.cos(RAMP_ANGLE)
        ramp_center_z = RAMP_LENGTH / 2 * np.sin(RAMP_ANGLE)
        # Calculate ramp center position in world space
        ramp_center = wp.vec3(0.0, ramp_center_y, ramp_center_z)

        # Create tilted ramp using a plane (static)
        # Rotation: tilt the ramp so it slopes down from +y to -y
        ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(RAMP_ANGLE))

        builder.add_shape_plane(
            body=-1,  # Static body (attached to world)
            xform=wp.transform(p=ramp_center, q=ramp_quat),
            width=0,
            length=0,
        )

        # Compute coordinate system vectors for the tilted ramp (needed for side guides)
        # Forward vector points down the ramp (-y in world space)
        ramp_forward = wp.quat_rotate(ramp_quat, wp.vec3(0.0, -1.0, 0.0))
        # Up vector is perpendicular to ramp surface
        ramp_up = wp.quat_rotate(ramp_quat, wp.vec3(0.0, 0.0, 1.0))
        # Right vector is parallel to ramp surface, perpendicular to slope
        ramp_right = wp.quat_rotate(ramp_quat, wp.vec3(1.0, 0.0, 0.0))

        # For a plane, the center is already on the surface (no thickness)
        ramp_center_surface = ramp_center

        # Add side guide walls along the ramp
        guide_height = 0.3  # Height of the side walls
        guide_thickness = 0.1  # Thickness of the side walls
        
        # Left side guide wall
        left_guide_offset = (RAMP_WIDTH / 2 + guide_thickness / 2) * ramp_right
        left_guide_center = ramp_center + left_guide_offset + (guide_height / 2) * ramp_up
        builder.add_shape_box(
            body=-1,  # Static body (attached to world)
            xform=wp.transform(p=left_guide_center, q=ramp_quat),
            hx=guide_thickness / 2,
            hy=RAMP_LENGTH / 2,
            hz=guide_height / 2,
        )
        
        # Right side guide wall
        right_guide_offset = -(RAMP_WIDTH / 2 + guide_thickness / 2) * ramp_right
        right_guide_center = ramp_center + right_guide_offset + (guide_height / 2) * ramp_up
        builder.add_shape_box(
            body=-1,  # Static body (attached to world)
            xform=wp.transform(p=right_guide_center, q=ramp_quat),
            hx=guide_thickness / 2,
            hy=RAMP_LENGTH / 2,
            hz=guide_height / 2,
        )


        start_shift = 0.6*RAMP_LENGTH

        # Create end wall at the bottom of the ramp (static)
        # Position it at Y=0 (the lowest point where the ramp ends)
        tmp = ramp_center_surface + 0.5*CUBE_SIZE*(ramp_up+start_shift*ramp_forward)
        wall_y = tmp.y - CUBE_SIZE/2*1.4 - RAMP_THICKNESS / 2
        wall_z = tmp.z   # Center the wall vertically

        builder.add_shape_box(
            body=-1,  # Static body (attached to world)
            xform=wp.transform(p=wp.vec3(0.0, wall_y, wall_z), q=wp.quat_identity()),
            hx=RAMP_WIDTH / 2,
            hy=RAMP_THICKNESS / 2,
            hz=WALL_HEIGHT / 2,
        )

        # Rotate cubes to match ramp orientation (same rotation as ramp)
        cube_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(RAMP_ANGLE))


        offset_a = 0.5*CUBE_SIZE*(ramp_up+ramp_right+start_shift*ramp_forward)
        offset_b = 0.5*CUBE_SIZE*(ramp_up-ramp_right+start_shift*ramp_forward)      

       
        # Cube 1 (left side)
        body_cube1 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat)
        )
        builder.add_joint_free(body_cube1)
        builder.add_shape_box(
            body=body_cube1,
            hx=CUBE_SIZE / 2,
            hy=CUBE_SIZE / 2,
            hz=CUBE_SIZE / 2,
        )

        # Cube 2 (right side)
        body_cube2 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat)
        )
        builder.add_joint_free(body_cube2)
        builder.add_shape_box(
            body=body_cube2,
            hx=CUBE_SIZE / 2,
            hy=CUBE_SIZE / 2,
            hz=CUBE_SIZE / 2,
        )


        # MORE SHAPES HERE
        offset_a = 0.5*CUBE_SIZE*(ramp_up+ramp_right+(start_shift - 2.01)*ramp_forward)
        offset_b = 0.5*CUBE_SIZE*(ramp_up-ramp_right+(start_shift - 2.01)*ramp_forward)

        # Sphere 1 (left side)
        sphere_radius = CUBE_SIZE / 2
        body_sphere1 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat)
        )
        builder.add_joint_free(body_sphere1)
        builder.add_shape_sphere(
            body=body_sphere1,
            radius=sphere_radius,
        )

        # Sphere 2 (right side)
        body_sphere2 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat)
        )
        builder.add_joint_free(body_sphere2)
        builder.add_shape_sphere(
            body=body_sphere2,
            radius=sphere_radius,
        )

        # Capsule behind the spheres (centered on the ramp)
        capsule_radius = CUBE_SIZE / 2
        capsule_height = 2 * capsule_radius
        offset_capsule = 0.5*CUBE_SIZE*(ramp_up + (start_shift - 4.02)*ramp_forward)

        # Capsule axis is Z (using quat_between_axes), then apply ramp rotation
        capsule_local_quat = quat_between_axes(newton.Axis.Z, newton.Axis.X)
        capsule_quat = cube_quat * capsule_local_quat

        body_capsule = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_capsule, q=capsule_quat)
        )
        builder.add_joint_free(body_capsule)
        builder.add_shape_capsule(
            body=body_capsule,
            radius=capsule_radius,
            half_height=capsule_height / 2,
        )

        # Cylinder behind the capsule (centered on the ramp)
        cylinder_radius = CUBE_SIZE / 2
        cylinder_height = 4 * cylinder_radius
        offset_cylinder = 0.5*CUBE_SIZE*(ramp_up + (start_shift - 6.03)*ramp_forward)

        # Cylinder axis is Z (using quat_between_axes), then apply ramp rotation
        cylinder_local_quat = quat_between_axes(newton.Axis.Z, newton.Axis.X)
        cylinder_quat = cube_quat * cylinder_local_quat

        body_cylinder = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_cylinder, q=cylinder_quat)
        )
        builder.add_joint_free(body_cylinder)
        builder.add_shape_cylinder(
            body=body_cylinder,
            radius=cylinder_radius,
            half_height=cylinder_height / 2,
        )

        # Two more cubes after the cylinder
        offset_a = 0.5*CUBE_SIZE*(ramp_up+ramp_right+(start_shift - 8.04)*ramp_forward)
        offset_b = 0.5*CUBE_SIZE*(ramp_up-ramp_right+(start_shift - 8.04)*ramp_forward)

        # Cube 3 (left side)
        body_cube3 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat)
        )
        builder.add_joint_free(body_cube3)
        builder.add_shape_box(
            body=body_cube3,
            hx=CUBE_SIZE / 2,
            hy=CUBE_SIZE / 2,
            hz=CUBE_SIZE / 2,
        )

        # Cube 4 (right side)
        body_cube4 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat)
        )
        builder.add_joint_free(body_cube4)
        builder.add_shape_box(
            body=body_cube4,
            hx=CUBE_SIZE / 2,
            hy=CUBE_SIZE / 2,
            hz=CUBE_SIZE / 2,
        )

        # Two cones after the cubes (z-axis aligned with ramp_up)
        cone_radius = CUBE_SIZE / 2
        cone_height = 2 * cone_radius
        offset_a = 0.5*CUBE_SIZE*(ramp_up+ramp_right+(start_shift - 10.05)*ramp_forward)
        offset_b = 0.5*CUBE_SIZE*(ramp_up-ramp_right+(start_shift - 10.05)*ramp_forward)

        # Cone z-axis should align with ramp_up (which is the local Z of the ramp)
        # The cone's default axis is Z, and we want it to point along ramp_up
        # Since ramp_up is already the rotated Z axis of the ramp, we use cube_quat
        cone_quat = cube_quat

        # Cone 1 (left side)
        body_cone1 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_a, q=cone_quat)
        )
        builder.add_joint_free(body_cone1)
        builder.add_shape_cone(
            body=body_cone1,
            radius=cone_radius,
            half_height=cone_height / 2,
        )

        # Cone 2 (right side)
        body_cone2 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_b, q=cone_quat)
        )
        builder.add_joint_free(body_cone2)
        builder.add_shape_cone(
            body=body_cone2,
            radius=cone_radius,
            half_height=cone_height / 2,
        )

        # Two more cubes after the cones
        offset_a = 0.5*CUBE_SIZE*(ramp_up+ramp_right+(start_shift - 12.06)*ramp_forward)
        offset_b = 0.5*CUBE_SIZE*(ramp_up-ramp_right+(start_shift - 12.06)*ramp_forward)

        # Cube 5 (left side)
        body_cube5 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_a, q=cube_quat)
        )
        builder.add_joint_free(body_cube5)
        builder.add_shape_box(
            body=body_cube5,
            hx=CUBE_SIZE / 2,
            hy=CUBE_SIZE / 2,
            hz=CUBE_SIZE / 2,
        )

        # Cube 6 (right side)
        body_cube6 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_b, q=cube_quat)
        )
        builder.add_joint_free(body_cube6)
        builder.add_shape_box(
            body=body_cube6,
            hx=CUBE_SIZE / 2,
            hy=CUBE_SIZE / 2,
            hz=CUBE_SIZE / 2,
        )

        # Two cubes using convex hull representation (8 corner points)
        # Create a cube mesh with 8 vertices centered at origin
        cube_half = CUBE_SIZE / 2
        cube_vertices = np.array(
            [
                # Bottom face (z = -cube_half)
                [-cube_half, -cube_half, -cube_half],
                [cube_half, -cube_half, -cube_half],
                [cube_half, cube_half, -cube_half],
                [-cube_half, cube_half, -cube_half],
                # Top face (z = cube_half)
                [-cube_half, -cube_half, cube_half],
                [cube_half, -cube_half, cube_half],
                [cube_half, cube_half, cube_half],
                [-cube_half, cube_half, cube_half],
            ],
            dtype=np.float32,
        )

        # Cube faces (12 triangles, 2 per face)
        # Counter-clockwise winding when viewed from outside
        cube_indices = np.array(
            [
                # Bottom face (z = -cube_half, looking from below)
                0, 2, 1,
                0, 3, 2,
                # Top face (z = cube_half, looking from above)
                4, 5, 6,
                4, 6, 7,
                # Front face (y = -cube_half)
                0, 1, 5,
                0, 5, 4,
                # Right face (x = cube_half)
                1, 2, 6,
                1, 6, 5,
                # Back face (y = cube_half)
                2, 3, 7,
                2, 7, 6,
                # Left face (x = -cube_half)
                3, 0, 4,
                3, 4, 7,
            ],
            dtype=np.int32,
        )

        cube_mesh = newton.Mesh(cube_vertices, cube_indices)
        
        # Position cubes using body transform (same as normal cubes)
        offset_a = 0.5*CUBE_SIZE*(ramp_up+ramp_right+(start_shift - 14.07)*ramp_forward)
        offset_b = 0.5*CUBE_SIZE*(ramp_up-ramp_right+(start_shift - 14.07)*ramp_forward)

        # Cube z-axis should align with ramp_up (same as other shapes)
        convex_cube_quat = cube_quat

        # Convex Hull Cube 1 (left side)
        body_convex_cube1 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_a, q=convex_cube_quat)
        )
        builder.add_joint_free(body_convex_cube1)
        builder.add_shape_convex_hull(
            body=body_convex_cube1,
            mesh=cube_mesh,
            scale=(1.0, 1.0, 1.0),
        )

        # Convex Hull Cube 2 (right side)
        body_convex_cube2 = builder.add_body(
            xform=wp.transform(p=ramp_center_surface + offset_b, q=convex_cube_quat)
        )
        builder.add_joint_free(body_convex_cube2)
        builder.add_shape_convex_hull(
            body=body_convex_cube2,
            mesh=cube_mesh,
            scale=(1.0, 1.0, 1.0),
        )

        # Add ground plane (optional, for safety)
        builder.add_ground_plane()

        # Finalize model without pre-computed shape pairs (for CollisionPipeline2)
        self.model = builder.finalize(build_shape_contact_pairs=False)

        # Create CollisionPipeline2 explicitly with the selected broad phase mode
        print(f"Using CollisionPipeline2 with broad phase mode: {BROAD_PHASE_MODE.name}")
        self.collision_pipeline = CollisionPipeline2.from_model(
            self.model,
            rigid_contact_max_per_pair=10,
            rigid_contact_margin=0.01,
            broad_phase_mode=BROAD_PHASE_MODE,
        )

        # Initialize solver
        if SOLVER_TYPE == "XPBD":
            print("Using XPBD solver")
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=2)
        elif SOLVER_TYPE == "FEATHERSTONE":
            print("Using Featherstone solver")
            self.solver = newton.solvers.SolverFeatherstone(self.model)
        else:
            raise ValueError(f"Unknown solver type: {SOLVER_TYPE}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # Evaluate forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        if USE_CUDA_GRAPH and wp.get_device().is_cuda:
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

            # Compute contacts using CollisionPipeline2
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            # Step solver
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not self.viewer.is_paused():
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

            self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)