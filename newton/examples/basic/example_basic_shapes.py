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
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples

# Solver Selection
# =================
# Choose which solver to use for rigid body contact simulation:
SOLVER_TYPE = "XPBD"  # Options: "XPBD", "MUJOCO_NEWTON", "MUJOCO_NATIVE", "FEATHERSTONE"

# Solver descriptions:
# - "XPBD": Newton's native XPBD solver (fast, stable, good for general use)
# - "MUJOCO_NEWTON": MuJoCo Warp solver using Newton contacts (best of both worlds)
# - "MUJOCO_NATIVE": MuJoCo Warp solver using MuJoCo contacts (MuJoCo's native contact handling)
# - "FEATHERSTONE": Featherstone reduced-coordinate solver (good for articulated systems)


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        print("Example Basic Shapes")

        builder = newton.ModelBuilder()

        # replace ground plane with a large static box whose top face lies at z=0
        # attach directly to world (body = -1) so it is truly static
        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -50.0), q=wp.quat_identity()),
            hx=50.0,
            hy=50.0,
            hz=50.0,
        )
        # Add a ground plane at z=0
        # builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # z height to drop shapes from
        drop_z = 2.0

        # Note: Free joints are added to all bodies for MuJoCo compatibility.
        # MuJoCo requires explicit joints for all free-floating bodies.
        # XPBD solver doesn't require joints but ignores them if present.

        # SPHERE
        body_sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_sphere(body_sphere, radius=0.5)
        builder.add_joint_free(body_sphere)  # Add free joint for MuJoCo

        # CAPSULE
        body_capsule = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)
        builder.add_joint_free(body_capsule)  # Add free joint for MuJoCo

        # CYLINDER (no collision support)
        body_cylinder = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -4.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)
        builder.add_joint_free(body_cylinder)  # Add free joint for MuJoCo

        # BOX
        body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)
        builder.add_joint_free(body_box)  # Add free joint for MuJoCo

        # CONE (no collision support)
        body_cone = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -6.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)
        builder.add_joint_free(body_cone)  # Add free joint for MuJoCo

        # Three stacked cubes (small initial gaps), positioned at y = 6.0
        cube_h = 0.4
        gap = 0.02
        y_stack = 6.0
        z1 = cube_h + gap
        z2 = z1 + 2.0 * cube_h + gap
        z3 = z2 + 2.0 * cube_h + gap

        # Build a pyramid of cubes
        pyramid_size = 10  # Number of cubes at the base
        cube_spacing = 2.1 * cube_h  # Space between cube centers

        for level in range(pyramid_size):
            num_cubes_in_row = pyramid_size - level
            row_width = (num_cubes_in_row - 1) * cube_spacing

            for i in range(num_cubes_in_row):
                x_pos = -row_width / 2 + i * cube_spacing
                z_pos = level * cube_spacing + cube_h

                body = builder.add_body(xform=wp.transform(p=wp.vec3(x_pos, y_stack, z_pos), q=wp.quat_identity()))
                builder.add_shape_box(body, hx=cube_h, hy=cube_h, hz=cube_h)
                builder.add_joint_free(body)  # Add free joint for MuJoCo

        # # MESH (bunny)
        # usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        # usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        # mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        # mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        # demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        # body_mesh = builder.add_body(
        #     xform=wp.transform(p=wp.vec3(0.0, 4.0, drop_z - 0.5), q=wp.quat(0.5, 0.5, 0.5, 0.5))
        # )
        # builder.add_shape_mesh(body_mesh, mesh=demo_mesh)
        # builder.add_joint_free(body_mesh)  # Add free joint for MuJoCo

        # finalize model
        self.model = builder.finalize()

        # Initialize solver based on the selected type
        if SOLVER_TYPE == "XPBD":
            print("Using XPBD solver")
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=2, rigid_contact_relaxation=0.8)
        elif SOLVER_TYPE == "MUJOCO_NEWTON":
            print("Using MuJoCo Warp solver with Newton contacts")
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,  # Use Newton contacts instead of MuJoCo contacts
                iterations=20,
                ls_iterations=10,
                integrator="euler",
                solver="cg",
            )
        elif SOLVER_TYPE == "MUJOCO_NATIVE":
            print("Using MuJoCo Warp solver with MuJoCo contacts")
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=True,  # Use MuJoCo's native contact handling
                iterations=20,
                ls_iterations=10,
                integrator="euler",
                solver="cg",
            )
        elif SOLVER_TYPE == "FEATHERSTONE":
            print("Using Featherstone reduced-coordinate solver")
            self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.05, friction_smoothing=1.0)
        else:
            raise ValueError(
                f"Unknown solver type: {SOLVER_TYPE}. Choose from: XPBD, MUJOCO_NEWTON, MUJOCO_NATIVE, FEATHERSTONE"
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        # Disable graph capture: run simulation directly each step
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Compute contacts - needed for Newton contact solvers and XPBD
            # MuJoCo with native contacts computes its own contacts internally
            if SOLVER_TYPE in ["XPBD", "MUJOCO_NEWTON", "FEATHERSTONE"]:
                self.contacts = self.model.collide(self.state_0)
            else:
                self.contacts = None  # MuJoCo native contacts don't need Newton contacts

            # Step solver - all solvers use the same interface
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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

    newton.examples.run(example)
