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
# Example Basic URDF
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
# Users can pick bodies by right-clicking and dragging with the mouse.
#
# Command: python -m newton.examples basic_urdf
#
###########################################################################


import warp as wp

import newton
import newton.examples

# CUDA Graph Capture
# ==================
# Enable CUDA graph capture for better performance (CUDA devices only)
USE_CUDA_GRAPH = True  # Set to False to disable CUDA graph capture


class Example:
    def __init__(self, viewer, num_envs):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        self.viewer = viewer

        quadruped = newton.ModelBuilder()

        # set default parameters for the quadruped
        quadruped.default_body_armature = 0.01
        quadruped.default_joint_cfg.armature = 0.01
        quadruped.default_joint_cfg.mode = newton.JointMode.TARGET_POSITION
        quadruped.default_joint_cfg.target_ke = 2000.0
        quadruped.default_joint_cfg.target_kd = 1.0
        quadruped.default_shape_cfg.ke = 1.0e4
        quadruped.default_shape_cfg.kd = 1.0e2
        quadruped.default_shape_cfg.kf = 1.0e2
        quadruped.default_shape_cfg.mu = 1.0

        # parse the URDF file
        quadruped.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform([0.0, 0.0, 0.7], wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
        )

        # set initial joint positions
        quadruped.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        quadruped.joint_target[-12:] = quadruped.joint_q[-12:]

        # use "scene" for the entire set of environments
        scene = newton.ModelBuilder()

        # use the builder.replicate() function to create N copies of the environment
        scene.replicate(quadruped, self.num_envs)
        scene.add_ground_plane()

        # finalize model
        self.model = scene.finalize()

        # Create unified collision pipeline with NxN broad phase
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            rigid_contact_max_per_pair=10,
            rigid_contact_margin=0.01,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
        )

        # Use MuJoCo Warp solver with Newton contacts
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,  # Use Newton's collision pipeline instead of MuJoCo's
            iterations=100,
            ls_iterations=50,
            njmax=300,
            contact_stiffness_time_const=self.sim_dt,  # Match timestep for stiff contacts
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize contacts using unified collision pipeline
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # put graph capture into it's own function
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

            # Compute contacts using unified collision pipeline
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
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "quadruped links are not moving too fast",
            lambda q, qd: max(abs(qd)) < 0.1,
        )

        bodies_per_env = self.model.body_count // self.num_envs
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "quadrupeds have reached the terminal height",
            lambda q, qd: wp.abs(q[2] - 0.46) < 0.01,
            # only select the root body of each environment
            indices=[i * bodies_per_env for i in range(self.num_envs)],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=4, help="Total number of simulated environments.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    example = Example(viewer, args.num_envs)

    newton.examples.run(example, args)
