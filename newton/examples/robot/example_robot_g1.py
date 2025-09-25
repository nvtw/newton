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
# Example Robot G1
#
# Shows how to set up a simulation of a G1 robot articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_g1 --num-envs 8192 --num-frames 200
#
###########################################################################

import os

import warp as wp

import newton
import newton.examples
import newton.utils
from newton._src.utils.recorder import RecorderModelAndState


class Example:
    def __init__(self, viewer, num_envs=8192):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        self.viewer = viewer

        g1 = newton.ModelBuilder()
        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        g1.default_shape_cfg.ke = 5.0e4
        g1.default_shape_cfg.kd = 5.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")

        g1.add_usd(
            str(asset_path / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        for i in range(6, g1.joint_dof_count):
            g1.joint_target_ke[i] = 1000.0
            g1.joint_target_kd[i] = 5.0

        # approximate meshes for faster collision detection
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(g1, self.num_envs, spacing=(3, 3, 0))

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="euler",
            njmax=300,
            ncon_per_env=150,
            cone="elliptic",
            impratio=100,
            iterations=100,
            ls_iterations=50,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # Initialize recorder with ring buffer for 50 states
        self.recorder = RecorderModelAndState(max_history_size=5)
        self.recorder.record_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        # Record state for debugging
        self.recorder.record(self.state_0)

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        pass

    def save_recording(self):
        """Save the recording to a binary file in the specified directory."""
        output_dir = "C:/tmp/recording_tests"

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        filename = "robot_g1_recording.bin"
        filepath = os.path.join(output_dir, filename)

        # Save recording as binary file
        print(f"Saving recording to: {filepath}")
        self.recorder.save_to_file(filepath)
        print(f"Recording saved to: {filepath}")
        print(f"Recorded {len(self.recorder.history)} states (ring buffer capacity: 50)")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-envs", type=int, default=8192, help="Total number of simulated environments (default: 8192)."
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_envs)

    # Run simulation (default 100 frames from base parser)
    print(f"Running simulation with {example.num_envs} robots for {args.num_frames} frames...")

    newton.examples.run(example)

    # Save recording before exit
    example.save_recording()
