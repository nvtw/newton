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
# Example Suction Cup
#
# Demonstrates controllable suction-cup adhesion between rigid bodies.
# A resting box sits on the ground. Right-click-drag the second box
# near it; when they touch and the adhesion slider is above zero the
# boxes stick together and the resting box can be lifted.
#
# Command: python -m newton.examples suction_cup
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.adhesion_strength = 1.0

        box_size = 0.15
        adhesion_gain = 500.0

        default_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0)
        suction_cfg = newton.ModelBuilder.ShapeConfig(
            adhesion_gain=adhesion_gain,
            ka=0.05,
            mu=1.0,
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=default_cfg)

        # Box A – resting on the ground (no adhesion)
        body_a = builder.add_body(
            xform=wp.transform((0.0, 0.0, box_size), wp.quat_identity()),
            label="box_resting",
        )
        builder.add_shape_box(body_a, hx=box_size, hy=box_size, hz=box_size, cfg=default_cfg)

        # Box B – the suction-cup "tool" the user drags via right-click picking
        body_b = builder.add_body(
            xform=wp.transform((0.0, 0.5, box_size), wp.quat_identity()),
            label="box_tool",
        )
        builder.add_shape_box(body_b, hx=box_size, hy=box_size, hz=box_size, cfg=suction_cfg)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.5, -0.5, 0.6),
            pitch=-15.0,
            yaw=-160.0,
        )

    def gui(self, ui):
        ui.text("Suction Cup Controls")
        ui.separator()
        changed, self.adhesion_strength = ui.slider_float(
            "Adhesion Strength", self.adhesion_strength, 0.0, 1.0
        )
        ui.text("Right-click drag the offset box")
        ui.text("near the resting box to stick them.")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.control.shape_adhesion_ctrl.fill_(self.adhesion_strength)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Both boxes should still be above the ground after settling
        for i in range(2):
            z = float(self.state_0.body_q.numpy()[i][2])
            assert z > 0.0, f"Body {i} fell through the ground (z={z})"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
