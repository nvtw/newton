# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Pendulum Play
#
# Loads a trained ONNX policy and replays it on a single double pendulum.
#
# Command: python -m newton.examples basic_pendulum_play
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton._src.onnx_runtime import OnnxRuntime

_Q_STRIDE = 2
_QD_STRIDE = 2
_HX = 1.0


@wp.kernel
def _compute_obs_kernel(joint_q: wp.array[float], joint_qd: wp.array[float], obs: wp.array2d[float]):
    env = wp.tid()
    q = env * _Q_STRIDE
    qd = env * _QD_STRIDE
    obs[env, 0] = wp.sin(joint_q[q])
    obs[env, 1] = wp.cos(joint_q[q])
    obs[env, 2] = wp.sin(joint_q[q + 1])
    obs[env, 3] = wp.cos(joint_q[q + 1])
    obs[env, 4] = joint_qd[qd] * 0.1
    obs[env, 5] = joint_qd[qd + 1] * 0.1


@wp.kernel
def _apply_actions_kernel(actions: wp.array2d[float], joint_act: wp.array[float]):
    env = wp.tid()
    joint_act[env * _QD_STRIDE] = actions[env, 0] * 50.0


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = 1.0 / 100.0 / 10
        self.frame_dt = 1.0 / 100.0

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        hx, hy, hz = _HX, 0.1, 0.1
        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)
        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = builder.add_joint_revolute(
            parent=-1, child=link_0, axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=0.0, target_kd=0.1,
        )
        j1 = builder.add_joint_revolute(
            parent=link_0, child=link_1, axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=0.0, target_kd=0.1,
        )
        builder.add_articulation([j0, j1], label="pendulum")

        self.model = builder.finalize(device=self.device)
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        policy_path = getattr(args, "policy", "pendulum_trained.onnx")
        self.policy = OnnxRuntime(policy_path, device=str(self.device), batch_size=1)
        self.obs = wp.zeros((1, 6), dtype=wp.float32, device=self.device)

    def step(self):
        wp.launch(_compute_obs_kernel, dim=1,
                  inputs=[self.state_0.joint_q, self.state_0.joint_qd, self.obs], device=self.device)
        act = self.policy({"observation": self.obs})["action"]
        wp.launch(_apply_actions_kernel, dim=1,
                  inputs=[act, self.control.joint_act], device=self.device)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--policy", type=str, default="pendulum_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
