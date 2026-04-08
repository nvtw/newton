# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Play Trained Policy
#
# Replays a policy trained with PufferLib PPO (from train_onnx example).
# Applies tanh squashing to match the training setup.
#
# Command: python -m newton.examples robot_anymal_c_play_trained --policy anymal_best.onnx
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import GeoType, State
from newton._src.onnx_runtime import OnnxRuntime

_LAB_TO_MUJOCO = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=np.intp)
_MUJOCO_TO_LAB = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11], dtype=np.intp)


def quat_rotate_inverse(q, v):
    q_w = q[..., 3:4]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def compute_obs(actions, state: State, joint_pos_initial, command):
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    root_quat = q[3:7].reshape(1, 4)
    root_lin_vel = qd[:3].reshape(1, 3)
    root_ang_vel = qd[3:6].reshape(1, 3)
    joint_pos = q[7:].reshape(1, 12)
    joint_vel = qd[6:].reshape(1, 12)
    vel_b = quat_rotate_inverse(root_quat, root_lin_vel)
    a_vel_b = quat_rotate_inverse(root_quat, root_ang_vel)
    grav = quat_rotate_inverse(root_quat, np.array([[0.0, 0.0, -1.0]], dtype=np.float32))
    joint_pos_rel = joint_pos - joint_pos_initial
    rearranged_pos = joint_pos_rel[:, _LAB_TO_MUJOCO]
    rearranged_vel = joint_vel[:, _LAB_TO_MUJOCO]
    return np.concatenate([vel_b, a_vel_b, grav, command, rearranged_pos, rearranged_vel, actions], axis=1).astype(np.float32)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1e3, limit_kd=1e1)
        builder.default_shape_cfg.ke = 5e4
        builder.default_shape_cfg.kd = 5e2
        builder.default_shape_cfg.kf = 1e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        builder.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            xform=wp.transform(wp.vec3(0, 0, 0.62), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        for i in range(len(builder.shape_type)):
            if builder.shape_type[i] == GeoType.SPHERE:
                r = builder.shape_scale[i][0]
                builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)

        initial_q = {
            "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
            "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
        }
        for name, value in initial_q.items():
            idx = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}"))
            builder.joint_q[idx + 6] = value
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        builder.add_ground_plane()

        self.sim_substeps = 4
        self.sim_dt = 0.005
        self.frame_dt = self.sim_dt * self.sim_substeps
        self.sim_time = 0.0

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, solver="newton",
            ls_parallel=False, ls_iterations=50, njmax=50, nconmax=100,
        )
        self.viewer.set_model(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Load trained policy
        policy_path = getattr(args, "policy", "anymal_best.onnx")
        self.policy = OnnxRuntime(policy_path, device=str(self.device))

        self.joint_pos_initial = self.state_0.joint_q.numpy()[7:].reshape(1, 12).astype(np.float32)
        self.act_raw = np.zeros((1, 12), dtype=np.float32)
        # Command: forward velocity
        self.command = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.follow_cam = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            def toggle_follow_cam(imgui):
                changed, follow_cam = imgui.checkbox("Follow Camera", self.follow_cam)
                if changed:
                    self.follow_cam = follow_cam
            self.viewer.register_ui_callback(toggle_follow_cam, position="side")

        # CUDA graph capture
        if self.device.is_cuda:
            self.control.joint_target_pos = wp.zeros(18, dtype=wp.float32, device=self.device)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Observation uses raw (pre-tanh) actions as "previous actions"
        obs = compute_obs(self.act_raw, self.state_0, self.joint_pos_initial, self.command)
        obs_wp = wp.array(obs, dtype=wp.float32, device=self.device)
        act_wp = self.policy({"observation": obs_wp})["action"]
        act_raw = act_wp.numpy()

        # Store raw output for next observation
        self.act_raw = act_raw

        # Direct action mapping (matches IsaacLab and training env)
        rearranged = act_raw[:, _MUJOCO_TO_LAB]
        targets = self.joint_pos_initial + 0.5 * rearranged
        targets_padded = np.zeros(18, dtype=np.float32)
        targets_padded[6:] = targets.squeeze(0)
        wp.copy(self.control.joint_target_pos, wp.array(targets_padded, dtype=wp.float32, device=self.device))

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._step_count = getattr(self, "_step_count", 0) + 1
        if self._step_count <= 5 or self._step_count % 50 == 0:
            q = self.state_0.joint_q.numpy()
            print(f"step {self._step_count}: pos=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f}] act_raw={self.act_raw[0,:3]}", flush=True)

    def render(self):
        if self.follow_cam:
            pos = self.state_0.joint_q.numpy()[:3]
            self.viewer.set_camera(pos=wp.vec3(*pos) + wp.vec3(3.0, 0.0, 1.5), pitch=-15.0, yaw=-180.0)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--policy", type=str, default="anymal_best.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
