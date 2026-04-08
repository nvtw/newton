# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Play (ONNX)
#
# Loads a trained ONNX policy (with baked normalization) and replays it
# on a single ANYmal C.  Uses the exact same obs/action kernels as
# the training environment.
#
# Command: python -m newton.examples robot_anymal_c_play_onnx
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import GeoType
from newton._src.onnx_runtime import OnnxRuntime

# Import kernels and constants from training example
from newton.examples.robot.example_robot_anymal_c_train_onnx import (
    _ACT_DIM,
    _ACTION_SCALE,
    _INITIAL_JOINT_Q_NAMES,
    _MUJOCO_TO_LAB,
    _OBS_DIM,
    _QD_STRIDE,
    _Q_STRIDE,
    _compute_obs_kernel,
)


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[float],
    mujoco_to_lab: wp.array[int],
    joint_pos_initial: wp.array[float],
    joint_target_pos: wp.array[float],
):
    """Apply tanh squashing (matching training) then scale to joint targets."""
    env, j = wp.tid()
    lab_j = mujoco_to_lab[j]
    a = wp.tanh(actions[env, j])  # tanh bounding -- must match training
    joint_target_pos[env * _QD_STRIDE + 6 + lab_j] = joint_pos_initial[lab_j] + _ACTION_SCALE * a


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = 0.005
        self.frame_dt = self.sim_dt * self.sim_substeps

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        builder.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True, enable_self_collisions=False,
            collapse_fixed_joints=True, ignore_inertial_definitions=False,
        )
        for i in range(len(builder.shape_type)):
            if builder.shape_type[i] == GeoType.SPHERE:
                r = builder.shape_scale[i][0]
                builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)
        for name, value in _INITIAL_JOINT_Q_NAMES.items():
            idx = next((i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}")), None)
            if idx is None:
                raise ValueError(f"Joint '{name}' not found")
            builder.joint_q[idx + 6] = value
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        builder.add_ground_plane()

        self.model = builder.finalize(device=self.device)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, solver="newton",
            ls_parallel=False, ls_iterations=50, njmax=50, nconmax=100,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.follow_cam = True

        d = self.device
        self._lab_to_mujoco = wp.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=wp.int32, device=d)
        self._mujoco_to_lab = wp.array(_MUJOCO_TO_LAB, dtype=wp.int32, device=d)
        full_q = self.state_0.joint_q.numpy()
        self._joint_pos_initial = wp.array(full_q[7:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=d)

        policy_path = getattr(args, "policy", "anymal_raw.onnx")
        self.policy = OnnxRuntime(policy_path, device=str(d), batch_size=1)
        self.obs = wp.zeros((1, _OBS_DIM), dtype=wp.float32, device=d)
        self.norm_obs = wp.zeros((1, _OBS_DIM), dtype=wp.float32, device=d)
        self.last_actions = wp.zeros((1, _ACT_DIM), dtype=wp.float32, device=d)
        self.commands = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32, device=d)

        # Load normalizer stats from training
        from newton._src.ppo import ObsNormalizer  # noqa: PLC0415

        self.normalizer = ObsNormalizer(_OBS_DIM, device=str(d))
        norm_path = getattr(args, "norm", "anymal_norm.npz")
        try:
            data = np.load(norm_path)
            wp.copy(self.normalizer.mean, wp.array(data["mean"], dtype=wp.float32, device=d))
            wp.copy(self.normalizer.inv_std, wp.array(data["inv_std"], dtype=wp.float32, device=d))
            print(f"Loaded normalizer from {norm_path}")
        except FileNotFoundError:
            print(f"Warning: {norm_path} not found, using unnormalized obs")

        # Graph capture
        if d.is_cuda:
            self.control.joint_target_pos = wp.zeros(18, dtype=wp.float32, device=d)
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Same obs kernel as training
        wp.launch(_compute_obs_kernel, dim=1, inputs=[
            self.state_0.joint_q, self.state_0.joint_qd,
            self.last_actions, self.commands, self._joint_pos_initial,
            self._lab_to_mujoco,
            self.obs,
        ], device=self.device)

        # Normalize obs then run ONNX inference
        self.normalizer.normalize(self.obs, self.norm_obs)
        act = self.policy({"observation": self.norm_obs})["action"]

        # Copy to last_actions
        wp.copy(self.last_actions, act)

        # Apply actions
        wp.launch(_apply_actions_kernel, dim=(1, _ACT_DIM), inputs=[
            act, self._mujoco_to_lab, self._joint_pos_initial, self.control.joint_target_pos,
        ], device=self.device)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt
        self._step_count = getattr(self, "_step_count", 0) + 1
        if self._step_count % 50 == 0:
            q = self.state_0.joint_q.numpy()
            qd = self.state_0.joint_qd.numpy()
            h = q[2]
            fwd = qd[1]
            max_jvel = np.abs(qd[6:]).max()
            act_np = act.numpy()[0]
            print(
                f"t={self.sim_time:.1f}s h={h:.3f} fwd_vel={fwd:+.3f}"
                f" max_joint_vel={max_jvel:.1f} act=[{act_np.min():+.2f},{act_np.max():+.2f}]"
            )

    def render(self):
        if self.follow_cam:
            pos = self.state_0.joint_q.numpy()[:3]
            self.viewer.set_camera(
                pos=wp.vec3(*pos) + wp.vec3(10.0, 0.0, 2.0), pitch=0.0, yaw=-180.0,
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--policy", type=str, default="anymal_raw.onnx")
        parser.add_argument("--norm", type=str, default="anymal_norm.npz")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
