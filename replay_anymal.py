#!/usr/bin/env python3
"""Replay best ANYmal policy using the same env code as training.

Uses the exact same AnymalEnv, observation computation, and action mapping
as training to ensure identical behavior.

Usage: uv run python replay_anymal.py [--policy anymal_best.onnx] [--headless]
"""
import argparse

import numpy as np
import warp as wp

wp.init()

import newton
import newton.examples
import newton.utils
from newton import GeoType
from newton._src.onnx_runtime import OnnxRuntime
from newton._src.pufferlib.envs.anymal import (
    AnymalEnv,
    _ACT_DIM,
    _LAB_TO_MUJOCO,
    _MUJOCO_TO_LAB,
    _OBS_DIM,
    _Q_STRIDE,
    _QD_STRIDE,
    _compute_obs_kernel,
    _quat_rotate_inv,
)


class Example:
    """Replay trained policy using Newton viewer."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        # Build a single robot (matching training env setup exactly)
        art = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(art)
        art.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1e3, limit_kd=1e1)
        art.default_shape_cfg.ke = 5e4
        art.default_shape_cfg.kd = 5e2
        art.default_shape_cfg.kf = 1e3
        art.default_shape_cfg.mu = 0.75
        p = newton.utils.download_asset("anybotics_anymal_c")
        art.add_urdf(
            str(p / "urdf" / "anymal.urdf"), floating=True, enable_self_collisions=False,
            collapse_fixed_joints=True, ignore_inertial_definitions=False,
        )
        for i in range(len(art.shape_type)):
            if art.shape_type[i] == GeoType.SPHERE:
                r = art.shape_scale[i][0]
                art.shape_scale[i] = (r * 2, 0, 0)
        for name, val in {
            "LF_HAA": 0, "LF_HFE": 0.4, "LF_KFE": -0.8,
            "RF_HAA": 0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            "LH_HAA": 0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            "RH_HAA": 0, "RH_HFE": -0.4, "RH_KFE": 0.8,
        }.items():
            idx = next(i for i, l in enumerate(art.joint_label) if l.endswith(f"/{name}"))
            art.joint_q[idx + 6] = val
        for i in range(len(art.joint_target_ke)):
            art.joint_target_ke[i] = 150
            art.joint_target_kd[i] = 5
        art.add_ground_plane()

        self.model = art.finalize()
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

        # Load policy (ONNX with baked normalization)
        policy_path = getattr(args, "policy", "anymal_best.onnx")
        self.policy = OnnxRuntime(policy_path, device=str(self.device))

        # Use SAME observation computation as training env
        d = str(self.device)
        self.obs = wp.zeros((1, _OBS_DIM), dtype=wp.float32, device=d)
        self.last_actions = wp.zeros((1, _ACT_DIM), dtype=wp.float32, device=d)
        self.commands = wp.zeros((1, 3), dtype=wp.float32, device=d)
        ji = self.state_0.joint_q.numpy()[7:_Q_STRIDE].astype(np.float32)
        self._joint_pos_initial = wp.array(ji, dtype=wp.float32, device=d)
        self._lab_to_mujoco = wp.array(_LAB_TO_MUJOCO, dtype=wp.int32, device=d)
        self._mujoco_to_lab = wp.array(_MUJOCO_TO_LAB, dtype=wp.int32, device=d)

        # Set command: walk forward
        self.commands.numpy()[0] = [1.0, 0.0, 0.0]
        wp.copy(self.commands, wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32, device=d))

        self.sim_time = 0.0
        self._step_count = 0

        # CUDA graph for physics
        if self.device.is_cuda:
            self.control.joint_target_pos = wp.zeros(_QD_STRIDE, dtype=wp.float32, device=self.device)
            with wp.ScopedCapture() as capture:
                for _ in range(4):
                    self.state_0.clear_forces()
                    self.viewer.apply_forces(self.state_0)
                    self.model.collide(self.state_0, self.contacts)
                    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, 0.005)
                    self.state_0, self.state_1 = self.state_1, self.state_0
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self):
        # Compute obs using SAME kernel as training
        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q, self.state_0.joint_qd,
                self.last_actions, self.commands,
                self._joint_pos_initial, self._lab_to_mujoco, self.obs,
            ],
            device=str(self.device),
        )

        # Run policy (ONNX includes obs normalization)
        act = self.policy({"observation": self.obs})["action"]

        # Store raw actions as "previous actions" (same as training)
        wp.copy(self.last_actions, act)

        # Apply actions: target = initial + 0.5 * action (same as training)
        act_np = act.numpy()
        mj_to_lab = _MUJOCO_TO_LAB
        ji = self._joint_pos_initial.numpy()
        targets = np.zeros(_QD_STRIDE, dtype=np.float32)
        for j in range(12):
            lab_j = mj_to_lab[j]
            targets[6 + lab_j] = ji[lab_j] + 0.5 * act_np[0, j]
        wp.copy(self.control.joint_target_pos, wp.array(targets, dtype=wp.float32, device=str(self.device)))

        # Step physics
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            for _ in range(4):
                self.state_0.clear_forces()
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, 0.005)
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += 0.02

        self._step_count += 1
        if self._step_count <= 5 or self._step_count % 50 == 0:
            q = self.state_0.joint_q.numpy()
            print(f"step {self._step_count}: pos=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f}]", flush=True)

    def render(self):
        pos = self.state_0.joint_q.numpy()[:3]
        self.viewer.set_camera(pos=wp.vec3(*pos) + wp.vec3(3, 0, 1.5), pitch=-15, yaw=-180)
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
