# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Walk (ONNX)
#
# Shows how to simulate ANYmal C using SolverMuJoCo and control it with a
# policy loaded via the lightweight OnnxRuntime (no PyTorch dependency).
#
# Command: python -m newton.examples robot_anymal_c_walk_onnx
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

lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

_LAB_TO_MUJOCO = np.array(lab_to_mujoco, dtype=np.intp)
_MUJOCO_TO_LAB = np.array(mujoco_to_lab, dtype=np.intp)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate *v* by the inverse of quaternion *q* (XYZW convention)."""
    q_w = q[..., 3:4]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def compute_obs(actions: np.ndarray, state: State, joint_pos_initial: np.ndarray) -> np.ndarray:
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

    command = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    return np.concatenate([vel_b, a_vel_b, grav, command, rearranged_pos, rearranged_vel, actions], axis=1).astype(np.float32)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        for i in range(len(builder.shape_type)):
            if builder.shape_type[i] == GeoType.SPHERE:
                r = builder.shape_scale[i][0]
                builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)

        if not self.is_test:
            terrain_mesh = newton.Mesh.create_terrain(
                grid_size=(8, 3),
                block_size=(3.0, 3.0),
                terrain_types=["random_grid", "flat", "wave", "gap", "pyramid_stairs"],
                terrain_params={
                    "pyramid_stairs": {"step_width": 0.3, "step_height": 0.02, "platform_width": 0.6},
                    "random_grid": {"grid_width": 0.3, "grid_height_range": (0, 0.02)},
                    "wave": {"wave_amplitude": 0.1, "wave_frequency": 2.0},
                },
                seed=42,
                compute_inertia=False,
            )
            terrain_offset = wp.transform(p=wp.vec3(-5, -2.0, 0.01), q=wp.quat_identity())
            builder.add_shape_mesh(
                body=-1,
                mesh=terrain_mesh,
                xform=terrain_offset,
                cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False),
            )
        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for name, value in initial_q.items():
            idx = next(
                (i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}")),
                None,
            )
            if idx is None:
                raise ValueError(f"Joint '{name}' not found in builder.joint_label")
            builder.joint_q[idx + 6] = value

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        self.model = builder.finalize()
        use_mujoco_contacts = getattr(args, "use_mujoco_contacts", False)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=use_mujoco_contacts,
            solver="newton",
            ls_parallel=False,
            ls_iterations=50,
            njmax=50,
            nconmax=100,
        )

        self.viewer.set_model(self.model)

        self.follow_cam = True

        if isinstance(self.viewer, newton.viewer.ViewerGL):

            def toggle_follow_cam(imgui):
                changed, follow_cam = imgui.checkbox("Follow Camera", self.follow_cam)
                if changed:
                    self.follow_cam = follow_cam

            self.viewer.register_ui_callback(toggle_follow_cam, position="side")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        if use_mujoco_contacts:
            self.contacts = None
        else:
            self.contacts = self.model.contacts()

        # Load the ONNX policy via the lightweight OnnxRuntime
        default_policy = str(Path(__file__).resolve().parent.parent.parent / "_src" / "anymal_c_walking_policy.onnx")
        policy_path = getattr(args, "policy", None) or default_policy
        self.policy = OnnxRuntime(policy_path, device=str(self.device))

        self.joint_pos_initial = self.state_0.joint_q.numpy()[7:].reshape(1, 12).astype(np.float32)
        self.act = np.zeros((1, 12), dtype=np.float32)

        self.capture()

    def capture(self):
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

            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        obs = compute_obs(self.act, self.state_0, self.joint_pos_initial)
        obs_wp = wp.array(obs, dtype=wp.float32, device=self.device)
        act_wp = self.policy({"observation": obs_wp})["action"]
        act_np = act_wp.numpy()

        self.act = act_np
        rearranged = act_np[:, _MUJOCO_TO_LAB]
        targets = self.joint_pos_initial + 0.5 * rearranged
        targets_padded = np.zeros(18, dtype=np.float32)
        targets_padded[6:] = targets.squeeze(0)
        wp.copy(self.control.joint_target_pos, wp.array(targets_padded, dtype=wp.float32, device=self.device))

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.follow_cam:
            self.viewer.set_camera(
                pos=wp.vec3(*self.state_0.joint_q.numpy()[:3]) + wp.vec3(10.0, 0.0, 2.0), pitch=0.0, yaw=-180.0
            )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_names = [lbl.split("/")[-1] for lbl in self.model.body_label]
        assert body_names == [
            "base",
            "LF_HIP",
            "LF_THIGH",
            "LF_SHANK",
            "RF_HIP",
            "RF_THIGH",
            "RF_SHANK",
            "LH_HIP",
            "LH_THIGH",
            "LH_SHANK",
            "RH_HIP",
            "RH_THIGH",
            "RH_SHANK",
        ]
        joint_names = [lbl.split("/")[-1] for lbl in self.model.joint_label]
        assert joint_names == [
            "floating_base",
            "LF_HAA",
            "LF_HFE",
            "LF_KFE",
            "RF_HAA",
            "RF_HFE",
            "RF_KFE",
            "LH_HAA",
            "LH_HFE",
            "LH_KFE",
            "RH_HAA",
            "RH_HFE",
            "RH_KFE",
        ]

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 9.0,
        )

        forward_vel_min = wp.spatial_vector(-0.5, 0.9, -0.2, -0.8, -1.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.5, 1.1, 0.2, 0.8, 1.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.math.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.add_argument("--policy", type=str, default=None, help="Path to a custom ONNX policy file.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
