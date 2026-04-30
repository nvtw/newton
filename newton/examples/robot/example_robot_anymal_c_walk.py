# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Walk
#
# Shows how to simulate ANYmal C using SolverMuJoCo and control it with a
# policy trained in PhysX, exported to ONNX and run via Newton's
# Warp-backed :class:`~newton.utils.OnnxRuntime`.
#
# Command: python -m newton.examples robot_anymal_c_walk
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import GeoType, State

lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion (NumPy implementation).

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * (q_w * 2.0)[..., np.newaxis]
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def compute_obs(actions, state: State, joint_pos_initial, indices, gravity_vec, command):
    """Compute the 48-element observation vector using NumPy."""
    root_quat_w = np.asarray(state.joint_q.numpy()[3:7], dtype=np.float32).reshape(1, 4)
    root_lin_vel_w = np.asarray(state.joint_qd.numpy()[:3], dtype=np.float32).reshape(1, 3)
    root_ang_vel_w = np.asarray(state.joint_qd.numpy()[3:6], dtype=np.float32).reshape(1, 3)
    joint_pos_current = np.asarray(state.joint_q.numpy()[7:], dtype=np.float32).reshape(1, 12)
    joint_vel_current = np.asarray(state.joint_qd.numpy()[6:], dtype=np.float32).reshape(1, 12)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)

    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current

    rearranged_joint_pos_rel = joint_pos_rel[:, indices]
    rearranged_joint_vel_rel = joint_vel_rel[:, indices]

    return np.concatenate(
        [
            vel_b,
            a_vel_b,
            grav,
            command,
            rearranged_joint_pos_rel,
            rearranged_joint_vel_rel,
            actions,
        ],
        axis=1,
    ).astype(np.float32)


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

        # Enlarge foot collision spheres to improve walking stability on uneven terrain.
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

        # Load ONNX policy bundled with Newton (Warp-backed runtime, no torch dependency).
        policy_path = newton.examples.get_asset("rl_policies/anymal_walking_policy_physx.onnx")

        self.policy = newton.utils.OnnxRuntime(policy_path, device=str(self.device))
        self._policy_input_name = self.policy.input_names[0]
        self._policy_output_name = self.policy.output_names[0]

        self.joint_pos_initial = np.asarray(self.state_0.joint_q.numpy()[7:], dtype=np.float32).reshape(1, 12)
        self.act = np.zeros((1, 12), dtype=np.float32)

        self.lab_to_mujoco_indices = np.asarray(lab_to_mujoco, dtype=np.int64)
        self.mujoco_to_lab_indices = np.asarray(mujoco_to_lab, dtype=np.int64)
        self.gravity_vec = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        self.command = np.zeros((1, 3), dtype=np.float32)
        self.command[0, 0] = 1.0

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
        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.lab_to_mujoco_indices,
            self.gravity_vec,
            self.command,
        )

        obs_wp = wp.array(obs, dtype=wp.float32, device=self.device)
        out = self.policy({self._policy_input_name: obs_wp})
        self.act = out[self._policy_output_name].numpy().astype(np.float32)

        rearranged_act = self.act[:, self.mujoco_to_lab_indices]
        a = self.joint_pos_initial + 0.5 * rearranged_act
        a_with_zeros = np.concatenate([np.zeros(6, dtype=np.float32), a.squeeze(0)])
        a_wp = wp.array(a_with_zeros, dtype=wp.float32, device=self.device)
        wp.copy(self.control.joint_target_pos, a_wp)

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
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
