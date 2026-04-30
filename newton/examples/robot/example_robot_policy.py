# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot control via keyboard
#
# Shows how to control robots pretrained in IsaacLab with RL.  Policies are
# loaded from ONNX files and run via Newton's Warp-backed
# :class:`~newton.utils.OnnxRuntime` (no PyTorch dependency).
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# python -m newton.examples robot_policy --robot g1_29dof
# python -m newton.examples robot_policy --robot g1_23dof
# python -m newton.examples robot_policy --robot go2
# python -m newton.examples robot_policy --robot anymal
# python -m newton.examples robot_policy --robot anymal --physx
###########################################################################

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, State


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str
    yaml_path: str


# Policy paths are now ONNX files.  These names mirror the original ``.pt``
# layout under ``rl_policies/`` but with the ``.onnx`` extension.
ROBOT_CONFIGS = {
    "anymal": RobotConfig(
        asset_dir="anybotics_anymal_c",
        policy_path={"mjw": "rl_policies/mjw_anymal.onnx", "physx": "rl_policies/physx_anymal.onnx"},
        asset_path="usd/anymal_c.usda",
        yaml_path="rl_policies/anymal.yaml",
    ),
    "go2": RobotConfig(
        asset_dir="unitree_go2",
        policy_path={"mjw": "rl_policies/mjw_go2.onnx", "physx": "rl_policies/physx_go2.onnx"},
        asset_path="usd/go2.usda",
        yaml_path="rl_policies/go2.yaml",
    ),
    "g1_29dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_29DOF.onnx"},
        asset_path="usd/g1_isaac.usd",
        yaml_path="rl_policies/g1_29dof.yaml",
    ),
    "g1_23dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_23DOF.onnx", "physx": "rl_policies/physx_g1_23DOF.onnx"},
        asset_path="usd/g1_minimal.usd",
        yaml_path="rl_policies/g1_23dof.yaml",
    ),
}


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


def compute_obs(
    actions: np.ndarray,
    state: State,
    joint_pos_initial: np.ndarray,
    indices: np.ndarray,
    gravity_vec: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    """Compute observation for robot policy."""
    joint_q = state.joint_q.numpy() if state.joint_q is not None else np.zeros(7, dtype=np.float32)
    joint_qd = state.joint_qd.numpy() if state.joint_qd is not None else np.zeros(6, dtype=np.float32)

    root_quat_w = np.asarray(joint_q[3:7], dtype=np.float32).reshape(1, 4)
    root_lin_vel_w = np.asarray(joint_qd[:3], dtype=np.float32).reshape(1, 3)
    root_ang_vel_w = np.asarray(joint_qd[3:6], dtype=np.float32).reshape(1, 3)
    joint_pos_current = np.asarray(joint_q[7:], dtype=np.float32).reshape(1, -1)
    joint_vel_current = np.asarray(joint_qd[6:], dtype=np.float32).reshape(1, -1)

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


def load_policy_and_setup_arrays(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load ONNX policy and setup initial NumPy arrays for control."""
    print("[INFO] Loading policy from:", policy_path)
    example.policy = newton.utils.OnnxRuntime(policy_path, device=str(example.device))
    example.policy_input_name = example.policy.input_names[0]
    example.policy_output_name = example.policy.output_names[0]

    joint_q = example.state_0.joint_q.numpy() if example.state_0.joint_q is not None else np.zeros(7, dtype=np.float32)
    example.joint_pos_initial = np.asarray(joint_q[joint_pos_slice], dtype=np.float32).reshape(1, num_dofs)
    example.act = np.zeros((1, num_dofs), dtype=np.float32)


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


class Example:
    def __init__(
        self,
        viewer,
        robot_config: RobotConfig,
        config,
        asset_directory: str,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
    ):
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config

        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")

        builder.add_ground_plane()

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config["mjw_joint_pos"]

        for i in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=30,
            njmax=100,
        )

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.physx_to_mjc_indices = np.asarray(physx_to_mjc, dtype=np.int64)
        self.mjc_to_physx_indices = np.asarray(mjc_to_physx, dtype=np.int64)
        self.gravity_vec = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        self.command = np.zeros((1, 3), dtype=np.float32)
        self._reset_key_prev = False

        self.policy = None
        self.policy_input_name = None
        self.policy_output_name = None
        self.joint_pos_initial = None
        self.act = None

        self.capture()

    def capture(self):
        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            self.control.joint_target_pos = wp.zeros(self.config["num_dofs"] + 6, dtype=wp.float32, device=self.device)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            if need_state_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting example")
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.physx_to_mjc_indices,
            self.gravity_vec,
            self.command,
        )

        obs_wp = wp.array(obs, dtype=wp.float32, device=self.device)
        out = self.policy({self.policy_input_name: obs_wp})
        self.act = out[self.policy_output_name].numpy().astype(np.float32)

        rearranged_act = self.act[:, self.mjc_to_physx_indices]
        a = self.joint_pos_initial + self.config["action_scale"] * rearranged_act
        a_with_zeros = np.concatenate([np.zeros(6, dtype=np.float32), a.squeeze(0)])
        a_wp = wp.array(a_with_zeros, dtype=wp.float32, device=self.device)
        wp.copy(self.control.joint_target_pos, a_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--robot", type=str, default="g1_29dof", choices=list(ROBOT_CONFIGS.keys()), help="Robot name to load"
    )
    parser.add_argument("--physx", action="store_true", help="Run physX policy instead of MJWarp.")

    viewer, args = newton.examples.init(parser)

    if args.robot not in ROBOT_CONFIGS:
        print(f"[ERROR] Unknown robot: {args.robot}")
        print(f"[INFO] Available robots: {list(ROBOT_CONFIGS.keys())}")
        exit(1)

    robot_config = ROBOT_CONFIGS[args.robot]
    print(f"[INFO] Selected robot: {args.robot}")

    asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
    print(f"[INFO] Asset directory: {asset_directory}")

    yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    if args.physx:
        if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
            physx_robots = [name for name, cfg in ROBOT_CONFIGS.items() if "physx" in cfg.policy_path]
            print(f"[ERROR] PhysX policy not available for robot '{args.robot}'.")
            print(f"[INFO] Robots with PhysX support: {physx_robots}")
            exit(1)
        policy_path = newton.examples.get_asset(robot_config.policy_path["physx"])
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])
    else:
        policy_path = newton.examples.get_asset(robot_config.policy_path["mjw"])

    example = Example(viewer, robot_config, config, asset_directory, mjc_to_physx, physx_to_mjc)

    load_policy_and_setup_arrays(example, policy_path, config["num_dofs"], slice(7, None))

    newton.examples.run(example, args)
