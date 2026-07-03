# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

import newton
import newton.utils

from .anymal import ConfigEnvAnymalPhoenX, EnvAnymalPhoenX

# Policy order is right-hind, right-front, left-hind, left-front, grouped by
# hip/thigh/calf. This matches the shared quadruped symmetry and reward layout.
_GO2_POLICY_TO_MODEL_JOINT = (9, 3, 6, 0, 10, 4, 7, 1, 11, 5, 8, 2)
_GO2_INITIAL_JOINT_Q = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)


@dataclass
class ConfigEnvGo2PhoenX(ConfigEnvAnymalPhoenX):
    """Configuration for :class:`EnvGo2PhoenX`.

    The defaults follow the flat Unitree Go2 velocity task: a 50 Hz policy,
    four 5 ms physics steps, 0.5 rad position actions, and 25/0.5 implicit PD
    gains.
    """

    world_count: int = 4096
    frame_dt: float = 1.0 / 50.0
    sim_substeps: int = 4
    action_scale: float = 0.5
    target_base_height: float = 0.4
    min_base_height: float = 0.18
    min_upright_cos: float = 0.3
    max_episode_steps: int = 1000
    lin_vel_reward_scale: float = 1.5
    yaw_rate_reward_scale: float = 0.75
    lin_vel_tracking_sigma: float = 0.5
    yaw_rate_tracking_sigma: float = 0.5
    z_vel_reward_scale: float = -2.0
    ang_vel_reward_scale: float = -0.05
    action_rate_reward_scale: float = -0.01
    flat_orientation_reward_scale: float = -2.5
    actuator_ke: float = 25.0
    actuator_kd: float = 0.5
    ground_friction: float = 1.0


class EnvGo2PhoenX(EnvAnymalPhoenX):
    """Warp-only Unitree Go2 flat-terrain locomotion environment."""

    _policy_to_model_joint = _GO2_POLICY_TO_MODEL_JOINT

    def __init__(self, config: ConfigEnvGo2PhoenX | None = None, *, device: wp.context.Devicelike = None):
        super().__init__(config or ConfigEnvGo2PhoenX(), device=device)

    def _build_model(self):
        robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        robot.default_shape_cfg.ke = 5.0e4
        robot.default_shape_cfg.kd = 5.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = float(self.config.ground_friction)

        asset_path = newton.utils.download_asset("unitree_go2")
        robot.add_usd(
            str(asset_path / "usd" / "go2.usda"),
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        if len(robot.joint_q) != 19 or len(robot.joint_qd) != 18:
            raise RuntimeError(
                f"Expected Go2 coordinate/dof counts (19, 18), got ({len(robot.joint_q)}, {len(robot.joint_qd)})"
            )
        robot.joint_q[:7] = [0.0, 0.0, self.config.target_base_height, 0.0, 0.0, 0.0, 1.0]
        robot.joint_q[7:19] = _GO2_INITIAL_JOINT_Q
        for dof in range(6, 18):
            robot.joint_target_ke[dof] = float(self.config.actuator_ke)
            robot.joint_target_kd[dof] = float(self.config.actuator_kd)
            robot.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            robot.joint_effort_limit[dof] = 23.5
            robot.joint_velocity_limit[dof] = 30.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = float(self.config.ground_friction)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        return model
