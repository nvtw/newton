# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G1 standing-pose stiffness regression for ``SolverPhoenX``.

The ``robot_policy --solver phoenx`` G1 scene relies on the old rigid
joint formulation: armature is baked into the attached body inertias,
PD drive rows are prepared against the same joint targets as the
example, and the TGS relax pass runs before the post-integrate world
inertia refresh. Reordering that path made the torso/hip and ankle
pitch joints behave far too softly, and in the isolated no-contact case
could drive the first frame non-finite.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

_ROBOT_POLICY_MU = 0.75


def _g1_29dof_config() -> dict:
    """Load the G1 29-DoF policy config used by ``example_robot_policy``."""
    import yaml  # noqa: PLC0415

    import newton.utils  # noqa: PLC0415

    asset_dir = newton.utils.download_asset("unitree_g1")
    with open(asset_dir / "rl_policies" / "g1_29dof.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _g1_robot_policy_model() -> newton.Model:
    """Build the G1 PhoenX fixture from ``example_robot_policy``."""
    import newton.utils  # noqa: PLC0415

    cfg = _g1_29dof_config()
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
    builder.default_shape_cfg.mu = _ROBOT_POLICY_MU
    builder.default_shape_cfg.gap = 0.01

    asset_dir = newton.utils.download_asset("unitree_g1")
    builder.add_usd(
        str(asset_dir / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    builder.approximate_meshes("convex_hull")
    builder.add_ground_plane()

    for i in range(len(cfg["mjw_joint_stiffness"])):
        builder.joint_target_ke[i + 6] = cfg["mjw_joint_stiffness"][i]
        builder.joint_target_kd[i + 6] = cfg["mjw_joint_damping"][i]
        builder.joint_armature[i + 6] = cfg["mjw_joint_armature"][i]
        builder.joint_target_mode[i + 6] = int(newton.JointTargetMode.POSITION)

    builder.joint_q[:3] = [0.0, 0.0, 0.76]
    builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
    builder.joint_q[7:] = cfg["mjw_joint_pos"]

    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _standing_target(model: newton.Model) -> np.ndarray:
    target = np.zeros(int(model.joint_dof_count), dtype=np.float32)
    target[6:] = model.joint_q.numpy()[7:]
    return target


def _make_g1_hold_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=20,
        solver_iterations=8,
        velocity_iterations=2,
        velocity_readout="substep_end",
    )


def _run_g1_one_frame_without_contacts(model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
    solver = _make_g1_hold_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control.joint_target_q.assign(_standing_target(model))

    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, 1.0 / 200.0)
    return state_1.body_q.numpy(), state_1.body_qd.numpy()


def _run_g1_hold_pose(
    model: newton.Model,
    *,
    frames: int,
    decimation: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    solver = _make_g1_hold_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control.joint_target_q.assign(_standing_target(model))

    joint_q = wp.zeros(int(model.joint_coord_count), dtype=wp.float32, device=model.device)
    joint_qd = wp.zeros(int(model.joint_dof_count), dtype=wp.float32, device=model.device)
    q_history = np.empty((frames, int(model.joint_coord_count) - 7), dtype=np.float32)
    qd_history = np.empty((frames, int(model.joint_dof_count) - 6), dtype=np.float32)

    for frame in range(frames):
        for _ in range(decimation):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        newton.eval_ik(model, state_0, joint_q, joint_qd)
        q_history[frame] = joint_q.numpy()[7:]
        qd_history[frame] = joint_qd.numpy()[6:]

    return q_history, qd_history


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "G1 PhoenX standing-pose regression runs on CUDA only.",
)
class TestRobotPolicyStandingStiffness(unittest.TestCase):
    """The G1 standing pose should not collapse when held by fixed targets."""

    FRAMES = 40
    WARMUP_FRAMES = 10
    DECIMATION = 4
    DT = 1.0 / 200.0

    def test_g1_one_frame_with_armature_stays_finite(self) -> None:
        model = _g1_robot_policy_model()
        body_q, body_qd = _run_g1_one_frame_without_contacts(model)

        self.assertTrue(np.isfinite(body_q).all(), "G1 body poses became non-finite on the first frame")
        self.assertTrue(np.isfinite(body_qd).all(), "G1 body velocities became non-finite on the first frame")

    def test_g1_standing_pose_joints_remain_stiff(self) -> None:
        cfg = _g1_29dof_config()
        model = _g1_robot_policy_model()
        q_history, qd_history = _run_g1_hold_pose(
            model,
            frames=self.FRAMES,
            decimation=self.DECIMATION,
            dt=self.DT,
        )

        self.assertTrue(np.isfinite(q_history).all(), "G1 joint coordinates became non-finite")
        self.assertTrue(np.isfinite(qd_history).all(), "G1 joint velocities became non-finite")

        target = np.asarray(cfg["mjw_joint_pos"], dtype=np.float32)
        steady_q = q_history[self.WARMUP_FRAMES :]
        steady_qd = qd_history[self.WARMUP_FRAMES :]
        tracking_error = np.abs(steady_q - target)
        self.assertLess(
            float(tracking_error.mean()),
            0.04,
            "standing-pose target tracking drifted into the soft-joint regime",
        )
        self.assertLess(
            float(np.abs(steady_qd).max()),
            8.0,
            "standing-pose hold developed excessive joint speed",
        )

        max_hip_pitch_drift = math.radians(6.0)
        max_ankle_pitch_drift = math.radians(10.0)
        for name, max_drift in (
            ("left_hip_pitch_joint", max_hip_pitch_drift),
            ("right_hip_pitch_joint", max_hip_pitch_drift),
            ("left_ankle_pitch_joint", max_ankle_pitch_drift),
            ("right_ankle_pitch_joint", max_ankle_pitch_drift),
        ):
            dof = cfg["mjw_joint_names"].index(name)
            drift = float(np.abs(steady_q[:, dof] - target[dof]).max())
            self.assertLess(
                drift,
                max_drift,
                f"{name} drifted {math.degrees(drift):.1f} deg while holding the standing pose",
            )


if __name__ == "__main__":
    unittest.main()
