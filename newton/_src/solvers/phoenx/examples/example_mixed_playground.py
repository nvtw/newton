# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kapla tower, Hoberman spheres, standing Unitree G1 robots and bicycle drivetrains in one PhoenX world.

Hoberman joints stay local PGS blocks; the G1 trees and the closed chain loops
are solved as exact mechanisms between the color sweeps (``direct_joint_mask``).

Run with ``uv run -m newton._src.solvers.phoenx.examples.example_mixed_playground``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.examples.example_kapla_hoberman import Example as KaplaHobermanExample
from newton._src.solvers.phoenx.examples.example_kapla_tower import GROUND_HEIGHT
from newton._src.solvers.phoenx.rl_training.g1 import (
    _CTRL_HI_G1,
    _CTRL_LO_G1,
    _DEFAULT_JOINT_POS_G1,
    _NANOG1_ACTUATOR_FORCE_HI_G1,
    _NANOG1_DOF_ARMATURE_G1,
    _NANOG1_DOF_DAMPING_G1,
    ACTION_DIM_G1,
    _apply_nanog1_body_inertials,
    _g1_mjcf_with_visual_meshes_without_mesh_colliders,
)
from newton.examples.phoenx.example_phoenx_bike_transmission import DEFAULT_STARTUP_RAMP_TIME, build_scene

G1_COUNT = 50
G1_COLUMNS = 10
G1_SPACING = 1.4
G1_ROW_START_Y = 8.0
G1_ROOT_HEIGHT = 0.785
G1_STANDING_HEIGHT = 0.6
G1_HOLD_KE = 500.0
G1_HOLD_KD = 10.0
BIKE_COUNT = 5
BIKE_SPACING = 2.0
BIKE_ROW_Y = -14.0
BIKE_STAND_HEIGHT = 1.0
BIKE_CADENCE = -2.0 * np.pi  # crank target [rad/s], 60 rpm
BIKE_MAX_CHAIN_GAP = 0.002


@wp.kernel(enable_backward=False)
def _ramp_joint_velocities(targets: wp.array[wp.float32], dofs: wp.array[wp.int32], target: float, max_delta: float):
    dof = dofs[wp.tid()]
    targets[dof] += wp.clamp(target - targets[dof], -max_delta, max_delta)


def make_g1_builder() -> newton.ModelBuilder:
    """One G1 holding its default stance with position drives and primitive colliders."""
    g1 = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
    g1.default_shape_cfg.ke = 5.0e4
    g1.default_shape_cfg.kd = 5.0e2
    g1.default_shape_cfg.kf = 1.0e3
    g1.default_shape_cfg.mu = 0.75
    g1.default_shape_cfg.gap = 0.0
    mjcf_path = newton.utils.download_asset("unitree_g1") / "mjcf" / "g1_29dof.xml"
    g1.add_mjcf(
        _g1_mjcf_with_visual_meshes_without_mesh_colliders(mjcf_path),
        floating=None,
        enable_self_collisions=False,
        ignore_names=("floor", "ground"),
        parse_visuals=True,
        parse_meshes=True,
        ignore_inertial_definitions=False,
    )
    _apply_nanog1_body_inertials(g1)
    g1.joint_q[:3] = [0.0, 0.0, G1_ROOT_HEIGHT]
    g1.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    g1.joint_q[7 : 7 + ACTION_DIM_G1] = list(_DEFAULT_JOINT_POS_G1)
    g1.joint_target_q[7 : 7 + ACTION_DIM_G1] = list(_DEFAULT_JOINT_POS_G1)
    for i in range(ACTION_DIM_G1):
        dof = i + 6
        # Without a balance policy, a stiff position hold keeps the stance.
        g1.joint_target_ke[dof] = G1_HOLD_KE
        g1.joint_target_kd[dof] = G1_HOLD_KD
        g1.joint_limit_lower[dof] = _CTRL_LO_G1[i]
        g1.joint_limit_upper[dof] = _CTRL_HI_G1[i]
        g1.joint_effort_limit[dof] = _NANOG1_ACTUATOR_FORCE_HI_G1[i]
        g1.joint_damping[dof] = _NANOG1_DOF_DAMPING_G1[i]
        g1.joint_armature[dof] = _NANOG1_DOF_ARMATURE_G1[i]
        g1.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
    return g1


class Example(KaplaHobermanExample):
    """Add standing G1 robots and driven bicycle drivetrains to the Kapla tower and Hoberman ring."""

    # Chain-sprocket meshing needs the drivetrain's integration rate; the sweep
    # count per frame matches the Kapla+Hoberman scene.
    substeps = 8
    solver_iterations = 5
    # Exact mechanisms are projected mid-solve and after the final iteration.
    direct_joint_projection_passes = 2
    # Long drivetrain mesh manifolds become short columns; the 3-5 point
    # brick, sphere and robot manifolds are unaffected.
    contact_chunk_size = 8

    def add_extra_content(self, builder: newton.ModelBuilder) -> None:
        self._exact_joints = []
        first_joint = builder.joint_count
        g1 = make_g1_builder()
        self.g1_root_bodies = []
        for index in range(G1_COUNT):
            row, column = divmod(index, G1_COLUMNS)
            offset = wp.transform(
                p=wp.vec3(
                    (column - 0.5 * (G1_COLUMNS - 1)) * G1_SPACING,
                    G1_ROW_START_Y + row * G1_SPACING,
                    GROUND_HEIGHT,
                ),
                q=wp.quat_identity(),
            )
            first_shape = len(builder.shape_body)
            self.g1_root_bodies.append(builder.body_count)
            builder.add_builder(g1, xform=offset, label_prefix=f"g1_{index:02d}")
            # One negative group per robot: no self-contact, contact with everything else.
            group = -(1000 + index)
            builder.shape_collision_group[first_shape:] = [group] * (len(builder.shape_body) - first_shape)
        self.g1_root_bodies = np.asarray(self.g1_root_bodies, dtype=np.int32)
        self._exact_joints.append((first_joint, builder.joint_count))

        bike = build_scene()
        bike_labels = list(bike.joint_label)
        crank = next(i for i, label in enumerate(bike_labels) if label.endswith("/FrontGears/RevoluteJoint"))
        rear = next(i for i, label in enumerate(bike_labels) if label.endswith("/BackGears/LoadRevoluteJoint"))
        crank_dofs, self.bike_rear_bodies, self.bike_chain_joints = [], [], []
        for index in range(BIKE_COUNT):
            offset = wp.transform(
                p=wp.vec3((index - 0.5 * (BIKE_COUNT - 1)) * BIKE_SPACING, BIKE_ROW_Y, BIKE_STAND_HEIGHT),
                q=wp.quat_identity(),
            )
            first_joint = builder.joint_count
            builder.add_builder(bike, xform=offset, label_prefix=f"bike_{index:02d}")
            crank_dof = builder.joint_qd_start[first_joint + crank]
            builder.joint_target_qd[crank_dof] = 0.0  # ramped to the cadence in simulate_controls
            crank_dofs.append(crank_dof)
            self.bike_rear_bodies.append(builder.joint_child[first_joint + rear])
            self.bike_chain_joints.extend(first_joint + i for i, label in enumerate(bike_labels) if "/Chain/" in label)
            self._exact_joints.append((first_joint, builder.joint_count))
        self.bike_crank_dofs = wp.array(crank_dofs, dtype=wp.int32)

    def direct_joint_mask(self) -> np.ndarray:
        mask = np.zeros(self.model.joint_count, dtype=bool)
        for begin, end in self._exact_joints:
            mask[begin:end] = True
        return mask

    def simulate_controls(self) -> None:
        wp.launch(
            _ramp_joint_velocities,
            dim=len(self.bike_crank_dofs),
            inputs=[
                self.control.joint_target_qd,
                self.bike_crank_dofs,
                BIKE_CADENCE,
                abs(BIKE_CADENCE) * self.frame_dt / DEFAULT_STARTUP_RAMP_TIME,
            ],
        )

    def test_final(self) -> None:
        super().test_final()
        poses = self.state.body_q.numpy()
        root_heights = poses[self.g1_root_bodies, 2] - GROUND_HEIGHT
        fallen = int(np.count_nonzero(root_heights < G1_STANDING_HEIGHT))
        print(f"[PhoenX Mixed] g1_min_root_height={float(root_heights.min()):.6g} m")
        print(f"[PhoenX Mixed] g1_fallen={fallen}")
        assert fallen == 0, f"{fallen} G1 robots fell below {G1_STANDING_HEIGHT} m"

        velocities = self.state.body_qd.numpy()
        rear_speed = np.linalg.norm(velocities[self.bike_rear_bodies, 3:], axis=1)
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        parent_frames = self.model.joint_X_p.numpy()
        child_frames = self.model.joint_X_c.numpy()
        chain_gap = 0.0
        for joint in self.bike_chain_joints:
            anchors = []
            for body, frame in ((parents[joint], parent_frames[joint]), (children[joint], child_frames[joint])):
                pose = wp.transform(wp.vec3(*poses[body, :3]), wp.quat(*poses[body, 3:]))
                anchors.append(wp.transform_point(pose, wp.vec3(*frame[:3])))
            chain_gap = max(chain_gap, float(wp.length(anchors[0] - anchors[1])))
        print(f"[PhoenX Mixed] bike_rear_speed_min={float(rear_speed.min()):.6g} rad/s")
        print(f"[PhoenX Mixed] bike_chain_gap_max={chain_gap:.6g} m")
        assert chain_gap < BIKE_MAX_CHAIN_GAP, f"bike chain pins separated by {chain_gap:.6g} m"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
