# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference-style DR Legs validation for PhoenX articulation DVI."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton.utils

PHYSICS_DT = 0.004
SUBSTEPS = 4
FRAME_COUNT = 100
DRIVEN_KE = 5.0
DRIVEN_KD = 0.2
DRIVEN_EFFORT = 3.1
PASSIVE_EFFORT = 400.0
ACTION_SCALE = 0.3

DR_LEGS_ACTUATED_JOINTS = {
    "j1_l_i",
    "j2_l_i",
    "j6_l_i",
    "j7_l_i",
    "j2_l_o",
    "j7_l_o",
    "j1_r_i",
    "j2_r_i",
    "j6_r_i",
    "j7_r_i",
    "j2_r_o",
    "j7_r_o",
}


def _short_label(label: object) -> str:
    return str(label).rsplit("/", 1)[-1]


def _generate_actions(n_frames: int, n_dofs: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 1.5, size=n_dofs)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_dofs)
    times = np.arange(n_frames, dtype=np.float32) * PHYSICS_DT
    return np.sin(2.0 * np.pi * freqs[None, :] * times[:, None] + phases[None, :]).astype(np.float32)


def _transform_point(body_q: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    p = body_q[:3]
    x, y, z, w = (float(v) for v in body_q[3:7])
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return p + rotation @ local_point[:3]


def _loop_residual(model: newton.Model, state: newton.State) -> np.ndarray:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    joint_type = model.joint_type.numpy()
    free_type = int(newton.JointType.FREE)
    distance_type = int(newton.JointType.DISTANCE)

    residuals: list[float] = []
    for joint_index in range(len(joint_parent)):
        if int(joint_type[joint_index]) in (free_type, distance_type):
            continue
        parent = int(joint_parent[joint_index])
        child = int(joint_child[joint_index])
        parent_anchor = (
            joint_x_p[joint_index][:3] if parent < 0 else _transform_point(body_q[parent], joint_x_p[joint_index])
        )
        child_anchor = _transform_point(body_q[child], joint_x_c[joint_index])
        residuals.append(float(np.linalg.norm(parent_anchor - child_anchor)))
    return np.asarray(residuals if residuals else [0.0], dtype=np.float64)


def _build_hanging_dr_legs_model(device: str) -> tuple[newton.Model, list[int]]:
    asset_path = newton.utils.download_asset("disneyresearch")
    asset_file = str(asset_path / "dr_legs/usd/dr_legs_with_boxes.usda")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_shape_cfg.margin = 1.0e-6
    builder.default_shape_cfg.gap = 0.005
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No articulation was found.*")
        builder.add_usd(
            asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            floating=True,
        )

    pelvis = next((i for i, label in enumerate(builder.body_label) if "pelvis" in str(label).lower()), 0)
    builder.add_joint_fixed(parent=-1, child=pelvis, parent_xform=builder.body_q[pelvis], label="world_to_pelvis_fixed")

    model = builder.finalize(skip_validation_joints=True, device=device)

    target_ke = model.joint_target_ke.numpy()
    target_kd = model.joint_target_kd.numpy()
    target_mode = model.joint_target_mode.numpy()
    armature = model.joint_armature.numpy()
    effort_limit = model.joint_effort_limit.numpy()
    qd_start = model.joint_qd_start.numpy()
    actuated_dofs: list[int] = []

    for joint_index in range(model.joint_count):
        dof = int(qd_start[joint_index])
        if dof < 0 or dof >= len(target_ke):
            continue
        target_mode[dof] = int(newton.JointTargetMode.POSITION)
        armature[dof] = 0.0
        if _short_label(model.joint_label[joint_index]) in DR_LEGS_ACTUATED_JOINTS:
            target_ke[dof] = DRIVEN_KE
            target_kd[dof] = DRIVEN_KD
            effort_limit[dof] = DRIVEN_EFFORT
            actuated_dofs.append(dof)
        else:
            target_ke[dof] = 0.0
            target_kd[dof] = 0.0
            effort_limit[dof] = PASSIVE_EFFORT

    model.joint_target_ke.assign(target_ke)
    model.joint_target_kd.assign(target_kd)
    model.joint_target_mode.assign(target_mode)
    model.joint_armature.assign(armature)
    model.joint_effort_limit.assign(effort_limit)
    return model, sorted(actuated_dofs)


@unittest.skipUnless(wp.is_cuda_available(), "DR Legs PhoenX DVI reference validation requires CUDA")
class TestDrLegsDVIReference(unittest.TestCase):
    def test_hanging_dr_legs_full_coordinate_dvi_matches_reference_metrics(self) -> None:
        try:
            model, actuated_dofs = _build_hanging_dr_legs_model("cuda:0")
        except ImportError as e:
            raise unittest.SkipTest(f"DR Legs USD dependencies unavailable: {e}") from e

        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=SUBSTEPS,
            solver_iterations=1,
            velocity_iterations=0,
            step_layout="multi_world",
            articulation_dvi=True,
            articulation_dvi_solver="block_sparse",
        )

        topology = solver.world.articulation_topology
        self.assertEqual(model.body_count, 31)
        self.assertEqual(model.joint_count, 37)
        self.assertEqual(model.joint_dof_count, 36)
        self.assertEqual(len(actuated_dofs), 12)
        self.assertEqual(int(topology.active_joint_count), 37)
        self.assertEqual(int(topology.total_rows), 222)
        self.assertEqual(int(np.sum(solver.world.articulation_dvi_joint_mask)), 37)
        self.assertEqual(int(np.sum(solver.world._joint_pgs_enabled.numpy()[: solver.world.num_joints])), 0)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        action_scales = np.zeros(model.joint_dof_count, dtype=np.float32)
        action_scales[actuated_dofs] = ACTION_SCALE
        actions = _generate_actions(FRAME_COUNT, model.joint_dof_count)
        body_q0 = state_0.body_q.numpy()[:, :3].copy()
        max_loop_residual = float(_loop_residual(model, state_0).max())
        max_body_displacement = 0.0

        for frame in range(FRAME_COUNT):
            target = actions[frame] * action_scales
            target_q = control.joint_target_q.numpy()
            target_q[: model.joint_dof_count] = target
            control.joint_target_q.assign(target_q)

            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, PHYSICS_DT)
            state_0, state_1 = state_1, state_0

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            joint_qd = state_0.joint_qd.numpy()
            self.assertTrue(np.isfinite(body_q).all(), f"body_q became non-finite at frame {frame}")
            self.assertTrue(np.isfinite(body_qd).all(), f"body_qd became non-finite at frame {frame}")
            self.assertTrue(np.isfinite(joint_qd).all(), f"joint_qd became non-finite at frame {frame}")

            displacement = np.linalg.norm(body_q[:, :3] - body_q0, axis=1)
            max_body_displacement = max(max_body_displacement, float(displacement.max()))
            max_loop_residual = max(max_loop_residual, float(_loop_residual(model, state_0).max()))

        self.assertGreater(max_body_displacement, 5.0e-2)
        self.assertLess(max_loop_residual, 5.0e-5)


if __name__ == "__main__":
    unittest.main()
