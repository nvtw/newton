# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world contact isolation regressions for ``SolverPhoenX``."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode


class _Scene:
    def __init__(self, model, solver, state_0, state_1, control, contacts):
        self.model = model
        self.solver = solver
        self.state_0 = state_0
        self.state_1 = state_1
        self.control = control
        self.contacts = contacts


def _make_h1_isolation_scene(world_count: int) -> _Scene:
    h1 = newton.ModelBuilder()
    h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1.0e-5)
    h1.default_shape_cfg.ke = 2.0e3
    h1.default_shape_cfg.kd = 1.0e2
    h1.default_shape_cfg.kf = 1.0e3
    h1.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_h1")
    h1.add_usd(
        str(asset_path / "usd_structured" / "h1.usda"),
        ignore_paths=["/GroundPlane"],
        enable_self_collisions=False,
    )
    h1.approximate_meshes("bounding_box")
    for i in range(len(h1.joint_target_ke)):
        h1.joint_target_ke[i] = 150.0
        h1.joint_target_kd[i] = 5.0
        h1.joint_target_mode[i] = int(JointTargetMode.POSITION)

    builder = newton.ModelBuilder()
    builder.replicate(h1, world_count)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=8,
        velocity_iterations=1,
        multi_world_scheduler="fast_tail",
        prepare_refresh_stride=1,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts(collision_pipeline=getattr(model, "_collision_pipeline", None))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    return _Scene(model, solver, state_0, state_1, control, contacts)


def _step(scene: _Scene, force: tuple[float, float, float] | None = None) -> int:
    scene.model.collide(scene.state_0, scene.contacts)
    contact_count = int(scene.contacts.rigid_contact_count.numpy()[0])

    scene.state_0.clear_forces()
    if force is not None:
        body_f = np.zeros((scene.model.body_count, 6), dtype=np.float32)
        body_f[5, :3] = np.asarray(force, dtype=np.float32)
        scene.state_0.body_f.assign(body_f)

    scene.solver.step(scene.state_0, scene.state_1, scene.control, scene.contacts, 1.0 / 200.0)
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
    return contact_count


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX multi-world contact isolation requires CUDA")
class TestPhoenXMultiWorldContactIsolation(unittest.TestCase):
    def test_lower_world_contact_count_change_does_not_perturb_higher_worlds(self) -> None:
        baseline = _make_h1_isolation_scene(world_count=8)
        disturbed = _make_h1_isolation_scene(world_count=8)

        world_id = baseline.model.body_world.numpy()
        self.assertEqual(int(world_id[5]), 0)
        higher_world_bodies = world_id > 0

        contact_counts_differed = False
        max_higher_world_delta = 0.0
        max_world0_delta = 0.0
        for step in range(24):
            baseline_contacts = _step(baseline)
            force = (1500.0, 0.0, 2000.0) if step < 12 else None
            disturbed_contacts = _step(disturbed, force=force)
            contact_counts_differed = contact_counts_differed or baseline_contacts != disturbed_contacts

            baseline_x = baseline.state_0.body_q.numpy()[:, :3]
            disturbed_x = disturbed.state_0.body_q.numpy()[:, :3]
            max_higher_world_delta = max(
                max_higher_world_delta,
                float(np.max(np.abs(baseline_x[higher_world_bodies] - disturbed_x[higher_world_bodies]))),
            )
            max_world0_delta = max(
                max_world0_delta,
                float(np.max(np.abs(baseline_x[~higher_world_bodies] - disturbed_x[~higher_world_bodies]))),
            )

        self.assertTrue(contact_counts_differed, "world-0 disturbance did not change the contact layout")
        self.assertGreater(max_world0_delta, 0.1, "world-0 disturbance was too small for the isolation check")
        self.assertLess(
            max_higher_world_delta,
            1.0e-6,
            "lower-world contact compaction perturbed bodies in higher worlds",
        )


if __name__ == "__main__":
    unittest.main()
