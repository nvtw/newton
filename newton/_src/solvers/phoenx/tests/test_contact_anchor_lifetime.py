# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction references across separation while a collision generation is reused."""

import unittest

import numpy as np
import warp as wp

import newton


class _ContactScene:
    """One sphere over a fixed surface, with explicit collision generation reuse."""

    def __init__(self, *, mesh=False, friction="point"):
        builder = newton.ModelBuilder()
        builder.gravity = wp.vec3(0.0)
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, gap=0.02)
        if mesh:
            surface = newton.Mesh(
                vertices=[(-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (0.0, 2.0, 0.0)],
                indices=[0, 1, 2],
                compute_inertia=False,
            )
            builder.add_shape_mesh(-1, mesh=surface, cfg=cfg)
        else:
            builder.add_ground_plane(cfg=cfg)
        self.body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        builder.add_shape_sphere(self.body, radius=0.1, cfg=cfg)
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=8,
            velocity_iterations=0,
            step_layout="single_world",
            contact_friction_model=friction,
            enable_body_pair_grouping=False,
        )
        self.state, self.output = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.model._collision_pipeline.contacts()
        self.solver.collide(self.state, self.contacts)
        self.count = int(self.contacts.rigid_contact_count.numpy()[0])
        if self.count == 0:
            raise AssertionError("sphere must have an initial contact with its supporting surface")
        self.generation = int(self.contacts.contact_generation.numpy()[0])
        self.step()

    def place(self, *, x, gap, closing_speed=0.0):
        pose = self.state.body_q.numpy()
        pose[self.body] = [x, 0.0, 0.1 + gap, 0.0, 0.0, 0.0, 1.0]
        self.state.body_q.assign(pose)
        velocity = np.zeros((self.model.body_count, 6), dtype=np.float32)
        velocity[self.body, 2] = -closing_speed
        self.state.body_qd.assign(velocity)

    def step(self):
        self.solver.step(self.state, self.output, self.control, self.contacts, 1.0 / 60.0)
        self.state, self.output = self.output, self.state

    def anchors(self):
        return self.solver.world._contact_container.lambdas.numpy()[6:12, : self.count].copy()

    def impulses(self):
        return self.solver.world._contact_container.impulses.numpy()[:, : self.count].copy()


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact integration requires CUDA")
class TestContactAnchorLifetime(unittest.TestCase):
    """Retain near-contact references without adhesion or stale detached history."""

    def test_near_contact_retains_reference_without_unloaded_friction(self):
        """Preserve convex references inside the friction band while respecting zero normal load."""
        for friction in ("point", "patch"):
            with self.subTest(friction=friction):
                scene = _ContactScene(friction=friction)
                if friction == "patch":
                    self.assertGreater(int(scene.solver.world._contact_cols.patch.eligible.numpy().max()), 0)
                anchors = scene.anchors()
                # A small tangential displacement provides a nonzero restoring
                # error, but a stationary open contact has no compressive load.
                scene.place(x=0.0004, gap=0.0005)
                scene.step()
                np.testing.assert_allclose(scene.anchors(), anchors, atol=1.0e-7)
                np.testing.assert_allclose(scene.impulses(), 0.0, atol=1.0e-7)
                np.testing.assert_allclose(scene.state.body_qd.numpy()[scene.body], 0.0, atol=1.0e-7)
                self.assertAlmostEqual(float(scene.state.body_q.numpy()[scene.body, 0]), 0.0004, delta=1.0e-7)
                if friction == "patch":
                    np.testing.assert_allclose(
                        scene.solver.world._contact_cols.patch.impulse_world.numpy(), 0, atol=1e-7
                    )

                # Closing fast enough to activate the speculative normal row
                # supplies load; the retained reference now opposes the offset.
                scene.place(x=0.0004, gap=0.0005, closing_speed=0.1)
                scene.step()
                self.assertGreater(float(scene.impulses()[0].sum()), 0.0)
                self.assertLess(float(scene.state.body_qd.numpy()[scene.body, 0]), 0.0)
                self.assertEqual(int(scene.contacts.contact_generation.numpy()[0]), scene.generation)

    def test_detached_contact_rebases_before_recontact(self):
        """Forget the old tangential reference after definite separation without fresh collision detection."""
        scene = _ContactScene()
        anchors = scene.anchors()
        scene.place(x=0.0008, gap=0.01)
        scene.step()
        self.assertGreater(float(np.max(np.abs(scene.anchors() - anchors))), 0.0001)
        np.testing.assert_allclose(scene.impulses(), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(scene.state.body_qd.numpy()[scene.body], 0.0, atol=1.0e-7)

        scene.place(x=0.0008, gap=0.0, closing_speed=0.1)
        scene.step()
        self.assertGreater(float(scene.impulses()[0].sum()), 0.0)
        # The reused collision witness is now off-center, so its normal impulse
        # can couple translation and rotation. It must not pull toward the old
        # tangential reference at x=0 after the separation reset.
        self.assertGreaterEqual(float(scene.state.body_qd.numpy()[scene.body, 0]), -1.0e-6)
        self.assertGreaterEqual(float(scene.state.body_q.numpy()[scene.body, 0]), 0.0008 - 1.0e-7)
        self.assertEqual(int(scene.contacts.contact_generation.numpy()[0]), scene.generation)

    def test_stale_mesh_contact_keeps_strict_separation_reset(self):
        """Rebase stale mesh witnesses at a positive gap even inside the convex friction band."""
        scene = _ContactScene(mesh=True)
        anchors = scene.anchors()
        scene.place(x=0.0004, gap=0.0005)
        scene.step()
        self.assertGreater(float(np.max(np.abs(scene.anchors() - anchors))), 0.0001)
        np.testing.assert_allclose(scene.impulses(), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(scene.state.body_qd.numpy()[scene.body], 0.0, atol=1.0e-7)
        self.assertEqual(int(scene.contacts.contact_generation.numpy()[0]), scene.generation)


if __name__ == "__main__":
    unittest.main()
