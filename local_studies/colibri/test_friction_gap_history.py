"""Native preparation gates for material history across a tiny separation."""

import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.test_friction_break_state import _prepare
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


class TestGapHistory(unittest.TestCase):
    def test_small_gap_keeps_material_anchor_without_creating_impulse(self):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9999), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            step_layout="single_world",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            sor_boost=1.0,
        )
        state = model.state()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0e-6)
        world = solver.world
        cc = world._contact_container
        positions = world.bodies.position.numpy()
        dynamic_body = int(np.flatnonzero(world.bodies.inverse_mass.numpy() > 0)[0])
        positions[dynamic_body, 2] = 1.000001
        world.bodies.position.assign(positions)
        anchors = cc.lambdas.numpy().copy()
        anchors[6, 0] += 1.0e-5
        cc.lambdas.assign(anchors)
        cc.impulses.zero_()
        inputs = [
            world._contact_cols,
            world.bodies,
            world.particles or ParticleContainer(),
            world.num_bodies,
            cc,
            world._active_contact_views(),
            world._copy_state or CopyStateContainer(),
        ]
        witnesses = (contacts.rigid_contact_point0.numpy().copy(), contacts.rigid_contact_point1.numpy().copy())
        wp.launch(_prepare, dim=1, inputs=inputs, device=model.device)
        self.assertGreater(float(cc.derived.numpy()[3, 0]), 0.0)
        np.testing.assert_array_equal(cc.lambdas.numpy()[6:12, 0], anchors[6:12, 0])
        np.testing.assert_array_equal(cc.impulses.numpy()[:, 0], 0.0)
        np.testing.assert_array_equal(contacts.rigid_contact_point0.numpy(), witnesses[0])
        np.testing.assert_array_equal(contacts.rigid_contact_point1.numpy(), witnesses[1])
        # A real loss of geometric correlation must still discard the reference.
        positions[dynamic_body, 2] = 1.003
        world.bodies.position.assign(positions)
        wp.launch(_prepare, dim=1, inputs=inputs, device=model.device)
        self.assertFalse(
            np.array_equal(cc.lambdas.numpy()[6:12, 0], anchors[6:12, 0]),
            str((cc.derived.numpy()[:, 0], world.bodies.position.numpy(), world._contact_cols.data.numpy()[:, 0])),
        )
        np.testing.assert_array_equal(cc.impulses.numpy()[:, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
