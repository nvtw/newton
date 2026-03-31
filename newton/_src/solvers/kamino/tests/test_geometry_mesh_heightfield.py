# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mesh and heightfield collision support in Kamino.

Tests the unified collision pipeline, Newton-to-Kamino contact conversion,
and solver integration with mesh and heightfield shapes via the
``ModelKamino.from_newton()`` path.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from newton._src.solvers.kamino._src.geometry.unified import CollisionPipelineUnifiedKamino
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Scene Builders
###

SPHERE_RADIUS = 0.25
BOX_HALF = 0.5


def _build_sphere_on_heightfield() -> newton.ModelBuilder:
    """Sphere resting on a flat heightfield (elevation = 0 everywhere)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 8, 8
    elevation = np.zeros((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_sphere_on_mesh_box() -> newton.ModelBuilder:
    """Sphere resting on a box-shaped triangle mesh."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    mesh = newton.Mesh.create_box(BOX_HALF, BOX_HALF, BOX_HALF)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 0.02

    # Static mesh box centered at origin (top face at z = 0.5)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg)

    # Sphere slightly penetrating the mesh box top face
    body = builder.add_body(
        xform=wp.transform(p=(0.0, 0.0, BOX_HALF + SPHERE_RADIUS - 0.005), q=wp.quat_identity()),
    )
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_box_on_heightfield() -> newton.ModelBuilder:
    """Box resting on a flat heightfield."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 8, 8
    elevation = np.zeros((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, BOX_HALF), q=wp.quat_identity()))
    builder.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)

    return builder


def _build_mixed_scene() -> newton.ModelBuilder:
    """Scene with both primitive shapes and a mesh — sphere on mesh box + sphere on ground plane."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    mesh = newton.Mesh.create_box(BOX_HALF, BOX_HALF, BOX_HALF)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    # Static mesh box at x=2
    builder.add_shape_mesh(
        body=-1,
        mesh=mesh,
        xform=wp.transform(p=(2.0, 0.0, 0.0), q=wp.quat_identity()),
        cfg=cfg,
    )

    # Ground plane
    builder.add_ground_plane(cfg=cfg)

    # Sphere on ground plane at origin
    body_a = builder.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    builder.add_shape_sphere(body_a, radius=SPHERE_RADIUS, cfg=cfg)

    # Sphere on mesh box at x=2 (slightly penetrating)
    body_b = builder.add_body(
        xform=wp.transform(p=(2.0, 0.0, BOX_HALF + SPHERE_RADIUS - 0.005), q=wp.quat_identity()),
    )
    builder.add_shape_sphere(body_b, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_heightfield_terrain() -> newton.ModelBuilder:
    """Sphere on a non-flat heightfield with sine-wave terrain."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 20, 20
    x = np.linspace(-2.0, 2.0, ncol)
    y = np.linspace(-2.0, 2.0, nrow)
    xx, yy = np.meshgrid(x, y)
    elevation = (0.1 * np.sin(xx) * np.cos(yy)).astype(np.float32)
    # Elevation at center (0,0) ≈ 0
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=2.0, hy=2.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    # Sphere at the center, slightly above terrain surface (elevation ≈ 0)
    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_multi_world_heightfield(num_worlds: int = 3) -> newton.ModelBuilder:
    """Multi-world scene, each with sphere-on-flat-heightfield."""
    single = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 8, 8
    elevation = np.zeros((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    body = single.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    single.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    multi = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(num_worlds):
        multi.add_world(single)

    multi.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    return multi


###
# Helpers
###


def _finalize_and_get_kamino(builder, device):
    """Finalize Newton model, create Kamino model, data, and state."""
    newton_model = builder.finalize(device=device)
    model = ModelKamino.from_newton(newton_model)
    data = model.data(device=device)
    state = model.state(device=device)
    return newton_model, model, data, state


def _run_unified_pipeline(model, data, state, device):
    """Create unified pipeline, allocate contacts, run collision detection."""
    num_worlds = model.size.num_worlds
    pipeline = CollisionPipelineUnifiedKamino(
        model=model,
        broadphase="nxn",
        device=device,
    )
    contacts = ContactsKamino(capacity=[4096] * num_worlds, device=device)
    contacts.clear()
    pipeline.collide(data, state, contacts)
    return contacts


def _run_newton_cd_and_convert(newton_model, device):
    """Run Newton's CD pipeline and convert contacts to Kamino format."""
    # Normalize shape_world for single-world models
    if newton_model.world_count == 1:
        sw = newton_model.shape_world.numpy()
        if np.any(sw < 0):
            sw[sw < 0] = 0
            newton_model.shape_world.assign(sw)

    state = newton_model.state()
    newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, state)
    newton_contacts = newton_model.collide(state)

    nc = int(newton_contacts.rigid_contact_count.numpy()[0])
    kamino_contacts = ContactsKamino(capacity=[max(nc + 64, 256)], device=device)
    convert_contacts_newton_to_kamino(newton_model, state, newton_contacts, kamino_contacts)
    wp.synchronize()

    return kamino_contacts, nc


def _assert_contacts_valid(test, contacts, min_count=1):
    """Common assertions on Kamino contacts."""
    nc = int(contacts.model_active_contacts.numpy()[0])
    test.assertGreaterEqual(nc, min_count, f"Expected at least {min_count} contacts, got {nc}")

    gapfunc = contacts.gapfunc.numpy()[:nc]
    for i in range(nc):
        n = gapfunc[i, :3]
        norm = np.linalg.norm(n)
        test.assertTrue(np.isclose(norm, 1.0, atol=1e-4), f"Contact {i}: normal not unit (norm={norm})")

    return nc


###
# Test Classes
###


class TestUnifiedPipelineMeshHeightfield(unittest.TestCase):
    """Tests unified collision pipeline with mesh/heightfield shapes via from_newton()."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_sphere_on_flat_heightfield(self):
        """Sphere touching a flat heightfield must produce contacts with upward normal."""
        _, model, data, state = _finalize_and_get_kamino(
            _build_sphere_on_heightfield(), self.default_device
        )
        contacts = _run_unified_pipeline(model, data, state, self.default_device)
        nc = _assert_contacts_valid(self, contacts, min_count=1)

        gapfunc = contacts.gapfunc.numpy()[:nc]
        for i in range(nc):
            # Normal should point approximately upward (z-up)
            self.assertGreater(gapfunc[i, 2], 0.5, f"Contact {i}: normal z={gapfunc[i, 2]} not upward")

    def test_02_box_on_flat_heightfield(self):
        """Box touching a flat heightfield must produce multiple contacts."""
        _, model, data, state = _finalize_and_get_kamino(
            _build_box_on_heightfield(), self.default_device
        )
        contacts = _run_unified_pipeline(model, data, state, self.default_device)
        _assert_contacts_valid(self, contacts, min_count=1)

    def test_03_heightfield_terrain(self):
        """Sphere on non-flat terrain must produce contacts."""
        _, model, data, state = _finalize_and_get_kamino(
            _build_heightfield_terrain(), self.default_device
        )
        contacts = _run_unified_pipeline(model, data, state, self.default_device)
        _assert_contacts_valid(self, contacts, min_count=1)

    def test_04_multi_world_heightfield(self):
        """Multi-world sphere-on-heightfield must produce contacts in each world."""
        num_worlds = 3
        _, model, data, state = _finalize_and_get_kamino(
            _build_multi_world_heightfield(num_worlds), self.default_device
        )
        contacts = _run_unified_pipeline(model, data, state, self.default_device)
        nc = _assert_contacts_valid(self, contacts, min_count=num_worlds)

        world_counts = contacts.world_active_contacts.numpy()[:num_worlds]
        for w in range(num_worlds):
            self.assertGreater(
                int(world_counts[w]), 0, f"World {w}: expected contacts, got {world_counts[w]}"
            )


class TestNewtonCollisionPathMeshHeightfield(unittest.TestCase):
    """Tests Newton model.collide() -> convert_contacts_newton_to_kamino() path."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_newton_to_kamino_heightfield(self):
        """Newton CD on sphere-on-heightfield, converted to Kamino format."""
        builder = _build_sphere_on_heightfield()
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertGreater(newton_count, 0, "Newton must produce contacts")

        nc = _assert_contacts_valid(self, kamino_contacts, min_count=1)

        # Verify A/B convention: bid_B >= 0
        bid_AB = kamino_contacts.bid_AB.numpy()[:nc]
        for i in range(nc):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0")

    def test_02_newton_to_kamino_mesh(self):
        """Newton CD on sphere-on-mesh-box, converted to Kamino format."""
        builder = _build_sphere_on_mesh_box()
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertGreater(newton_count, 0, "Newton must produce contacts")

        nc = _assert_contacts_valid(self, kamino_contacts, min_count=1)

        bid_AB = kamino_contacts.bid_AB.numpy()[:nc]
        for i in range(nc):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0")

    def test_03_newton_to_kamino_mixed(self):
        """Newton CD on mixed scene (primitive + mesh), converted to Kamino format."""
        builder = _build_mixed_scene()
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertGreater(newton_count, 0, "Newton must produce contacts")
        # Two spheres: one on ground plane, one on mesh box
        _assert_contacts_valid(self, kamino_contacts, min_count=2)


class TestSolverWithMeshHeightfield(unittest.TestCase):
    """Tests Kamino solver stepping with mesh/heightfield contacts (end-to-end)."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def _step_with_newton_cd(self, builder, num_steps=200, dt=0.005):
        """Build scene, step solver with Newton CD, return final body z position."""
        newton_model = builder.finalize(device=self.default_device)

        # Normalize shape_world
        if newton_model.world_count == 1:
            sw = newton_model.shape_world.numpy()
            if np.any(sw < 0):
                sw[sw < 0] = 0
                newton_model.shape_world.assign(sw)

        newton_model.set_gravity((0.0, 0.0, -9.81))

        model = ModelKamino.from_newton(newton_model)
        model.time.set_uniform_timestep(dt)

        state_p = model.state(device=self.default_device)
        state_n = model.state(device=self.default_device)
        control = model.control(device=self.default_device)

        per_world = max(1024, newton_model.rigid_contact_max // max(newton_model.world_count, 1))
        contacts = ContactsKamino(capacity=[per_world], device=self.default_device)

        solver = SolverKaminoImpl(model=model, contacts=contacts)
        solver.reset(state_out=state_n)
        state_p.copy_from(state_n)

        newton_state = newton_model.state()
        newton_contacts = newton_model.contacts()

        from newton._src.solvers.kamino._src.core.bodies import convert_body_com_to_origin

        for _ in range(num_steps):
            state_p.copy_from(state_n)

            # Run Newton CD
            convert_body_com_to_origin(
                body_com=model.bodies.i_r_com_i,
                body_q_com=state_p.q_i,
                body_q=newton_state.body_q,
            )
            newton_model.collide(newton_state, newton_contacts)
            convert_contacts_newton_to_kamino(newton_model, newton_state, newton_contacts, contacts)

            solver.step(
                state_in=state_p,
                state_out=state_n,
                control=control,
                contacts=contacts,
                detector=None,
            )

        # Read final body positions (COM frame)
        q_i = state_n.q_i.numpy()
        return q_i

    def test_01_sphere_falls_onto_heightfield(self):
        """Sphere dropped onto flat heightfield must come to rest near the surface."""
        builder = _build_sphere_on_heightfield()
        # Move sphere up so it falls
        builder.body_q[0] = wp.transform(p=(0.0, 0.0, SPHERE_RADIUS + 0.5), q=wp.quat_identity())

        q_i = self._step_with_newton_cd(builder, num_steps=400, dt=0.005)

        # Body 0 is the sphere; its COM z should be near SPHERE_RADIUS (resting on z=0 surface)
        sphere_z = q_i[0, 2]
        self.assertGreater(sphere_z, -0.1, f"Sphere fell through surface: z={sphere_z}")
        self.assertLess(sphere_z, SPHERE_RADIUS + 0.5, f"Sphere didn't fall: z={sphere_z}")

    def test_02_sphere_falls_onto_mesh_box(self):
        """Sphere dropped onto mesh box must come to rest on top."""
        builder = _build_sphere_on_mesh_box()
        # Move sphere up so it falls
        builder.body_q[0] = wp.transform(
            p=(0.0, 0.0, BOX_HALF + SPHERE_RADIUS + 0.5), q=wp.quat_identity()
        )

        q_i = self._step_with_newton_cd(builder, num_steps=400, dt=0.005)

        sphere_z = q_i[0, 2]
        expected_rest = BOX_HALF + SPHERE_RADIUS
        self.assertGreater(sphere_z, expected_rest - 0.2, f"Sphere fell through mesh: z={sphere_z}")
        self.assertLess(sphere_z, expected_rest + 0.5, f"Sphere didn't fall: z={sphere_z}")


if __name__ == "__main__":
    unittest.main()
