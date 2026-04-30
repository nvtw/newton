# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-build wiring test for cloth-triangle constraint rows.

Commit 5 of :file:`PLAN_CLOTH_TRIANGLE.md`. Verifies that
:meth:`PhoenXWorld.populate_cloth_triangles_from_model` correctly
copies particle state and stamps cloth-triangle constraint rows
from a finalised Newton :class:`~newton.Model` built via
:meth:`~newton.ModelBuilder.add_cloth_grid`.

The end-to-end iterate / stepping behaviour is exercised in
Commit 6 (the cloth example + regression test); this file checks
the data-flow at the scene-build boundary.

CUDA-only by Newton convention.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_get_alpha_lambda,
    cloth_triangle_get_alpha_mu,
    cloth_triangle_get_body1,
    cloth_triangle_get_body2,
    cloth_triangle_get_body3,
    cloth_triangle_get_inv_rest,
    cloth_triangle_get_rest_area,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_TRIANGLE,
    ConstraintContainer,
    read_int,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@wp.kernel(enable_backward=False)
def _readback_cloth_rows_kernel(
    c: ConstraintContainer,
    cid_offset: wp.int32,
    out_type: wp.array[wp.int32],
    out_body1: wp.array[wp.int32],
    out_body2: wp.array[wp.int32],
    out_body3: wp.array[wp.int32],
    out_inv_rest: wp.array[wp.mat22f],
    out_rest_area: wp.array[wp.float32],
    out_alpha_lambda: wp.array[wp.float32],
    out_alpha_mu: wp.array[wp.float32],
):
    t = wp.tid()
    cid = cid_offset + t
    out_type[t] = read_int(c, wp.int32(0), cid)
    out_body1[t] = cloth_triangle_get_body1(c, cid)
    out_body2[t] = cloth_triangle_get_body2(c, cid)
    out_body3[t] = cloth_triangle_get_body3(c, cid)
    out_inv_rest[t] = cloth_triangle_get_inv_rest(c, cid)
    out_rest_area[t] = cloth_triangle_get_rest_area(c, cid)
    out_alpha_lambda[t] = cloth_triangle_get_alpha_lambda(c, cid)
    out_alpha_mu[t] = cloth_triangle_get_alpha_mu(c, cid)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-populate test requires CUDA")
class TestPopulateClothTrianglesFromModel(unittest.TestCase):
    """Round-trip a 2x2-cell cloth grid through Newton's
    :class:`~newton.ModelBuilder`, finalise to a :class:`~newton.Model`,
    and verify :meth:`PhoenXWorld.populate_cloth_triangles_from_model`
    transcribes the mesh into PhoenX constraint rows correctly."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        self.dim_x = 2
        self.dim_y = 2
        self.cell = 0.5
        self.mass = 0.1
        # Distinct stiffness values so the alpha derivation can be
        # cross-checked against tri_materials column 0 (k_mu) /
        # column 1 (k_lambda).
        self.tri_ke = 1234.0  # k_mu
        self.tri_ka = 9876.0  # k_lambda
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.mass,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
        )
        self.model = builder.finalize(device=self.device)
        # 3x3 = 9 particles, 2x2x2 = 8 triangles.
        self.expected_particles = (self.dim_x + 1) * (self.dim_y + 1)
        self.expected_triangles = self.dim_x * self.dim_y * 2
        self.assertEqual(self.model.particle_count, self.expected_particles)
        self.assertEqual(self.model.tri_count, self.expected_triangles)

    def _make_world(self, num_joints: int = 0) -> PhoenXWorld:
        bodies = body_container_zeros(max(1, num_joints + 1), device=self.device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            num_cloth_triangles=self.expected_triangles,
            device=self.device,
        )
        return PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=num_joints,
            num_particles=self.expected_particles,
            num_cloth_triangles=self.expected_triangles,
            num_worlds=1,
            step_layout="single_world",
            device=self.device,
        )

    def test_particle_state_copied(self) -> None:
        """Particle position / velocity / inverse-mass arrays mirror
        the model's after :meth:`populate_cloth_triangles_from_model`."""
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)

        np.testing.assert_allclose(
            world.particles.position.numpy(),
            self.model.particle_q.numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            world.particles.velocity.numpy(),
            self.model.particle_qd.numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            world.particles.inverse_mass.numpy(),
            self.model.particle_inv_mass.numpy(),
            atol=1e-6,
        )

    def test_constraint_type_tag_set(self) -> None:
        """Every cloth row carries
        :data:`CONSTRAINT_TYPE_CLOTH_TRIANGLE` at dword 0."""
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)
        type_arr = self._readback(world).type
        self.assertTrue(np.all(type_arr == int(CONSTRAINT_TYPE_CLOTH_TRIANGLE)))

    def test_endpoints_shifted_to_unified_indices(self) -> None:
        """Per-row body1/2/3 are the model's tri_indices shifted by
        ``num_bodies`` so they land in the particle range of the
        unified body-or-particle index space."""
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)
        rb = self._readback(world)
        nb = world.num_bodies
        tri_indices = self.model.tri_indices.numpy().reshape(-1, 3)
        np.testing.assert_array_equal(rb.body1, tri_indices[:, 0] + nb)
        np.testing.assert_array_equal(rb.body2, tri_indices[:, 1] + nb)
        np.testing.assert_array_equal(rb.body3, tri_indices[:, 2] + nb)
        # Endpoints land in the unified-index particle range.
        for arr in (rb.body1, rb.body2, rb.body3):
            self.assertTrue(np.all(arr >= nb))
            self.assertTrue(np.all(arr < nb + world.num_particles))

    def test_inv_rest_and_rest_area_transcribed(self) -> None:
        """``inv_rest`` is verbatim ``model.tri_poses[t]`` and
        ``rest_area`` is verbatim ``model.tri_areas[t]``."""
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)
        rb = self._readback(world)
        np.testing.assert_allclose(rb.inv_rest, self.model.tri_poses.numpy(), atol=1e-5)
        np.testing.assert_allclose(rb.rest_area, self.model.tri_areas.numpy(), atol=1e-6)

    def test_alpha_compliance_derived_from_materials(self) -> None:
        """``alpha_lambda = 1 / (k_lambda * area)``,
        ``alpha_mu = 1 / (k_mu * area)`` -- where the stiffness
        columns come from ``tri_materials[t, 1]`` (k_lambda /
        ``tri_ka``) and ``tri_materials[t, 0]`` (k_mu /
        ``tri_ke``)."""
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)
        rb = self._readback(world)
        materials = self.model.tri_materials.numpy()  # [tri, 5]
        areas = self.model.tri_areas.numpy()
        expected_alpha_mu = 1.0 / (materials[:, 0] * areas)
        expected_alpha_lambda = 1.0 / (materials[:, 1] * areas)
        np.testing.assert_allclose(rb.alpha_mu, expected_alpha_mu, rtol=1e-5)
        np.testing.assert_allclose(rb.alpha_lambda, expected_alpha_lambda, rtol=1e-5)

    def test_particle_count_mismatch_rejected(self) -> None:
        """If the model's particle count differs from
        ``world.num_particles`` the call raises rather than silently
        producing a misaligned state."""
        bodies = body_container_zeros(1, device=self.device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=self.expected_triangles,
            device=self.device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=self.expected_particles + 1,  # off-by-one
            num_cloth_triangles=self.expected_triangles,
            num_worlds=1,
            step_layout="single_world",
            device=self.device,
        )
        with self.assertRaises(ValueError):
            world.populate_cloth_triangles_from_model(self.model)

    def test_triangle_count_mismatch_rejected(self) -> None:
        """If the model's triangle count differs from
        ``world.num_cloth_triangles`` the call raises."""
        bodies = body_container_zeros(1, device=self.device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=self.expected_triangles + 1,
            device=self.device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=self.expected_particles,
            num_cloth_triangles=self.expected_triangles + 1,
            num_worlds=1,
            step_layout="single_world",
            device=self.device,
        )
        with self.assertRaises(ValueError):
            world.populate_cloth_triangles_from_model(self.model)

    # ------------------------------------------------------------------
    # Read-back helper.
    # ------------------------------------------------------------------

    class _Readback:
        __slots__ = (
            "alpha_lambda",
            "alpha_mu",
            "body1",
            "body2",
            "body3",
            "inv_rest",
            "rest_area",
            "type",
        )

    def _readback(self, world: PhoenXWorld) -> _Readback:
        n = self.expected_triangles
        out_type = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_body1 = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_body2 = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_body3 = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_inv_rest = wp.zeros(n, dtype=wp.mat22f, device=self.device)
        out_rest_area = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_alpha_lambda = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_alpha_mu = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.launch(
            kernel=_readback_cloth_rows_kernel,
            dim=n,
            inputs=[
                world.constraints,
                wp.int32(world.num_joints),
                out_type,
                out_body1,
                out_body2,
                out_body3,
                out_inv_rest,
                out_rest_area,
                out_alpha_lambda,
                out_alpha_mu,
            ],
            device=self.device,
        )
        rb = self._Readback()
        rb.type = out_type.numpy()
        rb.body1 = out_body1.numpy()
        rb.body2 = out_body2.numpy()
        rb.body3 = out_body3.numpy()
        rb.inv_rest = out_inv_rest.numpy()
        rb.rest_area = out_rest_area.numpy()
        rb.alpha_lambda = out_alpha_lambda.numpy()
        rb.alpha_mu = out_alpha_mu.numpy()
        return rb


if __name__ == "__main__":
    unittest.main()
