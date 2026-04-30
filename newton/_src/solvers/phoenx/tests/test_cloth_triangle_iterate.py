# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Math validation for the cloth-triangle prepare/iterate funcs.

Commit 2 of :file:`PLAN_CLOTH_TRIANGLE.md` -- the dispatcher
isn't wired in yet, so these tests run the
:func:`cloth_triangle_prepare_for_iteration_at` and
:func:`cloth_triangle_iterate_at` ``@wp.func``s through a test-only
harness kernel.

Three convergence tests covering every code path of the energy
density:

* :class:`TestRestPoseIsZeroEnergy` -- a triangle at its rest
  configuration produces zero constraint values, so iterating
  doesn't move particles.
* :class:`TestAreaPreservation` -- an inflated triangle relaxes
  back to its rest area under the lambda row alone (mu compliance
  set high enough that mu's effect is negligible).
* :class:`TestShearDecay` -- a sheared but area-preserving
  triangle relaxes back to a rotated copy of the rest shape under
  the mu row alone (lambda compliance set high enough that
  lambda's effect is negligible).

CUDA-only by Newton convention.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.body_or_particle import BodyOrParticleStore
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    CLOTH_TRIANGLE_DWORDS,
    cloth_triangle_iterate_at,
    cloth_triangle_prepare_for_iteration_at,
    cloth_triangle_set_alpha_lambda,
    cloth_triangle_set_alpha_mu,
    cloth_triangle_set_body1,
    cloth_triangle_set_body2,
    cloth_triangle_set_body3,
    cloth_triangle_set_inv_rest,
    cloth_triangle_set_rest_area,
    cloth_triangle_set_type,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.particle import particle_container_zeros


def _build_unit_right_triangle():
    """Three particle rest positions for an axis-aligned right
    triangle. Rest area = 0.5; ``inv_rest`` follows from the rest
    edges in the triangle's tangent frame."""
    rest_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rest_area = 0.5
    # Tangent frame: t1 = +X, t2 = +Y. Rest edges in that frame:
    #   eAB_2D = (1, 0), eAC_2D = (0, 1)
    # rest matrix D_m = [eAB_2D | eAC_2D] = [[1, 0], [0, 1]], so
    # inv_rest = inv(D_m) = identity 2x2.
    inv_rest = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return rest_positions, rest_area, inv_rest


def _build_store(particle_positions, device):
    """Wrap a particle-only :class:`BodyOrParticleStore` for the
    given particle positions. Uses a length-1 sentinel
    BodyContainer (anchor at slot 0) so ``num_bodies = 1`` and
    unified indices ``>= 1`` resolve to particle slots."""
    bodies = body_container_zeros(1, device=device)
    n = particle_positions.shape[0]
    particles = particle_container_zeros(n, device=device)
    particles.position.assign(particle_positions)
    particles.inverse_mass.assign(np.ones(n, dtype=np.float32))
    store = BodyOrParticleStore()
    store.bodies = bodies
    store.particles = particles
    store.num_bodies = wp.int32(1)
    return store, particles


def _stamp_triangle_row(constraints_cls, alpha_lambda, alpha_mu):
    """Helper: returns a wp.kernel that stamps one cloth triangle
    row into ``constraints_cls.container`` with the test fixture's
    rest geometry + the requested compliances. Used as a one-shot
    init kernel before the iterate sweeps."""

    @wp.kernel(enable_backward=False)
    def stamp(
        c: ConstraintContainer,
        body_a: wp.int32,
        body_b: wp.int32,
        body_c: wp.int32,
        inv_rest: wp.mat22f,
        rest_area: wp.float32,
        a_lam: wp.float32,
        a_mu: wp.float32,
    ):
        cid = wp.tid()
        cloth_triangle_set_type(c, cid)
        cloth_triangle_set_body1(c, cid, body_a)
        cloth_triangle_set_body2(c, cid, body_b)
        cloth_triangle_set_body3(c, cid, body_c)
        cloth_triangle_set_inv_rest(c, cid, inv_rest)
        cloth_triangle_set_rest_area(c, cid, rest_area)
        cloth_triangle_set_alpha_lambda(c, cid, a_lam)
        cloth_triangle_set_alpha_mu(c, cid, a_mu)

    return stamp


@wp.kernel(enable_backward=False)
def _prepare_kernel(
    c: ConstraintContainer,
    store: BodyOrParticleStore,
    idt: wp.float32,
):
    cid = wp.tid()
    cloth_triangle_prepare_for_iteration_at(c, cid, wp.int32(0), store, idt)


@wp.kernel(enable_backward=False)
def _iterate_kernel(
    c: ConstraintContainer,
    store: BodyOrParticleStore,
):
    cid = wp.tid()
    cloth_triangle_iterate_at(c, cid, wp.int32(0), store)


def _run_iterate_loop(device, container, store, num_iterations, dt):
    idt = 1.0 / dt
    wp.launch(
        kernel=_prepare_kernel,
        dim=1,
        inputs=[container, store, wp.float32(idt)],
        device=device,
    )
    for _ in range(num_iterations):
        wp.launch(
            kernel=_iterate_kernel,
            dim=1,
            inputs=[container, store],
            device=device,
        )


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-triangle math tests require CUDA")
class TestRestPoseIsZeroEnergy(unittest.TestCase):
    """A triangle at rest stays at rest -- the iterate doesn't
    move particles when both constraint rows evaluate to zero."""

    def test_no_motion_at_rest(self) -> None:
        device = wp.get_device()
        rest_positions, rest_area, inv_rest = _build_unit_right_triangle()
        container = constraint_container_zeros(
            num_constraints=1,
            num_dwords=CLOTH_TRIANGLE_DWORDS,
            device=device,
        )
        store, particles = _build_store(rest_positions.copy(), device)
        # Stiff cloth (low compliance).
        stamp = _stamp_triangle_row(None, alpha_lambda=1e-6, alpha_mu=1e-6)
        wp.launch(
            kernel=stamp,
            dim=1,
            inputs=[
                container,
                wp.int32(1),  # body_a (unified index = num_bodies + 0)
                wp.int32(2),
                wp.int32(3),
                wp.mat22f(inv_rest[0, 0], inv_rest[0, 1], inv_rest[1, 0], inv_rest[1, 1]),
                wp.float32(rest_area),
                wp.float32(1.0e-6),
                wp.float32(1.0e-6),
            ],
            device=device,
        )
        _run_iterate_loop(device, container, store, num_iterations=20, dt=1.0 / 60.0)
        positions_after = particles.position.numpy()
        np.testing.assert_allclose(positions_after, rest_positions, atol=1e-5)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-triangle math tests require CUDA")
class TestAreaPreservation(unittest.TestCase):
    """Inflate the rest triangle (multiply each particle's distance
    from centroid by 1.5 -> area = 2.25*rest_area). Run iterate
    sweeps with a stiff lambda row and a soft mu row. Area should
    converge back to the rest area within a few iterations."""

    def test_area_relaxes_to_rest(self) -> None:
        device = wp.get_device()
        rest_positions, rest_area, inv_rest = _build_unit_right_triangle()
        # Inflate by 1.5x about the centroid.
        centroid = rest_positions.mean(axis=0)
        inflated = (rest_positions - centroid) * 1.5 + centroid
        # Sanity: the inflated triangle has 1.5^2 = 2.25x area.
        e1 = inflated[1] - inflated[0]
        e2 = inflated[2] - inflated[0]
        inflated_area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        self.assertAlmostEqual(inflated_area / rest_area, 2.25, places=4)
        container = constraint_container_zeros(
            num_constraints=1,
            num_dwords=CLOTH_TRIANGLE_DWORDS,
            device=device,
        )
        store, particles = _build_store(inflated.astype(np.float32), device)
        # Stiff lambda (alpha_lambda small -> high stiffness), soft mu
        # so the shear row barely participates.
        stamp = _stamp_triangle_row(None, alpha_lambda=1e-9, alpha_mu=1.0)
        wp.launch(
            kernel=stamp,
            dim=1,
            inputs=[
                container,
                wp.int32(1),
                wp.int32(2),
                wp.int32(3),
                wp.mat22f(inv_rest[0, 0], inv_rest[0, 1], inv_rest[1, 0], inv_rest[1, 1]),
                wp.float32(rest_area),
                wp.float32(1.0e-9),
                wp.float32(1.0),
            ],
            device=device,
        )
        _run_iterate_loop(device, container, store, num_iterations=40, dt=1.0 / 60.0)
        positions_after = particles.position.numpy()
        e1f = positions_after[1] - positions_after[0]
        e2f = positions_after[2] - positions_after[0]
        final_area = 0.5 * np.linalg.norm(np.cross(e1f, e2f))
        # Should be very close to the rest area; XPBD with stiff
        # compliance + 40 iterations gets us within a few percent.
        self.assertAlmostEqual(final_area / rest_area, 1.0, delta=0.05)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-triangle math tests require CUDA")
class TestShearDecay(unittest.TestCase):
    """Apply pure shear (skew the rest triangle in-plane) and run
    iterate with stiff mu / soft lambda. The iterate should drive
    the deformation gradient toward a rotation-only state, which
    means the mu energy ``||F - R||_F`` decays over iterations.

    We don't check exact convergence to the rest pose because XPBD
    can leave a translation residual; we check that the mu energy
    monotonically decreases."""

    def test_shear_energy_decays(self) -> None:
        device = wp.get_device()
        rest_positions, rest_area, inv_rest = _build_unit_right_triangle()
        # Apply a pure shear: keep particles A (origin) and B fixed,
        # slide C along +X by 0.5 -- preserves area, introduces shear.
        sheared = rest_positions.copy()
        sheared[2, 0] += 0.5
        e1 = sheared[1] - sheared[0]
        e2 = sheared[2] - sheared[0]
        # Sanity: area unchanged.
        sheared_area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        self.assertAlmostEqual(sheared_area / rest_area, 1.0, places=5)
        container = constraint_container_zeros(
            num_constraints=1,
            num_dwords=CLOTH_TRIANGLE_DWORDS,
            device=device,
        )
        store, particles = _build_store(sheared.astype(np.float32), device)
        # Stiff mu, soft lambda.
        stamp = _stamp_triangle_row(None, alpha_lambda=1.0, alpha_mu=1e-9)
        wp.launch(
            kernel=stamp,
            dim=1,
            inputs=[
                container,
                wp.int32(1),
                wp.int32(2),
                wp.int32(3),
                wp.mat22f(inv_rest[0, 0], inv_rest[0, 1], inv_rest[1, 0], inv_rest[1, 1]),
                wp.float32(rest_area),
                wp.float32(1.0),
                wp.float32(1.0e-9),
            ],
            device=device,
        )

        def mu_energy_now() -> float:
            p = particles.position.numpy()
            ee1 = p[1] - p[0]
            ee2 = p[2] - p[0]
            f0 = inv_rest[0, 0] * ee1 + inv_rest[1, 0] * ee2
            f1 = inv_rest[0, 1] * ee1 + inv_rest[1, 1] * ee2
            m00 = float(np.dot(f0, f0))
            m01 = float(np.dot(f0, f1))
            m11 = float(np.dot(f1, f1))
            tr_m = m00 + m11
            det_m = m00 * m11 - m01 * m01
            sigma_prod = math.sqrt(max(det_m, 0.0))
            sigma_sum = math.sqrt(max(tr_m + 2 * sigma_prod, 0.0))
            return max(tr_m - 2 * sigma_sum + 2.0, 0.0)

        # Run prepare then 30 iterations; sample energy every 5 iters.
        idt = 1.0 / (1.0 / 60.0)
        wp.launch(
            kernel=_prepare_kernel,
            dim=1,
            inputs=[container, store, wp.float32(idt)],
            device=device,
        )
        energies = [mu_energy_now()]
        for it in range(30):
            wp.launch(
                kernel=_iterate_kernel,
                dim=1,
                inputs=[container, store],
                device=device,
            )
            if (it + 1) % 5 == 0:
                energies.append(mu_energy_now())
        # Energy must decrease (allowing a tiny float-32 wobble).
        for i in range(1, len(energies)):
            self.assertLessEqual(
                energies[i],
                energies[i - 1] + 1e-6,
                msg=f"mu energy increased at sample {i}: {energies[i - 1]} -> {energies[i]}",
            )
        # And end up materially smaller than the start (not just
        # noise-level identical).
        self.assertLess(energies[-1], 0.5 * energies[0])


if __name__ == "__main__":
    unittest.main()
