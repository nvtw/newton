# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Algebraic identity tests for the unified contact-endpoint helpers
(``contact_endpoint_velocity_at_point`` /
``contact_endpoint_inv_mass_along`` /
``contact_endpoint_apply_impulse``).

These helpers are the per-side primitives the cloth-aware contact
iterate (phase 5) is built on. The tests below exercise each helper
with hand-computed expected values:

#. **Rigid case** matches Jitter2 ``RigidTetContactManifold``'s rigid-
   side math (``invMass1 + (r x d) . invInertia . (r x d)`` for the
   row's mass term, ``v + omega x r`` for velocity-at-point) and the
   existing :func:`effective_mass_scalar`'s denominator.
#. **Cloth case** matches the barycentric formulas:
   ``bary . v`` for velocity, ``sum_i bary_i^2 * inv_m_i`` for mass,
   and ``v_i += bary_i * J * inv_m_i`` for impulse application.
#. **Cross-helper consistency**: ``inv_mass_along`` summed over both
   sides equals the existing ``effective_mass_scalar`` denominator
   in the rigid-rigid degenerate case.

Skipped on CPU (Warp kernels need a Warp-supported device).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.cloth_collision import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
    SHAPE_ENDPOINT_KIND_RIGID,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    contact_endpoint_apply_impulse,
    contact_endpoint_inv_mass_along,
    contact_endpoint_velocity_at_point,
)
from newton._src.solvers.phoenx.helpers.math_helpers import effective_mass_scalar
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer, copy_state_container_zeros
from newton._src.solvers.phoenx.particle import ParticleContainer, particle_container_zeros


@wp.kernel(enable_backward=False)
def _eval_velocity_at_point(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    contact_point: wp.vec3f,
    out: wp.array[wp.vec3f],
):
    out[0] = contact_endpoint_velocity_at_point(
        kind, nodes, bary, bodies, particles, copy_state, num_bodies, parallel_id, contact_point
    )


@wp.kernel(enable_backward=False)
def _eval_inv_mass_along(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    contact_point: wp.vec3f,
    direction: wp.vec3f,
    out: wp.array[wp.float32],
):
    out[0] = contact_endpoint_inv_mass_along(
        kind,
        nodes,
        bary,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        contact_point,
        direction,
    )


@wp.kernel(enable_backward=False)
def _eval_apply_impulse(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    contact_point: wp.vec3f,
    impulse: wp.vec3f,
):
    contact_endpoint_apply_impulse(
        kind,
        nodes,
        bary,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        contact_point,
        impulse,
    )


@wp.kernel(enable_backward=False)
def _eval_effective_mass_scalar_pair(
    direction: wp.vec3f,
    r1: wp.vec3f,
    r2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    out: wp.array[wp.float32],
):
    """Returns ``1 / effective_mass_scalar`` (the raw denominator)."""
    em = effective_mass_scalar(direction, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
    if em > wp.float32(0.0):
        out[0] = wp.float32(1.0) / em
    else:
        out[0] = wp.float32(0.0)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX contact-endpoint helpers run on CUDA only.",
)
class TestContactEndpointHelpers(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_preferred_device()
        # Two rigid bodies + one cloth triangle, deterministic state.
        self.num_bodies = 2
        self.num_particles = 3
        self.bodies = body_container_zeros(self.num_bodies, device=self.device)
        self.particles = particle_container_zeros(self.num_particles, device=self.device)

        # Body 0: at origin, mass 2.0, isotropic inertia I = 0.5
        # Body 1: at (1, 0, 0), mass 4.0, isotropic inertia I = 0.25
        self.body_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        self.body_vel = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        self.body_omega = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=np.float32)
        body_inv_mass = np.array([1.0 / 2.0, 1.0 / 4.0], dtype=np.float32)
        body_inv_inertia = np.zeros((2, 3, 3), dtype=np.float32)
        body_inv_inertia[0] = np.eye(3) * (1.0 / 0.5)
        body_inv_inertia[1] = np.eye(3) * (1.0 / 0.25)
        self.body_inv_mass = body_inv_mass
        self.body_inv_inertia = body_inv_inertia
        self.bodies.position.assign(self.body_pos)
        self.bodies.velocity.assign(self.body_vel)
        self.bodies.angular_velocity.assign(self.body_omega)
        self.bodies.inverse_mass.assign(body_inv_mass)
        self.bodies.inverse_inertia_world.assign(body_inv_inertia.reshape(2, 9))

        # Cloth particles 0/1/2: triangle nodes with distinct masses
        # and velocities; these are the cloth side of a contact.
        self.particle_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        self.particle_vel = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
        self.particle_inv_mass = np.array([1.0 / 0.1, 1.0 / 0.2, 1.0 / 0.3], dtype=np.float32)
        self.particles.position.assign(self.particle_pos)
        self.particles.velocity.assign(self.particle_vel)
        self.particles.inverse_mass.assign(self.particle_inv_mass)

        # Sentinel ``CopyStateContainer`` -- mass splitting disabled, so
        # every endpoint helper falls through to direct bodies / particles
        # reads/writes (``inv_factor == 1``, ``slot == -1``).
        self.copy_state = copy_state_container_zeros(capacity=1, num_nodes=1, device=self.device)

    @staticmethod
    def _nodes4(nodes):
        """Pad a 1/2/3-element node tuple to a ``wp.vec4i`` with ``-1`` sentinels."""
        padded = list(nodes) + [-1] * (4 - len(nodes))
        return wp.vec4i(*padded[:4])

    def _eval_v_at_p(self, kind, nodes, bary, contact_point):
        out = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        wp.launch(
            _eval_velocity_at_point,
            dim=1,
            inputs=[
                wp.int32(kind),
                self._nodes4(nodes),
                wp.vec3f(*bary),
                self.bodies,
                self.particles,
                self.copy_state,
                wp.int32(self.num_bodies),
                wp.int32(0),
                wp.vec3f(*contact_point),
            ],
            outputs=[out],
            device=self.device,
        )
        return out.numpy()[0]

    def _eval_inv_mass(self, kind, nodes, bary, contact_point, direction):
        out = wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(
            _eval_inv_mass_along,
            dim=1,
            inputs=[
                wp.int32(kind),
                self._nodes4(nodes),
                wp.vec3f(*bary),
                self.bodies,
                self.particles,
                self.copy_state,
                wp.int32(self.num_bodies),
                wp.int32(0),
                wp.vec3f(*contact_point),
                wp.vec3f(*direction),
            ],
            outputs=[out],
            device=self.device,
        )
        return float(out.numpy()[0])

    def test_rigid_velocity_at_point(self):
        """v_at_p = v + omega x r for a rigid body."""
        # contact point at body 0's origin
        p = (0.0, 0.5, 0.0)
        b = 0
        r = np.asarray(p, dtype=np.float32) - self.body_pos[b]
        expected = self.body_vel[b] + np.cross(self.body_omega[b], r)
        got = self._eval_v_at_p(SHAPE_ENDPOINT_KIND_RIGID, (b, -1, -1), (0.0, 0.0, 0.0), p)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_rigid_static_returns_zero_velocity(self):
        """Static-anchor (nodes[0] < 0) -> zero velocity."""
        got = self._eval_v_at_p(SHAPE_ENDPOINT_KIND_RIGID, (-1, -1, -1), (0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
        np.testing.assert_allclose(got, [0.0, 0.0, 0.0], atol=0.0)

    def test_cloth_velocity_at_point(self):
        """v_at_p = bary . v for a cloth-triangle endpoint."""
        bary = (0.2, 0.3, 0.5)
        # particles in unified-index space: num_bodies + p_id
        nodes = (self.num_bodies + 0, self.num_bodies + 1, self.num_bodies + 2)
        # Contact point unused for cloth (math is purely barycentric)
        expected = bary[0] * self.particle_vel[0] + bary[1] * self.particle_vel[1] + bary[2] * self.particle_vel[2]
        got = self._eval_v_at_p(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE, nodes, bary, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_rigid_inv_mass_along(self):
        """Side's mass term = inv_mass + (r x d) . invI . (r x d)."""
        b = 1
        p = (1.0, 0.5, 0.2)
        d = (0.0, 0.0, 1.0)
        r = np.asarray(p, dtype=np.float32) - self.body_pos[b]
        rc = np.cross(r, d)
        invI = self.body_inv_inertia[b]
        expected = float(self.body_inv_mass[b] + rc @ invI @ rc)
        got = self._eval_inv_mass(SHAPE_ENDPOINT_KIND_RIGID, (b, -1, -1), (0.0, 0.0, 0.0), p, d)
        self.assertAlmostEqual(got, expected, places=5)

    def test_rigid_static_inv_mass_zero(self):
        got = self._eval_inv_mass(
            SHAPE_ENDPOINT_KIND_RIGID, (-1, -1, -1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        self.assertAlmostEqual(got, 0.0, places=6)

    def test_cloth_inv_mass_along(self):
        """Side's mass term = sum_i bary_i^2 * inv_mass_i."""
        bary = (0.2, 0.3, 0.5)
        nodes = (self.num_bodies + 0, self.num_bodies + 1, self.num_bodies + 2)
        expected = float(
            bary[0] ** 2 * self.particle_inv_mass[0]
            + bary[1] ** 2 * self.particle_inv_mass[1]
            + bary[2] ** 2 * self.particle_inv_mass[2]
        )
        got = self._eval_inv_mass(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE, nodes, bary, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(got, expected, places=5)

    def test_rigid_rigid_pair_matches_effective_mass_scalar(self):
        """Critical cross-helper identity:

        ``side0_inv_mass_along + side1_inv_mass_along ==
        1 / effective_mass_scalar(d, r1, r2, ...)``

        This is what makes the rigid-rigid path through the new
        helpers byte-equivalent to the existing PhoenX rigid contact
        path -- swapping in the unified iterate (phase 5) for the
        rigid-rigid case must produce identical impulses.
        """
        d = np.array([0.7, 0.3, 0.65], dtype=np.float32)
        d /= np.linalg.norm(d)
        # Contact point somewhere off both bodies' COMs:
        p = (0.4, 0.5, 0.6)
        # Side 0 = body 0, side 1 = body 1.
        side0 = self._eval_inv_mass(SHAPE_ENDPOINT_KIND_RIGID, (0, -1, -1), (0.0, 0.0, 0.0), p, tuple(d))
        side1 = self._eval_inv_mass(SHAPE_ENDPOINT_KIND_RIGID, (1, -1, -1), (0.0, 0.0, 0.0), p, tuple(d))
        sum_helper = side0 + side1

        out = wp.zeros(1, dtype=wp.float32, device=self.device)
        r1 = np.asarray(p, dtype=np.float32) - self.body_pos[0]
        r2 = np.asarray(p, dtype=np.float32) - self.body_pos[1]
        wp.launch(
            _eval_effective_mass_scalar_pair,
            dim=1,
            inputs=[
                wp.vec3f(*d.tolist()),
                wp.vec3f(*r1.tolist()),
                wp.vec3f(*r2.tolist()),
                wp.float32(self.body_inv_mass[0]),
                wp.float32(self.body_inv_mass[1]),
                wp.mat33f(*self.body_inv_inertia[0].flatten().tolist()),
                wp.mat33f(*self.body_inv_inertia[1].flatten().tolist()),
            ],
            outputs=[out],
            device=self.device,
        )
        denom = float(out.numpy()[0])
        self.assertAlmostEqual(
            sum_helper, denom, places=5, msg=f"sum_helper={sum_helper} effective_mass_scalar denom={denom}"
        )

    def test_rigid_apply_impulse(self):
        """v_after - v_before == J / m ; w_after - w_before == invI . (r x J)."""
        b = 1
        p = np.array([0.4, 0.6, -0.2], dtype=np.float32)
        J = np.array([0.3, -0.5, 0.4], dtype=np.float32)
        v_before = self.bodies.velocity.numpy()[b].copy()
        w_before = self.bodies.angular_velocity.numpy()[b].copy()
        wp.launch(
            _eval_apply_impulse,
            dim=1,
            inputs=[
                wp.int32(SHAPE_ENDPOINT_KIND_RIGID),
                self._nodes4((b,)),
                wp.vec3f(0.0, 0.0, 0.0),
                self.bodies,
                self.particles,
                self.copy_state,
                wp.int32(self.num_bodies),
                wp.int32(0),
                wp.vec3f(*p.tolist()),
                wp.vec3f(*J.tolist()),
            ],
            device=self.device,
        )
        v_after = self.bodies.velocity.numpy()[b]
        w_after = self.bodies.angular_velocity.numpy()[b]
        expected_dv = J * self.body_inv_mass[b]
        r = p - self.body_pos[b]
        expected_dw = self.body_inv_inertia[b] @ np.cross(r, J)
        np.testing.assert_allclose(v_after - v_before, expected_dv, atol=1e-6)
        np.testing.assert_allclose(w_after - w_before, expected_dw, atol=1e-6)

    def test_cloth_apply_impulse(self):
        """v_i_after - v_i_before == bary_i * J * inv_mass_i."""
        bary = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        nodes = (self.num_bodies + 0, self.num_bodies + 1, self.num_bodies + 2)
        J = np.array([0.7, -0.4, 0.2], dtype=np.float32)
        v_before = self.particles.velocity.numpy().copy()
        wp.launch(
            _eval_apply_impulse,
            dim=1,
            inputs=[
                wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE),
                self._nodes4(nodes),
                wp.vec3f(*bary.tolist()),
                self.bodies,
                self.particles,
                self.copy_state,
                wp.int32(self.num_bodies),
                wp.int32(0),
                wp.vec3f(0.0, 0.0, 0.0),  # contact_point unused for cloth
                wp.vec3f(*J.tolist()),
            ],
            device=self.device,
        )
        v_after = self.particles.velocity.numpy()
        for i in range(3):
            expected_dv = bary[i] * J * self.particle_inv_mass[i]
            np.testing.assert_allclose(v_after[i] - v_before[i], expected_dv, atol=1e-6, err_msg=f"node {i}")


if __name__ == "__main__":
    unittest.main()
