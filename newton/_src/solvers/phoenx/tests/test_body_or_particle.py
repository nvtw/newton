# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified body-or-particle indexing scheme.

Three things to verify:

* :class:`TestBodyOrParticleAccessors` -- the
  :func:`get_position` / :func:`get_velocity` / :func:`get_orientation`
  / :func:`get_angular_velocity` / :func:`get_inverse_mass`
  accessors return the right SoA field on the body branch and the
  right (possibly degenerate) value on the particle branch.
  Setters route to the right SoA too.
* :class:`TestPhoenXWorldNumParticles` -- constructing
  :class:`PhoenXWorld` with ``num_particles=0`` (default) leaves
  ``world.particles is None`` and the world is otherwise unaffected;
  ``num_particles>0`` allocates a :class:`ParticleContainer` and the
  ``body_or_particle`` property returns a valid store.
* :class:`TestStoreSentinel` -- the ``body_or_particle`` property
  works on a rigid-only world too (no particles allocated; the
  store wraps a length-1 sentinel particle container that the
  threshold compare guarantees is never read).

All CUDA-only by Newton convention.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.body_or_particle import (
    BodyOrParticleStore,
    get_angular_velocity,
    get_inverse_mass,
    get_orientation,
    get_position,
    get_velocity,
    is_particle,
    set_position,
    set_velocity,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.particle import (
    particle_container_zeros,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX body-or-particle tests require CUDA")
class TestBodyOrParticleAccessors(unittest.TestCase):
    """The five getters + two setters branch on the threshold and
    return / write the right SoA field for both index regimes."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        self.num_bodies = 3
        self.num_particles = 2
        self.bodies = body_container_zeros(self.num_bodies, device=self.device)
        # Set distinct values per body for unambiguous assertions.
        self.bodies.position.assign(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32))
        self.bodies.orientation.assign(
            # Distinct unit quaternions: identity, +90 deg about X,
            # +90 deg about Z. The accessor must distinguish bodies
            # from each other (not just from particles).
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.7071068, 0.0, 0.0, 0.7071068],
                    [0.0, 0.0, 0.7071068, 0.7071068],
                ],
                dtype=np.float32,
            )
        )
        self.bodies.velocity.assign(np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32))
        self.bodies.angular_velocity.assign(
            np.array([[0.5, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.7]], dtype=np.float32)
        )
        self.bodies.inverse_mass.assign(np.array([1.0, 0.5, 0.25], dtype=np.float32))
        self.particles = particle_container_zeros(self.num_particles, device=self.device)
        self.particles.position.assign(np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float32))
        self.particles.velocity.assign(np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32))
        self.particles.inverse_mass.assign(np.array([2.0, 4.0], dtype=np.float32))
        self.store = BodyOrParticleStore()
        self.store.bodies = self.bodies
        self.store.particles = self.particles
        self.store.num_bodies = wp.int32(self.num_bodies)

    def _probe(self, kernel, n: int):
        """Launch ``kernel`` at ``dim=n`` against the test store; the
        kernel writes one output per unified index 0..n-1."""
        out_position = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        out_orientation = wp.zeros(n, dtype=wp.quatf, device=self.device)
        out_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        out_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        out_inverse_mass = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_is_particle = wp.zeros(n, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=kernel,
            dim=n,
            inputs=[
                self.store,
                out_position,
                out_orientation,
                out_velocity,
                out_angular_velocity,
                out_inverse_mass,
                out_is_particle,
            ],
            device=self.device,
        )
        return (
            out_position.numpy(),
            out_orientation.numpy(),
            out_velocity.numpy(),
            out_angular_velocity.numpy(),
            out_inverse_mass.numpy(),
            out_is_particle.numpy(),
        )

    def test_body_branch(self) -> None:
        """For unified indices ``[0, num_bodies)`` the accessors must
        return the corresponding body's SoA fields."""
        n = self.num_bodies + self.num_particles
        pos, orient, vel, ang_vel, inv_mass, is_p = self._probe(_probe_kernel, n)
        # Body branch: indices 0..2.
        np.testing.assert_allclose(pos[: self.num_bodies], self.bodies.position.numpy(), atol=1e-6)
        np.testing.assert_allclose(orient[: self.num_bodies], self.bodies.orientation.numpy(), atol=1e-6)
        np.testing.assert_allclose(vel[: self.num_bodies], self.bodies.velocity.numpy(), atol=1e-6)
        np.testing.assert_allclose(ang_vel[: self.num_bodies], self.bodies.angular_velocity.numpy(), atol=1e-6)
        np.testing.assert_allclose(inv_mass[: self.num_bodies], self.bodies.inverse_mass.numpy(), atol=1e-6)
        np.testing.assert_array_equal(is_p[: self.num_bodies], np.zeros(self.num_bodies, dtype=np.int32))

    def test_particle_branch(self) -> None:
        """For unified indices ``[num_bodies, num_bodies + num_particles)``
        the accessors must return the corresponding particle's SoA
        fields, with degenerate identity orientation + zero angular
        velocity."""
        n = self.num_bodies + self.num_particles
        pos, orient, vel, ang_vel, inv_mass, is_p = self._probe(_probe_kernel, n)
        # Particle branch: indices 3..4 (num_bodies..num_bodies+num_particles).
        np.testing.assert_allclose(pos[self.num_bodies :], self.particles.position.numpy(), atol=1e-6)
        np.testing.assert_allclose(vel[self.num_bodies :], self.particles.velocity.numpy(), atol=1e-6)
        np.testing.assert_allclose(inv_mass[self.num_bodies :], self.particles.inverse_mass.numpy(), atol=1e-6)
        # Degenerate fields for particles.
        identity_q = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.num_particles, 1))
        zero_v = np.zeros((self.num_particles, 3), dtype=np.float32)
        np.testing.assert_allclose(orient[self.num_bodies :], identity_q, atol=1e-6)
        np.testing.assert_allclose(ang_vel[self.num_bodies :], zero_v, atol=1e-6)
        np.testing.assert_array_equal(is_p[self.num_bodies :], np.ones(self.num_particles, dtype=np.int32))

    def test_setters_route_to_right_soa(self) -> None:
        """``set_position`` and ``set_velocity`` write into the right
        underlying container based on the unified index threshold."""
        n = self.num_bodies + self.num_particles
        new_positions = np.array(
            [[100.0, 0.0, 0.0], [200.0, 0.0, 0.0], [300.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1100.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        new_velocities = np.array(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [110.0, 0.0, 0.0], [120.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        wp.launch(
            kernel=_setter_kernel,
            dim=n,
            inputs=[
                self.store,
                wp.array(new_positions, dtype=wp.vec3f, device=self.device),
                wp.array(new_velocities, dtype=wp.vec3f, device=self.device),
            ],
            device=self.device,
        )
        np.testing.assert_allclose(self.bodies.position.numpy(), new_positions[: self.num_bodies], atol=1e-6)
        np.testing.assert_allclose(self.particles.position.numpy(), new_positions[self.num_bodies :], atol=1e-6)
        np.testing.assert_allclose(self.bodies.velocity.numpy(), new_velocities[: self.num_bodies], atol=1e-6)
        np.testing.assert_allclose(self.particles.velocity.numpy(), new_velocities[self.num_bodies :], atol=1e-6)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX body-or-particle tests require CUDA")
class TestPhoenXWorldNumParticles(unittest.TestCase):
    """``PhoenXWorld(num_particles=0)`` is bit-for-bit unchanged from
    today; ``num_particles>0`` allocates a ParticleContainer and the
    ``body_or_particle`` property returns a populated store."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def _make_world(self, num_particles: int) -> PhoenXWorld:
        bodies = body_container_zeros(2, device=self.device)
        constraints = constraint_container_zeros(num_constraints=1, num_dwords=7, device=self.device)
        return PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=num_particles,
            num_worlds=1,
            step_layout="single_world",
            device=self.device,
        )

    def test_default_num_particles_zero_no_container(self) -> None:
        """Existing rigid-only callers don't pass ``num_particles`` and
        get the default ``0``; ``self.particles`` stays ``None``."""
        world = self._make_world(num_particles=0)
        self.assertEqual(world.num_particles, 0)
        self.assertIsNone(world.particles)

    def test_num_particles_positive_allocates_container(self) -> None:
        world = self._make_world(num_particles=4)
        self.assertEqual(world.num_particles, 4)
        self.assertIsNotNone(world.particles)
        self.assertEqual(world.particles.position.shape[0], 4)
        self.assertEqual(world.particles.velocity.shape[0], 4)
        self.assertEqual(world.particles.inverse_mass.shape[0], 4)

    def test_negative_num_particles_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._make_world(num_particles=-1)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX body-or-particle tests require CUDA")
class TestStoreSentinel(unittest.TestCase):
    """The ``body_or_particle`` property works on a rigid-only world
    too: no particles allocated, but the property returns a store
    with a length-1 sentinel particle container whose only purpose
    is keeping the wp.struct field valid."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_rigid_only_world_yields_valid_store(self) -> None:
        bodies = body_container_zeros(2, device=self.device)
        constraints = constraint_container_zeros(num_constraints=1, num_dwords=7, device=self.device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=0,
            num_worlds=1,
            step_layout="single_world",
            device=self.device,
        )
        store = world.body_or_particle
        # ``wp.struct`` types aren't Python classes, so we can't
        # use ``isinstance``; check the structural fields instead.
        self.assertTrue(hasattr(store, "bodies"))
        self.assertTrue(hasattr(store, "particles"))
        self.assertTrue(hasattr(store, "num_bodies"))
        self.assertIsNone(world.particles)  # still None -- sentinel is internal
        # The store's num_bodies field gates the threshold; for a
        # rigid-only world it equals world.num_bodies.
        self.assertEqual(int(store.num_bodies), world.num_bodies)
        # The accessor is cached; second access returns the same object.
        self.assertIs(store, world.body_or_particle)


# ---------------------------------------------------------------------------
# Local probe / setter kernels (only used by the tests above).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _probe_kernel(
    store: BodyOrParticleStore,
    out_position: wp.array[wp.vec3f],
    out_orientation: wp.array[wp.quatf],
    out_velocity: wp.array[wp.vec3f],
    out_angular_velocity: wp.array[wp.vec3f],
    out_inverse_mass: wp.array[wp.float32],
    out_is_particle: wp.array[wp.int32],
):
    i = wp.tid()
    out_position[i] = get_position(store, i)
    out_orientation[i] = get_orientation(store, i)
    out_velocity[i] = get_velocity(store, i)
    out_angular_velocity[i] = get_angular_velocity(store, i)
    out_inverse_mass[i] = get_inverse_mass(store, i)
    if is_particle(store, i):
        out_is_particle[i] = wp.int32(1)
    else:
        out_is_particle[i] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _setter_kernel(
    store: BodyOrParticleStore,
    new_positions: wp.array[wp.vec3f],
    new_velocities: wp.array[wp.vec3f],
):
    i = wp.tid()
    set_position(store, i, new_positions[i])
    set_velocity(store, i, new_velocities[i])


if __name__ == "__main__":
    unittest.main()
