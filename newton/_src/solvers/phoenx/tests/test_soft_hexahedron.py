# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the 8-node soft-hexahedron ARAP constraint.

Three CUDA-only graph-captured tests covering the math primitives:

1. **rest_state_no_impulse**: hex stamped at rest, no gravity -- the
   constraint must produce ~zero impulse so corner positions don't
   drift across PGS sweeps.
2. **rigid_rotation_no_impulse**: rest corners rigidly rotated by an
   arbitrary R -- the polar decomposition must recover R, ``F - R``
   vanishes, and corner positions don't drift after a sweep.
3. **stretch_restoration**: two opposite body-diagonal corners pushed
   outward by 1 cm -- one PGS sweep must pull both endpoints back
   along the displacement axis (sign of the ARAP gradient is right).

All tests run on CUDA, build a captured graph for one step, and
replay it via :func:`wp.capture_launch` -- the same code path the
production solver runs.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float32,
)


def _build_world(
    device,
    *,
    rest_corners: np.ndarray,  # [8, 3]
    inv_mass: np.ndarray,  # [8]
    youngs_modulus: float = 5.0e6,
    poisson_ratio: float = 0.3,
    beta_mu: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    substeps: int = 4,
    solver_iterations: int = 8,
) -> PhoenXWorld:
    """Construct a 1-hex / 8-particle PhoenX world.

    Centralised so every test runs the same construction shape; the
    individual tests just feed it different rest poses / pin patterns /
    materials.
    """
    bodies = body_container_zeros(1, device=device)
    bodies.orientation.assign(
        wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (1, 1)).astype(np.float32), dtype=wp.quatf, device=device)
    )
    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0,
        num_soft_hexahedra=1,
        device=device,
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=8,
        num_soft_hexahedra=1,
        num_worlds=1,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=0,
        rigid_contact_max=0,
        step_layout="single_world",
        mass_splitting=False,
        partitioner_algorithm="greedy",
        device=device,
    )
    world.gravity.assign(np.array([list(gravity)], dtype=np.float32))

    k_mu, _ = soft_tet_lame_from_youngs_poisson(youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio)
    hex_indices = wp.array(np.arange(8, dtype=np.int32).reshape(1, 8), dtype=wp.int32, device=device)
    particle_q = wp.array(rest_corners, dtype=wp.vec3f, device=device)
    particle_qd = wp.zeros(8, dtype=wp.vec3f, device=device)
    particle_inv_mass = wp.array(inv_mass, dtype=wp.float32, device=device)
    hex_materials = wp.array(
        np.array([[k_mu, beta_mu]], dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
    world.populate_soft_hexahedra_from_arrays(
        hex_indices=hex_indices,
        particle_q=particle_q,
        hex_materials=hex_materials,
        particle_qd=particle_qd,
        particle_inv_mass=particle_inv_mass,
    )
    return world


def _capture_step_graph(world: PhoenXWorld, dt: float, device) -> wp.Graph:
    """Warm up (1 un-captured step) then capture a single step graph."""
    world.step(dt)
    with wp.ScopedCapture(device=device) as cap:
        world.step(dt)
    return cap.graph


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-hexahedron tests are CUDA-only (need graph capture).",
)
class TestSoftHexahedron(unittest.TestCase):
    def test_rest_state_no_impulse(self):
        """Hex stamped at rest, zero gravity -> negligible drift."""
        device = wp.get_preferred_device()
        rest_corners = 0.25 * _CORNER_SIGNS  # 0.5 m cube centered at origin
        rest_corners[:, 2] += 1.0
        # All corners free (no pins); zero gravity.
        inv_mass = np.full(8, 1.0, dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest_corners,
            inv_mass=inv_mass,
            beta_mu=0.0,
            gravity=(0.0, 0.0, 0.0),
        )
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        # 30 captured replays. With no gravity and no initial velocity
        # the rest configuration is a fixed point of the constraint;
        # any drift here would indicate the ARAP gradient is
        # mis-derived or polar decomposition is unstable at rest.
        for _ in range(30):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        positions = world.particles.position.numpy()
        max_drift = float(np.max(np.linalg.norm(positions - rest_corners, axis=1)))
        # Float32 polar decomposition + XPBD round-trip noise floor.
        # 1e-5 m absolute = 10 microns; tight enough to catch a real
        # gradient mistake, loose enough to absorb fp32 quirks.
        self.assertLess(max_drift, 1e-5, f"rest-state drift {max_drift:.2e} m > 1e-5")
        self.assertTrue(np.isfinite(positions).all())

    def test_rigid_rotation_no_impulse(self):
        """Rigidly rotate rest corners by R -> APD recovers R, no drift."""
        device = wp.get_preferred_device()
        rest = 0.25 * _CORNER_SIGNS
        # Pick an arbitrary rotation: 30 deg about (1, 1, 1) normalised.
        axis = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)
        angle = np.deg2rad(30.0)
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float32,
        )
        R = np.eye(3, dtype=np.float32) + s * K + (1.0 - c) * (K @ K)
        # ``populate_soft_hexahedra_from_arrays`` reads particle_q for
        # BOTH the rest-pose J build AND the current positions. To test
        # that a uniform rotation produces zero impulse we need the
        # CURRENT positions to be R * rest while the stored inv_rest
        # was built from rest. Easiest path: build with rest, then
        # overwrite particle.position before stepping.
        inv_mass = np.full(8, 1.0, dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            beta_mu=0.0,
            gravity=(0.0, 0.0, 0.0),
        )
        rotated = rest @ R.T  # rotate each rest corner by R
        # Overwrite particles' current positions (constraint's inv_rest
        # was built from the unrotated rest pose).
        world.particles.position.assign(wp.array(rotated, dtype=wp.vec3f, device=device))
        # Also seed position_prev_substep so the Macklin damping anchor
        # doesn't see a spurious initial delta.
        world.particles.position_prev_substep.assign(wp.array(rotated, dtype=wp.vec3f, device=device))

        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        for _ in range(30):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        positions = world.particles.position.numpy()
        # A pure rotation should leave each corner at its rotated rest
        # position. Tolerance is looser than the rest-state test
        # because APD Newton iterates from identity warm start on the
        # first prepare and only converges to ~6 digits in 6 iters.
        max_drift = float(np.max(np.linalg.norm(positions - rotated, axis=1)))
        self.assertLess(max_drift, 1e-4, f"rigid-rotation drift {max_drift:.2e} m > 1e-4")
        self.assertTrue(np.isfinite(positions).all())

    def test_stretch_restoration(self):
        """Push two opposite corners apart -> constraint pulls them back.

        Hangs the cube with all corners free in zero gravity, displaces
        opposite body-diagonal corners (0 and 7) outward by +/- delta,
        runs one captured step, and verifies the constraint has pulled
        them back toward rest (component of (x_new - x_perturbed)
        along the displacement is negative on both ends).

        This is the canonical "ARAP gradient direction is right" check:
        no integration over time, no equilibrium hunting -- just one
        sweep on a known-bad pose.
        """
        device = wp.get_preferred_device()
        cube_size = 0.1
        rest = 0.5 * cube_size * _CORNER_SIGNS
        # All free, equal mass.
        inv_mass = np.full(8, 100.0, dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            youngs_modulus=2.6e5,
            poisson_ratio=0.3,
            beta_mu=0.0,
            gravity=(0.0, 0.0, 0.0),
            substeps=1,
            solver_iterations=1,
        )

        # Perturb: stretch the body diagonal from corner 0 -> corner 7
        # by 1 cm at each end (so 2 cm total elongation).
        diag = rest[7] - rest[0]
        diag_unit = diag / np.linalg.norm(diag)
        delta = 0.01
        perturbed = rest.copy()
        perturbed[0] -= delta * diag_unit  # corner 0 outward
        perturbed[7] += delta * diag_unit  # corner 7 outward
        world.particles.position.assign(wp.array(perturbed, dtype=wp.vec3f, device=device))
        # Seed prev-substep so Macklin damping anchor is zero.
        world.particles.position_prev_substep.assign(wp.array(perturbed, dtype=wp.vec3f, device=device))

        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        after = world.particles.position.numpy()
        self.assertTrue(np.isfinite(after).all())

        # Both displaced corners must move back toward rest along
        # the diagonal: the projection of (after - perturbed) onto
        # the outward stretch direction is negative for each end.
        move_0_along = float(np.dot(after[0] - perturbed[0], -diag_unit))
        move_7_along = float(np.dot(after[7] - perturbed[7], +diag_unit))
        self.assertLess(
            move_0_along,
            0.0,
            f"corner 0 didn't move back toward rest along the stretch axis "
            f"(projection={move_0_along:.3e}); gradient direction wrong?",
        )
        self.assertLess(
            move_7_along,
            0.0,
            f"corner 7 didn't move back toward rest along the stretch axis "
            f"(projection={move_7_along:.3e}); gradient direction wrong?",
        )


if __name__ == "__main__":
    unittest.main()
