# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the 8-node soft-hexahedron block Neo-Hookean constraint.

Three CUDA-only graph-captured tests, mirroring
:mod:`test_soft_tet_neohookean`'s pattern:

1. **rest_pose_remains_bounded**: at the rest pose with zero gravity
   the stable Neo-Hookean energy gradient vanishes (mu*I + lambda*(1
   - gamma)*I = 0 by construction), so the cube must remain spatially
   bounded. Allows a small drift to the analytic ``F = gamma^(1/3) I``
   equilibrium for finite ``mu / lambda``.

2. **free_fall_integrates_gravity**: a soft cube in vacuum under
   gravity must fall ballistically without exploding; the block
   constraint is the only thing active.

3. **top_face_pin_hang_stable**: pin the 4 top-face corners, apply
   gravity, verify the bottom face hangs in a bounded elastic
   configuration (edge ratios stay within +/- 50% of rest, no NaN)
   over a long-ish horizon.

All tests run on CUDA and replay through ``wp.capture_launch`` --
the same code path the production solver uses.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
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
    k_mu: float = 1.0e3,
    k_lambda: float = 1.0e5,  # high lambda / low mu -> gamma ~ 1 -> small rest drift
    beta_h: float = 0.0,
    beta_d: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    substeps: int = 4,
    solver_iterations: int = 8,
) -> PhoenXWorld:
    """Construct a 1-hex / 8-particle PhoenX world."""
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

    hex_indices = wp.array(np.arange(8, dtype=np.int32).reshape(1, 8), dtype=wp.int32, device=device)
    particle_q = wp.array(rest_corners, dtype=wp.vec3f, device=device)
    particle_qd = wp.zeros(8, dtype=wp.vec3f, device=device)
    particle_inv_mass = wp.array(inv_mass, dtype=wp.float32, device=device)
    hex_materials = wp.array(
        np.array([[k_mu, k_lambda, beta_h, beta_d]], dtype=np.float32),
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
    def test_rest_pose_remains_bounded(self):
        """Rest pose under zero gravity -- bounded drift to gamma^(1/3) I."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS  # 0.1 m cube
        rest[:, 2] += 0.5  # lift off origin (irrelevant; just hygiene)
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        # High lambda / low mu so gamma ~ 1 and the natural Neo-Hookean
        # inflation is sub-percent.
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e3,
            k_lambda=1.0e5,
            gravity=(0.0, 0.0, 0.0),
        )
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        p_initial = world.particles.position.numpy().copy()
        for _ in range(20):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        # Bound the drift to a fraction of the cube edge. The stable
        # Neo-Hookean rest equilibrium ``F = gamma^(1/3) I`` produces
        # ~0.3% expansion at lambda/mu = 100; 0.05 * cube_size leaves
        # generous headroom for the explicit-Euler equilibration mode.
        drift = float(np.linalg.norm(positions - p_initial, axis=1).max())
        self.assertLess(
            drift,
            0.05 * 0.1,
            f"rest-pose drift {drift:.4f} m exceeds bound",
        )

    def test_free_fall_integrates_gravity(self):
        """Cube in vacuum under gravity falls ballistically (~exact)."""
        device = wp.get_preferred_device()
        rest = 0.05 * _CORNER_SIGNS
        rest[:, 2] += 1.0
        inv_mass = np.full(8, 1.0 / (200.0 * 0.001 / 8.0), dtype=np.float32)
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e3,
            k_lambda=1.0e5,
            gravity=(0.0, 0.0, -9.81),
        )
        initial_mean_z = float(world.particles.position.numpy()[:, 2].mean())
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        T = 0.25
        n_frames = int(T / dt)
        for _ in range(n_frames):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        final_mean_z = float(positions[:, 2].mean())
        elapsed = (n_frames + 1) * dt  # +1 for the warm-up step inside _capture_step_graph
        expected_drop = 0.5 * 9.81 * elapsed * elapsed
        actual_drop = initial_mean_z - final_mean_z
        # Looser than ARAP: the cube also breathes during self-equilibration.
        # 15% tolerance covers the explicit-Euler bias + equilibration.
        self.assertAlmostEqual(actual_drop, expected_drop, delta=0.15 * expected_drop)

    def test_top_face_pin_hang_stable(self):
        """Top-face pin + gravity -> bottom face hangs without exploding."""
        device = wp.get_preferred_device()
        cube_size = 0.1
        rest = 0.5 * cube_size * _CORNER_SIGNS
        rest[:, 2] += 0.5
        # Pin top face (corners 4..7).
        density = 200.0
        cube_volume = cube_size**3
        corner_mass = density * cube_volume / 8.0
        inv_mass = np.full(8, 1.0 / corner_mass, dtype=np.float32)
        inv_mass[4:8] = 0.0
        world = _build_world(
            device,
            rest_corners=rest,
            inv_mass=inv_mass,
            k_mu=1.0e4,
            k_lambda=1.0e6,
            beta_h=5.0,
            beta_d=5.0,
            gravity=(0.0, 0.0, -9.81),
            substeps=5,
            solver_iterations=8,
        )
        dt = 1.0 / 60.0
        graph = _capture_step_graph(world, dt, device)
        for _ in range(120):  # 2 s at 60 Hz
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        positions = world.particles.position.numpy()
        self.assertTrue(np.isfinite(positions).all())

        # Pinned top corners haven't drifted (inv_mass=0 => no update).
        pinned_drift = float(np.linalg.norm(positions[4:8] - rest[4:8], axis=1).max())
        self.assertLess(pinned_drift, 1e-4, f"pinned-corner drift {pinned_drift:.2e}")

        # Hex didn't explode: every edge stays within +/- 50% of rest.
        edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=np.int32,
        )
        rest_lens = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
        now_lens = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
        ratios = now_lens / rest_lens
        self.assertTrue(
            np.all((ratios > 0.5) & (ratios < 1.5)),
            f"hex edges blew up: ratios min={ratios.min():.3f} max={ratios.max():.3f}",
        )

        # Bottom face hangs below the pinned top.
        free_z = float(positions[:4, 2].mean())
        pin_z = float(positions[4:8, 2].mean())
        self.assertLess(free_z, pin_z, "bottom face didn't hang below pinned top")


if __name__ == "__main__":
    unittest.main()
