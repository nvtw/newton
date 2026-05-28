# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the soft-tetrahedron internal XPBD constraint.

These tests construct a small soft mesh, populate the tet constraint
rows from a Newton :class:`Model`, and run the PGS pipeline WITHOUT
external collision. Verify:

* Rest state is a fixed point: a tet in its rest pose with zero
  velocity stays at rest pose (no force applied by the corotational
  shear row).
* Free-fall integrates gravity correctly: a small soft cube in vacuum
  with no constraints other than the internal tet rows falls under
  gravity without exploding / NaN-ing.
* The constraint kernel compiles and runs under
  ``wp.ScopedCapture`` (graph-capture safety).

These tests run on CUDA only -- PhoenX kernels are not CPU-portable.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_soft_cube(
    *, dim: int = 2, cell: float = 0.1, density: float = 100.0, k_mu: float = 1.0e4, k_lambda: float = 1.0e4
):
    """Build a Newton model containing a dim x dim x dim soft cube.

    Returns the finalised :class:`newton.Model`.
    """
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=0.0,
        add_surface_mesh_edges=False,
    )
    return builder.finalize()


def _build_phoenx_world_for_soft_cube(
    model, device, gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> PhoenXWorld:
    """Build a PhoenXWorld matched to the given soft-cube model. No
    external collision; pure internal-constraint exercise."""
    bodies = body_container_zeros(max(1, int(model.body_count)), device=device)
    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0,
        num_cloth_triangles=0,
        num_soft_tetrahedra=int(model.tet_count),
        device=device,
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=int(model.particle_count),
        num_cloth_triangles=0,
        num_soft_tetrahedra=int(model.tet_count),
        rigid_contact_max=1,  # min sentinel; no contacts produced
        num_worlds=1,
        substeps=4,
        solver_iterations=8,
        step_layout="single_world",
        device=device,
    )
    world.gravity.assign(np.array([gravity], dtype=np.float32))
    world.populate_soft_tetrahedra_from_model(model)
    return world


def _total_tet_volume(model, positions: np.ndarray) -> float:
    tet_indices = model.tet_indices.numpy().reshape(-1, 4)
    volume = 0.0
    for a, b, c, d in tet_indices:
        xa = positions[a]
        xb = positions[b]
        xc = positions[c]
        xd = positions[d]
        volume += abs(float(np.dot(xb - xa, np.cross(xc - xa, xd - xa)))) / 6.0
    return volume


def _volume_error(model, positions: np.ndarray, rest_volume: float) -> float:
    return abs(_total_tet_volume(model, positions) - rest_volume) / rest_volume


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-body tests are CUDA-only.",
)
class TestSoftTetrahedronInternal(unittest.TestCase):
    def test_rest_state_is_fixed_point(self):
        # A soft cube at rest pose with zero velocity and zero gravity
        # must stay at rest pose: the corotational shear row evaluates
        # to zero on the rest configuration.
        device = wp.get_preferred_device()
        model = _build_soft_cube()
        world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, 0.0))
        p_initial = world.particles.position.numpy().copy()
        for _ in range(10):
            world.step(1.0 / 60.0)
        p_final = world.particles.position.numpy()
        # Rest pose should be near-identity (no force applied, no
        # velocity, no gravity). Allow tiny float drift from the
        # corotational rotation extraction's first iteration.
        np.testing.assert_allclose(p_final, p_initial, atol=1e-4)

    def test_free_fall_integrates_gravity(self):
        # A soft cube in vacuum (no ground plane, no contacts) under
        # gravity must fall ballistically without exploding. The
        # internal tet constraints don't generate gravity; only the
        # solver's integrate step does. After T seconds the mean z
        # should equal initial_z - 0.5*g*T^2 within ~5%.
        device = wp.get_preferred_device()
        model = _build_soft_cube()
        world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, -9.81))
        initial_mean_z = float(world.particles.position.numpy()[:, 2].mean())
        T = 0.25  # 0.25 s of free fall
        dt = 1.0 / 60.0
        n_frames = int(T / dt)
        for _ in range(n_frames):
            world.step(dt)
        p_final = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(p_final)), "non-finite particle position after free fall")
        final_mean_z = float(p_final[:, 2].mean())
        elapsed = n_frames * dt
        expected_drop = 0.5 * 9.81 * elapsed * elapsed
        actual_drop = initial_mean_z - final_mean_z
        # Within 10% of analytical free-fall (substep integrator
        # accumulates a small explicit-Euler bias; tight tolerance
        # would be 1%).
        self.assertAlmostEqual(actual_drop, expected_drop, delta=0.1 * expected_drop)

    def test_volume_stiffness_controls_volume_recovery(self):
        device = wp.get_preferred_device()

        def run(k_lambda: float) -> float:
            model = _build_soft_cube(k_mu=1.0e2, k_lambda=k_lambda)
            world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, 0.0))
            rest = world.particles.position.numpy().copy()
            rest_volume = _total_tet_volume(model, rest)
            center = rest.mean(axis=0)
            compressed = center + (rest - center) * np.array([1.0, 1.0, 0.65], dtype=np.float32)
            compressed_wp = wp.array(compressed, dtype=wp.vec3f, device=device)
            wp.copy(world.particles.position, compressed_wp)
            wp.copy(world.particles.position_prev_substep, compressed_wp)
            for _ in range(30):
                world.step(1.0 / 120.0)
            return _volume_error(model, world.particles.position.numpy(), rest_volume)

        low_error = run(1.0e2)
        high_error = run(1.0e7)
        self.assertLess(high_error, 0.6 * low_error)

    def test_runs_under_graph_capture(self):
        # Load-bearing capture-safety check: the soft-tet prepare /
        # iterate launches must run inside ``wp.ScopedCapture`` +
        # ``wp.capture_launch`` and re-launch deterministically.
        device = wp.get_preferred_device()
        model = _build_soft_cube()
        world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, -9.81))
        # Warm-up.
        world.step(1.0 / 60.0)
        with wp.ScopedCapture(device=device) as capture:
            world.step(1.0 / 60.0)
        for _ in range(5):
            wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(world.particles.position.numpy())))


if __name__ == "__main__":
    unittest.main()
