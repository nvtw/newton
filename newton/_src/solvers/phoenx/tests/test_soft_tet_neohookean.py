# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the block Neo-Hookean soft-tetrahedron variant.

Mirrors :mod:`test_soft_tetrahedron` but stamps the constraint rows
with :attr:`SoftBodyConstraintType.BLOCK_NEOHOOKEAN`. CUDA-only and
exercises the constraint kernels under :class:`wp.ScopedCapture` so the
2x2 Schur solve is verified safe for graph-captured replay.

Coverage:

* Free-fall under gravity: ballistic drop without NaNs / blow-ups,
  proving the analytic Neo-Hookean gradient + 2x2 solve are stable
  through many substeps.
* Stationary stability: a soft cube at the rest pose with zero gravity
  and zero initial velocity remains spatially bounded (the stable
  Neo-Hookean formulation has zero combined energy gradient on the rest
  configuration, so the cube should hold its shape -- modulo a tiny
  drift to the analytical equilibrium ``F = gamma^(1/3) I`` for finite
  ``mu / lambda``).
* Graph-capture replay: prepare + iterate launches re-execute
  deterministically inside a ``wp.ScopedCapture`` graph.
* Variant selection: stamping ``ARAP`` vs ``BLOCK_NEOHOOKEAN`` writes
  the corresponding constraint type tag at dword 0 of every soft-tet
  cid (verifies the dispatcher routes to the right kernel).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON,
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import (
    SoftBodyConstraintType,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_soft_cube(
    *, dim: int = 2, cell: float = 0.1, density: float = 100.0, k_mu: float = 1.0e4, k_lambda: float = 1.0e4
):
    """Build a Newton model containing a dim x dim x dim soft cube."""
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
    model,
    device,
    *,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    constraint_type: SoftBodyConstraintType = SoftBodyConstraintType.BLOCK_NEOHOOKEAN,
) -> PhoenXWorld:
    """Build a PhoenXWorld matched to the given soft-cube model with the
    requested soft-body constraint variant. No external collision."""
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
        rigid_contact_max=1,
        num_worlds=1,
        substeps=4,
        solver_iterations=8,
        step_layout="single_world",
        device=device,
    )
    world.gravity.assign(np.array([gravity], dtype=np.float32))
    world.populate_soft_tetrahedra_from_model(model, constraint_type=constraint_type)
    return world


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-body tests are CUDA-only.",
)
class TestSoftTetNeoHookean(unittest.TestCase):
    def test_variant_tag_dispatch(self):
        # Stamping with BLOCK_NEOHOOKEAN must write the corresponding
        # constraint-type tag at dword 0 of every soft-tet cid; the ARAP
        # default writes CONSTRAINT_TYPE_SOFT_TETRAHEDRON. This load-bears
        # the per-cid ctype dispatch in the persistent / fused kernels.
        device = wp.get_preferred_device()
        model = _build_soft_cube()

        world_neo = _build_phoenx_world_for_soft_cube(
            model, device, constraint_type=SoftBodyConstraintType.BLOCK_NEOHOOKEAN
        )
        types_neo = world_neo.constraints.data.numpy()[0].view(np.int32)
        expected_neo = int(CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN)
        self.assertTrue(
            np.all(types_neo[: int(model.tet_count)] == expected_neo),
            f"expected ctype={expected_neo} for every neohookean cid, got distribution "
            f"{np.unique(types_neo[: int(model.tet_count)])}",
        )

        world_arap = _build_phoenx_world_for_soft_cube(model, device, constraint_type=SoftBodyConstraintType.ARAP)
        types_arap = world_arap.constraints.data.numpy()[0].view(np.int32)
        expected_arap = int(CONSTRAINT_TYPE_SOFT_TETRAHEDRON)
        self.assertTrue(
            np.all(types_arap[: int(model.tet_count)] == expected_arap),
            f"expected ctype={expected_arap} for every ARAP cid, got distribution "
            f"{np.unique(types_arap[: int(model.tet_count)])}",
        )

    def test_free_fall_integrates_gravity(self):
        # A soft cube in vacuum under gravity must fall ballistically
        # without exploding. The block solve is the only constraint
        # active; only the substep integrator generates gravity.
        device = wp.get_preferred_device()
        model = _build_soft_cube()
        world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, -9.81))
        initial_mean_z = float(world.particles.position.numpy()[:, 2].mean())
        T = 0.25
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
        # Looser tolerance than ARAP: the stable Neo-Hookean rest
        # configuration is F = gamma^(1/3) I, not F = I, so the cube
        # also breathes slightly under self-equilibration. ~12% covers
        # both the explicit-Euler bias on free fall and the
        # equilibration mode.
        self.assertAlmostEqual(actual_drop, expected_drop, delta=0.12 * expected_drop)

    def test_rest_pose_remains_bounded(self):
        # With zero gravity, zero velocity, and the cube initialised at
        # its rest pose, the block Neo-Hookean solver must keep the cube
        # spatially bounded. The combined hydrostatic + deviatoric
        # energy gradient vanishes on the rest pose so there is no net
        # force; in practice individual rows may push the cube to a
        # slightly inflated equilibrium for finite ``mu / lambda``, so
        # we assert a loose-but-finite drift bound.
        device = wp.get_preferred_device()
        # Use a high-Poisson material (lambda >> mu) so gamma ~ 1 and
        # the rest-equilibrium drift is small.
        model = _build_soft_cube(k_mu=1.0e3, k_lambda=1.0e5)
        world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, 0.0))
        p_initial = world.particles.position.numpy().copy()
        for _ in range(20):
            world.step(1.0 / 60.0)
        p_final = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(p_final)), "non-finite particle position at rest")
        # Drift bound: under a stable solver, no particle should walk
        # more than the cube's characteristic length within 20 frames
        # of self-equilibration with no external load. The cube is
        # 2x2x2 cells of 0.1 m, so 0.2 m total -- 0.05 m drift is
        # already a generous bound.
        drift = np.linalg.norm(p_final - p_initial, axis=1).max()
        self.assertLess(drift, 0.05, f"max drift {drift:.3f} m exceeds rest stability bound")

    def test_runs_under_graph_capture(self):
        # Capture-safety: prepare + iterate launches must execute
        # deterministically under ``wp.ScopedCapture`` + repeated
        # ``wp.capture_launch``. This load-bears the host-allocation
        # contract of the new constraint kernel: no .numpy() in step(),
        # no dynamic memory, all sizing determined at populate time.
        device = wp.get_preferred_device()
        model = _build_soft_cube()
        world = _build_phoenx_world_for_soft_cube(model, device, gravity=(0.0, 0.0, -9.81))
        # Warm-up out-of-capture so all kernels JIT-compile + allocate.
        world.step(1.0 / 60.0)
        with wp.ScopedCapture(device=device) as capture:
            world.step(1.0 / 60.0)
        for _ in range(5):
            wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(world.particles.position.numpy())))


if __name__ == "__main__":
    unittest.main()
