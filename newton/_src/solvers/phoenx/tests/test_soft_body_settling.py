# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Settling regression tests for the soft-tetrahedron drop scene.

The dense soft-tet drop in :mod:`example_soft_body_drop` (15k tets
sharing ~7k particles via the standard 5-tets-per-voxel hex
decomposition) historically had ~10-20% of cubes still wiggling
visibly after several seconds of wall-time, even with the Tonge
mass-splitting Jacobi-block path running. The wiggling does decay
but slowly; without a regression guard, perf changes to either the
mass-splitting layout (``max_colored_partitions``,
``ms_batch_size``), the polar-decomposition rotation extraction
(Mueller / Kugelstadt APD), or the slot-aware access helpers
(``set_access_mode_with_slot`` / ``read_position_with_slot``) could
easily make settling *worse* without breaking the existing
momentum-balance or determinism asserts.

This file exercises a deliberately small, fully-self-contained
scene (4 piles x 2 cubes x ``cube_resolution=1`` = 40 tets) so the
test fits in the PhoenX 1-minute timeout budget. The scene is the
minimum that:

* exercises the mass-splitting overflow bucket (multiple soft-tet
  rows per particle when batch_size <= 5 tets/cube);
* exercises both colored and uncoloured paths via the
  ``max_colored_partitions=0`` overflow-only regression knob;
* leaves the cubes free-falling for a non-trivial amount before
  contact;

and the asserts check the property the perf work most directly
risks regressing -- the maximum residual particle speed after a
generous settle window.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.examples.example_common import init_phoenx_bodies_kernel
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_settling_scene(
    *,
    device,
    grid_xy: int = 2,
    num_cubes: int = 2,
    cube_size: float = 0.4,
    cube_resolution: int = 1,
    base_drop_height: float = 1.0,
    cube_spacing: float = 0.5,
    grid_spacing_mult: float = 1.5,
    youngs_modulus: float = 1.0e9,
    poisson_ratio: float = 0.3,
    density: float = 500.0,
    max_colored_partitions: int = 0,
    mass_splitting_batch_size: int = 2,
):
    """Replicates :class:`example_soft_body_drop.Example` with smaller
    defaults so the unit test fits the 1-minute timeout. Returns
    ``(model, world, contacts, collision_pipeline)``."""
    k_lambda, k_mu = soft_tet_lame_from_youngs_poisson(youngs_modulus, poisson_ratio)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=1.0)

    cube_xy_offset = -0.5 * cube_size
    cell_size = cube_size / cube_resolution
    grid_spacing = grid_spacing_mult * cube_size
    grid0 = -0.5 * (grid_xy - 1) * grid_spacing
    for gx in range(grid_xy):
        for gy in range(grid_xy):
            pile_cx = grid0 + gx * grid_spacing
            pile_cy = grid0 + gy * grid_spacing
            for i in range(num_cubes):
                drop_height = base_drop_height + i * cube_spacing
                builder.add_soft_grid(
                    pos=wp.vec3(pile_cx + cube_xy_offset, pile_cy + cube_xy_offset, drop_height),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=cube_resolution,
                    dim_y=cube_resolution,
                    dim_z=cube_resolution,
                    cell_x=cell_size,
                    cell_y=cell_size,
                    cell_z=cell_size,
                    density=density,
                    k_mu=k_mu,
                    k_lambda=k_lambda,
                    k_damp=0.0,
                    add_surface_mesh_edges=False,
                )
    model = builder.finalize(device=device)

    num_phoenx_bodies = max(1, int(model.body_count) + 1)
    bodies = body_container_zeros(num_phoenx_bodies, device=device)
    bodies.orientation.assign(
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
            dtype=wp.quatf,
            device=device,
        )
    )
    if model.body_count > 0:
        state_init = model.state()
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=model.body_count,
            inputs=[model.body_q, state_init.body_qd, model.body_com, model.body_inv_mass, model.body_inv_inertia],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=device,
        )

    num_piles = grid_xy * grid_xy
    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0, num_cloth_triangles=0, num_soft_tetrahedra=int(model.tet_count), device=device
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=int(model.particle_count),
        num_cloth_triangles=0,
        num_soft_tetrahedra=int(model.tet_count),
        num_worlds=1,
        substeps=5,
        solver_iterations=4,
        velocity_iterations=0,
        rigid_contact_max=2 * 1024 * cube_resolution * cube_resolution * num_piles,
        step_layout="single_world",
        mass_splitting=True,
        max_colored_partitions=max_colored_partitions,
        mass_splitting_batch_size=mass_splitting_batch_size,
        partitioner_algorithm="greedy",
        device=device,
    )
    world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
    world.populate_soft_tetrahedra_from_model(model)
    collision_pipeline = world.setup_cloth_collision_pipeline(
        model,
        soft_body_thickness=0.005,
        soft_body_gap=0.010,
        rigid_contact_max=1024 * cube_resolution * cube_resolution * num_piles,
    )
    contacts = collision_pipeline.contacts()
    state = model.state()
    return model, world, contacts, collision_pipeline, state


def _run_settling(world, contacts, collision_pipeline, state, *, frame_dt, num_frames):
    """Run ``num_frames`` of ``collide + step`` under a captured CUDA
    graph (the shipped example does the same)."""
    device = wp.get_device()

    def _one_frame():
        world.collide(state, contacts)
        world.step(frame_dt, contacts=contacts)

    # Warm-up + capture.
    _one_frame()
    with wp.ScopedCapture(device=device) as capture:
        _one_frame()
    for _ in range(num_frames):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-body tests are CUDA-only.",
)
class TestSoftBodySettling(unittest.TestCase):
    """Regression guard for the dense-soft-tet drop's residual motion.

    The thresholds are set with generous headroom vs. measured
    behaviour (current branch as of the perf-opts work; small scene
    4 piles x 2 cubes x ``cube_resolution=1``, 600 frames at 120 Hz):

    +-----------------------+----------+---------------+---------+
    | config                | max |v|  | %>0.1 m/s     | %>0.05  |
    +=======================+==========+===============+=========+
    | max=0, batch=2        | 0.24     | 14%           | 45%     |
    | max=8, batch=2        | 0.38     | 3%            | 30%     |
    | baseline (main),  max=0 | 0.24   | 20%           | 53%     |
    | baseline (main),  max=8 | 0.27   | 8%            | 31%     |
    +-----------------------+----------+---------------+---------+

    ``MAX_RESIDUAL_SPEED = 1.0`` and ``MAX_FRACTION_FAST = 0.30``
    (frac > 0.1 m/s) give ~3x headroom on current measurements and
    1.5x on the worst baseline measurement. A real regression (e.g.,
    a polar-decomp swap that returns the wrong rotation, or an
    average-and-broadcast change that drops some slot velocities)
    would push max |v| well past 1.0 or fraction well past 50%.

    Tightening below baseline would turn this into a noise-flaky
    test; the goal is to catch substantive regressions, not the
    pre-existing dense-soft-tet residual jitter.
    """

    NUM_SETTLE_FRAMES = 600  # 5.0 s at frame_dt = 1/120
    FRAME_DT = 1.0 / 120.0
    MAX_RESIDUAL_SPEED = 1.0
    FAST_SPEED_THRESHOLD = 0.1
    MAX_FRACTION_FAST = 0.30
    MED_SPEED_THRESHOLD = 0.05
    MAX_FRACTION_MED = 0.70

    def _assert_settled(self, world, label: str):
        vel = world.particles.velocity.numpy()
        pos = world.particles.position.numpy()
        speeds = np.linalg.norm(vel, axis=1)
        n = len(speeds)
        self.assertTrue(np.all(np.isfinite(pos)), f"{label}: non-finite particle position")
        self.assertTrue(np.all(np.isfinite(vel)), f"{label}: non-finite particle velocity")
        self.assertLess(
            float(speeds.max()),
            self.MAX_RESIDUAL_SPEED,
            f"{label}: max |v|={speeds.max():.4f} m/s exceeds {self.MAX_RESIDUAL_SPEED} after "
            f"{self.NUM_SETTLE_FRAMES} frames -- soft-body solver is not converging to rest.",
        )
        fraction_fast = float((speeds > self.FAST_SPEED_THRESHOLD).sum()) / n
        self.assertLess(
            fraction_fast,
            self.MAX_FRACTION_FAST,
            f"{label}: {100 * fraction_fast:.1f}% of particles moving fast "
            f"(|v| > {self.FAST_SPEED_THRESHOLD} m/s); expected < "
            f"{100 * self.MAX_FRACTION_FAST:.1f}%. Residual motion regression.",
        )
        fraction_med = float((speeds > self.MED_SPEED_THRESHOLD).sum()) / n
        self.assertLess(
            fraction_med,
            self.MAX_FRACTION_MED,
            f"{label}: {100 * fraction_med:.1f}% of particles moving "
            f"(|v| > {self.MED_SPEED_THRESHOLD} m/s); expected < "
            f"{100 * self.MAX_FRACTION_MED:.1f}%.",
        )
        # Cubes must remain above the ground plane (z >= 1.0 with a
        # margin for the soft-body contact margin).
        self.assertGreater(
            float(pos[:, 2].min()),
            0.5,
            f"{label}: a particle escaped below the ground plane (min z = {pos[:, 2].min():.4f}).",
        )

    def test_settles_with_overflow_only(self):
        """``max_colored_partitions=0`` -- overflow-only regression mode.
        All soft-tet rows route to the Jacobi-block overflow bucket;
        ``_average_and_broadcast_kernel`` reconciles batches between
        PGS iterations."""
        device = wp.get_preferred_device()
        _, world, contacts, collision_pipeline, state = _build_settling_scene(
            device=device, max_colored_partitions=0, mass_splitting_batch_size=2
        )
        _run_settling(
            world, contacts, collision_pipeline, state, frame_dt=self.FRAME_DT, num_frames=self.NUM_SETTLE_FRAMES
        )
        self._assert_settled(world, label="max=0/batch=2")

    def test_settles_with_eight_colors(self):
        """``max_colored_partitions=8`` -- the configuration mixed
        rigid-soft scenes need so the rigid contacts get Gauss-Seidel
        convergence via colours. Soft-tet rows that don't fit in 8
        colours spill to the overflow bucket and converge via the
        Jacobi-block path."""
        device = wp.get_preferred_device()
        _, world, contacts, collision_pipeline, state = _build_settling_scene(
            device=device, max_colored_partitions=8, mass_splitting_batch_size=2
        )
        _run_settling(
            world, contacts, collision_pipeline, state, frame_dt=self.FRAME_DT, num_frames=self.NUM_SETTLE_FRAMES
        )
        self._assert_settled(world, label="max=8/batch=2")


if __name__ == "__main__":
    unittest.main()
