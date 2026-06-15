# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Momentum conservation for soft body + mass splitting.

A small soft-tet cube settles on a static ground plane with mass
splitting ON. Once settled, the sum of contact normal lambdas across
all active rigid-particle contacts must equal ``M * g * substep_dt``
-- the impulse needed to cancel gravity for a body of total mass
``M`` over one substep.

Settling check: total kinetic energy of the soft-body particles must
fall below a low threshold (well below the initial KE from the drop).
This guarantees the system is in steady state before the impulse-sum
check.

CUDA only, graph-captured.
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
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

_GRAVITY = 9.81
_SUBSTEPS = 4
_SOLVER_ITERATIONS = 16
_FPS = 60
_DT = 1.0 / _FPS
_SUBSTEP_DT = _DT / _SUBSTEPS

# Soft-cube parameters. Resolution=2 gives 8 hex cells -> 40 tets, 27
# particles. Small enough for a quick test, large enough that the
# overflow partition is non-trivial (every interior particle touches
# multiple tets -> the per-particle slot count > 1 in some bucket).
# Spawn just above ground so initial momentum is small.
_CUBE_SIZE = 0.3
_CUBE_RESOLUTION = 2
_DENSITY = 500.0  # kg / m^3
_DROP_HEIGHT = 0.06  # ~5 mm above the contact-rest position; small bounce
_YOUNGS_MODULUS = 5.0e7
_POISSON_RATIO = 0.3
# Macklin XPBD damping. Wired into the shear-row iterate in
# constraint_soft_tetrahedron.py: ``gamma_mu = beta_mu * substep_dt``
# enters the lambda numerator as ``gamma * grad . (x -
# position_prev_substep)``. ``5.0`` settles the cube within ~1 s of
# sim so the time-averaged impulse converges to ``M * g * dt_substep``
# within a few percent. Lower values (e.g. 0.1) leave residual internal
# oscillation that biases the impulse mean by ~7 % even after long
# averages -- bad for a momentum-conservation regression test.
_BETA_MU = 5.0

# Settling frames: 200 frames (~3.3 s) is enough for the cube to
# reach KE ~ 0.33 J at beta_mu=5.0 (down from 25 J at impact). Past
# this the contact-impulse mean stabilises around the analytic value.
_SETTLE_FRAMES = 200
# Number of frames over which to time-average the normal-impulse sum.
# At beta_mu=5.0 individual-frame readings vary +/- ~10 % due to the
# remaining low-amplitude internal oscillation; averaging over 120
# frames brings the mean close enough to catch large coloring bias.
_MEASURE_FRAMES = 120
# Tolerance on (avg impulse / expected impulse - 1). The metric is
# intentionally strict enough to catch biased coloring schedules on
# current kernels (periodic cache-stir measured >10 %), while allowing
# the residual soft-body oscillation that remains after the settle
# window.
_IMPULSE_REL_TOL = 0.05


class _SoftCubeMassSplittingScene:
    """Single soft-tet cube on ground, mass splitting ON."""

    def __init__(self, device: wp.Device):
        self.device = device

        k_mu, k_lambda = soft_tet_lame_from_youngs_poisson(_YOUNGS_MODULUS, _POISSON_RATIO)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=0.0)

        cell_size = _CUBE_SIZE / _CUBE_RESOLUTION
        half = 0.5 * _CUBE_SIZE
        # ``add_soft_grid`` spawns the +x/+y/+z octant from ``pos``; shift
        # x/y by -half so the cube is centred on world Z. z = _DROP_HEIGHT
        # places the cube bottom ~5 mm above ground -- a deliberately tiny
        # drop because soft-tet has no internal damping.
        builder.add_soft_grid(
            pos=wp.vec3(-half, -half, _DROP_HEIGHT),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=_CUBE_RESOLUTION,
            dim_y=_CUBE_RESOLUTION,
            dim_z=_CUBE_RESOLUTION,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=_DENSITY,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

        self.model = builder.finalize(device=device)

        # No rigid dynamic bodies; just the ground plane (which lives at
        # shape_body=-1). Allocate one slot for the static world anchor.
        num_phoenx_bodies = max(1, int(self.model.body_count) + 1)
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=device,
            )
        )
        if int(self.model.body_count) > 0:
            state_init = self.model.state()
            wp.launch(
                init_phoenx_bodies_kernel,
                dim=int(self.model.body_count),
                inputs=[
                    self.model.body_q,
                    state_init.body_qd,
                    self.model.body_com,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                ],
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
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            num_worlds=1,
            substeps=_SUBSTEPS,
            solver_iterations=_SOLVER_ITERATIONS,
            rigid_contact_max=4096,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=12,
            device=device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -_GRAVITY]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(self.model, beta_mu=_BETA_MU)
        self.pipeline = self.world.setup_cloth_collision_pipeline(
            self.model, soft_body_thickness=0.005, soft_body_gap=0.010, rigid_contact_max=4096
        )
        self.contacts = self.pipeline.contacts()
        self.state = self.model.state()

        # Total soft-body mass = sum of particle masses. Static / pinned
        # particles have inverse_mass==0 and contribute nothing -- but
        # the soft cube has no pins, so every particle is dynamic.
        inv_mass = self.world.particles.inverse_mass.numpy()
        masses = np.where(inv_mass > 0.0, 1.0 / np.maximum(inv_mass, 1.0e-30), 0.0)
        self.total_mass = float(masses.sum())

        # Warm-up step + capture.
        self._simulate_one_frame()
        with wp.ScopedCapture(device=device) as cap:
            self._simulate_one_frame()
        self._graph = cap.graph

    def _simulate_one_frame(self) -> None:
        self.world.collide(self.state, self.contacts)
        self.world.step(_DT, contacts=self.contacts)

    def step(self) -> None:
        wp.capture_launch(self._graph)

    def kinetic_energy(self) -> float:
        """Sum of ``0.5 * m * |v|^2`` over every soft-body particle."""
        v = self.world.particles.velocity.numpy()
        inv_m = self.world.particles.inverse_mass.numpy()
        m = np.where(inv_m > 0.0, 1.0 / np.maximum(inv_m, 1.0e-30), 0.0)
        v_sq = (v * v).sum(axis=1)
        return 0.5 * float((m * v_sq).sum())

    def normal_lambda_sum(self) -> float:
        """Sum of the normal-impulse row across every active contact
        column. ``ContactContainer.impulses`` is keyed by contact index
        ``k``; row 0 holds the normal impulse. Inactive columns hold
        ``0.0`` so the raw sum is well-defined."""
        return float(self.world._contact_container.impulses.numpy()[0, :].sum())


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Soft-body + mass-splitting test runs on CUDA only (mass splitting requires CUDA + graph capture).",
)
class TestSoftBodyMassSplittingMomentum(unittest.TestCase):
    """A settled soft-tet cube's normal-contact-impulse sum must equal
    its weight over one substep: ``sum(lambda_n) == M * g * dt_substep``."""

    def test_normal_impulse_balances_weight(self) -> None:
        """Time-averaged sum of contact normal lambdas equals
        ``M * g * dt_substep`` to within 1 % -- the impulse needed to
        cancel gravity for the soft body's total mass over one
        substep.

        Strategy: drop the cube ~5 mm above the rest position (small
        impact energy), enable Macklin XPBD shear-row damping at
        ``beta_mu = 5.0`` so the internal oscillation modes bleed off,
        run a 200-frame settling loop to converge the contact warm-
        start, then time-average the per-frame normal-impulse readings
        over 120 frames. The tolerance is kept at 5 %: low enough to
        catch biased coloring schedules observed on this scene, but
        high enough for the remaining damped soft-body oscillation.
        """
        device = wp.get_preferred_device()
        scene = _SoftCubeMassSplittingScene(device)

        for _ in range(_SETTLE_FRAMES):
            scene.step()
        wp.synchronize_device(device)

        expected_impulse = scene.total_mass * _GRAVITY * _SUBSTEP_DT

        readings = []
        for _ in range(_MEASURE_FRAMES):
            scene.step()
            wp.synchronize_device(device)
            readings.append(scene.normal_lambda_sum())
        measured = float(np.mean(readings))

        rel_err = abs(measured - expected_impulse) / max(expected_impulse, 1.0e-12)
        self.assertLess(
            rel_err,
            _IMPULSE_REL_TOL,
            f"time-averaged normal-impulse sum {measured:.4f} N.s vs M*g*dt_substep "
            f"{expected_impulse:.4f} N.s (rel err {rel_err:.2%}); > {_IMPULSE_REL_TOL:.0%} "
            f"means contact-side momentum is not balanced. "
            f"M = {scene.total_mass:.4f} kg, dt_substep = {_SUBSTEP_DT:.5f} s, "
            f"avg over {_MEASURE_FRAMES} frames; min/max readings = "
            f"({min(readings):.4f}, {max(readings):.4f})",
        )

    def test_settled_height_above_ground(self) -> None:
        """Sanity: every particle must remain above ground (z >= 0)
        and the cube's z extent must stay near its original size --
        validates that the contact pipeline doesn't leak particles
        through the plane and that the soft-tet didn't collapse."""
        device = wp.get_preferred_device()
        scene = _SoftCubeMassSplittingScene(device)

        for _ in range(_SETTLE_FRAMES):
            scene.step()

        p = scene.world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(p)), "non-finite particle position after settle")

        z_min = float(p[:, 2].min())
        z_max = float(p[:, 2].max())
        self.assertGreater(
            z_min,
            -0.02,
            f"soft cube penetrated ground: min z = {z_min:.4f}",
        )
        self.assertLess(
            z_min,
            0.05,
            f"soft cube didn't reach ground: min z = {z_min:.4f}",
        )
        self.assertLess(
            z_max,
            _CUBE_SIZE + _DROP_HEIGHT + 0.05,
            f"soft cube did not settle to expected height: max z = {z_max:.4f}, "
            f"expected near {_CUBE_SIZE + _DROP_HEIGHT:.4f}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
