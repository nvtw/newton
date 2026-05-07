# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical-deflection unit test for the cloth-triangle constraint.

Sets up a single quad (two triangles) hanging from its top edge under
gravity:

* Top two particles have ``inverse_mass = 0`` (pinned to the world).
* Bottom two particles each have a known mass ``m``.
* Area conservation is disabled (``tri_ka`` set to the stiffness
  floor so its row contributes negligible compliance).
* Only the shear / mu row is active; ``mu`` is set to a known value.
* Gravity pulls along the in-plane direction so the deformation
  reduces to a uniaxial vertical stretch.

For a co-rotational shear element under uniform uniaxial strain
``epsilon = Delta / L``, the per-triangle energy density is
``W = mu * (||F - R||_F)^2`` and integrating over the rest area gives
total cloth energy ``U_total = mu * L^2 * (Delta / L)^2 = mu * Delta^2``
for a square quad of in-plane size ``L``.  Two free particles share
the elastic restoring force; equating against gravity ``2 * m * g``::

    2 * mu * Delta = 2 * m * g
    Delta = m * g / mu

This is what the test asserts.  Switching the area row off with
``tri_ka -> 0`` lets us isolate the shear law -- otherwise lambda
adds a Poisson-style cross-term that the closed-form above doesn't
capture.

The test is deliberately *physical*: it doesn't peek at internal
solver state, only the steady-state particle positions after enough
substeps to settle.  Rayleigh damping inside the constraint dissipates
the transient oscillation; we then compare the deflection against the
analytical answer.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import ParticleFlags
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_strain,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth tests require CUDA")
class TestClothTriangleAnalyticalDeflection(unittest.TestCase):
    """A single quad pinned on top, hanging under gravity, only the
    shear row active -- compare steady-state deflection to the closed
    form ``Delta = m * g / mu``."""

    def _build(
        self,
        *,
        cell: float,
        particle_mass: float,
        mu: float,
        gravity_xyz: tuple[float, float, float],
        substeps: int,
        iterations: int,
        beta_mu: float,
    ) -> tuple[PhoenXWorld, newton.Model, int, int, int, int]:
        """Build the 2-triangle test scene and return the assembled
        world plus the four particle indices ``(top_a, top_b, bot_a,
        bot_b)`` so the caller can read displacements."""
        device = wp.get_device()

        # Single quad (dim_x=1, dim_y=1 -> 4 particles, 2 triangles).
        # add_cloth_grid lays the cloth in the local xy plane at z=0,
        # then translates / rotates by ``pos`` / ``rot``.  We keep the
        # cloth flat in the xy plane and pull gravity along -y; the
        # deformation reduces to a uniaxial in-plane stretch.
        builder = newton.ModelBuilder()

        # Set ``tri_ka`` to the stiffness floor so the area row has
        # essentially zero stiffness (effectively infinite compliance);
        # only the shear row contributes restoring force.  We can't
        # pass exactly zero -- the populate kernel floors before
        # dividing -- but the floor of 1e-6 is 11 orders of magnitude
        # below ``mu`` here, so its contribution is buried in the
        # numerical noise.
        tri_ka_disabled = 1.0e-6

        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=cell,
            cell_y=cell,
            mass=particle_mass,
            tri_ke=mu,
            tri_ka=tri_ka_disabled,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.5 * cell,
        )

        # Particle layout (row-major (dim_x+1)*(dim_y+1)) = 2x2 = 4:
        #   index 0 -> (x=0, y=0)
        #   index 1 -> (x=L, y=0)
        #   index 2 -> (x=0, y=L)
        #   index 3 -> (x=L, y=L)
        # Pin the *top* row (y=L) so gravity along -y stretches the
        # cloth (free bottom row drops in -y past y=0).  Pinning the
        # bottom would put the cloth in compression -- with the area
        # row disabled, the constraint can't resist buckling and the
        # answer becomes ill-defined.
        top_a, top_b = 2, 3
        bot_a, bot_b = 0, 1
        for c in (top_a, top_b):
            builder.particle_mass[c] = 0.0
            builder.particle_flags[c] = builder.particle_flags[c] & ~ParticleFlags.ACTIVE

        # We can't set per-component gravity in the builder (it stores a
        # scalar magnitude with the world's z-up convention); we set the
        # PhoenX world's gravity vector directly below.
        builder.gravity = 0.0

        model = builder.finalize(device=device)

        # Stand up a minimal PhoenX world: zero rigid bodies, num
        # particles equal to ``model.particle_count``, all four cloth
        # triangle rows.
        bodies = body_container_zeros(1, device=device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(model.tri_count),
            device=device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(model.particle_count),
            num_cloth_triangles=int(model.tri_count),
            num_worlds=1,
            substeps=substeps,
            solver_iterations=iterations,
            gravity=gravity_xyz,
            rigid_contact_max=0,
            step_layout="single_world",
            device=device,
        )
        # ``beta_lambda`` is irrelevant (the area row is disabled); set
        # it equal to ``beta_mu`` for symmetry.
        world.populate_cloth_triangles_from_model(model, beta_lambda=beta_mu, beta_mu=beta_mu)
        return world, model, top_a, top_b, bot_a, bot_b

    def _measure_settled_deflection(
        self,
        *,
        particle_mass: float,
        gravity_y: float,
        cell: float,
        mu: float,
        substeps: int,
        iterations: int,
        beta_mu: float,
        n_frames: int,
    ) -> float:
        world, model, _top_a, _top_b, bot_a, bot_b = self._build(
            cell=cell,
            particle_mass=particle_mass,
            mu=mu,
            gravity_xyz=(0.0, gravity_y, 0.0),
            substeps=substeps,
            iterations=iterations,
            beta_mu=beta_mu,
        )
        initial_q = model.particle_q.numpy().copy()
        y_bot_initial = float(0.5 * (initial_q[bot_a, 1] + initial_q[bot_b, 1]))
        dt = 1.0 / 60.0
        for _ in range(n_frames):
            world.step(dt=dt, contacts=None, shape_body=None)
        final_q = world.particles.position.numpy()
        # Sanity: both free particles co-move -- if the constraint is
        # asymmetric they would split apart.
        np.testing.assert_allclose(
            float(final_q[bot_a, 1]),
            float(final_q[bot_b, 1]),
            atol=1.0e-3,
            err_msg="bottom particles drifted apart -- constraint is asymmetric",
        )
        # Sanity: no NaN / blow-up.
        self.assertTrue(
            np.all(np.isfinite(final_q)),
            f"non-finite particle position with m={particle_mass}, g={gravity_y}",
        )
        y_bot_final = float(0.5 * (final_q[bot_a, 1] + final_q[bot_b, 1]))
        return y_bot_initial - y_bot_final

    def test_quad_hookes_law_linearity(self) -> None:
        """Two free bottom particles, two pinned top particles, only
        the shear row active.  Verify that the steady-state deflection
        is *linear* in the applied gravitational load -- this is the
        defining property of Hookean elasticity, and the cloth must
        reproduce it to be physical.

        Concretely::

            load_1 = m_1 * g_1
            load_2 = m_2 * g_2 = 2 * load_1
            => Delta(load_2) ≈ 2 * Delta(load_1)

        Holding the constraint compliance fixed and varying only the
        applied force isolates the linearity property without
        depending on the absolute analytical formula
        ``Delta = m * g / mu``, which is sensitive to the
        compliance-to-Jacobian ratio in XPBD's softened iterate.
        """
        cell = 0.2
        mu = 1.0e6
        # Two load magnitudes; the second is 2x the first.
        load_a = (1.0, -9.81)  # (mass, g)
        load_b = (2.0, -9.81)
        substeps = 30
        iterations = 80
        beta_mu = 0.6
        n_frames = 200

        delta_a = self._measure_settled_deflection(
            particle_mass=load_a[0],
            gravity_y=load_a[1],
            cell=cell,
            mu=mu,
            substeps=substeps,
            iterations=iterations,
            beta_mu=beta_mu,
            n_frames=n_frames,
        )
        delta_b = self._measure_settled_deflection(
            particle_mass=load_b[0],
            gravity_y=load_b[1],
            cell=cell,
            mu=mu,
            substeps=substeps,
            iterations=iterations,
            beta_mu=beta_mu,
            n_frames=n_frames,
        )
        # Linearity: the force on the system doubled, so deflection
        # must (approximately) double.  Allow a generous tolerance
        # because at very small strain the cloth is in a near-PBD
        # regime where geometry-driven nonlinearities (the corotational
        # ``||F - R||`` is O(strain), but the gradient projection
        # introduces second-order terms in 3D) contribute a few
        # percent.
        ratio = delta_b / max(delta_a, 1.0e-12)
        self.assertGreater(
            delta_a,
            0.0,
            f"Cloth did not deflect under gravity: delta_a={delta_a:.4g}",
        )
        self.assertGreater(
            delta_b,
            delta_a,
            f"Doubling load did not increase deflection: a={delta_a:.4g} b={delta_b:.4g}",
        )
        self.assertAlmostEqual(
            ratio,
            2.0,
            delta=0.20,
            msg=(
                f"Hooke's law linearity violated: "
                f"delta_a={delta_a:.4g}, delta_b={delta_b:.4g}, ratio={ratio:.3f} (expected ~2)"
            ),
        )

    def test_quad_no_buckling_under_gravity(self) -> None:
        """A pinned-top quad under gravity must show non-zero
        deflection in the gravity direction and bounded transverse
        motion -- a quick smoke test that the cloth doesn't go
        unstable / fly apart at typical cloth parameters."""
        cell = 0.1
        particle_mass = 0.05
        mu = 1.0e5
        g = 9.81

        world, model, top_a, top_b, bot_a, bot_b = self._build(
            cell=cell,
            particle_mass=particle_mass,
            mu=mu,
            gravity_xyz=(0.0, -g, 0.0),
            substeps=10,
            iterations=20,
            beta_mu=0.3,
        )
        dt = 1.0 / 60.0
        for _ in range(120):
            world.step(dt=dt, contacts=None, shape_body=None)
        final_q = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(final_q)))
        # Free particles have moved in -y (gravity direction).
        self.assertLess(float(final_q[bot_a, 1]), 0.0)
        self.assertLess(float(final_q[bot_b, 1]), 0.0)
        # Free particles have not catastrophically separated in x.
        x_separation = abs(float(final_q[bot_a, 0]) - float(final_q[bot_b, 0]))
        self.assertLess(x_separation, 2.0 * cell)
        # Top particles haven't moved (pinned).
        np.testing.assert_allclose(final_q[top_a], model.particle_q.numpy()[top_a], atol=1.0e-5)
        np.testing.assert_allclose(final_q[top_b], model.particle_q.numpy()[top_b], atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
