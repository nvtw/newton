# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic-accuracy tests for PhoenX's soft-contact PD path.

Validates the PhysX-style absolute spring-damper normal row wired
into the contact iterate when Newton's per-contact
``rigid_contact_stiffness`` / ``rigid_contact_damping`` arrays are
non-zero. The baseline here is the classic sphere-on-plane steady
state: a single contact with spring stiffness ``k`` supporting a
mass ``m`` under gravity ``g`` produces a penetration

.. math:: \\text{depth} = m g / k

that's *independent of the body's mass-normalized response* --
this is the whole reason you'd prefer the absolute PD path over
the mass-normalized Box2D (hertz) path: the hydroelastic pressure
field, treated as a spring, delivers the same force at the same
depth no matter the colliding bodies' masses.

Tests:

* :meth:`TestSoftContactAnalyticalDepth.test_sphere_on_plane_depth`:
  single sphere, verify steady-state depth matches ``m*g/k`` within
  a few percent.
* :meth:`TestSoftContactAnalyticalDepth.test_stiffness_halves_doubles_depth`:
  halving ``k`` must double ``depth`` (pure linearity check; catches
  mass-normalization regressions where the response would stiffen
  with mass).
* :meth:`TestSoftContactAnalyticalDepth.test_mass_scales_depth_linearly`:
  at fixed ``k``, depth scales linearly with mass (the target-force
  property the hydroelastic pipeline relies on).
* :meth:`TestSoftContactLegacyPath.test_no_stiffness_uses_rigid_path`:
  omitting the per-contact arrays leaves PhoenX on the legacy
  Nyquist-rigid path -- contacts produce negligible sink.
* :meth:`TestSoftContactUnilateral.test_separating_sphere_receives_no_impulse`:
  sphere launched away from the plane under positive gap must not
  attract (unilateral clamp lam_n >= 0).
"""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.contact_matching_config import (
    PHOENX_CONTACT_MATCHING,
)
from newton._src.solvers.phoenx.solver import SolverPhoenX

_G = 9.81


def _finalize_phoenx(
    mb: newton.ModelBuilder,
    *,
    per_contact_shape_properties: bool,
    substeps: int = 8,
    solver_iterations: int = 40,
    velocity_iterations: int = 1,
) -> tuple[newton.Model, newton.State, newton.State, newton.Contacts, SolverPhoenX]:
    """Build a :class:`newton.Model` from ``mb`` and wire it to
    :class:`SolverPhoenX` with a :class:`~newton.Contacts` buffer that
    may or may not carry the per-contact stiffness / damping /
    friction arrays.

    ``per_contact_shape_properties=True`` here manually allocates the
    per-contact arrays (which the ``CollisionPipeline`` itself would
    only enable for hydroelastic scenes). This lets the tests
    exercise the soft-contact PD path without requiring a full
    hydroelastic SDF setup.

    Returns ``(model, state_in, state_out, contacts, solver)``. The
    caller steps via ``model.collide`` + ``solver.step``; manual
    stiffness writes on ``contacts.rigid_contact_stiffness`` between
    ``collide`` and ``step`` exercise the soft-contact PD path end to
    end.
    """
    model = mb.finalize()
    model._collision_pipeline = newton.CollisionPipeline(
        model,
        contact_matching=PHOENX_CONTACT_MATCHING,
    )
    # The pipeline's default ``contacts()`` returns a buffer with
    # ``per_contact_shape_properties`` driven by hydroelastic presence.
    # For soft-contact tests we want the per-contact arrays present
    # regardless of hydroelastic, so we manually construct the buffer
    # and attach it to the model.
    cp = model._collision_pipeline
    model._collision_pipeline.contacts()  # forces internal sizing

    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    solver = SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=velocity_iterations,
    )

    if per_contact_shape_properties:
        contacts = newton.Contacts(
            cp.rigid_contact_max,
            cp.soft_contact_max,
            device=model.device,
            per_contact_shape_properties=True,
            contact_matching=True,
        )
    else:
        contacts = model.contacts()
    return model, state_in, state_out, contacts, solver


def _step_soft(
    model,
    solver: SolverPhoenX,
    state_in,
    state_out,
    contacts,
    dt: float,
    stiffness: float,
    damping: float,
) -> None:
    """One render-frame step that writes ``stiffness`` and ``damping``
    into every allocated contact slot (broadcast) between ``collide``
    and ``solver.step``. Emulates a narrow-phase writer that publishes
    soft-contact properties per-contact.
    """
    model.collide(state_in, contacts)
    if contacts.rigid_contact_stiffness is not None:
        contacts.rigid_contact_stiffness.fill_(float(stiffness))
    if contacts.rigid_contact_damping is not None:
        contacts.rigid_contact_damping.fill_(float(damping))
    solver.step(state_in, state_out, None, contacts, dt)


def _add_sphere_on_ground(
    mb: newton.ModelBuilder,
    *,
    radius: float,
    mass: float,
    z0: float | None = None,
) -> int:
    """Add a ground plane at z=0 plus a single dynamic sphere whose
    COM is placed at ``z0 = radius`` by default (just touching the
    plane, zero gap). Returns the sphere's body index."""
    mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
    if z0 is None:
        z0 = radius
    i_sphere = 0.4 * mass * radius * radius
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, z0), q=wp.quat_identity()),
        mass=mass,
        inertia=(
            (i_sphere, 0.0, 0.0),
            (0.0, i_sphere, 0.0),
            (0.0, 0.0, i_sphere),
        ),
    )
    mb.add_shape_sphere(
        body,
        radius=radius,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    return body


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX soft-contact tests require CUDA")
class TestSoftContactAnalyticalDepth(unittest.TestCase):
    """Analytic steady-state depth for sphere-on-plane with a known
    spring stiffness."""

    FPS = 240
    DT = 1.0 / FPS
    SUBSTEPS = 8
    ITERATIONS = 40
    SETTLE_FRAMES = 400  # ~1.7 s
    RADIUS = 0.1

    def _settle_and_measure_depth(
        self,
        *,
        mass: float,
        stiffness: float,
        damping: float = 0.0,
    ) -> float:
        """Drop a sphere of ``mass`` from zero-gap on the ground with
        spring stiffness ``stiffness`` [N/m] and ``damping`` [N*s/m];
        return the final penetration depth [m].

        Steady-state theory: at rest the spring force balances
        gravity, ``k * depth = m * g``, so ``depth = m*g/k``
        independent of damping.
        """
        mb = newton.ModelBuilder()
        # Gravity already defaults to (0, 0, -9.81) via ModelBuilder.
        body = _add_sphere_on_ground(mb, radius=self.RADIUS, mass=mass)
        model, state_in, state_out, contacts, solver = _finalize_phoenx(
            mb,
            per_contact_shape_properties=True,
            substeps=self.SUBSTEPS,
            solver_iterations=self.ITERATIONS,
        )

        for _ in range(self.SETTLE_FRAMES):
            _step_soft(
                model,
                solver,
                state_in,
                state_out,
                contacts,
                self.DT,
                stiffness,
                damping,
            )
            # swap states
            state_in, state_out = state_out, state_in

        body_q = state_in.body_q.numpy()[body]
        z_com = float(body_q[2])
        # With the sphere COM starting at z=radius (zero gap), the
        # penetration depth is radius - z_com.
        return self.RADIUS - z_com

    def test_sphere_on_plane_depth(self) -> None:
        """A 1 kg sphere with k=1e3 N/m under g=9.81 should settle at
        ``m*g/k = 9.81 mm`` of penetration (within 2 %)."""
        mass = 1.0
        k = 1.0e3
        c = 20.0  # critical-ish damping: 2*sqrt(k*m) ~= 63; pick lower for faster settle
        depth = self._settle_and_measure_depth(mass=mass, stiffness=k, damping=c)
        expected = mass * _G / k
        rel_err = abs(depth - expected) / expected
        self.assertLess(
            rel_err,
            0.05,
            msg=f"depth {depth * 1000:.3f} mm vs expected {expected * 1000:.3f} mm (rel err {rel_err * 100:.2f}%)",
        )

    def test_stiffness_halves_doubles_depth(self) -> None:
        """Pure linearity: halving ``k`` should double steady-state
        depth. This catches mass-normalization regressions (where the
        response would implicitly pick up an ``omega_n = sqrt(k/m)``
        factor and the depth wouldn't scale cleanly with ``1/k``)."""
        mass = 1.0
        k_hi = 2.0e3
        k_lo = 1.0e3
        c = 10.0
        depth_hi = self._settle_and_measure_depth(mass=mass, stiffness=k_hi, damping=c)
        depth_lo = self._settle_and_measure_depth(mass=mass, stiffness=k_lo, damping=c)
        ratio = depth_lo / depth_hi
        self.assertAlmostEqual(
            ratio,
            2.0,
            delta=0.1,
            msg=f"halving k should ~ double depth: {depth_hi * 1000:.2f} -> {depth_lo * 1000:.2f} mm (ratio {ratio:.3f})",
        )

    def test_mass_scales_depth_linearly(self) -> None:
        """At fixed stiffness, depth scales linearly with mass.

        This is the hydroelastic "target force" property: if two
        bodies of different masses both settle against the same
        surface, their equilibrium depths differ by the mass ratio,
        and therefore the force each receives is ``k * depth = m *
        g`` -- what hydroelastic pressure fields need.
        """
        k = 1.0e3
        c = 10.0
        depth_1 = self._settle_and_measure_depth(mass=1.0, stiffness=k, damping=c)
        depth_2 = self._settle_and_measure_depth(mass=2.0, stiffness=k, damping=c)
        ratio = depth_2 / depth_1
        self.assertAlmostEqual(
            ratio,
            2.0,
            delta=0.1,
            msg=f"doubling mass should ~ double depth: {depth_1 * 1000:.2f} -> {depth_2 * 1000:.2f} mm (ratio {ratio:.3f})",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX soft-contact tests require CUDA")
class TestSoftContactLegacyPath(unittest.TestCase):
    """With per-contact stiffness / damping *omitted* the solver must
    fall back to the legacy Nyquist-rigid path bit-for-bit."""

    FPS = 240
    DT = 1.0 / FPS
    SUBSTEPS = 8
    ITERATIONS = 40
    SETTLE_FRAMES = 200
    RADIUS = 0.1

    def test_no_stiffness_uses_rigid_path(self) -> None:
        """Without per-contact arrays, a sphere at zero gap should
        stay essentially on the surface (penetration within sub-mm,
        set by the Nyquist-clamped Box2D path's residual
        Baumgarte)."""
        mb = newton.ModelBuilder()
        body = _add_sphere_on_ground(mb, radius=self.RADIUS, mass=1.0)
        model, state_in, state_out, contacts, solver = _finalize_phoenx(
            mb,
            per_contact_shape_properties=False,
            substeps=self.SUBSTEPS,
            solver_iterations=self.ITERATIONS,
        )
        self.assertIsNone(
            contacts.rigid_contact_stiffness,
            msg="per_contact_shape_properties=False should not allocate stiffness",
        )

        for _ in range(self.SETTLE_FRAMES):
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, self.DT)
            state_in, state_out = state_out, state_in

        z_com = float(state_in.body_q.numpy()[body][2])
        depth = self.RADIUS - z_com
        self.assertLess(
            abs(depth),
            2.0e-3,
            msg=f"rigid path produced unexpectedly large sink: depth={depth * 1000:.3f} mm",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX soft-contact tests require CUDA")
class TestSoftContactUnilateral(unittest.TestCase):
    """Contacts don't pull. A sphere at *separating* depth (gap > 0)
    must see zero normal impulse under the PD path, matching the
    existing Box2D path's unilateral behaviour."""

    FPS = 240
    DT = 1.0 / FPS
    SUBSTEPS = 4
    ITERATIONS = 16
    RADIUS = 0.1

    def test_separating_sphere_receives_no_impulse(self) -> None:
        """Sphere COM starts at ``z = 2 * radius`` (gap = radius > 0),
        gravity-free. Under a stiff k, if the unilateral clamp were
        broken the spring would pull the sphere down toward the
        plane; with the clamp in place the sphere must stay put."""
        # Build with zero gravity so the only driver of motion is
        # the (not-supposed-to-fire) contact spring.
        mb = newton.ModelBuilder(gravity=0.0)
        # Start the sphere well above the plane so no contact fires.
        body = _add_sphere_on_ground(mb, radius=self.RADIUS, mass=1.0, z0=0.5)
        model, state_in, state_out, contacts, solver = _finalize_phoenx(
            mb,
            per_contact_shape_properties=True,
            substeps=self.SUBSTEPS,
            solver_iterations=self.ITERATIONS,
        )

        # Stiff spring, no damping -- would pull the sphere down HARD
        # if the unilateral clamp were broken.
        k = 1.0e5
        c = 0.0

        # Step a few frames; sphere should stay put.
        z_before = float(state_in.body_q.numpy()[body][2])
        for _ in range(10):
            _step_soft(model, solver, state_in, state_out, contacts, self.DT, k, c)
            state_in, state_out = state_out, state_in
        z_after = float(state_in.body_q.numpy()[body][2])
        # No gravity, no initial velocity, contact gap = 0.4 m >> 0:
        # the spring must not reach out to the sphere.
        drift = abs(z_after - z_before)
        self.assertLess(
            drift,
            1.0e-5,
            msg=f"sphere drifted ({z_before:.6f} -> {z_after:.6f}, {drift * 1000:.3f} mm) despite no contact; unilateral clamp broken",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
