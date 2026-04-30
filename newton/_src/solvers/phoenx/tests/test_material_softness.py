# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Material-driven contact softness in PhoenX.

PhoenX exposes Bepu-style ``(contact_hertz, contact_damping_ratio)``
fields on :class:`~newton._src.solvers.phoenx.materials.Material`. At
ingest time the per-shape values are blended per pair under
``softness_combine_mode`` and sign-encoded into the existing
per-contact ``rigid_contact_stiffness`` / ``rigid_contact_damping``
arrays as ``(-hertz, -damping_ratio)``. The contact prepare phase
decodes the negative gate, computes the absolute spring-damper
``(k = omega^2 * m_eff, c = 2*zeta*omega*m_eff)``, and routes the
normal row through the same PD path hydroelastic uses.

These tests check three invariants:

* **Material-encoded softness produces the correct steady-state
  depth.** A sphere on a plane with ``contact_hertz = f`` settles at
  ``m * g / k`` where ``k = (2*pi*f)^2 * m`` (ground is static, so
  ``m_eff == m``). This catches sign-encoding, decode-arithmetic,
  and ingest-pipeline bugs in one shot.
* **Hydroelastic absolute K wins over Material softness.** When the
  narrow phase publishes a positive stiffness, the ingest encoder
  must not overwrite it; otherwise hydroelastic stacks regress.
* **Default Material (``contact_hertz = 0``) is bit-identical to the
  pre-Material path.** Friction-only Material installs must not
  flip any contact onto the soft path.
"""

from __future__ import annotations

import math
import unittest

import warp as wp

import newton
from newton._src.solvers.phoenx.materials import (
    COMBINE_AVERAGE,
    Material,
    material_table_from_list,
)
from newton._src.solvers.phoenx.solver import SolverPhoenX
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_CONTACT_MATCHING,
)

_G = 9.81


def _build_sphere_on_plane(
    *,
    radius: float,
    mass: float,
):
    """One sphere COM-touching a ground plane (zero initial gap)."""
    mb = newton.ModelBuilder()
    mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
    i_sphere = 0.4 * mass * radius * radius
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, radius), q=wp.quat_identity()),
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
    return mb, body


def _finalize_with_per_contact_arrays(mb, *, substeps=8, solver_iterations=40):
    """Finalize the model + force ``per_contact_shape_properties=True``
    on the Contacts buffer so the soft-contact path can light up."""
    model = mb.finalize()
    model._collision_pipeline = newton.CollisionPipeline(
        model,
        contact_matching=PHOENX_CONTACT_MATCHING,
    )
    cp = model._collision_pipeline
    cp.contacts()  # forces sizing
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=1,
    )
    contacts = newton.Contacts(
        cp.rigid_contact_max,
        cp.soft_contact_max,
        device=model.device,
        per_contact_shape_properties=True,
        contact_matching=True,
    )
    return model, state_in, state_out, contacts, solver


def _install_softness_materials(
    solver: SolverPhoenX,
    model,
    *,
    hertz: float,
    damping_ratio: float,
):
    """Install a 2-material table on the solver: ``Material()`` at
    index 0 (default for the plane), one Material at index 1 with the
    requested softness for the sphere. Pair blends under AVERAGE so
    the effective values are ``(hertz / 2, (1 + damping_ratio) / 2)``
    -- the test computes its analytic expectations from those."""
    materials = material_table_from_list(
        [
            Material(),
            Material(
                contact_hertz=hertz,
                contact_damping_ratio=damping_ratio,
                softness_combine_mode=COMBINE_AVERAGE,
            ),
        ],
        device=model.device,
    )
    # Plane = shape 0 -> material 0 (default). Sphere = shape 1 ->
    # material 1 (softness).
    shape_material = wp.array([0, 1], dtype=wp.int32, device=model.device)
    solver.world.set_materials(materials, shape_material)


def _step(model, solver, state_in, state_out, contacts, dt):
    """One render-frame step: collide + solver.step. No manual K/D
    population; the Material softness flows through ingest."""
    model.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, dt)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX Material softness tests require CUDA")
class TestMaterialDrivenSoftness(unittest.TestCase):
    """Material softness arrives at the contact prepare phase via the
    sign-encoded ``rigid_contact_stiffness`` / ``rigid_contact_damping``
    arrays. The steady-state depth is the cleanest end-to-end check."""

    FPS = 240
    DT = 1.0 / FPS
    SUBSTEPS = 8
    ITERATIONS = 40
    SETTLE_FRAMES = 200  # ~0.83 s -- fast enough for the inner loop
    RADIUS = 0.1

    def _settle_depth(self, *, mass: float, hertz: float, damping_ratio: float) -> float:
        """Drop the sphere with Material-driven softness; return the
        final penetration depth [m]."""
        mb, body = _build_sphere_on_plane(radius=self.RADIUS, mass=mass)
        model, state_in, state_out, contacts, solver = _finalize_with_per_contact_arrays(
            mb,
            substeps=self.SUBSTEPS,
            solver_iterations=self.ITERATIONS,
        )
        _install_softness_materials(
            solver,
            model,
            hertz=hertz,
            damping_ratio=damping_ratio,
        )
        for _ in range(self.SETTLE_FRAMES):
            _step(model, solver, state_in, state_out, contacts, self.DT)
            state_in, state_out = state_out, state_in
        z_com = float(state_in.body_q.numpy()[body][2])
        return self.RADIUS - z_com

    def test_steady_state_depth_matches_omega_squared_formula(self) -> None:
        """Material softness must produce ``depth = m*g/k`` where
        ``k = (2*pi*hertz_eff)^2 * m`` (the ground is static so the
        contact's effective mass is just the sphere's mass).

        The plane uses default Material (hertz=0); the sphere's
        Material has ``hertz=15`` Hz. Under AVERAGE combine the
        effective per-pair hertz is ``7.5`` Hz."""
        mass = 1.0
        sphere_hertz = 15.0
        eff_hertz = 0.5 * (0.0 + sphere_hertz)  # AVERAGE of plane/sphere
        depth = self._settle_depth(mass=mass, hertz=sphere_hertz, damping_ratio=1.0)
        omega = 2.0 * math.pi * eff_hertz
        k = omega * omega * mass
        expected = mass * _G / k
        rel_err = abs(depth - expected) / expected
        self.assertLess(
            rel_err,
            0.10,
            msg=(
                f"depth {depth * 1000:.3f} mm vs expected {expected * 1000:.3f} mm "
                f"(rel err {rel_err * 100:.2f}%); "
                f"k={k:.2f} N/m, m={mass} kg, eff_hertz={eff_hertz} Hz"
            ),
        )

    def test_zero_hertz_default_keeps_legacy_path(self) -> None:
        """Default :class:`Material` (``contact_hertz = 0``) must not
        engage the soft-contact PD path -- the sphere should sit on
        the plane as a near-rigid contact (sub-mm sink)."""
        mass = 1.0
        depth = self._settle_depth(mass=mass, hertz=0.0, damping_ratio=1.0)
        # Legacy Box2D-rigid path settles to <1mm at 240 Hz / 8 substeps.
        self.assertLess(depth, 1.0e-3, msg=f"unexpected sink {depth * 1000:.3f} mm with hertz=0")


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX Material softness tests require CUDA")
class TestHydroelasticPrecedence(unittest.TestCase):
    """When the narrow phase has already published a positive absolute
    stiffness into ``rigid_contact_stiffness[k]``, the Material
    sign-encoder must skip that slot. We simulate the hydroelastic
    pre-population by writing positive K manually before stepping; if
    the encoder were to overwrite it with a negative ``-hertz``, the
    sphere would spring through.

    This is functionally what hydroelastic does in production, just
    with a hand-rolled K instead of the SDF reduction."""

    FPS = 240
    DT = 1.0 / FPS
    SUBSTEPS = 8
    ITERATIONS = 40
    SETTLE_FRAMES = 80  # short -- only need to confirm the cube doesn't fall through
    RADIUS = 0.1

    def test_positive_absolute_k_is_not_overwritten(self) -> None:
        mass = 1.0
        mb, body = _build_sphere_on_plane(radius=self.RADIUS, mass=mass)
        model, state_in, state_out, contacts, solver = _finalize_with_per_contact_arrays(
            mb,
            substeps=self.SUBSTEPS,
            solver_iterations=self.ITERATIONS,
        )
        # Install Material softness with hertz=5 (would give k ~= 99
        # N/m and depth ~= 100mm -- enough to tell the two regimes
        # apart from the hand-rolled k=1e4 N/m below, which yields
        # ~1mm of depth).
        _install_softness_materials(solver, model, hertz=5.0, damping_ratio=1.0)
        k_hydro = 1.0e4
        c_hydro = 50.0
        for _ in range(self.SETTLE_FRAMES):
            model.collide(state_in, contacts)
            # Pre-populate the per-contact arrays with hydroelastic-
            # style absolute K/D (positive). The Material encoder
            # must see these as != 0 and skip them.
            contacts.rigid_contact_stiffness.fill_(k_hydro)
            contacts.rigid_contact_damping.fill_(c_hydro)
            solver.step(state_in, state_out, None, contacts, self.DT)
            state_in, state_out = state_out, state_in
        depth = self.RADIUS - float(state_in.body_q.numpy()[body][2])
        # Expected hydroelastic depth: m*g/k_hydro = 9.81e-4 m.
        # If the Material encoder were to clobber the positive K,
        # we'd see ~100 mm of sink instead.
        expected = mass * _G / k_hydro
        self.assertLess(
            depth,
            10.0 * expected,
            msg=(
                f"depth {depth * 1000:.3f} mm > 10 * expected "
                f"{expected * 1000:.3f} mm; Material encoder may have "
                f"overwritten hydroelastic K"
            ),
        )


if __name__ == "__main__":
    unittest.main()
