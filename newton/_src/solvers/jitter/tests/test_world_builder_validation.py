# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validation / robustness tests for :class:`WorldBuilder` body ingest.

These tests lock in the contract that ``WorldBuilder.add_body`` rejects
physically-invalid :class:`RigidBodyDescriptor`s *eagerly* (before
finalize) and that :func:`build_jitter_world_from_model` catches the
classic "``add_body(mass=M)`` was silently inflated by shape density"
failure when the caller opts in via ``expected_masses``. The historical
regression this guards against is summarised in
:mod:`test_contact_force_accuracy`: a sphere created with
``add_body(mass=2.0)`` then ``add_shape_sphere(radius=0.1)`` silently
ended up at ``6.188 kg`` (density 1000 * 4/3 pi r^3 ~ 4.189 kg layered
on top), producing a mysterious 3.094x contact-force multiplier that
took significant debugging to trace back to scene-setup.

All checks here run on CPU / host only -- they never finalize a
:class:`World` so Warp / CUDA isn't required.
"""

from __future__ import annotations

import math
import unittest

import warp as wp

import newton
from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
)
from newton._src.solvers.jitter.examples.example_jitter_common import (
    build_jitter_world_from_model,
)
from newton._src.solvers.jitter.world_builder import (
    RigidBodyDescriptor,
    WorldBuilder,
)


_ID = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ZERO = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class TestRigidBodyDescriptorValidation(unittest.TestCase):
    """Descriptor-level rules enforced by ``WorldBuilder.add_body``."""

    def setUp(self) -> None:
        self.wb = WorldBuilder()

    # ------------------------------------------------------------------
    # Happy-path cases (must be accepted)
    # ------------------------------------------------------------------

    def test_default_descriptor_accepted(self) -> None:
        # A bare ``RigidBodyDescriptor()`` must be a valid static body.
        # Historically the defaults mixed STATIC motion with IDENTITY
        # inverse inertia, which is a self-contradiction: this test
        # pins down that the defaults are now internally consistent.
        idx = self.wb.add_body(RigidBodyDescriptor())
        self.assertEqual(idx, 1)  # 0 is the auto-created world body

    def test_static_body_accepted(self) -> None:
        idx = self.wb.add_static_body(position=(1.0, 2.0, 3.0))
        self.assertEqual(idx, 1)

    def test_dynamic_body_accepted(self) -> None:
        idx = self.wb.add_dynamic_body(inverse_mass=0.5, inverse_inertia=_ID)
        self.assertEqual(idx, 1)

    def test_kinematic_body_accepted(self) -> None:
        idx = self.wb.add_kinematic_body(velocity=(1.0, 0.0, 0.0))
        self.assertEqual(idx, 1)

    # ------------------------------------------------------------------
    # Finite-value checks
    # ------------------------------------------------------------------

    def test_nan_position_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite.*position"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    position=(float("nan"), 0.0, 0.0),
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ID,
                )
            )

    def test_inf_velocity_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite.*velocity"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ID,
                    velocity=(float("inf"), 0.0, 0.0),
                )
            )

    # ------------------------------------------------------------------
    # Sign / range checks
    # ------------------------------------------------------------------

    def test_negative_inverse_mass_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"inverse_mass.*must be >= 0"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=-1.0,
                    inverse_inertia=_ID,
                )
            )

    def test_negative_diagonal_inertia_rejected(self) -> None:
        bad = (
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, r"inverse_inertia\[0\]\[0\]"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=bad,
                )
            )

    def test_asymmetric_inertia_rejected(self) -> None:
        bad = (
            (1.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=bad,
                )
            )

    def test_damping_out_of_range_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"linear_damping.*out of range"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ID,
                    linear_damping=2.0,
                )
            )

    def test_non_unit_quaternion_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "orientation.*norm"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    orientation=(1.0, 1.0, 0.0, 0.0),  # norm^2 = 2
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ID,
                )
            )

    # ------------------------------------------------------------------
    # Motion-type cross-checks (the main safety net)
    # ------------------------------------------------------------------

    def test_dynamic_with_zero_inverse_mass_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_type=DYNAMIC.*positive ``inverse_mass``"
        ):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=0.0,
                    inverse_inertia=_ID,
                )
            )

    def test_dynamic_with_zero_inertia_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_type=DYNAMIC.*positive diagonal"
        ):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_DYNAMIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ZERO,
                )
            )

    def test_static_with_mass_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_type=STATIC.*inverse_mass"
        ):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_STATIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ZERO,
                )
            )

    def test_static_with_inertia_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_type=STATIC.*inverse_inertia"
        ):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_STATIC),
                    inverse_mass=0.0,
                    inverse_inertia=_ID,
                )
            )

    def test_static_with_velocity_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_type=STATIC.*velocity"
        ):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_STATIC),
                    inverse_mass=0.0,
                    inverse_inertia=_ZERO,
                    velocity=(1.0, 0.0, 0.0),
                )
            )

    def test_kinematic_with_mass_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_type=KINEMATIC.*inverse_mass"
        ):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=int(MOTION_KINEMATIC),
                    inverse_mass=1.0,
                    inverse_inertia=_ZERO,
                )
            )

    def test_unknown_motion_type_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown motion_type 99"):
            self.wb.add_body(
                RigidBodyDescriptor(
                    motion_type=99,
                    inverse_mass=1.0,
                    inverse_inertia=_ID,
                )
            )


class TestBuildJitterWorldFromModel(unittest.TestCase):
    """Mass-mismatch guard in :func:`build_jitter_world_from_model`.

    This is the actual regression test for the ``add_shape_sphere``
    density-stack bug: if the caller passes ``expected_masses`` and the
    Newton-side mass has been silently inflated by shape density, we
    raise *with actionable guidance* instead of producing a wrong
    downstream simulation.
    """

    def _sphere_model(self, *, mass: float, radius: float, density: float | None):
        mb = newton.ModelBuilder()
        sphere = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.15), q=wp.quat_identity()
            ),
            mass=mass,
        )
        cfg = None
        if density is not None:
            cfg = newton.ModelBuilder.ShapeConfig(density=density)
        if cfg is not None:
            mb.add_shape_sphere(sphere, radius=radius, cfg=cfg)
        else:
            mb.add_shape_sphere(sphere, radius=radius)
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
        return mb.finalize(), sphere

    def test_density_zero_ingests_clean_mass(self) -> None:
        # Sanity: ``density=0`` means the shape adds no mass on top of
        # the explicit ``add_body(mass=...)`` value, so the guard must
        # *not* fire.
        MASS = 2.0
        model, sphere = self._sphere_model(
            mass=MASS, radius=0.1, density=0.0
        )
        # Should not raise.
        builder, n2j = build_jitter_world_from_model(
            model, expected_masses={sphere: MASS}
        )
        self.assertEqual(n2j[sphere], 1)

    def test_default_density_triggers_guard(self) -> None:
        # Default shape density is 1000 kg/m^3 -> a 10 cm sphere adds
        # ~4.189 kg on top, inflating the body to ~6.189 kg. Our guard
        # must fire with a message that names the remedy.
        MASS = 2.0
        model, sphere = self._sphere_model(
            mass=MASS, radius=0.1, density=None
        )
        with self.assertRaises(ValueError) as ctx:
            build_jitter_world_from_model(
                model, expected_masses={sphere: MASS}
            )
        msg = str(ctx.exception)
        self.assertIn("expected_masses", msg)
        self.assertIn("density", msg)  # guidance for remedy
        self.assertIn("ShapeConfig", msg)

    def test_no_expected_masses_opts_out(self) -> None:
        # The guard is strictly opt-in -- if the caller doesn't supply
        # ``expected_masses``, density-stacked scenes build silently
        # (same as before the hardening). Important for back-compat
        # with existing demo scripts that don't care about exact mass.
        MASS = 2.0
        model, _sphere = self._sphere_model(
            mass=MASS, radius=0.1, density=None
        )
        builder, _n2j = build_jitter_world_from_model(model)
        # No exception -- that's the whole assertion.
        self.assertIsNotNone(builder)

    def test_negative_expected_mass_rejected(self) -> None:
        MASS = 2.0
        model, sphere = self._sphere_model(
            mass=MASS, radius=0.1, density=0.0
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_jitter_world_from_model(
                model, expected_masses={sphere: -1.0}
            )


if __name__ == "__main__":
    unittest.main()
