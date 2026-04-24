# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-shape material tests for :mod:`solver_phoenx`.

Ports the jitter-side :mod:`test_materials`: the host-side combine
modes / validation tests copy over verbatim (they never touch the
solver), and the end-to-end friction-from-material checks run
against :class:`PhoenXWorld` through the shared
:class:`~newton._src.solvers.phoenx.tests.test_stacking._PhoenXScene`
harness.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.materials import (
    COMBINE_AVERAGE,
    COMBINE_MAX,
    COMBINE_MIN,
    COMBINE_MULTIPLY,
    Material,
    material_table_from_list,
)
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

_G = 9.81


def _combine_scalar(a: float, b: float, mode: int) -> float:
    """Host-side reference for the combine rules."""
    if mode == COMBINE_MIN:
        return min(a, b)
    if mode == COMBINE_MAX:
        return max(a, b)
    if mode == COMBINE_MULTIPLY:
        return a * b
    return 0.5 * (a + b)


class TestCombineModes(unittest.TestCase):
    """Host-side combine rule checks -- identical to the jitter
    :mod:`test_materials` suite. No device / solver involvement, so
    these tests verify the Python-level :class:`Material` plumbing
    without needing CUDA.
    """

    def test_average(self):
        self.assertAlmostEqual(_combine_scalar(0.2, 0.8, COMBINE_AVERAGE), 0.5)

    def test_min(self):
        self.assertAlmostEqual(_combine_scalar(0.2, 0.8, COMBINE_MIN), 0.2)

    def test_max(self):
        self.assertAlmostEqual(_combine_scalar(0.2, 0.8, COMBINE_MAX), 0.8)

    def test_multiply(self):
        self.assertAlmostEqual(_combine_scalar(0.5, 0.4, COMBINE_MULTIPLY), 0.2)

    def test_material_construction_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            Material(static_friction=-0.1)
        with self.assertRaises(ValueError):
            Material(dynamic_friction=-0.5)
        with self.assertRaises(ValueError):
            Material(restitution=-0.5)
        with self.assertRaises(ValueError):
            Material(restitution=1.5)
        with self.assertRaises(ValueError):
            Material(friction_combine_mode=99)


@unittest.skipUnless(wp.is_cuda_available(), "Material integration test requires CUDA")
class TestContactUsesMaterialFriction(unittest.TestCase):
    """End-to-end: solver friction must track the per-shape material
    table, not ``default_friction``.

    Single cube on a plane, ``default_friction = 0.0`` (so a fallback
    would be obvious), and material table installed via
    :meth:`PhoenXWorld.set_materials`. Applies a constant horizontal
    push of ``0.5 m g`` for 2 s and checks whether the cube slides
    (``mu_effective < 0.5``) or holds (``mu_effective > 0.5``).

    Same arithmetic as
    :mod:`test_friction.TestStaticFrictionThreshold` but
    driven from the material combine rule -- so a regression in the
    pack-kernel friction lookup surfaces here first.
    """

    N_FRAMES = 120

    def _run(self, plane_mu: float, cube_mu: float, combine_mode: int) -> float:
        scene = _PhoenXScene(
            fps=60,
            substeps=4,
            solver_iterations=16,
            friction=0.0,  # intentionally wrong -- material table must win
        )
        scene.add_ground_plane()
        he = 0.5
        box = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
            density=1000.0,
        )
        scene.finalize()

        # Install materials: plane (shape 0) -> material 1,
        # cube (shape 1) -> material 2. Material 0 is the default.
        device = scene.device
        materials = material_table_from_list(
            [
                Material(),
                Material(
                    dynamic_friction=plane_mu,
                    static_friction=plane_mu,
                    friction_combine_mode=combine_mode,
                ),
                Material(
                    dynamic_friction=cube_mu,
                    static_friction=cube_mu,
                    friction_combine_mode=combine_mode,
                ),
            ],
            device=device,
        )
        shape_material = wp.array(
            [1, 2], dtype=wp.int32, device=device
        )  # plane -> mat 1; cube -> mat 2
        scene.install_materials(materials, shape_material)

        # Settle vertical first.
        for _ in range(10):
            scene.step()

        # Apply horizontal push of 0.5 * m * g.
        cube_mass = 1000.0 * (2 * he) ** 3
        push = 0.5 * cube_mass * _G
        for _ in range(self.N_FRAMES):
            scene.apply_body_force(box, force=(push, 0.0, 0.0))
            scene.step()

        return float(scene.body_position(box)[0])

    def test_high_friction_holds(self):
        """mu_plane=0.7, mu_cube=0.7, AVERAGE -> 0.7 > 0.5 push, box stays."""
        x = self._run(0.7, 0.7, COMBINE_AVERAGE)
        self.assertLess(abs(x), 0.1, f"x={x:.4f} m")

    def test_low_friction_slides(self):
        """mu_plane=0.1, mu_cube=0.1, AVERAGE -> 0.1 < push, box slides."""
        x = self._run(0.1, 0.1, COMBINE_AVERAGE)
        self.assertGreater(x, 1.0, f"x={x:.4f} m")

    def test_mixed_min_slides(self):
        """mu=(0.1, 0.9) MIN -> 0.1 (slippery wins), box slides."""
        x = self._run(0.1, 0.9, COMBINE_MIN)
        self.assertGreater(x, 1.0, f"x={x:.4f} m")

    def test_mixed_max_holds(self):
        """mu=(0.1, 0.9) MAX -> 0.9 (grippy wins), box stays."""
        x = self._run(0.1, 0.9, COMBINE_MAX)
        self.assertLess(abs(x), 0.1, f"x={x:.4f} m")

    def test_mixed_multiply_slides(self):
        """mu=(0.3, 0.5) MULTIPLY -> 0.15 < push, box slides."""
        x = self._run(0.3, 0.5, COMBINE_MULTIPLY)
        self.assertGreater(x, 1.0, f"x={x:.4f} m")


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX material tests require CUDA")
class TestPhoenXStaticVsDynamicFriction(unittest.TestCase):
    """Validate that :class:`Material` exposes both ``static_friction``
    and ``dynamic_friction`` and that the contact solver distinguishes
    them in the tangent-row clamp.

    Setup: one cube on a plane with a horizontal push that sits
    *between* the static and kinetic budgets
    (``mu_k * m * g < F_push < mu_s * m * g``). In the static
    regime the cube stays stuck; once slipping, the kinetic clamp
    wouldn't stop it because gravity-along-ramp > mu_k budget.
    Here we replace the ramp with a horizontal plane + constant
    push so the threshold is crisp:

    * ``mu_s = 0.8``, ``mu_k = 0.2``: push at ``F = 0.4 * m * g``.
    * Static: 0.8 >= 0.4 -> cube stays stuck, no slip.
    * If the solver only used ``dynamic_friction = 0.2`` (the
      previous behaviour), 0.2 < 0.4 and the cube would slide.

    A passing test confirms the solver now uses ``static_friction``
    as the stick threshold and the two-regime clamp works.
    """

    def test_static_holds_above_kinetic_budget(self) -> None:
        mu_static = 0.8
        mu_kinetic = 0.2
        # Push at 0.4 * m * g (between kinetic and static budgets).
        push_ratio = 0.4
        scene = _PhoenXScene(
            fps=60, substeps=4, solver_iterations=16, friction=0.0
        )
        scene.add_ground_plane()
        he = 0.5
        box = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
            density=1000.0,
        )
        scene.finalize()
        device = scene.device
        # Install materials: both shapes use a material with the
        # split coefficients and COMBINE_AVERAGE (so the pair
        # friction is just the material's own).
        materials = material_table_from_list(
            [
                Material(),  # index 0 = default (unused)
                Material(
                    static_friction=mu_static,
                    dynamic_friction=mu_kinetic,
                    friction_combine_mode=COMBINE_AVERAGE,
                ),
            ],
            device=device,
        )
        # Plane is shape 0, cube is shape 1; both use material 1.
        scene.install_materials(
            materials,
            wp.array([1, 1], dtype=wp.int32, device=device),
        )
        # Settle the vertical normal first.
        for _ in range(10):
            scene.step()
        cube_mass = 1000.0 * (2 * he) ** 3
        push = push_ratio * cube_mass * 9.81
        for _ in range(120):  # 2 s
            scene.apply_body_force(box, force=(push, 0.0, 0.0))
            scene.step()
        x_final = float(scene.body_position(box)[0])
        # Static regime: cube doesn't slide. Allow 5 cm for numerics.
        self.assertLess(
            abs(x_final),
            0.05,
            f"cube slid to x={x_final:.4f} m with mu_s={mu_static} / "
            f"push_ratio={push_ratio}. The static threshold "
            f"({mu_static} * m * g) > push, so the cube must stay "
            "stuck -- if it slid, the solver is still clamping at "
            f"mu_k={mu_kinetic} instead of the static cone.",
        )

    def test_kinetic_clamp_applies_once_slipping(self) -> None:
        """Above the static threshold the cube slides with acceleration
        set by ``mu_kinetic``, not ``mu_static``. Push at ``1.2 *
        mu_static * m * g`` to guarantee the slide starts; measure
        deceleration-after-launch and check it matches
        ``mu_kinetic * g`` to within 30%.
        """
        mu_static = 0.5
        mu_kinetic = 0.1
        scene = _PhoenXScene(
            fps=120, substeps=8, solver_iterations=16, friction=0.0
        )
        scene.add_ground_plane()
        he = 0.5
        box = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
            density=1000.0,
        )
        scene.finalize()
        device = scene.device
        materials = material_table_from_list(
            [
                Material(),
                Material(
                    static_friction=mu_static,
                    dynamic_friction=mu_kinetic,
                    friction_combine_mode=COMBINE_AVERAGE,
                ),
            ],
            device=device,
        )
        scene.install_materials(
            materials,
            wp.array([1, 1], dtype=wp.int32, device=device),
        )
        # Settle; then launch the cube with a high initial velocity
        # so it's well past the static threshold and in pure kinetic
        # regime. Measure deceleration.
        for _ in range(10):
            scene.step()
        scene.set_body_velocity(box, (5.0, 0.0, 0.0))
        for _ in range(3):
            scene.step()
        v_start = float(scene.body_velocity(box)[0])
        measure_frames = 30
        for _ in range(measure_frames):
            scene.step()
        v_end = float(scene.body_velocity(box)[0])
        measured_decel = (v_start - v_end) / (measure_frames / 120.0)
        expected_decel_kinetic = mu_kinetic * 9.81
        expected_decel_static = mu_static * 9.81
        # Decel must be near the kinetic value, clearly distinct from
        # the static value. If the solver were still using only
        # dynamic_friction, measured_decel would also match kinetic.
        # If it only used static_friction, measured_decel would match
        # static (~5x larger). So we bound away from static.
        self.assertLess(
            measured_decel,
            0.5 * (expected_decel_kinetic + expected_decel_static),
            f"sliding cube decel {measured_decel:.3f} m/s^2 is closer "
            f"to the static friction value ({expected_decel_static:.3f}) "
            f"than to the kinetic value ({expected_decel_kinetic:.3f}) "
            "-- the solver is using mu_static when it should be using "
            "mu_kinetic.",
        )


if __name__ == "__main__":
    unittest.main()
