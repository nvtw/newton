# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-shape material tests for :mod:`solver_phoenx`.

Ports the jitter-side :mod:`test_materials`: the host-side combine
modes / validation tests copy over verbatim (they never touch the
solver), and the end-to-end friction-from-material checks run
against :class:`PhoenXWorld` through the shared
:class:`~newton._src.solvers.jitter.tests.test_phoenx_stacking._PhoenXScene`
harness.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.materials import (
    COMBINE_AVERAGE,
    COMBINE_MAX,
    COMBINE_MIN,
    COMBINE_MULTIPLY,
    Material,
    material_table_from_list,
)
from newton._src.solvers.jitter.tests.test_phoenx_stacking import _PhoenXScene

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
    :mod:`test_phoenx_friction.TestStaticFrictionThreshold` but
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


if __name__ == "__main__":
    unittest.main()
