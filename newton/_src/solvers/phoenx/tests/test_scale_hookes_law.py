# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hooke-law smoke tests for prismatic PD limits."""

from __future__ import annotations

import math
import unittest

import warp as wp

from newton._src.solvers.phoenx.tests._test_helpers import run_settle_loop
from newton._src.solvers.phoenx.world_builder import JointMode, WorldBuilder

_GRAVITY = 9.81
_LIMIT_LOWER = -0.05
_LIMIT_UPPER = 0.05
_FPS = 120
_SETTLE_FRAMES = 240
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _settle_slider(*, mass: float, limit_ke: float) -> tuple[float, float]:
    """Return final ``(z, vz)`` for a gravity-loaded limited slider."""
    b = WorldBuilder()
    body = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0 / mass,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=True,
    )
    b.add_joint(
        body1=b.world_body,
        body2=body,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 1.0),
        mode=JointMode.PRISMATIC,
        min_value=_LIMIT_LOWER,
        max_value=_LIMIT_UPPER,
        stiffness_limit=limit_ke,
        damping_limit=2.0 * math.sqrt(limit_ke * mass),
    )
    world = b.finalize(
        substeps=8,
        solver_iterations=16,
        gravity=(0.0, 0.0, -_GRAVITY),
        device=wp.get_preferred_device(),
    )
    run_settle_loop(world, _SETTLE_FRAMES, 1.0 / _FPS)
    position = world.bodies.position.numpy()[1]
    velocity = world.bodies.velocity.numpy()[1]
    return float(position[2]), float(velocity[2])


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX scale tests run on CUDA only (graph capture path).",
)
class TestScaleHookesLaw(unittest.TestCase):
    """Linear PD limit deflection should scale with ``force / ke``."""

    def test_deflection_matches_hookes_law(self) -> None:
        for mass, ke in ((1.0, 200.0), (1.0, 1000.0), (2.0, 5000.0)):
            with self.subTest(mass=mass, ke=ke):
                z, vz = _settle_slider(mass=mass, limit_ke=ke)
                self.assertLess(abs(vz), 1.0e-3, msg=f"slider not settled: vz={vz:.6f} m/s")
                expected = mass * _GRAVITY / ke
                actual = _LIMIT_LOWER - z
                self.assertAlmostEqual(
                    actual,
                    expected,
                    delta=max(0.06 * expected, 2.0e-4),
                    msg=f"deflection={actual:.6f} m, expected={expected:.6f} m for mass={mass}, ke={ke}",
                )

    def test_stiffer_limit_reduces_deflection(self) -> None:
        soft_z, _ = _settle_slider(mass=1.0, limit_ke=200.0)
        stiff_z, _ = _settle_slider(mass=1.0, limit_ke=5000.0)
        soft_deflection = _LIMIT_LOWER - soft_z
        stiff_deflection = _LIMIT_LOWER - stiff_z
        self.assertLess(stiff_deflection, 0.25 * soft_deflection)


if __name__ == "__main__":
    wp.init()
    unittest.main()
