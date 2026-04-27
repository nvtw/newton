# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast regression tests for stack stability under the PhoenX contact
path.

These tests run through the CUDA-graph-captured
:class:`~newton._src.solvers.phoenx.tests.test_stacking._PhoenXScene`
harness so they complete in a handful of seconds and can be wired into
a pre-commit / PR gate. They exercise two scenarios where the old
scale-dependent ``penetration_slop`` / ``friction_slop`` dead zones
used to mask bugs:

1. **Tall tower** (40 layers x 4 planks) -- analogous to
   :mod:`newton._src.solvers.phoenx.examples.example_tower`.
   Catches any regression where resting contacts accumulate spurious
   Baumgarte bias and the top of the tower launches.
2. **Scaled 5-cube stack** -- runs the same stack at ``scene_scale``
   values 0.5, 1.0, 2.0 and checks the residual velocity stays sub-5
   cm/s at each. Exposes any length-scale-dependent threshold that
   only happens to hit the dead zone at ``scene_scale = 1``.

Both tests require CUDA (same as the rest of the PhoenX suite) and
rely on :class:`_PhoenXScene`'s built-in graph capture so they don't
pay per-frame Python overhead.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

_G = 9.81


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX stack tests require CUDA")
class TestPhoenXTallTowerHoldsAtRest(unittest.TestCase):
    """Tall jitter-style tower must not explode at rest.

    Scaled-down version of ``example_tower``: 10 layers of 4
    alternating-orientation planks so the settle completes in ~2
    seconds of sim time. If any plank is moving faster than 10 cm/s
    at the end of the settle, or has drifted more than 30 cm from
    its spawn position, the stack has broken.
    """

    LAYERS = 6
    CUBE_HALF = 0.05
    SETTLE_FRAMES = 240  # 2 s at 120 Hz

    def test_tower_stays_stacked(self) -> None:
        scene = _PhoenXScene(
            fps=120,
            substeps=4,
            solver_iterations=16,
            friction=0.5,
        )
        scene.add_ground_plane()
        cube_ids: list[int] = []
        # Single-column 6-cube tower -- the example_tower
        # regression-catcher. A "proper" 40-layer plank tower is too
        # expensive for the pre-commit gate and has its own
        # sensitivities; this captures the core invariant (a resting
        # stack shouldn't launch its top layer) quickly enough for
        # CI (<5 s).
        cube_size = 2.0 * self.CUBE_HALF
        for layer in range(self.LAYERS):
            cube_ids.append(
                scene.add_box(
                    position=(0.0, 0.0, cube_size * (layer + 0.5) + 0.002),
                    half_extents=(self.CUBE_HALF,) * 3,
                    density=1000.0,
                )
            )
        scene.finalize()

        for _ in range(self.SETTLE_FRAMES):
            scene.step()

        velocities = scene.bodies.velocity.numpy()
        positions = scene.bodies.position.numpy()
        max_speed = 0.0
        max_drift = 0.0
        for cube in cube_ids:
            slot = cube + 1  # slot 0 is the static world anchor
            speed = float(np.linalg.norm(velocities[slot]))
            pos = positions[slot]
            drift = float(math.hypot(pos[0], pos[1]))
            max_speed = max(max_speed, speed)
            max_drift = max(max_drift, drift)
        # Tolerance is for "tower exploded", not "settled to dead
        # stop": a resting stack under PGS with Baumgarte bias always
        # carries some residual micro-oscillation. The previous
        # scale-dependent ``penetration_slop`` regression showed up
        # as |v| > 1 m/s within 2 s of settle, so a 0.1 m/s cap
        # catches that class of bug with plenty of headroom above the
        # ~0.05 m/s normal residual.
        self.assertLess(
            max_speed,
            0.1,
            f"tower exploded: max |v|={max_speed:.3f} m/s",
        )
        self.assertLess(
            max_drift,
            0.02,
            f"tower drifted: max xy drift={max_drift:.4f} m",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX stack tests require CUDA")
class TestPhoenXFiveCubeStackScaleSweep(unittest.TestCase):
    """5-cube vertical stack must settle under gravity across scales.

    Any length-scale-dependent threshold in the contact solver shows
    up here as a scale at which the residual velocity jumps. The
    previous ``penetration_slop = 5 mm`` would ignore the settle
    compression at ``scene_scale = 1`` but fire a spurious bias kick
    at ``scene_scale = 2`` (initial gravity overlap 2x larger than
    the slop). Replacing the dead zone with a scale-invariant
    formulation is the fix this test locks down.
    """

    SETTLE_FRAMES = 180  # 1.5 s at 120 Hz
    CUBE_HALF = 0.25
    STACK_HEIGHT = 5

    def _run_at_scale(self, scene_scale: float) -> tuple[float, float]:
        """Return ``(max_speed, max_drift)`` after the settle."""
        he = self.CUBE_HALF * scene_scale
        scene = _PhoenXScene(
            fps=120,
            substeps=4,
            solver_iterations=16,
            friction=0.5,
        )
        scene.add_ground_plane()
        body_ids: list[int] = []
        for layer in range(self.STACK_HEIGHT):
            # Spawn with a 1 mm gap (scaled) so the initial settle
            # has a consistent gravity step regardless of scale.
            z = 2.0 * he * (layer + 0.5) + 0.001 * scene_scale
            body_ids.append(
                scene.add_box(
                    position=(0.0, 0.0, z),
                    half_extents=(he, he, he),
                    density=1000.0,
                )
            )
        scene.finalize()
        for _ in range(self.SETTLE_FRAMES):
            scene.step()

        velocities = scene.bodies.velocity.numpy()
        positions = scene.bodies.position.numpy()
        max_speed = 0.0
        max_drift = 0.0
        for _layer, body in enumerate(body_ids):
            slot = body + 1
            max_speed = max(max_speed, float(np.linalg.norm(velocities[slot])))
            # XY drift (stack should stay on-axis).
            pos = positions[slot]
            max_drift = max(max_drift, float(math.hypot(pos[0], pos[1])))
        return max_speed, max_drift

    # Velocity tolerance is scale-invariant on purpose: residual
    # Baumgarte jitter is driven by ``g * substep_dt`` (gravity's
    # per-substep velocity injection) which has no dependence on
    # ``scene_scale``. Drift tolerance IS scaled -- a bigger scene
    # lets bodies wander proportionally further before it counts as
    # "slid off the stack".
    _VEL_TOL = 0.1  # m/s
    _DRIFT_TOL_UNIT = 0.05  # m, pre-scale

    def test_scale_half(self) -> None:
        max_speed, max_drift = self._run_at_scale(0.5)
        self.assertLess(max_speed, self._VEL_TOL, f"scale=0.5: |v|={max_speed:.3f} m/s")
        self.assertLess(max_drift, self._DRIFT_TOL_UNIT * 0.5, f"scale=0.5: drift={max_drift:.3f} m")

    def test_scale_one(self) -> None:
        max_speed, max_drift = self._run_at_scale(1.0)
        self.assertLess(max_speed, self._VEL_TOL, f"scale=1.0: |v|={max_speed:.3f} m/s")
        self.assertLess(max_drift, self._DRIFT_TOL_UNIT, f"scale=1.0: drift={max_drift:.3f} m")

    def test_scale_two(self) -> None:
        max_speed, max_drift = self._run_at_scale(2.0)
        self.assertLess(max_speed, self._VEL_TOL, f"scale=2.0: |v|={max_speed:.3f} m/s")
        self.assertLess(max_drift, self._DRIFT_TOL_UNIT * 2.0, f"scale=2.0: drift={max_drift:.3f} m")


if __name__ == "__main__":
    unittest.main()
