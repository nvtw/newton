# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression: every ported Box2D / JitterPhysics2 scene must reach a
solver-stable steady state within a few seconds.

Each scene is classified ``"settle"`` (resting stack / contact-bound
joints, expected to come fully to rest) or ``"swing"`` (free-swinging
chains / pendulums that conserve energy for many seconds). Per-class
final-velocity thresholds catch:

* divergence (NaN, exploding velocities) -- fails both classes
* unstable / jittering stacks -- fails ``settle``
* runaway joint dynamics -- fails ``swing`` only if velocities exceed a
  loose "no blow-up" bound

Importing the example modules directly (with a fake viewer) gives the
test a body-by-body view without duplicating the scene-construction
code.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp


class _FakeViewer:
    """Minimal viewer stub: every public method the ``PortedExample``
    base touches is a no-op so the example builds fully without a
    real GL window."""

    ui = None

    def set_model(self, m):
        pass

    def set_camera(self, **kw):
        pass

    def begin_frame(self, t):
        pass

    def log_state(self, s):
        pass

    def log_contacts(self, c, s):
        pass

    def end_frame(self):
        pass

    def register_ui_callback(self, *a, **kw):
        pass

    def apply_forces(self, s):
        pass


# Per-scene config: (module suffix, frames-to-run, class).
# ``settle`` -- final |v|, |w| must be near zero.
# ``swing``  -- final values must be finite and bounded (no blow-up).
_SCENES: list[tuple[str, int, str]] = [
    # -- Settle (stacks + driven joints) -------------------------------
    ("example_b2d_single_box", 240, "settle"),
    ("example_b2d_vertical_stack", 480, "settle"),
    ("example_b2d_large_pyramid", 600, "settle"),
    ("example_b2d_card_house", 480, "settle"),
    ("example_b2d_friction", 480, "settle"),
    ("example_b2d_restitution", 600, "settle"),
    ("example_b2d_arch", 480, "settle"),
    ("example_b2d_tilted_stack", 600, "settle"),
    ("example_b2d_many_pyramids", 600, "settle"),
    ("example_b2d_circle_stack", 600, "settle"),
    ("example_b2d_double_domino", 480, "settle"),
    ("example_b2d_drop", 240, "settle"),
    ("example_b2d_compound_shapes", 480, "settle"),
    ("example_b2d_explosion", 600, "settle"),
    ("example_jitter_ancient_pyramids", 600, "settle"),
    ("example_jitter_tower_of_jitter", 600, "settle"),
    ("example_jitter_restitution_and_friction", 480, "settle"),
    ("example_jitter_colosseum", 600, "settle"),
    ("example_b2d_door", 240, "settle"),
    ("example_b2d_revolute", 240, "settle"),
    ("example_b2d_prismatic", 240, "settle"),
    ("example_jitter_motor_and_limit", 240, "settle"),
    # -- Swing (free joint scenes -- energy conserved, no driver) -----
    ("example_b2d_ball_and_chain", 480, "swing"),
    ("example_b2d_ragdoll", 480, "swing"),
    ("example_b2d_chain_link", 480, "swing"),
    ("example_b2d_bridge", 480, "swing"),
    ("example_b2d_cantilever", 480, "swing"),
    ("example_jitter_double_pendulum", 240, "swing"),
]

# Steady-state thresholds.
_SETTLE_V = 0.5  # m/s
_SETTLE_W = 1.5  # rad/s
# Free-swinging chains can have meaningful residual motion; fail only
# at clear blow-up (1000x what a sane swing produces).
_SWING_V = 50.0
_SWING_W = 50.0


def _measure_scene(module_name: str, frames: int) -> dict:
    """Build the example, run ``frames`` steps, return final-frame
    state stats over all dynamic bodies."""
    mod = __import__(
        f"newton._src.solvers.phoenx.examples.{module_name}", fromlist=["Example"]
    )
    e = mod.Example(_FakeViewer(), None)
    for _ in range(frames):
        e.step()
    pos = e.bodies.position.numpy()
    vel = e.bodies.velocity.numpy()
    om = e.bodies.angular_velocity.numpy()
    motion_type = e.bodies.motion_type.numpy()
    dyn = motion_type == 2  # MOTION_DYNAMIC
    return {
        "finite": bool(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(om).all()),
        "n_dyn": int(dyn.sum()),
        "max_v": float(np.linalg.norm(vel[dyn], axis=1).max()) if dyn.any() else 0.0,
        "max_w": float(np.linalg.norm(om[dyn], axis=1).max()) if dyn.any() else 0.0,
        "max_pos": float(np.abs(pos).max()),
    }


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX example smoke tests run on CUDA only.",
)
class TestPortedExamplesReachSteadyState(unittest.TestCase):
    """One sub-test per ported scene; each runs the scene to its
    settle-frame budget and asserts the per-class velocity bounds."""

    def test_each_scene_settles(self) -> None:
        for module_name, frames, classification in _SCENES:
            with self.subTest(scene=module_name):
                stats = _measure_scene(module_name, frames)
                self.assertTrue(
                    stats["finite"],
                    msg=f"{module_name}: state went non-finite after {frames} frames",
                )
                if classification == "settle":
                    self.assertLess(
                        stats["max_v"],
                        _SETTLE_V,
                        msg=f"{module_name}: max |v| = {stats['max_v']:.3f} m/s "
                            f"after {frames} frames -- stack jittering or unstable",
                    )
                    self.assertLess(
                        stats["max_w"],
                        _SETTLE_W,
                        msg=f"{module_name}: max |w| = {stats['max_w']:.3f} rad/s "
                            f"after {frames} frames -- stack jittering or unstable",
                    )
                else:
                    # Swing class -- only catch outright blow-up.
                    self.assertLess(
                        stats["max_v"],
                        _SWING_V,
                        msg=f"{module_name}: max |v| = {stats['max_v']:.3f} m/s -- runaway",
                    )
                    self.assertLess(
                        stats["max_w"],
                        _SWING_W,
                        msg=f"{module_name}: max |w| = {stats['max_w']:.3f} rad/s -- runaway",
                    )


if __name__ == "__main__":
    wp.init()
    unittest.main()
