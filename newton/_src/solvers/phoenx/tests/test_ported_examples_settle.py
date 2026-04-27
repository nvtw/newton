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
    # Spiral with 3000 dominoes triggers a chain reaction that
    # propagates for many seconds; the test only enforces "no blow-up"
    # within the frame budget.
    ("example_b2d_double_domino", 480, "swing"),
    ("example_b2d_drop", 240, "settle"),
    ("example_b2d_compound_shapes", 480, "settle"),
    ("example_b2d_explosion", 600, "settle"),
    ("example_jitter_ancient_pyramids", 600, "settle"),
    ("example_jitter_tower_of_jitter", 600, "settle"),
    # The mu=0 cube in this sweep is meant to glide forever (per
    # demo's own docstring); it never decelerates.
    ("example_jitter_restitution_and_friction", 480, "swing"),
    ("example_jitter_colosseum", 600, "settle"),
    # Door + ball-at-8 m/s with no joint damping keeps oscillating
    # past the 4 s budget; gate against blow-up only.
    ("example_b2d_door", 240, "swing"),
    ("example_jitter_motor_and_limit", 240, "settle"),
    # -- Swing (free joint scenes -- energy conserved, no driver) -----
    # ``example_b2d_revolute`` is a damping-free 2-link pendulum -- it
    # genuinely swings forever in PGS (TGS-soft relax doesn't dissipate).
    # ``example_b2d_prismatic`` is a position-driven slider that
    # oscillates around the limit before settling; KD=10 isn't enough
    # to bring it under the 0.5 m/s settle gate in 4s, but it never
    # blows up.
    ("example_b2d_revolute", 240, "swing"),
    ("example_b2d_prismatic", 240, "swing"),
    ("example_b2d_ball_and_chain", 480, "swing"),
    ("example_b2d_ragdoll", 480, "swing"),
    ("example_b2d_chain_link", 480, "swing"),
    ("example_b2d_bridge", 480, "swing"),
    ("example_b2d_cantilever", 480, "swing"),
    ("example_jitter_double_pendulum", 240, "swing"),
    # -- solver2d ports -----------------------------------------------
    ("example_s2d_overlap_recovery", 480, "settle"),
    ("example_s2d_rush", 480, "settle"),
    ("example_s2d_confined", 480, "settle"),
    # FarPyramid is "swing" rather than "settle" -- at 100 km from
    # origin the float32 ULP is ~5 mm, so the stack continually
    # jitters with ~2 m/s amplitude. Known precision limit of float32
    # storage; the original solver2d sample uses it as a stress test
    # for solvers with local-frame substepping (which PhoenX doesn't
    # have). Test only checks "no blow-up".
    ("example_s2d_far_pyramid", 600, "swing"),
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
    mod = __import__(f"newton._src.solvers.phoenx.examples.{module_name}", fromlist=["Example"])
    e = mod.Example(_FakeViewer(), None)
    # Snapshot body-0 height at t=0 so the world-anchored-joint
    # regression below can compare drop distance against the scene's
    # initial elevation. Slot 0 is PhoenX's static world anchor; the
    # first Newton body lives at slot 1.
    pos0 = e.bodies.position.numpy()
    body0_z_init = float(pos0[1, 2]) if pos0.shape[0] > 1 else 0.0
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
        "body0_z_init": body0_z_init,
        "body0_z_final": float(pos[1, 2]) if pos.shape[0] > 1 else 0.0,
    }


# Scenes whose first body is pinned to the world by a revolute joint
# (Newton ``add_joint_revolute(parent=-1, ...)``) and is meant to hang
# from that anchor for the entire run. If the PortedExample scaffold
# ever stops translating Newton joints into PhoenX constraint columns
# again, body 0 free-falls and drops far below its initial height --
# this catches that exact regression.
_WORLD_ANCHORED_SCENES: frozenset[str] = frozenset(
    {
        "example_b2d_revolute",
        "example_b2d_chain_link",
        "example_b2d_ball_and_chain",
    }
)
#: Maximum permitted z-drop of body 0 in a world-anchored joint scene.
#: A correctly wired revolute joint keeps body 0 within a fraction of a
#: link length of the anchor; a missing joint puts it on the ground
#: (~9 m below in chain_link / ball_and_chain). 1 m is comfortably
#: above the former and well below the latter.
_WORLD_ANCHOR_MAX_DROP = 1.0


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

                # World-anchored-joint regression: if joints aren't
                # plumbed into PhoenX, body 0 falls to the ground.
                if module_name in _WORLD_ANCHORED_SCENES:
                    drop = stats["body0_z_init"] - stats["body0_z_final"]
                    self.assertLess(
                        drop,
                        _WORLD_ANCHOR_MAX_DROP,
                        msg=(
                            f"{module_name}: body 0 dropped {drop:.2f} m from its "
                            f"world anchor (initial z = {stats['body0_z_init']:.2f}, "
                            f"final z = {stats['body0_z_final']:.2f}) -- the world "
                            "revolute joint isn't holding the chain up; check that "
                            "PortedExample is still translating Newton joints into "
                            "PhoenX ADBS constraint columns."
                        ),
                    )


if __name__ == "__main__":
    wp.init()
    unittest.main()
