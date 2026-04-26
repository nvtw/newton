# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""High mass-ratio stability tests for :class:`PhoenXWorld`.

PGS solvers can struggle when bodies in contact differ in mass by orders
of magnitude: the impulse a heavy body applies on a light one must
resolve in a single substep, but the corresponding velocity correction
on the light body is huge (``acc * dt / m_small`` -> blow-up). Box2D's
classic ``Heavy On Light`` test is the standard regression -- a few
heavy bodies stacked on light ones should not crush them.

Three scenarios:

* :class:`TestHeavyOnLightStack` -- a 100x heavy cube rests on a 1x
  cube. The light cube must hold up the heavy one (no NaN, no
  penetration); the plane must carry both weights.
* :class:`TestLightOnHeavyStack` -- inverse: a 1x cube rests on a 100x
  cube. Stack must stay stable; the heavy cube barely moves under the
  light one's weight.
* :class:`TestHeavyPendulum` -- a 1000:1 mass-ratio pendulum: heavy
  hub holding a light bob via a 1 m revolute joint. Natural period
  must match ``2*pi*sqrt(L/g)`` (small-angle approximation) within
  ~5% -- a broken solver typically inflates the period or damps it.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.scene_registry import Scene, scene
from newton._src.solvers.phoenx.tests._test_helpers import STEP_LAYOUTS, run_settle_loop
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene
from newton._src.solvers.phoenx.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

GRAVITY = 9.81
HE = 0.5  # cube half-extent for stack tests
SETTLE_FRAMES_STACK = 240  # 4 s @ 60 Hz, plenty for any settle


# ---------------------------------------------------------------------------
# Heavy-on-light stack (Box2D-style)
# ---------------------------------------------------------------------------


def _plane_pair_fz_to_body(scene, body_newton_idx: int) -> float:
    """Extract the +z component of the plane->body contact force from
    the per-pair wrench arrays. The plane is PhoenX slot 0 (the static
    world body)."""
    pw, b1, b2, cnt = scene.gather_pair_wrenches_raw()
    body_slot = body_newton_idx + 1
    fz = 0.0
    for i in range(len(cnt)):
        if cnt[i] <= 0:
            continue
        if b1[i] == 0 and b2[i] == body_slot:
            fz += float(pw[i, 2])
        elif b1[i] == body_slot and b2[i] == 0:
            fz -= float(pw[i, 2])
    return fz


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping")
class TestHeavyOnLightStack(unittest.TestCase):
    """A heavy cube resting on a light cube must not crush it.

    Box2D's HighMassRatio2 sample (big box on small boxes) is the
    canonical regression for this configuration. Box2D-v3 handles
    400:1 mass ratios with its default 4 TGS substeps; PGS solvers
    like PhoenX need more (here 40 substeps x 32 iters). The test
    verifies the 100:1 case stably stacks and contact forces propagate
    to the plane.
    """

    def test_100x_on_1x(self) -> None:
        """100 kg cube atop a 1 kg cube on the plane.

        Settled state: bottom cube at z ~= HE, top at z ~= 3*HE; both
        velocities ~ 0. The plane->bottom pair force must equal
        (1 + 100)*g = 990.81 N.
        """
        scene = _PhoenXScene(substeps=40, solver_iterations=32)
        scene.add_ground_plane()
        bottom = scene.add_box(
            position=(0.0, 0.0, HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=1.0,
        )
        top = scene.add_box(
            position=(0.0, 0.0, 3 * HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=100.0,
        )
        scene.finalize()

        for _ in range(SETTLE_FRAMES_STACK):
            scene.step()

        p_bottom = scene.body_position(bottom)
        p_top = scene.body_position(top)
        v_bottom = scene.body_velocity(bottom)
        v_top = scene.body_velocity(top)

        # No NaN.
        self.assertTrue(np.isfinite(p_bottom).all())
        self.assertTrue(np.isfinite(p_top).all())

        # Z ordering: top must remain above bottom.
        self.assertGreater(
            float(p_top[2]),
            float(p_bottom[2]) + HE,
            msg=f"top sank into bottom: top.z={p_top[2]:.3f}, bottom.z={p_bottom[2]:.3f}",
        )

        # Both at rest -- velocity tolerance scaled for the heavy cube
        # since the same residual impulse moves a 1 kg cube 100x faster
        # than a 100 kg one.
        self.assertLess(float(np.linalg.norm(v_bottom)), 0.05)
        self.assertLess(float(np.linalg.norm(v_top)), 0.01)

        # Settled heights (allow a generous slack -- light cube can
        # compress slightly under 100x weight in soft-contact mode).
        self.assertAlmostEqual(float(p_bottom[2]), HE, delta=0.2)
        self.assertAlmostEqual(float(p_top[2]), 3 * HE, delta=0.3)

        # XY drift: heavy cube must not squirt out sideways.
        self.assertLess(float(np.hypot(p_top[0], p_top[1])), 0.2)

        # System-net invariant: sum of contact forces on every cube
        # must equal the system weight (Newton 3rd law on the plane).
        F_top, _, _ = scene.gather_contact_wrench_on_body(top)
        F_bot, _, _ = scene.gather_contact_wrench_on_body(bottom)
        net_fz = float(F_top[2]) + float(F_bot[2])
        expected_net = (1.0 + 100.0) * GRAVITY
        self.assertAlmostEqual(
            net_fz,
            expected_net,
            delta=0.05 * expected_net,
            msg=f"system net Fz = {net_fz:.2f} N vs (m_top+m_bot)*g = {expected_net:.2f} N",
        )

        # Cube-cube contact must transmit the heavy cube's full weight
        # downward. The top cube only touches the bottom cube, so its
        # gathered contact force IS the cube-cube force.
        self.assertAlmostEqual(
            float(F_top[2]),
            100.0 * GRAVITY,
            delta=0.05 * 100.0 * GRAVITY,
            msg=f"cube-cube contact Fz = {F_top[2]:.2f} N vs m_top*g = "
                f"{100.0 * GRAVITY:.2f} N -- heavy cube's weight not propagating",
        )

        # Plane->bottom pair force: must equal full stack weight.
        plane_fz = _plane_pair_fz_to_body(scene, bottom)
        self.assertAlmostEqual(
            plane_fz,
            expected_net,
            delta=0.05 * expected_net,
            msg=f"plane->bottom pair Fz = {plane_fz:.2f} N vs "
                f"(m_top+m_bot)*g = {expected_net:.2f} N",
        )


# ---------------------------------------------------------------------------
# Light-on-heavy stack
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping")
class TestLightOnHeavyStack(unittest.TestCase):
    """A light cube resting on a heavy cube. The mirror configuration:
    here the bottom cube has effectively infinite stiffness against the
    light cube, so the dominant failure mode is the heavy cube
    accelerating downward through its own contact bias."""

    def test_1x_on_100x(self) -> None:
        """1 kg cube atop a 100 kg cube.

        Cube-cube contact must carry m_top*g; system net Fz on both
        bodies must equal (m_top+m_bot)*g; plane-bottom pair Fz must
        match the system weight.
        """
        scene = _PhoenXScene(substeps=20, solver_iterations=12)
        scene.add_ground_plane()
        bottom = scene.add_box(
            position=(0.0, 0.0, HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=100.0,
        )
        top = scene.add_box(
            position=(0.0, 0.0, 3 * HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=1.0,
        )
        scene.finalize()

        for _ in range(SETTLE_FRAMES_STACK):
            scene.step()

        p_bottom = scene.body_position(bottom)
        p_top = scene.body_position(top)
        v_bottom = scene.body_velocity(bottom)
        v_top = scene.body_velocity(top)

        self.assertTrue(np.isfinite(p_bottom).all())
        self.assertTrue(np.isfinite(p_top).all())

        self.assertGreater(float(p_top[2]), float(p_bottom[2]) + HE)
        self.assertLess(float(np.linalg.norm(v_bottom)), 0.01)
        self.assertLess(float(np.linalg.norm(v_top)), 0.05)

        # Heavy cube barely moves -- expected within 5 cm of rest height.
        self.assertAlmostEqual(float(p_bottom[2]), HE, delta=0.05)
        self.assertAlmostEqual(float(p_top[2]), 3 * HE, delta=0.15)

        F_top, _, _ = scene.gather_contact_wrench_on_body(top)
        F_bot, _, _ = scene.gather_contact_wrench_on_body(bottom)
        expected_net = (100.0 + 1.0) * GRAVITY
        self.assertAlmostEqual(
            float(F_top[2]) + float(F_bot[2]),
            expected_net,
            delta=0.05 * expected_net,
        )
        self.assertAlmostEqual(
            float(F_top[2]),
            1.0 * GRAVITY,
            delta=0.5,  # 50 cN absolute tolerance on a 9.81 N expected
        )
        plane_fz = _plane_pair_fz_to_body(scene, bottom)
        self.assertAlmostEqual(
            plane_fz,
            expected_net,
            delta=0.05 * expected_net,
        )


# ---------------------------------------------------------------------------
# Sandwich: heavy-light-heavy stack
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping")
class TestSandwichedLightCube(unittest.TestCase):
    """Heavy-Light-Heavy sandwich: a 1 kg cube squeezed between two
    50 kg cubes. The middle cube must not get crushed or squirt out.
    """

    def test_50_1_50_sandwich(self) -> None:
        scene = _PhoenXScene(substeps=40, solver_iterations=32)
        scene.add_ground_plane()
        bottom = scene.add_box(
            position=(0.0, 0.0, HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=50.0,
        )
        middle = scene.add_box(
            position=(0.0, 0.0, 3 * HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=1.0,
        )
        top = scene.add_box(
            position=(0.0, 0.0, 5 * HE + 0.05),
            half_extents=(HE, HE, HE),
            mass=50.0,
        )
        scene.finalize()

        for _ in range(SETTLE_FRAMES_STACK):
            scene.step()

        p_bottom = scene.body_position(bottom)
        p_middle = scene.body_position(middle)
        p_top = scene.body_position(top)

        for label, p in (("bottom", p_bottom), ("middle", p_middle), ("top", p_top)):
            self.assertTrue(np.isfinite(p).all(), msg=f"{label} non-finite: {p}")

        # Z ordering preserved -- nothing crushes through.
        self.assertGreater(float(p_middle[2]), float(p_bottom[2]) + HE)
        self.assertGreater(float(p_top[2]), float(p_middle[2]) + HE)

        # Middle cube should not have squirted sideways.
        self.assertLess(
            float(np.hypot(p_middle[0], p_middle[1])),
            0.1,
            msg=f"middle cube ejected: xy={p_middle[0]:.3f}, {p_middle[1]:.3f}",
        )

        # System net Fz invariant + plane-pair Fz check.
        F_top, _, _ = scene.gather_contact_wrench_on_body(top)
        F_mid, _, _ = scene.gather_contact_wrench_on_body(middle)
        F_bot, _, _ = scene.gather_contact_wrench_on_body(bottom)
        net_fz = float(F_top[2]) + float(F_mid[2]) + float(F_bot[2])
        expected_net = (50.0 + 1.0 + 50.0) * GRAVITY
        self.assertAlmostEqual(
            net_fz,
            expected_net,
            delta=0.05 * expected_net,
            msg=f"system net Fz = {net_fz:.2f} N vs total weight = {expected_net:.2f} N",
        )
        plane_fz = _plane_pair_fz_to_body(scene, bottom)
        self.assertAlmostEqual(
            plane_fz,
            expected_net,
            delta=0.05 * expected_net,
            msg=f"plane->bottom pair Fz = {plane_fz:.2f} N vs total weight = "
                f"{expected_net:.2f} N",
        )


# ---------------------------------------------------------------------------
# High mass-ratio pendulum
# ---------------------------------------------------------------------------


_PENDULUM_LENGTH = 1.0
_PENDULUM_FPS = 240
_PENDULUM_SUBSTEPS = 8
_PENDULUM_ITERATIONS = 16


def _build_high_ratio_pendulum(
    device,
    *,
    hub_mass: float,
    bob_mass: float,
    initial_angle: float = 0.2,
    step_layout: str = "multi_world",
):
    """Hub body fixed to the world via a fixed joint, with a small bob
    hung off it via a +z revolute joint. ``initial_angle`` is the
    starting deflection [rad] from straight-down; the pendulum then
    swings under gravity.

    The hub is fixed via a separate :class:`JointMode.FIXED` joint to
    the world body so the bob's pendulum dynamics see the full hub
    inertia (vs. the world's infinite inertia). This mimics a robot
    arm's distal links carrying a small payload, where the inertia
    ratio between the parent link and the payload sets the natural
    frequency the controller must operate at.
    """
    b = WorldBuilder()
    anchor = b.world_body
    # Hub: massive, anchored to world by a fixed joint.
    hub_inv_mass = 1.0 / hub_mass
    hub_i = (1.0 / 6.0) * hub_mass * (2 * 0.2) * (2 * 0.2)
    hub_inv_inertia = ((1.0 / hub_i, 0.0, 0.0), (0.0, 1.0 / hub_i, 0.0), (0.0, 0.0, 1.0 / hub_i))
    hub = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=hub_inv_mass,
        inverse_inertia=hub_inv_inertia,
        affected_by_gravity=False,  # held by FIXED joint
    )
    # Bob: light, hangs below hub with initial deflection.
    bx = _PENDULUM_LENGTH * math.sin(initial_angle)
    by = -_PENDULUM_LENGTH * math.cos(initial_angle)
    # Cube approx: I = (1/6) * m * (2*he)^2 for a unit-aspect cube of
    # half-extent 0.05. Bob is essentially a point mass for the
    # pendulum dynamics.
    bob_he = 0.05
    bob_i = (1.0 / 6.0) * bob_mass * (2 * bob_he) * (2 * bob_he)
    bob_inv_inertia = ((1.0 / bob_i, 0.0, 0.0), (0.0, 1.0 / bob_i, 0.0), (0.0, 0.0, 1.0 / bob_i))
    bob = b.add_dynamic_body(
        position=(bx, by, 0.0),
        inverse_mass=1.0 / bob_mass,
        inverse_inertia=bob_inv_inertia,
        affected_by_gravity=True,
    )
    # FIXED joint world->hub keeps hub locked at origin.
    b.add_joint(
        body1=anchor,
        body2=hub,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 1.0),
        mode=JointMode.FIXED,
    )
    # Revolute hub->bob about +z; anchor at hub origin.
    b.add_joint(
        body1=hub,
        body2=bob,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 1.0),
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.OFF,
    )
    return b.finalize(
        substeps=_PENDULUM_SUBSTEPS,
        solver_iterations=_PENDULUM_ITERATIONS,
        gravity=(0.0, -GRAVITY, 0.0),
        step_layout=step_layout,
        device=device,
    )


@scene(
    "Mass ratio: 1000:1 pendulum",
    description=(
        "Pendulum on a 1000:1 mass-ratio hinge: heavy hub (1000 kg) "
        "fixed to the world, light bob (1 kg) hangs 1 m below via a "
        "revolute joint. Initial 0.2 rad deflection."
    ),
    tags=("mass_ratio", "pendulum"),
)
def build_high_ratio_pendulum_scene(device) -> Scene:
    world = _build_high_ratio_pendulum(device, hub_mass=1000.0, bob_mass=1.0)
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = 0.2  # hub
    he[2] = 0.05  # bob
    return Scene(
        world=world,
        body_half_extents=he,
        frame_dt=1.0 / _PENDULUM_FPS,
        substeps=_PENDULUM_SUBSTEPS,
    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestHeavyPendulum(unittest.TestCase):
    """High-mass-ratio pendulum natural period must match ``2*pi*sqrt(L/g)``.

    A correctly-conditioned hinge with a 1000:1 hub-to-bob mass ratio
    behaves as a fixed-pivot pendulum (the hub's inertia dominates so
    the bob swings against an effectively immovable anchor). A broken
    solver typically reports an inflated or NaN period, or damps the
    oscillation.
    """

    def test_natural_period_matches_analytic(self) -> None:
        device = wp.get_preferred_device()
        # Small initial deflection so the small-angle period
        # ``T = 2*pi*sqrt(L/g)`` is accurate.
        initial_angle = 0.15  # rad
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                world = _build_high_ratio_pendulum(
                    device,
                    hub_mass=1000.0,
                    bob_mass=1.0,
                    initial_angle=initial_angle,
                    step_layout=layout,
                )

                # Sample bob position over ~3 expected periods.
                T_expected = 2.0 * math.pi * math.sqrt(_PENDULUM_LENGTH / GRAVITY)
                n_frames = int(round(3.0 * T_expected * _PENDULUM_FPS))
                dt = 1.0 / _PENDULUM_FPS

                # Warm-up step + capture single step + replay.
                world.step(dt)
                with wp.ScopedCapture(device=device) as cap:
                    world.step(dt)
                graph = cap.graph

                # Initial bob xy from the post-warmup state.
                bob_xy = np.empty((n_frames + 1, 2), dtype=np.float32)
                positions = world.bodies.position.numpy()
                bob_xy[0] = positions[2, 0:2]
                for i in range(n_frames):
                    wp.capture_launch(graph)
                    positions = world.bodies.position.numpy()
                    bob_xy[i + 1] = positions[2, 0:2]

                # Pendulum angle: atan2(x, -y) where straight-down is 0.
                angles = np.arctan2(bob_xy[:, 0], -bob_xy[:, 1])
                # No NaN.
                self.assertTrue(np.isfinite(angles).all(), "bob angle went non-finite")

                # Detect zero-crossings (sign change in angle) to estimate
                # half-periods. Take the first 4 crossings -> 2 full periods.
                signs = np.sign(angles)
                crossings = np.where(np.diff(signs) != 0)[0]
                self.assertGreaterEqual(
                    len(crossings),
                    4,
                    msg=f"only {len(crossings)} zero-crossings in {n_frames} frames -- "
                        "pendulum may not be oscillating",
                )

                # Linear interpolate each crossing to sub-frame precision.
                cross_times = []
                for k in crossings[:4]:
                    a0 = float(angles[k])
                    a1 = float(angles[k + 1])
                    # frac in [0, 1]: where angle would be 0.
                    frac = a0 / (a0 - a1) if (a0 - a1) != 0.0 else 0.0
                    cross_times.append((k + frac) * dt)
                # Period = 2 * average half-period.
                half_periods = np.diff(cross_times)
                T_measured = 2.0 * float(np.mean(half_periods))

                # Tolerance: 5% on the period. A broken solver typically
                # misses by 20+%.
                rel_err = abs(T_measured - T_expected) / T_expected
                self.assertLess(
                    rel_err,
                    0.05,
                    msg=f"T_measured={T_measured:.4f} s, expected={T_expected:.4f} s, "
                        f"rel_err={rel_err * 100:.2f}%",
                )

                # Amplitude must remain meaningful (no spurious damping).
                # First-period peak amplitude should be within 10% of the
                # initial deflection.
                first_period_end = int(round(T_expected * _PENDULUM_FPS))
                peak_first = float(np.abs(angles[: first_period_end + 1]).max())
                self.assertGreater(
                    peak_first,
                    0.9 * initial_angle,
                    msg=f"peak amplitude {peak_first:.4f} rad < 0.9 * initial "
                        f"{initial_angle:.4f} rad -- spurious damping",
                )


if __name__ == "__main__":
    wp.init()
    unittest.main()
