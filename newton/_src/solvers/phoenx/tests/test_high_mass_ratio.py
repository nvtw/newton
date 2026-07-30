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

import newton
from newton._src.solvers.phoenx.tests._test_helpers import STEP_LAYOUTS
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

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
            msg=f"plane->bottom pair Fz = {plane_fz:.2f} N vs (m_top+m_bot)*g = {expected_net:.2f} N",
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
            msg=f"plane->bottom pair Fz = {plane_fz:.2f} N vs total weight = {expected_net:.2f} N",
        )


# ---------------------------------------------------------------------------
# High mass-ratio direct mechanism
# ---------------------------------------------------------------------------

PENDULUM_LENGTH = 1.0
PENDULUM_FPS = 240
PENDULUM_SUBSTEPS = 5


def _build_high_ratio_pendulum(*, hub_mass: float, bob_mass: float, initial_angle: float) -> tuple[newton.Model, int]:
    """Build a heavy fixed hub carrying a light revolute pendulum."""
    builder = newton.ModelBuilder(gravity=(0.0, -GRAVITY, 0.0), up_axis=newton.Axis.Y)
    hub_inertia = hub_mass * (0.4**2) / 6.0
    hub = builder.add_link(
        xform=wp.transform_identity(),
        mass=hub_mass,
        inertia=((hub_inertia, 0.0, 0.0), (0.0, hub_inertia, 0.0), (0.0, 0.0, hub_inertia)),
    )
    bob_inertia = bob_mass * (0.1**2) / 6.0
    bob = builder.add_link(
        xform=wp.transform_identity(),
        mass=bob_mass,
        inertia=((bob_inertia, 0.0, 0.0), (0.0, bob_inertia, 0.0), (0.0, 0.0, bob_inertia)),
    )
    fixed = builder.add_joint_fixed(parent=-1, child=hub)
    revolute = builder.add_joint_revolute(
        parent=hub,
        child=bob,
        axis=(0.0, 0.0, 1.0),
        child_xform=wp.transform((0.0, PENDULUM_LENGTH, 0.0), wp.quat_identity()),
        damping=0.0,
    )
    builder.add_articulation([fixed, revolute])
    model = builder.finalize(device=wp.get_preferred_device())
    model.joint_q.assign(np.asarray((initial_angle,), dtype=np.float32))
    return model, bob


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX mass-ratio tests require CUDA graphs.")
class TestHeavyPendulum(unittest.TestCase):
    """Check a poorly conditioned full-coordinate mechanism analytically."""

    def test_natural_period_matches_analytic(self) -> None:
        """Match the small-angle period at a 1000-to-1 adjacent mass ratio."""
        initial_angle = 0.15
        expected_period = 2.0 * math.pi * math.sqrt(PENDULUM_LENGTH / GRAVITY)
        frame_count = int(round(2.1 * expected_period * PENDULUM_FPS))
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, bob = _build_high_ratio_pendulum(
                    hub_mass=1000.0,
                    bob_mass=1.0,
                    initial_angle=initial_angle,
                )
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=PENDULUM_SUBSTEPS,
                    solver_iterations=1,
                    velocity_iterations=0,
                    articulation_mode="maximal",
                    step_layout=layout,
                )
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (11,))
                np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [0, 0])
                control = model.control()
                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    solver.step(state, state, control, None, 1.0 / PENDULUM_FPS)

                angles = np.empty(frame_count + 1, dtype=np.float32)
                position = state.body_q.numpy()[bob, :2]
                angles[0] = math.atan2(float(position[0]), -float(position[1]))
                for frame in range(frame_count):
                    wp.capture_launch(capture.graph)
                    position = state.body_q.numpy()[bob, :2]
                    angles[frame + 1] = math.atan2(float(position[0]), -float(position[1]))

                self.assertTrue(np.isfinite(angles).all())
                crossings = np.flatnonzero(np.diff(np.signbit(angles)))
                self.assertGreaterEqual(len(crossings), 4)
                times: list[float] = []
                for frame in crossings[:4]:
                    angle0 = float(angles[frame])
                    angle1 = float(angles[frame + 1])
                    fraction = angle0 / (angle0 - angle1)
                    times.append((frame + fraction) / PENDULUM_FPS)
                measured_period = 2.0 * float(np.mean(np.diff(times)))
                self.assertLess(abs(measured_period - expected_period) / expected_period, 0.05)
                first_period = int(round(expected_period * PENDULUM_FPS))
                self.assertGreater(float(np.max(np.abs(angles[: first_period + 1]))), 0.9 * initial_angle)


if __name__ == "__main__":
    unittest.main()
