# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for :class:`JointMode.CABLE` -- soft angular
spring + damper on top of a rigid ball-socket.

The cable joint decomposes the Darboux vector of the child body's
rotation (relative to its rest pose) into three independent scalar
rows:

* ``bend`` about the two axes perpendicular to the reference axis
  ``anchor1 -> anchor2``;
* ``twist`` along that axis.

Per-component ``(stiffness, damping)`` pairs drive each row with the
Jitter2 / Box2D soft-PD plumbing the actuated joint's drive / limit
rows use. The physical response matches the textbook damped harmonic
oscillator :math:`I \\ddot\theta + c \\dot\theta + k \theta = 0`:
natural frequency :math:`\\omega_n = \\sqrt{k/I}`, damping ratio
:math:`\\zeta = c / (2\\sqrt{kI})`. Tests validate oscillation
(undamped), exponential settling (critically damped / overdamped),
and stiffness / damping ordering.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.scene_registry import Scene, scene
from newton._src.solvers.phoenx.tests._test_helpers import run_settle_loop
from newton._src.solvers.phoenx.world_builder import (
    JointMode,
    WorldBuilder,
)

FPS = 240
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
HALF_EXTENT = 0.2
_INV_INERTIA_ROD = ((20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0))


def _axis_angle_quat(ax: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float, float]:
    """Right-hand rotation quaternion about ``ax`` by ``angle_rad``."""
    ax_np = np.asarray(ax, dtype=np.float32)
    nrm = float(np.linalg.norm(ax_np))
    if nrm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    ax_np = ax_np / nrm
    s = math.sin(angle_rad * 0.5)
    c = math.cos(angle_rad * 0.5)
    return (float(ax_np[0] * s), float(ax_np[1] * s), float(ax_np[2] * s), c)


def _build_cable_pendulum(
    device,
    *,
    bend_stiffness: float = 0.0,
    twist_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    twist_damping: float = 0.0,
    init_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    init_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = False,
    com_offset: float = 0.0,
):
    """World anchor + one dynamic rod pivoted at the origin via a cable
    joint. Reference axis is +x, so rotations about +y or +z are
    'bend' and rotations about +x are 'twist'.

    By default the rod's COM sits AT the anchor (``com_offset = 0``)
    so the ball-socket positional lock contributes no lever-arm
    coupling -- bend / twist rows dominate the dynamics. Callers that
    want a gravity-driven pendulum (positional lock check) can set
    ``com_offset`` to move the COM off the anchor.

    The ADBS init kernel snapshots the rest alignment quaternion at
    finalize() time from the bodies' current orientation. To set up a
    pre-deflected rod without overwriting that rest snapshot, we build
    the rod at identity (so rest = identity) and then overwrite the
    runtime orientation after finalize -- the first prepare pass sees
    ``q_align = init_rotation``, i.e. the desired deflection."""
    b = WorldBuilder()
    world = b.world_body
    rod = b.add_dynamic_body(
        position=(com_offset, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA_ROD,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=init_angular_velocity,
    )
    b.add_joint(
        body1=world,
        body2=rod,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),  # reference axis along +x
        mode=JointMode.CABLE,
        bend_stiffness=bend_stiffness,
        twist_stiffness=twist_stiffness,
        bend_damping=bend_damping,
        twist_damping=twist_damping,
    )
    world_out = b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, -9.81) if affected_by_gravity else (0.0, 0.0, 0.0),
        device=device,
    )
    # Post-finalize: overwrite the rod's orientation so it starts
    # deflected from the (identity) rest pose stored in the column.
    if init_rotation != (0.0, 0.0, 0.0, 1.0):
        orient_np = world_out.bodies.orientation.numpy()
        orient_np[rod] = np.asarray(init_rotation, dtype=np.float32)
        world_out.bodies.orientation.assign(orient_np)
    return world_out


def _rod_orientation(world, slot: int = 1) -> np.ndarray:
    """Return the rod body's quaternion ``[x, y, z, w]``."""
    return world.bodies.orientation.numpy()[slot].copy()


def _rod_angular_velocity(world, slot: int = 1) -> np.ndarray:
    return world.bodies.angular_velocity.numpy()[slot].copy()


def _rotation_angle_about(q: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (radians) of ``q`` about the unit ``axis``. Picks
    the shortest-path sign via the quaternion's w component."""
    # q = [x, y, z, w]; enforce shortest-path canonical (w >= 0).
    w = float(q[3])
    xyz = q[:3].astype(np.float64)
    if w < 0.0:
        w = -w
        xyz = -xyz
    # Shortest-path angle about axis: angle * axis ≈ 2 * xyz (small).
    # For the full formula, angle = 2 * atan2(|xyz along axis|, w).
    projection = float(np.dot(xyz, axis.astype(np.float64)))
    return 2.0 * math.atan2(projection, w)


@scene(
    "Cable: bent rod springs back (x pivot)",
    description="Rod pre-rotated about +z (bend), released; springs back to rest.",
    tags=("cable",),
)
def build_cable_bend_scene(device) -> Scene:
    init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(20.0))
    world = _build_cable_pendulum(
        device,
        bend_stiffness=40.0,
        twist_stiffness=40.0,
        bend_damping=0.2,
        twist_damping=0.2,
        init_rotation=init_q,
    )
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = (HALF_EXTENT, HALF_EXTENT * 0.2, HALF_EXTENT * 0.2)
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@scene(
    "Cable: twisted rod springs back",
    description="Rod pre-rotated about +x (twist), released; springs back.",
    tags=("cable",),
)
def build_cable_twist_scene(device) -> Scene:
    init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(30.0))
    world = _build_cable_pendulum(
        device,
        bend_stiffness=40.0,
        twist_stiffness=40.0,
        bend_damping=0.2,
        twist_damping=0.2,
        init_rotation=init_q,
    )
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = (HALF_EXTENT, HALF_EXTENT * 0.2, HALF_EXTENT * 0.2)
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cable tests run on CUDA only.",
)
class TestCableBendStiffness(unittest.TestCase):
    """Bend stiffness acts perpendicular to the reference axis."""

    def test_bend_springs_back(self) -> None:
        """Undamped bend spring: pre-rotate about +z, release. The
        restoring torque pulls the rod back through zero and the
        angle signs change (free oscillation)."""
        init_angle = math.radians(20.0)
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), init_angle)
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=60.0,
            twist_stiffness=60.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = []
        for _ in range(int(0.3 * FPS)):
            run_settle_loop(world, 1, dt=dt)
            angles.append(_rotation_angle_about(_rod_orientation(world), z_axis))
        angles = np.asarray(angles)
        self.assertTrue(np.isfinite(angles).all())
        self.assertLess(
            float(angles.min()),
            0.0,
            msg=f"bend spring never crossed zero, angles: min={angles.min():.3f}, max={angles.max():.3f}",
        )

    def test_bend_damping_decays(self) -> None:
        """Critically / over-damped bend: adding damping kills the
        oscillation envelope. The angle magnitude shrinks well below
        its starting value."""
        init_q = _axis_angle_quat((0.0, 1.0, 0.0), math.radians(20.0))
        # zeta ~ 1 for I = 1/20 = 0.05, k = 200, so c_crit = 2*sqrt(k*I) ~ 6.3.
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=200.0,
            twist_stiffness=200.0,
            bend_damping=7.0,
            twist_damping=7.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        y_axis = np.array([0.0, 1.0, 0.0])
        amp_start = abs(_rotation_angle_about(_rod_orientation(world), y_axis))
        run_settle_loop(world, int(1.0 * FPS), dt=dt)
        amp_end = abs(_rotation_angle_about(_rod_orientation(world), y_axis))
        self.assertLess(
            amp_end,
            amp_start * 0.1,
            msg=f"bend damping failed: |q| went {amp_start:.3f} -> {amp_end:.3f}",
        )

    def test_bend_stiffness_affects_period(self) -> None:
        """Higher bend stiffness -> shorter oscillation period:
        :math:`\\omega_n = \\sqrt{k/I}`, so doubling k shrinks the
        period by sqrt(2). Stiffer spring should cross zero more
        often in the same window."""
        dt = 1.0 / FPS
        window_s = 0.4
        n = int(window_s * FPS)
        y_axis = np.array([0.0, 1.0, 0.0])

        def count_zero_crossings(k: float) -> int:
            init_q = _axis_angle_quat((0.0, 1.0, 0.0), math.radians(10.0))
            world = _build_cable_pendulum(
                wp.get_preferred_device(),
                bend_stiffness=k,
                twist_stiffness=k,
                init_rotation=init_q,
            )
            prev = _rotation_angle_about(_rod_orientation(world), y_axis)
            crossings = 0
            for _ in range(n):
                run_settle_loop(world, 1, dt=dt)
                cur = _rotation_angle_about(_rod_orientation(world), y_axis)
                if (cur > 0) != (prev > 0):
                    crossings += 1
                prev = cur
            return crossings

        soft_crossings = count_zero_crossings(30.0)
        stiff_crossings = count_zero_crossings(300.0)
        self.assertGreater(
            stiff_crossings,
            soft_crossings,
            msg=f"stiffer spring should oscillate faster; soft={soft_crossings}, stiff={stiff_crossings}",
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cable tests run on CUDA only.",
)
class TestCableTwistStiffness(unittest.TestCase):
    """Twist stiffness acts along the reference axis."""

    def test_twist_springs_back(self) -> None:
        """Undamped twist spring: the rod oscillates about rest
        twist, crossing zero."""
        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(30.0))
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=60.0,
            twist_stiffness=60.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        x_axis = np.array([1.0, 0.0, 0.0])
        angles = []
        for _ in range(int(0.3 * FPS)):
            run_settle_loop(world, 1, dt=dt)
            angles.append(_rotation_angle_about(_rod_orientation(world), x_axis))
        angles = np.asarray(angles)
        self.assertTrue(np.isfinite(angles).all())
        self.assertLess(
            float(angles.min()),
            0.0,
            msg=f"twist spring never crossed zero; min={angles.min():.3f}, max={angles.max():.3f}",
        )

    def test_twist_damping_decays(self) -> None:
        """Twist damping kills the oscillation envelope."""
        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(25.0))
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=200.0,
            twist_stiffness=200.0,
            bend_damping=7.0,
            twist_damping=7.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        x_axis = np.array([1.0, 0.0, 0.0])
        amp_start = abs(_rotation_angle_about(_rod_orientation(world), x_axis))
        run_settle_loop(world, int(1.0 * FPS), dt=dt)
        amp_end = abs(_rotation_angle_about(_rod_orientation(world), x_axis))
        self.assertLess(
            amp_end,
            amp_start * 0.1,
            msg=f"twist damping failed: |q_twist| went {amp_start:.3f} -> {amp_end:.3f}",
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cable tests run on CUDA only.",
)
class TestCableDecoupling(unittest.TestCase):
    """Bend and twist are independent scalar rows; a pure bend
    deflection must not spring up a twist error and vice versa."""

    def test_pure_bend_produces_no_twist(self) -> None:
        """Initial rotation is pure bend (about +y). Without any
        initial twist deflection, twist angle should stay near zero
        through the oscillation."""
        init_q = _axis_angle_quat((0.0, 1.0, 0.0), math.radians(20.0))
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=80.0,
            twist_stiffness=80.0,
            bend_damping=1.0,
            twist_damping=1.0,
            init_rotation=init_q,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(0.5 * FPS), dt=dt)
        x_axis = np.array([1.0, 0.0, 0.0])
        twist = _rotation_angle_about(_rod_orientation(world), x_axis)
        self.assertLess(
            abs(twist),
            0.05,
            msg=f"pure bend deflection bled into twist: q_x={twist:.4f} rad",
        )

    def test_pure_twist_produces_no_bend(self) -> None:
        """Initial rotation is pure twist (about +x). Bend angle
        should stay near zero."""
        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(30.0))
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=80.0,
            twist_stiffness=80.0,
            bend_damping=1.0,
            twist_damping=1.0,
            init_rotation=init_q,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(0.5 * FPS), dt=dt)
        # Any rotation about +y or +z would be bend; measure both.
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        q_rod = _rod_orientation(world)
        bend_y = _rotation_angle_about(q_rod, y_axis)
        bend_z = _rotation_angle_about(q_rod, z_axis)
        self.assertLess(
            abs(bend_y),
            0.05,
            msg=f"pure twist bled into bend-y: {bend_y:.4f} rad",
        )
        self.assertLess(
            abs(bend_z),
            0.05,
            msg=f"pure twist bled into bend-z: {bend_z:.4f} rad",
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cable tests run on CUDA only.",
)
class TestCableDegenerateAndStability(unittest.TestCase):
    """Edge cases."""

    def test_zero_stiffness_matches_ball_socket(self) -> None:
        """With zero stiffness and zero damping on both axes, the
        cable joint degenerates to a plain ball-socket (the angular
        rows all contribute zero impulse). An initial spin must be
        preserved."""
        omega0 = (1.0, -0.5, 0.3)
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=0.0,
            twist_stiffness=0.0,
            bend_damping=0.0,
            twist_damping=0.0,
            init_angular_velocity=omega0,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(0.5 * FPS), dt=dt)
        omega = _rod_angular_velocity(world)
        drift = float(np.linalg.norm(omega - np.asarray(omega0)))
        self.assertLess(
            drift,
            0.1,
            msg=f"zero-stiffness cable decayed spin: drift={drift:.4f} (initial |omega|={np.linalg.norm(omega0):.3f})",
        )

    def test_natural_frequency_matches_theory(self) -> None:
        """Sanity-check the soft-PD plumbing numerically: an undamped
        spring with stiffness ``k`` and rod inertia ``I`` should
        oscillate at :math:`\\omega_n = \\sqrt{k/I}`. We measure the
        oscillation period from the first two zero crossings and
        check it's within ~15 % of the textbook value. The tolerance
        covers the Box2D implicit-Euler discretisation error, which
        scales with ``dt^2 * k / I`` (bounded by our substep rate)."""
        # Diagonal inverse-inertia of 20 on every axis -> I = 1/20.
        I_rod = 1.0 / 20.0
        k = 60.0
        theoretical_period = 2.0 * math.pi / math.sqrt(k / I_rod)

        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(15.0))
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = [_rotation_angle_about(_rod_orientation(world), z_axis)]
        for _ in range(int(1.0 * FPS)):
            run_settle_loop(world, 1, dt=dt)
            angles.append(_rotation_angle_about(_rod_orientation(world), z_axis))
        angles = np.asarray(angles)
        # Find zero-crossing indices.
        signs = np.sign(angles)
        crossings = np.where(np.diff(signs) != 0)[0]
        self.assertGreaterEqual(
            len(crossings),
            2,
            msg=f"expected >=2 zero crossings in 1 s; got {len(crossings)}",
        )
        # Period between two successive zero crossings is half the full
        # period (angle goes +->-; next crossing is -->+, one full period
        # would require the third crossing).
        measured_half_period = (crossings[1] - crossings[0]) * dt
        measured_period = 2.0 * measured_half_period
        self.assertAlmostEqual(
            measured_period,
            theoretical_period,
            delta=theoretical_period * 0.2,
            msg=(
                f"period mismatch: theoretical {theoretical_period * 1000:.2f} ms, "
                f"measured {measured_period * 1000:.2f} ms"
            ),
        )

    def test_large_bend_deflection_stable(self) -> None:
        """Release the rod from pi/3 (~60 deg) of bend with stiff
        bend and moderate damping. It must settle without NaN or
        runaway velocity."""
        init_q = _axis_angle_quat((0.0, 1.0, 0.0), math.pi / 3.0)
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=200.0,
            twist_stiffness=200.0,
            bend_damping=5.0,
            twist_damping=5.0,
            init_rotation=init_q,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(1.5 * FPS), dt=dt)
        q = _rod_orientation(world)
        omega = _rod_angular_velocity(world)
        self.assertTrue(np.isfinite(q).all() and np.isfinite(omega).all())
        y_axis = np.array([0.0, 1.0, 0.0])
        final_bend = abs(_rotation_angle_about(q, y_axis))
        self.assertLess(
            final_bend,
            0.15,
            msg=f"rod didn't settle: final bend={final_bend:.3f} rad",
        )
        self.assertLess(
            float(np.linalg.norm(omega)),
            0.5,
            msg=f"rod still spinning: |omega|={np.linalg.norm(omega):.3f} rad/s",
        )

    def test_rigid_ball_socket_still_holds_anchor(self) -> None:
        """The positional 3-row lock at anchor-1 is unchanged by cable
        mode. Under gravity the rod must hang from the anchor without
        its COM drifting away from the pendulum radius."""
        world = _build_cable_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=5.0,  # soft -- tests the *positional* lock
            twist_stiffness=5.0,
            bend_damping=1.0,
            twist_damping=1.0,
            affected_by_gravity=True,
            com_offset=0.5,  # pendulum arm so gravity creates a torque
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(2.0 * FPS), dt=dt)
        pos = world.bodies.position.numpy()[1]
        self.assertTrue(np.isfinite(pos).all())
        # Pendulum radius is |COM - anchor1| = 0.5 m; the ball-socket
        # lock must preserve that through the swing.
        radius = float(np.linalg.norm(pos))
        self.assertAlmostEqual(
            radius,
            0.5,
            delta=0.05,
            msg=f"positional lock leaked: radius={radius:.4f} m",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
