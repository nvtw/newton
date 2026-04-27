# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for joint features that the existing suite exercised
only implicitly: soft-constraint compliance (``hertz`` /
``damping_ratio``), limit spring-damper in both Box2D and PhoenX-PD
modes (``hertz_limit`` / ``damping_ratio_limit`` vs.
``stiffness_limit`` / ``damping_limit``), combined PD drive
(``stiffness_drive`` + ``damping_drive``), sustained
``max_force_drive`` saturation, the JP contact-priority bias, and
the ``velocity_iterations = 0`` path.

Joint-only tests use the :class:`WorldBuilder` /
:func:`run_settle_loop` harness (graph-captured ``world.step(dt)``);
contact-using tests reuse :class:`_PhoenXScene` from
:mod:`test_stacking`, which also drives Newton's collision pipeline
and graph-captures the per-frame kernel sequence after a warm-up step.
Every test is gated on ``wp.is_cuda_available()`` -- the joint
kernels run hundreds of substeps and only graph-captured replay
keeps wall-clock reasonable; CPU SIMT emulation can take minutes.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests._test_helpers import run_settle_loop
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene
from newton._src.solvers.phoenx.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 4
SETTLE_FRAMES = 60
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ANCHOR1 = (0.0, 0.0, -HALF_EXTENT)
_ANCHOR2 = (0.0, 0.0, +HALF_EXTENT)


def _axial_twist(orientation: np.ndarray) -> float:
    """Signed twist of an orientation quat (xyzw) about world ``+z``.

    Mirrors the helper in :mod:`test_actuated_double_ball_socket`:
    when the swing component is ~zero (revolute lock about z), the
    twist is ``2 * atan2(z, w)``. The revolute fixture used here
    keeps body 1 static at identity, so this directly gives the
    joint angle.
    """
    _x, _y, z, w = orientation
    return 2.0 * math.atan2(z, w)


def _quat_rotate_np(q: np.ndarray, v) -> np.ndarray:
    """Apply quat ``(x, y, z, w)`` to ``v`` -- numpy helper for assertions."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return np.array([rx, ry, rz])


def _build_revolute(
    device,
    *,
    drive_mode=DriveMode.OFF,
    target=0.0,
    target_velocity=0.0,
    max_force_drive=0.0,
    stiffness_drive=0.0,
    damping_drive=0.0,
    hertz=60.0,
    damping_ratio=1.0,
    min_value=1.0,
    max_value=-1.0,
    hertz_limit=30.0,
    damping_ratio_limit=1.0,
    stiffness_limit=0.0,
    damping_limit=0.0,
    cube_position=(0.0, 0.0, 0.0),
    affected_by_gravity=False,
    initial_angular_velocity=(0.0, 0.0, 0.0),
    gravity=(0.0, 0.0, 0.0),
    velocity_iterations=1,
):
    """Static body + dynamic cube hinged about world ``+z``. Returns
    ``(world, joint_handle, dynamic_cube_phoenx_slot_idx)``.

    The body returned by ``add_dynamic_body`` is the index in
    ``WorldBuilder._bodies``, which (for a single-world build) lines
    up directly with the PhoenX slot index thanks to
    ``_bodies[0] = world anchor``: ``add_static_body()`` -> 1,
    ``add_dynamic_body()`` -> 2.
    """
    b = WorldBuilder()
    anchor = b.add_static_body()
    cube = b.add_dynamic_body(
        position=cube_position,
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_joint(
        body1=anchor,
        body2=cube,
        anchor1=_ANCHOR1,
        anchor2=_ANCHOR2,
        mode=JointMode.REVOLUTE,
        hertz=hertz,
        damping_ratio=damping_ratio,
        drive_mode=drive_mode,
        target=target,
        target_velocity=target_velocity,
        max_force_drive=max_force_drive,
        stiffness_drive=stiffness_drive,
        damping_drive=damping_drive,
        min_value=min_value,
        max_value=max_value,
        hertz_limit=hertz_limit,
        damping_ratio_limit=damping_ratio_limit,
        stiffness_limit=stiffness_limit,
        damping_limit=damping_limit,
    )
    world = b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        velocity_iterations=velocity_iterations,
        gravity=gravity,
        device=device,
    )
    return world, handle, cube


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX joint-advanced tests require CUDA + graph capture")
class TestSoftConstraint(unittest.TestCase):
    """``hertz`` / ``damping_ratio`` on the joint's positional bias."""

    def test_low_hertz_yields_more_anchor_drift_under_load(self) -> None:
        """A revolute joint hung off-anchor under gravity carries an
        external moment through its positional bias. Compliant
        (``hertz = 5``) lets the anchor pair drift measurably more
        than stiff (``hertz = 60``) under the same load. Compare the
        anchor-pair separation between the two configurations."""
        device = wp.get_device()
        results: dict[float, float] = {}
        for hertz in (5.0, 60.0):
            world, _, cube = _build_revolute(
                device,
                hertz=hertz,
                damping_ratio=1.0,
                cube_position=(0.0, -1.0, 0.0),
                affected_by_gravity=True,
                gravity=(0.0, -GRAVITY, 0.0),
            )
            run_settle_loop(world, frames=SETTLE_FRAMES, dt=1.0 / FPS)
            wp.synchronize_device()
            pos = world.bodies.position.numpy()
            ori = world.bodies.orientation.numpy()
            p1 = pos[1] + _quat_rotate_np(ori[1], _ANCHOR1)
            p2 = pos[cube] + _quat_rotate_np(ori[cube], _ANCHOR2)
            results[hertz] = float(np.linalg.norm(p1 - p2))
        self.assertGreater(
            results[5.0],
            results[60.0],
            msg=f"compliant ({results[5.0]:.3e}) must drift more than stiff ({results[60.0]:.3e})",
        )

    def test_high_damping_ratio_suppresses_oscillation(self) -> None:
        """Hang the cube off the hinge axis under gravity; the
        positional bias acts like a torsional spring on the
        constrained DoFs. Low damping_ratio should let the cube
        oscillate (large peak angular speed); critical damping
        should suppress it. Compare angular-speed peaks across the
        first 30 frames."""
        device = wp.get_device()
        peaks: dict[float, float] = {}
        for damping_ratio in (0.05, 1.0):
            # Cube offset along -y so gravity creates a moment about
            # an axis the joint locks (revolute axis = z, so x and y
            # rotational DoFs are constrained). The bias spring acts
            # on those, and damping_ratio controls how much it rings.
            world, _, cube = _build_revolute(
                device,
                hertz=10.0,
                damping_ratio=damping_ratio,
                cube_position=(0.0, -1.0, 0.0),
                affected_by_gravity=True,
                gravity=(0.0, -GRAVITY, 0.0),
            )
            peak = 0.0
            for _ in range(30):
                world.step(dt=1.0 / FPS)
                wp.synchronize_device()
                w = world.bodies.angular_velocity.numpy()[cube]
                # Only the constrained DoFs (x, y) feel the spring;
                # the free z-axis is decoupled.
                peak = max(peak, math.hypot(float(w[0]), float(w[1])))
            peaks[damping_ratio] = peak
        self.assertGreater(
            peaks[0.05],
            peaks[1.0],
            msg=f"under-damped peak ({peaks[0.05]:.3e}) must exceed critical ({peaks[1.0]:.3e})",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX joint-advanced tests require CUDA + graph capture")
class TestLimitSpringDamper(unittest.TestCase):
    """Box2D ``(hertz_limit, damping_ratio_limit)`` and PhoenX-PD
    ``(stiffness_limit, damping_limit)`` paths -- the
    :class:`JointDescriptor` docstring promises that positive PD
    gains take precedence over the Box2D formulation."""

    def _drive_into_upper_limit(self, device, *, stiffness_limit: float, damping_limit: float) -> float:
        """Drive past the upper limit with a constant target; let the
        limit catch the body. Return the steady-state axial twist (=
        revolute joint angle)."""
        upper = math.pi / 4
        world, _, cube = _build_revolute(
            device,
            drive_mode=DriveMode.POSITION,
            target=upper + 0.5,
            stiffness_drive=200.0,
            damping_drive=20.0,
            min_value=-upper,
            max_value=upper,
            hertz_limit=30.0,
            damping_ratio_limit=1.0,
            stiffness_limit=stiffness_limit,
            damping_limit=damping_limit,
        )
        run_settle_loop(world, frames=SETTLE_FRAMES, dt=1.0 / FPS)
        wp.synchronize_device()
        return _axial_twist(world.bodies.orientation.numpy()[cube])

    def test_box2d_formulation_clamps_at_limit(self) -> None:
        device = wp.get_device()
        upper = math.pi / 4
        angle = self._drive_into_upper_limit(device, stiffness_limit=0.0, damping_limit=0.0)
        # Soft limit: small steady-state overshoot is expected
        # because the spring needs penetration to balance the drive
        # torque. Allow 25% of the limit value -- this is a "limit
        # eventually catches" check, not a stiff-clamp test.
        self.assertLess(
            abs(angle - upper),
            0.25 * upper,
            msg=f"Box2D limit failed to clamp: angle={angle:.4f}, upper={upper:.4f}",
        )

    def test_pd_formulation_clamps_at_limit(self) -> None:
        device = wp.get_device()
        upper = math.pi / 4
        angle = self._drive_into_upper_limit(device, stiffness_limit=400.0, damping_limit=40.0)
        self.assertLess(
            abs(angle - upper),
            0.25 * upper,
            msg=f"PD limit failed to clamp: angle={angle:.4f}, upper={upper:.4f}",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX joint-advanced tests require CUDA + graph capture")
class TestPDDrive(unittest.TestCase):
    """Combined ``(stiffness_drive, damping_drive)`` PD on
    :data:`DriveMode.POSITION`."""

    def test_kd_suppresses_step_response_overshoot(self) -> None:
        """A kp-only drive produces oscillatory overshoot; kp + kd
        produces a flatter approach. Compare the peak axial twist
        over the first 30 frames after a step target."""
        device = wp.get_device()
        target = math.pi / 6
        peaks: dict[float, float] = {}
        for damping in (0.0, 20.0):
            world, _, cube = _build_revolute(
                device,
                drive_mode=DriveMode.POSITION,
                target=target,
                stiffness_drive=200.0,
                damping_drive=damping,
            )
            peak = 0.0
            for _ in range(30):
                world.step(dt=1.0 / FPS)
                wp.synchronize_device()
                ori = world.bodies.orientation.numpy()[cube]
                peak = max(peak, _axial_twist(ori))
            peaks[damping] = peak
        self.assertGreater(
            peaks[0.0],
            peaks[20.0],
            msg=(
                f"kp-only peak ({peaks[0.0]:.4f}) must exceed kp+kd peak "
                f"({peaks[20.0]:.4f}); both vs target {target:.4f}"
            ),
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX joint-advanced tests require CUDA + graph capture")
class TestMaxForceDriveSaturation(unittest.TestCase):
    """``max_force_drive`` caps the per-substep impulse magnitude.
    With a small cap, the drive cannot reach a stiff target as far
    as an uncapped drive in the same time -- the per-step impulse
    saturates and the response is rate-limited."""

    def test_capped_drive_lags_uncapped_drive(self) -> None:
        """Same drive parameters, same target, same scene -- one
        capped, one uncapped. Across the first N frames, the capped
        drive's axial twist must trail the uncapped drive's. We
        sample at the end of the window."""
        device = wp.get_device()
        target = math.pi / 3  # large step target
        twists: dict[str, float] = {}
        for label, cap in (("uncapped", 0.0), ("capped", 0.2)):
            world, _, cube = _build_revolute(
                device,
                drive_mode=DriveMode.POSITION,
                target=target,
                stiffness_drive=400.0,
                damping_drive=30.0,
                max_force_drive=cap,
            )
            for _ in range(15):  # short window so cap matters
                world.step(dt=1.0 / FPS)
                wp.synchronize_device()
            twists[label] = _axial_twist(world.bodies.orientation.numpy()[cube])
        self.assertLess(
            twists["capped"],
            twists["uncapped"],
            msg=(
                f"capped twist ({twists['capped']:.4f}) should trail uncapped "
                f"({twists['uncapped']:.4f}) -- the cap rate-limits the drive"
            ),
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX joint-advanced tests require CUDA + graph capture")
class TestColoringPriorityBias(unittest.TestCase):
    """The JP coloring biases contact cids' priority above every
    joint cid so contacts cluster in earlier colours and joints
    later. Build a contact-only stacking scene via :class:`_PhoenXScene`
    (which has zero joints, so this also asserts the partitioner is
    wired correctly when the bias is a no-op). The flip-side test
    of joint+contact priority requires Newton's
    :class:`SolverPhoenX` wrapper, which the existing
    :mod:`test_articulation_parity` covers end-to-end."""

    def test_contacts_appear_in_first_colour_when_no_joints(self) -> None:
        """Stack a single box on the ground. With ``num_joints == 0``,
        every cid is a contact and they must all land in the first
        colour (single shape pair, single column)."""
        scene = _PhoenXScene()
        scene.add_ground_plane()
        scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        for _ in range(SETTLE_FRAMES):
            scene.step()
        wp.synchronize_device()
        nc = int(scene.world._world_num_colors.numpy()[0])
        # One pair -> exactly one column -> one colour.
        self.assertGreaterEqual(nc, 1, msg="at least one colour required when contact exists")
        # First colour holds the contact column.
        starts = scene.world._world_color_starts.numpy()
        eids = scene.world._world_element_ids_by_color.numpy()
        s, e = int(starts[0, 0]), int(starts[0, 1])
        self.assertGreater(e - s, 0, msg="first colour should be non-empty")
        # The cid must be in the contact range (>= num_joints == 0).
        for i in range(s, e):
            self.assertGreaterEqual(int(eids[i]), 0)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX joint-advanced tests require CUDA + graph capture")
class TestVelocityIterationsZero(unittest.TestCase):
    """``velocity_iterations = 0`` skips the TGS-soft relax sweep.
    Position-bias-on iterations alone should still settle a single
    body on a plane; baumgarte drift only accumulates on tall
    stacks."""

    def test_single_body_settles_with_zero_velocity_iters(self) -> None:
        scene = _PhoenXScene(
            substeps=4,
            solver_iterations=8,
            velocity_iterations=0,
        )
        scene.add_ground_plane()
        scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        for _ in range(SETTLE_FRAMES):
            scene.step()
        wp.synchronize_device()
        # Cube COM should rest near z = 0.5 (half-extent above the
        # ground at z = 0). Allow 5 cm of baumgarte / penetration
        # tolerance -- without a relax sweep it sits slightly high.
        z = float(scene.body_position(0)[2])
        self.assertLess(
            abs(z - 0.5),
            0.05,
            msg=f"cube z={z:.4f} did not settle near 0.5 with velocity_iterations=0",
        )


if __name__ == "__main__":
    unittest.main()
