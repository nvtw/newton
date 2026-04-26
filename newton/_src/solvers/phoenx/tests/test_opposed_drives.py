# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steady-state Hooke's-law tests for revolute drives.

A PD position drive on a hinge with a known external torque ``tau_ext``
acting against it must settle at::

    theta_eq - target_pos = -tau_ext / ke

(rotary Hooke's law). The reported drive torque from
:meth:`PhoenXWorld.gather_constraint_wrenches` must equal ``tau_ext`` in
magnitude (Newton's third law on the joint), and -- in the two-joint
"opposed drive" variant -- the position-drive torque must equal the
velocity-drive torque in magnitude with opposite sign.

Two scenes are covered:

* :func:`build_pd_vs_gravity_scene` -- single revolute joint with a PD
  position drive, gravity acts on a 1 m lever to provide a known
  external torque ``m*g*L`` perpendicular to the lever.
* :func:`build_opposed_drives_scene` -- two coaxial revolute joints
  on the same body. Joint 0 is a PD position drive at ``target=0``;
  joint 1 is a velocity drive at ``target_velocity=omega*`` with damping
  ``kd_vel``. At rest the velocity drive applies a constant
  ``kd_vel*omega*`` torque (since ``omega_actual=0``) which the position
  drive must resist.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.scene_registry import Scene, scene
from newton._src.solvers.phoenx.tests._test_helpers import STEP_LAYOUTS, run_settle_loop
from newton._src.solvers.phoenx.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

GRAVITY = 9.81
FPS = 240
SUBSTEPS = 8
SOLVER_ITERATIONS = 32
SETTLE_FRAMES = 600  # 2.5 s @ 240 fps
HALF_EXTENT = 0.1
LEVER = 1.0
CUBE_INV_MASS = 1.0
CUBE_INV_INERTIA = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))


def _angle_about_z_from_quat(q: tuple[float, float, float, float]) -> float:
    """Recover hinge angle (rotation about world +z) from a quaternion
    ``[x, y, z, w]``. ``2 * atan2(qz, qw)`` lives in ``(-pi, pi]`` which
    is enough range for these settle-only tests."""
    return 2.0 * math.atan2(float(q[2]), float(q[3]))


# ---------------------------------------------------------------------------
# Scene A: single PD drive vs gravity (rotary Hooke's law)
# ---------------------------------------------------------------------------


def _build_pd_vs_gravity(
    device,
    *,
    target: float = 0.0,
    stiffness_drive: float,
    damping_drive: float,
    step_layout: str = "multi_world",
):
    """Cube hinged at origin via a +z revolute joint, COM at ``(L, 0, 0)``.

    Gravity ``-y`` torques the cube down (toward ``theta = -pi/2``);
    the PD drive at ``target=0`` resists. Steady state balances::

        ke*(target - theta_eq) + tau_grav = 0
        tau_grav = -m * g * L * cos(theta_eq)
    """
    b = WorldBuilder()
    anchor = b.world_body
    cube = b.add_dynamic_body(
        position=(LEVER, 0.0, 0.0),
        inverse_mass=CUBE_INV_MASS,
        inverse_inertia=CUBE_INV_INERTIA,
        affected_by_gravity=True,
    )
    b.add_joint(
        body1=anchor,
        body2=cube,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 1.0),  # +z hinge axis
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.POSITION,
        target=target,
        stiffness_drive=stiffness_drive,
        damping_drive=damping_drive,
        max_force_drive=1.0e4,
    )
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, -GRAVITY, 0.0),
        step_layout=step_layout,
        device=device,
    )


@scene(
    "Drive: PD vs gravity (rotary Hooke's law)",
    description=(
        "Hinge cube on a 1 m lever, PD position drive at theta*=0 vs "
        "gravity. Steady state: theta_eq ~ -mgL/ke."
    ),
    tags=("drive", "hooke"),
)
def build_pd_vs_gravity_scene(device) -> Scene:
    world = _build_pd_vs_gravity(
        device,
        target=0.0,
        stiffness_drive=200.0,
        damping_drive=20.0,
    )
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


# ---------------------------------------------------------------------------
# Scene B: two opposed drives on the same body
# ---------------------------------------------------------------------------


def _build_opposed_drives(
    device,
    *,
    target_pos: float = 0.0,
    target_vel: float,
    ke: float,
    kd_pos: float,
    kd_vel: float,
    step_layout: str = "multi_world",
):
    """Two coaxial revolute joints sharing the +z axis through a single
    cube. Joint 0 is a PD position drive at ``target_pos``; joint 1 is a
    velocity drive at ``target_vel`` with damping ``kd_vel``.

    Both joints lock the cube COM at the origin (redundant point lock,
    handled gracefully by Box2D soft constraints) and free rotation
    about z. At steady state ``omega = 0``, so the velocity drive
    applies ``kd_vel * target_vel`` torque about +z; the position drive
    must apply ``-kd_vel * target_vel`` to balance, deflecting the
    spring by ``kd_vel * target_vel / ke`` from ``target_pos``.
    """
    b = WorldBuilder()
    anchor = b.world_body
    cube = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=CUBE_INV_MASS,
        inverse_inertia=CUBE_INV_INERTIA,
        affected_by_gravity=False,
    )
    # Joint 0: PD position drive. Anchors on the +z axis below the cube.
    b.add_joint(
        body1=anchor,
        body2=cube,
        anchor1=(0.0, 0.0, -1.0),
        anchor2=(0.0, 0.0, 0.0),
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.POSITION,
        target=target_pos,
        stiffness_drive=ke,
        damping_drive=kd_pos,
        max_force_drive=1.0e4,
    )
    # Joint 1: velocity drive. Anchors on the +z axis above the cube
    # (consistent +z axis direction with joint 0 so torque signs agree).
    b.add_joint(
        body1=anchor,
        body2=cube,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 1.0),
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.VELOCITY,
        target_velocity=target_vel,
        damping_drive=kd_vel,
        max_force_drive=1.0e4,
    )
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, 0.0),
        step_layout=step_layout,
        device=device,
    )


@scene(
    "Drive: opposed PD vs velocity (Hooke's law)",
    description=(
        "Two coaxial revolute joints on a cube. Joint 0: PD position "
        "drive at theta*=0. Joint 1: velocity drive at omega*=0.5. "
        "Steady-state torque balance: theta_eq = kd_vel*omega* / ke."
    ),
    tags=("drive", "hooke"),
)
def build_opposed_drives_scene(device) -> Scene:
    world = _build_opposed_drives(
        device,
        target_pos=0.0,
        target_vel=0.5,
        ke=200.0,
        kd_pos=20.0,
        kd_vel=10.0,
    )
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestPDVsGravityHookesLaw(unittest.TestCase):
    """Single PD drive against gravity -- the deflection from target
    must obey the rotary Hooke's law within solver slop.

    Note on torques: ``gather_constraint_wrenches`` reports the *net*
    joint wrench on body 2 about its COM. With gravity at the COM the
    external angular impulse about the COM is zero, so at static
    equilibrium the reported joint torque is also ~0. The drive's axial
    contribution and the anchor-force lever-arm contribution cancel
    exactly inside the wrench, which makes the joint-torque API a poor
    probe for the drive torque alone here. The angle deflection
    ``theta_eq`` is the clean Hooke's-law observable -- so this test
    checks that.
    """

    def test_steady_state_angle_matches_hookes_law(self) -> None:
        device = wp.get_preferred_device()
        ke = 200.0
        kd = 20.0
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                world = _build_pd_vs_gravity(
                    device,
                    target=0.0,
                    stiffness_drive=ke,
                    damping_drive=kd,
                    step_layout=layout,
                )
                run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

                # Cube must be at rest.
                omega = world.bodies.angular_velocity.numpy()[1]
                self.assertLess(
                    abs(float(omega[2])),
                    0.05,
                    msg=f"cube not at rest: omega_z={omega[2]:.4f} rad/s",
                )

                # Recover hinge angle from the cube's quaternion (about +z).
                q = world.bodies.orientation.numpy()[1]
                theta_eq = _angle_about_z_from_quat(tuple(q))

                # Static balance: ke*(0 - theta_eq) + tau_grav_about_pivot = 0.
                # Gravity torque about the +z pivot at origin is
                #   tau_grav = -m * g * L * cos(theta_eq)
                # So: theta_eq = -m*g*L*cos(theta_eq) / ke
                # Self-consistent; iterate to convergence.
                m = 1.0 / CUBE_INV_MASS
                expected = 0.0
                for _ in range(40):
                    expected = -(m * GRAVITY * LEVER * math.cos(expected)) / ke

                self.assertAlmostEqual(
                    theta_eq,
                    expected,
                    delta=0.02,
                    msg=f"theta_eq={theta_eq:.4f} rad, expected ~{expected:.4f} rad",
                )

    def test_stiffness_inverse_proportional_deflection(self) -> None:
        """Hooke's law scaling: at the small-angle / stiff-spring limit
        the deflection ``|theta_eq|`` is inversely proportional to ``ke``
        (for a fixed external torque ``m*g*L``). Doubling ``ke`` should
        halve the deflection."""
        device = wp.get_preferred_device()
        kd = 30.0  # generous damping so each stiffness settles cleanly
        deflections: list[tuple[float, float]] = []
        for ke in (500.0, 1000.0, 2000.0, 4000.0):
            world = _build_pd_vs_gravity(
                device,
                target=0.0,
                stiffness_drive=ke,
                damping_drive=kd,
            )
            run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)
            q = world.bodies.orientation.numpy()[1]
            theta_eq = _angle_about_z_from_quat(tuple(q))
            deflections.append((ke, abs(theta_eq)))

        # ke * |theta_eq| should be ~ m*g*L*cos(theta_eq) ~ m*g*L
        # at small theta. Compare across stiffnesses.
        m = 1.0 / CUBE_INV_MASS
        ideal = m * GRAVITY * LEVER
        for ke, defl in deflections:
            product = ke * defl
            self.assertAlmostEqual(
                product,
                ideal,
                delta=0.10 * ideal,
                msg=f"ke={ke}: ke*|theta_eq|={product:.4f} N*m vs ideal "
                    f"{ideal:.4f} N*m -- Hooke's law scaling broken",
            )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestOpposedDrives(unittest.TestCase):
    """Two coaxial drives on one body. Velocity drive acts as a known
    constant-torque source; PD position drive must resist it. The two
    reported drive torques must match in magnitude with opposite sign."""

    def test_steady_state_torque_balance(self) -> None:
        device = wp.get_preferred_device()
        target_vel = 0.5  # rad/s
        ke = 200.0
        kd_vel = 10.0
        # Expected steady-state torque on each drive: kd_vel * target_vel.
        expected_torque = kd_vel * target_vel  # 5 N*m

        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                world = _build_opposed_drives(
                    device,
                    target_pos=0.0,
                    target_vel=target_vel,
                    ke=ke,
                    kd_pos=20.0,
                    kd_vel=kd_vel,
                    step_layout=layout,
                )
                run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

                # Cube must be (nearly) at rest -- velocity drive doesn't
                # spin the cube up because the position drive holds it.
                omega = world.bodies.angular_velocity.numpy()[1]
                self.assertLess(
                    abs(float(omega[2])),
                    0.05,
                    msg=f"cube still spinning: omega_z={omega[2]:.4f} rad/s",
                )

                # Hooke's law: theta_eq - target_pos = expected_torque / ke.
                q = world.bodies.orientation.numpy()[1]
                theta_eq = _angle_about_z_from_quat(tuple(q))
                expected_theta = expected_torque / ke  # 5/200 = 0.025 rad
                self.assertAlmostEqual(
                    theta_eq,
                    expected_theta,
                    delta=0.005,
                    msg=f"theta_eq={theta_eq:.5f} rad, expected ~{expected_theta:.5f} rad",
                )

                # Reported torques on both joints. spatial_vector layout
                # [fx, fy, fz, tx, ty, tz]; both joints' axes are +z.
                wrenches = wp.zeros(world.num_constraints, dtype=wp.spatial_vector, device=device)
                world.gather_constraint_wrenches(wrenches)
                w0 = wrenches.numpy()[0]  # PD position drive (joint 0)
                w1 = wrenches.numpy()[1]  # velocity drive (joint 1)
                tau_pos_z = float(w0[5])
                tau_vel_z = float(w1[5])

                # Velocity drive torque magnitude ~ kd_vel * target_vel.
                self.assertAlmostEqual(
                    abs(tau_vel_z),
                    expected_torque,
                    delta=0.10 * expected_torque,
                    msg=f"velocity drive torque |{tau_vel_z:.4f}| N*m vs "
                        f"expected {expected_torque:.4f} N*m",
                )

                # Position drive torque must equal velocity drive torque
                # in magnitude and oppose it in sign (Newton 3rd law on
                # the shared cube).
                self.assertAlmostEqual(
                    tau_pos_z + tau_vel_z,
                    0.0,
                    delta=0.10 * expected_torque,
                    msg=f"torque sum {tau_pos_z + tau_vel_z:.4f} N*m -- "
                        f"opposed drives not balancing (pos={tau_pos_z:.4f}, "
                        f"vel={tau_vel_z:.4f})",
                )


if __name__ == "__main__":
    wp.init()
    unittest.main()
