# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the generalised D6 (6-DoF) constraint.

A D6 owns *all* six relative degrees of freedom between body 1 and
body 2 and configures each of the 6 axes (3 angular + 3 linear)
independently via a :class:`D6AxisDrive`. Internally the joint
dispatches each block to one of two paths:

* **Translation block.** When all 3 linear axes are rigid the joint
  uses a fused 3x3 ``PointConstraintPart`` (Schur-style). Otherwise
  each linear axis becomes its own 1-DoF ``AxisConstraintPart``.
* **Rotation block.** Same dichotomy: a fused 3x3
  ``RotationEulerConstraintPart`` when every angular axis is rigid,
  or three independent 1-DoF ``AngleConstraintPart`` rows.

These tests exercise every documented per-axis state (rigid lock /
soft lock / position drive / velocity drive / free) on both blocks,
plus both dispatch paths (fused 3-DoF vs per-axis 1-DoF), the
wrench-reporting path, and Newton-3 momentum conservation when the
D6 is hung between two free bodies.

The most important regression they protect against is the
"world-body identity-inertia" footgun fixed in
:meth:`WorldBuilder.__init__`: if the auto-created ``world_body`` ever
goes back to having identity inverse inertia, every angular drive on
a D6 anchored to ``b.world_body`` will silently saturate at half its
target. :meth:`TestD6.test_velocity_motor_via_world_body` locks that
in. The picking / tilted-gravity hinge-chain explosion that motivated
the Jolt-style port is covered by
:meth:`TestD6.test_revolute_chain_under_tilted_gravity`.

All scenes are also registered with :func:`scene` so the test
visualizer can replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter._test_helpers import run_settle_loop
from newton._src.solvers.jitter.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import D6AxisDrive, WorldBuilder

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 120  # 2 s @ 60 fps -- PGS warm-start converges well within this
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_anchored_d6(
    device,
    *,
    angular: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
    linear: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
    cube_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = False,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_world_body: bool = False,
    gravity: tuple[float, float, float] = (0.0, -GRAVITY, 0.0),
):
    """Static body 1 + dynamic cube body 2 joined by a single D6.

    Defaults give a 6-DoF rigid weld (every axis defaults to
    :class:`D6AxisDrive` rigid). Pass per-axis overrides to soften /
    drive / free axes.

    ``use_world_body=True`` swaps the explicit static anchor for
    ``b.world_body``; this is the regression path for the world-body
    identity-inertia fix.
    """
    b = WorldBuilder()
    if use_world_body:
        anchor_body = b.world_body
    else:
        anchor_body = b.add_static_body()
    cube = b.add_dynamic_body(
        position=cube_position,
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        velocity=initial_velocity,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_d6(
        body1=anchor_body,
        body2=cube,
        anchor=(0.0, 0.0, 0.0),
        angular=angular,
        linear=linear,
    )
    world = b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
        gravity=gravity,
    )
    return world, cube, handle


def _build_two_body_d6(
    device,
    *,
    angular: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
    linear: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
    initial_angular_velocity_b1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_angular_velocity_b2: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Two free dynamic cubes joined by a single D6.

    Both bodies are gravity-free so the only thing that can change
    their state is the joint -- isolates Newton-3 momentum
    conservation.
    """
    b = WorldBuilder()
    b1 = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity_b1,
    )
    b2 = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity_b2,
    )
    # Anchor coincides with both COMs -> the linear rigid lock has no
    # lever arm to leak angular impulse through, so the motor is the
    # only thing exchanging axial angular momentum between the two
    # bodies (cleanly tests Newton 3 on the angular row).
    handle = b.add_d6(
        body1=b1, body2=b2, anchor=(0.0, 0.0, 0.0), angular=angular, linear=linear
    )
    return (
        b.finalize(
            substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
        ),
        b1,
        b2,
        handle,
    )


def _scene_he(num_bodies: int, *, skip_first: int = 1) -> np.ndarray:
    he = np.zeros((num_bodies, 3), dtype=np.float32)
    he[skip_first:] = HALF_EXTENT
    return he


# ---------------------------------------------------------------------------
# Visualizer scenes
# ---------------------------------------------------------------------------


@scene(
    "D6: rigid weld",
    description="Default D6 with all 6 axes rigid; cube is glued to anchor.",
    tags=("d6",),
)
def build_d6_rigid_weld_scene(device) -> Scene:
    world, _, _ = _build_anchored_d6(device, affected_by_gravity=True)
    return Scene(
        world=world, body_half_extents=_scene_he(3), frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "D6: revolute about +z",
    description=(
        "Rigid linear lock + 2 rigid angular axes + 1 free angular axis (+z); "
        "behaves like a hinge."
    ),
    tags=("d6", "revolute"),
)
def build_d6_revolute_scene(device) -> Scene:
    angular = (D6AxisDrive(), D6AxisDrive(), D6AxisDrive(max_force=0.0))
    world, _, _ = _build_anchored_d6(
        device, angular=angular, affected_by_gravity=True,
        cube_position=(0.0, -1.0, 0.0),
    )
    return Scene(
        world=world, body_half_extents=_scene_he(3), frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "D6: angular velocity motor (+z, target=2 rad/s)",
    description="5-axis lock + velocity motor on +z; cube spins up to 2 rad/s.",
    tags=("d6", "motor"),
)
def build_d6_velocity_motor_scene(device) -> Scene:
    motor = D6AxisDrive(hertz=10.0, target_velocity=2.0, max_force=20.0)
    angular = (D6AxisDrive(), D6AxisDrive(), motor)
    world, _, _ = _build_anchored_d6(device, angular=angular)
    return Scene(
        world=world, body_half_extents=_scene_he(3), frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


def _twist_about_z(q) -> float:
    """Signed rotation angle about world +z for a unit quaternion ``q`` (xyzw).

    Valid when the swing component is small (the case for every test
    here, where the angular DoF that's free is +z and the others are
    rigidly locked).
    """
    return 2.0 * math.atan2(float(q[2]), float(q[3]))


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestD6(unittest.TestCase):
    """End-to-end physics checks for :func:`WorldBuilder.add_d6`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        run_settle_loop(world, frames, dt=1.0 / FPS)

    # ------------------------------------------------------------------
    # Rigid weld -- both fused dispatch branches active
    # ------------------------------------------------------------------

    def test_rigid_weld_locks_all_six_dof(self):
        """Default D6 must hold body 2 glued to body 1 in all 6 DoF.

        Exercises the *fused* dispatch on both blocks at once:
        ``trans_mode=POINT`` (3-DoF point lock) and
        ``rot_mode=EULER`` (3-DoF Euler lock). Under gravity + a
        non-trivial off-axis spin the cube must stay at the anchor
        with its starting orientation.
        """
        device = wp.get_preferred_device()
        world, cube, _ = _build_anchored_d6(
            device,
            affected_by_gravity=True,
            initial_angular_velocity=(0.6, -0.4, 0.3),
        )
        self._step(world)

        positions = world.bodies.position.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        orientations = world.bodies.orientation.numpy()
        self.assertLess(
            np.linalg.norm(positions[cube]),
            0.02,
            msg=f"rigid D6 let cube drift: pos={positions[cube]}",
        )
        self.assertLess(
            np.linalg.norm(omegas[cube]),
            0.05,
            msg=f"rigid D6 left residual spin: omega={omegas[cube]}",
        )
        # Quaternion (xyzw) close to identity (0,0,0,1).
        q = orientations[cube]
        self.assertGreater(
            abs(float(q[3])),
            math.cos(math.radians(2.0)),
            msg=f"rigid D6 let cube rotate: q={q}",
        )

    def test_rigid_weld_carries_weight_in_wrench(self):
        """Gravity-loaded rigid weld reports +y reaction = mass * |g|.

        Anchored at the cube's COM, the weld must carry the cube's
        weight in pure +y force with negligible torque (no lever arm
        from the anchor to the COM).
        """
        device = wp.get_preferred_device()
        world, cube, handle = _build_anchored_d6(device, affected_by_gravity=True)
        self._step(world)

        out = wp.zeros(world.num_constraints, dtype=wp.spatial_vector, device=device)
        world.gather_constraint_wrenches(out)
        wrench = out.numpy()[handle.cid]
        fx, fy, fz, tx, ty, tz = wrench
        self.assertTrue(np.isfinite(wrench).all(), msg=f"non-finite wrench: {wrench}")
        self.assertAlmostEqual(
            fy,
            GRAVITY,
            delta=0.05,
            msg=f"expected fy~{GRAVITY:.3f} N, got {fy:.4f}",
        )
        for label, val in (("fx", fx), ("fz", fz), ("tx", tx), ("ty", ty), ("tz", tz)):
            self.assertLess(
                abs(val),
                0.5,
                msg=f"{label}={val:.4f} should be near zero for symmetric weld",
            )

    # ------------------------------------------------------------------
    # Free axes / per-axis dispatch
    # ------------------------------------------------------------------

    def test_revolute_free_z_preserves_spin(self):
        """``max_force=0`` on angular +z makes that axis fully free.

        The other 5 axes are rigid. Spinning the cube about +z must
        leave the spin untouched (no impulse on the free axis), and
        all other DoF must stay locked.
        """
        device = wp.get_preferred_device()
        omega_axial = 2.0
        angular = (D6AxisDrive(), D6AxisDrive(), D6AxisDrive(max_force=0.0))
        world, cube, _ = _build_anchored_d6(
            device,
            angular=angular,
            initial_angular_velocity=(0.0, 0.0, omega_axial),
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        positions = world.bodies.position.numpy()
        self.assertAlmostEqual(
            omegas[cube, 2],
            omega_axial,
            delta=0.05,
            msg=f"free +z bled spin to {omegas[cube, 2]}",
        )
        self.assertLess(
            math.hypot(omegas[cube, 0], omegas[cube, 1]),
            0.05,
            msg=f"locked perpendicular axes leaked spin: {omegas[cube]}",
        )
        self.assertLess(
            np.linalg.norm(positions[cube]),
            0.02,
            msg=f"linear lock failed: pos={positions[cube]}",
        )

    def test_free_linear_y_under_gravity(self):
        """``max_force=0`` on linear +y reduces the joint to a vertical slider.

        Exercises ``trans_mode=AXIS`` (per-axis 1-DoF rows) +
        ``rot_mode=EULER`` (fused 3-DoF angular lock). Under gravity
        the cube falls freely along +y at ``g``; lateral and angular
        DoF stay clamped.
        """
        device = wp.get_preferred_device()
        linear = (D6AxisDrive(), D6AxisDrive(max_force=0.0), D6AxisDrive())
        world, cube, _ = _build_anchored_d6(
            device, linear=linear, affected_by_gravity=True,
        )
        frames = 120  # 2 s
        t = frames / FPS
        self._step(world, frames=frames)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        expected_y = -0.5 * GRAVITY * t * t
        self.assertAlmostEqual(
            positions[cube, 1],
            expected_y,
            delta=0.05 * abs(expected_y),
            msg=f"expected y~{expected_y:.3f} m, got {positions[cube, 1]:.4f}",
        )
        self.assertLess(
            abs(positions[cube, 0]),
            0.02,
            msg=f"x leaked: {positions[cube, 0]}",
        )
        self.assertLess(
            abs(positions[cube, 2]),
            0.02,
            msg=f"z leaked: {positions[cube, 2]}",
        )
        self.assertLess(
            abs(velocities[cube, 0]) + abs(velocities[cube, 2]),
            0.05,
            msg=f"lateral velocity leaked: v={velocities[cube]}",
        )

    # ------------------------------------------------------------------
    # Drives (velocity / position / soft lock)
    # ------------------------------------------------------------------

    def test_velocity_motor_reaches_target(self):
        """Velocity motor on angular +z must drive the cube to the setpoint.

        With other 5 axes locked and the world body acting as an
        infinite rotational sink, the motor should saturate the
        relative axial velocity at ``target`` -- and because body 1
        is static, the cube's absolute angular velocity equals the
        relative one.
        """
        device = wp.get_preferred_device()
        target = 3.0
        motor = D6AxisDrive(hertz=10.0, target_velocity=target, max_force=200.0)
        angular = (D6AxisDrive(), D6AxisDrive(), motor)
        world, cube, _ = _build_anchored_d6(device, angular=angular)
        self._step(world, frames=120)

        omegas = world.bodies.angular_velocity.numpy()
        positions = world.bodies.position.numpy()
        self.assertAlmostEqual(
            omegas[cube, 2],
            target,
            delta=0.1,
            msg=f"velocity motor at {omegas[cube, 2]:.3f}, expected {target}",
        )
        self.assertLess(
            math.hypot(omegas[cube, 0], omegas[cube, 1]),
            0.05,
            msg=f"motor leaked into perpendicular axes: {omegas[cube]}",
        )
        self.assertLess(
            np.linalg.norm(positions[cube]),
            0.02,
            msg=f"linear lock failed under motor torque: pos={positions[cube]}",
        )

    def test_velocity_motor_via_world_body(self):
        """Velocity motor anchored to ``b.world_body`` must reach the target.

        Regression test for the world-body identity-inertia fix in
        :meth:`WorldBuilder.__init__`. Before the fix the auto-
        created world body had identity ``inverse_inertia`` and the
        motor's per-axis effective mass was ``1 / (Ia^-1 + Ib^-1) =
        0.5`` (instead of ``1.0`` for an "infinite sink"); the per-
        substep impulse cap then saturated the cube at half the
        target. If a future change reverts the seed to identity, the
        cube here will only reach ``target / 2``.
        """
        device = wp.get_preferred_device()
        target = 3.0
        motor = D6AxisDrive(hertz=10.0, target_velocity=target, max_force=200.0)
        angular = (D6AxisDrive(), D6AxisDrive(), motor)
        world, cube, _ = _build_anchored_d6(
            device, angular=angular, use_world_body=True,
        )
        self._step(world, frames=120)

        omegas = world.bodies.angular_velocity.numpy()
        self.assertAlmostEqual(
            omegas[cube, 2],
            target,
            delta=0.1,
            msg=(
                f"motor anchored to world_body landed at {omegas[cube, 2]:.4f}, "
                f"expected {target} (regression: world_body identity-inertia bug)"
            ),
        )

    def test_position_drive_reaches_target_angle(self):
        """Position drive on angular +z must seat the cube near ``target``.

        Critically-damped 2 Hz drive, generous force cap; after 4 s
        the relative twist must be within 0.05 rad of the target.
        """
        device = wp.get_preferred_device()
        target = math.pi / 2.0
        drive = D6AxisDrive(
            hertz=2.0, damping_ratio=1.0, target_position=target, max_force=200.0,
        )
        angular = (D6AxisDrive(), D6AxisDrive(), drive)
        world, cube, _ = _build_anchored_d6(device, angular=angular)
        self._step(world)

        orientations = world.bodies.orientation.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        twist = _twist_about_z(orientations[cube])
        self.assertAlmostEqual(
            twist,
            target,
            delta=0.05,
            msg=f"position drive landed at {twist:.4f} rad, expected {target:.4f}",
        )
        self.assertLess(
            np.linalg.norm(omegas[cube]),
            0.05,
            msg=f"position drive left residual spin: {omegas[cube]}",
        )

    def test_position_drive_reaches_target_linear(self):
        """Position drive on linear +y must seat the cube near ``target_position``.

        Exercises the ``AxisConstraintPart`` + soft-spring branch on
        the linear block.
        """
        device = wp.get_preferred_device()
        target = 0.4
        drive = D6AxisDrive(
            hertz=2.0, damping_ratio=1.0, target_position=target, max_force=200.0,
        )
        linear = (D6AxisDrive(), drive, D6AxisDrive())
        world, cube, _ = _build_anchored_d6(device, linear=linear)
        self._step(world)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        self.assertAlmostEqual(
            positions[cube, 1],
            target,
            delta=0.02,
            msg=f"linear pos drive landed at y={positions[cube, 1]:.4f}, want {target}",
        )
        self.assertLess(
            abs(positions[cube, 0]) + abs(positions[cube, 2]),
            0.02,
            msg=f"locked lateral axes drifted: pos={positions[cube]}",
        )
        self.assertLess(
            np.linalg.norm(velocities[cube]),
            0.05,
            msg=f"position drive left residual velocity: v={velocities[cube]}",
        )

    def test_soft_linear_lock_sags_under_gravity(self):
        """Soft (``hertz>0``, no target) linear +y matches analytic spring sag.

        Steady-state of an implicit spring with mass m=1 and effective
        stiffness ``k = m * (2*pi*hertz)^2`` under gravity g is
        ``y_ss = g / k`` (negative for ``g < 0``). After a long
        critically-damped settle the cube must sit near that value
        with zero residual velocity.
        """
        device = wp.get_preferred_device()
        hertz = 2.0
        soft = D6AxisDrive(hertz=hertz, damping_ratio=1.0)
        linear = (D6AxisDrive(), soft, D6AxisDrive())
        world, cube, _ = _build_anchored_d6(
            device, linear=linear, affected_by_gravity=True,
        )
        self._step(world, frames=300)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        omega = 2.0 * math.pi * hertz
        k = omega * omega  # mass = 1
        expected_y = -GRAVITY / k
        self.assertAlmostEqual(
            positions[cube, 1],
            expected_y,
            delta=0.02,
            msg=f"soft sag y={positions[cube, 1]:.4f} expected {expected_y:.4f}",
        )
        self.assertLess(
            abs(velocities[cube, 1]),
            0.05,
            msg=f"soft cube still moving: v_y={velocities[cube, 1]}",
        )

    def test_force_cap_saturates_velocity_motor(self):
        """A ``max_force`` smaller than equilibrium need must clamp the motor.

        Ramps the motor target to a high setpoint with a tiny
        ``max_force``; the motor cannot overcome the per-substep
        impulse cap and the steady-state spin sits well below the
        target -- but still in the right direction.
        """
        device = wp.get_preferred_device()
        target = 50.0  # huge target
        max_f = 0.05  # tiny force cap
        motor = D6AxisDrive(hertz=10.0, target_velocity=target, max_force=max_f)
        angular = (D6AxisDrive(), D6AxisDrive(), motor)
        world, cube, _ = _build_anchored_d6(device, angular=angular)
        self._step(world, frames=120)

        omegas = world.bodies.angular_velocity.numpy()
        # Per-substep cap: max_force * dt; over 120 frames * SUBSTEPS the
        # cube can pick up at most ~ max_f * (frames * SUBSTEPS) * dt rad/s
        # = max_f * (frames / FPS) = 0.05 * 2 = 0.1 rad/s. Allow some
        # slack for the implicit-Euler integrator.
        achievable = max_f * (120.0 / FPS) * 1.5
        self.assertLess(
            omegas[cube, 2],
            achievable,
            msg=f"force cap failed: omega_z={omegas[cube, 2]} >= {achievable}",
        )
        self.assertGreater(
            omegas[cube, 2],
            0.001,
            msg=f"capped motor went the wrong direction: omega_z={omegas[cube, 2]}",
        )

    # ------------------------------------------------------------------
    # Wrench reporting
    # ------------------------------------------------------------------

    def test_wrench_force_matches_weight_with_per_axis_lock(self):
        """Per-axis linear lock must still report the gravity reaction.

        Same physics as :meth:`test_rigid_weld_carries_weight_in_wrench`
        but now the linear block is in the per-axis ``trans_mode=AXIS``
        branch (because we soften +x slightly to force dispatch). The
        +y reaction must still match ``mass * |g|`` and lateral
        components must stay near zero.
        """
        device = wp.get_preferred_device()
        # Soften +x with a generous spring -- this kicks the linear
        # block out of the fused PointConstraint path and into the
        # per-axis AxisConstraint path. Lock +y rigidly so it carries
        # the weight cleanly.
        linear = (
            D6AxisDrive(hertz=50.0, damping_ratio=1.0),  # very stiff soft lock
            D6AxisDrive(),                               # rigid
            D6AxisDrive(),                               # rigid
        )
        world, cube, handle = _build_anchored_d6(
            device, linear=linear, affected_by_gravity=True,
        )
        self._step(world, frames=600)

        out = wp.zeros(world.num_constraints, dtype=wp.spatial_vector, device=device)
        world.gather_constraint_wrenches(out)
        wrench = out.numpy()[handle.cid]
        fx, fy, fz, _, _, _ = wrench
        self.assertAlmostEqual(
            fy,
            GRAVITY,
            delta=0.2,
            msg=f"per-axis path: expected fy~{GRAVITY:.3f} N, got {fy:.4f}",
        )
        self.assertLess(
            abs(fx) + abs(fz),
            0.5,
            msg=f"per-axis lateral force leaked: fx={fx:.3f}, fz={fz:.3f}",
        )

    # ------------------------------------------------------------------
    # Newton-3 / momentum conservation
    # ------------------------------------------------------------------

    def test_two_body_motor_conserves_axial_momentum(self):
        """Velocity motor between two free bodies must conserve momentum.

        D6 with rigid lock on every axis except the angular +z
        velocity-motor row. Gravity is off so the motor is the only
        thing applying torque. Whatever the motor adds to body 2's
        +z spin it must take from body 1 in equal measure (Newton 3).
        """
        device = wp.get_preferred_device()
        omega1, omega2 = -0.4, 1.6
        target = 5.0  # impossible target -> motor saturates torque budget
        motor = D6AxisDrive(hertz=10.0, target_velocity=target, max_force=10.0)
        angular = (D6AxisDrive(), D6AxisDrive(), motor)
        world, b1, b2, _ = _build_two_body_d6(
            device,
            angular=angular,
            initial_angular_velocity_b1=(0.0, 0.0, omega1),
            initial_angular_velocity_b2=(0.0, 0.0, omega2),
        )
        net_initial = omega1 + omega2
        self._step(world, frames=60)

        omegas = world.bodies.angular_velocity.numpy()
        net_final = omegas[b1, 2] + omegas[b2, 2]
        self.assertAlmostEqual(
            net_final,
            net_initial,
            delta=0.1,
            msg=(
                f"axial angular momentum drifted: net_initial={net_initial:.4f}, "
                f"net_final={net_final:.4f}"
            ),
        )

    # ------------------------------------------------------------------
    # Stability under load (the original explosion repro)
    # ------------------------------------------------------------------

    def test_revolute_chain_under_tilted_gravity(self):
        """A 5-cube D6_REVOLUTE chain under tilted gravity must not explode.

        Reproduces the failure mode that motivated the Jolt-style
        rewrite of :mod:`constraint_d6`. Each link is a D6 with the
        translation block fully locked and the +z angular axis free
        (a "revolute" via ``max_force=0``). Tilted gravity gives
        every joint a non-zero torque budget so the chain swings
        rather than sitting trivially at equilibrium. After several
        simulated seconds positions must stay finite and bounded.
        """
        device = wp.get_preferred_device()
        num_cubes = 5
        b = WorldBuilder()
        anchor = b.add_static_body()
        cubes = []
        for j in range(num_cubes):
            cubes.append(
                b.add_dynamic_body(
                    position=(0.0, -(2 * j + 1) * HALF_EXTENT, 0.0),
                    inverse_mass=1.0,
                    inverse_inertia=_INV_INERTIA,
                )
            )
        d6_lock = D6AxisDrive()
        d6_free = D6AxisDrive(max_force=0.0)
        d6_angular = (d6_lock, d6_lock, d6_free)
        d6_linear = (d6_lock, d6_lock, d6_lock)
        for k in range(num_cubes):
            body_a = anchor if k == 0 else cubes[k - 1]
            body_b = cubes[k]
            anchor_pos = (0.0, -k * 2.0 * HALF_EXTENT, 0.0)
            b.add_d6(
                body1=body_a, body2=body_b, anchor=anchor_pos,
                angular=d6_angular, linear=d6_linear,
            )
        world = b.finalize(
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
            device=device,
            gravity=(0.5 * GRAVITY, -0.866 * GRAVITY, 0.0),  # 30° tilt
        )
        # 4 simulated seconds; if the joint is unstable the chain
        # shoots off to non-finite positions within a handful of frames.
        self._step(world, frames=240)

        positions = world.bodies.position.numpy()
        # Skip body 0 (auto world) + body 1 (explicit static anchor).
        for i, cube in enumerate(cubes):
            self.assertTrue(
                np.isfinite(positions[cube]).all(),
                msg=f"cube {i} produced non-finite position: {positions[cube]}",
            )
            # No cube should travel further than ~ chain length below
            # the anchor; a soft bound catches the typical "rocket out
            # to infinity" failure mode without being brittle.
            self.assertLess(
                np.linalg.norm(positions[cube]),
                4.0 * num_cubes,
                msg=f"cube {i} fell unreasonably far: pos={positions[cube]}",
            )


if __name__ == "__main__":
    wp.init()
    unittest.main()
