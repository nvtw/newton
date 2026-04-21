# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-check tests: ActuatedDoubleBallSocket against the standalone
stand-ins it was meant to subsume.

The unified :meth:`WorldBuilder.add_joint` constraint is *supposed* to
behave identically to the "reference" composition of standalone pieces:

* ``REVOLUTE + PD drive``  ==  ``add_double_ball_socket + add_angular_motor``
* ``REVOLUTE + PD limit``  ==  ``add_double_ball_socket + add_angular_limit``
* ``PRISMATIC + PD drive`` ==  ``add_double_ball_socket_prismatic
                                 + add_linear_motor``
* ``PRISMATIC + PD limit`` ==  ``add_double_ball_socket_prismatic
                                 + add_linear_limit``

The standalone constraints (:mod:`constraint_angular_motor`,
:mod:`constraint_angular_limit`, :mod:`constraint_linear_motor`,
:mod:`constraint_linear_limit`) have their own dedicated analytical
test suites and are treated as ground truth here. Each test in this
module builds two worlds in parallel:

* A **reference** world that composes the two standalone pieces.
* An **ADBS** world that encodes the same physics through the unified
  :meth:`WorldBuilder.add_joint` API.

Both worlds are stepped for the same number of frames with identical
initial conditions, then the bob body's full state (position,
orientation, linear / angular velocity) is compared. Any disagreement
above a tight tolerance is a genuine behaviour bug in the unified
constraint -- the reference pieces have *already* been validated
against closed-form physics (DHO identity, Hooke's-law equilibrium)
in their own modules, so divergence is unambiguously attributable to
the unified path.

We deliberately avoid the Newton-model pipeline (:class:`_JitterScene`)
here. A bare :class:`WorldBuilder` plus direct :meth:`World.step` keeps
the mirror minimal and the failure mode as local as possible.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter._test_helpers import run_settle_loop
from newton._src.solvers.jitter.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

# Fine substep + many PGS iterations so the two paths *do* converge to
# the same steady state. The two constraint implementations share the
# same PGS driver but order constraints differently in the column
# array, so they can disagree slightly after only a handful of
# iterations; we want them to agree in the converged limit.
FPS = 240
SUBSTEPS = 8
SOLVER_ITERATIONS = 32
DT = 1.0 / FPS

# Tolerances. Both worlds run the same math at the same iteration
# count; differences are first-order in PGS residual, so sub-millimetre
# / sub-microradian agreement is expected once settled. We leave some
# slack for the slight ordering differences in the column pack.
POSITION_TOL = 1.0e-3    # m
ORIENTATION_TOL = 1.0e-3  # rad-equivalent quaternion dot deficit
VELOCITY_TOL = 5.0e-3    # m/s or rad/s

_GRAVITY_OFF = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finalize(builder: WorldBuilder, device, *, gravity):
    return builder.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=gravity,
        device=device,
    )


def _snapshot(world, body: int) -> dict[str, np.ndarray]:
    """Read body state out of a finalised world into host arrays."""
    return {
        "position": world.bodies.position.numpy()[body].copy(),
        "orientation": world.bodies.orientation.numpy()[body].copy(),
        "velocity": world.bodies.velocity.numpy()[body].copy(),
        "angular_velocity": world.bodies.angular_velocity.numpy()[body].copy(),
    }


def _quat_angle_between(qa: np.ndarray, qb: np.ndarray) -> float:
    """Angle (rad) between two unit quaternions, robust to double-cover.

    ``2 * acos(|qa . qb|)``; the absolute value folds the sign
    ambiguity (``q`` and ``-q`` represent the same rotation) so
    small-rotation cases return ~0 rather than ~2*pi.
    """
    dot = float(abs(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _assert_states_agree(
    test: unittest.TestCase,
    ref: dict[str, np.ndarray],
    adbs: dict[str, np.ndarray],
    label: str,
    *,
    pos_tol: float = POSITION_TOL,
    orient_tol: float = ORIENTATION_TOL,
    vel_tol: float = VELOCITY_TOL,
) -> None:
    """Compare the two snapshots. Differences above tolerance fail the
    test with a human-readable component breakdown."""
    d_pos = float(np.linalg.norm(ref["position"] - adbs["position"]))
    d_orient = _quat_angle_between(ref["orientation"], adbs["orientation"])
    d_vel = float(np.linalg.norm(ref["velocity"] - adbs["velocity"]))
    d_avel = float(
        np.linalg.norm(ref["angular_velocity"] - adbs["angular_velocity"])
    )
    test.assertLess(
        d_pos,
        pos_tol,
        msg=(
            f"[{label}] position disagreement {d_pos:.2e} m > {pos_tol:.2e} "
            f"(ref={ref['position']}, adbs={adbs['position']})"
        ),
    )
    test.assertLess(
        d_orient,
        orient_tol,
        msg=(
            f"[{label}] orientation disagreement {d_orient:.2e} rad > "
            f"{orient_tol:.2e} (ref={ref['orientation']}, "
            f"adbs={adbs['orientation']})"
        ),
    )
    test.assertLess(
        d_vel,
        vel_tol,
        msg=(
            f"[{label}] linear-velocity disagreement {d_vel:.2e} m/s > "
            f"{vel_tol:.2e} (ref={ref['velocity']}, adbs={adbs['velocity']})"
        ),
    )
    test.assertLess(
        d_avel,
        vel_tol,
        msg=(
            f"[{label}] angular-velocity disagreement {d_avel:.2e} rad/s > "
            f"{vel_tol:.2e} (ref={ref['angular_velocity']}, "
            f"adbs={adbs['angular_velocity']})"
        ),
    )


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------


def _add_bob(
    b: WorldBuilder,
    *,
    position,
    mass: float,
    inertia_cm: float,
    orientation=(0.0, 0.0, 0.0, 1.0),
    initial_omega_z: float = 0.0,
    initial_vx: float = 0.0,
    affected_by_gravity: bool = False,
) -> int:
    """Isotropic-inertia bob; used by both ref and ADBS scenes so the
    body parameters are identical modulo the constraint plumbing."""
    inv_i = 1.0 / inertia_cm
    return b.add_dynamic_body(
        position=position,
        orientation=orientation,
        inverse_mass=1.0 / mass,
        inverse_inertia=(
            (inv_i, 0.0, 0.0),
            (0.0, inv_i, 0.0),
            (0.0, 0.0, inv_i),
        ),
        linear_damping=1.0,
        angular_damping=1.0,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=(0.0, 0.0, float(initial_omega_z)),
        velocity=(float(initial_vx), 0.0, 0.0),
    )


# ----- Revolute PD drive scenes --------------------------------------------


def _build_revolute_drive_reference(
    device, *, mass, lever, inertia_cm, k, c, target_angle, omega0, gravity
):
    """Reference revolute + PD drive via standalone pieces."""
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(lever, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_omega_z=omega0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_double_ball_socket(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, -0.25),
        anchor2=(0.0, 0.0, +0.25),
        hertz=0.0,
    )
    b.add_angular_motor(
        body1=pivot,
        body2=bob,
        axis=(0.0, 0.0, 1.0),
        target_angle=target_angle,
        stiffness=k,
        damping=c,
        max_force=1.0e6,
    )
    return _finalize(b, device, gravity=gravity), bob


def _build_revolute_drive_adbs(
    device, *, mass, lever, inertia_cm, k, c, target_angle, omega0, gravity
):
    """Same physics encoded through the unified add_joint API."""
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(lever, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_omega_z=omega0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_joint(
        body1=pivot,
        body2=bob,
        # Two anchors on the +Z hinge axis at the pivot's XY origin, so
        # ``anchor2 - anchor1`` = +Z and the ADBS kernel's derived
        # n_hat matches the standalone angular-motor's ``axis = +Z``.
        anchor1=(0.0, 0.0, -0.25),
        anchor2=(0.0, 0.0, +0.25),
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.POSITION,
        target=target_angle,
        max_force_drive=1.0e6,
        stiffness_drive=k,
        damping_drive=c,
        # Match the reference's ``hertz=0.0`` (rigid lock) so the two
        # worlds stay trajectory-identical rather than differing by the
        # anchor's soft-spring response.
        hertz=0.0,
    )
    return _finalize(b, device, gravity=gravity), bob


# ----- Revolute PD limit scenes --------------------------------------------


def _build_revolute_limit_reference(
    device,
    *,
    mass,
    lever,
    inertia_cm,
    k_lim,
    c_lim,
    min_angle,
    max_angle,
    omega0,
    gravity,
):
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(lever, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_omega_z=omega0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_double_ball_socket(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, -0.25),
        anchor2=(0.0, 0.0, +0.25),
        hertz=0.0,
    )
    b.add_angular_limit(
        body1=pivot,
        body2=bob,
        axis=(0.0, 0.0, 1.0),
        min_value=min_angle,
        max_value=max_angle,
        stiffness=k_lim,
        damping=c_lim,
    )
    return _finalize(b, device, gravity=gravity), bob


def _build_revolute_limit_adbs(
    device,
    *,
    mass,
    lever,
    inertia_cm,
    k_lim,
    c_lim,
    min_angle,
    max_angle,
    omega0,
    gravity,
):
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(lever, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_omega_z=omega0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_joint(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, -0.25),
        anchor2=(0.0, 0.0, +0.25),
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.OFF,
        min_value=min_angle,
        max_value=max_angle,
        stiffness_limit=k_lim,
        damping_limit=c_lim,
        hertz=0.0,
    )
    return _finalize(b, device, gravity=gravity), bob


# ----- Prismatic PD drive scenes -------------------------------------------


def _build_prismatic_drive_reference(
    device, *, mass, inertia_cm, k, c, target_position, v0, gravity
):
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(0.0, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_vx=v0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_double_ball_socket_prismatic(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        hertz=0.0,
    )
    b.add_linear_motor(
        body1=pivot,
        body2=bob,
        axis=(1.0, 0.0, 0.0),
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 0.0),
        target_position=target_position,
        stiffness=k,
        damping=c,
        max_force=1.0e6,
    )
    return _finalize(b, device, gravity=gravity), bob


def _build_prismatic_drive_adbs(
    device, *, mass, inertia_cm, k, c, target_position, v0, gravity
):
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(0.0, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_vx=v0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_joint(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        mode=JointMode.PRISMATIC,
        drive_mode=DriveMode.POSITION,
        target=target_position,
        max_force_drive=1.0e6,
        stiffness_drive=k,
        damping_drive=c,
        hertz=0.0,
    )
    return _finalize(b, device, gravity=gravity), bob


# ----- Prismatic PD limit scenes -------------------------------------------


def _build_prismatic_limit_reference(
    device,
    *,
    mass,
    inertia_cm,
    k_lim,
    c_lim,
    min_pos,
    max_pos,
    v0,
    gravity,
):
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(0.0, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_vx=v0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_double_ball_socket_prismatic(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        hertz=0.0,
    )
    b.add_linear_limit(
        body1=pivot,
        body2=bob,
        axis=(1.0, 0.0, 0.0),
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 0.0),
        min_value=min_pos,
        max_value=max_pos,
        stiffness=k_lim,
        damping=c_lim,
    )
    return _finalize(b, device, gravity=gravity), bob


def _build_prismatic_limit_adbs(
    device,
    *,
    mass,
    inertia_cm,
    k_lim,
    c_lim,
    min_pos,
    max_pos,
    v0,
    gravity,
):
    b = WorldBuilder()
    pivot = b.world_body
    bob = _add_bob(
        b,
        position=(0.0, 0.0, 0.0),
        mass=mass,
        inertia_cm=inertia_cm,
        initial_vx=v0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
    )
    b.add_joint(
        body1=pivot,
        body2=bob,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        mode=JointMode.PRISMATIC,
        drive_mode=DriveMode.OFF,
        min_value=min_pos,
        max_value=max_pos,
        stiffness_limit=k_lim,
        damping_limit=c_lim,
        hertz=0.0,
    )
    return _finalize(b, device, gravity=gravity), bob


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "ADBS cross-check tests require CUDA (graph capture keeps runtime "
    "reasonable for the fine-grained substep counts used here).",
)
class TestActuatedDBSvsStandalone(unittest.TestCase):
    """Step-by-step equivalence between ADBS and standalone combos."""

    # --- Revolute drive ----------------------------------------------------

    def test_revolute_pd_drive_free_oscillation(self):
        """Release a bob from the PD target with an initial angular
        kick; compare trajectories after one DHO period. Tests both
        the bias-from-C = 0 branch (we're at the target) and the
        damping-driven decay.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 4.0e-3
        I_eff = inertia_cm + mass * lever * lever
        k = 400.0
        omega0 = math.sqrt(k / I_eff)
        zeta = 0.1
        c = 2.0 * zeta * omega0 * I_eff
        kick = 0.5

        for label, gravity in (
            ("gravity-off", _GRAVITY_OFF),
            ("gravity-on", (0.0, -9.81, 0.0)),
        ):
            world_ref, bob_ref = _build_revolute_drive_reference(
                device,
                mass=mass,
                lever=lever,
                inertia_cm=inertia_cm,
                k=k,
                c=c,
                target_angle=0.0,
                omega0=kick,
                gravity=gravity,
            )
            world_adbs, bob_adbs = _build_revolute_drive_adbs(
                device,
                mass=mass,
                lever=lever,
                inertia_cm=inertia_cm,
                k=k,
                c=c,
                target_angle=0.0,
                omega0=kick,
                gravity=gravity,
            )
            # 0.75 s: covers ~2 damped periods for this k / I_eff.
            run_settle_loop(world_ref, frames=int(0.75 * FPS), dt=DT)
            run_settle_loop(world_adbs, frames=int(0.75 * FPS), dt=DT)
            _assert_states_agree(
                self,
                _snapshot(world_ref, bob_ref),
                _snapshot(world_adbs, bob_adbs),
                f"revolute-drive-dho/{label}",
            )

    def test_revolute_pd_drive_nonzero_target(self):
        """Non-zero target angle: verifies the ``drive_C = theta -
        target`` term goes through both paths identically. Bob starts
        at theta=0 and the drive pulls it toward target_angle; after
        enough time the two worlds should agree along the full
        trajectory, not just at steady state.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 4.0e-3
        I_eff = inertia_cm + mass * lever * lever
        k = 400.0
        c = 2.0 * math.sqrt(k * I_eff)  # critically damped

        world_ref, bob_ref = _build_revolute_drive_reference(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            k=k,
            c=c,
            target_angle=0.3,
            omega0=0.0,
            gravity=_GRAVITY_OFF,
        )
        world_adbs, bob_adbs = _build_revolute_drive_adbs(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            k=k,
            c=c,
            target_angle=0.3,
            omega0=0.0,
            gravity=_GRAVITY_OFF,
        )
        # Sample at the midpoint of the approach where velocity is
        # non-trivial (not just steady-state; steady-state agreement
        # is a weaker claim).
        run_settle_loop(world_ref, frames=int(0.3 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.3 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "revolute-drive-nonzero-target",
        )

    # --- Revolute PD velocity drive ----------------------------------------

    def test_revolute_pd_velocity_drive(self):
        """PD velocity drive (``stiffness_drive=0, damping_drive=c>0``)
        must match the standalone AngularMotor's PD-damping path
        (``stiffness=0, damping=c, target_angle`` unused) when driven
        to the same velocity target.

        Both sides collapse to the same scalar iterate
        ``lam = -(1/(I_inv + 1/(c*dt))) * (jv - target_velocity)``;
        the cross-check guards the sign of ``target_velocity`` in the
        bias fold and the sign of the body-side impulse application
        in ADBS vs. the standalone constraint.

        Gravity loads the bob so the servo must actually work against
        an external torque (a free-spin servo would be trivially
        satisfied by any velocity-tracking implementation).
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 0.5
        # Solid sphere radius 0.1 m: inertia (2/5)*m*r^2 = 4e-3.
        inertia_cm = 4.0e-3
        # Damping gain high enough that the servo catches v_target in
        # ~2 substeps (time constant tau = I_eff / c ~ 1 ms << dt), so
        # the bob follows the prescribed velocity except where gravity
        # saturates against ``max_force_drive``.
        c = 50.0  # N*m*s/rad
        target_velocity = -2.0  # rad/s
        # Generous cap so the servo is not saturated -- we want to
        # verify the PD math, not the clamp logic (that is covered by
        # the saturated position-drive cross-check).
        tau = 100.0

        # Reference: double-ball + standalone PD-damping-only motor.
        b_ref = WorldBuilder()
        pivot_ref = b_ref.world_body
        bob_ref = _add_bob(
            b_ref,
            position=(lever, 0.0, 0.0),
            mass=mass,
            inertia_cm=inertia_cm,
            affected_by_gravity=True,
        )
        b_ref.add_double_ball_socket(
            body1=pivot_ref,
            body2=bob_ref,
            anchor1=(0.0, 0.0, -0.25),
            anchor2=(0.0, 0.0, +0.25),
            hertz=0.0,
        )
        # Standalone motor in PD mode: setting ``damping > 0`` selects
        # the PD path; ``stiffness = 0`` zeroes the spring term so
        # ``target_angle`` is irrelevant and the iterate reduces to the
        # same pure-damping form the ADBS VELOCITY mode uses.
        b_ref.add_angular_motor(
            body1=pivot_ref,
            body2=bob_ref,
            axis=(0.0, 0.0, 1.0),
            target_velocity=target_velocity,
            max_force=tau,
            stiffness=0.0,
            damping=c,
        )
        world_ref = _finalize(b_ref, device, gravity=(0.0, -9.81, 0.0))

        # ADBS: revolute + PD velocity drive with the same damping.
        b_adbs = WorldBuilder()
        pivot_adbs = b_adbs.world_body
        bob_adbs = _add_bob(
            b_adbs,
            position=(lever, 0.0, 0.0),
            mass=mass,
            inertia_cm=inertia_cm,
            affected_by_gravity=True,
        )
        b_adbs.add_joint(
            body1=pivot_adbs,
            body2=bob_adbs,
            anchor1=(0.0, 0.0, -0.25),
            anchor2=(0.0, 0.0, +0.25),
            mode=JointMode.REVOLUTE,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target_velocity,
            max_force_drive=tau,
            stiffness_drive=0.0,
            damping_drive=c,
            hertz=0.0,
        )
        world_adbs = _finalize(b_adbs, device, gravity=(0.0, -9.81, 0.0))

        run_settle_loop(world_ref, frames=int(0.5 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.5 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "revolute-pd-velocity-drive",
        )

    # --- Prismatic PD velocity drive --------------------------------------

    def test_prismatic_pd_velocity_drive(self):
        """Linear analogue of :meth:`test_revolute_pd_velocity_drive`:
        PD damping on the slide axis loaded by gravity.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        inertia_cm = 4.0e-3
        c = 50.0  # N*s/m; tau = m/c = 20 ms
        target_velocity = -1.0
        F = 100.0  # N; not saturated

        b_ref = WorldBuilder()
        pivot_ref = b_ref.world_body
        bob_ref = _add_bob(
            b_ref,
            position=(0.0, 0.0, 0.0),
            mass=mass,
            inertia_cm=inertia_cm,
            affected_by_gravity=True,
        )
        b_ref.add_double_ball_socket_prismatic(
            body1=pivot_ref,
            body2=bob_ref,
            anchor1=(0.0, 0.0, 0.0),
            anchor2=(1.0, 0.0, 0.0),
            hertz=0.0,
        )
        b_ref.add_linear_motor(
            body1=pivot_ref,
            body2=bob_ref,
            axis=(1.0, 0.0, 0.0),
            anchor1=(0.0, 0.0, 0.0),
            anchor2=(0.0, 0.0, 0.0),
            target_velocity=target_velocity,
            max_force=F,
            stiffness=0.0,
            damping=c,
        )
        world_ref = _finalize(b_ref, device, gravity=(9.81, 0.0, 0.0))

        b_adbs = WorldBuilder()
        pivot_adbs = b_adbs.world_body
        bob_adbs = _add_bob(
            b_adbs,
            position=(0.0, 0.0, 0.0),
            mass=mass,
            inertia_cm=inertia_cm,
            affected_by_gravity=True,
        )
        b_adbs.add_joint(
            body1=pivot_adbs,
            body2=bob_adbs,
            anchor1=(0.0, 0.0, 0.0),
            anchor2=(1.0, 0.0, 0.0),
            mode=JointMode.PRISMATIC,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target_velocity,
            max_force_drive=F,
            stiffness_drive=0.0,
            damping_drive=c,
            hertz=0.0,
        )
        world_adbs = _finalize(b_adbs, device, gravity=(9.81, 0.0, 0.0))

        run_settle_loop(world_ref, frames=int(0.5 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.5 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "prismatic-pd-velocity-drive",
        )

    # --- Revolute limit ----------------------------------------------------

    def test_revolute_pd_limit_hooke_equilibrium(self):
        """Gravity loads a horizontal pendulum into a one-sided upper
        limit. Both worlds must settle to the same deflection angle
        (``k * (theta - max) = m g L cos(theta)``).
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 0.5
        inertia_cm = 4.0e-3
        I_eff = inertia_cm + mass * lever * lever
        k_lim = 200.0
        c_lim = 2.0 * math.sqrt(k_lim * I_eff)  # critical
        # One-sided: huge negative lower bound so only the upper
        # bound engages when gravity rotates the bob about +Z into
        # positive theta.
        min_angle = -100.0
        max_angle = 0.01  # rad; gravity will press us past this

        gravity = (0.0, -9.81, 0.0)
        world_ref, bob_ref = _build_revolute_limit_reference(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_angle=min_angle,
            max_angle=max_angle,
            omega0=0.0,
            gravity=gravity,
        )
        world_adbs, bob_adbs = _build_revolute_limit_adbs(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_angle=min_angle,
            max_angle=max_angle,
            omega0=0.0,
            gravity=gravity,
        )
        run_settle_loop(world_ref, frames=int(2.0 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(2.0 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "revolute-limit-hooke",
        )

    def test_revolute_pd_limit_inactive_side_does_not_leak(self):
        """Kick the bob into the *open* side of a one-sided limit
        (``max_angle = +huge``, ``min_angle = -small``). The limit
        should be silent and the bob should coast. Any discrepancy
        between the two worlds means one of them is leaking impulse on
        the inactive side.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 4.0e-3
        k_lim = 200.0
        c_lim = 10.0
        # Kick in the +theta direction, so we move *away* from the
        # lower bound. Upper bound at +huge.
        world_ref, bob_ref = _build_revolute_limit_reference(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_angle=-0.01,
            max_angle=100.0,
            omega0=1.0,
            gravity=_GRAVITY_OFF,
        )
        world_adbs, bob_adbs = _build_revolute_limit_adbs(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_angle=-0.01,
            max_angle=100.0,
            omega0=1.0,
            gravity=_GRAVITY_OFF,
        )
        # 0.25 s of free coast: at omega=1 rad/s we rotate 0.25 rad,
        # which is well away from the lower bound.
        run_settle_loop(world_ref, frames=int(0.25 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.25 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "revolute-limit-open-side",
        )

    # --- Prismatic drive ---------------------------------------------------

    def test_prismatic_pd_drive_free_oscillation(self):
        """Linear analogue of test_revolute_pd_drive_free_oscillation.
        Horizontal slider + PD spring-damper + initial velocity kick.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        inertia_cm = 4.0e-3  # Harmless: no rotation on a prismatic.
        k = 400.0
        omega0 = math.sqrt(k / mass)
        zeta = 0.1
        c = 2.0 * zeta * omega0 * mass
        kick = 0.5  # m/s

        world_ref, bob_ref = _build_prismatic_drive_reference(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k=k,
            c=c,
            target_position=0.0,
            v0=kick,
            gravity=_GRAVITY_OFF,
        )
        world_adbs, bob_adbs = _build_prismatic_drive_adbs(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k=k,
            c=c,
            target_position=0.0,
            v0=kick,
            gravity=_GRAVITY_OFF,
        )
        run_settle_loop(world_ref, frames=int(0.75 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.75 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "prismatic-drive-dho",
        )

    def test_prismatic_pd_drive_nonzero_target(self):
        """Target offset: drive pulls the bob from x=0 to x=target."""
        device = wp.get_preferred_device()
        mass = 1.0
        inertia_cm = 4.0e-3
        k = 400.0
        c = 2.0 * math.sqrt(k * mass)  # critical

        world_ref, bob_ref = _build_prismatic_drive_reference(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k=k,
            c=c,
            target_position=0.1,
            v0=0.0,
            gravity=_GRAVITY_OFF,
        )
        world_adbs, bob_adbs = _build_prismatic_drive_adbs(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k=k,
            c=c,
            target_position=0.1,
            v0=0.0,
            gravity=_GRAVITY_OFF,
        )
        run_settle_loop(world_ref, frames=int(0.3 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.3 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "prismatic-drive-nonzero-target",
        )

    # --- Prismatic limit ---------------------------------------------------

    def test_prismatic_pd_limit_hooke_equilibrium(self):
        """Vertical slider: gravity loads a PD upper limit. Both
        worlds must settle to the same deflection
        (``k * (x - max) = m g``).
        """
        device = wp.get_preferred_device()
        mass = 1.0
        inertia_cm = 4.0e-3
        k_lim = 200.0
        c_lim = 2.0 * math.sqrt(k_lim * mass)  # critical
        # Axis is +X (from _build_prismatic_limit_*); tilt gravity
        # along +X so the slide is monotonically driven into the
        # upper bound.
        gravity = (9.81, 0.0, 0.0)

        world_ref, bob_ref = _build_prismatic_limit_reference(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_pos=-100.0,
            max_pos=0.01,
            v0=0.0,
            gravity=gravity,
        )
        world_adbs, bob_adbs = _build_prismatic_limit_adbs(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_pos=-100.0,
            max_pos=0.01,
            v0=0.0,
            gravity=gravity,
        )
        run_settle_loop(world_ref, frames=int(2.0 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(2.0 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "prismatic-limit-hooke",
        )

    def test_prismatic_pd_limit_inactive_side_does_not_leak(self):
        """Kick the bob away from the active bound; both worlds
        should coast identically."""
        device = wp.get_preferred_device()
        mass = 1.0
        inertia_cm = 4.0e-3
        k_lim = 200.0
        c_lim = 10.0

        world_ref, bob_ref = _build_prismatic_limit_reference(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_pos=-0.01,
            max_pos=100.0,
            v0=1.0,  # +x, away from lower bound
            gravity=_GRAVITY_OFF,
        )
        world_adbs, bob_adbs = _build_prismatic_limit_adbs(
            device,
            mass=mass,
            inertia_cm=inertia_cm,
            k_lim=k_lim,
            c_lim=c_lim,
            min_pos=-0.01,
            max_pos=100.0,
            v0=1.0,
            gravity=_GRAVITY_OFF,
        )
        run_settle_loop(world_ref, frames=int(0.25 * FPS), dt=DT)
        run_settle_loop(world_adbs, frames=int(0.25 * FPS), dt=DT)
        _assert_states_agree(
            self,
            _snapshot(world_ref, bob_ref),
            _snapshot(world_adbs, bob_adbs),
            "prismatic-limit-open-side",
        )


if __name__ == "__main__":
    unittest.main()
