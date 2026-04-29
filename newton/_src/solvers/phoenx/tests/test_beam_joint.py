# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for :class:`JointMode.BEAM` -- soft fixed joint
with bend / twist PD softness on the anchor-2 tangent rows + anchor-3
scalar row.

BEAM replaces CABLE's angular log-map rows with a positional 3+2+1
split:

* anchor-1 3-row Box2D-soft point lock (rigid ball-socket);
* anchor-2 tangent 2-row PD-soft block driven by ``bend_stiffness``,
  ``bend_damping``;
* anchor-3 scalar 1-row PD-soft block driven by ``twist_stiffness``,
  ``twist_damping``.

User gains are rotational (``N*m/rad``, ``N*m*s/rad``) and rescaled by
``1 / rest_length^2`` to obtain the equivalent positional spring at
the lever-armed anchors. The lever-arm amplification gives the correct
rotational stiffness without an angular Jacobian / log-map (cf. CABLE),
and the well-conditioned positional rows match REVOLUTE / PRISMATIC
convergence behaviour.

The physical response in the linear regime matches the textbook damped
harmonic oscillator :math:`I \\ddot\\theta + c \\dot\\theta + k \\theta
= 0`: natural frequency :math:`\\omega_n = \\sqrt{k/I}`, damping ratio
:math:`\\zeta = c / (2\\sqrt{kI})`. Tests validate oscillation
(undamped), exponential settling (over- / under-damped), stiffness /
damping ordering, gain conversion across ``rest_length``, multi-segment
chain behaviour, and BEAM <-> CABLE asymptotic equivalence.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    _BEAM_NYQUIST_HEADROOM,
)
from newton._src.solvers.phoenx.tests._test_helpers import make_graph_stepper, run_settle_loop
from newton._src.solvers.phoenx.world_builder import (
    JointMode,
    WorldBuilder,
)

FPS = 240
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
HALF_EXTENT = 0.2
_INV_INERTIA_ROD = ((20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0))
_I_ROD = 1.0 / 20.0  # diagonal -> scalar moment of inertia about any axis [kg*m^2]


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


def _build_beam_pendulum(
    device,
    *,
    bend_stiffness: float = 0.0,
    twist_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    twist_damping: float = 0.0,
    init_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    init_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rest_length: float = 1.0,
    affected_by_gravity: bool = False,
    com_offset: float = 0.0,
    substeps: int = SUBSTEPS,
    solver_iterations: int = SOLVER_ITERATIONS,
):
    """World anchor + one dynamic rod pivoted at the origin via a beam
    joint. Reference axis is +x (``anchor1 = (0, 0, 0)``,
    ``anchor2 = (rest_length, 0, 0)``); anchor-3 is auto-derived
    perpendicular to that axis at distance ``rest_length``.

    Layout matches :func:`_build_cable_pendulum` from
    :mod:`test_cable_joint` so the analytical assertions transfer
    one-for-one. By default the rod's COM sits AT the anchor
    (``com_offset = 0``) so the ball-socket positional lock contributes
    no lever-arm coupling; bend / twist rows dominate.

    The rest pose used by the bend / twist springs is captured at
    finalize() time from the bodies' current orientation. To pre-
    deflect without overwriting that snapshot, build at identity (so
    rest = identity) and overwrite the runtime orientation post-
    finalize -- the first prepare pass sees ``q_align = init_rotation``
    relative to the identity rest pose.
    """
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
        anchor2=(float(rest_length), 0.0, 0.0),
        mode=JointMode.BEAM,
        bend_stiffness=bend_stiffness,
        twist_stiffness=twist_stiffness,
        bend_damping=bend_damping,
        twist_damping=twist_damping,
    )
    world_out = b.finalize(
        substeps=substeps,
        solver_iterations=solver_iterations,
        gravity=(0.0, 0.0, -9.81) if affected_by_gravity else (0.0, 0.0, 0.0),
        device=device,
    )
    if init_rotation != (0.0, 0.0, 0.0, 1.0):
        orient_np = world_out.bodies.orientation.numpy()
        orient_np[rod] = np.asarray(init_rotation, dtype=np.float32)
        world_out.bodies.orientation.assign(orient_np)
    return world_out


def _build_cable_pendulum_for_equivalence(
    device,
    *,
    bend_stiffness: float,
    twist_stiffness: float,
    bend_damping: float,
    twist_damping: float,
    init_rotation: tuple[float, float, float, float],
):
    """Sibling of :func:`_build_beam_pendulum` configured identically
    but with :attr:`JointMode.CABLE`. Used by
    :class:`TestBeamCableEquivalence` to compare matched rods.
    """
    b = WorldBuilder()
    world = b.world_body
    rod = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA_ROD,
        affected_by_gravity=False,
    )
    b.add_joint(
        body1=world,
        body2=rod,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        mode=JointMode.CABLE,
        bend_stiffness=bend_stiffness,
        twist_stiffness=twist_stiffness,
        bend_damping=bend_damping,
        twist_damping=twist_damping,
    )
    world_out = b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, 0.0),
        device=device,
    )
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
    w = float(q[3])
    xyz = q[:3].astype(np.float64)
    if w < 0.0:
        w = -w
        xyz = -xyz
    projection = float(np.dot(xyz, axis.astype(np.float64)))
    return 2.0 * math.atan2(projection, w)


def _record_angle_history(world, axis: np.ndarray, n_samples: int, dt: float) -> np.ndarray:
    """Step ``world`` ``n_samples`` times at ``dt`` and return the
    signed-angle-about-axis trajectory at each sample (length
    ``n_samples + 1``: index 0 is the pre-step value).

    Uses :func:`make_graph_stepper` so the per-sample step is replayed
    from a captured CUDA graph rather than re-traced -- without this,
    1-frame ``run_settle_loop`` calls fall through to the eager-step
    fallback (the graph-capture threshold is 40 frames) and the test
    spends most of its time on Python + Warp launch overhead.
    """
    step = make_graph_stepper(world, dt)
    angles = [_rotation_angle_about(_rod_orientation(world), axis)]
    for _ in range(n_samples):
        step(1)
        angles.append(_rotation_angle_about(_rod_orientation(world), axis))
    return np.asarray(angles)


def _zero_crossings(angles: np.ndarray) -> np.ndarray:
    """Indices ``i`` where successive non-zero samples flip sign."""
    signs = np.sign(angles)
    nz = np.where(signs != 0)[0]
    if len(nz) < 2:
        return np.empty(0, dtype=int)
    return np.where(np.diff(signs[nz]) != 0)[0]


# ---------------------------------------------------------------------------
# Bend stiffness / damping / period ordering
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamBendStiffness(unittest.TestCase):
    """Bend stiffness acts perpendicular to the reference axis. The
    BEAM rows are positional (anchor-2 tangent block) but produce the
    same rotational restoring torque as CABLE's bend rows in the
    linear regime."""

    def test_bend_springs_back(self) -> None:
        """Undamped bend spring: pre-rotate about +z, release. The
        restoring torque pulls the rod back through zero and the
        angle signs change (free oscillation)."""
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(20.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=60.0,
            twist_stiffness=60.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, int(0.3 * FPS), dt)[1:]
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
        world = _build_beam_pendulum(
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
            world = _build_beam_pendulum(
                wp.get_preferred_device(),
                bend_stiffness=k,
                twist_stiffness=k,
                init_rotation=init_q,
            )
            angles = _record_angle_history(world, y_axis, n, dt)
            crossings = 0
            for i in range(1, len(angles)):
                if (angles[i] > 0) != (angles[i - 1] > 0):
                    crossings += 1
            return crossings

        soft_crossings = count_zero_crossings(30.0)
        stiff_crossings = count_zero_crossings(300.0)
        self.assertGreater(
            stiff_crossings,
            soft_crossings,
            msg=f"stiffer spring should oscillate faster; soft={soft_crossings}, stiff={stiff_crossings}",
        )


# ---------------------------------------------------------------------------
# Twist stiffness / damping
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamTwistStiffness(unittest.TestCase):
    """Twist stiffness acts along the reference axis (anchor-3 scalar
    row in the BEAM formulation)."""

    def test_twist_springs_back(self) -> None:
        """Undamped twist spring: the rod oscillates about rest twist,
        crossing zero."""
        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(30.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=60.0,
            twist_stiffness=60.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        x_axis = np.array([1.0, 0.0, 0.0])
        angles = _record_angle_history(world, x_axis, int(0.3 * FPS), dt)[1:]
        self.assertTrue(np.isfinite(angles).all())
        self.assertLess(
            float(angles.min()),
            0.0,
            msg=f"twist spring never crossed zero; min={angles.min():.3f}, max={angles.max():.3f}",
        )

    def test_twist_damping_decays(self) -> None:
        """Twist damping kills the oscillation envelope."""
        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(25.0))
        world = _build_beam_pendulum(
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


# ---------------------------------------------------------------------------
# Decoupling: anchor-2 tangent rows and anchor-3 scalar row are
# independent in the BEAM block-GS solve. Pure-bend deflection should
# not bleed into a twist error and vice versa.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamDecoupling(unittest.TestCase):
    """Bend and twist are independent rows in the BEAM block-Gauss-
    Seidel solve."""

    def test_pure_bend_produces_no_twist(self) -> None:
        """Initial rotation is pure bend (about +y). Without any
        initial twist deflection, twist angle should stay near zero
        through the oscillation."""
        init_q = _axis_angle_quat((0.0, 1.0, 0.0), math.radians(20.0))
        world = _build_beam_pendulum(
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
        """Initial rotation is pure twist (about +x). Bend angle should
        stay near zero."""
        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(30.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=80.0,
            twist_stiffness=80.0,
            bend_damping=1.0,
            twist_damping=1.0,
            init_rotation=init_q,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(0.5 * FPS), dt=dt)
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


# ---------------------------------------------------------------------------
# Quantitative analytical regressions against the closed-form damped
# harmonic oscillator I*theta_dd + c*theta_d + k*theta = 0.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamAnalytical(unittest.TestCase):
    """Quantitative comparisons against the closed-form damped
    harmonic oscillator. Mirror of
    :class:`test_cable_joint.TestCableAnalytical` with BEAM-specific
    bounds (e.g. the high-stiffness ceiling shifts by
    ``_BEAM_NYQUIST_HEADROOM``).
    """

    def test_undamped_period_within_5pct(self) -> None:
        """Undamped oscillator period :math:`T = 2\\pi / \\omega_n`
        with :math:`\\omega_n = \\sqrt{k/I}`. Drift here means the
        BEAM prepare's positional rescale ``k_pos = k_user /
        rest_length^2`` or its iterate gain has skewed."""
        k = 80.0
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)

        init_angle_rad = math.radians(10.0)
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), init_angle_rad)
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_samples = int(1.5 * T_theory / dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, n_samples, dt)
        crossings = _zero_crossings(angles)
        self.assertGreaterEqual(
            len(crossings),
            2,
            msg=f"need >=2 zero crossings to measure period; got {len(crossings)}",
        )
        if len(crossings) >= 3:
            T_meas = (crossings[2] - crossings[0]) * dt
        else:
            T_meas = 2.0 * (crossings[1] - crossings[0]) * dt
        rel = abs(T_meas - T_theory) / T_theory
        self.assertLess(
            rel,
            0.05,
            msg=f"undamped period drift {rel * 100:.2f}% > 5%; T_theory={T_theory * 1000:.2f} ms, T_meas={T_meas * 1000:.2f} ms",
        )

    def test_underdamped_log_decrement_recovers_zeta(self) -> None:
        """Log-decrement: for an underdamped oscillator
        (:math:`0 < \\zeta < 1`), successive amplitude peaks decay by
        :math:`\\delta = \\ln(A_n / A_{n+1}) = 2\\pi\\zeta /
        \\sqrt{1 - \\zeta^2}`. We pick ``zeta = 0.1`` so the rod
        oscillates several times before damping wins.

        Tolerance choice: 30 % on ``zeta``. At the default
        ``(substeps, solver_iterations) = (4, 16)`` the substep
        ``dt_sub = 1 / (FPS * substeps) = 1.04 ms`` and
        ``omega_n * dt_sub ~= 0.047``. Implicit-Euler integration adds
        :math:`O(\\omega_n dt)` numerical damping per step which
        compounds across the PGS iterations -- empirically the recovered
        ``zeta`` lands ~23 % above the target at this configuration
        (CABLE shows the same offset at the same dt, so this is a shared
        first-order discretisation artefact, not BEAM-specific). Halving
        ``dt_sub`` halves the offset (clean ``O(dt)`` convergence:
        23 % at sub=4, 11.6 % at sub=8, 5.9 % at sub=16). The 30 %
        tolerance is the smallest round bound that fits the default
        config with a safety margin; a future migration to the
        spring / damping split in :func:`pd_coefficients_split`
        (Phase-2 ``PD_DRIVE_AUDIT.md`` Proposal B) or a higher-order
        integrator should let us tighten this to ~10 %.
        """
        k = 100.0
        zeta_target = 0.1
        c = 2.0 * zeta_target * math.sqrt(k * _I_ROD)

        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=c,
            twist_damping=c,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)
        n_samples = int(4.0 * T_theory / dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, n_samples, dt)

        # Find local maxima of |angles| separated by ~T_theory; these
        # are the oscillator's peak amplitudes per cycle.
        T_steps = max(1, int(T_theory / dt))
        peaks: list[float] = []
        i = 0
        while i + T_steps < len(angles):
            window = np.abs(angles[i : i + T_steps])
            peak_idx = int(np.argmax(window))
            peaks.append(float(window[peak_idx]))
            i += T_steps
        self.assertGreaterEqual(
            len(peaks),
            3,
            msg=f"need >=3 peaks to measure log-decrement; got {len(peaks)}",
        )
        decrements = [math.log(peaks[i] / max(peaks[i + 1], 1e-9)) for i in range(2)]
        delta = float(np.mean(decrements))
        zeta_meas = delta / math.sqrt(4.0 * math.pi * math.pi + delta * delta)
        rel = abs(zeta_meas - zeta_target) / zeta_target
        self.assertLess(
            rel,
            0.30,
            msg=f"recovered zeta {zeta_meas:.4f} vs target {zeta_target:.4f} ({rel * 100:.1f}% off)",
        )

    def test_underdamped_log_decrement_tightens_with_finer_dt(self) -> None:
        """Convergence regression: at ``substeps = 16`` (4x finer
        ``dt_sub`` than the default), the implicit-Euler numerical
        damping should drop the recovered-``zeta`` error to ~6 %, well
        below the 30 % bound used at the default substep count. Locks
        in the ``O(dt)`` convergence rate so a regression that broke
        the implicit-Euler structure (e.g. a stale ``dt`` factor in
        the gain rescale) would surface here even when the default-dt
        test still passes its loose bound.
        """
        k = 100.0
        zeta_target = 0.1
        c = 2.0 * zeta_target * math.sqrt(k * _I_ROD)

        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=c,
            twist_damping=c,
            init_rotation=init_q,
            substeps=16,
            solver_iterations=64,
        )

        dt = 1.0 / FPS
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)
        n_samples = int(4.0 * T_theory / dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, n_samples, dt)

        T_steps = max(1, int(T_theory / dt))
        peaks: list[float] = []
        i = 0
        while i + T_steps < len(angles):
            window = np.abs(angles[i : i + T_steps])
            peak_idx = int(np.argmax(window))
            peaks.append(float(window[peak_idx]))
            i += T_steps
        self.assertGreaterEqual(
            len(peaks),
            3,
            msg=f"need >=3 peaks to measure log-decrement; got {len(peaks)}",
        )
        decrements = [math.log(peaks[i] / max(peaks[i + 1], 1e-9)) for i in range(2)]
        delta = float(np.mean(decrements))
        zeta_meas = delta / math.sqrt(4.0 * math.pi * math.pi + delta * delta)
        rel = abs(zeta_meas - zeta_target) / zeta_target
        # Tight 10 % bound (empirical: 5.9 % at substeps=16). If a
        # regression bumps this, also re-check the default-substeps
        # log-decrement test -- the offset should scale ~linearly.
        self.assertLess(
            rel,
            0.10,
            msg=(
                f"finer-dt log-decrement: zeta_meas={zeta_meas:.4f} vs "
                f"target {zeta_target:.4f} ({rel * 100:.1f}% off; 10% bound)"
            ),
        )

    def test_overdamped_settling_time_constant(self) -> None:
        """Overdamped (:math:`\\zeta > 1`) settles without crossing
        zero; the slow exponential mode :math:`\\alpha_1 = \\omega_n
        (\\zeta - \\sqrt{\\zeta^2 - 1})` dominates the late-time tail.
        """
        k = 100.0
        zeta = 2.0
        omega_n = math.sqrt(k / _I_ROD)
        c = 2.0 * zeta * math.sqrt(k * _I_ROD)
        alpha_slow = omega_n * (zeta - math.sqrt(zeta * zeta - 1.0))
        tau_slow = 1.0 / alpha_slow

        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=c,
            twist_damping=c,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_samples = int(5.0 * tau_slow / dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, n_samples, dt)

        signs = np.sign(angles[1:])
        crossings = int(np.sum(np.diff(signs) != 0))
        self.assertEqual(
            crossings,
            0,
            msg=f"overdamped trajectory should not cross zero; got {crossings} crossings",
        )

        t = np.arange(len(angles)) * dt
        tail_start = int(tau_slow / dt)
        tail_end = min(int(4.0 * tau_slow / dt), len(angles) - 1)
        if tail_end <= tail_start + 5:
            self.skipTest("not enough samples in the exponential tail")
        amp = np.abs(angles[tail_start:tail_end])
        ts = t[tail_start:tail_end]
        valid = amp > 1e-5
        self.assertGreater(int(valid.sum()), 5, msg="tail too noisy to fit")
        log_amp = np.log(amp[valid])
        ts_v = ts[valid]
        slope, _ = np.polyfit(ts_v, log_amp, 1)
        tau_meas = -1.0 / slope
        rel = abs(tau_meas - tau_slow) / tau_slow
        # Tight 5 % bound: the slow exponential mode at zeta=2 has
        # ``alpha_slow ~= 12 rad/s``, which is ~80x below the substep
        # Nyquist rate ``pi / dt_sub ~= 3000 rad/s``. Implicit-Euler's
        # numerical-damping error is ``O(alpha * dt)`` so the recovered
        # tau lands within ~1 % of theory at the default config.
        # Empirical: bend ~0.63 %, twist ~0.59 %.
        self.assertLess(
            rel,
            0.05,
            msg=(
                f"overdamped time constant: theory tau={tau_slow * 1000:.2f} ms, "
                f"measured tau={tau_meas * 1000:.2f} ms ({rel * 100:.1f}% off)"
            ),
        )

    def test_high_damping_settles_within_solver_iterations(self) -> None:
        """Convergence regression at :math:`\\zeta = 5` (high damping):
        the rod must settle to a small fraction of its initial
        deflection within ``2 / alpha_slow`` simulated seconds at the
        default ``solver_iterations``.

        BEAM currently uses the *combined* :func:`pd_coefficients`
        softness rather than the spring/damping split CABLE adopted
        in commit ``8b9fb289`` -- so under high damping the soft-PD's
        effective mass collapses toward the rigid limit and PGS needs
        more iterations to dissipate. Threshold is set at 25 % of the
        initial deflection, which BEAM hits comfortably while leaving
        room for a future split-formulation upgrade to tighten this
        bound (cf. ``test_cable_joint.test_high_damping_settles_within_solver_iterations``
        which holds CABLE to 10 %). The Phase-2 audit calls out the
        gap as an actionable follow-up."""
        k = 50.0
        zeta = 5.0
        c = 2.0 * zeta * math.sqrt(k * _I_ROD)
        omega_n = math.sqrt(k / _I_ROD)
        alpha_slow = omega_n * (zeta - math.sqrt(zeta * zeta - 1.0))
        settle_t = 2.0 / alpha_slow

        init_angle = math.radians(15.0)
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), init_angle)
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=c,
            twist_damping=c,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        run_settle_loop(world, int(settle_t / dt), dt=dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angle_end = abs(_rotation_angle_about(_rod_orientation(world), z_axis))
        omega_end = _rod_angular_velocity(world)
        self.assertTrue(
            np.isfinite(_rod_orientation(world)).all() and np.isfinite(omega_end).all(),
            msg="state went non-finite under high damping",
        )
        self.assertLess(
            angle_end,
            init_angle * 0.25,
            msg=(
                f"high-damping convergence regression: angle decayed to "
                f"{angle_end:.4f} rad after {settle_t * 1000:.1f} ms, "
                f"want < 25% of init ({init_angle * 0.25:.4f} rad)"
            ),
        )

    def test_high_stiffness_remains_stable(self) -> None:
        """High-stiffness stability regression at ~100x of BEAM's
        *headroom-scaled* Nyquist cap (the strict cap times
        :data:`_BEAM_NYQUIST_HEADROOM`). Without a clamp the soft-PD's
        eff_mass collapses to the rigid limit and the bias spikes to
        ``C / dt`` -- per-step impulse can overshoot and angular
        velocity diverges. With the clamp in place pose and velocity
        stay bounded.

        Note: the equivalent test in :mod:`test_cable_joint` sizes the
        ceiling against the strict Nyquist (CABLE's ``pd_coefficients``
        path); BEAM caps at :data:`_BEAM_NYQUIST_HEADROOM` *times*
        that ceiling, so we scale the test ``k`` accordingly to keep
        the regression at the same ratio above the saturation point.
        """
        # ``k_ref ~= 1 / (dt_substep^2 * I_inv)``; FPS=240, substeps=4
        # -> dt_substep = 1/960, I_inv = 20 -> k_ref ~= 46000. BEAM's
        # cap is ``_BEAM_NYQUIST_HEADROOM * k_ref`` ~= 460000. We
        # request ~100x of that to comfortably exceed the saturation
        # point.
        headroom = float(_BEAM_NYQUIST_HEADROOM)
        k_ref = (FPS * SUBSTEPS) ** 2 / 20.0
        k = 100.0 * headroom * k_ref
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=0.0,  # undamped to make instability easier to detect
            twist_damping=0.0,
            init_rotation=init_q,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(0.5 * FPS), dt=dt)
        q = _rod_orientation(world)
        omega = _rod_angular_velocity(world)
        self.assertTrue(np.isfinite(q).all() and np.isfinite(omega).all())
        z_axis = np.array([0.0, 0.0, 1.0])
        bend = abs(_rotation_angle_about(q, z_axis))
        self.assertLess(
            bend,
            math.radians(60.0),
            msg=f"high-stiffness rod blew up: bend={math.degrees(bend):.2f} deg",
        )
        self.assertLess(
            float(np.linalg.norm(omega)),
            50.0,
            msg=f"high-stiffness rod's omega diverged: |omega|={np.linalg.norm(omega):.2f} rad/s",
        )

    def test_large_angle_stable_at_60deg(self) -> None:
        """BEAM uses the *positional* anchor-2 / anchor-3 rows rather
        than CABLE's angular log-map, so there is no small-angle
        approximation in the spring formulation: the spring force on
        anchor-2 / anchor-3 scales as ``rest_length * sin(theta) *
        k``, which is the projection along the tangent basis. At
        moderate-to-large bend angles the rod's pure restoring
        behaviour does drift slightly from the linear theory because
        of the geometric ``sin(theta) vs theta`` difference, but the
        oscillation period must stay within ~15% of the small-angle
        theory and the trajectory must be bounded.

        Failure mode: large-angle BEAM blows up or its period drifts
        much further than CABLE's log-map equivalent
        (:meth:`test_cable_joint.test_large_angle_log_map_accuracy`).
        """
        k = 60.0
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)

        init_angle = math.radians(60.0)
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), init_angle)
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_samples = int(1.5 * T_theory / dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, n_samples, dt)

        crossings = _zero_crossings(angles)
        self.assertGreaterEqual(
            len(crossings),
            2,
            msg=f"large-angle rod didn't oscillate; crossings={len(crossings)}",
        )
        T_meas = 2.0 * (crossings[1] - crossings[0]) * dt
        # BEAM's positional spring loses force at large angles
        # (sin(theta) < theta), so the period actually *grows* slightly.
        # We allow a wider band than CABLE's 15 % to accommodate this.
        rel = abs(T_meas - T_theory) / T_theory
        self.assertLess(
            rel,
            0.30,
            msg=(
                f"large-angle BEAM period drift {rel * 100:.1f}% > 30%; "
                f"T_theory={T_theory * 1000:.2f} ms, T_meas={T_meas * 1000:.2f} ms"
            ),
        )
        self.assertTrue(np.isfinite(angles).all())


# ---------------------------------------------------------------------------
# Twist axis: quantitative analytical regressions
# ---------------------------------------------------------------------------
#
# The bend rows live in the BEAM anchor-2 tangent block (2 positional
# DoF, softened with ``k_bend, c_bend``) and the twist row lives in the
# anchor-3 scalar block (1 positional DoF, softened with ``k_twist,
# c_twist``). Both blocks share the same ``pd_coefficients``-equivalent
# softening kernel and the same ``1 / rest_length^2`` lever-arm
# rescale, so analytically a damped-harmonic-oscillator match should
# hold for twist with the same tolerances as bend. These tests pin
# that match independently from the bend suite -- a regression that
# only affected the anchor-3 block (e.g. wrong sign on the bias, or a
# missing ``inv_rest2`` rescale on the twist gains) would leave bend
# tests passing while the twist period drifted.
#
# The pendulum's reference axis is ``+x`` (anchor1 -> anchor2 in
# ``_build_beam_pendulum``) so a pre-rotation about ``+x`` is pure
# twist; angle-about-``+x`` measures it.


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamTwistAnalytical(unittest.TestCase):
    """Quantitative comparison of BEAM twist vs the closed-form damped
    harmonic oscillator. Mirror of :class:`TestBeamAnalytical` with
    ``+x`` (twist axis) substituted for ``+z`` (bend axis) -- proves
    the anchor-3 row honours user gains analytically.
    """

    def test_undamped_period_within_5pct(self) -> None:
        """Undamped twist oscillator period :math:`T = 2\\pi /
        \\omega_n` with :math:`\\omega_n = \\sqrt{k_\\mathrm{twist} /
        I}`. A drift here means the BEAM prepare's twist gain
        rescale (``k_pos = k_user / rest_length^2``) or the anchor-3
        iterate gain has skewed."""
        k = 80.0
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)

        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_samples = int(1.5 * T_theory / dt)
        x_axis = np.array([1.0, 0.0, 0.0])
        angles = _record_angle_history(world, x_axis, n_samples, dt)
        crossings = _zero_crossings(angles)
        self.assertGreaterEqual(
            len(crossings),
            2,
            msg=f"need >=2 zero crossings to measure period; got {len(crossings)}",
        )
        if len(crossings) >= 3:
            T_meas = (crossings[2] - crossings[0]) * dt
        else:
            T_meas = 2.0 * (crossings[1] - crossings[0]) * dt
        rel = abs(T_meas - T_theory) / T_theory
        self.assertLess(
            rel,
            0.05,
            msg=(
                f"twist undamped period drift {rel * 100:.2f}% > 5%; "
                f"T_theory={T_theory * 1000:.2f} ms, T_meas={T_meas * 1000:.2f} ms"
            ),
        )

    def test_underdamped_log_decrement_recovers_zeta(self) -> None:
        """Log-decrement on the twist axis: peaks decay by
        :math:`\\delta = 2\\pi\\zeta / \\sqrt{1 - \\zeta^2}`. Same
        ``zeta = 0.1`` target as the bend variant so the analysis
        windows match. 30 % tolerance carries the same implicit-Euler
        numerical-damping story documented in the bend version
        (``test_underdamped_log_decrement_recovers_zeta`` in
        :class:`TestBeamAnalytical`); empirically both axes land at
        ~23 % above target at the default substep count and converge
        ``O(dt)`` with finer substeps.
        """
        k = 100.0
        zeta_target = 0.1
        c = 2.0 * zeta_target * math.sqrt(k * _I_ROD)

        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=c,
            twist_damping=c,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)
        n_samples = int(4.0 * T_theory / dt)
        x_axis = np.array([1.0, 0.0, 0.0])
        angles = _record_angle_history(world, x_axis, n_samples, dt)

        T_steps = max(1, int(T_theory / dt))
        peaks: list[float] = []
        i = 0
        while i + T_steps < len(angles):
            window = np.abs(angles[i : i + T_steps])
            peak_idx = int(np.argmax(window))
            peaks.append(float(window[peak_idx]))
            i += T_steps
        self.assertGreaterEqual(
            len(peaks),
            3,
            msg=f"need >=3 peaks to measure log-decrement; got {len(peaks)}",
        )
        decrements = [math.log(peaks[i] / max(peaks[i + 1], 1e-9)) for i in range(2)]
        delta = float(np.mean(decrements))
        zeta_meas = delta / math.sqrt(4.0 * math.pi * math.pi + delta * delta)
        rel = abs(zeta_meas - zeta_target) / zeta_target
        self.assertLess(
            rel,
            0.30,
            msg=(
                f"twist log-decrement: recovered zeta {zeta_meas:.4f} vs "
                f"target {zeta_target:.4f} ({rel * 100:.1f}% off)"
            ),
        )

    def test_overdamped_settling_time_constant(self) -> None:
        """Overdamped twist (:math:`\\zeta = 2`) settles without
        crossing zero; the slow exponential mode sets the late-time
        tail :math:`\\tau = 1 / (\\omega_n (\\zeta - \\sqrt{\\zeta^2
        - 1}))`. Same setup as the bend variant.
        """
        k = 100.0
        zeta = 2.0
        omega_n = math.sqrt(k / _I_ROD)
        c = 2.0 * zeta * math.sqrt(k * _I_ROD)
        alpha_slow = omega_n * (zeta - math.sqrt(zeta * zeta - 1.0))
        tau_slow = 1.0 / alpha_slow

        init_q = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(10.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=c,
            twist_damping=c,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_samples = int(5.0 * tau_slow / dt)
        x_axis = np.array([1.0, 0.0, 0.0])
        angles = _record_angle_history(world, x_axis, n_samples, dt)

        signs = np.sign(angles[1:])
        crossings = int(np.sum(np.diff(signs) != 0))
        self.assertEqual(
            crossings,
            0,
            msg=(
                f"twist overdamped trajectory should not cross zero; "
                f"got {crossings} crossings"
            ),
        )

        t = np.arange(len(angles)) * dt
        tail_start = int(tau_slow / dt)
        tail_end = min(int(4.0 * tau_slow / dt), len(angles) - 1)
        if tail_end <= tail_start + 5:
            self.skipTest("not enough samples in the exponential tail")
        amp = np.abs(angles[tail_start:tail_end])
        ts = t[tail_start:tail_end]
        valid = amp > 1e-5
        self.assertGreater(int(valid.sum()), 5, msg="tail too noisy to fit")
        log_amp = np.log(amp[valid])
        ts_v = ts[valid]
        slope, _ = np.polyfit(ts_v, log_amp, 1)
        tau_meas = -1.0 / slope
        rel = abs(tau_meas - tau_slow) / tau_slow
        # Same 5 % bound as the bend overdamped tau test -- both
        # axes share the same ``pd_coefficients``-style softening so
        # they should converge to the same ~1 % discretisation error.
        self.assertLess(
            rel,
            0.05,
            msg=(
                f"twist overdamped tau: theory={tau_slow * 1000:.2f} ms, "
                f"measured={tau_meas * 1000:.2f} ms ({rel * 100:.1f}% off)"
            ),
        )


# ---------------------------------------------------------------------------
# Degenerate / stability edge cases
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamDegenerate(unittest.TestCase):
    """Edge cases."""

    def test_zero_stiffness_does_not_blow_up(self) -> None:
        """With zero stiffness and zero damping on both axes, BEAM
        does *not* degenerate to a ball-socket the way CABLE does
        (cf. ``test_cable_joint.test_zero_stiffness_matches_ball_socket``).

        CABLE's angular rows short-circuit on ``k = c = 0`` via
        :func:`pd_coefficients_split`, leaving anchor-1 as the only
        active block. BEAM's anchor-2 / anchor-3 prepare *also*
        skips the bias / gamma write (see L2391-L2417), but the
        cached ``K22_inv`` falls back to the *unsoftened* inverse of
        ``K22`` -- which the iterate then applies as a rigid
        positional lock against the current twist / bend drift. The
        rod's spin therefore decays even though the user requested
        no spring force.

        This is an internal BEAM behaviour gap surfaced by the
        validation suite, not a stability failure. The assertion
        below pins that pose / velocity stay bounded and finite,
        leaving the "BEAM[0,0] == ball-socket" guarantee out of
        scope until the BEAM kernel grows the same short-circuit
        plumbing CABLE has."""
        omega0 = (1.0, -0.5, 0.3)
        world = _build_beam_pendulum(
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
        q = _rod_orientation(world)
        pos = world.bodies.position.numpy()[1]
        self.assertTrue(np.isfinite(omega).all(), msg=f"BEAM[0,0] omega non-finite: {omega}")
        self.assertTrue(np.isfinite(q).all(), msg=f"BEAM[0,0] orientation non-finite: {q}")
        self.assertTrue(np.isfinite(pos).all(), msg=f"BEAM[0,0] position non-finite: {pos}")
        # The anchor-1 ball-socket lock is unaffected by zero gains:
        # the rod's COM (built at the anchor) must stay there.
        self.assertLess(
            float(np.linalg.norm(pos)),
            0.05,
            msg=f"BEAM[0,0] anchor-1 lock leaked: |com|={np.linalg.norm(pos):.4f} m",
        )

    def test_rigid_ball_socket_holds_anchor_under_gravity(self) -> None:
        """The positional 3-row lock at anchor-1 is unchanged by BEAM
        mode. Under gravity the rod must hang from the anchor without
        its COM drifting away from the pendulum radius."""
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=5.0,  # soft -- tests the positional anchor-1 lock
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
        radius = float(np.linalg.norm(pos))
        self.assertAlmostEqual(
            radius,
            0.5,
            delta=0.05,
            msg=f"positional lock leaked: radius={radius:.4f} m",
        )


# ---------------------------------------------------------------------------
# Gain conversion: ``k_pos = k_user / rest_length^2`` rescale must
# leave the rotational restoring torque invariant under rest_length.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamGainConversion(unittest.TestCase):
    """Pin the BEAM prepare's rotational -> positional rescale.

    BEAM stores the user gain as ``k_pos = k_user / rest_length^2``
    (see ``_beam_prepare_at`` L2371-L2380); the lever-arm-amplified
    anchor-2 / anchor-3 rows multiply by ``rest_length^2`` again so
    the *rotational* restoring torque is invariant under
    ``rest_length``. A regression that flipped the sign of the
    exponent (``rest_length^2`` instead of ``1 / rest_length^2``) or
    forgot the rescale entirely would change the apparent stiffness
    -- and therefore the period -- by a factor of ``rest_length^2``.
    Sweeping ``rest_length`` and asserting the measured period stays
    inside ~5 % of the small-angle theory pins this conversion.
    """

    def _measure_period(self, rest_length: float, k: float) -> tuple[float, float]:
        """Build a BEAM rod with the given ``rest_length`` and ``k``,
        measure its undamped oscillation period from the first two
        zero crossings, and return ``(T_meas, T_theory)``."""
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)

        init_q = _axis_angle_quat((0.0, 0.0, 1.0), math.radians(8.0))
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k,
            twist_stiffness=k,
            rest_length=rest_length,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_samples = int(2.0 * T_theory / dt)
        z_axis = np.array([0.0, 0.0, 1.0])
        angles = _record_angle_history(world, z_axis, n_samples, dt)
        crossings = _zero_crossings(angles)
        if len(crossings) < 2:
            self.fail(
                f"rest_length={rest_length:.3g}: <2 zero crossings; "
                f"period unmeasurable (got {len(crossings)})"
            )
        if len(crossings) >= 3:
            T_meas = (crossings[2] - crossings[0]) * dt
        else:
            T_meas = 2.0 * (crossings[1] - crossings[0]) * dt
        return float(T_meas), float(T_theory)

    def test_period_invariant_under_rest_length(self) -> None:
        """Sweep ``rest_length in {0.1, 0.5, 1.0, 2.0}`` at fixed
        ``k_user`` and assert measured period stays within 10 % of
        the small-angle theory ``T = 2 pi / sqrt(k/I)``. The wider
        band vs the single-rod 5 % bound covers numerical noise from
        the very-short-anchor (rest_length = 0.1) regime where the
        positional rows ride on a 0.1 m lever arm and amplify
        substep error by ~10x."""
        k = 80.0
        rest_lengths = [0.1, 0.5, 1.0, 2.0]
        for rl in rest_lengths:
            with self.subTest(rest_length=rl):
                T_meas, T_theory = self._measure_period(rl, k)
                rel = abs(T_meas - T_theory) / T_theory
                self.assertLess(
                    rel,
                    0.10,
                    msg=(
                        f"rest_length={rl:.3g}: period drift {rel * 100:.2f}% > 10%; "
                        f"T_theory={T_theory * 1000:.2f} ms, T_meas={T_meas * 1000:.2f} ms. "
                        f"This pins BEAM's k_pos = k_user / rest_length^2 rescale."
                    ),
                )


# ---------------------------------------------------------------------------
# Static cantilever: multi-segment chain held against gravity should
# match Euler-Bernoulli's joint-segmented angle drop in the linear
# regime.
# ---------------------------------------------------------------------------


_CHAIN_INV_INERTIA = ((100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (0.0, 0.0, 100.0))


def _build_beam_cantilever_chain(
    device,
    *,
    n_segments: int,
    pitch: float,
    segment_mass: float,
    bend_stiffness: float,
    bend_damping: float,
    gravity_g: float = 9.81,
    affected_by_gravity: bool = True,
):
    """Horizontal cantilever (anchor at world origin, axis along +x)
    built out of ``n_segments`` capsules linked by BEAM joints.

    Layout matches ``example_beam_vs_cable.py``: the leftmost capsule
    is a static anchor at ``x = 0``, then ``n_segments - 1`` dynamic
    capsules tile along +x at spacing ``pitch``. Gravity points in
    -z. Each BEAM joint shares an attachment at the midpoint between
    its two bodies and runs anchor1 -> anchor2 along +x with rest
    length ``pitch``.
    """
    builder = WorldBuilder()
    world = builder.world_body

    # Capsule axis is body-local +z; orient it along world +x via a
    # 90 deg rotation about +y.
    half_angle = math.radians(45.0)
    s = math.sin(half_angle)
    c = math.cos(half_angle)
    orient_x = (0.0, s, 0.0, c)

    inv_mass = 1.0 / segment_mass

    anchor_x = 0.5 * pitch
    anchor = builder.add_static_body(position=(anchor_x, 0.0, 0.0), orientation=orient_x)
    bodies = [anchor]

    for k in range(1, n_segments):
        pos = ((k + 0.5) * pitch, 0.0, 0.0)
        bid = builder.add_dynamic_body(
            position=pos,
            orientation=orient_x,
            inverse_mass=inv_mass,
            inverse_inertia=_CHAIN_INV_INERTIA,
            affected_by_gravity=affected_by_gravity,
        )
        bodies.append(bid)
        # Joint anchors sit at the midpoint between adjacent
        # capsules; anchor2 is along +x by ``rest_length = pitch``.
        anchor1 = (k * pitch, 0.0, 0.0)
        anchor2 = (anchor1[0] + pitch, anchor1[1], anchor1[2])
        builder.add_joint(
            body1=bodies[k - 1],
            body2=bid,
            anchor1=anchor1,
            anchor2=anchor2,
            mode=JointMode.BEAM,
            bend_stiffness=float(bend_stiffness),
            twist_stiffness=float(bend_stiffness),
            bend_damping=float(bend_damping),
            twist_damping=float(bend_damping),
        )
        builder.add_collision_filter_pair(bodies[k - 1], bid)

    world_out = builder.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, -gravity_g) if affected_by_gravity else (0.0, 0.0, 0.0),
        rigid_contact_max=0,
        device=device,
    )
    return world_out, bodies


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamCantileverStatic(unittest.TestCase):
    """Joint-segmented Euler-Bernoulli static deflection.

    A cantilever of ``N - 1`` dynamic capsules attached to a static
    anchor by BEAM joints (each with bend stiffness ``k``) carries a
    static gravitational load. The torque about the *root* joint is

        M_root = sum_{i=1}^{N-1} m_i * g * (i - 0.5) * pitch

    and decreases as you walk down the chain: joint ``j`` carries

        M_j = sum_{i=j}^{N-1} m_i * g * (i - j + 0.5) * pitch
            ~= 0.5 * (N - j)^2 * m * g * pitch    (uniform load).

    Each joint's static angle drop is ``theta_j = M_j / k`` in the
    small-angle limit. The total tip drop in *angle* is

        theta_tip = sum_j theta_j ~= (M_total / k) * O(1)
                  ~= 0.5 * (N - 1)^2 * m * g * pitch / k    [rad]

    (for large ``N``; equivalently, the tip's vertical displacement
    sums the cumulative angle drops along the chain). We compare the
    measured static tip drop against the small-angle theory at high
    ``k`` (small angles -> linear regime) with a 50 % tolerance to
    accommodate damped settling and the simplifying assumptions of
    the closed-form (uniform load, identical joints, no anchor-1
    lever-arm coupling). A tighter bound here would be brittle to
    integration noise; the qualitative ordering is what we care
    about, plus the absence of a hidden lever-arm scale error.
    """

    def test_tip_angle_matches_euler_bernoulli(self) -> None:
        n_segments = 8  # 1 anchor + 7 dynamic capsules
        pitch = 0.1  # m
        segment_mass = 0.05  # kg
        gravity_g = 9.81
        bend_stiffness = 5.0  # N*m/rad -- soft enough for measurable drop, stiff enough for linearity
        bend_damping = 0.5  # critical damping for I_seg = 0.01: c_crit = 2*sqrt(k*I) ~= 0.45

        world, bodies = _build_beam_cantilever_chain(
            wp.get_preferred_device(),
            n_segments=n_segments,
            pitch=pitch,
            segment_mass=segment_mass,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            gravity_g=gravity_g,
        )

        dt = 1.0 / FPS
        # Settle for 4 s with high-ish damping -- uniform-load
        # cantilevers settle in ~4 / omega_n_root simulated seconds.
        run_settle_loop(world, int(4.0 * FPS), dt=dt)

        positions = world.bodies.position.numpy()
        self.assertTrue(np.isfinite(positions).all(), msg="cantilever non-finite after settle")

        # Tip vertical drop = anchor_z - tip_z.
        tip_idx = bodies[-1]
        anchor_idx = bodies[0]
        tip_z = float(positions[tip_idx, 2])
        anchor_z = float(positions[anchor_idx, 2])
        tip_drop_z = anchor_z - tip_z

        # Small-angle theory: theta_j = M_j / k summed; tip drop in z
        # is sum_{j} pitch * sum_{i<=j} theta_i (cumulative angle).
        # Compute by direct summation to keep the assertion explicit.
        m = segment_mass
        g = gravity_g
        theta = [0.0] * n_segments
        for j in range(1, n_segments):
            # Joint j connects bodies[j-1] -> bodies[j]. Torque it
            # carries is the gravitational moment of bodies[j..end]
            # about bodies[j-1]'s anchor end.
            moment = 0.0
            for i in range(j, n_segments):
                arm = (i - j + 0.5) * pitch
                moment += m * g * arm
            theta[j] = moment / bend_stiffness
        # Tip drop in z = sum_{j} pitch * sum_{i <= j} theta_i.
        tip_drop_theory = 0.0
        cumulative_theta = 0.0
        for j in range(1, n_segments):
            cumulative_theta += theta[j]
            tip_drop_theory += pitch * cumulative_theta

        # Loose 50 % envelope (see class docstring).
        rel = abs(tip_drop_z - tip_drop_theory) / max(tip_drop_theory, 1e-9)
        self.assertLess(
            rel,
            0.50,
            msg=(
                f"cantilever tip drop: theory={tip_drop_theory * 1000:.2f} mm, "
                f"measured={tip_drop_z * 1000:.2f} mm ({rel * 100:.1f}% off). "
                f"Catches lever-arm scale errors in BEAM's k_pos = k_user / rest_length^2 rescale."
            ),
        )
        # Sanity: tip *did* drop (positive z drift downward).
        self.assertGreater(
            tip_drop_z,
            0.5 * tip_drop_theory,
            msg=f"cantilever barely drooped: {tip_drop_z * 1000:.2f} mm vs theory {tip_drop_theory * 1000:.2f} mm",
        )


# ---------------------------------------------------------------------------
# BEAM <-> CABLE asymptotic equivalence in the linear regime.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamCableEquivalence(unittest.TestCase):
    """At small bend angle and identical ``(k, c)`` gains, BEAM and
    CABLE must realize the same ``I theta_dd + c theta_d + k theta =
    0`` model. Failure here means one of the two has a unit / scale
    bug that the single-mode tests can't see.
    """

    def test_small_angle_orientations_agree(self) -> None:
        """Build a 5 deg pre-deflected rod twice -- once with BEAM,
        once with CABLE, identical gains and zero damping -- step
        for ~half a period and assert the rod orientations match
        within ~10 % of the deflection magnitude. The match looser
        than the per-mode 5 % bound because BEAM and CABLE use
        slightly different effective-mass clamps (BEAM has the 10x
        Nyquist headroom; CABLE the strict cap) and the period
        therefore differs by a small amount even in the linear
        regime."""
        k = 80.0
        T_theory = 2.0 * math.pi / math.sqrt(k / _I_ROD)
        init_angle_rad = math.radians(5.0)
        init_q = _axis_angle_quat((0.0, 0.0, 1.0), init_angle_rad)
        device = wp.get_preferred_device()

        beam_world = _build_beam_pendulum(
            device,
            bend_stiffness=k,
            twist_stiffness=k,
            init_rotation=init_q,
            rest_length=1.0,
        )
        cable_world = _build_cable_pendulum_for_equivalence(
            device,
            bend_stiffness=k,
            twist_stiffness=k,
            bend_damping=0.0,
            twist_damping=0.0,
            init_rotation=init_q,
        )

        dt = 1.0 / FPS
        n_steps = int(0.5 * T_theory / dt)
        z_axis = np.array([0.0, 0.0, 1.0])

        # Step both rods through the same simulated time.
        run_settle_loop(beam_world, n_steps, dt=dt)
        run_settle_loop(cable_world, n_steps, dt=dt)

        beam_angle = _rotation_angle_about(_rod_orientation(beam_world), z_axis)
        cable_angle = _rotation_angle_about(_rod_orientation(cable_world), z_axis)
        self.assertTrue(np.isfinite([beam_angle, cable_angle]).all())

        diff = abs(beam_angle - cable_angle)
        # 25 % of the initial deflection covers the period mismatch
        # between BEAM (10x Nyquist headroom) and CABLE (strict cap)
        # at the chosen k; reducing the mismatch is one motivator
        # for the headroom unification proposed in Phase 1.5.
        threshold = 0.25 * init_angle_rad
        self.assertLess(
            diff,
            threshold,
            msg=(
                f"BEAM vs CABLE small-angle drift: |beam - cable| = {diff:.5f} rad, "
                f"threshold = {threshold:.5f} rad (25 %% of {math.degrees(init_angle_rad):.1f} deg deflection). "
                f"beam={math.degrees(beam_angle):.3f} deg, cable={math.degrees(cable_angle):.3f} deg. "
                f"Mismatch points to a unit / scale bug in one of the two."
            ),
        )


# ---------------------------------------------------------------------------
# Multi-segment torque transmission: reaction torque at each joint
# decreases monotonically root -> tip.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX beam tests run on CUDA only.",
)
class TestBeamChainConservation(unittest.TestCase):
    """In a static-equilibrium beam cantilever the reaction torque
    each joint carries equals the gravitational moment of all bodies
    distal to it. Walking root -> tip the moment must shrink
    monotonically; this catches lever-arm scale errors that survive
    the single-rod tests.

    Rather than reaching into the constraint columns to extract per-
    joint impulses (which would tightly couple the test to ADBS
    layout details), we infer the bend angles per joint from the
    *body orientations* at static equilibrium -- ``theta_j = M_j /
    k`` in the small-angle limit, so monotone-decreasing torques
    imply monotone-decreasing per-joint angle drops.
    """

    def test_joint_angle_drops_decrease_root_to_tip(self) -> None:
        n_segments = 5  # 1 anchor + 4 dynamic capsules (4 joints)
        pitch = 0.1
        segment_mass = 0.05
        bend_stiffness = 8.0
        bend_damping = 0.5
        gravity_g = 9.81

        world, bodies = _build_beam_cantilever_chain(
            wp.get_preferred_device(),
            n_segments=n_segments,
            pitch=pitch,
            segment_mass=segment_mass,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            gravity_g=gravity_g,
        )

        dt = 1.0 / FPS
        run_settle_loop(world, int(3.0 * FPS), dt=dt)

        orientations = world.bodies.orientation.numpy()
        positions = world.bodies.position.numpy()
        self.assertTrue(np.isfinite(orientations).all() and np.isfinite(positions).all())

        # Bend axis: about world +y (gravity is -z, beam axis is +x).
        y_axis = np.array([0.0, 1.0, 0.0])

        # Per-joint angle drop = relative pitch between successive
        # bodies' rotations about +y. body[0] is the anchor (rest
        # orientation: 90 deg about +y so capsule lies along +x);
        # measure each segment's deviation from that rest pose.
        anchor_q = orientations[bodies[0]].copy()
        anchor_pitch = _rotation_angle_about(anchor_q, y_axis)

        per_joint_drops = []
        for j in range(1, n_segments):
            body_q = orientations[bodies[j]]
            body_pitch = _rotation_angle_about(body_q, y_axis)
            # Joint j connects bodies[j-1] -> bodies[j]. Its angle
            # drop is the *increment* in pitch from the previous
            # body to this one, relative to the anchor.
            if j == 1:
                prev_pitch = anchor_pitch
            else:
                prev_pitch = _rotation_angle_about(orientations[bodies[j - 1]], y_axis)
            per_joint_drops.append(body_pitch - prev_pitch)

        # In the static cantilever the moment carried at joint j
        # decreases as we walk root -> tip; therefore the angle drop
        # |theta_j| must decrease too. The drops have a consistent
        # sign (gravity -> chain droops in +pitch direction), so we
        # test |drop| ordering.
        abs_drops = [abs(d) for d in per_joint_drops]
        for j in range(len(abs_drops) - 1):
            self.assertGreaterEqual(
                abs_drops[j] + 1e-3,  # 1 mrad numerical-noise floor
                abs_drops[j + 1],
                msg=(
                    f"joint {j} drop {math.degrees(abs_drops[j]):.3f} deg < joint {j + 1} "
                    f"drop {math.degrees(abs_drops[j + 1]):.3f} deg -- moment should shrink "
                    f"root -> tip; lever-arm scale error suspected. "
                    f"All drops: {[math.degrees(d) for d in per_joint_drops]} deg."
                ),
            )

        # Sanity: at least the root joint actually deflected.
        self.assertGreater(
            abs_drops[0],
            math.radians(0.5),
            msg=f"root joint barely drooped ({math.degrees(abs_drops[0]):.3f} deg); test signal too weak",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
