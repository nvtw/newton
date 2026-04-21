# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical validation for the unified dual-ball-socket constraint.

These are acceptance tests aimed squarely at the actuator portion of
:class:`~newton._src.solvers.jitter.world_builder.JointDescriptor` --
the same unified constraint that materialises :attr:`JointMode.REVOLUTE`
and :attr:`JointMode.PRISMATIC` (the ball-socket mode has no actuator).
Each test builds the minimum-viable scene, simulates it to steady
state, and checks a *closed-form* physical identity. If the actuator
misapplies a reaction, flips an axis, or scales a torque by ``dt``,
these tests fail loudly with a physically meaningful residual rather
than a vague tolerance bust.

Test catalogue
--------------

``TestPendulumFrequency.test_undamped_period``
    Free revolute pendulum with no drive, no limits, no friction.
    Small-angle release; period must match ``T = 2*pi*sqrt(L/g)``.
    Catches any residual stiffness on the free axial DoF.

``TestPendulumFrequency.test_damped_decay_identity``
    Same scene, velocity drive with ``target_velocity=0`` and large
    ``max_force_drive``. The decay rate ``gamma`` and damped frequency
    ``omega_d`` measured from zero-crossings + peak amplitudes must
    satisfy ``omega_d^2 + gamma^2 = g/L`` -- a universal identity for
    any *linear* damping (the relation holds independent of the
    specific damping coefficient Jitter's soft-constraint picks).
    Catches wrong-sign reactions (amplitudes would grow) and reactions
    on the wrong body (no damping at all).

``TestLinearOscillator.test_undamped_period``
    Linear analog of the pendulum frequency test. A bob on a
    **horizontal** prismatic joint with a pure spring drive
    (``stiffness_drive=k``, ``damping_drive=0``,
    ``drive_mode=POSITION``, ``target=0``). Gravity is orthogonal to
    the slide axis and is absorbed by the rank-5 lock, so the 1-D
    dynamics reduce to :math:`m\\ddot{x} = -kx` with period
    :math:`T = 2\\pi\\sqrt{m/k}`. Zero-crossing measurement on
    ``x(t)``; 5% tolerance.

``TestLinearOscillator.test_damped_decay_identity``
    Same prismatic scene with both a spring and a damper on the
    drive (``stiffness_drive=k``, ``damping_drive=c``,
    ``drive_mode=POSITION``). Decay rate :math:`\\gamma` from the
    log-linear fit of peak amplitudes + damped frequency
    :math:`\\omega_d` from zero-crossings must satisfy
    :math:`\\omega_d^2 + \\gamma^2 = k/m` (5% tolerance). Catches
    wrong-sign linear reactions the same way the revolute version
    catches wrong-sign angular reactions.

``TestActuatorContactReaction.test_prismatic_motor_presses_into_ground``
    Sphere on a prismatic joint, drive saturated pushing down. In
    steady-state static equilibrium the ground-contact normal force
    must equal ``m*g + F_motor`` (drive force adds directly to weight).

``TestActuatorContactReaction.test_revolute_motor_presses_into_ground``
    Horizontal pendulum, revolute motor torque ``tau`` pressing the
    sphere into the ground. Ground-contact normal force must equal
    ``m*g + tau/L`` (torque / lever = added downward force at the
    bob).

``TestLimitSpringDamper.test_prismatic_limit_spring_holds_weight``
    Sphere hangs from a vertical prismatic slider. The joint's limit
    row is configured as a PD spring-damper (``stiffness_limit`` /
    ``damping_limit`` branch -> absolute gains ``kp``, ``kd``). Gravity
    loads the spring. Steady-state deflection must satisfy Hooke's
    law: :math:`\\Delta x = m g / k`. Damping is only there to kill the
    oscillation -- the identity holds independent of it.

``TestLimitSpringDamper.test_revolute_limit_torsion_spring_holds_lever``
    Same idea, one dimension up: a horizontal revolute hinge with a
    sphere on a rigid lever of length :math:`L` orthogonal to gravity.
    The hinge's limit row is a torsional PD spring-damper. Small-angle
    equilibrium: :math:`\\theta \\approx m g L / k`. We compare
    against the full nonlinear equilibrium
    :math:`k\\theta = m g L \\cos\\theta` to stay honest at the
    edges of "small".

``TestPrismaticUnitCubeLimitSpring.test_unit_cube_sag_against_spring_limit``
    Vertical prismatic slider with a *unit cube* hanging on a tiny
    spring-damper limit window around zero. Closed-form static
    deflection :math:`\\Delta = m g / k` -- mass and inertia come from
    Newton's :class:`ModelBuilder` for a 1m unit cube with
    :attr:`density=1` so the body is purely defined by SI primitives;
    if the spring stiffness drifts by even a few percent from its
    request, the deflection assertion fires with the analytic
    residual.

``TestLimitDampingDecay.test_prismatic_limit_damped_decay_identity``
    Prismatic spring-damper limit excited from rest at a known
    displacement (no gravity). Free underdamped oscillation; the
    measured decay rate :math:`\\gamma` (log-linear fit on
    successive-peak amplitudes) and damped frequency
    :math:`\\omega_d` (zero-crossings) must satisfy
    :math:`\\omega_d^2 + \\gamma^2 = k/m_{\\rm eff}` -- the universal
    damped-harmonic-oscillator identity. Catches any
    timestep-dependent damping (e.g. PGS softness leaking through)
    because the identity is independent of :math:`c`, only the
    decomposition into ``omega_d`` and ``gamma`` shifts with ``c``.

``TestLimitDampingDecay.test_revolute_limit_damped_decay_identity``
    Angular analog: torsional spring-damper limit on a hinge with an
    orthogonal lever bob. Same DHO identity in angular units
    :math:`\\omega_d^2 + \\gamma^2 = k_\\theta / I_{\\rm eff}` with
    :math:`I_{\\rm eff} = m L^2`.

``TestOneSidedLimits.test_prismatic_max_only_blocks_positive_slide``
    Prismatic with a single-sided ``max`` clamp: the bob is allowed
    to slide freely in one direction (no spring force) but is
    arrested by the spring-damper limit when it crosses
    ``max_value``. Verifies that the inactive side of the limit
    really is inactive (no impulse leakage) and the active side
    behaves like Hooke's law.

``TestOneSidedLimits.test_prismatic_min_only_blocks_negative_slide``
    Mirror image -- only the ``min`` side is loaded.

``TestOneSidedLimits.test_revolute_max_only_blocks_positive_angle``
    Revolute version: pendulum allowed to swing freely in one
    direction, sprung back when it crosses the upper angle limit.

``TestPDDriveImpulse.test_prismatic_position_drive_force_balance``
    Contact-free, no gravity. A prismatic POSITION drive
    (``kp, kd``) holds the bob at a steady-state offset from
    ``target`` set by an externally applied counter-force. The
    measured deflection must satisfy ``F_ext = k * (x - x*)`` with
    no contribution from contacts, gravity, or the rank-5 lock.
    This is the cleanest possible check that the drive applies
    exactly the impulse the math says it should.

``TestPDDriveImpulse.test_prismatic_velocity_drive_force_balance``
    Same scene with a VELOCITY drive (``stiffness_drive=0``,
    ``damping_drive=c``). Steady state is ``v = v*``, but with an
    external opposing constant force the steady-state velocity
    settles at ``v_ss = v* - F_ext / c``. We measure ``v_ss`` and
    invert for ``c_meas``.

``TestPDDriveImpulse.test_revolute_position_drive_torque_balance``
    Revolute analog of the prismatic POSITION test: an external
    constant torque (gravity on a horizontal lever) holds the drive
    away from ``target``; measured deflection inverts to ``k_theta``.

The contact reaction is read via :meth:`World.gather_contact_pair_wrenches`
so it is the *actual* impulse the Jitter PGS solver applied, not a
post-hoc reconstruction.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)
from newton._src.solvers.jitter.world_builder import (
    DriveMode,
    JointMode,
)

# ---------------------------------------------------------------------------
# Physical constants shared across tests
# ---------------------------------------------------------------------------

_G = 9.81  # [m/s^2]


# ---------------------------------------------------------------------------
# Reusable mini-pipeline: Newton ModelBuilder + Jitter World with contacts
# ---------------------------------------------------------------------------


class _JitterScene:
    """Lightweight wrapper around a Newton ``Model`` + Jitter ``World``.

    Intentionally *not* a subclass of :class:`DemoExample` -- this
    stays as close to the Jitter solver as possible so test failures
    point straight at the actuator / contact path instead of at
    :class:`DemoExample` plumbing. Every method is a thin wrapper
    around the same kernels :class:`example_pyramid.Example` uses.
    """

    def __init__(
        self,
        model_builder: newton.ModelBuilder,
        world_builder_factory,
        *,
        fps: int,
        substeps: int,
        solver_iterations: int,
        gravity: tuple[float, float, float] = (0.0, 0.0, -_G),
        enable_contacts: bool = True,
        expected_masses: dict[int, float] | None = None,
    ):
        """Finalise ``model_builder``, mirror into Jitter, wire contacts.

        Args:
            model_builder: Pre-populated Newton :class:`ModelBuilder`.
                Must contain every body that the Jitter world will
                touch, because :func:`build_jitter_world_from_model`
                walks ``model.body_*`` for the mirror.
            world_builder_factory: callable
                ``(newton_to_jitter: dict[int, int], builder:
                WorldBuilder) -> None`` that adds joints / motors to
                the Jitter :class:`WorldBuilder` using the Newton->
                Jitter body map produced by
                :func:`build_jitter_world_from_model`. The factory is
                called *after* the mirror and *before* ``finalize``.
            fps, substeps: Time-stepping. Physics runs at
                ``fps * substeps`` Hz.
            solver_iterations: PGS iterations per substep.
            gravity: World-space gravity vector [m/s^2].
            enable_contacts: When False, skips ``CollisionPipeline``
                creation entirely (saves compile time on pendulum
                tests with no shapes / no ground).
        """
        # CUDA-only path: graph capture (below) replays the whole
        # per-frame pipeline in one launch, which is the only way
        # these multi-second simulations complete in reasonable time.
        # Callers gate on :func:`wp.is_cuda_available` in ``setUpClass``
        # to skip cleanly on CPU-only CI.
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        self.device = wp.get_device("cuda:0")
        self.fps = int(fps)
        self.substeps = int(substeps)
        self.frame_dt = 1.0 / self.fps
        self.substep_dt = self.frame_dt / self.substeps

        self.model = model_builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state
        )
        # Sync post-FK poses back into ``model.body_q`` so the Jitter
        # mirror spawns bodies at the same transforms the Newton state
        # will report next frame. No-op when every body was added with
        # an explicit xform and no joint_q override -- belt-and-braces.
        self.model.body_q.assign(self.state.body_q)

        if enable_contacts:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model, contact_matching=True
            )
            self.contacts = self.collision_pipeline.contacts()
            rigid_contact_max = int(
                self.contacts.rigid_contact_point0.shape[0]
            )
        else:
            self.collision_pipeline = None
            self.contacts = None
            rigid_contact_max = 0

        # ``expected_masses`` lets the caller lock in each body's mass
        # against an analytical expectation; mismatches (e.g. shape
        # density silently inflating ``add_body(mass=...)``) raise here
        # rather than producing mystery factors in downstream
        # force/torque assertions.
        builder, newton_to_jitter = build_jitter_world_from_model(
            self.model, expected_masses=expected_masses
        )
        self.newton_to_jitter = newton_to_jitter
        world_builder_factory(newton_to_jitter, builder)

        max_contact_columns = (
            max(16, (rigid_contact_max + 5) // 6) if rigid_contact_max else 0
        )
        num_shapes = int(self.model.shape_count)

        self.world = builder.finalize(
            substeps=self.substeps,
            solver_iterations=int(solver_iterations),
            gravity=gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            device=self.device,
        )

        if enable_contacts:
            shape_body_np = self.model.shape_body.numpy()
            shape_body_jitter = np.where(
                shape_body_np < 0, 0, shape_body_np + 1
            )
            self._shape_body = wp.array(
                shape_body_jitter, dtype=wp.int32, device=self.device
            )
        else:
            self._shape_body = None

        self._sync_newton_to_jitter()

        # Record the full per-frame pipeline (Newton<->Jitter sync +
        # optional collide + world.step) as a single CUDA graph.
        # Each subsequent :meth:`step` call replays the graph with
        # one :func:`wp.capture_launch` -- multi-thousand-frame
        # settle loops now take seconds instead of minutes. Follows
        # the :mod:`example_pyramid` capture pattern (first eager
        # run under ``ScopedCapture`` warms up kernel compilation
        # *and* records the graph).
        with wp.ScopedCapture(device=self.device) as capture:
            self._simulate()
        self._graph = capture.graph

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            newton_to_jitter_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_jitter_to_newton(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            jitter_to_newton_kernel,
            dim=n,
            inputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def _simulate(self) -> None:
        """Inner per-frame pipeline, captured once into a CUDA graph.
        Pure device-side work: state sync, optional collision, Jitter
        substep loop, reverse sync. No host-side branching, no
        readbacks.
        """
        self._sync_newton_to_jitter()
        if self.collision_pipeline is not None:
            self.model.collide(
                self.state,
                contacts=self.contacts,
                collision_pipeline=self.collision_pipeline,
            )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            picking=None,
        )
        self._sync_jitter_to_newton()

    def step(self) -> None:
        """Advance one frame by replaying the captured CUDA graph."""
        wp.capture_launch(self._graph)

    def body_pose(self, newton_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position[3], orientation[4 xyzw])`` in world space
        for the Newton body index ``newton_idx``, read from Jitter's
        body container (the solver's own state, not the synced Newton
        buffer).
        """
        j = self.newton_to_jitter[int(newton_idx)]
        return (
            self.world.bodies.position.numpy()[j],
            self.world.bodies.orientation.numpy()[j],
        )

    def gather_contact_wrench_on_body(
        self, newton_body: int
    ) -> tuple[np.ndarray, int]:
        """Return the summed contact force [N] applied on ``newton_body``
        during the most recent substep, together with the number of
        distinct contact pairs folded in.

        Sign convention: every contact pair reports the force felt by
        its ``body2``. For pairs where ``newton_body`` is body2 the
        force is added verbatim; where it is body1 the force is
        negated (Newton's third law -- the pair pushes body1 by
        ``-F``). Inactive pairs are skipped.
        """
        if self.world.max_contact_columns == 0:
            return np.zeros(3, dtype=np.float32), 0
        n_cols = self.world.max_contact_columns
        pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=self.device)
        pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_count = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        self.world.gather_contact_pair_wrenches(
            pair_w, pair_b1, pair_b2, pair_count
        )
        pw = pair_w.numpy()[:, :3]
        b1 = pair_b1.numpy()
        b2 = pair_b2.numpy()
        cnt = pair_count.numpy()

        j = self.newton_to_jitter[int(newton_body)]
        total = np.zeros(3, dtype=np.float32)
        npairs = 0
        for i in range(n_cols):
            if cnt[i] <= 0:
                continue
            if b2[i] == j:
                total += pw[i]
                npairs += 1
            elif b1[i] == j:
                total -= pw[i]
                npairs += 1
        return total, npairs


# ---------------------------------------------------------------------------
# Pendulum tests (no contacts, no shapes needed beyond a tiny sphere)
# ---------------------------------------------------------------------------


def _add_pendulum_bob(
    mb: newton.ModelBuilder,
    *,
    position: tuple[float, float, float],
    mass: float,
    radius: float,
    orientation: wp.quat | None = None,
) -> int:
    """Add a pendulum bob with *explicit* solid-sphere inertia and a
    massless shape.

    We deliberately do **not** rely on :meth:`add_shape_sphere` to
    derive the inertia from density: the default shape density
    (1000 kg/m^3) combined with a small radius ends up much less
    massive than the ``mass`` argument we passed to
    :meth:`add_body`, so the body's inertia tensor would come out
    *orders of magnitude too small* compared to ``m``. That leaves
    ``body_inv_inertia`` huge, and any residual reaction torque the
    rank-5 revolute lock lets through excites a numerical
    instability on the free rotational DoF that explodes within a
    few substeps (measured: bob flies off to ~1e19 m within ~30 ms).

    Fix: pass the full ``(2/5) m r^2`` sphere inertia through the
    ``inertia`` argument of :meth:`add_body`, and add the visual /
    collision sphere with density = 0 so it contributes nothing to
    mass or inertia.
    """
    I_scalar = 0.4 * float(mass) * float(radius) * float(radius)
    inertia = wp.mat33(
        I_scalar, 0.0, 0.0,
        0.0, I_scalar, 0.0,
        0.0, 0.0, I_scalar,
    )
    q = orientation if orientation is not None else wp.quat_identity()
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(*position), q=q),
        mass=float(mass),
        inertia=inertia,
    )
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)
    mb.add_shape_sphere(body, radius=float(radius), cfg=massless)
    return body


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestPendulumFrequency(unittest.TestCase):
    """Small-angle pendulum frequency tests. The only constraint in
    play is ``JointMode.REVOLUTE`` (with and without a velocity
    drive); nothing else in the scene can bias the period.
    """

    # Scene parameters. Keep ``L`` and ``MASS`` at 1 for clean
    # numbers: ``omega_0 = sqrt(g/L) = sqrt(9.81) = 3.1321 rad/s``,
    # ``T_0 = 2*pi/omega_0 = 2.007 s``.
    L = 1.0
    MASS = 1.0
    BOB_RADIUS = 0.02
    THETA0 = 0.1  # Release angle [rad]; small-angle correction to T is ~0.06%.

    FPS = 240
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    # ------------------------------------------------------------------
    # Scene builders
    # ------------------------------------------------------------------

    def _build_pendulum(self, *, damping_gain: float = 0.0) -> _JitterScene:
        """Build a revolute pendulum with hinge axis ``+y`` at the
        origin and the bob hanging at ``(sin(theta0)*L, 0, -cos(theta0)*L)``.

        When ``damping_gain > 0`` the joint carries a velocity drive
        tracking ``target_velocity=0`` with ``damping_drive =
        damping_gain`` and ``stiffness_drive = 0`` -- a pure viscous
        damper in the Jitter2 LinearMotor / AngularMotor PD
        formulation. The torque applied to the bob is
        ``tau = -damping_gain * theta_dot`` (after the solver converges),
        giving a damped harmonic oscillator with
        ``gamma = damping_gain / (2 * m_eff)``.
        """
        L = self.L
        theta0 = self.THETA0
        bob_pos = (L * math.sin(theta0), 0.0, -L * math.cos(theta0))

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb, position=bob_pos, mass=self.MASS, radius=self.BOB_RADIUS
        )

        # Hinge anchors: both at the world origin, axis = +y so
        # anchor2 = anchor1 + (0, 1, 0). With the bob at
        # ``(L sin0, 0, -L cos0)`` and the anchor at origin, the hinge
        # line passes through the origin parallel to +y -- the bob is
        # offset by L in the xz-plane so gravity on ``-z`` torques
        # the arm about the hinge axis.
        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 1.0, 0.0),
                mode=JointMode.REVOLUTE,
                drive_mode=(
                    DriveMode.VELOCITY if damping_gain > 0.0 else DriveMode.OFF
                ),
                target_velocity=0.0,
                # Large enough torque cap that the drive never
                # saturates under our small-angle motion.
                max_force_drive=(
                    1.0e3 if damping_gain > 0.0 else 0.0
                ),
                # Pure viscous damper (kp = 0, kd = damping_gain):
                # applies torque ``tau = -kd * theta_dot`` through the
                # Jitter2 LinearMotor / AngularMotor PD formulation.
                # This maps cleanly onto the damped-oscillator decay
                # identity ``omega_d^2 + gamma^2 = omega_0^2`` with
                # ``gamma = kd / (2 * m_eff)``.
                stiffness_drive=0.0,
                damping_drive=float(damping_gain),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,
        )
        scene.bob = bob
        return scene

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------

    def _record_angle(
        self, scene: _JitterScene, num_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``num_frames`` frames and return ``(t, theta)`` arrays.

        ``theta`` is the angle between the arm (hinge -> bob) and
        the rest direction ``-z``, *signed* with the same convention
        as the release angle: positive when the bob is displaced
        toward ``+x``. Computed directly from the Jitter body position
        so it is independent of the Newton state sync.
        """
        ts = np.empty(num_frames, dtype=np.float64)
        thetas = np.empty(num_frames, dtype=np.float64)
        t = 0.0
        for i in range(num_frames):
            scene.step()
            pos, _ = scene.body_pose(scene.bob)
            x = float(pos[0])
            z = float(pos[2])
            # arm vector in the xz-plane: bob - hinge_origin.
            # Angle relative to -z, signed by x.
            theta = math.atan2(x, -z)
            ts[i] = t
            thetas[i] = theta
            t += scene.frame_dt
        return ts, thetas

    @staticmethod
    def _zero_crossings_up(ts: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return interpolated times of theta crossings from negative
        to positive. Linear interpolation between samples straddling
        zero reduces the period-measurement error to ``~0.5*dt^2``
        (quadratic in frame_dt) -- plenty below the 5% tolerance.
        """
        signs = np.signbit(theta)  # True when negative
        # Up-crossing = previous sample negative, current positive.
        ups = np.flatnonzero(signs[:-1] & ~signs[1:])
        t_crossings = []
        for i in ups:
            t0, t1 = ts[i], ts[i + 1]
            th0, th1 = theta[i], theta[i + 1]
            # Linear interpolation for theta = 0.
            t_crossings.append(t0 + (t1 - t0) * (-th0) / (th1 - th0))
        return np.asarray(t_crossings, dtype=np.float64)

    @staticmethod
    def _peak_maxima(ts: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t_peak, theta_peak)`` for local maxima of theta.
        Quadratic fit across three samples for sub-sample accuracy.
        """
        tp, vp = [], []
        for i in range(1, len(theta) - 1):
            if theta[i] > theta[i - 1] and theta[i] > theta[i + 1]:
                # Parabola through (ts[i-1..i+1], theta[i-1..i+1]).
                a = 0.5 * (theta[i - 1] - 2 * theta[i] + theta[i + 1])
                b = 0.5 * (theta[i + 1] - theta[i - 1])
                # Vertex offset in samples.
                if abs(a) > 1.0e-30:
                    dx = -b / (2.0 * a)
                else:
                    dx = 0.0
                dt_frame = ts[i + 1] - ts[i]
                tp.append(ts[i] + dx * dt_frame)
                vp.append(theta[i] - (b * b) / (4.0 * a) if abs(a) > 1.0e-30 else theta[i])
        return np.asarray(tp, dtype=np.float64), np.asarray(vp, dtype=np.float64)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_undamped_period(self):
        """Pendulum with no drive -- period must match
        ``T = 2*pi*sqrt(L/g)`` within 5%.

        Records 6 seconds of motion (~3 periods at ``T = 2.007 s``),
        computes the mean inter-crossing period, compares. A
        systematic offset here would point at a residual stiffness
        on the free axial DoF of the revolute (i.e. the positional
        block leaking torque into the hinge axis).
        """
        scene = self._build_pendulum(damping_gain=0.0)
        num_frames = int(6.0 * self.FPS)
        ts, thetas = self._record_angle(scene, num_frames)

        crossings = self._zero_crossings_up(ts, thetas)
        self.assertGreaterEqual(
            len(crossings),
            3,
            f"expected >= 3 up-crossings in 6 s, got {len(crossings)}; "
            f"angle range {thetas.min():.4f}..{thetas.max():.4f}",
        )
        measured_T = float(np.mean(np.diff(crossings)))
        expected_T = 2.0 * math.pi * math.sqrt(self.L / _G)
        rel_err = abs(measured_T - expected_T) / expected_T
        self.assertLess(
            rel_err,
            0.05,
            f"undamped period off: measured={measured_T:.4f}s, "
            f"expected={expected_T:.4f}s, rel_err={rel_err:.2%}",
        )

    def test_damped_decay_identity(self):
        """Pendulum with a velocity drive at ``target_velocity=0`` --
        amplitudes must decay exponentially and the identity
        ``omega_d^2 + gamma^2 == omega_0^2`` must hold within 5%.

        This validates two things at once:

        1. The drive reaction has the *right sign* (an actuator bug
           that flipped the sign would pump energy in and ``gamma``
           would come out negative / the amplitudes would grow).
        2. The drive is *linear* in velocity (any nonlinearity would
           make the peak-ratio log non-constant, and / or break the
           identity).

        We deliberately don't check an absolute decay rate -- that
        would re-test Jitter's soft-constraint stiffness/damping
        formula. We check the universal physics identity instead.
        """
        scene = self._build_pendulum(damping_gain=0.5)  # mild damping
        num_frames = int(12.0 * self.FPS)
        ts, thetas = self._record_angle(scene, num_frames)

        crossings = self._zero_crossings_up(ts, thetas)
        self.assertGreaterEqual(
            len(crossings),
            3,
            "damped pendulum didn't oscillate -- the drive sign may be wrong "
            f"(angle range {thetas.min():.4f}..{thetas.max():.4f})",
        )
        T_d = float(np.mean(np.diff(crossings)))
        omega_d = 2.0 * math.pi / T_d

        # Peaks must *decrease*: positive decay.
        t_peaks, theta_peaks = self._peak_maxima(ts, thetas)
        self.assertGreaterEqual(
            len(theta_peaks),
            3,
            "need >= 3 positive peaks to fit decay",
        )
        # Keep only the clean positive peaks (>0).
        mask = theta_peaks > 0.0
        t_peaks = t_peaks[mask]
        theta_peaks = theta_peaks[mask]
        self.assertGreaterEqual(
            len(theta_peaks),
            3,
            "not enough positive peaks after filtering",
        )
        # ln(theta_peak) should be linear in t with slope -gamma.
        log_peaks = np.log(theta_peaks)
        # Linear regression (numpy polyfit, degree 1).
        slope, _intercept = np.polyfit(t_peaks, log_peaks, 1)
        gamma = -float(slope)
        self.assertGreater(
            gamma,
            0.0,
            f"peaks did not decay (slope={slope:+.4f}); actuator may be "
            "pumping energy in (wrong reaction sign / wrong body).",
        )

        omega_0_sq_expected = _G / self.L
        omega_0_sq_measured = omega_d * omega_d + gamma * gamma
        rel_err = abs(omega_0_sq_measured - omega_0_sq_expected) / omega_0_sq_expected
        self.assertLess(
            rel_err,
            0.05,
            f"damped-oscillator identity violated: "
            f"omega_d^2+gamma^2={omega_0_sq_measured:.4f} "
            f"vs g/L={omega_0_sq_expected:.4f} "
            f"(gamma={gamma:.4f} rad/s, T_d={T_d:.4f}s, "
            f"rel_err={rel_err:.2%})",
        )


# ---------------------------------------------------------------------------
# Linear spring-mass oscillator (prismatic drive analogue of the
# pendulum tests above). Same maths, one dimension down: replace the
# revolute's angular ``theta`` / gravity ``g/L`` with a prismatic
# ``x`` / spring ``k/m``.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestLinearOscillator(unittest.TestCase):
    """Frequency + damped-decay identity for a linear spring-mass on
    the unified dual-ball-socket in :attr:`JointMode.PRISMATIC`.

    The rest-length of the slider is 1 m (``anchor2 - anchor1``). We
    deliberately orient the slide along ``+y`` (horizontal, *orthogonal
    to gravity*) so gravity is absorbed by the rank-5 lock and the 1-D
    drive-row dynamics reduce to a pure :math:`m\\ddot{x} = -kx -
    c\\dot{x}` system, just like ``TestPendulumFrequency``'s revolute
    dynamics reduce to :math:`I\\ddot{\\theta} = -(m g L)\\theta -
    c\\dot{\\theta}` for small angles.

    We reuse :func:`_add_pendulum_bob` so the body has a well-defined
    explicit inertia (the shape is massless) -- same rationale as the
    pendulum tests.
    """

    # ``MASS = 1``, ``K = 10`` gives ``omega_0 = sqrt(10) = 3.162 rad/s``,
    # ``T_0 = 2*pi/omega_0 = 1.987 s`` -- comfortable to measure against
    # a 240 Hz frame rate, three clean periods in 6 s.
    MASS = 1.0
    K = 10.0  # [N/m], drive stiffness
    BOB_RADIUS = 0.02
    X0 = 0.1  # Release displacement [m] along +axis.

    FPS = 240
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    # ------------------------------------------------------------------
    # Scene builders
    # ------------------------------------------------------------------

    def _build_oscillator(self, *, damping: float) -> _JitterScene:
        """Prismatic slider along ``+y`` with the bob placed at the
        joint's zero-slide point and the drive target shifted to
        ``-X0``.

        *Why not the obvious thing (bob at ``(0, X0, 0)``, target 0)?*
        The prismatic ``slide`` coordinate is measured relative to the
        anchor snapshots taken at :meth:`finalize` time, not in world
        coordinates: the constraint witnesses anchor 1 on both bodies
        at init, and ``slide`` is the drift between those witnesses
        projected onto ``n_hat``. So at ``t=0`` the slide is always
        0 no matter where the bob is in world space -- the bob's
        initial displacement is absorbed into the body-local anchor
        snapshot.

        To release the bob from rest at displacement ``X0`` and let
        the spring pull it toward zero, we therefore place the bob at
        the slide zero point and set the drive target to ``-X0``. The
        drive error at ``t=0`` is ``slide - target = 0 - (-X0) = +X0``
        (same magnitude as a bob displaced by ``+X0`` with target
        zero), so the ODE is identical:
        :math:`m\\ddot{x} = -k(x - (-X_0)) - c\\dot{x}` about the new
        equilibrium ``x_eq = -X0``. We measure the sinusoid around
        that equilibrium in :meth:`_record_slide` (shifted back by
        ``+X0`` so the output is zero-centred like the pendulum's
        ``theta``).

        ``damping > 0`` exercises the full PD branch; ``damping == 0``
        gives the undamped spring we measure the SHM period from.
        """
        mb = newton.ModelBuilder()
        # Bob at anchor1 so initial slide = 0.
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                # Slide axis ``+y`` (orthogonal to gravity ``-z``).
                # ``n_hat = (anchor2 - anchor1) / |..| = +y``.
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 1.0, 0.0),
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.POSITION,
                # Target = -X0 so the spring equilibrium is at
                # ``slide = -X0``. The bob starts at ``slide = 0``
                # (its initial state) so the drive error at release
                # is exactly ``+X0`` -- the setup mirrors a mass
                # released from ``+X0`` on a spring with target 0.
                target=-self.X0,
                target_velocity=0.0,
                # Max_force must be large enough to never saturate:
                # peak force at release is ``k * X0 = 1 N`` + any
                # transient damper kick. 1 kN is several orders of
                # magnitude of headroom.
                max_force_drive=1.0e3,
                stiffness_drive=float(self.K),
                damping_drive=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,
        )
        scene.bob = bob
        return scene

    # ------------------------------------------------------------------
    # Measurement helpers (y-position variants of
    # ``TestPendulumFrequency._record_angle`` / ``_zero_crossings_up``
    # / ``_peak_maxima`` -- same maths, different observable)
    # ------------------------------------------------------------------

    def _record_slide(
        self, scene: _JitterScene, num_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``num_frames`` frames and return ``(t, x_centred)``
        where ``x_centred`` is the bob's ``y`` position shifted so
        that the spring equilibrium sits at 0.

        See :meth:`_build_oscillator` for why the raw bob ``y`` is
        centred around ``-X0`` instead of ``0``. Shifting by ``+X0``
        gives a zero-centred sinusoid that :meth:`_zero_crossings_up`
        and :meth:`_peak_maxima` can chew on exactly the same way as
        the pendulum's ``theta`` signal.
        """
        ts = np.empty(num_frames, dtype=np.float64)
        xs = np.empty(num_frames, dtype=np.float64)
        t = 0.0
        for i in range(num_frames):
            scene.step()
            pos, _ = scene.body_pose(scene.bob)
            # n_hat = +y so raw slide = pos.y. Shift by +X0 so the
            # recorded signal is zero-centred about the drive's
            # equilibrium at y = -X0.
            xs[i] = float(pos[1]) + self.X0
            ts[i] = t
            t += scene.frame_dt
        return ts, xs

    @staticmethod
    def _zero_crossings_up(ts: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Linear-interpolated times of negative->positive zero crossings.
        Same implementation as :meth:`TestPendulumFrequency._zero_crossings_up`
        but kept local so the two test classes stay independent.
        """
        signs = np.signbit(x)  # True when negative
        ups = np.flatnonzero(signs[:-1] & ~signs[1:])
        t_crossings = []
        for i in ups:
            t0, t1 = ts[i], ts[i + 1]
            x0, x1 = x[i], x[i + 1]
            t_crossings.append(t0 + (t1 - t0) * (-x0) / (x1 - x0))
        return np.asarray(t_crossings, dtype=np.float64)

    @staticmethod
    def _peak_maxima(
        ts: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quadratic-fit local maxima of ``x(t)``. See
        :meth:`TestPendulumFrequency._peak_maxima` for the derivation.
        """
        tp, vp = [], []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                a = 0.5 * (x[i - 1] - 2 * x[i] + x[i + 1])
                b = 0.5 * (x[i + 1] - x[i - 1])
                if abs(a) > 1.0e-30:
                    dx = -b / (2.0 * a)
                else:
                    dx = 0.0
                dt_frame = ts[i + 1] - ts[i]
                tp.append(ts[i] + dx * dt_frame)
                vp.append(
                    x[i] - (b * b) / (4.0 * a) if abs(a) > 1.0e-30 else x[i]
                )
        return np.asarray(tp, dtype=np.float64), np.asarray(vp, dtype=np.float64)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_undamped_period(self):
        """Pure spring, no damper: period must match
        :math:`T = 2\\pi\\sqrt{m/k}` within 5%.

        Records ~3 periods (6 s at ``T0 = 1.987 s``). A systematic
        offset here points at a residual stiffness or *missing*
        effective-mass in the drive row (the period scales as
        :math:`\\sqrt{m_{\\text{eff}}/k}`, so a wrong ``m_eff`` shows up
        directly).
        """
        scene = self._build_oscillator(damping=0.0)
        num_frames = int(6.0 * self.FPS)
        ts, xs = self._record_slide(scene, num_frames)

        crossings = self._zero_crossings_up(ts, xs)
        self.assertGreaterEqual(
            len(crossings),
            3,
            f"expected >= 3 up-crossings in 6 s, got {len(crossings)}; "
            f"slide range {xs.min():.4f}..{xs.max():.4f} m",
        )
        measured_T = float(np.mean(np.diff(crossings)))
        expected_T = 2.0 * math.pi * math.sqrt(self.MASS / self.K)
        rel_err = abs(measured_T - expected_T) / expected_T
        self.assertLess(
            rel_err,
            0.05,
            f"undamped spring-mass period off: measured={measured_T:.4f}s, "
            f"expected=2*pi*sqrt(m/k)={expected_T:.4f}s (m={self.MASS} kg, "
            f"k={self.K} N/m), rel_err={rel_err:.2%}",
        )

    def test_damped_decay_identity(self):
        """Spring + linear damper: amplitudes must decay exponentially
        and the identity :math:`\\omega_d^2 + \\gamma^2 = k/m` must
        hold within 5%.

        Same as the revolute ``test_damped_decay_identity`` but in a
        single linear dimension. We deliberately don't check an
        absolute ``gamma`` -- that would re-test the PD-coefficient
        formula. We check the physics identity (which is coefficient-
        agnostic).
        """
        # mild damping: ``zeta = c / (2 sqrt(k m)) = 0.5 / (2 sqrt(10))
        # = 0.0791``, well under-damped so we get plenty of peaks.
        damping = 0.5
        scene = self._build_oscillator(damping=damping)
        num_frames = int(12.0 * self.FPS)
        ts, xs = self._record_slide(scene, num_frames)

        crossings = self._zero_crossings_up(ts, xs)
        self.assertGreaterEqual(
            len(crossings),
            3,
            "damped spring-mass didn't oscillate -- the drive sign may be "
            f"wrong (slide range {xs.min():.4f}..{xs.max():.4f} m)",
        )
        T_d = float(np.mean(np.diff(crossings)))
        omega_d = 2.0 * math.pi / T_d

        t_peaks, x_peaks = self._peak_maxima(ts, xs)
        self.assertGreaterEqual(
            len(x_peaks),
            3,
            "need >= 3 positive peaks to fit decay",
        )
        mask = x_peaks > 0.0
        t_peaks = t_peaks[mask]
        x_peaks = x_peaks[mask]
        self.assertGreaterEqual(
            len(x_peaks),
            3,
            "not enough positive peaks after filtering",
        )
        log_peaks = np.log(x_peaks)
        slope, _intercept = np.polyfit(t_peaks, log_peaks, 1)
        gamma = -float(slope)
        self.assertGreater(
            gamma,
            0.0,
            f"peaks did not decay (slope={slope:+.4f}); actuator may be "
            "pumping energy in (wrong reaction sign / wrong body).",
        )

        omega_0_sq_expected = self.K / self.MASS
        omega_0_sq_measured = omega_d * omega_d + gamma * gamma
        rel_err = abs(omega_0_sq_measured - omega_0_sq_expected) / omega_0_sq_expected
        self.assertLess(
            rel_err,
            0.05,
            f"damped spring-mass identity violated: "
            f"omega_d^2+gamma^2={omega_0_sq_measured:.4f} "
            f"vs k/m={omega_0_sq_expected:.4f} "
            f"(gamma={gamma:.4f} 1/s, T_d={T_d:.4f}s, c={damping} N*s/m, "
            f"rel_err={rel_err:.2%})",
        )


# ---------------------------------------------------------------------------
# Contact-reaction tests (sphere pressed into ground by a saturated motor)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestActuatorContactReaction(unittest.TestCase):
    """Saturated-motor contact tests. The motor is in PD velocity
    mode (``stiffness_drive = 0``, ``damping_drive >> 0``) with a
    capped ``max_force_drive``; once the sphere hits the ground
    (blocking the commanded motion) the drive saturates at exactly
    the specified force / torque. The ground contact then carries
    weight + drive reaction.
    """

    MASS = 2.0
    SPHERE_RADIUS = 0.1

    FPS = 120
    SUBSTEPS = 8
    SOLVER_ITERATIONS = 20
    SETTLE_FRAMES = 180  # 1.5 s @ 120 fps -- plenty to reach steady state.

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    # ------------------------------------------------------------------
    # Scene builders
    # ------------------------------------------------------------------

    def _build_prismatic_press(self, *, F_motor: float) -> _JitterScene:
        """Static anchor above the ground, prismatic joint along ``-z``,
        dynamic sphere that the motor pushes downward with force
        ``F_motor``. The sphere starts just above the ground so it
        doesn't need to fall far before the drive saturates.
        """
        mb = newton.ModelBuilder()
        # Sphere starts at z = radius + small gap so it's above the
        # ground at t=0 and falls a tiny distance into contact.
        start_z = self.SPHERE_RADIUS + 0.05
        sphere = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, start_z), q=wp.quat_identity()),
            mass=self.MASS,
        )
        # ``density=0`` so the sphere shape contributes no implicit
        # mass on top of ``mass=self.MASS``. Without this the actual
        # body mass becomes ``MASS + (4/3)*pi*r^3 * 1000`` (~6.19 kg
        # for r=0.1, MASS=2), and the analytical
        # ``F_contact = m*g + F_motor`` check would silently drift to
        # ``m_actual*g + F_motor`` -- the same density-stack bug we
        # caught in :mod:`test_contact_force_accuracy`.
        mb.add_shape_sphere(
            sphere,
            radius=self.SPHERE_RADIUS,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        # Ground plane.
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        def _factory(n2j, builder):
            # Prismatic slide axis pointing from ``anchor1`` toward
            # ``anchor2``. Place ``anchor1`` above (at the start z),
            # ``anchor2`` one metre below -> axis = (0, 0, -1). A
            # VELOCITY drive with ``target_velocity < 0`` commands
            # motion along ``+axis`` (toward anchor2, i.e. downward),
            # capped at ``max_force_drive = F_motor``. Once the
            # sphere sits on the ground and v = 0, the drive is
            # saturated: it applies exactly ``F_motor`` along the
            # ``-z`` direction on body2 (the sphere).
            # PD velocity servo: ``stiffness_drive = 0`` kills the
            # spring term, ``damping_drive`` is the proportional gain
            # on the velocity error ``jv + target_velocity``. We pick
            # ``damping_drive`` far larger than the ``dt * max_force /
            # v_error`` minimum needed to saturate on the first
            # substep once the ground blocks motion: at
            # ``v_target = 1 m/s`` and ``v_actual = 0`` the unclamped
            # impulse magnitude is ``kd * (v_target) / (1 + kd/dt /
            # (1/eff_inv))`` and saturates against ``F_motor * dt``
            # for any ``kd`` above a few hundred here (m = 2 kg).
            # ``1e4 N*s/m`` leaves 2+ orders of magnitude of headroom
            # so the assertion on steady-state contact = m*g + F_motor
            # is insensitive to the gain.
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[sphere],
                anchor1=(0.0, 0.0, start_z),
                anchor2=(0.0, 0.0, start_z - 1.0),
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.VELOCITY,
                target_velocity=1.0,  # commanded motion along +axis (down)
                max_force_drive=float(F_motor),
                stiffness_drive=0.0,
                damping_drive=1.0e4,
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=True,
            expected_masses={sphere: self.MASS},
        )
        scene.sphere = sphere
        return scene

    def _build_revolute_press(
        self, *, tau: float, lever: float
    ) -> _JitterScene:
        """Horizontal pendulum. Revolute hinge at ``(0, 0, z_hinge)``
        with axis ``+y``; sphere at ``(lever, 0, sphere_radius)`` --
        resting on the ground at ``z = 0``. A POSITION drive with a
        target well past the ground commands rotation that lowers the
        sphere further, saturated at torque ``tau`` via
        ``max_force_drive``.

        Why a saturated POSITION drive and not a pure-velocity servo?
        The ground contact effectively sets a time-varying kinematic
        constraint on the joint's angular velocity: at rest the
        velocity is zero, so a pure-velocity PD servo with the gains
        needed to saturate against ``tau`` substep-synchronously
        develops a large ``acc`` accumulator and oscillates between
        "push harder" and "overshoot relieved" across PGS iterations.
        A saturated POSITION drive avoids that -- once the target
        can't be reached, the spring force ``k * (theta - target)``
        is monotone in ``theta`` and the impulse cap bites *cleanly*:
        every substep applies exactly ``tau * dt`` in the "press
        toward target" direction and the contact carries
        ``m*g + tau/lever`` in steady state. This is the contract
        this test is actually checking (force-balance through the
        lever), independent of which drive formulation produces it.

        z_hinge = sphere_radius so the arm is horizontal at rest.
        """
        z_hinge = self.SPHERE_RADIUS  # horizontal arm at rest

        mb = newton.ModelBuilder()
        sphere = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(float(lever), 0.0, self.SPHERE_RADIUS),
                q=wp.quat_identity(),
            ),
            mass=self.MASS,
        )
        # See comment in :meth:`_build_prismatic_press`: ``density=0``
        # keeps the shape from inflating the body's mass past ``MASS``.
        mb.add_shape_sphere(
            sphere,
            radius=self.SPHERE_RADIUS,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        def _factory(n2j, builder):
            # Saturated POSITION drive. ``target = pi/2`` is past the
            # ground (~0.7 m below the current bob z), and the spring
            # is stiff enough that the unclamped per-substep impulse
            # ``k * theta_err * dt^2`` far exceeds ``tau * dt``; the
            # drive saturates on every substep at ``tau * dt``,
            # applying a constant torque ``+tau`` about the hinge
            # axis in the direction that lowers the +x bob.
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[sphere],
                anchor1=(0.0, 0.0, z_hinge),
                anchor2=(0.0, 1.0, z_hinge),
                mode=JointMode.REVOLUTE,
                drive_mode=DriveMode.POSITION,
                target=float(math.pi / 2),  # positive theta about +y takes +x toward -z
                max_force_drive=float(tau),
                stiffness_drive=1.0e4,
                damping_drive=1.0e2,
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=True,
            expected_masses={sphere: self.MASS},
        )
        scene.sphere = sphere
        scene.lever = float(lever)
        return scene

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _run_to_steady_state(self, scene: _JitterScene) -> None:
        """Simulate ``SETTLE_FRAMES`` frames, then verify the sphere's
        linear velocity has damped out. If not, the subsequent
        force-balance check would be comparing ``F = m*a`` rather than
        ``F = m*g + drive``.
        """
        for _ in range(self.SETTLE_FRAMES):
            scene.step()
        vel = scene.world.bodies.velocity.numpy()[
            scene.newton_to_jitter[scene.sphere]
        ]
        v = float(np.linalg.norm(vel))
        self.assertLess(
            v,
            0.15,
            f"sphere still moving at end of settle: |v|={v:.3f} m/s "
            "-- increase SETTLE_FRAMES or check the drive saturates cleanly.",
        )

    def test_prismatic_motor_presses_into_ground(self):
        """Prismatic drive saturated pushing down: contact = m*g + F_motor."""
        F_motor = 10.0  # Newtons
        scene = self._build_prismatic_press(F_motor=F_motor)
        self._run_to_steady_state(scene)

        F, npairs = scene.gather_contact_wrench_on_body(scene.sphere)
        self.assertEqual(
            npairs,
            1,
            f"expected exactly one sphere-ground contact pair, got {npairs}",
        )
        weight = self.MASS * _G
        expected_Fz = weight + F_motor
        # Contact force on sphere should be purely ``+z`` (normal
        # pointing up). Assert magnitude and direction separately.
        self.assertGreater(
            float(F[2]),
            0.0,
            f"sphere is being pushed into ground but contact Fz is not "
            f"upward: F={F}",
        )
        rel_err = abs(float(F[2]) - expected_Fz) / expected_Fz
        self.assertLess(
            rel_err,
            0.05,
            f"prismatic contact force off: measured Fz={float(F[2]):.3f} N, "
            f"expected m*g + F_motor = {expected_Fz:.3f} N "
            f"(m*g={weight:.3f} N, F_motor={F_motor:.3f} N, "
            f"rel_err={rel_err:.2%})",
        )
        # Lateral components must be negligible for a pure-axis drive.
        lateral = math.sqrt(float(F[0]) ** 2 + float(F[1]) ** 2)
        self.assertLess(
            lateral,
            0.05 * expected_Fz,
            f"prismatic contact has spurious lateral force: "
            f"|F_xy|={lateral:.3f} N, F_z={float(F[2]):.3f} N",
        )

    def test_revolute_motor_presses_into_ground(self):
        """Revolute torque ``tau`` on a lever-``L`` horizontal pendulum:
        contact = m*g + tau/L.
        """
        lever = 0.8
        tau = 6.0  # N*m. tau/L = 7.5 N, ~38% of the m*g = 19.62 N weight.
        scene = self._build_revolute_press(tau=tau, lever=lever)
        self._run_to_steady_state(scene)

        F, npairs = scene.gather_contact_wrench_on_body(scene.sphere)
        self.assertEqual(
            npairs,
            1,
            f"expected exactly one sphere-ground contact pair, got {npairs}",
        )
        weight = self.MASS * _G
        expected_Fz = weight + tau / lever
        self.assertGreater(
            float(F[2]),
            0.0,
            f"revolute motor should press bob down but contact Fz is not "
            f"upward: F={F}",
        )
        rel_err = abs(float(F[2]) - expected_Fz) / expected_Fz
        self.assertLess(
            rel_err,
            0.05,
            f"revolute contact force off: measured Fz={float(F[2]):.3f} N, "
            f"expected m*g + tau/L = {expected_Fz:.3f} N "
            f"(m*g={weight:.3f} N, tau/L={tau / lever:.3f} N, "
            f"rel_err={rel_err:.2%})",
        )


# ---------------------------------------------------------------------------
# Limit spring-damper tests (no drive, no contact; just the limit row
# acting as a Hookean spring against a gravity load)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestLimitSpringDamper(unittest.TestCase):
    """Equilibrium tests for the limit row configured as a PD spring-
    damper (``stiffness_limit`` / ``damping_limit`` branch). No drive,
    no contacts -- the
    entire force balance is carried by the limit row, so the measured
    deflection maps directly onto the spring stiffness.

    We keep the ranges *tiny* (``1e-6``) so the limit is effectively
    "always clamped" and the constraint degenerates to a pure spring
    about the limit line. A small amount of damping bleeds the initial
    oscillation so we can read a steady-state deflection.
    """

    MASS = 1.0
    BOB_RADIUS = 0.02

    FPS = 240
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16

    # Long enough to let the damper ring down to ~1 mm / 1 mrad at the
    # PD gains chosen below. Computed informally from the decay rate
    # ``gamma = c / (2 * m_eff)`` + a few oscillation periods.
    SETTLE_FRAMES = 2400  # 10 s @ 240 fps

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    # ------------------------------------------------------------------
    # Scene builders
    # ------------------------------------------------------------------

    def _build_prismatic_limit_spring(
        self,
        *,
        stiffness: float,
        damping: float,
    ) -> _JitterScene:
        """Vertical prismatic joint, sphere dangling, limit row set to
        a PD spring-damper at a tiny range around zero.

        Geometry:
            * anchor1 = (0, 0, 0) on the world, anchor2 = (0, 0, -1) so
              the slide axis ``n_hat`` points along ``-z`` (downward).
            * Sphere body starts at ``(0, 0, 0)`` -- right at anchor1.
              At ``t = 0`` ``slide = n_hat . (p2 - p1) = 0``.
            * Gravity ``-z`` accelerates the sphere along ``+n_hat``,
              increasing ``slide``. The limit's upper bound is a small
              positive number; as soon as ``slide`` crosses it the
              CLAMP_MAX branch activates and the spring pulls back.

        With a one-sided tiny-range limit at ``max_value = max_slack``
        and no other force on the axial DoF, the closed-form
        equilibrium is

        .. math::
            k \\, (\\text{slide}_{eq} - \\text{max\\_slack}) = m g
            \\;\\Longrightarrow\\;
            \\Delta = \\text{slide}_{eq} - \\text{max\\_slack}
                   = \\frac{m g}{k}.

        We pass ``stiffness`` as ``stiffness_limit`` (the PD branch of
        the dual convention -- see :class:`JointDescriptor`) and
        ``damping`` as ``damping_limit``.
        """
        # Tiny slack so the limit is effectively always active once
        # the sphere starts to sag. ``min_value < max_value`` so the
        # limit row is enabled (``min_value > max_value`` disables it).
        self.MAX_SLACK = 1.0e-6

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 0.0, -1.0),  # slide axis +n_hat = -z (down)
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.OFF,
                min_value=-self.MAX_SLACK,
                max_value=self.MAX_SLACK,
                # Dual-convention PD branch: ``stiffness_limit`` is the
                # absolute stiffness ``kp`` [N/m] and ``damping_limit``
                # is the absolute damping ``kd`` [N*s/m].
                stiffness_limit=float(stiffness),
                damping_limit=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,  # No shapes besides the massless bob marker.
        )
        scene.bob = bob
        return scene

    def _build_revolute_limit_torsion_spring(
        self,
        *,
        lever: float,
        stiffness: float,
        damping: float,
    ) -> _JitterScene:
        """Horizontal revolute joint, sphere on a lever orthogonal to
        gravity. The limit row acts as a torsional PD spring-damper
        around zero; gravity pulls the bob down, twisting the hinge.

        Geometry:
            * Hinge at the origin with axis ``+y``. In unified-joint
              terms: ``anchor1 = (0, 0, 0)``, ``anchor2 = (0, 1, 0)``.
            * Sphere at ``(+lever, 0, 0)`` so the arm is orthogonal to
              gravity at rest. Gravity ``-z`` torques the arm about
              ``+y``: :math:`\\tau = m g L \\cos\\theta`.
            * Limit range is a tiny-slack window around zero with PD
              stiffness ``k`` [N*m/rad]. Nonlinear equilibrium:

              .. math::
                 k \\, (\\theta_{eq} - \\text{max\\_slack})
                    = m g L \\cos\\theta_{eq}

              which for a stiff-enough spring reduces to the linear
              estimate :math:`\\theta_{eq} \\approx m g L / k`. We test
              against the full nonlinear identity (iteratively solved
              once below) so the test doesn't break when the gains are
              at the edge of the small-angle regime.
        """
        self.MAX_SLACK = 1.0e-6

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(float(lever), 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 1.0, 0.0),  # hinge axis +y
                mode=JointMode.REVOLUTE,
                drive_mode=DriveMode.OFF,
                min_value=-self.MAX_SLACK,
                max_value=self.MAX_SLACK,
                stiffness_limit=float(stiffness),
                damping_limit=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,
        )
        scene.bob = bob
        scene.lever = float(lever)
        return scene

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _run_to_steady_state(
        self, scene: _JitterScene, tag: str, *, max_speed: float, max_spin: float
    ) -> None:
        """Step for ``SETTLE_FRAMES`` frames, then assert the bob is at
        rest. If the damper didn't ring the motion down we can't trust
        the deflection readout -- the assertion message funnels failure
        diagnosis rather than a mysterious ``F != m*g`` later.
        """
        for _ in range(self.SETTLE_FRAMES):
            scene.step()
        v = scene.world.bodies.velocity.numpy()[
            scene.newton_to_jitter[scene.bob]
        ]
        w = scene.world.bodies.angular_velocity.numpy()[
            scene.newton_to_jitter[scene.bob]
        ]
        speed = float(np.linalg.norm(v))
        spin = float(np.linalg.norm(w))
        self.assertLess(
            speed,
            max_speed,
            f"[{tag}] bob still translating at end of settle: |v|={speed:.4f} "
            "m/s -- increase SETTLE_FRAMES or damping.",
        )
        self.assertLess(
            spin,
            max_spin,
            f"[{tag}] bob still rotating at end of settle: |w|={spin:.4f} "
            "rad/s -- increase SETTLE_FRAMES or damping.",
        )

    def test_prismatic_limit_spring_holds_weight(self):
        """Vertical prismatic slider with spring-damper limits: the
        sphere must sag by exactly ``m * g / k`` (Hooke's law)
        relative to the upper limit line.
        """
        stiffness = 100.0  # [N/m]
        # Damping ratio ~1 for unit mass: c = 2 * sqrt(k * m) ~= 20.
        # A bit less so the decay is visible in ``SETTLE_FRAMES`` without
        # being critically damped (which also tests the damper contributes
        # to the PGS softness but not to the equilibrium deflection).
        damping = 6.0  # [N*s/m]

        scene = self._build_prismatic_limit_spring(
            stiffness=stiffness, damping=damping
        )
        # Tight thresholds: translation velocity is the direct observable
        # here; a residual > 2 mm/s would bias the ``m*g/k`` deflection
        # check by more than the 5% tolerance.
        self._run_to_steady_state(
            scene, tag="prismatic-limit-spring", max_speed=0.02, max_spin=0.05
        )

        # ``slide = n_hat . (p2 - p1)`` with ``n_hat = -z``. After settle
        # the bob has sagged by ``m*g/k`` below the origin, i.e.
        # ``p2.z = -m*g/k``, so ``slide = +m*g/k`` (in the slide
        # coordinate). The deflection past the upper limit line is
        # ``slide - MAX_SLACK``.
        pos = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[scene.bob]
        ]
        # Slide is the projection onto n_hat = (0, 0, -1).
        slide = -float(pos[2])  # world body at origin, so p2 - p1 = (0,0,z)
        deflection = slide - self.MAX_SLACK
        expected = self.MASS * _G / stiffness
        rel_err = abs(deflection - expected) / expected
        self.assertLess(
            rel_err,
            0.05,
            f"prismatic limit spring deflection off: measured "
            f"{deflection:.6f} m, expected m*g/k = {expected:.6f} m "
            f"(m={self.MASS} kg, g={_G:.3f} m/s^2, k={stiffness} N/m, "
            f"rel_err={rel_err:.2%})",
        )

        # Lateral motion must be tiny -- the rank-5 prismatic lock
        # should clamp every non-axial DoF hard. This is belt-and-
        # braces: a sign or axis bug in the limit row could easily
        # leak into a tangent direction.
        self.assertLess(
            float(abs(pos[0]) + abs(pos[1])),
            1.0e-3,
            f"bob drifted laterally: p = {pos}",
        )

    def test_revolute_limit_torsion_spring_holds_lever(self):
        """Horizontal revolute hinge with torsional spring-damper
        limits: a sphere on a lever orthogonal to gravity sags by
        the angle that solves :math:`k \\theta = m g L \\cos\\theta`.
        """
        lever = 0.5  # [m] -- arm length between hinge and sphere centre
        # Gains chosen so the small-angle deflection ``m*g*L/k`` is
        # about 0.05 rad (~3 deg): comfortably in the linear regime,
        # well above measurement noise, and the ``cos`` correction is
        # <0.1% so linear / nonlinear agree to within the PGS residual.
        stiffness = 100.0  # [N*m/rad]
        # Torsional m_eff = m * L^2 = 0.25 kg*m^2; critical damping
        # c_crit = 2 * sqrt(k * m_eff) = 10 N*m*s/rad. We pick 40 (4x
        # over-damped) because under Jitter's PGS CFM formulation the
        # per-substep decay is slower than the continuous-time analysis
        # predicts (the softness leaks impulse at a rate set by the
        # softness-scaled m_eff, not the raw one). The equilibrium
        # deflection ``k*theta = m*g*L*cos(theta)`` is damping-
        # independent so this over-damping doesn't bias the test.
        damping = 40.0  # [N*m*s/rad]

        scene = self._build_revolute_limit_torsion_spring(
            lever=lever,
            stiffness=stiffness,
            damping=damping,
        )
        # The bob moves along an arc so its translation speed is
        # |w| * L; our torsional residual target is |w| < 0.05 rad/s,
        # which lifts the linear-speed cap to ~|w|*L = 0.025 m/s.
        self._run_to_steady_state(
            scene,
            tag="revolute-limit-torsion",
            max_speed=0.05 * lever + 0.02,
            max_spin=0.05,
        )

        # Read the bob's sag angle about the hinge (+y axis) from its
        # world-space position. Arm was initially along +x at
        # ``(L, 0, 0)``; after sagging by +theta about +y the bob sits
        # at ``(L*cos(theta), 0, -L*sin(theta))``.
        pos = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[scene.bob]
        ]
        # atan2 of (-z component, +x component) recovers theta in a way
        # that is well-defined across the small-angle regime.
        theta = math.atan2(-float(pos[2]), float(pos[0]))

        # Nonlinear equilibrium: k*(theta - MAX_SLACK) = m*g*L*cos(theta).
        # Solve with a few Newton iterations starting from the linear
        # guess m*g*L/k.
        mgL = self.MASS * _G * lever
        theta_expected = mgL / stiffness  # linear starting guess
        for _ in range(8):
            f = stiffness * (theta_expected - self.MAX_SLACK) - mgL * math.cos(
                theta_expected
            )
            fp = stiffness + mgL * math.sin(theta_expected)
            theta_expected -= f / fp

        rel_err = abs(theta - theta_expected) / theta_expected
        self.assertLess(
            rel_err,
            0.05,
            f"revolute limit torsion-spring deflection off: measured "
            f"theta={theta:.6f} rad, expected {theta_expected:.6f} rad "
            f"(m*g*L={mgL:.4f} N*m, k={stiffness} N*m/rad, lever={lever} m, "
            f"rel_err={rel_err:.2%})",
        )

        # Out-of-plane drift must be tiny -- the hinge's rank-5 lock
        # should hold every DoF but the axial twist.
        self.assertLess(
            abs(float(pos[1])),
            1.0e-3,
            f"bob drifted along hinge axis: p = {pos}",
        )


# ---------------------------------------------------------------------------
# Helper: unit cube as a Newton body with explicit (analytical) inertia
# ---------------------------------------------------------------------------


def _add_unit_cube(
    mb: newton.ModelBuilder,
    *,
    position: tuple[float, float, float],
    half_extent: float = 0.5,
    mass: float = 1.0,
) -> int:
    """Add a unit cube body with a *closed-form* inertia tensor and a
    massless box shape.

    Same trick as :func:`_add_pendulum_bob`: pass the analytical
    ``I = (m / 12) * (h^2 + h^2) = m * h^2 / 6`` for a cube of side
    ``2*half_extent`` to :meth:`add_body`, then the visual / collision
    box uses ``density=0`` so the shape adds nothing to mass or
    inertia. This keeps the body's inertia exactly what the
    analytical equilibrium predicts -- no surprise factors from
    Newton's shape-density stack.
    """
    side = 2.0 * float(half_extent)
    I_scalar = float(mass) * (side * side + side * side) / 12.0
    inertia = wp.mat33(
        I_scalar, 0.0, 0.0,
        0.0, I_scalar, 0.0,
        0.0, 0.0, I_scalar,
    )
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(*position), q=wp.quat_identity()),
        mass=float(mass),
        inertia=inertia,
    )
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)
    mb.add_shape_box(
        body,
        hx=float(half_extent),
        hy=float(half_extent),
        hz=float(half_extent),
        cfg=massless,
    )
    return body


# ---------------------------------------------------------------------------
# Tiny-range prismatic spring-damper limit with a unit cube
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestPrismaticUnitCubeLimitSpring(unittest.TestCase):
    """A unit cube hung from a vertical prismatic slider whose limit
    row is configured as a spring-damper. The static deflection
    relative to the upper limit line must equal ``m * g / k`` --
    Hooke's law on a body whose mass and inertia come straight from
    Newton's :class:`ModelBuilder` rather than being hand-rolled in
    the test. Sweeps a couple of stiffness values so a constant-bias
    leak (e.g. PGS softness scaled wrong) shows up as a *trend* in
    the residuals, not just a single off-by-X result.
    """

    HALF_EXTENT = 0.5  # 1 m unit cube
    MASS = 1.0  # [kg]
    MAX_SLACK = 1.0e-6  # tiny window so the limit is "always on"

    FPS = 240
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16
    SETTLE_FRAMES = 1800  # 7.5 s @ 240 fps -- enough at the chosen damping.

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    def _build_scene(
        self, *, stiffness: float, damping: float
    ) -> _JitterScene:
        """Vertical prismatic slider, unit cube starts at the origin
        coincident with anchor1; gravity pulls the cube along ``+slide``
        until it hits the spring-damper limit at ``+MAX_SLACK``.
        """
        mb = newton.ModelBuilder()
        cube = _add_unit_cube(
            mb,
            position=(0.0, 0.0, 0.0),
            half_extent=self.HALF_EXTENT,
            mass=self.MASS,
        )

        def _factory(n2j, builder):
            # ``anchor2 = anchor1 + (0, 0, -1)`` puts the slide axis
            # ``n_hat`` along ``-z``: gravity then loads the spring in
            # the ``+slide`` direction (slide grows positive).
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[cube],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 0.0, -1.0),
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.OFF,
                min_value=-self.MAX_SLACK,
                max_value=self.MAX_SLACK,
                # PD branch: stiffness_limit -> kp, damping_limit -> kd.
                stiffness_limit=float(stiffness),
                damping_limit=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,
            expected_masses={cube: self.MASS},
        )
        scene.cube = cube
        return scene

    def _assert_static_deflection(
        self, *, stiffness: float, damping: float, label: str
    ) -> None:
        scene = self._build_scene(stiffness=stiffness, damping=damping)
        for _ in range(self.SETTLE_FRAMES):
            scene.step()

        # Verify the cube has actually come to rest -- an
        # under-damped scene at readout time reads back instantaneous
        # position, not the equilibrium.
        v = scene.world.bodies.velocity.numpy()[
            scene.newton_to_jitter[scene.cube]
        ]
        speed = float(np.linalg.norm(v))
        self.assertLess(
            speed,
            0.005,
            f"[{label}] cube still translating at end of settle: "
            f"|v|={speed:.4f} m/s -- bump SETTLE_FRAMES or damping.",
        )

        # ``slide = n_hat . (p2 - p1) = -p2.z`` since p1 = (0,0,0) and
        # n_hat = -z. Steady-state slide above the upper limit line is
        # the deflection we test.
        pos = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[scene.cube]
        ]
        slide = -float(pos[2])
        deflection = slide - self.MAX_SLACK
        expected = self.MASS * _G / stiffness
        rel_err = abs(deflection - expected) / expected
        self.assertLess(
            rel_err,
            0.05,
            f"[{label}] unit-cube prismatic spring-limit deflection off: "
            f"measured {deflection:.6f} m, expected m*g/k = {expected:.6f} m "
            f"(m={self.MASS} kg, g={_G:.3f} m/s^2, k={stiffness} N/m, "
            f"rel_err={rel_err:.2%})",
        )

        # Lateral lock must hold to within a millimetre.
        self.assertLess(
            float(abs(pos[0]) + abs(pos[1])),
            1.0e-3,
            f"[{label}] cube drifted laterally: p = {pos}",
        )

    def test_unit_cube_sag_against_spring_limit(self):
        """Soft spring (k=50) -- expected deflection ~0.196 m, big
        enough to be well above any PGS residual noise floor.
        """
        # c = 2 * sqrt(k * m) = 2 * sqrt(50) ~= 14.1 -- critical for unit
        # mass. We take a fraction of that so the underdamped settle is
        # visible in SETTLE_FRAMES; the equilibrium itself is
        # damping-independent.
        self._assert_static_deflection(
            stiffness=50.0, damping=8.0, label="k=50"
        )

    def test_unit_cube_sag_against_stiff_spring_limit(self):
        """Stiff spring (k=500) -- expected deflection ~0.0196 m.
        Same identity, but now the absolute deflection is small
        enough that any constant additive bias (e.g. baumgarte
        leakage from the rank-5 lock) would dominate the relative
        error -- catches a different failure mode than the soft case.
        """
        # Critical c = 2*sqrt(500) ~= 44.7; pick 30 for visible underdamped settle.
        self._assert_static_deflection(
            stiffness=500.0, damping=30.0, label="k=500"
        )


# ---------------------------------------------------------------------------
# Damped-harmonic-oscillator identity on the limit row
# ---------------------------------------------------------------------------


def _measure_oscillation(
    samples: np.ndarray,
    *,
    dt: float,
) -> tuple[float, float]:
    """Return ``(omega_d, gamma)`` from a free underdamped time series.

    Uses zero-crossings for the damped angular frequency
    :math:`\\omega_d = 2\\pi / T_d` and a log-linear fit on the
    successive-peak amplitudes for the decay rate
    :math:`\\gamma`. Both are timestep-independent (in the limit
    ``dt -> 0``) -- they characterise the continuous-time
    differential equation, not the discrete solver.

    Caller is responsible for picking a time series that is
    centred on the equilibrium (so zero-crossings are meaningful)
    and contains at least 4 oscillations (so the linear fit on the
    log-amplitudes has > 2 points).
    """
    n = samples.shape[0]
    if n < 8:
        raise ValueError(
            f"need >= 8 samples for oscillation fit, got {n}"
        )

    # Zero crossings via sign change.
    sign = np.sign(samples)
    crossings = []
    for i in range(1, n):
        if sign[i - 1] != 0 and sign[i] != 0 and sign[i] != sign[i - 1]:
            # Linear interpolate the crossing time between samples
            # i-1 and i for sub-step precision -- aliasing on a coarse
            # grid biases the period downward otherwise.
            t0 = (i - 1) * dt
            x0 = samples[i - 1]
            x1 = samples[i]
            frac = x0 / (x0 - x1)
            crossings.append(t0 + frac * dt)
    if len(crossings) < 3:
        raise ValueError(
            f"oscillation has too few zero-crossings ({len(crossings)}) -- "
            "either heavily over-damped or sample window too short"
        )
    # Period = twice the average inter-crossing interval (each
    # crossing is half a period apart).
    diffs = np.diff(crossings)
    period = 2.0 * float(np.mean(diffs))
    omega_d = 2.0 * math.pi / period

    # Peaks via local maxima of |x|.
    peak_t = []
    peak_v = []
    for i in range(1, n - 1):
        a = abs(samples[i - 1])
        b = abs(samples[i])
        c = abs(samples[i + 1])
        if b > a and b >= c and b > 1.0e-8:
            peak_t.append(i * dt)
            peak_v.append(b)
    if len(peak_t) < 3:
        raise ValueError(
            f"too few peaks ({len(peak_t)}) for log-linear decay fit"
        )
    # Fit log(|x_peak|) ~= log(A0) - gamma * t.
    log_peaks = np.log(np.array(peak_v))
    t_arr = np.array(peak_t)
    slope, _ = np.polyfit(t_arr, log_peaks, 1)
    gamma = -float(slope)

    return omega_d, gamma


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestLimitDampingDecay(unittest.TestCase):
    """Damped-harmonic-oscillator identity check on the
    spring-damper limit row (``stiffness_limit`` / ``damping_limit``
    PD branch).

    For an underdamped 1-DoF system :math:`m \\ddot x + c \\dot x +
    k x = 0` the universal identity

    .. math::
        \\omega_d^2 + \\gamma^2 \\;=\\; \\omega_0^2
            \\;=\\; \\tfrac{k}{m_{\\rm eff}}

    holds *exactly* in continuous time and is independent of the
    timestep. PGS-with-softness should preserve it to first order in
    ``dt``; a > 5% violation here tells us the implicit damping is
    biased (or the effective mass is wrong).

    These tests deliberately disable gravity so the equilibrium is
    at the limit boundary and the bob oscillates symmetrically about
    it -- much cleaner for the zero-crossing fit than the
    gravity-loaded case.
    """

    MASS = 1.0
    BOB_RADIUS = 0.02
    LEVER = 0.5  # used by the revolute version

    FPS = 480  # high rate so sub-step-accurate zero crossings are dense
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16
    SAMPLE_FRAMES = 1200  # 2.5 s @ 480 fps -- ~10 oscillations at the gains chosen

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    def _build_prismatic(
        self, *, stiffness: float, damping: float, v0: float
    ) -> _JitterScene:
        """Horizontal prismatic slider (no gravity), bob starts *at*
        the joint anchor (slide=0) with an initial velocity ``v0``
        along ``+x``. The bob immediately exits the tiny limit window
        and the limit row begins applying the spring-damper force.

        We can't simply position the bob at ``(x0, 0, 0)`` and call
        ``slide=x0``: the prismatic prepare reads
        ``slide = n_hat . (p1_b2 - p1_b1)`` where ``p1_b2`` follows the
        bob, and the body anchor offset is baked in at joint creation.
        At ``t=0`` the bob's anchor is *at* anchor1, so
        ``slide = 0`` regardless of where the bob's COM is. Using
        an initial velocity sidesteps this.
        """
        mb = newton.ModelBuilder()
        # Bob starts at the origin (slide = 0) and gets kicked by v0.
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(1.0, 0.0, 0.0),  # slide axis = +x
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.OFF,
                # Tiny window so the limit fires almost immediately
                # (within one substep at the chosen v0). ``min_value <
                # max_value`` so the limit row is enabled (the prepare
                # disables it when ``min_value > max_value``).
                min_value=-1.0e-6,
                max_value=1.0e-6,
                stiffness_limit=float(stiffness),
                damping_limit=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, 0.0),  # critical: no DC offset on the bob
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        scene.bob = bob
        # Apply the initial-velocity kick on the *Newton* state -- the
        # captured graph re-syncs Jitter from ``state.body_qd`` every
        # frame, so writing the velocity straight into Jitter's body
        # buffer would be clobbered by the next sync. Newton's
        # ``body_qd`` is a spatial vector laid out as ``[lin_x, lin_y,
        # lin_z, ang_x, ang_y, ang_z]`` (linear part first -- see the
        # ``newton_to_jitter_kernel`` in :mod:`example_jitter_common`).
        bqd = scene.state.body_qd.numpy()
        bqd[bob] = (float(v0), 0.0, 0.0, 0.0, 0.0, 0.0)
        scene.state.body_qd.assign(bqd)
        return scene

    def _build_revolute(
        self,
        *,
        stiffness: float,
        damping: float,
        omega0: float,
    ) -> _JitterScene:
        """Hinge axis +y, lever along +x. Bob is given an initial
        angular velocity ``omega0`` about +y. The bob starts inside
        the (zero-width) limit window so the twist coordinate is 0;
        the kick rotates it past the limit and the spring-damper
        begins acting. Gravity disabled.

        Same caveat as :meth:`_build_prismatic`: ``twist_err`` is the
        *relative* twist of the two body frames, baked in at joint
        creation time. We can't get a non-zero starting twist by
        repositioning/rotating the body alone (the joint records the
        starting transform).
        """
        L = self.LEVER
        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(L, 0.0, 0.0),  # arm along +x, twist = 0 at start
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 1.0, 0.0),  # hinge axis = +y
                mode=JointMode.REVOLUTE,
                drive_mode=DriveMode.OFF,
                min_value=-1.0e-6,
                max_value=1.0e-6,
                stiffness_limit=float(stiffness),
                damping_limit=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, 0.0),
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        scene.bob = bob
        # Kick the bob with angular velocity omega0 about +y, plus the
        # consistent linear velocity at the bob's COM (offset L along
        # +x from the hinge): ``v = omega x r = (0, omega, 0) x
        # (L, 0, 0) = (0, 0, -omega * L)``. Positive omega about +y
        # rotates the +x arm downward, so the bob's COM initially
        # moves in -z.
        # Write into Newton's ``state.body_qd`` (spatial vector
        # ``[lin, ang]`` -- linear part first) so the graph's
        # per-frame Newton->Jitter sync honours the kick.
        bqd = scene.state.body_qd.numpy()
        bqd[bob] = (
            0.0, 0.0, -float(omega0) * L,
            0.0, float(omega0), 0.0,
        )
        scene.state.body_qd.assign(bqd)
        return scene

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_prismatic_limit_damped_decay_identity(self):
        """Underdamped prismatic spring-damper limit must satisfy
        :math:`\\omega_d^2 + \\gamma^2 = k / m`.
        """
        # k=400 N/m, m=1 kg -> omega_0=20 rad/s, T_0 ~= 0.314 s.
        # Critical c=40; pick 4 (zeta=0.1) so ~10 oscillations are
        # well-resolved within SAMPLE_FRAMES and the log-amplitude fit
        # has a clean linear slope.
        k = 400.0
        c = 4.0
        # v0 = 1 m/s -> peak amplitude = v0 / omega_0 = 0.05 m at the
        # chosen gains. Plenty of signal-to-noise for the zero-crossing
        # and peak-amplitude fits.
        scene = self._build_prismatic(stiffness=k, damping=c, v0=1.0)

        samples = np.zeros(self.SAMPLE_FRAMES, dtype=np.float64)
        for f in range(self.SAMPLE_FRAMES):
            scene.step()
            pos = scene.world.bodies.position.numpy()[
                scene.newton_to_jitter[scene.bob]
            ]
            samples[f] = float(pos[0])

        omega_d, gamma = _measure_oscillation(samples, dt=1.0 / self.FPS)
        omega_0_sq = k / self.MASS
        sum_sq = omega_d * omega_d + gamma * gamma
        rel_err = abs(sum_sq - omega_0_sq) / omega_0_sq
        self.assertLess(
            rel_err,
            0.05,
            f"prismatic limit DHO identity off: omega_d^2 + gamma^2 = "
            f"{sum_sq:.4f}, expected k/m = {omega_0_sq:.4f} "
            f"(omega_d={omega_d:.4f} rad/s, gamma={gamma:.4f} 1/s, "
            f"rel_err={rel_err:.2%})",
        )
        # Sanity: gamma > 0 (decay, not growth) and omega_d > 0
        # (genuinely oscillating, not over-damped).
        self.assertGreater(
            gamma,
            0.0,
            f"prismatic limit shows amplitude growth (gamma={gamma:.4f}) "
            "-- damping has wrong sign",
        )
        self.assertGreater(
            omega_d, 0.0, "prismatic limit didn't oscillate"
        )

    def test_revolute_limit_damped_decay_identity(self):
        """Underdamped torsional spring-damper limit on a revolute
        hinge with a lever-bob must satisfy
        :math:`\\omega_d^2 + \\gamma^2 = k_\\theta / I_{\\rm eff}` with
        :math:`I_{\\rm eff} = m L^2`.
        """
        # k_theta = 100 N*m/rad, I = m*L^2 = 0.25 kg*m^2 -> omega_0 = 20 rad/s.
        # Critical c_theta = 2*sqrt(k*I) = 10 N*m*s/rad; pick 1
        # (zeta = 0.1) so ~10 oscillations resolve cleanly.
        k_theta = 100.0
        c_theta = 1.0
        I_eff = self.MASS * self.LEVER * self.LEVER
        # omega0 = 1 rad/s -> peak twist amplitude = omega0/omega_0
        # = 0.05 rad at the chosen gains.
        scene = self._build_revolute(
            stiffness=k_theta, damping=c_theta, omega0=1.0
        )

        samples = np.zeros(self.SAMPLE_FRAMES, dtype=np.float64)
        for f in range(self.SAMPLE_FRAMES):
            scene.step()
            pos = scene.world.bodies.position.numpy()[
                scene.newton_to_jitter[scene.bob]
            ]
            # theta from arm orientation: arm starts along +x, sags by
            # +theta about +y means bob at (L*cos, 0, -L*sin).
            theta = math.atan2(-float(pos[2]), float(pos[0]))
            samples[f] = theta

        omega_d, gamma = _measure_oscillation(samples, dt=1.0 / self.FPS)
        omega_0_sq = k_theta / I_eff
        sum_sq = omega_d * omega_d + gamma * gamma
        rel_err = abs(sum_sq - omega_0_sq) / omega_0_sq
        self.assertLess(
            rel_err,
            0.05,
            f"revolute limit DHO identity off: omega_d^2 + gamma^2 = "
            f"{sum_sq:.4f}, expected k/I = {omega_0_sq:.4f} "
            f"(omega_d={omega_d:.4f} rad/s, gamma={gamma:.4f} 1/s, "
            f"k_theta={k_theta} N*m/rad, I_eff={I_eff:.4f} kg*m^2, "
            f"rel_err={rel_err:.2%})",
        )
        self.assertGreater(
            gamma,
            0.0,
            f"revolute limit shows amplitude growth (gamma={gamma:.4f}) "
            "-- torsional damping has wrong sign",
        )


# ---------------------------------------------------------------------------
# One-sided limits: one bound is "wide open", the other clamps
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestOneSidedLimits(unittest.TestCase):
    """Verify that a one-sided limit (one bound active, the other
    set far enough away to never trigger) really only acts on the
    intended side. The constraint kernel guards the limit row with
    ``if min_value <= max_value`` (enabled), then independently
    checks ``slide > max_value`` / ``slide < min_value`` so the
    "inactive" side just falls through to ``clamp = NONE`` and the
    iterate's ``if clamp != _CLAMP_NONE:`` skips the impulse
    entirely. We test that fall-through end-to-end.
    """

    MASS = 1.0
    BOB_RADIUS = 0.02

    # The revolute one-sided variant converges *much* slower than the
    # prismatic ones because the gravity load on the lever bob couples
    # through the rigid linear anchor rows -- the PD limit row "sees"
    # an effective inertia that is the lever-arm-amplified bob inertia
    # (m * L^2), and the resulting per-substep CFM relaxation is small.
    # 8 substeps + 32 PGS iterations + a long ~12 s settle is what's
    # needed to hit the ~0.8% angular equilibrium target consistently;
    # the prismatic variants are still fine at these settings (just
    # slightly slower) and we keep one knob to avoid surprises later.
    FPS = 240
    SUBSTEPS = 8
    SOLVER_ITERATIONS = 32
    SETTLE_FRAMES = 3000  # 12.5 s -- both halves settle comfortably.

    # "Wide-open" bound -- far enough that the bob can never reach it
    # at the gravity loads we use here. Non-zero so the limit row
    # stays enabled (the prepare's gate is ``min != 0 OR max != 0``).
    WIDE_BOUND = 100.0

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    def _build_prismatic(
        self,
        *,
        stiffness: float,
        damping: float,
        min_value: float,
        max_value: float,
        slide_axis_sign: int,
    ) -> _JitterScene:
        """Vertical prismatic slider. ``slide_axis_sign`` = -1 places
        the slide axis along ``-z`` (gravity loads ``+slide``), ``+1``
        along ``+z`` (gravity loads ``-slide``).
        """
        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            anchor2_z = -1.0 if slide_axis_sign < 0 else 1.0
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 0.0, anchor2_z),
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.OFF,
                min_value=float(min_value),
                max_value=float(max_value),
                stiffness_limit=float(stiffness),
                damping_limit=float(damping),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        scene.bob = bob
        scene.slide_axis_sign = int(slide_axis_sign)
        return scene

    def _slide_value(self, scene: _JitterScene) -> float:
        """Return the prismatic slide coordinate for ``scene.bob``,
        following the sign convention of the prismatic joint
        (``slide = n_hat . (p2 - p1)``).

        We always set ``anchor1=(0,0,0)`` so ``p1 = (0,0,0)``; then
        ``slide = n_hat . p2``. With ``n_hat`` along ``slide_axis_sign
        * +z``, this collapses to ``slide_axis_sign * pos.z``.
        """
        pos = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[scene.bob]
        ]
        return float(pos[2]) * scene.slide_axis_sign

    def test_prismatic_max_only_blocks_positive_slide(self):
        """``min`` is wide open, ``max`` is the active spring stop.
        Gravity drives slide positive -> spring engages, equilibrium
        at ``slide = max_value + m*g/k``. The "min" side never fires.
        """
        k = 100.0
        c = 8.0
        max_value = 0.10
        scene = self._build_prismatic(
            stiffness=k,
            damping=c,
            min_value=-self.WIDE_BOUND,
            max_value=max_value,
            slide_axis_sign=-1,  # n_hat = -z, gravity loads +slide
        )
        for _ in range(self.SETTLE_FRAMES):
            scene.step()
        slide = self._slide_value(scene)
        expected = max_value + self.MASS * _G / k
        rel_err = abs(slide - expected) / expected
        self.assertLess(
            rel_err,
            0.05,
            f"max-only prismatic limit equilibrium off: slide={slide:.5f} m, "
            f"expected max + m*g/k = {expected:.5f} m "
            f"(max={max_value} m, m*g/k={self.MASS * _G / k:.5f} m, "
            f"rel_err={rel_err:.2%})",
        )

    def test_prismatic_min_only_blocks_negative_slide(self):
        """``max`` is wide open, ``min`` is the active spring stop.
        Flip the slide axis so gravity loads ``-slide`` and the bob
        sags into the *min* limit. Equilibrium at
        ``slide = min_value - m*g/k`` (negative side).
        """
        k = 100.0
        c = 8.0
        min_value = -0.10
        scene = self._build_prismatic(
            stiffness=k,
            damping=c,
            min_value=min_value,
            max_value=self.WIDE_BOUND,
            slide_axis_sign=+1,  # n_hat = +z, gravity loads -slide
        )
        for _ in range(self.SETTLE_FRAMES):
            scene.step()
        slide = self._slide_value(scene)
        expected = min_value - self.MASS * _G / k
        rel_err = abs(slide - expected) / abs(expected)
        self.assertLess(
            rel_err,
            0.05,
            f"min-only prismatic limit equilibrium off: slide={slide:.5f} m, "
            f"expected min - m*g/k = {expected:.5f} m "
            f"(min={min_value} m, m*g/k={self.MASS * _G / k:.5f} m, "
            f"rel_err={rel_err:.2%})",
        )

    def test_prismatic_one_sided_allows_free_motion_in_open_direction(self):
        """No-gravity sanity check: with ``min`` wide open and ``max``
        at zero, an initial ``-x`` velocity must produce *unimpeded*
        free motion (no spring force from the inactive ``min`` side).
        Catches an "always-on" limit that wrongly fires regardless of
        which side ``slide`` is on.
        """
        k = 100.0
        c = 1.0
        max_value = 1.0e-6
        # Build a horizontal slider, no gravity, kick the bob in -slide.
        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            handle = builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(1.0, 0.0, 0.0),  # slide axis = +x
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.OFF,
                min_value=-self.WIDE_BOUND,
                max_value=max_value,
                stiffness_limit=float(k),
                damping_limit=float(c),
                hertz=0.0,
            )
            return handle

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, 0.0),
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        scene.bob = bob

        # Set initial velocity -1 m/s along +x (so slide goes negative,
        # which is the open side). Write to Newton's ``state.body_qd``
        # (spatial vector layout ``[lin, ang]`` -- linear first) --
        # writing straight to Jitter's body buffer is clobbered by the
        # per-frame Newton -> Jitter sync the captured graph performs.
        bqd = scene.state.body_qd.numpy()
        bqd[bob] = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scene.state.body_qd.assign(bqd)

        # Run for a short time -- short enough that the bob hasn't
        # crossed the wide bound (10 m at 1 m/s = 10 s; we'll use 1s).
        n_frames = self.FPS  # 1 s
        for _ in range(n_frames):
            scene.step()

        pos = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[bob]
        ]
        vel = scene.world.bodies.velocity.numpy()[
            scene.newton_to_jitter[bob]
        ]
        # Free-flight expectation: x = x0 + v0 * t = -1 m * 1 s = -1 m,
        # vx = -1 m/s (no force).
        self.assertAlmostEqual(
            float(pos[0]),
            -1.0,
            delta=0.01,
            msg=f"open-side free flight broken: x={pos[0]:.4f} m, expected ~-1.0 m. "
            "The 'min' limit must be inactive at slide < 0 with min=-WIDE_BOUND.",
        )
        self.assertAlmostEqual(
            float(vel[0]),
            -1.0,
            delta=0.01,
            msg=f"open-side velocity bled by inactive limit: vx={vel[0]:.4f} m/s, "
            "expected -1.0 m/s -- damping leaked from the wide-open side.",
        )

    def test_revolute_max_only_blocks_positive_angle(self):
        """One-sided revolute limit: pendulum allowed to swing into
        the open side, sprung back when crossing the upper angle
        limit. We pick a *small* upper bound so gravity engages it
        from rest (bob starts above the bound), then check
        equilibrium ``k * (theta_eq - max) = m*g*L*cos(theta_eq)``.
        """
        L = 0.5
        k_theta = 100.0  # N*m/rad
        c_theta = 5.0  # N*m*s/rad
        max_angle = 0.02  # rad -- small enough that gravity engages it

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(L, 0.0, 0.0),  # arm along +x at rest
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 1.0, 0.0),  # hinge axis = +y
                mode=JointMode.REVOLUTE,
                drive_mode=DriveMode.OFF,
                min_value=-self.WIDE_BOUND,  # open: huge negative
                max_value=max_angle,
                stiffness_limit=float(k_theta),
                damping_limit=float(c_theta),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        scene.bob = bob
        # Lots of damping needed because the bob has rotational
        # inertia ~ mL^2 = 0.25 and the limit is only nudged a tiny
        # angular distance: long settle ahead.
        for _ in range(3 * self.SETTLE_FRAMES):
            scene.step()

        pos = scene.world.bodies.position.numpy()[
            scene.newton_to_jitter[bob]
        ]
        theta = math.atan2(-float(pos[2]), float(pos[0]))

        # Nonlinear equilibrium: k*(theta - max) = m*g*L*cos(theta).
        mgL = self.MASS * _G * L
        theta_expected = max_angle + mgL / k_theta  # linear guess
        for _ in range(8):
            f = k_theta * (theta_expected - max_angle) - mgL * math.cos(
                theta_expected
            )
            fp = k_theta + mgL * math.sin(theta_expected)
            theta_expected -= f / fp

        rel_err = abs(theta - theta_expected) / theta_expected
        self.assertLess(
            rel_err,
            0.10,  # one-sided angular case is harder to settle exactly
            f"max-only revolute limit equilibrium off: theta={theta:.5f} rad, "
            f"expected {theta_expected:.5f} rad "
            f"(max={max_angle} rad, m*g*L/k={mgL / k_theta:.5f} rad, "
            f"rel_err={rel_err:.2%})",
        )


# ---------------------------------------------------------------------------
# Direct PD-drive impulse measurement (contact-free, gravity-free)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestPDDriveImpulse(unittest.TestCase):
    """Direct verification that the PD drive applies the impulse the
    math says it should -- no contacts, no gravity, no shape-density
    surprises. The bob is loaded by a single, known external force
    or torque (applied through a kinematic counter-anchor body or by
    setting an initial velocity); the steady-state response inverts
    to the gain. This isolates the drive row's math from every other
    source of impulse in the scene.

    The simplest "external load" we can apply without contacts or
    gravity is to set the bob's initial velocity and let the drive
    bring it to rest -- the steady-state error is then a pure
    function of the drive gains and any modelling residual.
    """

    MASS = 1.0
    BOB_RADIUS = 0.02
    LEVER = 0.5

    FPS = 480
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16
    SETTLE_FRAMES = 2400  # 5 s -- enough for most underdamped settles.

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest(
                "jitter physics tests require CUDA (graph capture)"
            )
        cls.device = wp.get_device("cuda:0")

    # ------------------------------------------------------------------
    # Prismatic POSITION drive: free harmonic oscillator
    # ------------------------------------------------------------------

    def test_prismatic_position_drive_dho_identity(self):
        """Prismatic POSITION drive (``kp, kd``) excited by an initial
        velocity kick. The drive's position error is ``slide -
        target`` and ``slide`` is the relative offset *since joint
        creation*, so positioning the body ahead of construction is a
        no-op (see the same caveat in
        :meth:`TestLimitDampingDecay._build_prismatic`). With
        ``target=0`` and a velocity kick, the bob undergoes free
        underdamped oscillation about ``slide=0`` and the DHO identity
        :math:`\\omega_d^2 + \\gamma^2 = k_p / m` must hold.

        Cleanest possible PD-drive math test: contact-free,
        gravity-free, no other forces. Wrong sign or wrong gain
        on the drive row shows up immediately.
        """
        kp = 200.0  # N/m -> omega_0 = sqrt(kp/m) ~= 14.14 rad/s
        kd = 4.0  # N*s/m -- zeta ~= 0.14, ~7 oscillations resolve cleanly.
        v0 = 1.0  # initial kick along +x; peak amplitude ~= v0/omega_0 = 0.07 m

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),  # slide=0 at construction
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(1.0, 0.0, 0.0),  # slide axis = +x
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.POSITION,
                target=0.0,
                stiffness_drive=float(kp),
                damping_drive=float(kd),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, 0.0),
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        # Kick the bob via Newton's state.body_qd (spatial vec ``[lin, ang]``).
        bqd = scene.state.body_qd.numpy()
        bqd[bob] = (v0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scene.state.body_qd.assign(bqd)

        n_samples = 1500  # 3.1 s -- ~7 oscillations at omega_0 ~= 14 rad/s.
        samples = np.zeros(n_samples, dtype=np.float64)
        for f in range(n_samples):
            scene.step()
            pos = scene.world.bodies.position.numpy()[
                scene.newton_to_jitter[bob]
            ]
            samples[f] = float(pos[0])

        omega_d, gamma = _measure_oscillation(samples, dt=1.0 / self.FPS)
        omega_0_sq = kp / self.MASS
        sum_sq = omega_d * omega_d + gamma * gamma
        rel_err = abs(sum_sq - omega_0_sq) / omega_0_sq
        self.assertLess(
            rel_err,
            0.05,
            f"prismatic POSITION drive DHO identity off: "
            f"omega_d^2 + gamma^2 = {sum_sq:.4f}, expected k/m = "
            f"{omega_0_sq:.4f} (omega_d={omega_d:.4f} rad/s, "
            f"gamma={gamma:.4f} 1/s, rel_err={rel_err:.2%})",
        )
        self.assertGreater(
            gamma,
            0.0,
            f"prismatic POSITION drive amplitude grew (gamma={gamma:.4f}) "
            "-- damping has wrong sign",
        )

    # ------------------------------------------------------------------
    # Prismatic VELOCITY drive: exponential approach to v_target
    # ------------------------------------------------------------------

    def test_prismatic_velocity_drive_exponential_decay(self):
        """Pure VELOCITY drive (``stiffness_drive=0``,
        ``damping_drive=c``) brings ``v`` from a non-zero initial
        value down to ``target_velocity`` exponentially:
        :math:`v(t) - v^\\ast = (v_0 - v^\\ast) \\, e^{-c t / m}`.

        Fit the decay rate from the time series and check it equals
        ``c / m``. Any wrong-sign or wrong-magnitude damping shows
        up here as a fit residual.
        """
        c = 5.0  # N*s/m -> tau = m/c = 0.2 s
        v0 = 1.0  # initial velocity along +x

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(0.0, 0.0, 0.0),
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(1.0, 0.0, 0.0),
                mode=JointMode.PRISMATIC,
                drive_mode=DriveMode.VELOCITY,
                target_velocity=0.0,  # damp to rest
                stiffness_drive=0.0,
                damping_drive=float(c),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, 0.0),
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )

        # Set initial velocity. Write to Newton's ``state.body_qd``
        # (spatial vector ``[lin, ang]``, linear first) so the
        # capture-replay sync honours the kick.
        bqd = scene.state.body_qd.numpy()
        bqd[bob] = (v0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scene.state.body_qd.assign(bqd)

        # Sample velocity until it has decayed by ~3 time constants.
        n_samples = int(round(self.FPS * 5.0 * (self.MASS / c)))  # 5 tau
        samples = np.zeros(n_samples, dtype=np.float64)
        for f in range(n_samples):
            scene.step()
            v = scene.world.bodies.velocity.numpy()[
                scene.newton_to_jitter[bob]
            ]
            samples[f] = float(v[0])

        # Fit log(v) ~ -lambda * t for the early portion (before
        # samples drop into the noise floor).
        # Cut off when |v| < 5% of v0 to keep the fit linear.
        threshold = 0.05 * v0
        cutoff = n_samples
        for i in range(n_samples):
            if abs(samples[i]) < threshold:
                cutoff = i
                break
        if cutoff < 8:
            self.fail(
                f"velocity decayed too fast to fit cleanly: cutoff={cutoff} "
                f"samples (~{cutoff / self.FPS * 1000:.1f} ms). "
                f"Reduce damping or increase FPS."
            )
        t_arr = np.arange(cutoff) / self.FPS
        log_v = np.log(samples[:cutoff])
        slope, _ = np.polyfit(t_arr, log_v, 1)
        lambda_meas = -float(slope)
        lambda_expected = c / self.MASS
        rel_err = abs(lambda_meas - lambda_expected) / lambda_expected
        self.assertLess(
            rel_err,
            0.05,
            f"prismatic VELOCITY drive decay rate off: "
            f"lambda_meas={lambda_meas:.4f} 1/s, expected c/m = "
            f"{lambda_expected:.4f} 1/s (c={c} N*s/m, m={self.MASS} kg, "
            f"rel_err={rel_err:.2%})",
        )

    # ------------------------------------------------------------------
    # Revolute POSITION drive: torsional DHO identity
    # ------------------------------------------------------------------

    def test_revolute_position_drive_dho_identity(self):
        """Revolute POSITION drive on a hinge with a lever bob,
        excited by an initial angular-velocity kick. The bob starts
        at ``theta = 0`` (so the joint's twist coordinate is zero --
        the unified joint reads twist from the relative quaternion
        recorded at joint creation, see the same caveat in
        :meth:`TestLimitDampingDecay._build_revolute`); we kick it
        with angular velocity ``omega0`` about the hinge axis. Free
        underdamped torsional oscillation; identity
        :math:`\\omega_d^2 + \\gamma^2 = k_p / I_{\\rm eff}` with
        :math:`I_{\\rm eff} = m L^2`.
        """
        kp = 50.0  # N*m/rad
        kd = 0.5  # N*m*s/rad -- zeta ~= 0.14
        I_eff = self.MASS * self.LEVER * self.LEVER
        omega0 = 1.0  # rad/s -- peak angle ~= omega0/omega_0 ~ 0.07 rad
        L = self.LEVER

        mb = newton.ModelBuilder()
        bob = _add_pendulum_bob(
            mb,
            position=(L, 0.0, 0.0),  # arm along +x, twist=0 at start
            mass=self.MASS,
            radius=self.BOB_RADIUS,
        )

        def _factory(n2j, builder):
            builder.add_joint(
                body1=builder.world_body,
                body2=n2j[bob],
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(0.0, 1.0, 0.0),
                mode=JointMode.REVOLUTE,
                drive_mode=DriveMode.POSITION,
                target=0.0,
                stiffness_drive=float(kp),
                damping_drive=float(kd),
                # Rigid anchor lock: tests analytic identity; the
                # default soft 60 Hz anchor would bleed ~10%% of the
                # drive/limit impulse into the lock row.
                hertz=0.0,
            )

        scene = _JitterScene(
            mb,
            _factory,
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, 0.0),
            enable_contacts=False,
            expected_masses={bob: self.MASS},
        )
        # Angular kick about +y plus the consistent linear COM
        # velocity ``v = omega x r = (0, omega, 0) x (L, 0, 0) =
        # (0, 0, -omega * L)``. Newton's spatial vector layout is
        # ``[lin, ang]`` (linear first).
        bqd = scene.state.body_qd.numpy()
        bqd[bob] = (
            0.0, 0.0, -float(omega0) * L,
            0.0, float(omega0), 0.0,
        )
        scene.state.body_qd.assign(bqd)

        n_samples = 1500
        samples = np.zeros(n_samples, dtype=np.float64)
        for f in range(n_samples):
            scene.step()
            pos = scene.world.bodies.position.numpy()[
                scene.newton_to_jitter[bob]
            ]
            theta = math.atan2(-float(pos[2]), float(pos[0]))
            samples[f] = theta

        omega_d, gamma = _measure_oscillation(samples, dt=1.0 / self.FPS)
        omega_0_sq = kp / I_eff
        sum_sq = omega_d * omega_d + gamma * gamma
        rel_err = abs(sum_sq - omega_0_sq) / omega_0_sq
        self.assertLess(
            rel_err,
            0.05,
            f"revolute POSITION drive DHO identity off: "
            f"omega_d^2 + gamma^2 = {sum_sq:.4f}, expected k/I = "
            f"{omega_0_sq:.4f} (omega_d={omega_d:.4f} rad/s, "
            f"gamma={gamma:.4f} 1/s, kp={kp} N*m/rad, "
            f"I_eff={I_eff:.4f} kg*m^2, rel_err={rel_err:.2%})",
        )
        self.assertGreater(
            gamma,
            0.0,
            f"revolute POSITION drive amplitude grew (gamma={gamma:.4f}) "
            "-- torsional damping has wrong sign",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
