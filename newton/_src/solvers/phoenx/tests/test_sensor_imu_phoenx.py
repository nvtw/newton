# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""IMU / ``body_qdd`` tests for :class:`SolverPhoenX`.

PhoenX is a velocity-based PGS integrator and does not natively expose
``body_qdd``. The solver finite-differences the pre-step and post-step
COM-frame linear / angular velocity over the outer ``dt`` and writes
the result to ``state_out.body_qdd`` when the user has requested the
``body_qdd`` extended state attribute (which :class:`SensorIMU` does
automatically). The Newton convention matches what the IMU kernel
consumes:

* ``spatial_top(body_qdd)`` -- linear acceleration in world frame,
  including gravity-induced terms (so an accelerometer reads
  ``a_world - gravity`` as specific force).
* ``spatial_bottom(body_qdd)`` -- angular acceleration in world frame.

Three layers of coverage, all CUDA + graph-capture only (PhoenX is
GPU-only and graph capture is the shipping execution mode -- a fix
that breaks capture must surface here):

* :class:`TestBodyQddBasics` -- direct readback of ``state.body_qdd``
  for analytical fixtures (free-fall, at rest, spin-up). Bypasses
  the IMU sensor and pins the FD wiring itself.

* :class:`TestSensorIMU` -- drives :class:`~newton.sensors.SensorIMU`
  end-to-end through PhoenX inside a CUDA graph. Verifies the
  textbook IMU invariants:

    - a body at rest reads ``+g`` along the up-axis (specific force);
    - a body in free fall reads ``~0`` (the iconic IMU property);
    - a spinning body reads its angular velocity through the gyroscope.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorIMU

_G = 9.81
_FPS = 240
_DT = 1.0 / _FPS


def _capture_step(
    model: newton.Model,
    solver: newton.solvers.SolverPhoenX,
    state_in: newton.State,
    state_out: newton.State,
    control: newton.Control | None,
    sensors: list[SensorIMU],
    *,
    n_frames: int,
    use_contacts: bool = True,
) -> None:
    """Run ``n_frames`` of (clear_forces, collide, step, sensor.update) through
    CUDA graph capture. Mirrors the graph-capture harness used by
    :mod:`test_sensor_contact`.

    ``use_contacts=False`` skips contact generation (free-fall fixtures that
    intentionally have no ground plane)."""
    device = wp.get_device()
    assert device.is_cuda, "graph-captured IMU tests require CUDA"

    contacts = model.contacts() if use_contacts else None

    def _frame() -> None:
        state_in.clear_forces()
        if contacts is not None:
            model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        for s in sensors:
            s.update(state_out)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        # body_qdd is the readout target; copying it back into state_in is
        # not strictly required (PhoenX overwrites it next step), but
        # keeps the two-state ping-pong tidy.
        if state_in.body_qdd is not None and state_out.body_qdd is not None:
            wp.copy(state_in.body_qdd, state_out.body_qdd)

    if n_frames < 1:
        return

    _frame()  # warm-up: compile kernels, prime lazy scratch
    if n_frames == 1:
        return

    with wp.ScopedCapture(device=device) as capture:
        _frame()
    graph = capture.graph

    for _ in range(n_frames - 2):
        wp.capture_launch(graph)


def _make_solver(model: newton.Model, *, substeps: int = 4) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=20,
        velocity_iterations=2,
    )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX IMU tests require CUDA")
class TestBodyQddBasics(unittest.TestCase):
    """Direct ``state.body_qdd`` readback. Pins the FD wiring itself
    (without involving :class:`SensorIMU`)."""

    def _build_free_body(self, *, z: float = 1.0) -> newton.Model:
        """One free-floating body well above any ground."""
        mass, radius = 1.0, 0.05
        builder = newton.ModelBuilder()
        ixx = 0.4 * mass * radius * radius
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, z), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.request_state_attributes("body_qdd")
        return builder.finalize()

    def test_free_fall_linear_acceleration(self) -> None:
        """A body falling under gravity must report ``body_qdd.top == (0,0,-g)``.

        The FD averages over the outer ``dt``. Starting from rest, the
        post-step velocity is ``-g * dt`` (no contacts, no drag), so
        ``(v_now - v_prev) / dt == -g``. Validates the sign convention
        and that PhoenX's gravity application reaches the FD readout.
        """
        model = self._build_free_body(z=2.0)
        solver = _make_solver(model)
        state_in = model.state()
        state_out = model.state()
        self.assertIsNotNone(state_in.body_qdd, "body_qdd must be allocated after request_state_attributes")
        self.assertIsNotNone(state_out.body_qdd, "body_qdd must be allocated after request_state_attributes")

        _capture_step(model, solver, state_in, state_out, None, [], n_frames=3, use_contacts=False)

        qdd = state_out.body_qdd.numpy()[0]
        lin = qdd[:3]
        ang = qdd[3:]
        np.testing.assert_allclose(
            lin,
            np.array([0.0, 0.0, -_G], dtype=np.float32),
            atol=1.0e-2,
            err_msg=f"free-fall linear acc should be (0,0,-g); got {lin}",
        )
        np.testing.assert_allclose(
            ang,
            np.zeros(3, dtype=np.float32),
            atol=1.0e-2,
            err_msg=f"free-fall angular acc should be ~0; got {ang}",
        )

    def test_at_rest_acceleration_is_zero(self) -> None:
        """A body settled on the ground must report ``body_qdd ~ 0``: net
        force is zero in equilibrium, so the FD between consecutive
        steady-state velocities is zero. A non-zero residual flags
        either incomplete settling or a wrong gravity-routing
        convention."""
        mass, radius = 1.0, 0.1
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ixx = 0.4 * mass * radius * radius
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, radius + 0.05), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.request_state_attributes("body_qdd")
        model = builder.finalize()

        solver = _make_solver(model)
        state_in = model.state()
        state_out = model.state()

        # Let the body settle on the ground -- 120 frames at 240 Hz is
        # well past the contact-spring transient.
        _capture_step(model, solver, state_in, state_out, None, [], n_frames=120)

        qdd = state_out.body_qdd.numpy()[0]
        # 5 % of g is the per-frame numerical-noise floor at this substep
        # count; well-converged sphere-on-plane contact normally lands
        # under 1 %.
        np.testing.assert_allclose(
            qdd,
            np.zeros(6, dtype=np.float32),
            atol=0.05 * _G,
            err_msg=f"at-rest body_qdd should be ~0; got {qdd}",
        )

    def test_body_qdd_skipped_when_not_requested(self) -> None:
        """When ``body_qdd`` is not requested on the model, ``state.body_qdd``
        is ``None`` and the solver must not crash trying to write into
        it. The FD launch is gated on the allocation existing on
        ``state_out``."""
        # Build a model WITHOUT request_state_attributes.
        mass, radius = 1.0, 0.05
        builder = newton.ModelBuilder()
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
            mass=mass,
            inertia=((1e-4, 0, 0), (0, 1e-4, 0), (0, 0, 1e-4)),
        )
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        model = builder.finalize()

        solver = _make_solver(model)
        state_in = model.state()
        state_out = model.state()
        self.assertIsNone(state_in.body_qdd, "body_qdd should be unallocated without request")
        self.assertIsNone(state_out.body_qdd, "body_qdd should be unallocated without request")

        # Step must succeed end-to-end -- this is the regression guard
        # against the ``state_out.body_qdd is not None`` check being
        # mishandled (e.g. an unconditional launch would fault on a
        # null array).
        _capture_step(model, solver, state_in, state_out, None, [], n_frames=3, use_contacts=False)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX IMU tests require CUDA")
class TestSensorIMU(unittest.TestCase):
    """End-to-end :class:`SensorIMU` integration. The sensor consumes
    ``body_qdd``; an accelerometer reading equal to the textbook
    invariants validates that the FD direction, magnitude, and frame
    convention all line up with what the IMU kernel expects."""

    def test_at_rest_accelerometer_reads_g_upward(self) -> None:
        """A stationary IMU on a body at rest under gravity must read
        ``+g`` along the up axis: ``a_world == 0`` at equilibrium, so
        the specific force ``a_world - gravity == -gravity == +g e_z``.
        This is *the* textbook accelerometer invariant; a sign flip or
        gravity-routing bug fails immediately."""
        mass, radius = 1.0, 0.1
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ixx = 0.4 * mass * radius * radius
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, radius + 0.05), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.add_site(body, label="imu_rest")
        model = builder.finalize()

        sensor = SensorIMU(model, sites="imu_rest", verbose=False)
        solver = _make_solver(model)
        state_in = model.state()
        state_out = model.state()

        _capture_step(model, solver, state_in, state_out, None, [sensor], n_frames=180)

        acc = sensor.accelerometer.numpy()[0]
        gyro = sensor.gyroscope.numpy()[0]
        # Specific force at rest is +g along +Z (up). Tolerance covers
        # per-frame settling noise + FD discretization (~ a few % of g).
        self.assertGreater(
            float(acc[2]),
            0.5 * _G,
            f"accelerometer Z should be ~+g (specific force points up at rest), got {acc}",
        )
        np.testing.assert_allclose(
            acc,
            np.array([0.0, 0.0, _G], dtype=np.float32),
            atol=0.5,
            err_msg=f"accelerometer at rest should read (0,0,+g); got {acc}",
        )
        # No rotation -> gyroscope is zero.
        self.assertLess(
            float(np.linalg.norm(gyro)),
            0.05,
            f"gyroscope at rest should be ~0; got {gyro}",
        )

    def test_free_fall_accelerometer_reads_zero(self) -> None:
        """Free-fall is the IMU's defining invariant: the accelerometer
        reads ``~0`` even though gravity is acting. ``a_world == -g e_z``
        and specific force ``a_world - gravity == 0``.

        Catches the failure mode where ``body_qdd`` doesn't include
        gravity-induced acceleration: the FD between (v=0) and
        (v=-g*dt) yields ``-g``, so without gravity in the FD output
        the IMU would read ``+g`` (a stationary reading) during free
        fall."""
        mass, radius = 1.0, 0.05
        builder = newton.ModelBuilder()
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 3.0), wp.quat_identity()),
            mass=mass,
            inertia=((1e-4, 0, 0), (0, 1e-4, 0), (0, 0, 1e-4)),
        )
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.add_site(body, label="imu_fall")
        # No ground -- pure free-fall.
        model = builder.finalize()

        sensor = SensorIMU(model, sites="imu_fall", verbose=False)
        solver = _make_solver(model)
        state_in = model.state()
        state_out = model.state()

        # Only a few frames so the body doesn't pick up enough velocity
        # to amplify numerical error. Free-fall is the cleanest possible
        # signal for the FD readout.
        _capture_step(model, solver, state_in, state_out, None, [sensor], n_frames=4, use_contacts=False)

        acc = sensor.accelerometer.numpy()[0]
        # Free-fall: specific force is ~0. A non-zero reading flags
        # either a sign error in body_qdd or a missing gravity term.
        self.assertLess(
            float(np.linalg.norm(acc)),
            0.5,
            f"accelerometer in free-fall should read ~0; got {acc} (|a|={np.linalg.norm(acc):.3f})",
        )

    def test_spinning_body_gyroscope(self) -> None:
        """A body spinning at known angular velocity ``omega`` about ``+Z``
        in zero gravity must report ``gyroscope == omega * e_z`` (in
        the IMU site frame, identity-aligned with the body frame
        here). Gyroscope reads ``body_qd_ang``, not ``body_qdd``, so
        this doubles as a check that adding the body_qdd readout
        didn't perturb the velocity export path."""
        mass, radius = 1.0, 0.05
        omega_z = 5.0  # rad/s

        builder = newton.ModelBuilder(gravity=0.0)  # no gravity to keep linear acc clean
        ixx = 0.4 * mass * radius * radius
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_sphere(body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.add_site(body, label="imu_spin")
        model = builder.finalize()

        sensor = SensorIMU(model, sites="imu_spin", verbose=False)
        solver = _make_solver(model, substeps=2)
        state_in = model.state()
        state_out = model.state()

        # Seed angular velocity about +Z.
        qd = state_in.body_qd.numpy()
        qd[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, omega_z], dtype=np.float32)
        state_in.body_qd.assign(qd)

        _capture_step(model, solver, state_in, state_out, None, [sensor], n_frames=4, use_contacts=False)

        gyro = sensor.gyroscope.numpy()[0]
        # Symmetric inertia + zero gravity -> angular velocity is conserved.
        # Site is identity-aligned with the body so gyro frame == world frame.
        np.testing.assert_allclose(
            gyro,
            np.array([0.0, 0.0, omega_z], dtype=np.float32),
            atol=0.1,
            err_msg=f"gyroscope should read (0,0,{omega_z}); got {gyro}",
        )


if __name__ == "__main__":
    unittest.main()
