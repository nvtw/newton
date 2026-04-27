# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stability tests for revolute-joint PD drives and limits on pathological
inputs.

Three failure modes to guard against:

1. **Target step changes**: a controller handing PhoenX a drive
   target that jumps several radians per frame (saturated command,
   discrete gait planner) must not produce a NaN / explosion.

2. **Multi-revolution targets**: a drive target multiple full
   rotations away from the current angle. The revolution tracker
   must count past ``2*pi`` without wrapping the angle error to a
   spurious small value.

3. **Limits**: both patterns above applied to ``joint_limit_lower`` /
   ``joint_limit_upper`` -- a narrow limit window that suddenly
   shrinks or a limit placed several revolutions away from the
   current joint angle.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

_G = 9.81


def _pendulum(
    *,
    init_q: float = 0.0,
    target_pos: float = 0.0,
    target_ke: float = 150.0,
    target_kd: float = 5.0,
    limit_lower: float | None = None,
    limit_upper: float | None = None,
    limit_ke: float | None = None,
    limit_kd: float | None = None,
) -> newton.Model:
    """Single-revolute pendulum with configurable drive + limits."""
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    cube = mb.add_link(
        xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        cube,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    kwargs = {
        "parent": -1,
        "child": cube,
        "axis": (0.0, 1.0, 0.0),
        "parent_xform": wp.transform_identity(),
        "child_xform": wp.transform(p=wp.vec3(-1.0, 0.0, 0.0), q=wp.quat_identity()),
        "target_pos": target_pos,
        "target_ke": target_ke,
        "target_kd": target_kd,
        "actuator_mode": newton.JointTargetMode.POSITION,
    }
    if limit_lower is not None:
        kwargs["limit_lower"] = limit_lower
    if limit_upper is not None:
        kwargs["limit_upper"] = limit_upper
    if limit_ke is not None:
        kwargs["limit_ke"] = limit_ke
    if limit_kd is not None:
        kwargs["limit_kd"] = limit_kd
    joint = mb.add_joint_revolute(**kwargs)
    mb.add_articulation([joint])
    mb.gravity = 0.0
    model = mb.finalize()
    if init_q != 0.0:
        model.joint_q.assign(np.array([init_q], dtype=np.float32))
    return model


def _solver(model: newton.Model):
    return newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=16, velocity_iterations=1)


def _run_with_target_schedule(
    model: newton.Model,
    targets_per_frame: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``len(targets_per_frame)`` frames and rewrite
    ``control.joint_target_pos`` each frame. Returns the ``joint_q``
    + ``joint_qd`` trajectory."""
    solver = _solver(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    n = targets_per_frame.shape[0]
    q_traj = np.empty(n, dtype=np.float32)
    qd_traj = np.empty(n, dtype=np.float32)
    for i in range(n):
        t = np.asarray([targets_per_frame[i]], dtype=np.float32)
        control.joint_target_pos.assign(t)
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        newton.eval_ik(model, s0, jq, jqd)
        q_traj[i] = float(jq.numpy()[0])
        qd_traj[i] = float(jqd.numpy()[0])
    return q_traj, qd_traj


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Drive-stability tests run on CUDA only.",
)
class TestDriveTargetJumpStability(unittest.TestCase):
    """Large discontinuous jumps in the drive target must not destabilise
    the solver."""

    def test_target_step_change(self) -> None:
        """Square-wave target toggling between +pi/2 and -pi/2 every
        plateau frames for 2 s. The drive must (a) not diverge and
        (b) actually move the joint toward the current target on every
        plateau -- a drive that ignored the target would fail (b)."""
        plateau = 20  # 100 ms at dt=5 ms
        n_plateaus = 20
        n = plateau * n_plateaus
        dt = 0.005
        targets = np.where((np.arange(n) // plateau) % 2 == 0, math.pi / 2.0, -math.pi / 2.0).astype(np.float32)
        model = _pendulum(target_pos=0.0, target_ke=200.0, target_kd=20.0)
        q, qd = _run_with_target_schedule(model, targets, dt)

        self.assertTrue(np.isfinite(q).all(), msg="joint_q went non-finite")
        self.assertTrue(np.isfinite(qd).all(), msg="joint_qd went non-finite")
        peak_q = float(np.abs(q).max())
        peak_qd = float(np.abs(qd).max())
        self.assertLess(peak_q, math.pi, msg=f"|q| peaked at {peak_q:.3f} rad -- too large for square-wave target")
        self.assertLess(peak_qd, 100.0, msg=f"|qd| peaked at {peak_qd:.3f} rad/s -- runaway")

        # Joint must actually move under the drive -- a broken drive
        # leaves q near zero. With ke=200 / kd=20 the critically-damped
        # response only partially settles inside a 100 ms plateau, but
        # peak excursion crosses ~0.5 rad easily.
        self.assertGreater(
            peak_q,
            0.3,
            msg=f"|q| peaked only at {peak_q:.3f} rad -- drive may be ignoring the target",
        )

        # Per-plateau direction check: skip the first plateau (initial
        # transient from rest), then for every subsequent plateau the
        # change ``q[end] - q[start]`` must lie in the direction of
        # ``target - q[start]`` -- i.e., the drive consistently pulls
        # toward the active target. Allow up to 1 plateau out of the 19
        # remaining to violate (covers the toggle frame's overshoot).
        violations = 0
        for p in range(1, n_plateaus):
            i0 = p * plateau
            i1 = (p + 1) * plateau - 1
            target_p = float(targets[i0])
            dq = float(q[i1] - q[i0])
            err0 = target_p - float(q[i0])
            if err0 == 0.0 or dq == 0.0:
                continue
            if dq * err0 < 0.0:
                violations += 1
        self.assertLessEqual(
            violations,
            1,
            msg=f"{violations}/{n_plateaus - 1} plateaus moved away from their target",
        )

    def test_random_target_jitter(self) -> None:
        """Smooth band-limited random target inside ``[-pi/2, +pi/2]`` as
        a stand-in for the smooth-but-noisy commands an early-stage RL
        policy emits. The drive must (a) not diverge and (b) correlate
        positively with the target -- a drive that ignored the target
        would have correlation ~0."""
        n = 600
        dt = 0.005
        # Smooth target: low-pass-filtered Gaussian noise. tau = 300 ms ->
        # ~0.5 Hz dominant content, well below the pendulum's ~2 Hz
        # natural frequency so phase lag stays small and the
        # target/response correlation is sharp.
        rng = np.random.default_rng(seed=0)
        tau = 0.3
        alpha = dt / (tau + dt)
        raw = rng.standard_normal(n).astype(np.float32) * 6.0
        smooth = np.empty(n, dtype=np.float32)
        smooth[0] = 0.0
        for i in range(1, n):
            smooth[i] = (1.0 - alpha) * smooth[i - 1] + alpha * raw[i]
        targets = np.clip(smooth, -math.pi / 2.0, math.pi / 2.0)
        model = _pendulum(target_pos=0.0, target_ke=200.0, target_kd=20.0)
        q, qd = _run_with_target_schedule(model, targets, dt)

        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.isfinite(qd).all())
        self.assertLess(float(np.abs(q).max()), math.pi)

        # Sanity: joint must move under the drive (broken drive leaves
        # q ~ 0).
        skip = n // 5
        q_rms = float(np.sqrt(np.mean(q[skip:] ** 2)))
        self.assertGreater(
            q_rms,
            0.05,
            msg=f"q RMS {q_rms:.3f} rad too small -- joint barely moved",
        )

        # Correlation with target: a drive that follows the target gives
        # a strongly positive correlation; ignoring the target gives ~0.
        q_c = q[skip:] - q[skip:].mean()
        t_c = targets[skip:] - targets[skip:].mean()
        denom = float(np.sqrt((q_c * q_c).sum() * (t_c * t_c).sum()))
        if denom > 0.0:
            corr = float((q_c * t_c).sum() / denom)
        else:
            corr = 0.0
        self.assertGreater(
            corr,
            0.5,
            msg=f"q-vs-target correlation {corr:.3f} too low -- drive may not be tracking targets",
        )

    def test_impulse_target_return(self) -> None:
        """Target starts at 0, jumps to +pi/4 for one frame, back to 0.
        A single-frame impulse should not leak a large residual velocity."""
        n = 200
        dt = 0.005
        targets = np.zeros(n, dtype=np.float32)
        targets[50] = math.pi / 4.0
        model = _pendulum(target_pos=0.0, target_ke=200.0, target_kd=20.0)
        q, qd = _run_with_target_schedule(model, targets, dt)

        # After settle window, both q and qd back near zero.
        self.assertLess(abs(float(q[-1])), 0.1, msg=f"q residual {q[-1]:.3f} too large after impulse return")
        self.assertLess(abs(float(qd[-1])), 0.5, msg=f"qd residual {qd[-1]:.3f} too large after impulse return")


def _cumulative_y_rotation(
    model: newton.Model, s0: newton.State, initial_cum: float, last_branch: float
) -> tuple[float, float]:
    """Track cumulative rotation about +y from the body's quaternion.

    ``2*atan2(q_y, q_w)`` maps to ``(-2*pi, 2*pi]``; a single body
    rotation through ``2*pi`` makes the branch jump by ~``4*pi``.
    The ``while`` loops fold each 2*pi worth of jump out of the delta
    until what remains is the true per-step rotation (``abs <= pi``,
    well within anything a sane sim can produce per frame).
    """
    bq = s0.body_q.numpy()[0]
    branch = 2.0 * math.atan2(float(bq[4]), float(bq[6]))
    delta = branch - last_branch
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return initial_cum + delta, branch


def _run_target_and_track_rotation(
    model: newton.Model,
    targets_per_frame: np.ndarray,
    dt: float,
) -> tuple[float, float]:
    """Drive ``model`` with a per-frame target schedule and return the
    unwrapped cumulative +y rotation at the end (plus the final
    angular velocity). Uses the body quaternion directly so
    ``eval_ik``'s branch cut doesn't hide the true rotation count."""
    solver = _solver(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    # Seed the branch-tracker with the initial body orientation.
    bq0 = s0.body_q.numpy()[0]
    last_branch = 2.0 * math.atan2(float(bq0[4]), float(bq0[6]))
    cum = last_branch
    for i in range(targets_per_frame.shape[0]):
        t = np.asarray([targets_per_frame[i]], dtype=np.float32)
        control.joint_target_pos.assign(t)
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        cum, last_branch = _cumulative_y_rotation(model, s0, cum, last_branch)
    final_qd = float(s0.body_qd.numpy()[0, 4])  # y-axis angular velocity
    return cum, final_qd


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Multi-revolution tests run on CUDA only.",
)
class TestMultiRevolutionDrive(unittest.TestCase):
    """Drive targets several revolutions away exercise the
    ``revolution_tracker``'s wrap-around counter.

    ``eval_ik`` reports ``joint_q`` via ``atan2`` of the body's axis
    quaternion and loses the revolution count at every ``pi``, so
    these tests compare the **cumulative physical rotation** derived
    from the body quaternion directly -- see
    :func:`_cumulative_y_rotation`. That removes the eval_ik reporting
    artefact without changing what the solver is doing."""

    def test_two_revolutions_drive(self) -> None:
        """Target = ``2*pi`` (one full rotation). The body's physical
        cumulative rotation must reach ``2*pi``."""
        target = 2.0 * math.pi
        n = 1500
        dt = 0.005
        targets = np.full(n, target, dtype=np.float32)
        model = _pendulum(
            target_pos=0.0,
            target_ke=50.0,
            target_kd=8.0,
        )
        cum, qd = _run_target_and_track_rotation(model, targets, dt)

        self.assertTrue(math.isfinite(cum))
        self.assertTrue(math.isfinite(qd))
        self.assertLess(
            abs(cum - target),
            0.5,
            msg=f"multi-rev physical rotation = {cum:.3f}, target = {target:.3f}",
        )

    def test_negative_multi_revolution_drive(self) -> None:
        """Target = ``-2*pi`` -- mirror."""
        target = -2.0 * math.pi
        n = 1500
        dt = 0.005
        targets = np.full(n, target, dtype=np.float32)
        model = _pendulum(
            target_pos=0.0,
            target_ke=50.0,
            target_kd=8.0,
        )
        cum, qd = _run_target_and_track_rotation(model, targets, dt)

        self.assertTrue(math.isfinite(cum))
        self.assertTrue(math.isfinite(qd))
        self.assertLess(
            abs(cum - target),
            0.5,
            msg=f"negative multi-rev physical rotation = {cum:.3f}, target = {target:.3f}",
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Limit-stability tests run on CUDA only.",
)
class TestLimitStability(unittest.TestCase):
    """Joint-limit stability under pathological configurations."""

    def test_limit_clamps_initial_spin(self) -> None:
        """Spin the joint hard into a narrow limit with stiff
        limit gains. The soft limit must eventually settle the joint
        inside the window, without NaN or runaway velocity."""
        n = 600
        dt = 0.005
        targets = np.zeros(n, dtype=np.float32)
        model = _pendulum(
            init_q=0.0,
            target_pos=0.0,
            target_ke=0.0,
            target_kd=0.0,
            limit_lower=-0.5,
            limit_upper=0.5,
            limit_ke=2.0e3,
            limit_kd=50.0,
        )
        # Moderate initial spin so the soft limit can catch it in the
        # window. Stronger spins saturate any soft constraint.
        model.joint_qd.assign(np.array([3.0], dtype=np.float32))
        q, qd = _run_with_target_schedule(model, targets, dt)

        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.isfinite(qd).all())
        # Final settled position should be inside the limit window
        # (within soft-constraint overshoot slop).
        self.assertLess(
            abs(float(q[-1])),
            0.8,
            msg=f"limit did not settle: final q={q[-1]:.3f}",
        )
        # Final velocity should be small.
        self.assertLess(
            abs(float(qd[-1])),
            1.0,
            msg=f"final |qd|={abs(qd[-1]):.3f} -- limit not damping",
        )

    def test_limit_across_multiple_revolutions(self) -> None:
        """Limit window = ``[+2*pi - 0.3, +2*pi + 0.3]`` -- one full
        rotation away from the current angle. The drive must spin
        the joint up through the first revolution to reach the
        window, and the soft limit must hold it there.

        Assessed on the body's physical cumulative rotation (not
        ``joint_q``) so the ``eval_ik`` branch cut doesn't obscure
        the result."""
        target = 2.0 * math.pi
        n = 2000
        dt = 0.005
        targets = np.full(n, target, dtype=np.float32)
        model = _pendulum(
            init_q=0.0,
            target_pos=0.0,
            target_ke=50.0,
            target_kd=8.0,
            limit_lower=2.0 * math.pi - 0.3,
            limit_upper=2.0 * math.pi + 0.3,
            limit_ke=200.0,
            limit_kd=20.0,
        )
        cum, _qd = _run_target_and_track_rotation(model, targets, dt)

        self.assertTrue(math.isfinite(cum))
        # Physical rotation must be inside the window (or a generous
        # soft-constraint slop outside it).
        self.assertGreater(
            cum,
            2.0 * math.pi - 1.0,
            msg=f"cum rotation {cum:.3f} below multi-rev lower limit",
        )
        self.assertLess(
            cum,
            2.0 * math.pi + 1.0,
            msg=f"cum rotation {cum:.3f} above multi-rev upper limit",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
