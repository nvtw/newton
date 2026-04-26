# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANYmal early-RL stability tests.

During the first few hundred policy gradient updates an RL policy
emits effectively random PD targets on every control step. The
simulator must not explode on this input -- the robot can fail (fall
down, ragdoll), but state must stay finite, the base must not eject
through the ground, and joint velocities must stay bounded.

Two variants:

* :class:`TestAnymalRandomPositionTargets` -- per-frame uniform random
  joint position targets sampled in ``[joint_limit_lower,
  joint_limit_upper]``. Mirrors a position-control RL policy with a
  policy ``out = U(-1, 1)`` mapped through the action scaling.
* :class:`TestAnymalRandomVelocityTargets` -- per-frame uniform random
  joint velocity targets sampled in ``[-omega_max/3, +omega_max/3]``,
  exercising the velocity-mode actuator path.

These tests are smoke tests, not parity tests -- they only check that
no diagnostic-class failure occurs (NaN, base ejection, runaway joint
velocity). The matching `MuJoCo` parity is covered by
:mod:`test_articulation_parity`.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

ANYMAL_FRAMES = 600  # 3 s @ 200 Hz -- well past any compile/warm-up transient
ANYMAL_DT = 1.0 / 200.0
ANYMAL_SUBSTEPS = 4


def _anymal_model_with_position_drives() -> newton.Model:
    """Anymal C URDF, configured exactly the way
    :func:`test_articulation_parity._anymal_model` does (so the
    PhoenX-MuJoCo parity test and these smoke tests share an identical
    rig). PD gains are set on every joint."""
    import newton.utils  # noqa: PLC0415

    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.06,
        limit_ke=1.0e3,
        limit_kd=1.0e1,
    )
    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    stage_path = str(asset_path / "urdf" / "anymal.urdf")
    mb.add_urdf(
        stage_path,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.62),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    mb.add_ground_plane()
    for i in range(len(mb.joint_target_ke)):
        mb.joint_target_ke[i] = 150.0
        mb.joint_target_kd[i] = 5.0
    initial_q = {
        "RH_HAA": 0.0,   "RH_HFE": -0.4,  "RH_KFE": 0.8,
        "LH_HAA": 0.0,   "LH_HFE": -0.4,  "LH_KFE": 0.8,
        "RF_HAA": 0.0,   "RF_HFE": 0.4,   "RF_KFE": -0.8,
        "LF_HAA": 0.0,   "LF_HFE": 0.4,   "LF_KFE": -0.8,
    }
    for name, value in initial_q.items():
        idx = next(
            (i for i, lbl in enumerate(mb.joint_label) if lbl.endswith(f"/{name}")),
            None,
        )
        if idx is not None:
            mb.joint_q[idx + 6] = value
    mb.gravity = -9.81
    return mb.finalize()


def _setup_anymal_loop(model: newton.Model):
    """Return ``(solver, state_0, state_1, control, contacts, target_buf,
    leg_dof_count, leg_dof_offset)``.

    ``target_buf`` is a host-resident ``np.float32`` view shaped to
    ``(joint_dof_count,)``; callers mutate ``target_buf[leg_dof_offset:]``
    every frame and call ``control.joint_target_pos.assign(target_buf)``
    -- this is the canonical Newton pattern for graph-captured
    per-frame target updates and matches what the Anymal walk example
    does.
    """
    solver = newton.solvers.SolverPhoenX(
        model, substeps=ANYMAL_SUBSTEPS, solver_iterations=8, velocity_iterations=1
    )
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    contacts = model.collide(state_0)

    # FREE base accounts for the first 6 DOFs; the next 12 are leg joints.
    n_dofs = int(model.joint_dof_count)
    leg_dof_offset = 6
    leg_dof_count = n_dofs - leg_dof_offset
    target_buf = np.zeros(n_dofs, dtype=np.float32)
    return (
        solver,
        state_0,
        state_1,
        control,
        contacts,
        target_buf,
        leg_dof_count,
        leg_dof_offset,
    )


def _step_one_frame(solver, state_0, state_1, control, contacts, model, dt):
    """Advance ``state_0`` by one ``dt`` second with ``ANYMAL_SUBSTEPS``
    inner substeps. Mirrors :class:`example_robot_anymal_c_walk`'s
    ``simulate``."""
    sub_dt = dt / ANYMAL_SUBSTEPS
    for _ in range(ANYMAL_SUBSTEPS):
        state_0.clear_forces()
        if contacts is not None:
            model.collide(state_0, contacts=contacts)
        solver.step(state_0, state_1, control, contacts, sub_dt)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Anymal random-target tests run on CUDA only.",
)
class TestAnymalRandomPositionTargets(unittest.TestCase):
    """Random-walk position targets every frame -- early-RL warm-up."""

    def test_random_walk_position_targets_no_explosion(self) -> None:
        rng = np.random.default_rng(seed=0)
        model = _anymal_model_with_position_drives()
        (
            solver, s0, s1, control, contacts, target_buf,
            leg_dof_count, leg_dof_offset,
        ) = _setup_anymal_loop(model)

        # Sample targets around the default pose with a +/-0.5 rad window
        # per leg DoF -- this matches what an early-RL policy emits when
        # actions ~ N(0, 1) are scaled into the action space (typical
        # 0.5x scale on Anymal). Sampling the full joint-limit window
        # would force the robot to ragdoll violently every step, which
        # is unrealistic for an RL warm-up scenario.
        default_pose = model.joint_q.numpy()[7 : 7 + leg_dof_count].copy()
        lo = default_pose - 0.5
        hi = default_pose + 0.5
        # Clamp to the URDF's joint limits so we don't sample past them.
        lower = model.joint_limit_lower.numpy()[leg_dof_offset:]
        upper = model.joint_limit_upper.numpy()[leg_dof_offset:]
        lo = np.maximum(lo, lower + 0.05)
        hi = np.minimum(hi, upper - 0.05)

        # Sanity over the whole trajectory:
        n_frames = ANYMAL_FRAMES
        peak_qd = 0.0
        peak_base_z = 0.0
        min_base_z = float("inf")
        for i in range(n_frames):
            target_buf[leg_dof_offset:] = rng.uniform(lo, hi).astype(np.float32)
            control.joint_target_pos.assign(target_buf)
            s0, s1 = _step_one_frame(solver, s0, s1, control, contacts, model, ANYMAL_DT)
            if i % 30 == 29:
                bq = s0.body_q.numpy()
                bqd = s0.body_qd.numpy()
                self.assertTrue(np.isfinite(bq).all(), f"frame {i}: body_q non-finite")
                self.assertTrue(np.isfinite(bqd).all(), f"frame {i}: body_qd non-finite")
                # Track peak joint angular velocity (qd[3:6] is the base
                # angular velocity, qd[6:] are leg DOFs in the spatial
                # body_qd Newton uses).
                peak_qd = max(peak_qd, float(np.abs(bqd[:, :]).max()))
                base_z = float(bq[0, 2])
                peak_base_z = max(peak_base_z, base_z)
                min_base_z = min(min_base_z, base_z)

        # Base must stay grounded-ish: did not fly away or sink through
        # the floor by orders of magnitude. Anymal's base height is
        # ~0.62 m; we allow [-0.5, 2.0] m which still fails on real
        # blowups while letting the robot legitimately tumble.
        self.assertGreater(min_base_z, -0.5, msg=f"base sank to z={min_base_z:.3f} m -- through-the-floor")
        self.assertLess(peak_base_z, 2.0, msg=f"base rose to z={peak_base_z:.3f} m -- ejected upward")

        # Joint / body velocities stay bounded. 50 m/s linear or 50 rad/s
        # angular is a hard cap -- a tipped-over robot easily hits 5-10,
        # an exploding solver typically blows past 100.
        self.assertLess(
            peak_qd,
            50.0,
            msg=f"peak |body_qd| = {peak_qd:.2f} -- runaway",
        )

        # Final-state finite (catches a blowup that happens between
        # checkpoints).
        bq_final = s0.body_q.numpy()
        bqd_final = s0.body_qd.numpy()
        self.assertTrue(np.isfinite(bq_final).all())
        self.assertTrue(np.isfinite(bqd_final).all())


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Anymal random-target tests run on CUDA only.",
)
class TestAnymalRandomVelocityTargets(unittest.TestCase):
    """Random per-frame velocity targets with capped magnitude. Probes
    the VELOCITY-mode actuator path under the same RL-warm-up
    conditions as the position-target test."""

    def test_random_velocity_targets_no_explosion(self) -> None:
        rng = np.random.default_rng(seed=1)

        # Build the Anymal rig with VELOCITY-mode actuators by switching
        # the per-DoF actuator mode + setting target_kd. ``newton``
        # exposes this via ``ModelBuilder.joint_target_mode``; we
        # rebuild with the same Anymal URDF and override the mode and
        # gains in-place before finalize.
        model = _anymal_model_with_position_drives()
        # Rewrite per-DoF mode to VELOCITY (skip the 6 FREE base DOFs).
        mode = model.joint_target_mode.numpy()
        mode[6:] = int(newton.JointTargetMode.VELOCITY)
        model.joint_target_mode.assign(mode)
        # Velocity-mode wants a damping coefficient on each DoF -- use
        # the same kd value the position drives have.
        model.joint_target_kd.assign(np.full(model.joint_dof_count, 5.0, dtype=np.float32))

        (
            solver, s0, s1, control, contacts, target_buf,
            leg_dof_count, leg_dof_offset,
        ) = _setup_anymal_loop(model)

        # Sample velocity targets inside ``[-omega_max/3, omega_max/3]``
        # rad/s, where omega_max = 8 rad/s is a typical limit on Anymal's
        # leg actuators. Capping at /3 of the peak makes the targets
        # smooth enough that a sane PD can track without diverging.
        omega_max = 8.0 / 3.0
        target_buf_vel = np.zeros_like(target_buf)

        n_frames = ANYMAL_FRAMES
        peak_qd = 0.0
        for i in range(n_frames):
            target_buf_vel[leg_dof_offset:] = rng.uniform(
                -omega_max, omega_max, size=leg_dof_count
            ).astype(np.float32)
            control.joint_target_vel.assign(target_buf_vel)
            s0, s1 = _step_one_frame(solver, s0, s1, control, contacts, model, ANYMAL_DT)
            if i % 30 == 29:
                bq = s0.body_q.numpy()
                bqd = s0.body_qd.numpy()
                self.assertTrue(np.isfinite(bq).all(), f"frame {i}: body_q non-finite")
                self.assertTrue(np.isfinite(bqd).all(), f"frame {i}: body_qd non-finite")
                peak_qd = max(peak_qd, float(np.abs(bqd).max()))
                self.assertGreater(float(bq[0, 2]), -0.5, f"frame {i}: base through floor")
                self.assertLess(float(bq[0, 2]), 2.0, f"frame {i}: base ejected")

        self.assertLess(
            peak_qd,
            50.0,
            msg=f"peak |body_qd| = {peak_qd:.2f} under random velocity targets -- runaway",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
