# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-DoF observation diff: PhoenX vs MuJoCo Warp on the G1 standing pose.

Diagnostic, not a unit test. Run via::

    uv run python -m newton._src.solvers.phoenx.tests._g1_obs_diff

The :mod:`example_robot_policy` policy reads its observations from
``state.joint_q`` and ``state.joint_qd`` only -- root quaternion,
root linear/angular velocity, per-DoF joint position and velocity,
plus a derived ``projected_gravity`` and the user command. Nothing
exotic, no force/torque sensors. So the question "is PhoenX
reporting the wrong sign / magnitude / frame to the policy?" reduces
to: do the two solvers populate ``state.joint_q`` / ``state.joint_qd``
with matching numerical values after the same number of steps from
the same starting pose?

This script answers that by replicating the harness from
:mod:`test_robot_policy_parity` and printing per-DoF state at
steps 0, 1, and 50:

* **Step 0** -- before any simulation. Both solvers should report
  the seeded ``model.joint_q`` exactly. Differences here would mean
  one solver's state init is broken.

* **Step 1** -- after one ``solver.step``. Catches sign flips and
  frame mismatches in the per-step writeback.

* **Step 50** -- after 0.25 s. Catches accumulated divergence from
  contact dynamics differences.

The output prints free-base and per-DoF q / qd side by side. The
G1 toes-down bug surfaces here as:

* ``left_ankle_pitch_joint`` and ``right_ankle_pitch_joint`` qd
  growing positive in PhoenX (extending the foot, pushing toes
  into the floor) while MuJoCo qd stays near 0.

* ``left_ankle_roll_joint`` / ``right_ankle_roll_joint`` qd jumping
  to ~+2 rad/s after the very first step in PhoenX with MuJoCo
  reporting ~0.

* The trunk's pitch component (``body_qd[trunk].angular.x``)
  acquiring opposite signs between solvers around step 50.

These divergences are *not* readback bugs -- step 0 matches exactly
and the conversion kernels in :mod:`example_common` use the public
COM-twist convention that ``eval_ik`` expects. They reflect a real
divergence in the simulated physics under foot-ground contact load.
The companion diagnostic :mod:`_actuator_compare` traces it back to
PhoenX's strict-vs-MuJoCo's-lossy stiction discretisation.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_robot_policy_parity import (
    _g1_29dof_yaml,
    _g1_robot_model,
    _mj_solver,
    _px_solver,
)


def _px_solver_fd(model: newton.Model):
    """PhoenX with the finite-difference velocity readout opt-in.

    Velocity stamped into ``state.body_qd`` is
    ``(body_q_now - body_q_prev) / dt_outer`` instead of the
    substep-end value. Smooths over PhoenX's per-substep contact-
    impulse ringing without changing the simulated physics.
    """
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=16,
        velocity_iterations=1,
        velocity_readout="finite_difference",
    )


def _step_n(solver_factory, n_frames: int) -> tuple[newton.Model, newton.State]:
    """Build a fresh G1 model, hold the standing pose target, step
    ``n_frames`` of size 1/200 s, return ``(model, state)``."""
    model = _g1_robot_model()
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    ctl = model.control()
    target = np.zeros(int(model.joint_dof_count), dtype=np.float32)
    target[6:] = model.joint_q.numpy()[7:]
    ctl.joint_target_pos.assign(target)

    is_phoenx = isinstance(solver, newton.solvers.SolverPhoenX)
    contacts = model.contacts() if is_phoenx else None
    dt = 1.0 / 200.0
    for _ in range(n_frames):
        s0.clear_forces()
        if is_phoenx:
            model.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts if is_phoenx else None, dt)
        s0, s1 = s1, s0
    return model, s0


def _quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy port of ``example_robot_policy.quat_rotate_inverse``.

    ``q`` is XYZW. Returns ``v`` rotated by the inverse of ``q``,
    matching what the torch policy gets after JIT-scripting.
    """
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    a_scale = 2.0 * qw * qw - 1.0
    a_vec = np.array([vx * a_scale, vy * a_scale, vz * a_scale])
    cross_qv = np.array([qy * vz - qz * vy, qz * vx - qx * vz, qx * vy - qy * vx])
    b_vec = cross_qv * (qw * 2.0)
    qv_dot_v = qx * vx + qy * vy + qz * vz
    c_vec = np.array([qx, qy, qz]) * (qv_dot_v * 2.0)
    return a_vec - b_vec + c_vec


def _make_obs(jq: np.ndarray, jqd: np.ndarray) -> dict[str, np.ndarray]:
    """Replicate the per-block observation construction in
    :func:`example_robot_policy.compute_obs` *without* the index-
    rearrangement (we want raw per-DoF order to compare like-for-like
    across solvers). Returns a dict so the diff printer can reach
    each block individually.
    """
    root_quat = jq[3:7]
    root_lin_vel_w = jqd[0:3]
    root_ang_vel_w = jqd[3:6]
    gravity_w = np.array([0.0, 0.0, -1.0])  # the example feeds (0,0,-1)

    return {
        "vel_b": _quat_rotate_inverse_np(root_quat, root_lin_vel_w),
        "a_vel_b": _quat_rotate_inverse_np(root_quat, root_ang_vel_w),
        "grav_b": _quat_rotate_inverse_np(root_quat, gravity_w),
        "joint_pos_rel": jq[7:].copy(),  # caller already has joint_pos_initial baseline
        "joint_vel_rel": jqd[6:].copy(),
    }


def _print_diff(n_frames: int) -> None:
    cfg = _g1_29dof_yaml()
    names = cfg["mjw_joint_names"]

    print(f"\n{'=' * 120}")
    print(f"AFTER {n_frames} STEPS  (dt = 1/200 s)")
    print(f"{'=' * 120}")

    _, s_mj = _step_n(_mj_solver, n_frames)
    _, s_px = _step_n(_px_solver, n_frames)
    _, s_fd = _step_n(_px_solver_fd, n_frames)
    jq_mj = s_mj.joint_q.numpy()
    jqd_mj = s_mj.joint_qd.numpy()
    jq_px = s_px.joint_q.numpy()
    jqd_px = s_px.joint_qd.numpy()
    jq_fd = s_fd.joint_q.numpy()
    jqd_fd = s_fd.joint_qd.numpy()

    obs_mj = _make_obs(jq_mj, jqd_mj)
    obs_px = _make_obs(jq_px, jqd_px)
    obs_fd = _make_obs(jq_fd, jqd_fd)

    print()
    print("Free base (positions identical across the two PhoenX modes -- only velocities differ):")
    print(f"  trunk pos     MJ = {jq_mj[0:3]}")
    print(f"  trunk pos     PX = {jq_px[0:3]}")
    print(f"  trunk pos     FD = {jq_fd[0:3]}")
    print(f"  trunk quat    MJ = {jq_mj[3:7]}")
    print(f"  trunk quat    PX = {jq_px[3:7]}")
    print(f"  trunk quat    FD = {jq_fd[3:7]}")
    print(f"  trunk lin vel MJ = {jqd_mj[0:3]}")
    print(f"  trunk lin vel PX = {jqd_px[0:3]}")
    print(f"  trunk lin vel FD = {jqd_fd[0:3]}")
    print(f"  trunk ang vel MJ = {jqd_mj[3:6]}")
    print(f"  trunk ang vel PX = {jqd_px[3:6]}")
    print(f"  trunk ang vel FD = {jqd_fd[3:6]}")

    print()
    print("Policy obs tensor blocks (body-frame, exactly what the network sees):")
    print(f"  vel_b  (root lin vel, body)    MJ = {obs_mj['vel_b']}")
    print(f"  vel_b                          PX = {obs_px['vel_b']}")
    print(f"  vel_b                          FD = {obs_fd['vel_b']}")
    print(f"  a_vel_b (root ang vel, body)   MJ = {obs_mj['a_vel_b']}")
    print(f"  a_vel_b                        PX = {obs_px['a_vel_b']}")
    print(f"  a_vel_b                        FD = {obs_fd['a_vel_b']}")
    print(f"  grav_b (projected gravity)     MJ = {obs_mj['grav_b']}")
    print(f"  grav_b                         PX = {obs_px['grav_b']}")
    print(f"  grav_b                         FD = {obs_fd['grav_b']}")
    print(
        f"  PX - MJ orientation drift (radians equivalent on grav_b deviation): "
        f"{np.linalg.norm(obs_mj['grav_b'] - obs_px['grav_b']):.5f}"
    )

    print()
    print(
        f"{'idx':>3} {'name':<32} "
        f"{'q_MJ(deg)':>10} {'q_PX(deg)':>10} {'q_FD(deg)':>10}    "
        f"{'qd_MJ':>9} {'qd_PX':>9} {'qd_FD':>9}    "
        f"{'qd_FD-MJ':>9}"
    )
    n_dofs = min(len(names), len(jq_mj) - 7)
    for i in range(n_dofs):
        qm = math.degrees(float(jq_mj[7 + i]))
        qp = math.degrees(float(jq_px[7 + i]))
        qf = math.degrees(float(jq_fd[7 + i]))
        qdm = float(jqd_mj[6 + i])
        qdp = float(jqd_px[6 + i])
        qdf = float(jqd_fd[6 + i])
        print(
            f"{i:>3} {names[i]:<32} "
            f"{qm:>+10.3f} {qp:>+10.3f} {qf:>+10.3f}    "
            f"{qdm:>+9.4f} {qdp:>+9.4f} {qdf:>+9.4f}    "
            f"{qdf - qdm:>+9.4f}"
        )


def main():
    wp.init()
    for n in (0, 1, 50):
        _print_diff(n)


if __name__ == "__main__":
    main()
