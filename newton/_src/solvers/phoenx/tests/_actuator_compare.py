# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actuator + friction comparison harness: PhoenX vs MuJoCo Warp.

Diagnostic, not a unit test. Run via::

    uv run python -m newton._src.solvers.phoenx.tests._actuator_compare

Motivation: ``robot_policy --solver phoenx --robot g1_29dof`` makes
the G1 humanoid push its toes into the floor, while
``--solver mujoco`` (the default) keeps the foot flat. The bug
report (saved at :mod:`test_robot_policy_parity`) localises this to
the G1 ankle pitch joint -- the lowest-stiffness joint in the rig
(``target_ke = 20 N*m/rad``).

The sweeps in this file isolate the divergence one layer at a time:

1. :func:`sweep_ke`, :func:`sweep_kd`, :func:`sweep_armature`,
   :func:`sweep_substeps` -- single-revolute pendulum, gravity load,
   no contacts. Both solvers track the analytic effective stiffness
   ``ke_eff = m*g*L*cos(q_ss)/|target - q_ss|`` to ~0.2 % of the
   nominal ``target_ke``. **The PD drive is fine in isolation.**

2. :func:`sweep_three_link_leg(with_foot_contact=False)` -- a
   3-DoF planar leg (hip + knee + ankle) holding a G1-like standing
   pose under gravity. Both solvers agree on every joint angle to
   ~0.005 deg. **Chain dynamics are fine.**

3. :func:`sweep_three_link_leg(with_foot_contact=True)` -- same
   leg, but the foot is a box on a ground plane. Drift between
   solvers grows to ~0.5-1.8 deg on the ankle. The bug is in the
   foot contact, not the actuator.

4. :func:`sweep_kinetic_friction` -- block sliding on flat ground
   with v0 = 3 m/s. **PhoenX matches the analytic stop distance
   to 0.5 %; MuJoCo Warp overshoots by ~5-15 % and is still moving
   at the end of the run window.**

5. :func:`sweep_static_friction_threshold` -- block on a tilted
   plane below the analytic break-away angle. **PhoenX correctly
   sticks; MuJoCo Warp creeps at every incline below the threshold.**

6. :func:`sweep_creep_under_load` -- 10 s creep test at
   mu = 0.75, incline = 30 deg (well below atan(0.75) = 36.87 deg).
   **MuJoCo Warp creeps 1.18 m at 0.13 m/s; PhoenX moves 0.6 mm
   and is fully stationary.**

The G1 toes-down behaviour is therefore not a PhoenX PD bug nor a
PhoenX friction bug. It's the *correct* PhoenX stiction locking the
foot to the ground while the MuJoCo-trained policy implicitly
relies on MuJoCo's lossy stiction (foot creeps a small amount each
step, unloading the ankle). When PhoenX enforces stiction strictly,
the entire torque load funnels into the lowest-stiffness DoF
(ankle pitch) and bends it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import warp as wp

import newton

# ---------------------------------------------------------------------------
# Test rig: single revolute joint, no gravity, no contacts
# ---------------------------------------------------------------------------


#: Pendulum arm length (joint axis to body CoM along +x) [m]
_ARM_LEN = 0.5
#: Pendulum body mass [kg]
_MASS = 1.0
#: Gravity magnitude [m/s^2] -- world -Z. Combined with the arm
#: length and mass this creates a constant gravitational torque
#: ``tau_grav = mass * g * arm_len * cos(q)`` around the joint's
#: +y axis, peaking at ``mass*g*arm_len = 4.9 N*m`` at q=0.
_G = 9.81


def _build_pendulum(
    *,
    init_q: float,
    target_ke: float,
    target_kd: float,
    armature: float,
) -> newton.Model:
    """One body hinged to world via a +y revolute joint at the
    origin. Pendulum arm length ``_ARM_LEN`` m along +x, mass
    ``_MASS`` kg. Gravity -Z creates the constant external load.

    ``init_q`` is BOTH the initial joint angle and the PD target
    written into ``joint_target_pos``. With the body starting at
    the target there's no transient "snap"; gravity slowly tips
    the body away from target until the PD spring balances it.
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=armature,
        # Wide-open limits so we never hit them in this sweep.
        limit_lower=-10.0,
        limit_upper=10.0,
        limit_ke=1.0e2,
        limit_kd=1.0e0,
    )

    link = mb.add_link(
        xform=wp.transform(p=wp.vec3(_ARM_LEN, 0.0, 0.0), q=wp.quat_identity()),
        mass=_MASS,
        inertia=((0.01, 0, 0), (0, 0.1, 0), (0, 0, 0.1)),
    )

    mb.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(-_ARM_LEN, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=init_q,
        target_ke=target_ke,
        target_kd=target_kd,
        actuator_mode=newton.JointTargetMode.POSITION,
    )

    mb.add_articulation([0])
    mb.gravity = -_G
    model = mb.finalize()
    q = model.joint_q.numpy()
    q[0] = init_q
    model.joint_q.assign(q)
    return model


def _gravity_torque_at(q: float) -> float:
    """Magnitude of the gravitational torque around the joint axis
    at angle ``q``: ``tau = m*g*L*cos(q)``. Always positive (the
    direction depends on the sign convention but the magnitude is
    what enters the steady-state balance)."""
    return _MASS * _G * _ARM_LEN * abs(math.cos(q))


def _mj_solver(model: newton.Model):
    return newton.solvers.SolverMuJoCo(model, solver="newton", nconmax=8, njmax=8)


def _px_solver(model: newton.Model, *, substeps: int = 4, iters: int = 32):
    return newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=iters,
        velocity_iterations=1,
    )


def _step(
    solver_factory: Callable[[newton.Model], object],
    model_factory: Callable[[], newton.Model],
    n_frames: int,
    dt: float,
    *,
    target_pos: float,
) -> tuple[float, float]:
    """Step ``n_frames`` with the pendulum + gravity rig from
    :func:`_build_pendulum`. The PD target is held at
    ``target_pos`` for every frame. Returns ``(q_ss, qd_ss)``
    averaged over the last 25 % of frames so any transient has
    decayed.
    """
    model = model_factory()
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    if control.joint_target_pos is not None:
        tgt = np.zeros(int(model.joint_dof_count), dtype=np.float32)
        tgt[0] = target_pos
        control.joint_target_pos.assign(tgt)

    jq_buf = wp.zeros(int(model.joint_coord_count), dtype=wp.float32, device=model.device)
    jqd_buf = wp.zeros(int(model.joint_dof_count), dtype=wp.float32, device=model.device)
    q_traj = np.empty(n_frames, dtype=np.float32)
    qd_traj = np.empty(n_frames, dtype=np.float32)

    for i in range(n_frames):
        s0.clear_forces()
        # No contacts; both solvers get ``contacts=None``.
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        newton.eval_ik(model, s0, jq_buf, jqd_buf)
        q_traj[i] = float(jq_buf.numpy()[0])
        qd_traj[i] = float(jqd_buf.numpy()[0])

    n_avg = max(1, n_frames // 4)
    return float(q_traj[-n_avg:].mean()), float(qd_traj[-n_avg:].mean())


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def _eff_ke(q_target: float, q_ss: float) -> float:
    """At equilibrium: ``ke_eff * (q_target - q_ss) = tau_grav(q_ss)``.
    Returns the inferred ``ke_eff`` [N*m/rad]."""
    err = q_target - q_ss
    if abs(err) < 1.0e-9:
        return float("inf")
    return _gravity_torque_at(q_ss) / abs(err)


def sweep_ke():
    """Vary ``target_ke``, hold ``target_kd`` fixed.

    Body starts at the PD target (= 0 rad horizontal). Gravity
    pulls the arm down (positive q rotates +x toward +z, gravity
    creates -y torque, so q drifts negative). Steady state:
    ``ke_nominal * |q_target - q_ss|  ==  m*g*L*cos(q_ss)``.

    Print MJ and PX side-by-side. ``ratio = eff_ke / target_ke``:
    1.00 means the solver realises the nominal gain; < 1.00 means
    it under-shoots and the joint settles further from the target.
    """
    print("=" * 92)
    print("Sweep: target_ke (body at q=0, gravity pulls down)")
    print(f"target_kd = 5, gravity load tau_max = {_MASS * _G * _ARM_LEN:.2f} N*m at q=0")
    print("=" * 92)

    target_kd = 5.0
    target_pos = 0.0
    init_q = 0.0
    armature = 0.1
    dt = 1.0 / 200.0
    n_frames = 2000  # 10 s

    print(
        f"\n{'ke_nom':>8} {'MJ q_ss(deg)':>12} {'MJ eff_ke':>10} {'MJ ratio':>9}    "
        f"{'PX q_ss(deg)':>12} {'PX eff_ke':>10} {'PX ratio':>9}"
    )
    for target_ke in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0):

        def make(_ke=target_ke, _kd=target_kd, _arm=armature, _q0=init_q):
            return _build_pendulum(
                init_q=_q0,
                target_ke=_ke,
                target_kd=_kd,
                armature=_arm,
            )

        q_mj, _ = _step(_mj_solver, make, n_frames, dt, target_pos=target_pos)
        q_px, _ = _step(_px_solver, make, n_frames, dt, target_pos=target_pos)
        eff_mj = _eff_ke(target_pos, q_mj)
        eff_px = _eff_ke(target_pos, q_px)
        ratio_mj = eff_mj / target_ke
        ratio_px = eff_px / target_ke
        print(
            f"{target_ke:>8.1f} {math.degrees(q_mj):>+12.3f} {eff_mj:>10.3f} {ratio_mj:>9.3f}    "
            f"{math.degrees(q_px):>+12.3f} {eff_px:>10.3f} {ratio_px:>9.3f}"
        )


def sweep_kd():
    """Vary ``target_kd`` with fixed ``target_ke = 20`` and gravity
    load. ``q_ss`` should be invariant in kd (the spring balance is
    purely positional); ``qd_ss`` should be 0 in both solvers."""
    print()
    print("=" * 92)
    print("Sweep: target_kd; target_ke = 20")
    print("=" * 92)

    target_ke = 20.0
    target_pos = 0.0
    init_q = 0.0
    armature = 0.1
    dt = 1.0 / 200.0
    n_frames = 2000

    print(f"\n{'kd':>6} {'MJ q_ss(deg)':>12} {'MJ qd_ss':>10}    {'PX q_ss(deg)':>12} {'PX qd_ss':>10}")
    for target_kd in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0):

        def make(_kd=target_kd):
            return _build_pendulum(
                init_q=init_q,
                target_ke=target_ke,
                target_kd=_kd,
                armature=armature,
            )

        q_mj, qd_mj = _step(_mj_solver, make, n_frames, dt, target_pos=target_pos)
        q_px, qd_px = _step(_px_solver, make, n_frames, dt, target_pos=target_pos)
        print(
            f"{target_kd:>6.1f} {math.degrees(q_mj):>+12.3f} {qd_mj:>+10.5f}    "
            f"{math.degrees(q_px):>+12.3f} {qd_px:>+10.5f}"
        )


def sweep_armature():
    """Vary ``armature`` with target_ke = 20, target_kd = 5, gravity
    load. ``q_ss`` is independent of inertia (spring balance is
    kinematic). If PhoenX's q_ss shifts with armature the constraint
    is leaking inertia into the spring."""
    print()
    print("=" * 92)
    print("Sweep: armature; target_ke = 20, target_kd = 5")
    print("=" * 92)

    target_ke = 20.0
    target_kd = 5.0
    target_pos = 0.0
    init_q = 0.0
    dt = 1.0 / 200.0
    n_frames = 2000

    print(f"\n{'arm':>6} {'MJ q_ss(deg)':>12} {'MJ eff_ke':>10}    {'PX q_ss(deg)':>12} {'PX eff_ke':>10}")
    for arm in (0.0, 0.05, 0.1, 0.5, 1.0):

        def make(_a=arm):
            return _build_pendulum(
                init_q=init_q,
                target_ke=target_ke,
                target_kd=target_kd,
                armature=_a,
            )

        q_mj, _ = _step(_mj_solver, make, n_frames, dt, target_pos=target_pos)
        q_px, _ = _step(_px_solver, make, n_frames, dt, target_pos=target_pos)
        eff_mj = _eff_ke(target_pos, q_mj)
        eff_px = _eff_ke(target_pos, q_px)
        print(
            f"{arm:>6.3f} {math.degrees(q_mj):>+12.3f} {eff_mj:>10.3f}    {math.degrees(q_px):>+12.3f} {eff_px:>10.3f}"
        )


def sweep_substeps():
    """Vary PhoenX substeps and solver_iterations. MuJoCo alone is
    reported as a reference (no analogous knob). If PhoenX's
    effective ke depends on substeps, the PD discretisation is off
    (typically: impulse applied once per outer step instead of once
    per substep, integrated torque ∝ 1/N_substeps)."""
    print()
    print("=" * 92)
    print("PhoenX substeps / iters sweep; target_ke = 20, target_kd = 5")
    print("=" * 92)

    target_ke = 20.0
    target_kd = 5.0
    armature = 0.1
    target_pos = 0.0
    init_q = 0.0
    dt = 1.0 / 200.0
    n_frames = 2000

    def make():
        return _build_pendulum(
            init_q=init_q,
            target_ke=target_ke,
            target_kd=target_kd,
            armature=armature,
        )

    q_mj, _ = _step(_mj_solver, make, n_frames, dt, target_pos=target_pos)
    eff_mj = _eff_ke(target_pos, q_mj)
    print(f"\n  MuJoCo (reference): q_ss = {math.degrees(q_mj):+.3f} deg  (eff_ke = {eff_mj:.3f})")
    print(f"\n{'substeps':>10} {'iters':>7} {'PX q_ss(deg)':>14} {'eff_ke':>10}")
    for substeps, iters in [
        (1, 16),
        (2, 16),
        (4, 16),
        (8, 16),
        (16, 16),
        (4, 32),
        (4, 64),
        (4, 128),
    ]:
        q_px, _ = _step(
            lambda m, s=substeps, it=iters: _px_solver(m, substeps=s, iters=it),
            make,
            n_frames,
            dt,
            target_pos=target_pos,
        )
        eff_px = _eff_ke(target_pos, q_px)
        print(f"{substeps:>10d} {iters:>7d} {math.degrees(q_px):>+14.3f} {eff_px:>10.3f}")


# ---------------------------------------------------------------------------
# 3-link planar leg: hip + knee + ankle, fixed base
# ---------------------------------------------------------------------------


def _build_three_link_leg(
    *,
    hip_q: float,
    knee_q: float,
    ankle_q: float,
    hip_ke: float,
    knee_ke: float,
    ankle_ke: float,
    hip_kd: float,
    knee_kd: float,
    ankle_kd: float,
    armature: float = 0.1,
    with_foot_contact: bool = False,
) -> newton.Model:
    """Three serial revolute joints (all +y axis) -> three rigid
    links along +x in the rest pose.

    Approximates a sagittal-plane G1 leg: ``hip_q`` ~ -0.1 rad,
    ``knee_q`` ~ +0.3 rad, ``ankle_q`` ~ -0.2 rad with the YAML's
    PD gains (200 / 200 / 20 N*m/rad). The base is fixed via the
    natural ``parent=-1`` to world (no add_link auto-free-joint
    because we use add_joint_revolute(parent=-1)).
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=armature,
        limit_lower=-10.0,
        limit_upper=10.0,
        limit_ke=1.0e2,
        limit_kd=1.0e0,
    )

    L = 0.3  # link length
    m = 1.0
    inertia = ((0.005, 0, 0), (0, 0.05, 0), (0, 0, 0.05))

    thigh = mb.add_link(
        xform=wp.transform(p=wp.vec3(L * 0.5, 0.0, 0.0), q=wp.quat_identity()),
        mass=m,
        inertia=inertia,
    )
    shank = mb.add_link(
        xform=wp.transform(p=wp.vec3(L * 1.5, 0.0, 0.0), q=wp.quat_identity()),
        mass=m,
        inertia=inertia,
    )
    foot = mb.add_link(
        xform=wp.transform(p=wp.vec3(L * 2.5, 0.0, 0.0), q=wp.quat_identity()),
        mass=m,
        inertia=inertia,
    )

    hip = mb.add_joint_revolute(
        parent=-1,
        child=thigh,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(-L * 0.5, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=hip_q,
        target_ke=hip_ke,
        target_kd=hip_kd,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    knee = mb.add_joint_revolute(
        parent=thigh,
        child=shank,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(L * 0.5, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-L * 0.5, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=knee_q,
        target_ke=knee_ke,
        target_kd=knee_kd,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    ankle = mb.add_joint_revolute(
        parent=shank,
        child=foot,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(L * 0.5, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-L * 0.5, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=ankle_q,
        target_ke=ankle_ke,
        target_kd=ankle_kd,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    mb.add_articulation([hip, knee, ankle])
    mb.gravity = -_G

    if with_foot_contact:
        # Box "foot" attached to the leaf body, touching the ground.
        # Same proportions as a G1 foot (10 cm long, 5 cm wide, 2 cm high).
        mb.default_shape_cfg.mu = 0.75
        mb.add_shape_box(
            foot,
            xform=wp.transform(p=wp.vec3(0.05, 0.0, -0.01), q=wp.quat_identity()),
            hx=0.05,
            hy=0.025,
            hz=0.01,
        )
        mb.add_ground_plane()

    model = mb.finalize()
    q = model.joint_q.numpy()
    q[0] = hip_q
    q[1] = knee_q
    q[2] = ankle_q
    model.joint_q.assign(q)
    return model


def sweep_three_link_leg(with_foot_contact: bool = False):
    """Hold the G1-like (hip=-0.1, knee=+0.3, ankle=-0.2) standing
    leg pose under gravity for 5 s. Compare per-joint steady-state
    angle between MuJoCo and PhoenX.

    Sweep ankle_ke -- the bug should worsen as the ankle gets
    softer relative to the load. ``with_foot_contact=True`` adds
    a foot-shaped box and a ground plane so the foot supports the
    leg's weight (the only thing left differentiating this from
    the no-contact run that already matched between solvers).
    """
    print()
    print("=" * 92)
    label = "WITH foot↔ground contact" if with_foot_contact else "no contacts (gravity only)"
    print(f"3-link leg, hold standing pose [{label}], fixed base")
    print("hip_ke=200, knee_ke=200, vary ankle_ke")
    print("=" * 92)

    hip_q, knee_q, ankle_q = -0.1, 0.3, -0.2
    hip_ke, knee_ke = 200.0, 200.0
    hip_kd, knee_kd, ankle_kd = 5.0, 5.0, 2.0
    armature = 0.1
    dt = 1.0 / 200.0
    n_frames = 1000  # 5 s

    print(
        f"\n{'ankle_ke':>10} {'MJ ankle(deg)':>14} {'MJ knee(deg)':>14} {'MJ hip(deg)':>13}    "
        f"{'PX ankle(deg)':>14} {'PX knee(deg)':>14} {'PX hip(deg)':>13}    {'ankle drift':>12}"
    )
    for ankle_ke in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0):

        def make(_aank=ankle_ke, _wfc=with_foot_contact):
            return _build_three_link_leg(
                hip_q=hip_q,
                knee_q=knee_q,
                ankle_q=ankle_q,
                hip_ke=hip_ke,
                knee_ke=knee_ke,
                ankle_ke=_aank,
                hip_kd=hip_kd,
                knee_kd=knee_kd,
                ankle_kd=ankle_kd,
                armature=armature,
                with_foot_contact=_wfc,
            )

        # Step both solvers, read final joint angles via eval_ik.
        def run(solver_factory, _wfc=with_foot_contact):
            model = make()
            solver = solver_factory(model)
            s0 = model.state()
            s1 = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            ctl = model.control()
            tgt = np.zeros(int(model.joint_dof_count), dtype=np.float32)
            tgt[0] = hip_q
            tgt[1] = knee_q
            tgt[2] = ankle_q
            ctl.joint_target_pos.assign(tgt)
            jq = wp.zeros(int(model.joint_coord_count), dtype=wp.float32, device=model.device)
            jqd = wp.zeros(int(model.joint_dof_count), dtype=wp.float32, device=model.device)
            traj = np.empty((n_frames, 3), dtype=np.float32)
            is_phoenx = isinstance(solver, newton.solvers.SolverPhoenX)
            contacts = model.contacts() if (_wfc and is_phoenx) else None
            for i in range(n_frames):
                s0.clear_forces()
                if _wfc and is_phoenx:
                    model.collide(s0, contacts)
                solver.step(s0, s1, ctl, contacts if (_wfc and is_phoenx) else None, dt)
                s0, s1 = s1, s0
                newton.eval_ik(model, s0, jq, jqd)
                traj[i] = jq.numpy()[:3]
            return tuple(float(x) for x in traj[-200:].mean(axis=0))

        h_mj, k_mj, a_mj = run(_mj_solver)
        h_px, k_px, a_px = run(_px_solver)
        ankle_drift = a_mj - a_px
        print(
            f"{ankle_ke:>10.1f} "
            f"{math.degrees(a_mj):>+14.3f} {math.degrees(k_mj):>+14.3f} {math.degrees(h_mj):>+13.3f}    "
            f"{math.degrees(a_px):>+14.3f} {math.degrees(k_px):>+14.3f} {math.degrees(h_px):>+13.3f}    "
            f"{math.degrees(ankle_drift):>+12.3f}"
        )


# ---------------------------------------------------------------------------
# Friction sweeps -- the second half of the bug story is at the
# foot-ground contact, not the actuator.
# ---------------------------------------------------------------------------


def _build_block_on_plane(mu: float, *, incline_deg: float = 0.0) -> newton.Model:
    """Single 1 m^3 box on a ground plane, mu shared. ``incline_deg``
    rotates *gravity* (not the plane) -- numerically identical to a
    tilted plane and avoids placing geometry off-axis.

    A free body (added via ``add_body``) gets an auto-attached free
    joint. The block's CoG starts at z = 0.5 m + 1 mm; both solvers
    settle the contact at z ≈ 0.5 m on a flat plane, then friction
    decides whether the block slides or sticks.
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu)
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5 + 1.0e-3), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)),
    )
    mb.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    mb.add_ground_plane()
    # Gravity tilted around +y by ``incline_deg``: equivalent to a
    # plane tilted around +y by the same angle. With m*g cos(theta)
    # normal load and m*g sin(theta) tangential, stiction breaks at
    # tan(theta) > mu.
    a = math.radians(incline_deg)
    mb.gravity = -_G  # the magnitude; we re-decompose via vec3 below
    model = mb.finalize()
    # Rewrite gravity as a vec3 so the tilt direction matters.
    model.set_gravity((-_G * math.sin(a), 0.0, -_G * math.cos(a)))
    return model


def _step_block(
    solver_factory: Callable[[newton.Model], object],
    model_factory: Callable[[], newton.Model],
    n_frames: int,
    dt: float,
    *,
    initial_v0_x: float = 0.0,
) -> tuple[float, float, float]:
    """Step ``n_frames`` of the block-on-plane scene. Returns
    ``(final_x, final_z, final_speed)`` of the body."""
    model = model_factory()
    if initial_v0_x != 0.0:
        # Free joint qd order = (linear, angular). Linear x is index 0.
        model.joint_qd.assign(np.array([initial_v0_x, 0, 0, 0, 0, 0], dtype=np.float32))
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    if initial_v0_x != 0.0:
        s0.joint_qd.assign(np.array([initial_v0_x, 0, 0, 0, 0, 0], dtype=np.float32))
        s1.joint_qd.assign(np.array([initial_v0_x, 0, 0, 0, 0, 0], dtype=np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    is_phoenx = isinstance(solver, newton.solvers.SolverPhoenX)
    contacts = model.contacts() if is_phoenx else None

    for _i in range(n_frames):
        s0.clear_forces()
        if is_phoenx:
            model.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts if is_phoenx else None, dt)
        s0, s1 = s1, s0

    bq = s0.body_q.numpy()[0]
    bqd = s0.body_qd.numpy()[0]
    return float(bq[0]), float(bq[2]), float(np.linalg.norm(bqd))


def sweep_kinetic_friction():
    """Block sliding on flat ground, v0 = 3 m/s along +x. Compare
    stop distance between solvers across a sweep of mu. Analytic
    expectation: ``d = v0**2 / (2 * mu * g)``."""
    print()
    print("=" * 92)
    print("Kinetic friction: block stop distance (v0 = 3 m/s, flat ground)")
    print("=" * 92)
    v0 = 3.0
    dt = 1.0 / 200.0

    print(f"\n{'mu':>5} {'analytic (m)':>12} {'MJ stop (m)':>11} {'MJ |v|':>9}    {'PX stop (m)':>11} {'PX |v|':>9}")
    for mu in (0.1, 0.3, 0.5, 0.75, 1.0, 2.0):
        analytic = (v0 * v0) / (2.0 * mu * _G)
        # Run long enough for both solvers to fully stop at the lowest mu.
        t_stop = v0 / (mu * _G)
        n_frames = int(np.ceil((2.5 * t_stop + 1.0) / dt))

        def make(_mu=mu):
            return _build_block_on_plane(_mu)

        x_mj, _, v_mj = _step_block(_mj_solver, make, n_frames, dt, initial_v0_x=v0)
        x_px, _, v_px = _step_block(_px_solver, make, n_frames, dt, initial_v0_x=v0)
        print(f"{mu:>5.2f} {analytic:>12.3f} {x_mj:>11.3f} {v_mj:>9.4f}    {x_px:>11.3f} {v_px:>9.4f}")


def sweep_static_friction_threshold():
    """Block on a flat ground; gravity tilted by ``incline_deg``.
    Stiction holds when ``tan(theta) < mu``, slips otherwise. Sweep
    incline angles around the analytic threshold for mu = 0.5
    (``theta_crit = atan(0.5) ≈ 26.57 deg``) and report final
    body x-speed: ~0 means stuck, > 0 means slipping."""
    print()
    print("=" * 92)
    print("Static friction threshold: incline sweep at mu = 0.5")
    print(f"Analytic break-away angle = atan(0.5) = {math.degrees(math.atan(0.5)):.2f} deg")
    print("=" * 92)
    mu = 0.5
    dt = 1.0 / 200.0
    n_frames = 600  # 3 s

    print(
        f"\n{'incline':>8} {'expect':>8} {'MJ x':>9} {'MJ |v|':>9} {'MJ verdict':>14}    {'PX x':>9} {'PX |v|':>9} {'PX verdict':>14}"
    )
    crit = math.degrees(math.atan(mu))
    for incline_deg in (15.0, 20.0, 25.0, 26.0, 27.0, 28.0, 30.0, 35.0):
        expect = "stick" if incline_deg < crit else "slip"

        def make(_inc=incline_deg):
            return _build_block_on_plane(mu, incline_deg=_inc)

        x_mj, _, v_mj = _step_block(_mj_solver, make, n_frames, dt)
        x_px, _, v_px = _step_block(_px_solver, make, n_frames, dt)

        def verdict(v: float) -> str:
            if v < 0.05:
                return "stick"
            if v > 0.5:
                return "slip"
            return "creep"

        print(
            f"{incline_deg:>8.1f} {expect:>8} {x_mj:>+9.3f} {v_mj:>9.4f} {verdict(v_mj):>14}    "
            f"{x_px:>+9.3f} {v_px:>9.4f} {verdict(v_px):>14}"
        )


def sweep_creep_under_load():
    """Block on a tilted plane below the breakaway angle. Stiction
    *should* hold but micro-creep may accumulate over long runs.
    This is the closest single-body proxy for the G1 ankle bug:
    the ankle puts a constant tangential load on the foot, friction
    is supposed to hold, and slow creep is what produces the drift.

    mu = 0.75 (matching robot_policy), incline_deg = 30 (well below
    atan(0.75) = 36.87 deg). 10 s window."""
    print()
    print("=" * 92)
    print("Creep under sub-critical load: mu = 0.75, incline = 30 deg, 10 s")
    print("=" * 92)
    mu = 0.75
    incline_deg = 30.0
    dt = 1.0 / 200.0
    n_frames = 2000  # 10 s
    print(f"\nBreakaway angle = atan(0.75) = {math.degrees(math.atan(mu)):.2f} deg (block well below).")
    print("Block should stay static; slow creep would mean the static-friction discretisation leaks.")
    print(f"\n{'solver':<10} {'final x (m)':>12} {'final |v|':>10} {'creep diag':>12}")

    def make():
        return _build_block_on_plane(mu, incline_deg=incline_deg)

    x_mj, _, v_mj = _step_block(_mj_solver, make, n_frames, dt)
    x_px, _, v_px = _step_block(_px_solver, make, n_frames, dt)
    print(f"{'MuJoCo':<10} {x_mj:>+12.5f} {v_mj:>10.5f} {abs(x_mj):>12.5f}")
    print(f"{'PhoenX':<10} {x_px:>+12.5f} {v_px:>10.5f} {abs(x_px):>12.5f}")


def main():
    wp.init()
    sweep_ke()
    sweep_kd()
    sweep_armature()
    sweep_substeps()
    sweep_three_link_leg(with_foot_contact=False)
    sweep_three_link_leg(with_foot_contact=True)
    sweep_kinetic_friction()
    sweep_static_friction_threshold()
    sweep_creep_under_load()
    print()
    print("Reading the tables:")
    print("  - 'eff_ke' = m*g*L*cos(q_ss) / |target - q_ss|  (the spring stiffness")
    print("    you'd compute from the steady-state torque balance).")
    print("  - 'ratio'  = eff_ke / nominal target_ke. 1.00 means the solver")
    print("    realises exactly the nominal gain. < 1.00 means the solver")
    print("    is producing less restoring torque than asked for.")
    print("  - 'ankle drift' (3-link leg) = MJ_ankle - PX_ankle in degrees.")
    print("    The G1 bug (toes pushing into floor) is a non-zero ankle drift")
    print("    here, growing as ankle_ke gets smaller relative to the load.")


if __name__ == "__main__":
    main()
