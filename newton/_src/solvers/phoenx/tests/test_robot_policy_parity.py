# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver parity tests motivated by the ``example_robot_policy`` bug
report: running ``robot_policy --solver phoenx`` (default robot is
``g1_29dof``) makes the humanoid push its toes into the floor.

The "toes" in question are the **G1 ankle pitch joints** -- the G1's
foot is rigid (no separate toe DoF), so toe-down corresponds to a
spurious negative drift on ``left_ankle_pitch_joint`` /
``right_ankle_pitch_joint``. The standing-pose YAML sets those at
init = -0.2 rad (rotated to keep the foot flat under the +0.3 rad
knee bend) with ``target_ke = 20 N*m/rad`` -- the lowest gain in the
whole rig. So any small init-offset, sign-flip, or contact-induced
perturbation that PhoenX interprets differently from MuJoCo first
shows up as the ankles giving way.

The Anymal walker (a quadruped with passive feet) does *not*
exhibit the bug; G1 does. Tests in this module therefore target G1
specifically.

Tests:

1. :class:`TestSlidingBlockFrictionParity` -- block with v0 = 3 m/s
   on a ground plane, mu = 0.75. Confirms friction is the same for
   both solvers in the simplest possible case (no actuators, no
   articulation).

2. :class:`TestFootPrimitiveSettleParity` -- sphere/capsule/box
   dropped onto a ground plane. Isolates "does the same primitive
   produce the same contact normal/penetration in both contact
   pipelines?".

3. :class:`TestG1HoldPoseParity` -- the actual G1 model from the
   ``robot_policy`` example, held at its standing-pose joint targets
   for 2 s with the YAML's PD gains and armature. Catches:

   * **Foot z drift** between solvers (one solver letting the foot
     dip while the other holds it).
   * **Ankle-pitch angle divergence** -- the smoking gun for the
     toes-down bug. ``targets`` for the ankle pitch are init = -0.2
     rad; if PhoenX drifts these toward +0.2 / 0.0 / etc. while
     MuJoCo holds them at -0.2, the foot rolls forward and the toes
     punch into the floor.

The tests are deliberately tolerant: PhoenX and MuJoCo Warp use
different stabilisation schemes (Baumgarte vs solref/solimp) and
different friction-cone discretisations, so we expect a few percent
of disagreement even on identical setups. What we want to catch is
*qualitative* divergence -- one solver letting the ankle drift by
several degrees while the other clamps it.

The module also exposes ``_diagnose_robot_contacts`` (run via
``python -m newton._src.solvers.phoenx.tests.test_robot_policy_parity``)
which prints the PhoenX-side contact set for the G1 standing pose so
the user can sanity-check what shapes are colliding and where.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

import newton

try:
    import mujoco  # noqa: F401
    import mujoco_warp  # noqa: F401

    _HAS_MJW = True
except ImportError:
    _HAS_MJW = False


# Friction value used by ``example_robot_policy.py``
# (``default_shape_cfg.mu = 0.75``). Pinned here so the tests document
# the exact case the user is running into.
_ROBOT_POLICY_MU = 0.75


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


def _mj_solver(model: newton.Model):
    return newton.solvers.SolverMuJoCo(model, solver="newton", nconmax=64, njmax=64)


def _px_solver(model: newton.Model):
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=16,
        velocity_iterations=1,
    )


def _run_with_contacts(
    solver_factory: Callable[[newton.Model], object],
    model_factory: Callable[[], newton.Model],
    n_frames: int,
    dt: float,
    initial_joint_qd: np.ndarray | None = None,
    target_pos: np.ndarray | None = None,
    record_joint_q: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Step ``n_frames`` of a contact-heavy scene and return
    ``(body_q[n_frames, body_count, 7], body_qd[n_frames, body_count, 6],
    joint_q[n_frames, joint_coord_count] | None)``.

    MuJoCo runs its own contact pipeline (``contacts=None``); PhoenX
    requires :func:`newton.Model.collide` per step.

    ``initial_joint_qd`` is written into ``model.joint_qd`` *before*
    the solver is built and into both ``state.joint_qd`` arrays
    *after* construction (MuJoCo's per-step coordinate sync reads
    ``state.joint_qd``). For a free-base body the 6 qd entries are
    ``(vx, vy, vz, wx, wy, wz)`` -- Newton's FREE-joint convention is
    (linear, angular) per ``ModelBuilder.add_body``.

    ``target_pos`` is broadcast to ``control.joint_target_pos`` once.

    When ``record_joint_q`` is set we run :func:`newton.eval_ik` after
    every step so per-joint angles can be compared between solvers
    (PhoenX maintains body_q as the source of truth and only fills
    joint_q when state.joint_q is set on entry; eval_ik gives a clean
    common readout regardless).
    """
    model = model_factory()
    if initial_joint_qd is not None:
        model.joint_qd.assign(initial_joint_qd.astype(np.float32))
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    if initial_joint_qd is not None:
        # ``state.joint_qd`` is a freshly allocated zero array; the
        # state copy of joint_qd is what the solvers read each step.
        # (``eval_fk`` propagates joint_q/joint_qd into body_q/body_qd
        # but does *not* copy joint_qd into the state.)
        s0.joint_qd.assign(initial_joint_qd.astype(np.float32))
        s1.joint_qd.assign(initial_joint_qd.astype(np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    control = model.control()
    if target_pos is not None and control.joint_target_pos is not None:
        control.joint_target_pos.assign(target_pos.astype(np.float32))

    is_phoenx = isinstance(solver, newton.solvers.SolverPhoenX)
    contacts = model.contacts() if is_phoenx else None

    bq_traj = np.empty((n_frames, model.body_count, 7), dtype=np.float32)
    bqd_traj = np.empty((n_frames, model.body_count, 6), dtype=np.float32)
    jq_traj: np.ndarray | None = None
    if record_joint_q and int(model.joint_coord_count) > 0:
        jq_traj = np.empty((n_frames, int(model.joint_coord_count)), dtype=np.float32)
        jq_buf = wp.zeros(int(model.joint_coord_count), dtype=wp.float32, device=model.device)
        jqd_buf = wp.zeros(int(model.joint_dof_count), dtype=wp.float32, device=model.device)

    for i in range(n_frames):
        s0.clear_forces()
        if is_phoenx:
            model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts if is_phoenx else None, dt)
        s0, s1 = s1, s0
        bq_traj[i] = s0.body_q.numpy()
        bqd_traj[i] = s0.body_qd.numpy()
        if jq_traj is not None:
            newton.eval_ik(model, s0, jq_buf, jqd_buf)
            jq_traj[i] = jq_buf.numpy()

    return bq_traj, bqd_traj, jq_traj


# ---------------------------------------------------------------------------
# 1. Sliding block: pure friction, no actuators
# ---------------------------------------------------------------------------


def _build_sliding_block(mu: float, v0: float) -> newton.Model:
    """Single 1 m^3 box on a ground plane, ``mu`` shared between
    both shapes, +X initial velocity ``v0``. Tests pure kinetic
    friction in isolation.

    ``builder.gravity`` is left at the default ``(0, 0, -9.81)`` so
    the box's normal load is ``m * g`` -- the analytic stop distance
    is ``v0**2 / (2 * mu * g)``.
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=1000.0,
        mu=mu,
    )

    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5 + 1.0e-3), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)),
    )
    mb.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    mb.add_ground_plane()
    mb.gravity = -9.81
    model = mb.finalize()
    # Stash v0 on the model so the harness can read it back; we can't
    # set body_qd at builder level for free joints (they seed via
    # joint_qd which has shape (joint_dof_count,)).
    model._test_initial_v0 = v0  # type: ignore[attr-defined]
    return model


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Parity tests run on CUDA only.",
)
class TestSlidingBlockFrictionParity(unittest.TestCase):
    """Both solvers must produce the same kinetic-friction stopping
    distance for a block with the same ``mu`` and initial velocity.

    The box has ``mu = 0.75`` (matching ``example_robot_policy``'s
    default), starts with ``v0 = 3 m/s`` along +X, and gravity is the
    only normal load. Analytic stop distance is
    ``v0**2 / (2 mu g) = 0.61 m``.

    Tolerance is wide (15 % between solvers, 25 % vs analytic) because
    each solver brings its own warm-up transient; the point of this
    test is to catch a *systematic* divergence, not bit-for-bit
    parity.
    """

    def test_stop_distance_matches(self) -> None:
        v0 = 3.0
        mu = _ROBOT_POLICY_MU
        dt = 1.0 / 200.0
        # ``2.5 t_stop + safety`` matches test_friction.TestKineticFrictionStopDistance.
        t_stop = v0 / (mu * 9.81)
        n_frames = int(np.ceil((2.5 * t_stop + 1.0) / dt))

        def factory():
            return _build_sliding_block(mu, v0)

        # Single free-base body -> joint_qd has 6 entries laid out
        # (linear, angular) per Newton's FREE-joint convention
        # (see ``ModelBuilder.add_body``). Linear x velocity is the
        # first 3 entries, angular the last 3.
        initial_qd = np.array([v0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bq_mj, _, _ = _run_with_contacts(_mj_solver, factory, n_frames, dt, initial_joint_qd=initial_qd)
        bq_px, _, _ = _run_with_contacts(_px_solver, factory, n_frames, dt, initial_joint_qd=initial_qd)

        stop_mj = float(bq_mj[-1, 0, 0])
        stop_px = float(bq_px[-1, 0, 0])
        analytic = (v0 * v0) / (2.0 * mu * 9.81)

        self.assertAlmostEqual(
            stop_mj,
            stop_px,
            delta=0.15 * analytic,
            msg=(
                f"stop distance disagreement: MuJoCo={stop_mj:.3f} m PhoenX={stop_px:.3f} m analytic={analytic:.3f} m"
            ),
        )
        # Both solvers should land within ~25 % of the analytic stop.
        for name, x in (("MuJoCo", stop_mj), ("PhoenX", stop_px)):
            rel_err = abs(x - analytic) / analytic
            self.assertLess(
                rel_err,
                0.25,
                f"{name} stop {x:.3f} m vs analytic {analytic:.3f} m (err {rel_err:+.0%})",
            )


# ---------------------------------------------------------------------------
# 2. Foot primitive settle: contact-pipeline parity
# ---------------------------------------------------------------------------


def _build_primitive_drop(shape_kind: str, drop_height: float) -> newton.Model:
    """Drop a single primitive (sphere | capsule | box) from
    ``drop_height`` above a ground plane. ``mu = 0.75`` shared.

    Each primitive's contact-with-ground gives the two collision
    pipelines a chance to disagree on contact count and normal:

    * sphere: 1 point, normal = world +Z.
    * capsule (axis +Z, dropped from h): 1 point at the lower hemisphere.
    * box: up to 4 corner contacts -- the most sensitive to GJK / SAT
      differences.
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=1000.0,
        mu=_ROBOT_POLICY_MU,
    )
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_height), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)),
    )
    if shape_kind == "sphere":
        mb.add_shape_sphere(body, radius=0.1)
    elif shape_kind == "capsule":
        mb.add_shape_capsule(body, radius=0.05, half_height=0.1)
    elif shape_kind == "box":
        mb.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    else:
        raise ValueError(f"unknown shape: {shape_kind}")
    mb.add_ground_plane()
    mb.gravity = -9.81
    return mb.finalize()


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Parity tests run on CUDA only.",
)
class TestFootPrimitiveSettleParity(unittest.TestCase):
    """Each primitive must settle to the same height in both solvers.

    Tolerance is 5 mm: with shape radii of 5-10 cm that is well below
    "the foot is pushing through the floor" but generous enough to
    absorb each solver's warm-up overshoot.
    """

    SHAPES_AND_EXPECTED_REST_Z = (
        # (kind, expected rest z of body origin -- equal to the half-extent
        # along -Z, since each primitive starts axis-aligned)
        ("sphere", 0.1),
        ("capsule", 0.15),  # 0.05 radius + 0.1 half-height along +Z
        ("box", 0.1),
    )

    def test_each_primitive_settles_in_both_solvers(self) -> None:
        dt = 1.0 / 200.0
        n_frames = 200  # 1 s -- plenty of time to settle

        for kind, expected_z in self.SHAPES_AND_EXPECTED_REST_Z:
            with self.subTest(shape=kind):

                def factory(_kind=kind):
                    return _build_primitive_drop(_kind, drop_height=0.5)

                bq_mj, _, _ = _run_with_contacts(_mj_solver, factory, n_frames, dt)
                bq_px, _, _ = _run_with_contacts(_px_solver, factory, n_frames, dt)

                z_mj = float(bq_mj[-1, 0, 2])
                z_px = float(bq_px[-1, 0, 2])

                self.assertAlmostEqual(
                    z_mj,
                    z_px,
                    delta=5.0e-3,
                    msg=(
                        f"{kind}: settle z disagreement MuJoCo={z_mj:.4f} "
                        f"PhoenX={z_px:.4f} (expected ~{expected_z:.3f})"
                    ),
                )
                # Both must actually be near the analytic rest height
                # (the floor is at z=0, the body origin sits one half-extent
                # above the contact surface).
                for name, z in (("MuJoCo", z_mj), ("PhoenX", z_px)):
                    self.assertAlmostEqual(
                        z,
                        expected_z,
                        delta=0.02,
                        msg=f"{name} {kind}: rest z {z:.4f} vs expected {expected_z:.3f}",
                    )


# ---------------------------------------------------------------------------
# 3. Real robot: hold standing pose, foot z parity
# ---------------------------------------------------------------------------


def _g1_29dof_yaml() -> dict:
    """Load the YAML config the ``robot_policy`` example reads for
    the G1 29-DoF rig."""
    import yaml  # noqa: PLC0415

    import newton.utils  # noqa: PLC0415

    asset_dir = newton.utils.download_asset("unitree_g1")
    with open(asset_dir / "rl_policies" / "g1_29dof.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _g1_robot_model() -> newton.Model:
    """Build the G1 29-DoF model the same way ``example_robot_policy``
    does for ``--robot g1_29dof`` (the default robot when no
    ``--robot`` flag is given).

    Mirrors the example one-for-one:

    * USD: ``unitree_g1/usd/g1_isaac.usd`` (the path the example uses).
    * Default joint cfg: ``armature=0.1, limit_ke=1e2, limit_kd=1e0``.
    * Default shape cfg: ``ke=5e4, kd=5e2, kf=1e3, mu=0.75``.
    * Per-DoF stiffness / damping / armature pulled from the YAML.
    * Per-DoF target_mode = POSITION.
    * Initial ``joint_q[7:]`` set from the YAML's ``mjw_joint_pos``.
    * Floating base at z = 0.76 with the same ``(0, 0, sqrt(2)/2,
      sqrt(2)/2)`` quaternion (90 deg yaw) the example sets.
    """
    import newton.utils  # noqa: PLC0415

    cfg = _g1_29dof_yaml()
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.1,
        limit_ke=1.0e2,
        limit_kd=1.0e0,
    )
    mb.default_shape_cfg.ke = 5.0e4
    mb.default_shape_cfg.kd = 5.0e2
    mb.default_shape_cfg.kf = 1.0e3
    mb.default_shape_cfg.mu = _ROBOT_POLICY_MU

    asset_dir = newton.utils.download_asset("unitree_g1")
    mb.add_usd(
        str(asset_dir / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    mb.approximate_meshes("convex_hull")
    mb.add_ground_plane()

    for i in range(len(cfg["mjw_joint_stiffness"])):
        mb.joint_target_ke[i + 6] = cfg["mjw_joint_stiffness"][i]
        mb.joint_target_kd[i + 6] = cfg["mjw_joint_damping"][i]
        mb.joint_armature[i + 6] = cfg["mjw_joint_armature"][i]
        mb.joint_target_mode[i + 6] = int(newton.JointTargetMode.POSITION)

    mb.joint_q[:3] = [0.0, 0.0, 0.76]
    mb.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
    mb.joint_q[7:] = cfg["mjw_joint_pos"]

    mb.gravity = -9.81
    return mb.finalize()


def _g1_foot_body_indices(model: newton.Model) -> list[int]:
    """Indices of the two foot bodies (left/right ankle_roll_link)
    inside ``model.body_q``. G1 has no separate "foot" body -- the
    ankle_roll_link IS the foot."""
    out: list[int] = []
    for i, label in enumerate(model.body_label):
        if any(label.endswith(s) for s in ("/left_ankle_roll_link", "/right_ankle_roll_link")):
            out.append(i)
    return out


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Parity tests run on CUDA only.",
)
class TestG1HoldPoseParity(unittest.TestCase):
    """Step the G1 29-DoF model with both solvers, holding the YAML's
    standing-pose joint targets for 2 s. Compare per-foot body z, the
    trunk z, and per-DoF joint angles -- with extra scrutiny on the
    ankle-pitch DoFs (the "toes" the bug report calls out)."""

    DT = 1.0 / 200.0
    N_FRAMES = 400  # 2 s

    def _make_target(self) -> np.ndarray:
        sample = _g1_robot_model()
        q = sample.joint_q.numpy()
        target = np.zeros(int(sample.joint_dof_count), dtype=np.float32)
        target[6:] = q[7:]
        return target

    @unittest.expectedFailure
    def test_standing_pose_held_two_seconds(self) -> None:
        """Body z and per-DoF joint angles must track between solvers
        when ``control.joint_target_pos`` holds the standing pose.

        Marked ``expectedFailure``: PhoenX's PD-drive restoring torque
        does not match MuJoCo's at the YAML's nominal target_ke, so
        the ankle-pitch DoFs diverge by ~30 deg over 2 s. Will flip
        to "unexpected success" once that mismatch is closed."""
        target = self._make_target()
        sample = _g1_robot_model()
        foot_indices = _g1_foot_body_indices(sample)
        self.assertEqual(len(foot_indices), 2, "expected 2 G1 foot (ankle_roll) bodies")

        bq_mj, _, jq_mj = _run_with_contacts(
            _mj_solver,
            _g1_robot_model,
            self.N_FRAMES,
            self.DT,
            target_pos=target,
            record_joint_q=True,
        )
        bq_px, _, jq_px = _run_with_contacts(
            _px_solver,
            _g1_robot_model,
            self.N_FRAMES,
            self.DT,
            target_pos=target,
            record_joint_q=True,
        )

        # Per-foot body z divergence at the final frame.
        for fi in foot_indices:
            z_mj = float(bq_mj[-1, fi, 2])
            z_px = float(bq_px[-1, fi, 2])
            self.assertAlmostEqual(
                z_mj,
                z_px,
                delta=0.05,
                msg=(f"foot body {fi}: z diverges by {z_mj - z_px:+.4f} m (MuJoCo={z_mj:.4f}, PhoenX={z_px:.4f})"),
            )

        # Trunk (body 0) z.
        self.assertAlmostEqual(
            float(bq_mj[-1, 0, 2]),
            float(bq_px[-1, 0, 2]),
            delta=0.10,
            msg=(f"trunk z diverges: MuJoCo={float(bq_mj[-1, 0, 2]):.4f} PhoenX={float(bq_px[-1, 0, 2]):.4f}"),
        )

        # Per-DoF angle divergence on the actuated joints (joint_q[7:]).
        # Tolerance is relaxed (15 deg) because the bug we're after
        # produces *much* bigger errors and a bunch of small ones is
        # acceptable. We separately assert the ankle-pitch DoFs to a
        # tighter tolerance below.
        dof_q_mj = jq_mj[:, 7:]
        dof_q_px = jq_px[:, 7:]
        per_dof_max = np.abs(dof_q_mj - dof_q_px).max(axis=0)
        worst = float(per_dof_max.max())
        worst_idx = int(per_dof_max.argmax())
        cfg = _g1_29dof_yaml()
        worst_name = cfg["mjw_joint_names"][worst_idx]
        self.assertLess(
            worst,
            math.radians(15.0),
            msg=(f"max per-DoF angle divergence {math.degrees(worst):.1f} deg on DoF {worst_idx} ({worst_name})"),
        )

    @unittest.expectedFailure
    def test_ankle_pitch_does_not_drift(self) -> None:
        """The smoking-gun test for the toes-down bug.

        Marked ``expectedFailure`` (see commit ``e6ca9048``): PhoenX's
        steady-state ankle pitch sits ~7 deg from the commanded
        target while MuJoCo holds it. Will flip to "unexpected
        success" once the PD-drive restoring torque matches.

        ``left_ankle_pitch_joint`` and ``right_ankle_pitch_joint`` start
        at -0.2 rad (rotated to keep the foot flat under the +0.3 rad
        knee bend) with target_ke = 20 N*m/rad (the lowest gain in the
        rig, set by the YAML).

        If PhoenX interprets the absolute target differently from MuJoCo
        -- via a sign-flip on the joint axis, an off-by-init_q error,
        a miswired actuator gain, etc. -- the ankle is the first DoF
        where it shows: the foot tips forward and the toe pushes into
        the floor. We assert the steady-state angle stays within 5 deg
        of the commanded -0.2 rad on both solvers, and that the two
        solvers agree on the ankle pitch within 5 deg.
        """
        target = self._make_target()
        cfg = _g1_29dof_yaml()
        ankle_l_dof = cfg["mjw_joint_names"].index("left_ankle_pitch_joint")
        ankle_r_dof = cfg["mjw_joint_names"].index("right_ankle_pitch_joint")
        ankle_target_l = float(cfg["mjw_joint_pos"][ankle_l_dof])
        ankle_target_r = float(cfg["mjw_joint_pos"][ankle_r_dof])

        _, _, jq_mj = _run_with_contacts(
            _mj_solver,
            _g1_robot_model,
            self.N_FRAMES,
            self.DT,
            target_pos=target,
            record_joint_q=True,
        )
        _, _, jq_px = _run_with_contacts(
            _px_solver,
            _g1_robot_model,
            self.N_FRAMES,
            self.DT,
            target_pos=target,
            record_joint_q=True,
        )

        # Steady-state = mean of the last 100 frames (0.5 s window) so
        # we don't latch a single-frame oscillation.
        ankle_l_mj = float(jq_mj[-100:, 7 + ankle_l_dof].mean())
        ankle_l_px = float(jq_px[-100:, 7 + ankle_l_dof].mean())
        ankle_r_mj = float(jq_mj[-100:, 7 + ankle_r_dof].mean())
        ankle_r_px = float(jq_px[-100:, 7 + ankle_r_dof].mean())

        tol = math.radians(5.0)

        # Each solver tracks the commanded target.
        for name, val, tgt in (
            ("MuJoCo L ankle pitch", ankle_l_mj, ankle_target_l),
            ("PhoenX L ankle pitch", ankle_l_px, ankle_target_l),
            ("MuJoCo R ankle pitch", ankle_r_mj, ankle_target_r),
            ("PhoenX R ankle pitch", ankle_r_px, ankle_target_r),
        ):
            self.assertAlmostEqual(
                val,
                tgt,
                delta=tol,
                msg=(
                    f"{name}: steady-state {math.degrees(val):+.2f} deg "
                    f"vs target {math.degrees(tgt):+.2f} deg "
                    f"(drift {math.degrees(val - tgt):+.2f} deg)"
                ),
            )

        # The two solvers agree.
        self.assertAlmostEqual(
            ankle_l_mj,
            ankle_l_px,
            delta=tol,
            msg=(
                f"L ankle pitch solver disagreement: "
                f"MuJoCo={math.degrees(ankle_l_mj):+.2f} deg "
                f"PhoenX={math.degrees(ankle_l_px):+.2f} deg"
            ),
        )
        self.assertAlmostEqual(
            ankle_r_mj,
            ankle_r_px,
            delta=tol,
            msg=(
                f"R ankle pitch solver disagreement: "
                f"MuJoCo={math.degrees(ankle_r_mj):+.2f} deg "
                f"PhoenX={math.degrees(ankle_r_px):+.2f} deg"
            ),
        )


# ---------------------------------------------------------------------------
# Diagnostic: contact-set inspector
# ---------------------------------------------------------------------------


def _diagnose_robot_contacts() -> None:
    """Print PhoenX contact set for the G1 standing pose plus the
    per-DoF angle a 1-second hold produces in each solver.

    MuJoCo's contact set isn't trivially exposed via the Newton wrapper
    (it lives inside ``mjw_data.contact``); this helper focuses on what
    Newton's :class:`CollisionPipeline` (which feeds PhoenX) produces
    so the user can sanity-check whether the foot contacts are where
    they should be. Run via::

        python -m newton._src.solvers.phoenx.tests.test_robot_policy_parity
    """
    print("=" * 70)
    print("G1 contact-set diagnostic (PhoenX path: Newton CollisionPipeline)")
    print("=" * 70)

    model = _g1_robot_model()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = model.contacts()
    model.collide(state, contacts)

    n = int(contacts.rigid_contact_count.numpy()[0])
    print(f"Total rigid contacts: {n}")
    if n == 0:
        print("  (no contacts -- robot may be floating above the floor)")
        return

    p0 = contacts.rigid_contact_point0.numpy()[:n]
    nrm = contacts.rigid_contact_normal.numpy()[:n]
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    bq = state.body_q.numpy()
    sb = model.shape_body.numpy()
    bl = model.body_label

    def _shape_name(shape_idx: int) -> str:
        # URDF-imported shapes don't always populate ``shape_label``;
        # fall back to "<body_label>:shape_<idx>" so the readout is
        # still meaningful.
        if shape_idx < 0:
            return f"shape_{shape_idx}"
        body_idx = int(sb[shape_idx])
        body_name = bl[body_idx] if 0 <= body_idx < len(bl) else "<no-body>"
        try:
            label = model.shape_label[shape_idx]
        except (IndexError, TypeError):
            label = ""
        if not label:
            label = f"shape_{shape_idx}"
        return f"{body_name}:{label}"

    print(f"{'idx':>3} {'shape0':>40} {'shape1':>40} {'normal':>22} {'world_p (z)':>10}")
    for i in range(n):
        b0 = int(sb[s0[i]])
        if b0 >= 0:
            tf = bq[b0]
            wp_xform = wp.transform(wp.vec3(*tf[:3]), wp.quat(*tf[3:7]))
            world_p = wp.transform_point(wp_xform, wp.vec3(*p0[i]))
            wz = float(world_p[2])
        else:
            wz = float(p0[i][2])
        nm = nrm[i]
        print(
            f"{i:>3} {_shape_name(int(s0[i])):>40s} {_shape_name(int(s1[i])):>40s} "
            f"({nm[0]:+.2f},{nm[1]:+.2f},{nm[2]:+.2f}) {wz:>+8.4f}"
        )

    foot_idx = set(_g1_foot_body_indices(model))
    foot_contact_count = sum(1 for i in range(n) if int(sb[s0[i]]) in foot_idx or int(sb[s1[i]]) in foot_idx)
    print(f"\nFoot-involved contacts: {foot_contact_count} / {n}")

    # Per-solver hold-pose angle drift on the ankle pitch joints.
    print()
    print("Steady-state ankle pitch (1 s hold-pose):")
    cfg = _g1_29dof_yaml()
    target = np.zeros(int(model.joint_dof_count), dtype=np.float32)
    target[6:] = model.joint_q.numpy()[7:]
    n_frames = 200
    dt = 1.0 / 200.0
    for solver_name, fac in (("MuJoCo", _mj_solver), ("PhoenX", _px_solver)):
        _, _, jq = _run_with_contacts(fac, _g1_robot_model, n_frames, dt, target_pos=target, record_joint_q=True)
        for label, joint_name in (
            ("L ankle pitch", "left_ankle_pitch_joint"),
            ("R ankle pitch", "right_ankle_pitch_joint"),
        ):
            d = cfg["mjw_joint_names"].index(joint_name)
            tgt = float(cfg["mjw_joint_pos"][d])
            ss = float(jq[-50:, 7 + d].mean())
            drift = ss - tgt
            print(
                f"  {solver_name:>6} {label}: {math.degrees(ss):+7.2f} deg (target {math.degrees(tgt):+5.2f}, drift {math.degrees(drift):+6.2f})"
            )


if __name__ == "__main__":
    wp.init()
    if _HAS_MJW:
        unittest.main(exit=False, argv=["test"])
    _diagnose_robot_contacts()
