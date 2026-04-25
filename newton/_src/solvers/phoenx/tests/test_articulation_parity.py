# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-body articulation parity: :class:`SolverPhoenX` vs
:class:`SolverMuJoCo`.

The single-joint pendulum parity tests in :mod:`test_mujoco_parity`
don't catch regressions that only fire with:

* Non-zero initial ``joint_q`` values (common in URDF-imported
  rigs -- Anymal sets HFE/KFE to non-zero before finalize).
* Multi-link chains where each body's rest pose depends on the
  joint tree (if PhoenX snapshots joint anchors from stale
  ``model.body_q``, every child body's local anchor is wrong).
* Joint axes that are *not* world-axis-aligned (URDF joints often
  have a ``parent_xform`` rotation that tilts the axis).

This module assembles a two-body revolute chain and the actual
Anymal URDF, then steps single joints in isolation to localise any
remaining discrepancy.
"""

from __future__ import annotations

import math
import unittest
from typing import Callable

import numpy as np
import warp as wp

import newton

try:
    import mujoco  # noqa: F401
    import mujoco_warp  # noqa: F401

    _HAS_MJW = True
except ImportError:
    _HAS_MJW = False


def _mj_factory(model: newton.Model):
    return newton.solvers.SolverMuJoCo(
        model, solver="newton", nconmax=64, njmax=64
    )


def _px_factory(model: newton.Model):
    return newton.solvers.SolverPhoenX(
        model, substeps=4, solver_iterations=16, velocity_iterations=1
    )


def _px_factory_singleworld(model: newton.Model):
    return newton.solvers.SolverPhoenX(
        model, substeps=4, solver_iterations=16, velocity_iterations=1,
        step_layout="single_world",
    )


def _run_joint_state(
    solver_factory: Callable[[newton.Model], object],
    model_factory: Callable[[], newton.Model],
    n_frames: int,
    dt: float,
    target_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Step ``n_frames`` and return ``(body_q[n_frames], joint_q[n_frames])``.

    ``target_pos`` (if provided) is written into ``control.joint_target_pos``
    every frame so PD drives track toward it.
    """
    model = model_factory()
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    if target_pos is not None:
        control.joint_target_pos.assign(target_pos.astype(np.float32))
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    body_q_traj = np.empty((n_frames, model.body_count, 7), dtype=np.float32)
    joint_q_traj = np.empty((n_frames, model.joint_coord_count), dtype=np.float32)
    for i in range(n_frames):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        newton.eval_ik(model, s0, jq, jqd)
        body_q_traj[i] = s0.body_q.numpy()
        joint_q_traj[i] = jq.numpy()
    return body_q_traj, joint_q_traj


# ---------------------------------------------------------------------------
# Two-body REVOLUTE chain (minimum that exposes "initial joint_q matters")
# ---------------------------------------------------------------------------


def _build_two_link_chain(
    *,
    init_q0: float = 0.0,
    init_q1: float = 0.0,
) -> newton.Model:
    """Two-link serial chain: world -> link_a (REVOLUTE about +y at origin)
    -> link_b (REVOLUTE about +y at link_a's tip).

    Each link is a thin 1 m rod along +x. The rest pose (joint_q=0)
    has link_a spanning (0..1 m) along +x and link_b spanning (1..2 m)
    along +x. Non-zero ``init_q0`` / ``init_q1`` rotate the links.
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)

    link_a = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.01, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        link_a, hx=0.5, hy=0.05, hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )

    link_b = mb.add_link(
        xform=wp.transform(p=wp.vec3(1.5, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.01, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        link_b, hx=0.5, hy=0.05, hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )

    # Joint 0: world -> link_a. Hinge at world origin.
    j0 = mb.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=0.0,
        target_ke=100.0,
        target_kd=5.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )

    # Joint 1: link_a -> link_b. Hinge at link_a's +x tip, i.e. world
    # (1, 0, 0) in the rest pose. In link_a's local frame that's
    # (+0.5, 0, 0); in link_b's local frame it's (-0.5, 0, 0).
    j1 = mb.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=0.0,
        target_ke=100.0,
        target_kd=5.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )

    mb.add_articulation([j0, j1])
    mb.gravity = 0.0
    model = mb.finalize()

    # Preload joint_q the way Anymal does -- builder.joint_q is a
    # plain Python list on the builder, mirrored to model.joint_q at
    # finalize() time. We edit the finalized array directly so both
    # cases are covered.
    q = model.joint_q.numpy()
    q[0] = init_q0
    q[1] = init_q1
    model.joint_q.assign(q)
    return model


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Articulation parity tests run on CUDA only.",
)
class TestTwoLinkChainParity(unittest.TestCase):
    """Two-link serial chain. Non-zero initial ``joint_q[0]`` exposes
    the stale-``body_q``-at-init bug that broke Anymal."""

    def test_rest_pose_holds(self) -> None:
        """joint_q=0 everywhere; PD drive holds both links still. Body
        poses should not move (within soft-constraint slop)."""

        def make():
            return _build_two_link_chain(init_q0=0.0, init_q1=0.0)

        bq_mj, jq_mj = _run_joint_state(_mj_factory, make, 60, 0.005)
        bq_px, jq_px = _run_joint_state(_px_factory, make, 60, 0.005)

        # After 0.3 s the body positions should still be near the
        # rest pose.
        for name, bq in (("MuJoCo", bq_mj), ("PhoenX", bq_px)):
            link_a_pos = bq[-1, 0, :3]
            link_b_pos = bq[-1, 1, :3]
            self.assertAlmostEqual(float(link_a_pos[0]), 0.5, delta=0.05,
                                   msg=f"{name}: link_a x drifted: {link_a_pos}")
            self.assertAlmostEqual(float(link_b_pos[0]), 1.5, delta=0.05,
                                   msg=f"{name}: link_b x drifted: {link_b_pos}")

    def test_init_q0_nonzero_matches_mujoco(self) -> None:
        """Start with ``joint_q[0] = pi/4`` preloaded via
        ``model.joint_q``. The PD drive commands the same target, so
        the chain stays near rest in the preloaded pose.

        This is the regression test for the stale-``body_q``-at-init
        bug: without the FK sync, PhoenX snapshots joint anchors at
        the URDF rest pose, so at the first step the solver yanks
        both links back toward q=0 -- visible as a large position
        divergence in the first few frames."""
        target = np.array([math.pi / 4.0, 0.0], dtype=np.float32)

        def make():
            return _build_two_link_chain(init_q0=math.pi / 4.0, init_q1=0.0)

        bq_mj, _ = _run_joint_state(_mj_factory, make, 30, 0.005, target_pos=target)
        bq_px, _ = _run_joint_state(_px_factory, make, 30, 0.005, target_pos=target)

        # Per-body max divergence across the whole 30-frame window
        # must stay under a few cm. Before the fix the divergence was
        # O(1 m) -- links flew off back toward rest.
        max_delta = float(np.abs(bq_mj[:, :, :3] - bq_px[:, :, :3]).max())
        self.assertLess(
            max_delta,
            0.05,
            msg=f"init-q0 body position max divergence {max_delta:.4f} m",
        )

    def test_init_q1_nonzero_matches_mujoco(self) -> None:
        """Non-zero ``joint_q[1]`` -- child-joint-angle preload. Needs
        the adapter to resolve joint 1's parent-body world pose from
        the FK'd ``model.body_q``, which depends on joint 0's joint_q
        (even though it's 0 here, joint 1's anchor is on link_a)."""
        target = np.array([0.0, math.pi / 4.0], dtype=np.float32)

        def make():
            return _build_two_link_chain(init_q0=0.0, init_q1=math.pi / 4.0)

        bq_mj, _ = _run_joint_state(_mj_factory, make, 30, 0.005, target_pos=target)
        bq_px, _ = _run_joint_state(_px_factory, make, 30, 0.005, target_pos=target)

        max_delta = float(np.abs(bq_mj[:, :, :3] - bq_px[:, :, :3]).max())
        self.assertLess(
            max_delta,
            0.05,
            msg=f"init-q1 body position max divergence {max_delta:.4f} m",
        )

    def test_pd_drive_single_joint_in_chain(self) -> None:
        """PD-drive joint 0 toward pi/4 while joint 1 is held at 0.
        The load is under-damped (ke=100, kd=5, I_eff ~ 1 kg m^2) so
        neither solver settles at the target within 1.2 s -- both
        oscillate. The parity requirement is that PhoenX tracks
        MuJoCo's trajectory, not that it reaches the target."""
        target = np.array([math.pi / 4.0, 0.0], dtype=np.float32)

        def make():
            return _build_two_link_chain(init_q0=0.0, init_q1=0.0)

        dt = 0.005
        n = 240  # 1.2 s
        _, jq_mj = _run_joint_state(_mj_factory, make, n, dt, target_pos=target)
        _, jq_px = _run_joint_state(_px_factory, make, n, dt, target_pos=target)

        # Trajectories track each other within 0.05 rad RMS and both
        # oscillate about the target (both pass through it at least
        # once in the window).
        rms = float(np.sqrt(np.mean((jq_mj[:, 0] - jq_px[:, 0]) ** 2)))
        self.assertLess(rms, 0.05, msg=f"joint-0 RMS divergence {rms:.4f} rad")
        self.assertGreater(
            float(np.max(jq_mj[:, 0])),
            target[0] * 0.8,
            msg="MuJoCo should swing past ~0.8*target in the window",
        )
        self.assertGreater(
            float(np.max(jq_px[:, 0])),
            target[0] * 0.8,
            msg="PhoenX should swing past ~0.8*target in the window",
        )


# ---------------------------------------------------------------------------
# Anymal URDF -- real-world per-joint parity
# ---------------------------------------------------------------------------


def _anymal_model() -> newton.Model:
    """Anymal C URDF built the exact way the walking example does
    (floating base + disabled self-collision + 3x PD gain per DOF),
    minus the policy / controller wiring."""
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
    # Anymal's canonical default pose -- the same one the walking
    # example sets. These initial angles are exactly what the stale-
    # body_q regression used to swallow silently.
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


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Anymal parity tests run on CUDA only.",
)
class TestAnymalArticulationParity(unittest.TestCase):
    """Parity on the real Anymal URDF. The walking rig is what we
    actually care about getting right; these tests catch anything
    the two-body chain missed."""

    def test_fk_matches_after_init(self) -> None:
        """After loading Anymal and running FK once, body positions
        must be identical between the two solvers at step 0 -- i.e.
        PhoenX must interpret the same joint_q / parent_xform /
        child_xform / axis conventions as MuJoCo does."""
        model_mj = _anymal_model()
        model_px = _anymal_model()

        _ = newton.solvers.SolverMuJoCo(model_mj, solver="newton", nconmax=100, njmax=50)
        _ = newton.solvers.SolverPhoenX(
            model_px, substeps=4, solver_iterations=16, velocity_iterations=1
        )

        state_mj = model_mj.state()
        state_px = model_px.state()
        newton.eval_fk(model_mj, model_mj.joint_q, model_mj.joint_qd, state_mj)
        newton.eval_fk(model_px, model_px.joint_q, model_px.joint_qd, state_px)

        bq_mj = state_mj.body_q.numpy()
        bq_px = state_px.body_q.numpy()
        max_pos = float(np.abs(bq_mj[:, :3] - bq_px[:, :3]).max())
        self.assertLess(
            max_pos,
            1.0e-4,
            msg=f"FK body positions diverge by {max_pos:.6f} m between MuJoCo/PhoenX init",
        )

    def test_initial_configuration_holds(self) -> None:
        """Step Anymal for 20 frames with ``control.joint_target_pos``
        held at the default joint_q. The robot must not fall through
        the floor or drift catastrophically. Per-body position
        divergence between MuJoCo and PhoenX must stay small.

        ``target_pos`` is sized by ``joint_dof_count`` (not ``joint_coord_count``);
        the 6 FREE base DOFs come first and are left at zero (no PD
        drive on the free base), then the 12 REVOLUTE leg DOFs carry
        the default leg angles.
        """
        dt = 1.0 / 200.0  # 5 ms, matching Anymal's sim_dt
        n = 20  # 0.1 s -- the policy hasn't even kicked in yet

        # ``joint_q`` has 7 FREE coords + 12 leg angles = 19.
        # ``joint_target_pos`` has 6 FREE DOFs + 12 leg DOFs = 18.
        # Leg q values start at joint_q[7] and leg target starts at
        # joint_target_pos[6], both length 12.
        model = _anymal_model()
        q = model.joint_q.numpy()
        target = np.zeros(int(model.joint_dof_count), dtype=np.float32)
        target[6:] = q[7:]

        bq_mj, _ = _run_joint_state(_mj_factory, lambda: _anymal_model(), n, dt, target_pos=target)
        bq_px, _ = _run_joint_state(_px_factory, lambda: _anymal_model(), n, dt, target_pos=target)

        max_delta = float(np.abs(bq_mj[-1, :, :3] - bq_px[-1, :, :3]).max())
        self.assertLess(
            max_delta,
            0.05,
            msg=(
                f"after {n} frames, max body position divergence = "
                f"{max_delta:.4f} m between MuJoCo and PhoenX (target = joint_q hold)"
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
