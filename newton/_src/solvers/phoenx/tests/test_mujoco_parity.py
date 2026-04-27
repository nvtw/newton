# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parity tests: :class:`SolverPhoenX` vs :class:`SolverMuJoCo` on
simple revolute-joint scenes.

Catches sign / convention / drive-direction regressions that would
prevent PhoenX from being a drop-in replacement for MjWarp in scenes
like the Anymal walking rig. Scenes are deliberately tiny so any
divergence points at a specific PhoenX bug.

Also includes a sweep over PhoenX ``substeps`` / ``solver_iterations``
that maps the stiffness regime: how many substeps + PGS iterations
does PhoenX need to track MuJoCo's response for a given PD gain?

All tests are skipped when ``mujoco`` / ``mujoco_warp`` are not
importable.
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


def _build_pendulum_model(
    *,
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    cube_world_position: tuple[float, float, float] = (1.0, 0.0, 0.0),
    target_pos: float = 0.0,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    limit_lower: float | None = None,
    limit_upper: float | None = None,
    init_q: float = 0.0,
    init_qd: float = 0.0,
    gravity: float = -9.81,
) -> newton.Model:
    """Single dynamic cube hinged to the world by one revolute joint.

    Convention note -- ``child_xform`` is the **inverse** of the
    desired cube offset. Newton's FK computes the child body pose as
    ``X_wc = X_wp * X_pj * X_j * inv(X_cj)``, so to land the cube at
    ``cube_world_position`` with hinge at the world origin and
    ``X_pj = identity`` we pass ``X_cj = -cube_world_position``.

    Defaults (axis=+y, cube at (+1, 0, 0), gravity=-z) produce a
    classic pendulum swinging in the xz plane; positive rotation
    about +y sweeps +x -> -z.
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)

    cube = mb.add_link(
        xform=wp.transform(p=wp.vec3(*cube_world_position), q=wp.quat_identity()),
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

    actuator_mode = (
        newton.JointTargetMode.POSITION if (target_ke > 0.0 or target_kd > 0.0) else newton.JointTargetMode.NONE
    )
    # Invert the cube position to get the joint anchor in child frame.
    neg = tuple(-c for c in cube_world_position)
    kwargs = {
        "parent": -1,
        "child": cube,
        "axis": axis,
        "parent_xform": wp.transform_identity(),
        "child_xform": wp.transform(p=wp.vec3(*neg), q=wp.quat_identity()),
        "target_pos": target_pos,
        "target_ke": target_ke,
        "target_kd": target_kd,
        "actuator_mode": actuator_mode,
    }
    if limit_lower is not None:
        kwargs["limit_lower"] = limit_lower
    if limit_upper is not None:
        kwargs["limit_upper"] = limit_upper

    joint = mb.add_joint_revolute(**kwargs)
    mb.add_articulation([joint])
    mb.gravity = gravity
    model = mb.finalize()

    if init_q != 0.0:
        model.joint_q.assign(np.array([init_q], dtype=np.float32))
    if init_qd != 0.0:
        model.joint_qd.assign(np.array([init_qd], dtype=np.float32))
    return model


def _run(
    solver_factory: Callable[[newton.Model], object],
    model_factory: Callable[[], newton.Model],
    n_frames: int,
    dt: float,
) -> np.ndarray:
    """Step ``n_frames`` and return an ``(n_frames, 2)`` trajectory of
    ``(joint_q, joint_qd)`` read back via :func:`eval_ik`."""
    model = model_factory()
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    traj = np.empty((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        newton.eval_ik(model, s0, jq, jqd)
        traj[i, 0] = float(jq.numpy()[0])
        traj[i, 1] = float(jqd.numpy()[0])
    return traj


def _mj_factory(model: newton.Model):
    return newton.solvers.SolverMuJoCo(model, solver="newton", nconmax=16, njmax=16)


#: Default PhoenX settings for parity tests. ``substeps=4`` + 16 PGS
#: iterations matches MuJoCo trajectories to under 1% RMS on a 1 m
#: pendulum at ``dt = 5 ms`` and ``ke=150`` -- see
#: :class:`TestPhoenXStiffnessSweep` for the full mapping. Heavier
#: stiffness / stiffer drives need more substeps (scaled roughly as
#: ``substeps ~ sqrt(ke)``).
_PX_SUBSTEPS = 4
_PX_ITERATIONS = 16


def _px_factory(model: newton.Model, substeps: int = _PX_SUBSTEPS, solver_iterations: int = _PX_ITERATIONS):
    return newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=1,
    )


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX / MuJoCo parity tests run on CUDA only.",
)
class TestPhoenXMuJoCoParity(unittest.TestCase):
    """Compare joint-coordinate trajectories between PhoenX and MuJoCo
    on the same Newton revolute scene."""

    # ------------------------------------------------------------------
    # Sign convention checks (the most important class of tests for a
    # policy-trained-on-MjWarp -> runs-on-PhoenX swap).
    # ------------------------------------------------------------------

    def test_positive_qd_rotates_positive_angle(self) -> None:
        """Positive initial ``joint_qd`` produces a positive
        ``joint_q`` trajectory under both solvers. Catches a swapped
        axis or inverted iterate."""

        def make():
            return _build_pendulum_model(init_qd=1.0, gravity=0.0)

        traj_mj = _run(_mj_factory, make, 40, 0.005)
        traj_px = _run(_px_factory, make, 40, 0.005)

        self.assertGreater(traj_mj[-1, 0], 0.15)
        self.assertGreater(traj_px[-1, 0], 0.15)
        self.assertAlmostEqual(
            float(traj_mj[-1, 0]),
            float(traj_px[-1, 0]),
            delta=0.01,
            msg=f"free-spin q mismatch: mj={traj_mj[-1, 0]:.5f}, px={traj_px[-1, 0]:.5f}",
        )

    def test_negative_qd_rotates_negative_angle(self) -> None:
        """Negative initial ``joint_qd`` -> negative ``joint_q``."""

        def make():
            return _build_pendulum_model(init_qd=-1.0, gravity=0.0)

        traj_mj = _run(_mj_factory, make, 40, 0.005)
        traj_px = _run(_px_factory, make, 40, 0.005)

        self.assertLess(traj_mj[-1, 0], -0.15)
        self.assertLess(traj_px[-1, 0], -0.15)
        self.assertAlmostEqual(
            float(traj_mj[-1, 0]),
            float(traj_px[-1, 0]),
            delta=0.01,
        )

    def test_free_spin_conserves_angular_velocity(self) -> None:
        """No drive, no gravity, ``qd0 = 1`` -- both solvers must keep
        ``joint_qd`` close to 1 across 100 frames."""

        def make():
            return _build_pendulum_model(init_qd=1.0, gravity=0.0)

        traj_mj = _run(_mj_factory, make, 100, 0.005)
        traj_px = _run(_px_factory, make, 100, 0.005)

        self.assertAlmostEqual(float(traj_mj[-1, 1]), 1.0, delta=0.05)
        self.assertAlmostEqual(float(traj_px[-1, 1]), 1.0, delta=0.05)

    # ------------------------------------------------------------------
    # Gravity pendulum.
    # ------------------------------------------------------------------

    def test_gravity_pendulum_matches_trajectory(self) -> None:
        """Pendulum released from rest at ``q=0`` under gravity. The
        trajectory is identical within solver substep noise."""

        def make():
            return _build_pendulum_model(gravity=-9.81)

        dt = 0.005
        n = 80  # 0.4 s -- first quarter swing for a 1 m pendulum.
        traj_mj = _run(_mj_factory, make, n, dt)
        traj_px = _run(_px_factory, make, n, dt)

        # Same sign at every frame (catches a sign inversion that
        # integrates in opposite directions).
        for i in range(n):
            if abs(traj_mj[i, 0]) > 1e-3:
                self.assertEqual(
                    np.sign(traj_mj[i, 0]),
                    np.sign(traj_px[i, 0]),
                    msg=f"sign mismatch at frame {i}: mj={traj_mj[i, 0]:+.4f}, px={traj_px[i, 0]:+.4f}",
                )
        rms = float(np.sqrt(np.mean((traj_mj[:, 0] - traj_px[:, 0]) ** 2)))
        self.assertLess(
            rms,
            0.01,
            msg=f"gravity-pendulum q RMS divergence {rms:.4f} rad",
        )

    # ------------------------------------------------------------------
    # PD drive.
    # ------------------------------------------------------------------

    def test_pd_drive_reaches_target(self) -> None:
        """PD drive at ``target=pi/4`` from rest. Both solvers settle
        at the target within a 1.2 s window."""

        target = math.pi / 4.0

        def make():
            return _build_pendulum_model(
                target_pos=target,
                target_ke=150.0,
                target_kd=5.0,
                gravity=0.0,
            )

        dt = 0.005
        n = 240  # 1.2 s
        traj_mj = _run(_mj_factory, make, n, dt)
        traj_px = _run(_px_factory, make, n, dt)

        self.assertAlmostEqual(float(traj_mj[-1, 0]), target, delta=0.05)
        self.assertAlmostEqual(float(traj_px[-1, 0]), target, delta=0.05)

    def test_pd_drive_negative_target(self) -> None:
        """PD drive at ``target=-pi/4`` -- mirror of the previous
        test. Catches a drive-sign bug only visible with negative
        targets."""
        target = -math.pi / 4.0

        def make():
            return _build_pendulum_model(
                target_pos=target,
                target_ke=150.0,
                target_kd=5.0,
                gravity=0.0,
            )

        dt = 0.005
        n = 240
        traj_mj = _run(_mj_factory, make, n, dt)
        traj_px = _run(_px_factory, make, n, dt)

        self.assertAlmostEqual(float(traj_mj[-1, 0]), target, delta=0.05)
        self.assertAlmostEqual(float(traj_px[-1, 0]), target, delta=0.05)

    def test_pd_drive_trajectory_rms(self) -> None:
        """Stronger: PhoenX's trajectory must track MuJoCo's RMS-close
        across the settle window."""
        target = math.pi / 4.0

        def make():
            return _build_pendulum_model(
                target_pos=target,
                target_ke=150.0,
                target_kd=5.0,
                gravity=0.0,
            )

        dt = 0.005
        n = 120  # 0.6 s -- first pass through the target.
        traj_mj = _run(_mj_factory, make, n, dt)
        traj_px = _run(_px_factory, make, n, dt)

        rms = float(np.sqrt(np.mean((traj_mj[:, 0] - traj_px[:, 0]) ** 2)))
        self.assertLess(
            rms,
            0.03,
            msg=f"PD trajectory RMS divergence {rms:.4f} rad exceeds 0.03",
        )

    # ------------------------------------------------------------------
    # Joint limits.
    # ------------------------------------------------------------------

    def test_limit_upper_clamps(self) -> None:
        """Release pendulum with ``qd=+3``, limits at ``+/-pi/6``.
        Both solvers should clamp near ``+pi/6`` or oscillate near it
        -- peak q should never exceed ``pi/6`` by much."""

        def make():
            return _build_pendulum_model(
                init_qd=3.0,
                limit_lower=-math.pi / 6.0,
                limit_upper=math.pi / 6.0,
                gravity=0.0,
            )

        dt = 0.005
        n = 200
        traj_mj = _run(_mj_factory, make, n, dt)
        traj_px = _run(_px_factory, make, n, dt)

        # Peak q must not leak past the upper limit by more than a
        # soft-constraint slop.
        for traj, name in ((traj_mj, "MuJoCo"), (traj_px, "PhoenX")):
            peak = float(np.max(traj[:, 0]))
            self.assertLess(
                peak,
                math.pi / 6.0 + 0.1,
                msg=f"{name}: peak q={peak:.4f} leaked past upper limit",
            )


@unittest.skipUnless(_HAS_MJW, "mujoco / mujoco_warp not available")
@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Stiffness sweep runs on CUDA only.",
)
class TestPhoenXStiffnessSweep(unittest.TestCase):
    """Map the ``(substeps, solver_iterations)`` regime PhoenX needs to
    track MuJoCo's PD response for representative gains.

    Runs each (substeps, iterations) configuration and reports the
    RMS ``joint_q`` divergence from MuJoCo across a 0.6 s PD settle
    window. A single ``assert`` at the end locks in the knee of the
    curve so a regression in the solver's stiffness handling fails
    the test.

    The printed table (visible in `-v` mode) doubles as documentation
    of the trade-off between substep count and PGS iterations."""

    def test_pd_stiffness_sweep(self) -> None:
        target = math.pi / 4.0
        dt = 0.005
        n = 120  # 0.6 s

        def make():
            return _build_pendulum_model(
                target_pos=target,
                target_ke=150.0,
                target_kd=5.0,
                gravity=0.0,
            )

        traj_mj = _run(_mj_factory, make, n, dt)

        configs = [
            (1, 8),
            (1, 16),
            (2, 8),
            (2, 16),
            (4, 8),
            (4, 16),
            (8, 16),
            (16, 16),
            (16, 32),
        ]
        results = []
        for substeps, iterations in configs:

            def factory(m, s=substeps, it=iterations):
                return _px_factory(m, substeps=s, solver_iterations=it)

            traj_px = _run(factory, make, n, dt)
            rms = float(np.sqrt(np.mean((traj_mj[:, 0] - traj_px[:, 0]) ** 2)))
            final_err = abs(float(traj_mj[-1, 0]) - float(traj_px[-1, 0]))
            results.append((substeps, iterations, rms, final_err))

        print("\nPhoenX vs MuJoCo stiffness sweep (PD target=pi/4, ke=150, kd=5, dt=5ms):")
        print(f"{'substeps':>10} {'iterations':>12} {'RMS q err':>14} {'final q err':>14}")
        for s, it, rms, fe in results:
            print(f"{s:>10d} {it:>12d} {rms:>14.5f} {fe:>14.5f}")

        # Lock in the performance knee: at substeps=4, iterations=16 the
        # RMS error must stay under 0.03 rad (~1.7 deg). Regressions in
        # the PGS convergence rate or effective-stiffness computation
        # will push this up.
        knee = next(r for r in results if r[0] == 4 and r[1] == 16)
        self.assertLess(
            knee[2],
            0.03,
            msg=f"PD stiffness knee regressed: substeps=4/iter=16 RMS={knee[2]:.5f} rad",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
