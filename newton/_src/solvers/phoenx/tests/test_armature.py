# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint armature unit tests for ``SolverPhoenX``.

Maximal PhoenX represents motor armature as stator-side parent inertia plus
gearbox-reflected rotor-side child inertia. These tests cover physical periods, chain composition, zero-armature
behavior, MuJoCo parity for anchored joints, and stability.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _build_gravity_pendulum(*, length: float, mass: float, armature: float) -> newton.Model:
    """1-DoF revolute pendulum with the bob at ``-z * length`` and the
    pivot at the world origin. Joint axis is ``+y`` so the bob swings
    in the ``x-z`` plane under ``-z`` gravity."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
    # Concentrate the mass via tiny COM-frame inertia so the pendulum
    # behaves close to a point-mass at distance ``length``: parallel-
    # axis dominates with ``I_pivot ~ m * length^2``.
    bob = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((1.0e-4, 0.0, 0.0), (0.0, 1.0e-4, 0.0), (0.0, 0.0, 1.0e-4)),
    )
    mb.add_shape_box(
        bob,
        hx=0.02,
        hy=0.02,
        hz=0.02,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_revolute(
        parent=-1,
        child=bob,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, float(length)), q=wp.quat_identity()),
    )
    mb.add_articulation([joint])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _measure_period_zero_crossings(signal: np.ndarray, dt: float) -> float:
    """Average period from rising zero-crossings of ``signal`` (mean-removed)."""
    s = signal - signal.mean()
    sgn = np.sign(s)
    rising = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
    if len(rising) < 2:
        return float("nan")
    return float(np.mean(np.diff(rising)) * dt)


def _run_phoenx_joint_history(
    model: newton.Model,
    frames: int,
    dt: float,
    init_q,
    *,
    substeps: int = 8,
    solver_iterations: int = 8,
    velocity_iterations: int = 1,
    record_qd: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Replay one captured PhoenX step and record joint coordinates."""
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=velocity_iterations,
    )
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    s0.joint_q.assign(np.asarray(init_q, dtype=np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)

    def step_once() -> None:
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)
        newton.eval_ik(model, s0, jq, jqd)

    with wp.ScopedCapture(device=model.device) as cap:
        step_once()
    graph = cap.graph

    q_history = np.empty((frames, model.joint_coord_count), dtype=np.float32)
    qd_history = np.empty((frames, model.joint_dof_count), dtype=np.float32) if record_qd else None
    for i in range(frames):
        wp.capture_launch(graph)
        q_history[i] = jq.numpy()
        if qd_history is not None:
            qd_history[i] = jqd.numpy()
    return q_history, qd_history


def _run_pendulum(model: newton.Model, frames: int, dt: float, init_angle: float = 0.05) -> np.ndarray:
    """Return captured-graph ``joint_q[0]`` history."""
    history, _ = _run_phoenx_joint_history(model, frames, dt, [float(init_angle)])
    return history[:, 0]


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX armature tests run on CUDA only (graph-capture path).",
)
class TestPendulumPeriod(unittest.TestCase):
    """Verify armature affects the small-angle pendulum period as
    ``T = 2*pi * sqrt((m*L^2 + I_com + armature) / (m*g*L))``."""

    def test_period_scaling_with_armature(self) -> None:
        mass = 1.0
        length = 1.0
        g = 9.81
        # Body COM-frame inertia from the fixture (1e-4 along the joint
        # axis), plus parallel-axis ``m*L^2``.
        I_chain = mass * length * length + 1.0e-4
        dt = 1.0 / 400.0
        # Long enough for two measured periods at the largest armature.
        frames = 5200

        for armature in (0.0, 0.5, 2.0, 8.0):
            with self.subTest(armature=armature):
                model = _build_gravity_pendulum(length=length, mass=mass, armature=armature)
                history = _run_pendulum(model, frames=frames, dt=dt)
                T_sim = _measure_period_zero_crossings(history, dt)

                T_expected = 2.0 * np.pi * np.sqrt((I_chain + armature) / (mass * g * length))
                rel_err = abs(T_sim - T_expected) / T_expected
                # 5 % is generous: the soft Baumgarte bias on the rigid
                # 5-row block plus the implicit-Euler integration both
                # introduce small numerical period errors that scale
                # with ``dt``.
                self.assertLess(
                    rel_err,
                    0.05,
                    f"armature={armature}: T_sim={T_sim:.4f} s vs T_expected={T_expected:.4f} s",
                )


def _build_two_body_torsion_chain(
    *,
    inertia: float,
    mass: float,
    target_ke: float,
    armature: float,
) -> newton.Model:
    """Two free bodies joined by a revolute about ``+z`` with a PD
    torsion spring. Symmetric inertias on both bodies so the chain
    effective inertia at the joint is ``I*I/(I+I) = I/2``.

    No gravity, no external loads, no ground -- the only dynamics is
    the joint angle oscillating against the PD spring with the
    armature-augmented effective inertia.
    """
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
    # Body A at the origin, body B 0.2 m along +x. Both share the
    # same diagonal inertia so I_axial is identical along the joint
    # axis (+z).
    body_a = mb.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        inertia=(
            (inertia, 0.0, 0.0),
            (0.0, inertia, 0.0),
            (0.0, 0.0, inertia),
        ),
    )
    body_b = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.2, 0.0, 0.0), q=wp.quat_identity()),
        mass=float(mass),
        inertia=(
            (inertia, 0.0, 0.0),
            (0.0, inertia, 0.0),
            (0.0, 0.0, inertia),
        ),
    )
    # Coincident anchors so the joint frame has no lever arm: a pure
    # torsion spring around +z, decoupled from the bodies' linear
    # dynamics.
    joint = mb.add_joint_revolute(
        parent=body_a,
        child=body_b,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-0.1, 0.0, 0.0), q=wp.quat_identity()),
        target_pos=0.0,
        target_ke=float(target_ke),
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    mb.add_articulation([joint])
    model = mb.finalize()
    # Zero gravity so the only restoring force on the joint is the PD
    # spring -- the analytical period formula doesn't have a gravity
    # term to subtract.
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _run_torsion_chain(model: newton.Model, frames: int, dt: float, init_angle: float = 0.05) -> np.ndarray:
    """Return captured-graph torsion angle history."""
    history, _ = _run_phoenx_joint_history(model, frames, dt, [float(init_angle)])
    return history[:, 0]


def _cuda_with_graph_capture() -> bool:
    """Phoenx tests are graph-capture-only on CUDA. Without the
    graph the per-frame Python overhead pushes the period sweep
    over the test-suite timeout budget."""
    device = wp.get_preferred_device()
    return device.is_cuda and wp.is_mempool_enabled(device)


@unittest.skipUnless(
    _cuda_with_graph_capture(),
    "PhoenX armature tests run on CUDA with graph-capture only.",
)
class TestSymmetricTwoBodyPeriod(unittest.TestCase):
    """Check physical stator/rotor inertia for two free symmetric bodies.

    With unit gear ratio, both motor sides have inertia I + a. Eliminating
    free global rotation gives relative-coordinate inertia (I + a) / 2.
    """

    def test_period_scaling_with_armature(self) -> None:
        inertia = 1.0  # symmetric I_A = I_B = 1.0 along +z
        mass = 1.0
        target_ke = 100.0  # PD stiffness on the torsion spring
        # Half-period at the largest armature (a = 8) is
        # ~pi*sqrt((0.5 + 8)/100) ~= 0.92 s, so 4 s of sim @ dt = 2.5 ms
        # (frames = 1600) gives roughly 4 oscillations -- enough for a
        # robust zero-crossing fit even at the slowest case.
        dt = 1.0 / 400.0
        frames = 1600

        for armature in (0.0, 0.5, 2.0, 8.0):
            with self.subTest(armature=armature):
                model = _build_two_body_torsion_chain(
                    inertia=inertia,
                    mass=mass,
                    target_ke=target_ke,
                    armature=armature,
                )
                history = _run_torsion_chain(model, frames=frames, dt=dt)
                T_sim = _measure_period_zero_crossings(history, dt)

                stator_side_inertia = inertia + armature
                rotor_side_inertia = inertia + armature
                effective_inertia = (
                    stator_side_inertia * rotor_side_inertia / (stator_side_inertia + rotor_side_inertia)
                )
                T_expected = 2.0 * np.pi * np.sqrt(effective_inertia / target_ke)
                rel_err = abs(T_sim - T_expected) / T_expected
                self.assertLess(
                    rel_err,
                    0.05,
                    f"armature={armature}: T_sim={T_sim:.4f} s vs "
                    f"T_expected={T_expected:.4f} s "
                    f"(reflected-body effective inertia={effective_inertia:.4f}).",
                )


def _build_two_body_prismatic_slide(
    *,
    mass: float,
    target_ke: float,
    armature: float,
) -> newton.Model:
    """Two free bodies joined by a prismatic slide along ``+z`` with
    a linear PD spring. Symmetric masses on both bodies so the chain
    effective mass at the slide is ``m*m/(m+m) = m/2``. Zero gravity
    -- the only restoring force is the slide spring."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
    body_a = mb.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        # Tiny inertia so angular DoFs aren't excited; the
        # prismatic constraint locks angular motion anyway.
        inertia=((1.0e-4, 0, 0), (0, 1.0e-4, 0), (0, 0, 1.0e-4)),
    )
    body_b = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((1.0e-4, 0, 0), (0, 1.0e-4, 0), (0, 0, 1.0e-4)),
    )
    joint = mb.add_joint_prismatic(
        parent=body_a,
        child=body_b,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
        target_ke=float(target_ke),
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    mb.add_articulation([joint])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _run_prismatic_slide(model: newton.Model, frames: int, dt: float, init_slide: float = 0.05) -> np.ndarray:
    """Captured-graph slide history of the single prismatic DoF."""
    history, _ = _run_phoenx_joint_history(model, frames, dt, [float(init_slide)])
    return history[:, 0]


@unittest.skipUnless(
    _cuda_with_graph_capture(),
    "PhoenX armature tests run on CUDA with graph-capture only.",
)
class TestPrismaticTwoBodyPeriod(unittest.TestCase):
    """Reject armature that cannot be represented by rigid-body inertia."""

    def test_nonzero_prismatic_armature_is_rejected(self) -> None:
        model = _build_two_body_prismatic_slide(mass=1.0, target_ke=100.0, armature=0.5)
        with self.assertRaisesRegex(NotImplementedError, "rotational motor rotors only"):
            newton.solvers.SolverPhoenX(model)


def _build_anchored_three_body_chain(
    *,
    inertia: float,
    mass: float,
    target_ke: float,
    armature: float,
) -> newton.Model:
    """Anchored ``world -- A -- B -- C`` chain about ``+z``: two
    revolute joints, both armatured and PD-driven. Equal axial
    inertia ``I`` on each link; tight COM-frame inertia keeps
    parallel-axis terms negligible (joint anchors coincide with COM
    so there's no lever arm)."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
    body_a = mb.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    body_b = mb.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    body_c = mb.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    j_world_a = mb.add_joint_revolute(
        parent=-1,
        child=body_a,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
        target_ke=float(target_ke),
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    j_a_b = mb.add_joint_revolute(
        parent=body_a,
        child=body_b,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
        target_ke=float(target_ke),
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    j_b_c = mb.add_joint_revolute(
        parent=body_b,
        child=body_c,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
        target_ke=float(target_ke),
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    mb.add_articulation([j_world_a, j_a_b, j_b_c])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _run_three_body_chain(
    model: newton.Model,
    frames: int,
    dt: float,
    init_q: np.ndarray,
) -> np.ndarray:
    """Captured-graph history of the three joint angles."""
    history, _ = _run_phoenx_joint_history(model, frames, dt, init_q, solver_iterations=16)
    return history


@unittest.skipUnless(
    _cuda_with_graph_capture(),
    "PhoenX armature tests run on CUDA with graph-capture only.",
)
class TestThreeBodyChain(unittest.TestCase):
    """Check stator/rotor body inertia in an anchored chain.

    Each internal revolute motor contributes stator inertia to its parent and
    reflected rotor inertia to its child.
    """

    def test_armature_composition(self) -> None:
        inertia = 1.0
        mass = 1.0
        target_ke = 100.0
        dt = 1.0 / 400.0

        for armature in (0.0, 1.0, 8.0):
            K = target_ke * np.eye(3, dtype=np.float64)
            # Every internal motor adds stator inertia to its parent and
            # rotor inertia to its child. For world--A--B--C at unit gear,
            # link inertias are [I+2a, I+2a, I+a].
            link_inertia = np.array(
                [inertia + 2.0 * armature, inertia + 2.0 * armature, inertia + armature],
                dtype=np.float64,
            )
            M_reflected = np.empty((3, 3), dtype=np.float64)
            for row in range(3):
                for col in range(3):
                    M_reflected[row, col] = np.sum(link_inertia[max(row, col) :])
            # Generalised eigenproblem: K v = omega^2 M_reflected v.
            w_sq, V = np.linalg.eig(np.linalg.solve(M_reflected, K))
            sort_idx = np.argsort(w_sq)
            w_sq = w_sq[sort_idx]
            V = V[:, sort_idx]
            T_modes = (2.0 * np.pi / np.sqrt(w_sq)).real

            for mode_idx, T_expected in enumerate(T_modes):
                with self.subTest(armature=armature, mode=mode_idx):
                    # Initial condition aligned with eigenvector =>
                    # only this mode oscillates. Scale to ~0.05 rad
                    # max amplitude.
                    eigvec = V[:, mode_idx].real
                    eigvec = eigvec / np.max(np.abs(eigvec)) * 0.05

                    # 8 oscillations of the chosen mode for a robust
                    # zero-crossing fit, with a floor so the fastest
                    # mode still gets enough samples.
                    target_duration = max(8.0 * T_expected, 4.0)
                    frames = int(np.ceil(target_duration / dt))

                    model = _build_anchored_three_body_chain(
                        inertia=inertia,
                        mass=mass,
                        target_ke=target_ke,
                        armature=armature,
                    )
                    history = _run_three_body_chain(model, frames=frames, dt=dt, init_q=eigvec)

                    # Track q1 (every mode has nonzero q1 component
                    # for this M / K, so the signal is unambiguous).
                    T_sim = _measure_period_zero_crossings(history[:, 0], dt)

                    rel_err = abs(T_sim - T_expected) / T_expected
                    self.assertLess(
                        rel_err,
                        0.05,
                        f"3-body chain armature={armature}, mode={mode_idx}: "
                        f"T_sim={T_sim:.4f} s vs T_expected={T_expected:.4f} s "
                        f"(all modes={T_modes.tolist()}, eigvec={eigvec.tolist()})",
                    )


def _run_pendulum_with_solver(
    model: newton.Model,
    frames: int,
    dt: float,
    solver_factory,
    init_angle: float = 0.05,
) -> np.ndarray:
    """Pendulum runner parameterised by solver factory. Used by the
    cross-solver MuJoCo / PhoenX comparison test."""
    solver = solver_factory(model)
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = None
    s0.joint_q.assign(np.array([float(init_angle)], dtype=np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    history = np.empty(frames, dtype=np.float32)
    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    for i in range(frames):
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        newton.eval_ik(model, s0, jq, jqd)
        history[i] = float(jq.numpy()[0])
    return history


@unittest.skipUnless(
    _cuda_with_graph_capture(),
    "PhoenX armature tests run on CUDA with graph-capture only.",
)
class TestArmatureMatchesMuJoCo(unittest.TestCase):
    """Cross-solver semantics check.

    Build the same anchored revolute pendulum model, run it through
    both ``SolverPhoenX`` and ``SolverMuJoCo``, and assert that the
    measured period agrees within 3 % across an armature sweep.
    Pinning the periods to within a few percent of each other is the
    strongest invariant we can assert without bit-identical
    integrators -- it locks the reduced-coord ``M_q + a`` semantics
    in PhoenX to MuJoCo's reference behaviour.

    Pendulum only (anchored). The two-body torsion case requires an
    explicit FREE base joint for MuJoCo (not for PhoenX), which would
    diverge the joint_q layouts; the pendulum/skinny-chain coverage
    is enough to pin the inertial-row response.
    """

    def test_pendulum_period_matches_mujoco(self) -> None:
        mass = 1.0
        length = 1.0
        g = 9.81
        I_chain = mass * length * length + 1.0e-4
        dt = 1.0 / 400.0
        frames = 5200

        for armature in (0.0, 8.0):
            with self.subTest(armature=armature):
                T_expected = 2.0 * np.pi * np.sqrt((I_chain + armature) / (mass * g * length))

                model_phoenx = _build_gravity_pendulum(length=length, mass=mass, armature=armature)
                history_phoenx = _run_pendulum(model_phoenx, frames=frames, dt=dt)
                T_phoenx = _measure_period_zero_crossings(history_phoenx, dt)

                # Fresh model for MuJoCo (solvers may modify model
                # state internally; cheaper to rebuild than to copy).
                model_mujoco = _build_gravity_pendulum(length=length, mass=mass, armature=armature)
                history_mujoco = _run_pendulum_with_solver(
                    model_mujoco,
                    frames=frames,
                    dt=dt,
                    solver_factory=newton.solvers.SolverMuJoCo,
                )
                T_mujoco = _measure_period_zero_crossings(history_mujoco, dt)

                rel_err_cross = abs(T_phoenx - T_mujoco) / T_mujoco
                rel_err_phoenx = abs(T_phoenx - T_expected) / T_expected
                rel_err_mujoco = abs(T_mujoco - T_expected) / T_expected
                self.assertLess(
                    rel_err_cross,
                    0.03,
                    f"armature={armature}: PhoenX T={T_phoenx:.4f} s vs MuJoCo T={T_mujoco:.4f} s "
                    f"(rel_err {rel_err_cross:.4f}). "
                    f"vs analytical T={T_expected:.4f} s: phoenx_err={rel_err_phoenx:.4f}, "
                    f"mujoco_err={rel_err_mujoco:.4f}",
                )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX armature tests run on CUDA only (graph-capture path).",
)
class TestArmatureNoOpAtZero(unittest.TestCase):
    """``armature == 0`` (the default) must leave body inertias and the
    resulting trajectory bit-identical to the pre-armature behaviour."""

    def test_zero_armature_matches_baseline(self) -> None:
        # Build identical models, one with explicit armature=0 and one
        # without ever touching the armature column. Body inertias on
        # both PhoenX body containers should match exactly after
        # construction.
        m1 = _build_gravity_pendulum(length=1.0, mass=1.0, armature=0.0)
        s1 = newton.solvers.SolverPhoenX(m1)

        m2 = _build_gravity_pendulum(length=1.0, mass=1.0, armature=0.0)
        s2 = newton.solvers.SolverPhoenX(m2)

        i1 = s1.bodies.inverse_inertia.numpy()
        i2 = s2.bodies.inverse_inertia.numpy()
        np.testing.assert_array_equal(i1, i2)

        # And a trajectory should be identical (deterministic seed).
        h1 = _run_pendulum(m1, frames=400, dt=1.0 / 200.0)
        h2 = _run_pendulum(m2, frames=400, dt=1.0 / 200.0)
        np.testing.assert_allclose(h1, h2, rtol=1e-6, atol=1e-7)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX armature tests run on CUDA only (graph-capture path).",
)
class TestSkinnyChainStability(unittest.TestCase):
    """Regression for the ``robot_policy --solver phoenx`` G1 bug:
    high-stiffness PD on a chain through a near-zero-inertia link
    diverges without armature, settles with it."""

    @staticmethod
    def _build_chain(*, intermediate_inertia: float, end_inertia: float, armature: float) -> newton.Model:
        mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
        mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=armature)
        # Heavy anchor at world origin.
        anchor = mb.add_link(
            xform=wp.transform_identity(),
            mass=10.0,
            inertia=((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
        )
        # Skinny intermediate link.
        intermediate = mb.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.2), q=wp.quat_identity()),
            mass=0.1,
            inertia=(
                (intermediate_inertia, 0, 0),
                (0, intermediate_inertia, 0),
                (0, 0, intermediate_inertia),
            ),
        )
        # Heavy end link.
        end = mb.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5), q=wp.quat_identity()),
            mass=8.0,
            inertia=(
                (end_inertia, 0, 0),
                (0, end_inertia, 0),
                (0, 0, end_inertia),
            ),
        )
        # FIXED to world (anchor stays put).
        mb.add_joint_fixed(parent=-1, child=anchor)
        # Skinny revolute: anchor -> intermediate, axis +y, target_ke=300.
        j1 = mb.add_joint_revolute(
            parent=anchor,
            child=intermediate,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0, 0, -0.1), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0, 0, 0.1), q=wp.quat_identity()),
            target_pos=0.0,
            target_ke=300.0,
            target_kd=5.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        # End revolute: intermediate -> end.
        j2 = mb.add_joint_revolute(
            parent=intermediate,
            child=end,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0, 0, -0.1), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0, 0, 0.2), q=wp.quat_identity()),
            target_pos=0.0,
            target_ke=300.0,
            target_kd=5.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        mb.add_articulation([j1, j2])
        model = mb.finalize()
        model.set_gravity((0.0, 0.0, -9.81))
        return model

    def _peak_inner_qd(self, model: newton.Model, frames: int, dt: float) -> float:
        """Step ``frames`` frames with an initial off-equilibrium pose.
        Returns peak ``|joint_qd|`` over the two revolute DoFs (NaN
        sentinels become +inf)."""
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=8,
            velocity_iterations=1,
        )
        s0 = model.state()
        s1 = model.state()
        control = model.control()
        # Perturb both inner joints by 0.2 rad off the PD target so the
        # drive has to actively pull them back. With a near-zero-inertia
        # intermediate link the PGS transient diverges before settling.
        s0.joint_q.assign(np.array([0.2, 0.2], dtype=np.float32))
        newton.eval_fk(model, s0.joint_q, model.joint_qd, s0)
        peak = 0.0
        jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
        jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)

        def step_once() -> None:
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)
            newton.eval_ik(model, s0, jq, jqd)

        with wp.ScopedCapture(device=model.device) as cap:
            step_once()

        for _ in range(frames):
            wp.capture_launch(cap.graph)
            qd = jqd.numpy()
            v = float(np.abs(qd).max())
            if v != v:  # NaN
                return float("inf")
            peak = max(peak, v)
        return peak

    def test_armature_stabilises_chain(self) -> None:
        """Adding ``armature = 0.05`` keeps the chain bounded."""
        model = self._build_chain(
            intermediate_inertia=1.0e-5,
            end_inertia=0.1,
            armature=0.05,
        )
        peak = self._peak_inner_qd(model, frames=240, dt=1.0 / 200.0)
        self.assertLess(
            peak,
            50.0,
            f"with armature, scene should stay below 50 rad/s, got peak={peak:.2f}",
        )


if __name__ == "__main__":
    unittest.main()
