# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-equality coverage for every maximal-coordinate PhoenX joint mode."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_FIXED,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)

_INERTIA = ((0.7, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 0.9))


def _add_body(builder: newton.ModelBuilder, position: tuple[float, float, float]) -> int:
    return builder.add_link(
        xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
        mass=1.0,
        inertia=_INERTIA,
    )


def _build_all_joint_types() -> tuple[newton.Model, dict[str, tuple[int, int]]]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    joints: dict[str, tuple[int, int]] = {}
    x = 0.0

    child = _add_body(builder, (x, 0.0, 0.0))
    joints["ball"] = (builder.add_joint_ball(parent=-1, child=child), child)
    x += 2.0

    child = _add_body(builder, (x, 0.0, 0.0))
    joints["revolute"] = (
        builder.add_joint_revolute(
            parent=-1,
            child=child,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-np.inf,
            limit_upper=np.inf,
        ),
        child,
    )
    x += 2.0

    child = _add_body(builder, (x, 0.0, 0.0))
    joints["prismatic"] = (
        builder.add_joint_prismatic(
            parent=-1,
            child=child,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-np.inf,
            limit_upper=np.inf,
        ),
        child,
    )
    x += 2.0

    child = _add_body(builder, (x, 0.0, 0.0))
    joints["fixed"] = (
        builder.add_joint_fixed(
            parent=-1,
            child=child,
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
        ),
        child,
    )
    x += 2.0

    child = _add_body(builder, (x, 0.0, 0.0))
    axes = [
        newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
    ]
    joints["universal"] = (
        builder.add_joint_d6(
            parent=-1,
            child=child,
            angular_axes=axes,
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
        ),
        child,
    )
    x += 2.0

    parent = _add_body(builder, (x, 0.0, 0.0))
    child = _add_body(builder, (x, 0.0, -1.0))
    root = builder.add_joint_fixed(
        parent=-1,
        child=parent,
        parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
    )
    cable = builder.add_joint_cable(
        parent=parent,
        child=child,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        bend_stiffness=50.0,
        bend_damping=2.0,
        twist_stiffness=50.0,
        twist_damping=2.0,
    )
    builder.add_articulation([root, cable])
    joints["cable_root"] = (root, parent)
    joints["cable"] = (cable, child)

    # Cable validation requires its pair to be declared, but maximal PhoenX
    # must still discover every direct mechanism from joint connectivity.
    return builder.finalize(), joints


def _make_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=5,
        solver_iterations=1,
        velocity_iterations=1,
        articulation_mode="maximal",
    )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX direct-joint tests require CUDA")
class TestDirectJointTypes(unittest.TestCase):
    def test_every_supported_bilateral_mode_uses_direct_rows(self) -> None:
        """Route every supported bilateral joint mode through direct rows."""
        model, joints = _build_all_joint_types()
        solver = _make_solver(model)
        direct = solver._direct_equality_system
        self.assertTrue(direct.enabled)
        self.assertEqual(direct.topology.dimensions, (3, 5, 5, 6, 4, 12))
        expected_modes = {
            "ball": int(JOINT_MODE_BALL_SOCKET),
            "revolute": int(JOINT_MODE_REVOLUTE),
            "prismatic": int(JOINT_MODE_PRISMATIC),
            "fixed": int(JOINT_MODE_FIXED),
            "universal": int(JOINT_MODE_UNIVERSAL),
            "cable": int(JOINT_MODE_CABLE),
        }
        for kind, expected_mode in expected_modes.items():
            with self.subTest(kind=kind):
                joint, _body = joints[kind]
                self.assertTrue(bool(direct.joint_mask[joint]))
                cid = int(solver._adbs.joint_idx_to_cid.numpy()[joint])
                self.assertGreaterEqual(cid, 0)
                self.assertEqual(int(solver._adbs.joint_mode.numpy()[cid]), expected_mode)
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[cid]), 0)

    def test_every_supported_bilateral_mode_rejects_locked_velocity(self) -> None:
        """Reject locked velocity components with one direct mechanism solve."""
        model, joints = _build_all_joint_types()
        solver = _make_solver(model)
        expected_velocity = {
            "ball": np.asarray([0.0, 0.0, 0.0, 0.7, -0.6, 0.5]),
            "revolute": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            "prismatic": np.asarray([0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
            "fixed": np.zeros(6),
        }
        initial_velocity = np.asarray([0.2, -0.4, 0.3, 0.7, -0.6, 0.5], dtype=np.float32)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_qd = state.body_qd.numpy()
        for kind in ("ball", "revolute", "prismatic", "fixed", "cable", "universal"):
            _joint, body = joints[kind]
            body_qd[body] = initial_velocity
        state.body_qd.assign(body_qd)
        control = model.control()

        with wp.ScopedCapture(model.device) as capture:
            solver.step(state, state, control, None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)
        result = state.body_qd.numpy()
        self.assertTrue(np.isfinite(result).all())
        for kind in ("ball", "revolute", "prismatic", "fixed", "cable", "universal"):
            with self.subTest(kind=kind):
                _joint, body = joints[kind]
                if kind in expected_velocity:
                    np.testing.assert_allclose(result[body], expected_velocity[kind], rtol=2.0e-3, atol=2.0e-3)
                else:
                    self.assertLess(float(np.max(np.abs(result[body]))), 0.8)

    def test_mixed_small_and_large_direct_blocks_match_dense_residuals(self) -> None:
        """Solve narrow and wide ill-conditioned blocks in one launch set."""
        dimensions = (3, 5, 17, 40)
        starts = np.cumsum(np.asarray((0, *dimensions), dtype=np.int32))
        permutation = np.concatenate([np.arange(dimension, dtype=np.int32) for dimension in dimensions])
        row_bodies = tuple(
            frozenset((mechanism,)) for mechanism, dimension in enumerate(dimensions) for _ in range(dimension)
        )
        panel = FixedPatternPanelLLT(
            dimensions,
            starts,
            permutation,
            row_bodies,
            device=wp.get_preferred_device(),
        )
        np.testing.assert_array_equal(panel.small_mechanism.numpy(), [0, 1])
        np.testing.assert_array_equal(panel.large_mechanism.numpy(), [2, 3])
        np.testing.assert_array_equal(panel.partial_large_mechanism.numpy(), [2, 3])
        self.assertEqual(panel.narrow_mechanism.size, 0)

        rng = np.random.default_rng(1234)
        matrices = []
        expected_rhs = []
        for dimension in dimensions:
            basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            eigenvalues = np.geomspace(1.0, 1.0e4, dimension)
            matrices.append(basis @ np.diag(eigenvalues) @ basis.T)
            expected_rhs.append(rng.normal(size=dimension))

        storage = np.zeros(panel.matrix.size, dtype=np.float32)
        for row, column, address in zip(
            panel.symbolic.matrix_row,
            panel.symbolic.matrix_column,
            panel.symbolic.matrix_storage,
            strict=True,
        ):
            mechanism = int(np.searchsorted(starts[1:], row, side="right"))
            local_row = int(row - starts[mechanism])
            local_column = int(column - starts[mechanism])
            storage[address] = matrices[mechanism][local_row, local_column]
        rhs_np = np.concatenate(expected_rhs).astype(np.float32)
        rhs = wp.array(rhs_np, dtype=wp.float32, device=wp.get_preferred_device())
        solution = wp.zeros_like(rhs)
        panel.matrix.assign(storage)

        with wp.ScopedCapture(wp.get_preferred_device()) as capture:
            panel.compute()
            panel.solve(rhs, solution)
        wp.capture_launch(capture.graph)
        solution_np = solution.numpy()

        for mechanism, matrix in enumerate(matrices):
            begin = int(starts[mechanism])
            end = int(starts[mechanism + 1])
            residual = matrix @ solution_np[begin:end] - rhs_np[begin:end]
            relative_residual = np.linalg.norm(residual) / np.linalg.norm(rhs_np[begin:end])
            self.assertLess(relative_residual, 5.0e-3)

    def test_free_joint_emits_no_direct_or_pgs_rows(self) -> None:
        """Leave a free joint outside both direct and PGS constraint paths."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        free_body = builder.add_body(
            xform=wp.transform_identity(),
            mass=1.0,
            inertia=_INERTIA,
        )
        model = builder.finalize()
        solver = _make_solver(model)
        free_joint = int(np.flatnonzero(model.joint_child.numpy() == free_body)[0])
        self.assertFalse(bool(solver._direct_equality_system.joint_mask[free_joint]))
        self.assertEqual(int(solver._adbs.joint_idx_to_cid.numpy()[free_joint]), -1)


if __name__ == "__main__":
    unittest.main()
