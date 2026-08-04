# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-equality coverage for every maximal-coordinate PhoenX joint mode."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import (
    GROUPED_RHS_ITEM_WIDTH,
    GROUPED_RHS_ITEMS_PER_TASK,
    FixedPatternPanelLLT,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
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

    child = _add_body(builder, (x, 0.0, 0.0))
    joints["cylindrical"] = (
        builder.add_joint_d6(
            parent=-1,
            child=child,
            linear_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z)],
            angular_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z)],
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
        ),
        child,
    )
    x += 2.0

    child = _add_body(builder, (x, 0.0, 0.0))
    joints["planar"] = (
        builder.add_joint_d6(
            parent=-1,
            child=child,
            linear_axes=[
                newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X),
                newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            ],
            angular_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z)],
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
    return builder.finalize(device=wp.get_device("cuda:0")), joints


def _build_closed_loop_contact_model() -> tuple[newton.Model, tuple[int, ...], int]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    identity = wp.quat_identity()
    quarter_turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi)
    poses = (
        ((0.0, 0.0, 1.0), identity),
        ((0.5, 0.5, 1.0), quarter_turn),
        ((0.0, 1.0, 1.0), identity),
        ((-0.5, 0.5, 1.0), quarter_turn),
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.8)
    bodies = []
    for position, orientation in poses:
        body = builder.add_link(xform=wp.transform(wp.vec3(*position), orientation))
        builder.add_shape_box(body, hx=0.5, hy=0.05, hz=0.05, cfg=shape_cfg)
        bodies.append(body)

    root = builder.add_joint_free(parent=-1, child=bodies[0])
    connections = (
        (0, 1, (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        (1, 2, (0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        (2, 3, (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        (3, 0, (-0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)),
    )
    joints = []
    for parent, child, parent_anchor, child_anchor in connections:
        joints.append(
            builder.add_joint_revolute(
                parent=bodies[parent],
                child=bodies[child],
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(*parent_anchor), identity),
                child_xform=wp.transform(wp.vec3(*child_anchor), identity),
                limit_lower=-np.inf,
                limit_upper=np.inf,
            )
        )
    builder.add_articulation([root, *joints[:3]])
    builder.joint_articulation[joints[3]] = -1
    builder.color()
    return builder.finalize(device=wp.get_preferred_device()), tuple(bodies), joints[3]


def _build_mechanism_free_body_contact_model() -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.8)
    mechanism_body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    free_body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.18), wp.quat_identity()))
    builder.add_shape_box(mechanism_body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_joint_fixed(parent=-1, child=mechanism_body)
    builder.add_joint_free(parent=-1, child=free_body)
    builder.color()
    return builder.finalize(device=wp.get_preferred_device()), mechanism_body, free_body


def _joint_anchor_error(model: newton.Model, state: newton.State, joint: int) -> float:
    body_q = state.body_q.numpy()
    parent = int(model.joint_parent.numpy()[joint])
    child = int(model.joint_child.numpy()[joint])
    parent_anchor = np.asarray(
        wp.transform_point(
            wp.transform(*body_q[parent]),
            wp.vec3(*model.joint_X_p.numpy()[joint, :3]),
        )
    )
    child_anchor = np.asarray(
        wp.transform_point(
            wp.transform(*body_q[child]),
            wp.vec3(*model.joint_X_c.numpy()[joint, :3]),
        )
    )
    return float(np.linalg.norm(parent_anchor - child_anchor))


def _make_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=5,
        solver_iterations=1,
        velocity_iterations=1,
        articulation_mode="maximal",
    )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX direct-joint tests require CUDA")
class TestDirectJointTypes(unittest.TestCase):
    def test_direct_mechanism_contact_with_free_body_uses_schur_response(self) -> None:
        """Own a mechanism-to-free-body contact with the exact Schur response."""
        model, mechanism_body, free_body = _build_mechanism_free_body_contact_model()
        solver = _make_solver(model)
        state = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline = model._collision_pipeline
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, 1.0 / 60.0)

        response = solver._direct_contact_response
        self.assertIsNotNone(response)
        body_mechanism = response.data.body_mechanism.numpy()
        self.assertGreaterEqual(int(body_mechanism[mechanism_body + 1]), 0)
        self.assertEqual(int(body_mechanism[free_body + 1]), -1)
        owner = solver.world._contact_cols.articulation_owner.numpy()
        count = int(solver.world._ingest_scratch.num_contact_columns.numpy()[0])
        self.assertGreater(count, 0)
        self.assertTrue(np.all(owner[:count] >= 0))

    def test_every_supported_bilateral_mode_uses_direct_rows(self) -> None:
        """Route every supported bilateral joint mode through direct rows."""
        model, joints = _build_all_joint_types()
        solver = _make_solver(model)
        direct = solver._direct_equality_system
        self.assertTrue(direct.enabled)
        self.assertEqual(direct.topology.dimensions, (3, 5, 5, 6, 4, 4, 3, 12))
        expected_modes = {
            "ball": int(JOINT_MODE_BALL_SOCKET),
            "revolute": int(JOINT_MODE_REVOLUTE),
            "prismatic": int(JOINT_MODE_PRISMATIC),
            "fixed": int(JOINT_MODE_FIXED),
            "universal": int(JOINT_MODE_UNIVERSAL),
            "cylindrical": int(JOINT_MODE_CYLINDRICAL),
            "planar": int(JOINT_MODE_PLANAR),
            "cable": int(JOINT_MODE_CABLE),
        }
        for kind, expected_mode in expected_modes.items():
            with self.subTest(kind=kind):
                joint, _body = joints[kind]
                self.assertTrue(bool(direct.joint_mask[joint]))
                cid = int(solver._joint_constraints.joint_idx_to_cid.numpy()[joint])
                self.assertGreaterEqual(cid, 0)
                self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[cid]), expected_mode)
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
            "cylindrical": np.asarray([0.0, 0.0, 0.3, 0.0, 0.0, 0.5]),
            "planar": np.asarray([0.2, -0.4, 0.0, 0.0, 0.0, 0.5]),
        }
        initial_velocity = np.asarray([0.2, -0.4, 0.3, 0.7, -0.6, 0.5], dtype=np.float32)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_qd = state.body_qd.numpy()
        for kind in ("ball", "revolute", "prismatic", "fixed", "cable", "universal", "cylindrical", "planar"):
            _joint, body = joints[kind]
            body_qd[body] = initial_velocity
        state.body_qd.assign(body_qd)
        control = model.control()

        with wp.ScopedCapture(model.device) as capture:
            solver.step(state, state, control, None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)
        result = state.body_qd.numpy()
        self.assertTrue(np.isfinite(result).all())
        for kind in ("ball", "revolute", "prismatic", "fixed", "cable", "universal", "cylindrical", "planar"):
            with self.subTest(kind=kind):
                _joint, body = joints[kind]
                if kind in expected_velocity:
                    np.testing.assert_allclose(result[body], expected_velocity[kind], rtol=2.0e-3, atol=2.0e-3)
                else:
                    self.assertLess(float(np.max(np.abs(result[body]))), 0.8)

    def test_closed_loop_contacts_settle_without_horizontal_drift(self) -> None:
        """Keep a metadata-independent closed loop drift-free on the ground."""
        model, bodies, loop_joint = _build_closed_loop_contact_model()
        solver = _make_solver(model)
        self.assertTrue(solver._direct_equality_system.topology.mechanism_has_cycle[0])
        self.assertIsNotNone(solver._direct_contact_response)
        self.assertIsNotNone(solver._direct_contact_schedule)
        self.assertFalse(solver._direct_tree_contacts)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        pipeline = model._collision_pipeline
        contacts = pipeline.contacts()
        dt = 1.0 / 60.0

        def advance_pair() -> None:
            nonlocal state_0, state_1
            for _ in range(2):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

        advance_pair()

        with wp.ScopedCapture(model.device) as capture:
            advance_pair()
        for _ in range(15):
            wp.capture_launch(capture.graph)

        response = solver._direct_contact_response
        direct = solver._direct_equality_system
        contact_mechanism = response.data.contact_mechanism.numpy()
        contact = int(np.flatnonzero(contact_mechanism >= 0)[0])
        mechanism = int(contact_mechanism[contact])
        dimension = direct.topology.dimensions[mechanism]
        row_start = int(direct.topology.mechanism_row_start[mechanism])
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        matrix_storage = direct.matrix.numpy()
        for row, column, address in zip(
            direct.solver.symbolic.matrix_row,
            direct.solver.symbolic.matrix_column,
            direct.solver.symbolic.matrix_storage,
            strict=True,
        ):
            if row_start <= row < row_start + dimension:
                local_row = int(row - row_start)
                local_column = int(column - row_start)
                matrix[local_row, local_column] = matrix_storage[address]
                matrix[local_column, local_row] = matrix_storage[address]
        workspace_offset = contact * response.contact_batch.item_workspace_stride
        rhs_storage = response.contact_batch.rhs.numpy()
        rhs = np.stack(
            [
                rhs_storage[
                    workspace_offset + GROUPED_RHS_ITEM_WIDTH * row : workspace_offset
                    + GROUPED_RHS_ITEM_WIDTH * row
                    + 3
                ]
                for row in range(dimension)
            ]
        )
        lambdas = solver.world._contact_container.lambdas.numpy()
        derived = solver.world._contact_container.derived.numpy()
        normal = lambdas[0:3, contact]
        tangent0 = lambdas[3:6, contact]
        directions = np.stack((normal, tangent0, np.cross(normal, tangent0)))
        r0 = derived[9:12, contact]
        r1 = derived[12:15, contact]
        body0 = int(response.contact_body0.numpy()[contact])
        body1 = int(response.contact_body1.numpy()[contact])
        inverse_mass = solver.bodies.inverse_mass.numpy()
        packed_inertia = solver.bodies.inverse_inertia_world.numpy()
        raw = np.zeros((3, 3), dtype=np.float64)
        for body, lever, sign in ((body0, r0, -1.0), (body1, r1, 1.0)):
            packed = packed_inertia[body]
            inverse_inertia = np.asarray(
                (
                    (packed[0], packed[3], packed[4]),
                    (packed[3], packed[1], packed[5]),
                    (packed[4], packed[5], packed[2]),
                )
            )
            forces = sign * directions
            torques = np.cross(np.broadcast_to(lever, forces.shape), forces)
            raw += inverse_mass[body] * forces @ forces.T + torques @ inverse_inertia @ torques.T
        expected_inverse = raw - rhs.T @ np.linalg.solve(matrix, rhs)
        mobility = response.data.mobility.numpy()[:, contact]
        actual_inverse = np.asarray(
            (
                (1.0 / mobility[0], mobility[3], mobility[4]),
                (mobility[3], 1.0 / mobility[1], mobility[5]),
                (mobility[4], mobility[5], 1.0 / mobility[2]),
            )
        )
        np.testing.assert_allclose(actual_inverse, expected_inverse, rtol=3.0e-4, atol=3.0e-5)

        for _ in range(45):
            wp.capture_launch(capture.graph)
        settled_center = np.mean(state_0.body_q.numpy()[list(bodies), :3], axis=0)
        for _ in range(60):
            wp.capture_launch(capture.graph)

        body_q = state_0.body_q.numpy()[list(bodies)]
        body_qd = state_0.body_qd.numpy()[list(bodies)]
        self.assertTrue(np.isfinite(body_q).all())
        self.assertTrue(np.isfinite(body_qd).all())
        center = np.mean(body_q[:, :3], axis=0)
        self.assertLess(float(np.linalg.norm(center[:2] - (0.0, 0.5))), 1.0e-2)
        self.assertLess(float(np.linalg.norm(center[:2] - settled_center[:2])), 2.0e-4)
        self.assertLess(float(np.max(np.abs(body_qd))), 2.0e-2)
        self.assertLess(_joint_anchor_error(model, state_0, loop_joint), 5.0e-4)

        column_count = int(solver.world._ingest_scratch.num_contact_columns.numpy()[0])
        self.assertGreater(column_count, 0)
        owners = solver.world._contact_cols.articulation_owner.numpy()[:column_count]
        self.assertTrue(np.all(owners >= 0))

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
        np.testing.assert_array_equal(panel.large_mechanism.numpy(), [2, 3])
        np.testing.assert_array_equal(panel.cooperative_mechanism.numpy(), [0, 1])
        self.assertTrue(panel._use_push_solve)
        self.assertEqual(panel.cooperative_factor_mechanism.size, 0)

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

    def test_panel_grouped_rhs_batch_preserves_mechanism_boundaries(self) -> None:
        """Solve padded contact groups across heterogeneous mechanisms."""
        dimensions = (5, 17, 40)
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
        batch = panel.create_grouped_rhs_batch(item_capacity=8, task_capacity=3)
        task_mechanisms = np.asarray((2, 0, 1), dtype=np.int32)
        task_items = np.full(3 * GROUPED_RHS_ITEMS_PER_TASK, -1, dtype=np.int32)
        for task, items in enumerate(((0, 1), (2,), (3, 4, 5, 6))):
            task_items[task * GROUPED_RHS_ITEMS_PER_TASK : task * GROUPED_RHS_ITEMS_PER_TASK + len(items)] = items
        batch.task_mechanism.assign(task_mechanisms)
        batch.task_item.assign(task_items)

        rng = np.random.default_rng(7281)
        matrices = []
        matrix_storage = np.zeros(panel.matrix.size, dtype=np.float32)
        for dimension in dimensions:
            basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            matrices.append(basis @ np.diag(np.geomspace(1.0, 1.0e3, dimension)) @ basis.T)
        for row, column, address in zip(
            panel.symbolic.matrix_row,
            panel.symbolic.matrix_column,
            panel.symbolic.matrix_storage,
            strict=True,
        ):
            mechanism = int(np.searchsorted(starts[1:], row, side="right"))
            matrix_storage[address] = matrices[mechanism][row - starts[mechanism], column - starts[mechanism]]
        panel.matrix.assign(matrix_storage)

        rhs_storage = np.zeros(batch.rhs.size, dtype=np.float32)
        expected_rhs = {}
        for task, mechanism in enumerate(task_mechanisms):
            for slot in range(GROUPED_RHS_ITEMS_PER_TASK):
                item = int(task_items[GROUPED_RHS_ITEMS_PER_TASK * task + slot])
                if item < 0:
                    continue
                rhs = rng.normal(size=(dimensions[mechanism], 3)).astype(np.float32)
                expected_rhs[item] = (int(mechanism), rhs)
                begin = item * batch.item_workspace_stride
                for row in range(rhs.shape[0]):
                    rhs_storage[begin + GROUPED_RHS_ITEM_WIDTH * row : begin + GROUPED_RHS_ITEM_WIDTH * row + 3] = rhs[
                        row
                    ]
        batch.rhs.assign(rhs_storage)

        with wp.ScopedCapture(wp.get_preferred_device()) as capture:
            panel.compute()
            batch.solve()
        wp.capture_launch(capture.graph)
        solution_storage = batch.solution.numpy()

        for item, (mechanism, rhs) in expected_rhs.items():
            begin = item * batch.item_workspace_stride
            solution = np.stack(
                [
                    solution_storage[begin + GROUPED_RHS_ITEM_WIDTH * row : begin + GROUPED_RHS_ITEM_WIDTH * row + 3]
                    for row in range(rhs.shape[0])
                ]
            )
            residual = matrices[mechanism] @ solution - rhs
            self.assertLess(float(np.linalg.norm(residual) / np.linalg.norm(rhs)), 5.0e-3)

    def test_wide_partial_panels_match_dense_residual(self) -> None:
        """Match a sparse residual with a partial cooperative panel."""
        dimension = 40
        starts = np.asarray((0, dimension), dtype=np.int32)
        permutation = np.arange(dimension, dtype=np.int32)
        row_bodies = tuple(frozenset((row, row + 1)) for row in range(dimension))
        panel = FixedPatternPanelLLT(
            (dimension,),
            starts,
            permutation,
            row_bodies,
            device=wp.get_preferred_device(),
        )
        self.assertEqual(panel.block_size, 16)
        np.testing.assert_array_equal(panel.cooperative_mechanism.numpy(), [0])
        self.assertFalse(panel._use_push_solve)

        rng = np.random.default_rng(5678)
        matrix = np.eye(dimension, dtype=np.float64) * 2.0
        matrix[np.arange(dimension - 1), np.arange(1, dimension)] = 0.1
        matrix[np.arange(1, dimension), np.arange(dimension - 1)] = 0.1
        rhs_np = rng.normal(size=dimension).astype(np.float32)

        storage = np.zeros(panel.matrix.size, dtype=np.float32)
        for row, column, address in zip(
            panel.symbolic.matrix_row,
            panel.symbolic.matrix_column,
            panel.symbolic.matrix_storage,
            strict=True,
        ):
            storage[address] = matrix[row, column]
        panel.matrix.assign(storage)
        rhs = wp.array(rhs_np, dtype=wp.float32, device=wp.get_preferred_device())
        solution = wp.zeros_like(rhs)

        with wp.ScopedCapture(wp.get_preferred_device()) as capture:
            panel.compute()
            panel.solve(rhs, solution)
        wp.capture_launch(capture.graph)

        residual = matrix @ solution.numpy() - rhs_np
        relative_residual = np.linalg.norm(residual) / np.linalg.norm(rhs_np)
        self.assertLess(relative_residual, 5.0e-3)

    def test_branching_panel_solve_uses_global_ready_queue(self) -> None:
        """Solve a branching panel graph deterministically through the global queue."""
        dimension = 64
        starts = np.asarray((0, dimension), dtype=np.int32)
        permutation = np.arange(dimension, dtype=np.int32)
        row_bodies = tuple(
            frozenset((tile,)) if tile < 3 else frozenset((0, 1, 2)) for tile in range(4) for _ in range(16)
        )
        panel = FixedPatternPanelLLT(
            (dimension,),
            starts,
            permutation,
            row_bodies,
            device=wp.get_preferred_device(),
        )
        self.assertTrue(panel._use_push_solve)
        self.assertEqual(panel._push_forward_schedule.max_ready_count, 3)
        self.assertEqual(panel._push_backward_schedule.max_ready_count, 3)

        matrix = np.eye(dimension, dtype=np.float32) * 2.0
        storage = np.zeros(panel.matrix.size, dtype=np.float32)
        for row, column, address in zip(
            panel.symbolic.matrix_row,
            panel.symbolic.matrix_column,
            panel.symbolic.matrix_storage,
            strict=True,
        ):
            if row != column:
                matrix[row, column] = 1.0e-3
                matrix[column, row] = 1.0e-3
            storage[address] = matrix[row, column]

        rng = np.random.default_rng(4321)
        rhs_np = rng.normal(size=dimension).astype(np.float32)
        rhs = wp.array(rhs_np, dtype=wp.float32, device=wp.get_preferred_device())
        solution = wp.zeros_like(rhs)
        panel.matrix.assign(storage)

        with wp.ScopedCapture(wp.get_preferred_device()) as capture:
            panel.compute()
            panel.solve(rhs, solution)
        wp.capture_launch(capture.graph)
        first_solution = solution.numpy()
        wp.capture_launch(capture.graph)
        second_solution = solution.numpy()

        np.testing.assert_array_equal(second_solution, first_solution)
        np.testing.assert_allclose(matrix @ second_solution, rhs_np, rtol=2.0e-4, atol=2.0e-4)

    def test_many_mechanisms_use_cooperative_factorization(self) -> None:
        """Factor a mechanism fleet without global queue scratch."""
        device = wp.get_preferred_device()
        mechanism_count = max(1, int(device.sm_count))
        dimension = 48
        dimensions = (dimension,) * mechanism_count
        starts = np.arange(mechanism_count + 1, dtype=np.int32) * dimension
        permutation = np.tile(np.arange(dimension, dtype=np.int32), mechanism_count)
        row_bodies = tuple(frozenset((mechanism,)) for mechanism in range(mechanism_count) for _ in range(dimension))
        panel = FixedPatternPanelLLT(
            dimensions,
            starts,
            permutation,
            row_bodies,
            device=device,
        )

        self.assertFalse(panel._use_product_factor)
        self.assertIsNone(panel._product_factor_schedule)
        self.assertFalse(panel._use_push_solve)
        self.assertIsNone(panel._push_forward_schedule)
        self.assertIsNone(panel._push_backward_schedule)
        self.assertEqual(panel.cooperative_factor_mechanism.size, mechanism_count)
        self.assertIsNone(panel._persistent_schedule)

        storage = np.zeros(panel.matrix.size, dtype=np.float32)
        diagonal = np.empty(mechanism_count * dimension, dtype=np.float32)
        for row, column, address in zip(
            panel.symbolic.matrix_row,
            panel.symbolic.matrix_column,
            panel.symbolic.matrix_storage,
            strict=True,
        ):
            if row == column:
                value = 2.0 + 0.01 * (row % dimension)
                storage[address] = value
                diagonal[row] = value
        rhs = wp.ones(mechanism_count * dimension, dtype=wp.float32, device=device)
        solution = wp.zeros_like(rhs)
        panel.matrix.assign(storage)

        with wp.ScopedCapture(device) as capture:
            panel.compute()
            panel.solve(rhs, solution)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(solution.numpy(), 1.0 / diagonal, rtol=2.0e-5, atol=2.0e-5)

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
        self.assertEqual(int(solver._joint_constraints.joint_idx_to_cid.numpy()[free_joint]), -1)


if __name__ == "__main__":
    unittest.main()
