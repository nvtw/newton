# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed-bound joint preparation preserves the dynamic-bound mass metric."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints.bilateral_joint import (
    BodyContainer,
    ConstraintContainer,
    CopyStateContainer,
    Mat66d,
    Vec6d,
    _dot_double,
    _prepare_bilateral_joint_blocks_cooperative,
    _response,
    constraint_get_body1,
    constraint_get_body2,
    iterate_bilateral_joint_block,
    prepare_bilateral_joint_blocks,
)
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@wp.kernel(enable_backward=False)
def _prepare_dynamic_reference(constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer):
    cid = wp.tid()
    data = constraints.bilateral
    count = data.row_count[cid]
    data.valid[cid] = wp.int32(0)
    if count == 0:
        return
    structural = data.structural_index[cid]
    body0 = constraint_get_body1(constraints, cid)
    body1 = constraint_get_body2(constraints, cid)
    factor0 = wp.float32(1.0)
    factor1 = wp.float32(1.0)
    if copy_state.highest_index_in_use[0] > 0:
        factor0 = wp.float32(wp.max(copy_state.count_per_node[body0], wp.int32(1)))
        factor1 = wp.float32(wp.max(copy_state.count_per_node[body1], wp.int32(1)))
    for i in range(count):
        row = data.row_indices[cid, i]
        local = data.row_local[row]
        data.response0[cid, i] = factor0 * _response(bodies, body0, data.wrench0[structural, local])
        data.response1[cid, i] = factor1 * _response(bodies, body1, data.wrench1[structural, local])

    matrix = Mat66d()
    for i in range(count):
        row = data.row_indices[cid, i]
        local = data.row_local[row]
        j0 = data.wrench0[structural, local]
        j1 = data.wrench1[structural, local]
        for j in range(i + 1):
            value = _dot_double(j0, data.response0[cid, j]) + _dot_double(j1, data.response1[cid, j])
            if i == j and data.row_dynamic[row]:
                value += wp.float64(1.0) / wp.float64(data.dynamic_mass[row])
            matrix[i, j] = value
            matrix[j, i] = value

    lower = Mat66d()
    diagonal = Vec6d()
    valid = wp.bool(True)
    for i in range(count):
        pivot = matrix[i, i]
        for j in range(i):
            pivot -= lower[i, j] * lower[i, j] * diagonal[j]
        diagonal[i] = pivot
        lower[i, i] = wp.float64(1.0)
        if pivot <= wp.float64(0.0):
            valid = False
        if valid:
            for j in range(i + 1, count):
                value = matrix[j, i]
                for k in range(i):
                    value -= lower[j, k] * lower[i, k] * diagonal[k]
                lower[j, i] = value / pivot
    data.lower[cid] = lower
    data.diagonal[cid] = diagonal
    if valid:
        data.valid[cid] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _iterate_overflow_joint(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copies: CopyStateContainer,
    num_bodies: int,
    cid: int,
    partition: int,
):
    iterate_bilateral_joint_block(constraints, cid, bodies, particles, copies, num_bodies, partition, False)


@unittest.skipUnless(_cuda_with_graph_capture(), "Preparation reference requires CUDA")
class TestBilateralPreparation(unittest.TestCase):
    def test_overflow_joint_factor_matches_unequal_copy_response(self):
        """A native overflow joint uses the same physical mass scaling in factor and scatter."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        links = []
        for x, mass in ((-1.0, 1.0), (0.0, 3.0), (1.0, 2.0)):
            links.append(
                builder.add_link(
                    xform=wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()),
                    mass=mass,
                    inertia=wp.mat33(mass * 0.2, 0, 0, 0, mass * 0.3, 0, 0, 0, mass * 0.4),
                )
            )
        joints = []
        for a in range(2):
            joints.append(
                builder.add_joint_revolute(
                    parent=links[a],
                    child=links[a + 1],
                    axis=(0, 0, 1),
                    parent_xform=wp.transform(wp.vec3(0.5, 0.15, -0.1), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.5, 0.15, -0.1), wp.quat_identity()),
                )
            )
        builder.add_articulation(joints)
        model = builder.finalize(device="cuda:0")
        solver = make_solver(model, mass_splitting=True, max_colored_partitions=0, mass_splitting_batch_size=1)
        state = model.state()
        solver.step(state, state, model.control(), None, 0.001)
        world = solver.world
        data, copies = world.constraints.bilateral, world._copy_state
        counts = copies.count_per_node.numpy()
        self.assertEqual(counts[1:4].tolist(), [1, 2, 1])
        ids = world._partitioner.element_ids_by_color.numpy()
        cid = 0
        partition = int(np.flatnonzero(ids[:2] == cid)[0])
        self.assertEqual(int(world._partitioner.interaction_id_to_partition.numpy()[cid]), 0)
        rows = data.row_indices.numpy()[cid, : int(data.row_count.numpy()[cid])]
        local = data.row_local.numpy()[rows]
        structural = int(data.structural_index.numpy()[cid])
        jacobians = [
            data.wrench0.numpy()[structural, local].astype(float),
            data.wrench1.numpy()[structural, local].astype(float),
        ]
        inverse_mass = world.bodies.inverse_mass.numpy().astype(float)
        inverse_inertia = inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()).astype(float)
        mobility = []
        for body in (1, 2):
            w = np.zeros((6, 6))
            w[:3, :3] = np.eye(3) * inverse_mass[body]
            w[3:, 3:] = inverse_inertia[body]
            mobility.append(w)
        initial = np.array([[0.2, -0.1, 0.3, 0.1, 0.2, -0.2], [-0.3, 0.15, -0.1, -0.2, 0.1, 0.3]])
        matrix = sum(j @ (counts[b] * w) @ j.T for j, w, b in zip(jacobians, mobility, (1, 2), strict=True))
        expected_lambda = np.linalg.solve(matrix, -sum(j @ v for j, v in zip(jacobians, initial, strict=True)))
        expected_change = [w @ j.T @ expected_lambda for w, j in zip(mobility, jacobians, strict=True)]
        highest = copies.highest_index_in_use.numpy().copy()
        outputs = []
        for wrong_factor in (True, False):
            velocity = np.zeros_like(world.bodies.velocity.numpy())
            spin = np.zeros_like(world.bodies.angular_velocity.numpy())
            velocity[1:3], spin[1:3] = initial[:, :3], initial[:, 3:]
            world.bodies.velocity.assign(velocity)
            world.bodies.angular_velocity.assign(spin)
            world._mass_splitting_broadcast()
            data.accumulated.zero_()
            if wrong_factor:
                copies.highest_index_in_use.zero_()  # Deliberate process-local count-one factor defect.
            try:
                wp.launch(
                    prepare_bilateral_joint_blocks,
                    world.num_joints,
                    [world.constraints, world.bodies, copies],
                    device=model.device,
                )
            finally:
                copies.highest_index_in_use.assign(highest)
            np.testing.assert_array_equal(copies.highest_index_in_use.numpy(), highest)
            np.testing.assert_array_equal(copies.count_per_node.numpy(), counts)
            wp.launch(
                _iterate_overflow_joint,
                1,
                [
                    world.constraints,
                    world.bodies,
                    world._particles_or_sentinel(),
                    copies,
                    world.num_bodies,
                    cid,
                    partition,
                ],
                device=model.device,
            )
            world._mass_splitting_writeback()
            actual = np.concatenate(
                (world.bodies.velocity.numpy()[1:3], world.bodies.angular_velocity.numpy()[1:3]), axis=1
            ).astype(float)
            outputs.append(actual - initial)
            self.assertTrue(np.all(np.isfinite(actual)))
            if not wrong_factor:
                np.testing.assert_allclose(data.accumulated.numpy()[rows], expected_lambda, atol=3e-7, rtol=2e-6)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(outputs[0], expected_change, atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(outputs[1], expected_change, atol=3e-7, rtol=2e-6)
        positions = world.bodies.position.numpy()[1:3].astype(float)
        impulses = [np.linalg.solve(w, delta) for w, delta in zip(mobility, outputs[1], strict=True)]
        linear = sum(p[:3] for p in impulses)
        angular = sum(p[3:] + np.cross(x, p[:3]) for x, p in zip(positions, impulses, strict=True))
        np.testing.assert_allclose(np.r_[linear, angular], 0, atol=5e-7, rtol=0)

    def test_fixed_bounds_match_active_rows_and_mass_scales(self):
        """All 84 active-row, copy-count, mass-ratio, and wrench-scale cases stay exact."""
        model = make_model(40.0)
        model.joint_effort_limit.assign(np.asarray([1000.0], dtype=np.float32))
        solver = make_solver(model, mass_splitting=True, max_colored_partitions=0)
        state = model.state()
        solver.step(state, state, model.control(), None, 0.01)
        world = solver.world
        data = world.constraints.bilateral
        rng = np.random.default_rng(7819)
        fields = ("response0", "response1", "lower", "diagonal", "valid")
        original_wrenches = {field: getattr(data, field).numpy().copy() for field in ("wrench0", "wrench1")}
        original_mass = world.bodies.inverse_mass.numpy().copy()
        original_inertia = world.bodies.inverse_inertia_world.numpy().copy()
        for geometry, ratio, split in itertools.product(("authored", "random", "scaled"), (1.0, 400.0), (False, True)):
            mass, inertia = original_mass.copy(), original_inertia.copy()
            mass[-1] /= ratio
            inertia[-1] /= ratio
            world.bodies.inverse_mass.assign(mass)
            world.bodies.inverse_inertia_world.assign(inertia)
            world._copy_state.highest_index_in_use.assign(np.asarray([int(split)], dtype=np.int32))
            counts = np.arange(len(world._copy_state.count_per_node), dtype=np.int32) * 9 + 1
            world._copy_state.count_per_node.assign(counts)
            for field, original in original_wrenches.items():
                value = original.copy()
                if geometry != "authored":
                    value = rng.normal(size=value.shape).astype(np.float32)
                    if geometry == "scaled":
                        value *= np.logspace(-2, 2, value.shape[1], dtype=np.float32)[None, :, None]
                getattr(data, field).assign(value)
            for count in range(7):
                data.row_count.assign(np.full(data.row_count.shape, count, dtype=np.int32))
                outputs = []
                for active, block in (
                    (_prepare_dynamic_reference, 0),
                    (prepare_bilateral_joint_blocks, 0),
                    (_prepare_bilateral_joint_blocks_cooperative, 32),
                    (_prepare_bilateral_joint_blocks_cooperative, 64),
                ):
                    for field in fields:
                        getattr(data, field).zero_()
                    wp.launch(
                        active,
                        world.num_joints * (8 if block else 1),
                        [world.constraints, world.bodies, world._copy_state],
                        device=model.device,
                        block_dim=block or 256,
                    )
                    outputs.append({field: getattr(data, field).numpy() for field in fields})
                case = {"count": count, "geometry": geometry, "mass_ratio": ratio, "split": split}
                for field in fields:
                    if not np.all(np.isfinite(outputs[0][field])):
                        raise AssertionError(f"Nonfinite reference: {case}, {field}")
                    for actual in outputs[1:]:
                        self.assertEqual(outputs[0][field].tobytes(), actual[field].tobytes(), f"{case}, field={field}")

    def test_cooperative_ragged_rows_invalid_pivots_and_capture(self):
        """Complete subgroups preserve stale zero-row fields and invalid factors."""
        model = make_model(40.0)
        solver = make_solver(model, mass_splitting=True, max_colored_partitions=0)
        state = model.state()
        solver.step(state, state, model.control(), None, 0.01)
        world = solver.world
        constraints = world.constraints
        data = constraints.bilateral
        count = 7
        constraints.data = wp.array(
            np.repeat(constraints.data.numpy()[:, :1], count, axis=1), dtype=wp.float32, device=model.device
        )
        data.row_count = wp.array(np.arange(count, dtype=np.int32), device=model.device)
        data.row_indices = wp.array(
            np.repeat(data.row_indices.numpy()[:1], count, axis=0), dtype=wp.int32, device=model.device
        )
        data.structural_index = wp.array(
            np.repeat(data.structural_index.numpy()[:1], count), dtype=wp.int32, device=model.device
        )
        fields = ("response0", "response1", "lower", "diagonal", "valid")
        for name, dtype, shape, scalar in (
            ("response0", wp.spatial_vector, (count, 6, 6), np.float32),
            ("response1", wp.spatial_vector, (count, 6, 6), np.float32),
            ("lower", Mat66d, (count, 6, 6), np.float64),
            ("diagonal", Vec6d, (count, 6), np.float64),
            ("valid", wp.int32, (count,), np.int32),
        ):
            setattr(data, name, wp.array(np.full(shape, 17, dtype=scalar), dtype=dtype, device=model.device))
        constraints.bilateral = data
        inputs = [constraints, world.bodies, world._copy_state]
        initial = {name: getattr(data, name).numpy() for name in fields}
        for invalid in (False, True):
            if invalid:
                data.wrench0.zero_()
                data.wrench1.zero_()
                data.row_dynamic.zero_()
            expected = None
            for block in (0, 32, 64):
                for name, values in initial.items():
                    getattr(data, name).assign(values)
                if block:
                    with wp.ScopedCapture(device=model.device) as capture:
                        wp.launch(
                            _prepare_bilateral_joint_blocks_cooperative,
                            8 * count,
                            inputs,
                            device=model.device,
                            block_dim=block,
                        )
                    wp.capture_launch(capture.graph)
                else:
                    wp.launch(prepare_bilateral_joint_blocks, count, inputs, device=model.device)
                actual = {name: getattr(data, name).numpy() for name in fields}
                if expected is None:
                    expected = actual
                else:
                    for name in fields:
                        self.assertEqual(
                            expected[name].tobytes(), actual[name].tobytes(), f"{invalid=}, {block=}, {name=}"
                        )
                for name in fields[:-1]:
                    self.assertEqual(actual[name][0].tobytes(), initial[name][0].tobytes(), name)
                self.assertEqual(int(actual["valid"][0]), 0)


class TestBilateralPreparationDispatch(unittest.TestCase):
    def test_cpu_scalar_and_cuda_complete_subgroups(self):
        """Select the scalar CPU kernel or complete CUDA eight-lane groups."""
        from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem

        world = SimpleNamespace(num_joints=7, constraints=object(), bodies=object(), _copy_state=object())
        for cuda in (False, True):
            device = SimpleNamespace(is_cuda=cuda)
            system = SimpleNamespace(enabled=True, _block_world=world, model=SimpleNamespace(device=device))
            with patch.object(wp, "launch") as launched:
                BlockJointSystem.prepare_and_factor(system, 100.0)
            kernel = _prepare_bilateral_joint_blocks_cooperative if cuda else prepare_bilateral_joint_blocks
            self.assertIs(launched.call_args.args[0], kernel)
            self.assertEqual(launched.call_args.kwargs["dim"], 56 if cuda else 7)
            if cuda:
                self.assertEqual(launched.call_args.kwargs["block_dim"], 32)


if __name__ == "__main__":
    unittest.main()
