# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared body response bases retain the column response and task ownership."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.direct_contact_gs import (
    _fill_direct_contact_rhs_tasks_kernel,
    _row_correction_from_wrenches,
)
from newton._src.solvers.phoenx.articulations.direct_contact_response import (
    DirectContactResponse,
    DirectContactResponseData,
    _build_grouped_endpoint_equality_rhs_kernel,
)
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, body_container_zeros, inertia_sym6_pack_np
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_normal,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1,
    contact_container_zeros,
)


@wp.kernel(enable_backward=False)
def _initialize_contacts(contacts: ContactContainer):
    contact = wp.tid()
    cc_set_normal(contacts, contact, wp.vec3(0.0, 0.0, 1.0))
    cc_set_tangent1(contacts, contact, wp.vec3(1.0, 0.0, 0.0))
    cc_set_r0(contacts, contact, wp.vec3(0.1, 0.2, -0.15))
    cc_set_r1(contacts, contact, wp.vec3(-0.15, 0.12, -0.13))


@wp.kernel(enable_backward=False)
def _evaluate_deferred_coefficients(response: DirectContactResponseData, coefficients: wp.array2d[wp.float32]):
    column, row = wp.tid()
    mechanism = wp.int32(0)
    local_row = row
    if row >= wp.int32(3):
        mechanism = wp.int32(1)
        local_row = row - wp.int32(3)
    coefficients[column, row] = _row_correction_from_wrenches(
        response,
        mechanism,
        column,
        local_row,
        wp.spatial_vector(0.3, -0.4, 0.7, 0.2, 0.5, -0.8),
        wp.spatial_vector(-0.3, 0.4, -0.7, -0.1, 0.6, 0.2),
    )


def _fixture(device, *, unconstrained_bodies=0):
    """Two floating D6 chains: two and three dynamic bodies, three rows/joint."""

    def array(values, dtype=wp.float32):
        return wp.array(np.asarray(values), dtype=dtype, device=device)

    parent = np.asarray([0, 2, 3], dtype=np.int32)
    child = parent + 1
    row_joint = np.repeat(np.arange(3, dtype=np.int32), 3)
    mass = np.asarray([1.0, 2.0, 1.5, 2.5, 3.0])
    inertia = mass[:, None, None] * np.diag([0.2, 0.3, 0.4])[None]
    inverse = np.zeros((30, 30))
    for body in range(5):
        inverse[6 * body : 6 * body + 3, 6 * body : 6 * body + 3] = np.eye(3) / mass[body]
        inverse[6 * body + 3 : 6 * body + 6, 6 * body + 3 : 6 * body + 6] = np.linalg.inv(inertia[body])
    jacobian = np.zeros((9, 30))
    wrench0 = np.zeros((3, 3, 6), dtype=np.float32)
    wrench1 = np.zeros_like(wrench0)
    for joint in range(3):
        for axis, direction in enumerate(np.eye(3)):
            wrench0[joint, axis] = np.r_[-direction, np.cross([0.15, 0.02, 0], -direction)]
            wrench1[joint, axis] = np.r_[direction, np.cross([-0.15, 0.02, 0], direction)]
            jacobian[3 * joint + axis, 6 * parent[joint] : 6 * parent[joint] + 6] = wrench0[joint, axis]
            jacobian[3 * joint + axis, 6 * child[joint] : 6 * child[joint] + 6] = wrench1[joint, axis]
    starts = np.asarray([0, 3, 9], dtype=np.int32)
    panel = FixedPatternPanelLLT(
        (3, 6),
        starts,
        np.r_[np.arange(3), np.arange(6)].astype(np.int32),
        tuple(frozenset((int(parent[joint]), int(child[joint]))) for joint in row_joint),
        device=device,
    )
    matrix = jacobian @ inverse @ jacobian.T
    packed = np.zeros(panel.matrix.shape[0], dtype=np.float32)
    packed[panel.symbolic.matrix_storage] = matrix[panel.symbolic.matrix_row, panel.symbolic.matrix_column]
    panel.matrix.assign(packed)
    body_count = 6 + unconstrained_bodies
    inverse_mass = np.r_[1 / mass, np.ones(unconstrained_bodies)].astype(np.float32)
    bodies = body_container_zeros(body_count, device=device)
    bodies.inverse_mass.assign(np.r_[0.0, inverse_mass].astype(np.float32))
    bodies.inverse_inertia_world.assign(
        inertia_sym6_pack_np(
            np.concatenate(
                (np.zeros((1, 3, 3)), np.linalg.inv(inertia), np.tile(np.eye(3), (unconstrained_bodies, 1, 1)))
            )
        )
    )
    bodies.motion_type.assign(np.asarray([0] + [int(MOTION_DYNAMIC)] * (body_count - 1), dtype=np.int32))
    body_rows = [
        [row for row, joint in enumerate(row_joint) if body in (parent[joint] + 1, child[joint] + 1)]
        for body in range(body_count)
    ]
    model = SimpleNamespace(
        device=wp.get_device(device),
        body_inv_mass=array(inverse_mass),
        joint_parent=array(parent, wp.int32),
        joint_child=array(child, wp.int32),
    )
    return SimpleNamespace(
        enabled=True,
        model=model,
        bodies=bodies,
        solver=panel,
        topology=SimpleNamespace(dimensions=(3, 6), mechanism_row_start=starts, row_joint=row_joint),
        body_row_start=array(np.r_[0, np.cumsum([len(rows) for rows in body_rows])], wp.int32),
        body_rows=array([row for rows in body_rows for row in rows], wp.int32),
        row_joint=array(row_joint, wp.int32),
        row_local=array(np.tile(np.arange(3), 3), wp.int32),
        joint_to_structural=array(np.arange(3), wp.int32),
        row_wrench0=array(wrench0, wp.spatial_vector),
        row_wrench1=array(wrench1, wp.spatial_vector),
        row_scale=array(np.ones(9)),
        accumulated_impulse=wp.zeros(9, device=device),
    )


def _compute_response(response, contacts, device):
    if not wp.get_device(device).is_cuda:
        # CPU launch_tiled does not execute the 64 cooperating CUDA lanes.
        # Populate every lane's RHS before testing the common solver math.
        wp.launch(
            _build_grouped_endpoint_equality_rhs_kernel,
            (response.contact_batch.task_capacity, 64),
            [
                response.data,
                response.direct.bodies,
                response.contact_batch.task_item,
                response.contact_batch.task_mechanism,
            ],
            device=device,
        )
    response.compute(contacts)


class TestDirectBodyResponse(unittest.TestCase):
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_storage_guard_includes_unconstrained_body_maps(self):
        """Retain column storage when world-sized lookup maps erase savings."""
        with wp.ScopedDevice("cpu"):
            direct = _fixture("cpu", unconstrained_bodies=1024)
            response = DirectContactResponse(direct, 8, 8)
            self.assertFalse(response.shared_body_basis)
            self.assertEqual(response.data.body_basis_item.size, 0)
            self.assertEqual(response.data.body_basis_task.size, 0)
            self.assertEqual(response.data.body_basis_active.size, 0)
            self.assertEqual(response.contact_batch.item_capacity, 32)

    def test_compact_body_tasks_and_storage_fallback(self):
        """Pack selected body bases and retain the smaller column fallback."""
        with wp.ScopedDevice(self.device):
            direct = _fixture(self.device)
            shared = DirectContactResponse(direct, 8, 8)
            # Old per-column implementation allocates 32 items and 16 tasks.
            self.assertEqual(shared.contact_batch.item_capacity, 10)
            self.assertEqual(shared.contact_batch.task_capacity, 3)
            items = shared.contact_batch.task_item.numpy().reshape(-1, 4)
            np.testing.assert_array_equal(items, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, -1]])
            np.testing.assert_array_equal(shared.contact_batch.task_mechanism.numpy(), [-1, -1, -1])
            np.testing.assert_array_equal(shared.data.body_basis_item.numpy(), [-1, 0, 2, 4, 6, 8])
            np.testing.assert_array_equal(shared.data.body_basis_task.numpy(), [-1, 0, 0, 1, 1, 2])
            small = DirectContactResponse(direct, 2, 2)
            self.assertFalse(small.shared_body_basis)
            self.assertEqual(small.contact_batch.item_capacity, 8)
            inactive = DirectContactResponse(direct, 8, 8, active_mechanisms=(False, True))
            np.testing.assert_array_equal(inactive.data.body_basis_item.numpy(), [-1, -1, -1, 0, 2, 4])
            self.assertEqual(inactive.contact_batch.task_capacity, 2)

    def test_grouped_response_matches_column_response(self):
        """Match endpoint responses and deferred joint reactions exactly."""
        with wp.ScopedDevice(self.device):
            direct = _fixture(self.device)
            shared = DirectContactResponse(direct, 8, 8)
            legacy = DirectContactResponse(direct, 8, 8, shared_body_basis=False)
            pairs = np.asarray([[1, 3], [2, 4], [0, 5], [2, 3], [1, 3], [2, 4], [0, 5], [1, 5]], dtype=np.int32)
            mechanisms = np.asarray([-1, 0, 0, 1, 1, 1], dtype=np.int32)
            contacts = contact_container_zeros(8, device=self.device)
            wp.launch(_initialize_contacts, 8, [contacts], device=self.device)
            for response in (shared, legacy):
                for prefix in ("column", "contact"):
                    getattr(response, prefix + "_body0").assign(pairs[:, 0])
                    getattr(response, prefix + "_body1").assign(pairs[:, 1])
                response.column_mechanism0.assign(mechanisms[pairs[:, 0]])
                response.column_mechanism1.assign(mechanisms[pairs[:, 1]])
                response.contact_column.assign(np.arange(8, dtype=np.int32))
                response.contact_mechanism.fill_(0)
                wp.launch(
                    _fill_direct_contact_rhs_tasks_kernel,
                    8,
                    [response.data, response.contact_batch.task_mechanism, response.contact_batch.task_item],
                    device=self.device,
                )
            direct.solver.compute()
            _compute_response(shared, contacts, self.device)
            _compute_response(legacy, contacts, self.device)
            for field in ("endpoint_response", "contact_response", "mobility"):
                np.testing.assert_array_equal(
                    getattr(shared.data, field).numpy().view(np.uint32),
                    getattr(legacy.data, field).numpy().view(np.uint32),
                    err_msg=field,
                )
            coefficients = []
            for response in (shared, legacy):
                output = wp.zeros((8, 9), device=self.device)
                wp.launch(_evaluate_deferred_coefficients, (8, 9), [response.data, output], device=self.device)
                coefficients.append(output.numpy())
            np.testing.assert_array_equal(coefficients[0].view(np.uint32), coefficients[1].view(np.uint32))

    def test_sparse_and_zero_contact_activation(self):
        """Skip inactive tasks and clear paired bases before solving."""
        with wp.ScopedDevice(self.device):
            direct = _fixture(self.device)
            shared = DirectContactResponse(direct, 8, 8)
            contacts = contact_container_zeros(8, device=self.device)
            direct.solver.compute()
            # Only body 3 is touched, though task 1 also owns body's 4 basis.
            shared.column_mechanism1.assign(np.asarray([1] + [-1] * 7, dtype=np.int32))
            shared.column_body1.assign(np.asarray([3] + [0] * 7, dtype=np.int32))
            wp.launch(
                _fill_direct_contact_rhs_tasks_kernel,
                8,
                [shared.data, shared.contact_batch.task_mechanism, shared.contact_batch.task_item],
                device=self.device,
            )
            np.testing.assert_array_equal(shared.contact_batch.task_mechanism.numpy(), [-1, 1, -1])
            np.testing.assert_array_equal(shared.data.body_basis_active.numpy(), [0, 0, 0, 1, 0, 0])
            shared.data.rhs.fill_(float("nan"))
            shared.data.solution.fill_(float("nan"))
            _compute_response(shared, contacts, self.device)
            rhs = shared.data.rhs.numpy().reshape(-1, int(shared.data.workspace_stride))
            solution = shared.data.solution.numpy().reshape(-1, int(shared.data.workspace_stride))
            # Zero the untouched neighboring body's live rows before solving.
            np.testing.assert_array_equal(rhs[6:8, :18], 0.0)
            np.testing.assert_array_equal(solution[6:8, :18], 0.0)
            self.assertTrue(np.isnan(solution[:4]).all())
            self.assertTrue(np.isfinite(solution[4:6, :18]).all())
            items = shared.contact_batch.task_item.numpy().copy()
            shared.reset_rhs_tasks()
            shared.column_mechanism1.fill_(-1)
            wp.launch(
                _fill_direct_contact_rhs_tasks_kernel,
                8,
                [shared.data, shared.contact_batch.task_mechanism, shared.contact_batch.task_item],
                device=self.device,
            )
            np.testing.assert_array_equal(shared.contact_batch.task_mechanism.numpy(), [-1, -1, -1])
            np.testing.assert_array_equal(shared.data.body_basis_active.numpy(), 0)
            np.testing.assert_array_equal(shared.contact_batch.task_item.numpy(), items)
            _compute_response(shared, contacts, self.device)
            np.testing.assert_array_equal(
                shared.data.solution.numpy().view(np.uint32), solution.reshape(-1).view(np.uint32)
            )


if __name__ == "__main__":
    unittest.main()
