# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Joint blocks sharing the ordinary rigid contact color sweeps."""

import numpy as np
import warp as wp

from ..body import BodyContainer, mat33_from_sym6
from ..constraints.bilateral_joint import _prepare_bilateral_joint_blocks_cooperative, prepare_bilateral_joint_blocks
from ..constraints.bilateral_joint_data import BilateralJointData, Mat66d, Vec6d
from .direct_equality import DirectEqualitySystem, _body_com_twist, _direct_wrench_response

_Vec6d = wp.types.vector(length=6, dtype=wp.float64)
_Mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


@wp.func
def _dot_double(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.float64:
    result = wp.float64(0.0)
    for component in range(6):
        result += wp.float64(a[component]) * wp.float64(b[component])
    return result


@wp.kernel(enable_backward=False)
def _block_sweep(
    bodies: BodyContainer,
    block_rows: wp.array2d[wp.int32],
    block_count: wp.array[wp.int32],
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    row_dynamic: wp.array[wp.bool],
    joint_to_structural: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    child: wp.array[wp.int32],
    wrench0: wp.array2d[wp.spatial_vector],
    wrench1: wp.array2d[wp.spatial_vector],
    bias: wp.array2d[wp.float32],
    reference: wp.array[wp.float32],
    dynamic_mass: wp.array[wp.float32],
    accumulated: wp.array[wp.float32],
    use_bias: wp.bool,
):
    for block in range(block_count.shape[0]):
        count = block_count[block]
        first_row = block_rows[block, 0]
        joint = row_joint[first_row]
        structural = joint_to_structural[joint]
        b0 = parent[joint] + wp.int32(1)
        b1 = child[joint] + wp.int32(1)
        inertia0 = mat33_from_sym6(bodies.inverse_inertia_world[b0])
        inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[b1])
        twist0 = _body_com_twist(bodies, b0)
        twist1 = _body_com_twist(bodies, b1)
        matrix = _Mat66d()
        rhs = _Vec6d()
        for i in range(count):
            row = block_rows[block, i]
            local = row_local[row]
            j0 = wrench0[structural, local]
            j1 = wrench1[structural, local]
            residual = _dot_double(j0, twist0) + _dot_double(j1, twist1)
            if row_dynamic[row]:
                residual += wp.float64(accumulated[row]) / wp.float64(dynamic_mass[row])
                residual -= wp.float64(reference[row])
            elif use_bias:
                residual += wp.float64(bias[structural, local])
            rhs[i] = -residual
            for j in range(i + 1):
                other_row = block_rows[block, j]
                other_local = row_local[other_row]
                k0 = wrench0[structural, other_local]
                k1 = wrench1[structural, other_local]
                response0 = _direct_wrench_response(k0, bodies.inverse_mass[b0], inertia0)
                response1 = _direct_wrench_response(k1, bodies.inverse_mass[b1], inertia1)
                value = _dot_double(j0, response0) + _dot_double(j1, response1)
                if i == j and row_dynamic[row]:
                    value += wp.float64(1.0) / wp.float64(dynamic_mass[row])
                matrix[i, j] = value
                matrix[j, i] = value

        # Physical joint block LDL^T, without mass scaling or added compliance.
        # Valid independent joint rows have strictly positive pivots.
        lower = _Mat66d()
        diagonal = _Vec6d()
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
        if valid:
            solution = _Vec6d()
            for i in range(count):
                value = rhs[i]
                for j in range(i):
                    value -= lower[i, j] * solution[j]
                solution[i] = value
            for i in range(count):
                solution[i] /= diagonal[i]
            for reverse in range(count):
                i = count - reverse - 1
                value = solution[i]
                for j in range(i + 1, count):
                    value -= lower[j, i] * solution[j]
                solution[i] = value

            impulse0 = wp.spatial_vector()
            impulse1 = wp.spatial_vector()
            for i in range(count):
                row = block_rows[block, i]
                local = row_local[row]
                impulse = wp.float32(solution[i])
                accumulated[row] += impulse
                impulse0 += impulse * wrench0[structural, local]
                impulse1 += impulse * wrench1[structural, local]
            response0 = _direct_wrench_response(impulse0, bodies.inverse_mass[b0], inertia0)
            response1 = _direct_wrench_response(impulse1, bodies.inverse_mass[b1], inertia1)
            bodies.velocity[b0] += wp.spatial_top(response0)
            bodies.angular_velocity[b0] += wp.spatial_bottom(response0)
            bodies.velocity[b1] += wp.spatial_top(response1)
            bodies.angular_velocity[b1] += wp.spatial_bottom(response1)


class BlockJointSystem(DirectEqualitySystem):
    """Reuse physical joint rows, with local blocks instead of a global solve.

    The owning world must bind this system before stepping. Ordinary color
    sweeps solve the blocks; only newly activated finite drive bounds require
    an additional serial physical-mass correction sweep.
    """

    @property
    def requires_global_projection(self):
        """Whether finite drive bounds need corrections on the original body state."""
        return self.has_bounded_drives

    @property
    def supports_async_factor(self) -> bool:
        """Local block preparation is one kernel and has no separate factor."""
        return False

    def bind_world(self, world, joint_idx_to_cid):
        self._block_world = world
        world._block_joint_contact_columns = wp.array(
            np.arange(world.max_contact_columns), dtype=wp.int32, device=self.model.device
        )
        self._block_mapping = np.asarray(joint_idx_to_cid, dtype=np.int32).copy()
        self._resolving_bounds = False
        count = world.num_joints
        indices = np.full((count, 6), -1, dtype=np.int32)
        counts = np.zeros(count, dtype=np.int32)
        structural = np.full(count, -1, dtype=np.int32)
        joint_structural = self.joint_to_structural.numpy()
        correction_groups = {}
        for row, joint in enumerate(self.topology.row_joint):
            cid = int(self._block_mapping[joint])
            if cid < 0:
                raise ValueError("Every block joint requires a coloring column")
            if counts[cid] == 6:
                raise ValueError("Joint blocks with more than six rows are unsupported")
            indices[cid, counts[cid]] = row
            counts[cid] += 1
            structural[cid] = joint_structural[joint]
            correction_groups.setdefault(int(joint), []).append(row)
        correction_counts = np.array([len(rows) for rows in correction_groups.values()], dtype=np.int32)
        correction_indices = np.full((len(correction_groups), 6), -1, dtype=np.int32)
        for index, rows in enumerate(correction_groups.values()):
            correction_indices[index, : len(rows)] = rows
        device = self.model.device
        self._correction_rows = wp.array(correction_indices, dtype=wp.int32, device=device)
        self._correction_counts = wp.array(correction_counts, dtype=wp.int32, device=device)
        data = BilateralJointData()
        data.enabled = 1
        data.row_count = wp.array(counts, dtype=wp.int32, device=device)
        data.row_indices = wp.array(indices, dtype=wp.int32, device=device)
        data.structural_index = wp.array(structural, dtype=wp.int32, device=device)
        data.row_local = self.row_local
        data.row_dynamic = self.row_dynamic
        data.wrench0 = self.row_wrench0
        data.wrench1 = self.row_wrench1
        data.bias = self.row_bias
        data.reference = self.velocity_reference
        data.dynamic_mass = self.dynamic_mass
        data.accumulated = self.accumulated_impulse
        data.response0 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
        data.response1 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
        data.lower = wp.zeros(count, dtype=Mat66d, device=device)
        data.diagonal = wp.zeros(count, dtype=Vec6d, device=device)
        data.valid = wp.zeros(count, dtype=wp.int32, device=device)
        world.constraints.bilateral = data
        ownership = world._joint_pgs_enabled.numpy().copy()
        ownership[counts > 0] = 1
        world.set_joint_pgs_ownership(ownership)

    def refresh_joint_properties(self):
        previous_rows = self.row_joint
        super().refresh_joint_properties()
        if self.row_joint is not previous_rows:
            self.bind_world(self._block_world, self._block_mapping)

    def prepare_and_factor(self, idt):
        if self.enabled:
            world = self._block_world
            if self.model.device.is_cuda:
                wp.launch(
                    _prepare_bilateral_joint_blocks_cooperative,
                    dim=8 * world.num_joints,
                    inputs=[world.constraints, world.bodies, world._copy_state],
                    device=self.model.device,
                    block_dim=32,
                )
            else:
                wp.launch(
                    prepare_bilateral_joint_blocks,
                    dim=world.num_joints,
                    inputs=[world.constraints, world.bodies, world._copy_state],
                    device=self.model.device,
                )

    def solve(self, *, use_bias):
        if not self.enabled or not self._resolving_bounds:
            return
        wp.launch(
            _block_sweep,
            dim=1,
            inputs=[
                self.bodies,
                self._correction_rows,
                self._correction_counts,
                self.row_joint,
                self.row_local,
                self.row_dynamic,
                self.joint_to_structural,
                self.model.joint_parent,
                self.model.joint_child,
                self.row_wrench0,
                self.row_wrench1,
                self.row_bias,
                self.velocity_reference,
                self.dynamic_mass,
                self.accumulated_impulse,
                wp.bool(use_bias),
            ],
            device=self.model.device,
        )

    def resolve_bounded_drives(self, idt, *, use_bias):
        self._resolving_bounds = True
        try:
            super().resolve_bounded_drives(idt, use_bias=use_bias)
        finally:
            self._resolving_bounds = False
