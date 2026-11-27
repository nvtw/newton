# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Joint blocks sharing the ordinary rigid contact color sweeps."""

import numpy as np
import warp as wp

from newton._src.sim import JointType

from ..body import BodyContainer, mat33_from_sym6
from ..constraints.bilateral_joint import (
    _prepare_bilateral_joint_blocks_cooperative,
    prepare_bilateral_joint_blocks,
)
from ..constraints.bilateral_joint_data import BilateralJointData, Mat66, Vec6
from ..constraints.constraint_container import ConstraintContainer
from ..sleeping_kernels import _phoenx_apply_island_wake_kernel
from .direct_equality import (
    DirectEqualitySystem,
    _apply_direct_equality_body,
    _body_com_twist,
    _direct_wrench_response,
)

_Vec6 = wp.types.vector(length=6, dtype=wp.float32)
_Mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.func
def _dot6(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.float32:
    result = wp.float32(0.0)
    for component in range(6):
        result += wp.float32(a[component]) * wp.float32(b[component])
    return result


@wp.kernel(enable_backward=False)
def _update_local_joint_enabled(
    joints: wp.array[wp.int32],
    cids: wp.array[wp.int32],
    enabled: wp.array[wp.bool],
    previous: wp.array[wp.bool],
    saved_counts: wp.array[wp.vec3i],
    saved_flags: wp.array2d[wp.int32],
    constraints: ConstraintContainer,
    ownership: wp.array[wp.int32],
    dynamic: wp.array[wp.bool],
    bounded: wp.array[wp.bool],
    parent: wp.array[wp.int32],
    child: wp.array[wp.int32],
    bodies: BodyContainer,
    wake: wp.array[wp.int32],
):
    index = wp.tid()
    joint = joints[index]
    active = enabled[joint]
    if active == previous[index]:
        return
    previous[index] = active
    cid = cids[joint]
    counts = saved_counts[index]
    bilateral = constraints.bilateral
    bilateral.row_count[cid] = wp.where(active, counts[0], 0)
    constraints.d6.row_count[cid] = wp.where(active, counts[1], 0)
    ownership[cid] = wp.where(active, counts[2], 0)
    bilateral.valid[cid] = 0
    constraints.d6.reaction_wrench[cid] = wp.spatial_vector()
    for local in range(6):
        if local < counts[0]:
            row = bilateral.row_indices[cid, local]
            flags = saved_flags[index, local]
            dynamic[row] = active and (flags & 1) != 0
            bounded[row] = active and (flags & 2) != 0
            bilateral.accumulated[row] = 0.0
        constraints.d6.lower_impulse[cid, local] = 0.0
        constraints.d6.upper_impulse[cid, local] = 0.0
        constraints.d6.friction_impulse[cid, local] = 0.0
    for side in range(2):
        body = wp.where(side == 0, parent[joint], child[joint]) + 1
        if body > 0:
            root = bodies.island_root[body]
            if root >= 0:
                wp.atomic_max(wake, root, 1)
            bodies.frames_below_threshold[body] = 0


@wp.kernel(enable_backward=False)
def _block_sweep(
    bodies: BodyContainer,
    component_start: wp.array[wp.int32],
    component_blocks: wp.array[wp.int32],
    block_rows: wp.array2d[wp.int32],
    block_count: wp.array[wp.int32],
    block_cid: wp.array[wp.int32],
    prepared_lower: wp.array[Mat66],
    prepared_diagonal: wp.array[Vec6],
    prepared_valid: wp.array[wp.int32],
    active_count: wp.array[wp.int32],
    copy_highest: wp.array[wp.int32],
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
    component = wp.tid()
    for entry in range(component_start[component], component_start[component + 1]):
        block = component_blocks[entry]
        if active_count[block_cid[block]] == 0:
            continue
        count = block_count[block]
        reuse_factor = copy_highest[0] == 0
        first_row = block_rows[block, 0]
        joint = row_joint[first_row]
        structural = joint_to_structural[joint]
        b0 = parent[joint] + wp.int32(1)
        b1 = child[joint] + wp.int32(1)
        inertia0 = mat33_from_sym6(bodies.inverse_inertia_world[b0])
        inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[b1])
        twist0 = _body_com_twist(bodies, b0)
        twist1 = _body_com_twist(bodies, b1)
        # Fixed bounds keep six-row temporaries statically indexed.
        matrix = _Mat66()
        rhs = _Vec6()
        for i in range(6):
            if i < count:
                row = block_rows[block, i]
                local = row_local[row]
                j0 = wrench0[structural, local]
                j1 = wrench1[structural, local]
                residual = _dot6(j0, twist0) + _dot6(j1, twist1)
                if row_dynamic[row]:
                    residual += wp.float32(accumulated[row]) / wp.float32(dynamic_mass[row])
                    residual -= wp.float32(reference[row])
                elif use_bias:
                    residual += wp.float32(bias[structural, local])
                rhs[i] = -residual
                if not reuse_factor:
                    for j in range(6):
                        if j < i + 1:
                            other_row = block_rows[block, j]
                            other_local = row_local[other_row]
                            k0 = wrench0[structural, other_local]
                            k1 = wrench1[structural, other_local]
                            response0 = _direct_wrench_response(k0, bodies.inverse_mass[b0], inertia0)
                            response1 = _direct_wrench_response(k1, bodies.inverse_mass[b1], inertia1)
                            value = _dot6(j0, response0) + _dot6(j1, response1)
                            if i == j and row_dynamic[row]:
                                value += wp.float32(1.0) / wp.float32(dynamic_mass[row])
                            matrix[i, j] = value
                            matrix[j, i] = value
        # Physical joint block LDL^T, without mass scaling or added compliance.
        # Valid independent joint rows have strictly positive pivots.
        lower = _Mat66()
        diagonal = _Vec6()
        valid = wp.bool(True)
        if reuse_factor:
            # With no mass-splitting copies, the ordinary block preparation
            # already factored this exact physical-mass matrix.
            cid = block_cid[block]
            lower = prepared_lower[cid]
            diagonal = prepared_diagonal[cid]
            valid = prepared_valid[cid] != 0
        else:
            for i in range(6):
                if i < count:
                    pivot = matrix[i, i]
                    for j in range(6):
                        if j < i:
                            pivot -= lower[i, j] * lower[i, j] * diagonal[j]
                    diagonal[i] = pivot
                    lower[i, i] = wp.float32(1.0)
                    if pivot <= wp.float32(0.0):
                        valid = False
                    if valid:
                        for j in range(6):
                            if j >= i + 1 and j < count:
                                value = matrix[j, i]
                                for k in range(6):
                                    if k < i:
                                        value -= lower[j, k] * lower[i, k] * diagonal[k]
                                lower[j, i] = value / pivot
        if valid:
            solution = _Vec6()
            for i in range(6):
                if i < count:
                    value = rhs[i]
                    for j in range(6):
                        if j < i:
                            value -= lower[i, j] * solution[j]
                    solution[i] = value
            for i in range(6):
                if i < count:
                    solution[i] /= diagonal[i]
            for reverse in range(6):
                i = 5 - reverse
                if i < count:
                    value = solution[i]
                    for j in range(6):
                        if j >= i + 1 and j < count:
                            value -= lower[j, i] * solution[j]
                    solution[i] = value

            impulse0 = wp.spatial_vector()
            impulse1 = wp.spatial_vector()
            for i in range(6):
                if i < count:
                    row = block_rows[block, i]
                    local = row_local[row]
                    impulse = wp.float32(solution[i])
                    accumulated[row] += impulse
                    impulse0 += impulse * wrench0[structural, local]
                    impulse1 += impulse * wrench1[structural, local]
            response0 = _direct_wrench_response(impulse0, bodies.inverse_mass[b0], inertia0)
            response1 = _direct_wrench_response(impulse1, bodies.inverse_mass[b1], inertia1)
            # Static anchors can be shared by independent components.
            if bodies.inverse_mass[b0] > 0.0:
                bodies.velocity[b0] += wp.spatial_top(response0)
                bodies.angular_velocity[b0] += wp.spatial_bottom(response0)
            if bodies.inverse_mass[b1] > 0.0:
                bodies.velocity[b1] += wp.spatial_top(response1)
                bodies.angular_velocity[b1] += wp.spatial_bottom(response1)


@wp.kernel(enable_backward=False)
def _finish_local_microstep(
    row_joint: wp.array[wp.int32],
    selected: wp.array[wp.bool],
    impulse: wp.array[wp.float32],
    total: wp.array[wp.float32],
    seed: wp.array[wp.float32],
):
    row = wp.tid()
    if selected[row_joint[row]]:
        seed[row] = impulse[row]
        impulse[row] += total[row]
    else:
        seed[row] = 0.0


@wp.kernel(enable_backward=False)
def _warm_start_local_microstep(
    body_row_start: wp.array[wp.int32],
    descriptor: wp.array[wp.vec2i],
    wrench0: wp.array[wp.spatial_vector],
    wrench1: wp.array[wp.spatial_vector],
    seed: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bodies: BodyContainer,
):
    body = wp.tid()
    if bodies.island_root[body] < 0:
        _apply_direct_equality_body(body, body_row_start, descriptor, wrench0, wrench1, seed, scale, bodies)


class BlockJointSystem(DirectEqualitySystem):
    """Reuse physical joint rows, with local blocks instead of a global solve.

    The owning world must bind this system before stepping. Ordinary color
    sweeps solve the blocks; only newly activated finite drive bounds require
    an additional physical-mass correction sweep, parallel across independent
    mechanisms and ordered within each mechanism.
    """

    @property
    def fuses_copy_average(self) -> bool:
        """Block sweeps never average mass-splitting copies; the dispatcher keeps its own average."""
        return False

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
            if self.fixed_joint_rows:
                # Every column owns its whole fixed block, so joints created at runtime only
                # change the used row count.
                if cid < 0:
                    continue
                local = int(self.topology.row_local[row])
                indices[cid, local] = row
                structural[cid] = joint_structural[joint]
                if local >= self.fixed_row_count[joint]:
                    continue
                counts[cid] += 1
                correction_groups.setdefault(int(joint), []).append(row)
                continue
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
        # Preserve Gauss-Seidel block order within each connected mechanism.
        # Contacts are handled in a separate sweep; only shared movable bodies
        # couple these corrections. World and kinematic anchors remain read-only.
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        movable = self.bodies.inverse_mass.numpy() > 0.0
        roots = list(range(len(correction_groups)))

        def root(index):
            while roots[index] != index:
                roots[index] = roots[roots[index]]
                index = roots[index]
            return index

        owner = {}
        for block, joint in enumerate(correction_groups):
            for body in (int(parents[joint]) + 1, int(children[joint]) + 1):
                if movable[body]:
                    if body in owner:
                        roots[root(block)] = root(owner[body])
                    else:
                        owner[body] = block
        components = {}
        for block in range(len(roots)):
            components.setdefault(root(block), []).append(block)
        joints = list(correction_groups)
        components = {
            key: blocks
            for key, blocks in components.items()
            if any(self.bounded_drive_joint_mask[joints[block]] for block in blocks)
        }
        # Bodies a bound correction sweep may write; see MixedJointSystem.bind_world.
        self._correction_bodies = frozenset(
            body
            for group in components.values()
            for block in group
            for body in (int(parents[joints[block]]) + 1, int(children[joints[block]]) + 1)
            if body > 0
        )
        starts = np.cumsum([0, *(len(blocks) for blocks in components.values())], dtype=np.int32)
        blocks = np.asarray([block for group in components.values() for block in group], dtype=np.int32)
        device = self.model.device
        self._correction_component_start = wp.array(starts, dtype=wp.int32, device=device)
        self._correction_component_blocks = wp.array(blocks, dtype=wp.int32, device=device)
        self._correction_cids = wp.array(
            [self._block_mapping[joint] for joint in correction_groups], dtype=wp.int32, device=device
        )
        self._correction_rows = wp.array(correction_indices, dtype=wp.int32, device=device)
        self._correction_counts = wp.array(correction_counts, dtype=wp.int32, device=device)
        data = BilateralJointData()
        data.enabled = 1
        data.row_capacity = 3 if counts.max(initial=0) <= 3 and not self.fixed_joint_rows else 6
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
        data.lower = wp.zeros(count, dtype=Mat66, device=device)
        data.diagonal = wp.zeros(count, dtype=Vec6, device=device)
        data.valid = wp.zeros(count, dtype=wp.int32, device=device)
        world.constraints.bilateral = data
        if np.count_nonzero(counts) >= world._fuse_threshold:
            # A six-row block carries substantially more work than one scalar
            # contact column. Let the cooperative grid process medium colors
            # instead of serializing hundreds of blocks in the one-block tail.
            width = 4 if data.row_capacity == 3 else 8
            world._fuse_threshold = min(world._fuse_threshold, world._fuse_tail_block_dim // width)
        ownership = world._joint_pgs_enabled.numpy().copy()
        ownership[counts > 0] = 1
        world.set_joint_pgs_ownership(ownership)
        # Reserved slots manage their own lifetime; only authored local columns
        # have immutable descriptors that can be restored after a live toggle.
        authored = int(self.model.joint_count)
        capacity = self.model._dynamic_capacity
        if capacity is not None:
            authored = capacity.joint_start
        joints = [j for j in range(authored) if not self._excluded_joint_mask[j] and self._block_mapping[j] >= 0]
        self._enabled_joints = wp.array(joints, dtype=wp.int32, device=device)
        self._enabled_previous = wp.ones(len(joints), dtype=wp.bool, device=device)
        cids = self._block_mapping[joints]
        self._enabled_counts = wp.array(
            np.stack((counts[cids], world.constraints.d6.row_count.numpy()[cids], ownership[cids]), axis=1),
            dtype=wp.vec3i,
            device=device,
        )
        flags = np.zeros((len(joints), 6), dtype=np.int32)
        dynamic = self.row_dynamic.numpy()
        bounded = self.row_bounded_drive.numpy()
        for index, cid in enumerate(cids):
            for local in range(counts[cid]):
                row = indices[cid, local]
                flags[index, local] = int(dynamic[row]) | (int(bounded[row]) << 1)
        self._enabled_flags = wp.array(flags, dtype=wp.int32, device=device)
        self._enabled_wake = wp.zeros(len(self.bodies.island_root), dtype=wp.int32, device=device)

    def configure_temporal_iterations(self):
        """Microstep authored scalar local drives, retaining outer schedules for coupled exact mechanisms."""
        world, model = self._block_world, self.model
        world._temporal_local_joints = None
        self._microstep_count = 1
        self._microstep_joint.zero_()
        self._microstep_sum.zero_()
        self._microstep_seed.zero_()
        self._microstep_previous.zero_()
        self._microstep_body_row_start = None
        self._microstep_descriptors = None
        if world.solver_iterations <= 1:
            return
        local = np.zeros(model.joint_count, dtype=bool)
        local[self._enabled_joints.numpy()] = True
        types = model.joint_type.numpy()
        # Fixed companions and unsupported rows retain the outer schedule for
        # their whole connected component; unrelated fixed joints are untouched.
        local &= np.isin(types, (JointType.REVOLUTE, JointType.PRISMATIC))
        local &= model.joint_enabled.numpy()
        friction = model.joint_friction.numpy()
        starts, dims = model.joint_qd_start.numpy(), model.joint_dof_dim.numpy()
        for joint in np.flatnonzero(local):
            if np.any(friction[starts[joint] : starts[joint] + sum(dims[joint])] > 0):
                local[joint] = False
        if self.has_direct_friction or not np.any(local[self.row_joint.numpy()] & self.row_dynamic.numpy()):
            return
        self._microstep_count = world.solver_iterations
        self._microstep_joint = wp.zeros(model.joint_count, dtype=wp.bool, device=model.device)
        self._microstep_local = wp.array(local, dtype=wp.bool, device=model.device)
        self._microstep_outer = wp.array(~local & (types != JointType.FREE), dtype=wp.bool, device=model.device)
        self._microstep_sum = wp.zeros_like(self.accumulated_impulse)
        self._microstep_seed = wp.zeros_like(self.accumulated_impulse)
        self._microstep_previous = wp.zeros(self.accumulated_impulse.size, dtype=wp.bool, device=model.device)
        # Reserved rows share placeholder endpoints and can otherwise create a
        # huge serial gather on one body. Keep authored incidence order intact.
        descriptors = self._body_row_wrench_descriptors.numpy()
        keep = local[self.row_joint.numpy()[descriptors[:, 0]]]
        prefix = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(keep, dtype=np.int32)))
        self._microstep_body_row_start = wp.array(
            prefix[self.body_row_start.numpy()], dtype=wp.int32, device=model.device
        )
        self._microstep_descriptors = wp.array(descriptors[keep], dtype=wp.vec2i, device=model.device)
        world._temporal_local_joints = self

    def begin_substep(self, idt, *, reuse_geometry=False):
        if self._microstep_count > 1:
            self._block_world.update_temporal_jointed_bodies(self.model, local=True)
            self._microstep_sum.zero_()
        super().begin_substep(idt, reuse_geometry=reuse_geometry)

    def advance_temporal_iteration(self):
        """Advance drive coordinates and sum physical microstep reactions."""
        self._snapshot_dynamic_rows(wp.float32(1.0 / self._block_world.substep_dt), advance=True)

    def warm_start_temporal_iteration(self):
        """Balance contact and joint warm starts before broadcasting physical velocities to copies."""
        wp.launch(
            _warm_start_local_microstep,
            dim=self.model.body_count + 1,
            inputs=[
                self._microstep_body_row_start,
                self._microstep_descriptors,
                self._apply_wrench0,
                self._apply_wrench1,
                self._microstep_seed,
                self.row_scale,
                self.bodies,
            ],
            device=self.model.device,
        )

    def warm_start_temporal_iteration_inputs(self):
        """Body count and arrays of :meth:`warm_start_temporal_iteration`, for a fused per-body launch."""
        return (
            int(self.model.body_count) + 1,
            self._microstep_body_row_start,
            self._microstep_descriptors,
            self._apply_wrench0,
            self._apply_wrench1,
            self._microstep_seed,
            self.row_scale,
        )

    def finish_temporal_iterations(self):
        """Export summed physical reactions and retain only the last microstep as a warm start."""
        wp.launch(
            _finish_local_microstep,
            dim=self.accumulated_impulse.size,
            inputs=[
                self.row_joint,
                self._microstep_joint,
                self.accumulated_impulse,
                self._microstep_sum,
                self._microstep_seed,
            ],
            device=self.model.device,
        )

    def update_enabled(self):
        """Apply live toggles of existing authored local columns without recapture."""
        if not self.enabled or not self._enabled_joints.size or self.model.joint_enabled is None:
            return
        world = self._block_world
        if world._sleeping_enabled:
            self._enabled_wake.zero_()
        wp.launch(
            _update_local_joint_enabled,
            dim=self._enabled_joints.size,
            inputs=[
                self._enabled_joints,
                self.constraint_cids,
                self.model.joint_enabled,
                self._enabled_previous,
                self._enabled_counts,
                self._enabled_flags,
                world.constraints,
                world._joint_pgs_enabled,
                self.row_dynamic,
                self.row_bounded_drive,
                self.model.joint_parent,
                self.model.joint_child,
                self.bodies,
                self._enabled_wake,
            ],
            device=self.model.device,
        )
        if world._sleeping_enabled:
            wp.launch(
                _phoenx_apply_island_wake_kernel,
                dim=self._enabled_wake.size,
                inputs=[self.bodies, self._enabled_wake],
                device=self.model.device,
            )

    def refresh_joint_properties(self):
        temporal = self._block_world.temporal_iterations and not self._block_world._temporal_joint_springs
        previous_rows = self.row_joint
        super().refresh_joint_properties()
        if self.row_joint is not previous_rows:
            self.bind_world(self._block_world, self._block_mapping)
        if temporal:
            self.configure_temporal_iterations()

    def prefactor_async(self, idt) -> bool:
        """Local blocks are prepared from the colouring, which is not ready before the substeps."""
        return False

    def prepare_and_factor(self, idt):
        if self.enabled:
            world = self._block_world
            prepare_inputs = [
                world.constraints,
                world.bodies,
                world._copy_state,
                world._partitioner.interaction_id_to_partition,
                wp.int32(world.max_colored_partitions or 0),
                wp.int32(int(world._singleworld_overflow_only_mass_splitting)),
            ]
            if self.model.device.is_cuda:
                wp.launch(
                    _prepare_bilateral_joint_blocks_cooperative,
                    dim=8 * world.num_joints,
                    inputs=prepare_inputs,
                    device=self.model.device,
                    block_dim=32,
                )
            else:
                wp.launch(
                    prepare_bilateral_joint_blocks,
                    dim=world.num_joints,
                    inputs=prepare_inputs,
                    device=self.model.device,
                )

    def solve(self, *, use_bias, refine=True, copy_state=None, average_copies=False):
        # Match the direct equality solve interface. Block systems always
        # perform their complete local correction, so refine has no effect,
        # and they never own the mass-splitting copy average.
        del refine, copy_state
        if average_copies:
            raise ValueError("block joint systems cannot average mass-splitting copies")
        if not self.enabled or not self._resolving_bounds:
            return
        wp.launch(
            _block_sweep,
            dim=self._correction_component_start.size - 1,
            inputs=[
                self.bodies,
                self._correction_component_start,
                self._correction_component_blocks,
                self._correction_rows,
                self._correction_counts,
                self._correction_cids,
                self._block_world.constraints.bilateral.lower,
                self._block_world.constraints.bilateral.diagonal,
                self._block_world.constraints.bilateral.valid,
                self._block_world.constraints.bilateral.row_count,
                self._block_world._copy_state.highest_index_in_use,
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
            block_dim=32 if self.model.device.is_cuda else 256,
        )

    def _correct_new_bounds(self, idt, *, use_bias, after_change=None):
        self._resolving_bounds = True
        try:
            super()._correct_new_bounds(idt, use_bias=use_bias, after_change=after_change)
        finally:
            self._resolving_bounds = False


class MixedJointSystem:
    """Exact mechanism solves for selected joints, local color blocks for the rest.

    Long chains, trees with large link mass ratios and closed linkages converge
    poorly as independent joint blocks. Their joints are owned by a
    :class:`DirectEqualitySystem`, whose solve runs between the color sweeps at
    the same points as in ``maximal_direct``; the remaining joints stay in
    :class:`BlockJointSystem`. Both share the world's bodies, so the mechanisms
    couple to contacts and block joints through the sweep order.
    """

    def __init__(self, model, bodies, *, excluded_joint_mask=None, exact_joint_mask, **kwargs):
        joint_count = int(model.joint_count)
        excluded = np.zeros(joint_count, dtype=bool) if excluded_joint_mask is None else np.asarray(excluded_joint_mask)
        exact = np.asarray(exact_joint_mask, dtype=bool) & ~excluded
        self.blocks = BlockJointSystem(model, bodies, excluded_joint_mask=excluded | exact, **kwargs)
        # Runtime joints keep fixed row addresses in the block system. Exact
        # mechanisms retain compact topology and never own reserved slots.
        exact_options = dict(kwargs)
        exact_options.pop("fixed_joint_rows", None)
        self.exact = DirectEqualitySystem(model, bodies, excluded_joint_mask=~exact, **exact_options)
        self._parts = tuple(part for part in (self.blocks, self.exact) if part.enabled)
        self.d6_inequality_count = None
        self._joint_bound_scan = False

    @property
    def enabled(self) -> bool:
        return bool(self._parts)

    @property
    def joint_mask(self) -> np.ndarray:
        return self.blocks.joint_mask | self.exact.joint_mask

    @property
    def direct_friction_dof_mask(self) -> np.ndarray:
        return self.blocks.direct_friction_dof_mask | self.exact.direct_friction_dof_mask

    @property
    def direct_limit_dof_mask(self) -> np.ndarray:
        return self.blocks.direct_limit_dof_mask | self.exact.direct_limit_dof_mask

    @property
    def has_bounded_drives(self) -> bool:
        return any(part.has_bounded_drives for part in self._parts)

    @property
    def requires_global_projection(self) -> bool:
        return self.exact.enabled or self.has_bounded_drives

    @property
    def supports_async_factor(self) -> bool:
        return False

    def bind_world(self, world, joint_idx_to_cid):
        if self.blocks.enabled:
            self.blocks.bind_world(world, joint_idx_to_cid)
        self._joint_bound_scan = False
        if self.blocks.enabled and self.blocks.has_bounded_drives and self.exact.has_bounded_drives:
            # The exact bound scan reads only its bounded rows' endpoint
            # velocities. When the block correction sweep cannot write those
            # bodies, both scans commute with the block correction, so one
            # combined flag and conditional guard both corrections.
            parents = self.exact.model.joint_parent.numpy()
            children = self.exact.model.joint_child.numpy()
            bounded = self.exact.row_bounded_drive.numpy()
            exact_bodies = {
                body
                for row, joint in enumerate(self.exact.topology.row_joint)
                if bounded[row]
                for body in (int(parents[joint]) + 1, int(children[joint]) + 1)
                if body > 0
            }
            if exact_bodies.isdisjoint(self.blocks._correction_bodies):
                self._joint_bound_scan = True
                self._any_bound_flag = wp.zeros(1, dtype=wp.int32, device=self.exact.model.device)

    def _forward(self, name, *args, **kwargs):
        for part in self._parts:
            getattr(part, name)(*args, **kwargs)

    def bind_constraint_indices(self, joint_idx_to_cid):
        self.blocks.bind_constraint_indices(joint_idx_to_cid)
        self.exact.bind_constraint_indices(joint_idx_to_cid)

    def set_temporal_substeps(self, substeps):
        self.blocks.set_temporal_substeps(substeps)
        self.exact.set_temporal_substeps(substeps)

    def set_control_targets(self, target_q, target_qd):
        self.blocks.set_control_targets(target_q, target_qd)
        self.exact.set_control_targets(target_q, target_qd)

    def refresh_joint_properties(self):
        self._forward("refresh_joint_properties")

    def refresh_cable_rest_state(self):
        self._forward("refresh_cable_rest_state")

    def gather_constraint_wrenches(self, out, idt):
        self._forward("gather_constraint_wrenches", out, idt)

    def refresh_geometry(self, idt):
        self._forward("refresh_geometry", idt)

    def begin_substep(self, idt, *, reuse_geometry=False):
        if reuse_geometry:
            # The reuse gate permits only an all-exact system. Block rows
            # have a separate per-color preparation lifecycle.
            if self.blocks.enabled:
                raise ValueError("Direct factor reuse requires no block-owned joints")
            self.exact.begin_substep(idt, reuse_geometry=True)
        else:
            self._forward("begin_substep", idt)

    def prepare_and_factor(self, idt):
        self._forward("prepare_and_factor", idt)

    def prefactor_async(self, idt) -> bool:
        """Start the exact first-substep factor on its private stream; see :meth:`wait_factor`."""
        return self.exact.prefactor_async(idt)

    def wait_factor(self):
        self.exact.wait_factor()

    def solve(self, *, use_bias, refine=True, copy_state=None, average_copies=False):
        if average_copies:
            if not self.fuses_copy_average:
                raise ValueError("average_copies requires the exact system to cover every dynamic body")
            self.exact.solve(use_bias=use_bias, refine=refine, copy_state=copy_state, average_copies=True)
            if self.blocks.enabled:
                self.blocks.solve(use_bias=use_bias, refine=refine)
            return
        self._forward("solve", use_bias=use_bias, refine=refine)

    @property
    def fuses_copy_average(self) -> bool:
        """Whether the exact projection can replace the standalone copy average."""
        return bool(self.exact.enabled and self.exact.fuses_copy_average)

    def resolve_bounded_drives(self, idt, *, use_bias, after_change=None):
        if not self._joint_bound_scan:
            self._forward("resolve_bounded_drives", idt, use_bias=use_bias, after_change=after_change)
            return
        # Like the scan flags of both parts, the shared flag is zero between scans.
        self.blocks._find_new_bounds(idt, self._any_bound_flag)
        self.exact._find_new_bounds(idt, self._any_bound_flag)

        def correct():
            self._any_bound_flag.zero_()
            self.blocks._correct_new_bounds(idt, use_bias=use_bias)
            self.exact._correct_new_bounds(idt, use_bias=use_bias)
            if after_change is not None:
                after_change()

        wp.capture_if(self._any_bound_flag, on_true=correct, on_false=None)
