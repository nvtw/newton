# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolate ordinary rigid contacts plus bilateral joint-row PGS.

This diagnostic reuses the existing physical joint Jacobians and implicit
drive equations. It replaces the global equality factor/solve with scalar
Gauss-Seidel updates, and returns contacts to the ordinary rigid solver.
It is a convergence/performance experiment, not a supported solver mode.
"""

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.direct_equality import _body_com_twist, _direct_wrench_response
from newton._src.solvers.phoenx.articulations.maximal_contact_gs import rebase_owned_contact_levers_kernel
from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6


@wp.kernel(enable_backward=False)
def _sweep(
    bodies: BodyContainer,
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
    for row in range(row_joint.shape[0]):
        joint = row_joint[row]
        structural = joint_to_structural[joint]
        local = row_local[row]
        b0 = parent[joint] + wp.int32(1)
        b1 = child[joint] + wp.int32(1)
        j0 = wrench0[structural, local]
        j1 = wrench1[structural, local]
        r0 = _direct_wrench_response(j0, bodies.inverse_mass[b0], mat33_from_sym6(bodies.inverse_inertia_world[b0]))
        r1 = _direct_wrench_response(j1, bodies.inverse_mass[b1], mat33_from_sym6(bodies.inverse_inertia_world[b1]))
        mobility = wp.dot(j0, r0) + wp.dot(j1, r1)
        residual = wp.dot(j0, _body_com_twist(bodies, b0)) + wp.dot(j1, _body_com_twist(bodies, b1))
        if row_dynamic[row]:
            compliance = wp.float32(1.0) / wp.max(dynamic_mass[row], wp.float32(1.0e-10))
            mobility += compliance
            residual += accumulated[row] * compliance - reference[row]
        elif use_bias:
            residual += bias[structural, local]
        if mobility > wp.float32(0.0):
            impulse = -residual / mobility
            accumulated[row] += impulse
            bodies.velocity[b0] += impulse * wp.spatial_top(r0)
            bodies.angular_velocity[b0] += impulse * wp.spatial_bottom(r0)
            bodies.velocity[b1] += impulse * wp.spatial_top(r1)
            bodies.angular_velocity[b1] += impulse * wp.spatial_bottom(r1)


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


def install(solver, *, block=False, sweeps=1, allow_mass_splitting=False):
    """Select the diagnostic before simulation/capture; reject unsupported rows."""
    if solver.articulation_mode != "maximal":
        raise ValueError("The bilateral PGS diagnostic requires maximal coordinates")
    direct = solver._direct_equality_system
    if direct is None or not direct.enabled:
        raise ValueError("The diagnostic requires joint equality rows")
    if np.any(direct.model.joint_type.numpy() == 7):
        raise ValueError("Cable joint compliance is outside this diagnostic")
    world = solver.world
    world.bodies.constraint_node.assign(np.arange(world.bodies.constraint_node.shape[0], dtype=np.int32))
    if world.mass_splitting_enabled and not allow_mass_splitting:
        raise ValueError("The diagnostic uses unsplit physical bodies")

    block_rows = None
    block_count = None
    if block:
        groups = {}
        for row, joint in enumerate(direct.row_joint.numpy()):
            groups.setdefault(int(joint), []).append(row)
        counts = np.array([len(rows) for rows in groups.values()], dtype=np.int32)
        if np.any(counts > 6):
            raise ValueError("Joint blocks with more than six rows are unsupported")
        indices = np.full((len(groups), 6), -1, dtype=np.int32)
        for index, rows in enumerate(groups.values()):
            indices[index, : len(rows)] = rows
        block_rows = wp.array(indices, dtype=wp.int32, device=direct.model.device)
        block_count = wp.array(counts, dtype=wp.int32, device=direct.model.device)

    def sweep_once(*, use_bias):
        wp.launch(
            _block_sweep if block else _sweep,
            dim=1,
            inputs=[direct.bodies]
            + ([block_rows, block_count] if block else [])
            + [
                direct.row_joint,
                direct.row_local,
                direct.row_dynamic,
                direct.joint_to_structural,
                direct.model.joint_parent,
                direct.model.joint_child,
                direct.row_wrench0,
                direct.row_wrench1,
                direct.row_bias,
                direct.velocity_reference,
                direct.dynamic_mass,
                direct.accumulated_impulse,
                wp.bool(use_bias),
            ],
            device=direct.model.device,
        )

    def sweep(*, use_bias):
        for _ in range(sweeps):
            sweep_once(use_bias=use_bias)

    # Prepare rows/drive coefficients, but do not factor a global joint matrix.
    direct.prepare_and_factor = direct.prepare_matrix
    direct.solve = sweep
    for owner in (solver, world):
        owner._direct_contact_response = None
        owner._direct_contact_schedule = None
        owner._maximal_contact_response = None
        owner._maximal_contact_schedule = None
        owner._maximal_tree_projector = None
        owner._direct_tree_contacts = False

    # Ordinary contacts also need current COM levers for velocity relaxation.
    original_refresh = world._refresh_owned_relax_geometry
    canonical_columns = wp.array(np.arange(world.max_contact_columns), dtype=wp.int32, device=direct.model.device)

    def refresh_relax_geometry(idt):
        original_refresh(idt)
        if world.velocity_iterations > 0 and world._contact_input_active_this_step:
            wp.launch(
                rebase_owned_contact_levers_kernel,
                dim=world.max_contact_columns,
                inputs=[
                    world._contact_cols,
                    world._contact_container,
                    world.bodies,
                    canonical_columns,
                    world._ingest_scratch.num_contact_columns,
                ],
                device=direct.model.device,
            )

    world._refresh_owned_relax_geometry = refresh_relax_geometry


def install_fused(solver, *, mass_splitting=False):
    """Opt in to bilateral blocks in the existing mixed rigid color sweeps."""
    from newton._src.solvers.phoenx.constraints.bilateral_joint import prepare_bilateral_joint_blocks
    from newton._src.solvers.phoenx.constraints.bilateral_joint_data import BilateralJointData, Mat66d, Vec6d

    if solver.world.step_layout != "single_world":
        raise ValueError("Fused bilateral blocks currently require single_world")
    install(solver, block=True, allow_mass_splitting=mass_splitting)
    direct = solver._direct_equality_system
    world = solver.world
    count = world.num_joints
    mapping = solver._joint_constraints.joint_idx_to_cid.numpy()
    indices = np.full((count, 6), -1, dtype=np.int32)
    counts = np.zeros(count, dtype=np.int32)
    structural = np.full(count, -1, dtype=np.int32)
    joint_structural = direct.joint_to_structural.numpy()
    for row, joint in enumerate(direct.row_joint.numpy()):
        cid = int(mapping[joint])
        if cid < 0:
            raise ValueError("Every prepared joint requires a coloring column")
        if counts[cid] == 6:
            raise ValueError("Joint blocks with more than six rows are unsupported")
        indices[cid, counts[cid]] = row
        counts[cid] += 1
        structural[cid] = joint_structural[joint]

    data = BilateralJointData()
    data.enabled = 1
    device = direct.model.device
    data.row_count = wp.array(counts, dtype=wp.int32, device=device)
    data.row_indices = wp.array(indices, dtype=wp.int32, device=device)
    data.structural_index = wp.array(structural, dtype=wp.int32, device=device)
    data.row_local = direct.row_local
    data.row_dynamic = direct.row_dynamic
    data.wrench0 = direct.row_wrench0
    data.wrench1 = direct.row_wrench1
    data.bias = direct.row_bias
    data.reference = direct.velocity_reference
    data.dynamic_mass = direct.dynamic_mass
    data.accumulated = direct.accumulated_impulse
    data.response0 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
    data.response1 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
    data.lower = wp.zeros(count, dtype=Mat66d, device=device)
    data.diagonal = wp.zeros(count, dtype=Vec6d, device=device)
    data.valid = wp.zeros(count, dtype=wp.int32, device=device)
    world.constraints.bilateral = data

    ownership = world._joint_pgs_enabled.numpy().copy()
    ownership[counts > 0] = 1
    world.set_joint_pgs_ownership(ownership)

    def prepare_blocks(idt):
        wp.launch(
            prepare_bilateral_joint_blocks,
            dim=count,
            inputs=[world.constraints, world.bodies, world._copy_state],
            device=device,
        )

    # Ordinary iterations now invoke the block through each joint's color.
    # Keep a real correction sweep for finite-effort active-set transitions.
    correction_sweep = direct.solve
    active_set = [False]

    def solve(*, use_bias):
        if active_set[0]:
            correction_sweep(use_bias=use_bias)

    resolve = direct.resolve_bounded_drives

    def resolve_bounds(idt, *, use_bias):
        active_set[0] = True
        try:
            resolve(idt, use_bias=use_bias)
        finally:
            active_set[0] = False

    direct.prepare_and_factor = prepare_blocks
    direct.solve = solve
    direct.resolve_bounded_drives = resolve_bounds
