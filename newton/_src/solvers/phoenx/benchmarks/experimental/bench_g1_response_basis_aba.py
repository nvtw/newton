# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure exact unit-wrench ABA bases on the live reduced G1 contact state."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _build_packed_generalized_contact_rows_kernel,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_response_basis_synthesis import (
    _BASIS,
    _COLUMN_TILE,
    _DOFS,
    _PACKED_COLUMNS,
    _ROWS,
    _synthesize_tile_kernel,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.rl_training import g1_recipe

_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_BASIS_GROUP = 16


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_classify")
def _classify_basis_kernel(
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    basis_body: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
):
    articulation = wp.tid()
    packed_articulation = articulation * wp.int32(2)
    body0 = row_body[packed_articulation, 0]
    body1 = wp.int32(-1)
    for row_offset in range(_ROWS):
        body = row_body[packed_articulation, wp.int32(row_offset)]
        if body != body0:
            body1 = body
    basis_body[articulation, 0] = body0
    basis_body[articulation, 1] = body1
    coefficient_start = articulation * wp.int32(_ROWS)
    for row_offset in range(_ROWS):
        row = wp.int32(row_offset)
        for basis_offset in range(_BASIS):
            coefficients[coefficient_start + row, wp.int32(basis_offset)] = wp.float32(0.0)
        body = row_body[packed_articulation, row]
        slot = wp.int32(0) if body == body0 else wp.int32(1)
        wrench = row_wrench[packed_articulation, row]
        for component_offset in range(6):
            component = wp.int32(component_offset)
            coefficients[coefficient_start + row, slot * wp.int32(6) + component] = wrench[component]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_classify_parallel")
def _classify_basis_bodies_kernel(
    row_body: wp.array2d[wp.int32],
    basis_body: wp.array2d[wp.int32],
):
    articulation = wp.tid()
    packed_articulation = articulation * wp.int32(2)
    body0 = row_body[packed_articulation, 0]
    body1 = wp.int32(-1)
    for row_offset in range(_ROWS):
        body = row_body[packed_articulation, wp.int32(row_offset)]
        if body != body0:
            body1 = body
    basis_body[articulation, 0] = body0
    basis_body[articulation, 1] = body1


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_coefficients")
def _build_basis_coefficients_kernel(
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    basis_body: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
):
    articulation, row = wp.tid()
    packed_articulation = articulation * wp.int32(2)
    coefficient_row = articulation * wp.int32(_ROWS) + row
    for basis_offset in range(_BASIS):
        coefficients[coefficient_row, wp.int32(basis_offset)] = wp.float32(0.0)
    body = row_body[packed_articulation, row]
    slot = wp.int32(0) if body == basis_body[articulation, 0] else wp.int32(1)
    wrench = row_wrench[packed_articulation, row]
    for component_offset in range(6):
        component = wp.int32(component_offset)
        coefficients[coefficient_row, slot * wp.int32(6) + component] = wrench[component]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_synthesis_split")
def _synthesize_split_tile_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float32],
    response: wp.array2d[wp.float32],
):
    articulation, output_kind = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(_ROWS, _BASIS),
        offset=(articulation * _ROWS, 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, output_kind * _DOFS),
        storage="shared",
    )
    result = wp.tile_matmul(coefficients_tile, basis_tile)
    packed_row = articulation * wp.int32(2 * 96)
    if output_kind == wp.int32(0):
        wp.tile_store(jacobian, result, offset=(packed_row, 0))
    else:
        wp.tile_store(response, result, offset=(packed_row, 0))


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_synthesis_split_96")
def _synthesize_split_tile_96_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float32],
    response: wp.array2d[wp.float32],
):
    articulation, output_kind = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(96, _BASIS),
        offset=(articulation * wp.int32(96), 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, output_kind * _DOFS),
        storage="shared",
    )
    result = wp.tile_matmul(coefficients_tile, basis_tile)
    packed_row = articulation * wp.int32(2 * 96)
    if output_kind == wp.int32(0):
        wp.tile_store(jacobian, result, offset=(packed_row, 0))
    else:
        wp.tile_store(response, result, offset=(packed_row, 0))


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_scatter")
def _scatter_basis_kernel(
    synthesized: wp.array2d[wp.float32],
    jacobian: wp.array3d[wp.float32],
    response: wp.array3d[wp.float32],
):
    articulation, row, dof = wp.tid()
    synthesized_row = articulation * wp.int32(_ROWS) + row
    jacobian[articulation, row, dof] = synthesized[synthesized_row, dof]
    response[articulation, row, dof] = synthesized[synthesized_row, wp.int32(_DOFS) + dof]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis")
def _build_unit_wrench_basis_kernel(
    bodies: BodyContainer,
    basis_body: wp.array2d[wp.int32],
    basis_packed: wp.array2d[wp.float32],
    joint_work: wp.array3d[wp.float32],
    body_response: wp.array3d[wp.spatial_vector],
):
    index = wp.tid()
    articulation = index // wp.int32(_BASIS_GROUP)
    basis_row = index - articulation * wp.int32(_BASIS_GROUP)
    if basis_row >= wp.int32(_BASIS):
        return
    body_slot = basis_row // wp.int32(6)
    component = basis_row - body_slot * wp.int32(6)
    source_body = basis_body[articulation, body_slot]
    if source_body < wp.int32(0):
        return
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    dof_start_articulation = data.joint_qd_start[start]
    dof_end_articulation = data.joint_qd_start[end]
    dof_count_articulation = dof_end_articulation - dof_start_articulation
    packed_basis_row = articulation * wp.int32(_BASIS) + basis_row
    for local_dof in range(dof_count_articulation):
        basis_packed[packed_basis_row, local_dof] = wp.float32(0.0)
        basis_packed[packed_basis_row, wp.int32(_DOFS) + local_dof] = wp.float32(0.0)
        joint_work[articulation, local_dof, basis_row] = wp.float32(0.0)

    source_wrench = wp.spatial_vector()
    source_wrench[component] = wp.float32(1.0)
    path_start = data.body_path_start[source_body]
    path_end = data.body_path_start[source_body + wp.int32(1)]
    propagated_wrench = source_wrench
    for reverse in range(path_end - path_start):
        path_index = path_end - wp.int32(1) - reverse
        joint = data.body_path_joint[path_index]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        projected = _vec6(0.0)
        reduced = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                projected[dof_row] = wp.dot(data.joint_s[dof], propagated_wrench)
                joint_work[articulation, dof - dof_start_articulation, basis_row] = projected[dof_row]
                basis_packed[packed_basis_row, dof - dof_start_articulation] = wp.dot(data.joint_s[dof], source_wrench)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        reduced[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * projected[dof_column]
                propagated_wrench -= data.joint_u[dof_start + wp.int32(dof_row)] * reduced[dof_row]

    for joint in range(start, end):
        local_joint = joint - start
        parent = data.joint_parent[joint]
        parent_delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_delta = body_response[articulation, data.body_joint[parent] - start, basis_row]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        generalized_delta = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                rhs[dof_row] = joint_work[articulation, dof - dof_start_articulation, basis_row] - wp.dot(
                    data.joint_u[dof], parent_delta
                )
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        generalized_delta[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * rhs[dof_column]
                dof = dof_start + wp.int32(dof_row)
                basis_packed[packed_basis_row, wp.int32(_DOFS) + dof - dof_start_articulation] = generalized_delta[
                    dof_row
                ]
                parent_delta += data.joint_s[dof] * generalized_delta[dof_row]
        body_response[articulation, local_joint, basis_row] = parent_delta


def _time_graph(device: wp.context.Device, launch: Callable[[], None], replays: int) -> float:
    with wp.ScopedCapture(device=device) as capture:
        launch()
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    return 1.0e6 * (time.perf_counter() - start) / float(replays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--basis-block-dim", type=int, default=96)
    parser.add_argument("--basis-launch-block-dim", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("live G1 response-basis benchmark requires CUDA")

    env = rl.EnvG1PhoenX(
        rl.ConfigEnvG1PhoenX(
            world_count=int(args.world_count),
            articulation_mode="reduced",
            sim_substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            contact_geometry=g1_recipe.CONTACT_GEOMETRY,
        ),
        device=device,
    )
    actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
    env.step(actions)
    wp.synchronize_device(device)
    reduced = env.solver._reduced_articulation
    if reduced is None:
        raise RuntimeError("reduced articulation was not initialized")
    block = reduced.contact_block_system
    if block.packed_jacobian is None or block.packed_response is None:
        raise RuntimeError("reduced contact buffers were not initialized")
    block.page_index.assign(np.asarray([0], dtype=np.int32))

    point_count = block.point_count.numpy()[::2]
    row_body = block.row_body.numpy()[::2]
    row_wrench = block.row_wrench.numpy()[::2]
    if np.any(point_count != 8):
        raise RuntimeError(f"expected the common eight-point G1 page, got {np.unique(point_count).tolist()}")
    basis_body_np = np.full((env.world_count, 2), -1, dtype=np.int32)
    coefficients_np = np.zeros((env.world_count, _ROWS, _BASIS), dtype=np.float32)
    for articulation in range(env.world_count):
        bodies = np.unique(row_body[articulation, :_ROWS])
        if bodies.size != 2:
            raise RuntimeError(f"articulation {articulation} has {bodies.size} contacted bodies")
        basis_body_np[articulation] = bodies
        for row in range(_ROWS):
            slot = int(np.nonzero(bodies == row_body[articulation, row])[0][0])
            coefficients_np[articulation, row, slot * 6 : slot * 6 + 6] = row_wrench[articulation, row]

    basis_body = wp.empty((env.world_count, 2), dtype=wp.int32, device=device)
    coefficients = wp.empty((env.world_count * _ROWS, _BASIS), dtype=wp.float32, device=device)
    coefficients_96 = wp.zeros((env.world_count * 96, _BASIS), dtype=wp.float32, device=device)
    basis_packed = wp.zeros((env.world_count * _BASIS, _PACKED_COLUMNS), dtype=wp.float32, device=device)
    coefficients_96_np = np.zeros((env.world_count, 96, _BASIS), dtype=np.float32)
    coefficients_96_np[:, :_ROWS] = coefficients_np
    coefficients_96.assign(coefficients_96_np.reshape(-1, _BASIS))
    synthesized = wp.zeros((env.world_count * _ROWS, _PACKED_COLUMNS), dtype=wp.float32, device=device)
    scattered_jacobian = wp.empty((env.world_count, _ROWS, _DOFS), dtype=wp.float32, device=device)
    scattered_response = wp.empty_like(scattered_jacobian)

    def launch_classify() -> None:
        wp.launch(
            _classify_basis_kernel,
            dim=env.world_count,
            inputs=[block.row_body, block.row_wrench],
            outputs=[basis_body, coefficients],
            device=device,
        )

    def launch_classify_parallel() -> None:
        wp.launch(
            _classify_basis_bodies_kernel,
            dim=env.world_count,
            inputs=[block.row_body],
            outputs=[basis_body],
            device=device,
        )
        wp.launch(
            _build_basis_coefficients_kernel,
            dim=(env.world_count, _ROWS),
            inputs=[block.row_body, block.row_wrench, basis_body],
            outputs=[coefficients],
            device=device,
        )

    def launch_direct() -> None:
        wp.launch(
            _build_packed_generalized_contact_rows_kernel,
            dim=(env.world_count, 96),
            block_dim=int(args.basis_block_dim),
            inputs=[
                env.solver.world.bodies,
                block.enabled,
                block.point_count,
                block.row_body,
                block.row_wrench,
                block.max_page_count,
                block.page_index,
                wp.bool(True),
            ],
            outputs=[block.packed_jacobian, block.packed_response, block.aba_joint_work, block.aba_body_response],
            device=device,
        )

    def launch_basis() -> None:
        wp.launch(
            _build_unit_wrench_basis_kernel,
            dim=env.world_count * _BASIS_GROUP,
            block_dim=int(args.basis_launch_block_dim),
            inputs=[env.solver.world.bodies, basis_body],
            outputs=[basis_packed, block.aba_joint_work, block.aba_body_response],
            device=device,
        )

    def launch_synthesis() -> None:
        wp.launch_tiled(
            _synthesize_tile_kernel,
            dim=[env.world_count, _PACKED_COLUMNS // _COLUMN_TILE],
            block_dim=64,
            inputs=[coefficients, basis_packed],
            outputs=[synthesized],
            device=device,
        )

    def launch_synthesis_split_96() -> None:
        wp.launch_tiled(
            _synthesize_split_tile_96_kernel,
            dim=[env.world_count, 2],
            block_dim=128,
            inputs=[coefficients_96, basis_packed],
            outputs=[block.packed_jacobian, block.packed_response],
            device=device,
        )

    def launch_scatter() -> None:
        wp.launch(
            _scatter_basis_kernel,
            dim=(env.world_count, _ROWS, _DOFS),
            inputs=[synthesized],
            outputs=[scattered_jacobian, scattered_response],
            device=device,
        )

    def launch_synthesis_split() -> None:
        wp.launch_tiled(
            _synthesize_split_tile_kernel,
            dim=[env.world_count, 2],
            block_dim=64,
            inputs=[coefficients, basis_packed],
            outputs=[block.packed_jacobian, block.packed_response],
            device=device,
        )

    launch_classify()
    wp.synchronize_device(device)
    np.testing.assert_array_equal(basis_body.numpy(), basis_body_np)
    np.testing.assert_allclose(coefficients.numpy(), coefficients_np.reshape(-1, _BASIS), rtol=0.0, atol=0.0)
    launch_direct()
    wp.synchronize_device(device)
    direct_jacobian = block.packed_jacobian.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    direct_response = block.packed_response.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    launch_basis()
    launch_synthesis()
    launch_scatter()
    wp.synchronize_device(device)
    synthesized_np = synthesized.numpy().reshape(env.world_count, _ROWS, _PACKED_COLUMNS)
    jacobian_error = float(np.max(np.abs(synthesized_np[:, :, :_DOFS] - direct_jacobian)))
    response_error = float(np.max(np.abs(synthesized_np[:, :, _DOFS:] - direct_response)))
    launch_synthesis_split()
    wp.synchronize_device(device)
    split_jacobian = block.packed_jacobian.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    split_response = block.packed_response.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    split_jacobian_error = float(np.max(np.abs(split_jacobian - direct_jacobian)))
    split_response_error = float(np.max(np.abs(split_response - direct_response)))
    direct_us = _time_graph(device, launch_direct, int(args.replays))
    classify_us = _time_graph(device, launch_classify, int(args.replays))
    classify_parallel_us = _time_graph(device, launch_classify_parallel, int(args.replays))
    basis_us = _time_graph(device, launch_basis, int(args.replays))
    synthesis_us = _time_graph(device, launch_synthesis, int(args.replays))
    synthesis_split_us = _time_graph(device, launch_synthesis_split, int(args.replays))
    synthesis_split_96_us = _time_graph(device, launch_synthesis_split_96, int(args.replays))
    scatter_us = _time_graph(device, launch_scatter, int(args.replays))
    adaptive_total_us = classify_us + basis_us + synthesis_us + scatter_us
    adaptive_parallel_total_us = classify_parallel_us + basis_us + synthesis_split_us
    print(
        json.dumps(
            {
                "basis_block_dim": int(args.basis_block_dim),
                "basis_launch_block_dim": int(args.basis_launch_block_dim),
                "basis_build_us": basis_us,
                "adaptive_parallel_projected_speedup": direct_us / adaptive_parallel_total_us,
                "adaptive_parallel_total_us": adaptive_parallel_total_us,
                "adaptive_projected_speedup": direct_us / adaptive_total_us,
                "adaptive_total_us": adaptive_total_us,
                "basis_plus_synthesis_us": basis_us + synthesis_us,
                "classify_parallel_us": classify_parallel_us,
                "classify_us": classify_us,
                "direct_us": direct_us,
                "jacobian_max_abs_error": jacobian_error,
                "projected_speedup": direct_us / (basis_us + synthesis_us),
                "response_max_abs_error": response_error,
                "scatter_us": scatter_us,
                "split_jacobian_max_abs_error": split_jacobian_error,
                "split_response_max_abs_error": split_response_error,
                "synthesis_split_96_us": synthesis_split_96_us,
                "synthesis_split_us": synthesis_split_us,
                "synthesis_us": synthesis_us,
                "world_count": int(args.world_count),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
