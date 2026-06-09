# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-shaped rigid rows3 world-loop unification benchmark.

This benchmark sits between ``bench_rigid_rows3_sidecar`` and the production
PhoenX fast-tail kernels. It keeps the production-like local scheduling shape:
one fixed lane group owns one world and loops that world's colors locally. The
work item, however, is the same local rigid three-row solve used by the sidecar
benchmark:

* ``split`` branches between contact-like point rows and angular joint-like
  rows, then calls the shared projection helper;
* ``frame`` uses the compact :class:`RigidFrameRows3` descriptor for both row
  families, so every lane performs the same residual/project/apply sequence;
* ``shape`` keeps the same compact descriptor and projection representation,
  but calls a contact-point or angular-direct local helper to avoid wasted
  arithmetic on homogeneous row shapes.

The scene presets are deliberately topology-shaped rather than full solver
runs. They use representative per-world color sizes and contact ratios from the
PhoenX robot/contact benchmark graphs so we can test whether full local
unification still looks promising once the scheduler shape is production-like.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.experimental.bench_rigid_rows3_sidecar import (
    _FAMILY_CONTACT,
    _alloc_mat,
    _alloc_vec,
    _bench,
    _init_rows_kernel,
    _make_projection_op,
    _max_err,
)
from newton._src.solvers.phoenx.constraints.constraint_block import (
    RigidFrameRows3,
    RigidFrameRows3State,
    block_solve_rigid_frame_rows3,
    block_solve_rigid_frame_rows3_angular,
    block_solve_rigid_frame_rows3_contact,
    block_solve_velocity_rows3_op,
)


@dataclass(frozen=True)
class ScenePreset:
    rows_per_color: tuple[int, ...]
    contact_ratio: float


_SCENE_PRESETS: dict[str, ScenePreset] = {
    "h1": ScenePreset((11, 6, 5, 2), 4.0 / 24.0),
    "g1": ScenePreset((19, 14, 8, 2), 0.0),
    "dr_legs": ScenePreset((14, 13, 9, 5, 1), 2.0 / 42.0),
    "tower": ScenePreset((64, 61, 58, 55, 52, 49, 46, 44, 42, 40, 38, 36, 34), 1.0),
}


@dataclass(frozen=True)
class ScheduleHost:
    row_ids: np.ndarray
    color_starts: np.ndarray
    world_color_starts: np.ndarray
    rows: int
    colors: int


def _build_schedule(preset: ScenePreset, worlds: int) -> ScheduleHost:
    row_ids: list[int] = []
    color_starts: list[int] = [0]
    world_color_starts: list[int] = [0]
    row = 0
    for _world in range(worlds):
        for count in preset.rows_per_color:
            for _ in range(count):
                row_ids.append(row)
                row += 1
            color_starts.append(len(row_ids))
        world_color_starts.append(len(color_starts) - 1)
    return ScheduleHost(
        row_ids=np.asarray(row_ids, dtype=np.int32),
        color_starts=np.asarray(color_starts, dtype=np.int32),
        world_color_starts=np.asarray(world_color_starts, dtype=np.int32),
        rows=row,
        colors=len(color_starts) - 1,
    )


@wp.kernel(enable_backward=False)
def _solve_split_world_loop_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    normal: wp.array[wp.vec3f],
    tangent1: wp.array[wp.vec3f],
    tangent2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                va = v_a[row]
                wa = w_a[row]
                vb = v_b[row]
                wb = w_b[row]

                if family[row] == _FAMILY_CONTACT:
                    n = normal[row]
                    t1 = tangent1[row]
                    t2 = tangent2[row]
                    rr0 = r0[row]
                    rr1 = r1[row]
                    rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                    residual = wp.vec3f(wp.dot(n, rel), wp.dot(t1, rel), wp.dot(t2, rel)) + bias[row]
                    op = _make_projection_op(
                        k_inv[row],
                        residual,
                        lambda_old[row],
                        mass_coeff[row],
                        impulse_coeff[row],
                        lambda_min[row],
                        lambda_max[row],
                        projection_mode[row],
                        friction_static[row],
                        friction_kinetic[row],
                    )
                    update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
                    impulse = update.delta[0] * n + update.delta[1] * t1 + update.delta[2] * t2
                    out_va[row] = va - inv_m_a[row] * impulse
                    out_wa[row] = wa - inv_i_a[row] @ wp.cross(rr0, impulse)
                    out_vb[row] = vb + inv_m_b[row] * impulse
                    out_wb[row] = wb + inv_i_b[row] @ wp.cross(rr1, impulse)
                    out_lambda[row] = update.lambda_new
                else:
                    a0 = axis0[row]
                    a1 = axis1[row]
                    a2 = axis2[row]
                    rel_w = wb - wa
                    residual = wp.vec3f(wp.dot(a0, rel_w), wp.dot(a1, rel_w), wp.dot(a2, rel_w)) + bias[row]
                    op = _make_projection_op(
                        k_inv[row],
                        residual,
                        lambda_old[row],
                        mass_coeff[row],
                        impulse_coeff[row],
                        lambda_min[row],
                        lambda_max[row],
                        projection_mode[row],
                        friction_static[row],
                        friction_kinetic[row],
                    )
                    update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
                    angular_impulse = update.delta[0] * a0 + update.delta[1] * a1 + update.delta[2] * a2
                    out_va[row] = va
                    out_wa[row] = wa - inv_i_a[row] @ angular_impulse
                    out_vb[row] = vb
                    out_wb[row] = wb + inv_i_b[row] @ angular_impulse
                    out_lambda[row] = update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_frame_world_loop_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    frame_mode: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                rows = RigidFrameRows3()
                rows.axis0 = axis0[row]
                rows.axis1 = axis1[row]
                rows.axis2 = axis2[row]
                rows.r0 = r0[row]
                rows.r1 = r1[row]
                rows.mode = frame_mode[row]
                rows.k_inv = k_inv[row]
                rows.bias = bias[row]
                rows.lambda_old = lambda_old[row]
                rows.mass_coeff = mass_coeff[row]
                rows.impulse_coeff = impulse_coeff[row]
                rows.lambda_min = lambda_min[row]
                rows.lambda_max = lambda_max[row]
                rows.projection_mode = projection_mode[row]
                rows.friction_static = friction_static[row]
                rows.friction_kinetic = friction_kinetic[row]

                state = RigidFrameRows3State()
                state.v_a = v_a[row]
                state.w_a = w_a[row]
                state.v_b = v_b[row]
                state.w_b = w_b[row]
                state.inv_m_a = inv_m_a[row]
                state.inv_m_b = inv_m_b[row]
                state.inv_i_a = inv_i_a[row]
                state.inv_i_b = inv_i_b[row]

                update = block_solve_rigid_frame_rows3(rows, state, wp.float32(1.0))
                out_va[row] = update.v_a
                out_wa[row] = update.w_a
                out_vb[row] = update.v_b
                out_wb[row] = update.w_b
                out_lambda[row] = update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_shape_world_loop_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    frame_mode: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                rows = RigidFrameRows3()
                rows.axis0 = axis0[row]
                rows.axis1 = axis1[row]
                rows.axis2 = axis2[row]
                rows.r0 = r0[row]
                rows.r1 = r1[row]
                rows.mode = frame_mode[row]
                rows.k_inv = k_inv[row]
                rows.bias = bias[row]
                rows.lambda_old = lambda_old[row]
                rows.mass_coeff = mass_coeff[row]
                rows.impulse_coeff = impulse_coeff[row]
                rows.lambda_min = lambda_min[row]
                rows.lambda_max = lambda_max[row]
                rows.projection_mode = projection_mode[row]
                rows.friction_static = friction_static[row]
                rows.friction_kinetic = friction_kinetic[row]

                state = RigidFrameRows3State()
                state.v_a = v_a[row]
                state.w_a = w_a[row]
                state.v_b = v_b[row]
                state.w_b = w_b[row]
                state.inv_m_a = inv_m_a[row]
                state.inv_m_b = inv_m_b[row]
                state.inv_i_a = inv_i_a[row]
                state.inv_i_b = inv_i_b[row]

                if family[row] == _FAMILY_CONTACT:
                    update = block_solve_rigid_frame_rows3_contact(rows, state, wp.float32(1.0))
                else:
                    update = block_solve_rigid_frame_rows3_angular(rows, state, wp.float32(1.0))
                out_va[row] = update.v_a
                out_wa[row] = update.w_a
                out_vb[row] = update.v_b
                out_wb[row] = update.w_b
                out_lambda[row] = update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


def _parse_scenes(value: str) -> tuple[str, ...]:
    scenes = tuple(raw.strip() for raw in value.split(",") if raw.strip())
    for scene in scenes:
        if scene not in _SCENE_PRESETS:
            raise ValueError(f"unknown scene {scene!r}; choices={tuple(_SCENE_PRESETS)}")
    return scenes


def _run_scene(args: argparse.Namespace, scene: str, device: wp.context.Devicelike) -> None:
    preset = _SCENE_PRESETS[scene]
    schedule = _build_schedule(preset, int(args.worlds))
    rows = schedule.rows
    contacts_per_period = int(round(preset.contact_ratio * float(args.period)))

    row_ids = wp.array(schedule.row_ids, dtype=wp.int32, device=device)
    color_starts = wp.array(schedule.color_starts, dtype=wp.int32, device=device)
    world_color_starts = wp.array(schedule.world_color_starts, dtype=wp.int32, device=device)

    family = wp.empty(rows, dtype=wp.int32, device=device)
    v_a = _alloc_vec(rows, device)
    w_a = _alloc_vec(rows, device)
    v_b = _alloc_vec(rows, device)
    w_b = _alloc_vec(rows, device)
    inv_m_a = wp.empty(rows, dtype=wp.float32, device=device)
    inv_m_b = wp.empty(rows, dtype=wp.float32, device=device)
    inv_i_a = _alloc_mat(rows, device)
    inv_i_b = _alloc_mat(rows, device)
    normal = _alloc_vec(rows, device)
    tangent1 = _alloc_vec(rows, device)
    tangent2 = _alloc_vec(rows, device)
    r0 = _alloc_vec(rows, device)
    r1 = _alloc_vec(rows, device)
    axis0 = _alloc_vec(rows, device)
    axis1 = _alloc_vec(rows, device)
    axis2 = _alloc_vec(rows, device)
    j_unused = [_alloc_vec(rows, device) for _ in range(12)]
    k_inv = _alloc_vec(rows, device)
    bias = _alloc_vec(rows, device)
    lambda_old = _alloc_vec(rows, device)
    mass_coeff = _alloc_vec(rows, device)
    impulse_coeff = _alloc_vec(rows, device)
    lambda_min = _alloc_vec(rows, device)
    lambda_max = _alloc_vec(rows, device)
    projection_mode = wp.empty(rows, dtype=wp.int32, device=device)
    friction_static = wp.empty(rows, dtype=wp.float32, device=device)
    friction_kinetic = wp.empty(rows, dtype=wp.float32, device=device)
    frame_mode = _alloc_vec(rows, device)

    split_va = _alloc_vec(rows, device)
    split_wa = _alloc_vec(rows, device)
    split_vb = _alloc_vec(rows, device)
    split_wb = _alloc_vec(rows, device)
    split_lambda = _alloc_vec(rows, device)
    frame_va = _alloc_vec(rows, device)
    frame_wa = _alloc_vec(rows, device)
    frame_vb = _alloc_vec(rows, device)
    frame_wb = _alloc_vec(rows, device)
    frame_lambda = _alloc_vec(rows, device)
    shape_va = _alloc_vec(rows, device)
    shape_wa = _alloc_vec(rows, device)
    shape_vb = _alloc_vec(rows, device)
    shape_wb = _alloc_vec(rows, device)
    shape_lambda = _alloc_vec(rows, device)

    wp.launch(
        _init_rows_kernel,
        dim=rows,
        inputs=[
            family,
            v_a,
            w_a,
            v_b,
            w_b,
            inv_m_a,
            inv_m_b,
            inv_i_a,
            inv_i_b,
            normal,
            tangent1,
            tangent2,
            r0,
            r1,
            axis0,
            axis1,
            axis2,
            *j_unused,
            k_inv,
            bias,
            lambda_old,
            mass_coeff,
            impulse_coeff,
            lambda_min,
            lambda_max,
            projection_mode,
            friction_static,
            friction_kinetic,
            frame_mode,
            int(args.period),
            contacts_per_period,
        ],
        device=device,
    )

    split_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        normal,
        tangent1,
        tangent2,
        r0,
        r1,
        axis0,
        axis1,
        axis2,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        split_va,
        split_wa,
        split_vb,
        split_wb,
        split_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]
    frame_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        frame_mode,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        frame_va,
        frame_wa,
        frame_vb,
        frame_wb,
        frame_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]
    shape_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        frame_mode,
        k_inv,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        shape_va,
        shape_wa,
        shape_vb,
        shape_wb,
        shape_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]

    def split_run() -> None:
        wp.launch(
            _solve_split_world_loop_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=split_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def frame_run() -> None:
        wp.launch(
            _solve_frame_world_loop_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=frame_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def shape_run() -> None:
        wp.launch(
            _solve_shape_world_loop_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=shape_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    split_run()
    frame_run()
    shape_run()
    frame_err = _max_err(
        (frame_va, split_va),
        (frame_wa, split_wa),
        (frame_vb, split_vb),
        (frame_wb, split_wb),
        (frame_lambda, split_lambda),
    )
    shape_err = _max_err(
        (shape_va, split_va),
        (shape_wa, split_wa),
        (shape_vb, split_vb),
        (shape_wb, split_wb),
        (shape_lambda, split_lambda),
    )
    split_ms, _ = _bench(split_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    frame_ms, _ = _bench(frame_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    shape_ms, _ = _bench(shape_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    frame_speedup = split_ms / frame_ms if frame_ms > 0.0 else float("nan")
    shape_speedup = split_ms / shape_ms if shape_ms > 0.0 else float("nan")
    print(
        f"{scene:8s} worlds={int(args.worlds):5d} rows={rows:7d} colors={schedule.colors:5d} "
        f"contact_ratio={preset.contact_ratio:5.3f} split={split_ms:8.4f}ms "
        f"frame={frame_ms:8.4f}ms shape={shape_ms:8.4f}ms "
        f"frame_speedup={frame_speedup:6.3f}x shape_speedup={shape_speedup:6.3f}x "
        f"frame_err={frame_err:.6g} shape_err={shape_err:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", default="h1,g1,dr_legs,tower")
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--threads-per-world", type=int, default=32)
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    print(
        f"device={device} worlds={args.worlds} iterations={args.iterations} "
        f"tpw={args.threads_per_world} n_runs={args.n_runs} trials={args.trials}"
    )
    for scene in _parse_scenes(args.scenes):
        _run_scene(args, scene, device)


if __name__ == "__main__":
    main()
