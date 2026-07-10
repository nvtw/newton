# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark: FP32 vs FP16 storage for the packed reduced-contact rows.

The reduced contact phase streams ``packed_jacobian`` and ``packed_response``
(rows x nv fp32 each) from the row builder into every tile-GS sweep plus the
transpose kernel, and is bandwidth-bound at large world counts. This bench
measures the construction-time FP16 row-storage variant
(``PHOENX_CONTACT_ROWS_FP16``, see ``reduced_contact_block.py``) against
production FP32 on the identical live standing-G1 scene and snapshot:

* value-range analysis of the row entries (justifies the FP16 format),
* same-snapshot generalized-velocity-delta deviation (max/rms),
* bit-stable determinism across two captured replays,
* graph-captured reversed-bracket timing of the full contact phase and the
  biased / relax passes separately.

Scene, snapshot, and timing machinery are reused from
``bench_warp_world_contact_oracle``.

Example::

    uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.experimental.bench_fp16_contact_rows
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _BLOCK_DIM,
    _advance_reduced_contact_page_cursor_kernel,
    _apply_generalized_contact_delta_kernel,
    _get_build_packed_rows_kernel,
    _get_transpose_response_kernel,
    _reset_reduced_contact_page_cursor_kernel,
    _solve_generalized_contact_tile_ops,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_warp_world_contact_oracle import (
    _build_scene,
    _time_graph_batches,
)


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    magnitudes = np.abs(values)
    return {
        "count": int(values.size),
        "min_abs_nonzero": float(np.min(magnitudes)),
        "p1_abs": float(np.percentile(magnitudes, 1)),
        "p50_abs": float(np.percentile(magnitudes, 50)),
        "p99_abs": float(np.percentile(magnitudes, 99)),
        "max_abs": float(np.max(magnitudes)),
        "below_fp16_min_normal_6e-5": float(np.mean(magnitudes < 6.1e-5)),
        "above_fp16_max_65504": float(np.mean(magnitudes > 65504.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument("--replays", type=int, default=300)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("this benchmark requires CUDA")

    model, solver, state, control, contacts = _build_scene(args.world_count, device)

    def step() -> None:
        model.collide(state, contacts)
        state.clear_forces()
        solver.step(state, state, control, contacts, args.dt)

    for _ in range(args.settle_steps):
        step()
    with wp.ScopedCapture(device=device) as step_capture:
        step()
    for _ in range(10):
        wp.capture_launch(step_capture.graph)
    wp.synchronize_device(device)

    world = solver.world
    reduced_system = solver._reduced_articulation
    block = reduced_system.contact_block_system
    assert block.packed_jacobian is not None and block.packed_response is not None
    assert not block.packed_rows_fp16, "run this benchmark with the FP16 flag OFF; it builds the variant itself"
    bodies = world.bodies
    reduced = bodies.reduced
    cc = world._contact_container
    columns = world._contact_cols
    contact_views = world._active_contact_views()
    idt = wp.float32(1.0 / args.dt)
    sor_boost = float(world.sor_boost)
    iterations_biased = 2
    iterations_relax = 1
    articulation_count = block.articulation_count
    nv_stride = int(block.contact_dof_width)
    max_pages = int(block.max_page_count.numpy()[0])

    # ---- two-page gather from one body state (same as the oracle bench) ----
    def launch_gather_pages() -> None:
        wp.launch(
            _reset_reduced_contact_page_cursor_kernel,
            dim=1,
            inputs=[block.max_page_count],
            outputs=[block.page_cursor, block.page_index],
            device=device,
        )
        for _page in range(max_pages):
            wp.launch(
                block.gather_kernel,
                dim=(articulation_count, block.gather_tile_width),
                block_dim=_BLOCK_DIM,
                inputs=[
                    block.schedule_section_end,
                    block.schedule_columns,
                    columns,
                    bodies,
                    idt,
                    wp.bool(True),
                    cc,
                    contact_views,
                    block.enabled,
                    block.total_point_count,
                    block.page_index,
                ],
                outputs=[
                    block.point_count,
                    block.point_contact,
                    block.point_column,
                    block.point0,
                    block.point1,
                    block.normal,
                    block.tangent0,
                    block.row_body,
                    block.row_wrench,
                    block.row_velocity,
                ],
                device=device,
            )
            wp.launch(
                _advance_reduced_contact_page_cursor_kernel,
                dim=1,
                inputs=[block.page_cursor, block.page_index],
                device=device,
            )
        wp.launch(
            _reset_reduced_contact_page_cursor_kernel,
            dim=1,
            inputs=[block.max_page_count],
            outputs=[block.page_cursor, block.page_index],
            device=device,
        )

    # Perturb in generalized coordinates so the snapshot contact solve does
    # real work (post-step snapshots are degenerate; see the oracle bench).
    rng = np.random.default_rng(11)
    perturbation = (args.perturbation * rng.standard_normal((articulation_count, nv_stride))).astype(np.float32)
    block.generalized_delta.assign(perturbation)
    block.generalized_body_delta.zero_()
    wp.launch(
        _apply_generalized_contact_delta_kernel,
        dim=articulation_count,
        inputs=[bodies, block.enabled, block.generalized_delta],
        outputs=[block.generalized_body_delta],
        device=device,
    )
    launch_gather_pages()
    wp.synchronize_device(device)

    # ---- FP16 variant buffers/kernels (aliasing nothing; same shapes) ----
    fp16_jacobian = wp.zeros(block.packed_jacobian.shape, dtype=wp.float16, device=device)
    fp16_response = wp.zeros_like(fp16_jacobian)
    pair_shape = (block.packed_jacobian.shape[0], nv_stride // 2)
    fp16_jacobian_pairs = wp.array(ptr=fp16_jacobian.ptr, dtype=wp.uint32, shape=pair_shape, device=device)
    fp16_response_pairs = wp.array(ptr=fp16_response.ptr, dtype=wp.uint32, shape=pair_shape, device=device)
    generalized_delta_pairs = wp.array(
        ptr=block.generalized_delta.ptr,
        dtype=wp.vec2,
        shape=(articulation_count, nv_stride // 2),
        device=device,
    )
    quad_shape = (block.packed_jacobian.shape[0], nv_stride // 4)
    fp16_jacobian_quads = wp.array(ptr=fp16_jacobian.ptr, dtype=wp.uint64, shape=quad_shape, device=device)
    fp16_response_quads = wp.array(ptr=fp16_response.ptr, dtype=wp.uint64, shape=quad_shape, device=device)
    generalized_delta_quads = wp.array(
        ptr=block.generalized_delta.ptr,
        dtype=wp.vec4,
        shape=(articulation_count, nv_stride // 4),
        device=device,
    )
    fp16_build = _get_build_packed_rows_kernel(True)
    fp16_transpose = _get_transpose_response_kernel(True)

    # variant -> (solve, build, transpose, store_j, store_r, solve_j, solve_r, delta)
    variants = {
        "fp32": (
            block.solve_kernel,
            block.build_rows_kernel,
            block.transpose_kernel,
            block.packed_jacobian,
            block.packed_response,
            block.packed_jacobian,
            block.packed_response,
            block.generalized_delta,
        ),
        "fp16": (
            _solve_generalized_contact_tile_ops(nv_stride, "fp16")[1],
            fp16_build,
            fp16_transpose,
            fp16_jacobian,
            fp16_response,
            fp16_jacobian,
            fp16_response,
            block.generalized_delta,
        ),
        "fp16x2": (
            _solve_generalized_contact_tile_ops(nv_stride, "fp16x2")[1],
            fp16_build,
            fp16_transpose,
            fp16_jacobian,
            fp16_response,
            fp16_jacobian_pairs,
            fp16_response_pairs,
            generalized_delta_pairs,
        ),
        "fp16x4": (
            _solve_generalized_contact_tile_ops(nv_stride, "fp16x4")[1],
            fp16_build,
            fp16_transpose,
            fp16_jacobian,
            fp16_response,
            fp16_jacobian_quads,
            fp16_response_quads,
            generalized_delta_quads,
        ),
        # Lossless: fp32 rows, body_response streamed as 3x8B words (bit-identical).
        "fp32w": (
            block.solve_kernel,
            _get_build_packed_rows_kernel(False, True),
            block.transpose_kernel,
            block.packed_jacobian,
            block.packed_response,
            block.packed_jacobian,
            block.packed_response,
            block.generalized_delta,
        ),
        "fp16x2w": (
            _solve_generalized_contact_tile_ops(nv_stride, "fp16x2")[1],
            _get_build_packed_rows_kernel(True, True),
            fp16_transpose,
            fp16_jacobian,
            fp16_response,
            fp16_jacobian_pairs,
            fp16_response_pairs,
            generalized_delta_pairs,
        ),
    }

    snapshot_sources = {
        "velocity": bodies.velocity,
        "angular_velocity": bodies.angular_velocity,
        "joint_qd": reduced.joint_qd,
        "cc_impulses": cc.impulses,
        "cc_prev_impulses": cc.prev_impulses,
        "cc_lambdas": cc.lambdas,
        "cc_prev_lambdas": cc.prev_lambdas,
        "cc_derived": cc.derived,
        "packed_jacobian": block.packed_jacobian,
        "packed_response": block.packed_response,
        "fp16_jacobian": fp16_jacobian,
        "fp16_response": fp16_response,
        "packed_previous_row_body": block.packed_previous_row_body,
        "page_index": block.page_index,
        "page_cursor": block.page_cursor,
        "generalized_delta": block.generalized_delta,
        "generalized_body_delta": block.generalized_body_delta,
    }
    snapshots = {name: wp.clone(array) for name, array in snapshot_sources.items()}

    def restore() -> None:
        for name, array in snapshot_sources.items():
            wp.copy(array, snapshots[name])

    def apply_variant(name: str) -> None:
        solve, build, transpose, store_j, store_r, solve_j, solve_r, delta = variants[name]
        block.solve_kernel = solve
        block.build_rows_kernel = build
        block.transpose_kernel = transpose
        block.packed_jacobian = store_j
        block.packed_response = store_r
        block.packed_jacobian_solve = solve_j
        block.packed_response_solve = solve_r
        block.generalized_delta_solve = delta

    def launch_biased() -> None:
        block.solve(columns, bodies, idt, sor_boost, cc, contact_views, iterations_biased, use_bias=True, prepare=True)

    def launch_relax() -> None:
        block.solve(columns, bodies, idt, sor_boost, cc, contact_views, iterations_relax, use_bias=False, prepare=False)

    def launch_phase() -> None:
        launch_biased()
        launch_relax()

    graphs: dict[str, object] = {}
    for variant in variants:
        apply_variant(variant)
        for phase_name, launch in (("biased", launch_biased), ("relax", launch_relax), ("phase", launch_phase)):
            restore()
            with wp.ScopedCapture(device=device) as capture:
                launch()
            graphs[f"{variant}_{phase_name}"] = capture.graph
    apply_variant("fp32")

    # ---- value-range analysis (fp32 baseline rows after one biased pass) ----
    restore()
    wp.capture_launch(graphs["fp32_biased"])
    wp.synchronize_device(device)
    jacobian_np = snapshot_sources["packed_jacobian"].numpy()
    response_np = snapshot_sources["packed_response"].numpy()
    value_ranges = {
        "packed_jacobian_nonzero": _percentiles(jacobian_np[jacobian_np != 0.0]),
        "packed_response_nonzero": _percentiles(response_np[response_np != 0.0]),
    }

    # ---- physical validation from identical snapshots ----
    qd_before = snapshots["joint_qd"].numpy()
    v_before = snapshots["velocity"].numpy()
    restore()
    wp.capture_launch(graphs["fp32_phase"])
    wp.synchronize_device(device)
    fp32_qd_delta = reduced.joint_qd.numpy() - qd_before
    fp32_v_delta = bodies.velocity.numpy() - v_before
    fp32_impulses = cc.impulses.numpy().copy()

    qd_scale = max(1.0e-6, float(np.max(np.abs(fp32_qd_delta))))
    v_scale = max(1.0e-6, float(np.max(np.abs(fp32_v_delta))))
    impulse_scale = max(1.0e-6, float(np.max(np.abs(fp32_impulses))))
    validation: dict[str, dict] = {"fp32_qd_delta_max_abs": qd_scale}
    for variant in variants:
        if variant == "fp32":
            continue
        restore()
        wp.capture_launch(graphs[f"{variant}_phase"])
        wp.synchronize_device(device)
        qd_delta = reduced.joint_qd.numpy() - qd_before
        qd_raw = reduced.joint_qd.numpy().copy()
        v_delta = bodies.velocity.numpy() - v_before
        impulses = cc.impulses.numpy().copy()
        if not np.isfinite(qd_delta).all():
            raise RuntimeError(f"{variant} produced non-finite generalized velocity deltas")
        qd_dev = np.abs(qd_delta - fp32_qd_delta)
        entry = {
            "qd_delta_max_abs_deviation": float(np.max(qd_dev)),
            "qd_delta_rms_deviation": float(np.sqrt(np.mean(qd_dev**2))),
            "qd_delta_max_rel_deviation": float(np.max(qd_dev)) / qd_scale,
            "qd_delta_rms_rel_deviation": float(np.sqrt(np.mean(qd_dev**2))) / qd_scale,
            "body_velocity_delta_max_rel_deviation": float(np.max(np.abs(v_delta - fp32_v_delta))) / v_scale,
            "impulse_max_rel_deviation": float(np.max(np.abs(impulses - fp32_impulses))) / impulse_scale,
        }
        # Determinism: a second captured replay from the restored snapshot
        # must be bit-identical.
        restore()
        wp.capture_launch(graphs[f"{variant}_phase"])
        wp.synchronize_device(device)
        entry["bit_stable_across_replays"] = bool(np.array_equal(reduced.joint_qd.numpy(), qd_raw))
        validation[variant] = entry

    # ---- graph-captured timing, reversed-order brackets ----
    order = list(graphs)
    times: dict[str, list[float]] = {name: [] for name in order}
    for direction in (order, list(reversed(order)), order, list(reversed(order))):
        for name in direction:
            restore()
            times[name].extend(_time_graph_batches(graphs[name], args.replays, args.batches, device))
    median_us = {name: float(np.median(values)) for name, values in times.items()}

    payload = {
        "schema": "phoenx_fp16_contact_rows_v1",
        "device": device.name,
        "world_count": args.world_count,
        "dt": args.dt,
        "nv_stride": nv_stride,
        "pages": max_pages,
        "iterations_biased": iterations_biased,
        "iterations_relax": iterations_relax,
        "value_ranges": value_ranges,
        "validation": validation,
        "median_us": median_us,
        "speedups_vs_fp32": {
            f"{variant}_{phase}": median_us[f"fp32_{phase}"] / median_us[f"{variant}_{phase}"]
            for variant in variants
            if variant != "fp32"
            for phase in ("biased", "relax", "phase")
        },
        "timing_batches_us": dict(times),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    code = main()
    print(f"total wall time: {time.perf_counter() - start:.1f} s")
    raise SystemExit(code)
