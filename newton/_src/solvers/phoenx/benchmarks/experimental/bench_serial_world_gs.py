# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Feasibility benchmark for no-color serial per-world rigid PGS.

This intentionally does not change the production solver. It compares the
existing fast-tail rigid solve against an experimental kernel where one CUDA
thread owns one world and walks that world's uncolored constraint stream
sequentially. The proxy keeps production prepare/iterate dispatch helpers and
fast-tail sweep counts so the measured variable is scheduler shape, not row
math.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Callable

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver, _force_tpw
from newton._src.solvers.phoenx.benchmarks.scenarios import g1_flat
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _make_multiworld_rigid_iterate_dispatch_funcs,
    _make_multiworld_rigid_prepare_dispatch_func,
)


@functools.cache
def _make_serial_world_prepare_plus_iterate_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_neohookean: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
    solve_joint_inner_sweeps: int = 3,
    solve_contact_inner_sweeps: int = 3,
    solve_outer_iteration_chunk: int = 1,
):
    """Build a timing-only one-thread-per-world rigid prepare+iterate kernel."""
    if cloth_support or soft_tet_neohookean:
        raise NotImplementedError("serial-world benchmark supports rigid scenes only")
    _ = fixed_tpw, guard_tpw, family_split

    (_dispatch_prepare_cid, _dispatch_prepare_joint, _dispatch_prepare_contact) = (
        _make_multiworld_rigid_prepare_dispatch_func(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_soft_contact_pd=has_soft_contact_pd,
            cached_prepare=cached_prepare,
            enable_column_timers=enable_column_timers,
        )
    )
    (_dispatch_iterate_cid, _dispatch_iterate_joint, _dispatch_iterate_contact) = (
        _make_multiworld_rigid_iterate_dispatch_funcs(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            enable_column_timers=enable_column_timers,
            use_bias=True,
        )
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        per_world_elements: wp.array[wp.int32],
        per_world_offsets: wp.array[wp.int32],
        per_world_counts: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
        num_bodies: wp.int32,
        copy_state: CopyStateContainer,
    ):
        world_id = wp.tid()
        if world_id >= num_worlds:
            return

        start = per_world_offsets[world_id]
        count = per_world_counts[world_id]

        i = wp.int32(0)
        while i < count:
            cid = per_world_elements[start + i]
            _dispatch_prepare_cid(
                constraints,
                contact_cols,
                bodies,
                particles,
                cc,
                contacts,
                copy_state,
                num_bodies,
                idt,
                cid,
                num_joints,
                joint_pgs_enabled,
            )
            i += wp.int32(1)

        solve_outer_chunk = wp.int32(solve_outer_iteration_chunk)
        solve_outer_iterations = (num_iterations + solve_outer_chunk - wp.int32(1)) / solve_outer_chunk
        it_outer = wp.int32(0)
        while it_outer < solve_outer_iterations:
            i = wp.int32(0)
            while i < count:
                cid = per_world_elements[start + i]
                if wp.static(solve_joint_inner_sweeps == solve_contact_inner_sweeps):
                    _dispatch_iterate_cid(
                        constraints,
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        num_joints,
                        joint_pgs_enabled,
                        wp.int32(solve_joint_inner_sweeps),
                    )
                elif wp.static(has_joints and not has_contacts):
                    if wp.static(not skip_joint_pgs):
                        if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] != wp.int32(0):
                            _dispatch_iterate_joint(
                                constraints,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                wp.int32(solve_joint_inner_sweeps),
                            )
                elif cid < num_joints:
                    if wp.static(not skip_joint_pgs):
                        if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] != wp.int32(0):
                            _dispatch_iterate_joint(
                                constraints,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                wp.int32(solve_joint_inner_sweeps),
                            )
                else:
                    _dispatch_iterate_contact(
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid - num_joints,
                        wp.int32(solve_contact_inner_sweeps),
                    )
                i += wp.int32(1)
            it_outer += wp.int32(1)

    return kernel


@functools.cache
def _make_serial_world_relax_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_neohookean: bool = False,
    enable_column_timers: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
    solve_joint_inner_sweeps: int = 3,
    solve_contact_inner_sweeps: int = 3,
    solve_outer_iteration_chunk: int = 1,
):
    """Build a timing-only one-thread-per-world rigid relax kernel."""
    if cloth_support or soft_tet_neohookean:
        raise NotImplementedError("serial-world benchmark supports rigid scenes only")
    _ = fixed_tpw, guard_tpw, family_split, solve_joint_inner_sweeps, solve_contact_inner_sweeps
    _ = solve_outer_iteration_chunk

    (_dispatch_iterate_cid, _dispatch_iterate_joint, _dispatch_iterate_contact) = (
        _make_multiworld_rigid_iterate_dispatch_funcs(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            enable_column_timers=enable_column_timers,
            use_bias=False,
        )
    )
    _ = _dispatch_iterate_joint, _dispatch_iterate_contact

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        per_world_elements: wp.array[wp.int32],
        per_world_offsets: wp.array[wp.int32],
        per_world_counts: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
        num_bodies: wp.int32,
        copy_state: CopyStateContainer,
    ):
        world_id = wp.tid()
        if world_id >= num_worlds:
            return

        start = per_world_offsets[world_id]
        count = per_world_counts[world_id]
        i = wp.int32(0)
        while i < count:
            cid = per_world_elements[start + i]
            _dispatch_iterate_cid(
                constraints,
                contact_cols,
                bodies,
                particles,
                cc,
                contacts,
                copy_state,
                num_bodies,
                idt,
                sor_boost,
                cid,
                num_joints,
                joint_pgs_enabled,
                num_iterations,
            )
            i += wp.int32(1)

    return kernel


def _bench_graph(fn: Callable[[], None], *, warmup: int, replays: int, trials: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    wp.synchronize_device()

    with wp.ScopedCapture() as capture:
        fn()
    graph = capture.graph
    wp.synchronize_device()

    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(replays):
            wp.capture_launch(graph)
        wp.synchronize_device()
        times.append((time.perf_counter() - t0) * 1000.0 / float(replays))
    arr = np.asarray(times, dtype=np.float64)
    return float(arr.min()), float(np.median(arr))


def _serial_launch(world, prepare_kernel, relax_kernel) -> None:
    idt = wp.float32(1.0 / world.substep_dt)
    contact_views = world._active_contact_views()
    common_inputs = [
        world.constraints,
        world._contact_cols,
        world.bodies,
        world._particles_or_sentinel(),
        idt,
        wp.float32(world.sor_boost),
        world._per_world_elements,
        world._per_world_element_offsets,
        world._per_world_element_count,
        world._contact_container,
        contact_views,
        wp.int32(world.solver_iterations),
        wp.int32(world.num_worlds),
        wp.int32(world.num_joints),
        world._joint_pgs_enabled,
        wp.int32(world.num_bodies),
        world._copy_state,
    ]
    wp.launch(prepare_kernel, dim=world.num_worlds, inputs=common_inputs, device=world.device)

    if world.velocity_iterations > 0:
        relax_inputs = list(common_inputs)
        relax_inputs[11] = wp.int32(world.velocity_iterations)
        wp.launch(relax_kernel, dim=world.num_worlds, inputs=relax_inputs, device=world.device)


def _fast_tail_launch(world) -> None:
    world._solve_main()
    world._relax_velocities()


def _parse_stride(value: str) -> int | str:
    return "auto" if value.strip().lower() == "auto" else int(value)


def _parse_threads_per_world(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    tpw = int(normalized)
    if tpw not in (8, 16, 32):
        raise argparse.ArgumentTypeError("threads-per-world must be 'auto' or one of 8, 16, 32")
    return tpw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--prepare-refresh-stride", type=_parse_stride, default=1)
    parser.add_argument("--threads-per-world", type=_parse_threads_per_world, default="auto")
    parser.add_argument("--prime-frames", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args()

    wp.init()
    handle = g1_flat.build(
        args.world_count,
        "phoenx",
        args.substeps,
        args.solver_iterations,
        velocity_iterations=args.velocity_iterations,
        prepare_refresh_stride=args.prepare_refresh_stride,
    )
    solver = _extract_solver(handle)
    world = solver.world
    world._configure_multi_world_scheduler("fast_tail")
    _force_tpw(solver, args.threads_per_world)

    for _ in range(args.prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    fixed_tpw = world._fast_tail_fixed_tpw()
    fast_tail_kw = world._fast_tail_kernel_flags(fixed_tpw, cached_prepare=False)
    prepare_kernel = _make_serial_world_prepare_plus_iterate_kernel(**fast_tail_kw)
    relax_kw = dict(fast_tail_kw)
    relax_kw.pop("cached_prepare", None)
    relax_kernel = _make_serial_world_relax_kernel(**relax_kw)

    coloring_min, coloring_med = _bench_graph(
        world._build_per_world_coloring,
        warmup=args.warmup,
        replays=args.replays,
        trials=args.trials,
    )
    fast_tail_min, fast_tail_med = _bench_graph(
        lambda: _fast_tail_launch(world),
        warmup=args.warmup,
        replays=args.replays,
        trials=args.trials,
    )
    serial_min, serial_med = _bench_graph(
        lambda: _serial_launch(world, prepare_kernel, relax_kernel),
        warmup=args.warmup,
        replays=args.replays,
        trials=args.trials,
    )

    result = {
        "world_count": args.world_count,
        "substeps": args.substeps,
        "solver_iterations": args.solver_iterations,
        "velocity_iterations": args.velocity_iterations,
        "prepare_refresh_stride": args.prepare_refresh_stride,
        "threads_per_world": args.threads_per_world,
        "effective_threads_per_world": fixed_tpw,
        "replays": args.replays,
        "trials": args.trials,
        "coloring_min_ms": coloring_min,
        "coloring_median_ms": coloring_med,
        "fast_tail_solve_min_ms": fast_tail_min,
        "fast_tail_solve_median_ms": fast_tail_med,
        "serial_solve_min_ms": serial_min,
        "serial_solve_median_ms": serial_med,
        "serial_vs_fast_tail_solve": serial_min / fast_tail_min if fast_tail_min > 0.0 else float("nan"),
        "coloring_vs_fast_tail_solve": coloring_min / fast_tail_min if fast_tail_min > 0.0 else float("nan"),
    }
    print(json.dumps(result, indent=args.json_indent, sort_keys=True))


if __name__ == "__main__":
    main()
