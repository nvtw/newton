# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Screen joint-row PGS with ordinary contacts on source Colibri geometry."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import contact_chunks
from local_studies.colibri.bilateral_pgs import install, install_fused
from local_studies.colibri.hard_rigid_contacts import install as install_hard_contacts
from local_studies.colibri.parallel_prepare import install as install_parallel_prepare
from local_studies.colibri.phoenx_scene import Example
from local_studies.colibri.temporal_schedule import install_temporal_schedule
from local_studies.colibri.validate_phoenx import ContactAudit
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-count", type=int, default=2)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10, help="Exclude these initial frames from warm timing only")
    parser.add_argument("--substeps", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--block", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--public-solver", action="store_true", help="Use constructor-owned joint/contact options")
    parser.add_argument("--parallel-prepare", action="store_true")
    parser.add_argument("--mass-splitting", action="store_true")
    parser.add_argument(
        "--contact-chunk-size",
        type=int,
        default=0,
        help="Experimental maximum points per contact column; 0 preserves grouping",
    )
    parser.add_argument("--max-colored-partitions", type=int, default=12)
    parser.add_argument("--mass-splitting-batch-size", type=int, default=8)
    parser.add_argument("--mass-splitting-color-group-size", type=int, default=0)
    parser.add_argument("--hard-contacts", action="store_true")
    parser.add_argument("--relax-every-substep", action="store_true")
    parser.add_argument("--contact-offset-map", type=Path, help="Use measured per-shape SI contact offsets")
    parser.add_argument("--speculative-contact-gap-max", type=float, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.contact_chunk_size < 0:
        parser.error("--contact-chunk-size must be nonnegative")
    if args.contact_chunk_size and not (args.fused or args.public_solver):
        parser.error("--contact-chunk-size requires --fused")
    if args.contact_chunk_size and not args.public_solver:
        contact_chunks.reserve_capacity()
    if args.hard_contacts:
        install_hard_contacts()
    if not args.relax_every_substep and not args.public_solver:
        install_temporal_schedule()
    args.mode = "maximal"
    args.no_graph = True
    args.layout = "single_world"
    args.outer_substeps = 2
    args.contact_gap = 0.001
    args.source_contact_offsets = True
    args.mesh_cylinders = True
    example = Example(ViewerNull(), args)
    if args.public_solver:
        pass
    elif args.fused:
        install_fused(example.solver, mass_splitting=args.mass_splitting)
    elif args.block:
        install(example.solver, block=True)
    else:
        install(example.solver)
    if args.contact_chunk_size and not args.public_solver:
        contact_chunks.install(example.solver, chunk_size=args.contact_chunk_size)
    if args.parallel_prepare and not args.public_solver:
        install_parallel_prepare(example.solver)
    with wp.ScopedCapture(device=example.model.device) as capture:
        example.simulate()
    example.graph = capture.graph
    args.no_graph = False
    audit = ContactAudit(example.model)
    output = args.output or Path(f"/tmp/colibri_bilateral_pgs_{args.body_count}_{time.time_ns()}.json")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    times = []
    maximum_depth = 0.0
    failure = None
    frame = 0
    metric = None
    try:
        for frame in range(args.frames + 1):
            if frame:
                start = time.perf_counter()
                example.step()
                wp.synchronize_device(example.model.device)
                times.append(time.perf_counter() - start)
            metric = audit.check(example.state_0)
            maximum_depth = max(maximum_depth, metric["depth_m"])
            example.test_post_step()
            if metric["depth_m"] > 0.001:
                raise AssertionError(f"Fresh depth {metric['depth_m']} m exceeds 1 mm")
            if frame % 10 == 0:
                print(json.dumps({"frame": frame, **metric}), flush=True)
    except AssertionError as error:
        failure = str(error)
        world = example.solver.world
        np.savez(
            output.with_suffix(".npz"),
            q=example.state_0.body_q.numpy(),
            qd=example.state_0.body_qd.numpy(),
            labels=example.model.body_label,
            contact_impulses=world._contact_container.impulses.numpy(),
            contact_derived=world._contact_container.derived.numpy(),
            contact_owner=world._contact_cols.articulation_owner.numpy(),
            constraint_node=world.bodies.constraint_node.numpy(),
            active_constraints=world._num_active_constraints.numpy(),
            config=json.dumps(config),
            frame=frame,
        )
    warm_times = times[max(args.warmup, 0) :]
    contact_count = int(example.contacts.rigid_contact_count.numpy()[0])
    impulses = example.solver.world._contact_container.impulses.numpy()[:, :contact_count]
    workload = {
        "generated_points": contact_count,
        "solver_columns": int(example.solver.world._ingest_scratch.num_contact_columns.numpy()[0]),
        "nonzero_normal_impulse_points": int(np.count_nonzero(impulses[0])),
        "nonzero_tangent_impulse_points": int(np.count_nonzero(np.any(impulses[1:3] != 0.0, axis=0))),
        "max_normal_impulse_Ns": float(np.max(impulses[0])) if contact_count else 0.0,
        "scope": "Cached contacts from last solve; fresh validation uses separate contacts; exact nonzero counts",
    }
    world = example.solver.world
    copy_counts = world._copy_state.count_per_node.numpy()
    values, frequencies = np.unique(copy_counts, return_counts=True)
    color_count = int(world._partitioner.num_colors.numpy()[0])
    color_starts = world._partitioner.color_starts.numpy()
    overflow_count = 0
    if world.max_colored_partitions is not None and color_count > world.max_colored_partitions:
        cap = world.max_colored_partitions
        overflow_count = int(color_starts[cap + 1] - color_starts[cap])
    copy_workload = {
        "slots_in_use": int(world._copy_state.highest_index_in_use.numpy()[0]),
        "max_copies_per_node": int(np.max(copy_counts)) if copy_counts.size else 0,
        "copy_count_histogram": {
            str(int(value)): int(frequency) for value, frequency in zip(values, frequencies, strict=True)
        },
        "colors": color_count,
        "overflow_constraints": overflow_count,
        "overflow_batches": (overflow_count + args.mass_splitting_batch_size - 1) // args.mass_splitting_batch_size,
    }
    report = {
        "status": "failed" if failure is not None else "passed",
        "failure": failure,
        "config": config,
        "completed_frames": frame,
        "max_depth_m": maximum_depth,
        "last_contact_check": metric,
        "solver_contact_workload": workload,
        "copy_workload": copy_workload,
        "physics_mean_ms_including_cold_frames": 1000.0 * float(np.mean(times)) if times else None,
        "warm_timing_frames": len(warm_times),
        "physics_mean_ms": 1000.0 * float(np.mean(warm_times)) if warm_times else None,
        "physics_p95_ms": 1000.0 * float(np.percentile(warm_times, 95)) if warm_times else None,
        "physics_fps": 1.0 / float(np.mean(warm_times)) if warm_times else None,
        "scope": "Diagnostic joint PGS and ordinary rigid contacts, no global equality solve; fresh physical checks",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
