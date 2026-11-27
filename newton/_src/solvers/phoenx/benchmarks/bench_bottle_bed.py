# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed bottle bed: the bottling plant's pasteuriser bed on its own.

Builds the plant's filling transport (single-file infeed, a funnel into a
packed bed of plant bottles on moving belts, the outfeed conveyor and the
accumulation pool) with the plant's bottle colliders, materials and solver
settings, but without the rest of the factory. The bed forms one giant
contact island, which the island sweep solves as its merged island, so the
benchmark isolates the latency-bound color steps of
``phoenx_iterate_islands`` and ``phoenx_relax_islands`` in a few minutes
instead of a full factory run. ``--bed`` scales the bed length and with it
the columns per color (see ``PERF_NOTES.md``, round 8).

Examples::

    # Captured frame time.
    python -m newton._src.solvers.phoenx.benchmarks.bench_bottle_bed --bed 6

    # Per-kernel times: profile one captured window with nsys.
    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -o bed \\
        python -m newton._src.solvers.phoenx.benchmarks.bench_bottle_bed --profile --repeats 1 --frames 120

    # Bitwise A/B: compare the final body state of two source trees.
    python -m newton._src.solvers.phoenx.benchmarks.bench_bottle_bed --frames 120 --warmup 120 --repeats 1 --dump a.npz

    # Uncaptured frames for Nsight Compute kernel replay
    # (see profile_bottle_bed_iterate_ncu.sh).
    python -m newton._src.solvers.phoenx.benchmarks.bench_bottle_bed --warmup 240 --eager 1
"""

import argparse
import ctypes
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_process import add_filling_bottles, add_filling_transport
from newton.examples.phoenx._bottling_scene import add_round_bottle

#: Collision rate [Hz]; each captured frame runs two steps (60 FPS).
_STEP_RATE = 120.0


def _build(bed_length: float, half_width: float):
    """Return the model and its belts: the plant's filling transport around a bed of ``bed_length`` m."""
    x0 = 1.0
    bed_start = x0 + 0.3 + half_width  # after the funnel
    bed_end = bed_start + bed_length
    layout = {
        "infeed": (
            np.array([x0, x0 + 0.3, bed_start, bed_start + 0.6]),
            np.array([0.12, 0.12, half_width, half_width]),
        ),
        "outfeed": (
            np.array([bed_start + 0.6, bed_end, bed_end + 1.1, bed_end + 1.4]),
            np.array([half_width, half_width, 0.12, 0.12]),
        ),
        "combiner_end": bed_end + 4.6,
        "dryer": bed_end + 2.0,
        "rails": (0.15, (0.05, 0.10)),
        "pool": (
            np.array([bed_end + 5.0, bed_end + 6.0, bed_end + 6.05, bed_end + 6.6]),
            np.array([0.35, 0.35, 0.15, 0.15]),
        ),
    }
    path = np.array([[0.0, 0.0, 0.95], [bed_end + 6.6, 0.0, 0.95]])

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    glass = builder.ShapeConfig(density=0.0, mu=0.16, mu_static=0.2, gap=0.002)
    # The source bottle templates the filled bottles.
    source = builder.add_body(
        xform=wp.transform((0.0, -3.0, 0.0), wp.quat_identity()),
        mass=0.570,
        com=wp.vec3(0.0, 0.0, 0.0834),
        inertia=wp.mat33(np.diag((0.00184, 0.00184, 0.000261))),
        lock_inertia=True,
    )
    ring = wp.transform((0.0, 0.0, 0.008), wp.quat_identity())
    builder.add_shape_cylinder(source, xform=ring, radius=0.0305, half_height=0.008, cfg=glass)
    add_round_bottle(builder, source, "/Bottle/Collision", cfg=glass)
    scene = SimpleNamespace(builder=builder, belts=[], metadata={}, bodies={}, bottle_bodies=[source])
    add_filling_transport(scene, path, motor_off=False, layout=layout, inlet_speed=0.14)
    add_filling_bottles(scene, path, layout=layout)
    model = builder.finalize()
    for mesh, velocity, _ in scene.belts:
        mesh.mesh.velocities.assign(velocity)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bed", type=float, default=6.0, help="pasteuriser bed length [m]")
    parser.add_argument("--half-width", type=float, default=1.15, help="bed half-width [m]")
    parser.add_argument("--warmup", type=int, default=240, help="frames before timing")
    parser.add_argument("--frames", type=int, default=240, help="timed frames per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="wrap the last repeat in cuProfilerStart/Stop")
    parser.add_argument("--dump", default=None, help="save the final body_q and body_qd to this .npz")
    parser.add_argument("--eager", type=int, default=0, help="step warmup + EAGER frames uncaptured, then exit")
    args = parser.parse_args()
    wp.config.quiet = True

    with wp.ScopedDevice("cuda:0"):
        model = _build(args.bed, args.half_width)
        print("bodies", model.body_count, "shapes", model.shape_count, flush=True)  # noqa: T201
        # The plant's solver settings (example_phoenx_bottling_plant).
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=3,
            solver_iterations=4,
            temporal_iterations=True,
            contact_hertz=80.0,
            contact_chunk_size=64,
            enable_body_pair_grouping=True,
            velocity_iterations=1,
            step_layout="single_world",
            sleeping_velocity_threshold=0.01,
            sleeping_frames_required=1,
            sleeping_motion_threshold=0.0002,
        )
        state = model.state()
        contacts = model._collision_pipeline.contacts()
        dt = 1.0 / _STEP_RATE

        def frame():
            for _ in range(2):
                solver.collide(state, contacts, dt=dt)
                solver.step(state, state, None, contacts, dt)

        def contact_count():
            return int(contacts.rigid_contact_count.numpy()[0])

        if args.eager:
            # Uncaptured launches, for per-kernel replay profilers such as Nsight Compute.
            for _ in range(args.warmup + args.eager):
                frame()
            wp.synchronize()
            print("contacts", contact_count(), flush=True)  # noqa: T201
            return
        with wp.ScopedCapture() as capture:
            frame()
        for _ in range(args.warmup):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        times = []
        for repeat in range(args.repeats):
            profiled = args.profile and repeat == args.repeats - 1
            if profiled:
                ctypes.CDLL("libcuda.so.1").cuProfilerStart()
            start = time.perf_counter()
            for _ in range(args.frames):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            times.append((time.perf_counter() - start) * 1000.0 / args.frames)
            if profiled:
                ctypes.CDLL("libcuda.so.1").cuProfilerStop()
        print("ms/frame", " ".join(f"{t:.3f}" for t in times), "contacts", contact_count(), flush=True)  # noqa: T201
        if args.dump:
            np.savez(args.dump, q=state.body_q.numpy(), qd=state.body_qd.numpy())


if __name__ == "__main__":
    main()
