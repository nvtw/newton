# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for SolverBox3D.

Run with: uv run python3 newton/tests/bench_box3d.py
"""

import time

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig


def _build_sphere_on_ground():
    """Simple scene: 1 sphere on ground plane."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)))
    builder.add_shape_sphere(body=b, radius=0.5)
    return builder


def _build_box_stack(n=10):
    """N-box vertical stack."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    h = 0.5
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    for i in range(n):
        shift = -0.01 if i % 2 == 0 else 0.01
        b = builder.add_body(xform=wp.transform(wp.vec3(shift, 0.0, h + 2.0 * h * i)))
        builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=cfg)
    return builder


def benchmark_multi_world(scene_builder, num_worlds, num_steps=100, warmup=10):
    """Benchmark multi-world simulation."""
    device = "cuda:0"

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    sub = scene_builder()
    for w in range(num_worlds):
        scene.add_world(sub)

    model = scene.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(
        num_substeps=4,
        num_velocity_iters=1,
        num_relaxation_iters=1,
        contact_hertz=30.0,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    contacts = pipeline.contacts()
    dt = 1.0 / 60.0

    # Warmup
    for _ in range(warmup):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    wp.synchronize_device(device)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_steps):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    wp.synchronize_device(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    step_ms = elapsed / num_steps * 1000.0
    fps = num_steps / elapsed
    body_count = model.body_count

    return {
        "worlds": num_worlds,
        "bodies": body_count,
        "steps": num_steps,
        "total_s": elapsed,
        "step_ms": step_ms,
        "fps": fps,
    }


def main():
    wp.init()
    print("=" * 70)
    print("Box3D Performance Benchmarks")
    print("=" * 70)

    # Sphere on ground — scaling from 1 to 1000 worlds
    print("\n--- Sphere on Ground (1 body/world) ---")
    print(f"{'Worlds':>8} {'Bodies':>8} {'Step (ms)':>10} {'FPS':>8} {'Bodies/s':>12}")
    print("-" * 56)
    for nw in [1, 10, 100, 500, 1000]:
        r = benchmark_multi_world(_build_sphere_on_ground, nw, num_steps=200, warmup=20)
        bodies_per_sec = r["bodies"] * r["fps"]
        print(f"{r['worlds']:>8} {r['bodies']:>8} {r['step_ms']:>10.3f} {r['fps']:>8.1f} {bodies_per_sec:>12.0f}")

    # 10-box stack — scaling
    print("\n--- 10-Box Stack (10 bodies/world) ---")
    print(f"{'Worlds':>8} {'Bodies':>8} {'Step (ms)':>10} {'FPS':>8} {'Bodies/s':>12}")
    print("-" * 56)
    for nw in [1, 10, 100, 500]:
        r = benchmark_multi_world(lambda: _build_box_stack(10), nw, num_steps=100, warmup=10)
        bodies_per_sec = r["bodies"] * r["fps"]
        print(f"{r['worlds']:>8} {r['bodies']:>8} {r['step_ms']:>10.3f} {r['fps']:>8.1f} {bodies_per_sec:>12.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
