# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for SolverBox3D with kernel-level timing.

Run with: uv run python3 newton/tests/bench_box3d.py
"""

import time
from collections import defaultdict

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig


def _build_sphere_on_ground():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)))
    b.add_shape_sphere(body=body, radius=0.5)
    return b


def _build_box_stack(n=10):
    b = newton.ModelBuilder()
    b.add_ground_plane()
    h = 0.5
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    for i in range(n):
        shift = -0.01 if i % 2 == 0 else 0.01
        body = b.add_body(xform=wp.transform(wp.vec3(shift, 0.0, h + 2.0 * h * i)))
        b.add_shape_box(body=body, hx=h, hy=h, hz=h, cfg=cfg)
    return b


def _build_joint_chain(n=10):
    b = newton.ModelBuilder()
    anchor = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 20.0)), is_kinematic=True)
    b.add_shape_sphere(body=anchor, radius=0.05)
    cfg = newton.ModelBuilder.ShapeConfig(density=10.0, mu=0.5)
    prev = anchor
    for i in range(n):
        link = b.add_body(xform=wp.transform(wp.vec3(0.5 + 1.0 * i, 0.0, 20.0)))
        b.add_shape_capsule(body=link, radius=0.1, half_height=0.4, cfg=cfg)
        p_xf = wp.transform_identity() if prev == anchor else wp.transform(wp.vec3(0.5, 0.0, 0.0))
        b.add_joint_revolute(parent=prev, child=link,
                             parent_xform=p_xf,
                             child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)),
                             axis=newton.Axis.Y)
        prev = link
    return b


def benchmark(scene_name, scene_fn, num_worlds, num_steps=200, warmup=20, profile_kernels=False):
    device = "cuda:0"

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    sub = scene_fn()
    for _ in range(num_worlds):
        scene.add_world(sub)

    model = scene.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(
        num_substeps=4, num_velocity_iters=1, num_relaxation_iters=1,
        contact_hertz=30.0, joint_hertz=60.0,
        linear_damping=0.1, angular_damping=0.1,
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
    kernel_timings = defaultdict(list)

    t0 = time.perf_counter()
    for step_i in range(num_steps):
        if profile_kernels:
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN)

        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

        if profile_kernels:
            results = wp.timing_end()
            for r in results:
                kernel_timings[r.name].append(r.elapsed)

    wp.synchronize_device(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    step_ms = elapsed / num_steps * 1000.0
    fps = num_steps / elapsed

    return {
        "name": scene_name,
        "worlds": num_worlds,
        "bodies": model.body_count,
        "joints": model.joint_count,
        "step_ms": step_ms,
        "fps": fps,
        "kernel_timings": dict(kernel_timings),
    }


def print_results(results, show_kernels=False):
    r = results
    bodies_per_sec = r["bodies"] * r["fps"]
    print(f"  {r['worlds']:>6} worlds | {r['bodies']:>6} bodies | {r['joints']:>6} joints | "
          f"{r['step_ms']:.3f} ms/step | {r['fps']:.0f} FPS | {bodies_per_sec:.0f} bodies/s")

    if show_kernels and r["kernel_timings"]:
        # Sort by total time
        sorted_kernels = sorted(r["kernel_timings"].items(),
                                key=lambda kv: sum(kv[1]), reverse=True)
        total_kernel_ms = sum(sum(v) for v in r["kernel_timings"].values())
        print(f"  {'Kernel':<60} {'Total ms':>10} {'Avg ms':>10} {'Count':>6} {'%':>6}")
        print(f"  {'-'*96}")
        for name, times in sorted_kernels[:15]:  # top 15
            total = sum(times)
            avg = total / len(times)
            pct = total / total_kernel_ms * 100 if total_kernel_ms > 0 else 0
            short_name = name.split("::")[-1] if "::" in name else name
            if len(short_name) > 58:
                short_name = short_name[:55] + "..."
            print(f"  {short_name:<60} {total:>10.3f} {avg:>10.4f} {len(times):>6} {pct:>5.1f}%")
        print(f"  {'TOTAL':<60} {total_kernel_ms:>10.3f}")
        print()


def main():
    wp.init()
    print("=" * 80)
    print("Box3D Performance Benchmarks (kernel-level timing)")
    print("=" * 80)

    # ── Contact-heavy: box stacks ──
    print("\n📦 Contact-heavy: 10-Box Stack per world")
    for nw in [100, 500, 1000]:
        r = benchmark("box_stack_10", lambda: _build_box_stack(10), nw,
                       num_steps=100, profile_kernels=(nw == 1000))
        print_results(r, show_kernels=(nw == 1000))

    # ── Joint-heavy: 20-link chains ──
    print("\n🔗 Joint-heavy: 20-link revolute chain per world")
    for nw in [100, 500, 1000]:
        r = benchmark("chain_20", lambda: _build_joint_chain(20), nw,
                       num_steps=100, profile_kernels=(nw == 1000))
        print_results(r, show_kernels=(nw == 1000))

    # ── Simple scaling: sphere on ground ──
    print("\n⚪ Simple: 1 sphere per world")
    for nw in [100, 1000, 2000, 4000]:
        r = benchmark("sphere", _build_sphere_on_ground, nw, num_steps=200)
        print_results(r)

    # ── Peak memory ──
    peak_mem = wp.get_mempool_used_mem_high("cuda:0")
    print(f"\n💾 Peak GPU memory: {peak_mem / 1024**2:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
