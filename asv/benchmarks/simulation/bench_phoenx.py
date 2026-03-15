# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for the PhoenX PGS constraint solver.

Builds a chain of rigid bodies connected by mixed joint types
(revolute, prismatic, ball-socket, fixed) and measures per-kernel
timing of the solver pipeline.
"""

import math
import time

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

from newton._src.solvers.phoenx.solver_phoenx import SolverState


def _build_chain(num_bodies, device):
    """Build a chain of num_bodies connected by alternating joint types.

    Body 0 is static (anchor). Subsequent bodies hang along -Z.
    Joint types cycle: revolute → prismatic → ball-socket → fixed.
    """
    joint_capacity = max(num_bodies - 1, 0)
    ss = SolverState(
        body_capacity=num_bodies,
        contact_capacity=1,
        shape_count=0,
        device=device,
        joint_capacity=joint_capacity,
    )

    spacing = 0.5  # metres between bodies along Z
    handles = []
    for i in range(num_bodies):
        h = ss.add_body(
            position=(0.0, 0.0, -i * spacing),
            inverse_mass=0.0 if i == 0 else 1.0,
            is_static=(i == 0),
        )
        handles.append(h)

    joint_types = ["revolute", "prismatic", "ball_socket", "fixed"]
    for i in range(1, num_bodies):
        anchor = (0.0, 0.0, -(i - 0.5) * spacing)
        jtype = joint_types[(i - 1) % len(joint_types)]
        if jtype == "revolute":
            ss.add_joint_revolute(handles[i - 1], handles[i], anchor, axis_world=(1.0, 0.0, 0.0))
        elif jtype == "prismatic":
            ss.add_joint_prismatic(handles[i - 1], handles[i], anchor, axis_world=(0.0, 0.0, 1.0))
        elif jtype == "ball_socket":
            ss.add_joint_ball_socket(handles[i - 1], handles[i], anchor)
        elif jtype == "fixed":
            ss.add_joint_fixed(handles[i - 1], handles[i], anchor)

    ss.update_world_inertia()
    return ss


# ---------------------------------------------------------------------------
# ASV benchmark class
# ---------------------------------------------------------------------------


class PhoenXConstraintChain:
    """Benchmark PhoenX PGS solver with a constraint chain."""

    params = ([32, 128, 512],)
    param_names = ["num_bodies"]
    repeat = 3
    number = 1
    warmup_time = 0

    def setup(self, num_bodies):
        if wp.get_cuda_device_count() == 0:
            from asv_runner.benchmarks.mark import SkipNotImplemented

            raise SkipNotImplemented
        self.device = wp.get_cuda_device(0)
        self.ss = _build_chain(num_bodies, self.device)
        # Warm up (compile kernels)
        self.ss.step(dt=1.0 / 240.0, gravity=(0.0, 0.0, -9.81), num_iterations=4)
        wp.synchronize_device(self.device)

    def time_step(self, num_bodies):
        self.ss.step(dt=1.0 / 240.0, gravity=(0.0, 0.0, -9.81), num_iterations=8)
        wp.synchronize_device(self.device)


# ---------------------------------------------------------------------------
# Standalone runner with per-phase timing
# ---------------------------------------------------------------------------


def _timed_step(ss, dt, gravity, num_iterations, device):
    """Run one solver step with per-phase CUDA event timing.

    Returns a dict mapping phase names to elapsed milliseconds.
    """
    inv_dt = 1.0 / dt if dt > 0.0 else 0.0

    def _record():
        e = wp.Event(enable_timing=True)
        wp.record_event(e)
        return e

    phases = {}

    def _phase(name):
        """Context manager-like helper: call _phase('x'), do work, call _end()."""
        phases[name] = _record()

    def _end(name):
        phases[name + "_end"] = _record()

    # -- integrate velocities --
    _phase("integrate_velocities")
    ss.integrate_velocities(gravity, dt)
    _end("integrate_velocities")

    # -- partition --
    _phase("partition")
    ss._partition_contacts()
    _end("partition")

    # -- mass splitting --
    from newton._src.solvers.phoenx.kernels import clear_contact_count_kernel, count_contacts_per_body_kernel

    bs = ss.body_store
    cs = ss.contact_store

    if ss.joint_store is not None and ss._joint_count > 0:
        ss._cached_contact_count = int(cs.count.numpy()[0])
    else:
        ss._cached_contact_count = 0

    _phase("mass_splitting")
    wp.launch(clear_contact_count_kernel, dim=bs.capacity,
              inputs=[ss._contact_count_per_body, bs.count], device=device)
    wp.launch(count_contacts_per_body_kernel, dim=cs.capacity,
              inputs=[cs.column_of("body0"), cs.column_of("body1"), cs.count, ss._contact_count_per_body],
              device=device)
    _end("mass_splitting")

    max_slots = ss.graph_coloring.max_colors + 1

    # Read partition sizes once (matches optimized step())
    p_ends = ss.graph_coloring.partition_ends.numpy()
    active_slots = []
    for p in range(max_slots):
        p_start = int(p_ends[p - 1]) if p > 0 else 0
        p_end = int(p_ends[p])
        size = p_end - p_start
        if size > 0:
            active_slots.append((p, size))

    # -- prepare --
    _phase("prepare_constraints")
    for p, size in active_slots:
        ss._launch_prepare(p, inv_dt, dim=size)
        ss._launch_prepare_constraints(p, dim=size)
    _end("prepare_constraints")

    # -- position iterations --
    _phase("solve_position")
    for _ in range(num_iterations):
        for p, size in active_slots:
            ss._launch_solve(p, 1, dim=size)
            ss._launch_solve_constraints(p, 1, dim=size)
    _end("solve_position")

    # -- integrate positions --
    _phase("integrate_positions")
    ss.integrate_positions(dt)
    _end("integrate_positions")

    wp.synchronize_device(device)

    # Compute elapsed ms
    results = {}
    for name in ["integrate_velocities", "partition", "mass_splitting",
                  "prepare_constraints", "solve_position", "integrate_positions"]:
        beg = phases[name]
        end = phases[name + "_end"]
        results[name] = wp.get_event_elapsed_time(beg, end)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PhoenX solver benchmark with per-phase GPU timing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--num-bodies", type=int, default=128, help="Number of bodies in the chain.")
    parser.add_argument("-s", "--steps", type=int, default=60, help="Number of steps to run.")
    parser.add_argument("-i", "--iterations", type=int, default=8, help="PGS iterations per step.")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Time step [s].")
    parser.add_argument("--asv", action="store_true", help="Run ASV-style time_step benchmark.")
    args = parser.parse_args()

    if args.asv:
        from newton.utils import run_benchmark

        run_benchmark(PhoenXConstraintChain)
    else:
        device = wp.get_cuda_device(0) if wp.get_cuda_device_count() > 0 else wp.get_device("cpu")
        print(f"Building chain with {args.num_bodies} bodies on {device}...")
        ss = _build_chain(args.num_bodies, device)

        # Warm up
        ss.step(dt=args.dt, gravity=(0.0, 0.0, -9.81), num_iterations=args.iterations)
        wp.synchronize_device(device)

        # Timed run
        accum = {}
        wall_start = time.perf_counter()
        for step_i in range(args.steps):
            t = _timed_step(ss, args.dt, (0.0, 0.0, -9.81), args.iterations, device)
            for k, v in t.items():
                accum[k] = accum.get(k, 0.0) + v
        wall_elapsed = time.perf_counter() - wall_start

        print(f"\n{'Phase':<25s} {'Total (ms)':>10s} {'Avg (ms)':>10s} {'%':>6s}")
        print("-" * 55)
        total_gpu = sum(accum.values())
        for name in ["integrate_velocities", "partition", "mass_splitting",
                      "prepare_constraints", "solve_position", "integrate_positions"]:
            total_ms = accum[name]
            avg_ms = total_ms / args.steps
            pct = 100.0 * total_ms / total_gpu if total_gpu > 0 else 0.0
            print(f"{name:<25s} {total_ms:>10.2f} {avg_ms:>10.3f} {pct:>5.1f}%")
        print("-" * 55)
        print(f"{'GPU total':<25s} {total_gpu:>10.2f} {total_gpu / args.steps:>10.3f}")
        print(f"{'Wall clock':<25s} {wall_elapsed * 1000:>10.2f} {wall_elapsed * 1000 / args.steps:>10.3f}")
        print(f"\nThroughput: {args.steps / wall_elapsed:.1f} steps/s")
