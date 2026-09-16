# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate Colibri geometry while measuring physics separately from checks.

Run with ``uv run --no-project /path/to/newton/.venv/bin/python -m
local_studies.colibri.validate_phoenx`` from the PhoenX worktree. The scene
uses source geometry and SI units; no USD loading occurs during simulation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.phoenx_scene import Example
from newton.viewer import ViewerNull


@wp.kernel(enable_backward=False)
def _contact_separation(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    separation: wp.array[wp.float32],
):
    i = wp.tid()
    b0 = shape_body[shape0[i]]
    b1 = shape_body[shape1[i]]
    p0 = point0[i]
    p1 = point1[i]
    if b0 >= 0:
        p0 = wp.transform_point(body_q[b0], p0)
    if b1 >= 0:
        p1 = wp.transform_point(body_q[b1], p1)
    separation[i] = wp.dot(p1 - p0, normal[i]) - margin0[i] - margin1[i]


class ContactAudit:
    """Recompute contacts independently of solver caches and warm starts."""

    def __init__(self, model, *, capacity=16384):
        self.model = model
        self.capacity = capacity
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=capacity, contact_matching="disabled")
        self.contacts = self.pipeline.contacts()
        self.separation = wp.zeros(capacity, dtype=wp.float32, device=model.device)
        self.filtered_pairs = set(model.shape_collision_filter_pairs)

    def check(self, state):
        self.pipeline.collide(state, self.contacts)
        c = self.contacts
        count = int(c.rigid_contact_count.numpy()[0])
        if count >= self.capacity:
            raise AssertionError(f"Contact audit capacity reached: {count}/{self.capacity}")
        if count == 0:
            return {"count": 0, "depth_m": 0.0, "worst_shapes": []}
        wp.launch(
            _contact_separation,
            dim=count,
            inputs=[
                state.body_q,
                self.model.shape_body,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
            ],
            outputs=[self.separation],
            device=self.model.device,
        )
        gaps = self.separation.numpy()[:count]
        shape0 = c.rigid_contact_shape0.numpy()[:count]
        shape1 = c.rigid_contact_shape1.numpy()[:count]
        if not np.isfinite(gaps).all():
            raise AssertionError("Nonfinite fresh contact separation")
        for s0, s1 in zip(shape0, shape1, strict=True):
            if tuple(sorted((s0, s1))) in self.filtered_pairs:
                raise AssertionError(f"Filtered shape pair generated a contact: {s0}, {s1}")
        worst = int(np.argmin(gaps))
        return {
            "count": count,
            "depth_m": max(0.0, -float(gaps[worst])),
            "worst_shapes": [self.model.shape_label[shape0[worst]], self.model.shape_label[shape1[worst]]],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-count", type=int, default=15)
    parser.add_argument("--mode", choices=("maximal", "reduced"), default="reduced")
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per 120 Hz collision update")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--max-depth", type=float, default=0.001, help="Penetration failure threshold [m]")
    parser.add_argument("--contact-gap", type=float, default=0.001, help="Unauthored contact offset fallback [m]")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--forest", action="store_true", help="Use the reduced forest contact prototype")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames <= 0 or args.substeps <= 0 or args.max_depth < 0.0:
        parser.error("Require positive frames/substeps and nonnegative depth threshold")
    if args.forest and args.mode != "reduced":
        parser.error("--forest requires reduced mode")
    args.outer_substeps = 2
    args.layout = "single_world"
    args.source_contact_offsets = True
    args.mesh_cylinders = True
    args.source_damping = False
    example = Example(ViewerNull(), args)
    audit = ContactAudit(example.model)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    output = args.output or Path(f"/tmp/colibri_{args.mode}_{args.body_count}_{time.time_ns()}.json")
    timings = []
    peak_depth = 0.0
    peak_linear = 0.0
    peak_angular = 0.0
    status = "passed"
    failure = None
    metric = None
    frame = 0
    start = time.perf_counter()
    try:
        for frame in range(args.frames + 1):
            if frame:
                tick = time.perf_counter()
                example.step()
                wp.synchronize_device(example.model.device)
                elapsed = time.perf_counter() - tick
                if frame > args.warmup_frames:
                    timings.append(elapsed)
            metric = audit.check(example.state_0)
            peak_depth = max(peak_depth, metric["depth_m"])
            qd = example.state_0.body_qd.numpy()
            peak_linear = max(peak_linear, float(np.max(np.linalg.norm(qd[:, :3], axis=1))))
            peak_angular = max(peak_angular, float(np.max(np.linalg.norm(qd[:, 3:], axis=1))))
            example.test_post_step()
            if metric["depth_m"] > args.max_depth:
                raise AssertionError(f"Fresh penetration {metric['depth_m']:.7f} m exceeds {args.max_depth} m")
            if frame % 30 == 0:
                print(json.dumps({"frame": frame, **metric}), flush=True)
    except AssertionError as error:
        status = "failed"
        failure = str(error)
        c = audit.contacts
        count = min(int(c.rigid_contact_count.numpy()[0]), audit.capacity)
        np.savez(
            output.with_suffix(".npz"),
            q=example.state_0.body_q.numpy(),
            qd=example.state_0.body_qd.numpy(),
            initial_q=example.initial_q,
            body_labels=example.model.body_label,
            shape_labels=example.model.shape_label,
            shape0=c.rigid_contact_shape0.numpy()[:count],
            shape1=c.rigid_contact_shape1.numpy()[:count],
            point0=c.rigid_contact_point0.numpy()[:count],
            point1=c.rigid_contact_point1.numpy()[:count],
            normal=c.rigid_contact_normal.numpy()[:count],
            separation=audit.separation.numpy()[:count],
            config=json.dumps(config),
            frame=frame,
        )
    report = {
        "status": status,
        "failure": failure,
        "config": config,
        "completed_frames": frame,
        "peak_depth_m": peak_depth,
        "peak_linear_speed_m_s": peak_linear,
        "peak_angular_speed_rad_s": peak_angular,
        "last_contact_check": metric,
        "timed_frames": len(timings),
        "physics_mean_ms": 1000.0 * float(np.mean(timings)) if timings else None,
        "physics_p95_ms": 1000.0 * float(np.percentile(timings, 95)) if timings else None,
        "total_seconds_with_checks": time.perf_counter() - start,
        "timing_scope": "Physics and collision detection; synchronized frames; excludes validation and rendering",
        "acceptance_scope": "Finite/bounded motion, joint error, collision filters and fresh contact depth; conservation tested separately",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"REPORT {output}", flush=True)
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
