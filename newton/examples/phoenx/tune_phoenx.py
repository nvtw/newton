# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded, quality-gated PhoenX tuning for user-supplied scenes.

Run with: uv run --extra examples -m newton.examples.phoenx.tune_phoenx module:make_scene
The factory returns TuningScene. See AUTOTUNE.md in this directory for details.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import warp as wp

import newton


@wp.kernel
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
    body0 = shape_body[shape0[i]]
    body1 = shape_body[shape1[i]]
    p0 = point0[i]
    p1 = point1[i]
    if body0 >= 0:
        p0 = wp.transform_point(body_q[body0], p0)
    if body1 >= 0:
        p1 = wp.transform_point(body_q[body1], p1)
    separation[i] = wp.dot(p1 - p0, normal[i]) - margin0[i] - margin1[i]


@dataclass
class TuningScene:
    """A complete, finalized model and its unchanged physical configuration.

    collision_updates is the number of collision refreshes per displayed frame.
    before_update may apply time-dependent forces or controls.
    extra_metrics returns nonnegative error measures, with smaller being better.
    """

    model: newton.Model
    frame_dt: float
    collision_updates: int = 1
    solver_options: dict = field(default_factory=dict)
    pipeline_factory: Callable | None = None
    before_update: Callable | None = None
    extra_metrics: Callable | None = None


@dataclass(frozen=True)
class Settings:
    layout: str
    substeps: int
    iterations: int
    scheduler: str = "auto"
    mass_splitting: bool = False
    max_colors: int = 12
    threads_per_world: int | str = "auto"


def _may_toggle_mass(options: dict) -> bool:
    """Avoid trials that violate a fixed solver feature's mass-splitting requirement."""
    return not (
        options.get("solver_scheme") == "tgs"
        or options.get("contact_friction_model") == "patch"
        or options.get("joint_mode") in ("maximal_projected", "maximal_articulated", "hybrid", "reduced")
        or options.get("mass_splitting_color_group_size", 0)
        or options.get("joint_refinement_iterations", 0)
        or options.get("direct_joint_projection_passes", 1) > 1
    )


def candidate_settings(base: Settings, world_count: int, mode: str = "default", *, toggle_mass=True) -> list[Settings]:
    """Return an ordered, bounded search with no cartesian-product sweep."""
    if mode not in ("fast", "default", "thorough"):
        raise ValueError(f"Unknown tuning mode: {mode}")
    layouts = ["single_world", "multi_world"] if world_count > 1 else ["single_world"]
    layouts = [base.layout, *(layout for layout in layouts if layout != base.layout)]
    candidates = [base]
    candidates.extend(
        replace(base, layout=layout)
        for layout in layouts
        if layout != base.layout and not (base.layout == "auto" and layout == "multi_world" and world_count > 1)
    )
    if toggle_mass:
        candidates.append(replace(base, mass_splitting=not base.mass_splitting))
    color_candidates = []
    if base.mass_splitting or toggle_mass:
        colored = replace(base, mass_splitting=True)
        if mode != "fast":
            color_candidates.extend(
                (
                    replace(colored, max_colors=max(1, base.max_colors // 2)),
                    replace(colored, max_colors=min(32, base.max_colors + 4)),
                )
            )
        if mode == "thorough":
            color_candidates.append(replace(colored, max_colors=max(1, base.max_colors // 4)))
    if world_count > 1 and mode != "fast":
        candidates.extend(
            replace(base, layout="multi_world", scheduler=scheduler) for scheduler in ("fast_tail", "block_world")
        )
    if world_count > 1 and mode == "thorough":
        candidates.extend(replace(base, layout="multi_world", threads_per_world=threads) for threads in (8, 16, 32))
    cheaper = [
        (max(1, base.substeps // 2), base.iterations),
        (base.substeps, max(1, base.iterations // 2)),
    ]
    if mode != "fast":
        cheaper.append((max(1, base.substeps // 2), max(1, base.iterations // 2)))
    if mode == "thorough":
        cheaper.extend(
            [
                (max(1, base.substeps // 4), base.iterations),
                (base.substeps, max(1, base.iterations // 4)),
                (max(1, base.substeps // 2), base.iterations * 2),
                (base.substeps * 2, max(1, base.iterations // 2)),
            ]
        )
    for substeps, iterations in cheaper:
        candidates.append(replace(base, substeps=substeps, iterations=iterations))
    candidates.extend(color_candidates)
    return list(dict.fromkeys(candidates))


def _quat_rotate(q, v):
    xyz = q[:3]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + q[3] * v)


def _quat_mul(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - np.dot(a[:3], b[:3])]


def _anchor(body, local):
    if body is None:
        return local[:3], local[3:]
    return body[:3] + _quat_rotate(body[3:], local[:3]), _quat_mul(body[3:], local[3:])


def _joint_data(model):
    return tuple(
        getattr(model, name).numpy()
        for name in (
            "joint_parent",
            "joint_child",
            "joint_X_p",
            "joint_X_c",
            "joint_type",
            "joint_axis",
            "joint_qd_start",
            "joint_dof_dim",
            "joint_enabled",
        )
    )


def _joint_errors(model, poses, data=None):
    if not model.joint_count:
        return 0.0, 0.0, set()
    parent, child, xp, xc, types, axes, starts, dims, enabled = _joint_data(model) if data is None else data
    linear = angular = 0.0
    unscored = set()
    for j in range(model.joint_count):
        if not enabled[j]:
            continue
        kind = int(types[j])
        if kind in (int(newton.JointType.FREE), int(newton.JointType.DISTANCE), int(newton.JointType.ROD)):
            unscored.add(newton.JointType(kind).name)
            continue
        p, qp = _anchor(poses[parent[j]] if parent[j] >= 0 else None, xp[j])
        c, qc = _anchor(poses[child[j]] if child[j] >= 0 else None, xc[j])
        delta = c - p
        allowed_linear, allowed_angular = (int(x) for x in dims[j])
        first = int(starts[j])
        local_linear = axes[first : first + allowed_linear]
        local_angular = axes[first + allowed_linear : first + allowed_linear + allowed_angular]
        world_axes = [_quat_rotate(qp, axis) for axis in local_linear]
        for axis in world_axes[:allowed_linear]:
            delta -= np.dot(delta, axis) * axis
        linear = max(linear, float(np.linalg.norm(delta)))
        if kind == int(newton.JointType.BALL):
            continue
        relative = _quat_mul(np.r_[-qp[:3], qp[3]], qc)
        if relative[3] < 0:
            relative = -relative
        angle = 2.0 * math.atan2(float(np.linalg.norm(relative[:3])), float(relative[3]))
        if kind == int(newton.JointType.REVOLUTE):
            u, v = _quat_rotate(qp, local_angular[0]), _quat_rotate(qc, local_angular[0])
            angle = math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v)))
        elif kind == int(newton.JointType.D6) and allowed_angular:
            rotation = relative[:3] * (angle / max(float(np.linalg.norm(relative[:3])), 1.0e-12))
            for axis in local_angular:
                rotation -= np.dot(rotation, axis) * axis
            angle = float(np.linalg.norm(rotation))
        angular = max(angular, angle)
    return linear, angular, unscored


class _Probe:
    def __init__(self, scene):
        self.scene = scene
        self.joint_data = _joint_data(scene.model) if scene.model.joint_count else None
        self.pipeline = (
            scene.pipeline_factory()
            if scene.pipeline_factory
            else newton.CollisionPipeline(scene.model, contact_matching="sticky")
        )
        self.contacts = self.pipeline.contacts()
        self.separation = wp.empty(self.contacts.rigid_contact_max, dtype=wp.float32, device=scene.model.device)

    def measure(self, state):
        model = self.scene.model
        poses = state.body_q.numpy()
        velocities = state.body_qd.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            return {"joint_m": math.inf, "joint_rad": math.inf, "penetration_m": math.inf}, {"nonfinite"}
        linear, angular, unscored = _joint_errors(model, poses, self.joint_data)
        self.pipeline.collide(state, self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count > self.contacts.rigid_contact_max:
            penetration = math.inf
            unscored.add("contact_overflow")
        elif count:
            wp.launch(
                _contact_separation,
                dim=count,
                inputs=[
                    state.body_q,
                    model.shape_body,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.contacts.rigid_contact_point0,
                    self.contacts.rigid_contact_point1,
                    self.contacts.rigid_contact_normal,
                    self.contacts.rigid_contact_margin0,
                    self.contacts.rigid_contact_margin1,
                    self.separation,
                ],
            )
            penetration = max(0.0, -float(self.separation.numpy()[:count].min()))
        else:
            penetration = 0.0
        metrics = {"joint_m": linear, "joint_rad": angular, "penetration_m": penetration}
        if self.scene.extra_metrics:
            metrics.update(self.scene.extra_metrics(state, self.contacts))
        return metrics, unscored


def run_trial(scene, settings, frames, sample_stride):
    """Measure one setting from the same model initial state; exclude setup time."""
    start = time.perf_counter()
    options = dict(scene.solver_options)
    options.update(
        substeps=settings.substeps,
        solver_iterations=settings.iterations,
        step_layout=settings.layout,
        multi_world_scheduler=settings.scheduler,
        mass_splitting=settings.mass_splitting,
        max_colored_partitions=settings.max_colors,
        threads_per_world=settings.threads_per_world,
    )
    pipeline = (
        scene.pipeline_factory()
        if scene.pipeline_factory
        else newton.CollisionPipeline(scene.model, contact_matching="sticky")
    )
    solver = newton.solvers.SolverPhoenX(scene.model, collision_pipeline=pipeline, **options)
    contacts = pipeline.contacts()
    control = scene.model.control()
    states = [scene.model.state(), scene.model.state()]
    probe = _Probe(scene)
    wp.synchronize()
    setup_s = time.perf_counter() - start
    state_index = 0
    metrics = {}
    unscored = set()
    elapsed = 0.0
    timed_frames = 0
    warmup_frames = min(5, max(1, frames // 10))
    update_dt = scene.frame_dt / scene.collision_updates
    for frame in range(frames):
        tick = time.perf_counter()
        for update in range(scene.collision_updates):
            current, next_state = states[state_index], states[1 - state_index]
            current.clear_forces()
            if scene.before_update:
                scene.before_update(current, control, frame * scene.collision_updates + update, update_dt)
            pipeline.collide(current, contacts)
            solver.step(current, next_state, control, contacts, update_dt)
            state_index = 1 - state_index
        wp.synchronize()
        if frame >= warmup_frames:
            elapsed += time.perf_counter() - tick
            timed_frames += 1
        if frame % sample_stride == 0 or frame == frames - 1:
            measured, flags = probe.measure(states[state_index])
            unscored.update(flags)
            for name, value in measured.items():
                metrics[name] = max(metrics.get(name, 0.0), float(value))
    return {
        "settings": settings,
        "fps": timed_frames / elapsed,
        "metrics": metrics,
        "setup_s": setup_s,
        "trial_s": time.perf_counter() - start,
        "unscored": sorted(unscored),
    }


def _passes(metrics, limits):
    return all(math.isfinite(metrics.get(name, math.inf)) and metrics[name] <= cap for name, cap in limits.items())


def tune(scene, *, mode="default", frames=None, sample_stride=10, slack=0.1, limits=None, time_budget_s=None):
    """Tune against a reference run and optional user-supplied absolute caps."""
    if mode not in ("fast", "default", "thorough"):
        raise ValueError(f"Unknown tuning mode: {mode}")
    if frames is None:
        frames = {"fast": 60, "default": 120, "thorough": 240}[mode]
    if time_budget_s is None:
        time_budget_s = {"fast": 20.0, "default": 90.0, "thorough": 300.0}[mode]
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")
    if frames < 2 or sample_stride < 1 or scene.frame_dt <= 0 or scene.collision_updates < 1 or slack < 0:
        raise ValueError(
            "frames >= 2, sample_stride >= 1, frame_dt > 0, collision_updates >= 1, and slack >= 0 are required"
        )
    options = scene.solver_options
    base = Settings(
        options.get("step_layout", "auto"),
        options.get("substeps", 1),
        options.get("solver_iterations", 8),
        options.get("multi_world_scheduler", "auto"),
        options.get("mass_splitting", False),
        options.get("max_colored_partitions", 12),
        options.get("threads_per_world", "auto"),
    )
    if options.get("velocity_iterations", 1) != 1:
        raise ValueError("This tuner requires one fixed velocity iteration")
    rows = []
    candidates = candidate_settings(base, max(1, scene.model.world_count), mode, toggle_mass=_may_toggle_mass(options))
    tuning_start = time.perf_counter()
    for settings in candidates:
        if rows:
            baseline = rows[0]
            work_ratio = (settings.substeps * settings.iterations) / (base.substeps * base.iterations)
            predicted_s = baseline["setup_s"] + frames / baseline["fps"] * (0.35 + 0.65 * work_ratio) + 0.5
            if time.perf_counter() - tuning_start + predicted_s > time_budget_s:
                break
        trial_start = time.perf_counter()
        try:
            result = run_trial(scene, settings, frames, sample_stride)
        except Exception as exc:
            if not rows:
                raise RuntimeError("Reference PhoenX setting failed; fix the scene factory before tuning") from exc
            rows.append(
                {
                    "settings": settings,
                    "fps": 0.0,
                    "metrics": dict.fromkeys(limits, math.inf),
                    "setup_s": 0.0,
                    "trial_s": time.perf_counter() - trial_start,
                    "unscored": [],
                    "passes": False,
                    "error": str(exc),
                }
            )
            continue
        rows.append(result)
        if len(rows) == 1:
            reference = result["metrics"]
            relative = {
                name: value * (1.0 + slack) + (1.0e-4 if name.endswith("_m") else math.radians(0.1))
                for name, value in reference.items()
            }
            limits = {name: min(cap, (limits or {}).get(name, math.inf)) for name, cap in relative.items()}
            limits.update({name: cap for name, cap in (limits or {}).items() if name not in limits})
        result["passes"] = _passes(result["metrics"], limits)
        if len(rows) == 1 and not result["passes"]:
            candidates[1:1] = [
                replace(base, substeps=base.substeps * 2),
                replace(base, iterations=base.iterations * 2),
            ]
    winner = max((row for row in rows if row["passes"]), key=lambda row: row["fps"], default=None)
    return rows, limits, winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory", help="Python module:function returning TuningScene")
    parser.add_argument("--mode", choices=("fast", "default", "thorough"), default="default")
    parser.add_argument("--time-budget-s", type=float, help="Override the mode's soft wall-time budget")
    parser.add_argument("--frames", type=int, help="Override the mode's simulated frames per candidate")
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--relative-slack", type=float, default=0.1)
    parser.add_argument("--max-joint-mm", type=float)
    parser.add_argument("--max-joint-deg", type=float)
    parser.add_argument("--max-penetration-mm", type=float)
    parser.add_argument("--json", type=str, help="Write machine-readable results to this path")
    args = parser.parse_args()
    module, name = args.factory.split(":", 1)
    scene = getattr(importlib.import_module(module), name)()
    absolute = {}
    if args.max_joint_mm is not None:
        absolute["joint_m"] = args.max_joint_mm / 1000.0
    if args.max_joint_deg is not None:
        absolute["joint_rad"] = math.radians(args.max_joint_deg)
    if args.max_penetration_mm is not None:
        absolute["penetration_m"] = args.max_penetration_mm / 1000.0
    rows, limits, winner = tune(
        scene,
        mode=args.mode,
        frames=args.frames,
        sample_stride=args.sample_stride,
        slack=args.relative_slack,
        limits=absolute,
        time_budget_s=args.time_budget_s,
    )
    print(
        "layout           scheduler   threads  split   colors  substeps  iterations    FPS  joint mm  joint deg  penetration mm  pass"
    )
    for row in rows:
        s, m = row["settings"], row["metrics"]
        print(
            f"{s.layout:16} {s.scheduler:11} {s.threads_per_world!s:8} {s.mass_splitting!s:7}"
            f" {str(s.max_colors) if s.mass_splitting else '-':8}"
            f"{s.substeps:8d}{s.iterations:12d}{row['fps']:7.1f}{m['joint_m'] * 1000:10.3f}"
            f"{math.degrees(m['joint_rad']):11.3f}{m['penetration_m'] * 1000:16.3f}  {row['passes']}"
        )
    print("Limits:", limits)
    print("Recommended:", winner["settings"] if winner else "none passed")
    print(f"Evaluated {len(rows)} candidates in {sum(row['trial_s'] for row in rows):.1f} s")
    for row in rows:
        if "error" in row:
            print(f"Unsupported or failed {row['settings']}: {row['error']}")
    if any(row["unscored"] for row in rows):
        print("Unscored joint types or errors:", sorted(set().union(*(row["unscored"] for row in rows))))
    if args.json:
        json_rows = [
            {
                **row,
                "settings": vars(row["settings"]),
                "metrics": {name: value if math.isfinite(value) else None for name, value in row["metrics"].items()},
            }
            for row in rows
        ]
        with open(args.json, "w") as output:
            json.dump(
                {
                    "limits": {name: value if math.isfinite(value) else None for name, value in limits.items()},
                    "winner": str(winner["settings"]) if winner else None,
                    "rows": json_rows,
                },
                output,
                indent=2,
                allow_nan=False,
            )


if __name__ == "__main__":
    main()
