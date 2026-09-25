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
    observables returns motion quantities averaged over the second half of a
    trial; observable_tolerances limits their difference from the reference.
    in_place and initialize_state reproduce examples that step one state buffer
    or derive body poses from joint coordinates before the first update.
    """

    model: newton.Model
    frame_dt: float
    collision_updates: int = 1
    solver_options: dict = field(default_factory=dict)
    pipeline_factory: Callable | None = None
    before_update: Callable | None = None
    extra_metrics: Callable | None = None
    observables: Callable | None = None
    observable_tolerances: dict = field(default_factory=dict)
    in_place: bool = False
    initialize_state: Callable | None = None


@dataclass(frozen=True)
class Settings:
    layout: str
    substeps: int
    iterations: int
    scheduler: str = "auto"
    mass_splitting: bool = False
    max_colors: int = 12
    threads_per_world: int | str = "auto"
    color_group_size: int = 0
    splitting_batch_size: int = 8
    prepare_refresh_stride: int | str = "auto"
    parallel_contact_prepare: bool = False
    contact_chunk_size: int = 0


def _may_toggle_mass(options: dict) -> bool:
    """Avoid trials that violate a fixed solver feature's mass-splitting requirement."""
    return not (
        options.get("solver_scheme") == "tgs"
        or options.get("contact_friction_model") == "patch"
        or options.get("joint_mode") in ("maximal_projected", "maximal_articulated", "hybrid", "reduced")
        or options.get("joint_refinement_iterations", 0)
        or options.get("direct_joint_projection_passes", 1) > 1
        or (
            options.get("joint_mode") == "maximal_direct"
            and options.get("contact_chunk_size", 0) > 0
            and not options.get("enable_body_pair_grouping", False)
        )
    )


def _neighbor_sizes(value: int, *, minimum: int = 1, maximum: int = 32) -> tuple[int, ...]:
    """Try half/current/double near an authored GPU group or batch size."""
    return tuple(sorted({max(minimum, value // 2), value, min(maximum, value * 2)}))


def candidate_settings(
    base: Settings,
    world_count: int,
    mode: str = "default",
    *,
    toggle_mass=True,
    allow_ungrouped=True,
    allow_stride_two=True,
    allow_multi_layout=True,
    min_iterations=1,
) -> list[Settings]:
    """Start at authored settings, then try nearby scheduling and work budgets.

    Large vectorized workloads skip global single-world coloring; half/double
    size changes probe GPU occupancy without a cartesian-product sweep.
    """
    if mode not in ("fast", "default", "thorough"):
        raise ValueError(f"Unknown tuning mode: {mode}")
    layouts = (
        ["single_world"]
        if world_count == 1
        else ["single_world", "multi_world"]
        if world_count <= 16
        else ["multi_world"]
    )
    if not allow_multi_layout:
        layouts = [base.layout]
    layouts = [base.layout, *(layout for layout in layouts if layout != base.layout)]
    candidates = [base]
    candidates.extend(
        replace(base, layout=layout)
        for layout in layouts
        if layout != base.layout and not (base.layout == "auto" and layout == "multi_world" and world_count > 1)
    )
    if toggle_mass:
        candidates.append(
            replace(base, mass_splitting=False, color_group_size=0)
            if base.mass_splitting
            else replace(base, mass_splitting=True)
        )
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
    if world_count > 1 and mode != "fast" and allow_multi_layout:
        candidates.extend(
            replace(base, layout="multi_world", scheduler=scheduler) for scheduler in ("fast_tail", "block_world")
        )
    if world_count > 1 and mode == "thorough" and allow_multi_layout:
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
    if mode == "thorough":
        if base.mass_splitting:
            candidates.extend(
                replace(base, splitting_batch_size=size) for size in _neighbor_sizes(base.splitting_batch_size)
            )
            if base.color_group_size:
                group_sizes = _neighbor_sizes(base.color_group_size)
                if allow_ungrouped:
                    group_sizes = (0, *group_sizes)
                candidates.extend(replace(base, color_group_size=size) for size in group_sizes)
        candidates.append(replace(base, prepare_refresh_stride=1))
        if base.substeps >= 4 and allow_stride_two:
            candidates.append(replace(base, prepare_refresh_stride=2))
        candidates.append(replace(base, parallel_contact_prepare=not base.parallel_contact_prepare))
        candidates.append(replace(base, contact_chunk_size=0 if base.contact_chunk_size else 64))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate.iterations >= min_iterations))


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
        self.pipeline.collide(state, self.contacts, dt=self.scene.frame_dt / self.scene.collision_updates)
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
        mass_splitting_color_group_size=settings.color_group_size,
        mass_splitting_batch_size=settings.splitting_batch_size,
        prepare_refresh_stride=settings.prepare_refresh_stride,
        parallel_contact_prepare=settings.parallel_contact_prepare,
        contact_chunk_size=settings.contact_chunk_size,
    )
    pipeline = (
        scene.pipeline_factory()
        if scene.pipeline_factory
        else newton.CollisionPipeline(scene.model, contact_matching="sticky")
    )
    solver = newton.solvers.SolverPhoenX(scene.model, collision_pipeline=pipeline, **options)
    contacts = pipeline.contacts()
    control = scene.model.control()
    states = [scene.model.state()] if scene.in_place else [scene.model.state(), scene.model.state()]
    if scene.initialize_state:
        scene.initialize_state(states[0])
    probe = _Probe(scene)
    wp.synchronize()
    setup_s = time.perf_counter() - start
    state_index = 0
    metrics = {}
    observable_samples = {}
    unscored = set()
    elapsed = 0.0
    timed_frames = 0
    warmup_frames = min(5, max(1, frames // 10))
    update_dt = scene.frame_dt / scene.collision_updates
    for frame in range(frames):
        tick = time.perf_counter()
        for update in range(scene.collision_updates):
            current = states[state_index]
            next_state = current if scene.in_place else states[1 - state_index]
            current.clear_forces()
            if scene.before_update:
                scene.before_update(current, control, frame * scene.collision_updates + update, update_dt)
            pipeline.collide(current, contacts, dt=update_dt)
            solver.step(current, next_state, control, contacts, update_dt)
            if not scene.in_place:
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
            if scene.observables and frame >= frames // 2:
                for name, value in scene.observables(states[state_index]).items():
                    observable_samples.setdefault(name, []).append(float(value))
    return {
        "settings": settings,
        "fps": timed_frames / elapsed,
        "metrics": metrics,
        "observables": {name: float(np.mean(values)) for name, values in observable_samples.items()},
        "setup_s": setup_s,
        "trial_s": time.perf_counter() - start,
        "unscored": sorted(unscored),
    }


def _passes(metrics, limits):
    return all(math.isfinite(metrics.get(name, math.inf)) and metrics[name] <= cap for name, cap in limits.items())


def tune(
    scene, *, mode="default", frames=None, sample_stride=10, slack=0.1, limits=None, time_budget_s=None, min_gain=0.05
):
    """Tune against a reference run and optional user-supplied absolute caps."""
    if mode not in ("fast", "default", "thorough"):
        raise ValueError(f"Unknown tuning mode: {mode}")
    if frames is None:
        frames = {"fast": 60, "default": 120, "thorough": 240}[mode]
    if time_budget_s is None:
        time_budget_s = {"fast": 20.0, "default": 90.0, "thorough": 300.0}[mode]
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")
    if (
        frames < 2
        or sample_stride < 1
        or scene.frame_dt <= 0
        or scene.collision_updates < 1
        or slack < 0
        or min_gain < 0
    ):
        raise ValueError(
            "frames >= 2, sample_stride >= 1, frame_dt > 0, collision_updates >= 1, slack >= 0, "
            "and min_gain >= 0 are required"
        )
    options = scene.solver_options
    observable_tolerances = getattr(scene, "observable_tolerances", {})
    if any(value < 0 or not math.isfinite(value) for value in observable_tolerances.values()):
        raise ValueError("observable tolerances must be finite and nonnegative")
    base = Settings(
        options.get("step_layout", "auto"),
        options.get("substeps", 1),
        options.get("solver_iterations", 8),
        options.get("multi_world_scheduler", "auto"),
        options.get("mass_splitting", False),
        options.get("max_colored_partitions", 12),
        options.get("threads_per_world", "auto"),
        options.get("mass_splitting_color_group_size", 0),
        options.get("mass_splitting_batch_size", 8),
        options.get("prepare_refresh_stride", "auto"),
        options.get("parallel_contact_prepare", False),
        options.get("contact_chunk_size", 0),
    )
    rows = []
    candidates = candidate_settings(
        base,
        max(1, scene.model.world_count),
        mode,
        toggle_mass=_may_toggle_mass(options),
        allow_ungrouped=not (
            options.get("solver_scheme") == "tgs"
            or options.get("joint_refinement_iterations", 0)
            or options.get("direct_joint_projection_passes", 1) > 1
        ),
        allow_stride_two=not base.mass_splitting and options.get("solver_scheme") != "tgs",
        allow_multi_layout=not (
            base.color_group_size
            or options.get("solver_scheme") == "tgs"
            or options.get("joint_mode") == "maximal_pgs"
            or options.get("joint_refinement_iterations", 0)
            or options.get("direct_joint_projection_passes", 1) > 1
        ),
        min_iterations=options.get("direct_joint_projection_passes", 1),
    )
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
        reference_observables = rows[0].get("observables", {}) if rows else result.get("observables", {})
        for name in observable_tolerances:
            if name not in reference_observables:
                raise ValueError(f"Reference trial did not report observable {name!r}")
            current = result.get("observables", {}).get(name, math.inf)
            result["metrics"][f"motion_delta:{name}"] = abs(current - reference_observables[name])
        rows.append(result)
        if len(rows) == 1:
            reference = result["metrics"]
            near_zero_floor = {"joint_m": 1.0e-4, "joint_rad": math.radians(0.1), "penetration_m": 1.0e-4}
            relative = {
                name: max(value * (1.0 + slack), near_zero_floor.get(name, 0.0))
                for name, value in reference.items()
                if not name.startswith("motion_delta:")
            }
            limits = {name: min(cap, (limits or {}).get(name, math.inf)) for name, cap in relative.items()}
            limits.update({f"motion_delta:{name}": tolerance for name, tolerance in observable_tolerances.items()})
            limits.update({name: cap for name, cap in (limits or {}).items() if name not in limits})
        result["passes"] = _passes(result["metrics"], limits)
        if len(rows) == 1 and not result["passes"]:
            candidates[1:1] = [
                replace(base, substeps=base.substeps * 2),
                replace(base, iterations=base.iterations * 2),
            ]
    winner = max((row for row in rows if row["passes"]), key=lambda row: row["fps"], default=None)
    if winner and rows[0]["passes"] and winner["fps"] < rows[0]["fps"] * (1.0 + min_gain):
        winner = rows[0]
    return rows, limits, winner


def format_comparison(reference, recommended):
    """Compare the authored setting with the selected passing trial."""
    width = max(
        23,
        *(len(name) + 2 for name in (*reference["metrics"], *reference.get("observables", {}))),
    )
    lines = ["Reference vs recommended (simulation only)", f"{'Metric':{width}}Reference  Recommended"]
    for label, key, scale in (
        ("FPS", "fps", 1.0),
        ("Max joint error (mm)", "joint_m", 1000.0),
        ("Max joint error (deg)", "joint_rad", 180.0 / math.pi),
        ("Max penetration (mm)", "penetration_m", 1000.0),
    ):
        old = reference["fps"] if key == "fps" else reference["metrics"][key] * scale
        new = recommended["fps"] if key == "fps" else recommended["metrics"][key] * scale
        lines.append(f"{label:{width}}{old:9.3f}  {new:11.3f}")
    for key in sorted(reference["metrics"].keys() - {"joint_m", "joint_rad", "penetration_m"}):
        old, new = reference["metrics"][key], recommended["metrics"].get(key, math.nan)
        lines.append(f"{key:{width}}{old:9.3g}  {new:11.3g}")
    for key in sorted(reference.get("observables", {})):
        old, new = reference["observables"][key], recommended.get("observables", {}).get(key, math.nan)
        lines.append(f"{key:{width}}{old:9.3g}  {new:11.3g}")
    changes = [
        f"{name}: {old} -> {new}"
        for name, old, new in (
            (name, getattr(reference["settings"], name), getattr(recommended["settings"], name))
            for name in vars(reference["settings"])
        )
        if old != new
    ]
    lines.append("Suggested setting changes: " + (", ".join(changes) if changes else "none; keep authored settings"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory", help="Python module:function returning TuningScene")
    parser.add_argument("--mode", choices=("fast", "default", "thorough"), default="default")
    parser.add_argument("--time-budget-s", type=float, help="Override the mode's soft wall-time budget")
    parser.add_argument("--frames", type=int, help="Override the mode's simulated frames per candidate")
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--relative-slack", type=float, default=0.1)
    parser.add_argument("--min-gain", type=float, default=0.05, help="Minimum FPS gain before recommending a change")
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
        min_gain=args.min_gain,
    )
    print(
        "layout           scheduler   lanes  split  cap/grp/batch  prep  parallel  chunk  sub  iter    FPS  joint mm  joint deg  pen mm  pass"
    )
    for row in rows:
        s, m = row["settings"], row["metrics"]
        colors = f"{s.max_colors}/{s.color_group_size}/{s.splitting_batch_size}" if s.mass_splitting else "-"
        print(
            f"{s.layout:16} {s.scheduler:11} {s.threads_per_world!s:4} {s.mass_splitting!s:6} {colors:14}"
            f" {s.prepare_refresh_stride!s:5} {s.parallel_contact_prepare!s:9} {s.contact_chunk_size:6d}"
            f" {s.substeps:4d} {s.iterations:5d}{row['fps']:7.1f}{m['joint_m'] * 1000:10.3f}"
            f"{math.degrees(m['joint_rad']):11.3f}{m['penetration_m'] * 1000:8.3f}  {row['passes']}"
        )
        for name, value in row.get("observables", {}).items():
            delta_name = f"motion_delta:{name}"
            if delta_name in m:
                print(f"  {name}: mean={value:.3g}, delta={m[delta_name]:.3g}, limit={limits[delta_name]:.3g}")
            else:
                print(f"  {name}: mean={value:.3g}")
    print("Limits:", limits)
    print("Recommended:", winner["settings"] if winner else "none passed")
    if winner:
        print(format_comparison(rows[0], winner))
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
                "observables": {
                    name: value if math.isfinite(value) else None for name, value in row.get("observables", {}).items()
                },
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
