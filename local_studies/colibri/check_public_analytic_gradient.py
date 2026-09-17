# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the public Colibri CLI with read-only test-metric reporting."""

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from local_studies.colibri.drive_motion import measure
from local_studies.colibri.joint_accuracy import measure as measure_joint_accuracy
from newton.examples.kamino.example_kamino_colibri import BODY_ORDER

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", type=int, default=1200)
parser.add_argument("--body-count", type=int, default=len(BODY_ORDER))
parser.add_argument("--geometric-candidates", action="store_true")
parser.add_argument("--substeps", type=int, default=24)
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--warmup", type=int, default=30, help="Frames excluded from synchronized step timing")
parser.add_argument("--output", type=Path)
parser.add_argument("--save-contacts", action="store_true", help="Save final contact data for CPU residual audits")
parser.add_argument("--save-history", action="store_true", help="Save unique post-step poses and report joint motion")
options = parser.parse_args()
output = options.output or Path(f"/tmp/colibri_phoenx_analytic_gradient_public_{options.frames}.json")
original_run = newton.examples.run


def reported_run(example, args):
    report = {
        "frames_requested": args.num_frames,
        "body_labels": list(example.model.body_label),
        "joint_labels": list(example.model.joint_label),
        "substeps_per_collision_update": args.substeps,
        "solver_iterations": args.iterations,
        "collision_hz": 120,
        "color_group_size": example.solver.world.mass_splitting_color_group_size,
        "speculative_contact_velocity_filter": example.collision_pipeline.speculative_contact_velocity_filter,
        "checks": 0,
        "peak_depth_m": 0.0,
        "peak_anchor_m": 0.0,
        "peak_axis_rad": 0.0,
        "passed": False,
        "passed_scope": "Joint, fresh-contact, settled-support and crank-tracking checks",
    }
    original_check = example.test_post_step
    original_step = example.step
    step_times = []

    def timed_step():
        wp.synchronize_device(example.model.device)
        started = time.perf_counter()
        original_step()
        wp.synchronize_device(example.model.device)
        step_times.append(time.perf_counter() - started)

    example.step = timed_step
    model = example.model
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    xp = model.joint_X_p.numpy()
    xc = model.joint_X_c.numpy()
    types = model.joint_type.numpy()
    axes = model.joint_axis.numpy()
    starts = model.joint_qd_start.numpy()
    history = []
    velocity_history = []
    history_times = []
    base_index = list(model.body_label).index("FrameGround")
    base_initial = example.initial_q[base_index].astype(float)
    base_initial_rotation = base_initial[3:] / np.linalg.norm(base_initial[3:])
    base_metrics = {
        "scope": "Absolute motion from the initial pose, including the failing post-step pose",
        "max_horizontal_displacement_m": 0.0,
        "max_rotation_rad": 0.0,
    }
    report["base_motion"] = base_metrics

    def check():
        q = example.state_0.body_q.numpy()
        base = q[base_index].astype(float)
        delta = base[:3] - base_initial[:3]
        rotation = base[3:] / np.linalg.norm(base[3:])
        angle = float(2 * np.arccos(np.clip(abs(rotation @ base_initial_rotation), 0, 1)))
        horizontal = float(np.linalg.norm(delta[:2]))
        base_metrics["final_displacement_m"] = delta.tolist()
        base_metrics["final_rotation_rad"] = angle
        base_metrics["max_horizontal_displacement_m"] = max(base_metrics["max_horizontal_displacement_m"], horizontal)
        base_metrics["max_rotation_rad"] = max(base_metrics["max_rotation_rad"], angle)
        original_check()
        # test_final calls this check again at the same time; do not duplicate it.
        if options.save_history and (not history_times or example.sim_time != history_times[-1]):
            history.append(q.copy())
            velocity_history.append(example.state_0.body_qd.numpy().copy())
            history_times.append(example.sim_time)
        count = int(example._audit_contacts.rigid_contact_count.numpy()[0])
        if count:
            depth = max(0.0, -float(example._audit_separation.numpy()[:count].min()))
            report["peak_depth_m"] = max(report["peak_depth_m"], depth)
        for j, (a, b) in enumerate(zip(parents, children, strict=True)):
            if types[j] == int(newton.JointType.FREE):
                continue
            pa = wp.transform_identity() if a < 0 else wp.transform(wp.vec3(q[a, :3]), wp.quat(q[a, 3:]))
            pb = wp.transform(wp.vec3(q[b, :3]), wp.quat(q[b, 3:]))
            delta = np.asarray(wp.transform_point(pa, wp.vec3(xp[j, :3]))) - np.asarray(
                wp.transform_point(pb, wp.vec3(xc[j, :3]))
            )
            report["peak_anchor_m"] = max(report["peak_anchor_m"], float(np.linalg.norm(delta)))
            if types[j] == int(newton.JointType.REVOLUTE):
                axis = wp.vec3(axes[starts[j]])
                aa = np.asarray(wp.quat_rotate(pa.q * wp.quat(xp[j, 3:]), axis))
                ab = np.asarray(wp.quat_rotate(pb.q * wp.quat(xc[j, 3:]), axis))
                angle = 2 * np.arcsin(min(1.0, float(np.linalg.norm(aa - ab)) * 0.5))
                report["peak_axis_rad"] = max(report["peak_axis_rad"], float(angle))
        report["checks"] += 1
        if report["checks"] % 300 == 0:
            print("GRADIENT_CHECK", report, flush=True)

    example.test_post_step = check
    try:
        result = original_run(example, args)
        report["passed"] = True
        return result
    except Exception as error:
        report["failure"] = repr(error)
        raise
    finally:
        report["sim_time"] = example.sim_time
        measured = np.asarray(step_times[max(0, options.warmup) :])
        report["timing"] = {
            "scope": "Synchronized public example step; excludes rendering and physical audits",
            "warmup_frames": max(0, options.warmup),
            "measured_frames": len(measured),
            "mean_ms": float(measured.mean() * 1000) if len(measured) else None,
            "p95_ms": float(np.percentile(measured, 95) * 1000) if len(measured) else None,
            "fps": float(1 / measured.mean()) if len(measured) else None,
        }
        arrays = {
            "q": example.state_0.body_q.numpy(),
            "qd": example.state_0.body_qd.numpy(),
            "initial_q": example.initial_q.copy(),
        }
        displacement = np.linalg.norm(arrays["q"][:, :3] - arrays["initial_q"][:, :3], axis=1)
        farthest = int(np.argmax(displacement))
        report["final_motion_bounds"] = {
            "all_finite": bool(np.isfinite(arrays["q"]).all() and np.isfinite(arrays["qd"]).all()),
            "max_displacement_from_start_m": float(displacement[farthest]),
            "max_displacement_body": model.body_label[farthest],
            "max_linear_speed_m_s": float(np.linalg.norm(arrays["qd"][:, :3], axis=1).max()),
            "max_angular_speed_rad_s": float(np.linalg.norm(arrays["qd"][:, 3:], axis=1).max()),
        }
        if options.save_contacts:
            world = example.solver.world
            views = world._contact_views
            arrays.update(
                contact_count=views.rigid_contact_count.numpy(),
                contact_shape0=views.rigid_contact_shape0.numpy(),
                contact_shape1=views.rigid_contact_shape1.numpy(),
                contact_point0=views.rigid_contact_point0.numpy(),
                contact_point1=views.rigid_contact_point1.numpy(),
                contact_normal=views.rigid_contact_normal.numpy(),
                contact_margin0=views.rigid_contact_margin0.numpy(),
                contact_margin1=views.rigid_contact_margin1.numpy(),
                contact_impulses=world._contact_container.impulses.numpy(),
                contact_derived=world._contact_container.derived.numpy(),
                contact_anchors=world._contact_container.lambdas.numpy(),
                contact_headers=world._contact_cols.data.numpy(),
                contact_column_count=world._ingest_scratch.num_contact_columns.numpy(),
                shape_body=model.shape_body.numpy(),
                shape_labels=np.asarray(model.shape_label),
                body_labels=np.asarray(model.body_label),
                body_com=model.body_com.numpy(),
                body_mass=model.body_mass.numpy(),
                body_inertia=model.body_inertia.numpy(),
            )
            strong = world._temporal_contact_state
            if strong is None:
                strong = getattr(world._contact_cols, "strong", None)
            if strong is not None and hasattr(strong, "current"):
                arrays.update(
                    patch_membership=strong.current.point_patch.numpy(),
                    patch_member_next=strong.current.point_next.numpy(),
                    patch_point_count=strong.current.patch_count.numpy(),
                    patch_group_first=strong.current.group_first.numpy(),
                    patch_anchor_count=strong.anchors.count.numpy(),
                    patch_history_source=strong.anchors.source.numpy(),
                    patch_broken=strong.anchors.broken.numpy(),
                    patch_impulses=strong.impulse.numpy(),
                    patch_keys=strong.keys.numpy(),
                )
        if not report["passed"]:
            # The first assertion can precede worse errors on later joints.
            report["failed_pose_joint_accuracy"] = measure_joint_accuracy(arrays["q"], model.body_label)
        if options.save_history:
            labels = list(model.body_label)
            arrays.update(
                q_history=np.asarray(history),
                qd_history=np.asarray(velocity_history),
                history_times=np.asarray(history_times),
                labels=labels,
                body_mass=model.body_mass.numpy(),
                body_inertia=model.body_inertia.numpy(),
                body_com=model.body_com.numpy(),
            )
            if len(history) > 1 and options.body_count == len(BODY_ORDER):
                intervals = np.diff(history_times)
                np.testing.assert_allclose(intervals, intervals[0], rtol=1e-9, atol=1e-12)
                report["motion"] = measure(history, labels, fps=1.0 / intervals[0])
        output.write_text(json.dumps(report, indent=2))
        np.savez(output.with_suffix(".npz"), **arrays)
        print("GRADIENT_FINAL", json.dumps(report), flush=True)


newton.examples.run = reported_run
sys.argv = [
    "phoenx_colibri",
    "--viewer",
    "null",
    "--num-frames",
    str(options.frames),
    "--substeps",
    str(options.substeps),
    "--body-count",
    str(options.body_count),
    "--iterations",
    str(options.iterations),
    "--test",
]
if options.geometric_candidates:
    sys.argv.append("--geometric-candidates")
runpy.run_module("newton.examples.phoenx.example_phoenx_colibri", run_name="__main__")
