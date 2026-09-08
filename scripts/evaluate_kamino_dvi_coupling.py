# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate DVI block-coupling methods on the Unitree G1 workload.

This script holds contact, friction, warm-starting, model, timestep, and
projected-sweep count fixed while comparing these coupling methods:

``schur``
    Eliminate bilateral rows and solve the reduced unilateral problem, as in
    the candidate implementation.

``alternating``
    Run projected unilateral sweeps against the current full-system residual,
    then reuse the factored bilateral block for a forward/back substitution.
    This reproduces the coupling schedule before the Schur-complement change.

Run from the repository root::

    uv run scripts/evaluate_kamino_dvi_coupling.py --method schur
    uv run scripts/evaluate_kamino_dvi_coupling.py --method alternating

The JSON result includes formulation-independent terminal DVI residuals,
contact-state markers, post-settle G1 drift, and steady contact-rich runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode

wp.config.log_level = wp.LOG_WARNING


class G1CouplingWorkload:
    """Headless G1 example with parameterized DVI work."""

    def __init__(self, method: str, iterations: int, sweeps: int, omega: float, initial_tilt_deg: float):
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        g1 = newton.ModelBuilder()
        newton.solvers.SolverKamino.register_custom_attributes(g1)
        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            friction=1.0e-5,
        )
        g1.default_shape_cfg.ke = 1.0e3
        g1.default_shape_cfg.kd = 2.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75
        asset_path = newton.utils.download_asset("unitree_g1")
        g1.add_usd(
            str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.2)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        root_joint = next(
            index
            for index, (joint_type, parent) in enumerate(zip(g1.joint_type, g1.joint_parent, strict=True))
            if joint_type == int(newton.JointType.FREE) and parent == -1
        )
        root_coord = g1.joint_q_start[root_joint]
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(initial_tilt_deg))
        g1.joint_q[root_coord + 3 : root_coord + 7] = [tilt[index] for index in range(4)]
        for dof in range(6, g1.joint_dof_count):
            g1.joint_target_ke[dof] = 500.0
            g1.joint_target_kd[dof] = 10.0
            g1.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.add_world(g1)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 2.0e2
        builder.add_ground_plane()
        self.model = builder.finalize()

        config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver="dvi",
            sparse_dynamics=True,
            sparse_jacobian=True,
        )
        config.dvi.max_alternating_iterations = iterations
        config.dvi.inequality_sweeps_per_iteration = sweeps
        config.dvi.omega = omega
        config.dvi.use_schur_complement = method == "schur"
        # Schur folds the bilateral response into every unilateral update.
        # Alternating instead refreshes the direct bilateral solution after
        # every projected block, matching the historical interval-one method.
        config.dvi.bilateral_solve_interval = 1 if method == "alternating" else iterations
        config.dvi.bilateral_solver_type = "LLTBRCM"
        self.solver = newton.solvers.SolverKamino(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        with wp.ScopedCapture() as capture:
            self._simulate_frame()
        self.graph = capture.graph

    def _simulate_frame(self) -> None:
        """Advance one 60 Hz display frame."""
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        """Replay one captured display frame."""
        wp.capture_launch(self.graph)


def _ground_contact_groups(labels: list[str], body_pairs: np.ndarray) -> set[str]:
    """Return left/right hand and foot groups touching the ground."""
    groups = set()
    for body_a, body_b in body_pairs:
        body_id = int(body_b if body_a == -1 else body_a if body_b == -1 else -1)
        if body_id < 0:
            continue
        label = labels[body_id].lower()
        side = "left" if "left" in label else "right" if "right" in label else "unknown"
        if any(token in label for token in ("hand", "palm", "wrist")):
            groups.add(f"{side}_hand")
        if any(token in label for token in ("foot", "ankle")):
            groups.add(f"{side}_foot")
    return groups


def _summary(values: list[float]) -> dict[str, float]:
    """Return robust summary statistics for one metric series."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def run(args: argparse.Namespace) -> dict:
    """Run one coupling configuration and return JSON-compatible metrics."""
    workload = G1CouplingWorkload(args.method, args.iterations, args.sweeps, args.omega, args.initial_tilt_deg)
    labels = [label.rsplit("/", 1)[-1] for label in workload.model.body_label]
    pelvis = next((index for index, label in enumerate(labels) if "pelvis" in label.lower()), 0)
    solver_fd = workload.solver._solver_kamino.solver_fd
    contacts_kamino = workload.solver._contacts_kamino

    positions = []
    residuals = defaultdict(list)
    convergence = []
    reported_sweeps = []
    any_hand_foot_frame = None
    all_limbs_frame = None
    contact_groups_seen = set()
    contact_counts = []

    for frame in range(args.frames):
        workload.step()
        status = solver_fd.data.status.numpy()[0]
        positions.append(workload.state_0.body_q.numpy()[pelvis, :3].astype(np.float64))
        for name in ("r_p", "r_d", "r_c", "r_b"):
            residuals[name].append(float(status[name]))
        convergence.append(int(status["converged"]))
        reported_sweeps.append(int(status["iterations"]))

        contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
        contact_counts.append(contact_count)
        groups = _ground_contact_groups(labels, contacts_kamino.bid_AB.numpy()[:contact_count])
        contact_groups_seen.update(groups)
        has_hand = any(group.endswith("_hand") for group in groups)
        has_foot = any(group.endswith("_foot") for group in groups)
        if any_hand_foot_frame is None and has_hand and has_foot:
            any_hand_foot_frame = frame
        required_groups = {"left_hand", "right_hand", "left_foot", "right_foot"}
        if all_limbs_frame is None and required_groups <= groups:
            all_limbs_frame = frame

    positions_np = np.asarray(positions)
    onset = any_hand_foot_frame if any_hand_foot_frame is not None else max(0, args.frames // 2)
    settled = min(args.frames - 1, onset + args.settle_frames)
    settled_xy = positions_np[settled:, :2]
    elapsed = np.arange(settled_xy.shape[0], dtype=np.float64) * workload.frame_dt
    velocity_xy = np.zeros(2)
    if settled_xy.shape[0] >= 2:
        velocity_xy = np.array([np.polyfit(elapsed, settled_xy[:, axis], 1)[0] for axis in range(2)])

    # Time an already fallen, contact-rich continuation without per-frame host
    # copies. The preceding rollout is the warmup and establishes the state.
    timing_ms_per_frame = []
    for _ in range(args.timing_trials):
        wp.synchronize_device()
        start = time.perf_counter()
        for _ in range(args.timing_frames):
            workload.step()
        wp.synchronize_device()
        timing_ms_per_frame.append((time.perf_counter() - start) * 1000.0 / args.timing_frames)

    if any_hand_foot_frame is None and not args.allow_missing_hand_foot_contact:
        raise RuntimeError(
            "G1 never reached simultaneous hand-and-foot ground contact; "
            f"observed groups were {sorted(contact_groups_seen)}"
        )

    drift_xy = positions_np[-1, :2] - positions_np[settled, :2]
    return {
        "method": args.method,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "device": str(workload.model.device),
        "device_name": workload.model.device.name,
        "iterations": args.iterations,
        "sweeps_per_iteration": args.sweeps,
        "omega": args.omega,
        "initial_tilt_deg": args.initial_tilt_deg,
        "frames": args.frames,
        "simulated_seconds": args.frames * workload.frame_dt,
        "body_labels_in_ground_contact": sorted(contact_groups_seen),
        "first_any_hand_and_foot_frame": any_hand_foot_frame,
        "first_all_hands_and_feet_frame": all_limbs_frame,
        "settled_measurement_frame": settled,
        "pelvis_final_xyz_m": positions_np[-1].tolist(),
        "pelvis_post_settle_drift_xy_m": drift_xy.tolist(),
        "pelvis_post_settle_drift_norm_m": float(np.linalg.norm(drift_xy)),
        "pelvis_post_settle_lateral_drift_m": float(abs(drift_xy[1])),
        "pelvis_post_settle_velocity_xy_m_per_s": velocity_xy.tolist(),
        "pelvis_post_settle_lateral_speed_m_per_s": float(abs(velocity_xy[1])),
        "converged_frame_fraction": float(np.mean(convergence)),
        "reported_sweeps_mean": float(np.mean(reported_sweeps)),
        "active_contacts": _summary(contact_counts),
        "residuals": {name: _summary(values) for name, values in residuals.items()},
        "post_hand_foot_contact_residuals": (
            {name: _summary(values[any_hand_foot_frame:]) for name, values in residuals.items()}
            if any_hand_foot_frame is not None
            else None
        ),
        "timing_trials_ms_per_frame": timing_ms_per_frame,
        "timing_median_ms_per_frame": float(np.median(timing_ms_per_frame)),
        "timing_min_ms_per_frame": float(np.min(timing_ms_per_frame)),
        "timing_median_ms_per_substep": float(np.median(timing_ms_per_frame)) / workload.sim_substeps,
    }


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("schur", "alternating", "both"), default="both")
    parser.add_argument("--device", help="Warp CUDA device, for example cuda:0")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--omega", type=float, default=1.2)
    parser.add_argument("--initial-tilt-deg", type=float, default=20.0)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--timing-frames", type=int, default=120)
    parser.add_argument("--timing-trials", type=int, default=5)
    parser.add_argument("--allow-missing-hand-foot-contact", action="store_true")
    parser.add_argument("--output", type=Path, help="Write clean JSON to this path")
    return parser.parse_args()


def main() -> None:
    """Run one method or emit a paired comparison with useful ratios."""
    args = parse_args()
    if args.device:
        wp.set_device(args.device)
    if not wp.get_device().is_cuda:
        raise RuntimeError("The G1 coupling benchmark requires a CUDA device.")
    for name in ("iterations", "sweeps", "frames", "settle_frames", "timing_frames", "timing_trials"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.omega <= 2.0:
        raise ValueError("--omega must lie in (0, 2]")
    if args.method != "both":
        serialized = json.dumps(run(args), indent=2)
        if args.output:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return

    results = {}
    for method in ("schur", "alternating"):
        method_args = argparse.Namespace(**vars(args))
        method_args.method = method
        results[method] = run(method_args)
    schur = results["schur"]
    alternating = results["alternating"]
    results["ratios_alternating_over_schur"] = {
        "post_settle_drift_norm": (
            alternating["pelvis_post_settle_drift_norm_m"] / schur["pelvis_post_settle_drift_norm_m"]
        ),
        "post_settle_lateral_drift": (
            alternating["pelvis_post_settle_lateral_drift_m"] / schur["pelvis_post_settle_lateral_drift_m"]
        ),
        "post_settle_lateral_speed": (
            alternating["pelvis_post_settle_lateral_speed_m_per_s"] / schur["pelvis_post_settle_lateral_speed_m_per_s"]
        ),
        "timing_median": (alternating["timing_median_ms_per_frame"] / schur["timing_median_ms_per_frame"]),
    }
    serialized = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
