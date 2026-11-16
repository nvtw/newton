# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure resting-contact jitter from simple supports to a loaded container.

These asset-free cylinders are controlled proxies, not the authored bottle USD.
The default ladder contains fixed/free single-cylinder controls, four and twelve
cylinders on a free plate, and twelve cylinders inside a free box shell. All
free supports default to 0.42 kg with geometry-derived inertia; each cylinder weighs
0.57 kg. No sleeping, added damping, attachments, or pose resets are used.

Report motion in the translating/rotating support frame after settling. Runtime
includes collision detection and diagnostic readbacks, excludes construction,
and is NOT rendered factory FPS or a CUDA-graph throughput benchmark. Record
the same arguments and device when comparing solver changes. A quieter result
still requires independent sliding, separation, force, and penetration checks.

Run the complete three-second ladder::

    uv run --no-sync -m newton._src.solvers.phoenx.benchmarks.experimental.bench_container_jitter \
        --output /tmp/container-jitter.json

Compare a longer run or another solver schedule::

    uv run --no-sync -m newton._src.solvers.phoenx.benchmarks.experimental.bench_container_jitter \
        --case box_12 --duration 10 --settle 2 --substeps 8 --iterations 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverPhoenX

CASES = {
    "fixed_1": (1, False, False),
    "plate_1": (1, True, False),
    "plate_4": (4, True, False),
    "plate_12": (12, True, False),
    "box_12": (12, True, True),
}


def _rotate(q, vectors):
    return vectors + 2 * np.cross(q[:3], np.cross(q[:3], vectors) + q[3] * vectors)


def _build_scene(name, args):
    count, dynamic, walls = CASES[name]
    builder = newton.ModelBuilder()
    length, width, height, thickness = 0.286, 0.216, 0.261, 0.0032
    volume = length * width * thickness
    if walls:
        volume += 2 * thickness * (width + length - 2 * thickness) * (height - thickness)
    material = {"mu": 0.4, "mu_static": 0.45, "gap": args.gap, "margin": 0, "restitution": 0}
    cfg = builder.ShapeConfig(density=0, **material)
    support_cfg = builder.ShapeConfig(density=args.support_mass / volume if dynamic else 0, **material)
    support = builder.add_body(xform=wp.transform((0, 0, 0.001), wp.quat_identity())) if dynamic else -1
    builder.add_shape_box(
        support,
        xform=wp.transform((0, 0, thickness / 2 + (0 if dynamic else 0.001)), wp.quat_identity()),
        hx=length / 2,
        hy=width / 2,
        hz=thickness / 2,
        cfg=support_cfg,
    )
    if walls:
        for axis in (0, 1):
            for sign in (-1, 1):
                center = [0, 0, (height + thickness) / 2]
                half = [length / 2, width / 2, (height - thickness) / 2]
                center[axis] = sign * (half[axis] - thickness / 2)
                half[axis] = thickness / 2
                if axis == 1:
                    half[0] -= thickness
                builder.add_shape_box(
                    support,
                    xform=wp.transform(center, wp.quat_identity()),
                    hx=half[0],
                    hy=half[1],
                    hz=half[2],
                    cfg=support_cfg,
                )
    mass, radius, cylinder_height = 0.57, 0.0304, 0.25
    inertia = np.diag([mass * (3 * radius**2 + cylinder_height**2) / 12] * 2 + [mass * radius**2 / 2])
    nx, ny = {1: (1, 1), 4: (2, 2), 12: (4, 3)}[count]
    contents = []
    for x in range(nx):
        for y in range(ny):
            position = (
                (x - (nx - 1) / 2) * 0.0687,
                (y - (ny - 1) / 2) * 0.0687,
                0.001 + thickness + cylinder_height / 2 + 0.0001,
            )
            body = builder.add_body(
                xform=wp.transform(position, wp.quat_identity()), mass=mass, inertia=wp.mat33(inertia)
            )
            builder.add_shape_cylinder(body, radius=radius, half_height=cylinder_height / 2, cfg=cfg)
            contents.append(body)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=args.device)
    if dynamic:
        np.testing.assert_allclose(model.body_mass.numpy()[support], args.support_mass, rtol=1e-5)
    return model, support, contents


def run_case(name, args):
    model, support, contents = _build_scene(name, args)
    state, control = model.state(), model.control()
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching=args.matching,
        rigid_contact_max=4096,
        speculative_contact_gap_max=args.speculative_extension,
    )
    contacts = pipeline.contacts()
    solver = SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        substeps=args.solver_substeps,
        solver_iterations=args.iterations,
        velocity_iterations=1,
        contact_hertz=args.contact_hertz,
        step_layout="single_world",
        enable_body_pair_grouping=True,
        mass_splitting=False,
        sleeping_velocity_threshold=0,
    )
    com = model.body_com.numpy()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    shape_body = model.shape_body.numpy()
    # The builder adds the world floor last, separately from the fixed base.
    floor_shape = model.shape_count - 1
    direct_floor_forces = []
    positions, speeds, angular_speeds, counts = [], [], [], []
    minimum_floor_clearance = float("inf")
    minimum_support_clearance = float("inf")
    outside_footprint = 0
    dt = 1 / (60 * args.substeps)
    steps, settle_steps = round(args.duration / dt), round(args.settle / dt)
    wp.synchronize_device(model.device)
    start = time.perf_counter()
    for step in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts, dt=dt)
        solver.step(state, state, control, contacts, dt, observables=observables)
        if step < settle_steps:
            continue
        q, v = state.body_q.numpy(), state.body_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError(f"{name}: nonfinite state at step {step}")
        if support >= 0:
            rotation = q[support, 3:]
            origin = q[support, :3] + _rotate(rotation, com[support])
            linear, angular = v[support, :3], v[support, 3:]
        else:
            rotation, origin = np.array([0, 0, 0, 1]), np.array([0, 0, 0.001])
            linear = angular = np.zeros(3)
        # Cylinder origins are their centers of mass.
        offset = q[contents, :3] - origin
        relative_velocity = v[contents, :3] - linear - np.cross(angular, offset)
        inverse = rotation * np.array([-1, -1, -1, 1])
        local = _rotate(inverse, offset)
        positions.append(local)
        # Measure against the base's top face in its own frame, including tilt.
        axes = np.asarray([_rotate(quat, np.array([0.0, 0.0, 1.0])) for quat in q[contents, 3:]])
        relative_axes = _rotate(inverse, axes)
        axis_normal = np.clip(relative_axes[:, 2], -1, 1)
        support_extent = 0.125 * np.abs(axis_normal) + 0.0304 * np.sqrt(np.maximum(0, 1 - axis_normal**2))
        support_com_z = com[support, 2] if support >= 0 else 0.0
        support_clearance = local[:, 2] + support_com_z - 0.0032 - support_extent
        minimum_support_clearance = min(minimum_support_clearance, float(support_clearance.min()))
        axis_z = 1 - 2 * np.sum(q[contents, 3:5] ** 2, axis=1)
        vertical_extent = 0.125 * np.abs(axis_z) + 0.0304 * np.sqrt(np.maximum(0, 1 - axis_z**2))
        minimum_floor_clearance = min(minimum_floor_clearance, float(np.min(q[contents, 2] - vertical_extent)))
        outside_footprint = max(
            outside_footprint, int(np.count_nonzero(np.any(np.abs(local[:, :2]) > (0.143, 0.108), axis=1)))
        )
        speeds.append(np.sum(relative_velocity**2, axis=1))
        angular_speeds.append(np.sum((v[contents, 3:] - angular) ** 2, axis=1))
        count = int(contacts.rigid_contact_count.numpy()[0])
        counts.append(count)
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        direct_floor = ((shape0 == floor_shape) & np.isin(shape_body[shape1], contents)) | (
            (shape1 == floor_shape) & np.isin(shape_body[shape0], contents)
        )
        forces = observables.contact_f.numpy()[:count, :3]
        direct_floor_forces.append(float(np.linalg.norm(forces[direct_floor], axis=1).sum()))
    wp.synchronize_device(model.device)
    elapsed = time.perf_counter() - start
    positions = np.asarray(positions)
    contact = solver.world._contact_container
    return {
        "case": name,
        "bodies": model.body_count,
        "shapes": model.shape_count,
        "support_mass_kg": float(model.body_mass.numpy()[support]) if support >= 0 else 0.0,
        "relative_speed_rms_m_s": float(np.sqrt(np.mean(speeds))),
        "relative_angular_speed_rms_rad_s": float(np.sqrt(np.mean(angular_speeds))),
        "relative_excursion_m": float(np.linalg.norm(positions - positions[0], axis=2).max()),
        "contacts_mean": float(np.mean(counts)),
        "contacts_min": int(np.min(counts)),
        "contacts_max": int(np.max(counts)),
        "measured_samples": len(counts),
        "minimum_contents_floor_clearance_m": minimum_floor_clearance,
        "minimum_contents_support_clearance_m": minimum_support_clearance,
        "direct_contents_floor_force_magnitude_mean_N": float(np.mean(direct_floor_forces)),
        "contents_centers_outside_support_max": outside_footprint,
        "contact_soft_coefficients": {
            "capped": int(contact.soft_capped),
            "bias_dt": float(contact.soft_bias_dt),
            "mass": float(contact.soft_mass_coeff),
            "impulse": float(contact.soft_impulse_coeff),
        },
        "diagnostic_wall_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=tuple(CASES), action="append", dest="cases")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--settle", type=float, default=1)
    parser.add_argument("--substeps", type=int, default=4, help="Collision passes per 60 Hz frame.")
    parser.add_argument("--solver-substeps", type=int, default=1, help="Physics substeps per collision pass.")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--contact-hertz", type=float, default=80)
    parser.add_argument("--support-mass", type=float, default=0.42, help="Mass of each free support [kg].")
    parser.add_argument("--gap", type=float, default=0.002)
    parser.add_argument("--speculative-extension", type=float, default=0.005)
    parser.add_argument("--matching", choices=("sticky", "latest", "disabled"), default="sticky")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.substeps, args.solver_substeps, args.iterations) < 1 or not 0 <= args.settle < args.duration:
        parser.error("positive substeps/iterations and 0 <= settle < duration are required")
    if args.gap < 0 or args.speculative_extension < 0 or args.contact_hertz <= 0:
        parser.error("gaps must be nonnegative and contact frequency must be positive")
    if not np.isfinite(args.support_mass) or args.support_mass <= 0:
        parser.error("support mass must be finite and positive")
    if round(args.duration * 60 * args.substeps) <= round(args.settle * 60 * args.substeps):
        parser.error("measurement window must contain at least one step")
    wp.init()
    with wp.ScopedDevice(args.device):
        results = [run_case(case, args) for case in (args.cases or CASES)]
    settings = {key: value for key, value in vars(args).items() if key != "output"}
    report = {"settings": settings, "results": results}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
