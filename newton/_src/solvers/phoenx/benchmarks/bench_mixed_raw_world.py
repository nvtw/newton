# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Raw PhoenX mixed joint/contact benchmark.

This benchmark intentionally bypasses :class:`SolverPhoenX` and drives
``PhoenXWorld`` directly. It exists to measure the multi-world fast-tail path
when active joint rows and active contact rows share the same coloured stream,
without the higher-level state wrapper adding noise.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_mixed_raw_world
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _bench
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode


def _filled_float(value: float, count: int, device: wp.context.Devicelike) -> wp.array:
    return wp.array(np.full(count, value, dtype=np.float32), dtype=wp.float32, device=device)


def _filled_int(value: int, count: int, device: wp.context.Devicelike) -> wp.array:
    return wp.array(np.full(count, value, dtype=np.int32), dtype=wp.int32, device=device)


def _build_one_world_builder(pairs_per_world: int = 1) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 5.0e3
    builder.default_shape_cfg.mu = 0.7
    builder.add_ground_plane()

    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, gap=0.01)
    pairs = max(1, int(pairs_per_world))
    side = int(np.ceil(np.sqrt(pairs)))
    for pair in range(pairs):
        row = pair // side
        col = pair - row * side
        dx = 1.35 * float(col)
        dy = 0.45 * float(row)

        anchor = builder.add_body(
            xform=wp.transform(p=wp.vec3(dx, dy, 1.0), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(anchor, radius=0.08, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

        pendulum = builder.add_body(
            xform=wp.transform(p=wp.vec3(dx + 0.25, dy, 0.78), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(pendulum, radius=0.10, cfg=sphere_cfg)

        box = builder.add_body(
            xform=wp.transform(p=wp.vec3(dx + 0.75, dy, 0.10), q=wp.quat_identity()),
        )
        builder.add_shape_box(
            box,
            hx=0.10,
            hy=0.10,
            hz=0.10,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, gap=0.01),
        )

    return builder


def _force_tpw(world: PhoenXWorld, tpw) -> None:
    if tpw == "auto":
        world._tpw_auto = True
    else:
        world._tpw_auto = False
        world._tpw_choice.assign([int(tpw)])


def build_raw_mixed_world(
    *,
    num_worlds: int,
    substeps: int,
    solver_iterations: int,
    prepare_refresh_stride: int,
    pairs_per_world: int = 1,
    tpw="auto",
    device: wp.context.Devicelike = None,
) -> tuple[PhoenXWorld, Callable[[], None]]:
    if device is None:
        device = wp.get_device("cuda:0")

    pairs_per_world = max(1, int(pairs_per_world))
    one_world = _build_one_world_builder(pairs_per_world)
    builder = newton.ModelBuilder()
    builder.replicate(one_world, num_worlds)
    model = builder.finalize(skip_shape_contact_pairs=True)
    collision_pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = collision_pipeline.contacts()
    rigid_contact_max = int(contacts.rigid_contact_point0.shape[0])

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    model.body_q.assign(state.body_q)

    num_phx_bodies = int(model.body_count) + 1
    bodies = body_container_zeros(num_phx_bodies, device=device)
    wp.copy(
        bodies.orientation,
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_phx_bodies, 1)).astype(np.float32),
            dtype=wp.quatf,
            device=device,
        ),
    )
    wp.launch(
        init_phoenx_bodies_kernel,
        dim=model.body_count,
        inputs=[
            model.body_q,
            state.body_qd,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
        ],
        outputs=[
            bodies.position,
            bodies.orientation,
            bodies.velocity,
            bodies.angular_velocity,
            bodies.inverse_mass,
            bodies.inverse_inertia,
            bodies.inverse_inertia_world,
            bodies.motion_type,
            bodies.body_com,
        ],
        device=device,
    )

    shape_body_np = model.shape_body.numpy()
    shape_body = wp.array(np.where(shape_body_np < 0, 0, shape_body_np + 1), dtype=wp.int32, device=device)

    num_joints = num_worlds * pairs_per_world
    constraints = PhoenXWorld.make_constraint_container(num_joints=num_joints, device=device)
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=1,
        prepare_refresh_stride=prepare_refresh_stride,
        gravity=(0.0, 0.0, -9.81),
        rigid_contact_max=rigid_contact_max,
        num_joints=num_joints,
        num_worlds=num_worlds,
        step_layout="multi_world",
        device=device,
    )
    _force_tpw(world, tpw)

    # The replicated builder lays out each pair as anchor, pendulum, contact box.
    body1 = np.empty(num_joints, dtype=np.int32)
    body2 = np.empty(num_joints, dtype=np.int32)
    anchor1 = np.zeros((num_joints, 3), dtype=np.float32)
    anchor2 = np.zeros((num_joints, 3), dtype=np.float32)
    side = int(np.ceil(np.sqrt(pairs_per_world)))
    for w in range(num_worlds):
        world_body_base = 3 * pairs_per_world * w
        for pair in range(pairs_per_world):
            row = pair // side
            col = pair - row * side
            dx = 1.35 * float(col)
            dy = 0.45 * float(row)
            jid = w * pairs_per_world + pair
            body_base = world_body_base + 3 * pair
            body1[jid] = body_base + 1
            body2[jid] = body_base + 2
            anchor1[jid] = (dx, dy, 1.0)

    world.initialize_actuated_double_ball_socket_joints(
        body1=wp.array(body1, dtype=wp.int32, device=device),
        body2=wp.array(body2, dtype=wp.int32, device=device),
        anchor1=wp.array(anchor1, dtype=wp.vec3f, device=device),
        anchor2=wp.array(anchor2, dtype=wp.vec3f, device=device),
        hertz=_filled_float(float(DEFAULT_HERTZ_LINEAR), num_joints, device),
        damping_ratio=_filled_float(float(DEFAULT_DAMPING_RATIO), num_joints, device),
        joint_mode=_filled_int(int(JointMode.BALL_SOCKET), num_joints, device),
        drive_mode=_filled_int(int(DriveMode.OFF), num_joints, device),
        target=_filled_float(0.0, num_joints, device),
        target_velocity=_filled_float(0.0, num_joints, device),
        max_force_drive=_filled_float(0.0, num_joints, device),
        stiffness_drive=_filled_float(0.0, num_joints, device),
        damping_drive=_filled_float(0.0, num_joints, device),
        min_value=_filled_float(1.0, num_joints, device),
        max_value=_filled_float(-1.0, num_joints, device),
        hertz_limit=_filled_float(float(DEFAULT_HERTZ_LINEAR), num_joints, device),
        damping_ratio_limit=_filled_float(float(DEFAULT_DAMPING_RATIO), num_joints, device),
        stiffness_limit=_filled_float(0.0, num_joints, device),
        damping_limit=_filled_float(0.0, num_joints, device),
    )

    dt = 1.0 / 60.0

    def simulate_one_frame() -> None:
        wp.launch(
            newton_to_phoenx_kernel,
            dim=model.body_count,
            inputs=[state.body_q, state.body_qd, model.body_com],
            outputs=[bodies.position[1:], bodies.orientation[1:], bodies.velocity[1:], bodies.angular_velocity[1:]],
            device=device,
        )
        model.collide(state, contacts=contacts, collision_pipeline=collision_pipeline)
        world.step(dt=dt, contacts=contacts, shape_body=shape_body)
        wp.launch(
            phoenx_to_newton_kernel,
            dim=model.body_count,
            inputs=[
                bodies.position[1:],
                bodies.orientation[1:],
                bodies.velocity[1:],
                bodies.angular_velocity[1:],
                model.body_com,
            ],
            outputs=[state.body_q, state.body_qd],
            device=device,
        )

    return world, simulate_one_frame


def _run_case(
    *,
    num_worlds: int,
    tpw,
    substeps: int,
    solver_iterations: int,
    prepare_refresh_stride: int,
    pairs_per_world: int,
    n_runs: int,
    warmup: int,
    trials: int,
) -> tuple[float, float, int]:
    world, simulate = build_raw_mixed_world(
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        prepare_refresh_stride=prepare_refresh_stride,
        pairs_per_world=pairs_per_world,
        tpw=tpw,
    )
    for _ in range(5):
        simulate()
    wp.synchronize_device()
    chosen = int(world._tpw_choice.numpy()[0]) if tpw == "auto" else int(tpw)
    min_ms, med_ms = _bench(simulate, n_runs=n_runs, warmup=warmup, trials=trials)
    return min_ms, med_ms, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-worlds", type=int, default=512)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prepare-refresh-stride", type=int, default=1)
    parser.add_argument("--pairs-per-world", type=int, default=1)
    args = parser.parse_args()

    wp.init()
    print(f"device: {wp.get_device()}  sm_count={getattr(wp.get_device(), 'sm_count', 'N/A')}")
    print(f"\n=== raw_mixed num_worlds={args.num_worlds} n_runs={args.n_runs} trials={args.trials} ===")
    results: dict[str, tuple[float, float, int]] = {}
    for tpw in ("auto", 32, 16, 8):
        min_ms, med_ms, chosen = _run_case(
            num_worlds=args.num_worlds,
            tpw=tpw,
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
            prepare_refresh_stride=args.prepare_refresh_stride,
            pairs_per_world=args.pairs_per_world,
            n_runs=args.n_runs,
            warmup=args.warmup,
            trials=args.trials,
        )
        results[str(tpw)] = (min_ms, med_ms, chosen)
        print(
            f"  tpw={tpw!s:>5s}  chosen={chosen:>2d}  "
            f"min={min_ms:8.2f} ms  med={med_ms:8.2f} ms  "
            f"({1000.0 * min_ms / args.n_runs:7.2f} us/frame @min)"
        )

    base_min = results["32"][0]
    print(f"  (rel. to forced tpw=32 baseline of {base_min:.2f} ms)")
    for key, (min_ms, _, _) in results.items():
        if key == "32":
            continue
        speedup = base_min / min_ms if min_ms > 0.0 else float("nan")
        print(f"    tpw={key:>5s}: {speedup:.3f}x  ({100.0 * (speedup - 1.0):+.1f}%)")


if __name__ == "__main__":
    main()
