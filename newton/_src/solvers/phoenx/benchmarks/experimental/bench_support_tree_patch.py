# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a full-patch support-tree contact preconditioner prototype.

The prototype assumes every edge in a grounded support tree is a full-rank,
sticking contact patch. One thread computes the exact patch wrenches for one
tree by accumulating the spatial momentum change of each subtree. A
minimum-norm force distribution provides a conservative acceptance test: the
tree is eligible only when every reconstructed point force lies inside its
Coulomb cone. Rejected trees would fall back to ordinary PGS in a production
implementation.

This benchmark measures only the solve and acceptance kernel. It does not
include dynamic contact-forest construction or apply results to PhoenX state.

Example::

    uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.experimental.bench_support_tree_patch
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time

import numpy as np
import warp as wp


@functools.cache
def _make_support_tree_kernel(body_count: int, half_x: float, half_y: float):
    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        mass: wp.array[wp.float32],
        inertia: wp.array[wp.vec3f],
        position: wp.array[wp.vec3f],
        velocity: wp.array[wp.vec3f],
        angular_velocity: wp.array[wp.vec3f],
        friction: wp.float32,
        eligible: wp.array[wp.int32],
        wrench: wp.array[wp.spatial_vector],
    ):
        world = wp.tid()
        base = world * wp.int32(body_count)
        linear = wp.vec3f(0.0, 0.0, 0.0)
        angular_origin = wp.vec3f(0.0, 0.0, 0.0)
        accepted = wp.int32(1)
        for reverse_index in range(body_count):
            body = wp.int32(body_count - 1 - reverse_index)
            index = base + body
            body_linear = -mass[index] * velocity[index]
            body_angular = -wp.cw_mul(inertia[index], angular_velocity[index]) + wp.cross(position[index], body_linear)
            linear += body_linear
            angular_origin += body_angular
            patch_point = wp.vec3f(0.0, 0.0, wp.float32(body))
            torque = angular_origin - wp.cross(patch_point, linear)
            wrench[index] = wp.spatial_vector(linear, torque)

            dual_torque = wp.vec3f(
                torque[0] / wp.float32(4.0 * half_y * half_y),
                torque[1] / wp.float32(4.0 * half_x * half_x),
                torque[2] / wp.float32(4.0 * (half_x * half_x + half_y * half_y)),
            )
            for corner in range(4):
                x = wp.float32(-half_x if (corner & 1) == 0 else half_x)
                y = wp.float32(-half_y if (corner & 2) == 0 else half_y)
                force = wp.float32(0.25) * linear + wp.cross(dual_torque, wp.vec3f(x, y, 0.0))
                friction_limit = friction * force[2]
                if force[2] < wp.float32(0.0) or force[0] * force[0] + force[1] * force[1] > (
                    friction_limit * friction_limit
                ):
                    accepted = wp.int32(0)
        eligible[world] = accepted

    return kernel


def _reference_trees(
    mass: np.ndarray,
    inertia: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    friction: float,
    half_x: float,
    half_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    world_count, body_count = mass.shape
    wrench = np.zeros((world_count, body_count, 6), dtype=np.float64)
    eligible = np.ones(world_count, dtype=np.int32)
    for world in range(world_count):
        linear = np.zeros(3, dtype=np.float64)
        angular_origin = np.zeros(3, dtype=np.float64)
        for body in reversed(range(body_count)):
            body_linear = -float(mass[world, body]) * velocity[world, body].astype(np.float64)
            body_angular = -inertia[world, body].astype(np.float64) * angular_velocity[world, body].astype(
                np.float64
            ) + np.cross(position[world, body], body_linear)
            linear += body_linear
            angular_origin += body_angular
            patch_point = np.array((0.0, 0.0, float(body)))
            torque = angular_origin - np.cross(patch_point, linear)
            wrench[world, body, :3] = linear
            wrench[world, body, 3:] = torque
            dual_torque = np.array(
                (
                    torque[0] / (4.0 * half_y * half_y),
                    torque[1] / (4.0 * half_x * half_x),
                    torque[2] / (4.0 * (half_x * half_x + half_y * half_y)),
                )
            )
            for corner in range(4):
                x = -half_x if (corner & 1) == 0 else half_x
                y = -half_y if (corner & 2) == 0 else half_y
                force = 0.25 * linear + np.cross(dual_torque, np.array((x, y, 0.0)))
                if force[2] < 0.0 or float(force[:2] @ force[:2]) > (friction * force[2]) ** 2:
                    eligible[world] = 0
    return eligible, wrench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--bodies", type=int, default=5)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if not wp.is_cuda_available():
        raise RuntimeError("the support-tree prototype requires CUDA graph capture")
    if args.worlds < 1 or args.bodies < 1 or args.replays < 1 or args.batches < 1:
        parser.error("worlds, bodies, replays, and batches must be positive")
    if args.friction < 0.0:
        parser.error("friction must be non-negative")

    rng = np.random.default_rng(args.seed)
    mass_np = (10.0 ** rng.uniform(-2.0, 4.0, size=(args.worlds, args.bodies))).astype(np.float32)
    position_np = np.zeros((args.worlds, args.bodies, 3), dtype=np.float32)
    position_np[:, :, 2] = np.arange(args.bodies, dtype=np.float32) + 0.5
    velocity_np = (0.001 * rng.standard_normal((args.worlds, args.bodies, 3))).astype(np.float32)
    velocity_np[:, :, 2] -= np.float32(9.81 / 300.0)
    angular_np = (0.001 * rng.standard_normal((args.worlds, args.bodies, 3))).astype(np.float32)
    inertia_np = np.repeat((mass_np / 6.0)[..., None], 3, axis=2).astype(np.float32)

    device = wp.get_device("cuda:0")
    mass = wp.array(mass_np.reshape(-1), dtype=wp.float32, device=device)
    inertia = wp.array(inertia_np.reshape(-1, 3), dtype=wp.vec3f, device=device)
    position = wp.array(position_np.reshape(-1, 3), dtype=wp.vec3f, device=device)
    velocity = wp.array(velocity_np.reshape(-1, 3), dtype=wp.vec3f, device=device)
    angular_velocity = wp.array(angular_np.reshape(-1, 3), dtype=wp.vec3f, device=device)
    eligible = wp.zeros(args.worlds, dtype=wp.int32, device=device)
    wrench = wp.zeros(args.worlds * args.bodies, dtype=wp.spatial_vector, device=device)
    kernel = _make_support_tree_kernel(args.bodies, 0.4, 0.4)

    def launch() -> None:
        wp.launch(
            kernel,
            dim=args.worlds,
            inputs=[mass, inertia, position, velocity, angular_velocity, wp.float32(args.friction), eligible, wrench],
            device=device,
        )

    launch()
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as capture:
        launch()
    graph = capture.graph
    wp.capture_launch(graph)
    wp.synchronize_device(device)

    times_us = []
    for _ in range(args.batches):
        start = time.perf_counter()
        for _ in range(args.replays):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        times_us.append((time.perf_counter() - start) * 1.0e6 / args.replays)

    eligible_np = eligible.numpy()
    wrench_np = wrench.numpy().reshape(args.worlds, args.bodies, 6)
    if not np.isfinite(wrench_np).all():
        raise RuntimeError("support-tree prototype produced a non-finite wrench")
    validated_worlds = min(64, args.worlds)
    expected_eligible, expected_wrench = _reference_trees(
        mass_np[:validated_worlds],
        inertia_np[:validated_worlds],
        position_np[:validated_worlds],
        velocity_np[:validated_worlds],
        angular_np[:validated_worlds],
        float(args.friction),
        0.4,
        0.4,
    )
    np.testing.assert_array_equal(eligible_np[:validated_worlds], expected_eligible)
    np.testing.assert_allclose(
        wrench_np[:validated_worlds],
        expected_wrench,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    payload = {
        "schema": "phoenx_support_tree_patch_v1",
        "device": device.name,
        "worlds": args.worlds,
        "bodies_per_tree": args.bodies,
        "mass_range": [0.01, 10000.0],
        "eligible": int(np.sum(eligible_np)),
        "eligible_fraction": float(np.mean(eligible_np)),
        "validated_worlds": validated_worlds,
        "median_graph_replay_us": float(np.median(times_us)),
        "batch_graph_replay_us": times_us,
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
