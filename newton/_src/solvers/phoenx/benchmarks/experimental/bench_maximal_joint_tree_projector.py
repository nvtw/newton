# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark an exact one-warp maximal-coordinate joint-tree projection.

The free-root/revolute prototype uses persistent topology, projects body
twists in the true kinetic-energy metric, and recovers every joint reaction.
It validates against a dense generalized-coordinate solve and runs entirely
inside CUDA graph capture.

Example:
    ``uv run --extra dev -m
    newton._src.solvers.phoenx.benchmarks.experimental.bench_maximal_joint_tree_projector
    --worlds 8192 --bodies 29 --replays 100``

"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

_SYNC_WARP = """__syncwarp();"""


@wp.func_native(_SYNC_WARP)
def sync_warp(): ...


@wp.func
def solve_spd6(a: wp.spatial_matrixf, b: wp.spatial_vectorf):
    l = wp.spatial_matrixf(0.0)
    d = wp.spatial_vectorf(0.0)
    for i in range(6):
        diagonal = a[i, i]
        for k in range(i):
            diagonal -= l[i, k] * l[i, k] * d[k]
        d[i] = diagonal
        l[i, i] = wp.float32(1.0)
        for j in range(i + 1, 6):
            value = a[j, i]
            for k in range(i):
                value -= l[j, k] * l[i, k] * d[k]
            l[j, i] = value / diagonal
    z = wp.spatial_vectorf(0.0)
    for i in range(6):
        value = b[i]
        for k in range(i):
            value -= l[i, k] * z[k]
        z[i] = value
    w = wp.spatial_vectorf(0.0)
    for i in range(6):
        w[i] = z[i] / d[i]
    x = wp.spatial_vectorf(0.0)
    for reverse_i in range(6):
        i = wp.int32(5) - reverse_i
        value = w[i]
        for k in range(i + 1, 6):
            value -= l[k, i] * x[k]
        x[i] = value
    return x


@wp.func
def inverse_spd6(a: wp.spatial_matrixf):
    result = wp.spatial_matrixf(0.0)
    for column in range(6):
        basis = wp.spatial_vectorf(0.0)
        basis[column] = wp.float32(1.0)
        solution = solve_spd6(a, basis)
        for row in range(6):
            result[row, column] = solution[row]
    return result


@wp.kernel(enable_backward=False)
def compute_diagonal_mobility(
    body_count: wp.int32,
    max_depth: wp.int32,
    depth: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    transform: wp.array2d[wp.spatial_matrixf],
    motion: wp.array2d[wp.spatial_vectorf],
    articulated: wp.array2d[wp.spatial_matrixf],
    inv_d: wp.array2d[wp.float32],
    conditional_map: wp.array2d[wp.spatial_matrixf],
    mobility: wp.array2d[wp.spatial_matrixf],
):
    tid = wp.tid()
    world = tid // wp.int32(32)
    lane = tid - world * wp.int32(32)
    identity = wp.spatial_matrixf(0.0)
    for diagonal in range(6):
        identity[diagonal, diagonal] = wp.float32(1.0)

    if lane == wp.int32(0):
        mobility[world, lane] = inverse_spd6(articulated[world, lane])
        conditional_map[world, lane] = identity
    sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and depth[lane] == current_depth:
            s = motion[world, lane]
            conditional_mobility = inv_d[world, lane] * wp.outer(s, s)
            mapping = (identity - conditional_mobility @ articulated[world, lane]) @ transform[world, lane]
            conditional_map[world, lane] = mapping
            parent_mobility = mobility[world, parent[lane]]
            mobility[world, lane] = mapping @ parent_mobility @ wp.transpose(mapping) + conditional_mobility
        sync_warp()
        current_depth += wp.int32(1)


@wp.kernel(enable_backward=False)
def apply_factored_tree_impulse(
    body_count: wp.int32,
    max_depth: wp.int32,
    depth: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    child_start: wp.array[wp.int32],
    child_index: wp.array[wp.int32],
    transform: wp.array2d[wp.spatial_matrixf],
    motion: wp.array2d[wp.spatial_vectorf],
    articulated: wp.array2d[wp.spatial_matrixf],
    inv_d: wp.array2d[wp.float32],
    mobility: wp.array2d[wp.spatial_matrixf],
    impulse: wp.array2d[wp.spatial_vectorf],
    response_bias: wp.array2d[wp.spatial_vectorf],
    parent_bias: wp.array2d[wp.spatial_vectorf],
    response_velocity: wp.array2d[wp.spatial_vectorf],
):
    tid = wp.tid()
    world = tid // wp.int32(32)
    lane = tid - world * wp.int32(32)
    if lane < body_count:
        response_bias[world, lane] = wp.spatial_vectorf(0.0)
        parent_bias[world, lane] = wp.spatial_vectorf(0.0)
        response_velocity[world, lane] = wp.spatial_vectorf(0.0)
    sync_warp()

    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and depth[lane] == current_depth:
            rhs = impulse[world, lane]
            begin = child_start[lane]
            end = child_start[lane + wp.int32(1)]
            for cursor in range(begin, end):
                rhs += parent_bias[world, child_index[cursor]]
            response_bias[world, lane] = rhs
            if lane != wp.int32(0):
                s = motion[world, lane]
                u = articulated[world, lane] @ s
                projected_rhs = rhs - inv_d[world, lane] * wp.dot(s, rhs) * u
                parent_bias[world, lane] = wp.transpose(transform[world, lane]) @ projected_rhs
        sync_warp()
        current_depth -= wp.int32(1)

    if lane == wp.int32(0):
        response_velocity[world, lane] = mobility[world, lane] @ response_bias[world, lane]
    sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and depth[lane] == current_depth:
            parent_velocity = response_velocity[world, parent[lane]]
            base = transform[world, lane] @ parent_velocity
            s = motion[world, lane]
            qd = inv_d[world, lane] * wp.dot(
                s,
                response_bias[world, lane] - articulated[world, lane] @ base,
            )
            response_velocity[world, lane] = base + qd * s
        sync_warp()
        current_depth += wp.int32(1)


@wp.kernel(enable_backward=False)
def project_revolute_trees(
    body_count: wp.int32,
    max_depth: wp.int32,
    floating_root: wp.bool,
    depth: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    child_start: wp.array[wp.int32],
    child_index: wp.array[wp.int32],
    transform: wp.array2d[wp.spatial_matrixf],
    motion: wp.array2d[wp.spatial_vectorf],
    offset: wp.array2d[wp.spatial_vectorf],
    mass: wp.array2d[wp.spatial_matrixf],
    velocity_in: wp.array2d[wp.spatial_vectorf],
    articulated: wp.array2d[wp.spatial_matrixf],
    bias: wp.array2d[wp.spatial_vectorf],
    inv_d: wp.array2d[wp.float32],
    parent_articulated: wp.array2d[wp.spatial_matrixf],
    parent_bias: wp.array2d[wp.spatial_vectorf],
    velocity_out: wp.array2d[wp.spatial_vectorf],
    reaction: wp.array2d[wp.spatial_vectorf],
):
    tid = wp.tid()
    world = tid // wp.int32(32)
    lane = tid - world * wp.int32(32)
    if lane < body_count:
        body_mass = mass[world, lane]
        articulated[world, lane] = body_mass
        bias[world, lane] = body_mass @ velocity_in[world, lane]
        parent_articulated[world, lane] = wp.spatial_matrixf(0.0)
        parent_bias[world, lane] = wp.spatial_vectorf(0.0)
        inv_d[world, lane] = wp.float32(0.0)
    sync_warp()

    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and depth[lane] == current_depth:
            a = articulated[world, lane]
            h = bias[world, lane]
            begin = child_start[lane]
            end = child_start[lane + wp.int32(1)]
            for cursor in range(begin, end):
                child = child_index[cursor]
                a += parent_articulated[world, child]
                h += parent_bias[world, child]
            articulated[world, lane] = a
            bias[world, lane] = h
            if lane != wp.int32(0):
                s = motion[world, lane]
                u = a @ s
                reciprocal_d = wp.float32(1.0) / wp.dot(s, u)
                inv_d[world, lane] = reciprocal_d
                projected = a - reciprocal_d * wp.outer(u, u)
                projected_bias = h - reciprocal_d * wp.dot(s, h) * u
                x = transform[world, lane]
                c = offset[world, lane]
                parent_articulated[world, lane] = wp.transpose(x) @ projected @ x
                parent_bias[world, lane] = wp.transpose(x) @ (projected_bias - projected @ c)
        sync_warp()
        current_depth -= wp.int32(1)

    if lane == wp.int32(0):
        if floating_root:
            velocity_out[world, lane] = solve_spd6(articulated[world, lane], bias[world, lane])
        else:
            velocity_out[world, lane] = wp.spatial_vectorf(0.0)
    sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and depth[lane] == current_depth:
            base = transform[world, lane] @ velocity_out[world, parent[lane]] + offset[world, lane]
            s = motion[world, lane]
            qd = inv_d[world, lane] * wp.dot(s, bias[world, lane] - articulated[world, lane] @ base)
            velocity_out[world, lane] = base + qd * s
        sync_warp()
        current_depth += wp.int32(1)

    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and depth[lane] == current_depth:
            impulse = mass[world, lane] @ (velocity_out[world, lane] - velocity_in[world, lane])
            begin = child_start[lane]
            end = child_start[lane + wp.int32(1)]
            for cursor in range(begin, end):
                child = child_index[cursor]
                impulse += wp.transpose(transform[world, child]) @ reaction[world, child]
            reaction[world, lane] = impulse
        sync_warp()
        current_depth -= wp.int32(1)


@wp.kernel(enable_backward=False)
def copy_velocity(src: wp.array2d[wp.spatial_vectorf], dst: wp.array2d[wp.spatial_vectorf]):
    world, body = wp.tid()
    dst[world, body] = src[world, body]


def make_tree(n: int):
    parent = np.full(n, -1, np.int32)
    # Branched humanoid-like topology: torso chain plus four limbs.
    for i in range(1, n):
        if i < 5:
            parent[i] = i - 1
        else:
            branch = (i - 5) % 4
            segment = (i - 5) // 4
            parent[i] = 1 + branch if segment == 0 else i - 4
    depth = np.zeros(n, np.int32)
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    start = np.zeros(n + 1, np.int32)
    flat = []
    for i, row in enumerate(children):
        flat.extend(row)
        start[i + 1] = len(flat)
    return parent, depth, start, np.asarray(flat, np.int32)


def dense_oracle(parent, transform, motion, mass, velocity):
    n = len(parent)
    generalized_size = 6 + n - 1
    K = [np.zeros((6, generalized_size)) for _ in range(n)]
    K[0][:, :6] = np.eye(6)
    for i in range(1, n):
        K[i][:] = transform[i] @ K[parent[i]]
        K[i][:, 6 + i - 1] = motion[i]
    K = np.vstack(K)
    M = np.zeros((6 * n, 6 * n))
    for i in range(n):
        M[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = mass[i]
    v = velocity.reshape(-1)
    return (K @ np.linalg.solve(K.T @ M @ K, K.T @ M @ v)).reshape(n, 6), K, M


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("maximal joint-tree benchmark requires a CUDA device")
    body_count = int(args.bodies)
    world_count = int(args.worlds)
    if body_count < 2 or body_count > 32:
        raise ValueError("bodies must be in [2, 32]")
    if world_count < 1 or int(args.replays) < 1 or int(args.batches) < 1:
        raise ValueError("worlds, replays, and batches must be positive")

    parent, depth, child_start, child_index = make_tree(body_count)
    rng = np.random.default_rng(args.seed)
    transforms = np.zeros((body_count, 6, 6), np.float32)
    motion = np.zeros((body_count, 6), np.float32)
    transforms[0] = np.eye(6, dtype=np.float32)
    for body in range(1, body_count):
        lever = rng.normal(0.0, 0.25, 3).astype(np.float32)
        transforms[body] = np.eye(6, dtype=np.float32)
        transforms[body, :3, 3:] = np.asarray(wp.skew(wp.vec3f(*lever)), dtype=np.float32).reshape(3, 3)
        axis = rng.normal(size=3).astype(np.float32)
        axis /= np.linalg.norm(axis)
        motion[body, :3] = np.cross(lever, axis)
        motion[body, 3:] = axis

    masses = np.zeros((world_count, body_count, 6, 6), np.float32)
    velocities = rng.normal(size=(world_count, body_count, 6)).astype(np.float32)
    for body in range(body_count):
        body_mass = np.float32(10.0 ** rng.uniform(-2.0, 2.0))
        basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        inertia = basis @ np.diag(10.0 ** rng.uniform(-2.0, 2.0, size=3)) @ basis.T
        masses[:, body, :3, :3] = body_mass * np.eye(3, dtype=np.float32)
        masses[:, body, 3:, 3:] = inertia.astype(np.float32)

    transform_all = np.broadcast_to(transforms, (world_count, body_count, 6, 6)).copy()
    motion_all = np.broadcast_to(motion, (world_count, body_count, 6)).copy()
    shape = (world_count, body_count)
    transform_wp = wp.array(transform_all, dtype=wp.spatial_matrixf, device=device)
    motion_wp = wp.array(motion_all, dtype=wp.spatial_vectorf, device=device)
    offset_wp = wp.zeros(shape, dtype=wp.spatial_vectorf, device=device)
    mass_wp = wp.array(masses, dtype=wp.spatial_matrixf, device=device)
    velocity_wp = wp.array(velocities, dtype=wp.spatial_vectorf, device=device)
    articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
    bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    inverse_d = wp.empty(shape, dtype=wp.float32, device=device)
    parent_articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
    parent_bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    velocity_out = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    reaction = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    conditional_map = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
    mobility = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
    impulses = rng.normal(size=(world_count, body_count, 6)).astype(np.float32)
    impulse_wp = wp.array(impulses, dtype=wp.spatial_vectorf, device=device)
    response_bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    response_parent_bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    response_velocity = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
    depth_wp = wp.array(depth, device=device)
    parent_wp = wp.array(parent, device=device)
    child_start_wp = wp.array(child_start, device=device)
    child_index_wp = wp.array(child_index, device=device)
    inputs = [
        body_count,
        int(depth.max()),
        True,
        depth_wp,
        parent_wp,
        child_start_wp,
        child_index_wp,
        transform_wp,
        motion_wp,
        offset_wp,
        mass_wp,
        velocity_wp,
        articulated,
        bias,
        inverse_d,
        parent_articulated,
        parent_bias,
        velocity_out,
        reaction,
    ]
    wp.launch(
        project_revolute_trees,
        dim=world_count * 32,
        block_dim=32,
        inputs=inputs,
        device=device,
    )
    mobility_inputs = [
        body_count,
        int(depth.max()),
        depth_wp,
        parent_wp,
        transform_wp,
        motion_wp,
        articulated,
        inverse_d,
        conditional_map,
        mobility,
    ]
    wp.launch(
        compute_diagonal_mobility,
        dim=world_count * 32,
        block_dim=32,
        inputs=mobility_inputs,
        device=device,
    )
    response_inputs = [
        body_count,
        int(depth.max()),
        depth_wp,
        parent_wp,
        child_start_wp,
        child_index_wp,
        transform_wp,
        motion_wp,
        articulated,
        inverse_d,
        mobility,
        impulse_wp,
        response_bias,
        response_parent_bias,
        response_velocity,
    ]
    wp.launch(
        apply_factored_tree_impulse,
        dim=world_count * 32,
        block_dim=32,
        inputs=response_inputs,
        device=device,
    )
    actual = velocity_out.numpy()[0]
    actual_mobility = mobility.numpy()[0]
    actual_conditional_map = conditional_map.numpy()[0]
    expected, jacobian, mass_stacked = dense_oracle(parent, transforms, motion, masses[0], velocities[0])
    dense_mobility = jacobian @ np.linalg.solve(jacobian.T @ mass_stacked @ jacobian, jacobian.T)
    expected_response = (dense_mobility @ impulses[0].reshape(-1)).reshape(body_count, 6)
    response_error = float(np.max(np.abs(response_velocity.numpy()[0] - expected_response)))
    diagonal_mobility_error = max(
        np.max(np.abs(actual_mobility[body] - dense_mobility[6 * body : 6 * body + 6, 6 * body : 6 * body + 6]))
        for body in range(body_count)
    )

    def transfer_to_ancestor(body: int, ancestor: int) -> np.ndarray:
        mapping = np.eye(6, dtype=np.float32)
        path = []
        while body != ancestor:
            path.append(body)
            body = int(parent[body])
        for path_body in reversed(path):
            mapping = actual_conditional_map[path_body] @ mapping
        return mapping

    cross_mobility_error = 0.0
    for body_a in range(body_count):
        ancestors_a = set()
        cursor = body_a
        while cursor >= 0:
            ancestors_a.add(cursor)
            cursor = int(parent[cursor])
        for body_b in range(body_count):
            cursor = body_b
            while cursor not in ancestors_a:
                cursor = int(parent[cursor])
            lca = cursor
            predicted = transfer_to_ancestor(body_a, lca) @ actual_mobility[lca] @ transfer_to_ancestor(body_b, lca).T
            expected_cross = dense_mobility[
                6 * body_a : 6 * body_a + 6,
                6 * body_b : 6 * body_b + 6,
            ]
            cross_mobility_error = max(cross_mobility_error, float(np.max(np.abs(predicted - expected_cross))))

    correction = (actual - velocities[0]).reshape(-1)
    allowed_impulse = jacobian.T @ mass_stacked @ correction
    max_root_reaction = float(np.max(np.abs(reaction.numpy()[:, 0])))

    with wp.ScopedCapture(device=device) as capture:
        wp.launch(
            project_revolute_trees,
            dim=world_count * 32,
            block_dim=32,
            inputs=inputs,
            device=device,
        )
    with wp.ScopedCapture(device=device) as mobility_capture:
        wp.launch(
            compute_diagonal_mobility,
            dim=world_count * 32,
            block_dim=32,
            inputs=mobility_inputs,
            device=device,
        )
    with wp.ScopedCapture(device=device) as response_capture:
        wp.launch(
            apply_factored_tree_impulse,
            dim=world_count * 32,
            block_dim=32,
            inputs=response_inputs,
            device=device,
        )
    with wp.ScopedCapture(device=device) as combined_capture:
        wp.launch(
            project_revolute_trees,
            dim=world_count * 32,
            block_dim=32,
            inputs=inputs,
            device=device,
        )
        wp.launch(
            compute_diagonal_mobility,
            dim=world_count * 32,
            block_dim=32,
            inputs=mobility_inputs,
            device=device,
        )

    copy_out = wp.empty_like(velocity_out)
    with wp.ScopedCapture(device=device) as copy_capture:
        wp.launch(copy_velocity, dim=shape, inputs=[velocity_wp, copy_out], device=device)

    def time_graph(graph: wp.Graph) -> list[float]:
        samples = []
        for _ in range(args.batches):
            wp.synchronize_device(device)
            start = time.perf_counter()
            for _ in range(args.replays):
                wp.capture_launch(graph)
            wp.synchronize_device(device)
            samples.append(1.0e6 * (time.perf_counter() - start) / args.replays)
        return samples

    solve_us = time_graph(capture.graph)
    mobility_us = time_graph(mobility_capture.graph)
    response_us = time_graph(response_capture.graph)
    combined_us = time_graph(combined_capture.graph)
    copy_us = time_graph(copy_capture.graph)
    return {
        "device": device.name,
        "worlds": world_count,
        "bodies": body_count,
        "max_depth": int(depth.max()),
        "max_dense_error": float(np.max(np.abs(actual - expected))),
        "max_diagonal_mobility_error": float(diagonal_mobility_error),
        "max_cross_mobility_error": float(cross_mobility_error),
        "max_factored_response_error": response_error,
        "max_generalized_impulse": float(np.max(np.abs(allowed_impulse))),
        "max_root_reaction": max_root_reaction,
        "solve_us": solve_us,
        "median_solve_us": float(np.median(solve_us)),
        "median_mobility_us": float(np.median(mobility_us)),
        "median_factored_response_us": float(np.median(response_us)),
        "median_project_plus_mobility_us": float(np.median(combined_us)),
        "median_copy_us": float(np.median(copy_us)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--bodies", type=int, default=29)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(benchmark(args), indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
