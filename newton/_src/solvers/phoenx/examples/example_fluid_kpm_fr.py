# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import KPMFR3D, KPMFR3DConfig, rasterize_obstacles

_VOLUME_TRANSFER_TABLE = np.array(
    (
        (0.00, 1.00, 0.83, 0.95),
        (0.00, 0.84, 1.00, 0.72),
        (0.00, 0.53, 1.00, 0.32),
        (0.00, 0.50, 1.00, 0.08),
        (0.00, 0.63, 1.00, 0.012),
        (0.00, 0.15, 0.24, 0.0),
        (0.00, 0.04, 0.06, 0.0),
        (0.00, 0.00, 0.00, 0.0),
        (0.00, 0.00, 0.00, 0.0),
        (0.00, 0.00, 0.00, 0.0),
        (0.08, 0.002, 0.00, 0.0),
        (0.32, 0.015, 0.00, 0.0),
        (0.82, 0.09, 0.00, 0.012),
        (1.00, 0.20, 0.00, 0.08),
        (1.00, 0.32, 0.00, 0.32),
        (1.00, 0.42, 0.00, 0.72),
        (1.00, 0.52, 0.006, 0.95),
    ),
    dtype=np.float32,
)


def _camera_angles(position, target):
    direction = np.asarray(target) - np.asarray(position)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-20)
    yaw = math.degrees(math.atan2(float(direction[0]), float(direction[2])))
    pitch = math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))
    return tuple(position), yaw, pitch


class FreeCameraController:
    def __init__(self, viewer, api, position, yaw, pitch, fov, speed):
        self.window = viewer.window
        self.pyglet = viewer.pyglet
        self.api = api
        self.position = np.asarray(position, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov)
        self.speed = float(speed)
        self.keys = set()
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)
        self.window.push_handlers(self)

    def update(self, dt):
        key = self.pyglet.window.key
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        forward = np.array(
            (math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)),
            dtype=np.float32,
        )
        right = np.cross(forward, (0.0, 1.0, 0.0))
        right /= max(float(np.linalg.norm(right)), 1.0e-20)
        move = np.zeros(3, dtype=np.float32)
        move += forward * ((key.W in self.keys) - (key.S in self.keys))
        move += right * ((key.D in self.keys) - (key.A in self.keys))
        move[1] += (key.E in self.keys) - (key.Q in self.keys)
        length = float(np.linalg.norm(move))
        if length > 0.0:
            boost = 4.0 if key.LSHIFT in self.keys or key.RSHIFT in self.keys else 1.0
            self.position += move / length * self.speed * boost * min(max(dt, 0.0), 0.1)
            self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def on_key_press(self, symbol, _modifiers):
        self.keys.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self.keys.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        if buttons & self.pyglet.window.mouse.LEFT:
            self.yaw -= 0.1 * float(dx)
            self.pitch = np.clip(self.pitch + 0.1 * float(dy), -89.0, 89.0)
            self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self.fov = np.clip(self.fov - 2.0 * float(scroll_y), 15.0, 90.0)
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)


@wp.kernel(enable_backward=False)
def _initialize_flow(
    state: wp.array2d[float],
    volume_fraction: wp.array[wp.float16],
):
    p = wp.tid()
    rho = 1.0
    state[0, p] = rho
    state[1, p] = rho * 0.28
    state[2, p] = 0.0
    state[3, p] = 0.0
    volume_fraction[p] = wp.float16(0.0)


@wp.kernel(enable_backward=False)
def _apply_inflow(
    state: wp.array2d[float],
    points: wp.array[float],
    resolution: wp.vec3i,
    size: wp.vec3,
    order: int,
    time: float,
    coherent: int,
):
    p = wp.tid()
    k3 = order * order * order
    sp = p % k3
    cell = p / k3
    i = cell / (resolution[1] * resolution[2])
    if i < 2:
        j = (cell / resolution[2]) % resolution[1]
        k = cell % resolution[2]
        sy = (sp / order) % order
        sz = sp / (order * order)
        y = ((float(j) + 0.5 * (points[sy] + 1.0)) / float(resolution[1]) - 0.5) * size[1]
        z = ((float(k) + 0.5 * (points[sz] + 1.0)) / float(resolution[2]) - 0.5) * size[2]
        v = 0.012 * wp.sin(8.0 * y + 5.0 * z + 1.7 * time) + 0.006 * wp.sin(15.0 * z - 0.9 * time)
        w = 0.012 * wp.cos(7.0 * z - 4.0 * y - 1.3 * time) + 0.006 * wp.sin(13.0 * y + 1.1 * time)
        if coherent != 0:
            v = 0.0025 * wp.sin(2.2 * time + 3.0 * z)
            w = 0.0025 * wp.cos(1.7 * time - 3.0 * y)
        state[0, p] = 1.0
        state[1, p] = 0.28
        state[2, p] = v
        state[3, p] = w


@wp.func
def _point_index(gx: int, gy: int, gz: int, resolution: wp.vec3i, order: int):
    nx = resolution[0] * order
    ny = resolution[1] * order
    nz = resolution[2] * order
    x = (gx + nx) % nx
    y = (gy + ny) % ny
    z = (gz + nz) % nz
    i = x / order
    j = y / order
    k = z / order
    sx = x % order
    sy = y % order
    sz = z % order
    return ((i * resolution[1] + j) * resolution[2] + k) * order * order * order + sx + sy * order + sz * order * order


@wp.func
def _sample_velocity(
    state: wp.array2d[float],
    position: wp.vec3,
    resolution: wp.vec3i,
    size: wp.vec3,
    points: wp.array[float],
    order: int,
):
    grid = wp.vec3(
        (position[0] / size[0] + 0.5) * float(resolution[0]),
        (position[1] / size[1] + 0.5) * float(resolution[1]),
        (position[2] / size[2] + 0.5) * float(resolution[2]),
    )
    cell = wp.vec3i(
        wp.clamp(int(wp.floor(grid[0])), 0, resolution[0] - 1),
        wp.clamp(int(wp.floor(grid[1])), 0, resolution[1] - 1),
        wp.clamp(int(wp.floor(grid[2])), 0, resolution[2] - 1),
    )
    local = wp.vec3(
        wp.clamp(grid[0] - float(cell[0]), 0.0, 1.0),
        wp.clamp(grid[1] - float(cell[1]), 0.0, 1.0),
        wp.clamp(grid[2] - float(cell[2]), 0.0, 1.0),
    )
    coordinate = 2.0 * local - wp.vec3(1.0)
    low = wp.vec3i(0)
    weight = wp.vec3(0.0)
    for axis in range(3):
        for candidate in range(5):
            if candidate < order - 1 and coordinate[axis] > points[candidate + 1]:
                low[axis] = candidate + 1
        if low[axis] > order - 2:
            low[axis] = order - 2
        a = points[low[axis]]
        b = points[low[axis] + 1]
        weight[axis] = wp.clamp((coordinate[axis] - a) / (b - a), 0.0, 1.0)
    velocity = wp.vec3(0.0)
    k3 = order * order * order
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                point = low + wp.vec3i(dx, dy, dz)
                sp = point[0] + point[1] * order + point[2] * order * order
                p = ((cell[0] * resolution[1] + cell[1]) * resolution[2] + cell[2]) * k3 + sp
                wx = weight[0] if dx == 1 else 1.0 - weight[0]
                wy = weight[1] if dy == 1 else 1.0 - weight[1]
                wz = weight[2] if dz == 1 else 1.0 - weight[2]
                velocity += wp.vec3(state[1, p], state[2, p], state[3, p]) / wp.max(state[0, p], 1.0e-6) * wx * wy * wz
    return velocity


@wp.func
def _sample_state_features(
    state: wp.array2d[float],
    position: wp.vec3,
    resolution: wp.vec3i,
    size: wp.vec3,
    points: wp.array[float],
    order: int,
    h: float,
):
    vxm = _sample_velocity(state, position - wp.vec3(h, 0.0, 0.0), resolution, size, points, order)
    vxp = _sample_velocity(state, position + wp.vec3(h, 0.0, 0.0), resolution, size, points, order)
    vym = _sample_velocity(state, position - wp.vec3(0.0, h, 0.0), resolution, size, points, order)
    vyp = _sample_velocity(state, position + wp.vec3(0.0, h, 0.0), resolution, size, points, order)
    vzm = _sample_velocity(state, position - wp.vec3(0.0, 0.0, h), resolution, size, points, order)
    vzp = _sample_velocity(state, position + wp.vec3(0.0, 0.0, h), resolution, size, points, order)
    dv_dx = (vxp - vxm) / (2.0 * h)
    dv_dy = (vyp - vym) / (2.0 * h)
    dv_dz = (vzp - vzm) / (2.0 * h)
    curl = wp.vec3(dv_dy[2] - dv_dz[1], dv_dz[0] - dv_dx[2], dv_dx[1] - dv_dy[0])
    s01 = 0.5 * (dv_dx[1] + dv_dy[0])
    s02 = 0.5 * (dv_dx[2] + dv_dz[0])
    s12 = 0.5 * (dv_dy[2] + dv_dz[1])
    strain2 = (
        dv_dx[0] * dv_dx[0] + dv_dy[1] * dv_dy[1] + dv_dz[2] * dv_dz[2] + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12)
    )
    q = 0.25 * wp.dot(curl, curl) - 0.5 * strain2
    return wp.vec4(curl[0], curl[1], curl[2], q)


@wp.kernel(enable_backward=False)
def _advect_filaments(
    vertices: wp.array[float],
    radii: wp.array[float],
    strengths: wp.array[float],
    generations: wp.array[wp.uint32],
    state: wp.array2d[float],
    resolution: wp.vec3i,
    size: wp.vec3,
    points: wp.array[float],
    order: int,
    points_per_strand: int,
    dt: float,
    coherent: int,
):
    strand = wp.tid()
    base = strand * points_per_strand
    head = 3 * (base + points_per_strand - 1)
    position = wp.vec3(vertices[head], vertices[head + 1], vertices[head + 2])
    velocity = _sample_velocity(state, position, resolution, size, points, order)
    midpoint = position + 0.5 * dt * velocity
    midpoint_velocity = _sample_velocity(state, midpoint, resolution, size, points, order)
    next_position = position + dt * midpoint_velocity
    h = 0.035
    features = _sample_state_features(state, midpoint, resolution, size, points, order, h)
    curl = wp.vec3(features[0], features[1], features[2])
    curl_length = wp.length(curl)
    q = features[3]
    core = wp.clamp((wp.sqrt(wp.max(q, 0.0)) - 0.005) * 5.0, 0.0, 1.0)
    shear = wp.clamp((curl_length - 0.01) * 2.0, 0.0, 1.0)
    curl_strength = wp.max(core, 0.35 * shear)
    trace_velocity = 0.35 * midpoint_velocity + (0.18 + 0.52 * core) * curl / wp.max(curl_length, 0.05)
    if coherent != 0:
        trace_velocity = midpoint_velocity + 0.12 * core * curl / wp.max(curl_length, 0.05)
    next_position = position + dt * trace_velocity
    radius = wp.length(next_position)
    reset = (
        next_position[0] > 0.46 * size[0]
        or wp.abs(next_position[1]) > 0.47 * size[1]
        or wp.abs(next_position[2]) > 0.47 * size[2]
        or (radius < 0.36 and wp.length(midpoint_velocity) < 0.04)
    )
    if reset:
        generation = generations[strand] + wp.uint32(1)
        generations[strand] = generation
        rng = wp.rand_init(wp.int32(generation) + 17, strand)
        best_score = -1.0
        for _candidate_index in range(12):
            candidate_u = wp.randf(rng)
            candidate_x = -0.18 + 2.82 * candidate_u * candidate_u
            candidate_angle = 2.0 * wp.pi * wp.randf(rng)
            candidate_width = 0.34 + 0.14 * wp.max(candidate_x, 0.0)
            candidate_radius = candidate_width * wp.sqrt(wp.randf(rng))
            if coherent != 0:
                candidate_x = 0.08 + 2.55 * candidate_u * candidate_u
                candidate_width = 0.34 + 0.10 * candidate_x
                candidate_radius = candidate_width * (0.72 + 0.36 * wp.randf(rng))
            candidate = wp.vec3(
                candidate_x,
                candidate_radius * wp.cos(candidate_angle),
                candidate_radius * wp.sin(candidate_angle),
            )
            candidate_features = _sample_state_features(state, candidate, resolution, size, points, order, h)
            candidate_curl = wp.length(wp.vec3(candidate_features[0], candidate_features[1], candidate_features[2]))
            candidate_score = wp.sqrt(wp.max(candidate_features[3], 0.0)) + 0.04 * candidate_curl
            if wp.length(candidate) > 0.38 and candidate_score > best_score:
                best_score = candidate_score
                next_position = candidate
        for point in range(points_per_strand):
            dst = 3 * (base + point)
            vertices[dst] = next_position[0] - 0.002 * float(points_per_strand - point)
            vertices[dst + 1] = next_position[1]
            vertices[dst + 2] = next_position[2]
            strengths[base + point] = 0.0
    else:
        for point in range(points_per_strand - 1):
            dst = 3 * (base + point)
            src = dst + 3
            vertices[dst] = vertices[src]
            vertices[dst + 1] = vertices[src + 1]
            vertices[dst + 2] = vertices[src + 2]
            strengths[base + point] = strengths[base + point + 1]
        vertices[head] = next_position[0]
        vertices[head + 1] = next_position[1]
        vertices[head + 2] = next_position[2]
        strengths[base + points_per_strand - 1] = curl_strength
    base_radius = 0.0012 + 0.0018 * float((strand * 17) % 29) / 28.0
    for point in range(points_per_strand):
        vertex = 3 * (base + point)
        x = vertices[vertex]
        fade_in = wp.clamp((x + 0.42) / 0.32, 0.0, 1.0)
        fade_out = wp.clamp((2.92 - x) / 0.22, 0.0, 1.0)
        age = 0.28 + 0.72 * float(point + 1) / float(points_per_strand)
        visibility = wp.sqrt(strengths[base + point])
        radii[base + point] = base_radius * fade_in * fade_out * age * visibility


@wp.kernel(enable_backward=False)
def _write_uniform_velocity(
    output: wp.array[wp.vec3],
    state: wp.array2d[float],
    resolution: wp.vec3i,
    size: wp.vec3,
    points: wp.array[float],
    order: int,
    volume_resolution: wp.vec3i,
):
    gx, gy, gz = wp.tid()
    position = wp.vec3(
        (float(gx) / float(volume_resolution[0] - 1) - 0.5) * size[0],
        (float(gy) / float(volume_resolution[1] - 1) - 0.5) * size[1],
        (float(gz) / float(volume_resolution[2] - 1) - 0.5) * size[2],
    )
    index = (gx * volume_resolution[1] + gy) * volume_resolution[2] + gz
    output[index] = _sample_velocity(state, position, resolution, size, points, order)


@wp.kernel(enable_backward=False)
def _smooth_uniform_velocity(
    output: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    volume_resolution: wp.vec3i,
):
    gx, gy, gz = wp.tid()
    ny = volume_resolution[1]
    nz = volume_resolution[2]
    index = (gx * ny + gy) * nz + gz
    if gx == 0 or gy == 0 or gz == 0 or gx == volume_resolution[0] - 1 or gy == ny - 1 or gz == nz - 1:
        output[index] = velocity[index]
        return
    output[index] = 0.4 * velocity[index] + 0.1 * (
        velocity[index - ny * nz]
        + velocity[index + ny * nz]
        + velocity[index - nz]
        + velocity[index + nz]
        + velocity[index - 1]
        + velocity[index + 1]
    )


@wp.func
def _sample_uniform_scalar(
    field: wp.array[float],
    grid: wp.vec3,
    resolution: wp.vec3i,
):
    base = wp.vec3i(
        wp.clamp(int(wp.floor(grid[0])), 0, resolution[0] - 2),
        wp.clamp(int(wp.floor(grid[1])), 0, resolution[1] - 2),
        wp.clamp(int(wp.floor(grid[2])), 0, resolution[2] - 2),
    )
    fraction = wp.vec3(
        wp.clamp(grid[0] - float(base[0]), 0.0, 1.0),
        wp.clamp(grid[1] - float(base[1]), 0.0, 1.0),
        wp.clamp(grid[2] - float(base[2]), 0.0, 1.0),
    )
    value = 0.0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                x = base[0] + dx
                y = base[1] + dy
                z = base[2] + dz
                weight = fraction[0] if dx == 1 else 1.0 - fraction[0]
                weight *= fraction[1] if dy == 1 else 1.0 - fraction[1]
                weight *= fraction[2] if dz == 1 else 1.0 - fraction[2]
                value += field[(x * resolution[1] + y) * resolution[2] + z] * weight
    return value


@wp.kernel(enable_backward=False)
def _advect_tracer(
    output: wp.array[float],
    tracer: wp.array[float],
    velocity: wp.array[wp.vec3],
    size: wp.vec3,
    resolution: wp.vec3i,
    dt: float,
):
    gx, gy, gz = wp.tid()
    index = (gx * resolution[1] + gy) * resolution[2] + gz
    if gx == 0 or gy == 0 or gz == 0 or gx == resolution[0] - 1 or gy == resolution[1] - 1 or gz == resolution[2] - 1:
        output[index] = 0.0
        return
    spacing = wp.vec3(
        size[0] / float(resolution[0] - 1),
        size[1] / float(resolution[1] - 1),
        size[2] / float(resolution[2] - 1),
    )
    grid = wp.vec3(float(gx), float(gy), float(gz)) - dt * wp.cw_div(velocity[index], spacing)
    value = 0.9992 * _sample_uniform_scalar(tracer, grid, resolution)
    position = wp.vec3(
        (float(gx) / float(resolution[0] - 1) - 0.5) * size[0],
        (float(gy) / float(resolution[1] - 1) - 0.5) * size[1],
        (float(gz) / float(resolution[2] - 1) - 0.5) * size[2],
    )
    radius = wp.length(wp.vec2(position[1], position[2]))
    if wp.abs(position[0] + 0.58) < 1.6 * spacing[0] and radius < 0.62:
        bands = wp.abs(wp.sin(13.0 * position[1] + 4.0 * position[2]) * wp.cos(11.0 * position[2] - 3.0 * position[1]))
        source = 0.18 + 0.82 * bands * bands * bands
        value = wp.max(value, source)
    output[index] = value


@wp.kernel(enable_backward=False)
def _write_volume(
    volume: wp.uint64,
    velocity: wp.array[wp.vec3],
    tracer: wp.array[float],
    size: wp.vec3,
    volume_resolution: wp.vec3i,
):
    gx, gy, gz = wp.tid()
    nx = volume_resolution[0]
    ny = volume_resolution[1]
    nz = volume_resolution[2]
    if gx == 0 or gy == 0 or gz == 0 or gx == nx - 1 or gy == ny - 1 or gz == nz - 1:
        wp.volume_store_v(volume, gx, gy, gz, wp.vec3(0.0, 0.0, 1.0))
        return
    index = (gx * ny + gy) * nz + gz
    dx = size[0] / float(nx - 1)
    dy = size[1] / float(ny - 1)
    dz = size[2] / float(nz - 1)
    dv_dx = (velocity[index + ny * nz] - velocity[index - ny * nz]) / (2.0 * dx)
    dv_dy = (velocity[index + nz] - velocity[index - nz]) / (2.0 * dy)
    dv_dz = (velocity[index + 1] - velocity[index - 1]) / (2.0 * dz)
    curl = wp.vec3(dv_dy[2] - dv_dz[1], dv_dz[0] - dv_dx[2], dv_dx[1] - dv_dy[0])
    s01 = 0.5 * (dv_dx[1] + dv_dy[0])
    s02 = 0.5 * (dv_dx[2] + dv_dz[0])
    s12 = 0.5 * (dv_dy[2] + dv_dz[1])
    strain2 = wp.dot(wp.vec3(dv_dx[0], dv_dy[1], dv_dz[2]), wp.vec3(dv_dx[0], dv_dy[1], dv_dz[2])) + 2.0 * (
        s01 * s01 + s02 * s02 + s12 * s12
    )
    q = 0.25 * wp.dot(curl, curl) - 0.5 * strain2
    curl_length = wp.length(curl)
    q_strength = wp.sqrt(wp.max(q, 0.0))
    position = wp.vec3(
        (float(gx) / float(nx - 1) - 0.5) * size[0],
        (float(gy) / float(ny - 1) - 0.5) * size[1],
        (float(gz) / float(nz - 1) - 0.5) * size[2],
    )
    core = wp.clamp((q_strength - 0.018) * 18.0, 0.0, 1.0)
    shear = wp.clamp((curl_length - 0.05) * 4.0, 0.0, 1.0)
    wake = wp.clamp((position[0] + 0.45) / 0.80, 0.0, 1.0)
    fade_out = wp.clamp((2.78 - position[0]) / 0.32, 0.0, 1.0)
    clearance = wp.clamp((wp.length(position) - 0.36) / 0.12, 0.0, 1.0)
    wake_width = 0.42 + 0.13 * wp.max(position[0], 0.0)
    wake_profile = wp.clamp((wake_width - wp.length(wp.vec2(position[1], position[2]))) / 0.18, 0.0, 1.0)
    core_ridge = core * core * wp.sqrt(core)
    shear_ridge = shear * shear * shear
    magnitude = (
        tracer[index] * (0.015 + 1.35 * core_ridge + 0.18 * shear_ridge) * wake * fade_out * clearance * wake_profile
    )
    local_velocity = velocity[index]
    fluctuation = local_velocity - wp.vec3(0.28, 0.0, 0.0)
    helicity = wp.dot(fluctuation, curl) / wp.max(wp.length(fluctuation) * curl_length, 1.0e-6)
    orientation = (curl[1] + 0.45 * curl[2]) / wp.max(curl_length, 1.0e-6)
    signed_feature = wp.clamp(0.68 * helicity + 0.32 * orientation - 0.07, -1.0, 1.0)
    feature = wp.sqrt(wp.abs(signed_feature))
    if signed_feature < 0.0:
        feature = -feature
    wp.volume_store_v(volume, gx, gy, gz, wp.vec3(magnitude, feature, 1.0))


@wp.kernel(enable_backward=False)
def _write_volume_shadow(
    volume: wp.uint64,
    size: wp.vec3,
    volume_resolution: wp.vec3i,
):
    gx, gy, gz = wp.tid()
    value = wp.volume_lookup_v(volume, gx, gy, gz)
    if value[0] < 1.0e-4:
        wp.volume_store_v(volume, gx, gy, gz, wp.vec3(value[0], value[1], 1.0))
        return
    position = wp.vec3(
        (float(gx) / float(volume_resolution[0] - 1) - 0.5) * size[0],
        (float(gy) / float(volume_resolution[1] - 1) - 0.5) * size[1],
        (float(gz) / float(volume_resolution[2] - 1) - 0.5) * size[2],
    )
    direction = wp.normalize(wp.vec3(-0.45, 0.82, 0.35))
    optical_depth = 0.0
    for step in range(12):
        probe = position + direction * (0.06 + 0.12 * float(step))
        index = wp.volume_world_to_index(volume, probe)
        optical_depth += wp.max(wp.volume_sample_v(volume, index, wp.Volume.LINEAR)[0], 0.0)
    value[2] = wp.exp(-0.22 * optical_depth)
    wp.volume_store_v(volume, gx, gy, gz, value)


@wp.kernel(enable_backward=False)
def _pack_display(src: wp.array2d[wp.vec4], dst: wp.array[wp.uint32], width: int, height: int):
    x, y = wp.tid()
    if x >= width or y >= height:
        return
    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    dst[y * width + x] = wp.uint32(255) << wp.uint32(24) | b << wp.uint32(16) | g << wp.uint32(8) | r


def initialize_flow(solver):
    points = wp.array(solver.points, dtype=float, device=solver.device)
    wp.launch(
        _initialize_flow,
        dim=solver.state.shape[1],
        inputs=[solver.state, solver.volume_fraction],
        device=solver.device,
    )
    return points


def make_volume(solver, samples_per_cell, world_transform=None):
    config = solver.config
    order = config.order
    resolution = np.asarray(config.resolution, dtype=np.int32) * samples_per_cell
    size = np.asarray(config.size, dtype=np.float32)
    bounds_min = -0.5 * size
    bounds_max = 0.5 * size
    voxel_size = tuple(size / (resolution - 1))
    tile_axes = [np.arange(0, resolution[i], 8, dtype=np.int32) for i in range(3)]
    tile_coords = np.stack(np.meshgrid(*tile_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    allocation = {"voxel_size": voxel_size, "translation": tuple(bounds_min)}
    render_bounds_min = bounds_min
    render_bounds_max = bounds_max
    if world_transform is not None:
        matrix = np.asarray(world_transform, dtype=np.float32)
        rotation = matrix[:3, :3]
        allocation = {
            "transform": rotation @ np.diag(voxel_size),
            "translation": tuple(rotation @ bounds_min + matrix[:3, 3]),
        }
        corners = (
            np.array(np.meshgrid(*zip(bounds_min, bounds_max, strict=True), indexing="ij"), dtype=np.float32)
            .reshape(3, -1)
            .T
        )
        corners = corners @ rotation.T + matrix[:3, 3]
        render_bounds_min = corners.min(axis=0)
        render_bounds_max = corners.max(axis=0)
    volume = wp.Volume.allocate_by_tiles(
        wp.array(tile_coords, dtype=wp.vec3i, device="cuda"),
        bg_value=wp.vec3(0.0, 0.0, 1.0),
        device="cuda",
        **allocation,
    )
    point_count = int(np.prod(resolution))
    velocity = wp.empty(point_count, dtype=wp.vec3, device="cuda")
    smooth_velocity = wp.empty_like(velocity)
    tracer = wp.zeros(point_count, dtype=float, device="cuda")
    next_tracer = wp.empty_like(tracer)
    points = wp.array(solver.points, dtype=float, device="cuda")
    wp.launch(
        _write_uniform_velocity,
        dim=tuple(int(v) for v in resolution),
        inputs=[
            velocity,
            solver.state,
            config.resolution,
            config.size,
            points,
            order,
            tuple(int(v) for v in resolution),
        ],
        device="cuda",
    )
    wp.launch(
        _smooth_uniform_velocity,
        dim=tuple(int(v) for v in resolution),
        inputs=[smooth_velocity, velocity, tuple(int(v) for v in resolution)],
        device="cuda",
    )
    for _ in range(420):
        wp.launch(
            _advect_tracer,
            dim=tuple(int(v) for v in resolution),
            inputs=[next_tracer, tracer, smooth_velocity, config.size, tuple(int(v) for v in resolution), 0.03],
            device="cuda",
        )
        wp.copy(tracer, next_tracer)
    wp.launch(
        _write_volume,
        dim=tuple(int(v) for v in resolution),
        inputs=[
            volume.id,
            smooth_velocity,
            tracer,
            config.size,
            tuple(int(v) for v in resolution),
        ],
        device="cuda",
    )
    wp.launch(
        _write_volume_shadow,
        dim=tuple(int(v) for v in resolution),
        inputs=[volume.id, config.size, tuple(int(v) for v in resolution)],
        device="cuda",
    )
    return (
        volume,
        render_bounds_min,
        render_bounds_max,
        velocity,
        smooth_velocity,
        tracer,
        next_tracer,
        tuple(int(v) for v in resolution),
    )


def step_flow(solver, points, sim_time, coherent=0):
    solver.step()
    sim_time += solver.dt
    wp.launch(
        _apply_inflow,
        dim=solver.state.shape[1],
        inputs=[
            solver.state,
            points,
            solver.config.resolution,
            solver.config.size,
            solver.config.order,
            sim_time,
            coherent,
        ],
        device=solver.device,
    )
    return sim_time


def update_volume(solver, points, volume_data):
    volume, _, _, velocity, smooth_velocity, tracer, next_tracer, resolution = volume_data
    wp.launch(
        _write_uniform_velocity,
        dim=resolution,
        inputs=[
            velocity,
            solver.state,
            solver.config.resolution,
            solver.config.size,
            points,
            solver.config.order,
            resolution,
        ],
        device=solver.device,
    )
    wp.launch(
        _smooth_uniform_velocity,
        dim=resolution,
        inputs=[smooth_velocity, velocity, resolution],
        device=solver.device,
    )
    wp.launch(
        _advect_tracer,
        dim=resolution,
        inputs=[next_tracer, tracer, smooth_velocity, solver.config.size, resolution, 0.03],
        device=solver.device,
    )
    wp.copy(tracer, next_tracer)
    wp.launch(
        _write_volume,
        dim=resolution,
        inputs=[volume.id, smooth_velocity, tracer, solver.config.size, resolution],
        device=solver.device,
    )
    wp.launch(
        _write_volume_shadow,
        dim=resolution,
        inputs=[volume.id, solver.config.size, resolution],
        device=solver.device,
    )


def bind_volume(renderer, volume_data, *, realtime=True, bounds=None):
    volume, bounds_min, bounds_max, *_ = volume_data
    if bounds is not None:
        bounds_min, bounds_max = bounds
    renderer.set_volume(
        volume,
        bounds_min,
        bounds_max,
        density_scale=0.45,
        step_size=0.026 if realtime else 0.016,
        anisotropy=0.35,
        emission=0.0,
        transfer_table=_VOLUME_TRANSFER_TABLE,
        density_feature=True,
    )


def make_filaments(api, solver, material, strand_count, points_per_strand):
    rng = np.random.default_rng(17)
    head_x = np.full(strand_count, 3.0, dtype=np.float32)
    trail = np.linspace(-0.18, 0.0, points_per_strand, dtype=np.float32)
    vertices = np.empty((strand_count, points_per_strand, 3), dtype=np.float32)
    vertices[:, :, 0] = head_x[:, None] + trail[None, :]
    vertices[:, :, 1:] = 0.0
    taper = np.linspace(0.0015, 0.0055, points_per_strand, dtype=np.float32)
    radii = taper[None, :] * rng.uniform(0.72, 1.28, (strand_count, 1))
    segment_indices = (
        np.arange(strand_count, dtype=np.uint32)[:, None] * points_per_strand
        + np.arange(points_per_strand - 1, dtype=np.uint32)[None, :]
    ).reshape(-1)
    geometry_id = api.create_curve(
        vertices.reshape(-1, 3),
        radii.reshape(-1),
        segment_indices,
        material_id=material,
        dynamic=True,
    )
    api.create_instance(geometry_id)
    return (
        geometry_id,
        wp.array(vertices.reshape(-1), dtype=float, device="cuda"),
        wp.array(radii.reshape(-1), dtype=float, device="cuda"),
        wp.zeros(strand_count * points_per_strand, dtype=float, device="cuda"),
        wp.zeros(strand_count, dtype=wp.uint32, device="cuda"),
    )


def main(*, volume_only=False):
    import warp_optix as woptix  # noqa: PLC0415
    from PIL import Image
    from warp_optix.pathtracing import Mesh, PathTracerAPI  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("volume.png"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-dlss-rr", action="store_true")
    parser.add_argument("--camera-speed", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=2226)
    parser.add_argument("--elements", type=int, default=64)
    parser.add_argument("--order", type=int, choices=range(3, 7), default=6)
    parser.add_argument("--filaments", type=int, default=4096)
    parser.add_argument("--trail-points", type=int, default=64)
    parser.add_argument("--filament-warmup", type=int, default=480)
    parser.add_argument("--volume-samples-per-cell", type=int, default=0)
    parser.add_argument(
        "--wake-style",
        choices=("turbulent", "coherent"),
        default="turbulent",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--volume", action="store_true", default=volume_only)
    args = parser.parse_args()
    coherent = int(args.wake_style == "coherent")

    wp.init()
    resolution = (args.elements, args.elements // 2, args.elements // 2)
    solver = KPMFR3D(
        KPMFR3DConfig(
            resolution,
            size=(6.0, 3.0, 3.0),
            order=args.order,
            reference_velocity=0.28,
            reynolds=5_000.0 if coherent else 100_000.0,
            cfl=0.45,
        ),
        device="cuda",
    )
    points = initialize_flow(solver)
    obstacle_builder = newton.ModelBuilder()
    obstacle_builder.add_shape_sphere(body=-1, radius=0.34)
    obstacle_model = obstacle_builder.finalize(device="cuda")
    rasterize_obstacles(
        solver,
        obstacle_model,
        origin=tuple(-0.5 * np.asarray(solver.config.size)),
    )
    sim_time = 0.0
    for _ in range(args.warmup_steps):
        sim_time = step_flow(solver, points, sim_time, coherent)
    volume_data = None
    if args.volume:
        volume_data = make_volume(solver, args.volume_samples_per_cell or solver.config.order)
    realtime = args.live or args.max_frames > 1
    graph_capture = volume_only
    api = PathTracerAPI(
        args.width,
        args.height,
        enable_dlss_rr=not args.no_dlss_rr,
        enable_set=True,
        enable_cuda_graphs=graph_capture,
        dlss_quality="quality",
        samples_per_frame=1 if realtime else 8,
        max_bounces=2 if realtime else 3,
    )
    if not api.initialize():
        raise RuntimeError("Failed to initialize path tracing")
    ground_color = (0.447988, 0.447988, 0.447988)
    sphere_color = (0.50, 0.54, 0.60)
    ground = api.create_pbr_material(
        ground_color,
        0.8,
        0.0,
        ior=1.46,
        specular=0.75,
        clearcoat=0.03,
        clearcoat_roughness=0.4,
        u_subdiv=40.0,
        v_subdiv=24.0,
    )
    sphere = api.create_pbr_material(
        sphere_color,
        0.34,
        0.0,
        ior=1.46,
        specular=0.72,
        clearcoat=0.12,
        clearcoat_roughness=0.22,
    )
    flow = api.create_pbr_material(
        (0.0, 0.32, 1.0),
        0.26,
        0.05,
        base_color_scale=1.0,
        emissive=(0.0, 0.05, 0.14),
    )
    floor_vertices = np.array(
        ((-10.0, -0.70, -6.0), (10.0, -0.70, -6.0), (10.0, -0.70, 6.0), (-10.0, -0.70, 6.0)),
        dtype=np.float32,
    )
    floor_mesh = Mesh(
        floor_vertices,
        np.array(((0, 2, 1), (0, 3, 2)), dtype=np.uint32),
        normals=np.tile((0.0, 1.0, 0.0), (4, 1)).astype(np.float32),
        texcoords=np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float32),
        material_id=ground,
    )
    api.scene.add_instance(api.scene.add_mesh(floor_mesh))
    api.add_sphere((0.0, 0.0, 0.0), 0.34, 96, sphere)
    api.scene.add_light_sphere((-1.4, 1.8, 2.0), 0.55, (1.0, 0.93, 0.82), 12.0)
    api.scene.usd_ambient_light = (0.025, 0.028, 0.034)
    filament_data = None
    if not volume_only:
        filament_data = make_filaments(api, solver, flow, args.filaments, args.trail_points)
        curve_id, vertices, radii, strengths, generations = filament_data
        filament_inputs = [
            vertices,
            radii,
            strengths,
            generations,
            solver.state,
            solver.config.resolution,
            solver.config.size,
            points,
            solver.config.order,
            args.trail_points,
            0.12,
            coherent,
        ]
        for _ in range(args.filament_warmup):
            wp.launch(
                _advect_filaments,
                dim=args.filaments,
                inputs=filament_inputs,
                device="cuda",
            )
        if args.diagnostics:
            host_radii = radii.numpy()
            host_strengths = strengths.numpy()
            host_vertices = vertices.numpy().reshape(-1, 3)
            active = host_radii > 0.0
            print(
                "filaments",
                f"active={np.mean(active):.4f}",
                f"radius_max={np.max(host_radii):.6f}",
                f"strength_p95={np.percentile(host_strengths, 95):.6f}",
                f"bounds={host_vertices[active].min(0) if np.any(active) else None}..",
                f"{host_vertices[active].max(0) if np.any(active) else None}",
            )
    api.build_scene()
    if filament_data is not None:
        api.update_curve_device(
            curve_id,
            vertices,
            radii=radii,
            rebuild_tlas=True,
        )
    camera_position = (0.75, 0.30, 3.65) if volume_only else (0.75, 0.35, 4.6)
    camera_target = (0.72, 0.0, 0.0) if volume_only else (0.55, 0.0, 0.0)
    api.set_camera_look_at(camera_position, camera_target, fov=38.0)
    api.set_use_procedural_sky(True)
    api.set_sky_parameters(
        (-0.45, 0.82, 0.35),
        multiplier=1.0,
        haze=0.0,
        saturation=1.0,
        ground_color=(0.665185, 0.665185, 0.665185),
        grayscale=0.0,
    )
    if args.volume:
        render_bounds = ((-0.80, -0.95, -0.95), (2.98, 0.95, 0.95)) if volume_only else None
        bind_volume(api, volume_data, realtime=realtime, bounds=render_bounds)
    api.tonemap_exposure = 0.68
    api.tonemap_contrast = 1.08
    api.tonemap_saturation = 1.1

    def advance():
        nonlocal sim_time
        sim_time = step_flow(solver, points, sim_time, coherent)
        if filament_data is not None:
            wp.launch(
                _advect_filaments,
                dim=args.filaments,
                inputs=filament_inputs,
                device="cuda",
            )
            api.update_curve_device(
                curve_id,
                vertices,
                radii=radii,
                rebuild_tlas=False,
            )
        if args.volume:
            update_volume(solver, points, volume_data)

    if args.live:
        render_width, render_height = args.width, args.height

        def resize(width, height):
            nonlocal render_width, render_height
            render_width, render_height = width, height
            api.resize(width, height)

        viewer = woptix.GLInteropViewer(
            width=args.width,
            height=args.height,
            device="cuda",
            title=f"KPM-FR live — {'DLSS-RR' if api.dlss_enabled else 'native'} — WASD + drag",
            fps=args.fps,
            on_resize=resize,
            vsync=False,
        )

        position = camera_position
        position, yaw, pitch = _camera_angles(position, camera_target)
        controller = FreeCameraController(viewer, api, position, yaw, pitch, 38.0, args.camera_speed)
        last_elapsed = 0.0

        def render(mapped, _frame, elapsed):
            nonlocal last_elapsed, sim_time
            advance()
            controller.update(elapsed - last_elapsed)
            last_elapsed = elapsed
            api.render_frame()
            wp.launch(
                _pack_display,
                dim=(render_width, render_height),
                inputs=[
                    api.viewer.tonemapped_output,
                    mapped,
                    render_width,
                    render_height,
                ],
                device="cuda",
            )

        live_start = time.perf_counter()
        viewer.run(render, max_frames=args.max_frames)
        live_elapsed = time.perf_counter() - live_start
        print(f"Live: {viewer.frame_index} frames, {viewer.frame_index / live_elapsed:.1f} FPS")
    else:
        for _ in range(max(args.max_frames, 1)):
            advance()
            api.render_frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(api.get_frame_uint8()[..., :3]), mode="RGB").save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
