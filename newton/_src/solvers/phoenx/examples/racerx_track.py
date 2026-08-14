# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Procedural spline track used by the PhoenX RacerX example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

# Match the extended rainbow palette used by the OptiX viewer for the Kapla
# tower. Explicit colors are needed because the track is logged separately
# from the retained USD scene.
KAPLA_PALETTE = np.asarray(
    (
        (0.86, 0.06, 0.02),
        (0.92, 0.22, 0.01),
        (0.94, 0.42, 0.01),
        (0.88, 0.62, 0.01),
        (0.78, 0.72, 0.01),
        (0.46, 0.72, 0.01),
        (0.03, 0.60, 0.12),
        (0.00, 0.58, 0.38),
        (0.00, 0.62, 0.68),
        (0.00, 0.53, 0.88),
        (0.06, 0.30, 0.88),
        (0.22, 0.12, 0.78),
        (0.42, 0.10, 0.76),
        (0.58, 0.08, 0.68),
    ),
    dtype=np.float32,
)

# The last control point approaches the start from -X, making the car's spawn
# direction (+X) tangent to the circuit. The broad lobes and tighter infield
# give the roughly 210 m lap a useful mix of fast and technical sections.
DEFAULT_CONTROL_POINTS = np.asarray(
    (
        (0.0, 0.0),
        (8.0, 0.0),
        (15.0, 4.0),
        (20.0, 12.0),
        (17.0, 22.0),
        (8.0, 28.0),
        (-4.0, 30.0),
        (-14.0, 25.0),
        (-20.0, 16.0),
        (-19.0, 7.0),
        (-13.0, -1.0),
        (-18.0, -10.0),
        (-13.0, -19.0),
        (-3.0, -24.0),
        (9.0, -22.0),
        (18.0, -16.0),
        (22.0, -8.0),
        (16.0, -3.0),
        (7.0, -5.0),
        (-2.0, -8.0),
        (-8.0, -5.0),
        (-7.0, 0.0),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class TrackLayout:
    """Store uniformly spaced road and barrier transforms."""

    length: float
    centerline: np.ndarray
    tangents: np.ndarray
    road_poses: np.ndarray
    barrier_poses: np.ndarray
    barrier_colors: np.ndarray


def _closed_catmull_rom(control_points: np.ndarray, samples_per_segment: int = 32) -> np.ndarray:
    """Sample a closed, interpolating Catmull-Rom spline densely."""
    points = np.asarray(control_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("A closed track requires at least four 2D control points")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")

    samples = []
    t = np.arange(samples_per_segment, dtype=np.float32) / samples_per_segment
    t2 = t * t
    t3 = t2 * t
    for index in range(len(points)):
        p0 = points[(index - 1) % len(points)]
        p1 = points[index]
        p2 = points[(index + 1) % len(points)]
        p3 = points[(index + 2) % len(points)]
        segment = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        samples.append(segment)
    return np.concatenate(samples, axis=0)


def _resample_closed_curve(points: np.ndarray, spacing: float) -> tuple[np.ndarray, float]:
    """Resample a closed polyline at nearly uniform arc-length intervals."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    closed = np.concatenate((points, points[:1]), axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    total_length = float(cumulative[-1])
    sample_count = max(4, int(np.ceil(total_length / spacing)))
    distances = np.linspace(0.0, total_length, sample_count, endpoint=False)
    segment_indices = np.minimum(np.searchsorted(cumulative, distances, side="right") - 1, len(points) - 1)
    local = (distances - cumulative[segment_indices]) / np.maximum(lengths[segment_indices], 1.0e-8)
    samples = closed[segment_indices] + local[:, None] * (closed[segment_indices + 1] - closed[segment_indices])
    return samples.astype(np.float32), total_length


def _poses(points: np.ndarray, tangents: np.ndarray, height: float) -> np.ndarray:
    """Convert planar positions and tangents to Warp transform rows."""
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    poses = np.zeros((len(points), 7), dtype=np.float32)
    poses[:, :2] = points
    poses[:, 2] = height
    poses[:, 5] = np.sin(0.5 * yaw)
    poses[:, 6] = np.cos(0.5 * yaw)
    return poses


def build_track_layout(
    *,
    control_points: np.ndarray = DEFAULT_CONTROL_POINTS,
    spacing: float = 0.32,
    half_width: float = 0.6,
    barrier_half_height: float = 0.05,
    road_height: float = 0.002,
) -> TrackLayout:
    """Build a closed road with two uniformly sampled barrier rows."""
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    if barrier_half_height <= 0.0:
        raise ValueError("barrier_half_height must be positive")

    dense = _closed_catmull_rom(control_points)
    centerline, length = _resample_closed_curve(dense, spacing)
    tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-8)
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)

    left = centerline + half_width * normals
    right = centerline - half_width * normals
    barrier_points = np.concatenate((left, right), axis=0)
    barrier_tangents = np.concatenate((tangents, tangents), axis=0)
    color_indices = np.concatenate((np.arange(len(centerline)), np.arange(len(centerline)) + 4))
    colors = KAPLA_PALETTE[color_indices % len(KAPLA_PALETTE)]

    return TrackLayout(
        length=length,
        centerline=centerline,
        tangents=tangents,
        road_poses=_poses(centerline, tangents, road_height),
        barrier_poses=_poses(barrier_points, barrier_tangents, barrier_half_height),
        barrier_colors=colors.astype(np.float32),
    )


def add_track_barriers(
    builder: newton.ModelBuilder,
    layout: TrackLayout,
    *,
    half_extents: tuple[float, float, float] = (0.05, 0.05, 0.05),
    density: float = 450.0,
    contact_gap: float = 0.003,
) -> list[int]:
    """Add every barrier block as an independent dynamic rigid body."""
    hx, hy, hz = half_extents
    if min(hx, hy, hz) <= 0.0:
        raise ValueError("barrier half extents must be positive")
    cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=0.8, gap=contact_gap)
    body_indices = []
    for index, (pose, color) in enumerate(zip(layout.barrier_poses, layout.barrier_colors, strict=True)):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
            label=f"track_barrier_{index}",
        )
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=wp.vec3(*color))
        body_indices.append(body)
    return body_indices
