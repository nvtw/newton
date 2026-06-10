# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX "Another Arena"
#
# A procedurally-generated kapla-style arena: three concentric rings of
# bricks stacked on top of one another. Brick dimensions match the
# kapla tower / kapla arena demos so contact tuning carries over.
#
# Unlike :mod:`example_kapla_arena` (which loads a baked USD point
# instancer), this scene is built from scratch via two small helpers:
#
# * :func:`place_boxes_on_circle` -- emit one brick per slot around a
#   circle of given ``radius``, ``count``, and vertical ``z``. Each
#   brick is oriented so its local +Y faces outward (radial). A
#   per-brick ``local_orient`` quaternion is composed *on top* of the
#   ring-tangent orientation, allowing the caller to tip / spin every
#   brick in its own local frame without recomputing tangents.
# * :func:`build_arena_ring` -- one full horizontal ring at a given
#   height, layered into ``layers`` rows of staggered bricks.
#
# Three such rings are stacked vertically, each rotated half a brick
# around the ring axis so the seams don't line up between layers.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_another_arena
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

# Kapla brick full extents [m] -- same prototype as the kapla tower /
# arena demos (``BRICK_FULL_EXTENTS = (0.2, 2.59875, 0.625)`` at
# ``GLOBAL_SCALING = 0.1``).
BRICK_HX: float = 0.5 * 0.02
BRICK_HY: float = 0.5 * 0.259875
BRICK_HZ: float = 0.5 * 0.0625

# Ring geometry: radius is to the brick centre. Three rings stacked
# vertically; each ring is itself ``LAYERS_PER_RING`` rows of bricks.
RING_RADIUS: float = 1.2
BRICKS_PER_LAYER: int = 80
LAYERS_PER_RING: int = 6
NUM_RINGS: int = 3

# Vertical spacing between brick layers within a ring, and between
# rings. A small additive gap absorbs the USD-style "kissing contact"
# overlaps that would otherwise launch the first solver step.
LAYER_GAP: float = 0.002
RING_GAP: float = 0.005

# Starting height: slightly above the ground so the first layer has a
# millimetre of clearance to settle into.
BASE_Z: float = 0.01


def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    """Hamilton product ``a * b`` for :class:`wp.quat` (x, y, z, w).

    Used by :func:`place_boxes_on_circle` to compose the per-brick
    ``local_orient`` on top of the ring-tangent orientation. We avoid
    :func:`wp.quat_multiply` here because we're in host Python (not a
    Warp kernel) and want a dependency-free helper.
    """
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_axis_angle(axis: tuple[float, float, float], angle: float) -> wp.quat:
    """Build a unit quaternion from an axis-angle pair.

    ``axis`` is assumed unit length; the caller is responsible for
    normalisation (all call sites here pass canonical basis vectors).
    """
    half = 0.5 * angle
    s = math.sin(half)
    c = math.cos(half)
    return wp.quat(axis[0] * s, axis[1] * s, axis[2] * s, c)


def place_boxes_on_circle(
    builder: newton.ModelBuilder,
    *,
    radius: float,
    count: int,
    z: float,
    half_extents: tuple[float, float, float],
    local_orient: wp.quat | None = None,
    phase: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> list[int]:
    """Place ``count`` boxes evenly around a horizontal circle.

    Each brick's body frame is oriented so that local +X points
    tangentially (along the circle) and local +Y points radially
    outward; local +Z stays world-up. The brick's longest axis
    (``half_extents[1]`` -> local Y) therefore points outward, which
    keeps the bricks tightly packed when ``count`` is sized to the
    circumference.

    If ``local_orient`` is given it is post-multiplied onto the
    ring-tangent quaternion, i.e. the rotation is applied in the
    brick's own local frame *after* the tangent rotation. This lets
    callers tip each brick about its own axes (e.g. tilt outward,
    spin around its long axis) without re-deriving tangents.

    Args:
        builder: Newton :class:`ModelBuilder` to append bodies to.
        radius: Circle radius [m] measured to brick centres.
        count: Number of bricks. Must be >= 1.
        z: Z height [m] of every brick centre in this ring.
        half_extents: ``(hx, hy, hz)`` brick half-extents [m]. Passed
            straight through to :meth:`add_shape_box`.
        local_orient: Optional per-brick rotation in the brick-local
            frame, applied on top of the ring-tangent orientation.
            ``None`` means identity (no extra rotation).
        phase: Angular offset [rad] of slot 0 around the circle. Used
            by stacked layers to stagger seams.
        center: ``(cx, cy)`` world-space centre of the circle [m].

    Returns:
        Newton body ids of the placed bricks, in slot order.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1 (got {count})")

    hx, hy, hz = half_extents
    cx, cy = center
    body_ids: list[int] = []

    for i in range(count):
        theta = phase + (2.0 * math.pi * i) / count
        px = cx + radius * math.cos(theta)
        py = cy + radius * math.sin(theta)

        # Ring-tangent orientation: rotate about +Z by ``theta`` so
        # the brick's local +Y points radially outward (the body's
        # default +Y axis rotated by ``theta`` is ``(-sin, cos, 0)``,
        # i.e. perpendicular to the radius vector ``(cos, sin, 0)``).
        # We use ``theta + pi/2`` so local +Y -> ``(cos, sin, 0)``
        # (radially outward); local +X -> ``(-sin, cos, 0)``
        # (tangent, ccw).
        q_ring = _quat_axis_angle((0.0, 0.0, 1.0), theta + 0.5 * math.pi)
        if local_orient is not None:
            q = _quat_mul(q_ring, local_orient)
        else:
            q = q_ring

        body = builder.add_body(xform=wp.transform(p=wp.vec3(px, py, z), q=q))
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
        body_ids.append(body)

    return body_ids


def build_arena_ring(
    builder: newton.ModelBuilder,
    *,
    radius: float,
    bricks_per_layer: int,
    layers: int,
    base_z: float,
    half_extents: tuple[float, float, float],
    layer_gap: float = 0.0,
) -> list[int]:
    """Stack ``layers`` rows of bricks on a circular ring.

    Alternating layers are rotated by half a slot (``pi /
    bricks_per_layer``) so the vertical seams between adjacent bricks
    don't align across layers -- the classic brick-bond pattern that
    keeps a kapla wall standing.

    Returns the flat list of body ids in (layer, slot) order.
    """
    _, _, hz = half_extents
    layer_height = 2.0 * hz + layer_gap

    body_ids: list[int] = []
    for layer in range(layers):
        z = base_z + hz + layer * layer_height
        phase = (math.pi / bricks_per_layer) if (layer % 2) else 0.0
        body_ids.extend(
            place_boxes_on_circle(
                builder,
                radius=radius,
                count=bricks_per_layer,
                z=z,
                half_extents=half_extents,
                phase=phase,
            )
        )
    return body_ids


class Example(PortedExample):
    """Three stacked kapla-brick rings forming a tall arena wall."""

    fps = 120
    sim_substeps = 5
    solver_iterations = 10
    velocity_iterations = 1
    default_friction = 0.6
    step_layout = "single_world"
    broad_phase = "sap"
    # ``bricks_per_layer * layers_per_ring * num_rings`` is ~1440
    # bodies -- well below the SAP default but the brick AABBs are
    # nearly co-planar in each ring so candidate counts stay modest.
    shape_pairs_max = 200_000
    start_paused = True

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()

        half_extents = (BRICK_HX, BRICK_HY, BRICK_HZ)
        ring_height = LAYERS_PER_RING * (2.0 * BRICK_HZ + LAYER_GAP)

        extents: list = []
        total = 0
        for ring in range(NUM_RINGS):
            base_z = BASE_Z + ring * (ring_height + RING_GAP)
            ids = build_arena_ring(
                builder,
                radius=RING_RADIUS,
                bricks_per_layer=BRICKS_PER_LAYER,
                layers=LAYERS_PER_RING,
                base_z=base_z,
                half_extents=half_extents,
                layer_gap=LAYER_GAP,
            )
            total += len(ids)
            extents.extend(default_box_half_extents(*half_extents) for _ in ids)

        print(
            f"[PhoenX AnotherArena] rings={NUM_RINGS} "
            f"layers_per_ring={LAYERS_PER_RING} "
            f"bricks_per_layer={BRICKS_PER_LAYER} total_bricks={total}"
        )
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(3.5, -3.5, 1.6), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
