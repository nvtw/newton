# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Chain mesh
#
# A short anchor-style chain made of 10 interlocked rings dropped onto a
# ground plane. Each ring is a single rigid body whose collision geometry
# is 6 capsules arranged around a circle to approximate a torus -- the
# capsule hemispheres overlap so the loop is closed without holes.
# Consecutive rings are rotated 90 degrees about the chain axis so they
# interlock like real chain links.
#
# Ring outer diameter: 10 cm (centre-line radius R = 5 cm).
# Tube (small torus) diameter: 2.5 cm (capsule radius r = 1.25 cm).
#
# Run: python -m newton._src.solvers.phoenx.examples.example_chain_mesh
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    run_ported_example,
)

#: Centre-line radius of a ring [m]. Outer diameter 10 cm -> R = 0.05 m.
RING_RADIUS = 0.05
#: Capsule (tube) radius [m]. Small torus diameter 2 cm -> r = 0.01 m.
TUBE_RADIUS = 0.01
#: Number of capsule segments per ring.
SEGMENTS_PER_RING = 6
#: Number of interlocked rings in each chain.
N_RINGS = 31
#: Number of parallel chains laid out along world +y.
N_CHAINS = 15

#: Capsule cylindrical half-height. Adjacent segment centres on the ring
#: are separated by a chord of length ``2 * R * sin(pi / N)``; setting
#: the half-height to half that chord puts the cylindrical mid-points of
#: neighbouring capsules exactly back-to-back, so their hemispherical
#: end-caps overlap and seal the torus surface.
SEGMENT_HALF_HEIGHT = RING_RADIUS * math.sin(math.pi / SEGMENTS_PER_RING)

#: Spacing between consecutive ring centres along the chain (world +x).
#: Rings lie flat in the world x-y plane; spacing = 2 (R + r) + small
#: gap so adjacent rings don't overlap.
RING_SPACING = 2.0 * (RING_RADIUS + TUBE_RADIUS) + 0.005 - 0.05

#: Drop height: ring centre starts just above the tube radius so the
#: flat ring clears the ground plane.
DROP_HEIGHT = TUBE_RADIUS + 0.01


def _add_chain_ring(builder: newton.ModelBuilder, body: int, *, static: bool = False) -> None:
    """Attach 6 capsules to ``body`` arranged around a ring of radius
    :data:`RING_RADIUS` in the body-local x-y plane (torus axis = body
    +z).

    Args:
        builder: Target :class:`newton.ModelBuilder`.
        body: Index of the rigid body the capsules attach to.
        static: When ``True``, capsules are added with ``density = 0``
            so they contribute no mass; combined with ``add_link``
            (which skips the auto-free-joint), the body's
            ``inverse_mass`` stays zero and PhoenX treats it as
            ``MOTION_STATIC``.
    """
    cfg: newton.ModelBuilder.ShapeConfig | None = None
    if static:
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
            gap=builder.default_shape_cfg.gap,
        )

    for i in range(SEGMENTS_PER_RING):
        angle = (2.0 * math.pi * i) / SEGMENTS_PER_RING
        cx = RING_RADIUS * math.cos(angle)
        cy = RING_RADIUS * math.sin(angle)

        # Capsule's default axis is body-local +z. We need it tangent to
        # the ring at (cx, cy, 0), i.e. along (-sin a, cos a, 0). Build
        # that as: first rotate -pi/2 about body +x to send +z -> +y,
        # then rotate by ``angle`` about body +z. Composed:
        #   q = q_z(angle) * q_x(-pi/2)
        qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)
        q = qz * qx

        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(cx, cy, 0.0), q=q),
            radius=TUBE_RADIUS,
            half_height=SEGMENT_HALF_HEIGHT,
            cfg=cfg,
        )


class Example(PortedExample):
    sim_substeps = 30
    solver_iterations = 5
    velocity_iterations = 1
    default_friction = 0.4
    start_paused = True
    broad_phase = "sap"
    step_layout = "single_world"
    # Skip contact arrows so ``viewer.log_state`` stays on ViewerGL's
    # CUDA-OpenGL interop path (no per-frame host sync).
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        builder.default_shape_cfg.gap = 0.01
        builder.add_ground_plane(height=-2.5)
        extents: list = []

        q_rot_x_90 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        ring_extent = (
            float(RING_RADIUS + TUBE_RADIUS),
            float(RING_RADIUS + TUBE_RADIUS),
            float(TUBE_RADIUS),
        )
        first_dynamic_ring: int | None = None
        for c in range(N_CHAINS):
            y = c * (2.0 * RING_SPACING)
            for i in range(N_RINGS):
                x = i * RING_SPACING
                q = q_rot_x_90 if (i % 2 == 1) else wp.quat_identity()
                # Static anchors: the first and last ring of the first
                # and last chain. ``add_link`` skips the auto-free-joint
                # and ``static=True`` zeroes the density of the
                # capsules, so the body keeps ``inverse_mass = 0`` and
                # PhoenX classifies it as MOTION_STATIC.
                is_anchor = (c in (0, N_CHAINS - 1)) and (i in (0, N_RINGS - 1))
                xform = wp.transform(p=wp.vec3(x, y, DROP_HEIGHT), q=q)
                if is_anchor:
                    body = builder.add_link(xform=xform)
                else:
                    body = builder.add_body(xform=xform)
                _add_chain_ring(builder, body, static=is_anchor)
                if not is_anchor and first_dynamic_ring is None:
                    first_dynamic_ring = body
                extents.append(None if is_anchor else ring_extent)

        # Big sphere falling onto the centre of the net.
        sphere_radius = 0.45
        net_cx = 0.5 * (N_RINGS - 1) * RING_SPACING
        net_cy = 0.5 * (N_CHAINS - 1) * 2.0 * RING_SPACING
        sphere_body = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(net_cx, net_cy, DROP_HEIGHT + sphere_radius + TUBE_RADIUS),
                q=wp.quat_identity(),
            ),
        )
        sphere_cfg = newton.ModelBuilder.ShapeConfig(
            density=100.0,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
            gap=builder.default_shape_cfg.gap,
        )
        builder.add_shape_sphere(sphere_body, radius=sphere_radius, cfg=sphere_cfg)
        extents.append((float(sphere_radius), float(sphere_radius), float(sphere_radius)))

        # Connection rings: one between every pair of adjacent chains,
        # at every second ring position along the chain. Sits at the
        # midpoint in y; rotated 90 degrees about world +y so the ring
        # plane stands vertical, threading through the flat chain rings
        # on either side.
        q_rot_y_90 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
        for c in range(N_CHAINS - 1):
            y_mid = (c + 0.5) * (2.0 * RING_SPACING)
            for i in range(0, N_RINGS, 2):
                x = i * RING_SPACING
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, y_mid, DROP_HEIGHT), q=q_rot_y_90),
                )
                _add_chain_ring(builder, body)
                extents.append(ring_extent)

        ring_mass = builder.body_mass[first_dynamic_ring] if first_dynamic_ring is not None else 0.0
        sphere_mass = builder.body_mass[sphere_body]
        g = abs(self.gravity[2])
        print(f"Ring weight: mass = {ring_mass:.4f} kg, weight = {ring_mass * g:.4f} N")
        print(f"Sphere weight: mass = {sphere_mass:.4f} kg, weight = {sphere_mass * g:.4f} N")

        return extents

    def configure_camera(self, viewer):
        chain_length = (N_RINGS - 1) * RING_SPACING
        chain_width = (N_CHAINS - 1) * 2.0 * RING_SPACING
        viewer.set_camera(
            pos=wp.vec3(0.5 * chain_length, 0.5 * chain_width - 1.6, 1.0),
            pitch=-25.0,
            yaw=90.0,
        )


if __name__ == "__main__":
    run_ported_example(Example)
