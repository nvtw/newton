# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# solver2d port: ``Contact / Rush`` (3D)
#
# 3D extension of solver2d's planar ``Rush`` sample: spheres are
# arranged on Archimedean spirals stacked along Z, around a static
# central sphere. No gravity, no initial velocity. Each step we apply
# a constant-magnitude inward radial force toward the origin (full
# XYZ, not just XY) to every dynamic body; bodies closer than 0.1
# from the origin are skipped. Tests broad-phase + warm-start on a
# very dense, near-overlapping 3D cluster collapsing inward.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_s2d_rush
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

R = 0.5
#: Spheres per spiral layer.
N_PER_LAYER = 200
#: Number of stacked spiral layers along Z.
NUM_LAYERS = 5
#: Vertical spacing between layers; slightly more than the sphere
#: diameter so adjacent layers don't start interpenetrating.
LAYER_DZ = 1.1 * (2.0 * R)
RUSH_FORCE = 1000.0  # newtons; magnitude of the inward radial pull
RUSH_DEAD_RADIUS = 0.1  # bodies closer than this to the origin are skipped


@wp.kernel(enable_backward=False)
def _apply_rush_force_kernel(
    bodies: BodyContainer,
    first_body: wp.int32,
    last_body: wp.int32,
    force_mag: wp.float32,
    dead_radius: wp.float32,
):
    """Add a constant-magnitude inward radial force toward the origin
    (full XYZ) to every dynamic body in ``[first_body, last_body)``.

    Generalises solver2d's ``Rush::Step`` force branch from 2D to 3D:
    for each body ``i``, ``f = -(force_mag / |p|) * p``, skipping
    bodies whose distance to the origin is below ``dead_radius``.
    Force is additive (atomic) so it stacks with picking forces if
    the user grabs a body."""
    tid = wp.tid()
    bid = first_body + tid
    if bid >= last_body:
        return

    p = bodies.position[bid]
    r = wp.length(p)
    if r < dead_radius:
        return

    f = p * (-force_mag / r)
    wp.atomic_add(bodies.force, bid, f)


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 4
    gravity = (0.0, 0.0, 0.0)
    # 1000 dynamic spheres tank O(N^2) NXN broad-phase; switch to SAP.
    broad_phase = "sap"
    # Single big world: PhoenX's ``single_world`` layout drives the
    # global Jones-Plassmann colouring with per-colour persistent grid
    # launches via ``wp.capture_while`` -- the regime that wins for one
    # large world (vs. ``multi_world``'s per-world fast-tail kernels
    # tuned for thousands of small worlds).
    step_layout = "single_world"
    # Skip contact arrows so ``viewer.log_state`` stays on ViewerGL's
    # CUDA-OpenGL interop path (no per-frame host sync). With ~1000
    # spheres in dense contact the arrows would be unreadable anyway.
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        # No ground -- this is a free-floating cluster.
        extents: list = []
        # Static central sphere (anchor).
        builder.add_shape_sphere(
            -1,
            xform=wp.transform_identity(),
            radius=R,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        # Stack ``NUM_LAYERS`` copies of the 2D spiral along Z, centred
        # on z = 0. ``layer_offset`` flips sign every other layer so the
        # spirals don't all wind in phase (slightly less symmetric, more
        # interesting collapse).
        z0 = -0.5 * (NUM_LAYERS - 1) * LAYER_DZ
        for layer in range(NUM_LAYERS):
            z = z0 + layer * LAYER_DZ
            distance = 5.0
            delta_angle = 1.0 / distance
            delta_distance = 0.05
            angle = 0.5 * math.pi * layer  # phase offset per layer
            for _ in range(N_PER_LAYER):
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                )
                cfg = newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.2)
                builder.add_shape_sphere(body, radius=R, cfg=cfg)
                extents.append(default_sphere_half_extents(R))
                angle += delta_angle
                distance += delta_distance
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -40.0, 30.0), pitch=-30.0, yaw=90.0)

    def simulate(self) -> None:
        # Inline the base pipeline so the rush force is written into
        # ``bodies.force`` *after* :meth:`_sync_newton_to_phoenx`
        # refreshes ``bodies.position`` (otherwise we'd compute the
        # radial direction from last-frame positions). Order mirrors
        # :meth:`PortedExample.simulate` plus our extra force kernel
        # right before ``world.step``.
        self._sync_newton_to_phoenx()
        self.model.collide(self.state, contacts=self.contacts, collision_pipeline=self.collision_pipeline)
        n = self.model.body_count
        if n > 0:
            # PhoenX body indices are Newton body index + 1 (slot 0 is
            # the world anchor); the ``N`` dynamic spheres live
            # contiguously in ``[1, 1 + body_count)``.
            wp.launch(
                _apply_rush_force_kernel,
                dim=n,
                inputs=[
                    self.bodies,
                    wp.int32(1),
                    wp.int32(1 + n),
                    wp.float32(RUSH_FORCE),
                    wp.float32(RUSH_DEAD_RADIUS),
                ],
                device=self.device,
            )
        self.picking.apply_force()
        self.world.step(dt=self.frame_dt, contacts=self.contacts, shape_body=self._shape_body)
        self._sync_phoenx_to_newton()


if __name__ == "__main__":
    run_ported_example(Example)
