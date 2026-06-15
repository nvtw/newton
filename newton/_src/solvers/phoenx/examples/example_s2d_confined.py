# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# solver2d port: ``Contact / Confined``
#
# Direct replica of solver2d's ``Confined`` sample. 25 x 25 = 625
# circles (radius 0.5, diameter 1.0) packed at 0.72 center spacing --
# i.e. starting *overlapping* -- inside a rectangular cage with
# inner surfaces at x = +-10 and y in [0.5, 20]. Original is 2D: we
# build the same XY layout in a single Z slice and sandwich it
# between two static walls a tiny clearance above/below the spheres
# so they're constrained to one layer.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_s2d_confined
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

#: Original ``Confined`` constants verbatim.
R = 0.5
GRID_COUNT = 25
SPACING = 18.0 / GRID_COUNT  # = 0.72; sub-diameter -> spawn overlap.

#: Wall thickness (the original uses radius-0.5 capsules; box walls
#: are easier to read in 3D).
WALL_HE = 0.5

#: XY cage interior matches the 2D inner surfaces:
#:   x in [-10, 10] (capsule centerlines at +-10.5, radius 0.5)
#:   y in [0.5, 20] (capsule centerlines at y=0 and y=20.5, radius 0.5)
X_INNER_HALF = 10.0
Y_INNER_LO = 0.5
Y_INNER_HI = 20.0
Y_INNER_HALF = 0.5 * (Y_INNER_HI - Y_INNER_LO)
Y_INNER_MID = 0.5 * (Y_INNER_HI + Y_INNER_LO)

#: Z-axis sandwich: two static walls just clear of the sphere layer
#: (sphere centers sit at z=0, so radius + epsilon clearance keeps
#: them from interpenetrating the walls at spawn).
Z_CLEARANCE = R + 0.01


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 6
    gravity = (0.0, 0.0, 0.0)
    # 625 dynamic spheres in dense overlap tank O(N^2) NXN broad-phase;
    # switch to SAP.
    broad_phase = "sap"
    # Single big world: PhoenX's ``single_world`` layout drives the
    # global Jones-Plassmann colouring with per-colour persistent grid
    # launches via ``wp.capture_while`` -- the regime that wins for one
    # large world (vs. ``multi_world``'s per-world fast-tail kernels
    # tuned for thousands of small worlds).
    step_layout = "single_world"
    # Skip contact arrows so ``viewer.log_state`` stays on ViewerGL's
    # CUDA-OpenGL interop path (no per-frame host sync). With 625
    # spheres in dense overlap the arrows would be unreadable anyway.
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)

        # XY cage: 4 box walls aligned with the 2D capsule cage.
        # Inner surfaces at x = +- X_INNER_HALF and y = Y_INNER_LO /
        # Y_INNER_HI. Walls are tall enough on Z to seal the layer.
        wall_hz = Z_CLEARANCE + WALL_HE
        # Left / right walls (constant x).
        for sign in (-1.0, +1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    p=wp.vec3(sign * (X_INNER_HALF + WALL_HE), Y_INNER_MID, 0.0),
                    q=wp.quat_identity(),
                ),
                hx=WALL_HE,
                hy=Y_INNER_HALF + 2.0 * WALL_HE,
                hz=wall_hz,
                cfg=wall_cfg,
            )
        # Bottom (y = Y_INNER_LO) and top (y = Y_INNER_HI) walls.
        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, Y_INNER_LO - WALL_HE, 0.0), q=wp.quat_identity()),
            hx=X_INNER_HALF,
            hy=WALL_HE,
            hz=wall_hz,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, Y_INNER_HI + WALL_HE, 0.0), q=wp.quat_identity()),
            hx=X_INNER_HALF,
            hy=WALL_HE,
            hz=wall_hz,
            cfg=wall_cfg,
        )

        # Z sandwich: two large flat walls clamping the sphere layer
        # to z = 0. Span the whole XY interior so spheres can't squeeze
        # past them sideways during the initial overlap-recovery
        # explosion.
        for sign in (-1.0, +1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    p=wp.vec3(0.0, Y_INNER_MID, sign * (Z_CLEARANCE + WALL_HE)),
                    q=wp.quat_identity(),
                ),
                hx=X_INNER_HALF + 2.0 * WALL_HE,
                hy=Y_INNER_HALF + 2.0 * WALL_HE,
                hz=WALL_HE,
                cfg=wall_cfg,
            )

        # 25 x 25 spheres at the *original* positions:
        #   x = -8.75 + col * 0.72
        #   y =  1.5  + row * 0.72
        # All in the single Z = 0 layer.
        extents: list = []
        for col in range(GRID_COUNT):
            x = -8.75 + col * SPACING
            for row in range(GRID_COUNT):
                y = 1.5 + row * SPACING
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, y, 0.0), q=wp.quat_identity()),
                )
                builder.add_shape_sphere(body, radius=R)
                extents.append(default_sphere_half_extents(R))
        return extents

    def configure_camera(self, viewer):
        # Look along +z down onto the XY layer; centre on the cage.
        viewer.set_camera(
            pos=wp.vec3(0.0, Y_INNER_MID, 28.0),
            pitch=-89.0,
            yaw=90.0,
        )


if __name__ == "__main__":
    run_ported_example(Example)
