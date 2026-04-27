# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Double Domino`` (spiral variant)
#
# 3000 dominoes laid out along an Archimedean spiral in the X-Y plane.
# Each domino is rotated so its thin face (the one that gets hit) points
# along the spiral tangent. The first domino is kicked forward (rotating
# about its center-bottom edge, see derivation below) and topples into
# its neighbour, triggering the spiral-shaped chain.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_double_domino
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

HX = 0.04
HY = 0.4
HZ = 0.6
N_DOMINOES = 3000
SPACING = 0.6

#: Spiral parameters. ``r = R0 + B * theta`` with ``B`` chosen so each
#: full turn moves outward by 1.5 m -- comfortably larger than the
#: domino depth (2 * HY = 0.8 m) so adjacent arms can't collide. ``R0``
#: is the inner radius (start of the spiral); kept large enough to give
#: the first few dominoes enough room not to bump the second turn.
SPIRAL_R0 = 5.0
SPIRAL_ARM_GAP = 1.5
SPIRAL_B = SPIRAL_ARM_GAP / (2.0 * math.pi)

#: Tip-over angular velocity [rad/s] about the spiral-tangent's left
#: normal in the X-Y plane. Picked so the first domino rotates forward
#: fast enough to reach its neighbour.
TIP_OMEGA = 4.0


def _arc_length(theta: float) -> float:
    """Closed-form arc length of ``r = R0 + B*theta`` from 0 to theta.

    s(theta) = (1/(2B)) * (r * sqrt(r^2 + B^2) + B^2 * asinh(r/B)
                            - R0 * sqrt(R0^2 + B^2) - B^2 * asinh(R0/B))
    where r = R0 + B*theta. Derived by integrating sqrt(r^2 + B^2).
    """
    r = SPIRAL_R0 + SPIRAL_B * theta
    f_upper = r * math.sqrt(r * r + SPIRAL_B * SPIRAL_B) + SPIRAL_B * SPIRAL_B * math.asinh(r / SPIRAL_B)
    f_lower = SPIRAL_R0 * math.sqrt(SPIRAL_R0 * SPIRAL_R0 + SPIRAL_B * SPIRAL_B) + SPIRAL_B * SPIRAL_B * math.asinh(
        SPIRAL_R0 / SPIRAL_B
    )
    return (f_upper - f_lower) / (2.0 * SPIRAL_B)


def _spiral_pose(s: float) -> tuple[float, float, float]:
    """Position and yaw of a domino at arc length ``s`` along the spiral.

    Returns ``(x, y, yaw)`` where yaw is the angle (about world +z) of
    the spiral tangent at that point.
    """
    # Closed-form quadratic seed (ignores the +B^2 term under the
    # square root, exact when B << r). Then a few Newton iterations on
    # the exact arc-length integral nail the residual to machine
    # precision.
    discriminant = SPIRAL_R0 * SPIRAL_R0 + 2.0 * SPIRAL_B * s
    theta = (-SPIRAL_R0 + math.sqrt(discriminant)) / SPIRAL_B
    for _ in range(6):
        residual = _arc_length(theta) - s
        if abs(residual) < 1e-9:
            break
        r = SPIRAL_R0 + SPIRAL_B * theta
        ds_dtheta = math.sqrt(r * r + SPIRAL_B * SPIRAL_B)
        theta -= residual / ds_dtheta
    r = SPIRAL_R0 + SPIRAL_B * theta
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    # Tangent direction: d/dtheta (r cos theta, r sin theta).
    tx = SPIRAL_B * math.cos(theta) - r * math.sin(theta)
    ty = SPIRAL_B * math.sin(theta) + r * math.cos(theta)
    yaw = math.atan2(ty, tx)
    return x, y, yaw


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 4
    default_friction = 0.6
    # 3000 bodies makes O(N^2) NXN broad-phase prohibitive; switch to SAP.
    broad_phase = "sap"
    # Single big world: PhoenX's ``single_world`` layout drives the
    # global Jones-Plassmann colouring with per-colour persistent grid
    # launches via ``wp.capture_while`` -- the regime that wins for one
    # large world (vs. ``multi_world``'s per-world fast-tail kernels
    # tuned for thousands of small worlds).
    step_layout = "single_world"
    # Skip contact arrows so ``viewer.log_state`` stays on ViewerGL's
    # CUDA-OpenGL interop path (no per-frame host sync). With 3000
    # bricks the arrows would be unreadable anyway.
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        for i in range(N_DOMINOES):
            x, y, yaw = _spiral_pose(i * SPACING)
            # Yaw quaternion about world +z.
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)

            # Kick the first domino so its center-bottom edge stays
            # instantaneously at rest. With angular velocity ``omega``
            # about the spiral's *left-normal* axis (perpendicular to
            # tangent, in the X-Y plane) and pivot offset r = (0, 0, -HZ)
            # from the COM, v_pivot = v_com + omega x r. The COM
            # velocity needed for v_pivot = 0 is therefore HZ * omega
            # rotated 90 deg from the omega direction in the X-Y plane,
            # which conveniently equals HZ * TIP_OMEGA along the spiral
            # tangent.
            if i == 0:
                tx, ty = math.cos(yaw), math.sin(yaw)
                # Left-normal of the tangent (rotate tangent +90 deg
                # about world +z) is the toppling-rotation axis.
                nx, ny = -ty, tx
                lin_v: tuple[float, float, float] | None = (
                    TIP_OMEGA * HZ * tx,
                    TIP_OMEGA * HZ * ty,
                    0.0,
                )
                ang_v: tuple[float, float, float] | None = (
                    TIP_OMEGA * nx,
                    TIP_OMEGA * ny,
                    0.0,
                )
            else:
                lin_v = None
                ang_v = None
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, y, HZ + 0.02), q=q),
                linear_velocity=lin_v,
                angular_velocity=ang_v,
            )
            builder.add_shape_box(body, hx=HX, hy=HY, hz=HZ)
            extents.append(default_box_half_extents(HX, HY, HZ))
        return extents

    def configure_camera(self, viewer):
        # Outer radius of the spiral after N_DOMINOES bricks (closed-form
        # solution of the quadratic R0 + B*theta = sqrt(R0^2 + 2*B*s)).
        r_outer = math.sqrt(SPIRAL_R0 * SPIRAL_R0 + 2.0 * SPIRAL_B * N_DOMINOES * SPACING)
        viewer.set_camera(pos=wp.vec3(0.0, -r_outer * 1.4, r_outer * 0.9), pitch=-35.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
