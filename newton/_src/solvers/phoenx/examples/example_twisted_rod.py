# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX twisted rod
#
# A slender elastic rod hangs between two fixed supports in a double-looped
# shape.  The centerline and material frames are assembled through the public
# newton.Rod API; its circular section and nylon-like material determine the
# joint stretch, shear, bend, and twist stiffnesses.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_twisted_rod
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_capsule_half_extents,
    run_ported_example,
)

ROD_RADIUS = 0.018
ROD_DENSITY = 1100.0
ROD_YOUNGS_MODULUS = 1.0e9
ROD_POISSONS_RATIO = 0.40

NUM_SEGMENTS = 96
SUPPORT_HALF_EXTENTS = (0.16, 0.20, 0.20)

ROD_COLORS = ((0.90, 0.82, 0.25), (0.30, 0.58, 0.63))
SUPPORT_COLOR = (0.72, 0.65, 0.57)


def _catmull_rom(control_points: np.ndarray, samples_per_span: int = 20) -> np.ndarray:
    """Sample an interpolating Catmull-Rom curve through control points."""
    padded = np.vstack((control_points[0], control_points, control_points[-1]))
    samples = []
    for span in range(len(control_points) - 1):
        p0, p1, p2, p3 = padded[span : span + 4]
        for value in np.linspace(0.0, 1.0, samples_per_span, endpoint=False):
            value2 = value * value
            value3 = value2 * value
            samples.append(
                0.5
                * (
                    2.0 * p1
                    + (-p0 + p2) * value
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * value2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * value3
                )
            )
    samples.append(control_points[-1])
    return np.asarray(samples, dtype=np.float64)


def _resample_curve(points: np.ndarray, segment_count: int) -> np.ndarray:
    """Resample a polyline at nearly uniform arc-length intervals."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], segment_count + 1)
    result = np.empty((segment_count + 1, 3), dtype=np.float32)
    for axis in range(3):
        result[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return result


def _double_loop_centerline() -> np.ndarray:
    """Return the smooth, slightly out-of-plane double-loop centerline."""
    control_points = np.asarray(
        [
            (-1.55, 0.00, 1.35),
            (-1.30, 0.00, 1.29),
            (-1.03, 0.01, 1.08),
            (-0.72, 0.05, 0.88),
            (-0.61, 0.07, 0.50),
            (-0.82, 0.04, 0.22),
            (-1.08, -0.03, 0.40),
            (-1.02, -0.07, 0.80),
            (-0.70, -0.05, 1.08),
            (-0.36, -0.01, 1.23),
            (-0.04, 0.01, 1.12),
            (0.19, 0.05, 0.89),
            (0.39, 0.07, 0.59),
            (0.28, 0.04, 0.30),
            (0.01, -0.03, 0.36),
            (-0.07, -0.07, 0.70),
            (0.17, -0.05, 1.00),
            (0.55, -0.01, 1.18),
            (0.94, 0.00, 1.27),
            (1.25, 0.00, 1.30),
            (1.55, 0.00, 1.35),
        ],
        dtype=np.float64,
    )
    return _resample_curve(_catmull_rom(control_points), NUM_SEGMENTS)


class Example(PortedExample):
    """Simulate a double-looped elastic rod suspended between supports."""

    fps = 60
    sim_substeps = 8
    solver_iterations = 8
    velocity_iterations = 1
    default_friction = 0.6
    broad_phase = "sap"
    step_layout = "single_world"
    shape_pairs_max = 16384
    show_contacts = False
    evaluate_fk = False
    step_report_label = "TwistedRod"

    def build_scene(self, builder: newton.ModelBuilder):
        """Build the material-defined rod and its two fixed supports."""
        points = _double_loop_centerline()
        rod = newton.Rod(
            points,
            radius=ROD_RADIUS,
            youngs_modulus=ROD_YOUNGS_MODULUS,
            poissons_ratio=ROD_POISSONS_RATIO,
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=ROD_DENSITY,
            mu=self.default_friction,
            restitution=0.0,
            gap=0.002,
        )
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=shape_cfg,
            stretch_damping=2.0,
            shear_damping=2.0,
            bend_damping=0.25,
            twist_damping=0.10,
            label="twisted_rod",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )
        self.rod_bodies = [int(body) for body in bodies]

        # The support-connected end capsules are kinematic anchors. Their rod
        # joints still transmit all four elastic modes to the dynamic span.
        self.anchor_bodies = (self.rod_bodies[0], self.rod_bodies[-1])
        for body in self.anchor_bodies:
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        # Alternating colors make the rod deformation and individual elements
        # readable, as in the reference rendering.
        for index, body in enumerate(self.rod_bodies):
            color = ROD_COLORS[(index // 2) % len(ROD_COLORS)]
            for shape in builder.body_shapes[body]:
                builder.shape_color[shape] = color

        static_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=self.default_friction,
            restitution=0.0,
            gap=0.002,
        )
        hx, hy, hz = SUPPORT_HALF_EXTENTS
        for index, point in enumerate((points[0], points[-1])):
            side = -1.0 if index == 0 else 1.0
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    wp.vec3(float(point[0] + side * hx), float(point[1]), float(point[2])),
                    wp.quat_identity(),
                ),
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=static_cfg,
                color=SUPPORT_COLOR,
                label=f"support_{index}",
            )

        self.initial_anchor_poses = np.asarray(
            [builder.body_q[body] for body in self.anchor_bodies],
            dtype=np.float32,
        )
        self.initial_centerline_min_z = float(points[:, 2].min())
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return [default_capsule_half_extents(ROD_RADIUS, 0.5 * float(length)) for length in segment_lengths]

    def configure_camera(self, viewer) -> None:
        """Frame both suspended loops and their supports."""
        viewer.set_camera(
            pos=wp.vec3(0.0, -4.2, 1.45),
            pitch=-2.0,
            yaw=90.0,
        )

    def test_final(self) -> None:
        """Verify the rod stays finite, bounded, and attached at both ends."""
        super().test_final()
        body_q = self.state.body_q.numpy()
        rod_q = body_q[self.rod_bodies]
        anchor_q = body_q[list(self.anchor_bodies)]

        np.testing.assert_allclose(anchor_q, self.initial_anchor_poses, rtol=0.0, atol=1.0e-5)
        if float(np.max(np.abs(rod_q[:, :3]))) > 4.0:
            raise AssertionError("twisted rod left the expected scene bounds")
        min_z = float(np.min(rod_q[:, 2]))
        if min_z < self.initial_centerline_min_z - 0.75:
            raise AssertionError(
                "twisted rod sagged beyond the supported span "
                f"(initial_min_z={self.initial_centerline_min_z:.3f}, final_min_z={min_z:.3f})"
            )


def _configure_parser(parser) -> None:
    parser.set_defaults(viewer="optix", num_frames=300)


if __name__ == "__main__":
    run_ported_example(Example, _configure_parser)
