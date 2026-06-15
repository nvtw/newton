# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Cable Lattice
#
# Large Cosserat-rod-style lattice built from rigid capsule segments joined
# by PhoenX cable joints. Each cable joint is a rigid ball socket with soft
# bend and twist rows (JointMode.CABLE).
#
# The defaults mirror the reference video layout: 16 stacked layers, each
# layer containing 75 strands, each strand containing 256 capsule segments.
# That is intentionally a stress scene, so the command-line arguments can
# shrink the lattice for quick iteration.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_cable_lattice
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.picking import Picking, register_with_viewer_gl
from newton._src.solvers.phoenx.solver_phoenx import pack_body_xforms_kernel
from newton._src.solvers.phoenx.world_builder import JointMode, WorldBuilder

# Reference-scene counts.
LAYERS = 1  # 16
STRANDS_PER_LAYER = 5  # 75
SEGMENTS_PER_STRAND = 256

# Segment geometry. The capsule is aligned with body-local +z and rotated
# into the strand direction by each body's orientation. Capsules are longer
# than their center-to-center pitch so neighboring links overlap and render as
# a continuous strand.
SEGMENT_PITCH = 0.012
SEGMENT_RADIUS = 0.005
SEGMENT_OVERLAP = 2.0 * SEGMENT_RADIUS
SEGMENT_HALF_HEIGHT = max(0.5 * (SEGMENT_PITCH + SEGMENT_OVERLAP) - SEGMENT_RADIUS, 0.0)
SEGMENT_MASS = 0.01

# Stiffness/damping for PhoenX cable mode. Bend applies in the two directions
# perpendicular to the strand axis; twist applies along the strand axis.
BEND_STIFFNESS = 0.02
TWIST_STIFFNESS = 0.02
BEND_DAMPING = 0.002
TWIST_DAMPING = 0.002

FPS = 60
SUBSTEPS = 20
SOLVER_ITERATIONS = 5
VELOCITY_ITERATIONS = 1
STEP_LAYOUT = "single_world"

_LAYER_PALETTE = (
    (0.15, 0.85, 0.95),  # cyan
    (0.95, 0.75, 0.22),  # yellow
    (0.85, 0.35, 0.58),  # rose
    (0.55, 0.48, 0.68),  # slate-purple
)


def _quat_from_z_axis(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return a quaternion rotating body-local +z onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm <= 1.0e-12:
        raise ValueError("direction must be non-zero")
    d /= norm

    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if dot > 1.0 - 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)

    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * math.acos(dot)
    s = math.sin(half_angle)
    return (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(math.cos(half_angle)))


def _capsule_inverse_inertia(
    mass: float,
    radius: float,
    half_height: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Approximate inverse inertia for a solid capsule aligned along +z."""
    cylinder_height = 2.0 * half_height
    sphere_volume = (4.0 / 3.0) * math.pi * radius**3
    cylinder_volume = math.pi * radius**2 * cylinder_height
    total_volume = sphere_volume + cylinder_volume
    if total_volume <= 0.0:
        raise ValueError("capsule volume must be positive")

    density = mass / total_volume
    ms = density * sphere_volume
    mc = density * cylinder_volume
    ia = mc * (0.25 * radius * radius + (1.0 / 12.0) * cylinder_height * cylinder_height) + ms * (
        0.4 * radius * radius + 0.375 * radius * cylinder_height + 0.25 * cylinder_height * cylinder_height
    )
    ib = (mc * 0.5 + ms * 0.4) * radius * radius
    return (
        (1.0 / ia, 0.0, 0.0),
        (0.0, 1.0 / ia, 0.0),
        (0.0, 0.0, 1.0 / ib),
    )


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


class Example:
    """PhoenX cable-joint lattice made from capsule segments."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = int(getattr(args, "fps", FPS))
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "substeps", SUBSTEPS))
        self.solver_iterations = int(getattr(args, "solver_iterations", SOLVER_ITERATIONS))
        self.velocity_iterations = int(getattr(args, "velocity_iterations", VELOCITY_ITERATIONS))

        self._build_scene()

    def _build_scene(self) -> None:
        layers = int(self.args.layers)
        strands_per_layer = int(self.args.strands_per_layer)
        segments_per_strand = int(self.args.segments_per_strand)
        if layers < 1 or strands_per_layer < 1 or segments_per_strand < 1:
            raise ValueError("layers, strands-per-layer, and segments-per-strand must all be >= 1")

        radius = float(self.args.segment_radius)
        pitch = float(self.args.segment_pitch)
        overlap = float(self.args.segment_overlap)
        if radius <= 0.0 or pitch <= 0.0 or overlap < 0.0:
            raise ValueError("segment-radius and segment-pitch must be > 0, and segment-overlap must be >= 0")
        half_height = max(0.5 * (pitch + overlap) - radius, 0.0)
        layer_spacing = float(self.args.layer_spacing if self.args.layer_spacing is not None else 2.4 * radius)
        # Make each layer a square sheet: 256 segments set one dimension,
        # 75 strands are spread across the other.
        span = segments_per_strand * pitch
        strand_spacing = float(
            self.args.strand_spacing if self.args.strand_spacing is not None else span / max(strands_per_layer - 1, 1)
        )

        inv_mass = 1.0 / float(self.args.segment_mass)
        inv_inertia = _capsule_inverse_inertia(float(self.args.segment_mass), radius, half_height)

        builder = WorldBuilder()
        world_body = builder.world_body
        segment_orientation = {
            "x": _quat_from_z_axis((1.0, 0.0, 0.0)),
            "y": _quat_from_z_axis((0.0, 1.0, 0.0)),
        }

        colors = np.empty((layers * strands_per_layer * segments_per_strand, 3), dtype=np.float32)
        segment_body_ids: list[int] = []
        color_i = 0
        pin_ends = not bool(self.args.no_pin_ends)
        affected_by_gravity = abs(float(self.args.gravity)) > 0.0
        num_strands = layers * strands_per_layer
        expected_inter_segment_joints = num_strands * (segments_per_strand - 1)
        expected_pin_joints = 2 * num_strands if pin_ends else 0
        expected_joints = expected_inter_segment_joints + expected_pin_joints

        for layer in range(layers):
            axis_name = "x" if layer % 2 == 0 else "y"
            direction = (1.0, 0.0, 0.0) if axis_name == "x" else (0.0, 1.0, 0.0)
            transverse = (0.0, 1.0, 0.0) if axis_name == "x" else (1.0, 0.0, 0.0)
            orientation = segment_orientation[axis_name]
            z = (layer - 0.5 * (layers - 1)) * layer_spacing + radius
            layer_color = _LAYER_PALETTE[layer % len(_LAYER_PALETTE)]

            for strand in range(strands_per_layer):
                transverse_offset = (strand - 0.5 * (strands_per_layer - 1)) * strand_spacing
                previous_body: int | None = None
                first_body: int | None = None
                last_body: int | None = None

                for segment in range(segments_per_strand):
                    longitudinal_offset = -0.5 * span + (segment + 0.5) * pitch
                    pos = _vec_add(
                        _vec_add(_vec_scale(direction, longitudinal_offset), _vec_scale(transverse, transverse_offset)),
                        (0.0, 0.0, z),
                    )
                    body = builder.add_dynamic_body(
                        position=pos,
                        orientation=orientation,
                        inverse_mass=inv_mass,
                        inverse_inertia=inv_inertia,
                        affected_by_gravity=affected_by_gravity,
                    )
                    builder.add_shape_capsule(body, radius=radius, half_height=half_height)
                    segment_body_ids.append(body)
                    colors[color_i] = layer_color
                    color_i += 1

                    if first_body is None:
                        first_body = body
                    if previous_body is not None:
                        anchor = _vec_add(
                            _vec_add(
                                _vec_scale(direction, -0.5 * span + segment * pitch),
                                _vec_scale(transverse, transverse_offset),
                            ),
                            (0.0, 0.0, z),
                        )
                        builder.add_joint(
                            previous_body,
                            body,
                            anchor1=anchor,
                            anchor2=_vec_add(anchor, direction),
                            mode=JointMode.CABLE,
                            bend_stiffness=float(self.args.bend_stiffness),
                            twist_stiffness=float(self.args.twist_stiffness),
                            bend_damping=float(self.args.bend_damping),
                            twist_damping=float(self.args.twist_damping),
                        )
                        builder.add_collision_filter_pair(previous_body, body)
                    previous_body = body
                    last_body = body

                if pin_ends and first_body is not None and last_body is not None:
                    start_anchor = _vec_add(
                        _vec_add(_vec_scale(direction, -0.5 * span), _vec_scale(transverse, transverse_offset)),
                        (0.0, 0.0, z),
                    )
                    end_anchor = _vec_add(
                        _vec_add(_vec_scale(direction, 0.5 * span), _vec_scale(transverse, transverse_offset)),
                        (0.0, 0.0, z),
                    )
                    for body, anchor in ((first_body, start_anchor), (last_body, end_anchor)):
                        builder.add_joint(
                            world_body,
                            body,
                            anchor1=anchor,
                            anchor2=_vec_add(anchor, direction),
                            mode=JointMode.CABLE,
                            bend_stiffness=float(self.args.bend_stiffness),
                            twist_stiffness=float(self.args.twist_stiffness),
                            bend_damping=float(self.args.bend_damping),
                            twist_damping=float(self.args.twist_damping),
                        )

        self.world = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -float(self.args.gravity)),
            rigid_contact_max=0,
            step_layout=STEP_LAYOUT,
            device=self.device,
        )
        self._xforms = wp.zeros(self.world.num_bodies, dtype=wp.transform, device=self.device)
        self._capsule_xforms = self._xforms[1:]
        self._colors = wp.array(colors, dtype=wp.vec3, device=self.device)

        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        half_extents_np[1:] = (radius, radius, half_height + radius)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        self._radius = radius
        self._half_height = half_height
        self._overlap = overlap
        self._span = span
        self._layers = layers
        self._strands_per_layer = strands_per_layer
        self._segments_per_strand = segments_per_strand
        self._num_segments = len(segment_body_ids)
        self._num_strands = layers * strands_per_layer

        if int(self.world.num_joints) != expected_joints:
            raise RuntimeError(
                f"PhoenX cable lattice: expected {expected_joints} joints "
                f"(={expected_inter_segment_joints} inter-segment + {expected_pin_joints} pin), "
                f"but the world reports {int(self.world.num_joints)}"
            )
        print(
            f"[PhoenX CableLattice] layers={layers} strands/layer={strands_per_layer} "
            f"segments/strand={segments_per_strand} bodies={self.world.num_bodies} "
            f"segments={self._num_segments} joints={self.world.num_joints} "
            f"(inter_segment={expected_inter_segment_joints}, pin={expected_pin_joints}) "
            f"span={span:.3f}m segment_overlap={overlap:.4f}m "
            f"pin_ends={'yes' if pin_ends else 'no'}"
        )

        self.viewer.set_camera(
            pos=wp.vec3(0.75 * span, -1.0 * span, 0.45 * span),
            pitch=-24.0,
            yaw=140.0,
        )

        self.graph = None
        if self.device.is_cuda and not bool(self.args.no_cuda_graph):
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        self.world.step(dt=self.frame_dt, contacts=None, shape_body=None, picking=self.picking)

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=self.world.num_bodies,
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_shapes(
            "/world/cable_lattice",
            newton.GeoType.CAPSULE,
            (self._radius, self._half_height),
            self._capsule_xforms,
            colors=self._colors,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        assert np.isfinite(positions).all(), "non-finite body position in cable lattice"
        envelope = max(self._span, 1.0) * 2.0
        max_abs = float(np.max(np.abs(positions[1:]))) if self._num_segments > 0 else 0.0
        assert max_abs < envelope, f"cable lattice escaped its spawn envelope (max_abs={max_abs:.3f})"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--layers", type=int, default=LAYERS, help="Number of stacked cable layers.")
    parser.add_argument(
        "--strands-per-layer", type=int, default=STRANDS_PER_LAYER, help="Parallel strands in each layer."
    )
    parser.add_argument(
        "--segments-per-strand",
        type=int,
        default=SEGMENTS_PER_STRAND,
        help="Capsule segments per strand.",
    )
    parser.add_argument(
        "--segment-pitch", type=float, default=SEGMENT_PITCH, help="Center-to-center segment spacing [m]."
    )
    parser.add_argument("--segment-radius", type=float, default=SEGMENT_RADIUS, help="Capsule radius [m].")
    parser.add_argument(
        "--segment-overlap",
        type=float,
        default=SEGMENT_OVERLAP,
        help="Intentional axial overlap between neighboring capsules [m].",
    )
    parser.add_argument("--segment-mass", type=float, default=SEGMENT_MASS, help="Mass per capsule segment [kg].")
    parser.add_argument(
        "--strand-spacing",
        type=float,
        default=None,
        help="Spacing between parallel strands [m]. Defaults to a square sheet.",
    )
    parser.add_argument(
        "--layer-spacing",
        type=float,
        default=None,
        help="Vertical spacing between layers [m]. Defaults to 2.4 * segment radius.",
    )
    parser.add_argument("--bend-stiffness", type=float, default=BEND_STIFFNESS, help="Cable bend stiffness [N*m/rad].")
    parser.add_argument(
        "--twist-stiffness", type=float, default=TWIST_STIFFNESS, help="Cable twist stiffness [N*m/rad]."
    )
    parser.add_argument("--bend-damping", type=float, default=BEND_DAMPING, help="Cable bend damping [N*m*s/rad].")
    parser.add_argument("--twist-damping", type=float, default=TWIST_DAMPING, help="Cable twist damping [N*m*s/rad].")
    parser.add_argument("--gravity", type=float, default=0.0, help="Downward gravity magnitude [m/s^2].")
    parser.add_argument("--fps", type=int, default=FPS, help="Render frames per second.")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS, help="PhoenX solver substeps per frame.")
    parser.add_argument("--solver-iterations", type=int, default=SOLVER_ITERATIONS, help="PhoenX PGS iterations.")
    parser.add_argument(
        "--velocity-iterations", type=int, default=VELOCITY_ITERATIONS, help="PhoenX velocity iterations."
    )
    parser.add_argument("--no-pin-ends", action="store_true", help="Do not pin strand endpoints to the static world.")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture for the frame step.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
