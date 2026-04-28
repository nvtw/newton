# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Beam vs Cable
#
# Side-by-side bend-stiffness sweep comparing the new
# ``JointMode.BEAM`` against the existing ``JointMode.CABLE``. We spawn
# ``2 * NUM_STRANDS`` horizontal cantilevers, paired top-to-bottom along
# world +y: BEAM strands above, CABLE strands below, stepping through
# the same geometric ramp of bend stiffness in both halves. Each strand
# is a static anchor capsule on the left + ``SEGMENTS - 1`` dynamic
# capsules tiled along world +x; gravity points in -z.
#
# What you should see:
#
#   * At LOW bend stiffness (left in the sweep, blue), both BEAM and
#     CABLE strands sag heavily; the soft regime is similar.
#   * At MODERATE bend stiffness, BEAM holds shape better than CABLE
#     -- positional anchor-2 / anchor-3 rows have lever-arm
#     amplification that the angular log-map rows in CABLE lack.
#   * At HIGH bend stiffness (right, red), BEAM converges to a clean
#     near-rigid horizontal cantilever (matches a chain of REVOLUTE
#     joints), while CABLE struggles -- this is the regime BEAM was
#     designed for.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_beam_vs_cable
###########################################################################

from __future__ import annotations

import colorsys
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.picking import Picking, register_with_viewer_gl
from newton._src.solvers.phoenx.solver_phoenx import pack_body_xforms_kernel
from newton._src.solvers.phoenx.world_builder import JointMode, WorldBuilder

# Per-strand layout. The strand runs along world +x and is anchored at
# its leftmost capsule. ``SEGMENTS`` includes the static anchor.
SEGMENTS = 32
SPAN = 0.8

# Capsule geometry. Body-local +z is the strand axis.
SEGMENT_RADIUS = 0.005
SEGMENT_OVERLAP_FACTOR = 2.0
SEGMENT_MASS = 0.005

# Bend-stiffness sweep. Same shape as
# ``example_bend_stiffness_sweep`` so users can A/B straight across.
NUM_STRANDS = 5
BEND_STIFFNESS_MIN = 1.0e-2
BEND_STIFFNESS_MAX = 1.0e4
# Per-strand bend damping is derived from bend stiffness.
BEND_DAMPING_RATIO = 0.01

# Twist gains held constant across strands so the visual A/B isolates
# bend behaviour.
TWIST_STIFFNESS = 0.5
TWIST_DAMPING = 0.05

# World layout. Two banks of strands stacked in +y -- BEAM above, CABLE
# below. Per-bank, strands are spaced in +y by ``STRAND_SPACING_FACTOR``.
ANCHOR_HEIGHT = 1.2
STRAND_SPACING_FACTOR = 0.15
BANK_GAP_FACTOR = 0.5  # extra +y gap between BEAM bank and CABLE bank
GROUND_HEIGHT = 0.0
GRAVITY = 9.81

SIM_FPS = 240
RENDER_FPS = 60
SUBSTEPS = 30
SOLVER_ITERATIONS = 4
VELOCITY_ITERATIONS = 1
STEP_LAYOUT = "single_world"

_ANCHOR_COLOR = (0.85, 0.85, 0.85)


def _quat_from_z_axis(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
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


def _palette(num_strands: int) -> list[tuple[float, float, float]]:
    palette: list[tuple[float, float, float]] = []
    for i in range(max(num_strands, 1)):
        t = i / max(num_strands - 1, 1)
        hue = 0.6 * (1.0 - t)
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))
    return palette


def _log_ramp(num: int, vmin: float, vmax: float) -> list[float]:
    if num < 1:
        raise ValueError("num must be >= 1")
    if vmin <= 0.0 or vmax <= 0.0:
        raise ValueError("vmin and vmax must both be > 0")
    if num == 1:
        return [vmax]
    log_min = math.log(vmin)
    log_max = math.log(vmax)
    return [math.exp(log_min + (log_max - log_min) * i / (num - 1)) for i in range(num)]


class Example:
    """Side-by-side BEAM vs CABLE bend-stiffness sweep."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.sim_fps = int(getattr(args, "sim_fps", SIM_FPS))
        self.render_fps = int(getattr(args, "render_fps", RENDER_FPS))
        if self.sim_fps <= 0 or self.render_fps <= 0:
            raise ValueError("sim-fps and render-fps must both be > 0")
        if self.sim_fps % self.render_fps != 0:
            raise ValueError(
                f"sim-fps ({self.sim_fps}) must be an integer multiple of render-fps ({self.render_fps})"
            )
        self.sim_ticks_per_frame = self.sim_fps // self.render_fps
        self.frame_dt = 1.0 / self.sim_fps
        self.render_dt = 1.0 / self.render_fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "substeps", SUBSTEPS))
        self.solver_iterations = int(getattr(args, "solver_iterations", SOLVER_ITERATIONS))
        self.velocity_iterations = int(getattr(args, "velocity_iterations", VELOCITY_ITERATIONS))

        self._build_scene()

    def _build_scene(self) -> None:
        segments = int(self.args.segments)
        if segments < 3:
            raise ValueError("segments must be >= 3 (need >= 1 anchor + 2 dynamic capsules)")
        num_strands = int(self.args.num_strands)

        radius = float(self.args.segment_radius)
        span = float(self.args.span)
        if radius <= 0.0 or span <= 0.0:
            raise ValueError("segment-radius and span must both be > 0")

        pitch = span / segments
        overlap = float(self.args.segment_overlap_factor) * radius
        half_height = max(0.5 * (pitch + overlap) - radius, 0.0)

        anchor_height = float(self.args.anchor_height)
        ground_z = float(self.args.ground_height)
        if anchor_height - radius <= ground_z:
            raise ValueError(
                f"anchor-height ({anchor_height:.3f} m) must leave at least one capsule radius above ground"
            )

        gravity_g = float(self.args.gravity)

        mass = float(self.args.segment_mass)
        inv_mass = 1.0 / mass
        inv_inertia = _capsule_inverse_inertia(mass, radius, half_height)

        bend_stiffnesses = _log_ramp(
            num_strands,
            float(self.args.bend_stiffness_min),
            float(self.args.bend_stiffness_max),
        )
        damping_ratio = float(self.args.bend_damping_ratio)
        if damping_ratio < 0.0:
            raise ValueError("bend-damping-ratio must be >= 0")
        bend_dampings = [damping_ratio * k for k in bend_stiffnesses]
        palette = _palette(num_strands)

        builder = WorldBuilder()
        world_body = builder.world_body
        builder.add_shape_plane(world_body, normal=(0.0, 0.0, 1.0), offset=ground_z)

        orient_x = _quat_from_z_axis((1.0, 0.0, 0.0))

        spacing = float(self.args.strand_spacing_factor) * span
        bank_gap = float(self.args.bank_gap_factor) * span

        # Two banks: BEAM (top) and CABLE (bottom). Each bank centred
        # around its own y origin so the camera can frame both.
        cy_per_bank = (num_strands - 1) * 0.5
        bank_specs = [
            ("BEAM", JointMode.BEAM, +bank_gap),
            ("CABLE", JointMode.CABLE, -bank_gap),
        ]

        colors: list[tuple[float, float, float]] = []
        per_strand_dynamic_count = segments - 1
        total_chain_joints = 0

        for bank_label, mode_enum, y_bank_offset in bank_specs:
            for strand_idx in range(num_strands):
                y_strand = y_bank_offset + (strand_idx - cy_per_bank) * spacing
                strand_color = palette[strand_idx]

                anchor_x = -0.5 * span + 0.5 * pitch
                anchor_pos = (anchor_x, y_strand, anchor_height)
                anchor_bid = builder.add_static_body(position=anchor_pos, orientation=orient_x)
                builder.add_shape_capsule(anchor_bid, radius=radius, half_height=half_height)
                colors.append(_ANCHOR_COLOR)

                previous_bid = anchor_bid
                for k in range(1, segments):
                    x_centre = -0.5 * span + (k + 0.5) * pitch
                    pos = (x_centre, y_strand, anchor_height)
                    bid = builder.add_dynamic_body(
                        position=pos,
                        orientation=orient_x,
                        inverse_mass=inv_mass,
                        inverse_inertia=inv_inertia,
                        affected_by_gravity=True,
                    )
                    builder.add_shape_capsule(bid, radius=radius, half_height=half_height)
                    colors.append(strand_color)

                    anchor1 = (-0.5 * span + k * pitch, y_strand, anchor_height)
                    anchor2 = (anchor1[0] + 1.0, anchor1[1], anchor1[2])
                    builder.add_joint(
                        previous_bid,
                        bid,
                        anchor1=anchor1,
                        anchor2=anchor2,
                        mode=mode_enum,
                        bend_stiffness=float(bend_stiffnesses[strand_idx]),
                        twist_stiffness=float(self.args.twist_stiffness),
                        bend_damping=float(bend_dampings[strand_idx]),
                        twist_damping=float(self.args.twist_damping),
                    )
                    builder.add_collision_filter_pair(previous_bid, bid)
                    total_chain_joints += 1
                    previous_bid = bid

        expected_total = 2 * num_strands * per_strand_dynamic_count

        self.world = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -gravity_g),
            rigid_contact_max=0,
            step_layout=STEP_LAYOUT,
            device=self.device,
        )

        if int(self.world.num_joints) != expected_total:
            raise RuntimeError(
                f"PhoenX BeamVsCable: expected {expected_total} joints "
                f"(2 banks x {num_strands} strands x {per_strand_dynamic_count} per strand), "
                f"got {int(self.world.num_joints)}"
            )

        self._radius = radius
        self._half_height = half_height
        self._segments_per_strand = segments
        self._num_strands = num_strands
        self._span = span
        self._anchor_height = anchor_height
        self._bend_stiffnesses = bend_stiffnesses
        self._capsules_per_strand = segments
        self._num_capsules = 2 * num_strands * segments

        self._xforms = wp.zeros(self.world.num_bodies, dtype=wp.transform, device=self.device)
        self._capsule_xforms = self._xforms[1 : 1 + self._num_capsules]
        self._colors = wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3, device=self.device)

        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        half_extents_np[1 : 1 + self._num_capsules] = (radius, radius, half_height + radius)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        substep_dt = self.frame_dt / max(self.sim_substeps, 1)
        print(
            f"[PhoenX BeamVsCable] strands_per_bank={num_strands} segments/strand={segments} "
            f"span={span:.3f}m radius={radius * 1000.0:.1f}mm pitch={pitch:.4f}m "
            f"anchor_height={anchor_height:.3f}m gravity={gravity_g:.2f}m/s^2 "
            f"sim_fps={self.sim_fps}Hz render_fps={self.render_fps}Hz "
            f"substeps/tick={self.sim_substeps} substep_dt={substep_dt * 1000.0:.3f}ms"
        )
        print("  Bank layout (along world +y): BEAM (top, +y) | CABLE (bottom, -y).")
        print("  In each bank, strands run from low (blue) to high (red) bend stiffness.")
        for s, (k_bend, d_bend, color) in enumerate(zip(bend_stiffnesses, bend_dampings, palette, strict=False)):
            print(
                f"    strand[{s}] bend_stiffness={k_bend:.4g} N*m/rad "
                f"bend_damping={d_bend:.4g} N*m*s/rad "
                f"color=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
            )

        scene_y_extent = max(span, 2 * (cy_per_bank * spacing + bank_gap))
        cam_pos = wp.vec3(0.0, -1.6 * scene_y_extent, anchor_height + 0.1 * scene_y_extent)
        self.viewer.set_camera(pos=cam_pos, pitch=-15.0, yaw=90.0)

        self.graph = None
        if self.device.is_cuda and not bool(self.args.no_cuda_graph):
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        for _ in range(self.sim_ticks_per_frame):
            self.world.step(dt=self.frame_dt, contacts=None, shape_body=None, picking=self.picking)

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.render_dt

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=self.world.num_bodies,
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_shapes(
            "/world/beam_vs_cable",
            newton.GeoType.CAPSULE,
            (self._radius, self._half_height),
            self._capsule_xforms,
            colors=self._colors,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        assert np.isfinite(positions).all(), "non-finite body position in beam-vs-cable"
        capsule_z = positions[1 : 1 + self._num_capsules, 2]
        max_lateral = float(np.max(np.abs(positions[1 : 1 + self._num_capsules, :2])))
        assert max_lateral < 8.0 * self._span, (
            f"beam-vs-cable escaped its envelope (max_lateral={max_lateral:.3f})"
        )
        max_drop = float(self._anchor_height - np.min(capsule_z))
        assert max_drop < 4.0 * self._span, (
            f"beam-vs-cable produced an unreasonable drop (max_drop={max_drop:.3f})"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-strands",
        type=int,
        default=NUM_STRANDS,
        help="Number of strands per bank; the BEAM bank and the CABLE bank each get this many.",
    )
    parser.add_argument("--segments", type=int, default=SEGMENTS, help="Capsules per strand (incl. the static anchor).")
    parser.add_argument("--span", type=float, default=SPAN, help="Length of each horizontal cantilever [m].")
    parser.add_argument("--segment-radius", type=float, default=SEGMENT_RADIUS, help="Capsule radius [m].")
    parser.add_argument(
        "--segment-overlap-factor",
        type=float,
        default=SEGMENT_OVERLAP_FACTOR,
        help="Axial overlap between neighbouring capsules expressed as a multiple of segment-radius.",
    )
    parser.add_argument("--segment-mass", type=float, default=SEGMENT_MASS, help="Mass per dynamic capsule [kg].")
    parser.add_argument(
        "--bend-stiffness-min",
        type=float,
        default=BEND_STIFFNESS_MIN,
        help="Bend stiffness at the floppy end [N*m/rad].",
    )
    parser.add_argument(
        "--bend-stiffness-max",
        type=float,
        default=BEND_STIFFNESS_MAX,
        help="Bend stiffness at the rigid end [N*m/rad].",
    )
    parser.add_argument(
        "--bend-damping-ratio",
        type=float,
        default=BEND_DAMPING_RATIO,
        help="Per-strand bend damping = bend-damping-ratio * bend-stiffness.",
    )
    parser.add_argument(
        "--twist-stiffness",
        type=float,
        default=TWIST_STIFFNESS,
        help="Twist stiffness shared across all strands [N*m/rad].",
    )
    parser.add_argument(
        "--twist-damping",
        type=float,
        default=TWIST_DAMPING,
        help="Twist damping shared across all strands [N*m*s/rad].",
    )
    parser.add_argument(
        "--strand-spacing-factor",
        type=float,
        default=STRAND_SPACING_FACTOR,
        help="+y spacing between adjacent strands within a bank, as a multiple of --span.",
    )
    parser.add_argument(
        "--bank-gap-factor",
        type=float,
        default=BANK_GAP_FACTOR,
        help="+y gap between the BEAM bank and the CABLE bank, as a multiple of --span.",
    )
    parser.add_argument("--anchor-height", type=float, default=ANCHOR_HEIGHT, help="World z of the static anchor [m].")
    parser.add_argument("--ground-height", type=float, default=GROUND_HEIGHT, help="World z of the ground plane [m].")
    parser.add_argument("--gravity", type=float, default=GRAVITY, help="Downward gravity magnitude [m/s^2].")
    parser.add_argument("--sim-fps", type=int, default=SIM_FPS, help="Outer simulation tick rate [Hz].")
    parser.add_argument("--render-fps", type=int, default=RENDER_FPS, help="Rendered frames per second.")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS, help="PhoenX solver substeps per outer sim tick.")
    parser.add_argument("--solver-iterations", type=int, default=SOLVER_ITERATIONS, help="PhoenX PGS iterations.")
    parser.add_argument(
        "--velocity-iterations", type=int, default=VELOCITY_ITERATIONS, help="PhoenX velocity iterations."
    )
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture for the frame step.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
