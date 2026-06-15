# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Bend-Stiffness Sweep
#
# Educational A/B/C/... scene for getting a feel for the PhoenX
# ``JointMode.CABLE`` bend-stiffness knob. We spawn ``NUM_STRANDS``
# horizontal cantilevers stacked along world +y, each strand identical
# except for its per-cable-joint ``bend_stiffness``, and let gravity
# pull them down. The leftmost capsule of every strand is a static
# anchor (zero inverse mass), so each strand is a single-ended
# cantilever rather than a hanging chord -- that's deliberately the
# layout where bend stiffness matters most:
#
#   * Very high bend_stiffness  -> strand stays nearly horizontal.
#   * Moderate bend_stiffness   -> strand sags into a smooth curve.
#   * Very low bend_stiffness   -> strand drapes straight down from the
#                                  anchor (collapses onto the +z axis).
#
# Twist stiffness / damping and bend damping are held constant across
# strands so the visual comparison isolates bend stiffness.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_bend_stiffness_sweep
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
# its leftmost capsule. ``SEGMENTS`` includes the static anchor, so the
# number of dynamic capsules per strand is ``SEGMENTS - 1``.
SEGMENTS = 64
SPAN = 0.8  # length of the cantilever [m] (anchor to free tip when straight)

# Capsule geometry. Body-local +z is the strand axis; capsules overlap
# their neighbours along the chord so the strand reads as continuous.
SEGMENT_RADIUS = 0.005  # 10 mm rope diameter
SEGMENT_OVERLAP_FACTOR = 2.0
SEGMENT_MASS = 0.005

# Bend-stiffness sweep. Strands get bend_stiffness values geometrically
# ramped between ``BEND_STIFFNESS_MIN`` and ``BEND_STIFFNESS_MAX``.
#
# Sizing rationale: at the default geometry (64 segments, 5 g per
# segment, 0.8 m span) the static torque the cantilever loads on its
# root joint is roughly
#
#   M_grav = 0.5 * (N-1) * N * m_seg * g * pitch
#          ~= 0.5 * 63 * 64 * 0.005 * 9.81 * 0.0125
#          ~= 1.2 N*m
#
# so to hold the strand within e.g. 5 deg of horizontal you need a
# bend stiffness on the order of M_grav / 0.087 ~= 14 N*m/rad. The
# sweep therefore spans 4 decades centred on that scale: 1e-2 (limp
# rope) -> 1e2 (effectively rigid). Damping is scaled with the same
# range so each strand stays critically damped enough to settle in
# under a second.
NUM_STRANDS = 6
BEND_STIFFNESS_MIN = 1.0e-2
BEND_STIFFNESS_MAX = 1.0e2
# Per-strand bend damping is derived from the strand's bend stiffness
# as ``damping = BEND_DAMPING_RATIO * stiffness``. 0.3 keeps every
# strand slightly sub-critical (one overshoot, settle within ~1 s)
# instead of the long over-damped crawl you get when damping is
# large relative to ``stiffness * dt``. Raise this toward 1.0+ for a
# calmer settle; lower it (e.g. 0.05) to see pronounced ringing. The
# ratio is unitless: damping has units N*m*s/rad and stiffness has
# units N*m/rad, so the ratio carries an implicit ``[s]`` time scale.
BEND_DAMPING_RATIO = 0.3

# Twist stiffness / damping are held constant across strands so the
# user can stare at a single variable. Twist doesn't have anything to
# do here (no motors), but a small twist stiffness keeps the rest pose
# from drifting under solver noise.
TWIST_STIFFNESS = 0.5
TWIST_DAMPING = 0.05

# World layout. The strand anchors sit at ``ANCHOR_HEIGHT`` above the
# ground; we stack the strands along +y so all of them are visible side
# by side. Spacing is sized off ``SPAN`` so neighbouring cantilevers
# don't collide.
ANCHOR_HEIGHT = 1.2
STRAND_SPACING_FACTOR = 0.15  # multiplier on SPAN for the +y offset between strands
GROUND_HEIGHT = 0.0
GRAVITY = 9.81

# Time stepping. Same ``SIM_FPS / RENDER_FPS / SUBSTEPS`` idiom as
# ``example_twisted_thread`` and ``example_phoenx_scale``: outer ticks
# at 240 Hz, ``SIM_FPS / RENDER_FPS`` ticks per rendered frame,
# ``SUBSTEPS`` solver substeps per outer tick.
SIM_FPS = 240
RENDER_FPS = 60
SUBSTEPS = 20
SOLVER_ITERATIONS = 6
VELOCITY_ITERATIONS = 1
STEP_LAYOUT = "single_world"

_ANCHOR_COLOR = (0.85, 0.85, 0.85)


def _quat_from_z_axis(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Quaternion rotating body-local +z onto ``direction`` (unit vector)."""
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
    """Inverse inertia for a solid capsule along body-local +z."""
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


def _bend_stiffness_palette(num_strands: int) -> list[tuple[float, float, float]]:
    """HSV ramp from cool blue (lowest stiffness) to warm red (highest).

    Aligns the colour with the eye-balled "softness" of the strand:
    blue strands are floppy, red strands are rigid.
    """
    palette: list[tuple[float, float, float]] = []
    for i in range(max(num_strands, 1)):
        t = i / max(num_strands - 1, 1)
        # Hue runs from 0.6 (blue) at t=0 to 0.0 (red) at t=1. Slight
        # saturation/value floor so dark blues remain visible.
        hue = 0.6 * (1.0 - t)
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))
    return palette


def _log_ramp(num_strands: int, vmin: float, vmax: float, name: str) -> list[float]:
    """Geometric (log-spaced) ramp of values for the strand sweep."""
    if num_strands < 1:
        raise ValueError("num-strands must be >= 1")
    if vmin <= 0.0 or vmax <= 0.0:
        raise ValueError(f"{name}-min and {name}-max must both be > 0")
    if num_strands == 1:
        return [vmax]
    log_min = math.log(vmin)
    log_max = math.log(vmax)
    return [math.exp(log_min + (log_max - log_min) * i / (num_strands - 1)) for i in range(num_strands)]


def _bend_stiffness_values(num_strands: int, k_min: float, k_max: float) -> list[float]:
    """Backward-compatible alias for the bend-stiffness ramp."""
    return _log_ramp(num_strands, k_min, k_max, "bend-stiffness")


class Example:
    """Side-by-side cantilevers spanning a bend-stiffness sweep."""

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
                f"sim-fps ({self.sim_fps}) must be an integer multiple of render-fps "
                f"({self.render_fps}) so each rendered frame contains a whole number of sim ticks"
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

        # ``segments`` capsules tile the chord end-to-end; ``pitch`` is
        # their centre-to-centre spacing along world +x.
        pitch = span / segments
        overlap = float(self.args.segment_overlap_factor) * radius
        half_height = max(0.5 * (pitch + overlap) - radius, 0.0)

        anchor_height = float(self.args.anchor_height)
        ground_z = float(self.args.ground_height)
        if anchor_height - radius <= ground_z:
            raise ValueError(
                f"anchor-height ({anchor_height:.3f} m) must leave at least one capsule radius "
                f"({radius:.4f} m) above the ground at z={ground_z:.3f} m"
            )

        gravity_g = float(self.args.gravity)

        mass = float(self.args.segment_mass)
        inv_mass = 1.0 / mass
        inv_inertia = _capsule_inverse_inertia(mass, radius, half_height)

        bend_stiffnesses = _log_ramp(
            num_strands,
            float(self.args.bend_stiffness_min),
            float(self.args.bend_stiffness_max),
            "bend-stiffness",
        )
        damping_ratio = float(self.args.bend_damping_ratio)
        if damping_ratio < 0.0:
            raise ValueError("bend-damping-ratio must be >= 0")
        bend_dampings = [damping_ratio * k for k in bend_stiffnesses]
        palette = _bend_stiffness_palette(num_strands)

        builder = WorldBuilder()
        world_body = builder.world_body
        builder.add_shape_plane(world_body, normal=(0.0, 0.0, 1.0), offset=ground_z)

        orient_x = _quat_from_z_axis((1.0, 0.0, 0.0))

        spacing = float(self.args.strand_spacing_factor) * span
        # Centre the stack of strands around y=0 so the camera frame
        # stays symmetric.
        cy = (num_strands - 1) * 0.5

        colors: list[tuple[float, float, float]] = []
        per_strand_dynamic_count = segments - 1
        total_chain_joints = 0

        for strand_idx in range(num_strands):
            y_strand = (strand_idx - cy) * spacing
            strand_color = palette[strand_idx]

            # Anchor capsule (static). Centred at (-0.5*span + 0.5*pitch, y, z).
            anchor_x = -0.5 * span + 0.5 * pitch
            anchor_pos = (anchor_x, y_strand, anchor_height)
            anchor_bid = builder.add_static_body(position=anchor_pos, orientation=orient_x)
            builder.add_shape_capsule(anchor_bid, radius=radius, half_height=half_height)
            colors.append(_ANCHOR_COLOR)

            # Dynamic capsules tile the rest of the chord.
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

                # Cable joint at the shared capsule end-point. The
                # twist axis (``anchor1 -> anchor2``) starts horizontal
                # along world +x; the cable joint snapshots that as
                # the rest pose, so under gravity the strand bends
                # away from horizontal and the bend rows pull back.
                anchor1 = (-0.5 * span + k * pitch, y_strand, anchor_height)
                anchor2 = (anchor1[0] + 1.0, anchor1[1], anchor1[2])
                builder.add_joint(
                    previous_bid,
                    bid,
                    anchor1=anchor1,
                    anchor2=anchor2,
                    mode=JointMode.CABLE,
                    bend_stiffness=float(bend_stiffnesses[strand_idx]),
                    twist_stiffness=float(self.args.twist_stiffness),
                    bend_damping=float(bend_dampings[strand_idx]),
                    twist_damping=float(self.args.twist_damping),
                )
                builder.add_collision_filter_pair(previous_bid, bid)
                total_chain_joints += 1
                previous_bid = bid

        expected_total = num_strands * per_strand_dynamic_count

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
                f"PhoenX BendStiffnessSweep: expected {expected_total} cable joints "
                f"({num_strands} strands x {per_strand_dynamic_count} per strand), "
                f"but the world reports {int(self.world.num_joints)}"
            )

        self._radius = radius
        self._half_height = half_height
        self._segments_per_strand = segments
        self._num_strands = num_strands
        self._span = span
        self._anchor_height = anchor_height
        self._bend_stiffnesses = bend_stiffnesses
        # Capsule slot layout: 1 anchor + (segments - 1) dynamic per
        # strand, with strands appended in order. Slot 0 is the world
        # body.
        self._capsules_per_strand = segments
        self._num_capsules = num_strands * segments

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
            f"[PhoenX BendStiffnessSweep] strands={num_strands} segments/strand={segments} "
            f"span={span:.3f}m radius={radius * 1000.0:.1f}mm pitch={pitch:.4f}m "
            f"anchor_height={anchor_height:.3f}m gravity={gravity_g:.2f}m/s^2 "
            f"sim_fps={self.sim_fps}Hz render_fps={self.render_fps}Hz "
            f"sim_ticks/frame={self.sim_ticks_per_frame} substeps/tick={self.sim_substeps} "
            f"substep_dt={substep_dt * 1000.0:.3f}ms"
        )
        for s, (k_bend, d_bend, color) in enumerate(zip(bend_stiffnesses, bend_dampings, palette, strict=False)):
            print(
                f"  strand[{s}] bend_stiffness={k_bend:.4g} N*m/rad "
                f"bend_damping={d_bend:.4g} N*m*s/rad "
                f"(ratio={damping_ratio:.3f}) "
                f"color=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
            )

        # Camera framing: look along -y at the centred stack. The
        # cantilevers will droop in -z, so we pull the camera up a bit
        # to keep both anchors and free tips visible.
        scene_y_extent = max(span, (num_strands - 1) * spacing)
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
            "/world/bend_sweep",
            newton.GeoType.CAPSULE,
            (self._radius, self._half_height),
            self._capsule_xforms,
            colors=self._colors,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        assert np.isfinite(positions).all(), "non-finite body position in bend-stiffness sweep"
        capsule_z = positions[1 : 1 + self._num_capsules, 2]
        # ``rigid_contact_max=0`` means the strand passes through the
        # plane visually if it falls too far, so we only check that
        # nothing has shot off to infinity laterally and that the
        # vertical drop stays bounded.
        max_lateral = float(np.max(np.abs(positions[1 : 1 + self._num_capsules, :2])))
        assert max_lateral < 8.0 * self._span, (
            f"bend-stiffness sweep escaped its envelope (max_lateral={max_lateral:.3f})"
        )
        max_drop = float(self._anchor_height - np.min(capsule_z))
        # Even the floppiest strand can drop at most one strand length
        # plus a few capsule lengths of overshoot; any more means the
        # solver blew up.
        assert max_drop < 4.0 * self._span, (
            f"bend-stiffness sweep produced an unreasonable drop (max_drop={max_drop:.3f})"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-strands",
        type=int,
        default=NUM_STRANDS,
        help="Number of side-by-side cantilevers, each spanning a different bend-stiffness value.",
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
        help="Bend stiffness at the floppy end of the sweep [N*m/rad]; geometrically ramped to --bend-stiffness-max.",
    )
    parser.add_argument(
        "--bend-stiffness-max",
        type=float,
        default=BEND_STIFFNESS_MAX,
        help="Bend stiffness at the rigid end of the sweep [N*m/rad].",
    )
    parser.add_argument(
        "--bend-damping-ratio",
        type=float,
        default=BEND_DAMPING_RATIO,
        help=(
            "Per-strand bend damping = bend-damping-ratio * bend-stiffness. ~0.3 settles "
            "with one overshoot; ~1+ is over-damped; <0.1 rings."
        ),
    )
    parser.add_argument(
        "--twist-stiffness",
        type=float,
        default=TWIST_STIFFNESS,
        help="Twist stiffness shared across all strands [N*m/rad]; held constant to isolate bend.",
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
        help="+y spacing between adjacent strands as a multiple of --span.",
    )
    parser.add_argument("--anchor-height", type=float, default=ANCHOR_HEIGHT, help="World z of the static anchor [m].")
    parser.add_argument("--ground-height", type=float, default=GROUND_HEIGHT, help="World z of the ground plane [m].")
    parser.add_argument("--gravity", type=float, default=GRAVITY, help="Downward gravity magnitude [m/s^2].")
    parser.add_argument(
        "--sim-fps",
        type=int,
        default=SIM_FPS,
        help="Outer simulation tick rate [Hz]; one PhoenX world.step per tick. Must be a multiple of --render-fps.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=RENDER_FPS,
        help="Rendered frames per second; sim runs (sim-fps / render-fps) ticks per rendered frame.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=SUBSTEPS,
        help="PhoenX solver substeps per outer sim tick (each substep dt = 1 / (sim-fps * substeps)).",
    )
    parser.add_argument("--solver-iterations", type=int, default=SOLVER_ITERATIONS, help="PhoenX PGS iterations.")
    parser.add_argument(
        "--velocity-iterations", type=int, default=VELOCITY_ITERATIONS, help="PhoenX velocity iterations."
    )
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture for the frame step.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
