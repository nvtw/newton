# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Twisted Thread
#
# A single slack capsule strand suspended between two motorized hinges.
# Each end of the strand is wired to the static world body through a
# ``JointMode.REVOLUTE`` joint whose hinge axis runs along the strand
# (world +x). The two end motors are velocity-driven in opposite
# directions, so they pump twist into the strand from both sides.
#
# Combined with the strand's bend / twist stiffness (PhoenX
# ``JointMode.CABLE`` between consecutive segments) and a small amount
# of slack, the counter-rotating motors progressively coil the strand
# into the helical "thread twisting" pattern you see when over-twisting
# a hanging rope.
#
# Geometry note: ``segment_radius`` defaults to 2.5 mm so the visible
# rope is 5 mm in diameter.
#
# Time stepping: the simulation ticks at ``SIM_FPS`` (240 Hz default)
# with ``SUBSTEPS`` solver substeps per outer tick (7 by default), and
# the renderer is driven at ``RENDER_FPS`` (60 Hz). Each rendered frame
# triggers ``SIM_FPS / RENDER_FPS`` consecutive sim ticks (4 by
# default), all captured in a single CUDA graph. Same idiom as
# ``example_phoenx_scale`` / ``example_pyramid``.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_twisted_thread
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.picking import Picking, register_with_viewer_gl
from newton._src.solvers.phoenx.solver_phoenx import pack_body_xforms_kernel
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode, WorldBuilder

# Strand layout. The strand runs along world +x between two end pins.
# The motors sit at +/- 0.5 * span in world x, so shrinking ``span``
# moves them closer together. ``SLACK`` is the extra arc length on
# top of the chord; for a generously hanging rope we want the arc
# length to be a sizeable fraction of the chord (e.g. 100% extra ->
# rope is twice as long as the chord).
SEGMENTS = 256
SPAN = 0.1  # end-to-end chord [m]
SLACK = 0.5  # extra arc length beyond the chord [m]; rope is 2x the chord at default settings

# Capsule geometry. Body-local +z is the strand axis; each capsule
# overlaps its neighbours by ``2 * radius`` so the rendered tube reads
# as continuous.
SEGMENT_RADIUS = 0.0025  # 5 mm rope diameter
SEGMENT_OVERLAP_FACTOR = 2.0
SEGMENT_MASS = 0.002

# PhoenX cable joint stiffness / damping for the inter-segment links.
# Bend governs the two axes perpendicular to the strand; twist governs
# the axis along it. The strand needs enough twist stiffness that the
# torque pumped in by the end motors propagates through (and stays
# stored as elastic energy that buckles into helices), but not so much
# that the chain becomes a rigid bar. Damping kills the high-frequency
# ringing the rigid ball-sockets would otherwise inject as the helix
# forms.
BEND_STIFFNESS = 50
TWIST_STIFFNESS = 0.5
BEND_DAMPING = 0.3*BEND_STIFFNESS
TWIST_DAMPING = 0.3*TWIST_STIFFNESS

# End-cap revolute motor knobs. ``DriveMode.VELOCITY`` requires
# ``damping_drive > 0`` (PD velocity servo); the resulting torque per
# joint is roughly ``damping_drive * (target - current)``.
END_TARGET_VELOCITY = 4.0  # [rad/s]; positive at the +x end, negated at the -x end
END_DAMPING_DRIVE = 0.5  # PD gain on the velocity error [N*m*s/rad]
END_STIFFNESS_DRIVE = 0.0  # pure velocity servo
END_MAX_TORQUE = 5.0  # generous torque cap [N*m]

# World layout. The strand sits at ``STRAND_HEIGHT`` above the ground
# plane; we leave plenty of headroom so a parabolic sag never reaches
# the ground (validated below).
GROUND_HEIGHT = 0.0
STRAND_HEIGHT_SAFETY = 0.1  # extra clearance above the predicted sag [m]
GRAVITY = 9.81

# Time stepping. The simulation runs at ``SIM_FPS`` outer ticks per
# second; each tick calls :meth:`PhoenXWorld.step` once, which itself
# expands into ``SUBSTEPS`` solver substeps. The renderer is driven at
# ``RENDER_FPS``, so each rendered frame triggers
# ``SIM_FPS / RENDER_FPS`` consecutive sim ticks (4 by default at
# 240 Hz / 60 Hz). This pattern matches ``example_phoenx_scale`` /
# ``example_pyramid``: a fast inner sim with a slower visual frame.
SIM_FPS = 240
RENDER_FPS = 60
SUBSTEPS = 7
SOLVER_ITERATIONS = 6
VELOCITY_ITERATIONS = 1
STEP_LAYOUT = "single_world"

_STRAND_COLOR_A = (0.95, 0.55, 0.20)  # warm orange
_STRAND_COLOR_B = (0.20, 0.65, 0.95)  # cool blue


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


def _solve_catenary_a(span: float, arc_length: float) -> float:
    """Solve for the catenary scale ``a`` (in ``y = a * cosh(x/a)``).

    The arc length between ``+/- span/2`` of ``y = a * cosh(x/a)`` is
    ``L = 2 * a * sinh(span / (2*a))``. We invert this for ``a`` given
    target ``L = arc_length`` via bisection; ``a -> +inf`` gives
    ``L -> span`` (taut limit) and ``a -> 0`` gives ``L -> +inf``
    (very droopy), so the function is strictly monotonic in ``1/a``
    and bisection converges robustly.

    Args:
        span: End-to-end horizontal separation [m]; must be > 0.
        arc_length: Total rope length [m]; must satisfy
            ``arc_length > span`` (strictly slack).

    Returns:
        The positive scale parameter ``a`` [m].
    """
    if span <= 0.0:
        raise ValueError("span must be > 0")
    if arc_length <= span:
        raise ValueError(f"arc_length ({arc_length}) must exceed span ({span}) for a slack rope")

    half_span = 0.5 * span
    # Bracket: ``a`` decreases as the rope gets droopier.
    lo = 1.0e-6 * span
    hi = 1.0e6 * span
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        length_mid = 2.0 * mid * math.sinh(half_span / mid)
        if length_mid > arc_length:
            # ``a`` too small (rope too long) -> grow ``a``.
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-12 * max(hi, 1.0):
            break
    return 0.5 * (lo + hi)


def _catenary_x_for_arclength(a: float, target_s: float) -> float:
    """Invert the catenary arc-length integral.

    Arc length from ``x = 0`` along ``y = a * cosh(x/a)`` is
    ``s(x) = a * sinh(x/a)``, which inverts in closed form to
    ``x = a * asinh(s / a)``. We use it to space capsule centres at
    equal arc-length intervals along the rope.
    """
    return a * math.asinh(target_s / a)


def _catenary_sample(a: float, x: float) -> tuple[float, float, tuple[float, float, float]]:
    """Sample the catenary at signed horizontal position ``x``.

    Returns:
        Tuple ``(y, dy_dx, tangent)`` where ``y = a * cosh(x/a) - a``
        is the height drop relative to the endpoints, ``dy_dx =
        sinh(x/a)`` is the local slope, and ``tangent`` is the unit
        tangent ``(1, 0, dy_dx) / sqrt(1 + dy_dx^2)`` in the strand's
        ``xz`` plane (we set ``y = 0`` since the rope hangs in the
        ``xz`` plane in this scene).
    """
    y = a * math.cosh(x / a) - a
    slope = math.sinh(x / a)
    norm = math.sqrt(1.0 + slope * slope)
    return y, slope, (1.0 / norm, 0.0, slope / norm)


def _quat_from_tangent(tangent: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Quaternion rotating body-local +z onto ``tangent`` (unit)."""
    return _quat_from_z_axis(tangent)


class Example:
    """Counter-rotating end motors twisting a slack capsule strand."""

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
        # ``frame_dt`` is the **simulation** tick (inner step), not the
        # render frame; ``render_dt`` is what ``self.sim_time`` is
        # advanced by in :meth:`step`.
        self.frame_dt = 1.0 / self.sim_fps
        self.render_dt = 1.0 / self.render_fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "substeps", SUBSTEPS))
        self.solver_iterations = int(getattr(args, "solver_iterations", SOLVER_ITERATIONS))
        self.velocity_iterations = int(getattr(args, "velocity_iterations", VELOCITY_ITERATIONS))

        self._build_scene()

    def _build_scene(self) -> None:
        segments = int(self.args.segments)
        if segments < 4:
            raise ValueError("segments must be >= 4 (need room for two end joints)")

        radius = float(self.args.segment_radius)
        span = float(self.args.span)
        slack = float(self.args.slack)
        if radius <= 0.0 or span <= 0.0 or slack < 0.0:
            raise ValueError("segment-radius and span must be > 0; slack must be >= 0")

        # Each capsule has total length ``pitch + overlap``. ``pitch``
        # is the segment centre-to-centre spacing along the strand; the
        # full rope arc length is therefore ``segments * pitch`` by
        # construction (same definition as ``example_cable_lattice``).
        # We size ``pitch`` from the requested slack so that the rope
        # is exactly ``span + slack`` long and then re-derive the
        # catenary against this.
        arc_length = span + slack
        pitch = arc_length / segments
        overlap = float(self.args.segment_overlap_factor) * radius
        half_height = max(0.5 * (pitch + overlap) - radius, 0.0)

        # Solve for the catenary scale ``a`` so that ``y = a *
        # cosh(x/a)`` between ``+/- span/2`` has total arc length
        # ``arc_length`` -- i.e. the chain made of ``segments``
        # capsules of length ``pitch`` lies exactly on this curve.
        cat_a = _solve_catenary_a(span, arc_length)
        sag = cat_a * (math.cosh(0.5 * span / cat_a) - 1.0)
        clearance = float(self.args.strand_height_safety)
        ground_z = float(self.args.ground_height)
        # ``strand_z`` is the height of the two end pins; the lowest
        # point of the catenary sits at ``strand_z - sag``. We keep at
        # least one capsule radius plus ``clearance`` between that
        # bottom point and the ground.
        strand_z = ground_z + sag + clearance + radius
        if strand_z - sag - radius <= ground_z:
            raise ValueError(
                f"computed strand height {strand_z:.3f} m would let the sagging strand "
                f"(sag~{sag:.3f} m, radius={radius:.4f} m) touch the ground at z={ground_z:.3f}"
            )

        gravity_g = float(self.args.gravity)

        mass = float(self.args.segment_mass)
        inv_mass = 1.0 / mass
        inv_inertia = _capsule_inverse_inertia(mass, radius, half_height)

        builder = WorldBuilder()
        world_body = builder.world_body
        # Static ground plane attached to the world body so falling
        # bodies have something to land on if the strand tears free.
        builder.add_shape_plane(world_body, normal=(0.0, 0.0, 1.0), offset=ground_z)

        # Catenary sample helpers. ``x_at(s)`` inverts the catenary
        # arc-length integral from ``x=0`` (rope midpoint) and returns
        # the signed horizontal position; ``point_at(s)`` and
        # ``tangent_at(s)`` return the world-space rope point and its
        # local unit tangent at signed arc length ``s`` measured from
        # the midpoint.
        def x_at(s_signed: float) -> float:
            return _catenary_x_for_arclength(cat_a, s_signed)

        def point_at(s_signed: float) -> tuple[float, float, float]:
            x = x_at(s_signed)
            y, _slope, _tangent = _catenary_sample(cat_a, x)
            # ``y`` is height *above* the rope's lowest point, so the
            # world-space z is ``(strand_z - sag) + y``: the lowest
            # point sits at ``strand_z - sag`` and the two pins sit
            # at ``strand_z`` (where ``y == sag`` by construction).
            return (x, 0.0, (strand_z - sag) + y)

        def tangent_at(s_signed: float) -> tuple[float, float, float]:
            x = x_at(s_signed)
            _y, _slope, tangent = _catenary_sample(cat_a, x)
            return tangent

        # Spawn the strand directly on the catenary so frame zero
        # already shows the rope hanging through. The ``s`` parameter
        # runs from ``-arc_length/2`` (left pin) to ``+arc_length/2``
        # (right pin); each capsule centre sits at a half-pitch
        # offset, and its body-local +z is aligned with the local
        # rope tangent.
        segment_body_ids: list[int] = []
        colors: list[tuple[float, float, float]] = []
        for k in range(segments):
            s_centre = -0.5 * arc_length + (k + 0.5) * pitch
            pos = point_at(s_centre)
            tangent = tangent_at(s_centre)
            orientation = _quat_from_tangent(tangent)
            bid = builder.add_dynamic_body(
                position=pos,
                orientation=orientation,
                inverse_mass=inv_mass,
                inverse_inertia=inv_inertia,
                affected_by_gravity=True,
            )
            builder.add_shape_capsule(bid, radius=radius, half_height=half_height)
            segment_body_ids.append(bid)
            # Alternate colours so the helix reads visually as the
            # strand twists.
            colors.append(_STRAND_COLOR_A if (k & 1) == 0 else _STRAND_COLOR_B)

        # Inter-segment cable joints. The shared anchor between
        # capsules ``k`` and ``k+1`` is the catenary point at signed
        # arc length ``s = -L/2 + (k+1) * pitch``; ``anchor2`` is set
        # one unit further along the local tangent so the cable's
        # twist axis matches the rope direction at that anchor.
        for k in range(segments - 1):
            s_anchor = -0.5 * arc_length + (k + 1) * pitch
            anchor1 = point_at(s_anchor)
            tangent = tangent_at(s_anchor)
            anchor2 = (anchor1[0] + tangent[0], anchor1[1] + tangent[1], anchor1[2] + tangent[2])
            builder.add_joint(
                segment_body_ids[k],
                segment_body_ids[k + 1],
                anchor1=anchor1,
                anchor2=anchor2,
                mode=JointMode.CABLE,
                bend_stiffness=float(self.args.bend_stiffness),
                twist_stiffness=float(self.args.twist_stiffness),
                bend_damping=float(self.args.bend_damping),
                twist_damping=float(self.args.twist_damping),
            )
            builder.add_collision_filter_pair(segment_body_ids[k], segment_body_ids[k + 1])

        # End motors. The hinge axis is ``anchor1 -> anchor2`` along
        # the local rope tangent at the pin (so the motor spins the
        # rope about its own axis even though the rope is no longer
        # horizontal at the endpoints). ``DriveMode.VELOCITY`` with
        # opposite ``target_velocity`` signs pumps twist into the
        # strand from both ends.
        target_velocity = float(self.args.end_target_velocity)
        damping_drive = float(self.args.end_damping_drive)
        stiffness_drive = float(self.args.end_stiffness_drive)
        max_torque = float(self.args.end_max_torque)
        if damping_drive <= 0.0:
            raise ValueError("end-damping-drive must be > 0 (PhoenX VELOCITY drive needs PD damping)")

        s_left = -0.5 * arc_length
        s_right = +0.5 * arc_length
        end_anchor_left = point_at(s_left)
        end_tangent_left = tangent_at(s_left)
        end_anchor_left_axis = (
            end_anchor_left[0] + end_tangent_left[0],
            end_anchor_left[1] + end_tangent_left[1],
            end_anchor_left[2] + end_tangent_left[2],
        )
        end_anchor_right = point_at(s_right)
        end_tangent_right = tangent_at(s_right)
        end_anchor_right_axis = (
            end_anchor_right[0] + end_tangent_right[0],
            end_anchor_right[1] + end_tangent_right[1],
            end_anchor_right[2] + end_tangent_right[2],
        )

        builder.add_joint(
            world_body,
            segment_body_ids[0],
            anchor1=end_anchor_left,
            anchor2=end_anchor_left_axis,
            mode=JointMode.REVOLUTE,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=+target_velocity,
            stiffness_drive=stiffness_drive,
            damping_drive=damping_drive,
            max_force_drive=max_torque,
        )
        builder.add_joint(
            world_body,
            segment_body_ids[-1],
            anchor1=end_anchor_right,
            anchor2=end_anchor_right_axis,
            mode=JointMode.REVOLUTE,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=-target_velocity,
            stiffness_drive=stiffness_drive,
            damping_drive=damping_drive,
            max_force_drive=max_torque,
        )

        expected_inter_segment = segments - 1
        expected_end = 2
        expected_total = expected_inter_segment + expected_end

        self.world = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -gravity_g),
            rigid_contact_max=5000,
            step_layout=STEP_LAYOUT,
            device=self.device,
        )

        if int(self.world.num_joints) != expected_total:
            raise RuntimeError(
                f"PhoenX TwistedThread: expected {expected_total} joints "
                f"(={expected_inter_segment} cable + {expected_end} end-revolute), "
                f"but the world reports {int(self.world.num_joints)}"
            )

        self._radius = radius
        self._half_height = half_height
        self._segments = segments
        self._span = span
        self._slack = slack
        self._sag = sag
        self._strand_z = strand_z

        self._xforms = wp.zeros(self.world.num_bodies, dtype=wp.transform, device=self.device)
        self._capsule_xforms = self._xforms[1:]
        self._colors = wp.array(np.asarray(colors, dtype=np.float32), dtype=wp.vec3, device=self.device)

        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        half_extents_np[1:] = (radius, radius, half_height + radius)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        substep_dt = self.frame_dt / max(self.sim_substeps, 1)
        print(
            f"[PhoenX TwistedThread] segments={segments} span={span:.3f}m slack={slack:.3f}m "
            f"arc_length={arc_length:.3f}m radius={radius * 1000.0:.1f}mm pitch={pitch:.4f}m "
            f"overlap={overlap:.4f}m strand_z={strand_z:.3f}m catenary_a={cat_a:.4f}m "
            f"sag={sag:.3f}m end_omega=+/-{target_velocity:.2f}rad/s "
            f"joints={int(self.world.num_joints)} "
            f"sim_fps={self.sim_fps}Hz render_fps={self.render_fps}Hz "
            f"sim_ticks/frame={self.sim_ticks_per_frame} substeps/tick={self.sim_substeps} "
            f"substep_dt={substep_dt * 1000.0:.3f}ms"
        )

        # Camera framing has to account for the sag: the rope can hang
        # well below the motor line when ``slack`` is large, so size
        # the framing on the larger of chord and total drop.
        scene_extent = max(span, sag + clearance)
        cam_y = -1.6 * scene_extent
        cam_z = max(strand_z + 0.25 * span, ground_z + 0.25 * scene_extent)
        self.viewer.set_camera(
            pos=wp.vec3(0.05 * span, cam_y, cam_z),
            pitch=-12.0,
            yaw=92.0,
        )

        self.graph = None
        if self.device.is_cuda and not bool(self.args.no_cuda_graph):
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        # Run ``sim_ticks_per_frame`` outer ticks per rendered frame.
        # Each ``world.step`` internally expands into ``sim_substeps``
        # solver substeps of size ``frame_dt / sim_substeps``. The
        # whole loop is captured in a single CUDA graph below so the
        # 4 sim-ticks-per-render cost amounts to one graph launch.
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
            "/world/twisted_thread",
            newton.GeoType.CAPSULE,
            (self._radius, self._half_height),
            self._capsule_xforms,
            colors=self._colors,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        assert np.isfinite(positions).all(), "non-finite body position in twisted thread"
        # The strand must stay within a generous box around its spawn
        # chord and -- crucially -- never dip below the ground plane.
        capsule_z = positions[1:, 2]
        min_z = float(np.min(capsule_z))
        ground_z = float(self.args.ground_height)
        assert min_z > ground_z - self._radius, (
            f"strand dipped below the ground plane (min_z={min_z:.3f}, ground_z={ground_z:.3f})"
        )
        max_lateral = float(np.max(np.abs(positions[1:, :2])))
        assert max_lateral < 4.0 * self._span, f"twisted thread escaped its envelope (max_lateral={max_lateral:.3f})"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--segments", type=int, default=SEGMENTS, help="Number of capsule segments along the strand.")
    parser.add_argument("--span", type=float, default=SPAN, help="End-to-end chord length [m].")
    parser.add_argument(
        "--slack",
        type=float,
        default=SLACK,
        help="Extra arc length beyond the chord [m]; controls how loose the strand hangs.",
    )
    parser.add_argument(
        "--segment-radius",
        type=float,
        default=SEGMENT_RADIUS,
        help="Capsule radius [m]. Default = 2.5 mm = 5 mm rope diameter.",
    )
    parser.add_argument(
        "--segment-overlap-factor",
        type=float,
        default=SEGMENT_OVERLAP_FACTOR,
        help="Axial overlap between neighbouring capsules expressed as a multiple of segment-radius.",
    )
    parser.add_argument("--segment-mass", type=float, default=SEGMENT_MASS, help="Mass per capsule segment [kg].")
    parser.add_argument("--bend-stiffness", type=float, default=BEND_STIFFNESS, help="Cable bend stiffness [N*m/rad].")
    parser.add_argument("--twist-stiffness", type=float, default=TWIST_STIFFNESS, help="Cable twist stiffness [N*m/rad].")
    parser.add_argument("--bend-damping", type=float, default=BEND_DAMPING, help="Cable bend damping [N*m*s/rad].")
    parser.add_argument("--twist-damping", type=float, default=TWIST_DAMPING, help="Cable twist damping [N*m*s/rad].")
    parser.add_argument(
        "--end-target-velocity",
        type=float,
        default=END_TARGET_VELOCITY,
        help="End-motor target angular velocity [rad/s]; the two ends spin at +omega and -omega.",
    )
    parser.add_argument(
        "--end-damping-drive",
        type=float,
        default=END_DAMPING_DRIVE,
        help="PD damping gain on the end motors [N*m*s/rad]; PhoenX VELOCITY drives need this > 0.",
    )
    parser.add_argument(
        "--end-stiffness-drive",
        type=float,
        default=END_STIFFNESS_DRIVE,
        help="PD stiffness gain on the end motors [N*m/rad]; 0 keeps them as pure velocity servos.",
    )
    parser.add_argument(
        "--end-max-torque",
        type=float,
        default=END_MAX_TORQUE,
        help="Saturation torque cap for each end motor [N*m].",
    )
    parser.add_argument("--ground-height", type=float, default=GROUND_HEIGHT, help="World z of the ground plane [m].")
    parser.add_argument(
        "--strand-height-safety",
        type=float,
        default=STRAND_HEIGHT_SAFETY,
        help="Extra clearance above the predicted parabolic sag when picking the strand spawn height [m].",
    )
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
