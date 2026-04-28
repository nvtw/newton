# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Capsule Net
#
# Cloth-style 2D grid of rigid capsule segments connected by PhoenX
# ball-socket joints. Each grid cell shares its corners with up to four
# neighbors; horizontal capsules link the column-direction pairs and
# vertical capsules link the row-direction pairs, with a single
# ball-socket joint per shared corner. Ball-socket joints lock all 3
# translational DoFs at the corner but leave every rotation free, so
# the net behaves like a chain-mail / cargo-net with no bend or twist
# stiffness.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_capsule_net
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

# Net dimensions.
GRID_NX = 40
GRID_NY = 40

# Capsule geometry. Aligned with body-local +z; each segment spans one
# grid cell so neighboring capsules share their endpoints.
CELL_PITCH = 0.05
SEGMENT_RADIUS = 0.008
SEGMENT_OVERLAP = SEGMENT_RADIUS  # axial overlap so links read as continuous
SEGMENT_MASS = 0.05

GRAVITY = 9.81
PIN_TOP_CORNERS = True

# Multi-cloth defaults. Spacing is given as a multiplier of ``cell_pitch``;
# values smaller than the net's own y-span keep neighbours close enough to
# interact during swing while still spawning without geometric overlap.
NUM_NETS = 3
NET_SPACING_CELLS = 1.5

FPS = 60
SUBSTEPS = 16
SOLVER_ITERATIONS = 5
VELOCITY_ITERATIONS = 1
STEP_LAYOUT = "single_world"

# Per-net hue palette (HSV ramp); we tint each cloth so neighbouring cloths
# read as visually distinct without retuning the per-axis colour scheme.
_NET_HUE_OFFSETS = (0.0, 0.18, 0.36, 0.54, 0.72, 0.10, 0.28, 0.46, 0.64, 0.82)
_AXIS_BASE_COLOR = {
    "x": (0.95, 0.55, 0.20),  # warm orange for row capsules
    "y": (0.20, 0.65, 0.95),  # cool blue for column capsules
}


def _shift_hue(rgb: tuple[float, float, float], hue_offset: float) -> tuple[float, float, float]:
    """Rotate the hue of an RGB triple by ``hue_offset`` (in [0, 1])."""
    import colorsys  # noqa: PLC0415 -- localised to keep module import light

    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + hue_offset) % 1.0
    return colorsys.hsv_to_rgb(h, s, v)


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
    """Inverse inertia for a solid capsule aligned along body-local +z.

    Uses the closed-form mass-weighted sum of a cylinder (length ``2 *
    half_height``, radius ``radius``) and two hemispherical caps; matches
    :func:`newton._src.geometry.inertia.compute_inertia_capsule`."""
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


class Example:
    """PhoenX 2D grid net of capsules tied by ball-socket joints."""

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
        nx = int(self.args.grid_nx)
        ny = int(self.args.grid_ny)
        if nx < 2 or ny < 2:
            raise ValueError("grid-nx and grid-ny must both be >= 2")

        radius = float(self.args.segment_radius)
        pitch = float(self.args.cell_pitch)
        overlap = float(self.args.segment_overlap)
        if radius <= 0.0 or pitch <= 0.0 or overlap < 0.0:
            raise ValueError("segment-radius and cell-pitch must be > 0; segment-overlap must be >= 0")
        # Total capsule length is ``pitch + overlap``; half_height is the
        # length minus the two hemispherical caps.
        half_height = max(0.5 * (pitch + overlap) - radius, 0.0)
        height_z = float(self.args.height)
        gravity_g = float(self.args.gravity)

        num_nets = int(self.args.num_nets)
        if num_nets < 1:
            raise ValueError("num-nets must be >= 1")
        # Net y-span (corner to corner). Spawn neighbouring cloths with a
        # gap so their capsule shapes don't overlap at rest, but small
        # enough that a swing brings them into contact.
        net_y_span = (ny - 1) * pitch
        net_spacing = float(
            self.args.net_spacing
            if self.args.net_spacing is not None
            else (net_y_span + max(float(self.args.net_spacing_cells) * pitch, 4.0 * radius))
        )

        mass = float(self.args.segment_mass)
        inv_mass = 1.0 / mass
        inv_inertia = _capsule_inverse_inertia(mass, radius, half_height)

        builder = WorldBuilder()
        world_body = builder.world_body
        orient_x = _quat_from_z_axis((1.0, 0.0, 0.0))
        orient_y = _quat_from_z_axis((0.0, 1.0, 0.0))

        all_colors: list[tuple[float, float, float]] = []
        total_chain_joints = 0
        total_pin_joints = 0
        net_centers_y: list[float] = []

        for net_idx in range(num_nets):
            # Spread cloths around y=0 so the camera frame stays centred.
            center_y = (net_idx - 0.5 * (num_nets - 1)) * net_spacing
            net_centers_y.append(center_y)
            hue_offset = _NET_HUE_OFFSETS[net_idx % len(_NET_HUE_OFFSETS)] if num_nets > 1 else 0.0
            color_x = _shift_hue(_AXIS_BASE_COLOR["x"], hue_offset)
            color_y = _shift_hue(_AXIS_BASE_COLOR["y"], hue_offset)

            # Grid corner ``(i, j)`` lives at world (x_i, y_j, height_z).
            cx = (nx - 1) * 0.5
            cy = (ny - 1) * 0.5

            def x_of(i: int, _cx: float = cx) -> float:
                return (i - _cx) * pitch

            def y_of(j: int, _cy: float = cy, _center_y: float = center_y) -> float:
                return (j - _cy) * pitch + _center_y

            cap_x_id: dict[tuple[int, int], int] = {}
            cap_y_id: dict[tuple[int, int], int] = {}

            for j in range(ny):
                for i in range(nx - 1):
                    pos = (
                        0.5 * (x_of(i) + x_of(i + 1)),
                        y_of(j),
                        height_z,
                    )
                    bid = builder.add_dynamic_body(
                        position=pos,
                        orientation=orient_x,
                        inverse_mass=inv_mass,
                        inverse_inertia=inv_inertia,
                        affected_by_gravity=True,
                    )
                    builder.add_shape_capsule(bid, radius=radius, half_height=half_height)
                    cap_x_id[(i, j)] = bid
                    all_colors.append(color_x)

            for j in range(ny - 1):
                for i in range(nx):
                    pos = (
                        x_of(i),
                        0.5 * (y_of(j) + y_of(j + 1)),
                        height_z,
                    )
                    bid = builder.add_dynamic_body(
                        position=pos,
                        orientation=orient_y,
                        inverse_mass=inv_mass,
                        inverse_inertia=inv_inertia,
                        affected_by_gravity=True,
                    )
                    builder.add_shape_capsule(bid, radius=radius, half_height=half_height)
                    cap_y_id[(i, j)] = bid
                    all_colors.append(color_y)

            # Walk every grid corner and tie together every pair of adjacent
            # capsule bodies that touches it. A ball-socket joint locks the
            # 3 translational DoFs at the shared anchor and leaves all 3
            # rotations free.
            for j in range(ny):
                for i in range(nx):
                    anchor = (x_of(i), y_of(j), height_z)
                    incident: list[int] = []
                    if (i - 1, j) in cap_x_id:
                        incident.append(cap_x_id[(i - 1, j)])
                    if (i, j) in cap_x_id:
                        incident.append(cap_x_id[(i, j)])
                    if (i, j - 1) in cap_y_id:
                        incident.append(cap_y_id[(i, j - 1)])
                    if (i, j) in cap_y_id:
                        incident.append(cap_y_id[(i, j)])
                    # Chain consecutive bodies via the same anchor; a binary
                    # tie of every pair would be redundant because each link
                    # already enforces coincidence at the shared point.
                    for k in range(len(incident) - 1):
                        a = incident[k]
                        b = incident[k + 1]
                        builder.add_joint(a, b, anchor1=anchor, mode=JointMode.BALL_SOCKET)
                        builder.add_collision_filter_pair(a, b)
                        total_chain_joints += 1

            # Optional anchor pins so the net hangs from its top edge instead
            # of free-falling. Pin corners are evaluated per-net so each
            # cloth gets its own pair of anchors.
            pin_corners: list[tuple[int, int]] = []
            if bool(self.args.pin_top_corners):
                pin_corners = [(0, ny - 1), (nx - 1, ny - 1)]
            elif bool(self.args.pin_top_edge):
                pin_corners = [(i, ny - 1) for i in range(nx)]

            for (i, j) in pin_corners:
                anchor = (x_of(i), y_of(j), height_z)
                # Any incident capsule works as the body to pin; the
                # ball-socket lock at the corner anchors the whole star.
                incident: list[int] = []
                if (i - 1, j) in cap_x_id:
                    incident.append(cap_x_id[(i - 1, j)])
                if (i, j) in cap_x_id:
                    incident.append(cap_x_id[(i, j)])
                if (i, j - 1) in cap_y_id:
                    incident.append(cap_y_id[(i, j - 1)])
                if (i, j) in cap_y_id:
                    incident.append(cap_y_id[(i, j)])
                if not incident:
                    continue
                builder.add_joint(world_body, incident[0], anchor1=anchor, mode=JointMode.BALL_SOCKET)
                total_pin_joints += 1

        # Sanity assertion: each net contributes the same per-corner chain
        # joint count and the same number of pin joints.
        per_net_chain_joints = 0
        for j in range(ny):
            for i in range(nx):
                deg = (
                    (1 if i > 0 else 0)
                    + (1 if i < nx - 1 else 0)
                    + (1 if j > 0 else 0)
                    + (1 if j < ny - 1 else 0)
                )
                per_net_chain_joints += max(deg - 1, 0)
        expected_chain_joints = num_nets * per_net_chain_joints
        expected_total = expected_chain_joints + total_pin_joints
        joints_built = total_chain_joints + total_pin_joints
        if joints_built != expected_total:
            raise RuntimeError(
                f"PhoenX CapsuleNet: built {joints_built} joints but expected "
                f"{expected_total} ({expected_chain_joints} chain + {total_pin_joints} pin) "
                f"across {num_nets} net(s)"
            )

        self.world = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -gravity_g),
            rigid_contact_max=0,
            step_layout=STEP_LAYOUT,
            device=self.device,
        )

        self._radius = radius
        self._half_height = half_height
        self._pitch = pitch
        self._nx = nx
        self._ny = ny
        self._height = height_z
        self._num_nets = num_nets
        self._net_spacing = net_spacing
        self._net_centers_y = net_centers_y
        self._num_capsules = len(all_colors)
        self._num_pin_corners = total_pin_joints

        self._xforms = wp.zeros(self.world.num_bodies, dtype=wp.transform, device=self.device)
        self._capsule_xforms = self._xforms[1 : 1 + self._num_capsules]
        self._colors = wp.array(np.asarray(all_colors, dtype=np.float32), dtype=wp.vec3, device=self.device)

        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        half_extents_np[1 : 1 + self._num_capsules] = (radius, radius, half_height + radius)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        print(
            f"[PhoenX CapsuleNet] nets={num_nets} grid={nx}x{ny} capsules={self._num_capsules} "
            f"bodies={self.world.num_bodies} joints={self.world.num_joints} "
            f"(chain={expected_chain_joints}, pin={total_pin_joints}) "
            f"pitch={pitch:.3f}m radius={radius:.3f}m net_spacing={net_spacing:.3f}m "
            f"gravity={gravity_g:.2f}m/s^2"
        )

        net_width = (nx - 1) * pitch
        net_height = (ny - 1) * pitch
        diag = float(math.sqrt(net_width**2 + net_height**2))
        # Pull the camera further back when several cloths are stacked along y.
        scene_y_extent = max(diag, (num_nets - 1) * net_spacing + net_height)
        cam_pos = wp.vec3(0.0, -1.6 * scene_y_extent, height_z + 0.2 * diag)
        self.viewer.set_camera(pos=cam_pos, pitch=-15.0, yaw=90.0)

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
            "/world/capsule_net",
            newton.GeoType.CAPSULE,
            (self._radius, self._half_height),
            self._capsule_xforms,
            colors=self._colors,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        assert np.isfinite(positions).all(), "non-finite body position in capsule net"
        # Without pinning, the nets free-fall under gravity but should
        # still be near their spawn footprint; we just require they have
        # not exploded laterally. The envelope grows with the multi-net
        # spread along y.
        net_width = max(self._nx, self._ny) * self._pitch
        scene_extent = max(net_width, (self._num_nets - 1) * self._net_spacing + net_width)
        max_lateral = float(np.max(np.abs(positions[1 : 1 + self._num_capsules, :2])))
        assert max_lateral < 8.0 * scene_extent, (
            f"capsule net escaped its envelope (max_lateral={max_lateral:.3f}, scene_extent={scene_extent:.3f})"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--grid-nx", type=int, default=GRID_NX, help="Number of grid corners along x.")
    parser.add_argument("--grid-ny", type=int, default=GRID_NY, help="Number of grid corners along y.")
    parser.add_argument("--cell-pitch", type=float, default=CELL_PITCH, help="Distance between adjacent corners [m].")
    parser.add_argument("--segment-radius", type=float, default=SEGMENT_RADIUS, help="Capsule radius [m].")
    parser.add_argument(
        "--segment-overlap",
        type=float,
        default=SEGMENT_OVERLAP,
        help="Axial overlap between neighboring capsules so the strands look continuous [m].",
    )
    parser.add_argument("--segment-mass", type=float, default=SEGMENT_MASS, help="Mass per capsule [kg].")
    parser.add_argument("--height", type=float, default=1.0, help="Spawn height of the net [m].")
    parser.add_argument("--gravity", type=float, default=GRAVITY, help="Downward gravity magnitude [m/s^2].")
    parser.add_argument(
        "--num-nets",
        type=int,
        default=NUM_NETS,
        help="Number of identical cloths stacked along y. >1 spawns parallel cloths close enough to interact.",
    )
    parser.add_argument(
        "--net-spacing",
        type=float,
        default=None,
        help=(
            "Center-to-center y-spacing between cloths [m]. Defaults to the cloth's y-span plus "
            "``net-spacing-cells`` * ``cell-pitch`` so neighbours start without overlap but can swing into contact."
        ),
    )
    parser.add_argument(
        "--net-spacing-cells",
        type=float,
        default=NET_SPACING_CELLS,
        help="Extra gap between adjacent cloths, expressed as a multiple of cell-pitch.",
    )
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--pin-top-corners",
        action="store_true",
        default=PIN_TOP_CORNERS,
        help="Pin the top-left and top-right corners of the net to the world.",
    )
    pin_group.add_argument(
        "--pin-top-edge",
        action="store_true",
        default=False,
        help="Pin the entire top edge of the net to the world.",
    )
    pin_group.add_argument(
        "--no-pin",
        dest="pin_top_corners",
        action="store_false",
        help="Disable pinning so the net free-falls under gravity.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Render frames per second.")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS, help="PhoenX solver substeps per frame.")
    parser.add_argument("--solver-iterations", type=int, default=SOLVER_ITERATIONS, help="PhoenX PGS iterations.")
    parser.add_argument("--velocity-iterations", type=int, default=VELOCITY_ITERATIONS, help="PhoenX velocity iterations.")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture for the frame step.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
