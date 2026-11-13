# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drop a fan of interlocking steel chains from an anchor ring with PhoenX.

Command: python -m newton.examples phoenx_chain_fan

Chains of round steel links hang from a ring of kinematic first links (bolted eyelets); with many
chains, neighbors alternate between a few staggered heights so the ring stays small. Every
chain starts laid out straight and level, pointing radially away from the ring, ends in a
pentagonal pendant threaded through its last link, and swings down under gravity with a
gentle initial spin. After six seconds, the anchors spread outward smoothly over eight
seconds. The chains meet below the ring, so links of different chains collide, while each chain holds
together only through the contacts between its interlocked links. Links and pendants are
watertight meshes (``assets/chain_fan``) and
collide through Newton's mesh signed distance fields; there are no joints between links.

Success criteria: in test mode, every body stays finite, the chains swing down below the
ring, and no chain comes apart: the wire centerlines of neighboring links stay linked
(signed intersection count +-1), so no wire passed through another, and neighboring links never
move farther apart than their taut pitch allows.

Runs:
- `python -m newton.examples phoenx_chain_fan`
- `python -m newton.examples phoenx_chain_fan --chains 240`
  - The full fan of 240 chains.
- `python -m newton.examples phoenx_chain_fan --viewer null --chains 240 --num-frames 600`
"""

import colorsys
import math
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX

ASSETS = Path(newton.examples.get_asset_directory()) / "chain_fan"

# [m] per model unit: 3.4 cm links of 5 mm wire, like the 3/16" chains of a disc golf basket
# (24 mm inside length, 12 mm inside width).
SCALE = 0.042
# Link geometry in model units, see assets/chain_fan/generate_meshes.py.
WIRE_RADIUS = 0.06
BEND_RADIUS = 0.2
HALF_STRAIGHT = 0.15
END_WIRE = HALF_STRAIGHT + BEND_RADIUS  # link center to the wire centerline at either end
INNER_HALF = HALF_STRAIGHT + BEND_RADIUS - WIRE_RADIUS  # link center to the inner end of the opening
TAUT_PITCH = INNER_HALF - WIRE_RADIUS + END_WIRE  # center distance of two interlocked links pulled taut
PITCH = 0.52  # initial center distance, a little slack below TAUT_PITCH
PENDANT_HOLE = 0.15  # pendant hole center from the pendant's base, model units
LINK_SPACING = 0.6  # model units of ring circumference given to each chain
RING_HEIGHT = 2.5  # [m]
RING_RADIUS_MAX = 0.25  # [m] beyond this, chains alternate between staggered heights
LEVEL_STEP = 2.0 * (BEND_RADIUS + WIRE_RADIUS) * SCALE + 0.004  # [m] a link's width plus clearance
DENSITY = 7800.0  # steel [kg/m^3]
SDF_RESOLUTION = 128
CONTACTS_PER_BODY = 128  # rigid contact capacity per body
SDF_INNER_BAND = 0.06  # model units of signed distance kept inside a surface (the wire radius)
SDF_OUTER_BAND = 0.3  # model units of signed distance kept outside a surface
SPECULATIVE_GAP_MAX = 0.1  # [m] largest speculative extension of a contact search


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangle indices of a triangulated OBJ file."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _sdf_mesh(path: Path, gap: float) -> tuple[newton.Mesh, np.ndarray]:
    """A mesh centered on its bounds with a signed distance field, and the removed center.

    The field reaches ``gap`` [m] outside the surface, so speculative contacts up to the
    contact gap read true distances.
    """
    vertices, faces = _load_obj(path)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    mesh = newton.Mesh(vertices - center, faces.flatten())
    band = (-SDF_INNER_BAND, max(SDF_OUTER_BAND, 1.5 * gap / SCALE))
    mesh.build_sdf(
        max_resolution=SDF_RESOLUTION,
        narrow_band_range=band,
        margin=gap / SCALE,
        cache_dir=str(Path(tempfile.gettempdir()) / "newton_chain_fan_sdf"),
    )
    return mesh, center


def _link_centerline(samples: int = 96) -> np.ndarray:
    """Points on a link's wire centerline [m], in link coordinates."""
    straight, bend = 2.0 * HALF_STRAIGHT, math.pi * BEND_RADIUS
    points = []
    for arc in np.linspace(0.0, 2.0 * (straight + bend), samples, endpoint=False):
        side = 1.0 if arc < straight + bend else -1.0  # second half mirrors the first through the center
        s = arc if side > 0.0 else arc - straight - bend
        if s < straight:
            x, y = -HALF_STRAIGHT + s, -BEND_RADIUS
        else:
            angle = -0.5 * math.pi + (s - straight) / BEND_RADIUS
            x, y = HALF_STRAIGHT + BEND_RADIUS * math.cos(angle), BEND_RADIUS * math.sin(angle)
        points.append((side * x, side * y, 0.0))
    return np.asarray(points) * SCALE


def _link_intersections(poses_a: np.ndarray, poses_b: np.ndarray, curve_b: np.ndarray) -> np.ndarray:
    """Count signed crossings of B's wire through A's planar stadium opening."""

    def rotate(quaternions, points):
        imaginary, real = quaternions[:, None, :3], quaternions[:, None, 3:4]
        twice = 2.0 * np.cross(imaginary, points)
        return points + real * twice + np.cross(imaginary, twice)

    # Express B's polygonal centerline in A's frame. A spans a planar stadium
    # disk, so oriented segment/disk intersections give the integer linking number.
    points = rotate(poses_b[:, 3:], curve_b[None]) + (poses_b[:, :3] - poses_a[:, :3])[:, None]
    inverse = poses_a[:, 3:].copy()
    inverse[:, :3] *= -1.0
    points = rotate(inverse, points)
    ends = np.roll(points, -1, axis=1)
    crossed = (points[:, :, 2] > 0.0) != (ends[:, :, 2] > 0.0)
    t = points[:, :, 2] / np.where(crossed, points[:, :, 2] - ends[:, :, 2], 1.0)
    x = points[:, :, 0] + t * (ends[:, :, 0] - points[:, :, 0])
    y = points[:, :, 1] + t * (ends[:, :, 1] - points[:, :, 1])
    dx = np.maximum(np.abs(x) - HALF_STRAIGHT * SCALE, 0.0)
    inside = dx * dx + y * y < (BEND_RADIUS * SCALE) ** 2
    return np.where(crossed & inside, np.sign(ends[:, :, 2] - points[:, :, 2]), 0.0).sum(axis=1)


def _chain_color(chain: int, chains: int) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb(chain / max(chains, 1), 0.65, 0.68)


@wp.kernel
def _move_anchors(
    time: wp.array[float],
    offset: float,
    dt: float,
    anchors: wp.array[int],
    initial: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    body = anchors[i]
    pose = initial[i]
    position = wp.transform_get_translation(pose)
    radial = wp.normalize(wp.vec3(position[0], position[1], 0.0))
    # Smooth start and stop: no instantaneous velocity change at either end.
    t = wp.clamp((time[0] + offset - 6.0) / 8.0, 0.0, 1.0)
    position += radial * (0.65 * t * t * (3.0 - 2.0 * t))
    velocity = (position - wp.transform_get_translation(body_q[body])) / dt
    body_q[body] = wp.transform(position, wp.transform_get_rotation(pose))
    body_qd[body] = wp.spatial_vector(velocity, wp.vec3(0.0))


class Example:
    def __init__(self, viewer, args):
        if args.chains < 1 or args.links < 3:
            raise ValueError("Use at least one chain and three links per chain.")
        if min(args.collide_steps, args.substeps, args.iterations) < 1:
            raise ValueError("Collision steps, substeps, and iterations must be positive.")
        if any(not math.isfinite(value) or value < 0.0 for value in (args.gap, args.speculative_max)):
            raise ValueError("Contact gaps must be finite and nonnegative.")
        if not math.isfinite(args.spin):
            raise ValueError("Initial spin must be finite.")
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        # Collision detection runs every sim step; fast chain ends travel far within a frame.
        self.sim_steps = args.collide_steps
        self.sim_dt = self.frame_dt / self.sim_steps
        gap = args.gap
        self.sim_time = 0.0
        self.chains = args.chains
        self.links = args.links

        link_mesh, link_center = _sdf_mesh(ASSETS / "chain_link.obj", gap)
        pendant_mesh, pendant_center = _sdf_mesh(ASSETS / "chain_pendant.obj", gap)
        assert np.allclose(link_center, 0.0, atol=1.0e-4), "the link mesh must be centered"

        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=0.5, gap=gap)
        # Every chain gets an arc of the ring as wide as a link plus some clearance. Many chains
        # alternate between staggered heights so the ring stays small and the chains still meet.
        arc = self.chains * LINK_SPACING * SCALE / (2.0 * math.pi)  # [m] ring radius on one level
        levels = max(1, math.ceil(arc / RING_RADIUS_MAX))
        ring = max(0.15, arc / levels)  # [m]
        self.ring_radius = ring
        upright = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)  # link plane XY -> XZ

        self.chain_bodies: list[list[int]] = []
        anchors = []
        for chain in range(self.chains):
            angle = 2.0 * math.pi * chain / self.chains
            turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            radial = wp.vec3(math.cos(angle), math.sin(angle), 0.0)
            color = _chain_color(chain, self.chains)

            height = RING_HEIGHT - (chain % levels) * LEVEL_STEP

            def place(distance: float, rotation, turn=turn, radial=radial, height=height):
                return wp.transform(radial * distance + wp.vec3(0.0, 0.0, height), turn * rotation)

            # The first link is a bolted eyelet; all remaining links are free.
            bodies = []
            for link in range(self.links):
                distance = (ring / SCALE + link * PITCH) * SCALE
                rotation = upright if link % 2 == 0 else wp.quat_identity()
                body = builder.add_body(
                    xform=place(distance, rotation), is_kinematic=link == 0, label=f"chain{chain}_link{link}"
                )
                if link == 0:
                    anchors.append(body)
                builder.add_shape_mesh(body, mesh=link_mesh, scale=(SCALE, SCALE, SCALE), cfg=cfg, color=color)
                bodies.append(body)
            # The pendant plate crosses the last link's plane for either link-count parity.
            hole = ring / SCALE + (self.links - 1) * PITCH + END_WIRE
            base = hole - PENDANT_HOLE
            pendant = builder.add_body(
                xform=place(
                    (base + float(pendant_center[0])) * SCALE, wp.quat_identity() if self.links % 2 else upright
                ),
                label=f"chain{chain}_pendant",
            )
            light = tuple(0.12 + 0.8 * c for c in color)
            builder.add_shape_mesh(pendant, mesh=pendant_mesh, scale=(SCALE, SCALE, SCALE), cfg=cfg, color=light)
            bodies.append(pendant)
            self.chain_bodies.append(bodies)
            # Rotate each released chain about its anchor, avoiding a sideways
            # kick at the attachment. Linear velocity is measured at the COM.
            omega = wp.vec3(0.0, 0.0, args.spin)
            origin = wp.transform_get_translation(builder.body_q[bodies[0]])
            for body in bodies[1:]:
                com = wp.transform_point(builder.body_q[body], builder.body_com[body])
                builder.body_qd[body] = wp.spatial_vector(*wp.cross(omega, com - origin), *omega)

        self.model = builder.finalize()
        self.anchor_index = np.asarray(anchors, dtype=np.int32)
        self.anchors = wp.array(self.anchor_index, dtype=int, device=self.model.device)
        self.anchor_initial = wp.array(
            [builder.body_q[body] for body in anchors], dtype=wp.transform, device=self.model.device
        )
        self.time = wp.zeros(1, dtype=float, device=self.model.device)
        self.link_curve = _link_centerline()
        angles = np.linspace(0.0, 2.0 * math.pi, len(self.link_curve), endpoint=False)
        self.pendant_curve = (
            np.column_stack((PENDANT_HOLE + 0.11 * np.cos(angles), 0.11 * np.sin(angles), np.zeros_like(angles)))
            - pendant_center
        ) * SCALE
        # A small authored gap keeps contacts to touching links; moving links extend their
        # search by their closing distance over the collision update, up to 10 cm.
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            # A link touches its two neighbors and a few links of other chains; the automatic
            # estimate assumes any mesh pair may touch and would reserve far more.
            rigid_contact_max=CONTACTS_PER_BODY * self.model.body_count,
            contact_matching="sticky",
            speculative_contact_gap_max=args.speculative_max,
            contact_reduction_voxel_depth=True,
        )
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_0.body_qd.assign(np.asarray(builder.body_qd, dtype=np.float32))
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.link_index = np.asarray([bodies[:-1] for bodies in self.chain_bodies], dtype=np.int64)

        if hasattr(viewer, "tonemap_saturation"):
            viewer.tonemap_saturation = 1.0
            viewer.exposure = 0.45
        viewer.set_model(self.model)
        reach = ring + self.links * PITCH * SCALE
        viewer.set_camera(wp.vec3(0.0, -2.8 * reach, RING_HEIGHT + 0.7 * reach), pitch=-30.0, yaw=90.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for step in range(self.sim_steps):
            wp.launch(
                _move_anchors,
                dim=self.chains,
                inputs=[
                    self.time,
                    (step + 1) * self.sim_dt,
                    self.sim_dt,
                    self.anchors,
                    self.anchor_initial,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                ],
                device=self.model.device,
            )
            self.pipeline.collide(self.state_0, self.contacts, dt=self.sim_dt)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # Copy back instead of swapping so the captured graph always reads state_0.
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)

    def step(self):
        self.time.fill_(self.sim_time)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        ui.text(f"Contacts: {count:,}")
        ui.text(f"Chains: {self.chains} x {self.links} links")

    def link_stretch(self) -> float:
        """Largest center distance of neighboring links, in taut pitches."""
        positions = self.state_0.body_q.numpy()[:, :3]
        links = positions[self.link_index]
        gaps = np.linalg.norm(links[:, 1:] - links[:, :-1], axis=-1)
        return float(gaps.max() / (TAUT_PITCH * SCALE))

    def unlinked_pairs(self) -> list[tuple[int, int]]:
        """(chain, link) of every link whose wire no longer threads through the next link."""
        poses = self.state_0.body_q.numpy()
        links = self.link_index
        linking = _link_intersections(poses[links[:, :-1].ravel()], poses[links[:, 1:].ravel()], self.link_curve)
        linked = np.isfinite(linking) & (np.abs(linking) >= 0.5)
        return [tuple(int(i) for i in pair) for pair in np.argwhere(~linked.reshape(len(links), -1))]

    def test_post_step(self):
        """Every tenth frame, check finite poses and that no wire passed through its neighbor."""
        if round(self.sim_time / self.frame_dt) % 10 != 0:
            return
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "nonfinite body pose"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "nonfinite body velocity"
        unlinked = self.unlinked_pairs()
        assert not unlinked, f"links passed through their neighbors at (chain, link) {unlinked[:8]}"
        stretch = self.link_stretch()
        assert stretch < 1.15, f"a chain came apart: neighboring links {stretch:.2f} taut pitches apart"
        self._check_pendants()
        self._check_anchors()

    def _check_anchors(self):
        """Check that the bolted first links follow the prescribed anchor motion."""
        positions = self.state_0.body_q.numpy()[self.anchor_index, :3]
        initial = self.anchor_initial.numpy()[:, :3]
        t = np.clip((self.sim_time - 6.0) / 8.0, 0.0, 1.0)
        expected = initial.copy()
        expected[:, :2] += (
            initial[:, :2] / np.linalg.norm(initial[:, :2], axis=1)[:, None] * (0.65 * t * t * (3.0 - 2.0 * t))
        )
        np.testing.assert_allclose(positions, expected, atol=2.0e-6)

    def _check_pendants(self):
        """Check that each pendant still threads through its last link."""
        poses = self.state_0.body_q.numpy()
        linking = _link_intersections(
            poses[[bodies[-2] for bodies in self.chain_bodies]],
            poses[[bodies[-1] for bodies in self.chain_bodies]],
            self.pendant_curve,
        )
        assert np.all(np.isfinite(linking) & (np.abs(linking) > 0.5)), "a pendant passed through its last link"

    def test_final(self):
        """The chains have swung down below the ring and still hold together."""
        positions = self.state_0.body_q.numpy()[:, :3]
        pendants = positions[[bodies[-1] for bodies in self.chain_bodies]]
        print(f"PhoenX chain fan: pendants at {pendants[:, 2].min():.2f}..{pendants[:, 2].max():.2f} m")
        assert pendants[:, 2].max() < RING_HEIGHT - 0.25, "a chain did not swing down"
        assert self.link_stretch() < 1.15
        assert not self.unlinked_pairs(), "a chain came apart"
        self._check_pendants()
        self._check_anchors()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # Path-traced rendering, like the Kapla tower.
        parser.set_defaults(viewer="optix", optix_max_instances=65536)
        parser.add_argument("--chains", type=int, default=240, help="Chains around the anchor ring.")
        parser.add_argument("--spin", type=float, default=0.65, help="Initial rotation about each anchor [rad/s].")
        parser.add_argument("--links", type=int, default=40, help="Links per chain.")
        parser.add_argument("--collide-steps", type=int, default=4, help="Collision updates per frame (240 Hz).")
        parser.add_argument("--substeps", type=int, default=16, help="Physics substeps per collision update.")
        parser.add_argument("--gap", type=float, default=0.002, help="Authored contact gap [m].")
        parser.add_argument("--iterations", type=int, default=4, help="Solver iterations per substep.")
        parser.add_argument(
            "--speculative-max",
            type=float,
            default=SPECULATIVE_GAP_MAX,
            help="Largest speculative extension of a contact search [m].",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
