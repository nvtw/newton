# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Stand a four-level house of playing cards in PhoenX, then knock it down.

Command: python -m newton.examples phoenx_house_of_cards

Twenty-six playing cards (63.5 x 88.9 mm, 0.7 mm thick by default, 1.8 g) form the classic
four-level house: ten tents of two leaning cards with six flat cards bridging their tops,
held up only by friction (mu = 0.8). The house must stand still under gravity, which needs
accurate static friction at the thin card edges. At 6.7 s four projectiles are created at
runtime and fired into one side; the failure starts there and propagates through the
house. ``--impact-time`` delays the shot, and a negative value keeps the house standing.
Drag a card with the right mouse button to pick it. The scene follows the house-of-cards benchmark of Song et al., "A Splitting
Architecture for Exact Reduced Coulomb Friction" (2026), also shown as a MuJoCo demo.
``--card-thickness 0.3`` uses the thickness of real playing cards.

The solver runs sixteen substeps of six iterations. Compared with eight substeps, they
halve the penetration that an impact on a light card builds up within one substep and
quarter the compliance of loaded contacts, which keeps 0.3 mm cards from passing through
each other; six iterations are the fewest that hold the 0.7 mm house up while it settles.

Success criteria: in test mode, the house settles by closing its initial clearances, no
card creeps more than 0.5 mm while it stands before the impact, all states stay finite,
the projectiles knock cards down, and at the end no card corner lies deeper than half a
card thickness below the ground or inside another card.
"""

import itertools
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX, phoenx

CARD_WIDTH, CARD_HEIGHT = 0.0635, 0.0889  # poker card [m]
CARD_MASS = 0.0018  # [kg]
LEAN = math.radians(20.0)  # tilt of each tent card from vertical
LEVELS = 4
FRICTION = 0.8
PITCH = 0.85 * CARD_HEIGHT  # tent spacing; flat cards overhang each supporting apex [m]
CLEARANCE = 1.0e-4  # initial gap between touching cards [m]
PROJECTILES = 4
PROJECTILE_RADIUS = 0.012  # [m]
PROJECTILE_MASS = 0.01  # [kg]
PROJECTILE_SPEED = 2.5  # [m/s]


def card_poses(thickness: float, levels: int = LEVELS) -> list[tuple[wp.transform, bool]]:
    """Poses of all cards and whether each one lies flat, built level by level."""
    h, t = 0.5 * CARD_HEIGHT, 0.5 * thickness
    s, c = math.sin(LEAN), math.cos(LEAN)
    lean_left = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), LEAN)
    lean_right = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -LEAN)
    flat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
    poses = []
    base = 0.0
    for level in range(levels):
        tents = levels - level
        centers = [(i - 0.5 * (tents - 1)) * PITCH for i in range(tents)]
        # Each card stands on its lowest edge and touches its partner's top edge at the apex.
        z = base + h * c + t * s
        dx = h * s + t * c + CLEARANCE
        for x in centers:
            poses.append((wp.transform(wp.vec3(x - dx, 0.0, z), lean_left), False))
            poses.append((wp.transform(wp.vec3(x + dx, 0.0, z), lean_right), False))
        # Flat cards bridge neighbouring apexes; the outer top corners are the highest points.
        top = base + 2.0 * h * c + 2.0 * t * s
        for a, b in itertools.pairwise(centers):
            poses.append((wp.transform(wp.vec3(0.5 * (a + b), 0.0, top + t + CLEARANCE), flat), True))
        base = top + 2.0 * t + CLEARANCE
    return poses


def _rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate row vectors by an ``(x, y, z, w)`` quaternion."""
    xyz, w = quaternion[:3], quaternion[3]
    t = 2.0 * np.cross(xyz, vectors)
    return vectors + w * t + np.cross(xyz, t)


@wp.kernel
def _projectile_xforms_kernel(
    body_q: wp.array[wp.transform],
    projectiles: wp.array[wp.int32],
    xforms: wp.array[wp.transform],
):
    """Instance transforms of the projectiles; ones not fired yet are parked out of sight."""
    i = wp.tid()
    body = projectiles[i]
    xforms[i] = wp.transform(wp.vec3(0.0, 0.0, -1000.0), wp.quat_identity())
    if body >= 0:
        xforms[i] = body_q[body]


@wp.kernel
def _launch_kernel(
    dyn: phoenx.DynamicsData,
    target: wp.vec3,
    projectiles: wp.array[wp.int32],
):
    """Create the projectiles beside the house, moving into its lower right side."""
    i = wp.tid()
    radius = wp.float32(PROJECTILE_RADIUS)
    inertia = wp.float32(0.4 * PROJECTILE_MASS) * radius * radius
    start = target + wp.vec3(0.15, (wp.float32(i) - 1.5) * 2.5 * radius, wp.float32(i % 2) * 3.0 * radius)
    desc = phoenx.body_desc(
        wp.transform(start, wp.quat_identity()), wp.float32(PROJECTILE_MASS), wp.identity(3, dtype=wp.float32) * inertia
    )
    desc.velocity = wp.spatial_vector(wp.normalize(target - start) * wp.float32(PROJECTILE_SPEED), wp.vec3(0.0))
    body = phoenx.add_body(dyn, desc)
    projectiles[i] = body
    if body >= 0:
        shape = phoenx.shape_desc(wp.int32(newton.GeoType.SPHERE), wp.vec3(radius, 0.0, 0.0))
        shape.mu = wp.float32(FRICTION)
        phoenx.add_shape(dyn, body, shape)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.frame = 0
        # A negative impact time never fires.
        self.impact_frame = round(args.impact_time / self.frame_dt) if args.impact_time >= 0.0 else -1

        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=FRICTION))
        thickness = args.card_thickness * 1.0e-3
        self.thickness = thickness
        card = newton.ModelBuilder.ShapeConfig(
            density=CARD_MASS / (CARD_WIDTH * CARD_HEIGHT * thickness), mu=FRICTION, restitution=0.0
        )
        self.flat = []
        if args.levels < 1:
            raise ValueError("the house needs at least one level")
        self.levels = args.levels
        for pose, flat in card_poses(thickness, self.levels):
            body = builder.add_body(xform=pose)
            color = (0.93, 0.92, 0.88) if flat else (0.72, 0.1, 0.12)
            builder.add_shape_box(
                body, hx=0.5 * thickness, hy=0.5 * CARD_WIDTH, hz=0.5 * CARD_HEIGHT, cfg=card, color=color
            )
            self.flat.append(flat)
        self.card_count = builder.body_count
        builder.reserve_dynamic_capacity(bodies=PROJECTILES, shapes=PROJECTILES)
        self.model = builder.finalize()

        self.solver = SolverPhoenX(
            self.model,
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
        )
        self.dyn = self.solver.dynamics.data
        self.pipeline = self.model._collision_pipeline
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.projectiles = wp.full(PROJECTILES, -1, dtype=wp.int32)
        # Runtime bodies are not part of the model the viewer was given, so draw them here.
        self.projectile_xforms = wp.zeros(PROJECTILES, dtype=wp.transform)
        self.projectile_colors = wp.full(PROJECTILES, wp.vec3(0.15, 0.35, 0.85), dtype=wp.vec3)
        self.initial_q = self.state_0.body_q.numpy()[: self.card_count].copy()
        self.settled_q = None
        self.max_rest_drift = 0.0

        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(0.32, -0.42, 0.22), pitch=-18.0, yaw=128.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
        for name in ("body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))

    def step(self):
        if self.frame == self.impact_frame:
            # Aim at the lower edge of the right-most bottom tent.
            target = wp.vec3(0.5 * (self.levels - 1) * PITCH, 0.0, 0.045)
            wp.launch(_launch_kernel, dim=PROJECTILES, inputs=[self.dyn, target, self.projectiles])
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        wp.launch(
            _projectile_xforms_kernel,
            dim=PROJECTILES,
            inputs=[self.state_0.body_q, self.projectiles],
            outputs=[self.projectile_xforms],
        )
        self.viewer.log_shapes(
            "/projectiles", newton.GeoType.SPHERE, (PROJECTILE_RADIUS,), self.projectile_xforms, self.projectile_colors
        )
        self.viewer.end_frame()

    def card_displacement(self, reference: np.ndarray) -> np.ndarray:
        """Distance each card's center moved from ``reference`` [m]."""
        q = self.state_0.body_q.numpy()[: self.card_count]
        return np.linalg.norm(q[:, :3] - reference[:, :3], axis=1)

    def test_post_step(self):
        """Before the impact the settled house stands still; afterwards every state stays finite."""
        q = self.state_0.body_q.numpy()
        assert np.isfinite(q).all(), "nonfinite body pose"
        if self.frame == 60:
            self.settled_q = q[: self.card_count].copy()
            # Settling only closes the initial clearances.
            assert self.card_displacement(self.initial_q).max() < 2.0e-3, "the house did not settle in place"
        elif self.settled_q is not None and (self.impact_frame < 0 or self.frame <= self.impact_frame):
            self.max_rest_drift = max(self.max_rest_drift, float(self.card_displacement(self.settled_q).max()))
            assert self.max_rest_drift < 0.5e-3, f"a standing card crept {1e3 * self.max_rest_drift:.2f} mm"

    def deepest_card_overlaps(self) -> tuple[float, float]:
        """Deepest card corner below the ground and deepest card-card contact [m], both <= 0."""
        q = self.state_0.body_q.numpy()
        half = np.array([0.5 * self.thickness, 0.5 * CARD_WIDTH, 0.5 * CARD_HEIGHT])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        ground = min(
            [0.0] + [float(_rotate(q[i, 3:], signs * half)[:, 2].min() + q[i, 2]) for i in range(self.card_count)]
        )
        self.pipeline.collide(self.state_0, self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        body0 = self.model.shape_body.numpy()[self.contacts.rigid_contact_shape0.numpy()[:count]]
        body1 = self.model.shape_body.numpy()[self.contacts.rigid_contact_shape1.numpy()[:count]]
        point0 = self.contacts.rigid_contact_point0.numpy()[:count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:count]
        normal = self.contacts.rigid_contact_normal.numpy()[:count]
        cards = (body0 >= 0) & (body0 < self.card_count) & (body1 >= 0) & (body1 < self.card_count)
        between = 0.0
        for k in np.flatnonzero(cards):
            a, b = body0[k], body1[k]
            world0 = q[a, :3] + _rotate(q[a, 3:], point0[k][None])[0]
            world1 = q[b, :3] + _rotate(q[b, 3:], point1[k][None])[0]
            between = min(between, float(np.dot(world1 - world0, normal[k])))
        return ground, between

    def test_final(self):
        """The house stood, the projectiles knocked it down, and no card ended inside another or the ground."""
        moved = self.card_displacement(self.initial_q)
        fallen = int(np.count_nonzero(moved > 0.5 * CARD_HEIGHT))
        ground, between = self.deepest_card_overlaps()
        print(
            f"PhoenX house of cards: rest drift {1e3 * self.max_rest_drift:.3f} mm, "
            f"{fallen} of {self.card_count} cards knocked down, deepest corner below ground "
            f"{-1e3 * ground:.3f} mm, deepest card-card overlap {-1e3 * between:.3f} mm"
        )
        assert int(self.solver.dynamics.error_count.numpy()[0]) == 0
        if 0 <= self.impact_frame < self.frame:
            assert (self.projectiles.numpy() >= 0).all(), "a projectile was not created"
            assert fallen > 0, "the projectiles did not knock any card down"
        assert ground > -0.5 * self.thickness, f"a card corner ended {-1e3 * ground:.2f} mm below the ground"
        assert between > -0.5 * self.thickness, f"two cards ended {-1e3 * between:.2f} mm inside each other"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix", num_frames=600)
        parser.add_argument("--card-thickness", type=float, default=0.7, help="Card thickness [mm].")
        parser.add_argument("--substeps", type=int, default=16, help="Physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=6, help="Solver iterations per substep.")
        parser.add_argument("--levels", type=int, default=LEVELS, help="Levels of the house (26 cards at 4).")
        parser.add_argument(
            "--impact-time", type=float, default=6.7, help="Time the projectiles are fired [s]; negative never fires."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
