# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run packets over a cascade of large conveyor belts with PhoenX.

Command: python -m newton.examples phoenx_conveyor

Three static belt meshes carry packets with their surface velocity: 32 m, 30 m, and 34 m long
and 4 m wide, each feeding the next one over a drop. Packets of three sizes are created at the
start of the first belt from device code, each spawn spot waiting until its previous packet has
moved clear. A scanner gate on every belt is a tracking volume: the collision pipeline computes
exact contacts with it, PhoenX gives them no response, and a kernel reads them to recolor and
count the packets that pass. A sink volume below the last drop removes every packet that
enters it. Spawning, scanning, removal, collision, and stepping all run in one CUDA graph.

Success criteria: in test mode, packet states stay finite, packets ride the first belt at its
speed, every gate counts packets once they had time to reach it, the sink removes them, and
the packet count stays below the reserved capacity.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX, phoenx

BELT_WIDTH = 4.0  # [m]
BELT_THICKNESS = 0.2  # [m]
RAIL_HEIGHT = 0.3  # [m]
# (start, end, top height [m], speed [m/s]); each belt drops onto the next one.
BELTS = (
    ((-17.0, 0.0), (15.3, 0.0), 3.0, 3.0),
    ((17.0, -2.0), (17.0, 28.0), 1.5, 2.5),
    ((19.0, 30.0), (-15.0, 30.0), 0.5, 3.0),
)
GATES = ((0.0, 0.0), (17.0, 13.0), (2.0, 30.0))  # one scanner per belt, at these (x, y)
GATE_COLORS = ((1.0, 0.55, 0.1), (0.1, 0.8, 0.9), (0.9, 0.2, 0.75))
SINK = len(GATES)
PACKET_HALF_EXTENTS = ((0.25, 0.2, 0.15), (0.3, 0.3, 0.2), (0.2, 0.2, 0.2))
PACKET_DENSITY = 200.0  # [kg/m^3]
SPAWN_X = -15.5  # [m]
SPAWN_LANES = (-1.35, -0.45, 0.45, 1.35)  # [m], wider than two yawed packets
SPAWN_CLEARANCE = 1.1  # [m], more than two packet bounding radii
CAPACITY = 1024


def _box_mesh(center, half_extents, velocity):
    """Static box mesh whose surface moves with ``velocity`` [m/s]."""
    box = newton.Mesh.create_box(
        *half_extents, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
    )
    mesh = newton.Mesh(
        np.asarray(box.vertices) + np.asarray(center, dtype=np.float32),
        np.asarray(box.indices),
        compute_inertia=False,
        enable_surface_velocity=True,
    )
    return mesh, wp.vec3(*velocity)


@wp.kernel
def _spawn_kernel(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    lanes: wp.array[wp.vec3],
    lane_body: wp.array[wp.int32],
    half_extents: wp.array[wp.vec3],
    belt_velocity: wp.vec3,
    frame: wp.array[wp.int32],
    packet_kind: wp.array[wp.int32],
    packet_gate: wp.array[wp.int32],
    packet_claim: wp.array[wp.int32],
    body_start: int,
):
    lane = wp.tid()
    origin = lanes[lane]
    last = lane_body[lane]
    if last >= 0 and phoenx.is_body_active(dyn, last):
        if wp.length(wp.transform_get_translation(body_q[last]) - origin) < SPAWN_CLEARANCE:
            return
    rng = wp.rand_init(4242, frame[0] * lanes.shape[0] + lane)
    kind = wp.int32(wp.randi(rng, 0, half_extents.shape[0]))
    h = half_extents[kind]
    mass = PACKET_DENSITY * 8.0 * h[0] * h[1] * h[2]
    body_desc = phoenx.BodyDesc()
    body_desc.pose = wp.transform(
        origin + wp.vec3(0.0, 0.0, h[2]), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.randf(rng, -0.4, 0.4))
    )
    body_desc.velocity = wp.spatial_vector(belt_velocity, wp.vec3(0.0))
    body_desc.mass = mass
    body_desc.inertia = wp.diag(
        mass / 3.0 * wp.vec3(h[1] * h[1] + h[2] * h[2], h[0] * h[0] + h[2] * h[2], h[0] * h[0] + h[1] * h[1])
    )
    body = phoenx.add_body(dyn, body_desc)
    if body < 0:
        return
    shape = phoenx.ShapeDesc()
    shape.type = int(newton.GeoType.BOX)
    shape.transform = wp.transform_identity()
    shape.scale = h
    shape.gap = 0.02
    shape.mu = 0.6
    shape.collision_group = 1
    shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape.prototype = -1
    phoenx.add_shape(dyn, body, shape)
    lane_body[lane] = body
    slot = body - body_start
    packet_kind[slot] = kind
    packet_gate[slot] = -1
    packet_claim[slot] = 0


@wp.kernel
def _scan_kernel(
    dyn: phoenx.DynamicsData,
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_volume: wp.array[wp.int32],
    body_start: int,
    packet_gate: wp.array[wp.int32],
    packet_claim: wp.array[wp.int32],
    gate_count: wp.array[wp.int32],
):
    """Turn overlapping volume contacts into gate passes and sink removals."""
    k = wp.tid()
    if k >= count[0]:
        return
    a = shape0[k]
    b = shape1[k]
    volume = shape_volume[a]
    other = b
    if volume < 0:
        volume = shape_volume[b]
        other = a
    if volume < 0:
        return
    # Exact overlap: the contact's separation is not positive.
    p0 = point0[k]
    if shape_body[a] >= 0:
        p0 = wp.transform_point(body_q[shape_body[a]], p0)
    p1 = point1[k]
    if shape_body[b] >= 0:
        p1 = wp.transform_point(body_q[shape_body[b]], p1)
    if wp.dot(p1 - p0, normal[k]) - margin0[k] - margin1[k] > 0.0:
        return
    body = shape_body[other]
    slot = body - body_start
    if slot < 0 or slot >= packet_gate.shape[0]:
        return
    if volume == SINK:
        if wp.atomic_cas(packet_claim, slot, 0, 1) == 0:
            phoenx.remove_body(dyn, body)
    elif wp.atomic_max(packet_gate, slot, volume) < volume:
        wp.atomic_add(gate_count, volume, 1)


@wp.kernel
def _advance_frame_kernel(frame: wp.array[wp.int32]):
    frame[0] = frame[0] + 1


@wp.kernel
def _packet_instances_kernel(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    body_start: int,
    packet_kind: wp.array[wp.int32],
    packet_gate: wp.array[wp.int32],
    gate_colors: wp.array[wp.vec3],
    kind: int,
    xforms: wp.array[wp.transform],
    colors: wp.array[wp.vec3],
):
    """Instances of one packet size; other sizes and unused rows are parked out of view."""
    slot = wp.tid()
    body = body_start + slot
    xforms[slot] = wp.transform(wp.vec3(0.0, 0.0, -1000.0), wp.quat_identity())
    if phoenx.is_body_active(dyn, body) and packet_kind[slot] == kind:
        xforms[slot] = body_q[body]
    gate = packet_gate[slot]
    colors[slot] = wp.vec3(0.92)
    if gate >= 0:
        colors[slot] = gate_colors[gate]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        belt_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
        rail_cfg = newton.ModelBuilder.ShapeConfig(mu=0.05)
        visual = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        volume_cfg = newton.ModelBuilder.ShapeConfig(is_visible=False)
        self.belt_meshes = []
        for (x0, y0), (x1, y1), top, speed in BELTS:
            direction = np.array([x1 - x0, y1 - y0, 0.0])
            length = float(np.linalg.norm(direction))
            direction /= length
            along_x = abs(direction[0]) > 0.5
            half = (length / 2, BELT_WIDTH / 2) if along_x else (BELT_WIDTH / 2, length / 2)
            center = ((x0 + x1) / 2, (y0 + y1) / 2, top - BELT_THICKNESS / 2)
            mesh, velocity = _box_mesh(center, (*half, BELT_THICKNESS / 2), speed * direction)
            builder.add_shape_mesh(-1, mesh=mesh, cfg=belt_cfg, label="belt")
            self.belt_meshes.append((mesh, velocity))
            # Side rails along the belt; its ends stay open for the drops.
            side = np.array([-direction[1], direction[0], 0.0]) * (BELT_WIDTH / 2 + 0.05)
            rail_half = (length / 2, 0.05, RAIL_HEIGHT / 2) if along_x else (0.05, length / 2, RAIL_HEIGHT / 2)
            for sign in (-1.0, 1.0):
                rail_center = np.array((*center[:2], top + RAIL_HEIGHT / 2)) + sign * side
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(rail_center, wp.quat_identity()),
                    hx=rail_half[0],
                    hy=rail_half[1],
                    hz=rail_half[2],
                    cfg=rail_cfg,
                )

        # Scanner gates: an invisible tracking volume across each belt, inside a visual arch.
        volume_shapes = {}
        for gate, ((gx, gy), (start, end, top, _speed)) in enumerate(zip(GATES, BELTS, strict=True)):
            along_x = abs(end[0] - start[0]) > abs(end[1] - start[1])
            hx, hy = (0.1, BELT_WIDTH / 2) if along_x else (BELT_WIDTH / 2, 0.1)
            volume_shapes[
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((gx, gy, top + 0.8), wp.quat_identity()),
                    hx=hx,
                    hy=hy,
                    hz=0.8,
                    cfg=volume_cfg,
                )
            ] = gate
            color = GATE_COLORS[gate]
            post = np.array([0.0, BELT_WIDTH / 2 + 0.3, 0.0]) if along_x else np.array([BELT_WIDTH / 2 + 0.3, 0.0, 0.0])
            for sign in (-1.0, 1.0):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(np.array((gx, gy, top + 1.0)) + sign * post, wp.quat_identity()),
                    hx=0.12,
                    hy=0.12,
                    hz=1.0,
                    cfg=visual,
                    color=color,
                )
            builder.add_shape_box(
                -1,
                xform=wp.transform((gx, gy, top + 2.1), wp.quat_identity()),
                hx=hx if along_x else BELT_WIDTH / 2 + 0.42,
                hy=BELT_WIDTH / 2 + 0.42 if along_x else hy,
                hz=0.12,
                cfg=visual,
                color=color,
            )
        # The sink catches everything that leaves the last belt (or any belt).
        volume_shapes[
            builder.add_shape_box(
                -1, xform=wp.transform((0.0, 15.0, -4.0), wp.quat_identity()), hx=60.0, hy=60.0, hz=1.0, cfg=volume_cfg
            )
        ] = SINK
        builder.reserve_dynamic_capacity(bodies=CAPACITY, shapes=CAPACITY)
        self.model = builder.finalize()
        self.body_start = self.model._dynamic_capacity.body_start
        for mesh, velocity in self.belt_meshes:
            mesh.mesh.velocities.fill_(velocity)
        shape_volume = np.full(self.model.shape_count, -1, dtype=np.int32)
        for shape, volume in volume_shapes.items():
            shape_volume[shape] = volume
        self.shape_volume = wp.array(shape_volume, dtype=wp.int32)

        self.solver = SolverPhoenX(
            self.model,
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
            tracking_volumes=list(volume_shapes),
        )
        self.dyn = self.solver.dynamics.data
        self.pipeline = self.model._collision_pipeline
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        top = BELTS[0][2]
        self.lanes = wp.array([(SPAWN_X, y, top) for y in SPAWN_LANES], dtype=wp.vec3)
        self.belt_velocity = wp.vec3(BELTS[0][3], 0.0, 0.0)
        self.lane_body = wp.full(len(SPAWN_LANES), -1, dtype=wp.int32)
        self.half_extents = wp.array(PACKET_HALF_EXTENTS, dtype=wp.vec3)
        self.frame = wp.zeros(1, dtype=wp.int32)
        self.packet_kind = wp.zeros(CAPACITY, dtype=wp.int32)
        self.packet_gate = wp.full(CAPACITY, -1, dtype=wp.int32)
        self.packet_claim = wp.zeros(CAPACITY, dtype=wp.int32)
        self.gate_count = wp.zeros(len(GATES), dtype=wp.int32)
        self.gate_colors = wp.array(GATE_COLORS, dtype=wp.vec3)
        self.instance_xforms = [wp.zeros(CAPACITY, dtype=wp.transform) for _ in PACKET_HALF_EXTENTS]
        self.instance_colors = [wp.zeros(CAPACITY, dtype=wp.vec3) for _ in PACKET_HALF_EXTENTS]

        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(-20.0, -16.0, 30.0), pitch=-42.0, yaw=45.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(
            _spawn_kernel,
            dim=self.lanes.shape[0],
            inputs=[
                self.dyn,
                self.state_0.body_q,
                self.lanes,
                self.lane_body,
                self.half_extents,
                self.belt_velocity,
                self.frame,
                self.packet_kind,
                self.packet_gate,
                self.packet_claim,
                self.body_start,
            ],
        )
        self.pipeline.collide(self.state_0, self.contacts)
        contacts = self.contacts
        wp.launch(
            _scan_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                self.dyn,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.state_0.body_q,
                self.shape_volume,
                self.body_start,
                self.packet_gate,
                self.packet_claim,
                self.gate_count,
            ],
        )
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
        wp.copy(self.state_0.body_q, self.state_1.body_q)
        wp.copy(self.state_0.body_qd, self.state_1.body_qd)
        wp.launch(_advance_frame_kernel, dim=1, inputs=[self.frame])

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for kind, half_extents in enumerate(PACKET_HALF_EXTENTS):
            wp.launch(
                _packet_instances_kernel,
                dim=CAPACITY,
                inputs=[
                    self.dyn,
                    self.state_0.body_q,
                    self.body_start,
                    self.packet_kind,
                    self.packet_gate,
                    self.gate_colors,
                    kind,
                    self.instance_xforms[kind],
                    self.instance_colors[kind],
                ],
            )
            self.viewer.log_shapes(
                f"/packets_{kind}",
                newton.GeoType.BOX,
                half_extents,
                self.instance_xforms[kind],
                self.instance_colors[kind],
            )
        self.viewer.end_frame()

    def live_packets(self) -> np.ndarray:
        """Body indices of the packets currently on the belts."""
        state = self.solver.dynamics.bodies.data.state.numpy()[:CAPACITY]
        return self.body_start + np.flatnonzero((state == 1) | (state == 2))

    def test_post_step(self):
        """Every second, check finite state, belt speed on the first belt, and the capacity."""
        if round(self.sim_time / self.frame_dt) % 60 != 0:
            return
        live = self.live_packets()
        q = self.state_0.body_q.numpy()[live]
        qd = self.state_0.body_qd.numpy()[live]
        assert np.isfinite(q).all() and np.isfinite(qd).all(), "nonfinite packet state"
        assert len(live) < CAPACITY - 16, f"packets are piling up: {len(live)}"
        on_first = (q[:, 0] > SPAWN_X + 2.0) & (q[:, 0] < 15.0) & (q[:, 2] > BELTS[0][2])
        if np.any(on_first):
            speed = qd[on_first, 0].mean()
            assert abs(speed - BELTS[0][3]) < 0.2, f"packets do not ride the belt: {speed:.3f} m/s"

    def test_final(self):
        """Gates count packets once they had time to reach them, and the sink removes them."""
        counts = self.gate_count.numpy()
        live = len(self.live_packets())
        print(f"PhoenX conveyor: {live} packets, gate passes {counts.tolist()} after {self.sim_time:.1f} s")
        assert int(self.solver.dynamics.error_count.numpy()[0]) == 0
        assert live > 0
        # Arrival times at each gate at belt speed, with margin for the drops.
        for gate, arrival in enumerate((6.0, 20.0, 34.0)):
            if self.sim_time > arrival:
                assert counts[gate] > 0, f"gate {gate} saw no packets"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # Path-traced rendering, like the Kapla tower, which also suits video recording.
        parser.set_defaults(viewer="optix")
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=4, help="Solver iterations per substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
