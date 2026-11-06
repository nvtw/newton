# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sort around ten thousand suitcases in an airport baggage hall with PhoenX.

Command: python -m newton.examples phoenx_baggage_hall

Eight sorting lines run side by side. On each line, eight check-in counters put suitcases with
a flight tag on feeder belts that drop them onto a long collector belt. The collector turns into
a sorter with one diverter per flight: a tracking volume across the belt detects each passing
suitcase exactly, and when the tag matches its flight it pushes the suitcase sideways onto that
flight's chute. Chutes drop the suitcases into baggage carts. A cart departs once it is full, at
a randomized fill level so carts do not leave in lockstep, and its suitcases disappear; the
cart walls brighten as they fill. Suitcases that miss their diverter, or fall off a belt, end
in sinks. Belts are static meshes with surface velocity.

Suitcases are created and removed at runtime from device code. Spawning, diverting, cart
dispatch, collision, and stepping all run in one CUDA graph without host synchronization. The
hall settles at roughly ten thousand live suitcases after about two simulated minutes.

Success criteria: in test mode, suitcase states stay finite, suitcases reach the carts of their
flight, the live count stays below the reserved capacity, and carts depart once they had time
to fill.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.dynamic import (
    BodyDesc,
    DynamicsData,
    ShapeDesc,
    phoenx_add_body,
    phoenx_add_shape,
    phoenx_body_is_active,
    phoenx_remove_body,
)
from newton.solvers import SolverPhoenX

LINES = 8
FLIGHTS = 8  # per line: one counter, diverter, chute, and cart each
LINE_SPACING = 28.0  # [m]

FEEDER_TOP = 2.8  # [m]
FEEDER_WIDTH = 1.1
FEEDER_SPEED = 1.5  # [m/s]
FEEDER_START = 11.0  # [m], distance of the counters from the sorter centerline
COUNTER_SPACING = 4.5  # [m]
SORTER_TOP = 2.5
SORTER_WIDTH = 2.4
COLLECTOR_SPEED = 1.6
SORTER_SPEED = 4.0  # faster than the collector, which opens gaps between suitcases
SORTER_START = -2.0  # [m] along x
SORTER_END = 106.0
DIVERTER_START = 46.0  # x of the first diverter [m]
DIVERTER_SPACING = 7.5
CHUTE_OFFSET = 1.6  # chutes sit downstream of their diverter, where pushed suitcases arrive [m]
FUNNEL_LENGTH = 8.0  # guides at the sorter entrance center suitcases away from the chute mouths [m]
FUNNEL_HALF_WIDTH = 0.85
GUIDE_HALF_WIDTH = 0.5  # side guides steer undiverted suitcases beyond this back to the middle [m]
GUIDE_SPEED = 0.8  # [m/s]
PUSH_SPEED = 2.0  # sideways diverter push [m/s]
CHUTE_TOP = 2.15
CHUTE_WIDTH = 2.6
CHUTE_SPEED = 2.0
CHUTE_END = 8.6  # [m] from the sorter centerline, overhanging the cart wall
CART_NEAR = 8.0  # [m] from the sorter centerline
CART_DEPTH = 4.0
CART_FLOOR = 0.4  # [m] cart floor height; the hall floor below is a sink
CART_WALL = 1.6
CART_FILL_MIN = 120  # carts depart at a random fill level in this range
CART_FILL_MAX = 220
THICKNESS = 0.2
RAIL = 0.25

CASE_HALF_EXTENTS = ((0.35, 0.24, 0.12), (0.29, 0.2, 0.1), (0.21, 0.16, 0.09))
CASE_DENSITY = 160.0  # [kg/m^3]
SPAWN_PROBABILITY = 0.03  # per frame and free counter
SPAWN_CLEARANCE = 1.0  # [m]
CAPACITY = 14000
MAX_LIVE = 12000  # counters pause above this
SPECULATIVE_GAP = 0.3  # [m], speculative contact search for dropping suitcases
CONTACT_MAX = 400_000

GATE, CART, SINK = 0, 1, 2
FLIGHT_COLORS = (
    (0.95, 0.3, 0.25),
    (1.0, 0.65, 0.1),
    (0.95, 0.9, 0.2),
    (0.35, 0.85, 0.3),
    (0.2, 0.8, 0.85),
    (0.3, 0.45, 1.0),
    (0.7, 0.35, 0.95),
    (0.95, 0.4, 0.75),
)


def _moving_box(builder, belts, center, half_extents, velocity, color):
    """Static belt: a box mesh whose surface moves with ``velocity``.

    Belts of one size and velocity share a mesh, which keeps the viewer at a few draw batches.
    """
    key = (tuple(half_extents), tuple(velocity))
    if key not in belts:
        box = newton.Mesh.create_box(
            *half_extents, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
        )
        mesh = newton.Mesh(box.vertices, box.indices, compute_inertia=False, enable_surface_velocity=True)
        belts[key] = (mesh, wp.vec3(*velocity))
    builder.add_shape_mesh(
        -1,
        xform=wp.transform(center, wp.quat_identity()),
        mesh=belts[key][0],
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.9),
        color=color,
    )


def _box(builder, center, half_extents, cfg, color=None):
    return builder.add_shape_box(
        -1,
        xform=wp.transform(center, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        color=color,
    )


@wp.kernel
def _spawn_kernel(
    dyn: DynamicsData,
    body_q: wp.array[wp.transform],
    counters: wp.array[wp.vec3],
    counter_body: wp.array[wp.int32],
    half_extents: wp.array[wp.vec3],
    frame: wp.array[wp.int32],
    body_start: int,
    case_kind: wp.array[wp.int32],
    case_flight: wp.array[wp.int32],
    case_diverted: wp.array[wp.int32],
    case_claimed: wp.array[wp.int32],
    case_cart: wp.array[wp.int32],
):
    """Check in a suitcase at a counter whose last suitcase has moved clear."""
    counter = wp.tid()
    origin = counters[counter]
    last = counter_body[counter]
    if last >= 0 and phoenx_body_is_active(dyn, last):
        if wp.length(wp.transform_get_translation(body_q[last]) - origin) < SPAWN_CLEARANCE:
            return
    if dyn.body_pool.counts[0] >= MAX_LIVE:
        return
    rng = wp.rand_init(97, frame[0] * counters.shape[0] + counter)
    if wp.randf(rng) > SPAWN_PROBABILITY:
        return
    kind = wp.int32(wp.randi(rng, 0, half_extents.shape[0]))
    h = half_extents[kind]
    mass = CASE_DENSITY * 8.0 * h[0] * h[1] * h[2]
    body_desc = BodyDesc()
    body_desc.pose = wp.transform(
        origin + wp.vec3(0.0, 0.0, h[2]), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.randf(rng, -0.3, 0.3))
    )
    body_desc.velocity = wp.spatial_vector(0.0, -FEEDER_SPEED, 0.0, 0.0, 0.0, 0.0)
    body_desc.mass = mass
    body_desc.inertia = wp.diag(
        mass / 3.0 * wp.vec3(h[1] * h[1] + h[2] * h[2], h[0] * h[0] + h[2] * h[2], h[0] * h[0] + h[1] * h[1])
    )
    body = phoenx_add_body(dyn, body_desc)
    if body < 0:
        return
    shape = ShapeDesc()
    shape.type = int(newton.GeoType.BOX)
    shape.transform = wp.transform_identity()
    shape.scale = h
    shape.gap = 0.02
    shape.mu = 0.7
    shape.collision_group = 1
    shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape.prototype = -1
    phoenx_add_shape(dyn, body, shape)
    counter_body[counter] = body
    slot = body - body_start
    case_kind[slot] = kind
    line = counter / FLIGHTS
    case_flight[slot] = line * FLIGHTS + wp.int32(wp.randi(rng, 0, FLIGHTS))
    case_diverted[slot] = 0
    case_claimed[slot] = 0
    case_cart[slot] = -1


@wp.func
def _remove_case(dyn: DynamicsData, body: wp.int32, slot: wp.int32, case_claimed: wp.array[wp.int32]):
    if wp.atomic_cas(case_claimed, slot, 0, 1) == 0:
        phoenx_remove_body(dyn, body)


@wp.kernel
def _scan_kernel(
    dyn: DynamicsData,
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
    volume_kind: wp.array[wp.int32],
    volume_index: wp.array[wp.int32],
    body_start: int,
    frame: wp.array[wp.int32],
    case_flight: wp.array[wp.int32],
    case_diverted: wp.array[wp.int32],
    case_claimed: wp.array[wp.int32],
    case_seen: wp.array[wp.int32],
    case_cart: wp.array[wp.int32],
    cart_count: wp.array[wp.int32],
    cart_departing: wp.array[wp.int32],
):
    """Act on suitcases that overlap a diverter, a cart, or a sink this frame."""
    k = wp.tid()
    if k >= count[0]:
        return
    a = shape0[k]
    b = shape1[k]
    volume = a
    case_shape = b
    if volume_kind[a] < 0:
        volume = b
        case_shape = a
    kind = volume_kind[volume]
    if kind < 0:
        return
    body = shape_body[case_shape]
    slot = body - body_start
    if slot < 0 or slot >= case_flight.shape[0]:
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
    index = volume_index[volume]
    if kind == SINK:
        _remove_case(dyn, body, slot, case_claimed)
    elif kind == GATE:
        if case_flight[slot] == index:
            case_diverted[slot] = 1
    elif kind == CART:
        if cart_departing[index] > 0:
            _remove_case(dyn, body, slot, case_claimed)
        elif wp.atomic_exch(case_seen, slot, frame[0]) != frame[0]:
            wp.atomic_add(cart_count, index, 1)
            case_cart[slot] = index


@wp.kernel
def _pusher_kernel(
    dyn: DynamicsData,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_start: int,
    case_flight: wp.array[wp.int32],
    case_diverted: wp.array[wp.int32],
):
    """Sweep diverted suitcases across the sorter onto their chute, like a pusher arm, and
    steer all others back to the middle, like side guides.

    One kick would not do: belt friction stops the sideways motion within a few centimeters.
    """
    slot = wp.tid()
    body = body_start + slot
    if not phoenx_body_is_active(dyn, body):
        return
    p = wp.transform_get_translation(body_q[body])
    line = wp.round(p[1] / LINE_SPACING)
    offset = p[1] - line * LINE_SPACING
    if wp.abs(offset) > SORTER_WIDTH / 2.0 or p[2] < SORTER_TOP - 0.05 or p[0] < DIVERTER_START - 3.0:
        return
    # A suitcase that passed its chute mouth goes on to the overflow sink, never into another cart.
    mouth = DIVERTER_START + DIVERTER_SPACING * float(case_flight[slot] % FLIGHTS) + CHUTE_OFFSET
    if case_diverted[slot] != 0 and p[0] > mouth + CHUTE_WIDTH / 2.0:
        case_diverted[slot] = 0
    qd = body_qd[body]
    if case_diverted[slot] != 0:
        # Odd flights leave to the left of the travel direction, even ones to the right.
        side = wp.where(case_flight[slot] % 2 == 0, -1.0, 1.0)
        body_qd[body] = wp.spatial_vector(SORTER_SPEED, side * PUSH_SPEED, qd[2], qd[3], qd[4], qd[5])
    elif wp.abs(offset) > GUIDE_HALF_WIDTH:
        body_qd[body] = wp.spatial_vector(qd[0], -wp.sign(offset) * GUIDE_SPEED, qd[2], qd[3], qd[4], qd[5])


@wp.kernel
def _cart_kernel(
    frame: wp.array[wp.int32],
    cart_count: wp.array[wp.int32],
    cart_threshold: wp.array[wp.int32],
    cart_departing: wp.array[wp.int32],
    cart_departures: wp.array[wp.int32],
    cart_wall_start: wp.array[wp.int32],
    flight_colors: wp.array[wp.vec3],
    shape_color: wp.array[wp.vec3],
):
    """Dispatch full carts and tint cart walls by fill level."""
    cart = wp.tid()
    count = cart_count[cart]
    cart_count[cart] = 0
    fill = wp.min(float(count) / float(cart_threshold[cart]), 1.0)
    if cart_departing[cart] > 0:
        cart_departing[cart] = cart_departing[cart] - 1
        fill = 1.5
    elif count >= cart_threshold[cart]:
        # Keep removing for a few frames so every suitcase inside is caught.
        cart_departing[cart] = 3
        rng = wp.rand_init(331, frame[0] * cart_count.shape[0] + cart)
        cart_threshold[cart] = wp.randi(rng, CART_FILL_MIN, CART_FILL_MAX)
        wp.atomic_add(cart_departures, 0, 1)
    color = flight_colors[cart % FLIGHTS] * (0.3 + 0.55 * fill)
    for wall in range(5):
        shape_color[cart_wall_start[cart] + wall] = color


@wp.kernel
def _advance_frame_kernel(frame: wp.array[wp.int32]):
    frame[0] = frame[0] + 1


@wp.kernel
def _case_instances_kernel(
    dyn: DynamicsData,
    body_q: wp.array[wp.transform],
    body_start: int,
    case_kind: wp.array[wp.int32],
    case_flight: wp.array[wp.int32],
    flight_colors: wp.array[wp.vec3],
    half_extents: wp.array[wp.vec3],
    xforms: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    """One unit-box instance per suitcase slot, scaled to its size; unused slots vanish."""
    slot = wp.tid()
    body = body_start + slot
    xforms[slot] = body_q[body]
    scales[slot] = wp.vec3(0.0)
    if phoenx_body_is_active(dyn, body):
        scales[slot] = half_extents[case_kind[slot]]
    shade = 0.75 + 0.25 * float(slot % 5) / 4.0
    colors[slot] = flight_colors[case_flight[slot] % FLIGHTS] * shade


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.add_ground_plane(color=(0.32, 0.33, 0.35))
        rail_cfg = newton.ModelBuilder.ShapeConfig(mu=0.1)
        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        visual = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        volume_cfg = newton.ModelBuilder.ShapeConfig(is_visible=False)
        belt_color = (0.16, 0.17, 0.19)
        rail_color = (0.85, 0.75, 0.2)
        volumes = {}  # shape -> (kind, index)
        self.belts = {}
        counters = []
        cart_wall_start = []
        collector_end = DIVERTER_START - 3.0
        for line in range(LINES):
            y0 = line * LINE_SPACING
            # Collector, then a faster sorter; both along +x. The collector is guarded on the far
            # side of the counters.
            for start, end, speed in (
                (SORTER_START, collector_end, COLLECTOR_SPEED),
                (collector_end, SORTER_END, SORTER_SPEED),
            ):
                _moving_box(
                    builder,
                    self.belts,
                    ((start + end) / 2, y0, SORTER_TOP - THICKNESS / 2),
                    ((end - start) / 2, SORTER_WIDTH / 2, THICKNESS / 2),
                    (speed, 0, 0),
                    belt_color,
                )
            _box(
                builder,
                ((SORTER_START + collector_end) / 2, y0 - SORTER_WIDTH / 2 - 0.05, SORTER_TOP + RAIL / 2),
                ((collector_end - SORTER_START) / 2, 0.05, RAIL / 2),
                rail_cfg,
                rail_color,
            )
            for flight in range(FLIGHTS):
                index = line * FLIGHTS + flight
                color = FLIGHT_COLORS[flight]
                # Check-in counter and its feeder belt toward the collector.
                x = 1.5 + COUNTER_SPACING * flight
                feeder_near = y0 + SORTER_WIDTH / 2 + 0.05
                feeder_len = FEEDER_START - feeder_near + y0 + 0.8
                feeder_center = (x, feeder_near + feeder_len / 2, FEEDER_TOP - THICKNESS / 2)
                _moving_box(
                    builder,
                    self.belts,
                    feeder_center,
                    (FEEDER_WIDTH / 2, feeder_len / 2, THICKNESS / 2),
                    (0, -FEEDER_SPEED, 0),
                    belt_color,
                )
                for sign in (-1.0, 1.0):
                    _box(
                        builder,
                        (x + sign * (FEEDER_WIDTH / 2 + 0.05), feeder_center[1], FEEDER_TOP + RAIL / 2),
                        (0.05, feeder_len / 2, RAIL / 2),
                        rail_cfg,
                        rail_color,
                    )
                _box(builder, (x, y0 + FEEDER_START + 1.2, 1.0), (1.2, 0.35, 1.0), visual, (0.6, 0.62, 0.66))
                _box(builder, (x, y0 + FEEDER_START + 1.2, 2.05), (1.0, 0.36, 0.06), visual, color)
                counters.append((x, y0 + FEEDER_START, FEEDER_TOP))

                # Diverter: a tracking volume across the sorter and a visual gantry in the flight color.
                side = -1.0 if flight % 2 == 0 else 1.0
                gx = DIVERTER_START + DIVERTER_SPACING * flight
                gate = _box(builder, (gx, y0, SORTER_TOP + 0.5), (0.3, SORTER_WIDTH / 2, 0.5), volume_cfg)
                volumes[gate] = (GATE, index)
                for post in (-1.0, 1.0):
                    _box(
                        builder,
                        (gx, y0 + post * (SORTER_WIDTH / 2 + 0.2), SORTER_TOP + 0.6),
                        (0.08, 0.08, 0.6),
                        visual,
                        color,
                    )
                _box(builder, (gx, y0, SORTER_TOP + 1.25), (0.1, SORTER_WIDTH / 2 + 0.28, 0.06), visual, color)

                # Chute from the sorter edge to the cart, offset downstream to catch pushed suitcases.
                cx = gx + CHUTE_OFFSET
                chute_near = SORTER_WIDTH / 2 + 0.02
                chute_len = CHUTE_END - chute_near
                chute_center = (cx, y0 + side * (chute_near + chute_len / 2), CHUTE_TOP - THICKNESS / 2)
                _moving_box(
                    builder,
                    self.belts,
                    chute_center,
                    (CHUTE_WIDTH / 2, chute_len / 2, THICKNESS / 2),
                    (0, side * CHUTE_SPEED, 0),
                    belt_color,
                )
                for sign in (-1.0, 1.0):
                    _box(
                        builder,
                        (cx + sign * (CHUTE_WIDTH / 2 + 0.05), chute_center[1], CHUTE_TOP + RAIL / 2),
                        (0.05, chute_len / 2, RAIL / 2),
                        rail_cfg,
                        rail_color,
                    )

                # Baggage cart: floor and four walls, tinted by fill level, around a tracking volume.
                cy = y0 + side * (CART_NEAR + CART_DEPTH / 2)
                cart_half = (CHUTE_WIDTH / 2 + 0.3, CART_DEPTH / 2)
                floor_shape = _box(builder, (cx, cy, CART_FLOOR - 0.1), (cart_half[0], cart_half[1], 0.1), wall_cfg)
                cart_wall_start.append(floor_shape)
                wall_z = CART_FLOOR + CART_WALL / 2
                _box(builder, (cx - cart_half[0], cy, wall_z), (0.06, cart_half[1], CART_WALL / 2), wall_cfg)
                _box(builder, (cx + cart_half[0], cy, wall_z), (0.06, cart_half[1], CART_WALL / 2), wall_cfg)
                _box(builder, (cx, cy - cart_half[1], wall_z), (cart_half[0], 0.06, CART_WALL / 2), wall_cfg)
                _box(builder, (cx, cy + cart_half[1], wall_z), (cart_half[0], 0.06, CART_WALL / 2), wall_cfg)
                cart = _box(
                    builder, (cx, cy, wall_z), (cart_half[0] - 0.08, cart_half[1] - 0.08, CART_WALL / 2), volume_cfg
                )
                volumes[cart] = (CART, index)
                # Wheels under the cart floor, for looks.
                for wx in (-1.0, 1.0):
                    for wy in (-1.0, 1.0):
                        _box(
                            builder,
                            (cx + wx * (cart_half[0] - 0.4), cy + wy * (cart_half[1] - 0.5), 0.15),
                            (0.12, 0.15, 0.15),
                            visual,
                            (0.1, 0.1, 0.1),
                        )

            # Funnel at the sorter entrance: suitcases keep their sideways position on a belt, so
            # centering them keeps undiverted suitcases away from the chute mouths.
            narrowing = SORTER_WIDTH / 2 - FUNNEL_HALF_WIDTH
            yaw = float(np.arctan2(narrowing, FUNNEL_LENGTH))
            guide_half = (0.5 * float(np.hypot(FUNNEL_LENGTH, narrowing)), 0.05, RAIL / 2)
            for side in (-1.0, 1.0):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(
                        (
                            collector_end + FUNNEL_LENGTH / 2,
                            y0 + side * (FUNNEL_HALF_WIDTH + narrowing / 2),
                            SORTER_TOP + RAIL / 2,
                        ),
                        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -side * yaw),
                    ),
                    hx=guide_half[0],
                    hy=guide_half[1],
                    hz=guide_half[2],
                    cfg=rail_cfg,
                    color=rail_color,
                )

            # Guard rails along the sorter, open only at the chute mouths of each side.
            for side, first in ((-1.0, 0), (1.0, 1)):
                start = collector_end if side < 0 else SORTER_START
                mouths = [DIVERTER_START + DIVERTER_SPACING * f + CHUTE_OFFSET for f in range(first, FLIGHTS, 2)]
                for mouth in [*mouths, SORTER_END + CHUTE_WIDTH / 2 + 0.1]:
                    end = mouth - CHUTE_WIDTH / 2 - 0.1
                    if end > start:
                        _box(
                            builder,
                            ((start + end) / 2, y0 + side * (SORTER_WIDTH / 2 + 0.05), SORTER_TOP + RAIL / 2),
                            ((end - start) / 2, 0.05, RAIL / 2),
                            rail_cfg,
                            rail_color,
                        )
                    start = mouth + CHUTE_WIDTH / 2 + 0.1

            # Sink for suitcases that missed their diverter, past the sorter end.
            sink = _box(builder, (SORTER_END + 3.0, y0, 1.0), (2.5, 4.0, 1.0), volume_cfg)
            volumes[sink] = (SINK, line)
        # The hall floor itself is a sink for suitcases that fell off a belt.
        floor_sink = _box(
            builder, (50.0, LINE_SPACING * (LINES - 1) / 2, 0.05), (120.0, LINE_SPACING * LINES, 0.15), volume_cfg
        )
        volumes[floor_sink] = (SINK, -1)

        builder.reserve_dynamic_capacity(bodies=CAPACITY, shapes=CAPACITY)
        self.model = builder.finalize()
        self.body_start = self.model._dynamic_capacity.body_start
        for mesh, velocity in self.belts.values():
            mesh.mesh.velocities.fill_(velocity)
        volume_kind = np.full(self.model.shape_count, -1, dtype=np.int32)
        volume_index = np.full(self.model.shape_count, -1, dtype=np.int32)
        for shape, (kind, index) in volumes.items():
            volume_kind[shape] = kind
            volume_index[shape] = index
        self.volume_kind = wp.array(volume_kind, dtype=wp.int32)
        self.volume_index = wp.array(volume_index, dtype=wp.int32)

        # Speculative contacts catch suitcases dropping from feeders and chutes early.
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="sticky",
            rigid_contact_max=CONTACT_MAX,
            speculative_contact_gap_max=SPECULATIVE_GAP,
        )
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
            tracking_volumes=list(volumes),
        )
        self.dyn = self.solver.dynamics.data
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        carts = LINES * FLIGHTS
        rng = np.random.default_rng(5)
        self.counters = wp.array(counters, dtype=wp.vec3)
        self.counter_body = wp.full(len(counters), -1, dtype=wp.int32)
        self.half_extents = wp.array(CASE_HALF_EXTENTS, dtype=wp.vec3)
        self.frame = wp.zeros(1, dtype=wp.int32)
        self.case_kind = wp.zeros(CAPACITY, dtype=wp.int32)
        self.case_flight = wp.zeros(CAPACITY, dtype=wp.int32)
        self.case_diverted = wp.zeros(CAPACITY, dtype=wp.int32)
        self.case_claimed = wp.zeros(CAPACITY, dtype=wp.int32)
        self.case_seen = wp.full(CAPACITY, -1, dtype=wp.int32)
        self.case_cart = wp.full(CAPACITY, -1, dtype=wp.int32)
        self.cart_count = wp.zeros(carts, dtype=wp.int32)
        self.cart_threshold = wp.array(rng.integers(CART_FILL_MIN, CART_FILL_MAX, size=carts), dtype=wp.int32)
        self.cart_departing = wp.zeros(carts, dtype=wp.int32)
        self.cart_departures = wp.zeros(1, dtype=wp.int32)
        self.cart_wall_start = wp.array(cart_wall_start, dtype=wp.int32)
        self.flight_colors = wp.array(FLIGHT_COLORS, dtype=wp.vec3)
        self.instance_xforms = wp.zeros(CAPACITY, dtype=wp.transform)
        self.instance_scales = wp.zeros(CAPACITY, dtype=wp.vec3)
        self.instance_colors = wp.zeros(CAPACITY, dtype=wp.vec3)
        self.instance_materials = wp.full(CAPACITY, wp.vec4(0.55, 0.0, 0.0, 0.0), dtype=wp.vec4)

        viewer.set_model(self.model)
        # One unit box serves every suitcase size through per-instance scales.
        viewer.log_geo("/suitcase", newton.GeoType.BOX, (1.0, 1.0, 1.0), 0.0, True, hidden=True)
        viewer.set_camera(wp.vec3(40.0, -12.0, 14.0), pitch=-28.0, yaw=70.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(
            _spawn_kernel,
            dim=self.counters.shape[0],
            inputs=[
                self.dyn,
                self.state_0.body_q,
                self.counters,
                self.counter_body,
                self.half_extents,
                self.frame,
                self.body_start,
                self.case_kind,
                self.case_flight,
                self.case_diverted,
                self.case_claimed,
                self.case_cart,
            ],
        )
        self.pipeline.collide(self.state_0, self.contacts, dt=self.frame_dt)
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
                self.volume_kind,
                self.volume_index,
                self.body_start,
                self.frame,
                self.case_flight,
                self.case_diverted,
                self.case_claimed,
                self.case_seen,
                self.case_cart,
                self.cart_count,
                self.cart_departing,
            ],
        )
        wp.launch(
            _pusher_kernel,
            dim=CAPACITY,
            inputs=[
                self.dyn,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.body_start,
                self.case_flight,
                self.case_diverted,
            ],
        )
        wp.launch(
            _cart_kernel,
            dim=self.cart_count.shape[0],
            inputs=[
                self.frame,
                self.cart_count,
                self.cart_threshold,
                self.cart_departing,
                self.cart_departures,
                self.cart_wall_start,
                self.flight_colors,
                self.model.shape_color,
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
        wp.launch(
            _case_instances_kernel,
            dim=CAPACITY,
            inputs=[
                self.dyn,
                self.state_0.body_q,
                self.body_start,
                self.case_kind,
                self.case_flight,
                self.flight_colors,
                self.half_extents,
            ],
            outputs=[self.instance_xforms, self.instance_scales, self.instance_colors],
        )
        self.viewer.log_instances(
            "/suitcases",
            "/suitcase",
            self.instance_xforms,
            self.instance_scales,
            self.instance_colors,
            self.instance_materials,
        )
        self.viewer.end_frame()

    def live_cases(self) -> np.ndarray:
        """Body indices of the suitcases currently in the hall."""
        state = self.solver.dynamics.bodies.data.state.numpy()[:CAPACITY]
        return self.body_start + np.flatnonzero((state == 1) | (state == 2))

    def test_post_step(self):
        """Every two seconds, check finite state and the capacity."""
        if round(self.sim_time / self.frame_dt) % 120 != 0:
            return
        live = self.live_cases()
        assert np.isfinite(self.state_0.body_q.numpy()[live]).all(), "nonfinite suitcase pose"
        assert np.isfinite(self.state_0.body_qd.numpy()[live]).all(), "nonfinite suitcase velocity"
        assert len(live) < CAPACITY, f"suitcases exceed the reserved capacity: {len(live)}"

    def sort_accuracy(self) -> tuple[int, float]:
        """Live suitcases inside a cart, and the fraction of them in the cart of their flight."""
        slots = self.live_cases() - self.body_start
        carts = self.case_cart.numpy()[slots]
        carted = carts >= 0
        if not carted.any():
            return 0, 1.0
        return int(carted.sum()), float((carts[carted] == self.case_flight.numpy()[slots][carted]).mean())

    def test_final(self):
        """Suitcases reach the carts of their flight, and full carts depart."""
        live = len(self.live_cases())
        carted, accuracy = self.sort_accuracy()
        departures = int(self.cart_departures.numpy()[0])
        print(
            f"PhoenX baggage hall: {live} suitcases, {carted} in carts ({accuracy:.1%} sorted correctly), "
            f"{departures} cart departures after {self.sim_time:.1f} s"
        )
        assert int(self.solver.dynamics.error_count.numpy()[0]) == 0
        if self.sim_time > 40.0:
            assert carted > 0, "no suitcase reached a cart"
            assert accuracy > 0.95, f"only {accuracy:.1%} of carted suitcases are in the cart of their flight"
        if self.sim_time > 300.0:
            assert departures > 0, "no cart departed"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # Path-traced rendering, like the Kapla tower, which also suits video recording.
        # Reserved shape rows count as viewer instances too.
        parser.set_defaults(viewer="optix", optix_max_instances=2 * CAPACITY + 8192)
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=4, help="Solver iterations per substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
