# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Kapla City, a skyline assembled from loose wooden planks under gravity.

Run with python -m newton.examples phoenx_kapla_city. The stepped spires,
observation towers, gateway and apartment blocks should retain their silhouettes.
Every rigid body models one 117 x 23.4 x 7.8 mm plank weighing 11 g.
Friction (0.5) and zero restitution approximate dry wood in a resting stack.
"""

import itertools
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX

# Layout coordinates use the original drawing scale. Convert every position and
# extent at the model boundary so gravity, inertia and contacts use SI units.
METRES_PER_UNIT = 0.117 / 0.30
PLANK_MASS = 0.011  # kg, nominal natural pine plank
LENGTH, WIDTH, THICK = 0.30, 0.06, 0.02
FLOOR = WIDTH + THICK
FLAT = (LENGTH, WIDTH, THICK)
EDGE = (LENGTH, THICK, WIDTH)
WOOD = ((0.86, 0.79, 0.66), (0.91, 0.85, 0.74), (0.82, 0.76, 0.65))


# Ground-plane calibration of the 2264 x 1028 oblique overview. Pixel anchors
# locate the feet of buildings, so their heights do not displace their lots.
VIEW_PITCH = math.radians(31)
VIEW_DISTANCE = 62.0
FOCAL_PIXELS = 116 * VIEW_DISTANCE


def ground(u, v):
    rise = (514 - v) / FOCAL_PIXELS
    y = VIEW_DISTANCE * rise / (math.sin(VIEW_PITCH) - rise * math.cos(VIEW_PITCH))
    return ((u - 1132) * (VIEW_DISTANCE + y * math.cos(VIEW_PITCH)) / FOCAL_PIXELS, y)


class City:
    def __init__(self, builder):
        self.builder = builder
        self.footprints = []
        self.lot = (0, 0, 0)
        volume = LENGTH * WIDTH * THICK * METRES_PER_UNIT**3
        self.material = newton.ModelBuilder.ShapeConfig(density=PLANK_MASS / volume, mu=0.5, restitution=0, gap=0.0006)

    def place(self, u, v, degrees=0):
        self.lot = (*ground(u, v), math.radians(degrees))

    def plank(self, x, y, base, dims=FLAT, yaw=0.0, roll=0.0):
        dx, dy, dz = dims
        cx, cy, a = self.lot
        c, s = math.cos(a), math.sin(a)
        x, y = cx + c * x - s * y, cy + s * x + c * y
        yaw += a
        height = dy * abs(math.sin(roll)) + dz * abs(math.cos(roll))
        if base + (dz - height) / 2 < WIDTH:
            plan_width = dy * abs(math.cos(roll)) + dz * abs(math.sin(roll))
            self.footprints.append((x, y, dx, plan_width, yaw))
        pose = wp.transform(
            wp.vec3(x, y, base + dz / 2) * METRES_PER_UNIT,
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), roll),
        )
        body = self.builder.add_body(xform=pose)
        self.builder.add_shape_box(
            body,
            hx=dx * METRES_PER_UNIT / 2,
            hy=dy * METRES_PER_UNIT / 2,
            hz=dz * METRES_PER_UNIT / 2,
            cfg=self.material,
            color=WOOD[body % 3],
        )

    def block(self, x, y, floors, bays=1, base=0.0, pitch=LENGTH):
        for story in range(floors):
            z = base + story * FLOOR
            for bay in range(bays):
                cx = x + (bay - (bays - 1) / 2) * pitch
                for offset in (-0.12, 0, 0.12):
                    self.plank(cx + offset, y, z, EDGE, math.pi / 2)
                for row in range(-2, 3):
                    self.plank(cx, y + row * WIDTH, z + WIDTH)

    def spire(self, x, y, layers, twist, lattice=False):
        # Successive square pinwheels cross at their corners.
        for level in range(layers):
            if lattice and level >= layers // 2:
                angle = (level % 2) * math.pi / 2
                for offset in (-0.11, 0.11):
                    self.plank(x - offset * math.sin(angle), y + offset * math.cos(angle), level * WIDTH, EDGE, angle)
                continue
            for side in range(4):
                a = level * twist + side * math.pi / 2
                c, s = math.cos(a), math.sin(a)
                self.plank(x + 0.15 * s - 0.01 * c, y - 0.15 * c - 0.01 * s, level * WIDTH, EDGE, a)

    def observatory(self, x, y, floors):
        self.block(x, y, floors)
        direction = -1 if floors == 19 else 1
        self.block(x, y + direction * LENGTH, max(1, floors - 4))
        if floors != 19:
            self.block(x, y + direction * 2 * LENGTH, max(2, (floors - 4) // 2))
        z = floors * FLOOR
        # Four offset planks interlock around the mast. Turning each course
        # distributes its load across the neighbouring sides of the lower ring.
        radii = (
            0.135,
            0.15,
            0.17,
            0.19,
            0.21,
            0.23,
            0.25,
            0.27,
            0.29,
            0.31,
            0.315,
            0.30,
            0.28,
            0.26,
            0.24,
            0.22,
            0.20,
            0.18,
            0.16,
            0.14,
        )
        for level, radius in enumerate(radii):
            shift = max(0.0, 0.20 - radius)
            for side in range(4):
                angle = (level % 2) * math.pi / 4 + side * math.pi / 2
                c, s = math.cos(angle), math.sin(angle)
                self.plank(x + radius * c - shift * s, y + radius * s + shift * c, z, FLAT, angle + math.pi / 2)
            z += THICK
        for row in range(-2, 3):
            self.plank(x + row * WIDTH, y, z, FLAT, math.pi / 2)
        z += THICK
        for rows in (3, 1):
            for row in range(rows):
                self.plank(x + (row - (rows - 1) / 2) * WIDTH, y, z, FLAT, math.pi / 2)
            z += THICK
        self.plank(x, y, z, (WIDTH, THICK, LENGTH))

    def house(self, x, y, floors, bays=1):
        lower = max(0, floors - 4)
        for offset in (-LENGTH / 2, LENGTH / 2):
            cy = y + offset
            self.block(x, cy, lower, bays)
            for bay in range(bays):
                cx = x + (bay - (bays - 1) / 2) * LENGTH
                for dx in (-0.12, 0, 0.12):
                    for dy in (-0.12, 0.12):
                        self.plank(cx + dx, cy + dy, lower * FLOOR, (THICK, WIDTH, LENGTH))
                    self.plank(cx + dx, cy, lower * FLOOR + LENGTH, FLAT, math.pi / 2)
                for row in range(-2, 3):
                    self.plank(cx, cy + row * WIDTH, lower * FLOOR + LENGTH + THICK)
        deck = lower * FLOOR + LENGTH + 2 * THICK
        # Five central courses carry the ridge. Each full-length roof board
        # bears on the top outside edge of the ridge and on the eaves.
        angle = math.asin(0.4)
        for bay in range(bays):
            cx = x + (bay - (bays - 1) / 2) * LENGTH
            for side in (-1, 1):
                for layer in range(5):
                    self.plank(cx, y + side * WIDTH / 2, deck + layer * THICK)
                eave = WIDTH + 5 * THICK / math.tan(angle)
                mid = eave - LENGTH * math.cos(angle) / 2 + THICK * math.sin(angle) / 2
                center_z = deck + (LENGTH * math.sin(angle) + THICK * math.cos(angle)) / 2
                for row in range(-2, 3):
                    self.plank(
                        cx + row * WIDTH,
                        y + side * mid,
                        center_z - THICK / 2,
                        (WIDTH, LENGTH, THICK),
                        roll=-side * angle,
                    )

    def frame_tower(self, x, y, stories):
        for story in range(stories):
            z = story * (LENGTH + 2 * THICK)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    self.plank(x + sx * 0.115, y + sy * 0.115, z, (WIDTH, THICK, LENGTH))
                self.plank(x + sx * 0.115, y, z + LENGTH, FLAT, math.pi / 2)
            for row in range(-2, 3):
                self.plank(x, y + row * WIDTH, z + LENGTH + THICK)

    def palm(self, x, y):
        self.plank(x, y, 0, (WIDTH, THICK, LENGTH))
        for level in range(3):
            self.plank(x, y, LENGTH + level * THICK, FLAT, level * math.pi / 3)

    def round_tower(self, x, y, floors):
        # The inner core carries the roof. Each outer ring bridges the
        # joints of the course below, with its bearing walls inset from the rim.
        self.block(x, y, floors)
        for floor in range(floors):
            base = floor * FLOOR
            for side in range(8):
                a = side * math.pi / 4
                self.plank(x + 0.378 * math.cos(a), y + 0.378 * math.sin(a), base, EDGE, a + math.pi / 2)
                tangent = a + math.pi / 8
                self.plank(
                    x + 0.40 * math.cos(tangent),
                    y + 0.40 * math.sin(tangent),
                    base + WIDTH,
                    FLAT,
                    tangent + math.pi / 2,
                )
        roof = floors * FLOOR
        for side in range(4):
            angle = side * math.pi / 2
            c, s = math.cos(angle), math.sin(angle)
            for row in range(-2, 3):
                offset = row * WIDTH + 0.045
                self.plank(x + 0.255 * c - offset * s, y + 0.255 * s + offset * c, roof, FLAT, angle)
        for row in range(-6, 7):
            centers = (-0.15, 0.15) if abs(row) <= 4 else (0,)
            for offset in centers:
                self.plank(x + offset, y + row * WIDTH, roof + THICK)

    def paving(self, x, y, dims, yaw=0.0):
        # Leave openings where streets meet buildings, walls or other streets.
        dx, dy, _ = dims
        for bx, by, sx, sy, angle in self.footprints:
            if abs(x - bx) > 2 * LENGTH or abs(y - by) > 2 * LENGTH:
                continue
            for axis in (yaw, yaw + math.pi / 2, angle, angle + math.pi / 2):
                separation = abs((x - bx) * math.cos(axis) + (y - by) * math.sin(axis))
                width = dx * abs(math.cos(yaw - axis)) + dy * abs(math.sin(yaw - axis))
                other = sx * abs(math.cos(angle - axis)) + sy * abs(math.sin(angle - axis))
                if separation >= (width + other) / 2 - 1e-7:
                    break
            else:
                return
        self.plank(x, y, 0, dims, yaw)

    def street(self, points, curb=False):
        # Catmull-Rom centerline sampled in the ground plane, then spaced by
        # physical arc length. Straight-sided courtyard walls use linear runs.
        pts = np.array([ground(*pt) for pt in points])
        samples = []
        for i, (a, b) in enumerate(itertools.pairwise(pts)):
            before = pts[max(0, i - 1)]
            after = pts[min(len(pts) - 1, i + 2)]
            for t in np.arange(48) / 48:
                if curb:
                    samples.append(a + t * (b - a))
                else:
                    samples.append(
                        0.5
                        * (
                            2 * a
                            + (-before + b) * t
                            + (2 * before - 5 * a + 4 * b - after) * t * t
                            + (-before + 3 * a - 3 * b + after) * t**3
                        )
                    )
        samples = np.array([*samples, pts[-1]])
        ds = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        arc = np.r_[0, np.cumsum(ds)]
        pitch = LENGTH + 0.008 if curb else 0.102
        for d in np.arange(pitch / 2, arc[-1], pitch):
            i = min(np.searchsorted(arc, d) - 1, len(ds) - 1)
            tangent = (samples[i + 1] - samples[i]) / ds[i]
            pos = samples[i] + (d - arc[i]) * tangent
            angle = math.atan2(tangent[1], tangent[0])
            self.paving(*pos, EDGE if curb else FLAT, angle if curb else angle + math.pi / 2)

    def gateway(self):
        # Successive corbels from the two piers support the high lintel.
        for x in (-0.56, 0.56):
            self.block(x, 0, 13, 2)
            for level in range(5):
                shift = -math.copysign((level + 1) * 0.031, x)
                self.block(x + shift, 0, 1, 2, (13 + level) * FLOOR)
        for bay in range(-2, 3):
            for row in range(-2, 3):
                self.plank(bay * LENGTH, row * WIDTH, 18 * FLOOR)
        self.block(0, 0, 3, 5, 18 * FLOOR + THICK)
        # The lower, perpendicular portal projects towards the viewer from
        # the right pier and can be seen through the taller opening.
        for y in (-0.84, -0.305):
            self.block(0.52, y, 9)
        for row in range(-2, 3):
            self.plank(0.52 + row * WIDTH, -0.5725, 9 * FLOOR, FLAT, math.pi / 2)
        self.block(0.52, -0.5725, 3, base=9 * FLOOR + THICK)

    def build(self):
        # The stepped foreground fan. Each entry records a visible footing
        # and the number of interlocking courses above it.
        for u, v, layers in (
            (389, 740, 4),
            (449, 681, 7),
            (442, 611, 5),
            (504, 657, 9),
            (526, 792, 5),
            (537, 715, 14),
            (567, 687, 18),
            (586, 650, 23),
            (575, 599, 13),
            (607, 735, 12),
            (614, 690, 29),
            (633, 653, 44),
            (664, 720, 24),
            (682, 674, 31),
            (704, 638, 24),
            (731, 693, 21),
            (761, 632, 18),
            (783, 596, 12),
            (810, 658, 12),
            (852, 644, 8),
            (898, 628, 5),
            (931, 727, 4),
            (856, 751, 7),
            (749, 817, 4),
            (718, 771, 7),
        ):
            self.place(u, v, 12)
            self.spire(0, 0, layers, math.pi / 4, lattice=True)
        # The central needle has open cross-laid courses above its shoulders.
        self.place(633, 653, 12)
        for i in range(6):
            for side in (-1, 1):
                self.plank(side * 0.105, 0, 44 * WIDTH + i * FLOOR, EDGE, math.pi / 2)
            for side in (-1, 1):
                self.plank(0, side * 0.105, 44 * WIDTH + i * FLOOR + WIDTH, FLAT)

        # Five domes, all on the same walled island. Narrow upper shafts rise
        # from stepped windowed bases and a chain of lower gallery bays.
        for u, v, h in ((837, 314, 14), (894, 337, 19), (968, 343, 12), (947, 410, 13), (1043, 354, 5)):
            self.place(u, v, 30)
            self.observatory(0, 0, h)
        self.place(1379, 231, -42)
        self.gateway()

        # The right-hand business district includes two open frame towers,
        # spiral columns, narrow apartment shafts and pitched crowns.
        for u, v, h, bays in (
            (1619, 375, 16, 2),
            (1464, 431, 8, 1),
            (1525, 419, 8, 1),
            (1693, 353, 8, 1),
            (1810, 380, 9, 1),
        ):
            self.place(u, v, 19)
            self.block(0, 0, h, bays)
        for u, v, stories in ((1666, 429, 2), (1749, 465, 2)):
            self.place(u, v, 18)
            self.frame_tower(0, 0, stories)
        for u, v, layers in ((1542, 352, 13), (1595, 463, 12)):
            self.place(u, v, 13)
            self.spire(0, 0, layers, 0.20)
        for u, v, n in ((1740, 339, 5), (1687, 494, 4)):
            self.place(u, v, 65)
            self.house(0, 0, n)

        # The long slab is seen almost end-on in the overview.
        self.place(2073, 463, 72)
        self.block(0, 0, 16, 4)
        for x in (-0.77, 0.77):
            self.round_tower(x, -0.60, 15)

        # West end, central market, then the row along the palm court.
        for u, v, levels, bays, heading in (
            (183, 395, 6, 2, 32),
            (220, 508, 3, 1, -36),
            (443, 501, 12, 2, 40),
            (506, 352, 3, 1, 26),
            (604, 380, 4, 1, -24),
            (944, 533, 3, 1, -30),
            (987, 494, 4, 1, -30),
            (1072, 502, 3, 1, 62),
            (1062, 582, 4, 1, -22),
            (1128, 530, 3, 1, -22),
            (1237, 475, 3, 1, 63),
            (1409, 694, 3, 1, 16),
            (1571, 684, 6, 2, 17),
            (1692, 668, 4, 1, 18),
        ):
            self.place(u, v, heading)
            self.house(0, 0, levels, bays)
        for u, v, heading, bays in (
            (475, 306, 22, 3),
            (213, 447, -32, 2),
            (328, 475, 32, 2),
            (491, 411, 32, 2),
            (1107, 620, -20, 1),
            (1474, 660, 9, 2),
            (1628, 648, 22, 2),
        ):
            self.place(u, v, heading)
            for bay in range(bays):
                self.frame_tower((bay - (bays - 1) / 2) * LENGTH, 0, 1)

        for u, v in (
            (199, 492),
            (258, 534),
            (324, 510),
            (379, 445),
            (458, 329),
            (548, 434),
            (1099, 425),
            (1190, 454),
            (1116, 364),
            (1430, 524),
            (1461, 481),
            (1507, 540),
            (1533, 513),
            (1616, 525),
            (1363, 602),
            (1462, 607),
            (1544, 598),
            (1623, 578),
            (2009, 393),
            (1948, 523),
        ):
            self.place(u, v, 15)
            self.palm(0, 0)

        # Ramp, elevated crossing, paired portal towers and descending ramp.
        x0, y0 = ground(1080, 700)
        x1, y1 = ground(1338, 467)
        angle = math.atan2(y1 - y0, x1 - x0)
        self.lot = (x0, y0, angle)
        spans = round(math.hypot(x1 - x0, y1 - y0) / LENGTH)
        ramp_spans = 4
        angle_ramp = math.asin(WIDTH / (LENGTH - THICK))
        pitch = LENGTH * math.cos(angle_ramp) + THICK * math.sin(angle_ramp) + 0.002
        bearing = WIDTH / math.tan(angle_ramp)
        center_spans = spans - 2 * ramp_spans
        bridge_length = 2 * ramp_spans * pitch + center_spans * LENGTH
        for direction in (1, -1):
            start = 0 if direction == 1 else bridge_length
            for i in range(ramp_spans):
                low = i * pitch
                cx = low + (LENGTH * math.cos(angle_ramp) - THICK * math.sin(angle_ramp)) / 2
                cz = i * WIDTH + (LENGTH * math.sin(angle_ramp) + THICK * math.cos(angle_ramp)) / 2
                for row in range(-2, 3):
                    self.plank(
                        start + direction * cx,
                        row * WIDTH,
                        cz - THICK / 2,
                        (WIDTH, LENGTH, THICK),
                        math.pi / 2,
                        -direction * angle_ramp,
                    )
                for course in range(i + 1):
                    for offset in (THICK / 2, 1.5 * THICK):
                        self.plank(start + direction * (low + bearing + offset), 0, course * WIDTH, EDGE, math.pi / 2)
        for bay in range(center_spans):
            x = ramp_spans * pitch + (bay + 0.5) * LENGTH
            if bay + 1 < center_spans:
                for course in range(ramp_spans):
                    self.plank(x + LENGTH / 2, 0, course * WIDTH, EDGE, math.pi / 2)
            for row in range(-2, 3):
                self.plank(x, row * WIDTH, ramp_spans * WIDTH)
        for sign in (-1, 1):
            tx, ty = spans * LENGTH * 0.52, sign * 0.307
            self.frame_tower(tx, ty, 2)
            z = 2 * (LENGTH + 2 * THICK)
            for rows in (5, 3, 1):
                for row in range(rows):
                    self.plank(tx, ty + (row - (rows - 1) / 2) * WIDTH, z)
                z += THICK

        self.lot = (0, 0, 0)
        self.street(
            [
                (39, 481),
                (197, 401),
                (415, 319),
                (623, 264),
                (835, 235),
                (1060, 233),
                (1180, 258),
                (1294, 329),
                (1366, 415),
                (1338, 467),
            ]
        )
        self.street(
            [
                (39, 481),
                (182, 427),
                (318, 451),
                (485, 483),
                (644, 451),
                (761, 449),
                (864, 474),
                (977, 535),
                (1127, 573),
                (1253, 590),
                (1371, 632),
                (1530, 649),
                (1704, 616),
                (1840, 520),
                (1917, 451),
                (1944, 389),
            ]
        )
        self.street(
            [
                (469, 510),
                (394, 552),
                (313, 635),
                (277, 728),
                (310, 835),
                (405, 900),
                (533, 919),
                (700, 896),
                (872, 846),
                (1007, 767),
                (1080, 700),
            ]
        )
        self.street([(1185, 262), (1243, 272), (1315, 282), (1384, 268), (1410, 247)])
        self.street([(1917, 451), (1973, 454), (2012, 476)])
        for path in (
            [(728, 256), (905, 445), (1287, 351), (1197, 275)],
            [(1891, 526), (2124, 591), (2208, 410)],
            [(1308, 533), (1391, 442), (1536, 481), (1748, 536), (1687, 601)],
            [(987, 537), (983, 584), (1056, 616), (1129, 602), (1195, 538)],
            [(309, 389), (433, 430), (534, 398)],
            [(107, 465), (220, 519), (336, 475)],
            [(1111, 366), (1228, 310), (1195, 275)],
        ):
            self.street(path, curb=True)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.dt = 1 / 120
        self.time = 0.0
        builder = newton.ModelBuilder()
        builder.add_ground_plane(color=(0.34, 0.36, 0.39))
        City(builder).build()
        self.model = builder.finalize(skip_shape_contact_pairs=True)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching="sticky",
            shape_pairs_max=12 * self.model.body_count,
            rigid_contact_max=32 * self.model.body_count,
        )
        # Use eight substeps at 120 Hz for the tall plank stacks.
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            step_layout="single_world",
            substeps=8,
            solver_iterations=10,
            sleeping_velocity_threshold=0.025,
            sleeping_frames_required=45,
        )
        self.contacts = self.pipeline.contacts()
        self.current, self.next = self.model.state(), self.model.state()
        self.initial = self.current.body_q.numpy()[:, :3].copy()
        viewer.set_model(self.model)
        if hasattr(viewer, "camera"):
            viewer.camera.fov = math.degrees(2 * math.atan(637 / FOCAL_PIXELS))
        viewer.set_camera(
            wp.vec3(0, -math.cos(VIEW_PITCH), math.sin(VIEW_PITCH)) * VIEW_DISTANCE * METRES_PER_UNIT,
            pitch=-math.degrees(VIEW_PITCH),
            yaw=90,
        )

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.current.clear_forces()
        self.viewer.apply_forces(self.current)
        self.pipeline.collide(self.current, self.contacts, dt=self.dt)
        self.solver.step(self.current, self.next, None, self.contacts, self.dt)
        wp.copy(self.current.body_q, self.next.body_q)
        wp.copy(self.current.body_qd, self.next.body_qd)

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.time += self.dt

    def render(self):
        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.current)
        self.viewer.end_frame()

    def test_final(self):
        # Audit the finalized rigid bodies, including planks stood on edge or end.
        model = self.model
        half_extents = model.shape_scale.numpy()[model.shape_body.numpy() >= 0]
        lengths = 2 * half_extents
        np.testing.assert_allclose(np.sort(lengths, axis=1), np.tile([0.0078, 0.0234, 0.117], (len(lengths), 1)))
        np.testing.assert_allclose(model.body_mass.numpy(), PLANK_MASS, rtol=1e-6)
        inertia = PLANK_MASS / 12 * (np.sum(lengths**2, axis=1, keepdims=True) - lengths**2)
        np.testing.assert_allclose(np.diagonal(model.body_inertia.numpy(), axis1=1, axis2=2), inertia, rtol=2e-5)
        positions = self.current.body_q.numpy()[:, :3]
        drift = np.linalg.norm(positions - self.initial, axis=1)
        worst = np.argmax(drift)
        assert drift[worst] < THICK * METRES_PER_UNIT, (self.initial[worst], drift[worst])


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
