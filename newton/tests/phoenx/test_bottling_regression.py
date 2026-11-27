# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast regressions of the bottling plant's physical failure modes.

Each test reproduces one mode seen in long runs of the bottling plant on a
small scene with the plant's colliders, materials and physics settings (see
:mod:`newton.tests.phoenx.plant_scenes`). ``scripts/run_bottling_regression.py``
runs these together with the earlier regressions of the same modes.
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_infeed import SINGLE_FILE_PITCH
from newton.examples.phoenx._bottling_process import (
    _BELT_SPEED,
    _add_table,
    _arc,
    _packed_field,
    add_filling_transport,
)
from newton.tests.phoenx.plant_scenes import (
    PlantPhysics,
    add_belt,
    add_plant_bottle,
    belt,
    guide,
    kinetic_energy,
    strip,
    tilt,
)

_DECK = 0.95  # m, height of the plant's process decks and bottle lanes
_PITCH = 0.063  # m, centre spacing of a packed bed


def _bed(scene, x, half, y, count=None, clearance=0.0, seed=0, lean=0.0):
    """Fill an outline with a packed bed of plant bottles, leaning by up to ``lean`` [rad]."""
    rng = np.random.default_rng(seed)
    bodies = []
    for i, (bx, by) in enumerate(_packed_field((x, half), y, count, clearance)):
        axis = rng.normal(size=2)
        q = wp.quat_from_axis_angle(wp.vec3(*axis, 0.0) / float(np.linalg.norm(axis)), float(lean * rng.random()))
        xform = wp.transform((float(bx), float(by), _DECK + 0.001), q)
        bodies.append(add_plant_bottle(scene.builder, xform, f"Bed_{i:04d}"))
    return np.asarray(bodies)


def _scene():
    builder = newton.ModelBuilder()
    return SimpleNamespace(builder=builder, belts=[], metadata={}, bodies={}, bottle_bodies=[])


def _physics(scene, **kw):
    return PlantPhysics(scene.builder, [(mesh, velocity) for mesh, velocity, *_ in scene.belts], **kw)


def _end_stop(builder, x, y, half):
    """A static wall across the line at ``x`` that holds the stock back."""
    builder.add_shape_box(
        -1,
        xform=wp.transform((x + 0.01, y, _DECK + 0.1), wp.quat_identity()),
        hx=0.01,
        hy=half,
        hz=0.1,
        cfg=guide(builder),
        label="/Plant/Process/Stop/collision",
    )


@unittest.skipUnless(wp.is_cuda_available(), "The plant's physics runs CUDA graphs")
class TestBottlingRegression(unittest.TestCase):
    def test_loaded_pallet_crosses_belt_seam(self):
        """A loaded pallet crosses the Z1/Z2 seam of wrap line WL_U without pitching or dropping its load.

        In the plant the pallet jolted sideways, pitched 12 degrees and
        dropped its load at that seam while the infeed ran at 0.2 m/s. The
        Z1 chain strands end in a free end and Z2's start behind a gap; the
        pallet's 31 source colliders ride them with a 270 kg load.
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        strand_cfg = builder.ShapeConfig(density=0.0, mu=0.7, gap=0.01, margin=0.0)
        box = newton.Mesh.create_box(0.5, compute_inertia=False)
        belts, speed = [], 0.2
        # Zone centres along the flow and strand lengths [m]; strands 1.075 m apart, tops at z = 0.5.
        for center, length in ((0.0, 1.8596), (1.5022, 1.085)):
            for y in (-0.5375, 0.5375):
                mesh = newton.Mesh(box.vertices, box.indices, compute_inertia=False, enable_surface_velocity=True)
                scale = (length, 0.036, 0.04)
                builder.add_shape_mesh(
                    -1,
                    mesh=mesh,
                    xform=wp.transform((center, y, 0.48), wp.quat_identity()),
                    scale=scale,
                    cfg=strand_cfg,
                )
                # Surface velocities are in the unscaled mesh frame.
                velocity = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
                velocity[:, 0] = speed / length
                belts.append((mesh, velocity))
        cfg = builder.ShapeConfig(density=0.0, mu=0.7, gap=0.001, margin=0.0)
        pallet = builder.add_body(xform=wp.transform((0.2, 0.0, 0.4999), wp.quat_identity()), label="/Plant/Pallet")

        def board(x, y, z, hx, hy, hz):
            builder.add_shape_box(
                pallet, xform=wp.transform((x, y, z), wp.quat_identity()), hx=hx, hy=hy, hz=hz, cfg=cfg
            )

        # The source pallet's bottom and top deck boards, stringer boards, blocks and upper stringers.
        for x in (-0.4498, -0.3213, -0.1928, -0.0643, 0.0642, 0.1927, 0.3212, 0.4497):
            board(x, 0.0, 0.0101, 0.05, 0.6, 0.01)
            board(x, 0.0, 0.1517, 0.05, 0.6, 0.01)
        for y in (-0.5596, 0.0, 0.5595):
            board(0.0, y, 0.0312, 0.49945, 0.0398, 0.01)
            board(0.0, y, 0.1312, 0.49945, 0.0398, 0.01)
            for x in (-0.4496, 0.0, 0.4497):
                board(x, y, 0.0809, 0.04975, 0.0396, 0.04)
        mass = 25.0
        inertia = np.diag(
            (mass / 12 * (1.2**2 + 0.16**2), mass / 12 * (1.0**2 + 0.16**2), mass / 12 * (1.0**2 + 1.2**2))
        )
        builder.body_mass[pallet], builder.body_inv_mass[pallet] = mass, 1.0 / mass
        builder.body_inertia[pallet] = wp.mat33(inertia)
        builder.body_inv_inertia[pallet] = wp.mat33(np.linalg.inv(inertia))
        builder.body_com[pallet] = wp.vec3(0.0, 0.0, 0.08)
        load = builder.add_body(xform=wp.transform((0.2, 0.0, 0.4999 + 0.162 + 0.26), wp.quat_identity()))
        builder.add_shape_box(
            load,
            hx=0.5,
            hy=0.6,
            hz=0.26,
            cfg=builder.ShapeConfig(density=270.0 / (1.0 * 1.2 * 0.52), mu=0.5, gap=0.002),
        )
        sim = PlantPhysics(builder, belts)
        worst = {"vy": 0.0, "tilt": 0.0, "low": 1.0, "slip": 0.0}

        def sample(_, q, qd):
            worst["vy"] = max(worst["vy"], abs(float(qd[pallet, 1])))
            worst["tilt"] = max(worst["tilt"], float(tilt(q[[pallet, load]]).max()))
            worst["low"] = min(worst["low"], float(q[pallet, 2]))
            worst["slip"] = max(worst["slip"], float(np.linalg.norm(q[load, :2] - q[pallet, :2])))

        final, _ = sim.run(480, 1, sample)
        self.assertGreater(final[pallet, 0], 1.5, "the pallet did not cross onto Z2")
        # The seam kicked the pallet sideways at up to 1.68 m/s; carried, it drifts at about 0.01 m/s.
        self.assertLess(worst["vy"], 0.05, f"the seam kicked the pallet sideways at {worst['vy']:.3f} m/s")
        self.assertLess(worst["tilt"], math.radians(1.0), f"the pallet tilted {math.degrees(worst['tilt']):.2f} deg")
        self.assertGreater(worst["low"], 0.49, "the pallet dropped")
        self.assertLess(worst["slip"], 0.02, f"the load slid {1e3 * worst['slip']:.1f} mm on the pallet")

    def test_bottle_heels_hold_on_deck_seams(self):
        """A pressed bed with heels on the coplanar seams of a process deck stands on the deck.

        Every triangle under a tilted heel reported its witness on a shared
        internal edge; rejecting those seam normals dropped all of the heel's
        contacts, and pasteuriser bottles sank the 16 mm of their base ring.
        The deck is triangulated like the plant's, at the plant's coordinates
        and full width, so long thin triangles meet on seams under the bed.
        That witness geometry is rare; the collision regressions in
        ``test_convex_refinement_regressions`` pin the recorded cases, and
        this bed checks the plant's physics keeps every heel on the deck.
        """
        scene = _scene()
        builder = scene.builder
        y = 38.4
        x = np.round(np.arange(4.4, 4.81, 0.1), 6)
        half = np.full_like(x, 1.15)
        speed = 0.0383  # m/s, the pasteuriser bed
        _add_table(
            scene,
            "Pasteuriser",
            (x, half),
            (0.0, y, _DECK),
            lambda xs: np.full_like(xs, speed),
            motor_off=False,
            deck_cfg=belt(builder),
            guard_cfg=guide(builder),
        )
        _end_stop(builder, x[-1], y, 1.2)
        bed = _bed(scene, x, half, y, lean=0.003)
        sim = _physics(scene)
        low = []
        sim.run(360, 30, lambda _, q, qd: low.append(float(q[bed, 2].min()) - _DECK))
        # A heel that loses the deck drops 16 mm onto the body cylinder.
        self.assertGreater(min(low), -0.002, f"a bottle sank {-1e3 * min(low):.1f} mm into the deck")

    def test_pressed_bottles_do_not_climb_at_neck(self):
        """A packed field pressed by the accumulation table into its neck comes to rest without climbing.

        The table belt runs under the field, which stands against a closed
        outlet, so the bottles at the neck's corners are pressed from behind
        and from the side. With side contacts collapsed to single wandering
        points, pressed bottles gained rocking energy while the belt ran and
        climbed out over their neighbours.
        """
        scene = _scene()
        builder = scene.builder
        y = 38.4
        outline = (np.array([0.0, 1.0, 1.05, 1.6]), np.array([0.35, 0.35, 0.15, 0.15]))
        _add_table(
            scene,
            "AccumulationTable",
            outline,
            (0.0, y, _DECK),
            lambda xs: np.full_like(xs, 0.1),
            motor_off=False,
            deck_cfg=belt(builder),
            guard_cfg=guide(builder),
        )
        _end_stop(builder, outline[0][-1], y, 0.2)
        bed = _bed(scene, *outline, y, count=260)
        sim = _physics(scene, sleeping=False)
        high, energy, leaning = [], [], []

        def sample(frame, q, qd):
            high.append(float(q[bed, 2].max()) - _DECK)
            leaning.append(float(tilt(q[bed]).max()))
            if frame > 240:
                energy.append(kinetic_energy(qd[bed] * [0, 0, 0, 1, 1, 0]))

        sim.run(480, 6, sample)
        self.assertLess(max(high), 0.002, f"a bottle climbed {1e3 * max(high):.1f} mm")
        self.assertLess(max(leaning), math.radians(5.0))
        # A field at rest keeps about 0.1 uJ of rocking energy; on wandering point contacts it kept 58 mJ.
        rocking = float(np.mean(energy))
        self.assertLess(rocking, 1.0e-5, f"the pressed field keeps {1e3 * rocking:.3f} mJ of rocking energy")

    def test_pasteuriser_infeed_spin_is_bounded(self):
        """A single file jams into the blocked pasteuriser infeed without spinning faster than it can roll.

        The driven guides of the single file run past the stalled bottles of
        the jam. A bottle rolling on a guide spins at most at the belt speed
        over its radius, 16.5 rad/s; transients may exceed that briefly.
        Bottles whose collider pieces met on planes at full radius touched
        with vertical normals and spun at 40-120 rad/s until they left the
        line.
        """
        scene = _scene()
        builder = scene.builder
        # The plant's low turn into the infeed chute, and its funnel at the
        # source's flare, cut short to a bed of about 150 bottles.
        y = 38.4
        turn = _arc((1.96, y + 0.16), 0.16, math.pi, 1.5 * math.pi, 1)
        xy = np.vstack(((1.8, y + 1.6), turn, (3.0, y)))
        path = np.column_stack((xy, np.full(len(xy), _DECK)))
        add_filling_transport(scene, path, motor_off=False)
        table = (np.array([3.0, 3.3, 3.65, 4.0]), np.array([0.12, 0.12, 0.45, 0.45]))
        chute = min(2.0 * 425.0 / 60.0 * 0.5 * math.sqrt(3.0) * _PITCH**2 / 0.24, _BELT_SPEED)
        _add_table(
            scene,
            "Pasteuriser",
            table,
            (0.0, y, _DECK),
            lambda xs: np.full_like(xs, chute),
            inflow=0.04,
            motor_off=False,
            deck_cfg=belt(builder),
            guard_cfg=guide(builder),
        )
        _end_stop(builder, table[0][-1], y, 0.5)
        bed = _bed(scene, table[0][1:], table[1][1:], y)
        # The single file stands on the route at the admission pitch, moving with its belt.
        length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        stations = np.arange(0.1, length[-1] - 0.05, SINGLE_FILE_PITCH)
        centers = np.column_stack([np.interp(stations, length, column) for column in xy.T])
        heading = np.gradient(xy, length, axis=0)
        heading = np.column_stack([np.interp(stations, length, column) for column in heading.T])
        heading /= np.linalg.norm(heading, axis=1, keepdims=True)
        file = [
            add_plant_bottle(
                builder,
                wp.transform((float(cx), float(cy), _DECK + 0.001), wp.quat_identity()),
                f"File_{i:03d}",
                velocity=(*(_BELT_SPEED * h), 0.0),
            )
            for i, ((cx, cy), h) in enumerate(zip(centers, heading, strict=True))
        ]
        bottles = np.r_[bed, file]
        sim = _physics(scene)
        spin, low = [], []

        def sample(_, q, qd):
            spin.append(float(np.linalg.norm(qd[bottles, 3:], axis=1).max()))
            low.append(float(q[bottles, 2].min()) - _DECK)

        sim.run(480, 2, sample)
        self.assertLess(max(spin), 1.25 * _BELT_SPEED / 0.0305, f"a bottle spun at {max(spin):.1f} rad/s")
        self.assertGreater(min(low), -0.002)

    def test_cased_bottles_ride_a_conveyor(self):
        """Bottles stay in a case carried across a belt transfer into a standing stop.

        The case's floor is its two major flaps with the 1.5 mm seam between
        them, as in the plant. The case crosses from one belt to the next and
        runs into a stop at the case conveyors' 0.375 m/s.
        """
        builder = newton.ModelBuilder()
        belts = []
        speed, top = 0.375, 0.7
        for i, (begin, end) in enumerate(((-1.0, 0.0), (0.0, 1.0))):
            x = np.array([begin, end])
            vertices, indices = strip(
                np.column_stack((x, np.full(2, -0.2), np.full(2, top))),
                np.column_stack((x, np.full(2, 0.2), np.full(2, top))),
            )
            add_belt(builder, belts, vertices, indices, (speed, 0.0, 0.0), f"/Plant/Line/CaseBelt_{i}")
        rail = guide(builder)
        for side in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform((0.0, side * 0.125, top + 0.08), wp.quat_identity()),
                hx=1.0,
                hy=0.005,
                hz=0.02,
                cfg=rail,
            )
        builder.add_shape_box(
            -1, xform=wp.transform((0.81, 0.0, top + 0.1), wp.quat_identity()), hx=0.01, hy=0.2, hz=0.1, cfg=rail
        )
        length, width, height, board, seam = 0.286, 0.216, 0.261, 0.0032, 0.0015
        area = 2.0 * (length * width + (length + width) * height)
        cfg = builder.ShapeConfig(density=0.42 / (area * board), mu=0.4, mu_static=0.45, gap=0.002)
        case = builder.add_body(xform=wp.transform((-0.7, 0.0, top + 0.0005), wp.quat_identity()))
        flap = 0.25 * width - 0.25 * seam
        boxes = [
            ((0.0, side * (0.5 * seam + flap), 0.5 * board), (0.5 * length, flap, 0.5 * board)) for side in (-1, 1)
        ]
        for axis, extent in ((0, length), (1, width)):
            for side in (-1, 1):
                center, half = [0.0, 0.0, 0.5 * height], [0.5 * length, 0.5 * width, 0.5 * height]
                center[axis] = side * 0.5 * (extent - board)
                half[axis] = 0.5 * board
                boxes.append((tuple(center), tuple(half)))
        for center, half in boxes:
            builder.add_shape_box(
                case, xform=wp.transform(center, wp.quat_identity()), hx=half[0], hy=half[1], hz=half[2], cfg=cfg
            )
        cells = [(cx, cy) for cx in (-1.5, -0.5, 0.5, 1.5) for cy in (-1.0, 0.0, 1.0)]
        bottles = np.asarray(
            [
                add_plant_bottle(
                    builder,
                    wp.transform((-0.7 + 0.0685 * cx, 0.0685 * cy, top + board + 0.001), wp.quat_identity()),
                    f"Cased_{i:02d}",
                )
                for i, (cx, cy) in enumerate(cells)
            ]
        )
        sim = PlantPhysics(builder, belts)
        worst = {"low": 0.0, "out": -1.0, "pitch": 0.0}

        def sample(_, q, qd):
            pose = wp.transform(q[case, :3], q[case, 3:])
            local = np.array([wp.transform_point(wp.transform_inverse(pose), wp.vec3(*p)) for p in q[bottles, :3]])
            worst["low"] = min(worst["low"], float(local[:, 2].min()) - board)
            worst["out"] = max(worst["out"], float((np.abs(local[:, :2]) - [0.5 * length, 0.5 * width]).max()))
            worst["pitch"] = max(worst["pitch"], float(tilt(q[case : case + 1])[0]))

        final, _ = sim.run(420, 3, sample)
        self.assertGreater(final[case, 0], 0.6, "the case did not reach the stop")
        self.assertGreater(worst["low"], -0.002, "a bottle sank through the case floor")
        self.assertLess(worst["out"], 0.0, "a bottle left the case")
        self.assertLess(worst["pitch"], math.radians(5.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
