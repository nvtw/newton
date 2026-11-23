# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the example's USD adapter without requiring the private plant assets."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_control import _Belt, _belt_speeds, _seal_cases, _target_coordinates
from newton.examples.phoenx._bottling_fleet import FleetController, _Transfer
from newton.examples.phoenx._bottling_infeed import BottleSupply, InfeedController, StandingQueue
from newton.examples.phoenx._bottling_io import (
    _CASE_PHOTOEYE_BATCH_SIZE,
    _PHOTOEYE_BATCH_SIZE,
    _case_drives,
    _case_photoeyes,
    _photoeyes,
    _valves,
)
from newton.examples.phoenx._bottling_process import (
    PASTEURISER_CAPACITY,
    _arc,
    add_filling_bottles,
    add_filling_transport,
)
from newton.examples.phoenx._bottling_scene import PlantScene, load_plant
from newton.examples.phoenx._bottling_transport import _carton_support_quadrants, _placement_motions
from newton.examples.phoenx._bottling_wrap import (
    WrapperController,
    build_wrapper,
    correct_wrapper_frame,
    mesh_components,
)
from newton.examples.phoenx.example_phoenx_bottling_plant import (
    Example,
    _fleet_direct_contact_mask,
    _local_fixture_drives,
)
from newton.tests.unittest_utils import requires
from newton.viewer import ViewerNull


@requires("pxr.Usd")
class TestBottlingPlant(unittest.TestCase):
    @staticmethod
    def _bottle_line(path, layout=None):
        """Build source-sized glass bottles on the filling transport and capture one 120 Hz step on CUDA."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        glass = builder.ShapeConfig(density=0.0, mu=0.16, mu_static=0.2, gap=0.002)
        # The source bottle's mass properties; its centre of mass sits low in the thick base.
        source = builder.add_body(
            xform=wp.transform((0.0, -3.0, 0.0), wp.quat_identity()),
            mass=0.570,
            com=wp.vec3(0.0, 0.0, 0.0834),
            inertia=wp.mat33(np.diag((0.00184, 0.00184, 0.000261))),
            lock_inertia=True,
        )
        bottle = wp.transform((0.0, 0.0, 0.115), wp.quat_identity())
        builder.add_shape_cylinder(source, xform=bottle, radius=0.0305, half_height=0.115, cfg=glass)
        scene = SimpleNamespace(builder=builder, belts=[], metadata={}, bodies={}, bottle_bodies=[source])
        add_filling_transport(scene, path, motor_off=False, layout=layout, inlet_speed=0.14)
        add_filling_bottles(scene, path, layout=layout)
        model = builder.finalize()
        for mesh, velocity, _ in scene.belts:
            mesh.mesh.velocities.assign(velocity)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=3,
            solver_iterations=4,
            temporal_iterations=True,
            contact_hertz=80.0,
            step_layout="single_world",
        )
        state = model.state()
        contacts = model._collision_pipeline.contacts()
        with wp.ScopedCapture() as capture:
            solver.collide(state, contacts, dt=1.0 / 120.0)
            solver.step(state, state, None, contacts, 1.0 / 120.0)
        return scene, state, np.asarray(scene.bottle_bodies[1:]), capture.graph

    def test_accumulation_table_feeds_the_ifm_with_its_drive(self):
        """Run the IFM's feed with its infeed drive and stop it for a standing mass at PE-A1."""
        infeed = SimpleNamespace(
            sensor_slices={"PE-A1": slice(0, 2)},
            zone_eyes={"line": 2, "pool": 0},
            zone_monitor={"line": StandingQueue(), "pool": StandingQueue()},
            zone_speed={"line": 0.0, "pool": 0.0},
        )
        hits = np.zeros(4, dtype=bool)
        speeds = []
        for frame in range(6 * 60):
            time = frame / 60.0
            # A moving mass passes PE-A1 for 3 s; then it stands while M1 is stopped.
            moving = time < 3.0
            hits[0], hits[1] = True, (frame // 4) % 2 == 0 if moving else True
            InfeedController.update_zones(infeed, time, hits, 0.5 if moving else 0.0)
            speeds.append((infeed.zone_speed["pool"], infeed.zone_speed["line"]))
        speeds = np.asarray(speeds)
        # The table follows the half-speed infeed drive through its 1 s ramp,
        # then stops with M1 and stays stopped while the mass at PE-A1 stands.
        np.testing.assert_allclose(speeds[60:180, 0], 0.5)
        np.testing.assert_allclose(speeds[250:, 0], 0.0)
        # A zone with its own clear eye runs at full speed.
        np.testing.assert_allclose(speeds[60:, 1], 1.0)

    def test_feed_zone_stops_only_for_a_standing_queue(self):
        """Keep a feed running past a dense moving file and stop it for a standing queue."""
        eye, dt = StandingQueue(), 1.0 / 60.0
        # A moving file blocks the body beam but flickers the neck beam between necks.
        moving = [eye.update(i * dt, True, (i // 4) % 2 == 0) for i in range(300)]
        self.assertFalse(any(moving))
        standing = [eye.update(5.0 + i * dt, True, True) for i in range(120)]
        self.assertFalse(any(standing[:85]))
        self.assertTrue(all(standing[95:]))
        self.assertFalse(eye.update(7.1, False, False))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_filling_line_delivers_rated_output(self):
        """Carry production stock upright through a transfer arc at the line's rated output."""
        with wp.ScopedDevice("cuda:0"):
            # Enter a 0.2 m pitch-circle transfer, as between the rinser and filler wheels.
            arc = _arc((1.0, 0.2), 0.2, -0.5 * np.pi, 0.5 * np.pi, 1)
            xy = np.vstack(((0.0, 0.0), arc, (-2.0, 0.4)))
            path = np.column_stack((xy, np.full(len(xy), 0.95)))
            _, state, bottles, graph = self._bottle_line(path)
            dt = 1.0 / 120.0
            gate, passed, crossings = -1.2, np.zeros(len(bottles), dtype=bool), []
            for frame in range(1, 480):
                wp.capture_launch(graph)
                if frame % 4:
                    continue
                poses = state.body_q.numpy()[bottles]
                crossed = (poses[:, 0] < gate) & (poses[:, 1] > 0.2) & ~passed
                crossings.extend([frame * dt] * int(crossed.sum()))
                passed |= crossed
            poses = state.body_q.numpy()[bottles]

        self.assertGreaterEqual(len(crossings), 8)
        rate = 60.0 * (len(crossings) - 1) / (crossings[-1] - crossings[0])
        # The single file must carry at least the plant's rated 425 bottles/min.
        self.assertGreater(rate, 0.95 * 425.0)
        # Every bottle short of the gate, near the route's open end, stands upright on its deck.
        route = ~passed
        self.assertGreater(int(route.sum()), 10)
        self.assertTrue(np.all(poses[route, 2] > 0.945))
        # Same tilt bound as the example's wet-end handoff check.
        self.assertTrue(np.all(np.linalg.norm(poses[route, 3:5], axis=1) < np.sin(np.radians(7.5))))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_wet_end_tables_carry_packed_beds(self):
        """Carry a packed bed through the pasteuriser, its single filer and the accumulation table."""
        # The source's chute, funnel and table outlines, scaled down in width and length.
        layout = {
            "infeed": (np.array([1.0, 1.3, 1.6, 2.0]), np.array([0.12, 0.12, 0.35, 0.35])),
            "outfeed": (np.array([4.0, 4.4, 4.7, 5.0]), np.array([0.35, 0.35, 0.12, 0.12])),
            "combiner_end": 6.4,
            "pool": (np.array([6.8, 7.8, 7.85, 8.4]), np.array([0.35, 0.35, 0.15, 0.15])),
        }
        path = np.array([[0.0, 0.0, 0.95], [8.4, 0.0, 0.95]])
        dt = 1.0 / 120.0
        with wp.ScopedDevice("cuda:0"):
            _, state, bottles, graph = self._bottle_line(path, layout)
            start = state.body_q.numpy()[bottles]
            filed, emptied, tilt = set(), set(), 0.0
            for frame in range(1, 3000):
                wp.capture_launch(graph)
                if frame % 30 == 0:
                    poses = state.body_q.numpy()[bottles]
                    # Bottles tip once they drop off the open outlet; judge the line before it.
                    on_line = (poses[:, 0] < path[-1, 0] - 0.1) & (poses[:, 2] > 0.9)
                    tilt = max(tilt, float(np.linalg.norm(poses[on_line, 3:5], axis=1).max()))
                    if frame * dt > 10.0:
                        filed.update(np.flatnonzero(on_line & (poses[:, 0] > 6.6) & (poses[:, 0] < 6.8)))
                    emptied.update(np.flatnonzero(poses[:, 0] > path[-1, 0]))
            poses = state.body_q.numpy()[bottles]

        # The pasteuriser starts full and its bed advances at its rated speed.
        tunnel = (start[:, 0] > 2.0) & (start[:, 0] < 4.0)
        self.assertGreater(int(tunnel.sum()), 300)
        speed = PASTEURISER_CAPACITY * 425.0 / 60.0 * 0.5 * np.sqrt(3.0) * 0.063**2 / 0.7
        advance = poses[tunnel, 0] - start[tunnel, 0]
        self.assertGreater(float(np.median(advance)), 0.9 * speed * 3000 * dt)
        # The bed feeds the single filer, which files it for the accumulation
        # table at the rated output or above.
        self.assertGreater(len(filed), 0.9 * 425.0 / 60.0 * 15.0)
        # The accumulation table discharges its packed field through the neck.
        self.assertGreater(len(emptied), 100)
        # Bottles on every table and conveyor stay upright, and none leaves the line.
        self.assertLess(tilt, np.sin(np.radians(7.5)))
        kept = np.ones(len(bottles), dtype=bool)
        kept[list(emptied)] = False
        self.assertTrue(np.all(poses[kept, 2] > 0.94))

    def test_wrapper_checks_actual_carton_corner_speed(self):
        """Admit a seated carton only when every corner meets the motion limit."""
        signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = 0.1 * signs
        controller = SimpleNamespace(
            carton_roots=np.array([1]),
            pallet_corners={0: signs * [0.4, 0.3, 0.1]},
            carton_corners=np.array([corners]),
            layer_heights=np.array([0.1]),
            body_com=np.zeros((2, 3)),
            data={"sealed": np.ones(1), "claimed": np.zeros(2)},
            wrapped=set(),
        )
        heading = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.0)
        displacement = np.asarray(wp.quat_rotate(heading, wp.vec3(0.05, 0, 0.2)))
        poses = np.array([[0.0, 0.0, 0.0, *heading], [*displacement, *heading]])
        velocities = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.5]] * 2)
        velocities[1, :3] += np.cross(velocities[0, 3:], displacement)
        velocities[1, 5] += 0.05
        np.testing.assert_array_equal(WrapperController._load(controller, {"pallet": 0}, poses, velocities), [0])

        velocities[1, 5] += 0.03
        self.assertIsNone(WrapperController._load(controller, {"pallet": 0}, poses, velocities))

    def test_loaded_transfer_contact_ownership(self):
        fleet = "/Plant/Logistics/Fleet/"
        joints = [
            fleet + "ab_U/lift_joint",
            fleet + "ab_U/back_wheel_joints/back_wheel_drive",
            fleet + "ab_D/lift_joint",
            fleet + "ab_D/back_wheel_joints/back_wheel_drive",
            fleet + "c_U/lift_joint",
            fleet + "c_D/lift_joint",
            fleet + "c_D/disabled_joint",
            "/Plant/Line/Palletizers/PZ1/drive_joint",
        ]
        exact = np.array([True, True, True, True, True, True, False, True])
        contacts = _fleet_direct_contact_mask(joints, exact)
        np.testing.assert_array_equal(contacts, [False, False, False, False, True, True, False, False])
        np.testing.assert_array_equal(exact, [True, True, True, True, True, True, False, True])
        self.assertIsNone(_fleet_direct_contact_mask(joints, np.zeros_like(exact)))

    def test_packing_fixture_drive_ownership(self):
        """Only independent roller and stop axes on a world-fixed mount use local contact steps."""
        builder = newton.ModelBuilder()
        expected = []
        mounts = []
        for kind in ("packing", "linked", "unwelded", "conveyor"):
            mount, roller, stop = (builder.add_link() for _ in range(3))
            mounts.append(mount)
            if kind != "unwelded":
                builder.add_joint_fixed(-1, mount)
            spindle = builder.add_joint_revolute(mount, roller)
            if kind != "conveyor":
                slide = builder.add_joint_prismatic(mount, stop, target_ke=1000.0)
                if kind == "packing":
                    expected.extend((spindle, slide))
                elif kind == "linked":
                    builder.add_joint_fixed(stop, builder.add_link())

        self.assertEqual(_local_fixture_drives(builder), expected)
        self.assertEqual(builder.body_flags[mounts[0]], newton.BodyFlags.KINEMATIC)
        self.assertTrue(all(builder.body_flags[m] == newton.BodyFlags.DYNAMIC for m in mounts[1:]))

    def test_palletizer_places_off_center_grasps(self):
        """Keep the carried carton centered through a turn without changing the authored timing or tool height."""
        times = np.array([0.0, 0.5, 1.0])
        pick = np.radians([15, -70, 85, 10])
        above = np.radians([-35, -65, 80, -10])
        down = np.radians([-35, -58, 87, -10])
        limits = np.radians([[-155, -120, 15, -350], [155, -15, 145, 350]])
        offset = np.array([0.016, -0.004])

        def flange(q):
            a1, a2, a3, _ = q.T
            radius = 0.42 + 0.85 * (np.cos(a2) + np.cos(a2 + a3))
            return np.column_stack((radius * np.cos(a1), -radius * np.sin(a1), -0.85 * (np.sin(a2) + np.sin(a2 + a3))))

        for orientation, wrist_turn in (("X", 0.0), ("Y", np.pi / 2)):
            with self.subTest(orientation=orientation):
                picked = pick.copy()
                picked[3] += wrist_turn
                original = {
                    f"{phase}_{orientation}_4": (times, np.linspace(start, end, len(times)))
                    for phase, start, end in (
                        ("transfer", picked, above),
                        ("descend", above, down),
                        ("lift", down, above),
                    )
                }
                self.assertIs(_placement_motions(original, orientation, 4, picked, np.zeros(2), limits), original)
                motions = _placement_motions(original, orientation, 4, picked, offset, limits)
                for key, (stamp, q) in motions.items():
                    self.assertIs(stamp, times)
                    authored = original[key][1]
                    np.testing.assert_allclose(flange(q)[:, 2], flange(authored)[:, 2], atol=1e-12)
                    np.testing.assert_allclose(q[:, 0] + q[:, 3], authored[:, 0] + authored[:, 3], atol=1e-12)
                descent = f"descend_{orientation}_4"
                yaw = picked[0] + picked[3] - above[0] - above[3]
                world_offset = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ offset
                np.testing.assert_allclose(
                    flange(motions[descent][1])[:, :2] + world_offset, flange(original[descent][1])[:, :2], atol=1e-12
                )
                transfer, lift = motions[f"transfer_{orientation}_4"][1], motions[f"lift_{orientation}_4"][1]
                np.testing.assert_allclose(transfer[[0, -1]], [picked, motions[descent][1][0]], atol=1e-12)
                np.testing.assert_allclose(lift[[0, -1]], [motions[descent][1][-1], above], atol=1e-12)
                self.assertIs(_placement_motions(original, orientation, 4, picked, offset * 300, limits), original)

    def test_finite_bottle_supply(self):
        """Admit supported compound bottles only through a clear inlet and stop at the partial final row."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            newton.solvers.SolverPhoenX.register_custom_attributes(builder)
            builder.add_shape_box(-1, hx=1, hy=1, hz=0.01, xform=wp.transform((0, 0, 0.94), wp.quat_identity()))
            label = "/Plant/Sim/Product/Bottles/B00000"
            source = builder.add_link(label=label, xform=wp.transform((0.12, 0, 0.951), wp.quat_identity()))
            hull = newton.Mesh.create_box(0.025, 0.025, 0.04)
            for i in range(2):
                builder.add_shape_convex_hull(
                    source,
                    mesh=hull,
                    xform=wp.transform((0, 0, 0.04 + 0.08 * i), wp.quat_identity()),
                    cfg=builder.ShapeConfig(mu=0.16, mu_static=0.20, gap=0.002, is_visible=False),
                    label=label + f"/Collision/Hull_0{i}/collision",
                    custom_attributes={"phoenx:restitution_combine_mode": i},
                )
            scene = PlantScene(builder, bottle_bodies=[source])
            supply = BottleSupply(scene, capacity=9, joints=1)
            memberships = np.zeros(builder.shape_count, dtype=np.uint64)
            memberships[supply.shapes] = [2, 4]
            filtered = memberships.copy()
            filtered[supply.shapes] = [6, 2]
            supply.copy_masks(memberships, filtered)
            np.testing.assert_array_equal(memberships[supply.first_shape :], np.tile([2, 4], 9))
            np.testing.assert_array_equal(filtered[supply.first_shape :], np.tile([6, 2], 9))
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model, broad_phase="sap", contact_matching="sticky", rigid_contact_max=1024
            )
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                substeps=3,
                solver_iterations=6,
                joint_mode="maximal_pgs",
                step_layout="single_world",
            )
            state = model.state()
            picking = newton.viewer.Picking(model)
            plant = SimpleNamespace(
                state=state,
                model=model,
                solver=solver,
                bottles=wp.array([source] + [-1] * 9, dtype=int),
                bottle_count=wp.array([1], dtype=int),
            )
            supply.bind(plant, np.eye(4), np.linspace(0, 1, 9))
            contacts = pipeline.contacts()

            def advance(frames=1):
                supply.admit()
                for _ in range(frames):
                    pipeline.collide(state, contacts)
                    solver.step(state, state, None, contacts, 1.0 / 60.0)

            supply.request(0, False)
            advance()
            self.assertEqual(plant.bottle_count.numpy()[0], 1)  # source bottle occupies the feed slot
            supply.request(1, True)
            advance()
            poses = state.body_q.numpy().copy()
            poses[source, 0] = 0.8
            state.body_q.assign(poses)
            supply.request(3, True)
            advance()
            self.assertEqual(plant.bottle_count.numpy()[0], 1)  # sustained PE-A1 coverage
            supply.request(4, False)
            self.assertEqual(plant.bottle_count.numpy()[0], 1)  # post-step observers still see committed stock
            advance(90)
            self.assertEqual(plant.bottle_count.numpy()[0], 8)
            row = plant.bottles.numpy()[1:8]
            modes = solver.world._materials.numpy()["restitution_combine_mode"]
            np.testing.assert_array_equal(modes[supply.first_shape :], np.tile([0, 1], 9))
            poses = state.body_q.numpy().copy()
            np.testing.assert_allclose(poses[row, 2], 0.95, atol=0.002)
            np.testing.assert_allclose(model.body_mass.numpy()[row], model.body_mass.numpy()[source])
            self.assertTrue(np.all(model.body_world.numpy()[row] == 0))
            dormant = np.setdiff1d(np.arange(supply.first_body, supply.first_body + supply.capacity), row)
            dormant_shapes = np.isin(model.shape_body.numpy(), dormant)
            self.assertEqual(dormant_shapes.sum(), 4)
            np.testing.assert_array_equal(model.shape_collision_group.numpy()[dormant_shapes], 0)
            np.testing.assert_array_equal(model.body_mass.numpy()[dormant], 0.0)
            poses[row, 0] += 0.4
            state.body_q.assign(poses)
            supply.request(5, False)
            advance(90)
            supply.request(6, False)
            advance()
            ids = plant.bottles.numpy()
            self.assertEqual(plant.bottle_count.numpy()[0], 10)
            self.assertEqual(len(set(ids)), 10)
            np.testing.assert_allclose(state.body_q.numpy()[ids, 2], 0.95, atol=0.002)
            self.assertEqual(int(solver.dynamics.error_count.numpy()[0]), 0)
            # The viewer was open before the inlet activated this body slot.
            bottle = ids[-1]
            x, y, z = state.body_q.numpy()[bottle, :3]
            picking.pick(state, wp.vec3(x, y, z + 1.0), wp.vec3(0.0, 0.0, -1.0))
            self.assertEqual(picking.pick_body, bottle)
            picking.pick_target = picking.pick_point + wp.vec3(0.0, 0.0, 0.5)
            state.clear_forces()
            picking.apply_force(state)
            advance()
            self.assertGreater(state.body_q.numpy()[bottle, 2], z + 0.001)

    @staticmethod
    def _wrapper_components(side):
        components = []
        for name, center, half in (
            ("steel", (0, 0, 2.9585), (0.34, 0.34, 0.0585)),
            ("steel", (-0.95, -0.95, 1.722), (0.065, 0.065, 1.105)),
            ("wrapline_film_roll", (-1.05, -0.53, 0.927), (0.125, 0.125, 0.25)),
            ("machine", (-0.25, 0, 3.025), (0.06, 1.425, 0.125)),
            ("machine", (0.25, 0, 3.025), (0.06, 1.425, 0.125)),
            ("machine", (-0.85, -1.5, 1.5825), (0.075, 0.075, 1.5675)),
            ("machine", (-0.85, 1.5, 1.5825), (0.075, 0.075, 1.5675)),
            ("machine", (0.85, -1.5, 1.5825), (0.075, 0.075, 1.5675)),
            ("machine", (0.85, 1.5, 1.5825), (0.075, 0.075, 1.5675)),
            ("galv", (-0.85, -1.5, 0.0075), (0.125, 0.125, 0.0075)),
            ("galv", (-0.85, 1.5, 0.0075), (0.125, 0.125, 0.0075)),
            ("galv", (0.85, -1.5, 0.0075), (0.125, 0.125, 0.0075)),
            ("galv", (0.85, 1.5, 0.0075), (0.125, 0.125, 0.0075)),
        ):
            mesh = newton.Mesh.create_box(*half, compute_inertia=False)
            translated = np.asarray(center) * (1, side, 1) + (5, side * 10, 0)
            components.append((name, mesh.copy(vertices=mesh.vertices + translated)))
        return components

    def test_palletizer_detects_carton_support(self):
        """Release needs upward support around a slow carton during the placement stroke."""
        with wp.ScopedDevice("cpu"):
            # The carton root is 0.2 m above the pallet origin. Its welded flap has a distinct body.
            corners = np.array([[-0.09, -0.06, 0], [0.09, -0.06, 0], [-0.09, 0.06, 0], [0.09, 0.06, 0]])
            poses = wp.array([[0, 0, z, 0, 0, 0, 1] for z in (0, 0.2, 0.2)], dtype=wp.transform)
            held = wp.array([1, -1, -1, -1], dtype=int)
            bodies = wp.array([0, 1, 2], dtype=int)
            lower = wp.zeros(4, dtype=int)
            flap = wp.full(4, 2, dtype=int)
            mask = wp.zeros(4, dtype=int)
            margin = wp.full(4, 0.0001, dtype=float)

            for support, gap, speed, armed, count, reverse, expected in (
                (-2, 0.0006, 0.0, 1, 4, False, 15),  # surfaces are 0.4 mm apart after margins
                (0, 0.0006, 0.0, 1, 4, True, 15),  # another carton, with the witnesses exchanged
                (-2, 0.001, 0.0, 1, 4, False, 0),
                (-2, 0.0006, 0.0, 1, 3, False, 7),
                (-2, 0.0006, 0.2, 1, 4, False, 0),
                (-2, 0.0006, 0.0, 2, 4, False, 0),
                (-1, 0.0006, 0.0, 1, 4, False, 0),
            ):
                with self.subTest(support=support, gap=gap, speed=speed, armed=armed, count=count, reverse=reverse):
                    mask.zero_()
                    top = wp.array(corners + np.array([0, 0, gap - 0.2]), dtype=wp.vec3)
                    bottom = wp.array(corners, dtype=wp.vec3)
                    velocity = wp.array([[0, 0, z, 0, 0, 0] for z in (0, speed, speed)], dtype=wp.spatial_vector)
                    wp.launch(
                        _carton_support_quadrants,
                        4,
                        inputs=[
                            wp.array([count], dtype=int),
                            flap if reverse else lower,
                            lower if reverse else flap,
                            bodies,
                            wp.array([support, 1, 1], dtype=int),
                            held,
                            wp.array([armed], dtype=int),
                            poses,
                            velocity,
                            top if reverse else bottom,
                            bottom if reverse else top,
                            wp.array([[0, 0, -1.0 if reverse else 1.0]] * 4, dtype=wp.vec3),
                            margin,
                            margin,
                            mask,
                            4,
                        ],
                    )
                    np.testing.assert_array_equal(mask.numpy(), [expected, 0, 0, 0])

    def test_wrapper_frame_clearance(self):
        """Remove source crossmember interference without altering moving mass or geometry."""
        with wp.ScopedDevice("cpu"):
            for side in (-1, 1):
                components = self._wrapper_components(side)
                original = newton.ModelBuilder()
                _, _, assignment = build_wrapper(original, components, label="/Wrapper", visuals=True)
                fixed = [
                    i
                    for i, label in enumerate(original.shape_label)
                    if label.endswith("/collision") and original.shape_body[i] < 0
                ]
                self.assertEqual(len(fixed), 10, "Every fixed frame component needs an actual collider")
                for shape, shape_label in enumerate(original.shape_label):
                    if shape_label.endswith("/collision"):
                        self.assertEqual(original.shape_gap[shape], 0.002)

                corrected, info = correct_wrapper_frame(components, label="/Wrapper")
                self.assertAlmostEqual(info["raise_m"], 0.127, places=5)
                builder = newton.ModelBuilder()
                build_wrapper(builder, corrected, label="/Wrapper", visuals=True)
                np.testing.assert_array_equal(builder.body_mass, original.body_mass)
                np.testing.assert_array_equal(builder.body_inertia, original.body_inertia)
                np.testing.assert_array_equal(builder.body_q, original.body_q)
                for i, ((name, before), (_, after)) in enumerate(zip(components, corrected, strict=True)):
                    if assignment[i] >= 0:
                        np.testing.assert_array_equal(before.vertices, after.vertices)
                    else:
                        collision = builder.shape_label.index(f"/Wrapper/{name}_{i}/collision")
                        visual = builder.shape_label.index(f"/Wrapper/{name}_{i}/visual")
                        np.testing.assert_array_equal(
                            builder.shape_source[collision].vertices, builder.shape_source[visual].vertices
                        )

                for corrected_mode, scene in ((False, original), (True, builder)):
                    model = scene.finalize()
                    state = model.state()
                    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=1024)
                    contacts = pipeline.contacts()
                    shape_body = model.shape_body.numpy()
                    q = model.joint_q.numpy()
                    penetrations = 0
                    for angle in range(0, 360, 45) if corrected_mode else (0,):
                        q[0] = np.radians(angle)
                        model.joint_q.assign(q)
                        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                        pipeline.collide(state, contacts)
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        a = contacts.rigid_contact_shape0.numpy()[:count]
                        b = contacts.rigid_contact_shape1.numpy()[:count]
                        p0 = contacts.rigid_contact_point0.numpy()[:count]
                        p1 = contacts.rigid_contact_point1.numpy()[:count]
                        normals = contacts.rigid_contact_normal.numpy()[:count]
                        poses = state.body_q.numpy()
                        for k in range(count):
                            if (shape_body[a[k]] >= 0) == (shape_body[b[k]] >= 0):
                                continue
                            points = []
                            for shape, point in ((a[k], p0[k]), (b[k], p1[k])):
                                body = shape_body[shape]
                                points.append(
                                    np.asarray(
                                        wp.transform_point(
                                            wp.transform(poses[body, :3], poses[body, 3:]), wp.vec3(point)
                                        )
                                    )
                                    if body >= 0
                                    else point
                                )
                            penetrations += np.dot(points[1] - points[0], normals[k]) < -1e-4
                    if corrected_mode:
                        self.assertEqual(penetrations, 0)
                    else:
                        self.assertGreater(penetrations, 0, "The source geometry must expose the actual interference")

    def test_wrapper_portal_proxy_ownership(self):
        """Replace portal proxies only when exact wrapper colliders own their contacts."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Plant").GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            label = "/Plant/WrapLine/Lines/WL_U/Wrapper"
            groups = {}
            for name, mesh in self._wrapper_components(1):
                groups.setdefault(name, []).append(mesh)
            for name, meshes in groups.items():
                prim = UsdGeom.Mesh.Define(stage, label + "/" + name)
                points = np.concatenate([mesh.vertices for mesh in meshes])
                starts = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
                indices = np.concatenate([mesh.indices + offset for mesh, offset in zip(meshes, starts, strict=True)])
                prim.CreatePointsAttr(points.tolist())
                prim.CreateFaceVertexIndicesAttr(indices.tolist())
                prim.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            portals = []
            centers = [(4.15, 8.5), (4.15, 11.5), (5.85, 8.5), (5.85, 11.5)]
            for suffix, (x, y) in zip(("da", "df", "ua", "uf"), centers, strict=True):
                portal_path = label.removesuffix("/Wrapper") + "/Physics/Static/portal_" + suffix
                portal = UsdGeom.Cube.Define(stage, portal_path)
                portal.CreateSizeAttr(1.0)
                portal.AddTranslateOp().Set((x, y, 1.575))
                portal.AddScaleOp().Set((0.15, 0.15, 3.15))
                UsdPhysics.CollisionAPI.Apply(portal.GetPrim())
                portals.append(portal_path)
            line = label.removesuffix("/Wrapper")
            for i in range(1, 6):
                zone_root = UsdGeom.Xform.Define(stage, f"{line}/Physics/Conveyors/Z{i}")
                zone = UsdGeom.Cube.Define(stage, zone_root.GetPath().AppendChild("Strand_0"))
                zone.CreateSizeAttr(1.0)
                zone.AddTranslateOp().Set((float(i), 10.0, 0.4))
                zone.AddScaleOp().Set((1.0, 1.0, 0.2))
                UsdPhysics.CollisionAPI.Apply(zone.GetPrim())

            navigation = "/Plant/Physics/NavProxies"
            wrappers = f"{label}/machine; {label}/steel"
            proxy_owners = {
                "Wrapper": wrappers,
                "Mixed": f"{wrappers}; {line}/Conveyor/galv",
                "External": f"{wrappers}; /Plant/Building/Wall",
            }
            for name, owners in proxy_owners.items():
                proxy = UsdGeom.Cube.Define(stage, f"{navigation}/{name}")
                proxy.CreateSizeAttr(1.0)
                proxy.AddTranslateOp().Set((5.0, 12.0 if name == "External" else 10.0, 1.5))
                proxy.GetPrim().CreateAttribute("plant:owner", Sdf.ValueTypeNames.String).Set(owners)
                UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())

            for reconstructed, belts in ((True, True), (True, False), (False, True)):
                stage.GetPrimAtPath(label).SetActive(reconstructed)
                stage.GetPrimAtPath(f"{line}/Physics/Conveyors").SetActive(belts)
                stage.GetRootLayer().Save()
                scene = load_plant(root, visuals=False, motor_off=True)
                builder = scene.builder
                probes = []
                for x, y in centers:
                    body = builder.add_body(xform=wp.transform((x + 0.08, y, 1.0), wp.quat_identity()))
                    probes.append(builder.add_shape_sphere(body, radius=0.01))
                tunnel = builder.add_body(xform=wp.transform((5.0, 10.0, 1.5), wp.quat_identity()))
                tunnel_probe = builder.add_shape_sphere(tunnel, radius=0.01)
                model = builder.finalize()
                pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                first = contacts.rigid_contact_shape0.numpy()[:count]
                second = contacts.rigid_contact_shape1.numpy()[:count]
                for probe in probes:
                    hit_shapes = set(second[first == probe]) | set(first[second == probe])
                    self.assertEqual(len(hit_shapes), 1, "Each post must have one collider owner, without a gap")
                if reconstructed:
                    self.assertTrue({path + "/collision" for path in portals}.isdisjoint(builder.shape_label))
                    self.assertEqual(
                        scene.metadata["wrapper_frame_corrections"][label]["replaced_collision_proxies"], portals
                    )
                else:
                    self.assertTrue({path + "/collision" for path in portals}.issubset(builder.shape_label))
                for name, covered in (
                    ("Wrapper", reconstructed),
                    ("Mixed", reconstructed and belts),
                    ("External", False),
                ):
                    self.assertEqual(f"{navigation}/{name}/collision" in builder.shape_label, not covered)
                tunnel_hits = set(second[first == tunnel_probe]) | set(first[second == tunnel_probe])
                navigation_hits = {builder.shape_label[i] for i in tunnel_hits} & {
                    f"{navigation}/Wrapper/collision",
                    f"{navigation}/Mixed/collision",
                }
                self.assertEqual(len(navigation_hits), int(not reconstructed) + int(not (reconstructed and belts)))

    def test_rack_inventory(self):
        """Import wrapped, open and empty stock on the rack rows, floor and in pallet stacks."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        def box(stage, path, center, size):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(center)
            cube.AddScaleOp().Set(size)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            return cube.GetPrim()

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            unit_file = root / "assets/pallet_load/pallet_load.usda"
            unit_file.parent.mkdir(parents=True)
            unit = Usd.Stage.CreateNew(str(unit_file))
            unit_root = UsdGeom.Xform.Define(unit, "/Unit").GetPrim()
            unit.SetDefaultPrim(unit_root)
            UsdPhysics.RigidBodyAPI.Apply(unit_root)
            UsdPhysics.MassAPI.Apply(unit_root).CreateMassAttr(300)
            pallet = UsdGeom.Xform.Define(unit, "/Unit/Pallet/Geometry/sm_pallet_block_a08_obj_00").GetPrim()
            UsdPhysics.MassAPI.Apply(pallet).CreateMassAttr(60)
            for i, x in enumerate((-0.4, 0.4)):
                box(unit, pallet.GetPath().AppendChild(f"Runner{i}"), (x, 0, 0.05), (0.1, 1.2, 0.1))
            box(unit, "/Unit/Collision/LoadBlock", (0, 0, 0.4), (1, 1.2, 0.6))
            unit.GetRootLayer().Save()

            supply_file = root / "assets/living_plant/carton_blanks_load/carton_blanks_load.usda"
            supply_file.parent.mkdir(parents=True)
            source = Usd.Stage.CreateNew(str(supply_file))
            payload = UsdGeom.Xform.Define(source, "/Blanks").GetPrim()
            source.SetDefaultPrim(payload)
            UsdPhysics.MassAPI.Apply(payload).CreateMassAttr(120)
            box(source, "/Blanks/Collision/box_0", (0, 0, 1), (1, 1, 2))
            source.GetRootLayer().Save()

            dimensions = root / "assets/beerbox/dimensions.json"
            dimensions.parent.mkdir()
            dimensions.write_text('{"outside_m": [0.4, 0.3, 0.2]}')
            stage_file = root / "scenarios/line_sim/line_sim.usda"
            stage_file.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(stage_file))
            stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Plant").GetPrim())
            UsdGeom.SetStageUpAxis(stage, "Z")
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            beam = box(stage, "/Plant/Physics/Racks/rack_1_11_0/Bay_00/Beam", (5, 3, 0.5), (3, 1.5, 1))
            UsdGeom.Imageable(beam).CreatePurposeAttr("guide")
            beam = box(stage, "/Plant/Physics/Racks/second/Bay_02/Beam", (5, 7, 0.5), (3, 1.5, 1))
            UsdGeom.Imageable(beam).CreatePurposeAttr("guide")
            inventory = UsdGeom.PointInstancer.Define(stage, "/Plant/Logistics/PalletLoads")
            prototype = UsdGeom.Xform.Define(stage, inventory.GetPath().AppendChild("FinishedGoods"))
            frame = UsdGeom.Xform.Define(stage, prototype.GetPath().AppendChild("Pallet"))
            frame.AddRotateZOp().Set(90)
            UsdGeom.Cube.Define(stage, frame.GetPath().AppendChild("Stock")).CreateSizeAttr(0.4)
            partial = UsdGeom.Xform.Define(stage, inventory.GetPath().AppendChild("LoadPartial"))
            partial_pallet = UsdGeom.Xform.Define(stage, partial.GetPath().AppendChild("Pallet"))
            UsdGeom.Cube.Define(stage, partial_pallet.GetPath().AppendChild("Wood")).CreateSizeAttr(0.1)
            case = UsdGeom.Xform.Define(stage, partial.GetPath().AppendPath("Cases/C0"))
            case.AddTranslateOp().Set((0.2, 0, 0.1))
            UsdGeom.Cube.Define(stage, case.GetPath().AppendChild("Box")).CreateSizeAttr(0.3)
            empty = UsdGeom.Xform.Define(stage, inventory.GetPath().AppendChild("EmptyPallet"))
            UsdGeom.Xform.Define(stage, empty.GetPath().AppendChild("Pallet"))
            blanks = UsdGeom.Xform.Define(stage, inventory.GetPath().AppendChild("CartonBlanks"))
            UsdGeom.Xform.Define(stage, blanks.GetPath().AppendChild("Pallet"))
            cargo = UsdGeom.Cube.Define(stage, blanks.GetPath().AppendChild("Geo"))
            cargo.CreateSizeAttr(1)
            cargo.AddTranslateOp().Set((0, 0, 0.5))
            cargo.AddScaleOp().Set((0.8, 0.7, 0.8))
            inventory.GetPrototypesRel().SetTargets(
                [prototype.GetPath(), partial.GetPath(), empty.GetPath(), blanks.GetPath()]
            )
            inventory.CreateIdsAttr([7, 2, 9, 5, 3])
            inventory.CreateProtoIndicesAttr([0, 0, 1, 2, 3])
            inventory.CreatePositionsAttr([(5, 3, 1), (9, 3, 1), (5, 7, 1), (6, 7, 1), (9, 7, 0)])
            for name, kind, center in (
                ("occupied", "packaging_store", (9, 7, 1)),
                ("unused", "packaging_store", (30, 7, 1)),
                ("fixture", "machine", (9, 7, 1)),
            ):
                proxy = box(stage, "/Plant/Physics/Equipment/" + name, center, (2, 2, 2))
                proxy.CreateAttribute("plant:kind", Sdf.ValueTypeNames.String).Set(kind)
                UsdGeom.Imageable(proxy).CreatePurposeAttr("guide")
            for identifier, x, y, kind in (
                (7, 5, 3, "FinishedGoods"),
                (2, 9, 3, "FinishedGoods"),
                (9, 5, 7, "LoadPartial"),
                (5, 6, 7, "EmptyPallet"),
                (3, 9, 7, "CartonBlanks"),
            ):
                proxy = box(stage, f"/Plant/Physics/Loads/load_{identifier:05d}", (x, y, 1.4), (1.2, 1, 0.8))
                proxy.CreateAttribute("plant:kind", Sdf.ValueTypeNames.String).Set(kind)
                UsdGeom.Imageable(proxy).CreatePurposeAttr("guide")
            placed = "/Plant/LivingPlant/Warehouse/W4_EmptyPallets/pallet_empty_1_01"
            direct = UsdGeom.Xform.Define(stage, placed).GetPrim()
            direct.CreateAttribute("living:asset", Sdf.ValueTypeNames.String).Set("pallet")
            physics = direct.GetVariantSets().AddVariantSet("physics")
            for variant in ("none", "dynamic"):
                physics.AddVariant(variant)
            physics.SetVariantSelection("dynamic")
            with physics.GetVariantEditContext():
                UsdPhysics.RigidBodyAPI.Apply(direct)
                box(stage, placed + "/Board", (0, 0, 0.05), (0.1, 1.2, 0.1))
            physics.SetVariantSelection("none")
            supply = "/Plant/LivingPlant/PackingHall/P2_CartonSupply/slip_sheet_load_pz_u"
            sheet = UsdGeom.Xform.Define(stage, supply)
            sheet.GetPrim().GetReferences().AddInternalReference(placed)
            sheet.GetPrim().GetVariantSet("physics").SetVariantSelection("none")
            sheet.GetPrim().CreateAttribute("living:asset", Sdf.ValueTypeNames.String).Set("slip_sheet_load")
            sheet.AddTranslateOp().Set((3, 0, 0.15))
            box(stage, supply + "/Collision/box_0", (0, 0, 0.3), (1, 1.2, 0.64))
            for name, owners in (("Stock", placed + "/Board"), ("Shared", placed + "/Board;/Plant/Other")):
                proxy = box(stage, f"/Plant/Physics/NavProxies/{name}", (0, 0, 0.5), (1, 1, 1))
                proxy.CreateAttribute("plant:owner", Sdf.ValueTypeNames.String).Set(owners)
            stage.GetRootLayer().Save()

            scene = load_plant(root, motor_off=True)
            b = scene.builder
            self.assertIn(placed, scene.bodies)
            self.assertIn(supply, scene.bodies)
            sheet_body = scene.bodies[supply]
            self.assertEqual(b.shape_body[b.shape_label.index(supply + "/Board/collision")], sheet_body)
            np.testing.assert_allclose(b.body_q[sheet_body].p, (3, 0, 0.15), atol=1e-6)
            shape = b.shape_label.index(supply + "/Collision/box_0/collision")
            z, half = b.shape_transform[shape].p[2], b.shape_scale[shape][2]
            np.testing.assert_allclose((z - half, z + half), (0, 0.62), atol=1e-6)
            self.assertNotIn("/Plant/Physics/Equipment/occupied/collision", b.shape_label)
            self.assertIn("/Plant/Physics/Equipment/unused/collision", b.shape_label)
            self.assertIn("/Plant/Physics/Equipment/fixture/collision", b.shape_label)
            self.assertNotIn("/Plant/Physics/NavProxies/Stock/collision", b.shape_label)
            self.assertIn("/Plant/Physics/NavProxies/Shared/collision", b.shape_label)
            payload = scene.bodies["/Plant/Sim/FloorStock/load_00003"]
            self.assertAlmostEqual(b.body_mass[payload], 180, places=3)
            cargo_shape = b.shape_label.index("/Plant/Sim/FloorStock/load_00003/Payload/Collision/box_0/collision")
            np.testing.assert_allclose(b.shape_scale[cargo_shape], (0.4, 0.35, 0.4), atol=1e-6)
            np.testing.assert_allclose(b.shape_transform[cargo_shape].p, (0, 0, 0.5), atol=1e-6)
            body = scene.bodies["/Plant/Sim/RackStock/load_00007"]
            self.assertAlmostEqual(b.body_mass[body], 300, places=3)
            np.testing.assert_allclose(b.body_com[body], (0, 0, 0.33), atol=1e-5)
            np.testing.assert_allclose(b.body_q[body].p, (5, 3, 1), atol=1e-6)
            self.assertNotIn("/Plant/Physics/Loads/load_00007/collision", b.shape_label)
            floor = scene.bodies["/Plant/Sim/FloorStock/load_00002"]
            self.assertNotIn("/Plant/Physics/Loads/load_00002/collision", b.shape_label)
            self.assertAlmostEqual(b.body_mass[floor], 300, places=3)
            np.testing.assert_allclose(b.body_q[floor].p, (9, 3, 1), atol=1e-6)
            visuals = [i for i, label in enumerate(b.shape_label) if label.endswith("/instance")]
            partial_body = scene.bodies["/Plant/Sim/RackStock/load_00009"]
            empty_body = scene.bodies["/Plant/Sim/RackStock/load_00005"]
            carton = scene.bodies["/Plant/Sim/RackStock/load_00009/Cases/C0"]
            self.assertCountEqual([b.shape_body[i] for i in visuals], [body, floor, partial_body, carton, payload])
            self.assertAlmostEqual(b.body_mass[empty_body], 60, places=3)
            self.assertAlmostEqual(b.body_mass[partial_body], 60, places=3)
            self.assertAlmostEqual(b.body_mass[carton], 240 / 34, places=3)
            np.testing.assert_allclose(b.body_q[carton].p, (5.2, 7, 1.1), atol=1e-6)
            for identifier in (9, 5):
                self.assertNotIn(f"/Plant/Physics/Loads/load_{identifier:05d}/collision", b.shape_label)
            world = wp.transform_multiply(b.body_q[body], b.shape_transform[visuals[0]])
            np.testing.assert_allclose(world.p, (5, 3, 1), atol=1e-6)
            np.testing.assert_allclose(wp.quat_rotate(world.q, wp.vec3(1, 0, 0)), (1, 0, 0), atol=1e-6)
            self.assertEqual(
                sum(b.shape_body[i] == body and label.endswith("/collision") for i, label in enumerate(b.shape_label)),
                3,
            )

    def test_wrapper_infeed_stops_loaded_pallets_after_withdrawal(self):
        """Transport both line orientations using measured rays and stop before wrapping."""
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            folder = root / "plant/lines_1x8/wrapline"
            folder.mkdir(parents=True)
            builder = newton.ModelBuilder()
            channels, ports, pallets, strands, rotors = {}, {}, [], [], []
            for line, y, side in (("WL_U", 2.0, 1), ("WL_D", -2.0, -1)):
                prefix = f"/Plant/WrapLine/Lines/{line}"
                pallet = builder.add_body(
                    xform=wp.transform(wp.vec3(0, y, 0.501)), mass=61.0, inertia=wp.mat33(np.eye(3) * 10)
                )
                pallets.append(pallet)
                builder.add_shape_box(
                    pallet,
                    xform=wp.transform(wp.vec3(0, 0, 0.04)),
                    hx=0.5,
                    hy=0.6,
                    hz=0.04,
                    cfg=builder.ShapeConfig(density=0.0, mu=0.7, gap=0.002),
                )
                load = builder.add_body(xform=wp.transform(wp.vec3(0, y, 1.082)))
                builder.add_shape_box(
                    load,
                    hx=0.45,
                    hy=0.55,
                    hz=0.5,
                    cfg=builder.ShapeConfig(density=250.0 / (0.9 * 1.1), mu=0.7, gap=0.002),
                )
                rotor = builder.add_link(xform=wp.transform(wp.vec3(10, y, 3)), mass=1.0, inertia=wp.mat33(np.eye(3)))
                rotors.append(rotor)
                joint = builder.add_joint_revolute(
                    -1,
                    rotor,
                    parent_xform=builder.body_q[rotor],
                    axis=(0, 0, 1),
                    target_ke=1000.0,
                    target_kd=100.0,
                    label=prefix + "/Wrapper/rotation",
                )
                builder.add_articulation([joint])
                zones = {f"Z{i}": {"prim": prefix + f"/Physics/Conveyors/Z{i}"} for i in range(1, 6)}
                for zone in zones.values():
                    channels[zone["prim"]] = len(channels)
                for offset in (-0.5375, 0.5375):
                    box = newton.Mesh.create_box(1.5, 0.018, 0.02, compute_inertia=False)
                    mesh = newton.Mesh(box.vertices, box.indices, compute_inertia=False, enable_surface_velocity=True)
                    builder.add_shape_mesh(
                        -1,
                        mesh=mesh,
                        xform=wp.transform(wp.vec3(0.5, y + offset, 0.48)),
                        cfg=builder.ShapeConfig(mu=0.7, gap=0.002),
                    )
                    strands.append((mesh, channels[zones["Z1"]["prim"]]))
                ports[line + "_drop"] = SimpleNamespace(xy=(0, y), z=0.5, yaw=90.0 * side)
                spec = {
                    "unit_root": prefix,
                    "zones": zones,
                    "ports": {"drop": {"xy": [0, y], "z": 0.5}},
                    "sensors": {
                        name: {"origin": [x, y + side * 0.615, 0.532], "direction": [0, -side, 0], "distance": 0.5}
                        for name, x in (("PE_DROP", 0), ("PE_WRAP", 1.0), ("PE_PICK", 1.8))
                    },
                    "timers_s": {"infeed_timeout_s": 8.0, "settle_s": 0.5},
                }
                (folder / f"{line}.json").write_text(json.dumps(spec))
            model = builder.finalize()
            state, control = model.state(), model.control()
            fleet = FleetController.__new__(FleetController)
            fleet.pallet_bodies, fleet.state = np.asarray(pallets), state
            fleet.transport = SimpleNamespace(case_roots=np.empty(0, dtype=int))
            fleet.navigation = SimpleNamespace(ports=ports)
            fleet.missions = {
                name: _Transfer("source", name + "_drop", i, phase="home") for i, name in enumerate(("WL_U", "WL_D"))
            }
            wrapper = WrapperController(model, state, control, root, channels, fleet)
            solver = newton.solvers.SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                contact_hertz=80,
                joint_mode="maximal_pgs",
                step_layout="single_world",
            )
            contacts = solver._collision_pipeline.contacts()
            for frame in range(600):
                if frame == 60:
                    fleet.missions.clear()
                if frame == 62:
                    control.joint_target_q.fill_(0.5)
                if frame == 90:
                    control.joint_target_q.zero_()
                speed = np.ones(len(channels), dtype=np.float32)
                wrapper.before_step(frame / 120, speed, wrapper.observe())
                if frame < 60:
                    np.testing.assert_array_equal(speed, 0)
                if frame == 61:
                    for line in wrapper.report().values():
                        self.assertEqual(line["state"], "WAIT_DROP", line)
                for label, channel in channels.items():
                    if label.endswith(("Z4", "Z5")):
                        self.assertEqual(speed[channel], 0)
                for mesh, channel in strands:
                    velocity = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
                    velocity[:, 0] = 0.2 * speed[channel]
                    mesh.mesh.velocities.assign(velocity)
                state.clear_forces()
                if frame == 60:
                    # The rotor is initially at home but has not established
                    # quiet support. A real torque exposes premature admission.
                    forces = np.zeros((model.body_count, 6), dtype=np.float32)
                    forces[rotors, 5] = 10.0
                    state.body_f.assign(forces)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            for line in wrapper.report().values():
                self.assertEqual(line["state"], "WAIT_WRAP_PROCESS", line)
                self.assertIsNone(line["fault"])
            np.testing.assert_allclose(state.body_q.numpy()[pallets, 0], 0.5, atol=0.01)
            np.testing.assert_array_equal(speed, 0)

    def test_sleep_colors_preserve_material_and_render_snapshot(self):
        """Dim real sleeping bodies and restore their tint after a force wakes them."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
            )
            mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_inertia=False)
            mesh.texture = np.full((2, 2, 4), 255, dtype=np.uint8)
            mesh.color = (0.8, 0.6, 0.4)
            shape = builder.add_shape_mesh(body, mesh=mesh, cfg=builder.ShapeConfig(density=0.0))
            builder.add_ground_plane()
            model = builder.finalize()
            solver = newton.solvers.SolverPhoenX(
                model,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                sleeping_velocity_threshold=0.01,
                sleeping_frames_required=2,
            )
            example = Example.__new__(Example)
            example.model, example.solver = model, solver
            example.state_0 = model.state()
            example.contacts = model._collision_pipeline.contacts()
            example.viewer = ViewerNull()
            example.viewer.set_model(model)
            example.viewer.show_contacts = False
            example._render_states = (model.state(), model.state())
            example._render_state_done = (None, None)
            example._render_contacts = [None, None]
            example._render_contact_snapshot = None
            example._render_state_index = 0
            example._render_state_prepared = False
            example._sleep_color_active = wp.clone(model.shape_color)
            example._render_sleep_roots = tuple(wp.empty_like(solver.bodies.island_root) for _ in range(2))
            example.sim_time = 0.0
            active = model.shape_color.numpy().copy()
            for _ in range(4):
                solver.collide(example.state_0, example.contacts)
                solver.step(example.state_0, example.state_0, model.control(), example.contacts, 1.0 / 60.0)
            self.assertGreaterEqual(int(solver.bodies.island_root.numpy()[body + 1]), 0)
            example.prepare_render_state()
            forces = np.zeros((model.body_count, 6), dtype=np.float32)
            forces[body, 3] = 1.0
            example.state_0.body_f.assign(forces)
            solver.wake_on_external_input(example.state_0)
            self.assertLess(int(solver.bodies.island_root.numpy()[body + 1]), 0)
            example.render()
            np.testing.assert_allclose(model.shape_color.numpy()[shape], 0.55 * active[shape], atol=1e-6)
            np.testing.assert_allclose(model.shape_color.numpy()[-1], active[-1])
            example.prepare_render_state()
            example.render()
            np.testing.assert_array_equal(model.shape_color.numpy(), active)
            self.assertIs(model.shape_source[shape].texture, mesh.texture)

    def test_contact_overlay_uses_render_snapshot(self):
        """Render contact geometry and keep it independent of advancing physics."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.49), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5)
            builder.add_ground_plane()
            model = builder.finalize()
            example = Example.__new__(Example)
            example.model = model
            example.state_0 = model.state()
            pipeline = newton.CollisionPipeline(model)
            example.contacts = pipeline.contacts()
            pipeline.collide(example.state_0, example.contacts)
            count = int(example.contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0)
            example.viewer = ViewerNull()
            example.viewer.set_model(model)
            example.viewer.show_contacts = True
            example.viewer.show_contact_disks = False
            example.viewer.show_contact_forces = False
            example._render_states = (model.state(), model.state())
            example._render_state_done = (None, None)
            example._render_contacts = [None, None]
            example._render_contact_snapshot = None
            example._sleep_color_active = None
            example._render_sleep_roots = None
            example._render_state_index = 0
            example._render_state_prepared = False
            example.sim_time = 0.0
            example.render()
            self.assertIsNotNone(example.viewer._contact_points0)
            expected = example.viewer._contact_points0.numpy()[:count].copy()
            lengths = np.linalg.norm(example.viewer._contact_points1.numpy()[:count] - expected, axis=1)
            self.assertTrue(np.all(lengths > 0.0))
            example.prepare_render_state()
            snapshot = example._render_contact_snapshot
            self.assertIsNot(snapshot, example.contacts)
            # Physics may overwrite its buffers while the render stream consumes
            # the snapshot. The displayed arrows must retain the old geometry.
            example.contacts.rigid_contact_count.zero_()
            example.contacts.rigid_contact_point0.fill_(wp.vec3(100, 100, 100))
            example.state_0.body_q.fill_(wp.transform(wp.vec3(100, 0, 0), wp.quat_identity()))
            example.render()
            np.testing.assert_allclose(example.viewer._contact_points0.numpy()[:count], expected)
            self.assertEqual(int(snapshot.rigid_contact_count.numpy()[0]), count)
            example.viewer.show_contacts = False
            example.prepare_render_state()
            self.assertIsNone(example._render_contact_snapshot)
            # Enabling in begin-frame UI after snapshotting must defer contacts,
            # not read live buffers on the concurrently running physics stream.
            example.viewer.show_contacts = True
            example.render()
            np.testing.assert_allclose(example.viewer._contact_points0.numpy()[:count], expected)

    def test_offset_box_hulls_preserve_exact_geometry(self):
        """Recognize float32 box corners without replacing tapered hulls."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            corners = np.asarray(
                [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            )
            points = np.asarray(corners * (0.142875, 0.0015875, 0.127) + (0, 0.1063625, 0.130175), dtype=np.float32)
            for name in ("Box", "Tapered"):
                mesh = UsdGeom.Mesh.Define(stage, f"/Plant/{name}")
                vertices = points.copy()
                if name == "Tapered":
                    vertices[4:, 0] *= 0.99
                    mesh.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
                mesh.CreatePointsAttr(vertices.tolist())
                mesh.CreateFaceVertexCountsAttr([4] * 6)
                mesh.CreateFaceVertexIndicesAttr(
                    [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
                )
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            stage.GetRootLayer().Save()

            builder = load_plant(root, visuals=False, motor_off=True).builder
            box = builder.shape_label.index("/Plant/Box/collision")
            tapered = builder.shape_label.index("/Plant/Tapered/collision")
            self.assertEqual(builder.shape_type[box], newton.GeoType.BOX)
            self.assertEqual(builder.shape_type[tapered], newton.GeoType.CONVEX_MESH)
            self.assertAlmostEqual(builder.shape_gap[box], 0.001)
            self.assertAlmostEqual(builder.shape_gap[tapered], 0.002)
            center = np.asarray(builder.shape_transform[box].p)
            half = np.asarray(builder.shape_scale[box])
            np.testing.assert_allclose(center - half, points.min(axis=0), rtol=0, atol=2e-8)
            np.testing.assert_allclose(center + half, points.max(axis=0), rtol=0, atol=2e-8)

    def test_wrapper_components_preserve_material_seams(self):
        """Split disconnected parts without welding away authored UV seams."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
            dtype=np.float32,
        )
        uvs = np.arange(18, dtype=np.float32).reshape(-1, 2) / 18
        normals = np.tile([0, 0, 1], (9, 1)).astype(np.float32)
        mesh = newton.Mesh(
            vertices,
            np.arange(9),
            normals=normals,
            uvs=uvs,
            color=(0.2, 0.4, 0.6),
            roughness=0.3,
            metallic=0.7,
            opacity=0.8,
            texture="wrapper.png",
            texture_transform=((2, 0, 0.1), (0, 3, 0.2)),
            compute_inertia=False,
        )
        parts = mesh_components(mesh)
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0].vertices), 6)
        for part, indices in zip(parts, (np.arange(6), np.arange(6, 9)), strict=True):
            np.testing.assert_array_equal(part.vertices, vertices[indices])
            np.testing.assert_array_equal(part.uvs, uvs[indices])
            np.testing.assert_array_equal(part.normals, normals[indices])
            np.testing.assert_array_equal(part.color, mesh.color)
            np.testing.assert_array_equal(part.texture_transform, mesh.texture_transform)
            self.assertEqual(part.texture, mesh.texture)
            self.assertEqual(part.roughness, mesh.roughness)
            self.assertEqual(part.metallic, mesh.metallic)
            self.assertEqual(part.opacity, mesh.opacity)
        triangles = np.concatenate([part.vertices[part.indices] for part in parts])
        np.testing.assert_array_equal(triangles, mesh.vertices[mesh.indices])
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.uvs, uvs)

    @unittest.skipUnless(wp.is_cuda_available(), "Wrapper direct-contact validation requires CUDA")
    def test_wrapper_homes_and_holds_lift(self):
        """Finite drives park the arm and support the carriage under gravity."""
        with wp.ScopedDevice("cuda:0"):
            components = []
            for name, center, half in (
                ("steel", (0, 0, 3), (0.34, 0.34, 0.0585)),
                ("steel", (-0.95, -0.95, 1.7), (0.065, 0.065, 1.105)),
                ("wrapline_film_roll", (-1.05, -0.53, 0.927), (0.125, 0.125, 0.25)),
            ):
                mesh = newton.Mesh.create_box(*half, compute_inertia=False)
                components.append((name, mesh.copy(vertices=mesh.vertices + np.asarray(center))))
            builder = newton.ModelBuilder()
            bodies, joints, _ = build_wrapper(builder, components, label="/Wrapper", visuals=False)
            model = builder.finalize()
            self.assertTrue(np.all(model.body_mass.numpy() > 0.0))
            self.assertTrue(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()) > 0.0))
            solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=4)
            state = model.state()
            control = model.control()
            targets = control.joint_target_q.numpy()
            targets[model.joint_target_q_start.numpy()[joints[1]]] = 0.3
            control.joint_target_q.assign(targets)
            contacts = model._collision_pipeline.contacts()
            for _ in range(240):
                solver.collide(state, contacts)
                solver.step(state, state, control, contacts, 1.0 / 60.0)
            newton.eval_ik(model, state, model.joint_q, model.joint_qd)
            q = model.joint_q.numpy()
            self.assertAlmostEqual(float(q[model.joint_q_start.numpy()[joints[0]]]), np.pi / 4, delta=0.03)
            self.assertAlmostEqual(float(q[model.joint_q_start.numpy()[joints[1]]]), 0.3, delta=0.005)
            np.testing.assert_allclose(model.joint_qd.numpy(), 0.0, atol=0.02)
            self.assertEqual(len(bodies), 2)
            self.assertTrue(np.isfinite(state.body_q.numpy()).all())

            # Continuous targets must retain phase through two revolutions.
            # Position-only actuation ignores velocity feed-forward and trails
            # by kd * angular_speed / ke even after acceleration has settled.
            rotor_target = model.joint_target_q_start.numpy()[joints[0]]
            speeds = control.joint_target_qd.numpy()
            speeds[model.joint_qd_start.numpy()[joints[0]]] = np.pi / 2
            speeds[model.joint_qd_start.numpy()[joints[1]]] = 0.1
            control.joint_target_qd.assign(speeds)
            for frame in range(480):
                targets[rotor_target] = np.pi / 4 + (frame + 1) / 60 * np.pi / 2
                targets[model.joint_target_q_start.numpy()[joints[1]]] = 0.3 + (frame + 1) / 60 * 0.1
                control.joint_target_q.assign(targets)
                solver.collide(state, contacts)
                solver.step(state, state, control, contacts, 1.0 / 60.0)
            newton.eval_ik(model, state, model.joint_q, model.joint_qd)
            self.assertAlmostEqual(
                float(model.joint_q.numpy()[model.joint_q_start.numpy()[joints[1]]]), 1.1, delta=0.005
            )
            angle = float(model.joint_q.numpy()[model.joint_q_start.numpy()[joints[0]]])
            error = angle - targets[rotor_target]
            self.assertAlmostEqual(float(np.arctan2(np.sin(error), np.cos(error))), 0.0, delta=0.05)
            self.assertAlmostEqual(
                float(model.joint_qd.numpy()[model.joint_qd_start.numpy()[joints[0]]]), np.pi / 2, delta=0.03
            )

    def test_disabled_cube_belt_can_start(self):
        """Retain a stopped USD cube conveyor's runtime surface-velocity support."""
        from pxr import Sdf, Usd

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            path.write_text("""#usda 1.0
(defaultPrim = "Plant"; metersPerUnit = 1; upAxis = "Z")
def Xform "Plant" {
 def Xform "Belt" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
  bool physics:kinematicEnabled = true
  custom bool physxSurfaceVelocity:surfaceVelocityEnabled = false
  custom bool physxSurfaceVelocity:surfaceVelocityLocalSpace = true
  custom vector3f physxSurfaceVelocity:surfaceVelocity = (0.2, 0, 0)
  double3 xformOp:scale = (2, 1, 1)
  uniform token[] xformOpOrder = ["xformOp:scale"]
  def Cube "Strand" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
   double size = 2
   double3 xformOp:scale = (2, 0.025, 0.01)
   uniform token[] xformOpOrder = ["xformOp:scale"]
  }
 }
}
""")
            scene = load_plant(root, visuals=False, motor_off=True)
            self.assertEqual(len(scene.belts), 1)
            mesh, initial_velocity, label = scene.belts[0]
            self.assertEqual(label, "/Plant/Belt")
            np.testing.assert_array_equal(initial_velocity, 0.0)
            shape = scene.builder.shape_source.index(mesh)
            body = scene.builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity()))
            scene.builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
            model = scene.builder.finalize()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state = model.state()
            for speed in (0.0, 0.2, 0.0):
                velocity = np.zeros_like(initial_velocity)
                velocity[:, 0] = speed / scene.builder.shape_scale[shape][0]
                mesh.mesh.velocities.assign(velocity)
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                first = contacts.rigid_contact_shape0.numpy()[:count] == shape
                signs = np.where(first, 1.0, -1.0)
                np.testing.assert_allclose(
                    signs * contacts.rigid_contact_surface_velocity.numpy()[:count, 0], -speed, atol=1e-6
                )
            # Starting enabled must also honor authored linear speed and the
            # schema's degrees/second angular speed, rather than zeroing them.
            stage = Usd.Stage.Open(str(path))
            belt = stage.GetPrimAtPath("/Plant/Belt")
            belt.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled").Set(True)
            belt.CreateAttribute("physxSurfaceVelocity:surfaceAngularVelocity", Sdf.ValueTypeNames.Vector3f).Set(
                (0.0, 0.0, 90.0)
            )
            stage.GetRootLayer().Save()
            vendor = root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
            for module, data in (("ifm_v01", '{"belts":[],"speeds_mps":{}}'), ("ctp_v01", '{"modules":{}}')):
                folder = vendor / module
                folder.mkdir(parents=True)
                (folder / "interface.json").write_text(data)
            enabled_scene = load_plant(root, visuals=False)
            moving, velocity, _ = enabled_scene.belts[0]
            scale = np.asarray(enabled_scene.builder.shape_scale[0])
            expected = np.array([0.4, 0.0, 0.0]) + np.cross(
                np.array([0.0, 0.0, np.pi / 2]), np.asarray(moving.vertices) * scale
            )
            np.testing.assert_allclose(velocity * scale, expected, atol=1e-7)

            # Wrapper sidecars provide operating speeds even though their USD
            # layers author both the enable flag and the velocity as zero.
            belt.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled").Set(False)
            belt.GetAttribute("physxSurfaceVelocity:surfaceVelocity").Set((0.0, 0.0, 0.0))
            belt.GetAttribute("physxSurfaceVelocity:surfaceAngularVelocity").Set((0.0, 0.0, 0.0))
            stage.GetRootLayer().Save()
            sidecar = root / "plant/lines_1x8/wrapline/WL_U.json"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text(
                json.dumps(
                    {
                        "zones": {
                            "Z1": {
                                "prim": "/Plant/Belt",
                                "local_axis": [1, 0, 0],
                                "speed_mps": 0.2,
                            }
                        }
                    }
                )
            )
            wrapper_scene = load_plant(root, visuals=False)
            wrapper_mesh, nominal, wrapper_label = wrapper_scene.belts[0]
            self.assertEqual(
                wrapper_scene.metadata["wrapper_belt_zones"][wrapper_label],
                {
                    "line": "WL_U",
                    "zone": "Z1",
                    "speed_mps": 0.2,
                },
            )
            wrapper_shape = wrapper_scene.builder.shape_source.index(wrapper_mesh)
            load = wrapper_scene.builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity()))
            wrapper_scene.builder.add_shape_box(load, hx=0.01, hy=0.01, hz=0.01)
            wrapper_model = wrapper_scene.builder.finalize()
            np.testing.assert_array_equal(wrapper_mesh.mesh.velocities.numpy(), 0.0)
            physical_velocity = nominal * np.asarray(wrapper_scene.builder.shape_scale[0])
            np.testing.assert_allclose(physical_velocity[:, 0], 0.2, atol=1e-7)
            np.testing.assert_array_equal(physical_velocity[:, 1:], 0.0)
            descriptor = _Belt()
            descriptor.velocity = wrapper_mesh.mesh.velocities
            descriptor.nominal = wp.array(nominal, dtype=wp.vec3)
            descriptor.channel = 1
            descriptors = wp.array([descriptor], dtype=_Belt)
            multiplier = wp.array([1.0, 0.0], dtype=float)
            wrapper_pipeline = newton.CollisionPipeline(wrapper_model)
            wrapper_contacts = wrapper_pipeline.contacts()
            wrapper_state = wrapper_model.state()
            for enabled in (0.0, 1.0, 0.0):
                multiplier.assign(np.array([1.0, enabled], dtype=np.float32))
                wp.launch(_belt_speeds, dim=(1, len(nominal)), inputs=[descriptors, multiplier])
                wrapper_pipeline.collide(wrapper_state, wrapper_contacts)
                count = int(wrapper_contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                first = wrapper_contacts.rigid_contact_shape0.numpy()[:count] == wrapper_shape
                signs = np.where(first, 1.0, -1.0)
                np.testing.assert_allclose(
                    signs * wrapper_contacts.rigid_contact_surface_velocity.numpy()[:count, 0],
                    -0.2 * enabled,
                    atol=1e-6,
                )
            stopped_scene = load_plant(root, visuals=False, motor_off=True)
            np.testing.assert_array_equal(stopped_scene.belts[0][1], 0.0)

    def test_optix_capacity_covers_the_plant(self):
        """Size OptiX instances for the plant's shapes in both modes and refuse a capacity that is too small."""
        from pxr import Usd, UsdGeom, UsdPhysics

        # Shapes the full plant builds with visuals (example log, 2026-11-23).
        logged = {False: 276_775, True: 592_302}
        parser = Example.create_parser
        for filling, shapes in logged.items():
            flag = ["--filling-room-physics"] if filling else []
            self.assertGreaterEqual(parser().parse_args(flag).optix_max_instances, 1.2 * shapes)
        # A capacity the caller chooses stays, in either order.
        for argv in (
            ["--filling-room-physics", "--optix-max-instances", "123"],
            ["--optix-max-instances", "123", "--filling-room-physics"],
        ):
            self.assertEqual(parser().parse_args(argv).optix_max_instances, 123)

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            for label in ("/Plant/Line/A", "/Plant/Line/B"):
                cube = UsdGeom.Cube.Define(stage, label)
                cube.CreateSizeAttr(0.2)
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            stage.GetRootLayer().Save()
            argv = ["--asset-root", str(root), "--viewer", "optix", "--no-controllers", "--motor-off"]
            with self.assertRaisesRegex(ValueError, "shapes, more than --optix-max-instances"):
                Example(ViewerNull(), parser().parse_args([*argv, "--optix-max-instances", "1"]))

    def test_authored_collision_groups(self):
        """Keep vehicle obstacles while honoring the line's USD exclusions."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            paths = ["/Plant/Line/Item", "/Plant/Line/Allowed", "/Plant/Fleet/Truck", "/Plant/Keepouts/Box"]
            for index, label in enumerate(paths):
                cube = UsdGeom.Cube.Define(stage, label)
                cube.CreateSizeAttr(0.2)
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                if index < 3:
                    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
            line = UsdPhysics.CollisionGroup.Define(stage, "/Plant/Groups/Line")
            line.GetCollidersCollectionAPI().GetIncludesRel().SetTargets(["/Plant/Line"])
            line.GetCollidersCollectionAPI().GetExcludesRel().SetTargets([paths[1]])
            obstacles = UsdPhysics.CollisionGroup.Define(stage, "/Plant/Groups/Obstacles")
            obstacles.GetCollidersCollectionAPI().GetIncludesRel().SetTargets(["/Plant/Keepouts"])
            obstacles.CreateFilteredGroupsRel().SetTargets([line.GetPath()])
            stage.GetRootLayer().Save()
            args = Example.create_parser().parse_args(
                [
                    "--asset-root",
                    str(root),
                    "--viewer",
                    "null",
                    "--no-controllers",
                    "--motor-off",
                ]
            )
            example = Example(ViewerNull(), args)
            example.pipeline.collide(example.state_0, example.contacts, dt=1.0 / 240.0)
            count = int(example.contacts.rigid_contact_count.numpy()[0])
            first = example.contacts.rigid_contact_shape0.numpy()[:count]
            second = example.contacts.rigid_contact_shape1.numpy()[:count]
            labels = example.model.shape_label
            pairs = {frozenset((labels[a], labels[b])) for a, b in zip(first, second, strict=True)}
            paths = [path + "/collision" for path in paths]
            self.assertNotIn(frozenset((paths[0], paths[3])), pairs)
            self.assertIn(frozenset((paths[1], paths[3])), pairs)
            self.assertIn(frozenset((paths[2], paths[3])), pairs)
            self.assertIn(frozenset((paths[0], paths[2])), pairs)

    def test_valves_hold_stroke_limits_under_load(self):
        """Hold both cylinder stops under opposing loads with the source valve law."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(
                -1,
                body,
                axis=newton.Axis.X,
                limit_lower=0.0,
                limit_upper=0.05,
                target_kd=100.0,
                effort_limit=10.0,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
            builder.add_articulation([joint])
        model = builder.finalize()
        state, control = model.state(), model.control()
        solver = newton.solvers.SolverPhoenX(
            model, joint_mode="maximal_pgs", step_layout="single_world", solver_iterations=4
        )
        wp.launch(
            _valves,
            dim=(1, 2),
            inputs=[
                wp.array([[0, 1]], dtype=int),
                wp.array([[0.05, 0.0]], dtype=float),
                wp.array([0.2, 0.2], dtype=float),
                control.joint_target_qd,
            ],
        )
        state.body_f.assign(np.array([[-2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], dtype=np.float32))
        for _ in range(240):
            solver.step(state, state, control, None, 1.0 / 240.0)
        np.testing.assert_allclose(state.joint_q.numpy(), (0.05, 0.0), atol=1e-5)

    def test_grasp_ray_respects_physical_reach(self):
        """Acquisition requires a real surface hit within the authored 6 mm."""
        mesh = newton.Mesh.create_box(0.03, 0.03, 0.115, compute_inertia=False)
        mesh_id = mesh.finalize()
        # Exercise partial warps, several bottle batches, and products
        # assigned to different ray lanes.
        product_count, sensor_count = 37 + _PHOTOEYE_BATCH_SIZE, 77
        origins = np.asarray(
            [(0.1 * ((i * 7) % product_count), 0.0, 0.2305) for i in range(sensor_count)], dtype=np.float32
        )
        origins[1::3, 2] = 0.24
        origins[2::3, 0] += 0.035
        hits = wp.zeros((sensor_count // 7, 7), dtype=float)
        wp.launch(
            _photoeyes,
            dim=(sensor_count, (product_count + _PHOTOEYE_BATCH_SIZE - 1) // _PHOTOEYE_BATCH_SIZE, 32),
            block_dim=32,
            inputs=[
                wp.array(
                    [wp.transform(wp.vec3(0.1 * i, 0.0, 0.0), wp.quat_identity()) for i in range(product_count)],
                    dtype=wp.transform,
                ),
                wp.array(np.arange(product_count), dtype=int),
                wp.array([product_count], dtype=int),
                wp.array(origins, dtype=wp.vec3),
                wp.array([(0.0, 0.0, -1.0)] * sensor_count, dtype=wp.vec3),
                wp.full(sensor_count, 0.006, dtype=float),
                wp.array([mesh_id], dtype=wp.uint64),
                wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.115), wp.quat_identity())], dtype=wp.transform),
                wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3),
                7,
                hits,
            ],
        )
        np.testing.assert_array_equal(hits.numpy().reshape(-1), np.arange(sensor_count) % 3 == 0)

    def test_case_photoeyes_partition_shape_batches(self):
        """Preserve geometry hits across batches, duplicate hits, and partial warps."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                count = 517
                mesh = newton.Mesh.create_box(0.03, 0.03, 0.115, compute_inertia=False)
                mesh_id = mesh.finalize()
                centers = [0.2 * i for i in range(count)]
                centers[257] = centers[513] = centers[1]
                origins = np.asarray([(centers[i], 0.0, 0.2) for i in (0, 1, 255, 256, 511, 512, 10, 11)])
                origins[6, 1] = 0.08
                hits = wp.zeros(8, dtype=int)
                inputs = [
                    wp.array([wp.transform((x, 0.0, 0.0), wp.quat_identity()) for x in centers], dtype=wp.transform),
                    wp.array(np.arange(count), dtype=int),
                    wp.array([wp.transform_identity()] * count, dtype=wp.transform),
                    wp.array([(1.0, 1.0, 1.0)] * count, dtype=wp.vec3),
                    wp.array(
                        [int(newton.GeoType.BOX if i % 2 == 0 else newton.GeoType.MESH) for i in range(count)],
                        dtype=int,
                    ),
                    wp.full(count, mesh_id, dtype=wp.uint64),
                    wp.array(np.arange(count), dtype=int),
                    wp.array([(-0.03, -0.03, -0.115)] * count, dtype=wp.vec3),
                    wp.array([(0.03, 0.03, 0.115)] * count, dtype=wp.vec3),
                    wp.array(origins, dtype=wp.vec3),
                    wp.array([(0.0, 0.0, -1.0)] * 8, dtype=wp.vec3),
                    wp.array([0.1] * 7 + [0.01], dtype=float),
                    hits,
                ]
                batches = (count + _CASE_PHOTOEYE_BATCH_SIZE - 1) // _CASE_PHOTOEYE_BATCH_SIZE
                wp.launch(_case_photoeyes, dim=(8, batches, 32), block_dim=32, inputs=inputs)
                np.testing.assert_array_equal(hits.numpy(), [1, 1, 1, 1, 1, 1, 0, 0])
                if wp.get_device().is_cuda:
                    with wp.ScopedCapture() as capture:
                        hits.zero_()
                        wp.launch(_case_photoeyes, dim=(8, batches, 32), block_dim=32, inputs=inputs)
                    for _ in range(3):
                        wp.capture_launch(capture.graph)
                        np.testing.assert_array_equal(hits.numpy(), [1, 1, 1, 1, 1, 1, 0, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "Runtime sealing requires CUDA")
    def test_tape_bonds_require_closed_flaps(self):
        """Open flaps stay free; closed, taped flaps resist hinge torque."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        root = builder.add_link(mass=0.2, inertia=wp.mat33(np.eye(3) * 0.01))
        joints = [builder.add_joint_fixed(-1, root)]
        for i, point in enumerate(((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0))):
            rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2) if i == 0 else wp.quat_identity()
            flap = builder.add_link(
                xform=wp.transform(wp.vec3(point), rotation), mass=0.015, inertia=wp.mat33(np.eye(3) * 0.001)
            )
            joint = builder.add_joint_revolute(
                root, flap, axis=newton.Axis.X, parent_xform=wp.transform(wp.vec3(point), wp.quat_identity())
            )
            joints.append(joint)
            if i == 0:
                builder.joint_q[builder.joint_q_start[joint]] = np.pi / 2
        builder.add_articulation(joints)
        builder.reserve_dynamic_capacity(joints=4)
        model = builder.finalize()
        exact = np.zeros(model.joint_count, dtype=bool)
        exact[joints[0]] = True
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="maximal_pgs",
            step_layout="single_world",
            direct_joint_mask=exact,
            substeps=4,
            solver_iterations=4,
        )
        state = model.state()
        bodies = wp.array([[0, 1, 2, 3, 4]], dtype=int)
        requests, sealed = wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)
        attachments = wp.full((1, 4), -1, dtype=int)
        inputs = [
            solver.dynamics.data,
            state.body_q,
            state.body_qd,
            bodies,
            requests,
            sealed,
            attachments,
            wp.array([joints[1:]], dtype=int),
            model.joint_enabled,
        ]
        wp.launch(_seal_cases, dim=1, inputs=inputs)
        self.assertEqual(int(sealed.numpy()[0]), 0)
        np.testing.assert_array_equal(attachments.numpy(), -1)
        poses = state.body_q.numpy()
        poses[1, 3:] = (0.0, 0.0, 0.0, 1.0)
        state.body_q.assign(poses)
        wp.launch(_seal_cases, dim=1, inputs=inputs)
        self.assertEqual(int(sealed.numpy()[0]), 1)
        self.assertTrue((attachments.numpy() >= 0).all())
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[1, 3] = 0.01
        for _ in range(20):
            state.body_f.assign(force)
            solver.step(state, state, model.control(), None, 1.0 / 60.0)
        np.testing.assert_allclose(state.body_q.numpy()[1, 3:], (0.0, 0.0, 0.0, 1.0), atol=1e-3)
        self.assertEqual(int(solver.dynamics.error_count.numpy()[0]), 0)

    def test_scaled_belt_preserves_metres_per_second(self):
        """Preserve a moving belt's authored hull and its scaled 0.2 m/s speed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            path.write_text("""#usda 1.0
(defaultPrim = "Plant"; metersPerUnit = 1; upAxis = "Z")
def Xform "Plant" {
 def Material "Acetal" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {
  float physics:restitution = 0
  custom token physxMaterial:restitutionCombineMode = "min"
 }
 def Xform "Line" {
  def Scope "Packers" {
   def Xform "PU1" {
    def Xform "Inlet" {
     def Scope "Belts" {
      def Xform "IN_A" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
       bool physics:kinematicEnabled = true
       custom bool physxSurfaceVelocity:surfaceVelocityEnabled = true
       def Mesh "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]) {
        rel material:binding:physics = </Plant/Acetal>
        uniform token physics:approximation = "convexHull"
        point3f[] points = [(-0.5,-0.5,-0.01), (0.5,-0.5,-0.01), (0.5,0.5,-0.01), (-0.5,0.5,-0.01),
                           (-0.5,-0.5,0.01), (0.5,-0.5,0.01), (0.5,0.5,0.01), (-0.5,0.5,0.01)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0,1,2,3]
        double3 xformOp:scale = (4,0.05,1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
       }
      }
     }
    }
   }
  }
 }
}
""")
            vendor = root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
            for module, data in (("ifm_v01", '{"belts":[],"speeds_mps":{}}'), ("ctp_v01", '{"modules":{}}')):
                folder = vendor / module
                folder.mkdir(parents=True)
                (folder / "interface.json").write_text(data)
            scene = load_plant(root, cutaway=False, visuals=False)
            self.assertEqual(len(scene.belts), 1)
            mesh, velocities, _ = scene.belts[0]
            shape = scene.builder.shape_source.index(mesh)
            np.testing.assert_allclose(
                velocities.mean(axis=0) * scene.builder.shape_scale[shape], (0.2, 0.0, 0.0), atol=1e-7
            )
            # Convex cooking uses all points, even when source triangles only
            # describe the bottom. Surface velocity must not bypass cooking.
            body = scene.builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity()))
            scene.builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
            model = scene.builder.finalize()
            solver = newton.solvers.SolverPhoenX(model)
            material_modes = solver.world._materials.numpy()["restitution_combine_mode"]
            self.assertEqual(material_modes[shape], 1, "the belt's minimum restitution combine mode")
            self.assertEqual(material_modes[-1], 0, "the unbound box's default average combine mode")
            mesh.mesh.velocities.assign(velocities)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0, "the authored convex belt must have a top support face")
            belt_first = contacts.rigid_contact_shape0.numpy()[:count] == shape
            signs = np.where(belt_first, 1.0, -1.0)
            self.assertTrue(np.all(signs * contacts.rigid_contact_normal.numpy()[:count, 2] > 0.99))
            belt_points = np.where(
                belt_first[:, None],
                contacts.rigid_contact_point0.numpy()[:count],
                contacts.rigid_contact_point1.numpy()[:count],
            )
            np.testing.assert_allclose(belt_points[:, 2], 0.01, atol=1e-5)
            np.testing.assert_allclose(
                signs * contacts.rigid_contact_surface_velocity.numpy()[:count, 0], -0.2, atol=1e-6
            )

    def test_position_drive_after_free_root(self):
        """A case clamp still receives its position command after a free root."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        free = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_free(free)])
        slider = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_prismatic(
            -1,
            slider,
            axis=newton.Axis.X,
            target_ke=1000.0,
            target_kd=50.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        dof = int(model.joint_qd_start.numpy()[joint])
        coordinate = int(_target_coordinates(model)[dof])
        self.assertNotEqual(dof, coordinate)
        control, state = model.control(), model.state()
        wp.launch(
            _case_drives,
            dim=1,
            inputs=[
                wp.array([dof], dtype=int),
                wp.array([coordinate], dtype=int),
                wp.array([0.1], dtype=float),
                wp.zeros(1, dtype=float),
                control.joint_target_q,
                control.joint_target_qd,
            ],
        )
        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="maximal_pgs",
            step_layout="single_world",
            substeps=2,
            solver_iterations=2,
        )
        for _ in range(30):
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 60.0)
        self.assertAlmostEqual(float(state.body_q.numpy()[slider, 0]), 0.1, delta=1e-3)
        np.testing.assert_allclose(state.body_q.numpy()[free], model.body_q.numpy()[free], atol=1e-6)

    def test_composed_geometry_and_physics(self):
        """Preserve inertial frames, static joint anchors, and point instancing."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            layer = stage.GetRootLayer()
            layer.ImportFromString("""#usda 1.0
(
    defaultPrim = "Plant"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Plant" {
    def Xform "Machine" {
        double3 xformOp:translate = (4, 2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "Mount" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            float physics:mass = 20
            float3 physics:diagonalInertia = (1, 2, 3)
            point3f physics:centerOfMass = (0, 0, 0.1)
            def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
                double size = 0.4
            }
        }
        def Xform "Arm" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
            double3 xformOp:translate = (1, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            float physics:mass = 2
            float3 physics:diagonalInertia = (0.1, 0.2, 0.3)
            def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
                double size = 0.2
            }
        }
        def Scope "Joints" {
        def PhysicsFixedJoint "Anchor" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
            custom bool physxArticulation:enabledSelfCollisions = false
            rel physics:body0 = </Plant/Machine>
            rel physics:body1 = </Plant/Machine/Mount>
            point3f physics:localPos0 = (0, 0, 1)
        }
        }
        def PhysicsRevoluteJoint "Hinge" (prepend apiSchemas = ["PhysicsDriveAPI:angular"]) {
            rel physics:body0 = </Plant/Machine/Mount>
            rel physics:body1 = </Plant/Machine/Arm>
            point3f physics:localPos0 = (1, 0, 0)
            uniform token physics:axis = "Z"
            float physics:lowerLimit = -45
            float physics:upperLimit = 45
            float drive:angular:physics:stiffness = 2
            float drive:angular:physics:targetPosition = 10
        }
    }
    def Xform "Flap" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 0.015
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 1
            double3 xformOp:scale = (0.2, 0.08, 0.0015)
            double3 xformOp:translate = (0.1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
    def PointInstancer "Copies" {
        rel prototypes = </Plant/Copies/Prototype>
        int[] protoIndices = [0, 0]
        point3f[] positions = [(10, 0, 0), (20, 0, 0)]
        def Xform "Prototype" {
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Mesh "Triangle" {
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                uniform token subdivisionScheme = "none"
                color3f[] primvars:displayColor = [(0.2, 0.4, 0.6)]
            }
            def Mesh "Guide" {
                point3f[] points = [(0, 0, 100), (1, 0, 100), (0, 1, 100)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                uniform token purpose = "guide"
            }
        }
    }
    def Scope "Sim" {
        def Scope "Product" {
            def Scope "Bottles" {
                def Xform "Parked" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
                    double3 xformOp:translate = (0, 0, -3)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
                }
            }
        }
    }
}
""")
            blade = UsdGeom.Xform.Define(stage, "/Plant/OffsetBlade").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(blade)
            UsdPhysics.MassAPI.Apply(blade).CreateMassAttr(0.35)
            collider = UsdGeom.Mesh.Define(stage, "/Plant/OffsetBlade/Collider")
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            half = np.asarray((0.003, 0.06, 0.008))
            corners = np.asarray(
                [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            )
            collider.CreatePointsAttr((corners * half + (5.0, 0.1, 1.0)).tolist())
            collider.CreateFaceVertexCountsAttr([3] * 12)
            collider.CreateFaceVertexIndicesAttr(
                [
                    0,
                    2,
                    1,
                    0,
                    3,
                    2,
                    4,
                    5,
                    6,
                    4,
                    6,
                    7,
                    0,
                    1,
                    5,
                    0,
                    5,
                    4,
                    1,
                    2,
                    6,
                    1,
                    6,
                    5,
                    2,
                    3,
                    7,
                    2,
                    7,
                    6,
                    3,
                    0,
                    4,
                    3,
                    4,
                    7,
                ]
            )
            layer.Save()
            scene = load_plant(root, cutaway=False, motor_off=True)
            builder = scene.builder
            self.assertEqual(len(scene.bodies), 4)
            mount = scene.bodies["/Plant/Machine/Mount"]
            np.testing.assert_allclose(builder.body_q[mount].p, (4, 2, 1))
            self.assertEqual(builder.body_mass[mount], 20.0)
            np.testing.assert_allclose(np.asarray(builder.body_inertia[mount]).reshape(3, 3), np.diag((1, 2, 3)))
            np.testing.assert_allclose(builder.body_com[mount], (0, 0, 0.1), atol=1e-7)
            self.assertIn("/Plant/Machine", scene.self_collision_scopes)
            anchor = scene.joints["/Plant/Machine/Joints/Anchor"]
            np.testing.assert_allclose(builder.joint_X_p[anchor].p, (4, 2, 1))
            hinge = scene.joints["/Plant/Machine/Hinge"]
            dof = builder.joint_qd_start[hinge]
            self.assertAlmostEqual(builder.joint_limit_lower[dof], -np.pi / 4)
            self.assertAlmostEqual(builder.joint_target_q[dof], np.pi / 18)
            flap = scene.bodies["/Plant/Flap"]
            self.assertAlmostEqual(builder.body_mass[flap], 0.015, places=7)
            np.testing.assert_allclose(builder.body_com[flap], (0.1, 0, 0), atol=1e-7)
            dims = np.asarray((0.2, 0.08, 0.0015)) ** 2
            expected = 0.015 / 12.0 * (dims.sum() - dims)
            np.testing.assert_allclose(
                np.diag(np.asarray(builder.body_inertia[flap]).reshape(3, 3)), expected, rtol=1e-6
            )
            blade = scene.bodies["/Plant/OffsetBlade"]
            tensor = np.asarray(builder.body_inertia[blade]).reshape(3, 3)
            expected = 0.35 / 3.0 * ((half**2).sum() - half**2)
            np.testing.assert_allclose(np.diag(tensor), expected, rtol=2e-4)
            self.assertGreater(float(np.linalg.eigvalsh(tensor).min()), 0.0)
            np.testing.assert_allclose(builder.body_com[blade], (5.0, 0.1, 1.0), atol=1e-6)
            copies = [i for i, name in enumerate(builder.shape_label) if name == "/Plant/Copies/instance"]
            self.assertEqual(len(copies), 2)
            self.assertIs(builder.shape_source[copies[0]], builder.shape_source[copies[1]])
            positions = []
            for shape in copies:
                mesh = builder.shape_source[shape]
                self.assertEqual(len(mesh.vertices), 3)
                positions.append(np.min(mesh.vertices[:, 0]) + builder.shape_transform[shape].p[0])
            np.testing.assert_allclose(positions, (11, 21))


if __name__ == "__main__":
    unittest.main()
