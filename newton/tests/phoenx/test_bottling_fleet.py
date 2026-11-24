# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check physical inventory and placement gates without private factory assets."""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_control import PlantController
from newton.examples.phoenx._bottling_fleet import (
    FleetController,
    _observe_port,
    _placement_valid,
    _Transfer,
    put_away,
    shelf_slot,
)
from newton.examples.phoenx._bottling_transport import TransportController


class TestBottlingFleet(unittest.TestCase):
    def test_wrapper_dispatch_requires_ready_inventory_and_free_resources(self):
        """Gate route C on process readiness, physical stock and exclusive ports."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            pallet = builder.add_body(xform=wp.transform(wp.vec3(4, 7, 0.5), wp.quat_identity()))
            obstruction = builder.add_body(xform=wp.transform(wp.vec3(10, 7, 0.2), wp.quat_identity()))
            model = builder.finalize()
            fleet = FleetController.__new__(FleetController)
            fleet.state = model.state()
            fleet.pallet_bodies = np.array([pallet], dtype=np.int32)
            fleet.transport = SimpleNamespace(case_roots=np.array([obstruction]), names=(), station_state=())
            fleet.navigation = SimpleNamespace(
                ports={
                    "WL_U_pick": SimpleNamespace(xy=(4, 7), z=0.5, yaw=90.0, side="U"),
                    "dropoff_U_drop": SimpleNamespace(xy=(10, 7), z=0.0, yaw=90.0),
                }
            )
            fleet.roles = {"c_U": {("C", "U")}, "ab_U": {("A", "U")}}
            fleet.missions = {}
            fleet.wrapper_ready = {}
            self.assertEqual(list(fleet._transfer_requests("c_U")), [])
            fleet.notify_wrapper_ready("U", pallet)
            fleet.notify_wrapper_ready("U", pallet)
            requests = list(fleet._transfer_requests("c_U"))
            self.assertEqual(requests, [("WL_U_pick", "dropoff_U_drop", 0, False)])
            self.assertEqual(list(fleet._transfer_requests("ab_U")), [])
            source, destination, identity, empty = requests[0]
            poses = fleet.state.body_q.numpy()
            self.assertIsNone(fleet._available_transfer(source, destination, poses, pallet=identity, empty=empty))
            poses[obstruction, 0] += 3.0
            fleet.state.body_q.assign(poses)
            mission = fleet._available_transfer(source, destination, poses, pallet=identity, empty=empty)
            self.assertEqual((mission.source, mission.destination, mission.pallet), (source, destination, 0))
            self.assertIsNone(fleet._available_transfer(source, destination, poses, pallet=1))
            for reserved in (
                mission,
                _Transfer("another", destination, 1),
                _Transfer(source, "another", 1),
                _Transfer(destination, "another", 1),
                _Transfer("another", source, 1),
                _Transfer("another", "elsewhere", 0),
            ):
                with self.subTest(reserved=reserved):
                    fleet.missions["other"] = reserved
                    self.assertIsNone(fleet._available_transfer(source, destination, poses, pallet=identity))
                    if reserved.pallet == identity:
                        with self.assertRaises(ValueError):
                            fleet.notify_wrapper_ready("U", pallet)
            fleet.missions.clear()
            self.assertIsNotNone(fleet._available_transfer(source, destination, poses, pallet=identity))
            poses[pallet, 2] += 0.2
            fleet.state.body_q.assign(poses)
            self.assertIsNone(fleet._available_transfer(source, destination, poses, pallet=identity))
            with self.assertRaises(ValueError):
                fleet.notify_wrapper_ready("U", pallet)
            with self.assertRaises(ValueError):
                fleet.notify_wrapper_ready("U", obstruction)

    def test_station_waits_for_withdrawal_before_reset(self):
        """Hold an empty seated replacement until its exchange truck withdraws."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            pallet = builder.add_body(xform=wp.transform(wp.vec3(4, 7, 0), wp.quat_identity()))
            model = builder.finalize()
            plant = PlantController.__new__(PlantController)
            plant.state = model.state()
            fleet = FleetController.__new__(FleetController)
            fleet.missions = {"truck": _Transfer("EP_PZ1_pick", "PZ1_A", 0, phase="place")}
            plant.fleet = fleet
            transport = TransportController.__new__(TransportController)
            transport.plant = plant
            transport.names = ("PZ1",)
            transport.pallet_bodies = np.array([pallet], dtype=np.int32)
            transport.case_roots = np.empty(0, dtype=np.int32)
            transport.station_positions = np.array([[4, 7, 0]])
            transport.station_state = ["EMPTY"]
            transport.station_ready_since = [None]
            # Only counter/state data is needed; the actual observer and fleet
            # interlock run against a real Newton body's measured transform.
            counters = SimpleNamespace(slot=34, n_slots=34, state="WAIT_CASE")
            transport.palletizers = {"PZ1": counters}
            transport._observe_stations(0.0)
            transport._observe_stations(0.5)
            self.assertEqual(transport.station_state, ["EMPTY"])
            self.assertEqual(counters.slot, 34)
            fleet.missions.clear()
            transport._observe_stations(0.6)
            self.assertEqual(transport.station_state, ["EMPTY"])
            transport._observe_stations(0.9)
            self.assertEqual(transport.station_state, ["LOADING"])
            self.assertEqual(counters.slot, 0)

    def test_inventory_requires_a_seated_empty_pallet(self):
        """Distinguish usable empty stock from raised, loaded and ambiguous pallets."""
        pallets = np.array([[4.0, 7.0, 0.5, 0, 0, 0, 1]], dtype=float)
        cases = np.empty((0, 3))
        observation = _observe_port(pallets, cases, (4, 7), 0.5, math.pi / 2)
        self.assertEqual(observation.pallet, 0)
        self.assertTrue(observation.empty)
        self.assertFalse(observation.clear)
        loaded = _observe_port(pallets, np.array([[4, 7, 0.8]]), (4, 7), 0.5, math.pi / 2)
        self.assertEqual(loaded.pallet, 0)
        self.assertFalse(loaded.empty)
        self.assertFalse(loaded.clear)
        pallets[0, 2] += 0.2
        raised = _observe_port(pallets, cases, (4, 7), 0.5, math.pi / 2)
        self.assertIsNone(raised.pallet)
        self.assertFalse(raised.empty)
        self.assertFalse(raised.clear)
        pallets[0, 2] = 0.5
        duplicate = _observe_port(np.repeat(pallets, 2, axis=0), cases, (4, 7), 0.5, math.pi / 2)
        self.assertIsNone(duplicate.pallet)
        self.assertFalse(duplicate.empty)

    def test_landing_space_blocks_loose_cases_and_misaligned_pallets(self):
        """Keep the landing blocked until obstructing geometry actually departs."""
        cases = np.array([[4.5, 7, 0.7]])
        absent = np.empty((0, 7))
        self.assertFalse(_observe_port(absent, cases, (4, 7), 0.5, math.pi / 2).clear)
        cases[0, 0] += 2.0
        self.assertTrue(_observe_port(absent, cases, (4, 7), 0.5, math.pi / 2).clear)
        pallets = np.array([[4, 7, 0.5, 0, 0, math.sin(0.1), math.cos(0.1)]])
        rotated = _observe_port(pallets, cases, (4, 7), 0.5, math.pi / 2)
        self.assertIsNone(rotated.pallet)
        self.assertFalse(rotated.clear)
        pallets[0, 3:] = [math.sin(0.1), 0, 0, math.cos(0.1)]
        tilted = _observe_port(pallets, cases, (4, 7), 0.5, math.pi / 2)
        self.assertIsNone(tilted.pallet)
        self.assertFalse(tilted.clear)
        # Pallet pockets are equivalent after a half-turn.
        pallets[0, 3:] = [0, 0, 1, 0]
        self.assertEqual(_observe_port(pallets, cases, (4, 7), 0.5, math.pi / 2).pallet, 0)

    def test_placement_rejects_unseated_or_tilted_load(self):
        """Reject failure in each measured placement dimension after withdrawal."""
        error = {"xy_m": 0.002, "yaw_deg": 0.1, "z_m": 0.001, "tilt_deg": 0.2}
        self.assertTrue(_placement_valid(error))
        for field, value in (("xy_m", 0.031), ("yaw_deg", -3.1), ("z_m", 0.04), ("tilt_deg", 3.1)):
            with self.subTest(field=field):
                self.assertFalse(_placement_valid({**error, field: value}))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_drop_off_is_put_away_after_the_source_delay(self):
        """Clear a drop-off 120 s after a wrapped pallet lands, carrying its whole load to the rack shelf at rest."""
        written = {}
        fleet = SimpleNamespace(
            putaways=[(130.0, "dropoff_U_drop")],
            shelved=0,
            navigation=SimpleNamespace(ports={"dropoff_U_drop": SimpleNamespace(xy=(90.0, 44.0), z=0.0)}),
            link=SimpleNamespace(write=lambda array, values: written.__setitem__(id(array), np.asarray(values))),
            putaway_active=wp.zeros(1, dtype=int, device="cpu"),
            putaway_drop=wp.zeros(1, dtype=wp.vec3, device="cpu"),
            putaway_delta=wp.zeros(1, dtype=wp.vec3, device="cpu"),
        )
        FleetController._put_away(fleet, 129.9)
        self.assertEqual(written[id(fleet.putaway_active)][0], 0)
        FleetController._put_away(fleet, 130.0)
        self.assertEqual(written[id(fleet.putaway_active)][0], 1)
        self.assertEqual((fleet.shelved, fleet.putaways), (1, []))
        delta = written[id(fleet.putaway_delta)][0]
        np.testing.assert_allclose(np.array((90.0, 44.0, 0.0)) + delta, shelf_slot(0), atol=1e-5)
        # Neighbouring shelf slots leave room for a 1.2 m pallet with its load.
        self.assertGreaterEqual(
            min(np.linalg.norm(shelf_slot(i + 1)[:2] - shelf_slot(i)[:2]) for i in range(30)), 1.6 - 1e-6
        )

        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            # The pallet, a case and a bottle on the drop-off; a truck beside it; a fixed part over it.
            positions = [(90.0, 44.0, 0.08), (90.2, 44.3, 0.5), (90.25, 44.3, 0.6), (92.0, 44.0, 0.5)]
            bodies = [builder.add_body(xform=wp.transform(p, wp.quat_identity()), mass=1.0) for p in positions]
            fixed = builder.add_body(xform=wp.transform((90.0, 44.0, 1.0), wp.quat_identity()), mass=0.0)
            model = builder.finalize()
            state = model.state()
            state.body_qd.fill_(wp.spatial_vector(0.1, 0.0, 0.0, 0.0, 0.0, 0.2))
            active = wp.array([1], dtype=int)
            drop = wp.array([(90.0, 44.0, 0.0)], dtype=wp.vec3)
            shift = wp.array([delta], dtype=wp.vec3)
            with wp.ScopedCapture() as capture:
                wp.launch(
                    put_away, model.body_count, [active, drop, shift, model.body_inv_mass, state.body_q, state.body_qd]
                )
            wp.capture_launch(capture.graph)
            active.zero_()
            wp.capture_launch(capture.graph)
            q, qd = state.body_q.numpy(), state.body_qd.numpy()

        np.testing.assert_allclose(q[bodies[:3], :3], np.asarray(positions[:3]) + delta, atol=1e-4)
        np.testing.assert_array_equal(qd[bodies[:3]], 0.0)
        np.testing.assert_allclose(q[[bodies[3], fixed], :3], [positions[3], (90.0, 44.0, 1.0)], atol=1e-6)
        np.testing.assert_allclose(qd[bodies[3]], (0.1, 0.0, 0.0, 0.0, 0.0, 0.2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
