# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind the source transport PLC and KR40 sequencers to the physical plant."""

import copy
import json

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_io import (
    _CASE_PHOTOEYE_BATCH_SIZE,
    _case_drives,
    _case_photoeyes,
    _load_controller,
    _ray_box,
)
from newton.solvers import phoenx


@wp.kernel
def _arm_targets(
    offset: int,
    target_coordinates: wp.array[int],
    arm_coordinates: wp.array[int],
    joint_q: wp.array[float],
    positions: wp.array[float],
    integral: wp.array[float],
    target_q: wp.array[float],
):
    """Add the palletizer arms' integral tracking correction from the current joint positions."""
    i = wp.tid()
    target = positions[offset + i]
    actual = joint_q[arm_coordinates[i]]
    actual += 2.0 * wp.pi * wp.round((target - actual) / (2.0 * wp.pi))
    integral[i] += 4.0 * (target - actual) / 60.0
    target_q[target_coordinates[offset + i]] = target + integral[i]


@wp.kernel
def _vacuum(
    dynamics: phoenx.DynamicsData,
    poses: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_xform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_type: wp.array[int],
    shape_mesh: wp.array[wp.uint64],
    shapes: wp.array[int],
    bounds_lo: wp.array[wp.vec3],
    bounds_hi: wp.array[wp.vec3],
    shape_case: wp.array[int],
    case_bodies: wp.array2d[int],
    sealed: wp.array[int],
    grippers: wp.array[int],
    commands: wp.array[int],
    held: wp.array[int],
    joints: wp.array[int],
    claimed: wp.array[int],
):
    cell = wp.tid()
    parent = grippers[cell]
    if commands[cell] < 0 and held[cell] >= 0:
        phoenx.remove_joint(dynamics, joints[cell])
        claimed[held[cell]] = 0
        held[cell] = -1
        joints[cell] = -1
    elif commands[cell] > 0 and held[cell] < 0:
        candidate = int(-1)
        direction = wp.transform_vector(poses[parent], wp.vec3(0.0, 0.0, 1.0))
        for cup in range(4):
            point = wp.vec3(-0.09 if cup % 2 == 0 else 0.09, -0.06 if cup / 2 == 0 else 0.06, 0.18)
            point = wp.transform_point(poses[parent], point)
            for i in range(shapes.shape[0]):
                case = shape_case[i]
                body = case_bodies[case, 0]
                if sealed[case] == 0 or claimed[body] != 0:
                    continue
                shape = shapes[i]
                world = wp.transform_multiply(poses[shape_body[shape]], shape_xform[shape])
                inverse = wp.transform_inverse(world)
                origin = wp.transform_point(inverse, point)
                ray = wp.transform_vector(inverse, direction)
                hit = _ray_box(origin, ray, bounds_lo[i], bounds_hi[i], 0.012)
                if hit and shape_type[shape] != int(newton.GeoType.BOX):
                    query = wp.mesh_query_ray(
                        shape_mesh[shape],
                        wp.cw_div(origin, shape_scale[shape]),
                        wp.cw_div(ray, shape_scale[shape]),
                        0.012,
                    )
                    hit = query.result
                if hit:
                    # Sealed flaps are welded to this root, so the assembly is
                    # the same rigid payload as the source's sealed-case proxy.
                    candidate = body
                    break
            if candidate >= 0:
                break
        if candidate >= 0:
            frame = wp.transform_multiply(wp.transform_inverse(poses[parent]), poses[candidate])
            desc = phoenx.joint_desc(parent, candidate, frame, wp.transform_identity())
            desc.collide_parent = False
            joint = phoenx.add_joint(dynamics, desc)
            if joint >= 0:
                joints[cell] = joint
                held[cell] = candidate
                claimed[candidate] = parent + 1


class TransportController:
    """Use source zero-pressure accumulation, merge, and palletizer logic."""

    def __init__(self, plant, vendor):
        self.plant = plant
        scene, model = plant.scene, plant.model
        module = _load_controller("_newton_berlin_transport", vendor / "ctp_v01/controller.py")
        interface = copy.deepcopy(json.loads((vendor / "ctp_v01/interface.json").read_text()))
        prefix = "/Plant/Line/CTP"
        active = {
            label[len(prefix) + 1 :].split("/", 1)[0] for label in model.shape_label if label.startswith(prefix + "/")
        }
        interface["modules"] = {k: v for k, v in interface["modules"].items() if k in active}
        interface["zones"] = {k: v for k, v in interface["zones"].items() if v["module"] in active}
        interface["graph"] = {
            k: [v for v in values if v in interface["zones"]]
            for k, values in interface["graph"].items()
            if k in interface["zones"]
        }
        interface["eyes"] = {k: v for k, v in interface["eyes"].items() if v["zone"] in interface["zones"]}
        self.interface = interface
        self.conveyors = module.Conveyors(interface)
        self.beams = {}
        frame = scene.frames[prefix]
        self.eye_names = tuple(interface["eyes"])
        self.origins = wp.array(
            [np.asarray(e["origin"]) @ frame[:3, :3] + frame[3, :3] for e in interface["eyes"].values()], dtype=wp.vec3
        )
        self.directions = wp.array(
            [np.asarray(e["dir"]) @ frame[:3, :3] for e in interface["eyes"].values()], dtype=wp.vec3
        )
        self.distances = wp.array([e["dist"] for e in interface["eyes"].values()], dtype=float)
        self.hits = wp.zeros(len(self.eye_names), dtype=int)
        self.channels = {}
        self.zone_speeds = {}
        for item in interface["modules"].values():
            for belt in item["belts"]:
                self.channels[prefix + belt["prim"]] = belt["zone"]
                self.zone_speeds[belt["zone"]] = belt["speed"]
        self.zone_channels = {zone: 17 + i for i, zone in enumerate(self.zone_speeds)}
        starts = model.joint_qd_start.numpy()
        self.mechanisms = {}
        for item in interface["modules"].values():
            for mechanism in item["mechanisms"]:
                joint = scene.joints[prefix + mechanism["joint"]]
                self.mechanisms[mechanism["name"]] = (mechanism, joint, int(starts[joint]))
        self.names = tuple(interface["stations"])
        libraries = json.loads((vendor / "ctp_v01/kr40_motions.json").read_text())
        self.palletizers = {
            name: module.Palletizer(name, libraries[name], libraries[name]["timing"]) for name in self.names
        }
        self.arm_dofs = []
        grippers = []
        for name in self.names:
            root = f"/Plant/Line/Palletizers/{name}/KR40"
            self.arm_dofs.append(
                [int(starts[scene.joints[f"{root}/Joints/{axis}"]]) for axis in ("A1", "A2", "A3", "A6")]
            )
            grippers.append(scene.bodies[root + "/link_6/ToolMount/Gripper/gripper_body"])
        self.grippers = wp.array(grippers, dtype=int)
        self.held = wp.full(4, -1, dtype=int)
        self.joints = wp.full(4, -1, dtype=int)
        self.commands = wp.zeros(4, dtype=int)
        self.closing_until = np.zeros(4)
        ownership = {}
        for key, index in plant.case_index.items():
            cell_name, box = key.split("_", 1)
            for body in plant.case_body_groups[plant.names.index(cell_name), box]:
                ownership[body] = index
        self.shape_case = wp.array(
            [ownership[body] for body in model.shape_body.numpy()[plant.case_shapes.numpy()]], dtype=int
        )
        self.drive_dofs = [record[2] for record in self.mechanisms.values()] + [d for row in self.arm_dofs for d in row]
        self.dofs = wp.array(self.drive_dofs, dtype=int)
        self.coordinates = wp.array(plant.target_coordinates[self.drive_dofs], dtype=int)
        self.positions = wp.zeros(len(self.drive_dofs), dtype=float)
        self.velocities = wp.zeros(len(self.drive_dofs), dtype=float)
        self.q_starts = model.joint_q_start.numpy()
        initial = model.body_q.numpy()
        self.pallet_bodies = np.asarray(
            [body for label, body in scene.bodies.items() if label.endswith("/sm_pallet_block_a08_obj_00")],
            dtype=np.int32,
        )
        self.station_positions = np.asarray(
            [
                initial[
                    next(
                        body
                        for label, body in scene.bodies.items()
                        if label.startswith(f"/Plant/Line/Palletizers/{name}/PalletA/")
                        and label.endswith("/sm_pallet_block_a08_obj_00")
                    ),
                    :3,
                ]
                for name in self.names
            ]
        )
        self.case_roots = np.asarray([bodies[0] for bodies in plant.case_body_groups.values()], dtype=np.int32)
        self.arm_coordinates = [
            [
                self.q_starts[scene.joints[f"/Plant/Line/Palletizers/{name}/KR40/Joints/{a}"]]
                for a in ("A1", "A2", "A3", "A6")
            ]
            for name in self.names
        ]
        self.arm_coordinate_array = wp.array(np.ravel(self.arm_coordinates), dtype=int)
        # The arms' integral tracking term must see the current joint positions;
        # on the one-frame-late host snapshot it winds up and pushes placed cases.
        self.integral = wp.zeros(len(self.arm_coordinate_array), dtype=float)
        plant.link.watch("transport_hits", self.hits)
        plant.link.watch("transport_held", self.held)
        plant.link.watch("body_q", plant.state.body_q, np.r_[self.pallet_bodies, self.case_roots])
        plant.link.watch(
            "joint_q",
            plant.state.joint_q,
            np.r_[[self.q_starts[joint] for _, joint, _ in self.mechanisms.values()], np.ravel(self.arm_coordinates)],
        )
        self.station_state = ["LOADING"] * len(self.names)
        self.station_ready_since = [None] * len(self.names)

    def _observe_stations(self, time, poses=None):
        """Require a seated empty replacement after a completed pallet departs."""
        if poses is None:
            poses = self.plant.state.body_q.numpy()
        for i, name in enumerate(self.names):
            offset = poses[self.pallet_bodies, :3] - self.station_positions[i]
            present = bool(np.any((np.linalg.norm(offset[:, :2], axis=1) < 0.08) & (np.abs(offset[:, 2]) < 0.03)))
            cases = poses[self.case_roots, :3] - self.station_positions[i]
            occupied = bool(
                np.any(
                    (np.abs(cases[:, 0]) < 0.65)
                    & (np.abs(cases[:, 1]) < 0.65)
                    & (cases[:, 2] > 0.1)
                    & (cases[:, 2] < 0.9)
                )
            )
            palletizer = self.palletizers[name]
            state = self.station_state[i]
            if state == "LOADING" and palletizer.slot >= palletizer.n_slots and palletizer.state == "WAIT_CASE":
                state = "FULL"
            if not present:
                if state == "LOADING" and (palletizer.slot > 0 or palletizer.state not in ("INIT", "WAIT_CASE")):
                    raise RuntimeError(f"Palletizer {name}: pallet removed before its load was complete")
                state = "EMPTY"
                self.station_ready_since[i] = None
            elif state == "EMPTY" and not occupied and not self.plant.fleet.station_busy(name):
                if self.station_ready_since[i] is None:
                    self.station_ready_since[i] = time
                elif time - self.station_ready_since[i] >= 0.25:
                    palletizer.slot = 0
                    state = "LOADING"
            else:
                self.station_ready_since[i] = None
            self.station_state[i] = state

    def sense(self):
        """Evaluate the conveyor photo-eyes on the device; graph-capturable."""
        plant, model = self.plant, self.plant.model
        self.hits.zero_()
        wp.launch(
            _case_photoeyes,
            dim=(
                len(self.eye_names),
                (len(plant.case_shapes) + _CASE_PHOTOEYE_BATCH_SIZE - 1) // _CASE_PHOTOEYE_BATCH_SIZE,
                32,
            ),
            block_dim=32,
            inputs=[
                plant.state.body_q,
                model.shape_body,
                model.shape_transform,
                model.shape_scale,
                model.shape_type,
                model.shape_source_ptr,
                plant.case_shapes,
                plant.case_bounds_lo,
                plant.case_bounds_hi,
                self.origins,
                self.directions,
                self.distances,
                self.hits,
            ],
        )

    def before_step(self, time, multipliers, data):
        """Step the source conveyor and palletizer PLCs on a host snapshot."""
        plant, model = self.plant, self.plant.model
        coordinates = data["joint_q"]
        self._observe_stations(time, data["body_q"])
        beams = dict(zip(self.eye_names, data["transport_hits"] != 0, strict=True))
        self.beams = beams
        position = {name: float(coordinates[self.q_starts[joint]]) for name, (_, joint, _) in self.mechanisms.items()}
        output = self.conveyors.step(time, beams, position)
        for zone, speed in output["speeds"].items():
            multipliers[self.zone_channels[zone]] = speed / self.zone_speeds[zone]
        q, qd = [], []
        for name, (mechanism, _, dof) in self.mechanisms.items():
            target, velocity = float(plant.initial_targets[plant.target_coordinates[dof]]), 0.0
            if name in output["blades"]:
                target = mechanism["stroke"] if output["blades"][name] else 0.0
            elif name in output["pushers"]:
                velocity = output["pushers"][name] * mechanism["dir"] * mechanism["speed"]
            q.append(target)
            qd.append(velocity)
        held = data["transport_held"]
        commands = np.zeros(4, dtype=np.int32)
        for i, (name, palletizer) in enumerate(self.palletizers.items()):
            station = self.conveyors.stations[name]
            ready = self.conveyors.held(station["pick_eye"], 0.5) and not self.conveyors.run.get(
                station["spur"][-1], True
            )
            ready = ready and self.station_state[i] == "LOADING" and not plant.fleet.station_busy(name)
            fifo = self.conveyors.spur_fifo[name]
            previous = palletizer.state
            result = palletizer.step(
                time,
                ready,
                "Closed" if held[i] >= 0 else "Open",
                [model.body_label[held[i]]] if held[i] >= 0 else [],
                fifo[0] if fifo else "X",
            )
            if previous == "GRIPPING" and palletizer.state == "LIFT_PICK" and fifo:
                fifo.pop(0)
            if result["cmd"] == "close":
                self.closing_until[i] = time + 0.5
            elif result["cmd"] == "open":
                self.closing_until[i] = 0.0
            commands[i] = -1 if result["cmd"] == "open" else int(held[i] < 0 and time < self.closing_until[i])
            q.extend(result["q"])
            qd.extend(result["qd"])
            palletizer.events.clear()
        plant.link.write(self.commands, commands)
        plant.link.write(self.positions, q)
        plant.link.write(self.velocities, qd)
        self.conveyors.events.clear()

    def can_accept(self, packer):
        """Match the source CTP outfeed's sensor and zone-run handshake."""
        zone = f"OF_{packer}_Z0"
        return not self.beams.get(f"{zone}:end", False) and self.conveyors.run.get(zone, True)

    def simulate(self):
        plant = self.plant
        wp.launch(
            _case_drives,
            dim=len(self.drive_dofs),
            inputs=[
                self.dofs,
                self.coordinates,
                self.positions,
                self.velocities,
                plant.control.joint_target_q,
                plant.control.joint_target_qd,
            ],
        )
        wp.launch(
            _arm_targets,
            dim=len(self.arm_coordinate_array),
            inputs=[
                len(self.mechanisms),
                self.coordinates,
                self.arm_coordinate_array,
                plant.state.joint_q,
                self.positions,
                self.integral,
                plant.control.joint_target_q,
            ],
        )
        wp.launch(
            _vacuum,
            dim=4,
            inputs=[
                plant.solver.dynamics.data,
                plant.state.body_q,
                plant.model.shape_body,
                plant.model.shape_transform,
                plant.model.shape_scale,
                plant.model.shape_type,
                plant.model.shape_source_ptr,
                plant.case_shapes,
                plant.case_bounds_lo,
                plant.case_bounds_hi,
                self.shape_case,
                plant.seal_bodies,
                plant.sealed,
                self.grippers,
                self.commands,
                self.held,
                self.joints,
                plant.claimed,
            ],
        )

    def report(self):
        return {
            name: {
                "state": p.state,
                "station": self.station_state[i],
                "placed_cases": len(p.placed),
                "fault": p.fault,
            }
            for i, (name, p) in enumerate(self.palletizers.items())
        }
