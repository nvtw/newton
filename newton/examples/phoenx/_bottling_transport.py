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


def zero_pressure_conveyors(module):
    """Return the vendor CTP ``Conveyors`` with zero-pressure accumulation at its bridges and pusher stations.

    The vendor's PS-x1 decides each new case once its blade area is clear:
    ``divert = not near_spur_full and (toggle or far_spur_full)``, else it
    passes the case on toward the terminal station. While the terminal
    palletizer waits for its pallet exchange, its spur and then the spine
    fill. A case passed into the backed-up spine stays in the station zone,
    and the station makes no further decision until the far spur drains,
    starving its own palletizer. The source's 64-case CTP run exchanged no
    pallets and never met this. As on a zero-pressure line, a station never
    releases a case into a backed-up spine: while any spine zone on the way
    to the terminal station stands, or the terminal station holds a case
    for its full spur, it diverts the case to its own spur, and while that
    spur is full too it holds the case at its blade until one has room.
    """

    class Conveyors(module.Conveyors):
        def __init__(self, interface):
            super().__init__(interface)
            # The vendor runs a zone unless its end eye is blocked while its
            # downstream zone stands, so a zone without an end eye always runs.
            # Each lane's short level bridge (BT_x) between the incline and the
            # curve has none: it kept driving into a stopped curve, and the
            # incline behind it never stopped, pressing the queue into the
            # cases at the crest until one was levered up and tipped. A bridge
            # cannot accumulate a case of its own, so it runs with its
            # downstream zone: presented with an occupied end eye, the vendor
            # rule runs it exactly while its downstream accepts.
            self.bridge_eyes = {
                f"{zone}:end": True
                for zone in self.zones
                if zone not in self.end_eye and len(self.graph.get(zone) or ()) == 1
            }
            self.end_eye.update({eye[: -len(":end")]: eye for eye in self.bridge_eyes})
            terminals = {station["station_zone"]: station for station in self.stations.values() if station["terminal"]}
            self.spine_ahead = {}
            for pzid, station in self.stations.items():
                if station["terminal"]:
                    continue
                zones, zone = [], station["station_zone"]
                while zone not in terminals:
                    zone = next(z for z in self.graph[zone] if z not in station["spur"])
                    zones.append(zone)
                self.spine_ahead[pzid] = (zones, terminals[zone])

        def step(self, t, beams, pusher_pos):
            return super().step(t, {**beams, **self.bridge_eyes}, pusher_pos)

        def _stations(self, t, beams, ppos):
            seen = dict(beams)
            for pzid, (zones, terminal) in self.spine_ahead.items():
                station = self.stations[pzid]
                backed_up = terminal["state"] != "IDLE" or not all(self.run.get(zone, True) for zone in zones)
                if station["state"] != "IDLE" or not backed_up:
                    continue
                if self._spur_full(pzid, beams):
                    # Input override: the vendor has no hold for a case whose both exits
                    # are full; it decides as soon as its blade-area eye (<zone>:clear)
                    # reads clear, and then passes. Only for this scan's decision, that
                    # eye is presented to the vendor logic as occupied, so it keeps the
                    # case pending and its zone stopped. The physical eye and every other
                    # reader of it are unaffected.
                    seen[f"{station['station_zone']}:clear"] = True
                else:
                    station["toggle"] = True
            super()._stations(t, seen, ppos)

    return Conveyors


def _placement_motions(authored, orientation, slot, pick_q, offset_xy, limits):
    """Place the carton center on the source trajectory using its measured horizontal offset in the robot base."""
    if not np.any(offset_xy):
        return authored
    motions = authored.copy()
    for phase in ("transfer", "descend", "lift"):
        key = f"{phase}_{orientation}_{slot}"
        times, q = authored[key]
        a1, a2, a3, a6 = q.T
        # The source kr40_plan uses two 850 mm links and 420 mm of radial offsets.
        # Keep the flange height and yaw while shifting its XY position opposite the load.
        link = 0.850
        radius = 0.420 + link * (np.cos(a2) + np.cos(a2 + a3))
        height = -link * (np.sin(a2) + np.sin(a2 + a3))
        yaw = pick_q[0] + pick_q[3] - a1 - a6
        dx = np.cos(yaw) * offset_xy[0] - np.sin(yaw) * offset_xy[1]
        dy = np.sin(yaw) * offset_xy[0] + np.cos(yaw) * offset_xy[1]
        blend = 1.0
        if phase != "descend":
            t = times / times[-1]
            blend = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
            if phase == "lift":
                blend = 1.0 - blend
        x = radius * np.cos(a1) - blend * dx
        y = -radius * np.sin(a1) - blend * dy
        radial = np.hypot(x, y) - 0.420
        reach = np.hypot(radial, height) / (2.0 * link)
        if np.any(reach > 1.0):
            return authored
        elbow = 2.0 * np.arccos(reach)
        base = -np.arctan2(y, x)
        shifted = np.column_stack((base, -np.arctan2(height, radial) - 0.5 * elbow, elbow, a6 + a1 - base))
        if np.any(shifted < limits[0]) or np.any(shifted > limits[1]):
            # Preserve the feasible, continuous source cycle for an off-center grip outside the workspace.
            return authored
        motions[key] = times, shifted
    return motions


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
def _carton_support_quadrants(
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    support_owner: wp.array[int],
    held: wp.array[int],
    armed: wp.array[int],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    quadrants: wp.array[int],
    stride: int,
):
    """Mark the quadrants of each held carton supported by a pallet or a lower carton."""
    # Returning at once costs less than a graph conditional around the launch.
    if armed[0] == 0:
        return
    for k in range(wp.tid(), wp.min(contact_count[0], shape0.shape[0]), stride):
        sa = shape0[k]
        sb = shape1[k]
        if sa < 0 or sb < 0:
            continue
        a = shape_body[sa]
        b = shape_body[sb]
        if a < 0 or b < 0:
            continue
        owner_a = support_owner[a]
        owner_b = support_owner[b]
        if owner_a == -1 or owner_b == -1 or owner_a == owner_b:
            continue
        for cell in range(4):
            carton = held[cell]
            if (armed[0] & (1 << cell)) == 0 or carton < 0 or (owner_a != carton and owner_b != carton):
                continue
            if wp.abs(wp.spatial_top(velocities[carton])[2]) >= 0.12:
                continue
            peer = wp.where(owner_a == carton, owner_b, owner_a)
            if (
                peer >= 0
                and wp.transform_get_translation(poses[peer])[2] + 0.12
                >= wp.transform_get_translation(poses[carton])[2]
            ):
                continue
            normal = normals[k]
            if wp.where(owner_a == carton, -normal[2], normal[2]) < 0.9:
                continue
            x = wp.transform_point(poses[a], point0[k])
            y = wp.transform_point(poses[b], point1[k])
            if wp.dot(normal, y - x) - margin0[k] - margin1[k] > 0.0005:
                continue
            local = wp.transform_point(wp.transform_inverse(poses[carton]), (x + y) * 0.5)
            # Exclude the center strips so four near-coincident witnesses cannot qualify the carton.
            if wp.abs(local[0]) >= 0.015 and wp.abs(local[1]) >= 0.015:
                quadrant = wp.int32(local[0] > 0.0) + 2 * wp.int32(local[1] > 0.0)
                wp.atomic_or(quadrants, cell, 1 << quadrant)


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
    armed: wp.array[int],
    quadrants: wp.array[int],
):
    cell = wp.tid()
    parent = grippers[cell]
    supported = (armed[0] & (1 << cell)) != 0 and quadrants[cell] == 15
    quadrants[cell] = 0
    if (commands[cell] < 0 or supported) and held[cell] >= 0:
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
        self.conveyors = zero_pressure_conveyors(module)(interface)
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
        self.authored_motions = {name: palletizer.m for name, palletizer in self.palletizers.items()}
        self.arm_base_axes = [scene.frames[f"/Plant/Line/Palletizers/{name}"][:2, :2].T for name in self.names]
        self.arm_dofs = []
        grippers = []
        for name in self.names:
            root = f"/Plant/Line/Palletizers/{name}/KR40"
            self.arm_dofs.append(
                [int(starts[scene.joints[f"{root}/Joints/{axis}"]]) for axis in ("A1", "A2", "A3", "A6")]
            )
            grippers.append(scene.bodies[root + "/link_6/ToolMount/Gripper/gripper_body"])
        self.grippers = wp.array(grippers, dtype=int)
        self.gripper_bodies = grippers
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
        self.arm_limits = np.stack((model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()))[:, self.arm_dofs]
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
        # A carton root owns its welded flaps. Pallets share the sentinel -2.
        support_owner = np.full(model.body_count, -1, dtype=np.int32)
        support_owner[self.pallet_bodies] = -2
        for bodies in plant.case_body_groups.values():
            support_owner[np.asarray(bodies)] = bodies[0]
        self.support_owner = wp.array(support_owner, dtype=int)
        self.release_armed = wp.zeros(1, dtype=int)
        self.support_quadrants = wp.zeros(len(self.names), dtype=int)
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
        plant.link.watch("body_q", plant.state.body_q, np.r_[self.pallet_bodies, self.case_roots, grippers])
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

    def sense_pallet_support(self, contacts):
        """Sample the last substep before the next vacuum actuation."""
        threads = min(4096, contacts.rigid_contact_max)
        if threads:
            wp.launch(
                _carton_support_quadrants,
                threads,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    self.plant.model.shape_body,
                    self.support_owner,
                    self.held,
                    self.release_armed,
                    self.plant.state.body_q,
                    self.plant.state.body_qd,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    self.support_quadrants,
                    threads,
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
        release_armed = 0
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
            if previous == "GRIPPING" and palletizer.state == "LIFT_PICK":
                poses = data["body_q"]
                offset = (poses[held[i], :2] - poses[self.gripper_bodies[i], :2]) @ self.arm_base_axes[i]
                palletizer.m = _placement_motions(
                    self.authored_motions[name],
                    palletizer.o,
                    palletizer.slot,
                    palletizer.q,
                    offset,
                    self.arm_limits[:, i],
                )
                if fifo:
                    fifo.pop(0)
            if result["cmd"] == "close":
                self.closing_until[i] = time + 0.5
            elif result["cmd"] == "open":
                self.closing_until[i] = 0.0
            commands[i] = -1 if result["cmd"] == "open" else int(held[i] < 0 and time < self.closing_until[i])
            if palletizer.state == "SET_DOWN" or (
                palletizer.state == "DESCEND_PLACE"
                and palletizer.cur is not None
                and time >= palletizer.t0 + palletizer.m[palletizer.cur][0][-1] - 0.18
            ):
                release_armed |= 1 << i
            q.extend(result["q"])
            qd.extend(result["qd"])
            palletizer.events.clear()
        plant.link.write(self.commands, commands)
        plant.link.write(self.release_armed, [release_armed])
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
                self.release_armed,
                self.support_quadrants,
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
