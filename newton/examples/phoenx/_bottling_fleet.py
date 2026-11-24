# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run source forklift drive sequences using measured Newton body and joint state."""

import json
import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from newton.examples.phoenx._bottling_io import _case_drives, _load_controller

# The source does not simulate its warehouse leg: a reach truck takes a wrapped
# pallet off a drop-off 120 s after it lands and puts it into the rack, which
# the source keeps as a hidden shelf below the floor (palletflow dispatcher.py
# ``putaway_s``; isaac_exec.py ``PalletPool``: shelf at (0, -60), z -5, 1.6 m
# pitch, 20 columns). The plant runs that leg at the source's timing.
_PUTAWAY_S = 120.0
_SHELF_XY, _SHELF_Z, _SHELF_PITCH, _SHELF_COLUMNS, _SHELF_ROWS = (0.0, -60.0), -5.0, 1.6, 20, 4
# m, the box above a drop-off that holds a wrapped load
_LOAD_HALF, _LOAD_HEIGHT = wp.constant(0.8), wp.constant(2.6)


def add_rack_shelf(builder):
    """Add the source's hidden rack shelf, on which put-away loads rest out of the plant."""
    span = _SHELF_PITCH * np.array((_SHELF_COLUMNS, _SHELF_ROWS))
    center = (
        _SHELF_XY[0] + 0.5 * (span[0] - _SHELF_PITCH),
        _SHELF_XY[1] - 0.5 * (span[1] - _SHELF_PITCH),
        _SHELF_Z - 0.05,
    )
    builder.add_shape_box(
        -1,
        xform=wp.transform(center, wp.quat_identity()),
        hx=0.5 * span[0],
        hy=0.5 * span[1],
        hz=0.05,
        cfg=builder.ShapeConfig(mu=0.5, is_visible=False),
        label="/Plant/Logistics/RackShelf/collision",
    )


def shelf_slot(index):
    """Return the plant position of the source shelf's ``index``-th pallet slot."""
    row, column = divmod(index, _SHELF_COLUMNS)
    return np.array((_SHELF_XY[0] + column * _SHELF_PITCH, _SHELF_XY[1] - row * _SHELF_PITCH, _SHELF_Z))


@wp.kernel
def put_away(
    active: wp.array[int],
    drop: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Carry every free body of the wrapped load on a drop-off, at rest, by ``delta``."""
    b = wp.tid()
    if active[0] == 0 or inv_mass[b] == 0.0:
        return
    pose = body_q[b]
    offset = wp.transform_get_translation(pose) - drop[0]
    if (
        wp.abs(offset[0]) > _LOAD_HALF
        or wp.abs(offset[1]) > _LOAD_HALF
        or offset[2] < -0.05
        or offset[2] > _LOAD_HEIGHT
    ):
        return
    body_q[b] = wp.transform(wp.transform_get_translation(pose) + delta[0], wp.transform_get_rotation(pose))
    body_qd[b] = wp.spatial_vector()


@wp.kernel
def _hold_wheels(
    coordinates: wp.array2d[int],
    targets: wp.array2d[int],
    joint_q: wp.array[float],
    joint_target_q: wp.array[float],
):
    # The source velocity-drives each wheel from its current angle.
    truck = wp.tid()
    joint_target_q[targets[truck, 0]] = joint_q[coordinates[truck, 0]]


@dataclass
class _PortObservation:
    pallet: int | None
    clear: bool
    empty: bool


def _observe_port(pallets, cases, xy, z, yaw):
    """Identify a seated pallet and conservatively block occupied landing space."""
    offset = pallets[:, :3] - np.array([*xy, z])
    near = (np.linalg.norm(offset[:, :2], axis=1) < 1.3) & (np.abs(offset[:, 2]) < 0.5)
    seated = (np.linalg.norm(offset[:, :2], axis=1) < 0.08) & (np.abs(offset[:, 2]) < 0.03)
    indices = np.flatnonzero(seated)
    pallet = None
    if len(indices) == 1:
        k = int(indices[0])
        q = pallets[k, 3:7]
        up = 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
        angle = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        # The source pallet's long axis is local +Y; ports specify fork-entry
        # heading. Either end of the pallet admits the same fork pockets.
        pallet_heading = angle + math.pi / 2
        error = (pallet_heading - yaw + math.pi / 2) % math.pi - math.pi / 2
        if up >= math.cos(math.radians(3)) and abs(error) <= math.radians(3):
            pallet = k
    relative_cases = cases[:, :3] - np.array([*xy, z])
    occupied = bool(
        np.any(
            (np.linalg.norm(relative_cases[:, :2], axis=1) < 1.0)
            & (relative_cases[:, 2] > -0.05)
            & (relative_cases[:, 2] < 2.5)
        )
    )
    return _PortObservation(pallet, not np.any(near) and not occupied, pallet is not None and not occupied)


def _placement_valid(error):
    """Accept a released pallet only when it is seated, upright and aligned."""
    return (
        error["xy_m"] <= 0.03
        and abs(error["yaw_deg"]) <= 3.0
        and abs(error["z_m"]) <= 0.03
        and error["tilt_deg"] <= 3.0
    )


@dataclass
class _Transfer:
    source: str
    destination: str
    pallet: int
    phase: str = "to_pick"
    record: dict = field(default_factory=dict)


class FleetController:
    """Apply force-limited wheel, steering, and lift targets to the authored trucks."""

    def __init__(self, model, state, control, asset_root, *, dt=1.0 / 60.0, transport=None, link=None):
        self.model, self.state, self.control = model, state, control
        self.link, self.poses = link, None
        root = asset_root / "scenarios/logistics/palletflow"
        package = "_newton_bottling_fleet"
        _load_controller(package, root / "__init__.py")
        self.planner = _load_controller(package + ".planner", root / "planner.py")
        source_model = _load_controller(package + ".model", root / "model.py")
        self.source = _load_controller(package + ".fleet_isaac", root / "fleet_isaac.py")
        prefix = "/Plant/Logistics/Fleet/"
        paths = {
            label[len(prefix) : -5]: label[:-5]
            for label in model.body_label
            if label.startswith(prefix) and label.endswith("/body")
        }
        self.pallet_bodies = np.asarray(
            [i for i, label in enumerate(model.body_label) if label.endswith("/sm_pallet_block_a08_obj_00")],
            dtype=np.int32,
        )
        fleet = self

        class MeasuredFleet(self.source.Fleet):
            def pallets(self):
                """Read physical pallet poses for the source pickup/place feedback."""
                return fleet.poses[fleet.pallet_bodies]

        self.driver = MeasuredFleet(None, paths, [model.body_label[i] for i in self.pallet_bodies], 1.0 / dt)
        self.trucks = self.driver.by_name
        self.bodies = [model.body_label.index(path + "/body") for path in paths.values()]
        joint_index = {
            (label[len(prefix) :].split("/", 1)[0], label.rsplit("/", 1)[-1]): i
            for i, label in enumerate(model.joint_label)
            if label.startswith(prefix)
        }
        joints = np.asarray(
            [
                [joint_index[truck, name] for name in ("back_wheel_drive", "back_wheel_swivel", "lift_joint")]
                for truck in paths
            ],
            dtype=np.int32,
        ).reshape((-1, 3))
        self.q_indices = model.joint_q_start.numpy()[joints]
        self.qd_indices = model.joint_qd_start.numpy()[joints]
        self.wheel_q = wp.array(self.q_indices, dtype=int, device=model.device)
        self.wheel_targets = wp.array(model.joint_target_q_start.numpy()[joints], dtype=int, device=model.device)
        self.dofs = wp.array(self.qd_indices.ravel(), dtype=int, device=model.device)
        self.coordinates = wp.array(model.joint_target_q_start.numpy()[joints].ravel(), dtype=int, device=model.device)
        self.positions = wp.array(
            control.joint_target_q.numpy()[model.joint_target_q_start.numpy()[joints]].ravel(),
            dtype=float,
            device=model.device,
        )
        self.velocities = wp.array(
            control.joint_target_qd.numpy()[self.qd_indices].ravel(), dtype=float, device=model.device
        )
        self.previous = None
        self.dt = dt
        initial = model.joint_q.numpy()
        for truck, indices in zip(self.driver.trucks, self.q_indices, strict=True):
            truck.lift_cmd = float(initial[indices[2]])
        self.transport = transport
        self.missions = {}
        self.completed = []
        self.waiting = {}
        self.wrapper_ready = {}
        # Wrapped pallets waiting on a drop-off for the source's timed put-away.
        self.putaways, self.shelved = [], 0
        self.putaway_active = wp.zeros(1, dtype=int, device=model.device)
        self.putaway_drop = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.putaway_delta = wp.zeros(1, dtype=wp.vec3, device=model.device)
        if transport is not None:
            data = json.loads((root.parent / "palletflow_model.json").read_text())
            self.navigation = source_model.from_json(
                data["model"], asset_root / "plant/lines_1x8/nav/occupancy_lgv_multi.yaml"
            )
            self.roles = {row["name"]: {tuple(role) for role in row["roles"]} for row in data["fleet"]}
            self.homes = {row["name"]: row["home"] for row in data["fleet"]}
            traffic = _load_controller(package + ".traffic", root / "traffic.py")
            self.traffic = traffic.Traffic()
            for truck in self.driver.trucks:
                truck.node = self.homes[truck.name]

    def station_busy(self, name):
        """Keep palletizing inhibited until the exchange truck has withdrawn."""
        port = name + "_A"
        return any(port in (mission.source, mission.destination) for mission in self.missions.values())

    def _port_observation(self, name, poses):
        port = self.navigation.ports[name]
        return _observe_port(
            poses[self.pallet_bodies], poses[self.transport.case_roots], port.xy, port.z, math.radians(port.yaw)
        )

    def _route(self, start, end, loaded):
        if start == end:
            return None
        plan = self.navigation.plan(start, end, loaded=loaded)
        if plan["n_map_hits"]:
            raise RuntimeError(f"Forklift route {start} -> {end} intersects the plant")
        return plan

    def _queue_route(self, truck, target, loaded):
        plan = self._route(truck.node, target, loaded)
        if plan is not None:
            truck.ops.extend(self.source.drive_ops(plan, loaded, f"{truck.node}->{target}"))

    def _mission_cells(self, truck, source, destination):
        """Reserve the complete exchange and home route before entering an aisle."""
        cells = set()
        for start, end, loaded in (
            (truck.node, source, False),
            (source, destination, True),
            (destination, self.homes[truck.name], False),
        ):
            plan = self._route(start, end, loaded)
            if plan is not None:
                for leg in plan["legs"]:
                    for _, footprint in self.traffic.fp.leg_samples(leg, loaded):
                        cells.update(footprint)
        for name in (source, destination):
            port = self.navigation.ports[name]
            # The source sequencer can pull back before aligning an insertion.
            outer = port.lineup_ref - max(self.source.PRE_D, self.source.M.PRE_INSERT) * port.u
            cells.update(self.traffic.fp.line_cells(outer, port.ref_at_tips(port.pick_tips_s), port.truck_yaw, True))
        return cells

    def notify_wrapper_ready(self, side, pallet_body, poses=None):
        """Release a physically seated pallet after wrapping and labelling finish.

        The wrapper process must call this only after its physical completion
        checks pass. Occupancy alone never establishes process completion.
        Repeated notification of the same waiting pallet is harmless.
        """
        source = f"WL_{side}_pick"
        destination = f"dropoff_{side}_drop"
        if source not in self.navigation.ports or destination not in self.navigation.ports:
            raise ValueError(f"Unknown wrapper side: {side}")
        indices = np.flatnonzero(self.pallet_bodies == pallet_body)
        if len(indices) != 1:
            raise ValueError(f"Unknown wrapper pallet body: {pallet_body}")
        pallet = int(indices[0])
        if poses is None:
            poses = self.state.body_q.numpy()
        if self._port_observation(source, poses).pallet != pallet:
            raise ValueError(f"Wrapper {side}: completed pallet is not seated at its pickup")
        if any(mission.pallet == pallet for mission in self.missions.values()):
            raise ValueError(f"Wrapper {side}: completed pallet is already reserved")
        if source in self.wrapper_ready and self.wrapper_ready[source] != pallet:
            raise ValueError(f"Wrapper {side}: a different completed pallet is still waiting")
        self.wrapper_ready[source] = pallet

    def _transfer_requests(self, truck_name):
        """Enumerate physical A/B requests and explicitly completed C requests."""
        roles = self.roles[truck_name]
        for name, state in zip(self.transport.names, self.transport.station_state, strict=True):
            station = name + "_A"
            port = self.navigation.ports[station]
            if state == "FULL":
                source, destination, role = station, f"WL_{port.side}_drop", "A"
            elif state == "EMPTY":
                source, destination, role = f"EP_{name}_pick", station, "B"
            else:
                continue
            if (role, port.side) in roles or (role, name) in roles:
                yield source, destination, None, role == "B"
        for source, pallet in self.wrapper_ready.items():
            side = self.navigation.ports[source].side
            if ("C", side) in roles:
                yield source, f"dropoff_{side}_drop", pallet, False

    def _available_transfer(self, source, destination, poses, *, pallet=None, empty=False):
        """Require measured inventory and exclusive ownership of both ports."""
        src, dst = self._port_observation(source, poses), self._port_observation(destination, poses)
        if (
            src.pallet is None
            or (pallet is not None and src.pallet != pallet)
            or (empty and not src.empty)
            or not dst.clear
        ):
            return None
        for mission in self.missions.values():
            if mission.pallet == src.pallet or {source, destination}.intersection(
                (mission.source, mission.destination)
            ):
                return None
        return _Transfer(source, destination, src.pallet)

    def _dispatch(self, senses, poses):
        """Dispatch only observed physical stock; source timers/pools are never run."""
        for truck, sense in zip(self.driver.trucks, senses, strict=True):
            if truck.name not in self.missions:
                cells = self.traffic.fp.cells(sense.rp, sense.yaw, truck.load is not None)
                self.traffic.keep_only(truck.name, cells)
                if self.traffic.hold(truck.name, cells):
                    raise RuntimeError(f"Forklift {truck.name}: occupied traffic reservation")
        for truck in self.driver.trucks:
            if not truck.idle() or truck.name in self.missions:
                continue
            for source, destination, pallet, empty in self._transfer_requests(truck.name):
                mission = self._available_transfer(source, destination, poses, pallet=pallet, empty=empty)
                if mission is None:
                    self.waiting[truck.name] = (
                        f"{source}->{destination}: stock, landing or transfer reservation unavailable"
                    )
                    continue
                cells = self._mission_cells(truck, source, destination)
                if not self.traffic.try_acquire(truck.name, cells, self.driver.t):
                    self.waiting[truck.name] = f"{source}->{destination}: traffic blocked"
                    continue
                self.missions[truck.name] = mission
                self.wrapper_ready.pop(source, None)
                self.waiting.pop(truck.name, None)
                self._queue_route(truck, source, False)
                break

    def _advance_missions(self, poses, senses):
        for truck, sense in zip(self.driver.trucks, senses, strict=True):
            mission = self.missions.get(truck.name)
            if mission is None or not truck.idle():
                continue
            if mission.phase == "to_pick":
                truck.node = mission.source
                observed = self._port_observation(mission.source, poses)
                if observed.pallet != mission.pallet:
                    self.waiting[truck.name] = f"{mission.source}: reserved pallet is no longer seated"
                    continue
                truck.ops.extend(
                    self.source.pick_ops(
                        self.driver, truck, self.navigation.ports[mission.source], mission.pallet, mission.record
                    )
                )
                mission.phase = "pick"
            elif mission.phase == "pick":
                if truck.load != mission.pallet:
                    raise RuntimeError(f"Forklift {truck.name}: pickup finished without its reserved pallet")
                relative = self.driver.rel_pallet(truck, sense, mission.pallet)
                nominal = self.navigation.ports[mission.source].pick_tips_s - self.source.M.TIPS_FROM_REF
                rise = poses[self.pallet_bodies[mission.pallet], 2] - mission.record["pallet_z0"]
                if (
                    rise < 0.10
                    or abs(relative["a"] - nominal) > 0.15
                    or abs(relative["b"]) > 0.08
                    or relative["tilt_deg"] > 3.0
                ):
                    raise RuntimeError(f"Forklift {truck.name}: pallet is not supported after pickup withdrawal")
                mission.phase = "to_place"
                self._queue_route(truck, mission.destination, True)
            elif mission.phase == "to_place":
                truck.node = mission.destination
                if not self._port_observation(mission.destination, poses).clear:
                    self.waiting[truck.name] = f"{mission.destination}: landing space occupied"
                    continue
                mission.record = {}
                truck.ops.extend(
                    self.source.place_ops(
                        self.driver, truck, self.navigation.ports[mission.destination], mission.record
                    )
                )
                mission.phase = "place"
            elif mission.phase == "place":
                error = self.driver.port_error(mission.pallet, self.navigation.ports[mission.destination])
                if truck.load is not None or not _placement_valid(error):
                    raise RuntimeError(f"Forklift {truck.name}: placement failed after withdrawal: {error}")
                self.completed.append(
                    {"truck": truck.name, "source": mission.source, "destination": mission.destination, "error": error}
                )
                if mission.destination.startswith("dropoff_"):
                    self.putaways.append((self.driver.t + _PUTAWAY_S, mission.destination))
                mission.phase = "home"
                self._queue_route(truck, self.homes[truck.name], False)
            elif mission.phase == "home":
                truck.node = self.homes[truck.name]
                del self.missions[truck.name]
                self.waiting.pop(truck.name, None)

    def _put_away(self, time):
        """Queue the source's put-away of the next due wrapped pallet from its drop-off into the rack shelf."""
        active, drop, delta = 0, np.zeros(3), np.zeros(3)
        if self.putaways and time >= self.putaways[0][0]:
            port = self.navigation.ports[self.putaways.pop(0)[1]]
            drop = np.array((*port.xy, port.z))
            delta = shelf_slot(self.shelved) - drop
            self.shelved += 1
            active = 1
        self.link.write(self.putaway_active, [active])
        self.link.write(self.putaway_drop, drop[None].astype(np.float32))
        self.link.write(self.putaway_delta, delta[None].astype(np.float32))

    def report(self):
        """Report physical transfers and unresolved stock/station interlocks."""
        return {
            "completed": len(self.completed),
            "put_away": self.shelved,
            "active": {name: mission.phase for name, mission in self.missions.items()},
            "waiting": dict(self.waiting),
        }

    def before_step(self, time, data):
        """Advance queued operations from a host snapshot and queue drive targets."""
        if not self.driver.trucks:
            return
        self.poses = poses = data["body_q"]
        q = data["joint_q"][self.q_indices]
        qd = data["joint_qd"][self.qd_indices]
        senses = []
        for i, pose in enumerate(poses[self.bodies]):
            sense = self.source.Sense()
            sense.t = time
            sense.pos = pose[:3]
            sense.yaw, sense.tilt = self.source.quat_yaw_tilt(pose[3:])
            forward = np.array([math.cos(sense.yaw), math.sin(sense.yaw)])
            sense.rp = sense.pos[:2] + self.planner.FK_REAR_X * forward
            sense.vm = 0.0 if self.previous is None else float(np.linalg.norm(sense.rp - self.previous[i])) / self.dt
            sense.lift, sense.lift_v = float(q[i, 2]), float(qd[i, 2])
            senses.append(sense)
        self.previous = [sense.rp.copy() for sense in senses]
        self.driver.t = time
        if self.transport is not None:
            self._advance_missions(poses, senses)
            self._dispatch(senses, poses)
            self._put_away(time)
        counts = [len(truck.log) for truck in self.driver.trucks]
        # Only reuse the operation sequencer: source Fleet.tick() also corrects
        # pallet poses, which would bypass the physical fork contacts.
        self.driver.run_ops(senses)
        position = np.zeros_like(q)
        velocity = np.zeros_like(q)
        for i, truck in enumerate(self.driver.trucks):
            for result in truck.log[counts[i] :]:
                if result.get("reached") is False:
                    truck.failed = f"{result['label']}: route target was not reached"
                if result.get("op") == "lift" and abs(result["final_m"] - result["target_m"]) > 0.004:
                    truck.failed = f"{result['label']}: lift target was not reached"
            if truck.failed:
                raise RuntimeError(f"Forklift {truck.name}: {truck.failed}")
            position[i] = (q[i, 0], self.source.STEER_SIGN * truck.steer, truck.lift_cmd)
            velocity[i, 0] = truck.v / (self.planner.FK_WHEEL_R * max(math.cos(truck.steer), 0.3))
        self.link.write(self.positions, position.ravel())
        self.link.write(self.velocities, velocity.ravel())

    def simulate(self):
        """Apply the current commands using the existing graph-capturable drive I/O."""
        if self.transport is not None:
            wp.launch(
                put_away,
                dim=self.model.body_count,
                inputs=[
                    self.putaway_active,
                    self.putaway_drop,
                    self.putaway_delta,
                    self.model.body_inv_mass,
                    self.state.body_q,
                    self.state.body_qd,
                ],
                device=self.model.device,
            )
        if self.dofs.size:
            wp.launch(
                _case_drives,
                dim=self.dofs.size,
                inputs=[
                    self.dofs,
                    self.coordinates,
                    self.positions,
                    self.velocities,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
                device=self.model.device,
            )
            wp.launch(
                _hold_wheels,
                dim=len(self.q_indices),
                inputs=[self.wheel_q, self.wheel_targets, self.state.joint_q, self.control.joint_target_q],
                device=self.model.device,
            )
