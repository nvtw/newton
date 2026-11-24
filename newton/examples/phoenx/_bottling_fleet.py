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
# The source dispatcher sends an idle truck home after this long (dispatcher.home_after_s).
_HOME_AFTER_S = 20.0
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


def _occupied_drop_offs(ports, poses, movable):
    """Return the drop-off ports with a movable body in their load box."""
    occupied = []
    for name, port in ports.items():
        if name.startswith("dropoff_") and name.endswith("_drop"):
            offset = poses[movable, :3] - np.array((*port.xy, port.z))
            inside = np.all(np.abs(offset[:, :2]) <= float(_LOAD_HALF), axis=1)
            inside &= (offset[:, 2] > -0.05) & (offset[:, 2] <= float(_LOAD_HEIGHT))
            if np.any(inside):
                occupied.append(name)
    return occupied


@wp.kernel
def put_away(
    active: wp.array[int],
    drop: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    movable: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Carry every movable body of the wrapped load on a drop-off, at rest, by ``delta``."""
    b = wp.tid()
    if active[0] == 0 or movable[b] == 0:
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


def _observe_port(pallets, cases, xy, z, yaw, carried=None):
    """Identify a seated pallet and conservatively block occupied landing space.

    A truck lined up to set down its load already holds that pallet and its
    cases over the landing space; ``carried`` excludes them.
    """
    if carried is not None:
        load = pallets[carried, :3]
        rides = (np.linalg.norm(cases[:, :2] - load[:2], axis=1) < 0.8) & (cases[:, 2] > load[2] - 0.05)
        cases = cases[~rides]
        pallets = np.delete(pallets, carried, axis=0)
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
    phase: str = "go_source"
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
        self.des = _load_controller(package + ".des", root / "des.py")
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
        # The source executor reserves one drive segment at a time (isaac_exec.IsaacExecutor):
        # truck -> (des.Plan, end-operation cells, log length at start, target node).
        self.plans, self.segments, self.idle_since = {}, {}, {}
        self.missions = {}
        self.completed = []
        self.waiting = {}
        self.wrapper_ready = {}
        # Wrapped pallets waiting on a drop-off for the source's timed put-away.
        self.putaways, self.shelved = [], 0
        self.putaway_active = wp.zeros(1, dtype=int, device=model.device)
        self.putaway_drop = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.putaway_delta = wp.zeros(1, dtype=wp.vec3, device=model.device)
        # The put-away carries free product and pallets, never the plant's own trucks.
        movable = (model.body_inv_mass.numpy() > 0.0) & ~np.array(
            [label.startswith(prefix) for label in model.body_label]
        )
        self.putaway_movable = wp.array(movable.astype(np.int32), dtype=int, device=model.device)
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
            # The plant starts in production: a wrapped load may already stand on a
            # drop-off, waiting for the same reach truck.
            for name in _occupied_drop_offs(self.navigation.ports, model.body_q.numpy(), movable):
                self.putaways.append((_PUTAWAY_S, name))

    def station_busy(self, name):
        """Keep palletizing inhibited until the exchange truck has withdrawn."""
        port = name + "_A"
        return any(port in (mission.source, mission.destination) for mission in self.missions.values())

    def _port_observation(self, name, poses, carried=None):
        port = self.navigation.ports[name]
        return _observe_port(
            poses[self.pallet_bodies],
            poses[self.transport.case_roots],
            port.xy,
            port.z,
            math.radians(port.yaw),
            carried,
        )

    def _plan(self, start, end, loaded):
        key = (start, end, bool(loaded))
        if key not in self.plans:
            plan = self.des.Plan(self.navigation, self.traffic.fp, start, end, bool(loaded))
            if plan.pl["n_map_hits"]:
                raise RuntimeError(f"Forklift route {start} -> {end} intersects the plant")
            self.plans[key] = plan
        return self.plans[key]

    def _operation_cells(self, name):
        port = self.navigation.ports[name]
        # The source sequencer can pull back before aligning an insertion.
        outer = port.lineup_ref - max(self.source.PRE_D, self.source.M.PRE_INSERT) * port.u
        return self.traffic.fp.line_cells(outer, port.ref_at_tips(port.pick_tips_s), port.truck_yaw, True)

    def _start_segment(self, truck, target, loaded, operation):
        """Reserve one drive segment and the fork operation at its end, as the source executor does."""
        end = self._operation_cells(target) if operation else set()
        plan = None if truck.node == target else self._plan(truck.node, target, loaded)
        cells = end if plan is None else plan.cells | end
        if not self.traffic.try_acquire(truck.name, cells, self.driver.t):
            self.waiting[truck.name] = f"{truck.node}->{target}: traffic blocked"
            return False
        self.waiting.pop(truck.name, None)
        self.segments[truck.name] = (plan, end, len(truck.log), target)
        if plan is not None:
            truck.ops.extend(self.source.drive_ops(plan.pl, loaded, f"{truck.node}->{target}"))
        return True

    def _release_along(self, truck):
        """Give back the cells a driving truck has passed (isaac_exec._release_along)."""
        plan, end, logged, _target = self.segments[truck.name]
        op = truck.op
        if plan is None or not isinstance(op, (self.source.Track, self.source.Leg)):
            return
        leg = sum(1 for entry in truck.log[logged:] if entry.get("op") == "leg")
        if leg < len(plan.legs):
            progress = float(op.s[min(op.idx, len(op.s) - 1)])
            self.traffic.keep_only(truck.name, plan.needed_at(leg, progress) | end)

    def _finish_segment(self, truck, sense):
        _plan, end, _logged, target = self.segments.pop(truck.name)
        truck.node = target
        self._keep_footprint(truck, sense, end)

    def _keep_footprint(self, truck, sense, extra=frozenset()):
        cells = self.traffic.fp.cells(sense.rp, sense.yaw, truck.load is not None)
        self.traffic.keep_only(truck.name, cells | set(extra))

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
            or any(destination == port for _, port in self.putaways)
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
        time = self.driver.t
        blocking = {blocker for _since, blockers in self.traffic.waiting.values() for blocker in blockers}
        for truck, sense in zip(self.driver.trucks, senses, strict=True):
            if truck.name in self.missions:
                self.idle_since.pop(truck.name, None)
                continue
            cells = self.traffic.fp.cells(sense.rp, sense.yaw, truck.load is not None)
            self.traffic.keep_only(truck.name, cells)
            if self.traffic.hold(truck.name, cells):
                raise RuntimeError(f"Forklift {truck.name}: occupied traffic reservation")
            if not truck.idle():
                continue
            idle_since = self.idle_since.setdefault(truck.name, time)
            for source, destination, pallet, empty in self._transfer_requests(truck.name):
                mission = self._available_transfer(source, destination, poses, pallet=pallet, empty=empty)
                if mission is None:
                    self.waiting[truck.name] = (
                        f"{source}->{destination}: stock, landing or transfer reservation unavailable"
                    )
                    continue
                self.missions[truck.name] = mission
                self.wrapper_ready.pop(source, None)
                self.waiting.pop(truck.name, None)
                break
            else:
                # The source dispatcher sends an idle truck home at once when it stands
                # in a waiting truck's way, else after home_after_s.
                home = self.homes[truck.name]
                if truck.node != home and (truck.name in blocking or time - idle_since >= _HOME_AFTER_S):
                    self.missions[truck.name] = _Transfer(home, home, -1, phase="go_home")

    def _advance_missions(self, poses, senses):
        for truck, sense in zip(self.driver.trucks, senses, strict=True):
            mission = self.missions.get(truck.name)
            if truck.name in self.segments and not truck.idle():
                self._release_along(truck)
            if mission is None or not truck.idle():
                continue
            if mission.phase == "go_source":
                if self._start_segment(truck, mission.source, False, True):
                    mission.phase = "to_pick"
            elif mission.phase == "to_pick":
                if truck.name in self.segments:
                    self._finish_segment(truck, sense)
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
                self._keep_footprint(truck, sense)
                mission.phase = "go_destination"
            elif mission.phase == "go_destination":
                if self._start_segment(truck, mission.destination, True, True):
                    mission.phase = "to_place"
            elif mission.phase == "to_place":
                if truck.name in self.segments:
                    self._finish_segment(truck, sense)
                if not self._port_observation(mission.destination, poses, carried=truck.load).clear:
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
                self._keep_footprint(truck, sense)
                mission.phase = "go_home"
            elif mission.phase == "go_home":
                if self._start_segment(truck, self.homes[truck.name], False, False):
                    mission.phase = "home"
            elif mission.phase == "home":
                self._finish_segment(truck, sense)
                del self.missions[truck.name]
                self.waiting.pop(truck.name, None)
        self._resolve_deadlock()

    def _resolve_deadlock(self):
        """Break a wait-for cycle: a truck in it that carries nothing gives its task back (dispatcher._resolve)."""
        cycle = self.traffic.cycle() if self.traffic.waiting else None
        if not cycle:
            return
        for name in cycle:
            mission = self.missions.get(name)
            if mission is None or mission.phase != "go_source" or self.trucks[name].load is not None:
                continue
            if mission.source in self.navigation.ports and mission.source.startswith("WL_"):
                self.wrapper_ready[mission.source] = mission.pallet
            del self.missions[name]
            self.traffic.waiting.pop(name, None)
            self.waiting[name] = "yielded a deadlocked task"
            return

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
                    truck.failed = f"{result['label']}: route target was not reached ({result})"
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
                    self.putaway_movable,
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
