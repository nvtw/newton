# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX bindings for the plant's inlet PLC and authored KR10 motion library."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_fleet import FleetController
from newton.examples.phoenx._bottling_infeed import InfeedController
from newton.examples.phoenx._bottling_io import (
    _actuator_observations,
    _bottle_ray,
    _case_drives,
    _case_photoeyes,
    _load_controller,
    _photoeyes,
    _valves,
)
from newton.examples.phoenx._bottling_transport import TransportController
from newton.solvers import phoenx


@wp.struct
class _Belt:
    velocity: wp.array[wp.vec3]
    nominal: wp.array[wp.vec3]
    channel: int


@wp.kernel
def _belt_speeds(belts: wp.array[_Belt], multipliers: wp.array[float]):
    belt, vertex = wp.tid()
    item = belts[belt]
    if vertex < item.velocity.shape[0]:
        item.velocity[vertex] = item.nominal[vertex] * multipliers[item.channel]


@wp.kernel
def _case_status(joint_q: wp.array[float], clamp: wp.array[int], placed: wp.array[int], observation: wp.array2d[float]):
    cell = wp.tid()
    observation[cell, 13] = float(placed[cell])
    observation[cell, 14] = joint_q[clamp[cell]]


@wp.kernel
def _seal_cases(
    dynamics: phoenx.DynamicsData,
    poses: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    bodies: wp.array2d[int],
    requests: wp.array[int],
    sealed: wp.array[int],
    joints: wp.array2d[int],
):
    case = wp.tid()
    if requests[case] == 0 or sealed[case] != 0:
        return
    root = bodies[case, 0]
    normal = wp.quat_rotate(poses[root].q, wp.vec3(0.0, 0.0, 1.0))
    closed = bool(True)
    for flap in range(4):
        body = bodies[case, flap + 1]
        other = wp.quat_rotate(poses[body].q, wp.vec3(0.0, 0.0, 1.0))
        relative = wp.spatial_bottom(velocity[body]) - wp.spatial_bottom(velocity[root])
        if wp.abs(wp.dot(normal, other)) < 0.9961947 or wp.length(relative) > 0.5:
            closed = False
    if closed:
        # The source replaces this assembly with a rigid sealed-case proxy.
        # Fixed tape bonds retain the actual case, flaps, and physical contents.
        for flap in range(4):
            body = bodies[case, flap + 1]
            frame = wp.transform_multiply(wp.transform_inverse(poses[root]), poses[body])
            desc = phoenx.joint_desc(root, body, frame, wp.transform_identity())
            desc.collide_parent = False
            joints[case, flap] = phoenx.add_joint(dynamics, desc)
        sealed[case] = 1


@wp.kernel
def _supply_cases(
    enabled: wp.array[int],
    bodies: wp.array2d[int],
    local: wp.array2d[wp.transform],
    target: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    cell, slot = wp.tid()
    if enabled[cell] != 0:
        body = bodies[cell, slot]
        body_q[body] = wp.transform_multiply(target[cell], local[cell, slot])
        body_qd[body] = wp.spatial_vector()


@wp.kernel
def _robots(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    bottles: wp.array[int],
    hull_mesh: wp.array[wp.uint64],
    hull_xform: wp.array[wp.transform],
    hull_scale: wp.array[wp.vec3],
    arm_dof: wp.array2d[int],
    arm_coordinate: wp.array2d[int],
    gripper: wp.array[int],
    targets: wp.array[float],
    target_velocity: wp.array[float],
    motion_q: wp.array2d[float],
    motion_qd: wp.array2d[float],
    offsets: wp.array[int],
    lengths: wp.array[int],
    ready: wp.array[int],
    mode: wp.array[int],
    age: wp.array[float],
    rows: wp.array[int],
    held: wp.array2d[int],
    attachments: wp.array2d[int],
    claimed: wp.array[int],
    placed: wp.array[int],
):
    cell = wp.tid()
    state = mode[cell]
    timer = age[cell]
    row = rows[cell] % 3
    motion = int(0)
    if state == 1:
        motion = 0
        timer = float(lengths[0] - 1) / 60.0
        if ready[cell] != 0:
            mode[cell] = 2
            age[cell] = 0.0
    elif state == 2 or state == 3:
        motion = 1
        if state == 3:
            timer = float(lengths[1] - 1) / 60.0
    elif state == 4 or state == 5:
        motion = 2 + row * 2
        if state == 5:
            timer = float(lengths[motion] - 1) / 60.0
    elif state == 6:
        motion = 3 + row * 2
    index = wp.min(int(timer * 60.0 + 0.0001), lengths[motion] - 1)
    for j in range(6):
        targets[arm_coordinate[cell, j]] = motion_q[offsets[motion] + index, j]
        target_velocity[arm_dof[cell, j]] = motion_qd[offsets[motion] + index, j]

    if state == 3 and ready[cell] != 0:
        selected = wp.vec4i(-1, -1, -1, -1)
        complete = bool(True)
        for head in range(4):
            point = wp.transform_point(body_q[gripper[cell]], wp.vec3(-0.1053 + float(head) * 0.0702, 0.0, 0.12664))
            nearest = int(-1)
            direction = wp.transform_vector(body_q[gripper[cell]], wp.vec3(0.0, 0.0, 1.0))
            closest = float(0.006)
            for i in range(bottles.shape[0]):
                bottle = bottles[i]
                if claimed[bottle] != 0:
                    continue
                distance = _bottle_ray(body_q[bottle], point, direction, 0.006, hull_mesh, hull_xform, hull_scale)
                used = bool(False)
                for previous in range(head):
                    if selected[previous] == bottle:
                        used = True
                if distance < closest and not used:
                    nearest, closest = bottle, distance
            selected[head] = nearest
            if nearest < 0:
                complete = False
        if complete:
            for head in range(4):
                bottle = selected[head]
                parent = gripper[cell]
                frame = wp.transform_multiply(wp.transform_inverse(body_q[parent]), body_q[bottle])
                desc = phoenx.joint_desc(parent, bottle, frame, wp.transform_identity())
                desc.collide_parent = False
                attachments[cell, head] = phoenx.add_joint(dyn, desc)
                held[cell, head] = bottle
                claimed[bottle] = parent + 1
            mode[cell] = 4
            age[cell] = 0.0
    elif state == 5 and timer >= float(lengths[motion] - 1) / 60.0 and age[cell] >= 0.1:
        mode[cell] = 6
        age[cell] = 0.0
    elif state != 1 and state != 3 and state != 5 and index == lengths[motion] - 1:
        if state == 0:
            mode[cell] = 1
            age[cell] = 0.0
        elif state == 2:
            mode[cell] = 3
            age[cell] = 0.0
        elif state == 4:
            for head in range(4):
                joint = attachments[cell, head]
                if joint >= 0:
                    phoenx.remove_joint(dyn, joint)
                claimed[held[cell, head]] = 0
                attachments[cell, head] = -1
                held[cell, head] = -1
            placed[cell] += 4
            mode[cell] = 5
            age[cell] = 0.0
        elif state == 6:
            rows[cell] += 1
            mode[cell] = 1
            age[cell] = 0.0
    age[cell] += 1.0 / 60.0


def _load_case_controllers(vendor):
    """Load the simulator-independent source modules without their Kit package."""
    aliases = ("design", "controller", "lines", "lines_l02")
    saved = {name: sys.modules.get(name) for name in aliases}
    saved_path = list(sys.path)
    loaded = {}
    try:
        for alias, relative in zip(
            aliases,
            (
                "case_conveyor_cc01/design.py",
                "case_conveyor_cc01/controller.py",
                "packer_l02/lines.py",
                "packer_l02/lines_l02.py",
            ),
            strict=True,
        ):
            name = "_newton_berlin_" + alias
            path = vendor / relative
            module = _load_controller(name, path)
            sys.modules[alias] = module
            loaded[alias] = module
    finally:
        sys.path[:] = saved_path
        for alias, original in saved.items():
            if original is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = original
    return SimpleNamespace(**loaded)


def _target_coordinates(model):
    """Map velocity DOFs to position-target coordinates, including free roots."""
    starts = model.joint_qd_start.numpy()
    coordinates = model.joint_target_q_start.numpy()
    indices = np.zeros(model.joint_dof_count, dtype=np.int32)
    for joint, dimensions in enumerate(model.joint_dof_dim.numpy()):
        for offset in range(int(sum(dimensions))):
            indices[starts[joint] + offset] = coordinates[joint] + offset
    return indices


class PlantController:
    """Bind the source inlet PLC to real ray tests and force-limited actuators."""

    def __init__(self, scene, model, solver, state, control, asset_root, claimed):
        self.scene, self.model, self.solver, self.state, self.control = scene, model, solver, state, control
        vendor = asset_root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
        module = _load_controller("_newton_berlin_inlet", vendor / "inlet_pki1/controller.py")
        self.interface = json.loads((vendor / "inlet_pki1/interface.json").read_text())
        self.request = json.loads((vendor.parents[3] / "config/requests/packer_l02_plant.json").read_text())
        self.plcs = [module.Controller(self.interface, 1.0 / 60.0) for _ in range(8)]
        self.names = ("PU1", "PU2", "PU3", "PU4", "PD1", "PD2", "PD3", "PD4")
        self.act_names = tuple(self.interface["actuators"])
        self.sensor_names = tuple(self.interface["sensors"])
        starts = model.joint_qd_start.numpy()
        self.target_coordinates = _target_coordinates(model)
        dofs, grippers, actuator_dof, actuator_body, inverse = [], [], [], [], []
        origins, directions, distances = [], [], []
        for name in self.names:
            root = "/Plant/Line/Packers/" + name
            matrix = scene.frames[root]
            inverse_matrix = np.linalg.inv(matrix)
            q = wp.quat_from_matrix(wp.mat33(inverse_matrix[:3, :3].T))
            inverse.append(wp.transform(inverse_matrix[3, :3], q))
            dofs.append([starts[scene.joints[f"{root}/KR10/Physics/joint_{j}"]] for j in range(1, 7)])
            grippers.append(
                next(
                    body
                    for label, body in scene.bodies.items()
                    if label.startswith(root + "/KR10/") and label.endswith("/gripper_body")
                )
            )
            actuator_dof.append(
                [
                    starts[scene.joints[root + "/Inlet" + self.interface["actuators"][n]["joint"][4:]]]
                    for n in self.act_names
                ]
            )
            actuator_body.append(
                [scene.bodies[root + "/Inlet" + self.interface["actuators"][n]["body"][4:]] for n in self.act_names]
            )
            for sensor in self.interface["sensors"].values():
                origins.append(np.asarray(sensor["origin"]) @ matrix[:3, :3] + matrix[3, :3])
                directions.append(np.asarray(sensor["direction"]) @ matrix[:3, :3])
                distances.append(sensor["distance"])
        self.bottles = wp.array(scene.bottle_bodies, dtype=int)
        self.origins = wp.array(origins, dtype=wp.vec3)
        self.directions = wp.array(directions, dtype=wp.vec3)
        self.distances = wp.array(distances, dtype=float)
        body_index = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        shapes = np.flatnonzero(
            (body_index == scene.bottle_bodies[0]) & ((flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)
        )
        self.hull_mesh = wp.array(model.shape_source_ptr.numpy()[shapes], dtype=wp.uint64)
        self.hull_xform = wp.array(model.shape_transform.numpy()[shapes], dtype=wp.transform)
        self.hull_scale = wp.array(model.shape_scale.numpy()[shapes], dtype=wp.vec3)
        self.arm_dof = wp.array(dofs, dtype=int)
        self.arm_coordinate = wp.array(self.target_coordinates[np.asarray(dofs)], dtype=int)
        self.grippers = wp.array(grippers, dtype=int)
        self.actuator_dof = wp.array(actuator_dof, dtype=int)
        self.actuator_body = wp.array(actuator_body, dtype=int)
        self.inverse = wp.array(inverse, dtype=wp.transform)
        self.axes = wp.array([self.interface["actuators"][n]["axis_world"] for n in self.act_names], dtype=wp.vec3)
        # The source readback is displacement from the retracted pose,
        # rather than the mechanism body's absolute coordinate in the cell.
        initial = model.body_q.numpy()
        initial_q = model.joint_q.numpy()
        q_starts = model.joint_q_start.numpy()
        zeros = np.zeros((8, 4), dtype=np.float32)
        for cell, name in enumerate(self.names):
            frame = inverse[cell]
            for i, actuator in enumerate(self.act_names):
                p = wp.transform_point(frame, wp.vec3(initial[actuator_body[cell][i], :3]))
                joint = scene.joints[
                    f"/Plant/Line/Packers/{name}/Inlet" + self.interface["actuators"][actuator]["joint"][4:]
                ]
                zeros[cell, i] = (
                    np.dot(p, self.interface["actuators"][actuator]["axis_world"]) - initial_q[q_starts[joint]]
                )
        self.actuator_zero = wp.array(zeros, dtype=float)
        self.valve_speeds = wp.array([self.interface["actuators"][n]["speed_mps"] for n in self.act_names], dtype=float)
        self.observation = wp.zeros((8, 15), dtype=float)
        self.valve_targets = wp.zeros((8, 4), dtype=float)
        self.ready = wp.zeros(8, dtype=int)
        self.mode = wp.zeros(8, dtype=int)
        self.age = wp.zeros(8, dtype=float)
        self.rows = wp.zeros(8, dtype=int)
        self.held = wp.full((8, 4), -1, dtype=int)
        self.attachments = wp.full((8, 4), -1, dtype=int)
        self.claimed = claimed
        self.placed = wp.zeros(8, dtype=int)
        library = json.loads((vendor / "packer_l02/motions_l02.json").read_text())
        motions = ["init_to_hover", "nest_down"]
        for row in ("far", "mid", "near"):
            motions.extend(("loaded_" + row, "empty_hover_" + row))
        q, qd, offsets, lengths = [], [], [], []
        for name in motions:
            move = library["motions"][name]
            t = np.asarray(move["t"])
            samples = np.linspace(0.0, t[-1], int(np.ceil(t[-1] * 60)) + 1)
            positions = np.asarray(move["q"])
            ff = np.asarray(move.get("q_ff", np.zeros_like(positions)))
            sampled = np.stack([np.interp(samples, t, positions[:, i] + ff[:, i]) for i in range(6)], axis=1)
            offsets.append(len(q))
            lengths.append(len(sampled))
            q.extend(sampled)
            velocity = np.gradient(sampled, 1.0 / 60.0, axis=0)
            velocity[[0, -1]] = 0.0
            qd.extend(velocity)
        self.motion_q = wp.array(q, dtype=float)
        self.motion_qd = wp.array(qd, dtype=float)
        self.offsets = wp.array(offsets, dtype=int)
        self.lengths = wp.array(lengths, dtype=int)
        self._setup_cases(vendor, starts)
        self.transport = TransportController(self, vendor)
        self.infeed = InfeedController(self, vendor, 17 + len(self.transport.zone_channels))
        self.fleet = FleetController(model, state, control, asset_root, transport=self.transport)
        self.wrapper_channels = {
            label: self.infeed.channel_end + i for i, label in enumerate(scene.metadata.get("wrapper_belt_zones", {}))
        }
        descriptors = []
        self.nominal_arrays = []
        self.belt_channels = []
        for mesh, velocities, label in scene.belts:
            channel = 0
            for cell, name in enumerate(self.names):
                if f"/Packers/{name}/Inlet/Belts/" in label:
                    channel = 1 + 2 * cell + int(label.endswith("IN_B"))
                elif f"/BottleLanes/{name}/" in label:
                    channel = self.infeed.lane_channels[name]
            if label in self.transport.channels:
                channel = self.transport.zone_channels[self.transport.channels[label]]
            if label in self.infeed.channels:
                channel = self.infeed.channels[label]
            if label in self.wrapper_channels:
                channel = self.wrapper_channels[label]
            item = _Belt()
            item.velocity = mesh.mesh.velocities
            item.nominal = wp.array(velocities, dtype=wp.vec3)
            item.channel = channel
            descriptors.append(item)
            self.nominal_arrays.append(item.nominal)
            self.belt_channels.append(channel)
        self.belts = wp.array(descriptors, dtype=_Belt)
        self.max_belt_vertices = max(d.velocity.shape[0] for d in descriptors)
        multipliers = np.ones(self.infeed.channel_end + len(self.wrapper_channels), dtype=np.float32)
        # Keep wrapper zones stopped until their physical process controller
        # takes ownership; the generic always-on conveyor channel is unsafe.
        multipliers[self.infeed.channel_end :] = 0.0
        self.multipliers = wp.array(multipliers, dtype=float)

    def _setup_cases(self, vendor, dof_starts):
        """Bind the source CC01 and sealer PLCs to optical case sensors."""
        model, scene = self.model, self.scene
        self.case_modules = _load_case_controllers(vendor)
        design = self.case_modules.design
        self.supply = [[case for case in scene.metadata["case_inventory"][name] if case != "C0"] for name in self.names]
        self.case_lines = [self.case_modules.lines_l02.CaseLineL02(len(supply)) for supply in self.supply]
        self.sealers = [self.case_modules.lines.Sealer(process_s=1.5) for _ in self.names]
        for line in self.case_lines:
            line.seed("Z2", "C0", holding=True)
        self.case_ack = [0] * 8
        self.closed_cases = set()
        self.case_clock = 0.0
        self.zone_velocity = [{} for _ in self.names]
        self.case_sensor_names = [*design.SENSORS, "PE_TS_IN", "PE_TS_SWAP", "PE_TS_OUT", "SPAWN"]
        origins, directions, distances, target, clamps = [], [], [], [], []
        self.case_body_groups = {}
        self.case_local_groups = {}
        initial = model.body_q.numpy()
        q_starts = model.joint_q_start.numpy()
        angle = np.radians(self.request["line_yaw_deg"])
        line_rotation = np.asarray(
            ((np.cos(angle), np.sin(angle), 0.0), (-np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
        )

        def point(matrix, x, y, z):
            local = np.asarray((x, y, z)) @ line_rotation + np.asarray(self.request["line_origin"])
            return local @ matrix[:3, :3] + matrix[3, :3]

        for cell, name in enumerate(self.names):
            root = "/Plant/Line/Packers/" + name
            matrix = scene.frames[root]
            for x in [*design.SENSORS.values(), 4.95, 5.74, 6.3, design.SPAWN["x"]]:
                origins.append(point(matrix, x, design.GUIDE_Y - 0.001, design.BEAM_Z))
                directions.append(np.asarray((0.0, -1.0, 0.0)) @ line_rotation @ matrix[:3, :3])
                distances.append(2.0 * design.GUIDE_Y)
            rotation = line_rotation @ matrix[:3, :3]
            target.append(
                wp.transform(
                    point(matrix, design.SPAWN["x"], design.SPAWN["y"], design.Z_TOP + design.SPAWN["z_drop"]),
                    wp.quat_from_matrix(wp.mat33(rotation.T)),
                )
            )
            clamps.append(q_starts[scene.joints[root + "/Line/CC01/Zones/Z3/Joints/P_Clamp"]])
            for box in ["C0", *self.supply[cell]]:
                prefix = f"/Plant/Sim/Product/Cases/{name}_{box}"
                ids = [scene.bodies[prefix]] + [
                    body for label, body in scene.bodies.items() if label.startswith(prefix + "/")
                ]
                if len(ids) != 5:
                    raise ValueError(f"Expected a case with four hinged flaps: {prefix}")
                inverse = wp.transform_inverse(wp.transform(initial[ids[0], :3], initial[ids[0], 3:]))
                local = [wp.transform_multiply(inverse, wp.transform(initial[i, :3], initial[i, 3:])) for i in ids]
                self.case_body_groups[cell, box] = ids
                self.case_local_groups[cell, box] = local
        self.case_index = {f"{self.names[cell]}_{box}": i for i, (cell, box) in enumerate(self.case_body_groups)}
        self.seal_bodies = wp.array(list(self.case_body_groups.values()), dtype=int)
        self.seal_requests = wp.zeros(len(self.case_index), dtype=int)
        self.seal_requested = np.zeros(len(self.case_index), dtype=np.int32)
        self.sealed = wp.zeros(len(self.case_index), dtype=int)
        self.seal_joints = wp.full((len(self.case_index), 4), -1, dtype=int)
        self.case_origins = wp.array(origins, dtype=wp.vec3)
        self.case_directions = wp.array(directions, dtype=wp.vec3)
        self.case_distances = wp.array(distances, dtype=float)
        self.case_hits = wp.zeros(len(origins), dtype=int)
        self.clamp_coordinates = wp.array(clamps, dtype=int)
        flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
        case_bodies = {b for label, b in scene.bodies.items() if label.startswith("/Plant/Sim/Product/Cases/")}
        shapes = [
            i
            for i, body in enumerate(bodies)
            if body in case_bodies and flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        kinds, scales = model.shape_type.numpy(), model.shape_scale.numpy()
        lo, hi = [], []
        for shape in shapes:
            if kinds[shape] == int(newton.GeoType.BOX):
                lo.append(-scales[shape])
                hi.append(scales[shape])
            elif kinds[shape] in (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)):
                vertices = np.asarray(model.shape_source[shape].vertices) * scales[shape]
                lo.append(vertices.min(axis=0))
                hi.append(vertices.max(axis=0))
            else:
                raise ValueError(f"Unsupported case sensor geometry: {model.shape_label[shape]}")
        self.case_shapes = wp.array(shapes, dtype=int)
        self.case_bounds_lo, self.case_bounds_hi = wp.array(lo, dtype=wp.vec3), wp.array(hi, dtype=wp.vec3)
        self.case_drive_records = []
        closed = model.joint_X_p.numpy()
        axes, joint_type = model.joint_axis.numpy(), model.joint_type.numpy()
        self.initial_targets = self.control.joint_target_q.numpy()
        for cell, name in enumerate(self.names):
            prefix = f"/Plant/Line/Packers/{name}/Line/"
            for label, joint in scene.joints.items():
                if label.startswith(prefix) and "/Zones/" in label:
                    zone = label.split("/Zones/", 1)[1].split("/", 1)[0]
                    dof = int(dof_starts[joint])
                    kind = joint_type[joint]
                    short = label.rsplit("/", 1)[-1]
                    if kind == int(newton.JointType.REVOLUTE) and short.startswith("R_"):
                        self.case_drive_records.append((cell, "roller", zone, dof, 0.0))
                    elif kind == int(newton.JointType.PRISMATIC):
                        self.case_drive_records.append((cell, "actuator", short.removeprefix("P_"), dof, 0.0))
                elif label.startswith(f"/Plant/Sim/Product/Cases/{name}_") and "/FlapJoints/" in label:
                    dof = int(dof_starts[joint])
                    vector, w = closed[joint, 3:6], closed[joint, 6]
                    angle = -2.0 * np.arctan2(np.dot(vector, axes[dof]), w)
                    box = label.split("/Cases/", 1)[1].split("/", 1)[0]
                    self.case_drive_records.append((cell, "flap", box, dof, angle))
        self.case_dofs = wp.array([r[3] for r in self.case_drive_records], dtype=int)
        self.case_coordinates = wp.array(self.target_coordinates[self.case_dofs.numpy()], dtype=int)
        self.case_positions = wp.zeros(len(self.case_drive_records), dtype=float)
        self.case_velocities = wp.zeros(len(self.case_drive_records), dtype=float)
        self.spawn_enabled = wp.zeros(8, dtype=int)
        self.spawn_bodies = wp.zeros((8, 5), dtype=int)
        self.spawn_local = wp.empty((8, 5), dtype=wp.transform)
        self.spawn_target = wp.array(target, dtype=wp.transform)

    def before_step(self):
        """Supply physical photo-eyes and cylinder positions to the source PLCs."""
        self.observation.zero_()
        self.case_hits.zero_()
        wp.launch(
            _photoeyes,
            dim=(len(self.origins), 32),
            block_dim=32,
            inputs=[
                self.state.body_q,
                self.bottles,
                self.origins,
                self.directions,
                self.distances,
                self.hull_mesh,
                self.hull_xform,
                self.hull_scale,
                7,
                self.observation,
            ],
        )
        wp.launch(
            _actuator_observations,
            dim=8,
            inputs=[
                self.state.body_q,
                self.actuator_body,
                self.inverse,
                self.axes,
                self.actuator_zero,
                self.mode,
                self.age,
                self.observation,
            ],
        )
        wp.launch(
            _case_status, dim=8, inputs=[self.state.joint_q, self.clamp_coordinates, self.placed, self.observation]
        )
        wp.launch(
            _case_photoeyes,
            dim=(len(self.case_origins), 32),
            block_dim=32,
            inputs=[
                self.state.body_q,
                self.model.shape_body,
                self.model.shape_transform,
                self.model.shape_scale,
                self.model.shape_type,
                self.model.shape_source_ptr,
                self.case_shapes,
                self.case_bounds_lo,
                self.case_bounds_hi,
                self.case_origins,
                self.case_directions,
                self.case_distances,
                self.case_hits,
            ],
        )
        observations = self.observation.numpy()
        case_hits = self.case_hits.numpy().reshape(8, -1)
        self.case_clock += 1.0 / 60.0
        case_outputs, sealer_outputs = [], []
        sealed = self.sealed.numpy()
        spawn_flags = np.zeros(8, dtype=np.int32)
        spawn_ids = np.zeros((8, 5), dtype=np.int32)
        spawn_local = np.zeros((8, 5, 7), dtype=np.float32)
        for cell, (line, sealer) in enumerate(zip(self.case_lines, self.sealers, strict=True)):
            beams = dict(zip(self.case_sensor_names, case_hits[cell] != 0, strict=True))
            completed = int(observations[cell, 13]) // 12
            if completed > self.case_ack[cell]:
                line.robot_done(self.case_clock, line.occ["Z3"])
                self.case_ack[cell] = completed
            output = line.step(self.case_clock, beams, observations[cell, 14], sealer_ready=sealer.ready())
            if line.release_started:
                sealer.expect(self.case_clock)
            for box in line.handed_off:
                sealer.receive(self.case_clock, box)
            line.handed_off.clear()
            out_ts = sealer.step(self.case_clock, beams, outfeed_can_accept=self.transport.can_accept(self.names[cell]))
            if sealer.state == "PROCESS":
                self.closed_cases.add(f"{self.names[cell]}_{sealer.box}")
            if sealer.swap_request is not None:
                box = sealer.swap_request
                index = self.case_index[f"{self.names[cell]}_{box}"]
                self.seal_requested[index] = 1
                if sealed[index] != 0:
                    sealer.swapped(self.case_clock, box)
            case_outputs.append(output)
            sealer_outputs.append(out_ts)
            if self.supply[cell] and line.can_spawn(self.case_clock, not beams["SPAWN"]):
                box = self.supply[cell].pop(0)
                spawn_flags[cell] = 1
                spawn_ids[cell] = self.case_body_groups[cell, box]
                spawn_local[cell] = np.asarray(self.case_local_groups[cell, box])
                line.on_spawn(self.case_clock, box)
            target_speeds = dict(output["zone_speed"])
            target_speeds.update(out_ts["speed"])
            for zone, target in target_speeds.items():
                current = self.zone_velocity[cell].get(zone, 0.0)
                ramp_s = (
                    self.case_modules.design.RAMP_UP_S
                    if abs(target) > abs(current)
                    else self.case_modules.design.RAMP_DOWN_S
                )
                delta = self.case_modules.design.SPEED["transport"] / (ramp_s * 60.0)
                self.zone_velocity[cell][zone] = current + np.clip(target - current, -delta, delta)
        self.seal_requests.assign(self.seal_requested)
        self.spawn_enabled.assign(spawn_flags)
        if spawn_flags.any():
            self.spawn_bodies.assign(spawn_ids)
            self.spawn_local.assign(spawn_local)
        positions, velocities = [], []
        for cell, kind, name, dof, closed in self.case_drive_records:
            position, velocity = self.initial_targets[self.target_coordinates[dof]], 0.0
            if kind == "roller":
                position = 0.0
                velocity = self.zone_velocity[cell].get(name, 0.0) / self.case_modules.design.ROLLER_R
            elif kind == "actuator":
                position = sealer_outputs[cell]["stop"] if name == "SwapStop" else case_outputs[cell]["actuators"][name]
            elif name in self.closed_cases:
                position = closed
            positions.append(position)
            velocities.append(velocity)
        self.case_positions.assign(np.asarray(positions, dtype=np.float32))
        self.case_velocities.assign(np.asarray(velocities, dtype=np.float32))
        targets, ready = [], []
        commands = {}
        speed = np.ones(len(self.multipliers), dtype=np.float32)
        speed[self.infeed.channel_end :] = 0.0
        for cell, plc in enumerate(self.plcs):
            obs = {
                "beams": dict(zip(self.sensor_names, observations[cell, :7] != 0, strict=True)),
                "actuator_pos": dict(zip(self.act_names, observations[cell, 7:11], strict=True)),
            }
            command = plc.step(
                obs, robot_mode=True, robot_gripped=observations[cell, 11] != 0, robot_clear=observations[cell, 12] != 0
            )
            commands[self.names[cell]] = command
            targets.append([command["targets"][name] for name in self.act_names])
            line = self.case_lines[cell]
            ready.append(int(command["group_ready"] and line.ready and line.station == "LOCKED"))
            speed[1 + cell * 2] = command["speeds"]["INA"] / 0.2
            speed[2 + cell * 2] = command["speeds"]["INB"] / 0.4
        self.valve_targets.assign(np.asarray(targets, dtype=np.float32))
        self.ready.assign(np.asarray(ready, dtype=np.int32))
        # All controllers observe the same physical state before simulation.
        poses = self.state.body_q.numpy()
        coordinates = self.state.joint_q.numpy()
        self.transport.before_step(self.case_clock, speed, poses=poses, coordinates=coordinates)
        self.infeed.before_step(self.case_clock, commands, speed, coordinates=coordinates)
        self.fleet.before_step(self.case_clock, poses=poses, coordinates=coordinates)
        self.multipliers.assign(speed)

    def simulate(self):
        """Update belt friction, valve drives, arm targets, and grasp constraints."""
        wp.launch(
            _seal_cases,
            dim=len(self.case_index),
            inputs=[
                self.solver.dynamics.data,
                self.state.body_q,
                self.state.body_qd,
                self.seal_bodies,
                self.seal_requests,
                self.sealed,
                self.seal_joints,
            ],
        )
        self.transport.simulate()
        self.infeed.simulate()
        self.fleet.simulate()
        wp.launch(
            _supply_cases,
            dim=(8, 5),
            inputs=[
                self.spawn_enabled,
                self.spawn_bodies,
                self.spawn_local,
                self.spawn_target,
                self.state.body_q,
                self.state.body_qd,
            ],
        )
        wp.launch(
            _case_drives,
            dim=len(self.case_dofs),
            inputs=[
                self.case_dofs,
                self.case_coordinates,
                self.case_positions,
                self.case_velocities,
                self.control.joint_target_q,
                self.control.joint_target_qd,
            ],
        )
        wp.launch(_belt_speeds, dim=(len(self.belts), self.max_belt_vertices), inputs=[self.belts, self.multipliers])
        wp.launch(
            _valves,
            dim=(8, 4),
            inputs=[
                self.actuator_dof,
                self.valve_targets,
                self.valve_speeds,
                self.control.joint_target_qd,
            ],
        )
        wp.launch(
            _robots,
            dim=8,
            inputs=[
                self.solver.dynamics.data,
                self.state.body_q,
                self.bottles,
                self.hull_mesh,
                self.hull_xform,
                self.hull_scale,
                self.arm_dof,
                self.arm_coordinate,
                self.grippers,
                self.control.joint_target_q,
                self.control.joint_target_qd,
                self.motion_q,
                self.motion_qd,
                self.offsets,
                self.lengths,
                self.ready,
                self.mode,
                self.age,
                self.rows,
                self.held,
                self.attachments,
                self.claimed,
                self.placed,
            ],
        )

    def report(self):
        """Return inlet states and the number of physically released bottles."""
        return {
            "inlets": {
                name: {"state": plc.state, "fault": plc.fault, "groups": plc.groups_picked}
                for name, plc in zip(self.names, self.plcs, strict=True)
            },
            "placed_bottles": self.placed.numpy().tolist(),
            "robot_modes": self.mode.numpy().tolist(),
            "case_stations": [line.station for line in self.case_lines],
            "reserve_cartons": {name: len(supply) for name, supply in zip(self.names, self.supply, strict=True)},
            "sealers": [sealer.state for sealer in self.sealers],
            "palletizers": self.transport.report(),
            "fleet": self.fleet.report(),
            "infeed": self.infeed.status,
            "sealed_cases": int(self.sealed.numpy().sum()),
        }
