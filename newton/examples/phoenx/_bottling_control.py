# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX bindings for the plant's inlet PLC and authored KR10 motion library."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_case_plc import (
    ACTUATOR_NAMES,
    CASE_STATIONS,
    SEALER_STATES,
    ZONE_NAMES,
    case_drive_targets,
    case_line_plcs,
    make_case_lines,
)
from newton.examples.phoenx._bottling_fleet import FleetController
from newton.examples.phoenx._bottling_infeed import InfeedController
from newton.examples.phoenx._bottling_io import (
    _CASE_PHOTOEYE_BATCH_SIZE,
    _PHOTOEYE_BATCH_SIZE,
    _actuator_observations,
    _bottle_ray,
    _case_drives,
    _case_photoeyes,
    _HostLink,
    _load_controller,
    _photoeyes,
    _valves,
)
from newton.examples.phoenx._bottling_transport import TransportController
from newton.examples.phoenx._bottling_wrap import WrapperController
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
    hinges: wp.array2d[int],
    joint_enabled: wp.array[wp.bool],
):
    # Each station owns a consecutive range of case IDs. Station order therefore
    # preserves tape allocation order even when several requests arrive at once.
    if wp.tid() != 0:
        return
    for station in range(requests.shape[0]):
        case = requests[station]
        if case < 0 or sealed[case] != 0:
            continue
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
            failed = wp.bool(False)
            for flap in range(4):
                body = bodies[case, flap + 1]
                frame = wp.transform_multiply(wp.transform_inverse(poses[root]), poses[body])
                desc = phoenx.joint_desc(root, body, frame, wp.transform_identity())
                desc.collide_parent = False
                joint = phoenx.add_joint(dynamics, desc)
                joints[case, flap] = joint
                if joint < 0:
                    failed = True
            if failed:
                for flap in range(4):
                    joint = joints[case, flap]
                    if joint >= 0:
                        phoenx.remove_joint(dynamics, joint)
                    joints[case, flap] = -1
                continue
            # Tape fixes the current flap pose and supersedes its folding hinge.
            # Disable the hinge only after the whole replacement batch succeeds.
            for flap in range(4):
                joint_enabled[hinges[case, flap]] = False
            sealed[case] = 1


@wp.kernel
def _attach_sealed_flaps(
    poses: wp.array[wp.transform],
    bodies: wp.array2d[int],
    sealed: wp.array[int],
    flap_shapes: wp.array2d[int],
    attached: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    solver_shape_body: wp.array[wp.int32],
):
    """Carry sealed flap collision shapes on the case body, as the source's sealed proxy does.

    Taped flaps already move rigidly with their case. Their shapes keep their
    current pose; the flap bodies keep their mass on the tape bonds. Stacked
    cases then load the case body directly, not a 20 g flap through its bond.
    """
    case = wp.tid()
    if sealed[case] == 0 or attached[case] != 0:
        return
    root = bodies[case, 0]
    to_root = wp.transform_inverse(poses[root])
    for k in range(flap_shapes.shape[1]):
        shape = flap_shapes[case, k]
        if shape < 0:
            continue
        flap = shape_body[shape]
        shape_transform[shape] = wp.transform_multiply(
            to_root, wp.transform_multiply(poses[flap], shape_transform[shape])
        )
        shape_body[shape] = root
        # PhoenX indexes bodies from one; slot zero is the world.
        solver_shape_body[shape] = wp.int32(root + 1)
    attached[case] = 1


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


@wp.func_native("""
#ifdef __CUDA_ARCH__
    for (int offset = 16; offset; offset >>= 1) {
        float d = __shfl_down_sync(0xffffffffu, distance, offset);
        int i = __shfl_down_sync(0xffffffffu, index, offset);
        if (i >= 0 && (index < 0 || d < distance || (d == distance && i < index))) {
            index = i;
            distance = d;
        }
    }
    return __shfl_sync(0xffffffffu, index, 0);
#else
    return index;
#endif
""")
def _nearest_pickup_lane(distance: float, index: int) -> int: ...


@wp.kernel
def _robots(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    bottles: wp.array[int],
    bottle_count: wp.array[int],
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
    lanes: int,
):
    cell = wp.tid() // lanes
    lane = wp.tid() % lanes
    state = mode[cell]
    picking = state == 3 and ready[cell] != 0
    selected = wp.vec4i(-1, -1, -1, -1)
    complete = bool(True)
    if picking:
        # Cups choose in order. Equal distances retain the original bottle-list order.
        for head in range(4):
            point = wp.transform_point(body_q[gripper[cell]], wp.vec3(-0.1053 + float(head) * 0.0702, 0.0, 0.12664))
            nearest = int(-1)
            direction = wp.transform_vector(body_q[gripper[cell]], wp.vec3(0.0, 0.0, 1.0))
            closest = float(0.006)
            for i in range(lane, bottle_count[0], lanes):
                bottle = bottles[i]
                if claimed[bottle] != 0:
                    continue
                distance = _bottle_ray(body_q[bottle], point, direction, 0.006, hull_mesh, hull_xform, hull_scale)
                used = bool(False)
                for previous in range(head):
                    if selected[previous] == bottle:
                        used = True
                if distance < closest and not used:
                    nearest, closest = i, distance
            if lanes == 32:
                nearest = _nearest_pickup_lane(closest, nearest)
            if nearest >= 0:
                nearest = bottles[nearest]
            selected[head] = nearest
            if nearest < 0:
                complete = False
    if lane != 0:
        return
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

    if picking:
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


# Inlet PLC states and fault reasons, in the source controller's names.
_INLET_STATES = (
    "SETTLE",
    "WAIT_QUEUE",
    "CLAMP",
    "RELEASE",
    "GATE_CLOSING",
    "SETTLE_GROUP",
    "NESTING",
    "READY",
    "ROBOT_UNNEST",
    "ROBOT_PICK",
    "FAULT",
)
_INLET_FAULTS = (
    None,
    "stops_not_closed_at_start",
    "holdback_did_not_meet_bottle",
    "holdback_slip",
    "count_exceeded",
    "gate_zone_occupied",
    "release_timeout",
    "gate_blocked",
    "group_did_not_reach_end_stop",
    "nest_not_seated",
    "comb_did_not_open",
)
# Debounced beams: queue_full, gate_seen, five_seen, nest1, nest4, nest_clear.
_INLET_DEBOUNCES = 6


@wp.struct
class _Inlet:
    """Per-cell state and interface constants of the source PKI-1 inlet PLC."""

    state: wp.array[int]
    fault: wp.array[int]
    running: wp.array[int]
    count: wp.array[int]
    picked: wp.array[int]
    last_count: wp.array[int]
    t: wp.array[wp.float64]
    t_state: wp.array[wp.float64]
    soft_until: wp.array[wp.float64]
    inb_drive: wp.array[wp.float64]
    debounce: wp.array2d[int]
    debounce_t: wp.array2d[wp.float64]
    debounce_on: wp.array[wp.float64]
    debounce_off: wp.array[wp.float64]
    status: wp.array2d[int]
    dt: wp.float64
    drive_time: wp.float64
    soft_advance: wp.float64
    nesting: wp.float64
    ina: wp.float64
    inb: wp.float64
    n_group: int
    gate: int
    holdback: int
    endstop: int
    comb: int
    gate_stroke: float
    holdback_stroke: float
    endstop_stroke: float
    comb_stroke: float
    gate_closed: float
    endstop_closed: float
    holdback_max: float
    comb_seated: float
    comb_tol: float


@wp.func
def _debounce(inlet: _Inlet, cell: int, k: int, raw: bool) -> bool:
    """Source ``Debounce.update``: hold a changed input for its on/off time."""
    state = inlet.debounce[cell, k] != 0
    if raw != state:
        timer = inlet.debounce_t[cell, k] + inlet.dt
        limit = inlet.debounce_off[k]
        if raw:
            limit = inlet.debounce_on[k]
        if timer >= limit:
            inlet.debounce[cell, k] = int(raw)
            timer = wp.float64(0.0)
        inlet.debounce_t[cell, k] = timer
    else:
        inlet.debounce_t[cell, k] = wp.float64(0.0)
    return inlet.debounce[cell, k] != 0


@wp.kernel
def _inlet_plcs(
    inlet: _Inlet,
    observation: wp.array2d[float],
    line_ready: wp.array[int],
    targets: wp.array2d[float],
    ready: wp.array[int],
    multipliers: wp.array[float],
):
    """One source inlet PLC scan per packer cell, in robot mode.

    A line-by-line port of ``Controller.step`` in ``inlet_pki1/controller.py``.
    The test-bypass states are unreachable in robot mode and are omitted.
    """
    cell = wp.tid()
    dt = inlet.dt
    t = inlet.t[cell] + dt
    t_state = inlet.t_state[cell] + dt
    queue_beam = observation[cell, 1] != 0.0
    gate_beam = observation[cell, 2] != 0.0
    five_beam = observation[cell, 3] != 0.0
    count_beam = observation[cell, 4] != 0.0
    nest1_beam = observation[cell, 5] != 0.0
    nest4_beam = observation[cell, 6] != 0.0
    gate = observation[cell, 7 + inlet.gate]
    holdback = observation[cell, 7 + inlet.holdback]
    endstop = observation[cell, 7 + inlet.endstop]
    comb = observation[cell, 7 + inlet.comb]
    gripped = observation[cell, 11] != 0.0
    robot_clear = observation[cell, 12] != 0.0
    qfull = _debounce(inlet, cell, 0, queue_beam)
    gate_seen = _debounce(inlet, cell, 1, gate_beam)
    five_seen = _debounce(inlet, cell, 2, five_beam)
    n1 = _debounce(inlet, cell, 3, nest1_beam)
    n4 = _debounce(inlet, cell, 4, nest4_beam)
    nest_clear = _debounce(inlet, cell, 5, not nest1_beam and not nest4_beam and not count_beam)
    last = inlet.last_count[cell] != 0
    rising = count_beam and not last
    falling = not count_beam and last
    inlet.last_count[cell] = int(count_beam)

    state = inlet.state[cell]
    next_state = state
    fault = int(0)
    if state == 0:  # SETTLE
        targets[cell, inlet.gate] = inlet.gate_stroke
        targets[cell, inlet.endstop] = inlet.endstop_stroke
        if t_state >= wp.float64(1.0):
            if gate >= inlet.gate_closed and endstop >= inlet.endstop_closed:
                inlet.running[cell] = 1
                next_state = 1
            else:
                fault = 1
    elif state == 1:  # WAIT_QUEUE
        if gate_seen and five_seen and nest_clear and inlet.inb_drive[cell] >= inlet.drive_time:
            targets[cell, inlet.holdback] = inlet.holdback_stroke
            next_state = 2
    elif state == 2:  # CLAMP
        if t_state >= wp.float64(0.15):
            if holdback >= 0.004 and holdback <= inlet.holdback_max:
                targets[cell, inlet.gate] = 0.0
                inlet.count[cell] = 0
                next_state = 3
            else:
                fault = 2
    elif state == 3:  # RELEASE
        if not five_beam and t_state > wp.float64(0.1):
            fault = 3
        else:
            if rising:
                inlet.count[cell] += 1
                if inlet.count[cell] > inlet.n_group:
                    fault = 4
            if fault == 0:
                if inlet.count[cell] == inlet.n_group and falling:
                    if gate_beam:
                        fault = 5
                    else:
                        targets[cell, inlet.gate] = inlet.gate_stroke
                        next_state = 4
                elif t_state > wp.float64(6.0):
                    fault = 6
    elif state == 4:  # GATE_CLOSING
        if gate >= inlet.gate_closed and t_state >= wp.float64(0.25):
            targets[cell, inlet.holdback] = 0.0
            # Soft advance: the freed queue closes up on the gate at half speed.
            inlet.soft_until[cell] = t + inlet.soft_advance
            next_state = 5
        elif t_state > wp.float64(1.5):
            fault = 7
    elif state == 5:  # SETTLE_GROUP
        if n1 and n4:
            targets[cell, inlet.comb] = inlet.comb_stroke
            next_state = 6
        elif t_state > wp.float64(6.0):
            fault = 8
    elif state == 6:  # NESTING
        if t_state >= inlet.nesting:
            if wp.abs(comb - inlet.comb_seated) <= inlet.comb_tol and nest1_beam and nest4_beam:
                next_state = 7
            elif t_state >= inlet.nesting + wp.float64(0.4):
                fault = 9
    elif state == 7:  # READY
        if gripped:
            # Release to the gripper: comb only.
            targets[cell, inlet.comb] = 0.0
            next_state = 8
    elif state == 8:  # ROBOT_UNNEST
        if comb <= 0.004 and t_state >= wp.float64(0.25):
            next_state = 9
        elif t_state > wp.float64(1.5):
            fault = 10
    elif state == 9:  # ROBOT_PICK
        if nest_clear and robot_clear:
            inlet.picked[cell] += 1
            next_state = 1
    else:  # FAULT
        inlet.running[cell] = 0
    if fault != 0:
        inlet.fault[cell] = fault
        next_state = 10
    if next_state != state:
        t_state = wp.float64(0.0)
    inlet.state[cell] = next_state
    inlet.t[cell] = t
    inlet.t_state[cell] = t_state

    # Source ``_outputs``: chain speeds, drive-time proof and port status.
    run = inlet.running[cell] != 0 and next_state != 10
    ina = wp.float64(0.0)
    inb = wp.float64(0.0)
    if run:
        if not qfull:
            ina = inlet.ina
        inb = inlet.inb
        if t < inlet.soft_until[cell]:
            inb = inlet.inb * wp.float64(0.5)
    if inb >= inlet.inb * wp.float64(0.99):
        inlet.inb_drive[cell] = inlet.inb_drive[cell] + dt
    else:
        inlet.inb_drive[cell] = wp.float64(0.0)
    multipliers[1 + cell * 2] = float(ina / wp.float64(0.2))
    multipliers[2 + cell * 2] = float(inb / wp.float64(0.4))
    ready[cell] = int(next_state == 7 and line_ready[cell] != 0)
    inlet.status[cell, 0] = next_state
    inlet.status[cell, 1] = inlet.fault[cell]
    inlet.status[cell, 2] = inlet.picked[cell]
    inlet.status[cell, 3] = int(run and qfull)


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


class _InletStatus:
    """Host view of one device inlet PLC, refreshed from the previous frame."""

    state = "SETTLE"
    fault = None
    groups_picked = 0

    def update(self, status):
        self.state = _INLET_STATES[status[0]]
        self.fault = _INLET_FAULTS[status[1]]
        self.groups_picked = int(status[2])
        return {"packer_ready": self.state != "FAULT", "packer_backlog": bool(status[3])}


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

    def __init__(self, scene, model, solver, state, control, asset_root, claimed, supply=None):
        self.scene, self.model, self.solver, self.state, self.control = scene, model, solver, state, control
        vendor = asset_root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
        self.link = _HostLink(model.device)
        self.interface = json.loads((vendor / "inlet_pki1/interface.json").read_text())
        self.request = json.loads((vendor.parents[3] / "config/requests/packer_l02_plant.json").read_text())
        self.plcs = [_InletStatus() for _ in range(8)]
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
        bottles = np.full(len(scene.bottle_bodies) + (supply.capacity if supply is not None else 0), -1, dtype=np.int32)
        bottles[: len(scene.bottle_bodies)] = scene.bottle_bodies
        self.bottles = wp.array(bottles, dtype=int, device=model.device)
        self.bottle_count = wp.array([len(scene.bottle_bodies)], dtype=int, device=model.device)
        self.origins = wp.array(origins, dtype=wp.vec3)
        self.directions = wp.array(directions, dtype=wp.vec3)
        self.distances = wp.array(distances, dtype=float)
        hulls = scene.metadata["bottle_sensor_hulls"]
        self.hull_mesh = wp.array([mesh.finalize(model.device) for mesh, _, _ in hulls], dtype=wp.uint64)
        self.hull_xform = wp.array([pose for _, pose, _ in hulls], dtype=wp.transform)
        self.hull_scale = wp.array([scale for _, _, scale in hulls], dtype=wp.vec3)
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
        self.inlet = self._setup_inlet()
        self.line_ready = wp.zeros(8, dtype=int)
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
        self.infeed = InfeedController(self, vendor, 17 + len(self.transport.zone_channels), supply)
        self.fleet = FleetController(model, state, control, asset_root, transport=self.transport, link=self.link)
        self.wrapper_channels = {
            label: self.infeed.channel_end + i for i, label in enumerate(scene.metadata.get("wrapper_belt_zones", {}))
        }
        self.wrapper = WrapperController(
            model, state, control, asset_root, self.wrapper_channels, self.fleet, plant=self
        )
        channel_count = self.infeed.channel_end + len(self.wrapper_channels)
        # Process feed zones stop while their eye sees a standing queue.
        zones = scene.metadata.get("process_zones", {})
        self.zone_channels = {name: channel_count + i for i, name in enumerate(zones)}
        zone_of = {label: name for name, (labels, _) in zones.items() for label in labels}
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
            if label in zone_of:
                channel = self.zone_channels[zone_of[label]]
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
        multipliers = np.ones(channel_count + len(zones), dtype=np.float32)
        # Infeed starts only after measured placement, truck withdrawal and homing.
        multipliers[self.infeed.channel_end :] = 0.0
        self.multipliers = wp.array(multipliers, dtype=float)
        self._watch()

    def _setup_inlet(self):
        """Bind the source inlet PLC's interface constants to the device port."""
        actuators, timers = self.interface["actuators"], self.interface["timers_s"]
        inlet = _Inlet()
        for name, dtype in (("state", int), ("fault", int), ("running", int), ("count", int), ("picked", int)):
            setattr(inlet, name, wp.zeros(8, dtype=dtype))
        inlet.last_count = wp.zeros(8, dtype=int)
        for name in ("t", "t_state", "soft_until", "inb_drive"):
            setattr(inlet, name, wp.zeros(8, dtype=wp.float64))
        inlet.soft_until.fill_(-1.0)
        inlet.debounce = wp.zeros((8, _INLET_DEBOUNCES), dtype=int)
        inlet.debounce_t = wp.zeros((8, _INLET_DEBOUNCES), dtype=wp.float64)
        steady, beam_on = timers["queue_steady_s"], timers["nest_beam_on_s"]
        inlet.debounce_on = wp.array(
            [1.0, steady, steady, beam_on, beam_on, timers["nest_clear_on_s"]], dtype=wp.float64
        )
        inlet.debounce_off = wp.array([1.0, 0.05, 0.05, 0.05, 0.05, 0.0], dtype=wp.float64)
        inlet.status = wp.zeros((8, 4), dtype=int)
        inlet.dt = 1.0 / 60.0
        inlet.drive_time, inlet.soft_advance = timers["drive_time_s"], timers["soft_advance_s"]
        inlet.nesting = timers["nesting_s"]
        inlet.ina, inlet.inb = self.interface["speeds_mps"]["INA"], self.interface["speeds_mps"]["INB"]
        inlet.n_group = len(self.interface["nest_x"])
        for name in ("gate", "holdback", "endstop", "comb"):
            actuator = {"gate": "Gate", "holdback": "Holdback", "endstop": "EndStop", "comb": "NestComb"}[name]
            setattr(inlet, name, self.act_names.index(actuator))
            setattr(inlet, name + "_stroke", actuators[actuator]["stroke"])
        # The source compares float32 readbacks against these float64 limits.
        inlet.gate_closed = float(np.float32(actuators["Gate"]["stroke"] - 0.006))
        inlet.endstop_closed = float(np.float32(actuators["EndStop"]["stroke"] - 0.006))
        inlet.holdback_max = float(np.float32(actuators["Holdback"]["stroke"] - 0.003))
        inlet.comb_seated = actuators["NestComb"]["seated_stroke"]
        inlet.comb_tol = actuators["NestComb"]["seated_tol"]
        return inlet

    def _setup_cases(self, vendor, dof_starts):
        """Bind the source CC01 and sealer PLCs to optical case sensors."""
        model, scene = self.model, self.scene
        self.case_modules = _load_case_controllers(vendor)
        design = self.case_modules.design
        self.supply = [[case for case in scene.metadata["case_inventory"][name] if case != "C0"] for name in self.names]
        # Host views of the device case-line and sealer PLCs, refreshed each frame.
        self.case_lines = [SimpleNamespace(station="EMPTY") for _ in self.names]
        self.sealers = [SimpleNamespace(state="EMPTY") for _ in self.names]
        self.case_clock = 0.0
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

        cases = "/Plant/Sim/Product/Cases/"
        case_parts = {}
        for label, body in scene.bodies.items():
            if label.startswith(cases):
                case, inside, _ = label[len(cases) :].partition("/")
                if inside:
                    case_parts.setdefault(case, []).append(body)
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
                prefix = f"{cases}{name}_{box}"
                ids = [scene.bodies[prefix], *case_parts.get(f"{name}_{box}", ())]
                if len(ids) != 5:
                    raise ValueError(f"Expected a case with four hinged flaps: {prefix}")
                inverse = wp.transform_inverse(wp.transform(initial[ids[0], :3], initial[ids[0], 3:]))
                local = [wp.transform_multiply(inverse, wp.transform(initial[i, :3], initial[i, 3:])) for i in ids]
                self.case_body_groups[cell, box] = ids
                self.case_local_groups[cell, box] = local
        self.case_index = {f"{self.names[cell]}_{box}": i for i, (cell, box) in enumerate(self.case_body_groups)}
        self.seal_bodies = wp.array(list(self.case_body_groups.values()), dtype=int)
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        types = model.joint_type.numpy()
        hinge_for_body = {
            int(child): joint for joint, child in enumerate(children) if types[joint] == int(newton.JointType.REVOLUTE)
        }
        hinges = []
        for ids in self.case_body_groups.values():
            row = [hinge_for_body[body] for body in ids[1:]]
            if any(parents[joint] != ids[0] for joint in row):
                raise ValueError("Case flap hinges must connect to their case root")
            hinges.append(row)
        self.seal_hinges = wp.array(hinges, dtype=int)
        self.sealed = wp.zeros(len(self.case_index), dtype=int)
        self.seal_joints = wp.full((len(self.case_index), 4), -1, dtype=int)
        self.case_origins = wp.array(origins, dtype=wp.vec3)
        self.case_directions = wp.array(directions, dtype=wp.vec3)
        self.case_distances = wp.array(distances, dtype=float)
        self.case_hits = wp.zeros(len(origins), dtype=int)
        self.clamp_coordinates = wp.array(clamps, dtype=int)
        flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
        collide = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        flap_shapes = [
            [shape for flap in ids[1:5] for shape in np.flatnonzero((bodies == flap) & collide)]
            for ids in self.case_body_groups.values()
        ]
        width = max(1, *(len(row) for row in flap_shapes))
        self.flap_shapes = wp.array([row + [-1] * (width - len(row)) for row in flap_shapes], dtype=int)
        self.flaps_attached = wp.zeros(len(self.case_index), dtype=int)
        case_bodies = {b for label, b in scene.bodies.items() if label.startswith(cases)}
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
        records = self.case_drive_records
        self.case_lines_plc = make_case_lines(
            design,
            self.case_modules.controller,
            self.case_modules.lines.Sealer(process_s=1.5),
            [
                [self.case_index[f"{name}_{box}"] for box in supply]
                for name, supply in zip(self.names, self.supply, strict=True)
            ],
            [self.case_index[f"{name}_C0"] for name in self.names],
            model.device,
        )
        self.case_local = wp.array(list(self.case_local_groups.values()), dtype=wp.transform)
        self.case_closed = wp.zeros(len(self.case_index), dtype=int)
        self.outfeed_ok = wp.zeros(8, dtype=int)
        self.case_record_cells = wp.array([r[0] for r in records], dtype=int)
        self.case_record_kinds = wp.array([("roller", "actuator", "flap").index(r[1]) for r in records], dtype=int)
        self.case_record_items = wp.array(
            [
                (ZONE_NAMES.index(r[2]) if r[2] in ZONE_NAMES else -1)
                if r[1] == "roller"
                else ACTUATOR_NAMES.index(r[2])
                if r[1] == "actuator"
                else self.case_index[r[2]]
                for r in records
            ],
            dtype=int,
        )
        self.case_record_closed = wp.array([r[4] for r in records], dtype=float)
        self.case_dofs = wp.array([r[3] for r in self.case_drive_records], dtype=int)
        self.case_coordinates = wp.array(self.target_coordinates[self.case_dofs.numpy()], dtype=int)
        self.case_positions = wp.zeros(len(self.case_drive_records), dtype=float)
        self.case_velocities = wp.zeros(len(self.case_drive_records), dtype=float)
        self.case_record_initial = wp.array(
            self.initial_targets[self.target_coordinates[self.case_dofs.numpy()]], dtype=float
        )
        self.spawn_enabled = wp.zeros(8, dtype=int)
        self.spawn_bodies = wp.zeros((8, 5), dtype=int)
        self.spawn_local = wp.empty((8, 5), dtype=wp.transform)
        self.spawn_target = wp.array(target, dtype=wp.transform)
        self.case_speeds = wp.zeros((8, len(ZONE_NAMES)), dtype=wp.float64)

    def _watch(self):
        """Declare the device data the host PLCs read each frame."""
        link = self.link
        link.watch("sealed", self.sealed)
        link.watch("inlet", self.inlet.status)
        link.watch("cases", self.case_lines_plc.status)
        fleet = self.fleet
        link.watch("body_q", self.state.body_q, np.r_[fleet.bodies, fleet.pallet_bodies])
        link.watch("joint_q", self.state.joint_q, fleet.q_indices)
        link.watch("joint_qd", self.state.joint_qd, fleet.qd_indices)

    def sense(self):
        """Evaluate photo-eyes and readbacks after the frame's physics; graph-capturable."""
        self.observation.zero_()
        self.case_hits.zero_()
        wp.launch(
            _photoeyes,
            dim=(len(self.origins), (len(self.bottles) + _PHOTOEYE_BATCH_SIZE - 1) // _PHOTOEYE_BATCH_SIZE, 32),
            block_dim=32,
            inputs=[
                self.state.body_q,
                self.bottles,
                self.bottle_count,
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
            dim=(
                len(self.case_origins),
                (len(self.case_shapes) + _CASE_PHOTOEYE_BATCH_SIZE - 1) // _CASE_PHOTOEYE_BATCH_SIZE,
                32,
            ),
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
        self.infeed.sense()
        self.transport.sense()
        self.wrapper.sense()
        self.link.gather()

    def after_step(self):
        """Run the host PLCs on the previous frame while the GPU steps this one.

        The sensors and the inlet, case-line and sealer PLCs run on the device
        inside the frame. The transport, infeed, fleet and wrapper PLCs see the
        previous frame's sensors, and their commands reach the next frame.
        """
        self.link.publish()
        data = self.link.latest()
        commands = {
            name: plc.update(status) for name, plc, status in zip(self.names, self.plcs, data["inlet"], strict=True)
        }
        for line, sealer, status in zip(self.case_lines, self.sealers, data["cases"], strict=True):
            line.station, sealer.state = CASE_STATIONS[status[0]], SEALER_STATES[status[1]]
        self.case_clock += 1.0 / 60.0
        # The device inlet PLCs overwrite their own chain channels each frame.
        speed = np.ones(len(self.multipliers), dtype=np.float32)
        speed[self.infeed.channel_end :] = 0.0
        # All host controllers observe the same physical state.
        self.transport.before_step(self.case_clock, speed, data)
        self.infeed.before_step(self.case_clock, commands, speed, data)
        self.fleet.before_step(self.case_clock, data)
        self.wrapper.before_step(self.case_clock, speed, data)
        for name, channel in self.zone_channels.items():
            speed[channel] = self.infeed.zone_speed[name]
        link = self.link
        link.write(self.multipliers, speed)
        link.write(self.outfeed_ok, [self.transport.can_accept(name) for name in self.names])

    def simulate(self):
        """Update belt friction, valve drives, arm targets, and grasp constraints."""
        lines = self.case_lines_plc
        wp.launch(
            case_line_plcs,
            dim=8,
            inputs=[
                lines,
                self.case_hits.reshape((8, -1)),
                self.observation,
                self.sealed,
                self.outfeed_ok,
                self.seal_bodies,
                self.case_local,
            ],
            outputs=[
                self.case_speeds,
                self.case_closed,
                self.spawn_enabled,
                self.spawn_bodies,
                self.spawn_local,
                self.line_ready,
            ],
        )
        wp.launch(
            case_drive_targets,
            dim=len(self.case_dofs),
            inputs=[
                lines,
                self.case_record_cells,
                self.case_record_kinds,
                self.case_record_items,
                self.case_record_closed,
                self.case_record_initial,
                self.case_closed,
                self.case_modules.design.ROLLER_R,
            ],
            outputs=[self.case_positions, self.case_velocities],
        )
        wp.launch(
            _inlet_plcs,
            dim=8,
            inputs=[self.inlet, self.observation, self.line_ready],
            outputs=[self.valve_targets, self.ready, self.multipliers],
        )
        wp.launch(
            _seal_cases,
            dim=1,
            inputs=[
                self.solver.dynamics.data,
                self.state.body_q,
                self.state.body_qd,
                self.seal_bodies,
                lines.swap_request,
                self.sealed,
                self.seal_joints,
                self.seal_hinges,
                self.model.joint_enabled,
            ],
        )
        wp.launch(
            _attach_sealed_flaps,
            dim=len(self.case_index),
            inputs=[
                self.state.body_q,
                self.seal_bodies,
                self.sealed,
                self.flap_shapes,
                self.flaps_attached,
                self.model.shape_body,
                self.model.shape_transform,
                self.solver._shape_body,
            ],
        )
        self.transport.simulate()
        self.infeed.simulate()
        self.fleet.simulate()
        self.wrapper.simulate()
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
        lanes = 32 if self.state.body_q.device.is_cuda else 1
        wp.launch(
            _robots,
            dim=8 * lanes,
            inputs=[
                self.solver.dynamics.data,
                self.state.body_q,
                self.bottles,
                self.bottle_count,
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
                lanes,
            ],
        )

    def report(self):
        """Return inlet states and the number of physically released bottles."""
        for plc, status in zip(self.plcs, self.inlet.status.numpy(), strict=True):
            plc.update(status)
        return {
            "inlets": {
                name: {"state": plc.state, "fault": plc.fault, "groups": plc.groups_picked}
                for name, plc in zip(self.names, self.plcs, strict=True)
            },
            "placed_bottles": self.placed.numpy().tolist(),
            "robot_modes": self.mode.numpy().tolist(),
            "case_stations": [line.station for line in self.case_lines],
            "reserve_cartons": dict(zip(self.names, self.case_lines_plc.magazine.numpy().tolist(), strict=True)),
            "sealers": [sealer.state for sealer in self.sealers],
            "palletizers": self.transport.report(),
            "fleet": self.fleet.report(),
            "wrapper": self.wrapper.report(),
            "infeed": self.infeed.status,
            "bottles_fed": int(self.bottle_count.numpy()[0]) - len(self.scene.bottle_bodies),
            "sealed_cases": int(self.sealed.numpy().sum()),
        }
