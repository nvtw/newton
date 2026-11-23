# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical sensor and actuator bindings for the source infeed PLC."""

import json

import numpy as np
import warp as wp

from newton.examples.phoenx._bottling_io import _PHOTOEYE_BATCH_SIZE, _case_drives, _load_controller, _photoeyes
from newton.solvers import phoenx

_ROW_SIZE = 7
_BOTTLE_PITCH = 0.0616  # metres, from the IFM supply interface
BOTTLES_PER_MINUTE = 425.0  # rated line output, from the source line_full scenario
SINGLE_FILE_PITCH = 0.071  # metres, including the upstream clearance check.
_ZONE_RAMP = 1.0  # s, for a process feed zone's drive to start or stop


@wp.kernel
def _check_feed_space(
    poses: wp.array[wp.transform],
    bottles: wp.array[int],
    count: wp.array[int],
    inlet_inverse: wp.transform,
    row_size: int,
    blocked: wp.array[int],
):
    index = wp.tid()
    if index >= count[0] or count[0] == bottles.shape[0]:
        return
    body = bottles[index]
    frame = inlet_inverse * poses[body]
    # Bound the entire swept bottle, including one tipped across the feed slot.
    center = wp.transform_point(frame, wp.vec3(0.0, 0.0, 0.115))
    tip = wp.abs(wp.quat_rotate(wp.transform_get_rotation(frame), wp.vec3(0.0, 0.0, 0.115)))
    radius = 0.031
    pitch = wp.static(_BOTTLE_PITCH)
    if (
        wp.abs(center[0]) <= pitch * 0.6 + tip[0] + radius
        and wp.abs(center[1]) <= pitch * 0.5 * float(row_size) + 0.02 + tip[1] + radius
        and wp.abs(center[2] - 0.12) <= 0.12 + tip[2] + radius
    ):
        wp.atomic_max(blocked, 0, 1)


@wp.kernel
def _feed_row(
    dynamics: phoenx.DynamicsData,
    inlet: wp.transform,
    blocked: wp.array[int],
    count: wp.array[int],
    bottles: wp.array[int],
    original: int,
    row_size: int,
    yaws: wp.array[float],
    desc: phoenx.BodyDesc,
    source_shapes: wp.array[int],
    first_body: int,
    first_shape: int,
    groups: wp.array[int],
):
    if blocked[0] != 0:
        return
    for lane in range(wp.min(row_size, bottles.shape[0] - count[0])):
        slot = count[0] - original
        desc.pose = inlet * wp.transform(
            wp.vec3(0.0, (float(lane) - 0.5 * float(row_size - 1)) * wp.static(_BOTTLE_PITCH), 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaws[slot]),
        )
        body = phoenx.add_body(dynamics, desc)
        if body < 0:
            return
        start = first_shape + (body - first_body) * source_shapes.shape[0]
        for shape in range(source_shapes.shape[0]):
            groups[start + shape] = groups[source_shapes[shape]]
        bottles[count[0]] = body
        count[0] += 1


class BottleSupply:
    """Admit finite bottle stock to the active line entrance when it has room."""

    def __init__(self, scene, capacity, joints):
        builder = scene.builder
        source = scene.bottle_bodies[0]
        self.capacity = capacity
        self.original = len(scene.bottle_bodies)
        self.first_body = builder.body_count
        self.first_shape = builder.shape_count
        self.shapes = builder.body_shapes[source]
        self.process_inlet = scene.metadata.get("process_inlet")
        self.row_size = 1 if self.process_inlet is not None else _ROW_SIZE
        self.period = self.row_size * 60.0 / BOTTLES_PER_MINUTE
        self.desc = phoenx.BodyDesc()
        self.desc.mass = builder.body_mass[source]
        self.desc.inertia = builder.body_inertia[source]
        self.desc.com = builder.body_com[source]
        self.desc.flags = builder.body_flags[source]
        # The source stage can use global colliders in its sole physical world.
        self.desc.world = max(0, builder.body_world[source])
        builder.reserve_dynamic_capacity(bodies=capacity, joints=joints)
        shapes = bottle_shape_templates(builder, source, dormant=True)
        label_prefix = "WetEndSupply" if self.process_inlet is not None else "Infeed"
        for slot in range(capacity):
            body = self.first_body + slot
            label = f"/Plant/Sim/Product/Bottles/{label_prefix}_{slot:05d}"
            builder.body_label[body] = label
            # Separate dormant bounds for SAP. Group zero disables every contact until admission.
            builder.body_q[body] = wp.transform(
                (200.0 + slot % 64 * 0.12, 200.0 + slot // 64 * 0.12, -10.0), wp.quat_identity()
            )
            add_bottle_shapes(builder, body, label, shapes)

    def copy_masks(self, membership, filtered):
        """Give every reserve hull the same USD collection memberships as its live source hull."""
        for mask in (membership, filtered):
            mask[self.first_shape :].reshape(self.capacity, len(self.shapes))[:] = mask[self.shapes]

    def bind(self, plant, matrix, yaws):
        self.plant = plant
        if self.process_inlet is None:
            self.frame = wp.transform(
                np.asarray([0.12, 0.0, 0.9508]) @ matrix[:3, :3] + matrix[3, :3],
                wp.quat_from_matrix(wp.mat33(matrix[:3, :3].T)),
            )
        else:
            self.frame, velocity = self.process_inlet
            self.desc.velocity = wp.spatial_vector(*velocity, 0.0, 0.0, 0.0)
            self.period = max(self.period, SINGLE_FILE_PITCH / float(np.linalg.norm(velocity)))
        device = plant.model.device
        self.blocked = wp.zeros(1, dtype=int, device=device)
        self.source_shapes = wp.array(self.shapes, dtype=int, device=device)
        self.yaws = wp.array(yaws, dtype=float, device=device)
        self.last = -float("inf")
        self.blocked_since = None
        self.pending = False

    def request(self, time, covered):
        # The physical wet end holds several minutes of transit stock. Its
        # local clearance check provides backpressure from an occupied line.
        if self.process_inlet is not None or not covered:
            self.blocked_since = None
        elif self.blocked_since is None:
            self.blocked_since = time
        if time - self.last >= self.period:
            self.last = time
            self.pending = self.blocked_since is None or time - self.blocked_since < 1.5

    def admit(self):
        """Stage requested bottles immediately before the solver commits their body states."""
        if not self.pending:
            return
        self.pending = False
        plant = self.plant
        self.blocked.zero_()
        wp.launch(
            _check_feed_space,
            len(plant.bottles),
            inputs=[
                plant.state.body_q,
                plant.bottles,
                plant.bottle_count,
                wp.transform_inverse(self.frame),
                self.row_size,
                self.blocked,
            ],
        )
        wp.launch(
            _feed_row,
            1,
            inputs=[
                plant.solver.dynamics.data,
                self.frame,
                self.blocked,
                plant.bottle_count,
                plant.bottles,
                self.original,
                self.row_size,
                self.yaws,
                self.desc,
                self.source_shapes,
                self.first_body,
                self.first_shape,
                plant.model.shape_collision_group,
            ],
        )


def bottle_shape_templates(builder, source, *, dormant=False):
    """Reuse the source bottle's full collision and visual shapes for additional stock."""
    modes = builder.custom_attributes.get("phoenx:restitution_combine_mode")
    templates = []
    for shape in builder.body_shapes[source]:
        cfg = builder.ShapeConfig(
            margin=builder.shape_margin[shape],
            gap=builder.shape_gap[shape],
            mu=builder.shape_material_mu[shape],
            mu_static=builder.shape_material_mu_static[shape],
            restitution=builder.shape_material_restitution[shape],
        )
        cfg.flags = builder.shape_flags[shape]
        cfg.density = 0.0
        cfg.collision_group = 0 if dormant else builder.shape_collision_group[shape]
        suffix = builder.shape_label[shape].removeprefix(builder.body_label[source])
        templates.append(
            (
                builder.shape_world[shape],
                suffix,
                {
                    "type": builder.shape_type[shape],
                    "src": builder.shape_source[shape],
                    "scale": builder.shape_scale[shape],
                    "xform": builder.shape_transform[shape],
                    "cfg": cfg,
                    "color": builder.shape_color[shape],
                    "opacity": builder.shape_opacity[shape],
                    "custom_attributes": {modes.key: modes.values[shape]} if modes and shape in modes.values else None,
                },
            )
        )
    return templates


def add_bottle_shapes(builder, body, label, templates):
    """Allocate shapes in the same order as the source bottle."""
    for world, suffix, config in templates:
        shape = builder.add_shape(body=body, label=label + suffix, **config)
        builder.shape_world[shape] = world


class StandingQueue:
    """Detect a standing queue at a through-beam eye, as the source IFM's motion monitor does.

    The neck beam flickers between the necks of a moving file. A queue stands
    once the body beam has been blocked and the neck beam unchanged for 1.5 s.
    """

    def __init__(self):
        self.covered = None
        self.neck = None
        self.changed = 0.0

    def update(self, time, body, neck):
        if not body:
            self.covered = None
        elif self.covered is None:
            self.covered = time
        if neck != self.neck:
            self.neck, self.changed = neck, time
        return self.covered is not None and time - max(self.covered, self.changed) >= 1.5


class InfeedController:
    """Run source splitter oscillators, jam relief, and port backpressure."""

    def __init__(self, plant, vendor, first_channel, supply=None):
        self.plant = plant
        self.interface = json.loads((vendor / "ifm_v01/interface.json").read_text())
        module = _load_controller("_newton_berlin_infeed", vendor / "ifm_v01/controller.py")
        self.plc = module.InfeedController(
            self.interface, 1.0 / 60.0, packers_attached={p["port"]: True for p in self.interface["ports"]}
        )
        self.channels = {}
        self.belt_records = []
        for belt in self.interface["belts"]:
            channel = first_channel + len(self.belt_records)
            label = belt["prim"].replace(self.interface["root"], "/Plant/Line/IFM", 1)
            self.channels[label] = channel
            self.belt_records.append((channel, belt["drive"], self.interface["speeds_mps"][belt["speed_key"]]))
        next_channel = first_channel + len(self.belt_records)
        self.lane_channels = {name: next_channel + i for i, name in enumerate(plant.names)}
        self.channel_end = next_channel + len(plant.names)
        self.clear_since = dict.fromkeys(plant.names)
        matrix = plant.scene.frames["/Plant/Line/IFM"]
        self.supply = supply
        if supply is not None:
            scenario = _load_controller("_newton_berlin_scenario", vendor.parent / "scenario.py")
            runtime = json.loads(plant.scene.metadata.get("bottling_plant_runtime", "{}"))
            step = float(runtime.get("bottle_yaw_step_deg") or 0.0)
            # Follow the source bottle sequence independently of the solver's free-slot order.
            labels = (
                f"/Plant/Sim/Product/Bottles/B{i:05d}"
                for i in range(supply.original, supply.original + supply.capacity)
            )
            supply.bind(plant, matrix, np.radians([scenario.bottle_yaw(label, step) for label in labels]))
        origins, directions, distances = [], [], []
        self.sensor_slices = {}
        for name, sensor in self.interface["sensors"].items():
            start = len(origins)
            for ray in sensor["rays"]:
                origins.append(np.asarray(ray["origin"]) @ matrix[:3, :3] + matrix[3, :3])
                directions.append(np.asarray(ray["direction"]) @ matrix[:3, :3])
                distances.append(ray["distance"])
            self.sensor_slices[name] = slice(start, len(origins))
        # Process feed zones watch their own body and neck beams.
        self.zone_eyes = {}
        for name, (_, eye) in plant.scene.metadata.get("process_zones", {}).items():
            self.zone_eyes[name] = len(origins)
            for origin, direction, distance in eye:
                origins.append(np.asarray(origin))
                directions.append(np.asarray(direction))
                distances.append(distance)
        self.zone_monitor = {name: StandingQueue() for name in self.zone_eyes}
        self.zone_speed = dict.fromkeys(self.zone_eyes, 0.0)
        self.origins = wp.array(origins, dtype=wp.vec3)
        self.directions = wp.array(directions, dtype=wp.vec3)
        self.distances = wp.array(distances, dtype=float)
        self.observation = wp.zeros((len(origins), 1), dtype=float)
        joints = [
            plant.scene.joints[o["joint"].replace(self.interface["root"], "/Plant/Line/IFM", 1)]
            for o in self.interface["oscillators"]
        ]
        starts = plant.model.joint_qd_start.numpy()
        self.dofs = wp.array(starts[joints], dtype=int)
        self.coordinates = wp.array(plant.target_coordinates[starts[joints]], dtype=int)
        self.positions = wp.zeros(len(joints), dtype=float)
        self.velocities = wp.zeros(len(joints), dtype=float)
        self.state_coordinates = plant.model.joint_q_start.numpy()[joints]
        self.oscillators = [o["name"] for o in self.interface["oscillators"]]
        self.status = None
        plant.link.watch("infeed_hits", self.observation)
        plant.link.watch("joint_q", plant.state.joint_q, self.state_coordinates)

    def sense(self):
        """Evaluate the infeed photo-eyes on the device; graph-capturable."""
        plant = self.plant
        self.observation.zero_()
        wp.launch(
            _photoeyes,
            dim=(len(self.origins), (len(plant.bottles) + _PHOTOEYE_BATCH_SIZE - 1) // _PHOTOEYE_BATCH_SIZE, 32),
            block_dim=32,
            inputs=[
                plant.state.body_q,
                plant.bottles,
                plant.bottle_count,
                self.origins,
                self.directions,
                self.distances,
                plant.hull_mesh,
                plant.hull_xform,
                plant.hull_scale,
                1,
                self.observation,
            ],
        )

    def before_step(self, time, commands, multipliers, data):
        """Step the source infeed PLC on a host snapshot."""
        hits = data["infeed_hits"][:, 0] != 0
        if self.supply is not None:
            self.supply.request(time, hits[self.sensor_slices["PE-A1"].start])
        ports = {
            port["port"]: {
                "PACKER_READY": commands[port["packer"]]["packer_ready"],
                "PACKER_BACKLOG": commands[port["packer"]]["packer_backlog"],
            }
            for port in self.interface["ports"]
        }
        angles = np.degrees(data["joint_q"][self.state_coordinates])
        angles = (angles + 180.0) % 360.0 - 180.0
        output = self.plc.step(
            {
                "beams": {name: hits[span].tolist() for name, span in self.sensor_slices.items()},
                "ports": ports,
                "osc_angle": dict(zip(self.oscillators, angles, strict=True)),
            }
        )
        for channel, drive, nominal in self.belt_records:
            multipliers[channel] = output["speeds"][drive] / nominal
        # A process feed zone stops while its eye has seen a standing queue for
        # 1.5 s: body beam blocked, neck beam unchanged. Its drive ramps like a VFD.
        for name, ray in self.zone_eyes.items():
            stop = self.zone_monitor[name].update(time, hits[ray], hits[ray + 1])
            step = (1.0 / 60.0) / _ZONE_RAMP
            self.zone_speed[name] = float(np.clip(self.zone_speed[name] + (-step if stop else step), 0.0, 1.0))
        for name, channel in self.lane_channels.items():
            command = commands[name]
            if command["packer_backlog"] or not command["packer_ready"]:
                self.clear_since[name] = None
                multipliers[channel] = 0.0
            else:
                if self.clear_since[name] is None:
                    self.clear_since[name] = time
                multipliers[channel] = float(time - self.clear_since[name] >= 0.3)
        self.plant.link.write(self.positions, np.radians([output["osc"].get(name, 0.0) for name in self.oscillators]))
        self.status = output["state"]
        self.plc.events.clear()
        if len(self.plc.history) > 2000:
            del self.plc.history[:1000]

    def simulate(self):
        control = self.plant.control
        wp.launch(
            _case_drives,
            dim=len(self.dofs),
            inputs=[
                self.dofs,
                self.coordinates,
                self.positions,
                self.velocities,
                control.joint_target_q,
                control.joint_target_qd,
            ],
        )
