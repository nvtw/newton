# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical sensor and actuator bindings for the source infeed PLC."""

import json

import numpy as np
import warp as wp

from newton.examples.phoenx._bottling_io import _case_drives, _load_controller, _photoeyes


class InfeedController:
    """Run source splitter oscillators, jam relief, and port backpressure."""

    def __init__(self, plant, vendor, first_channel):
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
        origins, directions, distances = [], [], []
        self.sensor_slices = {}
        for name, sensor in self.interface["sensors"].items():
            start = len(origins)
            for ray in sensor["rays"]:
                origins.append(np.asarray(ray["origin"]) @ matrix[:3, :3] + matrix[3, :3])
                directions.append(np.asarray(ray["direction"]) @ matrix[:3, :3])
                distances.append(ray["distance"])
            self.sensor_slices[name] = slice(start, len(origins))
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
            dim=(len(self.origins), 32),
            block_dim=32,
            inputs=[
                plant.state.body_q,
                plant.bottles,
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
