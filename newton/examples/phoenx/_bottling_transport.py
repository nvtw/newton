# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind the source transport PLC and KR40 sequencers to the physical plant."""

import copy
import json

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_io import _case_drives, _case_photoeyes, _load_controller, _ray_box
from newton.solvers import phoenx


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
        self.integral = np.zeros((4, 4))
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

    def before_step(self, time, multipliers):
        plant, model = self.plant, self.plant.model
        self.hits.zero_()
        wp.launch(
            _case_photoeyes,
            dim=(len(self.eye_names), 32),
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
        beams = dict(zip(self.eye_names, self.hits.numpy() != 0, strict=True))
        self.beams = beams
        coordinates = plant.state.joint_q.numpy()
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
        held = self.held.numpy()
        commands = np.zeros(4, dtype=np.int32)
        for i, (name, palletizer) in enumerate(self.palletizers.items()):
            station = self.conveyors.stations[name]
            ready = self.conveyors.held(station["pick_eye"], 0.5) and not self.conveyors.run.get(
                station["spur"][-1], True
            )
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
            actual = [
                coordinates[self.q_starts[plant.scene.joints[f"/Plant/Line/Palletizers/{name}/KR40/Joints/{a}"]]]
                for a in ("A1", "A2", "A3", "A6")
            ]
            actual = np.asarray(actual)
            actual += 2.0 * np.pi * np.round((result["q"] - actual) / (2.0 * np.pi))
            self.integral[i] += 4.0 * (result["q"] - actual) / 60.0
            q.extend(result["q"] + self.integral[i])
            qd.extend(result["qd"])
            palletizer.events.clear()
        self.commands.assign(commands)
        self.positions.assign(np.asarray(q, dtype=np.float32))
        self.velocities.assign(np.asarray(qd, dtype=np.float32))
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
            name: {"state": p.state, "placed_cases": len(p.placed), "fault": p.fault}
            for name, p in self.palletizers.items()
        }
