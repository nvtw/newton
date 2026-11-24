# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct moving wrap-line assemblies from the source material meshes."""

import json

import numpy as np
import warp as wp

import newton
import newton.geometry
from newton.examples.phoenx._bottling_io import _CASE_PHOTOEYE_BATCH_SIZE, _case_photoeyes
from newton.solvers import phoenx


@wp.kernel
def _bond_cartons(
    dynamics: phoenx.DynamicsData,
    poses: wp.array[wp.transform],
    pallets: wp.array[int],
    cartons: wp.array2d[int],
    requests: wp.array[int],
    status: wp.array[int],
    joints: wp.array2d[int],
):
    """Create an idempotent batch of ideal wrap bonds without moving bodies."""
    line = wp.tid()
    if requests[line] != 0:
        status[line] = 1
        for slot in range(cartons.shape[1]):
            joints[line, slot] = -1
    if status[line] != 1:
        return
    parent = pallets[line]
    failed = wp.bool(False)
    for slot in range(cartons.shape[1]):
        child = cartons[line, slot]
        frame = wp.transform_multiply(wp.transform_inverse(poses[parent]), poses[child])
        desc = phoenx.joint_desc(parent, child, frame, wp.transform_identity())
        joint = phoenx.add_joint(dynamics, desc)
        joints[line, slot] = joint
        if joint < 0:
            failed = True
            break
    if failed:
        for slot in range(cartons.shape[1]):
            joint = joints[line, slot]
            if joint >= 0:
                phoenx.remove_joint(dynamics, joint)
                joints[line, slot] = -1
        status[line] = -1
    else:
        status[line] = 2


@wp.kernel
def _process_drives(
    driven: wp.array[int],
    dofs: wp.array[int],
    coordinates: wp.array[int],
    position: wp.array[float],
    velocity: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    """Apply rotor and lift targets only on frames where the process commands them."""
    i = wp.tid()
    if driven[i / 2] != 0:
        target_q[coordinates[i]] = position[i]
        target_qd[dofs[i]] = velocity[i]


def _angle_error(angle, reference):
    """Signed angle error in [-pi, pi); measured rotor angles are wrapped, targets accumulate turns."""
    return (angle - reference + np.pi) % (2 * np.pi) - np.pi


def _rotation_profile(time, distance, speed, acceleration):
    """A rest-to-rest trapezoid with continuous angle and bounded acceleration."""
    speed = min(speed, np.sqrt(distance * acceleration))
    ramp = speed / acceleration
    cruise = max(0.0, distance / speed - ramp)
    duration = 2 * ramp + cruise
    t = float(np.clip(time, 0.0, duration))
    if t < ramp:
        angle, velocity = 0.5 * acceleration * t * t, acceleration * t
    elif t < ramp + cruise:
        angle, velocity = 0.5 * speed * ramp + speed * (t - ramp), speed
    else:
        remaining = duration - t
        angle, velocity = distance - 0.5 * acceleration * remaining * remaining, acceleration * remaining
    return angle, velocity, duration


def _rotate_vectors(quaternion, vectors):
    """Rotate NumPy batches by the simulation's xyzw quaternions."""
    xyz = quaternion[..., :3]
    twice_cross = 2.0 * np.cross(xyz, vectors)
    return vectors + quaternion[..., 3:] * twice_cross + np.cross(xyz, twice_cross)


def _payload_seated(lower, upper, deck, pallet_lo, pallet_hi, layer_heights):
    """Require a complete footprint-contained stack supported from the deck up."""
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return False
    if np.any(lower[:, :2] < pallet_lo[:2] - 0.01) or np.any(upper[:, :2] > pallet_hi[:2] + 0.01):
        return False
    heights, counts = np.unique(layer_heights, return_counts=True)
    assignment = np.argmin(abs(lower[:, 2, None] - heights[None, :]), axis=1)
    if np.any(abs(lower[:, 2] - heights[assignment]) > 0.01):
        return False
    if not np.array_equal(np.bincount(assignment, minlength=len(heights)), counts):
        return False
    for k in range(len(lower)):
        if abs(lower[k, 2] - deck) < 0.01:
            continue
        overlap = np.minimum(upper[:, :2], upper[k, :2]) - np.maximum(lower[:, :2], lower[k, :2])
        supports = (abs(upper[:, 2] - lower[k, 2]) < 0.01) & np.all(overlap > 0.0, axis=1)
        supports[k] = False
        # At least half the footprint must overlap supporting carton tops.
        area = np.prod(np.maximum(overlap[supports], 0.0), axis=1).sum()
        if area < 0.5 * np.prod(upper[k, :2] - lower[k, :2]):
            return False
    return True


class WrapperController:
    """Wrap with ideal carton bonds after measured machine motion, then dispatch.

    The bonds approximate applied film; film stress and labelling are not simulated.
    """

    def __init__(self, model, state, control, asset_root, channels, fleet, *, plant=None):
        self.model, self.state, self.control = model, state, control
        self.channels, self.fleet = channels, fleet
        self.plant = plant
        self.lines = {}
        origins, directions, distances = [], [], []
        self.pallets = np.asarray(fleet.pallet_bodies)
        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()
        shapes = np.flatnonzero(
            np.isin(shape_body, self.pallets) & ((shape_flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)
        )
        kinds, scales = model.shape_type.numpy(), model.shape_scale.numpy()
        if np.any(kinds[shapes] != int(newton.GeoType.BOX)):
            raise ValueError("Wrapper pallet photoeyes require the authored box colliders")
        self.shapes = wp.array(shapes, dtype=int)
        self.bounds_lo = wp.array(-scales[shapes], dtype=wp.vec3)
        self.bounds_hi = wp.array(scales[shapes], dtype=wp.vec3)
        transforms = model.shape_transform.numpy()
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        self.pallet_corners = {}
        for body in self.pallets:
            points = []
            for shape in shapes[shape_body[shapes] == body]:
                transform = wp.transform(transforms[shape, :3], transforms[shape, 3:])
                points.extend(np.asarray(wp.transform_point(transform, wp.vec3(p))) for p in corners * scales[shape])
            self.pallet_corners[body] = np.asarray(points)
        for name in ("WL_U", "WL_D"):
            if not any(f"/{name}/" in label for label in channels):
                continue
            spec = json.loads((asset_root / f"plant/lines_1x8/wrapline/{name}.json").read_text())
            if not any(zone["prim"] in channels for zone in spec["zones"].values()):
                continue
            joint = model.joint_label.index(spec["unit_root"] + "/Wrapper/rotation")
            self.lines[name] = {
                "spec": spec,
                "state": "WAIT_DROP",
                "pallet": None,
                "fault": None,
                "wrapped": False,
                "labelling": "not simulated",
                "since": 0.0,
                "quiet_since": None,
                "rotor_quiet_since": None,
                "rotor_q": int(model.joint_q_start.numpy()[joint]),
                "rotor_qd": int(model.joint_qd_start.numpy()[joint]),
            }
            for sensor in ("PE_DROP", "PE_WRAP", "PE_PICK"):
                beam = spec["sensors"][sensor]
                origins.append(beam["origin"])
                directions.append(beam["direction"])
                distances.append(beam["distance"])
        self.origins = wp.array(origins, dtype=wp.vec3)
        self.directions = wp.array(directions, dtype=wp.vec3)
        self.distances = wp.array(distances, dtype=float)
        self.hits = wp.zeros(len(origins), dtype=int)
        if plant is not None and self.lines:
            self._setup_process(asset_root)

    def _setup_process(self, asset_root):
        plant, model = self.plant, self.model
        self.carton_roots = plant.seal_bodies.numpy()[:, 0]
        library = json.loads(
            (
                asset_root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor/ctp_v01/kr40_motions.json"
            ).read_text()
        )
        slots = library[next(iter(library))]["meta"]["slots"]
        self.layer_heights = np.asarray([slot["bottom_z"] for slot in slots])
        count = len(slots)
        self.bond_pallets = wp.full(len(self.lines), -1, dtype=int)
        self.bond_cartons = wp.full((len(self.lines), count), -1, dtype=int)
        self.bond_status = wp.zeros(len(self.lines), dtype=int)
        self.bond_joints = wp.full((len(self.lines), count), -1, dtype=int)
        self.bond_requests = wp.zeros(len(self.lines), dtype=int)
        self.driven = wp.zeros(len(self.lines), dtype=int)
        self.bond_pallet_rows = self.bond_pallets.numpy()
        self.bond_carton_rows = self.bond_cartons.numpy()
        # Rotor and lift targets per line, written only while the process drives them.
        self.drive_dofs = wp.zeros(2 * len(self.lines), dtype=int)
        self.drive_coordinates = wp.zeros(2 * len(self.lines), dtype=int)
        self.drive_positions = wp.zeros(2 * len(self.lines), dtype=float)
        self.drive_velocities = wp.zeros(2 * len(self.lines), dtype=float)
        self.wrapped = set()
        self.carton_corners = []
        shape_body = model.shape_body.numpy()
        transforms = model.shape_transform.numpy()
        shapes = plant.case_shapes.numpy()
        lo, hi = plant.case_bounds_lo.numpy(), plant.case_bounds_hi.numpy()
        signs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        for body in self.carton_roots:
            points = []
            for i in np.flatnonzero(shape_body[shapes] == body):
                transform = wp.transform(transforms[shapes[i], :3], transforms[shapes[i], 3:])
                points.extend(
                    np.asarray(wp.transform_point(transform, wp.vec3(p))) for p in lo[i] + signs * (hi[i] - lo[i])
                )
            points = np.asarray(points)
            self.carton_corners.append(points.min(0) + signs * (points.max(0) - points.min(0)))
        self.carton_corners = np.asarray(self.carton_corners)
        qstarts, dstarts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        target_starts = model.joint_target_q_start.numpy()
        initial = model.body_q.numpy()
        masses, coms = model.body_mass.numpy(), model.body_com.numpy()
        inertias = model.body_inertia.numpy()
        for line in self.lines.values():
            prefix = line["spec"]["unit_root"] + "/Wrapper/"
            rotor, lift = [model.joint_label.index(prefix + suffix) for suffix in ("rotation", "lift")]
            rbody, lbody = model.joint_child.numpy()[[rotor, lift]]
            pivot = initial[rbody, :3]
            inertia = 0.0
            for body in (rbody, lbody):
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(initial[body, 3:]))).reshape(3, 3)
                offset = initial[body, :3] + rotation @ coms[body] - pivot
                inertia += (rotation @ inertias[body] @ rotation.T)[2, 2] + masses[body] * np.dot(
                    offset[:2], offset[:2]
                )
            dof = int(dstarts[rotor])
            torque = min(
                0.5 * model.joint_effort_limit.numpy()[dof], 0.5 * model.joint_target_ke.numpy()[dof] * np.radians(3.0)
            )
            acceleration = float(torque / inertia)
            if acceleration <= 0:
                raise ValueError("Wrapper requires positive rotor effort and inertia")
            lift_dof = int(dstarts[lift])
            line.update(
                lift_q=int(qstarts[lift]),
                lift_qd=lift_dof,
                rotor_target=int(target_starts[rotor]),
                lift_target=int(target_starts[lift]),
                acceleration=acceleration,
                lift_upper=float(model.joint_limit_upper.numpy()[lift_dof]),
                lift_speed=float(model.joint_velocity_limit.numpy()[lift_dof]),
                turns=0.0,
                wrapped=False,
                labelling="not simulated",
            )
        self.body_com = coms
        lines = self.lines.values()
        self.drive_dofs.assign([line[key] for line in lines for key in ("rotor_qd", "lift_qd")])
        self.drive_coordinates.assign([line[key] for line in lines for key in ("rotor_target", "lift_target")])
        link = plant.link
        link.watch("wrapper_hits", self.hits)
        link.watch("bond_status", self.bond_status)
        link.watch("body_q", self.state.body_q, np.r_[self.pallets, self.carton_roots])
        link.watch("body_qd", self.state.body_qd, np.r_[self.pallets, self.carton_roots])
        link.watch("claimed", plant.claimed, self.carton_roots)
        link.watch("joint_q", self.state.joint_q, [line[k] for line in lines for k in ("rotor_q", "lift_q")])
        link.watch("joint_qd", self.state.joint_qd, [line[k] for line in lines for k in ("rotor_qd", "lift_qd")])
        link.watch(
            "joint_target_q",
            self.control.joint_target_q,
            [line[k] for line in lines for k in ("rotor_q", "rotor_target", "lift_target")],
        )
        link.watch(
            "joint_target_qd",
            self.control.joint_target_qd,
            [line[k] for line in lines for k in ("rotor_qd", "lift_qd")],
        )

    def _load(self, line, poses, velocities):
        """Validate measured sealed cartons, never substitute layout stand-ins."""
        body = line["pallet"]
        pallet_pose = poses[body]
        pallet_rotation = pallet_pose[3:]
        inverse_rotation = pallet_rotation * [-1, -1, -1, 1]
        roots = self.carton_roots
        local = _rotate_vectors(inverse_rotation, poses[roots, :3] - pallet_pose[:3])
        pallet = self.pallet_corners[body]
        lo, hi = pallet.min(0), pallet.max(0)
        near = np.all((local[:, :2] > lo[:2] - 0.2) & (local[:, :2] < hi[:2] + 0.2), axis=1)
        near &= (local[:, 2] > hi[2] - 0.05) & (local[:, 2] < self.layer_heights.max() + 0.3)
        selected = np.flatnonzero(near)
        if len(selected) != len(self.layer_heights):
            return None
        cartons = roots[selected]
        if np.any(self.data["sealed"][selected] == 0) or np.any(self.data["claimed"][cartons] != 0):
            return None
        carton_rotation = poses[cartons, 3:]
        up = np.array([0.0, 0.0, 1.0])
        normal = _rotate_vectors(pallet_rotation, up)
        if not np.all(_rotate_vectors(carton_rotation, up) @ normal >= np.cos(np.radians(3.0))):
            return None
        corners = _rotate_vectors(carton_rotation[:, None], self.carton_corners[selected])
        carton_com = _rotate_vectors(carton_rotation, self.body_com[cartons])
        pallet_com = _rotate_vectors(pallet_rotation, self.body_com[body])
        delta = (poses[cartons, :3] - pallet_pose[:3]) + carton_com - pallet_com
        speed = velocities[cartons, :3] - velocities[body, :3] - np.cross(velocities[body, 3:], delta)
        angular = velocities[cartons, 3:] - velocities[body, 3:]
        # Rigid point velocity is affine, so its maximum norm on the carton is at a corner.
        corner_velocities = speed[:, None] + np.cross(angular[:, None], corners - carton_com[:, None])
        if not np.all(np.linalg.norm(corner_velocities, axis=2) <= 0.01):
            return None
        points = local[selected, None] + _rotate_vectors(inverse_rotation, corners)
        if not _payload_seated(points.min(1), points.max(1), hi[2], lo, hi, self.layer_heights):
            return None
        if any(int(carton) in self.wrapped for carton in cartons):
            return None
        return selected

    def _process(self, index, line, time, speed, poses, coordinates, velocities, hits, joint_velocity, targets):
        if self.plant is None:
            return
        state, spec = line["state"], line["spec"]
        pick_port = spec["item"] + "_pick"
        busy = any(pick_port in (m.source, m.destination) for m in self.fleet.missions.values())
        if state == "WAIT_WRAP_PROCESS":
            selected = self._load(line, poses, velocities)
            if selected is None:
                line["load_quiet_since"] = None
                return
            if line.get("load_quiet_since") is None:
                line["load_quiet_since"] = time
            if time - line["load_quiet_since"] < 0.5:
                return
            line.update(
                state="WRAPPING",
                since=time,
                cartons=selected,
                home=float(coordinates[line["rotor_q"]]),
                angle=float(coordinates[line["rotor_q"]]),
                turns=0.0,
                reference=None,
            )
            state = "WRAPPING"
        if state == "WRAPPING":
            # Source palletflow/sources.py: nine turns at 15 rpm plus clamp/cut.
            angle, rate, duration = _rotation_profile(time - line["since"], 18 * np.pi, np.pi / 2, line["acceleration"])
            t = np.clip(time - line["since"], 0, duration)
            lift = max(0.0, line["lift_upper"])
            lift = min(lift, line["lift_speed"] * duration / np.pi)
            targets[line["rotor_target"]] = line["home"] + angle
            targets[line["lift_target"]] = lift * (1 - np.cos(2 * np.pi * t / duration)) / 2
            self.process_speeds[line["rotor_qd"]] = rate
            self.process_speeds[line["lift_qd"]] = lift * np.pi / duration * np.sin(2 * np.pi * t / duration)
            self.driven_lines.add(index)
            # Commands reach the plant one frame after the measured state, so
            # compare against the previous command, which leads it as before.
            reference = line["reference"] or (targets[line["rotor_target"]], targets[line["lift_target"]])
            line["reference"] = (targets[line["rotor_target"]], targets[line["lift_target"]])
            measured = float(coordinates[line["rotor_q"]])
            line["turns"] += _angle_error(measured, line["angle"])
            line["angle"] = measured
            error = _angle_error(measured, reference[0])
            if abs(error) > np.radians(3) or abs(coordinates[line["lift_q"]] - reference[1]) > 0.01:
                line.update(state="FAULT", fault="Wrapper motor left its tracking envelope")
            elif (
                t >= duration
                and abs(line["turns"] - 18 * np.pi) < np.radians(2)
                and abs(joint_velocity[line["rotor_qd"]]) < 0.02
                and abs(coordinates[line["lift_q"]]) < 0.005
                and abs(joint_velocity[line["lift_qd"]]) < 0.01
            ):
                line.update(state="CLAMP_CUT", since=time)
            elif time - line["since"] > duration + 5:
                line.update(state="FAULT", fault="Wrapper did not finish the measured turns")
        elif state == "CLAMP_CUT" and time - line["since"] >= 6.0:
            selected = self._load(line, poses, velocities)
            if selected is None or not np.array_equal(selected, line["cartons"]):
                line.update(state="FAULT", fault="Cartons moved before wrap bonds were applied")
                return
            self.bond_pallet_rows[index] = line["pallet"]
            self.bond_carton_rows[index] = self.carton_roots[selected]
            self.plant.link.write(self.bond_pallets, self.bond_pallet_rows)
            self.plant.link.write(self.bond_cartons, self.bond_carton_rows)
            self.requests[index] = 1
            line["bond_stamp"] = self.plant.link.frame
            inverse = wp.transform_inverse(wp.transform(poses[line["pallet"], :3], poses[line["pallet"], 3:]))
            line["bond_frames"] = [
                wp.transform_multiply(inverse, wp.transform(poses[root, :3], poses[root, 3:]))
                for root in self.carton_roots[selected]
            ]
            line.update(state="BONDING", since=time)
        elif state == "BONDING":
            if not self.plant.link.observed(line["bond_stamp"]):
                return
            status = int(self.data["bond_status"][index])
            if status < 0:
                line.update(state="FAULT", fault="Wrap bond allocation failed")
            elif status == 2:
                roots = self.carton_roots[line["cartons"]]
                if not np.isfinite(poses[np.r_[line["pallet"], roots]]).all():
                    line.update(state="FAULT", fault="Non-finite wrapped load")
                    return
                inverse = wp.transform_inverse(wp.transform(poses[line["pallet"], :3], poses[line["pallet"], 3:]))
                for root, expected in zip(roots, line["bond_frames"], strict=True):
                    actual = wp.transform_multiply(inverse, wp.transform(poses[root, :3], poses[root, 3:]))
                    if np.linalg.norm(np.asarray(actual.p) - np.asarray(expected.p)) > 0.001 or abs(
                        np.dot(np.asarray(actual.q), np.asarray(expected.q))
                    ) < np.cos(np.radians(0.5)):
                        line.update(state="FAULT", fault="Wrapped carton bond did not hold its current frame")
                        return
                self.wrapped.update(int(self.carton_roots[i]) for i in line["cartons"])
                line.update(state="WAIT_OUTFEED", wrapped=True, since=time)
        elif state == "WAIT_OUTFEED":
            if not busy and self.fleet._port_observation(pick_port, poses).clear:
                line.update(state="OUTFEED", since=time)
        elif state == "OUTFEED":
            if busy:
                line.update(state="FAULT", fault="Wrapper pickup became reserved during outfeed")
            elif hits[2]:
                line.update(state="PICK_SETTLING", since=time)
            elif time - line["since"] > spec["timers_s"]["outfeed_timeout_s"]:
                line.update(state="FAULT", fault="Wrapped pallet did not reach pickup photoeye")
            else:
                for zone in ("Z3", "Z4", "Z5"):
                    speed[self.channels[spec["zones"][zone]["prim"]]] = 1.0
        elif state == "PICK_SETTLING":
            body = line["pallet"]
            seated = self.fleet._port_observation(pick_port, poses).pallet
            expected = int(np.flatnonzero(self.pallets == body)[0])
            radius = np.linalg.norm(self.pallet_corners[body], axis=1).max()
            quiet = np.linalg.norm(velocities[body, :3]) + radius * np.linalg.norm(velocities[body, 3:]) < 0.01
            if hits[2] and seated == expected and quiet and not busy:
                if line.get("pick_quiet_since") is None:
                    line["pick_quiet_since"] = time
                if time - line["pick_quiet_since"] >= spec["timers_s"]["settle_s"]:
                    self.fleet.notify_wrapper_ready(spec["item"][-1], body, poses)
                    line["state"] = "PICK_READY"
            else:
                line["pick_quiet_since"] = None
                if time - line["since"] > 5:
                    line.update(state="FAULT", fault="Wrapped pallet did not settle at pickup")
        elif (
            state == "PICK_READY" and not busy and not hits[2] and self.fleet._port_observation(pick_port, poses).clear
        ):
            line.update(
                state="WAIT_DROP",
                pallet=None,
                quiet_since=None,
                load_quiet_since=None,
                pick_quiet_since=None,
                wrapped=False,
            )

    def simulate(self):
        if self.plant is not None and self.lines:
            wp.launch(
                _bond_cartons,
                dim=len(self.lines),
                inputs=[
                    self.plant.solver.dynamics.data,
                    self.state.body_q,
                    self.bond_pallets,
                    self.bond_cartons,
                    self.bond_requests,
                    self.bond_status,
                    self.bond_joints,
                ],
            )
            wp.launch(
                _process_drives,
                dim=len(self.drive_dofs),
                inputs=[
                    self.driven,
                    self.drive_dofs,
                    self.drive_coordinates,
                    self.drive_positions,
                    self.drive_velocities,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )

    def sense(self):
        """Evaluate the pallet photo-eyes on the device; graph-capturable."""
        if not self.lines:
            return
        self.hits.zero_()
        model = self.model
        wp.launch(
            _case_photoeyes,
            dim=(len(self.hits), (len(self.shapes) + _CASE_PHOTOEYE_BATCH_SIZE - 1) // _CASE_PHOTOEYE_BATCH_SIZE, 32),
            block_dim=32,
            inputs=[
                self.state.body_q,
                model.shape_body,
                model.shape_transform,
                model.shape_scale,
                model.shape_type,
                model.shape_source_ptr,
                self.shapes,
                self.bounds_lo,
                self.bounds_hi,
                self.origins,
                self.directions,
                self.distances,
                self.hits,
            ],
        )

    def observe(self):
        """Evaluate and read back all wrapper inputs synchronously, for use without a plant."""
        self.sense()
        return {
            "wrapper_hits": self.hits.numpy(),
            "body_q": self.state.body_q.numpy(),
            "body_qd": self.state.body_qd.numpy(),
            "joint_q": self.state.joint_q.numpy(),
            "joint_qd": self.state.joint_qd.numpy(),
            "joint_target_q": self.control.joint_target_q.numpy(),
            "joint_target_qd": self.control.joint_target_qd.numpy(),
        }

    def before_step(self, time, speed, data):
        """Use actual pallet rays and motor state; never infer completion from a timer."""
        for channel in self.channels.values():
            speed[channel] = 0.0
        if not self.lines:
            return
        self.data = data
        hits = data["wrapper_hits"].reshape(-1, 3)
        poses, coordinates = data["body_q"], data["joint_q"]
        velocities = data["body_qd"]
        targets = data["joint_target_q"]
        joint_velocity = data["joint_qd"]
        self.process_speeds = data["joint_target_qd"]
        self.driven_lines = set()
        self.requests = np.zeros(len(self.lines), dtype=np.int32)
        for index, (name, line) in enumerate(self.lines.items()):
            spec = line["spec"]
            port = name + "_drop"
            reserved = any(port in (mission.source, mission.destination) for mission in self.fleet.missions.values())
            # Each wrap advances the target by whole turns; compare wrapped angles.
            rotor_home = abs(_angle_error(coordinates[line["rotor_q"]], targets[line["rotor_q"]])) < np.radians(2.0)
            rotor_home = rotor_home and abs(joint_velocity[line["rotor_qd"]]) < 0.02
            if line["state"] == "WAIT_DROP":
                if reserved or not rotor_home:
                    line["rotor_quiet_since"] = None
                    continue
                if line["rotor_quiet_since"] is None:
                    line["rotor_quiet_since"] = time
                if time - line["rotor_quiet_since"] < spec["timers_s"]["settle_s"]:
                    continue
                observed = self.fleet._port_observation(port, poses)
                if not hits[index, 0] or observed.pallet is None:
                    continue
                body = int(self.pallets[observed.pallet])
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(poses[body, 3:]))).reshape(3, 3)
                front = (self.pallet_corners[body] @ rotation.T)[:, 0].max()
                target_x = spec["sensors"]["PE_WRAP"]["origin"][0] - front
                others = self.pallets[self.pallets != body]
                blocked = (np.abs(poses[others, 0] - target_x) < 1.3) & (
                    np.abs(poses[others, 1] - spec["ports"]["drop"]["xy"][1]) < 0.8
                )
                if np.any(blocked):
                    continue
                line.update(state="INFEED", pallet=body, since=time, target_x=target_x, rotor_quiet_since=None)
            if line["state"] == "INFEED":
                if reserved or not rotor_home:
                    line.update(state="FAULT", fault="Infeed lost truck clearance or rotor home")
                elif hits[index, 1]:
                    line.update(state="SETTLING", since=time)
                elif time - line["since"] > spec["timers_s"]["infeed_timeout_s"]:
                    line.update(state="FAULT", fault="Pallet did not reach the wrap photoeye")
                else:
                    for zone in ("Z1", "Z2", "Z3"):
                        speed[self.channels[spec["zones"][zone]["prim"]]] = 1.0
            if line["state"] in ("SETTLING", "WAIT_WRAP_PROCESS"):
                body = line["pallet"]
                pose, velocity = poses[body], velocities[body]
                up = 1.0 - 2.0 * (pose[3] ** 2 + pose[4] ** 2)
                yaw = np.arctan2(2 * (pose[6] * pose[5] + pose[3] * pose[4]), 1 - 2 * (pose[4] ** 2 + pose[5] ** 2))
                yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
                positioned = (
                    abs(pose[0] - line["target_x"]) < 0.01
                    and abs(pose[1] - spec["ports"]["drop"]["xy"][1]) < 0.03
                    and abs(pose[2] - spec["ports"]["drop"]["z"]) < 0.03
                    and up > np.cos(np.radians(3.0))
                    and abs(yaw) < np.radians(3.0)
                )
                radius = np.linalg.norm(self.pallet_corners[body], axis=1).max()
                quiet = np.linalg.norm(velocity[:3]) + radius * np.linalg.norm(velocity[3:]) < 0.01
                if positioned and quiet and hits[index, 1]:
                    if line["quiet_since"] is None:
                        line["quiet_since"] = time
                    if time - line["quiet_since"] >= 0.5:
                        line["state"] = "WAIT_WRAP_PROCESS"
                else:
                    line["quiet_since"] = None
                    if time - line["since"] > 5.0:
                        line.update(state="FAULT", fault="Pallet did not settle at the measured wrap position")

            self._process(
                index, line, time, speed, poses, coordinates, velocities, hits[index], joint_velocity, targets
            )
        for index, line in enumerate(self.lines.values()):
            if line["state"] == "FAULT" and "rotor_target" in line:
                targets[line["rotor_target"]] = coordinates[line["rotor_q"]]
                targets[line["lift_target"]] = coordinates[line["lift_q"]]
                self.process_speeds[line["rotor_qd"]] = 0.0
                self.process_speeds[line["lift_qd"]] = 0.0
                self.driven_lines.add(index)
        if self.plant is not None:
            lines = self.lines.values()
            link = self.plant.link
            link.write(self.driven, [int(i in self.driven_lines) for i in range(len(self.lines))])
            link.write(
                self.drive_positions, [targets[line[k]] for line in lines for k in ("rotor_target", "lift_target")]
            )
            link.write(
                self.drive_velocities, [self.process_speeds[line[k]] for line in lines for k in ("rotor_qd", "lift_qd")]
            )
            link.write(self.bond_requests, self.requests)

    def report(self):
        """Report physical process status and the explicit labelling limitation."""
        return {
            name: {key: line[key] for key in ("state", "pallet", "fault", "wrapped", "labelling")}
            for name, line in self.lines.items()
        }


def mesh_components(mesh):
    """Separate disconnected surfaces while retaining vertex seams and materials.

    Coincident positions connect faces for component discovery only. Original
    vertices, normals, UVs, and winding remain intact in each returned mesh.
    Components follow their first source triangle, making selection repeatable.
    """
    triangles = np.asarray(mesh.indices).reshape(-1, 3)
    _, canonical = np.unique(mesh.vertices, axis=0, return_inverse=True)
    parents = np.arange(len(mesh.vertices))

    def root(vertex):
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    for a, b, c in canonical[triangles]:
        parent = root(a)
        parents[root(b)] = parent
        parents[root(c)] = parent

    groups = {}
    for index, triangle in enumerate(triangles):
        groups.setdefault(root(canonical[triangle[0]]), []).append(index)

    parts = []
    for faces in groups.values():
        vertices, remapped = np.unique(triangles[faces], return_inverse=True)
        parts.append(
            newton.Mesh(
                mesh.vertices[vertices],
                remapped,
                normals=mesh.normals[vertices] if mesh.normals is not None else None,
                uvs=mesh.uvs[vertices] if mesh.uvs is not None else None,
                color=mesh.color,
                roughness=mesh.roughness,
                metallic=mesh.metallic,
                texture=mesh.texture,
                texture_transform=mesh.texture_transform,
                opacity=mesh.opacity,
                is_solid=mesh.is_solid,
                compute_inertia=False,
            )
        )
    return parts


def _wrapper_groups(components, label):
    """Identify the source hub, carriage, and their material components."""
    centers = np.array([(m.vertices.min(axis=0) + m.vertices.max(axis=0)) * 0.5 for _, m in components])
    sizes = np.array([m.vertices.max(axis=0) - m.vertices.min(axis=0) for _, m in components])
    hubs = [
        i
        for i, (name, _) in enumerate(components)
        if name == "steel" and np.allclose(sizes[i], (0.68, 0.68, 0.117), atol=0.002)
    ]
    rolls = [i for i, (name, _) in enumerate(components) if name == "wrapline_film_roll"]
    if len(hubs) != 1 or len(rolls) != 1:
        raise ValueError(f"{label}: source rotary hub and film roll must be uniquely identifiable")
    pivot, lift_origin = centers[hubs[0]], centers[rolls[0]]
    assignment = np.full(len(components), -1, dtype=np.int32)
    for i, (name, _) in enumerate(components):
        center, size = centers[i], sizes[i]
        near_lift = np.linalg.norm(center[:2] - lift_origin[:2]) < 0.75
        if name in ("wrapline_film_roll", "wrapline_rubber", "arch_aluminium", "wrapline_product_film"):
            assignment[i] = 1
        elif name == "black":
            assignment[i] = 0
        elif name == "steel":
            if abs(center[2] - pivot[2]) < 0.2 or size[2] > 2.0:
                assignment[i] = 0
            elif near_lift and abs(center[2] - lift_origin[2]) < 0.4:
                assignment[i] = 1
        elif name == "machine" and near_lift and size[2] < 0.1:
            assignment[i] = 1
    return centers, sizes, pivot, lift_origin, assignment


def correct_wrapper_frame(components, *, label):
    """Raise the source's intersecting fixed frame, leaving the mechanism intact.

    The two authored 120 x 2850 x 250 mm crossmembers intersect the rotary
    hub. This explicit reconstruction extends the posts and raises the fixed
    upper frame/motor to provide 10 mm clearance. The ideal revolute bearing
    spans the increased gap; no shaft mass or stiffness is invented.
    """
    _, sizes, _, _, assignment = _wrapper_groups(components, label)
    beams = [
        i
        for i, (name, _) in enumerate(components)
        if assignment[i] < 0 and name == "machine" and np.allclose(sizes[i], (0.12, 2.85, 0.25), atol=0.002)
    ]
    if len(beams) != 2:
        raise ValueError(f"{label}: source upper-frame crossmembers must be uniquely identifiable")
    beam_bottom = min(components[i][1].vertices[:, 2].min() for i in beams)
    rotor_top = max(mesh.vertices[:, 2].max() for i, (_, mesh) in enumerate(components) if assignment[i] == 0)
    clearance = 0.01
    rise = max(0.0, float(rotor_top - beam_bottom + clearance))
    corrected = []
    for i, (name, mesh) in enumerate(components):
        transformed = mesh
        if assignment[i] < 0:
            vertices = mesh.vertices.copy()
            vertices[vertices[:, 2] >= beam_bottom - 1.0e-5, 2] += rise
            transformed = mesh.copy(vertices=vertices)
        corrected.append((name, transformed))
    return corrected, {
        "raise_m": rise,
        "rotor_clearance_m": clearance,
        "collision_gap_m": 0.002,
        "bearing_gap_increase_m": rise,
        "bearing_model": "ideal revolute joint; no reconstructed shaft mass or stiffness",
        "source_frame_modified": rise > 0.0,
    }


def build_wrapper(builder, components, *, label, visuals=True):
    """Build a rotary arm, lift, and fixed frame from world-space components.

    Masses are estimates: metal covers use 3 mm shells (steel 7850 and
    aluminium 2700 kg/m3); the roll and rubber use solid densities 920 and
    1100 kg/m3. These assumptions are explicit because the source wrapper
    has visual geometry but no authored masses, joints, or drive limits.
    Reconstructed colliders use a 2 mm contact gap, not continuous collision
    detection. Returns body/joint IDs and each source component's body group.
    """
    centers, sizes, pivot, lift_origin, assignment = _wrapper_groups(components, label)

    bodies = []
    for group, origin in enumerate((pivot, lift_origin)):
        masses, centers_of_mass, inertias = [], [], []
        for i, (name, mesh) in enumerate(components):
            if assignment[i] != group or name == "wrapline_product_film":
                continue
            local = newton.Mesh(mesh.vertices - origin, mesh.indices, compute_inertia=False)
            density = {"arch_aluminium": 2700.0, "wrapline_film_roll": 920.0, "wrapline_rubber": 1100.0}.get(
                name, 7850.0
            )
            solid = name in ("wrapline_film_roll", "wrapline_rubber")
            mass, com, inertia = newton.geometry.compute_inertia_shape(
                newton.GeoType.MESH,
                (1.0, 1.0, 1.0),
                local,
                density,
                is_solid=solid,
                thickness=0.003,
            )
            masses.append(mass)
            centers_of_mass.append(np.asarray(com))
            inertias.append(np.asarray(inertia).reshape(3, 3))
        total = sum(masses)
        if total <= 0.0:
            raise ValueError(f"{label}: moving assembly has no positive mass")
        com = np.average(centers_of_mass, axis=0, weights=masses)
        inertia = np.zeros((3, 3))
        for mass, center, tensor in zip(masses, centers_of_mass, inertias, strict=True):
            offset = center - com
            inertia += tensor + mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
        bodies.append(
            builder.add_link(
                xform=wp.transform(origin, wp.quat_identity()),
                mass=total,
                com=wp.vec3(com),
                inertia=wp.mat33(inertia),
                lock_inertia=True,
                label=label + ("/rotor" if group == 0 else "/carriage"),
            )
        )

    for i, (name, mesh) in enumerate(components):
        group = assignment[i]
        if name == "wrapline_product_film":
            continue
        if group < 0:
            builder.add_shape_mesh(
                -1,
                mesh=mesh,
                cfg=builder.ShapeConfig(density=0.0, mu=0.5, gap=0.002, is_visible=False),
                label=f"{label}/{name}_{i}/collision",
            )
            continue
        origin = pivot if group == 0 else lift_origin
        collider = newton.Mesh(mesh.vertices - origin, mesh.indices, compute_inertia=False)
        builder.add_shape_convex_hull(
            bodies[group],
            mesh=collider,
            cfg=builder.ShapeConfig(density=0.0, mu=0.5, gap=0.002, is_visible=False),
            label=f"{label}/{name}_{i}/collision",
        )
    if visuals:
        for i, (name, mesh) in enumerate(components):
            group = assignment[i]
            body = bodies[group] if group >= 0 else -1
            origin = (pivot if group == 0 else lift_origin) if group >= 0 else np.zeros(3)
            builder.add_shape_mesh(
                body,
                mesh=mesh.copy(vertices=mesh.vertices - origin),
                cfg=builder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
                label=f"{label}/{name}_{i}/visual",
            )

    masts = [i for i, (name, _) in enumerate(components) if name == "steel" and sizes[i, 2] > 2.0]
    if len(masts) != 1:
        raise ValueError(f"{label}: source lift mast must be uniquely identifiable")
    mast_offset = centers[masts[0]] - pivot
    # The source is a display pose, with the carriage across the infeed.
    # Park beside the conveyor using the motor, never by changing body poses.
    home = np.arctan2(np.sign(mast_offset[1]), 0.0) - np.arctan2(mast_offset[1], mast_offset[0])
    home = float((home + np.pi) % (2.0 * np.pi) - np.pi)
    rotor = builder.add_joint_revolute(
        -1,
        bodies[0],
        parent_xform=wp.transform(pivot, wp.quat_identity()),
        axis=(0.0, 0.0, 1.0),
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        target_q=home,
        target_qd=0.0,
        target_ke=2000.0,
        target_kd=1000.0,
        effort_limit=1000.0,
        velocity_limit=2.0,
        label=label + "/rotation",
    )
    mast = components[masts[0]][1]
    carriage_vertices = np.concatenate([mesh.vertices for i, (_, mesh) in enumerate(components) if assignment[i] == 1])
    lower = float(mast.vertices[:, 2].min() - carriage_vertices[:, 2].min())
    upper = float(mast.vertices[:, 2].max() - carriage_vertices[:, 2].max())
    if lower >= upper:
        raise ValueError(f"{label}: carriage does not fit within its lift mast")
    lift = builder.add_joint_prismatic(
        bodies[0],
        bodies[1],
        parent_xform=wp.transform(lift_origin - pivot, wp.quat_identity()),
        axis=(0.0, 0.0, 1.0),
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        target_q=0.0,
        target_ke=builder.body_mass[bodies[1]] * (2.0 * np.pi * 10.0) ** 2,
        target_kd=2.0 * builder.body_mass[bodies[1]] * (2.0 * np.pi * 10.0),
        effort_limit=3000.0,
        velocity_limit=0.3,
        limit_lower=lower,
        limit_upper=upper,
        label=label + "/lift",
    )
    builder.add_articulation([rotor, lift])
    return bodies, (rotor, lift), assignment
