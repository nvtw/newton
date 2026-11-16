# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct moving wrap-line assemblies from the source material meshes."""

import json

import numpy as np
import warp as wp

import newton
import newton.geometry
from newton.examples.phoenx._bottling_io import _case_photoeyes


class WrapperController:
    """Move seated pallets to the measured wrap position, then await the process.

    This implements infeed only. Film application and labelling remain absent;
    reaching the wrap position never releases an outbound pallet.
    """

    def __init__(self, model, state, control, asset_root, channels, fleet):
        self.model, self.state, self.control = model, state, control
        self.channels, self.fleet = channels, fleet
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
                "since": 0.0,
                "quiet_since": None,
                "rotor_q": int(model.joint_q_start.numpy()[joint]),
                "rotor_qd": int(model.joint_qd_start.numpy()[joint]),
            }
            for sensor in ("PE_DROP", "PE_WRAP"):
                beam = spec["sensors"][sensor]
                origins.append(beam["origin"])
                directions.append(beam["direction"])
                distances.append(beam["distance"])
        self.origins = wp.array(origins, dtype=wp.vec3)
        self.directions = wp.array(directions, dtype=wp.vec3)
        self.distances = wp.array(distances, dtype=float)
        self.hits = wp.zeros(len(origins), dtype=int)

    def before_step(self, time, speed, *, poses, coordinates):
        """Use actual pallet rays and motor state; never infer completion from a timer."""
        for channel in self.channels.values():
            speed[channel] = 0.0
        if not self.lines:
            return
        self.hits.zero_()
        model = self.model
        wp.launch(
            _case_photoeyes,
            dim=(len(self.hits), 32),
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
        hits = self.hits.numpy().reshape(-1, 2)
        velocities = self.state.body_qd.numpy()
        targets = self.control.joint_target_q.numpy()
        joint_velocity = self.state.joint_qd.numpy()
        for index, (name, line) in enumerate(self.lines.items()):
            spec = line["spec"]
            port = name + "_drop"
            reserved = any(port in (mission.source, mission.destination) for mission in self.fleet.missions.values())
            rotor_home = abs(coordinates[line["rotor_q"]] - targets[line["rotor_q"]]) < np.radians(2.0)
            rotor_home = rotor_home and abs(joint_velocity[line["rotor_qd"]]) < 0.02
            if line["state"] == "WAIT_DROP":
                observed = self.fleet._port_observation(port, poses)
                if reserved or not rotor_home or not hits[index, 0] or observed.pallet is None:
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
                line.update(state="INFEED", pallet=body, since=time, target_x=target_x)
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

    def report(self):
        """Expose missing wrapping as a waiting state, never as finished product."""
        return {name: {key: line[key] for key in ("state", "pallet", "fault")} for name, line in self.lines.items()}


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


def build_wrapper(builder, components, *, label, visuals=True):
    """Build a rotary arm and lift from world-space material components.

    Masses are estimates: metal covers use 3 mm shells (steel 7850 and
    aluminium 2700 kg/m3); the roll and rubber use solid densities 920 and
    1100 kg/m3. These assumptions are explicit because the source wrapper
    has visual geometry but no authored masses, joints, or drive limits.
    Returns body/joint IDs and the body assigned to each source component.
    """
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
        if group < 0 or name == "wrapline_product_film":
            continue
        origin = pivot if group == 0 else lift_origin
        collider = newton.Mesh(mesh.vertices - origin, mesh.indices, compute_inertia=False)
        builder.add_shape_convex_hull(
            bodies[group],
            mesh=collider,
            cfg=builder.ShapeConfig(density=0.0, mu=0.5, is_visible=False),
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
