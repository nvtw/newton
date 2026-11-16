# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct moving wrap-line assemblies from the source material meshes."""

import numpy as np
import warp as wp

import newton
import newton.geometry


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
