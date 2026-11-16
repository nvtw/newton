# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read the local Berlin plant's composed geometry and physics into a builder.

This is an example-specific adapter. It does not run Kit, execute embedded
scripts, or change the source layers. OpenUSD resolves references and variants;
Newton receives shared meshes, primitive colliders, rigid bodies, and joints.
"""

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.usd
from newton.examples.phoenx._bottling_wrap import build_wrapper, mesh_components


def _value(prim, name, default):
    value = prim.GetAttribute(name).Get()
    return default if value is None else value


def _quat(value):
    return wp.quat(*value.GetImaginary(), value.GetReal())


@dataclass
class PlantScene:
    """Builder and source-to-model maps used by the line controller and checks."""

    builder: newton.ModelBuilder
    bodies: dict = field(default_factory=dict)
    joints: dict = field(default_factory=dict)
    belts: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    bottle_bodies: list = field(default_factory=list)
    case_bodies: list = field(default_factory=list)
    body_mass: dict = field(default_factory=dict)
    joint_frames: list = field(default_factory=list)
    self_collision_scopes: list = field(default_factory=list)
    articulations: list = field(default_factory=list)
    frames: dict = field(default_factory=dict)
    collision_groups: list = field(default_factory=list)

    def collision_masks(self, labels):
        """Map the source collections to compact per-shape collision masks."""
        from pxr import Sdf  # noqa: PLC0415 - OpenUSD is optional.

        membership = np.zeros(len(labels), dtype=np.uint64)
        filtered = np.zeros(len(labels), dtype=np.uint64)
        # The adapter appends /collision or /visual to source geometry paths.
        paths = [Sdf.Path(label).GetParentPath() for label in labels]
        for index, (query, excluded) in enumerate(self.collision_groups):
            included = np.fromiter((query.IsPathIncluded(path) for path in paths), dtype=bool, count=len(paths))
            membership[included] |= np.uint64(1 << index)
            filtered[included] |= np.uint64(excluded)
        return membership, filtered


def _mesh_normals(mesh):
    """Keep valid authored normals, rebuilding missing or invalid normals."""
    vertices = np.asarray(mesh.vertices)
    if mesh.normals is not None:
        normals = np.asarray(mesh.normals)
        if normals.shape == vertices.shape and np.isfinite(normals).all():
            return normals
    faces = np.asarray(mesh.indices).reshape(-1, 3)
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    return normals


def load_plant(asset_root: Path, *, cutaway=True, visuals=True, motor_off=False) -> PlantScene:
    """Build the full line from local USD, retaining its authored SI properties."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415 - OpenUSD is optional.

    started = time.perf_counter()
    path = asset_root / "scenarios/line_sim/line_sim.usda"
    if not path.is_file():
        raise FileNotFoundError(
            f"No running-line stage at {path}. Download the documented 2.33 GB simulation subset and pass --asset-root."
        )
    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())
    fleet = asset_root / "scenarios/logistics/logistics_fleet.usda"
    if fleet.is_file():
        stage.GetSessionLayer().subLayerPaths.append(str(fleet.resolve()))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("The Berlin plant adapter expects its metre-scaled, Z-up source stage")
    plant = stage.GetPrimAtPath("/Plant")
    if cutaway:
        plant.GetVariantSet("view").SetVariantSelection("cutaway")
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    cache = UsdGeom.XformCache()
    scene = PlantScene(newton.ModelBuilder())
    builder = scene.builder
    collision_table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(stage)
    collision_groups = collision_table.GetGroups()
    if len(collision_groups) > 64:
        raise ValueError("The plant adapter supports at most 64 USD collision groups")
    for group in collision_groups:
        source_group = UsdPhysics.CollisionGroup(stage.GetPrimAtPath(group))
        if source_group.GetInvertFilteredGroupsAttr().Get():
            raise ValueError("The plant adapter does not support inverted collision groups")
        query = source_group.GetCollidersCollectionAPI().ComputeMembershipQuery()
        excluded = sum(
            1 << i for i, other in enumerate(collision_groups) if not collision_table.IsCollisionEnabled(group, other)
        )
        scene.collision_groups.append((query, excluded))
    for prim in prims:
        if _value(prim, "physxArticulation:enabledSelfCollisions", None) is False:
            root = prim
            if prim.IsA(UsdPhysics.Joint):
                # A fixed-root articulation may put its joint in a separate
                # folder. Its root body identifies the enclosing mechanism.
                targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
                root = stage.GetPrimAtPath(targets[0]).GetParent() if targets else prim.GetParent()
            scene.self_collision_scopes.append(str(root.GetPath()))
    scene.metadata = dict(stage.GetRootLayer().customLayerData)
    for path in ("/Plant/Line/Packers", "/Plant/Line/Palletizers", "/Plant/Line"):
        parent = stage.GetPrimAtPath(path)
        if parent:
            for prim in parent.GetChildren():
                scene.frames[str(prim.GetPath())] = np.asarray(cache.GetLocalToWorldTransform(prim)).copy()
    matrices = {}
    parked = set()
    body_prims = {}
    nominal = {}
    vendor = asset_root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
    if not motor_off:
        interface = json.loads((vendor / "ifm_v01/interface.json").read_text())
        for belt in interface["belts"]:
            speed = interface["speeds_mps"][belt["speed_key"]]
            linear, angular = (speed, 0.0, 0.0), (0.0, 0.0, 0.0)
            if belt["kind"] == "angular":
                linear = (0.0, 0.0, 0.0)
                angular = (0.0, 0.0, belt.get("omega_sign", 1.0) * speed / belt["radius"])
            nominal[belt["prim"].replace(interface["root"], "/Plant/Line/IFM", 1)] = (linear, angular)
        interface = json.loads((vendor / "ctp_v01/interface.json").read_text())
        for module in interface["modules"].values():
            for belt in module["belts"]:
                speed = belt["speed"]
                skew = math.radians(interface["zones"][belt["zone"]].get("skew_deg", 0.0))
                linear, angular = (speed * math.cos(skew), speed * math.sin(skew), 0.0), (0.0, 0.0, 0.0)
                if belt["kind"] == "angular":
                    linear = (0.0, 0.0, 0.0)
                    angular = (0.0, 0.0, belt.get("omega_sign", 1.0) * speed / belt["radius"])
                nominal["/Plant/Line/CTP" + belt["prim"]] = (linear, angular)

    def transform(matrix):
        parts = Gf.Transform(matrix)
        return wp.transform(parts.GetTranslation(), _quat(parts.GetRotation().GetQuat())), np.array(parts.GetScale())

    for prim in prims:
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not _value(prim, "physics:rigidBodyEnabled", True):
            continue
        label = str(prim.GetPath())
        matrix = cache.GetLocalToWorldTransform(prim)
        pose, scale = transform(matrix)
        if label.startswith("/Plant/Sim/Product/") and pose.p[2] < -2.0:
            parked.add(label)
            continue
        mass = float(_value(prim, "physics:mass", 0.0))
        inertia = np.asarray(_value(prim, "physics:diagonalInertia", (0.0, 0.0, 0.0)))
        com = np.asarray(_value(prim, "physics:centerOfMass", (0.0, 0.0, 0.0)))
        if not np.isfinite(com).all():
            com = np.zeros(3)
        axes = _quat(_value(prim, "physics:principalAxes", Gf.Quatf(1.0)))
        rotation = np.asarray(wp.quat_to_matrix(axes)).reshape(3, 3)
        tensor = rotation @ np.diag(inertia) @ rotation.T
        explicit = mass > 0 and np.all(inertia > 0)
        kinematic = bool(_value(prim, "physics:kinematicEnabled", False))
        if kinematic and not explicit:
            # Kinematic conveyor surfaces have infinite effective mass. Their
            # open triangle sheets have no volume from which to infer inertia.
            tensor = np.eye(3)
            explicit = True
        body = builder.add_link(
            xform=pose,
            mass=(mass or 1.0) if explicit else 0.0,
            inertia=wp.mat33(tensor) if explicit else None,
            com=wp.vec3(com * scale),
            lock_inertia=bool(explicit),
            is_kinematic=kinematic,
            label=label,
        )
        scene.bodies[label] = body
        scene.body_mass[body] = mass
        matrices[body] = Gf.Matrix4d(1.0)
        matrices[body].SetRotate(Gf.Quatd(pose.q[3], Gf.Vec3d(*pose.q[:3])))
        matrices[body].SetTranslateOnly(Gf.Vec3d(*pose.p))
        body_prims[body] = prim
        if label.startswith("/Plant/Sim/Product/Bottles/"):
            scene.bottle_bodies.append(body)
        elif label.startswith("/Plant/Sim/Product/Cases/"):
            scene.case_bodies.append(body)

    texture_files = {}
    if visuals:
        for file in asset_root.rglob("*"):
            if file.is_file():
                texture_files.setdefault(file.name, file)
    atlases = {}

    def resolve_texture(mesh):
        texture = mesh.texture
        if not isinstance(texture, str) or Path(texture).is_file():
            return
        name = Path(texture).name
        if "<UDIM>" not in name:
            if name in texture_files:
                mesh.texture = str(texture_files[name])
            return
        if name not in atlases:
            from PIL import Image  # noqa: PLC0415 - Decode only when UDIM textures are used.

            before, after = name.split("<UDIM>")
            tiles = {}
            for filename, file in texture_files.items():
                number = filename[len(before) : -len(after)] if after else filename[len(before) :]
                if filename.startswith(before) and filename.endswith(after) and number.isdigit():
                    tiles[int(number)] = file
            if not tiles:
                raise FileNotFoundError(f"No local tiles for source texture {texture}")
            coordinates = {tile: ((tile - 1001) % 10, (tile - 1001) // 10) for tile in tiles}
            columns = max(u for u, _ in coordinates.values()) + 1
            rows = max(v for _, v in coordinates.values()) + 1
            with Image.open(next(iter(tiles.values()))) as image:
                width, height = image.size
            atlas = np.zeros((rows * height, columns * width, 4), dtype=np.uint8)
            for tile, file in tiles.items():
                u, v = coordinates[tile]
                y, x = (rows - v - 1) * height, u * width
                with Image.open(file) as image:
                    if image.size != (width, height):
                        raise ValueError(f"Inconsistent UDIM tile size: {file}")
                    atlas[y : y + height, x : x + width] = np.asarray(image.convert("RGBA"))
            atlases[name] = atlas, columns, rows
        mesh.texture, columns, rows = atlases[name]
        matrix = np.asarray(mesh.texture_transform).copy()
        matrix[0] /= columns
        matrix[1] /= rows
        mesh.texture_transform = matrix

    meshes = {}
    owner_cache = {}

    def owner(prim):
        label = str(prim.GetPath())
        if label not in owner_cache:
            if label in scene.bodies:
                result = scene.bodies[label]
            elif label in parked:
                result = -2
            elif prim.IsPseudoRoot():
                result = -1
            else:
                result = owner(prim.GetParent())
            owner_cache[label] = result
        return owner_cache[label]

    def moving_mesh(mesh, velocity):
        mesh = newton.Mesh(mesh.vertices, mesh.indices, compute_inertia=False, enable_surface_velocity=True)
        linear, angular, shape_pose, scale, belt_label = velocity
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(shape_pose[3:]))).reshape(3, 3)
        points = np.asarray(mesh.vertices) * np.asarray(scale) @ rotation.T + np.asarray(shape_pose[:3])
        # Newton rotates and scales vertex velocities. Undo the shape transform
        # so the authored speed remains in metres per second in the body frame.
        speeds = ((np.asarray(linear) + np.cross(np.asarray(angular), points)) @ rotation) / np.asarray(scale)
        scene.belts.append((mesh, speeds.astype(np.float32), belt_label))
        return mesh

    def mesh_for(prim, *, visual, velocity=None):
        points = prim.GetAttribute("points")
        stack = points.GetPropertyStack()
        source = (stack[0].layer.identifier, str(stack[0].path)) if stack else str(prim.GetPath())
        # Instance proxies have distinct paths but share the same authored geometry.
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0] if visual else None
        material_key = None
        if material:
            material_stack = material.GetPrim().GetPrimStack()
            material_key = tuple((s.layer.identifier, str(s.path)) for s in material_stack)
        key = (source, visual, material_key, velocity)
        if key not in meshes:
            mesh = newton.usd.get_mesh(
                prim,
                load_normals=visual,
                load_uvs=visual,
                compute_inertia=False,
                load_visual_materials=visual,
            )
            if visual:
                resolve_texture(mesh)
            if visual and (
                mesh.normals is None or mesh.normals.shape != mesh.vertices.shape or not np.isfinite(mesh.normals).all()
            ):
                mesh = newton.Mesh(
                    mesh.vertices,
                    mesh.indices,
                    normals=_mesh_normals(mesh),
                    uvs=mesh.uvs,
                    color=mesh.color,
                    roughness=mesh.roughness,
                    metallic=mesh.metallic,
                    texture=mesh.texture,
                    texture_transform=mesh.texture_transform,
                    opacity=mesh.opacity,
                    compute_inertia=False,
                )
            if not visual and velocity is None:
                # Several source actuators bake metre-scale offsets into tiny
                # collider vertices. Integrating their moments about zero loses
                # enough precision to produce negative central inertias. Compute
                # moments near the solid and then translate only its COM back.
                vertices = np.asarray(mesh.vertices)
                center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
                centered = newton.Mesh(vertices - center, mesh.indices)
                mesh.has_inertia = True
                mesh.mass = centered.mass
                mesh.com = wp.vec3(np.asarray(centered.com) + center)
                mesh.inertia = centered.inertia
            if velocity is not None:
                if _value(prim, "physics:approximation", "none") == "convexHull":
                    # Cook the authored hull before assigning its velocity field.
                    mesh = mesh.compute_convex_hull()
                mesh = moving_mesh(mesh, velocity)
            meshes[key] = mesh
        return meshes[key]

    def add_geometry(prim, matrix, body, *, visual, collision):
        local = matrix * matrices[body].GetInverse() if body >= 0 else matrix
        pose, scale = transform(local)
        cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0 if collision else 0.0,
            gap=max(0.0, float(_value(prim, "physxCollision:contactOffset", 0.002))),
            margin=float(_value(prim, "physxCollision:restOffset", 0.0)),
            has_shape_collision=collision,
            has_particle_collision=False,
            is_visible=visual,
        )
        if collision:
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
            if material:
                cfg.mu = float(_value(material.GetPrim(), "physics:dynamicFriction", 0.5))
                cfg.mu_static = max(cfg.mu, float(_value(material.GetPrim(), "physics:staticFriction", cfg.mu)))
                cfg.restitution = float(_value(material.GetPrim(), "physics:restitution", 0.0))
                cfg.density = float(_value(material.GetPrim(), "physics:density", 1000.0)) or 1000.0
        kind = prim.GetTypeName()
        kw = {"xform": pose, "cfg": cfg, "label": str(prim.GetPath()) + ("/visual" if visual else "/collision")}
        color = _value(prim, "primvars:displayColor", None)
        if visual and color is not None and len(color):
            kw["color"] = tuple(color[0])
        velocity = None
        if collision and body >= 0:
            parent = body_prims[body]
            enabled = parent.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled")
            # A stopped conveyor still needs a velocity field for its PLC to
            # start it later; the authored enable flag controls its initial speed.
            if enabled:
                label = str(parent.GetPath())
                linear, angular = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                if enabled.Get() and not motor_off:
                    if label in nominal:
                        linear, angular = nominal[label]
                    else:
                        linear = _value(parent, "physxSurfaceVelocity:surfaceVelocity", linear)
                        angular = np.radians(_value(parent, "physxSurfaceVelocity:surfaceAngularVelocity", angular))
                        body_pose, body_scale = transform(cache.GetLocalToWorldTransform(parent))
                        if _value(parent, "physxSurfaceVelocity:surfaceVelocityLocalSpace", True):
                            linear = np.asarray(linear) * body_scale
                        else:
                            rotation = np.asarray(wp.quat_to_matrix(body_pose.q)).reshape(3, 3)
                            linear, angular = np.asarray(linear) @ rotation, np.asarray(angular) @ rotation
                    if label.startswith("/Plant/Line/BottleLanes/"):
                        linear = (float(_value(parent, "bottle_lanes:speedNominal", 0.2)), 0.0, 0.0)
                    if "/Packers/" in label and label.endswith(("/Belts/IN_A", "/Belts/IN_B")):
                        linear = (0.2 if label.endswith("IN_A") else 0.4, 0.0, 0.0)
                velocity = (tuple(linear), tuple(angular), tuple(pose.p) + tuple(pose.q), tuple(scale), label)
        if kind == "Mesh":
            mesh = mesh_for(prim, visual=visual, velocity=velocity)
            approximation = _value(prim, "physics:approximation", "none")
            if collision and approximation == "convexHull" and velocity is None and len(mesh.vertices) == 8:
                points = np.asarray(mesh.vertices)
                center = (points.min(axis=0) + points.max(axis=0)) / 2.0
                half = (points.max(axis=0) - points.min(axis=0)) / 2.0
                if np.all(half > 0.0) and np.allclose(np.abs(points - center), half, rtol=0.0, atol=1e-8):
                    # These are exactly boxes, rather than approximations of
                    # the source hulls; use their analytic contact path.
                    kw["xform"] = wp.transform_multiply(pose, wp.transform(center * scale, wp.quat_identity()))
                    h = np.abs(scale) * half
                    builder.add_shape_box(body, hx=h[0], hy=h[1], hz=h[2], **kw)
                    return
            add = (
                builder.add_shape_convex_hull
                if collision and approximation == "convexHull" and velocity is None
                else builder.add_shape_mesh
            )
            add(body, mesh=mesh, scale=scale, **kw)
        elif kind == "Cube":
            half = float(_value(prim, "size", 2.0)) / 2.0
            if velocity is not None:
                mesh = moving_mesh(newton.Mesh.create_box(half, compute_inertia=False), velocity)
                builder.add_shape_mesh(body, mesh=mesh, scale=scale, **kw)
            else:
                half = half * np.abs(scale)
                builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], **kw)
        elif kind == "Sphere":
            builder.add_shape_sphere(body, radius=float(_value(prim, "radius", 1.0)) * max(abs(scale)), **kw)
        elif kind == "Plane":
            axis = "XYZ".index(_value(prim, "axis", "Z"))
            if axis != 2:
                rotation = wp.quat_from_axis_angle(
                    wp.vec3(0.0, 1.0, 0.0) if axis == 0 else wp.vec3(1.0, 0.0, 0.0),
                    math.pi / 2.0 if axis == 0 else -math.pi / 2.0,
                )
                kw["xform"] = wp.transform(pose.p, pose.q * rotation)
            builder.add_shape_plane(body=body, width=0.0, length=0.0, **kw)
        elif kind in ("Cylinder", "Capsule", "Cone"):
            axis = _value(prim, "axis", "Z")
            i = "XYZ".index(axis)
            radius = float(_value(prim, "radius", 1.0)) * max(abs(scale[j]) for j in range(3) if j != i)
            half_height = float(_value(prim, "height", 2.0)) * abs(scale[i]) / 2.0
            add = {
                "Cylinder": builder.add_shape_cylinder,
                "Capsule": builder.add_shape_capsule,
                "Cone": builder.add_shape_cone,
            }[kind]
            if i != 2:
                rotation = (
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0)
                    if i == 0
                    else wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)
                )
                kw["xform"] = wp.transform(pose.p, pose.q * rotation)
            add(body, radius=radius, half_height=half_height, **kw)
        else:
            raise ValueError(f"Unsupported geometry {kind} at {prim.GetPath()}")

    wrappers = {}
    geometry_types = {"Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"}
    instancers = [UsdGeom.PointInstancer(p) for p in prims if p.IsA(UsdGeom.PointInstancer)]
    prototypes = {path for instancer in instancers for path in instancer.GetPrototypesRel().GetTargets()}
    for prim in prims:
        if prim.GetTypeName() not in geometry_types:
            continue
        if any(prim.GetPath().HasPrefix(path) for path in prototypes):
            continue
        body = owner(prim)
        if body == -2:
            continue
        imageable = UsdGeom.Imageable(prim)
        visible = (
            visuals
            and imageable.ComputeVisibility() != "invisible"
            and imageable.ComputePurpose() not in ("guide", "proxy")
        )
        collision = prim.HasAPI(UsdPhysics.CollisionAPI) and _value(prim, "physics:collisionEnabled", True)
        matrix = cache.GetLocalToWorldTransform(prim)
        parent_path = str(prim.GetParent().GetPath())
        if (
            prim.GetTypeName() == "Mesh"
            and parent_path.startswith("/Plant/WrapLine/Lines/")
            and parent_path.endswith("/Wrapper")
        ):
            if collision:
                raise ValueError(f"Wrapper visual mesh unexpectedly has authored collision: {prim.GetPath()}")
            mesh = mesh_for(prim, visual=visuals)
            world = np.asarray(matrix)
            for part in mesh_components(mesh):
                vertices = part.vertices @ world[:3, :3] + world[3, :3]
                component = part.copy(vertices=vertices)
                if part.normals is not None:
                    normals = part.normals @ np.linalg.inv(world[:3, :3]).T
                    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                    component.normals[:] = normals / np.maximum(lengths, 1.0e-12)
                wrappers.setdefault(parent_path, []).append((prim.GetName(), component))
            continue
        if collision:
            add_geometry(prim, matrix, body, visual=False, collision=True)
        if visible:
            add_geometry(prim, matrix, body, visual=True, collision=False)

    def prototype_meshes(root):
        """Bake a prototype once per material, retaining the placement instancing."""
        inverse = cache.GetLocalToWorldTransform(root).GetInverse()
        batches = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            kind = prim.GetTypeName()
            if kind not in geometry_types:
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy"):
                continue
            if kind == "Mesh":
                mesh = mesh_for(prim, visual=True)
            elif kind == "Cube":
                h = float(_value(prim, "size", 2.0)) / 2.0
                mesh = newton.Mesh.create_box(h, h, h, compute_inertia=False)
            elif kind == "Sphere":
                mesh = newton.Mesh.create_sphere(float(_value(prim, "radius", 1.0)), compute_inertia=False)
            else:
                create = {
                    "Cylinder": newton.Mesh.create_cylinder,
                    "Capsule": newton.Mesh.create_capsule,
                    "Cone": newton.Mesh.create_cone,
                }[kind]
                mesh = create(
                    float(_value(prim, "radius", 1.0)),
                    float(_value(prim, "height", 2.0)) / 2.0,
                    up_axis=newton.Axis("XYZ".index(_value(prim, "axis", "Z"))),
                    compute_inertia=False,
                )
            color = _value(prim, "primvars:displayColor", None)
            color = tuple(color[0]) if color is not None and len(color) else tuple(mesh.color or (0.65, 0.65, 0.65))
            key = (color, mesh.roughness, mesh.metallic, mesh.texture_hash, mesh.texture_transform, mesh.opacity)
            if key not in batches:
                batches[key] = ([], [], [], [], mesh)
            positions, faces, normals, uvs, _ = batches[key]
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
            linear = matrix[:3, :3]
            vertices = np.asarray(mesh.vertices) @ linear + matrix[3, :3]
            indices = np.asarray(mesh.indices).reshape(-1, 3)
            if np.linalg.det(linear) < 0:
                indices = indices[:, ::-1]
            offset = sum(len(p) for p in positions)
            positions.append(vertices)
            faces.append(indices + offset)
            n = _mesh_normals(mesh) @ np.linalg.inv(linear).T
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
            normals.append(n)
            uvs.append(np.zeros((len(vertices), 2)) if mesh.uvs is None else mesh.uvs)
        result = []
        for key, (positions, faces, normals, uvs, original) in batches.items():
            result.append(
                newton.Mesh(
                    np.concatenate(positions),
                    np.concatenate(faces).ravel(),
                    normals=np.concatenate(normals),
                    uvs=np.concatenate(uvs),
                    compute_inertia=False,
                    color=key[0],
                    roughness=original.roughness,
                    metallic=original.metallic,
                    texture=original.texture,
                    texture_transform=original.texture_transform,
                    opacity=original.opacity,
                )
            )
        return result

    prototype_cache = {}
    for instancer in instancers if visuals else ():
        if instancer.ComputeVisibility() == "invisible":
            continue
        targets = instancer.GetPrototypesRel().GetTargets()
        indices = instancer.GetProtoIndicesAttr().Get()
        transforms = instancer.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
        world = cache.GetLocalToWorldTransform(instancer.GetPrim())
        for path in targets:
            if path not in prototype_cache:
                prototype_cache[path] = prototype_meshes(stage.GetPrimAtPath(path))
        for index, matrix in zip(indices, transforms, strict=True):
            pose, scale = transform(matrix * world)
            for mesh in prototype_cache[targets[index]]:
                builder.add_shape_mesh(
                    -1,
                    mesh=mesh,
                    xform=pose,
                    scale=scale,
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=0.0, has_shape_collision=False, has_particle_collision=False
                    ),
                    label=str(instancer.GetPath()) + "/instance",
                )

    plans = []
    gripper_templates = [p.GetPath() for p in prims if p.GetTypeName() == "IsaacSurfaceGripper"]
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint) or not _value(prim, "physics:jointEnabled", True):
            continue
        # Kit's surface-gripper extension disables these attachment templates
        # while open and retargets them on grasp. Our runtime creates the grasp
        # constraints against the actual bottles instead.
        if any(prim.GetPath().HasPrefix(path) for path in gripper_templates):
            continue
        ids, frames = [], []
        for side in (0, 1):
            targets = prim.GetRelationship(f"physics:body{side}").GetTargets()
            target = str(targets[0]) if targets else None
            body = scene.bodies.get(target, -1)
            position = _value(prim, f"physics:localPos{side}", (0.0, 0.0, 0.0))
            rotation = _quat(_value(prim, f"physics:localRot{side}", Gf.Quatf(1.0)))
            if target and body < 0:
                target_prim = stage.GetPrimAtPath(target)
                if not target_prim:
                    raise ValueError(f"Joint {prim.GetPath()} references an unavailable frame: {target}")
                local = Gf.Matrix4d(1.0)
                local.SetRotate(Gf.Quatd(rotation[3], Gf.Vec3d(*rotation[:3])))
                local.SetTranslateOnly(Gf.Vec3d(*position))
                world = local * cache.GetLocalToWorldTransform(target_prim)
                position = world.ExtractTranslation()
                rotation = _quat(world.ExtractRotationQuat())
            ids.append(body)
            frames.append(wp.transform(position, rotation))
        if ids[1] < 0:
            if ids[0] < 0:
                continue
            ids.reverse()
            frames.reverse()
        plans.append((prim, ids, frames))

    def add_source_joint(prim, ids, frames):
        kw = {"parent_xform": frames[0], "child_xform": frames[1], "label": str(prim.GetPath())}
        kind = prim.GetTypeName()
        if kind == "PhysicsFixedJoint":
            joint = builder.add_joint_fixed(*ids, **kw)
        elif kind in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            angular = kind == "PhysicsRevoluteJoint"
            unit = math.pi / 180.0 if angular else 1.0
            gain_unit = 1.0 / unit if angular else 1.0
            drive = "drive:" + ("angular" if angular else "linear") + ":physics:"
            stiffness = float(_value(prim, drive + "stiffness", 0.0)) * gain_unit
            damping = float(_value(prim, drive + "damping", 0.0)) * gain_unit
            add = builder.add_joint_revolute if angular else builder.add_joint_prismatic
            joint = add(
                *ids,
                axis="XYZ".index(_value(prim, "physics:axis", "X")),
                limit_lower=float(_value(prim, "physics:lowerLimit", -math.inf)) * unit,
                limit_upper=float(_value(prim, "physics:upperLimit", math.inf)) * unit,
                target_q=float(_value(prim, drive + "targetPosition", 0.0)) * unit,
                target_qd=float(_value(prim, drive + "targetVelocity", 0.0)) * unit,
                target_ke=stiffness,
                target_kd=damping,
                effort_limit=float(_value(prim, drive + "maxForce", math.inf)),
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY
                if stiffness
                else newton.JointTargetMode.VELOCITY,
                **kw,
            )
        elif kind == "PhysicsJoint":
            axes = [[], []]
            for angular, names in enumerate((("transX", "transY", "transZ"), ("rotX", "rotY", "rotZ"))):
                unit = math.pi / 180.0 if angular else 1.0
                for axis, name in enumerate(names):
                    lo = float(_value(prim, f"limit:{name}:physics:low", -math.inf))
                    hi = float(_value(prim, f"limit:{name}:physics:high", math.inf))
                    if lo > hi:
                        continue
                    drive = f"drive:{name}:physics:"
                    stiffness = float(_value(prim, drive + "stiffness", 0.0)) / unit
                    axes[angular].append(
                        newton.ModelBuilder.JointDofConfig(
                            axis=axis,
                            limit_lower=lo * unit,
                            limit_upper=hi * unit,
                            target_q=float(_value(prim, drive + "targetPosition", 0.0)) * unit,
                            target_qd=float(_value(prim, drive + "targetVelocity", 0.0)) * unit,
                            target_ke=stiffness,
                            target_kd=float(_value(prim, drive + "damping", 0.0)) / unit,
                            effort_limit=float(_value(prim, drive + "maxForce", math.inf)),
                            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY
                            if stiffness
                            else newton.JointTargetMode.VELOCITY,
                        )
                    )
            joint = builder.add_joint_d6(*ids, linear_axes=axes[0], angular_axes=axes[1], **kw)
        else:
            raise ValueError(f"Unsupported joint {kind}: {prim.GetPath()}")
        scene.joints[str(prim.GetPath())] = joint
        scene.joint_frames.append((ids, frames, kind))
        return joint

    # Build each spanning tree contiguously. Keep the authored spring loops as
    # maximal constraints; trees can then use PhoenX's reduced-coordinate path.
    roots = list(range(builder.body_count))

    def find(body):
        while roots[body] != body:
            roots[body] = roots[roots[body]]
            body = roots[body]
        return body

    tree, constraints, incoming = [], [], set()
    for plan in plans:
        prim, (parent, child), _ = plan
        loop = child in incoming or (parent >= 0 and find(parent) == find(child))
        if loop or _value(prim, "physics:excludeFromArticulation", False):
            constraints.append(plan)
            continue
        incoming.add(child)
        tree.append(plan)
        if parent >= 0:
            roots[find(child)] = find(parent)
    components = {}
    for plan in tree:
        components.setdefault(find(plan[1][1]), []).append(plan)
    articulated = set()
    articulation_bodies = {}
    for component in components.values():
        children = {p[1][1] for p in component}
        parents = {p[1][0] for p in component if p[1][0] >= 0}
        floating = parents - children
        added = {-1}
        joints = []
        for body in sorted(floating):
            joints.append(builder.add_joint_free(child=body, label=builder.body_label[body] + "/floating_root"))
            added.add(body)
        remaining = list(component)
        while remaining:
            ready = [p for p in remaining if p[1][0] in added]
            if not ready:
                raise ValueError("The source articulation tree has no root")
            for plan in ready:
                joints.append(add_source_joint(*plan))
                added.add(plan[1][1])
                remaining.remove(plan)
        scene.articulations.append(builder.articulation_count)
        articulation_bodies[builder.articulation_count] = added - {-1}
        builder.add_articulation(joints)
        articulated.update(added - {-1})
    constraint_bodies = {b for _, bodies, _ in constraints for b in bodies if b >= 0}
    for body in sorted(constraint_bodies - articulated):
        joint = builder.add_joint_free(child=body, label=builder.body_label[body] + "/floating_root")
        builder.add_articulation([joint])
    with warnings.catch_warnings():
        # Multiple attachment springs between the same two gripper bodies are
        # intentional in the source. The maximal solver keeps every spring.
        warnings.filterwarnings("ignore", message="Adding a .*Parallel joints.*", category=newton.NewtonWarning)
        for plan in constraints:
            add_source_joint(*plan)
    unsupported = {body for prim, ids, _ in constraints if prim.GetTypeName() == "PhysicsJoint" for body in ids}
    scene.articulations = [a for a in scene.articulations if articulation_bodies[a].isdisjoint(unsupported)]

    for label, components in wrappers.items():
        bodies, joints, _ = build_wrapper(builder, components, label=label, visuals=visuals)
        scene.articulations.append(builder.articulation_count - 1)
        for body in bodies:
            scene.bodies[builder.body_label[body]] = body
            scene.body_mass[body] = builder.body_mass[body]
        for joint, kind in zip(joints, ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"), strict=True):
            scene.joints[builder.joint_label[joint]] = joint
            scene.joint_frames.append(
                (
                    [builder.joint_parent[joint], builder.joint_child[joint]],
                    [builder.joint_X_p[joint], builder.joint_X_c[joint]],
                    kind,
                )
            )
    if wrappers:
        scene.metadata["wrapper_mass_estimates"] = {
            "metal_shell_thickness_m": 0.003,
            "steel_density_kg_m3": 7850.0,
            "aluminium_density_kg_m3": 2700.0,
            "film_density_kg_m3": 920.0,
            "rubber_density_kg_m3": 1100.0,
            "rotor_effort_limit_nm": 1000.0,
            "lift_effort_limit_n": 3000.0,
            "lift_servo_frequency_hz": 10.0,
            "source_has_authored_wrapper_dynamics": False,
        }

    for body, mass in scene.body_mass.items():
        if not builder.body_lock_inertia[body] and mass > 0 and builder.body_mass[body] > 0:
            ratio = mass / builder.body_mass[body]
            builder.body_inertia[body] = builder.body_inertia[body] * ratio
            builder.body_mass[body] = mass
            builder.body_inv_mass[body] = 1.0 / mass
            builder.body_inv_inertia[body] = wp.mat33(
                np.linalg.inv(np.asarray(builder.body_inertia[body]).reshape(3, 3))
            )
    print(
        f"Berlin plant: {len(scene.bodies)} bodies, {len(scene.bottle_bodies)} live bottles, "
        f"{len(scene.joints)} joints, {builder.shape_count} shapes, {len(meshes)} shared meshes "
        f"loaded in {time.perf_counter() - started:.1f} s"
    )
    return scene
