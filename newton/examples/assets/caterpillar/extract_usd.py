# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Extract Caterpillar geometry and SI scene descriptors from its USD."""

import argparse
import gzip
import hashlib
import io
import pprint
from pathlib import Path

import numpy as np
import trimesh
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import newton.usd


def extract(source: Path, destination: Path, scene_output: Path):
    """Preserve authored body frames, joint data, materials, and collision meshes."""
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("This extraction expects the authored Z-up Caterpillar stage")
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if not np.isclose(unit, 1.0):
        raise ValueError(f"Caterpillar must be authored in meters, got metersPerUnit={unit}")
    cache = UsdGeom.XformCache()
    destination.mkdir(parents=True, exist_ok=True)
    bodies, shapes, joints = [], [], []
    body_ids, rotations, positions = {}, {}, {}

    def value(prim, name, default):
        attribute = prim.GetAttribute(name)
        result = attribute.Get() if attribute else None
        return default if result is None else result

    def pose(position, rotation):
        q = wp.quat_from_matrix(wp.mat33(rotation.astype(np.float32)))
        return [*map(float, position), *map(float, q)]

    def rigid(prim):
        matrix = cache.GetLocalToWorldTransform(prim)
        q = matrix.RemoveScaleShear().ExtractRotationQuat()
        r = np.asarray(wp.quat_to_matrix(wp.quat(*q.GetImaginary(), q.GetReal()))).reshape(3, 3)
        return np.asarray(matrix.ExtractTranslation()) * unit, r

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path = str(prim.GetPath())
            position, rotation = rigid(prim)
            body_ids[path] = len(bodies)
            positions[path], rotations[path] = position, rotation
            bodies.append(
                {
                    "label": path,
                    "pose": pose(position, rotation),
                    "kinematic": bool(value(prim, "physics:kinematicEnabled", False)),
                }
            )

    root_prefixes = ("/World/caterpillar_390f_lme/", "/World/Cat390F_L_upper/")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or not path.startswith(root_prefixes):
            continue
        ancestor = prim
        while ancestor and str(ancestor.GetPath()) not in body_ids:
            ancestor = ancestor.GetParent()
        body_path = str(ancestor.GetPath()) if ancestor else None
        body = body_ids.get(body_path, -1)
        rotation = rotations[body_path] if body >= 0 else np.eye(3)
        position = positions[body_path] if body >= 0 else np.zeros(3)
        mesh = newton.usd.get_mesh(
            prim,
            load_normals=True,
            load_uvs=True,
            face_varying_normal_conversion="vertex_splitting",
            vertex_splitting_angle_threshold_deg=0.0,
            compute_inertia=False,
            load_visual_materials=False,
        )
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        linear = rotation.T @ matrix[:3, :3].T * unit
        vertices = np.asarray(mesh.vertices, dtype=np.float64) @ linear.T
        vertices += (matrix[3, :3] * unit - position) @ rotation
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        vertices -= center
        normals = None
        if mesh.normals is not None:
            normals = np.asarray(mesh.normals) @ np.linalg.inv(linear)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)
        faces = np.asarray(mesh.indices).reshape(-1, 3).copy()
        if np.linalg.det(linear) < 0:
            faces = faces[:, ::-1]
        edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        faces = faces[np.linalg.norm(np.cross(edge0, edge1), axis=1) > 1.0e-16]
        uvs = mesh.uvs
        if normals is None:
            visual = trimesh.visual.texture.TextureVisuals(uv=uvs) if uvs is not None else None
            smooth = trimesh.graph.smooth_shade(
                trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False),
                angle=np.deg2rad(30.0),
            )
            vertices, faces, normals = smooth.vertices, smooth.faces, smooth.vertex_normals
            if uvs is not None:
                uvs = smooth.visual.uv
        used = np.unique(faces)
        if len(used) != len(vertices):
            remap = np.full(len(vertices), -1, dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            vertices, normals, faces = vertices[used], normals[used], remap[faces]
            if uvs is not None:
                uvs = np.asarray(uvs)[used]
        vertices = np.round(vertices, 7).astype(np.float32)
        digest = hashlib.sha256(
            vertices.tobytes()
            + faces.tobytes()
            + np.round(normals, 5).tobytes()
            + (b"" if uvs is None else np.asarray(uvs).tobytes())
        ).hexdigest()[:16]
        filename = f"mesh_{digest}.obj.gz"
        target = destination / filename
        if not target.exists():
            with (
                gzip.GzipFile(filename=target, mode="wb", compresslevel=9, mtime=0) as compressed,
                io.TextIOWrapper(compressed) as stream,
            ):
                stream.write("# Caterpillar USD extraction; meters; explicit surface normals\n")
                for vertex in vertices:
                    stream.write("v " + " ".join(f"{x:.9g}" for x in vertex) + "\n")
                for normal in normals:
                    stream.write("vn " + " ".join(f"{x:.9g}" for x in normal) + "\n")
                if uvs is not None:
                    for uv in uvs:
                        stream.write("vt " + " ".join(f"{x:.9g}" for x in uv) + "\n")
                for face in faces + 1:
                    stream.write("f " + " ".join(f"{i}/{i if uvs is not None else ''}/{i}" for i in face) + "\n")
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        color, roughness = (1.0, 1.0, 1.0), 0.5
        if material:
            for child in material.GetPrim().GetChildren():
                if child.IsA(UsdShade.Shader):
                    color = value(child, "inputs:diffuse_reflection_color", color)
                    roughness = float(value(child, "inputs:specular_reflection_roughness", roughness))
        color = np.asarray(color)
        color = np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)
        collision = prim.HasAPI(UsdPhysics.CollisionAPI) and bool(value(prim, "physics:collisionEnabled", True))
        material_kind = {"OmniGlass": "glass", "OmniSurface_Chrome": "chrome"}.get(
            material.GetPrim().GetName() if material else ""
        )
        shape = {
            "label": path,
            "body": body,
            "mesh": filename,
            "center": center.tolist(),
            "collision": collision,
            "density": 1000.0 * mass_unit / unit**3,
            "friction": 0.5,
            "approximation": str(value(prim, "physics:approximation", "none")),
            "visible": UsdGeom.Imageable(prim).ComputeVisibility() != "invisible",
            "color": color.tolist(),
            "roughness": roughness,
            "sdf_resolution": int(value(prim, "physxSDFMeshCollision:sdfResolution", 128)),
        }
        if material_kind is not None:
            shape["material"] = material_kind
        shapes.append(shape)

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type, drive_name = "revolute", "angular"
            axis = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
            gain_scale = mass_unit * unit**2 * 180.0 / np.pi
            position_scale = np.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type, drive_name = "prismatic", "linear"
            axis = str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get())
            gain_scale = mass_unit
            position_scale = unit
        elif prim.IsA(UsdPhysics.FixedJoint):
            joint_type, drive_name, axis = "fixed", "", "X"
            gain_scale = position_scale = 1.0
        else:
            raise ValueError(f"Unsupported joint type {prim.GetTypeName()}: {prim.GetPath()}")
        frames, ids = [], []
        for side in (0, 1):
            targets = prim.GetRelationship(f"physics:body{side}").GetTargets()
            target_path = str(targets[0]) if targets else None
            owner = stage.GetPrimAtPath(target_path) if target_path else None
            while owner and str(owner.GetPath()) not in body_ids:
                owner = owner.GetParent()
            body_path = str(owner.GetPath()) if owner else None
            ids.append(body_ids.get(body_path, -1))
            p = np.asarray(value(prim, f"physics:localPos{side}", (0, 0, 0)), dtype=float)
            q = value(prim, f"physics:localRot{side}", None)
            r = np.asarray(wp.quat_to_matrix(wp.quat(*q.GetImaginary(), q.GetReal()))).reshape(3, 3)
            if target_path:
                matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target_path)))
                p = (p @ matrix[:3, :3] + matrix[3, :3]) * unit
                target_rotation = rigid(stage.GetPrimAtPath(target_path))[1]
                r = target_rotation @ r
            if body_path:
                p = (p - positions[body_path]) @ rotations[body_path]
                r = rotations[body_path].T @ r
            elif not target_path:
                p *= unit
            frames.append(pose(p, r))
        drive = UsdPhysics.DriveAPI(prim, drive_name) if drive_name else None
        lower = float(value(prim, "physics:lowerLimit", -float("inf"))) * position_scale
        upper = float(value(prim, "physics:upperLimit", float("inf"))) * position_scale
        joints.append(
            {
                "label": str(prim.GetPath()),
                "type": joint_type,
                "parent": ids[0],
                "child": ids[1],
                "frames": frames,
                "axis": axis,
                "lower": lower,
                "upper": upper,
                "collision_enabled": bool(value(prim, "physics:collisionEnabled", False)),
                "drive_enabled": bool(drive),
                "stiffness": float(value(prim, f"drive:{drive_name}:physics:stiffness", 0.0)) * gain_scale,
                "damping": float(value(prim, f"drive:{drive_name}:physics:damping", 0.0)) * gain_scale,
                "target": float(value(prim, f"drive:{drive_name}:physics:targetPosition", 0.0)) * position_scale,
                "velocity": float(value(prim, f"drive:{drive_name}:physics:targetVelocity", 0.0)) * position_scale,
            }
        )
    payload = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "meters_per_unit": unit,
        "gravity": [0.0, 0.0, -9.81],
        "ground_height": 0.0,
        "bodies": bodies,
        "shapes": shapes,
        "joints": joints,
        "notes": "SI units, Z up. Authored body poses retained; saved simulation velocities intentionally reset.",
    }
    scene_output.parent.mkdir(parents=True, exist_ok=True)
    scene_output.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        '"""SI scene data extracted from Caterpillar.usd; see the asset README."""\n\n'
        "from math import inf\n\nSCENE = " + pprint.pformat(payload, width=120, sort_dicts=False) + "\n"
    )
    print(f"Extracted {len(bodies)} bodies, {len(shapes)} shapes, {len(joints)} joints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.output, args.scene_output)
