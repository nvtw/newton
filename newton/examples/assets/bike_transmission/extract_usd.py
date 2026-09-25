# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Extract BikeTransmission geometry and SI scene descriptors from its USD."""

import argparse
import hashlib
import pprint
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import newton.usd


def extract(source: Path, destination: Path):
    """Preserve authored frames, materials, normals, and collision meshes."""
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageUpAxis(stage) != "Y":
        raise ValueError("This extraction expects the authored Y-up BikeTransmission stage")
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    cache = UsdGeom.XformCache()
    destination.mkdir(parents=True, exist_ok=True)
    bodies, shapes, joints = [], [], []
    body_ids, rotations, positions = {}, {}, {}

    def value(prim, name, default):
        result = prim.GetAttribute(name).Get()
        return default if result is None else result

    def pose(position, rotation):
        q = wp.quat_from_matrix(wp.mat33(rotation.astype(np.float32)))
        return [*map(float, position), *map(float, q)]

    def rigid(prim):
        matrix = cache.GetLocalToWorldTransform(prim)
        q = matrix.ExtractRotationQuat()
        r = np.asarray(wp.quat_to_matrix(wp.quat(*q.GetImaginary(), q.GetReal()))).reshape(3, 3)
        return np.asarray(matrix.ExtractTranslation()) * unit @ up.T, up @ r

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

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith("/World/Xform/"):
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
        linear = rotation.T @ up @ matrix[:3, :3].T * unit
        vertices = np.asarray(mesh.vertices, dtype=np.float64) @ linear.T
        vertices += (matrix[3, :3] * unit @ up.T - position) @ rotation
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        vertices -= center
        normals = None
        if mesh.normals is not None:
            normals = np.asarray(mesh.normals) @ np.linalg.inv(linear)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        faces = np.asarray(mesh.indices).reshape(-1, 3).copy()
        if np.linalg.det(linear) < 0:
            faces = faces[:, ::-1]
        # Quantize below a micron to share repeated chain geometry despite USD float noise.
        vertices = np.round(vertices, 8).astype(np.float32)
        digest = hashlib.sha256(
            vertices.tobytes() + faces.tobytes() + (b"" if normals is None else np.round(normals, 5).tobytes())
        ).hexdigest()[:16]
        filename = f"mesh_{digest}.obj"
        target = destination / filename
        if not target.exists():
            with target.open("w") as stream:
                stream.write("# BikeTransmission USD extraction; meters; explicit authored normals\n")
                for v in vertices:
                    stream.write("v " + " ".join(f"{x:.9g}" for x in v) + "\n")
                if normals is not None:
                    for n in normals:
                        stream.write("vn " + " ".join(f"{x:.9g}" for x in n) + "\n")
                if mesh.uvs is not None:
                    for uv in mesh.uvs:
                        stream.write("vt " + " ".join(f"{x:.9g}" for x in uv) + "\n")
                for face in faces + 1:
                    stream.write(
                        "f "
                        + " ".join(
                            f"{i}/{i if mesh.uvs is not None else ''}/{i}" if normals is not None else str(i)
                            for i in face
                        )
                        + "\n"
                    )
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim))
        if subsets:
            materials = {
                str(UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0].GetPath()) for s in subsets
            }
            if len(materials) != 1:
                raise ValueError(f"Multiple face materials require splitting: {prim.GetPath()}")
            material = UsdShade.Material(stage.GetPrimAtPath(next(iter(materials))))
        color, roughness = (1.0, 1.0, 1.0), 0.5
        if material:
            for child in material.GetPrim().GetChildren():
                if child.IsA(UsdShade.Shader):
                    color = value(child, "inputs:diffuse_reflection_color", color)
                    roughness = float(value(child, "inputs:specular_reflection_roughness", roughness))
        color = np.asarray(color)
        color = np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)
        collision = prim.HasAPI(UsdPhysics.CollisionAPI) and bool(value(prim, "physics:collisionEnabled", True))
        shapes.append(
            {
                "label": str(prim.GetPath()),
                "body": body,
                "mesh": filename,
                "center": center.tolist(),
                "collision": collision,
                "visible": UsdGeom.Imageable(prim).ComputeVisibility() != "invisible",
                "color": color.tolist(),
                "roughness": roughness,
                "sdf_resolution": int(value(prim, "physxSDFMeshCollision:sdfResolution", 128)),
            }
        )

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        frames, ids = [], []
        for side in (0, 1):
            targets = prim.GetRelationship(f"physics:body{side}").GetTargets()
            path = str(targets[0]) if targets else None
            ids.append(body_ids.get(path, -1))
            p = np.asarray(value(prim, f"physics:localPos{side}", (0, 0, 0)), dtype=float)
            q = value(prim, f"physics:localRot{side}", None)
            r = np.asarray(wp.quat_to_matrix(wp.quat(*q.GetImaginary(), q.GetReal()))).reshape(3, 3)
            if path:
                matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                p = (p @ np.asarray(matrix)[:3, :3] + np.asarray(matrix)[3, :3]) * unit @ up.T
                _, body_rotation = rigid(stage.GetPrimAtPath(path))
                r = body_rotation @ r
                p = (p - positions[path]) @ rotations[path]
                r = rotations[path].T @ r
            else:
                p = p * unit @ up.T
                r = up @ r
            frames.append(pose(p, r))
        # USD angular gains use degrees; torque uses mass * length squared.
        gain_scale = mass_unit * unit**2 * 180.0 / np.pi
        joints.append(
            {
                "label": str(prim.GetPath()),
                "parent": ids[0],
                "child": ids[1],
                "frames": frames,
                "axis": str(joint.GetAxisAttr().Get()),
                "lower": float(joint.GetLowerLimitAttr().Get()) * np.pi / 180,
                "upper": float(joint.GetUpperLimitAttr().Get()) * np.pi / 180,
                "stiffness": float(value(prim, "drive:angular:physics:stiffness", 0)) * gain_scale,
                "damping": float(value(prim, "drive:angular:physics:damping", 0)) * gain_scale,
                "target": float(value(prim, "drive:angular:physics:targetPosition", 0)) * np.pi / 180,
                "velocity": float(value(prim, "drive:angular:physics:targetVelocity", 0)) * np.pi / 180,
            }
        )
    payload = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gravity": [0, 0, -9.8],
        "bodies": bodies,
        "shapes": shapes,
        "joints": joints,
        "notes": "SI units, Z up. Authored body poses retained; startup velocities intentionally zero.",
    }
    data_path = Path(__file__).resolve().parents[2] / "phoenx" / "bike_transmission_scene.py"
    data_path.write_text(
        '# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers\n# SPDX-License-Identifier: Apache-2.0\n"""SI scene data extracted from BikeTransmission.usd; see the asset README."""\n\nfrom math import inf\n\nSCENE = '
        + pprint.pformat(payload, width=120, sort_dicts=False)
        + "\n"
    )
    print(f"Extracted {len(bodies)} bodies, {len(shapes)} shapes, {len(joints)} joints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.output)
