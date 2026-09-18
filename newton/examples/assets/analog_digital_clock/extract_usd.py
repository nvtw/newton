# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Extract AnalogDigitalClock_SI_Units geometry and SI scene descriptors from its USD."""

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


def extract(source: Path, destination: Path):
    """Preserve authored frames, materials, normals, and collision meshes."""
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("This extraction expects the authored Z-up AnalogDigitalClock_SI_Units stage")
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    up = np.eye(3)
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
        q = matrix.RemoveScaleShear().ExtractRotationQuat()
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
                    "angular_damping": float(value(prim, "physxRigidBody:angularDamping", 0.05)),
                }
            )

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith("/World/Clock/"):
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
        uvs = mesh.uvs
        if normals is None:
            # The CAD exports omit normals. Split sharp edges before smoothing so
            # planar cam faces and curved rims keep their intended appearance.
            visual = trimesh.visual.texture.TextureVisuals(uv=uvs) if uvs is not None else None
            smooth = trimesh.graph.smooth_shade(
                trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False),
                angle=np.deg2rad(30),
            )
            vertices, faces, normals = smooth.vertices, smooth.faces, smooth.vertex_normals
            if uvs is not None:
                uvs = smooth.visual.uv
        # Quantize below a micron to share repeated geometry despite USD float noise.
        vertices = np.round(vertices, 8).astype(np.float32)
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
                stream.write("# AnalogDigitalClock_SI_Units USD extraction; meters; explicit surface normals\n")
                for v in vertices:
                    stream.write("v " + " ".join(f"{x:.9g}" for x in v) + "\n")
                if normals is not None:
                    for n in normals:
                        stream.write("vn " + " ".join(f"{x:.9g}" for x in n) + "\n")
                if uvs is not None:
                    for uv in uvs:
                        stream.write("vt " + " ".join(f"{x:.9g}" for x in uv) + "\n")
                for face in faces + 1:
                    stream.write(
                        "f "
                        + " ".join(
                            f"{i}/{i if uvs is not None else ''}/{i}" if normals is not None else str(i) for i in face
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
        physical_material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        density, friction = 1000.0, 0.5
        if physical_material:
            density = float(value(physical_material.GetPrim(), "physics:density", density))
            if physical_material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                friction = float(UsdPhysics.MaterialAPI(physical_material.GetPrim()).GetDynamicFrictionAttr().Get())
        shapes.append(
            {
                "label": str(prim.GetPath()),
                "body": body,
                "mesh": filename,
                "center": center.tolist(),
                "collision": collision,
                "density": density * mass_unit / unit**3,
                "friction": friction,
                "approximation": str(value(prim, "physics:approximation", "none")),
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
                "collision_enabled": bool(value(prim, "physics:collisionEnabled", False)),
                "drive_enabled": bool(UsdPhysics.DriveAPI(prim, "angular")),
                "stiffness": float(value(prim, "drive:angular:physics:stiffness", 0)) * gain_scale,
                "damping": float(value(prim, "drive:angular:physics:damping", 0)) * gain_scale,
                "target": float(value(prim, "drive:angular:physics:targetPosition", 0)) * np.pi / 180,
                "velocity": float(value(prim, "drive:angular:physics:targetVelocity", 0)) * np.pi / 180,
            }
        )
    payload = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gravity": [0, 0, -9.800000190734863],
        "ground_height": float(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/GroundPlane")).ExtractTranslation()[2]
        )
        * unit,
        "bodies": bodies,
        "shapes": shapes,
        "joints": joints,
        "notes": "SI units, Z up. Authored body poses retained; startup velocities intentionally zero.",
    }
    data_path = destination.parent.parent / "phoenx" / "analog_digital_clock_scene.py"
    data_path.write_text(
        '# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers\n# SPDX-License-Identifier: Apache-2.0\n"""SI scene data extracted from AnalogDigitalClock_SI_Units.usd; see the asset README."""\n\nfrom math import inf\n\nSCENE = '
        + pprint.pformat(payload, width=120, sort_dicts=False)
        + "\n"
    )
    print(f"Extracted {len(bodies)} bodies, {len(shapes)} shapes, {len(joints)} joints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    extract(args.source, args.output)
