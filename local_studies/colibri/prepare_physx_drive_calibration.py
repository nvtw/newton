"""Prepare a source-drive calibration with zero gravity and no contacts."""

# ruff: noqa: TID253 -- standalone optional-dependency diagnostic.

import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

output = "/tmp/colibri_physx_drive_calibration.usda"
stage = Usd.Stage.CreateNew(output)
stage.GetRootLayer().subLayerPaths = ["/tmp/colibri_physx_two_body_awake.usda"]
scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/Colibri/PhysicsScene"))
scene.CreateGravityMagnitudeAttr(0.0)
disabled = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        disabled.append(str(prim.GetPath()))
stage.GetRootLayer().Save()
joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
assert len(joints) == 1
joint = UsdPhysics.Joint(joints[0])
cache = UsdGeom.XformCache()
poses = []
for path in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets():
    matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    q = matrix.RemoveScaleShear().ExtractRotationQuat()
    poses.append([*[float(x) * 0.01 for x in matrix.ExtractTranslation()], *q.GetImaginary(), q.GetReal()])
Path(output).with_suffix(".fixture.json").write_text(
    json.dumps(
        {
            "usd": output,
            "source": list(stage.GetRootLayer().subLayerPaths),
            "gravity_m_s2": 0,
            "disabled_colliders": disabled,
            "joint": str(joints[0].GetPath()),
            "joint_attributes": {a.GetName(): str(a.Get()) for a in joints[0].GetAttributes()},
            "initial_source_body_poses_si": poses,
        },
        indent=2,
    )
)
print(output, "disabled colliders", len(disabled))
