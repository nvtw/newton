"""Make an exactly closed, zero-angle source hinge for drive calibration only."""

# ruff: noqa: TID253 -- standalone USD diagnostic.

import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

source = "/tmp/colibri_physx_drive_calibration.usda"
output = "/tmp/colibri_physx_closed_drive.usda"
stage = Usd.Stage.CreateNew(output)
stage.GetRootLayer().subLayerPaths = [source]
bodies = ["/World/Colibri/FrameAsm/FrameGround", "/World/Colibri/FrameAsm/Frame"]
cache = UsdGeom.XformCache()
source_scales = []
for path in bodies:
    prim = stage.GetPrimAtPath(path)
    original = cache.GetLocalToWorldTransform(prim)
    scale = Gf.Transform(original).GetScale()
    np.testing.assert_allclose(scale, [1.5, 1.5, 1.5], atol=1e-6, rtol=0)
    source_scales.append(list(scale))
    transform = UsdGeom.Xformable(prim)
    op = transform.MakeMatrixXform()
    op.Set(Gf.Matrix4d().SetScale(Gf.Vec3d(1.5)))
    transform.SetResetXformStack(True)
stage.GetRootLayer().Save()
cache.Clear()
world = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in bodies]
np.testing.assert_array_equal(np.array(world[0]), np.array(world[1]))
joint_prim = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))
joint = UsdPhysics.Joint(joint_prim)
np.testing.assert_array_equal(joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get())
assert joint.GetLocalRot0Attr().Get() == joint.GetLocalRot1Attr().Get()
anchors = [world[i].Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get())) for i in range(2)]
np.testing.assert_array_equal(anchors[0], anchors[1])
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        assert not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    if prim.IsA(UsdPhysics.Scene):
        assert UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Get() == 0.0
report = {
    "usd": output,
    "source": source,
    "source_world_scales": source_scales,
    "scope": "Calibration initializer only: both actors identity/origin, source1.5scale, no contacts/gravity; source inertial data/local joint frames/drive unchanged",
    "initial_source_body_poses_si": [[0, 0, 0, 0, 0, 0, 1]] * 2,
    "world_hinge_cm": list(anchors[0]),
    "joint_attributes": {a.GetName(): str(a.Get()) for a in joint_prim.GetAttributes()},
    "exact_composed_initial_closure": True,
}
Path(output).with_suffix(".fixture.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
