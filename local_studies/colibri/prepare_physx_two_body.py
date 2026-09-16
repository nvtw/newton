"""Make isolated source-USD two-body controls without editing source assets."""

import argparse
import json
from pathlib import Path


def main():
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--awake", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = "/tmp/colibri_physx_at_rest_no_damping.usda"
    stage = Usd.Stage.CreateNew(args.output)
    stage.GetRootLayer().subLayerPaths = [source]
    keep = {"/World/Colibri/FrameAsm/FrameGround", "/World/Colibri/FrameAsm/Frame"}
    disabled = [
        str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()) not in keep
    ]
    for path in disabled:
        stage.OverridePrim(path).SetActive(False)
    joints = []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = [str(p) for p in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()]
            if set(targets) != keep:
                prim.SetActive(False)
            else:
                joints.append(str(prim.GetPath()))
    assert len(joints) == 1
    for path in keep:
        p = stage.GetPrimAtPath(path)
        # Explicitly apply schemas so zero damping and optional sleep controls
        # are recognized even on source bodies lacking PhysxRigidBodyAPI.
        p.AddAppliedSchema("PhysxRigidBodyAPI")
        for name, value in (("linearDamping", 0.0), ("angularDamping", 0.0)):
            p.CreateAttribute("physxRigidBody:" + name, Sdf.ValueTypeNames.Float).Set(value)
        if args.awake:
            p.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    scene = stage.GetPrimAtPath("/World/Colibri/PhysicsScene")
    scene.AddAppliedSchema("PhysxSceneAPI")
    for name, value in (
        ("minPositionIterationCount", 30),
        ("maxPositionIterationCount", 30),
        ("minVelocityIterationCount", 1),
        ("maxVelocityIterationCount", 1),
    ):
        scene.CreateAttribute("physxScene:" + name, Sdf.ValueTypeNames.Int).Set(value)
    scene.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    scene.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    scene.CreateAttribute("physxScene:enableStabilization", Sdf.ValueTypeNames.Bool).Set(False)
    stage.GetRootLayer().Save()
    xcache = UsdGeom.XformCache()
    report = {
        "source": source,
        "output": args.output,
        "awake": args.awake,
        "kept_bodies": sorted(keep),
        "disabled_bodies": disabled,
        "joints": joints,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "scene": {a.GetName(): str(a.Get()) for a in scene.GetAttributes()},
        "body_attributes": {},
        "source_world_matrices": {},
    }
    for path in sorted(keep):
        prim = stage.GetPrimAtPath(path)
        report["body_attributes"][path] = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()}
        report["source_world_matrices"][path] = [list(row) for row in xcache.GetLocalToWorldTransform(prim)]
    report["joint_attributes"] = {a.GetName(): str(a.Get()) for a in stage.GetPrimAtPath(joints[0]).GetAttributes()}
    Path(args.output).with_suffix(".fixture.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("body_attributes", "source_world_matrices", "joint_attributes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
