"""Create an all-awake full-scene timing companion changing numerical flags only."""

import hashlib
import json
from pathlib import Path


def snapshot(stage):
    """Capture composed prim properties, relationships and applied API identities."""
    result = {}
    for prim in stage.Traverse():
        name = str(prim.GetPath())
        result[name] = {
            "type": prim.GetTypeName(),
            "apis": list(prim.GetAppliedSchemas()),
            "active": prim.IsActive(),
            "attributes": {a.GetName(): repr(a.Get()) for a in prim.GetAttributes()},
            "relationships": {r.GetName(): list(map(str, r.GetTargets())) for r in prim.GetRelationships()},
        }
    return result


def main():
    """Verify that only explicitly listed solver/sleep flags differ."""
    from pxr import Sdf, Usd, UsdPhysics

    source = Path("/tmp/colibri_physx_at_rest_no_damping.usda")
    output = Path("/tmp/colibri_physx_full37_awake.usda")
    assert not output.exists(), "Do not overwrite a prior diagnostic overlay"
    old = Usd.Stage.Open(str(source))
    before = snapshot(old)
    bodies = [p for p in old.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 37
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    allowed = set()
    for prim in bodies:
        name = str(prim.GetPath())
        p = stage.OverridePrim(name)
        p.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        allowed.add((name, "physxRigidBody:sleepThreshold"))
    scene_path = "/World/Colibri/PhysicsScene"
    scene = stage.GetPrimAtPath(scene_path)
    flags = {
        "physxScene:enableStabilization": (Sdf.ValueTypeNames.Bool, False),
        "physxScene:minPositionIterationCount": (Sdf.ValueTypeNames.Int, 30),
        "physxScene:maxPositionIterationCount": (Sdf.ValueTypeNames.Int, 30),
        "physxScene:minVelocityIterationCount": (Sdf.ValueTypeNames.Int, 1),
        "physxScene:maxVelocityIterationCount": (Sdf.ValueTypeNames.Int, 1),
    }
    for name, (kind, value) in flags.items():
        scene.CreateAttribute(name, kind).Set(value)
        allowed.add((scene_path, name))
    stage.GetRootLayer().Save()
    after = snapshot(Usd.Stage.Open(str(output)))
    assert before.keys() == after.keys()
    changes = []
    for path in before:
        for key in ("type", "apis", "active", "relationships"):
            assert before[path][key] == after[path][key], (path, key)
        a, b = before[path]["attributes"], after[path]["attributes"]
        for name in a.keys() | b.keys():
            if a.get(name) != b.get(name):
                assert (path, name) in allowed, (path, name, a.get(name), b.get(name))
                changes.append({"prim": path, "attribute": name, "before": a.get(name), "after": b.get(name)})
    for prim in bodies:
        assert stage.GetPrimAtPath(prim.GetPath()).GetAttribute("physxRigidBody:sleepThreshold").Get() == 0
    report = {
        "status": "only_allowed_numerical_flags_changed",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "body_count": len(bodies),
        "changes": changes,
        "authored_flags": {name: value for name, (_, value) in flags.items()},
        "all_body_sleep_threshold": 0,
        "unchanged": "All other composed attributes, relationships, prim types and APIs, including physical properties, collision geometry/filtering and gravity",
        "time_step": "Runner unchanged:2*1/120s per frame",
    }
    output.with_suffix(".fixture.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
