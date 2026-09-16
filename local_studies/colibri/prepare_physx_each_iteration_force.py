"""Author one PhysX scene flag and certify all other composed attributes."""

import hashlib
import json
from pathlib import Path

BASE = Path("/tmp/colibri_physx_two_body_awake_plane1mm.usda")
OUTPUT = Path("/tmp/colibri_physx_two_body_awake_plane1mm_eachforce.usda")
ATTRIBUTE = "/World/Colibri/PhysicsScene.physxScene:enableExternalForcesEveryIteration"


def snapshot(stage):
    """Capture composed typed values, relationships and physical prim identity."""
    out = {}
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        out[path + ":prim"] = (prim.GetTypeName(), prim.IsActive(), tuple(prim.GetAppliedSchemas()))
        for attr in prim.GetAttributes():
            out[str(attr.GetPath())] = (str(attr.GetTypeName()), str(attr.Get()))
        for rel in prim.GetRelationships():
            out[str(rel.GetPath())] = tuple(map(str, rel.GetTargets()))
    return out


def main():
    """Permit exactly the external-force timing boolean to change."""
    from pxr import Sdf, Usd

    old = Usd.Stage.Open(str(BASE))
    digest = hashlib.sha256(BASE.read_bytes()).hexdigest()
    expected = json.loads(Path("/tmp/colibri_physx_two_body_awake_plane1mm60.json").read_text())["source_sha256"]
    assert digest == expected
    layer = Sdf.Layer.CreateNew(str(OUTPUT))
    layer.subLayerPaths = [str(BASE)]
    stage = Usd.Stage.Open(layer)
    prim = stage.OverridePrim("/World/Colibri/PhysicsScene")
    prim.CreateAttribute("physxScene:enableExternalForcesEveryIteration", Sdf.ValueTypeNames.Bool, custom=False).Set(
        True
    )
    layer.Save()
    before, after = snapshot(old), snapshot(stage)
    differences = {
        k: [before.get(k), after.get(k)] for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    }
    assert set(differences) == {ATTRIBUTE}, differences
    assert old.GetPrimAtPath("/World/Colibri/PhysicsScene").GetAttribute(
        "physxScene:enableExternalForcesEveryIteration"
    ).Get() in (None, False)
    assert (
        stage.GetPrimAtPath("/World/Colibri/PhysicsScene")
        .GetAttribute("physxScene:enableExternalForcesEveryIteration")
        .Get()
        is True
    )
    report = {
        "base": str(BASE),
        "base_sha256": digest,
        "overlay": str(OUTPUT),
        "overlay_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "composed_differences": differences,
        "scope": "Single scene external-force timing flag; no other composed values changed",
    }
    OUTPUT.with_suffix(".fixture.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
