"""Diagnostic: authored two-body geometry and hinge, with its drive removed."""

import json
import runpy
import sys
from pathlib import Path

from newton.examples.kamino import example_kamino_colibri as source

original_joints = source.JOINTS
original_build = source.build_scene
output = Path(sys.argv[sys.argv.index("--output") + 1])
removed = [j[-1] for j in original_joints if j[0] == "FrameGround" and j[1] == "Frame"]
assert len(removed) == 1
source.JOINTS = [(*j[:-1], {}) if j[0] == "FrameGround" and j[1] == "Frame" else j for j in original_joints]
metadata = {
    "removed_drive": removed[0],
    "scope": "Diagnostic removal of spring and damping drive only; not a proposed scene fix",
}


def checked_build(*args, **kwargs):
    assert kwargs["body_count"] == 2
    builder = original_build(*args, **kwargs)
    assert all(x == 0.0 for x in builder.joint_target_ke)
    assert all(x == 0.0 for x in builder.joint_target_kd)
    metadata["target_ke"] = list(builder.joint_target_ke)
    metadata["target_kd"] = list(builder.joint_target_kd)
    print("UNACTUATED_CONTROL", json.dumps(metadata), flush=True)
    return builder


source.build_scene = checked_build
try:
    runpy.run_module("local_studies.colibri.staged_base_mechanism", run_name="__main__")
finally:
    source.build_scene = original_build
    source.JOINTS = original_joints
    output.with_suffix(".unactuated.json").write_text(json.dumps(metadata, indent=2))
