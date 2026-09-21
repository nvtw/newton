"""Run an authored connected prefix with noninteracting Flower geometry."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from newton.examples.kamino import example_kamino_colibri as source

original = source.build_scene
source_path = Path(source.__file__)
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
metadata = {}
original_argv = sys.argv.copy()
output_path = Path(original_argv[original_argv.index("--output") + 1]) if "--output" in original_argv else None


def build_scene(*args, **kwargs):
    """Keep authored dynamic bodies, joints and materials; disable only Flower shapes."""
    count = kwargs["body_count"]
    assert 2 <= count <= len(source.BODY_ORDER)
    assert not kwargs["fix_base"]
    builder = original(*args, **kwargs)
    flower = builder.body_label.index("Flower")
    removed = []
    for i, body in enumerate(builder.shape_body):
        if body == flower:
            builder.shape_flags[i] = 0
            removed.append(builder.shape_label[i])
    selected = set(source.BODY_ORDER[:count])
    metadata.update(
        scope="Authored connected prefix; dynamic base; only Flower shape flags disabled",
        body_count=count,
        bodies=list(builder.body_label),
        joints=list(builder.joint_label),
        authored_joints=[j for j in source.JOINTS if j[0] in selected and j[1] in selected],
        disabled_flower_shapes=removed,
        source=str(source_path),
        source_sha256=source_hash,
        argv=original_argv,
    )
    print("STAGED_BASE_METADATA", json.dumps(metadata), flush=True)
    return builder


source.build_scene = build_scene
try:
    runpy.run_module(
        os.environ.get("COLIBRI_STAGE_RUNNER", "local_studies.colibri.check_public_analytic_gradient"),
        run_name="__main__",
    )
finally:
    source.build_scene = original
    unchanged = hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    metadata["source_unchanged"] = unchanged
    if output_path is not None:
        output_path.with_suffix(".stage.json").write_text(json.dumps(metadata, indent=2))
    assert unchanged
