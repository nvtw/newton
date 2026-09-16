"""Run FrameGround without attached joints or interacting Flower geometry."""

import runpy

from newton.examples.kamino import example_kamino_colibri as source

original = source.build_scene

def build_scene(*args, **kwargs):
    assert kwargs["body_count"] == 1
    assert not kwargs["fix_base"]
    builder = original(*args, **kwargs)
    flower = builder.body_label.index("Flower")
    removed = 0
    for i, body in enumerate(builder.shape_body):
        if body == flower:
            builder.shape_flags[i] = 0
            removed += 1
    print("ISOLATED_BASE_DISABLED_FLOWER_SHAPES", removed, flush=True)
    return builder

source.build_scene = build_scene
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    source.build_scene = original
