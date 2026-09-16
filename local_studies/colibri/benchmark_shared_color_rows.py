"""Frozen actual Colibri endpoint graph timing; local candidate only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.check_shared_color_rows import oracle
from local_studies.colibri.shared_color_rows import build
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups

source = Path("/tmp/colibri_public_internals_first.internals.npz")
snapshot = np.load(source)
raw = snapshot["world._elements"]
count = int(snapshot["world._num_active_constraints"][0])
bodies = int(raw["bodies"][:count].max()) + 1
wp.init()
elements = wp.array(raw, dtype=ElementInteractionData, device="cuda:0")
active = wp.array([count], dtype=wp.int32, device="cuda:0")
states = [color_groups.allocate(len(raw), bodies, "cuda:0") for _ in range(2)]
expected = {k: v.numpy() for k, v in states[0].items()}
oracle(expected, raw["bodies"], count, 4)
graphs = {}
for name, state, builder in zip(("reference", "shared"), states, (color_groups.build, build), strict=True):
    builder(state, elements, active, 4, "cuda:0", rigid_only=True)
    for key, value in state.items():
        assert value.numpy().tobytes() == expected[key].tobytes(), (name, key)
    with wp.ScopedCapture(device="cuda:0") as capture:
        for _ in range(128):
            builder(state, elements, active, 4, "cuda:0", rigid_only=True)
    graphs[name] = capture.graph
    for _ in range(3):
        wp.capture_launch(capture.graph)
wp.synchronize_device("cuda:0")
samples = {name: [] for name in graphs}
stream = wp.get_stream("cuda:0")
for repeat in range(12):
    for name in list(graphs) if repeat % 2 == 0 else list(reversed(graphs)):
        start = wp.Event("cuda:0", enable_timing=True)
        end = wp.Event("cuda:0", enable_timing=True)
        stream.record_event(start)
        wp.capture_launch(graphs[name])
        stream.record_event(end)
        samples[name].append(wp.get_event_elapsed_time(start, end) / 128)
report = {name: {"median_ms": float(np.median(values)), "samples_ms": values} for name, values in samples.items()}
report.update(
    source=str(source),
    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    rows=count,
    capacity=len(raw),
    bodies=bodies,
    colors=int(expected["num_colors"][0]),
    scope="Historical actual first-frame Colibri graph; current canonical rigid builder vs local shared cache including same two zero fills; not current trajectory or full frame timing",
)
Path("/tmp/shared_color_rows_frozen_timing.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
