"""Compare first-fit coloring with an exact five-step first-free-bit search."""

import argparse
import json
import time

import numpy as np
import warp as wp

from local_studies.colibri import color_groups_linear_reference as reference
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups as candidate


def run(device, repeats):
    """Check full frozen topology and changing active prefixes before timing."""
    saved = np.load("/tmp/colibri_splitprep_trace_reference.npz")
    bodies = saved["0_before_constraint_bodies"]
    total = int(saved["0_before_active_count"][0])
    elements = wp.zeros(len(bodies), dtype=ElementInteractionData, device=device)
    host = elements.numpy()
    host["bodies"] = bodies
    elements.assign(host)
    active = wp.zeros(1, dtype=wp.int32, device=device)
    nodes = int(bodies.max()) + 1
    results = {}
    arrays = [m.allocate(len(bodies), nodes, device) for m in (reference, candidate)]
    for width in (1, 4, 8, 33):
        for count in (total, 33, 1, 0, total):
            active.assign(np.array([count], dtype=np.int32))
            for module, data in zip((reference, candidate), arrays, strict=True):
                module.build(data, elements, active, width, device)
            for key in arrays[0]:
                np.testing.assert_array_equal(arrays[0][key].numpy(), arrays[1][key].numpy(), err_msg=key)
    if device != "cpu":
        for label, module, data in zip(("linear", "binary"), (reference, candidate), arrays, strict=True):
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(100):
                    module.build(data, elements, active, 4, device)
            for _ in range(3):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                wp.capture_launch(capture.graph)
                wp.synchronize()
                samples.append((time.perf_counter() - start) * 10.0)
            results[label + "_ms"] = float(np.median(samples))
    return dict(device=device, active_rows=total, equality="all buffers exact", **results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="/tmp/colibri_color_binary.json")
    args = parser.parse_args()
    report = run(args.device, args.repeats)
    print(json.dumps(report, indent=2))
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
