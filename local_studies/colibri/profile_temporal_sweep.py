# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Time one temporal sweep on restored inputs; this is not a trajectory benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.freeze_support_star import collect
from newton._src.solvers.phoenx.dispatch.color_groups_tgs import get_sweep_kernel
from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerNull


class Frozen(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    example = Example(ViewerNull(), Example.create_parser().parse_args(["--num-worlds", "1"]))
    for _ in range(60):
        example.step()
    world = example.solver.world
    wanted = get_sweep_kernel("iterate", cooperative_joints=True, temporal_springs=world._temporal_joint_springs)
    original_launch = wp.launch
    captured = {}

    def intercept(kernel, *positional, **keywords):
        if kernel is wanted:
            captured.update(positional=positional, keywords=keywords)
            raise Frozen
        return original_launch(kernel, *positional, **keywords)

    wp.launch = intercept
    try:
        example.simulate()
    except Frozen:
        pass
    finally:
        wp.launch = original_launch
    assert captured, "Temporal iterate was not intercepted"
    saved = collect(world)
    outputs = {
        "velocity": world._copy_state.velocity,
        "angular_velocity": world._copy_state.angular_velocity,
        "contact_impulses": world._contact_container.impulses,
        "joint_impulses": world.constraints.bilateral.accumulated,
    }
    original_launch(wanted, *captured["positional"], **captured["keywords"])
    expected = {name: array.numpy().copy() for name, array in outputs.items()}
    start = wp.Event(world.device, enable_timing=True)
    end = wp.Event(world.device, enable_timing=True)
    with wp.ScopedCapture(device=world.device) as capture:
        for destination, source in saved:
            wp.copy(destination, source)
        wp.record_event(start, external=True)
        original_launch(wanted, *captured["positional"], **captured["keywords"])
        wp.record_event(end, external=True)
    times = []
    for _ in range(args.samples + 20):
        wp.capture_launch(capture.graph)
        times.append(wp.get_event_elapsed_time(start, end))
    for name, array in outputs.items():
        assert array.numpy().tobytes() == expected[name].tobytes(), name
    report = {
        "scope": "One frozen biased sweep after 60 warmup frames; restore copies excluded",
        "samples": args.samples,
        "restored_arrays": len(saved),
        "mean_ms": float(np.mean(times[20:])),
        "median_ms": float(np.median(times[20:])),
        "p95_ms": float(np.percentile(times[20:], 95)),
        "repeat_bits_equal": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), times_ms=times[20:], **expected)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
