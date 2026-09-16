# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Identical-input profile of canonical and cooperative-forward group sweeps."""

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from local_studies.colibri.cooperative_bilateral_forward import install
from local_studies.colibri.freeze_support_star import collect
from newton._src.solvers.phoenx.dispatch import color_groups
from newton.examples.phoenx.example_phoenx_colibri import Example


class Frozen(Exception):
    pass


def main():
    profile = "--profile" in sys.argv
    sys.argv = [sys.argv[0], "--viewer", "null", "--num-frames", "330"]
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    for _ in range(330):
        example.step()
    world = example.solver.world
    soft = bool(world._dispatch_specialization_flags()["has_soft_contact_pd"])
    baseline = color_groups.get_sweep_kernel("iterate", soft, cooperative_joints=True)
    original = wp.launch
    captured = {}

    def intercept(kernel, *positional, **keywords):
        if kernel is baseline:
            captured.update(positional=positional, keywords=keywords)
            raise Frozen
        return original(kernel, *positional, **keywords)

    wp.launch = intercept
    try:
        example.simulate()
    except Frozen:
        pass
    finally:
        wp.launch = original
    assert captured
    saved = collect(world)
    restore = install()
    candidate = color_groups.get_sweep_kernel("iterate", soft, cooperative_joints=True)
    restore()
    variants = {"canonical": baseline, "forward": candidate}
    expected = None
    for kernel in variants.values():
        for destination, source in saved:
            wp.copy(destination, source)
        original(kernel, *captured["positional"], **captured["keywords"])
        actual = [array.numpy().copy() for array, _ in saved]
        if expected is None:
            expected = actual
        else:
            assert all(a.tobytes() == b.tobytes() for a, b in zip(expected, actual, strict=True))
    driver = ctypes.CDLL("libcuda.so.1")
    if profile:
        assert driver.cuProfilerStart() == 0
    report = {"scope": "Same restored first biased group sweep after330frames", "arrays": len(saved), "variants": {}}
    for label, kernel in variants.items():
        start = wp.Event(world.device, enable_timing=True)
        end = wp.Event(world.device, enable_timing=True)
        with wp.ScopedCapture(device=world.device) as capture:
            for destination, source in saved:
                wp.copy(destination, source)
            wp.record_event(start, external=True)
            original(kernel, *captured["positional"], **captured["keywords"])
            wp.record_event(end, external=True)
        samples = []
        for _ in range(100):
            wp.capture_launch(capture.graph)
            samples.append(wp.get_event_elapsed_time(start, end))
        actual = [array.numpy() for array, _ in saved]
        assert all(a.tobytes() == b.tobytes() for a, b in zip(expected, actual, strict=True))
        report["variants"][label] = {
            "mean_us": float(np.mean(samples[10:]) * 1000),
            "median_us": float(np.median(samples[10:]) * 1000),
            "all_arrays_byte_exact": True,
        }
    if profile:
        assert driver.cuProfilerStop() == 0
    Path("/tmp/colibri_cooperative_forward_frozen.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
