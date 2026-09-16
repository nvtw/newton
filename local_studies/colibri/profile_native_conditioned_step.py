"""Profile one warmed native two-body frame after friction overlays install."""

import ctypes
import json
import os
import runpy
from pathlib import Path

import warp as wp

import newton.examples

frame_target = int(os.environ.get("COLIBRI_PROFILE_FRAME", "331"))
original_run = newton.examples.run
driver = ctypes.CDLL("libcuda.so.1")
driver.cuProfilerStart.restype = ctypes.c_int
driver.cuProfilerStop.restype = ctypes.c_int
captured = []


def run(example, args):
    """Profile only step execution, excluding post-step physical test queries."""
    original_step = example.step
    frame = 0

    def step():
        nonlocal frame
        frame += 1
        capture = frame == frame_target
        if capture:
            wp.synchronize_device(example.model.device)
            if driver.cuProfilerStart():
                raise RuntimeError("cuProfilerStart failed")
        try:
            original_step()
        finally:
            if capture:
                wp.synchronize_device(example.model.device)
                if driver.cuProfilerStop():
                    raise RuntimeError("cuProfilerStop failed")
                captured.append(frame)

    example.step = step
    return original_run(example, args)


newton.examples.run = run
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    newton.examples.run = original_run
    assert captured == [frame_target], captured
    # check_public_analytic_gradient changes argv; retain output via environment.
    output = Path(os.environ["COLIBRI_PROFILE_METADATA"])
    output.write_text(
        json.dumps(
            {
                "captured_frame": frame_target,
                "collision_intervals": 2,
                "substeps_per_interval": 30,
                "iterations": 1,
                "scope": "One warmed actual example.step, including collision and captured graph nodes; excludes physical post-step test queries. No solver changes.",
                "differential_reference": os.environ.get("COLIBRI_DIFFERENTIAL_REFERENCE") == "1",
            },
            indent=2,
        )
    )
