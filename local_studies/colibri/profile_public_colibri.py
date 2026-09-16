# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Profile a configurable warmed interval of the public Colibri example."""

import argparse
import ctypes
import runpy
import sys

import warp as wp

import newton.examples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--capture-frames", type=int, default=2)
options = parser.parse_args()
if options.warmup < 0 or options.capture_frames < 1:
    parser.error("Require nonnegative warmup and at least one captured frame")

driver = ctypes.CDLL("libcuda.so.1")
original_run = newton.examples.run


def run(example, args):
    original_step = example.step
    frame = 0

    def step():
        nonlocal frame
        frame += 1
        if frame == options.warmup + 1:
            wp.synchronize_device(example.model.device)
            if driver.cuProfilerStart():
                raise RuntimeError("cuProfilerStart failed")
        original_step()
        if frame == options.warmup + options.capture_frames:
            wp.synchronize_device(example.model.device)
            if driver.cuProfilerStop():
                raise RuntimeError("cuProfilerStop failed")

    example.step = step
    return original_run(example, args)


newton.examples.run = run
sys.argv = [
    "phoenx_colibri",
    "--viewer",
    "null",
    "--num-frames",
    str(options.warmup + options.capture_frames + 1),
    "--substeps",
    "30",
]
try:
    runpy.run_module("newton.examples.phoenx.example_phoenx_colibri", run_name="__main__")
finally:
    newton.examples.run = original_run
