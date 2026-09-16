# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated head-launch A/B using the existing single-world Kapla benchmark."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx import solver_phoenx
from newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla import _run_one
from newton._src.solvers.phoenx.examples import example_kapla_tower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-launches", type=int, choices=(1, 8), required=True)
    parser.add_argument("--mass-splitting", choices=("on", "off"), default="on")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    solver_phoenx.NUM_INNER_WHILE_ITERATIONS = args.head_launches
    instances = []
    original = example_kapla_tower.Example

    class CapturedExample(original):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            instances.append(self)

    example_kapla_tower.Example = CapturedExample
    mass_splitting = args.mass_splitting == "on"
    result = _run_one(
        mass_splitting=mass_splitting,
        substeps=6,
        solver_iterations=10 if mass_splitting else 6,
        max_colored_partitions=9,
        prepare_refresh_stride=1,
        warmup_frames=args.warmup,
        measured_frames=args.frames,
        grid_dims=(1, 1),
        blocks_per_sm=8,
        colored_contact_layout=True,
        partitioner_algorithm="endpoint_owner",
    )
    example = instances[-1]
    arrays = {
        name: getattr(example.bodies, name).numpy()
        for name in ("position", "orientation", "velocity", "angular_velocity")
    }
    arrays["contact_impulses"] = example.world._contact_container.impulses.numpy()
    arrays["contact_lambdas"] = example.world._contact_container.lambdas.numpy()
    report = asdict(result)
    report["head_launches"] = args.head_launches
    report["step_layout"] = example.world.step_layout
    report["source_sha256"] = hashlib.sha256(Path(solver_phoenx.__file__).read_bytes()).hexdigest()
    report["comparison"] = {}
    if args.reference:
        with np.load(args.reference.with_suffix(".npz")) as reference:
            for name, values in arrays.items():
                expected = reference[name]
                same = values.shape == expected.shape and np.array_equal(values.view(np.uint8), expected.view(np.uint8))
                report["comparison"][name] = {
                    "bitwise_equal": bool(same),
                    "max_abs_difference": float(np.max(np.abs(values - expected)))
                    if values.shape == expected.shape
                    else None,
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["finite"], "Nonfinite Kapla state"
    assert all(item["bitwise_equal"] for item in report["comparison"].values()), "Head launch schedules changed state"


if __name__ == "__main__":
    main()
