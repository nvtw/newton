# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run canonical cross-scene checks and compare saved physical arrays exactly."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--colibri-frames", type=int, default=18000)
    args = parser.parse_args()
    if args.colibri_frames != 18000:
        raise ValueError("The saved full-trajectory reference contains exactly18000frames")
    root = Path(__file__).resolve().parents[2]
    sources = [
        "newton/_src/solvers/phoenx/constraints/bilateral_joint.py",
        "newton/_src/solvers/phoenx/dispatch/color_groups.py",
        "newton/_src/solvers/phoenx/solver_phoenx.py",
    ]

    def fingerprints():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}

    original = fingerprints()
    report = {"source_sha256": original, "cases": {}, "passed": False}
    cases = [
        (
            "colibri_smoke",
            "local_studies.colibri.check_public_analytic_gradient",
            ["--frames", "330", "--substeps", "30", "--save-history"],
            "/tmp/colibri_fused_begin_baseline_ab330.npz",
        ),
        (
            "kapla",
            "local_studies.colibri.check_kapla_head_launches",
            ["--head-launches", "8", "--frames", "180", "--reference", "/tmp/kapla_rebased_packed_fixed.npz"],
            "/tmp/kapla_rebased_packed_fixed.npz",
        ),
        ("g1", "local_studies.colibri.check_g1_policy", ["--frames", "20"], "/tmp/g1_cache_only_candidate.npz"),
        (
            "colibri",
            "local_studies.colibri.check_public_analytic_gradient",
            ["--frames", "18000", "--substeps", "30", "--save-history"],
            "/tmp/colibri_public_geometric_relative18000.npz",
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, module, arguments, reference_path in cases:
            prefix = args.output.with_name(args.output.stem + "_" + name)
            result_path = prefix.with_suffix(".json")
            command = [
                "uv",
                "run",
                "--no-project",
                sys.executable,
                "-m",
                module,
                *arguments,
                "--output",
                str(result_path),
            ]
            print("START", name, flush=True)
            with prefix.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
            case = {"command": command, "returncode": result.returncode, "result": str(result_path)}
            report["cases"][name] = case
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {prefix.with_suffix('.log')}")
            with np.load(reference_path) as expected, np.load(prefix.with_suffix(".npz")) as actual:
                matches = {
                    key: key in actual.files
                    and expected[key].shape == actual[key].shape
                    and expected[key].dtype == actual[key].dtype
                    and expected[key].tobytes() == actual[key].tobytes()
                    for key in expected.files
                }
            case["reference"] = reference_path
            case["byte_equal"] = matches
            case["passed"] = all(matches.values())
            assert case["passed"], (name, matches)
            assert fingerprints() == original, "Production source changed during validation"
            args.output.write_text(json.dumps(report, indent=2))
            print("PASS", name, len(matches), "arrays", flush=True)
        report["passed"] = True
    finally:
        report["source_unchanged"] = fingerprints() == original
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
