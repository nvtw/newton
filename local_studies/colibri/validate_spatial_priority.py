# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate a changed contact manifold without requiring the old Colibri trajectory."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def compare_arrays(reference, candidate):
    """Require identical array keys, representations, and values."""
    with np.load(reference) as expected, np.load(candidate) as actual:
        return {
            key: key in expected.files
            and key in actual.files
            and expected[key].shape == actual[key].shape
            and expected[key].dtype == actual[key].dtype
            and expected[key].tobytes() == actual[key].tobytes()
            for key in sorted(set(expected.files) | set(actual.files))
        }


def main():
    """Check repeatability, existing cross-scene references, and optional long runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-frames", type=int, default=0)
    parser.add_argument(
        "--search-floor",
        action="store_true",
        help="Validate the local 2 mm search-extension floor without changing production defaults.",
    )
    args = parser.parse_args()
    if args.long_frames < 0:
        parser.error("--long-frames must be nonnegative")
    root = Path(__file__).resolve().parents[2]

    def fingerprints():
        paths = list((root / "newton").rglob("*.py"))
        paths += [p for p in (root / "newton/examples/assets/colibri").rglob("*") if p.is_file()]
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}

    original = fingerprints()
    report = {"source_asset_sha256": original, "search_floor": args.search_floor, "cases": {}, "passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run_case(name, module, arguments, reference=None):
        prefix = args.output.with_name(args.output.stem + "_" + name)
        prefix_modules = ("check_clean_velocity_iterations", "check_live_search_floor")
        output = str(prefix) if module.rsplit(".", 1)[-1] in prefix_modules else str(prefix) + ".json"
        command = ["uv", "run", "--no-project", sys.executable, "-m", module, *arguments, "--output", output]
        print("START", name, flush=True)
        with Path(str(prefix) + ".log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        case = {"command": command, "returncode": result.returncode, "output_prefix": str(prefix)}
        report["cases"][name] = case
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {prefix}.log")
        if reference is not None:
            case["byte_equal"] = compare_arrays(reference, str(prefix) + ".npz")
            if not case["byte_equal"] or not all(case["byte_equal"].values()):
                raise AssertionError(f"{name} changed saved arrays: {case['byte_equal']}")
        if fingerprints() != original:
            raise RuntimeError("Production source or scene assets changed during validation")
        case["passed"] = True
        args.output.write_text(json.dumps(report, indent=2))
        print("PASS", name, flush=True)
        return Path(str(prefix) + ".npz")

    try:
        module = (
            "local_studies.colibri.check_live_search_floor"
            if args.search_floor
            else "local_studies.colibri.check_clean_velocity_iterations"
        )
        for count in (1, 4):
            arguments = ["--velocity-iterations", str(count), "--frames", "330"]
            reference = run_case(f"velocity{count}_first", module, arguments)
            run_case(f"velocity{count}_repeat", module, arguments, reference)
        run_case(
            "kapla",
            "local_studies.colibri.check_kapla_head_launches",
            ["--head-launches", "8", "--frames", "180"],
            Path("/tmp/kapla_rebased_packed_fixed.npz"),
        )
        run_case(
            "g1",
            "local_studies.colibri.check_g1_policy",
            ["--frames", "20"],
            Path("/tmp/g1_cache_only_candidate.npz"),
        )
        if args.long_frames:
            for count in (1, 4):
                run_case(
                    f"velocity{count}_long",
                    module,
                    ["--velocity-iterations", str(count), "--frames", str(args.long_frames)],
                )
        report["passed"] = True
    finally:
        report["source_assets_unchanged"] = fingerprints() == original
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
