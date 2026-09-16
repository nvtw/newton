"""Run existing G1 timing/profile harnesses with source and reference gates."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/tmp/g1_current_cost_reaudit")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prefix = Path(args.prefix)
    asset = Path("/home/twidmer/.cache/newton/newton-assets_unitree_g1_2c175d66_f8fb7abc/unitree_g1")
    assert asset.is_dir(), "Expected already-cached G1 assets are missing"
    files = sorted((root / "newton").rglob("*.py")) + sorted(p for p in asset.rglob("*") if p.is_file())

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    before = hashes()
    report = {"source_asset_hashes": before, "passed": False, "commands": []}
    report_path = prefix.with_suffix(".validation.json")
    base = ["uv", "run", "--no-project", sys.executable]
    command = [
        *base,
        "-u",
        "-m",
        "local_studies.colibri.check_g1_policy",
        "--frames",
        "20",
        "--benchmark-frames",
        "100",
        "--output",
        str(prefix.with_suffix(".json")),
    ]
    try:
        hardware = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        report["hardware_before"] = hardware.stdout
        report["commands"].append(command)
        with prefix.with_suffix(".log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = np.load(prefix.with_suffix(".npz"))
        reference = np.load("/tmp/g1_cache_only_candidate.npz")
        assert set(actual.files) == set(reference.files)
        report["byte_reference"] = {}
        for key in reference.files:
            a, b = actual[key], reference[key]
            equal = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            report["byte_reference"][key] = equal
            assert equal, key
        report["timing"] = json.loads(prefix.with_suffix(".json").read_text())["timing"]
        assert hashes() == before
        if args.profile:
            command = [
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--cuda-graph-trace=node",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "-o",
                str(prefix) + "_nodes",
                *base,
                "-u",
                "-m",
                "local_studies.colibri.profile_g1_policy",
            ]
            report["commands"].append(command)
            with Path(str(prefix) + "_nodes.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        report["passed"] = True
    finally:
        report["source_assets_unchanged"] = hashes() == before
        report_path.write_text(json.dumps(report, indent=2))
        assert report["source_assets_unchanged"]


if __name__ == "__main__":
    main()
