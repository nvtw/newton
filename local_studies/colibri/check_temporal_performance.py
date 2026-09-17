# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check saved full-trajectory performance regressions on the same hardware.

Collect runs with profile_temporal_performance --validate. Bracket a control
with repeated candidates; this comparison does not remove machine noise.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.minimum_speedup) or args.minimum_speedup < 1.0:
        parser.error("minimum speedup must be finite and at least one")
    baseline = json.loads(args.baseline.with_suffix(".json").read_text())
    with np.load(args.baseline.with_suffix(".npz")) as reference:
        for path in args.candidate:
            report = json.loads(path.with_suffix(".json").read_text())
            for name in ("frames", "warmup", "collision_hz", "substeps_per_refresh", "rendering", "motor_enabled"):
                assert report[name] == baseline[name], (path, name)
            assert report.get("num_worlds", 1) == baseline.get("num_worlds", 1), (path, "num_worlds")
            assert not report["rendering"]
            for name in ("peak_depth_m", "joint_peaks", "support_failure"):
                assert report[name] == baseline[name], (path, name)
            assert report["peak_depth_m"] is not None, "Collect runs with --validate"
            with np.load(path.with_suffix(".npz")) as candidate:
                for name in ("labels", "poses", "velocities"):
                    a, b = reference[name], candidate[name]
                    assert a.shape == b.shape and a.dtype == b.dtype, (path, name)
                    assert a.tobytes() == b.tobytes(), (path, name)
                assert candidate["poses"].shape[0] == report["frames"] + report["warmup"]
                a, b = reference["times_ms"], candidate["times_ms"]
                assert a.shape == b.shape == (report["frames"],)
                assert np.all(np.isfinite(a)) and np.all(a > 0)
                assert np.all(np.isfinite(b)) and np.all(b > 0)
                speedup = float(np.mean(a) / np.mean(b))
                print(f"{path}: {speedup:.5f}x speedup; exact trajectory and quality parity", flush=True)
                assert speedup >= args.minimum_speedup, (path, speedup, args.minimum_speedup)


if __name__ == "__main__":
    main()
