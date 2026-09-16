"""Audit actual joint-recovery restoration after conditioned contact solves."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    x = np.load(args.snapshot)["history"].astype(float)
    assert x.shape[2] >= 102
    pairs = [(a, b) for a, b in itertools.pairwise(x) if int(a[0, 92]) == 42 and int(b[0, 92]) == 43]
    assert pairs
    restored = np.array([b[:4, 85:91] - a[:4, 85:91] for a, b in pairs])
    expected = np.array([b[:4, 96:102] for a, b in pairs])
    physical = np.array([a[:4, 85:91] for a, b in pairs])
    total = np.array([b[:4, 85:91] for a, b in pairs])
    report = {
        "restoration_pairs": len(pairs),
        "max_actual_minus_bias_mixed_units": float(np.max(abs(restored - expected))),
        "scope": "Linear velocities m/s then angular rad/s. Physical means before joint-recovery restoration; integrated means after restoration. Latest ring only, not a long-time integral.",
        "body_slots": {
            str(body): {
                "mean_bias_velocity": expected[:, body].mean(axis=0).tolist(),
                "mean_physical_velocity": physical[:, body].mean(axis=0).tolist(),
                "mean_integrated_velocity": total[:, body].mean(axis=0).tolist(),
                "max_bias_abs": abs(expected[:, body]).max(axis=0).tolist(),
            }
            for body in (1, 2)
        },
    }
    args.snapshot.with_suffix(".bias.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
