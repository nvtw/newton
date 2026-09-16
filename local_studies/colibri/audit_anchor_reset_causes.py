"""Classify actual material-anchor resets within a captured collision interval."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/colibri_base_frame_totalnormal_frame330.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_anchor_reset_causes.json"))
    args = parser.parse_args()
    records = []
    with np.load(args.source) as archive:
        data = {
            name: archive[name]
            for name in (
                "warm_before.impulses",
                "warm_solved.impulses",
                "warm_before.headers",
                "warm_before.column_count",
                "warm_before.lambdas",
                "warm_solved.lambdas",
                "warm_solved.derived",
            )
        }
        before = data["warm_before.impulses"]
        after = data["warm_solved.impulses"]
        for slot in range(1, len(before)):
            if slot % 30 == 0:
                continue  # Contact identity can change at collision refresh.
            headers = data["warm_before.headers"][slot]
            indices = headers.view(np.int32)
            for column in range(int(data["warm_before.column_count"][slot, 0])):
                bodies = indices[1:3, column]
                if 0 not in bodies:
                    continue
                mu = float(headers[3, column])
                for point in range(indices[5, column], indices[5, column] + indices[6, column]):
                    old = before[slot, :, point]
                    new = after[slot, :, point]
                    capacity = mu * float(old[0])
                    ratio = float(np.linalg.norm(old[1:]) / capacity) if capacity > 0 else 0.0
                    changed = not np.array_equal(
                        data["warm_before.lambdas"][slot, 6:12, point],
                        data["warm_solved.lambdas"][slot, 6:12, point],
                    )
                    bias = float(data["warm_solved.derived"][slot, 3, point])
                    records.append(
                        {
                            "slot": slot,
                            "point": point,
                            "body": int(max(bodies)),
                            "loaded": bool(old[0] > 0),
                            "anchor_changed": changed,
                            "tangent_cleared": bool(np.all(new[1:] == 0)),
                            "previous_cone_ratio": ratio,
                            "positive_gap": bool(bias > 0),
                            "positive_gap_m": max(0.0, bias) / 3600.0,
                            "normal_impulse_Ns": float(old[0]),
                        }
                    )
    summaries = {}
    for body in sorted({r["body"] for r in records}):
        rows = [r for r in records if r["body"] == body and r["loaded"]]
        resets = [r for r in rows if r["anchor_changed"] and r["tangent_cleared"]]
        gap_only = [r for r in resets if r["positive_gap"] and r["previous_cone_ratio"] < 0.98]
        summaries[str(body)] = {
            "loaded_point_substeps": len(rows),
            "reset_point_substeps": len(resets),
            "positive_gap_below_98pct_resets": len(gap_only),
            "positive_gap_below_98pct_max_m": max((r["positive_gap_m"] for r in gap_only), default=0.0),
            "reset_previous_normal_impulse_sum_Ns": sum(r["normal_impulse_Ns"] for r in resets),
            "loaded_previous_normal_impulse_sum_Ns": sum(r["normal_impulse_Ns"] for r in rows),
        }
    report = {
        "source": str(args.source),
        "scope": "Captured preparation only. Positive gap follows sign of prepared normal bias; distance assumes 3600 Hz and no positive bias clamp. Counts repeat contact points across substeps. Previous load does not prove current load.",
        "by_body": summaries,
        "rows": records,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
