"""Classify solve-generated break transitions without changing their semantics."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    x = np.load(args.snapshot)["history"]
    records = []
    transitions = []
    previous = None
    for event in x:
        phase = int(event[0, 92])
        count = int(event[0, 93])
        sequence = int(event[0, 94])
        headers = event[:, 46:78].T.copy()
        integer = headers.view(np.int32)
        contacts = {}
        for col in range(headers.shape[1]):
            first, n = integer[5:7, col]
            if n <= 0 or first < 0 or first + n > count:
                continue
            i, j = integer[1:3, col]
            for k in range(first, first + n):
                a = event[k]
                normal, t1 = a[:3], a[3:6]
                t2 = np.cross(normal, t1)
                vi = event[i, 85:88] + np.cross(event[i, 88:91], a[36:39])
                vj = event[j, 85:88] + np.cross(event[j, 88:91], a[39:42])
                vt = np.array([(vj - vi) @ t1, (vj - vi) @ t2])
                radius = headers[3, col] * a[43]
                contacts[k] = {
                    "point": k,
                    "body_slots": [int(i), int(j)],
                    "broken": bool(a[12]),
                    "reference": a[13:27].tolist(),
                    "normal_impulse_Ns": float(a[43]),
                    "cone_fraction": float(np.linalg.norm(a[44:46]) / radius) if radius > 0 else None,
                    "vn_m_s": float((vj - vi) @ normal),
                    "vt_m_s": vt.tolist(),
                    "bias_m_s": float(a[30]),
                    "tangent_bias_m_s": a[31:33].tolist(),
                    "biased_tangent_gradient_m_s": (vt + a[31:33]).tolist(),
                }
        assert len(contacts) == count, (sequence, len(contacts), count)
        record = {"event": sequence, "phase": phase, "contacts": list(contacts.values())}
        records.append(record)
        if previous is not None:
            before = {r["point"]: r for r in previous["contacts"]}
            for k, r in contacts.items():
                if k not in before:
                    continue
                b = before[k]
                same = b["reference"] == r["reference"] and b["body_slots"] == r["body_slots"]
                if same and b["broken"] != r["broken"]:
                    transitions.append(
                        {
                            "event": sequence,
                            "phase": phase,
                            "previous_phase": previous["phase"],
                            "point": k,
                            "before": b["broken"],
                            "after": r["broken"],
                            "normal_impulse_Ns": r["normal_impulse_Ns"],
                            "cone_fraction": r["cone_fraction"],
                        }
                    )
        previous = record
    report = {
        "scope": "Last device-ring events; actual phase state, same-reference bit transitions only. A final zero does not establish absence of earlier sliding.",
        "event_count": len(records),
        "phase_counts": {str(p): sum(r["phase"] == p for r in records) for p in (10, 11, 20, 21, 30, 31)},
        "transitions": transitions,
        "relax_clears": sum(t["phase"] == 21 and t["before"] and not t["after"] for t in transitions),
        "events": records,
    }
    args.snapshot.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k not in ("events", "transitions")})


if __name__ == "__main__":
    main()
