"""Classify retained bounded-solver histories without rerunning any solve."""

import json
from pathlib import Path

import numpy as np


def main():
    records = json.loads(Path("/tmp/high_mass_consistent960_replay.json").read_text())
    failed = {k: v for k, v in records.items() if not v["accepted"]}
    predicates = {
        "rank_abort": lambda a: any("rejected_response_null" in h for h in a["history"]),
        "line_search_abort": lambda a: any(h.get("accepted") is False for h in a["history"]),
        "converged_wrong_face": lambda a: a["kind"] == "face" and a["history"][-1]["face_equation_residual"] < 1e-8,
        "face_step_cap_with_descent": lambda a: (
            a["kind"] == "face"
            and len(a["history"]) == 8
            and a["history"][-1].get("accepted") is True
            and a["history"][-1]["face_equation_residual"] >= 1e-8
        ),
        "cycle_skip": lambda a: a.get("skipped_revisits", 0) > 0,
    }
    report = {"total_failed": len(failed), "categories_overlap": True, "categories": {}}
    for name, predicate in predicates.items():
        report["categories"][name] = {
            "cases": sum(any(predicate(a) for a in v["attempts"]) for v in failed.values()),
            "attempts": sum(predicate(a) for v in failed.values() for a in v["attempts"]),
        }
    report["representatives"] = {}
    for case in ("300_2", "362_5", "263_2"):
        d = np.load("/tmp/high_mass_consistent960_" + case + ".npz")
        value = d["solution"].reshape(-1, 3)
        gradient = (d["A"] @ d["solution"] + d["rhs"]).reshape(-1, 3)
        report["representatives"][case] = {
            "returned_residual": failed[case]["residual"],
            "best_attempt_residual": min(a["residual"] for a in failed[case]["attempts"]),
            "normal_impulse_Ns": value[:, 0].tolist(),
            "normal_velocity_m_s": gradient[:, 0].tolist(),
            "slip_speed_m_s": np.linalg.norm(gradient[:, 1:], axis=1).tolist(),
            "disk_margin_Ns": (0.5 * value[:, 0] - np.linalg.norm(value[:, 1:], axis=1)).tolist(),
            "attempts": failed[case]["attempts"],
        }
    Path("/tmp/high_mass_failure_diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "representatives"}, indent=2))


if __name__ == "__main__":
    main()
