"""Summarize the local collective relaxation reference without GPU access."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    """Report sample-window quality and all independent acceptance diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prefix = args.prefix + "live_"
    frames = json.loads(Path(prefix + "frames.json").read_text())
    records = json.loads(Path(prefix + "summary.json").read_text())
    phases = json.loads(Path(prefix + "phases.json").read_text())

    def window(rows):
        return {
            "samples": len(rows),
            "top_rms_speed_m_s": float(np.sqrt(np.mean([r["top_speed_m_s"] ** 2 for r in rows]))),
            "bottom_rms_speed_m_s": float(np.sqrt(np.mean([r["bottom_speed_m_s"] ** 2 for r in rows]))),
            "top_mean_load_relative_error": abs(float(np.mean([r["top_load_N"] for r in rows])) / 3924.0 - 1),
            "plane_mean_load_relative_error": abs(float(np.mean([r["plane_load_N"] for r in rows])) / 3933.81 - 1),
            "min_center_separation_m": min(r["separation_m"] for r in rows),
            "max_top_speed_m_s": max(r["top_speed_m_s"] for r in rows),
        }

    changes = [r["after"]["energy_J"] - r["before"]["energy_J"] for r in phases if r["phase"] == "relax"]
    report = {
        "scope": "CPU oracle; existing biased solves/integration unchanged; not a performance result",
        "full_window": window(frames),
        "last_half_window": window(frames[len(frames) // 2 :]),
        "accepted_relaxations": sum(r["physical_kkt_accepted"] for r in records),
        "attempted_relaxations": len(records),
        "max_normal_kkt_m_s": max(r["normal_natural_complementarity_m_s"] for r in records),
        "max_friction_kkt_m_s": max(r["max_friction_kkt_residual_m_s"] for r in records),
        "max_contact_work_J": max(r["physical_contact_work_J"] for r in records),
        "max_actual_relax_energy_change_J": max(changes),
        "sum_actual_relax_energy_change_J": sum(changes),
        "max_linear_reaction_balance": max(r["conservation"]["linear_reaction_balance"] for r in records),
        "max_angular_reaction_balance": max(r["conservation"]["angular_reaction_balance"] for r in records),
        "final_frame": frames[-1],
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
