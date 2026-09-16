"""CPU checks for final coupled friction-history ownership."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coupled_friction_history import final_break_flags


def main():
    """Exercise opening, sticking, sliding and resticking with the actual helper."""
    a = {"C": np.eye(3), "P": np.eye(3), "rhs": np.zeros(3), "vbar": np.zeros(3), "mu": np.array([0.5])}
    cases = [
        ("stick", [1, 0.1, 0], [0, 0, 0], 0),
        ("slide", [1, -0.5, 0], [0, 1, 0], 1),
        ("open_zero", [0, 0, 0], [1, 0, 0], 0),
        ("open_slipping", [0, 0, 0], [1, 1, 0], 1),
    ]
    records = []
    for name, lam, v, expected in cases:
        flags = final_break_flags(a, np.array(lam, dtype=float), np.array(v, dtype=float))
        stored = np.array([1.0], dtype=np.float32)
        stored[:] = flags
        assert stored[0] == expected, (name, stored, expected)
        records.append({"case": name, "expected": expected, "actual": float(stored[0])})
    Path("/tmp/colibri_coupled_friction_history_test.json").write_text(json.dumps(records, indent=2))
    print("PASS", records)


if __name__ == "__main__":
    main()
