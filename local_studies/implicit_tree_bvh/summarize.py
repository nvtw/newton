# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Print a compact summary of results.json."""

import json
import os

R = json.load(open(os.path.join(os.path.dirname(__file__), "results.json")))


def f(x):
    return f"{x:.3f}" if x < 10 else f"{x:.0f}"


for r in R:
    print(f"\n## {r['scene']} n={r['n']}")
    print("build:", {k: f(v) for k, v in r["build"].items()})
    print("refit:", {k: f(v) for k, v in r["refit"].items()})
    print("mem MB:", {k: round(v / 1e6, 2) for k, v in r["memory"].items()})
    print("capture:", r["capture"])
    for q, v in r["query"].items():
        print(q, {k: (f(x) if isinstance(x, float) else x) for k, x in v.items() if "mismatch" not in k})
