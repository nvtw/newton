# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Win/loss table: best implicit variant vs best Warp structure per category (results.json)."""

import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results.json")
R = json.load(open(path))


def split(values):
    ours = {k: v for k, v in values.items() if k.startswith("implicit") and isinstance(v, float) and "levels" not in k}
    theirs = {
        k: v
        for k, v in values.items()
        if (k.startswith("bvh") or k.startswith("mesh")) and isinstance(v, float) and "mismatch" not in k
    }
    return ours, theirs


rows = []
for r in R:
    cats = {}
    cats["build"] = {k: v for k, v in r["build"].items() if k != "tri_bounds_kernel"}
    cats["refit"] = r["refit"]
    cats["memory"] = {k: float(v) / 1e6 for k, v in r["memory"].items()}
    cats.update(r["query"])
    for name, values in cats.items():
        ours, theirs = split(values)
        if not ours or not theirs:
            continue
        ko, vo = min(ours.items(), key=lambda kv: kv[1])
        kt, vt = min(theirs.items(), key=lambda kv: kv[1])
        rows.append((r["scene"], r["n"], name, ko, vo, kt, vt, vt / vo))

w = max(len(x[2]) for x in rows)
print(f"{'scene':8} {'n':>8} {'category':{w}}  {'implicit (best)':>28}  {'warp (best)':>24}  speedup")
for scene, n, name, ko, vo, kt, vt, ratio in rows:
    mark = "" if ratio >= 0.97 else "   <-- behind"
    print(f"{scene:8} {n:8d} {name:{w}}  {ko:>18} {vo:9.3f}  {kt:>14} {vt:9.3f}  {ratio:6.2f}x{mark}")
