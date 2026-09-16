# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Save reachable runtime arrays to locate the first integration difference."""

import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import check_bilateral_pgs as runner

_instances = []
_original = runner.Example


class _DumpExample(_original):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        _instances.append(self)


def snapshot(example):
    arrays = {}
    seen = set()

    def visit(value, path, depth):
        if id(value) in seen or depth > 6:
            return
        if isinstance(value, wp.array):
            if value.size < 2000000:
                arrays[path] = value.numpy()
            return
        if value is None or isinstance(value, (str, int, float, bool, type, np.ndarray)) or callable(value):
            return
        seen.add(id(value))
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, (tuple, list)):
            items = enumerate(value)
        else:
            items = vars(value).items() if hasattr(value, "__dict__") else ()
        for name, child in items:
            if str(name) in ("model", "device", "_device", "module", "_cls", "cls", "func", "_struct"):
                continue
            visit(child, f"{path}.{name}", depth + 1)

    for name, obj in (
        ("world", example.solver.world),
        ("direct", example.solver._direct_equality_system),
        ("contacts", example.contacts),
        ("pipeline", example.collision_pipeline),
    ):
        visit(obj, name, 0)
    return arrays


if __name__ == "__main__":
    runner.Example = _DumpExample
    runpy.run_module("local_studies.colibri.check_joint_accuracy", run_name="__main__")
    destination = Path(sys.argv[sys.argv.index("--output") + 1]).with_suffix(".internals.npz")
    data = snapshot(_instances[0])
    np.savez_compressed(destination, **data)
    print(f"Saved {len(data)} internal arrays to {destination}")
