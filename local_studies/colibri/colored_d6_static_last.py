# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Two-body parity control: move joint-only last color before static contacts.

Retain scalar D6 rows and all original force equations. This is deliberately
restricted to the verified single-joint fixture, not a general scheduler.
"""

import hashlib
import importlib.abc
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

NAME = "newton._src.solvers.phoenx.solver_phoenx_kernels"
REPLACEMENT = """@wp.func
def _color_for_step(
    step_idx: wp.int32,
    n_colors: wp.int32,
    direction: wp.int32,
    max_colored_partitions: wp.int32,
) -> wp.int32:
    # Bounded experiment: original last color is the sole physical joint.
    wp.expect_eq(direction, wp.int32(0))
    wp.expect_eq(max_colored_partitions, wp.int32(-1))
    if step_idx == wp.int32(0):
        return n_colors - wp.int32(1)
    return step_idx - wp.int32(1)


"""


class Finder(importlib.abc.MetaPathFinder):
    """Load the isolated schedule without modifying canonical kernels."""

    def __init__(self):
        self.source = Path(__file__).resolve().parents[2] / "newton/_src/solvers/phoenx/solver_phoenx_kernels.py"
        self.original = self.source.read_text()
        start = self.original.index("@wp.func\ndef _color_for_step(")
        end = self.original.index("@wp.func\ndef _singleworld_color_range(", start)
        self.modified = self.original[:start] + REPLACEMENT + self.original[end:]
        self.path = Path(tempfile.mkdtemp(prefix="colibri_static_last_")) / "solver_phoenx_kernels.py"
        self.path.write_text(self.modified)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == NAME:
            return importlib.util.spec_from_file_location(fullname, self.path)
        return None


def finish(solver, output):
    """Verify the bounded color ownership and preserve actual provenance."""
    import numpy as np

    world = solver.world
    partitioner = world._partitioner
    count = int(partitioner.num_colors.numpy()[0])
    starts = partitioner.color_starts.numpy()[: count + 1]
    ids = partitioner.element_ids_by_color.numpy()[: starts[-1]]
    assert world.num_joints == 1 and not world.mass_splitting_enabled
    assert not world._colored_contact_headers and not world._colored_contact_rows
    assert int(partitioner.sweep_direction.numpy()[0]) == 0
    assert np.array_equal(ids[starts[-2] : starts[-1]], [0]), (starts, ids)
    assert np.all(ids[: starts[-2]] >= 1)
    from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_BODY1, _OFF_BODY2

    headers = world._contact_cols.data.numpy().view(np.int32)
    columns = ids[: starts[-2]] - world.num_joints
    endpoints = headers[[int(_OFF_BODY1), int(_OFF_BODY2)]][:, columns]
    inverse_mass = world.bodies.inverse_mass.numpy()
    assert np.all(np.any(inverse_mass[endpoints] == 0.0, axis=0)), endpoints
    result = {
        "scope": __doc__,
        "num_colors": count,
        "original_color_starts": starts.tolist(),
        "original_color_ids": ids.tolist(),
        "visited_color_order": [count - 1, *range(count - 1)],
        "static_contact_endpoints": endpoints.tolist(),
        "mass_splitting": False,
        "actual_kernel_module": sys.modules[NAME].__file__,
    }
    Path(output).with_suffix(".static_last.json").write_text(json.dumps(result, indent=2))


def main():
    """Run corrected scalar D6 with the isolated color permutation."""
    assert sys.argv[sys.argv.index("--body-count") + 1] == "2"
    assert NAME not in sys.modules
    finder = Finder()
    sys.meta_path.insert(0, finder)
    from .colored_d6_alternative import transformed_source
    from .colored_d6_scalar import install

    install()
    path, source = transformed_source()
    marker = '        print("COLORED_BLOCK_PGS_OWNERSHIP_GATE_PASS", flush=True)'
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        """        from local_studies.colibri.colored_d6_static_last import finish
        phase_audits.append(lambda: finish(solver, output))
"""
        + marker,
    )
    print("STATIC_LAST_SOURCE_SHA256", hashlib.sha256(finder.modified.encode()).hexdigest(), flush=True)
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    assert Path(sys.modules[NAME].__file__) == finder.path
    assert finder.source.read_text() == finder.original


if __name__ == "__main__":
    main()
