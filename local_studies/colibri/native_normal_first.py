# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Schedule all native normals before all original point friction rows.

Install as COLIBRI_STAGE_RUNNER under the corrected native runner. No point,
capacity, history, coefficient, drive equation, or application site is changed.
"""

import hashlib
import importlib.util
import inspect
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def split_sweeps(source):
    """Split one point sweep while preserving the original friction projection."""
    start = source.index("    for scheduled in range(begin, end):")
    header, loop = source[:start], source[start:]
    normal_start = loop.index("                    normal_delta = wp.dot(normal_impulse, normal)")
    update_end = loop.index("                response.contact_active[articulation]", normal_start)
    normal = loop[:normal_start] + "                    impulse = normal_impulse\n" + loop[update_end:]
    friction_start = loop.index("                    normal_impulse = -cc_get_normal_lambda")
    friction_end = loop.index("                    normal_delta = wp.dot(normal_impulse, normal)", friction_start)
    friction = loop[:friction_start] + "                    normal_impulse = wp.vec3f(0.0)\n" + loop[friction_end:]
    assert "contact_project_normal_velocity_update(" not in friction
    assert "contact_project_friction_metric" not in normal
    assert friction.count("contact_project_friction_metric") == loop.count("contact_project_friction_metric")
    return header + normal + "\n" + friction


def install():
    """Replace the actual owned callback, leaving dispatcher bias handling intact."""
    from newton._src.solvers.phoenx import solver_phoenx as dispatch
    from newton._src.solvers.phoenx.articulations import maximal_contact_gs as native

    original = dispatch.iterate_maximal_contact_runs_kernel
    assert original is native.iterate_maximal_contact_runs_kernel
    function = inspect.getsource(original.func)
    function = function[function.index("def iterate_maximal_contact_runs_kernel(") :]
    # Require the corrected total-load and explicit break-state implementation.
    assert "contact_project_friction_metric_with_break(" in function
    assert "normal_load += row_mass_coeff" not in function
    transformed = split_sweeps(function).replace(
        "def iterate_maximal_contact_runs_kernel(", "def iterate_normal_first_kernel("
    )
    full = Path(native.__file__).read_text() + "\n\n@wp.kernel(enable_backward=False)\n" + transformed
    path = Path(tempfile.mkdtemp(prefix="colibri_normal_first_")) / "maximal_contact_gs.py"
    path.write_text(full)
    name = "colibri_native_normal_first"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    dispatch.iterate_maximal_contact_runs_kernel = module.iterate_normal_first_kernel
    return {
        "generated_source": str(path),
        "sha256": hashlib.sha256(full.encode()).hexdigest(),
        "source_callback": inspect.getfile(original.func),
        "policy": __doc__,
    }


def main():
    """Run an unchanged fixture with the schedule-only callback replacement."""
    metadata = install()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.with_suffix(".normal_first.json").write_text(json.dumps(metadata, indent=2))
    print("ALL_ORIGINAL_POINTS_NORMAL_FIRST", json.dumps(metadata), flush=True)
    runpy.run_module(
        os.environ.get("COLIBRI_NORMAL_FIRST_RUNNER", "local_studies.colibri.check_public_analytic_gradient"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
