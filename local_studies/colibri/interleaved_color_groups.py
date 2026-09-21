# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local interleaved color partitions; intentionally changes finite-iteration order."""

import inspect
import runpy
import sys
import textwrap

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx import solver_phoenx
from newton._src.solvers.phoenx.dispatch import color_groups as dispatch
from newton._src.solvers.phoenx.mass_splitting import color_groups as topology


def install():
    source = textwrap.dedent(inspect.getsource(topology._color_rigid_rows.func))
    source = source.replace("def color_rows(", "def interleaved_color_rows(")
    source = source.replace("range(max_endpoints)", "range(2)")
    source = source.replace("        row_partition[row] = chosen / width\n", "")
    source = source.replace(
        "        color = row_color[row]",
        "        color = row_color[row]\n        row_partition[row] = color % ((colors + width - 1) / width)",
    )
    topology._color_rigid_rows = _compile_function(source, topology, "interleaved_color_rows")
    source = inspect.getsource(dispatch.get_sweep_kernel)
    source = source.replace("def get_sweep_kernel(", "def get_interleaved_sweep_kernel(")
    source = source.replace(
        "color = slab * slab_width + local_color",
        "color = slab + local_color * ((num_colors[0] + slab_width - 1) / slab_width)",
    )
    solver_phoenx.get_color_group_sweep_kernel = _compile_function(source, dispatch, "get_interleaved_sweep_kernel")


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
