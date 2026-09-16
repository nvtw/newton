# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local exact-order cooperative forward substitution, scalar back substitution."""

import inspect
import textwrap
import types

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.constraints import bilateral_joint
from newton._src.solvers.phoenx.dispatch import color_groups


def make_candidate():
    """Broadcast forward rows ascending; retain original back-subtraction order."""
    current = bilateral_joint.get_iterate_bilateral_joint_block(True)
    if "for forward_row in range(count)" in inspect.getsource(current.func):
        return current
    helper = inspect.getsource(bilateral_joint._solve_bilateral_impulses.func)
    helper = helper.replace("def _solve_bilateral_impulses(", "def _backward_bilateral_impulses(")
    start = helper.index("    solution = Vec6d()")
    end = helper.index("    # Preserve ascending inner subtraction order", start)
    helper = helper[:start] + "    solution = rhs\n" + helper[end:]
    namespace = types.SimpleNamespace(**vars(bilateral_joint))
    backward = _compile_function(helper, namespace, "_backward_bilateral_impulses")
    namespace._backward_bilateral_impulses = backward
    namespace.cooperative = True
    source = textwrap.dedent(inspect.getsource(bilateral_joint.get_iterate_bilateral_joint_block(True).func))
    source = source.replace("def iterate(", "def iterate_cooperative_forward(")
    marker = "        rhs = Vec6d()\n        for row_lane in range(count):"
    replacement = (
        "        lower = data.lower[cid]\n"
        "        for forward_row in range(count):\n"
        "            solved = _shuffle_bilateral_rhs(rhs_value, forward_row)\n"
        "            if tile_lane > forward_row and tile_lane < count:\n"
        "                rhs_value -= lower[tile_lane, forward_row] * solved\n"
        "        if tile_lane < count:\n"
        "            diagonal = data.diagonal[cid]\n"
        "            rhs_value /= diagonal[tile_lane]\n"
        "        rhs = Vec6d()\n"
        "        for row_lane in range(count):"
    )
    assert marker in source
    source = source.replace(marker, replacement, 1)
    source = source.replace("_solve_bilateral_impulses(", "_backward_bilateral_impulses(")
    return _compile_function(source, namespace, "iterate_cooperative_forward")


def install():
    """Replace only the private cooperative factory in this process."""
    original = color_groups.get_iterate_bilateral_joint_block
    candidate = make_candidate()

    def choose(cooperative=False):
        return candidate if cooperative else original(False)

    color_groups.get_iterate_bilateral_joint_block = choose
    color_groups.get_sweep_kernel.cache_clear()

    def restore():
        color_groups.get_iterate_bilateral_joint_block = original
        color_groups.get_sweep_kernel.cache_clear()

    return restore


if __name__ == "__main__":
    import runpy
    import sys

    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
