# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Change only independent color-group grid size for a canonical scene audit."""

import argparse
import runpy
import sys

import warp as wp

from newton._src.solvers.phoenx.dispatch.color_groups import get_sweep_kernel

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--group-blocks", type=int, required=True)
options, remaining = parser.parse_known_args()
replacements = {
    get_sweep_kernel(phase, soft): get_sweep_kernel(phase, soft, options.group_blocks)
    for phase in ("prepare", "cached_prepare", "iterate", "relax")
    for soft in (False, True)
}
original_launch = wp.launch


def launch(kernel, *args, **kwargs):
    replacement = replacements.get(kernel)
    if replacement is not None:
        if args:
            args = ((options.group_blocks, 32), *args[1:])
        else:
            kwargs["dim"] = (options.group_blocks, 32)
        kernel = replacement
    return original_launch(kernel, *args, **kwargs)


wp.launch = launch
sys.argv = [sys.argv[0], *remaining]
try:
    runpy.run_module("local_studies.colibri.check_canonical_groups", run_name="__main__")
finally:
    wp.launch = original_launch
