# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local FP64 reciprocal-diagonal experiment; no production changes.

Only this process stores inverse pivots in the existing diagonal buffer.
Both its producer and consumer are replaced together. A production change
would give the buffer an explicit inverse-diagonal name.
"""

import inspect
import runpy
import sys

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx import solver_phoenx_kernels
from newton._src.solvers.phoenx.articulations import block_joint_system
from newton._src.solvers.phoenx.constraints import bilateral_joint as module


def install():
    prepare = inspect.getsource(module.prepare_bilateral_joint_blocks.func)
    original = "    data.diagonal[cid] = diagonal"
    replacement = """    inverse_diagonal = Vec6d()
    if valid:
        for i in range(6):
            if i < count:
                inverse_diagonal[i] = wp.float64(1.0) / diagonal[i]
    data.diagonal[cid] = inverse_diagonal"""
    assert prepare.count(original) == 1
    prepare = prepare.replace(original, replacement).replace(
        "def prepare_bilateral_joint_blocks(", "def prepare_reciprocal_bilateral_joint_blocks("
    )
    iterate = inspect.getsource(module.iterate_bilateral_joint_block.func)
    original = "solution[i] /= diagonal[i]"
    assert iterate.count(original) == 1
    iterate = iterate.replace(original, "solution[i] *= diagonal[i]").replace(
        "def iterate_bilateral_joint_block(", "def iterate_reciprocal_bilateral_joint_block("
    )
    block_joint_system.prepare_bilateral_joint_blocks = _compile_function(
        prepare, module, "prepare_reciprocal_bilateral_joint_blocks"
    )
    solver_phoenx_kernels.iterate_bilateral_joint_block = _compile_function(
        iterate, module, "iterate_reciprocal_bilateral_joint_block"
    )


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
