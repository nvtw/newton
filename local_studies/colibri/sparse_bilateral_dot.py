# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local finite-state sparse FP64 wrench dot-product experiment."""

import inspect
import runpy
import sys

import warp as wp

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx import solver_phoenx_kernels
from newton._src.solvers.phoenx.constraints import bilateral_joint as module


@wp.func
def _dot_double_nonzero(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.float64:
    value = wp.float64(0.0)
    for component in range(6):
        if a[component] != wp.float32(0.0):
            value += wp.float64(a[component]) * wp.float64(b[component])
    return value


def install():
    module._dot_double_nonzero = _dot_double_nonzero
    source = inspect.getsource(module.iterate_bilateral_joint_block.func)
    source = source.replace("def iterate_bilateral_joint_block(", "def iterate_sparse_dot_bilateral_joint_block(")
    source = source.replace("_dot_double(", "_dot_double_nonzero(")
    solver_phoenx_kernels.iterate_bilateral_joint_block = _compile_function(
        source, module, "iterate_sparse_dot_bilateral_joint_block"
    )


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
