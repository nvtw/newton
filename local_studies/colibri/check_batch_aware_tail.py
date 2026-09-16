# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the full-scene validator with local batch-aware head/tail dispatch."""

import runpy
import sys

from local_studies.colibri import check_bilateral_pgs

_head_chunk = None
if "--head-launch-chunk" in sys.argv:
    _index = sys.argv.index("--head-launch-chunk")
    _head_chunk = int(sys.argv[_index + 1])
    del sys.argv[_index : _index + 2]
    if _head_chunk < 1:
        raise ValueError("Head launch chunk must be positive")
    from newton._src.solvers.phoenx import solver_phoenx

    solver_phoenx.NUM_INNER_WHILE_ITERATIONS = _head_chunk

_enabled = "--batch-aware-tail" in sys.argv
if _enabled:
    sys.argv.remove("--batch-aware-tail")
    from local_studies.colibri import batch_aware_tail
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    solver_phoenx_kernels._make_singleworld_persistent_kernel = batch_aware_tail._make_singleworld_persistent_kernel
    solver_phoenx_kernels._make_singleworld_fused_kernel = batch_aware_tail._make_singleworld_fused_kernel

if "--contact-chunk-size" in sys.argv:
    from local_studies.colibri import check_chunk_coloring  # noqa: F401

_original_example = check_bilateral_pgs.Example


class _RecordedExample(_original_example):
    def __init__(self, viewer, args):
        args.batch_aware_tail = _enabled
        args.head_launch_chunk = _head_chunk
        super().__init__(viewer, args)


if __name__ == "__main__":
    check_bilateral_pgs.Example = _RecordedExample
    runpy.run_module("local_studies.colibri.check_mu0", run_name="__main__")
