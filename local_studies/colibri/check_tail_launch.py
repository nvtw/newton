# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare fused-tail block sizes without changing constraint equations."""

import runpy
import sys

from local_studies.colibri import check_bilateral_pgs

_block_dim = 256
if "--tail-block-dim" in sys.argv:
    _index = sys.argv.index("--tail-block-dim")
    _block_dim = int(sys.argv[_index + 1])
    del sys.argv[_index : _index + 2]
if _block_dim not in (32, 64, 128, 256):
    raise ValueError("Tail block size must be 32, 64, 128 or 256")
_original_example = check_bilateral_pgs.Example


class _LaunchExample(_original_example):
    def __init__(self, viewer, args):
        args.tail_block_dim = _block_dim
        super().__init__(viewer, args)
        world = self.solver.world
        world._fuse_tail_block_dim = _block_dim
        # Every column accepted by the tail must have an executing lane.
        # Larger colors retain the existing persistent-head fallback.
        world._fuse_threshold = min(world._fuse_threshold, _block_dim)


if __name__ == "__main__":
    check_bilateral_pgs.Example = _LaunchExample
    runpy.run_module("local_studies.colibri.check_mu0", run_name="__main__")
