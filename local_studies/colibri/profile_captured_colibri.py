# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Capture a short warmed CUDA graph interval with Nsight Systems."""

import ctypes
import runpy
import sys

import warp as wp

from local_studies.colibri import check_bilateral_pgs

_first = 31
_count = 2
for _flag in ("--profile-first-frame", "--profile-frame-count"):
    if _flag in sys.argv:
        _index = sys.argv.index(_flag)
        _value = int(sys.argv[_index + 1])
        del sys.argv[_index : _index + 2]
        if _flag == "--profile-first-frame":
            _first = _value
        else:
            _count = _value
if _first < 1 or _count < 1:
    raise ValueError("Profile start and frame count must be positive")
_driver = ctypes.CDLL("libcuda.so.1")
_driver.cuProfilerStart.restype = ctypes.c_int
_driver.cuProfilerStop.restype = ctypes.c_int
_original = check_bilateral_pgs.Example


class _ProfiledExample(_original):
    def __init__(self, viewer, args):
        args.profile_first_frame = _first
        args.profile_frame_count = _count
        super().__init__(viewer, args)
        self._profile_frame = 0

    def step(self):
        self._profile_frame += 1
        if self._profile_frame == _first:
            wp.synchronize_device(self.model.device)
            result = _driver.cuProfilerStart()
            if result:
                raise RuntimeError(f"cuProfilerStart failed: {result}")
        super().step()
        if self._profile_frame == _first + _count - 1:
            wp.synchronize_device(self.model.device)
            result = _driver.cuProfilerStop()
            if result:
                raise RuntimeError(f"cuProfilerStop failed: {result}")


if __name__ == "__main__":
    check_bilateral_pgs.Example = _ProfiledExample
    runpy.run_module("local_studies.colibri.check_speculative_chunks", run_name="__main__")
