# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX unit tests require CUDA graph capture."""

from __future__ import annotations

import unittest

import warp as wp


def _require_cuda_graph_capture() -> None:
    device = wp.get_preferred_device()
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise unittest.SkipTest(
            f"PhoenX tests require CUDA graph capture with Warp mempool enabled (device: {device.name!r})."
        )


_require_cuda_graph_capture()
