# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX unit tests require CUDA graph capture."""

from __future__ import annotations

from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture

require_cuda_graph_capture()
