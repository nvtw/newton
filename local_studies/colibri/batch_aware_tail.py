# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the now-canonical batch-aware tail scheduling.

The old study flag no longer changes behavior; both factories now use the
validated batch-count handoff in production.
"""

from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _make_singleworld_fused_kernel,
    _make_singleworld_persistent_kernel,
)

__all__ = ["_make_singleworld_fused_kernel", "_make_singleworld_persistent_kernel"]
