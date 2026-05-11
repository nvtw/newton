# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os as _os

import warp as _wp

# Default parallel NVRTC compilation for any explicit ``wp.force_load`` /
# ``wp.load_module`` triggered while PhoenX is in use. ``0`` (Warp's default)
# means serial; we pick min(cpu_count, 8) unless the user already set a value.
if _wp.config.load_module_max_workers == 0:
    _wp.config.load_module_max_workers = min(_os.cpu_count() or 1, 8)

from newton._src.solvers.phoenx.solver import SolverPhoenX

__all__ = ["SolverPhoenX"]
