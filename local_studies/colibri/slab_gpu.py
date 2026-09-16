# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compatibility names for the maintained deterministic color-group builder."""

from newton._src.solvers.phoenx.mass_splitting.color_groups import allocate as _allocate
from newton._src.solvers.phoenx.mass_splitting.color_groups import build, color_rows
from newton._src.solvers.phoenx.mass_splitting.color_groups import emit_partition_pairs as emit_slab_pairs

__all__ = ["allocate", "build", "color_rows", "emit_slab_pairs"]


def allocate(capacity, num_bodies, device):
    """Keep the historical study dictionary key referring to the same array."""
    data = _allocate(capacity, num_bodies, device)
    data["row_slab"] = data["row_partition"]
    return data
