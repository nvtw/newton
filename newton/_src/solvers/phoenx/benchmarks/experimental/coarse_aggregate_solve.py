# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the promoted aggregate coarse solver."""

from newton._src.solvers.phoenx.articulations.coarse_aggregate import (
    CoarseAggregateSolver,
    parent_aggregate_mapping,
)

__all__ = ["CoarseAggregateSolver", "parent_aggregate_mapping"]
