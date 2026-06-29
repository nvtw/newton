# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Uncoloured scalar-row Jacobi implementation for PhoenX."""

from .dispatcher import SimplePhoenXDispatcher
from .rows import ScalarRowContainer, scalar_row_container_zeros

__all__ = ["ScalarRowContainer", "SimplePhoenXDispatcher", "scalar_row_container_zeros"]
