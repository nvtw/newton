# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level state-update infrastructure.

See :doc:`README.md` (in this directory) for the design overview.

Public entry point is :class:`PositionPass`; the two underlying
``@wp.kernel`` primitives are also re-exported for callers that want
to drive the snapshot / sync without going through the orchestrator
(e.g. graph-capture-friendly substep loops that already have their
own scratch buffers).
"""

from __future__ import annotations

from newton._src.solvers.phoenx.position_level.position_pass import PositionPass
from newton._src.solvers.phoenx.position_level.snapshot import (
    snapshot_pose_kernel,
    sync_position_to_velocity_kernel,
)

__all__ = [
    "PositionPass",
    "snapshot_pose_kernel",
    "sync_position_to_velocity_kernel",
]
