# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-constraint access-mode metadata.

Each constraint module exports a module-level ``ACCESS_MODE`` of
type :class:`ConstraintAccessMode`. Integer values match
:mod:`mass_splitting.state` so a future mass-splitting integration
routes the same tag through ``tiny_rigid_state_set_access_mode``
without translation. Contract: see ``CONSTRAINT_ACCESS_MODE.md``.
"""

from __future__ import annotations

import enum

from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_VELOCITY_LEVEL,
)

__all__ = [
    "ConstraintAccessMode",
]


class ConstraintAccessMode(enum.IntEnum):
    """Velocity-level vs position-level constraint integration.

    * :attr:`VELOCITY_LEVEL` -- reads / writes ``velocity`` /
      ``angular_velocity`` only (sequential-impulse).
    * :attr:`POSITION_LEVEL` -- reads / writes ``position`` /
      ``orientation`` (XPBD); the iterate must call
      ``writeback_position_to_velocity`` for every touched entity
      before returning, so downstream consumers see consistent
      velocities. Mixing both regimes within one substep / graph
      color is safe because every kernel is internally homogeneous.
    """

    VELOCITY_LEVEL = ACCESS_MODE_VELOCITY_LEVEL
    POSITION_LEVEL = ACCESS_MODE_POSITION_LEVEL
