# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base interface for Kamino forward-dynamics solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from ..core.data import DataKamino
    from ..core.model import ModelKamino
    from ..dynamics.dual import DualProblem
    from ..geometry.contacts import ContactsKamino
    from ..kinematics.limits import LimitsKamino


class ForwardDynamicsSolver(ABC):
    """Common interface implemented by Kamino forward-dynamics solvers."""

    @property
    @abstractmethod
    def data(self):
        """Return solver-owned data arrays."""

    @abstractmethod
    def reset(self, problem: DualProblem | None = None, world_mask: wp.array | None = None):
        """Reset solver state and cached solution data."""

    @abstractmethod
    def coldstart(self):
        """Prepare the next solve without warm-start data."""

    @abstractmethod
    def warmstart(
        self,
        problem: DualProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """Prepare the next solve using cached or container state."""

    @abstractmethod
    def solve(self, problem: DualProblem):
        """Solve the supplied dual forward-dynamics problem."""
