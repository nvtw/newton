# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim

__all__ = ["RigidBodySim"]


def __getattr__(name: str):
    if name != "RigidBodySim":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = importlib.import_module(".simulation", __name__).RigidBodySim
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
