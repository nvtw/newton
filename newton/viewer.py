# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

# Import all viewer classes (they handle missing dependencies at instantiation time)
from ._src.viewer import (
    Layer,
    ViewerBase,
    ViewerFile,
    ViewerGL,
    ViewerNull,
    ViewerRerun,
    ViewerRTX,
    ViewerUSD,
    ViewerViser,
)

if TYPE_CHECKING:
    from ._src.viewer.viewer_optix import ViewerOptix

__all__ = [
    "Layer",
    "ViewerBase",
    "ViewerFile",
    "ViewerGL",
    "ViewerNull",
    "ViewerOptix",
    "ViewerRTX",
    "ViewerRerun",
    "ViewerUSD",
    "ViewerViser",
]


def __getattr__(name: str):
    # The OptiX backend initializes Warp on import, so load it only when requested.
    if name == "ViewerOptix":
        from ._src.viewer.viewer_optix import ViewerOptix  # noqa: PLC0415

        globals()[name] = ViewerOptix
        return ViewerOptix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
