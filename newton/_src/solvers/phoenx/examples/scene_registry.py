# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decorator-based registry of loadable scenes for the phoenx solver.

Tests decorate a ``(device) -> Scene`` builder function with :func:`scene`;
:mod:`example_test_visualizer` reads the registry to populate a runtime
dropdown so any registered scene can be loaded into a live viewer without
writing a dedicated example.

Scene contract: the builder must return a :class:`Scene` carrying a
finalised :class:`PhoenXWorld`, a per-body half-extent array used for
rendering and picking, a suggested ``frame_dt``, and a substep count.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = [
    "Scene",
    "SceneBuilder",
    "discover_scenes",
    "registered_scenes",
    "scene",
]


@dataclass
class Scene:
    """State the visualizer needs to render + step a registered scene.

    Attributes:
        world: Finalised :class:`~newton._src.solvers.phoenx.solver_phoenx.PhoenXWorld`.
        body_half_extents: ``(num_bodies, 3)`` ``np.float32`` half-extents
            in body-local space. Bodies with any component ``<= 0``
            are treated as non-renderable / non-pickable (typical for
            the static world anchor at body 0).
        frame_dt: Suggested outer per-frame ``dt`` [s].
        substeps: ``world.step`` calls per rendered frame.
        description: Optional one-line summary for the UI.
    """

    world: object
    body_half_extents: np.ndarray
    frame_dt: float = 1.0 / 60.0
    substeps: int = 4
    description: str = ""


SceneBuilder = Callable[[object], Scene]


@dataclass
class _RegisteredScene:
    name: str
    builder: SceneBuilder
    source_module: str = ""
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, _RegisteredScene] = {}


def scene(
    name: str | None = None,
    *,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> Callable[[SceneBuilder], SceneBuilder]:
    """Register ``func`` as a loadable scene under ``name``.

    Raises:
        ValueError: If ``name`` is already registered (re-decoration
            of the same function is a no-op).
    """

    def deco(func: SceneBuilder) -> SceneBuilder:
        display = name if name is not None else func.__name__
        if display in _REGISTRY and _REGISTRY[display].builder is not func:
            raise ValueError(
                f"scene name {display!r} already registered by "
                f"{_REGISTRY[display].source_module!r}"
            )
        _REGISTRY[display] = _RegisteredScene(
            name=display,
            builder=func,
            source_module=getattr(func, "__module__", ""),
            description=description,
            tags=tuple(tags),
        )
        return func

    return deco


def registered_scenes() -> list[_RegisteredScene]:
    """Registered scenes, in insertion order."""
    return list(_REGISTRY.values())


def discover_scenes() -> list[_RegisteredScene]:
    """Import every ``test_*.py`` in the phoenx tests package so their
    ``@scene`` decorators run, then return the populated registry."""
    pkg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"
    )
    pkg_name = "newton._src.solvers.phoenx.tests"
    for mod_info in pkgutil.iter_modules([pkg_dir]):
        if not mod_info.name.startswith("test_"):
            continue
        if mod_info.name == "test_run_all":
            continue
        importlib.import_module(f"{pkg_name}.{mod_info.name}")
    return registered_scenes()
