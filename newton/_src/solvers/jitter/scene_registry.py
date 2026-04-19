# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decorator-based registry of "loadable" scenes for the jitter solver.

Tests (and one-off demos) decorate a zero-arg-or-``device``-arg builder
function with :func:`scene`; the decorator captures the function plus
some lightweight metadata (display name, body half-extents, optional
description) and inserts it into a process-global registry. The
:mod:`example_test_visualizer` reads that registry to populate a
runtime dropdown so any registered scene can be loaded into a live
viewer without having to write a dedicated example.

Why a registry instead of "import all builders by hand"?
--------------------------------------------------------
Every jitter test already builds its own scene. Manually maintaining a
parallel list in the visualizer would drift the moment a new test is
added. Instead each test imports :func:`scene`, decorates its
``build_*`` function, and the visualizer auto-discovers it via
:func:`discover_scenes` (which just imports every sibling ``test_*.py``
so the decorators run as a side-effect).

Scene contract
--------------
A registered builder must return a :class:`Scene` carrying:

* ``world`` -- a finalized :class:`World`.
* ``body_half_extents`` -- ``(num_bodies, 3)`` ``numpy.float32`` array
  giving each body's local-frame half extents. The world anchor (body
  0) should typically have ``(0, 0, 0)`` to mark it non-pickable; the
  visualizer renders bodies with a non-zero half-extent as boxes and
  uses the same array for picking.
* ``frame_dt`` -- preferred outer frame ``dt`` (the visualizer drives
  the substep loop itself; this is the wall-clock-ish per-frame step).
* ``substeps`` -- substeps per outer frame; multiplied into the
  per-substep ``dt`` passed to ``world.step``.

The builder receives the active Warp device so containers can be
allocated on the right GPU. It must *not* run any simulation steps; the
visualizer is responsible for the time loop.
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
    """Container holding everything the visualizer needs to render and
    step a registered scene.

    Attributes:
        world: The finalized :class:`~newton._src.solvers.jitter.solver_jitter.World`.
        body_half_extents: ``(num_bodies, 3)`` ``np.float32`` array of
            per-body half-extents in body-local space. Bodies with any
            component ``<= 0`` are treated as non-renderable /
            non-pickable (typical for the static world anchor at body
            index 0).
        frame_dt: Suggested outer per-frame ``dt`` [s]. The visualizer
            divides this by :attr:`substeps` and steps the world that
            many times per rendered frame.
        substeps: Number of ``world.step`` calls per rendered frame.
        description: Optional one-line summary shown in the UI.
    """

    world: object
    body_half_extents: np.ndarray
    frame_dt: float = 1.0 / 60.0
    substeps: int = 4
    description: str = ""


# Type alias for the builder callable -- one positional ``device`` arg
# (a Warp device-like) returning a :class:`Scene`. Module-level so
# users importing the registry can spell their own type hints with it.
SceneBuilder = Callable[[object], Scene]


@dataclass
class _RegisteredScene:
    """Internal record stored in the global registry.

    ``name`` is what the visualizer dropdown displays; ``builder`` is
    the decorated function. Kept as its own dataclass (rather than a
    plain tuple) so additions like "category" / "tags" don't break
    callers reading the registry.
    """

    name: str
    builder: SceneBuilder
    source_module: str = ""
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# Module-level registry. Keyed by display name to make duplicate
# registrations visible at decoration time (we raise on collision).
_REGISTRY: dict[str, _RegisteredScene] = {}


def scene(
    name: str | None = None,
    *,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> Callable[[SceneBuilder], SceneBuilder]:
    """Register ``func`` as a loadable scene under ``name``.

    Decorate a zero-arg (or single ``device``-arg) builder that returns
    a :class:`Scene`::

        @scene("Hanging chain (10 cubes)")
        def build_equilibrium_chain_scene(device) -> Scene:
            world = ...
            return Scene(world=world, body_half_extents=he, ...)

    Args:
        name: Display name shown in the visualizer dropdown. Defaults
            to the decorated function's ``__name__``.
        description: Optional one-line tooltip / sub-heading for the UI.
        tags: Optional free-form labels (currently unused; reserved for
            future filtering).

    Returns:
        The original function, unchanged. Registration is the side
        effect.

    Raises:
        ValueError: If ``name`` is already registered. Re-decoration
            (e.g. from a re-imported test module) is *not* silent --
            we want naming collisions to fail loudly.
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
    """Return the registered scenes in insertion order.

    Insertion order is dict-iteration order (Python >=3.7 contract);
    that's the order the visualizer dropdown displays them in, so
    putting ``@scene(...)`` near the top of a test module lifts it
    higher in the menu.
    """
    return list(_REGISTRY.values())


def discover_scenes() -> list[_RegisteredScene]:
    """Import every ``test_*.py`` in this package so their ``@scene``
    decorators run, then return the populated registry.

    Idempotent -- importing an already-imported test module is a no-op
    in the standard import system, and :func:`scene` rejects duplicate
    names so a module that's somehow loaded twice from the same source
    doesn't silently double-register.

    Skips :mod:`test_run_all` because that module re-imports the rest
    of the test files itself; running its discovery would duplicate
    work and risk re-entering the visualizer when it's launched from
    inside a test runner.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_name = __name__.rsplit(".", 1)[0]

    # ``pkgutil.iter_modules`` is the safe way to walk a package
    # directory without parsing filenames ourselves -- it respects
    # namespace packages, ``__init__`` presence, etc.
    for mod_info in pkgutil.iter_modules([pkg_dir]):
        if not mod_info.name.startswith("test_"):
            continue
        if mod_info.name == "test_run_all":
            continue
        importlib.import_module(f"{pkg_name}.{mod_info.name}")

    return registered_scenes()
