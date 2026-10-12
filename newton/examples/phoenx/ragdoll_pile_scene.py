# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-independent geometry for a six-box ragdoll stress scene.

The reference video specifies counts, but not dimensions, masses or joint
limits. These SI dimensions reconstruct its block figures; the five pivots
are free spherical joints. No assets or scene files are needed.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

DEFAULT_GRID = (40, 40, 20)
SPACING = (1.02, 0.38, 1.86)
SPAWN_CLEARANCE = 0.05
BODY_DENSITY = 200.0


@dataclass(frozen=True)
class BodySpec:
    name: str
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


BODIES = (
    BodySpec("torso", (0.0, 0.0, 1.10), (0.24, 0.13, 0.35)),
    BodySpec("head", (0.0, 0.0, 1.63), (0.17, 0.13, 0.17)),
    BodySpec("left_arm", (-0.37, 0.0, 1.10), (0.10, 0.12, 0.30)),
    BodySpec("right_arm", (0.37, 0.0, 1.10), (0.10, 0.12, 0.30)),
    BodySpec("left_leg", (-0.13, 0.0, 0.36), (0.10, 0.12, 0.36)),
    BodySpec("right_leg", (0.13, 0.0, 0.36), (0.10, 0.12, 0.36)),
)

# All pivots attach to body 0, the torso. Anchors are in the figure frame;
# each framework converts them to endpoint-local coordinates.
JOINTS = (
    (1, (0.0, 0.0, 1.455)),
    (2, (-0.27, 0.0, 1.40)),
    (3, (0.27, 0.0, 1.40)),
    (4, (-0.13, 0.0, 0.735)),
    (5, (0.13, 0.0, 0.735)),
)

PALETTE = (
    (0.91, 0.39, 0.08),
    (0.98, 0.68, 0.06),
    (0.39, 0.68, 0.13),
    (0.03, 0.54, 0.53),
    (0.24, 0.34, 0.78),
    (0.58, 0.31, 0.69),
)


def ragdoll_positions(grid=DEFAULT_GRID):
    """Yield a centered, non-overlapping block in stable layer-major order."""
    if len(grid) != 3 or any(size <= 0 for size in grid):
        raise ValueError("grid must contain three positive dimensions")
    nx, ny, nz = grid
    for z, y, x in product(range(nz), range(ny), range(nx)):
        yield (
            (x - 0.5 * (nx - 1)) * SPACING[0],
            (y - 0.5 * (ny - 1)) * SPACING[1],
            z * SPACING[2] + SPAWN_CLEARANCE,
        )


def scene_bounds(grid=DEFAULT_GRID):
    """Return the box union, including the initial clearance above the floor."""
    if len(grid) != 3 or any(size <= 0 for size in grid):
        raise ValueError("grid must contain three positive dimensions")
    centers = np.asarray([body.center for body in BODIES])
    extents = np.asarray([body.half_extents for body in BODIES])
    lower = (centers - extents).min(axis=0)
    upper = (centers + extents).max(axis=0)
    span = (np.asarray(grid) - 1) * np.asarray(SPACING)
    lower[:2] -= 0.5 * span[:2]
    upper[:2] += 0.5 * span[:2]
    lower[2] += SPAWN_CLEARANCE
    upper[2] += span[2] + SPAWN_CLEARANCE
    return lower, upper
