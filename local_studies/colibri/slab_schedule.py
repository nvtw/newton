# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU reference for conflict-free colors grouped into shared-copy slabs.

Each slab runs its colors sequentially. Different slabs use distinct body
copies and may run concurrently. A body has one copy per slab that touches
it, even when several colors in that slab touch the body. Copies start at
the same velocity; all slabs finish before averaging and rebroadcasting.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Schedule:
    """Explicit immutable-topology reference arrays, indexed by row or body."""

    colors: list[list[int]]
    row_color: np.ndarray
    row_slab: np.ndarray
    body_slabs: list[list[int]]
    colors_per_slab: int

    def summary(self):
        """Return work and copy counts without claiming runtime performance."""
        counts = np.array([len(slabs) for slabs in self.body_slabs])
        return {
            "rows": len(self.row_color),
            "colors": len(self.colors),
            "slabs": (len(self.colors) + self.colors_per_slab - 1) // self.colors_per_slab,
            "slots": int(counts.sum()),
            "maximum_body_copies": int(counts.max(initial=0)),
            "color_widths": [len(rows) for rows in self.colors],
            "copy_counts": counts.tolist(),
        }


def build_schedule(endpoints, num_bodies, colors_per_slab=8, order=None):
    """Greedily color every row, then group consecutive colors into slabs.

    Negative endpoints denote prescribed/world bodies and require no mutable
    copy. ``order`` is an optional permutation, never a filter of physical rows.
    This is an ordering experiment, not the current solver's exact coloring.
    """
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if endpoints.ndim != 2 or colors_per_slab < 1:
        raise ValueError("Expected a row-by-endpoint array and positive slab width")
    if np.any(endpoints >= num_bodies):
        raise ValueError("Endpoint outside body range")
    order = list(range(len(endpoints))) if order is None else list(order)
    if sorted(order) != list(range(len(endpoints))):
        raise ValueError("Order must cover every row exactly once")
    occupied = []
    colors = []
    row_color = np.full(len(endpoints), -1, dtype=np.int64)
    for row in order:
        nodes = {int(node) for node in endpoints[row] if node >= 0}
        color = next((i for i, seen in enumerate(occupied) if not nodes.intersection(seen)), len(colors))
        if color == len(colors):
            occupied.append(set())
            colors.append([])
        occupied[color].update(nodes)
        colors[color].append(row)
        row_color[row] = color
    row_slab = row_color // colors_per_slab
    body_slabs = [set() for _ in range(num_bodies)]
    for row, nodes in enumerate(endpoints):
        for node in nodes:
            if node >= 0:
                body_slabs[node].add(int(row_slab[row]))
    result = Schedule(colors, row_color, row_slab, [sorted(s) for s in body_slabs], colors_per_slab)
    validate_schedule(result, endpoints)
    return result


def validate_schedule(schedule, endpoints):
    """Check coverage, concurrent exclusivity, and exact body-copy ownership."""
    assert sorted(row for color in schedule.colors for row in color) == list(range(len(endpoints)))
    expected = [set() for _ in schedule.body_slabs]
    for color, rows in enumerate(schedule.colors):
        seen = set()
        for row in rows:
            nodes = {int(node) for node in endpoints[row] if node >= 0}
            assert not seen.intersection(nodes), "One color shares a mutable body"
            seen.update(nodes)
            assert schedule.row_color[row] == color
            assert schedule.row_slab[row] == color // schedule.colors_per_slab
            for node in nodes:
                expected[node].add(int(schedule.row_slab[row]))
    assert [sorted(s) for s in expected] == schedule.body_slabs


def project_scalar_rows(schedule, endpoints, inverse_mass, velocity, sweeps=1):
    """Reference unit-J fixed-relative-velocity solves with physical averaging.

    This scalar model checks the copy algebra independently of geometry. Each
    copy's inverse mass is its body's physical inverse mass times its number
    of active slabs. There is no added damping, regularization, or mass cap.
    """
    inverse_mass = np.asarray(inverse_mass, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64).copy()
    counts = np.array([len(s) for s in schedule.body_slabs])
    for _ in range(sweeps):
        copies = {(body, slab): velocity[body] for body, slabs in enumerate(schedule.body_slabs) for slab in slabs}
        for rows in schedule.colors:
            for row in rows:
                a, b = endpoints[row]
                slab = int(schedule.row_slab[row])
                wa = inverse_mass[a] * counts[a] if a >= 0 else 0.0
                wb = inverse_mass[b] * counts[b] if b >= 0 else 0.0
                va = copies[a, slab] if a >= 0 else 0.0
                vb = copies[b, slab] if b >= 0 else 0.0
                impulse = (vb - va) / (wa + wb) if wa + wb else 0.0
                if a >= 0:
                    copies[a, slab] += wa * impulse
                if b >= 0:
                    copies[b, slab] -= wb * impulse
        for body, slabs in enumerate(schedule.body_slabs):
            if slabs:
                velocity[body] = sum(copies[body, slab] for slab in slabs) / len(slabs)
    return velocity
