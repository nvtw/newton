# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deterministic color groups for small rigid mass-splitting graphs.

Rows sharing an endpoint receive different colors. Consecutive colors share
one partition, so a body's mass copy can serve several sequential colors.
The single-thread greedy builder targets small mechanisms; it is not a
replacement for the parallel coloring path used by large contact scenes.
"""

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import InteractionGraphScratch, emit_pair


def _make_color_rows(max_endpoints):
    """Specialize endpoint scans while preserving the generic greedy schedule."""

    @wp.kernel(enable_backward=False)
    def color_rows(
        elements: wp.array[ElementInteractionData],
        active: wp.array[wp.int32],
        width: wp.int32,
        masks: wp.array2d[wp.uint32],
        row_color: wp.array[wp.int32],
        row_partition: wp.array[wp.int32],
        counts: wp.array[wp.int32],
        starts: wp.array[wp.int32],
        cursors: wp.array[wp.int32],
        ids: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
    ):
        # One deterministic builder thread; 32-color masks avoid an O(rows*colors)
        # scan. This is graph-capturable and preserves CPU first-fit exactly.
        colors = wp.int32(0)
        for row in range(active[0]):
            chosen = wp.int32(-1)
            block = wp.int32(0)
            while chosen < 0:
                occupied = wp.uint32(0)
                for endpoint in range(max_endpoints):
                    body = elements[row].bodies[endpoint]
                    if body >= 0:
                        occupied = occupied | masks[body, block]
                if occupied != wp.uint32(0xFFFFFFFF):
                    available = ~occupied
                    bit = wp.int32(0)
                    if (available & wp.uint32(0xFFFF)) == wp.uint32(0):
                        available = available >> wp.uint32(16)
                        bit += 16
                    if (available & wp.uint32(0xFF)) == wp.uint32(0):
                        available = available >> wp.uint32(8)
                        bit += 8
                    if (available & wp.uint32(0xF)) == wp.uint32(0):
                        available = available >> wp.uint32(4)
                        bit += 4
                    if (available & wp.uint32(0x3)) == wp.uint32(0):
                        available = available >> wp.uint32(2)
                        bit += 2
                    if (available & wp.uint32(0x1)) == wp.uint32(0):
                        bit += 1
                    chosen = block * 32 + bit
                else:
                    block += 1
            for endpoint in range(max_endpoints):
                body = elements[row].bodies[endpoint]
                if body >= 0:
                    masks[body, block] = masks[body, block] | (wp.uint32(1) << wp.uint32(chosen % 32))
            row_color[row] = chosen
            row_partition[row] = chosen / width
            counts[chosen] += 1
            colors = wp.max(colors, chosen + 1)
        num_colors[0] = colors
        starts[0] = 0
        for color in range(colors):
            starts[color + 1] = starts[color] + counts[color]
            cursors[color] = starts[color]
        for row in range(active[0]):
            color = row_color[row]
            index = cursors[color]
            ids[index] = row
            cursors[color] = index + 1

    return color_rows


color_rows = _make_color_rows(8)
_color_rigid_rows = _make_color_rows(2)


@wp.kernel(enable_backward=False)
def emit_partition_pairs(
    elements: wp.array[ElementInteractionData],
    active: wp.array[wp.int32],
    row_partition: wp.array[wp.int32],
    scratch: InteractionGraphScratch,
):
    row = wp.tid()
    if row < active[0]:
        for endpoint in range(8):
            body = elements[row].bodies[endpoint]
            if body >= 0:
                emit_pair(scratch, body, row_partition[row])


def allocate(capacity, num_bodies, device):
    """Allocate topology buffers before CUDA capture."""
    if capacity < 1 or num_bodies < 1:
        raise ValueError("Color groups require positive row and node capacities")
    return {
        "masks": wp.zeros((num_bodies, (capacity + 31) // 32), dtype=wp.uint32, device=device),
        **{
            name: wp.zeros(capacity + 1, dtype=wp.int32, device=device)
            for name in ("row_color", "row_partition", "counts", "starts", "cursors", "ids")
        },
        "num_colors": wp.zeros(1, dtype=wp.int32, device=device),
    }


def build(data, elements, active, width, device, *, rigid_only=False):
    """Rebuild active row coverage without host reads or allocations."""
    if width < 1:
        raise ValueError("A color group must contain at least one color")
    data["masks"].zero_()
    data["counts"].zero_()
    wp.launch(
        _color_rigid_rows if rigid_only else color_rows,
        1,
        [
            elements,
            active,
            width,
            data["masks"],
            data["row_color"],
            data["row_partition"],
            data["counts"],
            data["starts"],
            data["cursors"],
            data["ids"],
            data["num_colors"],
        ],
        device=device,
    )
