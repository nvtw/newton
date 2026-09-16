# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local exact greedy-color specialization for rows with at most two endpoints."""

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData


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
            for endpoint in range(2):
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
        for endpoint in range(2):
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


if __name__ == "__main__":
    import runpy

    from newton._src.solvers.phoenx.mass_splitting import color_groups

    color_groups.color_rows = color_rows
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
