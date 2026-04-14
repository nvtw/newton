# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Store impulses for next-frame warm starting."""

from __future__ import annotations

import warp as wp


@wp.kernel
def store_impulses_2d(
    c_normal_impulse: wp.array2d(dtype=float),
    c_friction1_impulse: wp.array2d(dtype=float),
    c_friction2_impulse: wp.array2d(dtype=float),
    contact_count: wp.array[wp.int32],
    # Outputs (prev-frame buffers, stored in raw/sort order)
    prev_ni: wp.array2d(dtype=float),
    prev_fi1: wp.array2d(dtype=float),
    prev_fi2: wp.array2d(dtype=float),
    prev_count: wp.array[wp.int32],
):
    """Copy current solver impulses to previous-frame buffers.

    Launched with ``dim = (num_worlds, max_contacts_per_world)``.

    The impulses are in color order in the solver arrays.  For warm
    starting, we store them by raw contact index so they align with
    the sorted contact order that ``ContactMatcher`` produces.

    For now, we store directly from color order — the warm start
    loader will handle the mapping.  A future optimization can add
    a scatter via ``color_to_raw``.
    """
    wid, ci = wp.tid()
    nc = contact_count[wid]
    if ci >= nc:
        return

    prev_ni[wid, ci] = c_normal_impulse[wid, ci]
    prev_fi1[wid, ci] = c_friction1_impulse[wid, ci]
    prev_fi2[wid, ci] = c_friction2_impulse[wid, ci]

    # Store count (only thread 0)
    if ci == 0:
        prev_count[wid] = nc
