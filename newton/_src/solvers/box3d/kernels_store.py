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
    color_slot_to_raw: wp.array2d(dtype=wp.int32),
    contact_count: wp.array[wp.int32],
    # Outputs (prev-frame buffers, stored in raw order)
    prev_ni: wp.array2d(dtype=float),
    prev_fi1: wp.array2d(dtype=float),
    prev_fi2: wp.array2d(dtype=float),
    prev_count: wp.array[wp.int32],
):
    """Copy current solver impulses to previous-frame buffers (raw order).

    Launched with ``dim = (num_worlds, max_contacts_per_world)``.

    The impulses are in color order in the solver arrays. We scatter them
    back to raw order using ``color_slot_to_raw`` so they align with
    the contact matching indices from Newton's collision pipeline.
    """
    wid, ci = wp.tid()
    nc = contact_count[wid]
    if ci >= nc:
        return

    # ci is in color order; scatter to raw order for contact matching
    raw_idx = color_slot_to_raw[wid, ci]
    prev_ni[wid, raw_idx] = c_normal_impulse[wid, ci]
    prev_fi1[wid, raw_idx] = c_friction1_impulse[wid, ci]
    prev_fi2[wid, raw_idx] = c_friction2_impulse[wid, ci]

    # Store count (only thread 0)
    if ci == 0:
        prev_count[wid] = nc
