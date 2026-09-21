# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Broad-phase filtering for sleeping PhoenX rigid bodies."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import MOTION_STATIC

__all__ = [
    "PhoenXSleepingFilterData",
    "build_phoenx_sleeping_filter_data",
    "phoenx_sleeping_filter",
]


@wp.struct
class PhoenXSleepingFilterData:
    """Data used to suppress pairs between two frozen rigid bodies."""

    enabled: wp.int32
    phoenx_body_offset: wp.int32
    shape_body: wp.array[wp.int32]
    body_island_root: wp.array[wp.int32]
    body_motion_type: wp.array[wp.int32]


@wp.func
def phoenx_sleeping_filter(pair: wp.vec2i, data: PhoenXSleepingFilterData) -> wp.int32:
    """Return zero when both shapes belong to frozen rigid bodies."""

    if data.enabled == wp.int32(0):
        return wp.int32(1)
    body_a = data.shape_body[pair[0]]
    body_b = data.shape_body[pair[1]]
    slot_a = wp.int32(0)
    slot_b = wp.int32(0)
    if body_a >= 0:
        slot_a = body_a + data.phoenx_body_offset
    if body_b >= 0:
        slot_b = body_b + data.phoenx_body_offset
    frozen_a = (data.body_island_root[slot_a] >= wp.int32(0)) or (data.body_motion_type[slot_a] == MOTION_STATIC)
    frozen_b = (data.body_island_root[slot_b] >= wp.int32(0)) or (data.body_motion_type[slot_b] == MOTION_STATIC)
    if frozen_a and frozen_b:
        return wp.int32(0)
    return wp.int32(1)


def build_phoenx_sleeping_filter_data(
    *,
    sleeping_enabled: bool,
    phoenx_body_offset: int,
    shape_body: wp.array[wp.int32] | None,
    body_island_root: wp.array[wp.int32] | None,
    body_motion_type: wp.array[wp.int32] | None,
    device: wp.context.Devicelike,
) -> PhoenXSleepingFilterData:
    """Build broad-phase filter data, using inert sentinels when disabled."""

    data = PhoenXSleepingFilterData()
    data.enabled = wp.int32(1 if sleeping_enabled else 0)
    data.phoenx_body_offset = wp.int32(int(phoenx_body_offset))
    data.shape_body = shape_body if shape_body is not None else wp.zeros(1, dtype=wp.int32, device=device)
    data.body_island_root = (
        body_island_root if body_island_root is not None else wp.full(1, -1, dtype=wp.int32, device=device)
    )
    data.body_motion_type = (
        body_motion_type if body_motion_type is not None else wp.zeros(1, dtype=wp.int32, device=device)
    )
    return data
