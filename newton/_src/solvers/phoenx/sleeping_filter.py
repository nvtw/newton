# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Broad-phase filtering for sleeping PhoenX rigid bodies and runtime joint pairs."""

from __future__ import annotations

import warp as wp

from newton._src.geometry.flags import MeshProperties
from newton._src.solvers.phoenx.body import MOTION_STATIC

__all__ = [
    "PhoenXSleepingFilterData",
    "build_phoenx_sleeping_filter_data",
    "phoenx_passive_shape",
    "phoenx_sleeping_filter",
]


_SURFACE_VELOCITY = wp.constant(wp.int32(int(MeshProperties.SURFACE_VELOCITY)))


@wp.struct
class PhoenXSleepingFilterData:
    """Data used to suppress pairs between two frozen rigid bodies or two jointed bodies."""

    enabled: wp.int32
    #: Per Newton body, the parent of a runtime joint whose shapes it must not collide with
    #: (``-2`` for none), or empty.
    body_filter_parent: wp.array[wp.int32]
    phoenx_body_offset: wp.int32
    shape_body: wp.array[wp.int32]
    shape_mesh_properties: wp.array[wp.int32]
    body_island_root: wp.array[wp.int32]
    body_motion_type: wp.array[wp.int32]
    #: Per PhoenX body slot, nonzero when the sleeping island rooted there
    #: rests on a surface-velocity mesh, or empty.
    belt_island: wp.array[wp.int32]


@wp.func
def phoenx_passive_shape(shape: int, data: PhoenXSleepingFilterData) -> bool:
    """Return whether a shape can rely on active neighbors to initiate broad-phase queries."""
    if data.enabled == 0:
        return False
    body = data.shape_body[shape]
    slot = wp.int32(0)
    if body >= 0:
        slot = body + data.phoenx_body_offset
    root = data.body_island_root[slot]
    if root < 0 and data.body_motion_type[slot] != MOTION_STATIC:
        return False
    # A belt can wake its sleeping stack only after collision detection.
    if data.shape_mesh_properties.shape[0] > 0 and (data.shape_mesh_properties[shape] & _SURFACE_VELOCITY) != 0:
        return False
    return root < 0 or data.belt_island.shape[0] == 0 or data.belt_island[root] == 0


@wp.func
def phoenx_sleeping_filter(pair: wp.vec2i, data: PhoenXSleepingFilterData) -> wp.int32:
    """Return zero when both shapes belong to frozen rigid bodies or to a filtered joint pair."""

    body_a = data.shape_body[pair[0]]
    body_b = data.shape_body[pair[1]]
    if data.body_filter_parent.shape[0] > 0:
        if body_a >= 0 and data.body_filter_parent[body_a] == body_b:
            return wp.int32(0)
        if body_b >= 0 and data.body_filter_parent[body_b] == body_a:
            return wp.int32(0)
    if phoenx_passive_shape(pair[0], data) and phoenx_passive_shape(pair[1], data):
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
    body_filter_parent: wp.array[wp.int32] | None = None,
    shape_mesh_properties: wp.array[wp.int32] | None = None,
    belt_island: wp.array[wp.int32] | None = None,
) -> PhoenXSleepingFilterData:
    """Build broad-phase filter data, using inert sentinels when disabled."""

    data = PhoenXSleepingFilterData()
    data.enabled = wp.int32(1 if sleeping_enabled else 0)
    data.body_filter_parent = (
        body_filter_parent if body_filter_parent is not None else wp.empty(0, dtype=wp.int32, device=device)
    )
    data.shape_mesh_properties = (
        shape_mesh_properties if shape_mesh_properties is not None else wp.empty(0, dtype=wp.int32, device=device)
    )
    data.belt_island = belt_island if belt_island is not None else wp.empty(0, dtype=wp.int32, device=device)
    data.phoenx_body_offset = wp.int32(int(phoenx_body_offset))
    data.shape_body = shape_body if shape_body is not None else wp.zeros(1, dtype=wp.int32, device=device)
    data.body_island_root = (
        body_island_root if body_island_root is not None else wp.full(1, -1, dtype=wp.int32, device=device)
    )
    data.body_motion_type = (
        body_motion_type if body_motion_type is not None else wp.zeros(1, dtype=wp.int32, device=device)
    )
    return data
