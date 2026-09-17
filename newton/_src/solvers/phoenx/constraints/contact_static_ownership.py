# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Identify the physical body that owns a rigid dynamic/static contact."""

import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, MOTION_STATIC, BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_side0_kind,
    contact_get_side1_kind,
)


@wp.func
def static_owner(columns: ContactColumnContainer, cid: int, bodies: BodyContainer):
    owner = int(-1)
    if contact_get_side0_kind(columns, cid) == 0 and contact_get_side1_kind(columns, cid) == 0:
        a = contact_get_body1(columns, cid)
        b = contact_get_body2(columns, cid)
        if bodies.motion_type[a] == MOTION_DYNAMIC and bodies.motion_type[b] == MOTION_STATIC:
            owner = a
        elif bodies.motion_type[b] == MOTION_DYNAMIC and bodies.motion_type[a] == MOTION_STATIC:
            owner = b
    return owner
