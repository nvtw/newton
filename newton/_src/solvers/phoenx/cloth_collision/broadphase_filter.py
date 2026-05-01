# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX broadphase filter callback.

Plugs into the broadphase ``filter_func`` hook
(:mod:`newton._src.geometry.broad_phase_common` /
:class:`~newton._src.geometry.broad_phase_nxn.BroadPhaseAllPairs` /
:class:`~newton._src.geometry.broad_phase_sap.BroadPhaseSAP`) to drop
cloth-triangle pair candidates that we don't want the narrow phase
to spend time on:

* **Self-pair (same triangle vs itself).** The shape index space
  treats each cloth triangle as a virtual shape; a triangle's AABB
  trivially overlaps with itself.
* **Adjacent triangles (share at least one node).** Two triangles
  that share a vertex or an edge are mechanically coupled by the
  cloth's elasticity rows -- emitting a contact between them would
  produce a fictitious coupling and fight the elastic constraints.

The filter only fires when both shape indices are above the rigid
threshold ``num_rigid_shapes`` (i.e. both are cloth triangles) -- a
rigid-vs-triangle pair always survives.

Topology is read **directly from the cloth-triangle constraint
container** rather than a duplicate ``tri_indices`` buffer: every
cloth row already stores its three particle indices in the
``body1 / body2 / body3`` slots, so reusing them keeps the data in
one place and avoids any risk of the topology going out of sync.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_get_body1,
    cloth_triangle_get_body2,
    cloth_triangle_get_body3,
)
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer


@wp.struct
class PhoenxBroadphaseFilterData:
    """Runtime data the filter callback reads each launch.

    The struct is the broadphase-side analogue of NarrowPhase's
    ``ContactWriterData``: a single ``wp.struct`` packed once at
    finalize time and threaded through every broadphase launch.

    Attributes:
        num_rigid_shapes: Number of rigid shapes ``S``. Shape indices
            ``s < S`` are rigid; ``s >= S`` index a cloth triangle
            ``t = s - S``.
        constraints: Phoenx :class:`ConstraintContainer` whose cloth
            triangle rows hold the per-triangle particle indices.
        cloth_cid_offset: Constraint-id offset for the cloth-triangle
            block (= ``num_joints``). Triangle ``t``'s row lives at
            ``cid = cloth_cid_offset + t``.
    """

    num_rigid_shapes: wp.int32
    constraints: ConstraintContainer
    cloth_cid_offset: wp.int32


@wp.func
def phoenx_broadphase_filter(
    pair: wp.vec2i,
    ud: PhoenxBroadphaseFilterData,
) -> wp.int32:
    """Return ``1`` to keep the pair, ``0`` to drop it.

    A pair is dropped only when both shapes are cloth triangles and
    the two triangles share at least one particle node. Self-pairs
    (a triangle vs itself) are caught by this rule as well -- every
    node matches.
    """
    a = pair[0]
    b = pair[1]
    if a >= ud.num_rigid_shapes and b >= ud.num_rigid_shapes:
        cid_a = ud.cloth_cid_offset + (a - ud.num_rigid_shapes)
        cid_b = ud.cloth_cid_offset + (b - ud.num_rigid_shapes)
        ai0 = cloth_triangle_get_body1(ud.constraints, cid_a)
        ai1 = cloth_triangle_get_body2(ud.constraints, cid_a)
        ai2 = cloth_triangle_get_body3(ud.constraints, cid_a)
        bi0 = cloth_triangle_get_body1(ud.constraints, cid_b)
        bi1 = cloth_triangle_get_body2(ud.constraints, cid_b)
        bi2 = cloth_triangle_get_body3(ud.constraints, cid_b)
        # 9-pair node-share test. Both arrays already carry unified
        # body-or-particle indices (the cloth-row stamping kernel
        # added ``num_bodies`` once at populate time), so equality
        # is the same as comparing raw particle indices.
        if ai0 == bi0 or ai0 == bi1 or ai0 == bi2:
            return wp.int32(0)
        if ai1 == bi0 or ai1 == bi1 or ai1 == bi2:
            return wp.int32(0)
        if ai2 == bi0 or ai2 == bi1 or ai2 == bi2:
            return wp.int32(0)
    return wp.int32(1)
