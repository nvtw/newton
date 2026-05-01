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
"""

from __future__ import annotations

import warp as wp


@wp.struct
class PhoenxBroadphaseFilterData:
    """Runtime data the filter callback reads each launch.

    The struct is the broadphase-side analogue of NarrowPhase's
    ``ContactWriterData``: a single ``wp.struct`` packed once at
    finalize time and threaded through every broadphase launch.

    Attributes:
        num_rigid_shapes: Number of rigid shapes ``S``. Shape indices
            ``s < S`` are rigid; ``s >= S`` index a cloth triangle
            ``t = s - S`` in the ``tri_indices`` array.
        tri_indices: Per-triangle ``vec3i`` of particle indices, of
            length ``T``. Sourced from
            :attr:`newton.Model.tri_indices` at finalize.
    """

    num_rigid_shapes: wp.int32
    tri_indices: wp.array[wp.vec3i]


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
    a_is_tri = wp.int32(0)
    b_is_tri = wp.int32(0)
    if a >= ud.num_rigid_shapes:
        a_is_tri = wp.int32(1)
    if b >= ud.num_rigid_shapes:
        b_is_tri = wp.int32(1)

    if a_is_tri == wp.int32(1) and b_is_tri == wp.int32(1):
        ti = a - ud.num_rigid_shapes
        tj = b - ud.num_rigid_shapes
        ia = ud.tri_indices[ti]
        ib = ud.tri_indices[tj]
        # 9-iteration node-share test.  For non-adjacent triangle
        # pairs (the bulk on a typical cloth grid) every comparison
        # is false and the filter accepts.  For shared-node pairs
        # the early-out hits within the first few iterations.
        for u in range(3):
            for v in range(3):
                if ia[u] == ib[v]:
                    return wp.int32(0)
    return wp.int32(1)
