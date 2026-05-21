# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Supernodal ``ElementInteractionData`` emission from a cluster set.

Takes the output of :class:`ConstraintClusterBuilder` -- a ``vec4i`` per
cluster of constituent constraint ids -- and produces a parallel array
of ``ElementInteractionData`` indexed by cluster id, where each entry's
``bodies`` slot is the union of bodies referenced by all constraints in
that cluster.

The output is directly consumable by the existing graph coloring
partitioners: the supernodal graph has the same shape (one
``ElementInteractionData`` per node, edges defined by body sharing),
just with fewer, smaller-degree nodes. Wiring it through e.g.
:class:`IncrementalContactPartitioner` produces a colouring of clusters
which is the building block for the supernodal-XPBD sweep.

Determinism: the kernel walks members in vec4i slot order (which is
sorted ascending by constraint id from clustering's emit step) and
inserts bodies in first-encountered order. No atomics; pure per-thread
construction.

Graph capture safety: a single fixed-dim launch, gated on the active
``num_clusters[0]`` count.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.clustering.cluster_builder import MAX_CLUSTER_SIZE
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_add,
    element_interaction_data_contains,
    element_interaction_data_empty,
    element_interaction_data_get,
)

__all__ = ["SupernodalElements"]


@wp.kernel(enable_backward=False)
def _build_supernodal_elements_kernel(
    cluster_members: wp.array[wp.vec4i],
    num_clusters: wp.array[wp.int32],
    original_elements: wp.array[ElementInteractionData],
    # out
    supernodal_elements: wp.array[ElementInteractionData],
    # out: per-cluster active member count (1..MAX_CLUSTER_SIZE) -- handy
    # for the eventual PGS sweep that iterates "for cluster, for member".
    supernodal_member_counts: wp.array[wp.int32],
):
    """Per cluster, accumulate the body union of its members into one
    ``ElementInteractionData``. Slots past the active count and past
    ``num_clusters[0]`` are zeroed to an "empty" element (all -1) so
    downstream consumers can rely on the tail being well-formed.
    """
    cid = wp.tid()
    if cid >= cluster_members.shape[0]:
        return
    if cid >= num_clusters[0]:
        supernodal_elements[cid] = element_interaction_data_empty()
        supernodal_member_counts[cid] = wp.int32(0)
        return

    members = cluster_members[cid]
    out = element_interaction_data_empty()
    active_members = wp.int32(0)

    for slot in range(int(MAX_CLUSTER_SIZE)):
        m = members[slot]
        if m < 0:
            break
        active_members += wp.int32(1)
        el = original_elements[m]
        for j in range(MAX_BODIES):
            b = element_interaction_data_get(el, j)
            if b < 0:
                break
            if not element_interaction_data_contains(out, b):
                out = element_interaction_data_add(out, b)

    supernodal_elements[cid] = out
    supernodal_member_counts[cid] = active_members


class SupernodalElements:
    """Emits one ``ElementInteractionData`` per cluster, plus an active
    member count.

    Outputs (populated after :meth:`build`):

    :ivar elements: ``ElementInteractionData`` array, length
        ``max_num_clusters``. Each entry's ``bodies`` slot holds the
        union of bodies referenced by all constraints in that cluster
        (<= ``MAX_BODIES_PER_CLUSTER`` = 8 by construction). Entries
        past ``num_clusters[0]`` are zeroed to the empty element.
    :ivar num_clusters: device scalar, length 1. Mirrors the
        cluster builder's ``num_clusters``; surfaced here so a downstream
        graph-coloring partitioner can use this object as its element
        source without also wiring up the cluster builder.
    :ivar member_counts: int32 array, length ``max_num_clusters``. The
        number of constraint members per cluster (1..MAX_CLUSTER_SIZE,
        or 0 past ``num_clusters[0]``).

    Pre-allocates all device buffers; ``build`` is a single fixed-dim
    kernel launch, graph-capture safe and deterministic.
    """

    def __init__(
        self,
        max_num_clusters: int,
        device: wp.context.Devicelike = None,
    ) -> None:
        if max_num_clusters <= 0:
            raise ValueError(f"max_num_clusters must be > 0 (got {max_num_clusters})")
        self._max_num_clusters = int(max_num_clusters)
        self._device = wp.get_device(device)

        self.elements: wp.array[ElementInteractionData] = wp.zeros(
            self._max_num_clusters,
            dtype=ElementInteractionData,
            device=self._device,
        )
        self.member_counts: wp.array[wp.int32] = wp.zeros(self._max_num_clusters, dtype=wp.int32, device=self._device)
        # ``num_clusters`` is a passthrough handle -- :meth:`build`
        # stores a reference to the caller's array so downstream
        # consumers can read it via ``self.num_clusters`` after build.
        self.num_clusters: wp.array[wp.int32] | None = None

    @property
    def max_num_clusters(self) -> int:
        return self._max_num_clusters

    def build(
        self,
        cluster_members: wp.array,  # wp.array[wp.vec4i]
        num_clusters: wp.array[wp.int32],
        original_elements: wp.array,  # wp.array[ElementInteractionData]
    ) -> None:
        """Compute per-cluster supernodal elements.

        Args:
            cluster_members: ``vec4i`` per cluster from
                :class:`ConstraintClusterBuilder`. Length >=
                ``max_num_clusters``; the kernel handles the tail past
                ``num_clusters[0]`` by writing empty elements.
            num_clusters: device scalar holding the active cluster
                count.
            original_elements: ``ElementInteractionData`` array
                (per-constraint) that the clusters were built from.
        """
        self.num_clusters = num_clusters
        wp.launch(
            _build_supernodal_elements_kernel,
            dim=self._max_num_clusters,
            inputs=[cluster_members, num_clusters, original_elements],
            outputs=[self.elements, self.member_counts],
            device=self._device,
        )
