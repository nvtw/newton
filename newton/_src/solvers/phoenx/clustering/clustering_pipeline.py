# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Composed clustering pipeline: cluster builder + supernodal emitter.

Owns the two helpers that together turn a per-constraint
``ElementInteractionData`` graph into a supernodal one:

  1. :class:`ConstraintClusterBuilder` groups up to 4 constraints into
     each cluster, capped at 8 unique bodies per cluster.
  2. :class:`SupernodalElements` emits one ``ElementInteractionData``
     per cluster whose ``bodies`` slot is the body union of its
     members.

Single ``build()`` call drives both. Pre-allocates all scratch; safe
inside a captured Warp graph.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.clustering.cluster_builder import ConstraintClusterBuilder
from newton._src.solvers.phoenx.clustering.supernodal_elements import SupernodalElements

__all__ = ["ClusteringPipeline"]


class ClusteringPipeline:
    """Cluster builder + supernodal element emitter, glued together.

    After :meth:`build`:

    * :attr:`num_clusters`, :attr:`cluster_members`,
      :attr:`element_to_cluster` -- mirror the cluster builder's
      outputs.
    * :attr:`supernodal_elements`, :attr:`supernodal_member_counts` --
      mirror the supernodal builder's outputs. The supernodal element
      array can be fed straight into the existing graph coloring
      partitioner as a drop-in replacement for the per-constraint
      element array.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ) -> None:
        self._cluster_builder = ConstraintClusterBuilder(
            max_num_interactions=max_num_interactions,
            max_num_nodes=max_num_nodes,
            device=device,
            seed=seed,
        )
        # max_num_clusters == max_num_interactions: in the degenerate
        # case (all singletons) every constraint is its own cluster.
        self._supernodal = SupernodalElements(
            max_num_clusters=max_num_interactions,
            device=device,
        )

    def build(
        self,
        elements: wp.array,  # wp.array[ElementInteractionData]
        num_elements: wp.array[wp.int32],
    ) -> None:
        """Build clusters then emit supernodal elements.

        Both stages launch fixed-dim kernels gated by ``num_elements[0]``
        and the cluster count -- safe inside a captured graph. After
        the call, all output device arrays are populated.
        """
        self._cluster_builder.build_clusters(elements, num_elements)
        self._supernodal.build(
            self._cluster_builder.cluster_members,
            self._cluster_builder.num_clusters,
            elements,
        )

    # --- Cluster builder outputs (passthroughs) --------------------------

    @property
    def num_clusters(self) -> wp.array:
        """Device scalar (length 1): number of clusters from the last
        :meth:`build` call."""
        return self._cluster_builder.num_clusters

    @property
    def cluster_members(self) -> wp.array:
        """``vec4i`` per cluster of constraint ids (sorted ascending,
        ``-1`` for unused slots). Length ``max_num_interactions``;
        only the first ``num_clusters[0]`` entries are valid."""
        return self._cluster_builder.cluster_members

    @property
    def element_to_cluster(self) -> wp.array:
        """Per-element dense cluster id (``-1`` for inactive
        elements). Length ``max_num_interactions``."""
        return self._cluster_builder.element_to_cluster

    # --- Supernodal outputs (passthroughs) -------------------------------

    @property
    def supernodal_elements(self) -> wp.array:
        """``ElementInteractionData`` per cluster. Bodies slot is the
        union of cluster members' bodies (<= 8). Length
        ``max_num_interactions``; entries past ``num_clusters[0]`` are
        the empty element (all -1)."""
        return self._supernodal.elements

    @property
    def supernodal_member_counts(self) -> wp.array:
        """Per-cluster member count (1..4); 0 past ``num_clusters[0]``."""
        return self._supernodal.member_counts

    @property
    def max_num_interactions(self) -> int:
        return self._cluster_builder.max_num_interactions

    # --- Underlying components (for tests / advanced consumers) ----------

    @property
    def cluster_builder(self) -> ConstraintClusterBuilder:
        return self._cluster_builder

    @property
    def supernodal(self) -> SupernodalElements:
        return self._supernodal
