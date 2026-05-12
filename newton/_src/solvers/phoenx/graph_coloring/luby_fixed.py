# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-iteration Luby MIS coloring (C# PhoenX port).

Port of ``ContactPartitions.MaximalIndependentSetPartitioning2`` in
``experimentalsim/PhoenX/src/PhoenX/MassSplitting/ContactPartitions.cs``.
Runs **exactly** ``2 * MAX_LUBY_COLORS`` coloring kernel launches and
nothing else -- no ``capture_while``, no convergence check, no
``num_remaining`` atomic. Any element still uncoloured at the end is
written to the overflow colour, which mass splitting consumes via
copy-state slots.

Within each coloring launch, every uncoloured element checks neighbour
priorities for local-max status; on commit it marks not only itself
but also every uncoloured neighbour as "removed for this Luby
iteration" so the *next* iteration within the same colour can commit
those neighbours' non-conflicting ones. Two Luby iterations per
colour drains most conflicts; the remainder rolls to the next colour.

Compared to the greedy MIS in
:class:`IncrementalContactPartitioner`:

* Newton greedy MIS: per-launch one round of local-max commits, kept
  in a ``capture_while`` until ``num_remaining == 0``. Dense soft-tet
  contact graphs take 30+ outer rounds even with the safety cap.
* Luby-fixed: per-launch propagates commits via the per-iter
  removed-marker, so each launch makes more progress. Total launches
  ``= 2 * MAX_LUBY_COLORS`` (e.g. 16 for ``MAX_LUBY_COLORS = 8``),
  versus ~128 for greedy under the worst-case cap.

This implementation reuses the adjacency-build, histogram, scatter,
and locality-sort kernels from :mod:`graph_coloring_common` /
:mod:`graph_coloring_incremental`; only the coloring kernel itself is
new.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_contact_count,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    MAX_BODIES,
    ElementInteractionData,
    _COLOR_SHIFT,
    element_interaction_data_get,
    greedy_color_histogram_kernel,
    greedy_count_and_scan_color_starts_kernel,
    greedy_reset_init_kernel,
    greedy_scatter_elements_by_color_kernel,
    partitioning_adjacency_count_kernel,
    partitioning_adjacency_store_kernel,
    partitioning_prepare_kernel,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    MAX_COLORS,
    _GREEDY_BLOCK_DIM,
    _fill_cost_values_from_contacts_kernel,
    _locality_compute_keys_kernel,
    _locality_eid_keys_kernel,
    _locality_writeback_kernel,
    _greedy_coloring_grid_size,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import scan_variable_length

#: Default total colour budget for the fixed-iter build when no
#: ``max_colored_partitions`` is provided. Last slot
#: (``MAX_LUBY_COLORS``) is the overflow bucket the mass-splitting
#: copy-state machinery consumes. Without mass splitting the overflow
#: bucket is unsafe (constraints share bodies => solver races), so a
#: generous default is essential -- common dense scenes (soft tets,
#: cloth) only need 8-15 colours but constraint-graph chromatic number
#: can spike. Cost per colour is 2 launches.
MAX_LUBY_COLORS: int = 32

#: Sentinel marker meaning "not removed in any current Luby iter".
#: We compare ``removed_marker[i] in [lubyBase, lubyMarker)``; setting
#: marker to a value greater than the highest possible ``lubyMarker``
#: keeps the IsRemoved check false on fresh state.
_REMOVED_MARKER_NEVER: int = -1


@wp.kernel(enable_backward=False)
def _luby_reset_loop_state_kernel(
    partition_data_concat: wp.array[wp.int64],
    removed_marker: wp.array[wp.int32],
    interaction_id_to_partition: wp.array[wp.int32],
    num_elements: wp.array[wp.int32],
):
    """Stamp the loop-state arrays before each build. Mirrors
    ``incremental_reset_loop_state_kernel`` but for the Luby variant's
    state: no ``unpartitioned`` marker bit in ``partition_data_concat``
    -- we read ``rem == 0`` to mean "uncoloured"."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    partition_data_concat[tid] = wp.int64(tid)
    removed_marker[tid] = wp.int32(_REMOVED_MARKER_NEVER)
    interaction_id_to_partition[tid] = wp.int32(-1)


@wp.func
def _luby_is_removed(
    partition_data_concat: wp.array[wp.int64],
    removed_marker: wp.array[wp.int32],
    i: wp.int32,
    color: wp.int32,
    luby_base: wp.int32,
    luby_marker: wp.int32,
) -> bool:
    """Mirrors C# ``ContactPartitionsGpu.IsRemoved``. True iff element
    ``i`` is either marked-this-Luby-iter (``removed_marker[i] in
    [luby_base, luby_marker)``) or already committed to a previous
    colour (``partition_data_concat[i]`` high bits != 0 and != color+1).
    """
    marker = removed_marker[i]
    if marker >= luby_base and marker < luby_marker:
        return True
    pd = partition_data_concat[i]
    rem = wp.int32(pd >> _COLOR_SHIFT)
    # rem == 0 -> uncoloured. rem == color+1 -> already committed to
    # *this* colour (idempotent). rem == anything else -> committed to
    # an earlier colour, removed for this colour's iterations.
    if rem == wp.int32(0) or rem == color + wp.int32(1):
        return False
    return True


@wp.kernel(enable_backward=False)
def luby_coloring_kernel(
    partition_data_concat: wp.array[wp.int64],
    removed_marker: wp.array[wp.int32],
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[wp.int32],
    vertex_to_adjacent_elements: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    color: wp.int32,
    luby_base: wp.int32,
    luby_marker: wp.int32,
):
    """Port of C# ``PartitioningColoringKernel``. One call processes
    every still-uncoloured element under the current ``color``: tests
    local-max-priority and on commit marks the element + all uncoloured
    neighbours as "removed for this Luby iter" so the next call in the
    same colour skips them."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    if _luby_is_removed(partition_data_concat, removed_marker, tid, color, luby_base, luby_marker):
        return

    self_prio = (wp.int64(cost_values[tid]) << wp.int64(32)) | (wp.int64(random_values[tid]) & wp.int64(0xFFFFFFFF))

    el = elements[tid]
    is_local_max = bool(True)
    for j in range(MAX_BODIES):
        if not is_local_max:
            break
        v = element_interaction_data_get(el, j)
        if v < 0:
            break
        start = wp.int32(0)
        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if _luby_is_removed(
                partition_data_concat, removed_marker, neighbor, color, luby_base, luby_marker
            ):
                continue
            neigh_prio = (wp.int64(cost_values[neighbor]) << wp.int64(32)) | (
                wp.int64(random_values[neighbor]) & wp.int64(0xFFFFFFFF)
            )
            if neigh_prio > self_prio:
                is_local_max = False
                break

    if not is_local_max:
        return

    # Commit. Mark self + all uncoloured neighbours as removed for
    # this Luby iteration so the next call in the same colour skips
    # them. Then stamp the final colour in partition_data_concat.
    removed_marker[tid] = luby_marker
    for j in range(MAX_BODIES):
        v = element_interaction_data_get(el, j)
        if v < 0:
            break
        start = wp.int32(0)
        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if _luby_is_removed(
                partition_data_concat, removed_marker, neighbor, color, luby_base, luby_marker
            ):
                continue
            removed_marker[neighbor] = luby_marker

    partition_data_concat[tid] = (wp.int64(color + wp.int32(1)) << _COLOR_SHIFT) | wp.int64(tid)


@wp.kernel(enable_backward=False)
def _luby_spill_uncoloured_to_overflow_kernel(
    partition_data_concat: wp.array[wp.int64],
    num_elements: wp.array[wp.int32],
    overflow_color: wp.int32,
):
    """Final pass: any element still ``rem == 0`` (i.e. ``partition_data_concat
    [i] = i`` only) is stamped with the overflow colour. Mass splitting
    handles the within-bucket conflicts via copy-state slots."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    pd = partition_data_concat[tid]
    rem = wp.int32(pd >> _COLOR_SHIFT)
    if rem != wp.int32(0):
        return
    partition_data_concat[tid] = (wp.int64(overflow_color + wp.int32(1)) << _COLOR_SHIFT) | wp.int64(tid)


@wp.kernel(enable_backward=False)
def _luby_begin_sweep_kernel(num_colors: wp.array[wp.int32], color_cursor: wp.array[wp.int32]):
    """Copy num_colors -> color_cursor for the sweep loop."""
    if wp.tid() == 0:
        color_cursor[0] = num_colors[0]


class FixedIterationLubyPartitioner:
    """Fixed-iteration Luby MIS partitioner. Conforms to
    :class:`graph_coloring.base.ContactPartitioner`.

    Construction allocates all scratch sized to ``max_num_interactions``.
    Per step the solver calls :meth:`reset` with that step's elements +
    count, then :meth:`build_csr` to insert the coloring kernels into
    the captured graph.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.context.Devicelike = None,
        seed: int = 0,
        max_colored_partitions: int | None = None,
        max_luby_colors: int = MAX_LUBY_COLORS,
    ) -> None:
        if max_colored_partitions is not None and int(max_colored_partitions) != int(max_luby_colors):
            # The Luby loop uses ``max_luby_colors`` colours; the
            # overflow bucket = ``max_luby_colors``. ``max_colored_partitions``
            # was the API the existing partitioner exposed; we accept it
            # and use it as the colour budget directly.
            max_luby_colors = int(max_colored_partitions)
        if max_luby_colors < 1 or max_luby_colors >= int(GREEDY_MAX_COLORS):
            raise ValueError(
                f"max_luby_colors must be in [1, {int(GREEDY_MAX_COLORS)}); got {max_luby_colors}"
            )
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
        self.max_luby_colors = int(max_luby_colors)

        # JP priorities -- same construction as IncrementalContactPartitioner
        # (fixed seed makes the build deterministic).
        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)
        self._cost_values = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Per-element Luby state.
        self._removed_marker = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # partition_data_concat: high bits = colour+1, low 32 bits = eid.
        # Re-uses the int64 layout from graph_coloring_common so the
        # downstream histogram / scatter kernels work unchanged.
        self._partition_data_concat = wp.zeros(max_num_interactions, dtype=wp.int64, device=device)

        # Adjacency CSR (built per-step from elements).
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        # Worst case: every element claims every body slot.
        adjacency_capacity = max(1, max_num_interactions * int(MAX_BODIES))
        self._vertex_to_adjacent_elements = wp.zeros(adjacency_capacity, dtype=wp.int32, device=device)
        # ``partitioning_adjacency_store_kernel`` also writes a per-element
        # ``color_tags[tid] = 0`` slot. We don't read it from the Luby
        # path but it must be sized to ``max_num_interactions`` so the
        # write isn't OOB and doesn't corrupt neighbouring scratch.
        self._color_tags_dummy = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Persistent grid sizing -- same heuristic as the existing partitioner.
        self._coloring_grid_size: int = _greedy_coloring_grid_size(max_num_interactions, device)

        # Output CSR + sweep state. Sized to MAX_COLORS for symmetry with
        # the existing partitioner; we only fill the first
        # ``max_luby_colors + 1`` slots.
        self._element_ids_by_color = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._color_starts = wp.zeros(MAX_COLORS + 1, dtype=wp.int32, device=device)
        self._num_colors = wp.zeros(1, dtype=wp.int32, device=device)
        self._interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._color_cursor = wp.zeros(1, dtype=wp.int32, device=device)

        # Histogram / scatter scratch (reused from the existing common
        # kernels; sized at GREEDY_MAX_COLORS so the same kernels work).
        self._color_count = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        self._color_offsets = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        # Locality sort ping-pong (matches IncrementalContactPartitioner).
        self._locality_keys = wp.zeros(2 * max_num_interactions, dtype=wp.int64, device=device)
        self._locality_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)

        # Inputs set per step via reset().
        self._elements: wp.array | None = None
        self._num_elements: wp.array | None = None

        self._device = device

    # --- Public surface (matches ContactPartitioner protocol) -------------

    @property
    def element_ids_by_color(self) -> wp.array:
        return self._element_ids_by_color

    @property
    def color_starts(self) -> wp.array:
        return self._color_starts

    @property
    def num_colors(self) -> wp.array:
        return self._num_colors

    @property
    def interaction_id_to_partition(self) -> wp.array:
        return self._interaction_id_to_partition

    @property
    def color_cursor(self) -> wp.array:
        return self._color_cursor

    def reset(self, elements: wp.array, num_elements: wp.array) -> None:
        self._elements = elements
        self._num_elements = num_elements

    def set_costs_from_contacts(
        self,
        num_cids: int,
        num_contact_columns: wp.array,
        contact_cols: ContactColumnContainer,
    ) -> None:
        wp.launch(
            _fill_cost_values_from_contacts_kernel,
            dim=num_cids,
            inputs=[self._cost_values, contact_cols, num_contact_columns, wp.int32(0)],
            device=self._device,
        )

    def build_csr(self) -> None:
        """Insert the full coloring sequence into the current Warp graph."""
        assert self._elements is not None and self._num_elements is not None, (
            "FixedIterationLubyPartitioner.build_csr requires reset() first"
        )
        # --- Adjacency build (reused from common) ---
        wp.launch(
            partitioning_prepare_kernel,
            dim=max(self.max_num_interactions, self.max_num_nodes) + 1,
            inputs=[
                self._color_starts,
                self._num_colors,
                self._adjacency_section_end_indices,
                MAX_COLORS,
                self.max_num_nodes,
            ],
            device=self._device,
        )
        wp.launch(
            partitioning_adjacency_count_kernel,
            dim=self.max_num_interactions,
            inputs=[self._adjacency_section_end_indices, self._elements, self._num_elements],
            device=self._device,
        )
        # Exclusive scan: per-vertex count -> per-vertex START offset.
        # The store kernel below atomically advances each entry by the
        # per-vertex emit count, leaving the array with END-of-section
        # semantics that the coloring kernel reads as
        # ``adj[v-1] = start, adj[v] = end`` (matching the C# layout).
        scan_variable_length(
            self._adjacency_section_end_indices,
            self._num_elements,
            inclusive=False,
        )
        wp.launch(
            partitioning_adjacency_store_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._adjacency_section_end_indices,
                self._vertex_to_adjacent_elements,
                self._partition_data_concat,
                self._color_tags_dummy,
                self._elements,
                self._num_elements,
            ],
            device=self._device,
        )
        # --- Reset coloring state ---
        wp.launch(
            _luby_reset_loop_state_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._removed_marker,
                self._interaction_id_to_partition,
                self._num_elements,
            ],
            device=self._device,
        )
        wp.launch(
            greedy_reset_init_kernel,
            dim=int(GREEDY_MAX_COLORS),
            inputs=[
                self._num_colors,  # dummy overflow-flag stand-in (zeroed too, harmless)
                self._color_count,
                self._color_offsets,
                int(GREEDY_MAX_COLORS),
            ],
            device=self._device,
        )

        # --- Fixed Luby loop: 2 launches per colour, no convergence check ---
        for color in range(self.max_luby_colors):
            luby_base = 2 * color
            for luby in range(2):
                luby_marker = 2 * color + luby
                wp.launch(
                    luby_coloring_kernel,
                    dim=self.max_num_interactions,
                    inputs=[
                        self._partition_data_concat,
                        self._removed_marker,
                        self._random_values,
                        self._cost_values,
                        self._adjacency_section_end_indices,
                        self._vertex_to_adjacent_elements,
                        self._elements,
                        self._num_elements,
                        wp.int32(color),
                        wp.int32(luby_base),
                        wp.int32(luby_marker),
                    ],
                    device=self._device,
                )

        # --- Spill any still-uncoloured elements into the overflow bucket ---
        wp.launch(
            _luby_spill_uncoloured_to_overflow_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._num_elements,
                wp.int32(self.max_luby_colors),
            ],
            device=self._device,
        )

        # --- Histogram + colour-starts scan + scatter (reused) ---
        wp.launch(
            greedy_color_histogram_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._num_elements,
                self._color_count,
                self._interaction_id_to_partition,
            ],
            device=self._device,
        )
        wp.launch(
            greedy_count_and_scan_color_starts_kernel,
            dim=1,
            inputs=[
                self._color_count,
                self._color_starts,
                self._num_colors,
                int(GREEDY_MAX_COLORS),
            ],
            device=self._device,
        )
        wp.launch(
            greedy_scatter_elements_by_color_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._color_starts,
                self._color_offsets,
                self._element_ids_by_color,
                self._num_elements,
            ],
            device=self._device,
        )
        self._sort_csr_by_body_locality()

    def begin_sweep(self) -> None:
        wp.launch(
            _luby_begin_sweep_kernel,
            dim=1,
            inputs=[self._num_colors, self._color_cursor],
            device=self._device,
        )

    # --- Internal -----------------------------------------------------------

    def _sort_csr_by_body_locality(self) -> None:
        """Same 2-pass eid + (color, body_min) sort the
        IncrementalContactPartitioner uses for cache locality +
        deterministic overflow-bucket ordering."""
        n = self.max_num_interactions
        wp.launch(
            _locality_eid_keys_kernel,
            dim=2 * n,
            inputs=[
                self._element_ids_by_color,
                self._num_elements,
                self._locality_keys,
                self._locality_values,
            ],
            device=self._device,
        )
        wp.utils.radix_sort_pairs(self._locality_keys, self._locality_values, n)
        wp.launch(
            _locality_writeback_kernel,
            dim=n,
            inputs=[
                self._locality_values,
                self._num_elements,
                self._element_ids_by_color,
            ],
            device=self._device,
        )
        wp.launch(
            _locality_compute_keys_kernel,
            dim=2 * n,
            inputs=[
                self._elements,
                self._element_ids_by_color,
                self._interaction_id_to_partition,
                self._num_elements,
                self._locality_keys,
                self._locality_values,
            ],
            device=self._device,
        )
        wp.utils.radix_sort_pairs(self._locality_keys, self._locality_values, n)
        wp.launch(
            _locality_writeback_kernel,
            dim=n,
            inputs=[
                self._locality_values,
                self._num_elements,
                self._element_ids_by_color,
            ],
            device=self._device,
        )


__all__ = ["FixedIterationLubyPartitioner", "MAX_LUBY_COLORS", "luby_coloring_kernel"]
