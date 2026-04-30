# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent-set partitioning of the constraint graph.

Mirrors the C# ``ContactPartitions`` class (host) plus
``ContactPartitionsGpu`` (device view) in
``PhoenX/MassSplitting/ContactPartitions.cs``. The job is to assign
each constraint to a partition such that no two constraints in the
same partition share a body -- that's what makes a partition
parallel-safe.

## Design notes

The C# implementation runs Luby's algorithm on device with adjacency
arrays. This Python port supports **two equivalent build paths** so
the mass-splitting machinery can plug into whichever colored
constraint graph the caller already has:

1. **`build(constraint_bodies)`** -- host-side greedy MIS partitioner
   for isolated tests / small scenes (~ms on numpy at 100k
   constraints). Sort constraints by descending body-degree, place
   each in the lowest-indexed partition whose body set is disjoint
   from the constraint's. Overflow goes to a Jacobi-remainder
   "additional partition".

2. **`wrap_color_arrays(element_ids_by_color, color_starts, num_colors)``**
   -- adopt an existing :mod:`newton._src.solvers.phoenx.graph_coloring`
   output verbatim. ``element_ids_by_color`` *is*
   ``partition_data_concat`` and ``color_starts`` *is*
   ``partition_ends`` (modulo a one-slot prefix shift); the
   construction is just a wrapper allocating
   ``num_partitions[]`` / ``has_additional_partition[]`` slots.

Path (2) is the integration target: PhoenX already builds a colored
constraint graph every step (greedy ``incremental_tile_compact_csr_and_advance_kernel``
or JP fallback), and each color is by definition an independent set
of constraints touching disjoint body sets. That's exactly what mass
splitting needs as input -- no re-partitioning required.

## Output layout (matches the C# ``ContactPartitionsGpu`` struct)

* ``partition_data_concat`` (``wp.array[wp.int32]``): concatenated
  constraint IDs, grouped by partition. Length ==
  ``max_num_constraints``.
* ``partition_ends`` (``wp.array[wp.int32]``): inclusive scan of
  per-partition counts; ``partition_ends[p]`` is the end index of
  partition ``p`` in ``partition_data_concat``. Length ==
  ``max_num_partitions + 1``.
* ``num_partitions`` (length-1 ``wp.array[wp.int32]``): number of
  partitions actually used (<= ``max_num_partitions``).
* ``has_additional_partition`` (length-1 ``wp.array[wp.int32]``):
  ``1`` if a Jacobi-remainder partition exists past
  ``num_partitions``, ``0`` otherwise.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "ContactPartitions",
    "ContactPartitionsData",
    "partitions_get_partition_end",
    "partitions_get_partition_size",
    "partitions_get_partition_start",
]


@wp.struct
class ContactPartitionsData:
    """Device-visible view onto the partition arrays.

    Mirrors the C# ``ContactPartitionsGpu`` struct (the inner data
    portion of ``ContactPartitions.cs:107-150``). The constraint
    iterate kernel takes this by value and indexes into
    ``partition_data_concat`` to enumerate constraints in a given
    partition.
    """

    partition_data_concat: wp.array[wp.int32]
    partition_ends: wp.array[wp.int32]
    num_partitions: wp.array[wp.int32]
    has_additional_partition: wp.array[wp.int32]
    max_num_partitions: wp.int32


# ---------------------------------------------------------------------------
# Host-side builder.
# ---------------------------------------------------------------------------


class ContactPartitions:
    """Host-side partitioner + owner of :class:`ContactPartitionsData`.

    Lifecycle:

    1. Construct with capacities (``max_num_rigids``,
       ``max_num_constraints``, ``max_num_partitions``).
    2. ``build(constraint_bodies)`` once per scene-build (or per
       step if the constraint graph mutates). ``constraint_bodies``
       is a list of body-id tuples per constraint.
    3. ``data`` -- the device view, ready for the iterate kernel.
    4. ``iter_partition_constraints(p)`` -- host-side iteration of
       partition ``p``'s constraint IDs (useful for tests / dispatch
       layers).

    Attributes:
        max_num_rigids: Number of bodies the partitioner can address.
            Per-body ``occupied[partition]`` slot counts are bounded
            by this.
        max_num_constraints: Capacity of ``partition_data_concat``;
            the largest constraint count any single ``build()`` may
            see.
        max_num_partitions: Maximum number of partitions; constraints
            that don't fit go into the additional partition.
        device: Warp device the buffers live on.
    """

    #: Hard upper bound -- each body's "occupied partitions" bitmask
    #: is a Python int, so this is for sanity, not perf. C# enforces
    #: ``ContactPartitionsGpu.MaxNumSupportedPartitions = 64``.
    MAX_SUPPORTED_PARTITIONS: int = 64

    def __init__(
        self,
        max_num_rigids: int,
        max_num_constraints: int,
        max_num_partitions: int = 12,
        device: wp.context.Devicelike = None,
    ) -> None:
        if max_num_partitions > self.MAX_SUPPORTED_PARTITIONS:
            raise ValueError(
                f"max_num_partitions {max_num_partitions} exceeds "
                f"MAX_SUPPORTED_PARTITIONS {self.MAX_SUPPORTED_PARTITIONS}"
            )
        if max_num_partitions <= 0:
            raise ValueError(f"max_num_partitions must be > 0 (got {max_num_partitions})")
        self.max_num_rigids = int(max_num_rigids)
        self.max_num_constraints = int(max_num_constraints)
        self.max_num_partitions = int(max_num_partitions)
        self.device = device
        # Result of the most recent build; ``None`` until build().
        self._data: ContactPartitionsData | None = None
        self._partition_data_concat: wp.array | None = None
        self._partition_ends: wp.array | None = None
        self._num_partitions_arr: wp.array | None = None
        self._has_additional_partition_arr: wp.array | None = None
        self._partitions_host: list[list[int]] = []
        self._additional_host: list[int] = []

    # ------------------------------------------------------------------
    # Build.
    # ------------------------------------------------------------------

    def build(self, constraint_bodies) -> ContactPartitionsData:
        """Greedily partition the constraint graph.

        Args:
            constraint_bodies: Iterable of iterables; element ``i``
                is the tuple of body IDs that constraint ``i`` touches
                (typically 1 or 2 for joints / contacts; up to 8 for
                C#'s ``ElementInteractionData``). Constraint IDs are
                implicit in the iteration order.

        Returns:
            The :class:`ContactPartitionsData` device view -- the
            same object accessible later via :attr:`data`.

        Raises:
            ValueError: if the constraint count exceeds
                ``max_num_constraints``.

        Greedy MIS with descending-degree ordering. Empirically
        produces partition counts within a few of the chromatic
        number on typical scenes; for pathological cases the
        ``additional_partition`` absorbs the leftover constraints
        and the solver runs them with a Jacobi sweep
        (``inv_factor`` scaling already accounts for this).
        """
        constraints: list[list[int]] = [list(bodies) for bodies in constraint_bodies]
        num_constraints = len(constraints)
        if num_constraints > self.max_num_constraints:
            raise ValueError(
                f"constraint count {num_constraints} exceeds max_num_constraints {self.max_num_constraints}"
            )
        # Sort indices by descending body-touch count -- the harder
        # constraints to place go first, leaving the slack for easy
        # ones. Stable secondary key (constraint id) keeps the result
        # reproducible.
        order = sorted(
            range(num_constraints),
            key=lambda i: (-len(constraints[i]), i),
        )
        # Each partition tracks the set of bodies it has claimed so
        # far. A constraint goes into the lowest-indexed partition
        # whose body set is disjoint from the constraint's.
        partitions: list[list[int]] = [[] for _ in range(self.max_num_partitions)]
        partition_bodies: list[set[int]] = [set() for _ in range(self.max_num_partitions)]
        additional: list[int] = []
        for cid in order:
            placed = False
            cbodies = constraints[cid]
            cset = set(cbodies)
            for p in range(self.max_num_partitions):
                if cset.isdisjoint(partition_bodies[p]):
                    partitions[p].append(cid)
                    partition_bodies[p].update(cset)
                    placed = True
                    break
            if not placed:
                additional.append(cid)
        # Sort each partition's constraint IDs ascending so kernel
        # iteration order is deterministic and reproducible.
        for p in range(self.max_num_partitions):
            partitions[p].sort()
        additional.sort()
        # Drop trailing empty partitions to keep ``num_partitions``
        # tight (matches C#'s ``numPartitions[0]`` reflecting the
        # last *non-empty* coloring round, not the cap).
        while partitions and not partitions[-1]:
            partitions.pop()
        self._partitions_host = partitions
        self._additional_host = additional
        # Pack into device arrays.
        import numpy as np  # noqa: PLC0415 -- lazy

        flat = np.zeros(self.max_num_constraints, dtype=np.int32)
        ends = np.zeros(self.max_num_partitions + 1, dtype=np.int32)
        cursor = 0
        for p, part in enumerate(partitions):
            for cid in part:
                flat[cursor] = cid
                cursor += 1
            ends[p] = cursor
        # Tail: pad ``ends`` with the last cursor so out-of-range
        # reads stay sane.
        for p in range(len(partitions), self.max_num_partitions):
            ends[p] = cursor
        # Additional partition appended *after* the regular tail; its
        # range is ``[cursor_before, cursor_after)``.
        for cid in additional:
            flat[cursor] = cid
            cursor += 1
        ends[self.max_num_partitions] = cursor
        self._partition_data_concat = wp.array(flat, dtype=wp.int32, device=self.device)
        self._partition_ends = wp.array(ends, dtype=wp.int32, device=self.device)
        self._num_partitions_arr = wp.array(
            np.array([len(partitions)], dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._has_additional_partition_arr = wp.array(
            np.array([1 if additional else 0], dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        data = ContactPartitionsData()
        data.partition_data_concat = self._partition_data_concat
        data.partition_ends = self._partition_ends
        data.num_partitions = self._num_partitions_arr
        data.has_additional_partition = self._has_additional_partition_arr
        data.max_num_partitions = wp.int32(self.max_num_partitions)
        self._data = data
        return data

    # ------------------------------------------------------------------
    # Wrap an externally-built coloring (zero-copy where the dtypes
    # already match Newton's existing graph_coloring kernels).
    # ------------------------------------------------------------------

    def wrap_color_arrays(
        self,
        element_ids_by_color: wp.array,
        color_starts: wp.array,
        num_colors: int | wp.array,
        constraint_bodies=None,
    ) -> ContactPartitionsData:
        """Adopt an existing graph-coloring output as the partition table.

        Newton's
        :mod:`newton._src.solvers.phoenx.graph_coloring` produces a
        color-major CSR of constraint elements where each color is by
        definition an independent set (no two elements in the same
        color share a body). That's the same shape mass splitting
        needs as input, so this factory wraps the existing arrays
        instead of re-running a partitioning pass.

        Args:
            element_ids_by_color: ``wp.array[wp.int32]``;
                concatenated constraint IDs grouped by color. Same
                semantics as ``partition_data_concat``. Must live on
                ``self.device``.
            color_starts: ``wp.array[wp.int32]``; per-color *start*
                offsets into ``element_ids_by_color``. The Newton
                convention is ``color_starts[c]`` = first element of
                color ``c``, ``color_starts[c+1]`` = end (exclusive).
                We translate this to the
                C# ``partition_ends`` convention -- per-color *end*
                offsets, length ``max_num_partitions + 1`` -- in a
                small device kernel below.
            num_colors: number of regular partitions (excluding the
                Jacobi remainder). ``int`` for static configs,
                ``wp.array[wp.int32]`` length-1 for graph-captured
                rebuilds.
            constraint_bodies: optional iterable used to populate the
                companion :class:`InteractionGraph`. If provided, the
                caller can fetch ``self.last_constraint_bodies`` to
                feed the graph builder; if not, the caller is
                expected to register interaction-graph entries
                independently.

        Returns:
            The :class:`ContactPartitionsData` device view.
        """
        # Unify ``num_colors`` to a Python int for now -- the array
        # form is needed only at graph-capture rebuild time, which is
        # a follow-up. ``element_ids_by_color`` and ``color_starts``
        # however must be device arrays so the kernels see them.
        if isinstance(num_colors, wp.array):
            num_partitions_int = int(num_colors.numpy()[0])
        else:
            num_partitions_int = int(num_colors)
        if num_partitions_int > self.max_num_partitions:
            raise ValueError(f"num_colors {num_partitions_int} exceeds max_num_partitions {self.max_num_partitions}")
        # Build ``partition_ends`` from ``color_starts`` on the host.
        # This is a tiny array (max 64 partitions per the
        # MAX_SUPPORTED_PARTITIONS cap) so the cost is negligible.
        import numpy as np  # noqa: PLC0415

        color_starts_np = color_starts.numpy().astype(np.int32)
        partition_ends_np = np.zeros(self.max_num_partitions + 1, dtype=np.int32)
        # ``color_starts[c]`` is the start of color c; the end is
        # ``color_starts[c + 1]`` if it exists, else the total
        # count. Translate to per-color *end* indices.
        for c in range(self.max_num_partitions):
            if c < num_partitions_int:
                partition_ends_np[c] = (
                    color_starts_np[c + 1] if c + 1 < color_starts_np.shape[0] else color_starts_np[-1]
                )
            else:
                # Past the active partitions: pad with the last end.
                partition_ends_np[c] = (
                    color_starts_np[num_partitions_int]
                    if num_partitions_int < color_starts_np.shape[0]
                    else (color_starts_np[-1] if color_starts_np.shape[0] > 0 else 0)
                )
        # No additional partition for now -- Newton's coloring places
        # everything into a regular color (greedy with JP fallback).
        partition_ends_np[self.max_num_partitions] = partition_ends_np[self.max_num_partitions - 1]
        self._partition_data_concat = element_ids_by_color
        self._partition_ends = wp.array(partition_ends_np, dtype=wp.int32, device=self.device)
        self._num_partitions_arr = wp.array(
            np.array([num_partitions_int], dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._has_additional_partition_arr = wp.array(
            np.array([0], dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        # Reconstruct host-side partition lists for
        # ``iter_partition_constraints`` so callers that mix device
        # iteration and host inspection see a consistent picture.
        flat_np = element_ids_by_color.numpy().astype(np.int32)
        partitions_host: list[list[int]] = []
        for c in range(num_partitions_int):
            start = int(color_starts_np[c]) if c < color_starts_np.shape[0] else 0
            end = int(color_starts_np[c + 1]) if c + 1 < color_starts_np.shape[0] else int(color_starts_np[-1])
            partitions_host.append([int(x) for x in flat_np[start:end]])
        self._partitions_host = partitions_host
        self._additional_host = []
        if constraint_bodies is not None:
            self._last_constraint_bodies = [list(b) for b in constraint_bodies]
        else:
            self._last_constraint_bodies = None
        data = ContactPartitionsData()
        data.partition_data_concat = self._partition_data_concat
        data.partition_ends = self._partition_ends
        data.num_partitions = self._num_partitions_arr
        data.has_additional_partition = self._has_additional_partition_arr
        data.max_num_partitions = wp.int32(self.max_num_partitions)
        self._data = data
        return data

    @property
    def last_constraint_bodies(self):
        """The ``constraint_bodies`` argument from the most recent
        :meth:`wrap_color_arrays` call, or ``None`` if not supplied.
        Plumbing convenience for chaining into
        :meth:`InteractionGraph.add_entries`."""
        return getattr(self, "_last_constraint_bodies", None)

    # ------------------------------------------------------------------
    # Read-only host accessors.
    # ------------------------------------------------------------------

    @property
    def data(self) -> ContactPartitionsData:
        """Device view; requires :meth:`build` to have been called."""
        if self._data is None:
            raise RuntimeError("ContactPartitions.build(...) must be called before data is accessed")
        return self._data

    @property
    def num_partitions(self) -> int:
        """Number of regular (non-additional) partitions used."""
        return len(self._partitions_host)

    @property
    def has_additional_partition(self) -> bool:
        """``True`` iff some constraints overflowed into the Jacobi
        remainder partition."""
        return bool(self._additional_host)

    def iter_partition_constraints(self, partition_index: int):
        """Iterate constraint IDs in partition ``partition_index``.

        ``partition_index == self.max_num_partitions`` (or
        equivalently ``num_partitions``) yields the additional
        Jacobi-remainder partition; lower indices yield the regular
        partitions in order.
        """
        if partition_index < 0:
            raise IndexError(f"partition_index {partition_index} < 0")
        if partition_index >= len(self._partitions_host):
            if partition_index == len(self._partitions_host) and self._additional_host:
                yield from self._additional_host
                return
            return
        yield from self._partitions_host[partition_index]


# ---------------------------------------------------------------------------
# Device-side @wp.func helpers.
# ---------------------------------------------------------------------------


@wp.func
def partitions_get_partition_start(
    partitions: ContactPartitionsData,
    partition_index: wp.int32,
) -> wp.int32:
    """Start index of partition ``partition_index`` in
    ``partition_data_concat``."""
    if partition_index <= wp.int32(0):
        return wp.int32(0)
    return partitions.partition_ends[partition_index - wp.int32(1)]


@wp.func
def partitions_get_partition_end(
    partitions: ContactPartitionsData,
    partition_index: wp.int32,
) -> wp.int32:
    """End index (exclusive) of partition ``partition_index``."""
    return partitions.partition_ends[partition_index]


@wp.func
def partitions_get_partition_size(
    partitions: ContactPartitionsData,
    partition_index: wp.int32,
) -> wp.int32:
    """Number of constraints in partition ``partition_index``."""
    start = partitions_get_partition_start(partitions, partition_index)
    end = partitions_get_partition_end(partitions, partition_index)
    return end - start
