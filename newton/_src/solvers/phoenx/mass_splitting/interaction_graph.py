# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-body partition lookup for mass splitting.

Mirrors the C# ``MassSplittingRigidBodyInteractionGraph`` (host)
plus its device-resident ``MassSplittingRigidBodyInteractionGraphGpu``
companion in ``MassSplittingTypes.cuh``.

The data structure answers one question per call:

    "Given constraint ``c`` is touching body ``b``, which slot in
    ``tiny_states`` holds body ``b``'s velocity-state copy for
    constraint ``c``'s partition?"

…and a side-channel: ``inv_factor``, the count of partitions body
``b`` participates in. The constraint solve scales its impulse by
``1 / inv_factor`` so the periodic
:func:`~newton._src.solvers.phoenx.mass_splitting.kernels.average_and_broadcast`
recombines the per-partition copies into a single momentum-conserving
update.

## Layout

The two big arrays are kept *parallel*:

* ``partition_list`` -- ``wp.array[wp.int32]`` of length
  ``max_interactions``. Sorted constraint IDs grouped by body, with
  body ``i``'s slice at ``[state_section_end_indices[i-1],
  state_section_end_indices[i])`` (with the convention
  ``state_section_end_indices[-1] = 0``).
* ``tiny_states`` -- ``wp.array[TinyRigidState]`` of the same length.
  Slot ``k`` holds the velocity-state copy for the body / partition
  identified by ``partition_list[k]``.

``state_section_end_indices`` is a per-body inclusive scan of
"how many partitions does body ``i`` touch". One ``int`` per body.

``highest_index_in_use`` is a length-1 device array carrying the
plus-one of the highest body index that has any interactions; the
fan-out kernels gate on this so static / unused bodies cost nothing.

## Build pipeline (host-driven)

The C# code does the build entirely on device via a
``DuplicateFreeList``-on-device construction-helper plus three
``BuildInteractionGraph*Kernel`` passes (see
``MassSplittingRigidBodyInteractionGraph.cs:129-154``). This Python
port keeps the build *host-driven* for now -- a Python list
accumulates ``(rigid_id, constraint_id)`` pairs, sorts /
deduplicates, then a single device upload + tiny scan kernel
materialises the device arrays. Per-step code stays unchanged
between the two -- it only consumes the device arrays. Moving the
build itself onto the device is a perf follow-up; the host build is
~milliseconds even at 100k interactions.

## Per-step API

Once built, only :func:`graph_get_rigid_state_index` is hot; the
constraint kernels call it inside ``read_state`` /
``write_state``. Everything else is amortised per substep.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.mass_splitting.state import TinyRigidState

__all__ = [
    "InteractionGraph",
    "InteractionGraphData",
    "graph_get_rigid_state_index",
    "graph_get_state",
    "graph_set_state",
    "graph_state_section",
]


@wp.struct
class InteractionGraphData:
    """Device-visible view onto the interaction graph's arrays.

    Mirrors the C# ``MassSplittingRigidBodyInteractionGraphGpu``
    struct (``MassSplittingTypes.cuh:25-301``). The constraint /
    broadcast kernels take this as input by value; the underlying
    arrays are owned by :class:`InteractionGraph` on the host.

    Attributes:
        partition_list: Sorted constraint IDs per body. Length =
            ``max_interactions``. Body ``i``'s slice is
            ``[start, end)`` where ``end = state_section_end_indices[i]``
            and ``start = i > 0 ? state_section_end_indices[i-1] : 0``.
        tiny_states: Velocity-state copies, one per
            ``(body, partition)`` pair. Same length / indexing as
            ``partition_list``.
        state_section_end_indices: Inclusive scan of per-body
            partition counts. Length = ``max_rigid_bodies``.
        highest_index_in_use: Length-1 array. Plus-one of the highest
            body index with interactions. Static bodies fall through.
        max_interactions: Capacity of ``partition_list`` /
            ``tiny_states``; never changes after construction.
    """

    partition_list: wp.array[wp.int32]
    tiny_states: wp.array[TinyRigidState]
    state_section_end_indices: wp.array[wp.int32]
    highest_index_in_use: wp.array[wp.int32]
    max_interactions: wp.int32


# ---------------------------------------------------------------------------
# Host-side builder.
# ---------------------------------------------------------------------------


class InteractionGraph:
    """Host-side owner / builder of an :class:`InteractionGraphData`.

    Mirrors the C# ``MassSplittingRigidBodyInteractionGraph`` class
    (``MassSplittingRigidBodyInteractionGraph.cs:13-179``). Lifecycle:

    1. ``add_entry(rigid_id, constraint_id)`` once per
       (body, constraint) pair the partitioner will ever produce.
       O(1) host-side append.
    2. ``build(device)`` once after all entries are accumulated.
       Sorts / deduplicates the (rigid, constraint) pairs, uploads
       them to device, and runs an inclusive scan to populate
       ``state_section_end_indices`` / ``highest_index_in_use``.
    3. ``data`` -- read-only property returning the
       :class:`InteractionGraphData` device view, ready for kernels.

    A typical use does (1) once per scene-build, (2) once per
    scene-build, and then re-runs the per-substep kernels every
    step using the already-built ``data``.

    Attributes:
        max_rigid_bodies: Capacity of ``state_section_end_indices``.
        max_interactions: Capacity of ``partition_list`` /
            ``tiny_states`` (= the largest expected
            ``sum_b partitions_touched_by_body_b``).
        device: Warp device the buffers live on.
    """

    def __init__(
        self,
        max_rigid_bodies: int,
        max_interactions: int,
        device: wp.context.Devicelike = None,
    ) -> None:
        if max_rigid_bodies <= 0:
            raise ValueError(f"max_rigid_bodies must be > 0 (got {max_rigid_bodies})")
        if max_interactions <= 0:
            raise ValueError(f"max_interactions must be > 0 (got {max_interactions})")
        self.max_rigid_bodies = int(max_rigid_bodies)
        self.max_interactions = int(max_interactions)
        self.device = device
        # Host accumulator: list of (rigid_id, constraint_id) pairs,
        # mirroring C#'s ``DuplicateFreeList<long>`` pre-build state.
        self._entries: list[tuple[int, int]] = []
        # Device arrays -- allocated lazily in ``build()`` so empty
        # graphs (no entries) still produce a valid zero-length view.
        self._partition_list: wp.array | None = None
        self._tiny_states: wp.array | None = None
        self._state_section_end_indices: wp.array | None = None
        self._highest_index_in_use: wp.array | None = None
        self._built: bool = False
        self._data: InteractionGraphData | None = None

    # ------------------------------------------------------------------
    # Host accumulator.
    # ------------------------------------------------------------------

    def add_entry(self, rigid_id: int, constraint_id: int) -> None:
        """Record one ``(body, constraint)`` interaction.

        Mirrors the C# ``AddEntry(int rigidHandle, int constraintId)``
        overload. Callers typically iterate the partitioner's output
        and call this once per ``(constraint, touched-body)`` edge.
        Duplicates are tolerated -- :meth:`build` deduplicates -- but
        avoiding them on the host saves memory.
        """
        if rigid_id < 0:
            raise ValueError(f"rigid_id must be >= 0 (got {rigid_id})")
        if rigid_id >= self.max_rigid_bodies:
            raise ValueError(f"rigid_id {rigid_id} >= max_rigid_bodies {self.max_rigid_bodies}")
        if constraint_id < 0:
            raise ValueError(f"constraint_id must be >= 0 (got {constraint_id})")
        self._entries.append((int(rigid_id), int(constraint_id)))

    def add_entries(self, pairs) -> None:
        """Bulk-record entries; equivalent to a loop over
        :meth:`add_entry` but skips per-call validation in tight builds.
        """
        for rigid_id, constraint_id in pairs:
            self.add_entry(rigid_id, constraint_id)

    # ------------------------------------------------------------------
    # Build (device upload + scan).
    # ------------------------------------------------------------------

    def build(self) -> InteractionGraphData:
        """Materialise the device-resident graph.

        Sort + dedup the host accumulator, upload to device, run the
        inclusive scan that turns per-body partition *counts* into
        cumulative *end-indices*, write ``highest_index_in_use``,
        and return the resulting :class:`InteractionGraphData`.

        Idempotent: if already built, returns the cached view.

        Raises:
            ValueError: if the accumulated count exceeds
                ``max_interactions``.
        """
        if self._built and self._data is not None:
            return self._data
        # Sort + dedup. Sort key is ``(rigid_id, constraint_id)`` so
        # within each body the constraint IDs come out sorted -- the
        # binary-search invariant in :func:`graph_get_rigid_state_index`.
        unique = sorted(set(self._entries))
        if len(unique) > self.max_interactions:
            raise ValueError(
                f"interaction count {len(unique)} exceeds max_interactions {self.max_interactions}; "
                f"increase the capacity passed to InteractionGraph(...)"
            )
        # Build the per-body count array on host.
        import numpy as np  # noqa: PLC0415 -- intentionally lazy

        partition_list_np = np.zeros(self.max_interactions, dtype=np.int32)
        per_body_count = np.zeros(self.max_rigid_bodies, dtype=np.int32)
        highest_plus_one = 0
        for k, (rigid_id, constraint_id) in enumerate(unique):
            partition_list_np[k] = constraint_id
            per_body_count[rigid_id] += 1
            highest_plus_one = max(highest_plus_one, rigid_id + 1)
        # Inclusive scan over per-body count -> end indices.
        end_indices_np = np.cumsum(per_body_count, dtype=np.int32)
        # Allocate / upload device arrays. ``tiny_states`` is left
        # zero-initialised; the broadcast kernel will populate it
        # at the start of every substep.
        self._partition_list = wp.array(partition_list_np, dtype=wp.int32, device=self.device)
        self._tiny_states = wp.zeros(self.max_interactions, dtype=TinyRigidState, device=self.device)
        self._state_section_end_indices = wp.array(end_indices_np, dtype=wp.int32, device=self.device)
        self._highest_index_in_use = wp.array(
            np.array([highest_plus_one], dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        data = InteractionGraphData()
        data.partition_list = self._partition_list
        data.tiny_states = self._tiny_states
        data.state_section_end_indices = self._state_section_end_indices
        data.highest_index_in_use = self._highest_index_in_use
        data.max_interactions = wp.int32(self.max_interactions)
        self._data = data
        self._built = True
        return data

    @property
    def data(self) -> InteractionGraphData:
        """Device view -- requires :meth:`build` to have been called."""
        if not self._built or self._data is None:
            raise RuntimeError("InteractionGraph.build() must be called before data is accessed")
        return self._data

    @property
    def num_interactions(self) -> int:
        """Active interaction count (unique ``(rigid, constraint)``
        pairs accumulated via :meth:`add_entry`). Available pre-build."""
        return len(set(self._entries))


# ---------------------------------------------------------------------------
# Device-side accessors -- @wp.func helpers that read InteractionGraphData.
# ---------------------------------------------------------------------------
#
# These mirror the C# ``MassSplittingRigidBodyInteractionGraphGpu``
# member functions (``MassSplittingTypes.cuh``). Free functions
# rather than methods because Warp 1.13's ``@wp.struct`` doesn't
# support member functions yet.


@wp.func
def graph_state_section(
    graph: InteractionGraphData,
    rigid_body_index: wp.int32,
):
    """Return the ``[start, end)`` slice of ``tiny_states`` /
    ``partition_list`` belonging to ``rigid_body_index``.

    Mirrors the inline pattern at ``MassSplittingTypes.cuh:212-213``
    (and 253-254, 274-275, 287-288 -- it's repeated everywhere). This
    port factors it into one helper so the broadcast / average /
    write-back kernels share a single source of truth for the
    body-section indexing convention.
    """
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return wp.int32(0), wp.int32(0)
    start = wp.int32(0)
    if rigid_body_index > wp.int32(0):
        start = graph.state_section_end_indices[rigid_body_index - wp.int32(1)]
    end = graph.state_section_end_indices[rigid_body_index]
    return start, end


@wp.func
def graph_get_state(
    graph: InteractionGraphData,
    state_index: wp.int32,
) -> TinyRigidState:
    """Read one slot of ``tiny_states``.

    Negative indices are *not* checked here (they trip a
    out-of-bounds in Warp); :func:`graph_get_rigid_state_index`
    returns -1 for static bodies and the
    :func:`~newton._src.solvers.phoenx.mass_splitting.read_state.read_state`
    wrapper handles the static-body fallback before reaching this.
    """
    return graph.tiny_states[state_index]


@wp.func
def graph_set_state(
    graph: InteractionGraphData,
    state_index: wp.int32,
    new_state: TinyRigidState,
):
    """Write one slot of ``tiny_states``. ``state_index < 0`` is a
    no-op; mirrors the static-body short-circuit in
    ``MassSplittingTypes.cuh:39-62``."""
    if state_index >= wp.int32(0):
        graph.tiny_states[state_index] = new_state


@wp.func
def graph_get_rigid_state_index(
    graph: InteractionGraphData,
    constraint_index: wp.int32,
    rigid_body_index: wp.int32,
):
    """Look up the ``tiny_states`` index for ``(constraint, body)``.

    Mirrors the C# ``GetRigidStateIndex``
    (``MassSplittingTypes.cuh:166-204``). Returns
    ``(state_index, inv_factor)`` where:

    * ``state_index = -1, inv_factor = 0`` means the body is static
      (``rigid_body_index >= highest_index_in_use[0]``) or didn't
      register with this constraint -- the caller should fall back
      to the body store directly.
    * ``state_index >= 0`` is a valid index into ``tiny_states``;
      ``inv_factor`` is the number of partitions this body
      participates in (used to scale the impulse).

    The lookup is a binary search inside the body's section of
    ``partition_list``; each section is sorted by constraint ID
    (the build pipeline guarantees this).
    """
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return wp.int32(-1), wp.int32(0)
    start, end = graph_state_section(graph, rigid_body_index)
    count = end - start
    if count <= wp.int32(0):
        return wp.int32(-1), wp.int32(0)
    # Binary search ``partition_list[start..end)`` for ``constraint_index``.
    lo = start
    hi = end - wp.int32(1)
    found = wp.int32(-1)
    while lo <= hi:
        mid = (lo + hi) >> wp.int32(1)
        v = graph.partition_list[mid]
        if v < constraint_index:
            lo = mid + wp.int32(1)
        elif v > constraint_index:
            hi = mid - wp.int32(1)
        else:
            found = mid
            lo = hi + wp.int32(1)  # break
    if found < wp.int32(0):
        return wp.int32(-1), wp.int32(0)
    return found, count
