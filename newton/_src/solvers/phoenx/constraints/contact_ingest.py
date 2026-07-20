# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Convert Newton's sorted ``Contacts`` buffer into PhoenX contact columns.

Newton sorts Contacts by (shape_a, shape_b). Ingest marks only valid pair-run
boundaries, scans them once, and materializes compact contact columns directly.
Optional compound grouping retains its body-pair sort and two-stage compaction.

Graph-capture-safe: num_contact_columns is device-side; all kernels launch at
fixed sizes and gate on device counters.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    _col_write_int,
    _contact_uses_stale_anchor_start_gap,
    contact_set_contact_count,
    contact_set_contact_first,
    contact_set_count1,
    contact_set_count2,
    contact_set_friction,
    contact_set_friction_dynamic,
    contact_set_side0_counts_extra,
    contact_set_side0_slots_extra,
    contact_set_side1_counts_extra,
    contact_set_side1_slots_extra,
    contact_set_slot1,
    contact_set_slot2,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_BODY1_OFFSET,
    CONSTRAINT_BODY2_OFFSET,
    CONSTRAINT_TYPE_CONTACT,
    CONSTRAINT_TYPE_OFFSET,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_prev_normal_lambda,
    cc_get_prev_tangent1_lambda,
    cc_get_prev_tangent2_lambda,
    cc_set_normal,
    cc_set_normal_lambda,
    cc_set_start_gap,
    cc_set_tangent1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64
from newton._src.solvers.phoenx.materials import (
    MaterialData,
    resolve_friction_in_kernel,
    resolve_friction_static_in_kernel,
)

__all__ = [
    "IngestScratch",
    "allocate_ingest_scratch",
    "gather_contact_warmstart",
    "ingest_contacts",
    "stamp_forward_contact_map",
]


class IngestScratch:
    """Pre-allocated device buffers reused each step. Sized to
    rigid_contact_max (per-contact + per-pair, worst case) and
    max_contact_columns (per-column). Compound-grouping arrays are None
    when enable_body_pair_grouping is False."""

    __slots__ = (
        "_device",
        "body_pair_keys",
        "inv_sort_perm",
        "max_contact_columns",
        "num_contact_columns",
        "num_pairs",
        "pair_boundary",
        "pair_col_offset",
        "pair_columns",
        "pair_count",
        "pair_first",
        "pair_id",
        "pair_shape_a",
        "pair_shape_b",
        "pair_source_idx",
        "prev_inv_sort_perm",
        "rigid_contact_max",
        "sort_perm",
        "sorted_damping",
        "sorted_friction",
        "sorted_margin0",
        "sorted_margin1",
        "sorted_match_index",
        "sorted_normal",
        "sorted_point0",
        "sorted_point1",
        "sorted_shape0",
        "sorted_shape1",
        "sorted_stiffness",
    )

    def __init__(
        self,
        rigid_contact_max: int,
        max_contact_columns: int,
        device: wp.DeviceLike = None,
        enable_body_pair_grouping: bool = False,
    ) -> None:
        self._device = device
        self.rigid_contact_max = int(rigid_contact_max)
        self.max_contact_columns = int(max_contact_columns)
        n_pairs_max = max(1, self.rigid_contact_max)
        n_cols_max = max(1, self.max_contact_columns)

        # pair_boundary[i] = 1 iff contact i starts a new (shape_a, shape_b) run.
        self.pair_boundary = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # Inclusive scan of pair_boundary -> 1-based pair id per contact.
        self.pair_id = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # Per-pair arrays.
        self.pair_first = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_count = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_a = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_b = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        # 0/1 per pair: filtered pairs emit zero columns, others emit one.
        self.pair_columns = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_col_offset = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

        # Per-output-column arrays.
        self.pair_source_idx = wp.zeros(n_cols_max, dtype=wp.int32, device=device)

        # Device-held scalars.
        self.num_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        self.num_contact_columns = wp.zeros(1, dtype=wp.int32, device=device)

        # Compound-body grouping (opt-in): pre-sort contacts by body pair so
        # the boundary scan emits one column per body pair instead of per shape
        # pair (collapses a 2x2 compound's 4 colours to 1). int64 radix sort
        # needs 2*N ping-pong buffers.
        if enable_body_pair_grouping:
            n_sort = 2 * max(1, self.rigid_contact_max)
            n_perm = max(1, self.rigid_contact_max)
            self.body_pair_keys = wp.zeros(n_sort, dtype=wp.int64, device=device)
            self.sort_perm = wp.zeros(n_sort, dtype=wp.int32, device=device)
            self.inv_sort_perm = wp.full(n_perm, -1, dtype=wp.int32, device=device)
            #: Prev-frame inverse perm; lets warm-start translate Newton-order
            #: match indices to last frame's PhoenX-sorted index.
            self.prev_inv_sort_perm = wp.full(n_perm, -1, dtype=wp.int32, device=device)
            self.sorted_shape0 = wp.zeros(n_perm, dtype=wp.int32, device=device)
            self.sorted_shape1 = wp.zeros(n_perm, dtype=wp.int32, device=device)
            self.sorted_match_index = wp.zeros(n_perm, dtype=wp.int32, device=device)
            self.sorted_normal = wp.zeros(n_perm, dtype=wp.vec3f, device=device)
            self.sorted_point0 = wp.zeros(n_perm, dtype=wp.vec3f, device=device)
            self.sorted_point1 = wp.zeros(n_perm, dtype=wp.vec3f, device=device)
            self.sorted_margin0 = wp.zeros(n_perm, dtype=wp.float32, device=device)
            self.sorted_margin1 = wp.zeros(n_perm, dtype=wp.float32, device=device)
            self.sorted_stiffness = wp.zeros(n_perm, dtype=wp.float32, device=device)
            self.sorted_damping = wp.zeros(n_perm, dtype=wp.float32, device=device)
            self.sorted_friction = wp.zeros(n_perm, dtype=wp.float32, device=device)
        else:
            self.body_pair_keys = None
            self.sort_perm = None
            self.inv_sort_perm = None
            self.prev_inv_sort_perm = None
            self.sorted_shape0 = None
            self.sorted_shape1 = None
            self.sorted_match_index = None
            self.sorted_normal = None
            self.sorted_point0 = None
            self.sorted_point1 = None
            self.sorted_margin0 = None
            self.sorted_margin1 = None
            self.sorted_stiffness = None
            self.sorted_damping = None
            self.sorted_friction = None


def allocate_ingest_scratch(
    rigid_contact_max: int,
    max_contact_columns: int,
    device: wp.DeviceLike = None,
    enable_body_pair_grouping: bool = False,
) -> IngestScratch:
    """Factory alias matching the ``*_zeros`` style elsewhere in the solver."""
    return IngestScratch(
        rigid_contact_max,
        max_contact_columns,
        device=device,
        enable_body_pair_grouping=enable_body_pair_grouping,
    )


# Step 1: mark per-contact run boundaries directly from adjacency
# (contacts already arrive sorted by (shape_a, shape_b)).


@wp.func
def _clamp_contact_count(rigid_contact_count: wp.array[wp.int32], buffer_size: wp.int32) -> wp.int32:
    """Return ``min(rigid_contact_count[0], buffer_size)``.

    The narrow phase appends contacts via ``atomic_add`` into a fixed-size
    buffer; on overflow the counter keeps climbing past ``buffer_size`` while
    only the first ``buffer_size`` slots are actually written. Downstream
    kernels must clamp before using the count as a loop bound or index --
    otherwise inflated counts trigger OOB reads on the pair-grouping arrays
    (``pair_id[count - 1]``, ``pair_count[p]`` -> per-contact loops) and
    manifest as a CUDA illegal-memory-access crash. The narrow phase's
    ``wp.printf`` overflow warning still surfaces the underlying capacity
    problem so the user can bump ``rigid_contact_max``.
    """
    n = rigid_contact_count[0]
    if n > buffer_size:
        n = buffer_size
    return n


@wp.kernel(enable_backward=False)
def _contact_pair_boundary_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    # out
    pair_boundary: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """pair_boundary[i] = 1 iff contact i starts a new (shape_a, shape_b) run.
    Tail past rigid_contact_count[0] is zeroed."""
    tid = wp.tid()
    cid_of_contact[tid] = wp.int32(-1)
    count = _clamp_contact_count(rigid_contact_count, rigid_contact_shape0.shape[0])
    if tid >= count:
        pair_boundary[tid] = wp.int32(0)
        return
    if tid == wp.int32(0):
        pair_boundary[tid] = wp.int32(1)
        return
    prev_sa = rigid_contact_shape0[tid - 1]
    prev_sb = rigid_contact_shape1[tid - 1]
    cur_sa = rigid_contact_shape0[tid]
    cur_sb = rigid_contact_shape1[tid]
    if cur_sa != prev_sa or cur_sb != prev_sb:
        pair_boundary[tid] = wp.int32(1)
    else:
        pair_boundary[tid] = wp.int32(0)


# Compound-body grouping (opt-in). See docs/CONTACT_GROUP_COMPOUND_OPT.md.


@wp.kernel(enable_backward=False)
def _build_body_pair_keys_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    num_bodies: wp.int32,
    # out
    body_pair_keys: wp.array[wp.int64],
    sort_perm: wp.array[wp.int32],
):
    """Stamp ``body_pair_keys[k] = pack(min(b1, b2), max(b1, b2))`` and
    initialise ``sort_perm[k] = k`` (identity).

    After ``radix_sort_pairs(body_pair_keys, sort_perm, count)`` the
    ``sort_perm`` array becomes the permutation: ``sort_perm[sorted_k]``
    is the original Newton-order index that landed at position
    ``sorted_k`` after sorting by body pair.

    Tail entries past ``rigid_contact_count[0]`` get sentinel
    ``INT64_MAX`` keys so they sort to the end and the boundary scan
    can ignore them. ``sort_perm`` for tail entries is also initialised
    so the sort doesn't see undefined values.
    """
    tid = wp.tid()
    n = _clamp_contact_count(rigid_contact_count, rigid_contact_shape0.shape[0])
    if tid >= n:
        body_pair_keys[tid] = wp.int64(9223372036854775807)  # INT64_MAX
        sort_perm[tid] = tid
        return
    sa = rigid_contact_shape0[tid]
    sb = rigid_contact_shape1[tid]
    ba = shape_body[sa]
    bb = shape_body[sb]
    if ba <= bb:
        lo = ba
        hi = bb
    else:
        lo = bb
        hi = ba
    body_pair_keys[tid] = wp.int64(lo) * wp.int64(num_bodies) + wp.int64(hi)
    sort_perm[tid] = tid


@wp.kernel(enable_backward=False)
def _build_inv_sort_perm_kernel(
    rigid_contact_count: wp.array[wp.int32],
    sort_perm: wp.array[wp.int32],
    # out
    inv_sort_perm: wp.array[wp.int32],
):
    """``inv_sort_perm[sort_perm[sorted_k]] = sorted_k`` for the active
    range. Tail entries past the active count are left at their initial
    ``-1`` (the warm-start gather guards against this with
    ``prev_inv_sort_perm[prev_newton_k] >= 0`` checks indirectly via
    ``prev_cid_of_contact``).
    """
    tid = wp.tid()
    n = _clamp_contact_count(rigid_contact_count, sort_perm.shape[0])
    if tid >= n:
        return
    newton_k = sort_perm[tid]
    inv_sort_perm[newton_k] = tid


@wp.kernel(enable_backward=False)
def _gather_sorted_contacts_kernel(
    rigid_contact_count: wp.array[wp.int32],
    sort_perm: wp.array[wp.int32],
    # Newton-order narrow-phase arrays.
    src_shape0: wp.array[wp.int32],
    src_shape1: wp.array[wp.int32],
    src_match_index: wp.array[wp.int32],
    src_normal: wp.array[wp.vec3f],
    src_point0: wp.array[wp.vec3f],
    src_point1: wp.array[wp.vec3f],
    src_margin0: wp.array[wp.float32],
    src_margin1: wp.array[wp.float32],
    src_stiffness: wp.array[wp.float32],
    src_damping: wp.array[wp.float32],
    src_friction: wp.array[wp.float32],
    # ``prev_inv_sort_perm[prev_newton_k] = prev_sorted_k``. Used to
    # translate Newton-order match indices to the prev frame's
    # PhoenX-sorted index space, since ``cc.prev_*`` is keyed by the
    # prev frame's sorted-k.
    prev_inv_sort_perm: wp.array[wp.int32],
    # out (sorted-k indexed)
    sorted_shape0: wp.array[wp.int32],
    sorted_shape1: wp.array[wp.int32],
    sorted_match_index: wp.array[wp.int32],
    sorted_normal: wp.array[wp.vec3f],
    sorted_point0: wp.array[wp.vec3f],
    sorted_point1: wp.array[wp.vec3f],
    sorted_margin0: wp.array[wp.float32],
    sorted_margin1: wp.array[wp.float32],
    sorted_stiffness: wp.array[wp.float32],
    sorted_damping: wp.array[wp.float32],
    sorted_friction: wp.array[wp.float32],
):
    """Permute Newton's narrow-phase arrays into PhoenX-sorted order.

    For each ``sorted_k``, copy the slot from ``newton_k = sort_perm[sorted_k]``.
    The match index is additionally translated through
    ``prev_inv_sort_perm`` so the value stored at
    ``sorted_match_index[sorted_k]`` is the prev frame's sorted-k (or
    -1 for unmatched). Optional soft-contact arrays
    (``stiffness``/``damping``/``friction``) may be size-1 sentinel
    arrays when the user-supplied ``Contacts`` didn't allocate them;
    the gather respects that by branching on shape.
    """
    tid = wp.tid()
    n = _clamp_contact_count(rigid_contact_count, sort_perm.shape[0])
    if tid >= n:
        return
    newton_k = sort_perm[tid]
    sorted_shape0[tid] = src_shape0[newton_k]
    sorted_shape1[tid] = src_shape1[newton_k]
    # Translate match index through prev frame's inverse permutation:
    # match[newton_k] = prev_newton_k -> prev_sorted_k.
    raw_match = src_match_index[newton_k]
    if raw_match >= wp.int32(0):
        sorted_match_index[tid] = prev_inv_sort_perm[raw_match]
    else:
        sorted_match_index[tid] = raw_match
    sorted_normal[tid] = src_normal[newton_k]
    sorted_point0[tid] = src_point0[newton_k]
    sorted_point1[tid] = src_point1[newton_k]
    sorted_margin0[tid] = src_margin0[newton_k]
    sorted_margin1[tid] = src_margin1[newton_k]
    # Per-contact override arrays may be size-1 sentinel buffers; only
    # gather when the source array actually addresses this index.
    if newton_k < src_stiffness.shape[0]:
        sorted_stiffness[tid] = src_stiffness[newton_k]
    else:
        sorted_stiffness[tid] = wp.float32(0.0)
    if newton_k < src_damping.shape[0]:
        sorted_damping[tid] = src_damping[newton_k]
    else:
        sorted_damping[tid] = wp.float32(0.0)
    if newton_k < src_friction.shape[0]:
        sorted_friction[tid] = src_friction[newton_k]
    else:
        sorted_friction[tid] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _body_pair_boundary_kernel(
    rigid_contact_count: wp.array[wp.int32],
    body_pair_keys: wp.array[wp.int64],
    # out
    pair_boundary: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """Body-pair-grouping variant of :func:`_contact_pair_boundary_kernel`:
    emit a 1 wherever the (already-sorted) body-pair key changes,
    instead of comparing adjacent shape pairs.

    With contacts sorted by body pair, every body pair contributes one
    run regardless of how many shape pairs sit underneath it. The
    downstream scan + scatter / pack therefore emits one output
    contact column per body pair.
    """
    tid = wp.tid()
    cid_of_contact[tid] = wp.int32(-1)
    n = _clamp_contact_count(rigid_contact_count, body_pair_keys.shape[0])
    if tid >= n:
        pair_boundary[tid] = wp.int32(0)
        return
    if tid == wp.int32(0):
        pair_boundary[tid] = wp.int32(1)
        return
    if body_pair_keys[tid] != body_pair_keys[tid - 1]:
        pair_boundary[tid] = wp.int32(1)
    else:
        pair_boundary[tid] = wp.int32(0)


# ---------------------------------------------------------------------------
# Body-pair filter helper (shared by per-shape-pair and body-pair paths).
# ---------------------------------------------------------------------------


@wp.func
def _body_pair_filtered(
    filter_keys: wp.array[wp.int64],
    filter_count: wp.int32,
    key: wp.int64,
) -> wp.int32:
    """Binary search ``filter_keys[0:filter_count]`` for ``key``.

    Returns ``1`` if found, ``0`` otherwise.
    """
    if filter_count <= wp.int32(0):
        return wp.int32(0)
    lo = wp.int32(0)
    hi = filter_count
    while lo < hi:
        mid = (lo + hi) >> wp.int32(1)
        v = filter_keys[mid]
        if v < key:
            lo = mid + wp.int32(1)
        elif v > key:
            hi = mid
        else:
            return wp.int32(1)
    return wp.int32(0)


@wp.kernel(enable_backward=False, grid_stride=False)
def _mark_valid_shape_pair_runs_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_filter_id: wp.array[wp.int32],
    num_bodies: wp.int32,
    filter_keys: wp.array[wp.int64],
    filter_count: wp.int32,
    pair_boundary: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """Mark only shape-pair runs that will emit a contact column."""
    contact = wp.tid()
    cid_of_contact[contact] = wp.int32(-1)
    count = _clamp_contact_count(rigid_contact_count, rigid_contact_shape0.shape[0])
    if contact >= count:
        pair_boundary[contact] = wp.int32(0)
        return
    shape_a = rigid_contact_shape0[contact]
    shape_b = rigid_contact_shape1[contact]
    if contact > wp.int32(0):
        previous = contact - wp.int32(1)
        if shape_a == rigid_contact_shape0[previous] and shape_b == rigid_contact_shape1[previous]:
            pair_boundary[contact] = wp.int32(0)
            return
    body_a = shape_filter_id[shape_a]
    body_b = shape_filter_id[shape_b]
    if body_a == body_b:
        pair_boundary[contact] = wp.int32(0)
        return
    lo = wp.min(body_a, body_b)
    hi = wp.max(body_a, body_b)
    key = wp.int64(lo) * wp.int64(num_bodies) + wp.int64(hi)
    pair_boundary[contact] = wp.int32(_body_pair_filtered(filter_keys, filter_count, key) == wp.int32(0))


@wp.kernel(enable_backward=False, grid_stride=False)
def _materialize_valid_shape_pair_runs_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    pair_boundary: wp.array[wp.int32],
    pair_id: wp.array[wp.int32],
    max_columns: wp.int32,
    cid_base: wp.int32,
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    pair_source_idx: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
):
    """Materialize compact run metadata directly from the valid-boundary scan."""
    first = wp.tid()
    count = _clamp_contact_count(rigid_contact_count, rigid_contact_shape0.shape[0])
    if first == wp.int32(0):
        columns = wp.int32(0)
        if count > wp.int32(0):
            columns = wp.min(pair_id[count - wp.int32(1)], max_columns)
        num_pairs[0] = columns
        num_contact_columns[0] = columns
        num_active_constraints[0] = cid_base + columns
    if pair_boundary[first] == wp.int32(0):
        return
    column = pair_id[first] - wp.int32(1)
    if column >= max_columns:
        return
    shape_a = rigid_contact_shape0[first]
    shape_b = rigid_contact_shape1[first]
    end = first + wp.int32(1)
    while end < count and rigid_contact_shape0[end] == shape_a and rigid_contact_shape1[end] == shape_b:
        end += wp.int32(1)
    pair_first[column] = first
    pair_count[column] = end - first
    pair_shape_a[column] = shape_a
    pair_shape_b[column] = shape_b
    pair_source_idx[column] = column
    cid = cid_base + column
    for contact in range(first, end):
        cid_of_contact[contact] = cid


# ---------------------------------------------------------------------------
# Step 2: scatter per-pair metadata from run-boundary positions.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False, grid_stride=False)
def _scatter_pair_starts_kernel(
    rigid_contact_count: wp.array[wp.int32],
    pair_boundary: wp.array[wp.int32],
    pair_id: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    # out
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
):
    """At every boundary position ``i``, scatter ``(shape_a, shape_b, i)``
    into the unique slot ``pair_id[i] - 1``.

    Thread ``tid = 0`` also writes ``num_pairs[0]`` from the final
    inclusive-scan value. Pairs past ``num_pairs[0]`` get
    ``pair_columns = 0`` so the downstream scan is stable.
    """
    tid = wp.tid()
    count = _clamp_contact_count(rigid_contact_count, pair_id.shape[0])

    # Publish num_pairs via thread 0 before any reads/writes that depend
    # on it. The inclusive scan's last active value is the run count.
    if tid == wp.int32(0):
        if count > wp.int32(0):
            num_pairs[0] = pair_id[count - 1]
        else:
            num_pairs[0] = wp.int32(0)

    # Scatter per-pair metadata at run boundaries. Each boundary has a
    # unique ``pair_id`` so two threads never target the same output
    # slot -- fully deterministic, no atomics.
    if tid < count and pair_boundary[tid] == wp.int32(1):
        p = pair_id[tid] - wp.int32(1)
        pair_shape_a[p] = rigid_contact_shape0[tid]
        pair_shape_b[p] = rigid_contact_shape1[tid]
        pair_first[p] = tid

    # Tail-clear ``pair_columns`` past the live prefix. Launch dim is
    # ``rigid_contact_max`` (same as the boundary kernel) so covering
    # the tail is free.
    if tid >= count:
        pair_columns[tid] = wp.int32(0)


@wp.kernel(enable_backward=False, grid_stride=False)
def _pair_counts_and_columns_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_max: wp.int32,
    num_pairs: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    shape_filter_id: wp.array[wp.int32],
    num_bodies: wp.int32,
    filter_keys: wp.array[wp.int64],
    filter_count: wp.int32,
    # out
    pair_count: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
):
    """Per-pair: derive ``pair_count`` from adjacent ``pair_first`` and
    emit ``pair_columns = 1`` iff the pair is not self-contact /
    body-pair-filtered.

    ``shape_filter_id[s]`` is the per-shape "filter group" -- shapes
    sharing the same id collide as one body for the self-collision
    filter. For rigid-only scenes this is just ``model.shape_body``;
    for cloth-aware scenes the cloth-tri suffix gets unique negative
    ids so distinct cloth tris (each individually anchored to the
    world) don't collapse into a single "world body" group and lose
    cloth-vs-cloth + cloth-vs-static-rigid contacts.

    ``rigid_contact_max`` is the narrow-phase contact buffer capacity --
    used to clamp the inflated ``rigid_contact_count[0]`` on overflow so
    ``pair_count`` (which drives downstream per-contact loops) never
    exceeds the buffer size.
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        return

    # pair_count from adjacent pair_first values. Clamp the last pair's
    # upper bound against the narrow-phase buffer capacity so an
    # overflowed count doesn't inflate pair_count past the actual
    # written range.
    cur = pair_first[tid]
    if tid + wp.int32(1) < n:
        nxt = pair_first[tid + wp.int32(1)]
    else:
        nxt = _clamp_contact_count(rigid_contact_count, rigid_contact_max)
    pair_count[tid] = nxt - cur

    # pair_columns: 1 iff distinct filter groups + not body-pair-filtered.
    sa = pair_shape_a[tid]
    sb = pair_shape_b[tid]
    ba = shape_filter_id[sa]
    bb = shape_filter_id[sb]
    if ba == bb:
        pair_columns[tid] = wp.int32(0)
        return
    if ba <= bb:
        lo = ba
        hi = bb
    else:
        lo = bb
        hi = ba
    body_key = wp.int64(lo) * wp.int64(num_bodies) + wp.int64(hi)
    if _body_pair_filtered(filter_keys, filter_count, body_key) == wp.int32(1):
        pair_columns[tid] = wp.int32(0)
        return
    pair_columns[tid] = wp.int32(1)


# ---------------------------------------------------------------------------
# Step 3: total column count & per-output-column pair map.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _pair_source_idx_kernel(
    pair_col_offset: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    max_columns: wp.int32,
    active_constraint_base: wp.int32,
    # out
    pair_source_idx: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
):
    """Publish counts and map each output column to its pair."""
    tid = wp.tid()
    n = num_pairs[0]
    if tid == wp.int32(0):
        total = wp.int32(0)
        if n > wp.int32(0):
            total = pair_col_offset[n - 1] + pair_columns[n - 1]
            if total > max_columns:
                total = max_columns
            if total < wp.int32(0):
                total = wp.int32(0)
        num_contact_columns[0] = total
        num_active_constraints[0] = active_constraint_base + total

    if tid >= n:
        return
    if pair_columns[tid] == 0:
        return
    o = pair_col_offset[tid]
    if o < max_columns:
        pair_source_idx[o] = tid


# ---------------------------------------------------------------------------
# Step 4: per-column pack kernel -- writes the contact constraint header
# and the [contact_first, contact_count] range into the ConstraintContainer.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _contact_pack_columns_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_material: wp.array[wp.int32],
    materials: wp.array[MaterialData],
    num_contact_columns: wp.array[wp.int32],
    default_friction: wp.float32,
    # out
    contact_cols: ContactColumnContainer,
):
    """Materialise one contact column -- one thread per output column.

    Launched at ``dim == max_contact_columns`` and gates internally on
    ``num_contact_columns[0]`` so the launch size is captureable in a
    graph independently of the per-step column count.

    Each thread owns output column ``tid`` (local_cid). Writes the
    header + range. The contact range is ``[pair_first[p],
    pair_first[p] + pair_count[p])`` -- one column now covers an
    entire shape pair.

    Writes to :class:`ContactColumnContainer` directly -- contact
    column headers live in their own narrow storage so the
    joint-wide :class:`ConstraintContainer` can be sized to just
    ``num_joints``.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return

    p = pair_source_idx[tid]
    count = pair_count[p]
    first = pair_first[p]

    sa = pair_shape_a[p]
    sb = pair_shape_b[p]
    b1 = shape_body[sa]
    b2 = shape_body[sb]

    # Friction resolution via the per-shape materials table.
    mat_a = wp.int32(-1)
    mat_b = wp.int32(-1)
    if shape_material.shape[0] > sa:
        mat_a = shape_material[sa]
    if shape_material.shape[0] > sb:
        mat_b = shape_material[sb]
    mu_static = resolve_friction_static_in_kernel(materials, mat_a, mat_b, default_friction)
    mu_dynamic = resolve_friction_in_kernel(materials, mat_a, mat_b, default_friction)

    # Header is stored at offsets 0 / 1 / 2 (contract is
    # ``constraint_type / body1 / body2``). Dispatcher no longer reads
    # the type tag for contacts (cid-based dispatch) but the slot is
    # written anyway to preserve the shared header invariant for any
    # diagnostic / future caller that does.
    _col_write_int(contact_cols, CONSTRAINT_TYPE_OFFSET, tid, CONSTRAINT_TYPE_CONTACT)
    _col_write_int(contact_cols, CONSTRAINT_BODY1_OFFSET, tid, b1)
    _col_write_int(contact_cols, CONSTRAINT_BODY2_OFFSET, tid, b2)

    contact_set_friction(contact_cols, tid, mu_static)
    contact_set_friction_dynamic(contact_cols, tid, mu_dynamic)
    contact_set_contact_first(contact_cols, tid, first)
    contact_set_contact_count(contact_cols, tid, count)
    contact_set_slot1(contact_cols, tid, wp.int32(-1))
    contact_set_slot2(contact_cols, tid, wp.int32(-1))
    contact_set_side0_slots_extra(contact_cols, tid, wp.vec3i(-1, -1, -1))
    contact_set_side1_slots_extra(contact_cols, tid, wp.vec3i(-1, -1, -1))
    contact_set_count1(contact_cols, tid, wp.int32(1))
    contact_set_count2(contact_cols, tid, wp.int32(1))
    contact_set_side0_counts_extra(contact_cols, tid, wp.vec3i(1, 1, 1))
    contact_set_side1_counts_extra(contact_cols, tid, wp.vec3i(1, 1, 1))


# ---------------------------------------------------------------------------
# Warm-start gather kernel.
# ---------------------------------------------------------------------------


@wp.func
def _build_tangent1_from_normal(n: wp.vec3f) -> wp.vec3f:
    """Reconstruct a deterministic orthonormal tangent from a unit normal."""
    sign = wp.float32(1.0)
    if n[2] < wp.float32(0.0):
        sign = wp.float32(-1.0)
    a = wp.float32(-1.0) / (sign + n[2])
    return wp.vec3f(
        wp.float32(1.0) + sign * n[0] * n[0] * a,
        sign * n[0] * n[1] * a,
        -sign * n[0],
    )


@wp.kernel(enable_backward=False)
def _contact_warmstart_gather_kernel(
    pair_id: wp.array[wp.int32],
    pair_col_offset: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    max_contact_columns: wp.int32,
    rigid_contact_match_index: wp.array[wp.int32],
    prev_cid_of_contact: wp.array[wp.int32],
    reuse_contact_indices: wp.array[wp.int32],
    carry_impulses: wp.int32,
    bodies: BodyContainer,
    contacts: ContactViews,
    cid_base: wp.int32,
    direct_cid_map: wp.int32,
    cid_of_contact: wp.array[wp.int32],
    cc: ContactContainer,
):
    """Seed canonical contact rows from compact matched impulse history."""
    k = wp.tid()
    contact_count = _clamp_contact_count(contacts.rigid_contact_count, pair_id.shape[0])
    if k >= contact_count:
        return

    if direct_cid_map != wp.int32(0):
        if cid_of_contact[k] < cid_base:
            return
    else:
        p = pair_id[k] - wp.int32(1)
        if p < wp.int32(0) or pair_columns[p] == wp.int32(0) or pair_col_offset[p] >= max_contact_columns:
            return
        cid_of_contact[k] = cid_base + pair_col_offset[p]

    n = contacts.rigid_contact_normal[k]
    t1 = _build_tangent1_from_normal(n)
    prev_k = rigid_contact_match_index[k]
    reuse = reuse_contact_indices[0] != wp.int32(0)
    if reuse:
        prev_k = k
    prev_valid = prev_k >= wp.int32(0) and prev_cid_of_contact[prev_k] >= wp.int32(0)

    lambda_n = wp.float32(0.0)
    lambda_t1 = wp.float32(0.0)
    lambda_t2 = wp.float32(0.0)
    if prev_valid and carry_impulses != wp.int32(0):
        lambda_n = cc_get_prev_normal_lambda(cc, prev_k)
        lambda_t1 = cc_get_prev_tangent1_lambda(cc, prev_k)
        lambda_t2 = cc_get_prev_tangent2_lambda(cc, prev_k)

    uses_start_gap = _contact_uses_stale_anchor_start_gap(contacts, k)
    if uses_start_gap or reuse:
        local_p0 = contacts.rigid_contact_point0[k]
        local_p1 = contacts.rigid_contact_point1[k]
        sa = contacts.rigid_contact_shape0[k]
        sb = contacts.rigid_contact_shape1[k]
        b1 = contacts.shape_body[sa]
        b2 = contacts.shape_body[sb]
        r1 = wp.quat_rotate(bodies.orientation[b1], local_p0 - bodies.body_com[b1])
        r2 = wp.quat_rotate(bodies.orientation[b2], local_p1 - bodies.body_com[b2])
        gap = wp.dot((bodies.position[b2] + r2) - (bodies.position[b1] + r1), n) - (
            contacts.rigid_contact_margin0[k] + contacts.rigid_contact_margin1[k]
        )
        if uses_start_gap:
            cc_set_start_gap(cc, k, gap)
        if reuse and gap > wp.float32(0.0):
            lambda_n = wp.float32(0.0)
            lambda_t1 = wp.float32(0.0)
            lambda_t2 = wp.float32(0.0)

    cc_set_normal_lambda(cc, k, lambda_n)
    cc_set_tangent1_lambda(cc, k, lambda_t1)
    cc_set_tangent2_lambda(cc, k, lambda_t2)
    cc_set_normal(cc, k, n)
    cc_set_tangent1(cc, k, t1)


@wp.kernel(enable_backward=False)
def _stamp_cid_of_contact_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    # out
    cid_of_contact: wp.array[wp.int32],
):
    """For each contact ``k`` in this frame's sorted buffer, stamp ``cid``.

    The "forward" map the *next* frame consults when it swaps
    ``cid_of_contact`` into the "prev" role and looks up its own
    ``rigid_contact_match_index[k] = prev_k``. Written during ingest
    once the column layout is known. Per-pair design: no ``slot`` to
    stamp -- the next frame's gather indexes prev state directly by
    the contact index ``k``.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    p = pair_source_idx[tid]
    count = pair_count[p]
    start_contact = pair_first[p]
    cid = cid_base + tid
    for i in range(count):
        cid_of_contact[start_contact + i] = cid


# ---------------------------------------------------------------------------
# Host driver.
# ---------------------------------------------------------------------------


def ingest_contacts(
    contacts,  # newton._src.sim.contacts.Contacts
    shape_body: wp.array,
    contact_cols: ContactColumnContainer,
    scratch: IngestScratch,
    max_contact_columns: int,
    default_friction: float = 0.5,
    device: wp.DeviceLike = None,
    *,
    num_bodies: int = 0,
    filter_keys: wp.array | None = None,
    filter_count: int = 0,
    shape_material: wp.array | None = None,
    materials: wp.array | None = None,
    enable_body_pair_grouping: bool = False,
    shape_filter_id: wp.array | None = None,
    cid_of_contact: wp.array[wp.int32] | None = None,
    num_active_constraints: wp.array[wp.int32] | None = None,
    active_constraint_base: int = 0,
) -> None:
    """Materialise contact columns for one step.

    Graph-capture safe: no host readbacks, all launch sizes known at
    host time, step-varying counts kept on-device in :attr:`IngestScratch`.

    Args:
        contacts: Newton :class:`Contacts` buffer (must have been
            built with a non-disabled ``contact_matching`` mode).
        shape_body: ``model.shape_body`` array.
        contact_cols: Contact-only column storage. Writes at
            local_cid ``[0, num_contact_columns)``; dispatcher maps
            to global cid via ``cid < num_joints`` -> joint, else
            contact at ``cid - num_joints``.
        scratch: Reusable per-step scratch.
        max_contact_columns: Cap on contact columns per step (now
            sized by distinct shape pairs).
        default_friction: Fallback friction.
        device: Warp device for the launches.
        num_bodies: Total body count; used to pack
            ``(min_body, max_body)`` into int64 filter keys.
        filter_keys: Sorted int64 array of canonical body-pair keys
            to ignore.
        filter_count: Valid entries in ``filter_keys``.
        shape_material, materials: Optional per-shape material lookup.
        enable_body_pair_grouping: If ``True``, contacts are sorted by
            body-pair before pair-boundary detection so multiple
            shape-pair runs sharing one body pair collapse into a
            single contact column. Halves to quarters the graph-coloring
            color count on compound-body scenes (Kapla planks, ragdolls,
            decomposed concave hulls). Costs one int64 radix sort + one
            gather per ingest call. Requires the matching scratch
            arrays to be allocated (``IngestScratch(...,
            enable_body_pair_grouping=True)``) and a non-zero
            ``num_bodies``. See :file:`../docs/CONTACT_GROUP_COMPOUND_OPT.md`.
        cid_of_contact: Current-frame forward map cleared during the
            boundary pass. Defaults to scratch storage for standalone callers.
        num_active_constraints: Optional output for
            ``active_constraint_base + num_contact_columns``.
        active_constraint_base: Non-contact constraint count.
    """
    rigid_contact_max = int(contacts.rigid_contact_max)
    if cid_of_contact is None:
        cid_of_contact = scratch.pair_id
    if num_active_constraints is None:
        num_active_constraints = scratch.num_contact_columns
        active_constraint_base = 0

    if filter_keys is None:
        raise ValueError(
            "ingest_contacts: filter_keys must be non-None (use a size-1 "
            "sentinel array when no filters are registered; the kernel "
            "signature has no null-pointer path)."
        )

    if enable_body_pair_grouping:
        if scratch.body_pair_keys is None:
            raise RuntimeError(
                "ingest_contacts(enable_body_pair_grouping=True) needs an "
                "IngestScratch built with the same flag; got a scratch "
                "object without compound-grouping arrays."
            )
        # ---- Compound-body grouping path -----------------------------
        #
        # Sort contacts by body-pair key, gather narrow-phase data into
        # PhoenX scratch (translating match indices into prev-frame
        # sorted-k space along the way), then run the existing pipeline
        # against the sorted scratch.
        wp.launch(
            kernel=_build_body_pair_keys_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                shape_body,
                int(max(1, num_bodies)),
            ],
            outputs=[scratch.body_pair_keys, scratch.sort_perm],
            device=device,
        )
        # Sort body-pair keys (in place); ``sort_perm`` becomes the
        # permutation: ``sort_perm[sorted_k] = newton_k``.
        sort_variable_length_int64(
            scratch.body_pair_keys,
            scratch.sort_perm,
            contacts.rigid_contact_count,
        )
        # Build the inverse permutation for the warm-start gather to
        # translate prev-frame Newton-order match indices into
        # prev-frame PhoenX-sorted-k.
        wp.launch(
            kernel=_build_inv_sort_perm_kernel,
            dim=rigid_contact_max,
            inputs=[contacts.rigid_contact_count, scratch.sort_perm],
            outputs=[scratch.inv_sort_perm],
            device=device,
        )
        # Gather narrow-phase arrays into PhoenX scratch in sorted-k
        # order, applying the prev-frame inverse perm to the match
        # index so ``sorted_match_index[sorted_k]`` is the prev frame's
        # PhoenX-sorted-k (or -1).
        wp.launch(
            kernel=_gather_sorted_contacts_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                scratch.sort_perm,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_match_index,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_stiffness
                if getattr(contacts, "rigid_contact_stiffness", None) is not None
                else wp.zeros(0, dtype=wp.float32, device=device),
                contacts.rigid_contact_damping
                if getattr(contacts, "rigid_contact_damping", None) is not None
                else wp.zeros(0, dtype=wp.float32, device=device),
                contacts.rigid_contact_friction
                if getattr(contacts, "rigid_contact_friction", None) is not None
                else wp.zeros(0, dtype=wp.float32, device=device),
                scratch.prev_inv_sort_perm,
            ],
            outputs=[
                scratch.sorted_shape0,
                scratch.sorted_shape1,
                scratch.sorted_match_index,
                scratch.sorted_normal,
                scratch.sorted_point0,
                scratch.sorted_point1,
                scratch.sorted_margin0,
                scratch.sorted_margin1,
                scratch.sorted_stiffness,
                scratch.sorted_damping,
                scratch.sorted_friction,
            ],
            device=device,
        )
        # The downstream pipeline uses these sorted arrays in place of
        # Newton's narrow-phase arrays. ``shape0`` / ``shape1`` come
        # from the sorted scratch so material lookups land at the
        # *first* shape pair in each merged column (acceptable for
        # uniform-material compounds; see docs/CONTACT_GROUP_COMPOUND_OPT.md
        # Section 5).
        ingest_shape0 = scratch.sorted_shape0
        ingest_shape1 = scratch.sorted_shape1
    else:
        ingest_shape0 = contacts.rigid_contact_shape0
        ingest_shape1 = contacts.rigid_contact_shape1

    filter_id_arr = shape_filter_id if shape_filter_id is not None else shape_body
    if not enable_body_pair_grouping:
        wp.launch(
            kernel=_mark_valid_shape_pair_runs_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                filter_id_arr,
                int(num_bodies),
                filter_keys,
                int(filter_count),
            ],
            outputs=[scratch.pair_boundary, cid_of_contact],
            device=device,
        )
        wp.utils.array_scan(scratch.pair_boundary, scratch.pair_id, inclusive=True)
        wp.launch(
            kernel=_materialize_valid_shape_pair_runs_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                scratch.pair_boundary,
                scratch.pair_id,
                int(max_contact_columns),
                int(active_constraint_base),
            ],
            outputs=[
                scratch.pair_first,
                scratch.pair_count,
                scratch.pair_shape_a,
                scratch.pair_shape_b,
                scratch.pair_source_idx,
                cid_of_contact,
                scratch.num_pairs,
                scratch.num_contact_columns,
                num_active_constraints,
            ],
            device=device,
        )
    else:
        wp.launch(
            kernel=_body_pair_boundary_kernel,
            dim=rigid_contact_max,
            inputs=[contacts.rigid_contact_count, scratch.body_pair_keys],
            outputs=[scratch.pair_boundary, cid_of_contact],
            device=device,
        )

        wp.utils.array_scan(scratch.pair_boundary, scratch.pair_id, inclusive=True)
        wp.launch(
            kernel=_scatter_pair_starts_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                scratch.pair_boundary,
                scratch.pair_id,
                ingest_shape0,
                ingest_shape1,
            ],
            outputs=[
                scratch.pair_shape_a,
                scratch.pair_shape_b,
                scratch.pair_first,
                scratch.pair_columns,
                scratch.num_pairs,
            ],
            device=device,
        )
        wp.launch(
            kernel=_pair_counts_and_columns_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                wp.int32(rigid_contact_max),
                scratch.num_pairs,
                scratch.pair_first,
                scratch.pair_shape_a,
                scratch.pair_shape_b,
                filter_id_arr,
                int(num_bodies),
                filter_keys,
                int(filter_count),
            ],
            outputs=[scratch.pair_count, scratch.pair_columns],
            device=device,
        )
        wp.utils.array_scan(scratch.pair_columns, scratch.pair_col_offset, inclusive=False)
        wp.launch(
            kernel=_pair_source_idx_kernel,
            dim=scratch.pair_columns.shape[0],
            inputs=[
                scratch.pair_col_offset,
                scratch.pair_columns,
                scratch.num_pairs,
                int(max_contact_columns),
                int(active_constraint_base),
            ],
            outputs=[scratch.pair_source_idx, scratch.num_contact_columns, num_active_constraints],
            device=device,
        )

    # Step 5: write the contact column headers + ranges.
    if shape_material is None:
        shape_material = wp.array([-1], dtype=wp.int32, device=device)
    if materials is None:
        materials = wp.zeros(0, dtype=MaterialData, device=device)

    wp.launch(
        kernel=_contact_pack_columns_kernel,
        dim=max(1, max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_first,
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            shape_body,
            shape_material,
            materials,
            scratch.num_contact_columns,
            float(default_friction),
        ],
        outputs=[contact_cols],
        device=device,
    )


def stamp_forward_contact_map(
    cid_base: int,
    scratch: IngestScratch,
    cid_of_contact: wp.array,
    device: wp.DeviceLike = None,
) -> None:
    """Stamp this frame's contact-index -> cid lookup.

    :func:`ingest_contacts` clears the map in its boundary pass; this
    pass only writes active contact ranges for next frame's warm start.
    """
    wp.launch(
        kernel=_stamp_cid_of_contact_kernel,
        dim=max(1, scratch.max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_first,
            scratch.pair_count,
            scratch.num_contact_columns,
            int(cid_base),
        ],
        outputs=[cid_of_contact],
        device=device,
    )


def gather_contact_warmstart(
    scratch: IngestScratch,
    rigid_contact_match_index: wp.array,
    prev_cid_of_contact: wp.array,
    reuse_contact_indices: wp.array,
    bodies: BodyContainer,
    contacts: ContactViews,
    cid_base: int,
    cid_of_contact: wp.array,
    cc: ContactContainer,
    device: wp.DeviceLike = None,
    *,
    carry_impulses: bool = True,
    direct_shape_pair_runs: bool = False,
) -> None:
    """Copy prev-frame state into ``cc`` for matched contacts; initialise
    PhoenX-style for unmatched contacts.

    Launched contact-major so canonical SOA state is accessed coalesced and
    stamps the accepted contact-to-column map without a second pass.
    Called after contact history is copied into ``cc.prev_*`` but before
    :func:`contact_prepare_for_iteration_at`. Set ``carry_impulses=False`` to
    reuse matched contact geometry while cold-starting cross-frame impulse
    magnitudes.
    """
    device_obj = wp.get_device(device)
    max_blocks = max(1, int(getattr(device_obj, "sm_count", 1))) * 4
    wp.launch(
        kernel=_contact_warmstart_gather_kernel,
        dim=max(1, scratch.rigid_contact_max),
        inputs=[
            scratch.pair_id,
            scratch.pair_col_offset,
            scratch.pair_columns,
            wp.int32(scratch.max_contact_columns),
            rigid_contact_match_index,
            prev_cid_of_contact,
            reuse_contact_indices,
            wp.int32(1 if carry_impulses else 0),
            bodies,
            contacts,
            wp.int32(cid_base),
            wp.int32(1 if direct_shape_pair_runs else 0),
        ],
        outputs=[cid_of_contact, cc],
        device=device,
        max_blocks=max_blocks,
    )
