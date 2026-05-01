# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Convert Newton's sorted ``Contacts`` buffer into PhoenX contact columns.

Called once per :meth:`PhoenXWorld.step`. Newton's
:class:`~newton._src.sim.contacts.Contacts` is already sorted by
``(shape_a, shape_b)`` (contacts for one pair are contiguous), so
ingest detects pair boundaries by comparing adjacent
``(shape0, shape1)`` pairs, inclusive-scans the boundary marks into a
1-based run id per contact, and scatters per-pair metadata at those
boundaries to emit one :data:`CONSTRAINT_TYPE_CONTACT` column per
non-filtered shape pair with the pair's
``[contact_first, contact_first + contact_count)`` stamped into the
header.

Fully graph-capture-safe: ``num_contact_columns`` is a device-side
scalar, never host-read during the step; all kernels launch at fixed
sizes (``rigid_contact_max`` / ``max_contact_columns``) and gate
internally on device-held counters. The adjacency-mark design has no
int32 shape-count limit (an earlier ``sa * num_shapes + sb`` packing
silently wrapped above ~46 340 shapes — see Bug.md #1).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    _col_write_int,
    contact_set_contact_count,
    contact_set_contact_first,
    contact_set_friction,
    contact_set_friction_dynamic,
    contact_set_rigid_rigid_endpoints,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CONTACT,
    CONSTRAINT_TYPE_OFFSET,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_prev_local_p0,
    cc_get_prev_local_p1,
    cc_get_prev_normal,
    cc_get_prev_normal_lambda,
    cc_get_prev_tangent1,
    cc_get_prev_tangent1_lambda,
    cc_get_prev_tangent2_lambda,
    cc_set_local_p0,
    cc_set_local_p1,
    cc_set_normal,
    cc_set_normal_lambda,
    cc_set_tangent1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64
from newton._src.solvers.phoenx.materials import (
    MaterialData,
    resolve_friction_in_kernel,
    resolve_friction_static_in_kernel,
    resolve_softness_in_kernel,
)

__all__ = [
    "IngestScratch",
    "allocate_ingest_scratch",
    "gather_contact_warmstart",
    "ingest_contacts",
    "stamp_forward_contact_map",
]


# ---------------------------------------------------------------------------
# Host-side scratch -- reused across steps to avoid per-step allocs.
# ---------------------------------------------------------------------------


class IngestScratch:
    """Pre-allocated device buffers the ingest pipeline reuses each step.

    Sized to the upstream ``Contacts.rigid_contact_max`` (per-contact
    arrays) and to ``max_contact_columns`` (per-column arrays).
    Per-pair arrays are sized to ``rigid_contact_max`` too because in
    the worst case (every contact is its own shape pair)
    ``num_pairs == rigid_contact_max``.

    All scratch lives on ``device`` and is never resized after
    construction -- the shapes drive the launch sizes of every ingest
    kernel, which is what keeps the sequence graph-capture safe.
    """

    # Naturally sorted (RUF023). Compound-grouping scratch arrays
    # (``body_pair_keys``, ``inv_sort_perm``, ``prev_inv_sort_perm``,
    # ``sort_perm``, ``sorted_*``) are only allocated when the solver
    # opts in via ``enable_body_pair_grouping``; ``None`` otherwise so
    # memory cost is zero in the common single-shape case.
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

        # Per-contact run-start marker: ``pair_boundary[i] = 1`` iff
        # contact ``i`` starts a new ``(shape_a, shape_b)`` run (contact
        # 0 is always a boundary, tail past ``rigid_contact_count[0]``
        # is zeroed). Replaces a packed ``int32`` key array that
        # overflowed above ~46_340 shapes (see Bug.md #1).
        self.pair_boundary = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # Inclusive scan of ``pair_boundary``: ``pair_id[i]`` is the
        # 1-based run id that contact ``i`` belongs to. Used to scatter
        # each pair's metadata into its unique slot.
        self.pair_id = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # Per-pair arrays.
        self.pair_first = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_count = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_a = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_b = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        # 0 or 1 per pair in the new per-pair design -- the column-count
        # is binary (filtered pairs emit zero columns, everyone else
        # emits exactly one). Kept as ``int32`` so the same
        # :func:`wp.utils.array_scan` step that the old ceil-based code
        # used still applies.
        self.pair_columns = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_col_offset = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

        # Per-output-column arrays.
        self.pair_source_idx = wp.zeros(n_cols_max, dtype=wp.int32, device=device)

        # Device-held scalars.
        self.num_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        self.num_contact_columns = wp.zeros(1, dtype=wp.int32, device=device)

        # ---- Compound-body grouping scratch ---------------------------
        #
        # When ``enable_body_pair_grouping=True`` we precede the per-
        # shape-pair pipeline with a body-pair sort step: contacts that
        # share a ``(min(b1, b2), max(b1, b2))`` body pair group together
        # in the sorted order, so the pair-boundary scan emits one
        # column per *body pair* instead of per *shape pair*. That
        # collapses the 4-color forced spread of a 2x2 compound (A:{S1,S2}
        # touching B:{T1,T2}) down to 1 color. Cost: one ``int64``
        # radix sort + one gather kernel per ingest call.
        #
        # ``radix_sort_pairs`` requires ping-pong buffers at ``2 * N``;
        # the helper's ``count = keys.shape[0] // 2`` convention expects
        # the same. Sized to ``2 * rigid_contact_max`` accordingly.
        if enable_body_pair_grouping:
            n_sort = 2 * max(1, self.rigid_contact_max)
            n_perm = max(1, self.rigid_contact_max)
            self.body_pair_keys = wp.zeros(n_sort, dtype=wp.int64, device=device)
            self.sort_perm = wp.zeros(n_sort, dtype=wp.int32, device=device)
            self.inv_sort_perm = wp.full(n_perm, -1, dtype=wp.int32, device=device)
            #: Previous frame's inverse permutation, swapped each step.
            #: The warm-start gather translates Newton-order match
            #: indices through this so ``cc.prev_*`` lookups land at the
            #: prev frame's PhoenX-sorted index (which is what ``cc.*``
            #: was keyed by last frame).
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


# ---------------------------------------------------------------------------
# Step 1: mark per-contact run boundaries.
#
# Contacts arrive sorted by ``(shape_a, shape_b)`` from the narrow-phase
# pipeline (this was the whole reason the pre-Bug-#1 code RLE'd a packed
# key: the RLE only works if same-pair contacts are adjacent). So we can
# detect run boundaries directly from adjacency -- no key packing, no
# int32 overflow, cleanly scales to any shape count representable in
# int32.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _contact_pair_boundary_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    # out
    pair_boundary: wp.array[wp.int32],
):
    """Write ``pair_boundary[i] = 1`` iff contact ``i`` starts a new
    ``(shape_a, shape_b)`` run.

    Boundary rule: ``i == 0``, or ``(shape0[i], shape1[i])`` differs
    from ``(shape0[i - 1], shape1[i - 1])``. Tail entries past
    ``rigid_contact_count[0]`` are set to 0 so the downstream inclusive
    scan produces a stable result regardless of stale previous-frame
    data. Replaces the old ``sa * num_shapes + sb`` int32 key packing
    that silently wrapped at ~46_340 shapes (see Bug.md #1).
    """
    tid = wp.tid()
    count = rigid_contact_count[0]
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


# ---------------------------------------------------------------------------
# Compound-body grouping (opt-in): sort contacts by body pair before
# the boundary scan so multiple shape-pair runs sharing one body pair
# collapse into a single contact column. See
# :file:`CONTACT_GROUP_COMPOUND_OPT.md` for the design rationale.
# ---------------------------------------------------------------------------


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
    n = rigid_contact_count[0]
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
    n = rigid_contact_count[0]
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
    n = rigid_contact_count[0]
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
    n = rigid_contact_count[0]
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


# ---------------------------------------------------------------------------
# Step 2: scatter per-pair metadata from run-boundary positions.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
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
    count = rigid_contact_count[0]

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


@wp.kernel(enable_backward=False)
def _pair_counts_and_columns_kernel(
    rigid_contact_count: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
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

    Fuses the old ``_pair_counts_from_starts`` + ``_pair_columns_binary``
    kernels: they both launched at ``dim=rigid_contact_max``, both
    gated on ``tid < num_pairs[0]``, and their outputs had no
    cross-dependency (the binary kernel's ``pair_count > 0`` check
    used to guard zero-run pairs, but the adjacency-mark pipeline
    never produces those so the check was defensive-only and is
    dropped here).
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        return

    # pair_count from adjacent pair_first values.
    cur = pair_first[tid]
    if tid + wp.int32(1) < n:
        nxt = pair_first[tid + wp.int32(1)]
    else:
        nxt = rigid_contact_count[0]
    pair_count[tid] = nxt - cur

    # pair_columns: 1 iff distinct dynamic bodies + not body-pair-filtered.
    sa = pair_shape_a[tid]
    sb = pair_shape_b[tid]
    ba = shape_body[sa]
    bb = shape_body[sb]
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
def _total_contact_columns_kernel(
    pair_col_offset: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    max_columns: wp.int32,
    # out
    num_contact_columns: wp.array[wp.int32],
):
    """Derive ``num_contact_columns = sum(pair_columns)`` on-device."""
    tid = wp.tid()
    if tid != 0:
        return
    n = num_pairs[0]
    if n <= 0:
        num_contact_columns[0] = wp.int32(0)
        return
    total = pair_col_offset[n - 1] + pair_columns[n - 1]
    if total > max_columns:
        total = max_columns
    if total < wp.int32(0):
        total = wp.int32(0)
    num_contact_columns[0] = total


@wp.kernel(enable_backward=False)
def _pair_source_idx_kernel(
    pair_col_offset: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    max_columns: wp.int32,
    # out
    pair_source_idx: wp.array[wp.int32],
):
    """Write ``pair_source_idx[o] = p`` for every output column ``o``.

    One thread per pair ``p``. With the per-pair design,
    ``pair_columns[p]`` is either 0 (filtered) or 1, so each thread
    makes at most one write to ``pair_source_idx[pair_col_offset[p]]``.
    ``pair_col_offset`` is an exclusive prefix of ``pair_columns``, so
    the destination slot is unique to that pair -- two threads never
    race on the same output index, which makes this fully
    deterministic (no atomics, no cross-thread data dependency).

    Launched with ``dim=num_pairs_upper_bound`` (host-side cap). Threads
    past the live ``num_pairs[0]`` early-return via the bound check.
    Originally a single-thread kernel; profiling H1 @ 256 worlds
    showed 247 us on one thread -- parallel over ~10k pairs drops
    this to a rounding error.
    """
    tid = wp.tid()
    if tid >= num_pairs[0]:
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
    # in / out: per-contact softness arrays. Hydroelastic narrow
    # phases write absolute K/D (positive); this kernel encodes
    # Material-derived softness as (-hertz, -damping_ratio) into the
    # same slots when the narrow phase did not publish anything (i.e.,
    # when ``rigid_contact_stiffness[k] == 0``). Empty arrays
    # (length 0) opt out -- non-soft-contact scenes ignore both paths.
    rigid_contact_stiffness: wp.array[wp.float32],
    rigid_contact_damping: wp.array[wp.float32],
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
    ``num_joints``. Per-contact Material softness is sign-encoded into
    the contact stiffness / damping arrays in a per-pair fan-out loop;
    the prepare phase decodes the negative gate.
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

    # Friction + softness resolution via the per-shape materials table.
    mat_a = wp.int32(-1)
    mat_b = wp.int32(-1)
    if shape_material.shape[0] > sa:
        mat_a = shape_material[sa]
    if shape_material.shape[0] > sb:
        mat_b = shape_material[sb]
    mu_static = resolve_friction_static_in_kernel(materials, mat_a, mat_b, default_friction)
    mu_dynamic = resolve_friction_in_kernel(materials, mat_a, mat_b, default_friction)
    # ``softness`` packs (hertz, damping_ratio) blended under the
    # materials' ``softness_combine_mode``. For pre-material scenes
    # (no materials registered) this returns ``(0, 1)`` which keeps
    # every column on the legacy Box2D-rigid normal row -- bit-for-bit
    # unchanged.
    softness = resolve_softness_in_kernel(materials, mat_a, mat_b)

    # Header (constraint_type at dword 0). Dispatcher no longer reads
    # the type tag for contacts (cid-based dispatch) but the slot is
    # written anyway to preserve the shared header invariant for any
    # diagnostic / future caller that does.
    _col_write_int(contact_cols, CONSTRAINT_TYPE_OFFSET, tid, CONSTRAINT_TYPE_CONTACT)

    # Endpoint slots + subtype tag + barycentric weights. Rigid-rigid
    # is the only subtype the per-shape-pair narrow-phase ingest
    # produces today; triangle subtypes are emitted by the cloth-
    # narrow-phase work tracked separately. The unified iterate's
    # triangle-fetch helper still reads ``weights_a`` / ``weights_b``
    # on this column, so neutral ``(1, 0, 0)`` defaults are stamped
    # alongside the body indices.
    contact_set_rigid_rigid_endpoints(contact_cols, tid, b1, b2)

    contact_set_friction(contact_cols, tid, mu_static)
    contact_set_friction_dynamic(contact_cols, tid, mu_dynamic)
    contact_set_contact_first(contact_cols, tid, first)
    contact_set_contact_count(contact_cols, tid, count)

    # ---- Material softness sign-encoding -------------------------
    # When the resolved softness has a non-zero hertz, write
    # ``(-hertz, -damping_ratio)`` into every per-contact slot in
    # the pair whose stiffness is currently 0 (i.e., the narrow
    # phase did not publish hydroelastic absolute K). Empty arrays
    # (length 0) skip the loop entirely so soft-contact-disabled
    # scenes pay nothing.
    hertz = softness[0]
    damp_ratio = softness[1]
    if hertz > wp.float32(0.0) and rigid_contact_stiffness.shape[0] > 0:
        encoded_k = -hertz
        encoded_c = -damp_ratio
        for i in range(count):
            slot = first + i
            if slot < rigid_contact_stiffness.shape[0]:
                if rigid_contact_stiffness[slot] == wp.float32(0.0):
                    rigid_contact_stiffness[slot] = encoded_k
                    if slot < rigid_contact_damping.shape[0]:
                        rigid_contact_damping[slot] = encoded_c


# ---------------------------------------------------------------------------
# Warm-start gather kernel.
# ---------------------------------------------------------------------------


@wp.func
def _build_tangent1_from_velocity(n: wp.vec3f, dv: wp.vec3f) -> wp.vec3f:
    """Build a unit tangent1 from the tangential relative velocity.

    PhoenX's ``RRContactManifoldFunctions::Initialize`` tangent frame:
    project the contact-point relative velocity onto the contact plane
    and normalise. When there's no meaningful tangential slide, fall
    back to a branch-free orthonormal seed (Duff et al. 2017) so the
    basis still varies smoothly with ``n``.
    """
    v_n = wp.dot(dv, n) * n
    v_t = dv - v_n
    len_sq = wp.dot(v_t, v_t)
    if len_sq > wp.float32(1.0e-12):
        return v_t / wp.sqrt(len_sq)
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
    pair_source_idx: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    rigid_contact_match_index: wp.array[wp.int32],
    prev_cid_of_contact: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    bodies: BodyContainer,
    contacts: ContactViews,
    cc: ContactContainer,
):
    """Seed this frame's ``cc`` slots from the prev frame (PhoenX model).

    One thread per output column. For each contact ``k`` in the range:

    * Matched (``prev_k = rigid_contact_match_index[k] >= 0``):
      copy ``lambdas, normal, tangent1, local_p0, local_p1`` from
      ``cc.prev_*[prev_k]``. Fixed-frame warm-start preserves impulse
      meaning since the ``(n, t1, t2)`` basis is identical.
    * Unmatched (``< 0``): run PhoenX ``Initialize`` -- pull
      ``normal`` and body-local anchors from the upstream buffer,
      derive ``tangent1`` from tangential relative velocity, zero
      impulses.

    ``prev_cid_of_contact`` gates reads of stale prev slots that
    belonged to an already-overwritten frame.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return

    p = pair_source_idx[tid]
    count = pair_count[p]
    start_contact = pair_first[p]

    sa = pair_shape_a[p]
    sb = pair_shape_b[p]
    b1 = contacts.shape_body[sa]
    b2 = contacts.shape_body[sb]

    for i in range(count):
        k = start_contact + i

        prev_k = rigid_contact_match_index[k]
        prev_valid = wp.int32(0)
        if prev_k >= 0:
            if prev_cid_of_contact[prev_k] >= 0:
                prev_valid = wp.int32(1)

        fresh_local_p0 = contacts.rigid_contact_point0[k]
        fresh_local_p1 = contacts.rigid_contact_point1[k]
        fresh_n = contacts.rigid_contact_normal[k]

        # The narrow phase gives anchors in each body's *origin* frame
        # but :attr:`BodyContainer.position` is the body *COM*; subtract
        # ``body_com`` when building the world-space lever arm so
        # asymmetric meshes (bunny, nut) don't appear shifted by
        # ``|body_com|`` relative to where the narrow phase saw them.
        body_com1 = bodies.body_com[b1]
        body_com2 = bodies.body_com[b2]
        fresh_r1 = wp.quat_rotate(bodies.orientation[b1], fresh_local_p0 - body_com1)
        fresh_r2 = wp.quat_rotate(bodies.orientation[b2], fresh_local_p1 - body_com2)

        if prev_valid == wp.int32(1):
            prev_n = cc_get_prev_normal(cc, prev_k)
            prev_lp0 = cc_get_prev_local_p0(cc, prev_k)
            prev_lp1 = cc_get_prev_local_p1(cc, prev_k)
            prev_r1 = wp.quat_rotate(bodies.orientation[b1], prev_lp0 - body_com1)
            prev_r2 = wp.quat_rotate(bodies.orientation[b2], prev_lp1 - body_com2)
            prev_p1_world = bodies.position[b1] + prev_r1
            prev_p2_world = bodies.position[b2] + prev_r2
            prev_penetration = -wp.dot(prev_p2_world - prev_p1_world, prev_n)

            fresh_p1_world = bodies.position[b1] + fresh_r1
            fresh_p2_world = bodies.position[b2] + fresh_r2
            fresh_penetration = -wp.dot(fresh_p2_world - fresh_p1_world, fresh_n)

            if fresh_penetration > prev_penetration:
                # Prev frame has grown stale -- overwrite anchors /
                # normal / tangent but carry impulses forward.
                dv = (bodies.velocity[b2] + wp.cross(bodies.angular_velocity[b2], fresh_r2)) - (
                    bodies.velocity[b1] + wp.cross(bodies.angular_velocity[b1], fresh_r1)
                )
                fresh_t1 = _build_tangent1_from_velocity(fresh_n, dv)

                cc_set_normal_lambda(cc, k, cc_get_prev_normal_lambda(cc, prev_k))
                cc_set_tangent1_lambda(cc, k, cc_get_prev_tangent1_lambda(cc, prev_k))
                cc_set_tangent2_lambda(cc, k, cc_get_prev_tangent2_lambda(cc, prev_k))
                cc_set_normal(cc, k, fresh_n)
                cc_set_tangent1(cc, k, fresh_t1)
                cc_set_local_p0(cc, k, fresh_local_p0)
                cc_set_local_p1(cc, k, fresh_local_p1)
                continue

            # Prev frame still describes the contact accurately --
            # carry the full PhoenX state forward.
            cc_set_normal_lambda(cc, k, cc_get_prev_normal_lambda(cc, prev_k))
            cc_set_tangent1_lambda(cc, k, cc_get_prev_tangent1_lambda(cc, prev_k))
            cc_set_tangent2_lambda(cc, k, cc_get_prev_tangent2_lambda(cc, prev_k))
            cc_set_normal(cc, k, prev_n)
            cc_set_tangent1(cc, k, cc_get_prev_tangent1(cc, prev_k))
            cc_set_local_p0(cc, k, prev_lp0)
            cc_set_local_p1(cc, k, prev_lp1)
            continue

        # New contact -- PhoenX ``Initialize``.
        dv = (bodies.velocity[b2] + wp.cross(bodies.angular_velocity[b2], fresh_r2)) - (
            bodies.velocity[b1] + wp.cross(bodies.angular_velocity[b1], fresh_r1)
        )
        t1 = _build_tangent1_from_velocity(fresh_n, dv)

        cc_set_normal_lambda(cc, k, wp.float32(0.0))
        cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
        cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
        cc_set_normal(cc, k, fresh_n)
        cc_set_tangent1(cc, k, t1)
        cc_set_local_p0(cc, k, fresh_local_p0)
        cc_set_local_p1(cc, k, fresh_local_p1)


@wp.kernel(enable_backward=False)
def _reset_forward_map_kernel(
    # out
    cid_of_contact: wp.array[wp.int32],
):
    """Clear the forward map to ``-1`` before every stamp pass."""
    tid = wp.tid()
    cid_of_contact[tid] = wp.int32(-1)


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
            ``num_bodies``. See :file:`CONTACT_GROUP_COMPOUND_OPT.md`.
    """
    rigid_contact_max = int(contacts.rigid_contact_max)

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
        # uniform-material compounds; see CONTACT_GROUP_COMPOUND_OPT.md
        # Section 5).
        ingest_shape0 = scratch.sorted_shape0
        ingest_shape1 = scratch.sorted_shape1
    else:
        ingest_shape0 = contacts.rigid_contact_shape0
        ingest_shape1 = contacts.rigid_contact_shape1

    # Step 1: mark contacts where a new pair (shape-pair or body-pair,
    # depending on mode) run starts.
    if enable_body_pair_grouping:
        wp.launch(
            kernel=_body_pair_boundary_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                scratch.body_pair_keys,
            ],
            outputs=[scratch.pair_boundary],
            device=device,
        )
    else:
        wp.launch(
            kernel=_contact_pair_boundary_kernel,
            dim=rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
            ],
            outputs=[scratch.pair_boundary],
            device=device,
        )

    # Step 2: inclusive scan of the boundary marks -> 1-based run id
    # per contact. ``pair_id[count - 1]`` is the total run count.
    wp.utils.array_scan(scratch.pair_boundary, scratch.pair_id, inclusive=True)

    # Step 3a: scatter per-pair (shape_a, shape_b, pair_first) into
    # their unique slots, publish num_pairs, clear pair_columns tail.
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

    # Step 3b: derive per-pair contact counts from adjacent pair_first
    # AND compute the 0/1 pair_columns flag in one pass. Fused from two
    # back-to-back per-pair kernels that had identical launch geometry.
    wp.launch(
        kernel=_pair_counts_and_columns_kernel,
        dim=rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            scratch.num_pairs,
            scratch.pair_first,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            shape_body,
            int(num_bodies),
            filter_keys,
            int(filter_count),
        ],
        outputs=[scratch.pair_count, scratch.pair_columns],
        device=device,
    )

    # Step 3d: exclusive scan of pair_columns -> pair_col_offset.
    wp.utils.array_scan(scratch.pair_columns, scratch.pair_col_offset, inclusive=False)

    # Step 3e: reduce to the total column count on-device.
    wp.launch(
        kernel=_total_contact_columns_kernel,
        dim=1,
        inputs=[
            scratch.pair_col_offset,
            scratch.pair_columns,
            scratch.num_pairs,
            int(max_contact_columns),
        ],
        outputs=[scratch.num_contact_columns],
        device=device,
    )

    # Step 4: build the per-column -> pair map. One thread per
    # candidate pair; each does at most one conditional write to a
    # unique slot (pair_col_offset is a binary exclusive-prefix so
    # no two pairs share the same offset). Fully deterministic --
    # no atomics, no cross-thread data dependency.
    wp.launch(
        kernel=_pair_source_idx_kernel,
        dim=scratch.pair_columns.shape[0],
        inputs=[
            scratch.pair_col_offset,
            scratch.pair_columns,
            scratch.num_pairs,
            int(max_contact_columns),
        ],
        outputs=[scratch.pair_source_idx],
        device=device,
    )

    # Step 5: write the contact column headers + ranges + per-contact
    # softness encoding. The pack kernel writes ``-hertz`` /
    # ``-damping_ratio`` into the contact stiffness / damping arrays
    # for shape pairs whose materials carry softness; hydroelastic
    # narrow phases publish positive absolute K/D into the same slots
    # and that wins on a per-contact basis. The arrays the prepare
    # phase will later read depend on whether body-pair grouping is
    # on, so we route the encode writes to the matching scratch /
    # upstream buffer here.
    if shape_material is None:
        shape_material = wp.array([-1], dtype=wp.int32, device=device)
    if materials is None:
        materials = wp.zeros(0, dtype=MaterialData, device=device)
    if enable_body_pair_grouping:
        # Sorted scratch is what ContactViews binds to in this mode.
        encode_stiffness = scratch.sorted_stiffness
        encode_damping = scratch.sorted_damping
    else:
        encode_stiffness = (
            contacts.rigid_contact_stiffness
            if getattr(contacts, "rigid_contact_stiffness", None) is not None
            else wp.zeros(0, dtype=wp.float32, device=device)
        )
        encode_damping = (
            contacts.rigid_contact_damping
            if getattr(contacts, "rigid_contact_damping", None) is not None
            else wp.zeros(0, dtype=wp.float32, device=device)
        )

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
            encode_stiffness,
            encode_damping,
        ],
        outputs=[contact_cols],
        device=device,
    )


def stamp_forward_contact_map(
    rigid_contact_max: int,
    cid_base: int,
    scratch: IngestScratch,
    cid_of_contact: wp.array,
    device: wp.DeviceLike = None,
) -> None:
    """Fill the forward (sorted-index -> cid) lookup.

    Called after :func:`ingest_contacts`; the output array is consumed
    by the *next* step's warm-start gather as the "prev" side of the
    match map. Per-pair design: one lookup per contact ``k``; no slot
    because prev state is indexed by ``k`` directly.
    """
    wp.launch(
        kernel=_reset_forward_map_kernel,
        dim=rigid_contact_max,
        inputs=[],
        outputs=[cid_of_contact],
        device=device,
    )
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
    bodies: BodyContainer,
    contacts: ContactViews,
    cc: ContactContainer,
    device: wp.DeviceLike = None,
) -> None:
    """Copy prev-frame state into ``cc`` for matched contacts; initialise
    PhoenX-style for unmatched contacts.

    Called after the pointer-swap (``cc.prev_lambdas`` now holds last
    step's persistent state; ``cc.lambdas`` is scratch) but before
    :func:`contact_prepare_at_RR`.
    """
    wp.launch(
        kernel=_contact_warmstart_gather_kernel,
        dim=max(1, scratch.max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_first,
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            rigid_contact_match_index,
            prev_cid_of_contact,
            scratch.num_contact_columns,
            bodies,
            contacts,
        ],
        outputs=[cc],
        device=device,
    )
