# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Convert Newton's sorted ``Contacts`` buffer into Jitter contact columns.

Called once per :meth:`World.step` to materialise that step's
:data:`CONSTRAINT_TYPE_CONTACT` columns inside the shared
:class:`ConstraintContainer`. The upstream Newton
:class:`newton._src.sim.contacts.Contacts` buffer is already sorted by
``(shape_a, shape_b)`` when ``contact_matching`` is on (the
:func:`newton._src.geometry.contact_data.make_contact_sort_key`
invariant), so all contacts belonging to one shape-pair are
contiguous. Ingest therefore reduces to:

1. Segment the sorted active prefix of ``Contacts`` into shape-pair
   runs (``pair_first[p]``, ``pair_count[p]``, ``pair_shape_a/b[p]``).
2. For each pair ``p``, emit ``ceil(pair_count[p] / 6)`` adjacent
   contact columns with the appropriate ``[contact_first,
   contact_count]`` range and ``active_mask``.
3. Populate the :class:`ConstraintContainer` header
   (``constraint_type``, ``body1``, ``body2``) so the generic
   dispatcher routes each column to
   :func:`contact_prepare_for_iteration_at`.

The number of output columns is produced as a *device-side* scalar
(``IngestScratch.num_contact_columns``) and never read back to the
host during the step, which keeps the whole ingest sequence
graph-capture compatible. All kernels launch at fixed sizes
(``rigid_contact_max`` or ``max_contact_columns``) and gate
internally on the device-held counters. The only host reads happen
outside the captured region, in :meth:`World.step`, to decide how
many constraints to feed to the graph-coloring partitioner and the
iterate kernels.

Segmenting uses an ``(shape_a * num_shapes + shape_b)`` int32 key +
the graph-capture-safe
:func:`scan_and_sort.runlength_encode_variable_length` wrapper. The
int32 fits safely because Newton scenes with
``num_shapes >= 46340`` would overflow the key; the caller should
bound ``num_shapes`` accordingly at world construction time.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.constraints.constraint_contact import (
    CONTACT_MAX_SLOTS,
    contact_set_active_mask,
    contact_set_contact_first,
    contact_set_friction,
    contact_set_friction_dynamic,
)
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_BODY1_OFFSET,
    CONSTRAINT_BODY2_OFFSET,
    CONSTRAINT_TYPE_CONTACT,
    CONSTRAINT_TYPE_OFFSET,
    ConstraintContainer,
    write_int,
)
from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_contact import ContactViews
from newton._src.solvers.jitter.constraints.contact_container import (
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
from newton._src.solvers.jitter.helpers.scan_and_sort import (
    RLE_SENTINEL_INT32,
    runlength_encode_variable_length,
)
from newton._src.solvers.jitter.materials import (
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


# ---------------------------------------------------------------------------
# Host-side scratch -- reused across steps to avoid per-step allocs.
# ---------------------------------------------------------------------------


class IngestScratch:
    """Pre-allocated device buffers the ingest pipeline reuses each step.

    Sized to the upstream ``Contacts.rigid_contact_max`` (for per-
    contact arrays) and to ``max_contact_columns`` (for per-column
    arrays). Per-pair arrays are sized to ``rigid_contact_max`` too
    because in the worst case (every contact is its own shape pair)
    ``num_pairs == rigid_contact_max``.

    All scratch lives on ``device`` and is never resized after
    construction -- the shapes drive the launch sizes of every
    ingest kernel, which is what keeps the sequence graph-capture
    safe.
    """

    __slots__ = (
        "_device",
        "keys",
        "max_contact_columns",
        "num_contact_columns",
        "num_pairs",
        "pair_col_offset",
        "pair_columns",
        "pair_count",
        "pair_first",
        "pair_shape_a",
        "pair_shape_b",
        "pair_source_idx",
        "rigid_contact_max",
        "run_values",
    )

    def __init__(
        self,
        rigid_contact_max: int,
        max_contact_columns: int,
        device: wp.DeviceLike = None,
    ) -> None:
        self._device = device
        self.rigid_contact_max = int(rigid_contact_max)
        self.max_contact_columns = int(max_contact_columns)
        n_pairs_max = max(1, self.rigid_contact_max)
        n_cols_max = max(1, self.max_contact_columns)

        # Per-contact pair key. Bounded by ``rigid_contact_max``.
        self.keys = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # Output slot for RLE's unique-values array (keeps ``keys``
        # intact so callers can re-RLE / re-inspect if needed).
        self.run_values = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

        # Per-pair arrays.
        self.pair_first = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_count = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_a = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_b = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_columns = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_col_offset = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

        # Per-output-column arrays.
        self.pair_source_idx = wp.zeros(n_cols_max, dtype=wp.int32, device=device)

        # Device-held scalars. ``num_pairs`` is filled by the RLE pass;
        # ``num_contact_columns`` is filled by the total-columns
        # reduction kernel. Both are consumed by subsequent kernels
        # via a ``tid >= counter[0]`` gate, never read back host-side
        # inside the captured step.
        self.num_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        self.num_contact_columns = wp.zeros(1, dtype=wp.int32, device=device)


def allocate_ingest_scratch(
    rigid_contact_max: int,
    max_contact_columns: int,
    device: wp.DeviceLike = None,
) -> IngestScratch:
    """Factory alias matching the ``*_zeros`` style elsewhere in the solver."""
    return IngestScratch(rigid_contact_max, max_contact_columns, device=device)


# ---------------------------------------------------------------------------
# Step 1: build the int32 (shape_a, shape_b) key per active contact.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _contact_key_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    num_shapes: wp.int32,
    # out
    keys: wp.array[wp.int32],
):
    """Fill the per-contact key array for the RLE pass.

    The active prefix (``tid < rigid_contact_count[0]``) gets
    packed keys ``shape_a * num_shapes + shape_b``; the tail is
    zeroed and will be re-stamped with the sentinel by the RLE
    wrapper (:func:`runlength_encode_variable_length`) -- we don't
    stamp the sentinel here because the RLE wrapper needs to re-
    stamp it anyway to keep the mask/RLE pair self-contained.
    """
    tid = wp.tid()
    count = rigid_contact_count[0]
    if tid < count:
        sa = rigid_contact_shape0[tid]
        sb = rigid_contact_shape1[tid]
        keys[tid] = sa * num_shapes + sb
    else:
        # Stamp zero so the subsequent RLE-wrapper sentinel overwrite
        # can't accidentally leave stale data from a previous step
        # masquerading as a real key.
        keys[tid] = wp.int32(0)


@wp.func
def _body_pair_filtered(
    filter_keys: wp.array[wp.int64],
    filter_count: wp.int32,
    key: wp.int64,
) -> wp.int32:
    """Binary search ``filter_keys[0:filter_count]`` for ``key``.

    Returns ``1`` if found, ``0`` otherwise. ``filter_keys`` is a
    sorted int64 array of packed
    ``min(body_a, body_b) * num_bodies + max(body_a, body_b)`` keys;
    ``filter_count == 0`` short-circuits without touching the array
    (which still carries a size-1 sentinel so the pointer is
    non-null).
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
# Step 2: turn run_lengths into [pair_count, shape_a, shape_b, pair_columns].
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _pair_metadata_kernel(
    run_values: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    num_shapes: wp.int32,
    # out
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
):
    """Unpack ``run_values`` into per-pair shapes + column counts.

    ``pair_count`` is already the RLE ``run_lengths`` output; the RLE
    wrapper wrote it directly into ``scratch.pair_count`` so there's
    nothing to do for it here.

    Launches at ``dim == rigid_contact_max`` (the max possible pair
    count); threads past ``num_pairs[0]`` zero out the corresponding
    ``pair_columns`` entry so the subsequent exclusive scan produces
    a stable total (equal to the inclusive tail of the active
    prefix) regardless of leftover data from a previous step.
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        # Clear tail so the downstream scan is deterministic.
        pair_columns[tid] = wp.int32(0)
        return
    key = run_values[tid]
    sa = key // num_shapes
    sb = key - sa * num_shapes
    pair_shape_a[tid] = sa
    pair_shape_b[tid] = sb
    # Number of 6-slot contact columns needed for this pair.
    # ``pair_count[tid]`` already holds the run length written by
    # the RLE wrapper; read it directly.
    # (Kernel can't both read and write the same array at the same
    # index, so we don't take pair_count as an output.)


@wp.kernel(enable_backward=False)
def _pair_columns_from_count_kernel(
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    num_bodies: wp.int32,
    filter_keys: wp.array[wp.int64],
    filter_count: wp.int32,
    # out
    pair_columns: wp.array[wp.int32],
):
    """Compute ``pair_columns[p] = ceil(pair_count[p] / 6)`` for active pairs,
    collapsing filtered body pairs to zero columns.

    For each active shape pair (``tid < num_pairs[0]``) we resolve
    the two body ids via ``shape_body[pair_shape_{a,b}[p]]``, pack
    them into the canonical ``(min, max)`` int64 key, and check the
    sorted ``filter_keys`` array. Filtered pairs get
    ``pair_columns[p] = 0`` so the downstream exclusive scan treats
    them as taking zero output columns; :func:`_pair_source_idx_kernel`
    then never maps any output column to that pair, so no contact
    constraint is allocated or warm-started, and the dispatcher
    never sees it.

    Split out from :func:`_pair_metadata_kernel` because that kernel
    already has ``pair_columns`` bound as an output for the tail
    clear; keeping the filter test here makes both kernels'
    access patterns read-only except for one output array.
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        return
    sa = pair_shape_a[tid]
    sb = pair_shape_b[tid]
    ba = shape_body[sa]
    bb = shape_body[sb]
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
    length = pair_count[tid]
    pair_columns[tid] = (length + CONTACT_MAX_SLOTS - 1) // CONTACT_MAX_SLOTS


@wp.kernel(enable_backward=False)
def _pair_first_kernel(
    run_lengths: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    # out
    pair_first: wp.array[wp.int32],
):
    """Exclusive prefix sum of ``run_lengths`` -> ``pair_first``.

    Single-thread O(num_pairs) scan. ``num_pairs`` is small in
    practice (100s not millions) so the serial cost is negligible
    compared to launching a proper device scan, and this keeps the
    ingest sequence free of extra scratch allocations.
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = num_pairs[0]
    acc = wp.int32(0)
    for i in range(n):
        pair_first[i] = acc
        acc += run_lengths[i]


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
    """Derive ``num_contact_columns = sum(pair_columns)`` on-device.

    The exclusive scan of ``pair_columns`` has already been written
    to ``pair_col_offset`` by :func:`wp.utils.array_scan` before
    this kernel is launched, so we just read the last element of
    the scan + ``pair_columns[num_pairs - 1]`` and clamp against
    the caller's hard cap. Single-thread kernel; output is a
    device-held 1-element array consumed by all downstream
    ingest / gather kernels as their active-length gate.
    """
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
    """For each output column ``o``, write which pair ``p`` it comes from.

    Serial on one thread because ``num_pairs`` is small and the
    expansion is trivially bounded by ``max_columns``. No race, no
    atomics; clamps every write to ``[0, max_columns)``.
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = num_pairs[0]
    for p in range(n):
        off = pair_col_offset[p]
        cols = pair_columns[p]
        for k in range(cols):
            o = off + k
            if o < max_columns:
                pair_source_idx[o] = p


# ---------------------------------------------------------------------------
# Step 4: per-column pack kernel -- writes the contact constraint header
# and the [contact_first, contact_count] range into the ConstraintContainer.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _contact_pack_columns_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_col_offset: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_material: wp.array[wp.int32],
    materials: wp.array[MaterialData],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    default_friction: wp.float32,
    # out
    constraints: ConstraintContainer,
):
    """Materialise one contact column.

    Launched at ``dim == max_contact_columns`` (host-known) and
    gates internally on ``tid >= num_contact_columns[0]`` so the
    launch size is captureable in a graph independently of the
    per-step column count.

    Thread ``tid`` owns output column ``tid``. Looks up which pair
    ``p`` it belongs to via ``pair_source_idx``, then its sub-
    column index within that pair as ``tid - pair_col_offset[p]``.
    The contact range is ``[pair_first[p] + k*6, pair_first[p] +
    k*6 + slot_count)`` and the active mask is ``(1 << slot_count)
    - 1``. Contacts live at ``cid in [cid_base, cid_base +
    num_contact_columns)``.

    Friction resolution uses the per-shape materials table via
    :func:`resolve_friction_in_kernel`, combining the two materials
    under whichever ``friction_combine_mode`` is stricter. Callers
    that don't wire the materials table (``materials`` size 0, or
    ``shape_material[s]`` out-of-range) fall back to
    ``default_friction`` unchanged -- pre-material scenes behave
    identically.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    cid = cid_base + tid

    p = pair_source_idx[tid]
    col_in_pair = tid - pair_col_offset[p]
    total_in_pair = pair_count[p]
    start_contact = pair_first[p] + col_in_pair * CONTACT_MAX_SLOTS

    # Number of active slots in *this* column. Full 6 for interior
    # columns; the tail column gets what's left.
    remaining = total_in_pair - col_in_pair * CONTACT_MAX_SLOTS
    slot_count = remaining
    if slot_count > CONTACT_MAX_SLOTS:
        slot_count = CONTACT_MAX_SLOTS
    active_mask = (wp.int32(1) << slot_count) - wp.int32(1)

    # Resolve shape -> body once; both shapes within a pair map to
    # the same two bodies for every contact in the pair by
    # construction.
    sa = pair_shape_a[p]
    sb = pair_shape_b[p]
    b1 = shape_body[sa]
    b2 = shape_body[sb]

    # Resolve the pair's effective static + kinetic friction.
    # ``shape_material`` has size ``num_shapes``; if the caller passed
    # an empty sentinel (size-1 array of -1) both coefficients fall
    # back to ``default_friction`` so pre-material scenes behave
    # identically to before the two-regime plumbing was added.
    mat_a = wp.int32(-1)
    mat_b = wp.int32(-1)
    if shape_material.shape[0] > sa:
        mat_a = shape_material[sa]
    if shape_material.shape[0] > sb:
        mat_b = shape_material[sb]
    mu_static = resolve_friction_static_in_kernel(
        materials, mat_a, mat_b, default_friction
    )
    mu_dynamic = resolve_friction_in_kernel(
        materials, mat_a, mat_b, default_friction
    )

    # Header.
    write_int(constraints, CONSTRAINT_TYPE_OFFSET, cid, CONSTRAINT_TYPE_CONTACT)
    write_int(constraints, CONSTRAINT_BODY1_OFFSET, cid, b1)
    write_int(constraints, CONSTRAINT_BODY2_OFFSET, cid, b2)

    # Contact-specific header bits.
    contact_set_friction(constraints, cid, mu_static)
    contact_set_friction_dynamic(constraints, cid, mu_dynamic)
    contact_set_active_mask(constraints, cid, active_mask)
    contact_set_contact_first(constraints, cid, start_contact)


# ---------------------------------------------------------------------------
# Warm-start gather kernel.
# ---------------------------------------------------------------------------


@wp.func
def _build_tangent1_from_velocity(n: wp.vec3f, dv: wp.vec3f) -> wp.vec3f:
    """Build a unit tangent1 from the tangential relative velocity.

    Mirrors PhoenX's ``RigidRigidContactManifoldFunctions::Initialize``
    tangent frame: project the contact-point relative velocity onto the
    contact plane and normalise. When there's no meaningful tangential
    slide, fall back to a branch-free orthonormal seed (Duff et al.
    2017) so the basis still varies smoothly with ``n``.

    Aligning ``tangent1`` with the sliding direction means the friction
    row pre-soaks the instantaneous direction impulses need to oppose
    -- a fresh contact with tangential velocity immediately gets a
    non-zero friction budget along the "right" axis instead of two
    arbitrary orthogonal rows fighting each other for the first few
    iterations.
    """
    v_n = wp.dot(dv, n) * n
    v_t = dv - v_n
    len_sq = wp.dot(v_t, v_t)
    if len_sq > wp.float32(1.0e-12):
        return v_t / wp.sqrt(len_sq)
    # No slide -- fall back to a smooth orthonormal seed. Same Duff
    # 2017 construction used by ``_build_tangents``, repeated here
    # because this func lives in a different module and we want zero
    # cross-module dependency for this hot kernel.
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
    pair_col_offset: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    rigid_contact_match_index: wp.array[wp.int32],
    prev_slot_of_contact: wp.array[wp.int32],
    prev_cid_of_contact: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    bodies: BodyContainer,
    contacts: ContactViews,
    cc: ContactContainer,
):
    """Seed this frame's ``cc.*`` slots from the prev frame (PhoenX model).

    For each active slot of each current-frame contact column:

    * ``k = pair_first[p] + col_in_pair * 6 + slot`` is the contact's
      sorted-buffer index.
    * ``prev_k = rigid_contact_match_index[k]`` is the index in the
      *previous* frame's sorted buffer that the matcher paired us
      with, or :data:`MATCH_NOT_FOUND` / :data:`MATCH_BROKEN` (both
      ``< 0``).
    * When the matcher found a valid previous slot, copy **every**
      persistent field (``lambdas, normal, tangent1, local_p0,
      local_p1``) across from ``cc.prev_*``. This gives the PGS a
      fixed-frame warm-start: the contact's ``(n, t1, t2)`` basis is
      the one the previous frame's solution was written in, so the
      scalar impulses stay physically meaningful.
    * When there's no match -- contact is new or the prev point was
      already claimed -- do PhoenX's ``Initialize`` dance: pull
      ``normal`` and body-local anchors from the upstream buffer,
      derive ``tangent1`` from the tangential relative velocity at
      contact, and zero the impulses.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    cid = cid_base + tid

    p = pair_source_idx[tid]
    col_in_pair = tid - pair_col_offset[p]
    total_in_pair = pair_count[p]
    remaining = total_in_pair - col_in_pair * CONTACT_MAX_SLOTS
    slot_count = remaining
    if slot_count > CONTACT_MAX_SLOTS:
        slot_count = CONTACT_MAX_SLOTS
    start_contact = pair_first[p] + col_in_pair * CONTACT_MAX_SLOTS

    # Resolve this column's body pair once -- both bodies are shared by
    # every slot in the column.
    sa = pair_shape_a[p]
    sb = pair_shape_b[p]
    b1 = contacts.shape_body[sa]
    b2 = contacts.shape_body[sb]

    for s in range(CONTACT_MAX_SLOTS):
        if s >= slot_count:
            # Inactive slot -- zero everything so a stray read from
            # the prepare/iterate kernels degrades into a no-op.
            cc_set_normal_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent1_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent2_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_normal(cc, s, cid, wp.vec3f(0.0, 0.0, 0.0))
            cc_set_tangent1(cc, s, cid, wp.vec3f(0.0, 0.0, 0.0))
            cc_set_local_p0(cc, s, cid, wp.vec3f(0.0, 0.0, 0.0))
            cc_set_local_p1(cc, s, cid, wp.vec3f(0.0, 0.0, 0.0))
            continue

        k = start_contact + s

        prev_slot = wp.int32(-1)
        prev_cid = wp.int32(-1)
        prev_k = rigid_contact_match_index[k]
        if prev_k >= 0:
            prev_slot = prev_slot_of_contact[prev_k]
            prev_cid = prev_cid_of_contact[prev_k]

        # Precompute the fresh narrow-phase data; we'll either use it
        # wholesale (new contact, or matched-but-worse-penetration) or
        # replace a prev-frame frame that's grown stale.
        fresh_local_p0 = contacts.rigid_contact_point0[k]
        fresh_local_p1 = contacts.rigid_contact_point1[k]
        fresh_n = contacts.rigid_contact_normal[k]

        fresh_r1 = wp.quat_rotate(bodies.orientation[b1], fresh_local_p0)
        fresh_r2 = wp.quat_rotate(bodies.orientation[b2], fresh_local_p1)

        if prev_slot >= 0 and prev_cid >= 0:
            # Matched. Decide whether to carry the prev frame forward
            # or overwrite it with the fresh narrow-phase geometry.
            # PhoenX pattern: if the narrow phase now sees a deeper
            # penetration than the contact's stored anchors imply, the
            # contact's frozen geometry has grown stale and the fresh
            # detection is a better reflection of the current physical
            # contact -- swap it in. This prevents slowly-drifting
            # stacks from locking onto a frame that no longer matches
            # the real geometry, which otherwise manifests as
            # accumulated lateral drift over long runs.
            prev_n = cc_get_prev_normal(cc, prev_slot, prev_cid)
            prev_lp0 = cc_get_prev_local_p0(cc, prev_slot, prev_cid)
            prev_lp1 = cc_get_prev_local_p1(cc, prev_slot, prev_cid)
            prev_r1 = wp.quat_rotate(bodies.orientation[b1], prev_lp0)
            prev_r2 = wp.quat_rotate(bodies.orientation[b2], prev_lp1)
            prev_p1_world = bodies.position[b1] + prev_r1
            prev_p2_world = bodies.position[b2] + prev_r2
            # ``dot(p2 - p1, n)`` is positive when separated; the
            # negation below makes ``penetration`` positive when
            # bodies overlap (Phoenx's ``ComputeContactBias`` uses
            # the same convention).
            prev_penetration = -wp.dot(prev_p2_world - prev_p1_world, prev_n)

            fresh_p1_world = bodies.position[b1] + fresh_r1
            fresh_p2_world = bodies.position[b2] + fresh_r2
            fresh_penetration = -wp.dot(fresh_p2_world - fresh_p1_world, fresh_n)

            if fresh_penetration > prev_penetration:
                # Overwrite: the fresh narrow-phase detection sees a
                # deeper overlap than the stored frame can produce
                # once the bodies have moved into their current pose.
                # Carry the impulses forward for warm-start but rebuild
                # normal / tangent / anchors.
                dv = (bodies.velocity[b2] + wp.cross(bodies.angular_velocity[b2], fresh_r2)) - (
                    bodies.velocity[b1] + wp.cross(bodies.angular_velocity[b1], fresh_r1)
                )
                fresh_t1 = _build_tangent1_from_velocity(fresh_n, dv)

                cc_set_normal_lambda(cc, s, cid, cc_get_prev_normal_lambda(cc, prev_slot, prev_cid))
                cc_set_tangent1_lambda(cc, s, cid, cc_get_prev_tangent1_lambda(cc, prev_slot, prev_cid))
                cc_set_tangent2_lambda(cc, s, cid, cc_get_prev_tangent2_lambda(cc, prev_slot, prev_cid))
                cc_set_normal(cc, s, cid, fresh_n)
                cc_set_tangent1(cc, s, cid, fresh_t1)
                cc_set_local_p0(cc, s, cid, fresh_local_p0)
                cc_set_local_p1(cc, s, cid, fresh_local_p1)
                continue

            # Matched and prev frame still describes the contact
            # accurately -- carry the full PhoenX state forward.
            cc_set_normal_lambda(cc, s, cid, cc_get_prev_normal_lambda(cc, prev_slot, prev_cid))
            cc_set_tangent1_lambda(cc, s, cid, cc_get_prev_tangent1_lambda(cc, prev_slot, prev_cid))
            cc_set_tangent2_lambda(cc, s, cid, cc_get_prev_tangent2_lambda(cc, prev_slot, prev_cid))
            cc_set_normal(cc, s, cid, prev_n)
            cc_set_tangent1(cc, s, cid, cc_get_prev_tangent1(cc, prev_slot, prev_cid))
            cc_set_local_p0(cc, s, cid, prev_lp0)
            cc_set_local_p1(cc, s, cid, prev_lp1)
            continue

        # New contact -- initialize PhoenX-style from the upstream
        # narrow-phase output. Body-local anchors map straight from
        # Newton's per-contact point arrays (both are already in body-
        # origin frame); the normal is world-frame; ``tangent1`` gets
        # anchored to the sliding direction.
        dv = (bodies.velocity[b2] + wp.cross(bodies.angular_velocity[b2], fresh_r2)) - (
            bodies.velocity[b1] + wp.cross(bodies.angular_velocity[b1], fresh_r1)
        )
        t1 = _build_tangent1_from_velocity(fresh_n, dv)

        cc_set_normal_lambda(cc, s, cid, wp.float32(0.0))
        cc_set_tangent1_lambda(cc, s, cid, wp.float32(0.0))
        cc_set_tangent2_lambda(cc, s, cid, wp.float32(0.0))
        cc_set_normal(cc, s, cid, fresh_n)
        cc_set_tangent1(cc, s, cid, t1)
        cc_set_local_p0(cc, s, cid, fresh_local_p0)
        cc_set_local_p1(cc, s, cid, fresh_local_p1)


@wp.kernel(enable_backward=False)
def _reset_forward_map_kernel(
    # out
    slot_of_contact: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """Clear the forward map to ``-1`` before every stamp pass.

    Separate tiny kernel so the caller doesn't need a graph-
    breaking ``fill_`` between steps.
    """
    tid = wp.tid()
    slot_of_contact[tid] = wp.int32(-1)
    cid_of_contact[tid] = wp.int32(-1)


@wp.kernel(enable_backward=False)
def _stamp_slot_cid_of_contact_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_col_offset: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    # out
    slot_of_contact: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """For each contact in this frame's sorted buffer, stamp ``(slot, cid)``.

    This is the "forward" map the *next* frame will consult when it
    swaps ``(slot_of_contact, cid_of_contact)`` into the "prev" role
    and looks up its own ``rigid_contact_match_index[k] = prev_k``.
    Written during ingest once the column layout is known.

    Entries not covered by any column keep the ``-1`` written by
    :func:`_reset_forward_map_kernel`.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    p = pair_source_idx[tid]
    col_in_pair = tid - pair_col_offset[p]
    total_in_pair = pair_count[p]
    remaining = total_in_pair - col_in_pair * CONTACT_MAX_SLOTS
    slot_count = remaining
    if slot_count > CONTACT_MAX_SLOTS:
        slot_count = CONTACT_MAX_SLOTS
    start_contact = pair_first[p] + col_in_pair * CONTACT_MAX_SLOTS
    cid = cid_base + tid
    for s in range(slot_count):
        slot_of_contact[start_contact + s] = s
        cid_of_contact[start_contact + s] = cid


# ---------------------------------------------------------------------------
# Host driver.
# ---------------------------------------------------------------------------


def ingest_contacts(
    contacts,  # newton._src.sim.contacts.Contacts
    shape_body: wp.array,
    num_shapes: int,
    constraints: ConstraintContainer,
    scratch: IngestScratch,
    cid_base: int,
    max_contact_columns: int,
    default_friction: float = 0.5,
    device: wp.DeviceLike = None,
    *,
    num_bodies: int = 0,
    filter_keys: wp.array | None = None,
    filter_count: int = 0,
    shape_material: wp.array | None = None,
    materials: wp.array | None = None,
) -> None:
    """Materialise contact columns for one step.

    Graph-capture safe: no host readbacks, all kernel launches have
    sizes known at host time (``rigid_contact_max`` /
    ``max_contact_columns``), and all step-varying counts are
    kept on-device in :attr:`IngestScratch.num_pairs` /
    :attr:`IngestScratch.num_contact_columns`.

    After the call, ``scratch.num_contact_columns[0]`` holds the
    number of columns emitted; the caller reads this *outside* the
    captured region (e.g. right before the next step) to resize
    the partitioner's active length.

    Args:
        contacts: Newton :class:`Contacts` buffer. Must have been
            built with a non-disabled ``contact_matching`` mode
            (``"sticky"`` recommended for stable stacking).
        shape_body: ``model.shape_body`` array.
        num_shapes: Total shape count in the owning Newton model;
            only used to pack the (shape_a, shape_b) key into int32.
        constraints: Shared constraint storage. Contact columns are
            written at ``cid in [cid_base, cid_base +
            num_contact_columns)``.
        scratch: Reusable per-step scratch.
        cid_base: First cid in the container reserved for contacts.
        max_contact_columns: Hard cap on the number of contact
            columns this step can emit. Excess contacts are
            silently dropped by the final clamp.
        default_friction: Fallback friction coefficient when no
            material table is provided (or a shape's material index
            is out of range). Wired into every contact column's
            header when ``materials`` is ``None`` / empty or when
            either shape in a pair doesn't have a valid material
            assignment.
        device: Warp device for the launches.
        num_bodies: Total body count in the owning :class:`World`.
            Used to pack the ``(min_body, max_body)`` pair into an
            int64 key for the body-pair filter lookup.
        filter_keys: Sorted ``wp.int64`` array of packed canonical
            body-pair keys to ignore. May carry a trailing sentinel;
            only the first ``filter_count`` entries are searched. The
            array must always be non-null (size-1 sentinel array when
            no filters are registered) to satisfy the kernel binding.
        filter_count: Number of valid entries at the start of
            ``filter_keys``. ``0`` short-circuits the filter and
            matches the legacy (no-filter) behaviour exactly.
        shape_material: Optional ``wp.array[int32]`` of shape
            ``(num_shapes,)`` giving each shape's material index
            into ``materials``. ``None`` or any out-of-range index
            falls back to ``default_friction`` for that pair. A
            size-1 sentinel array ``[-1]`` is also accepted so
            graph-captured callers can hand a valid pointer when
            no materials are configured.
        shape_material: See ``shape_material``.
        materials: Optional ``wp.array[MaterialData]`` material
            table (see :mod:`newton._src.solvers.jitter.materials`).
            Empty or ``None`` means "use ``default_friction``
            everywhere". Entry 0 is conventionally the default
            material and is what unassigned shapes resolve to.
    """
    rigid_contact_max = int(contacts.rigid_contact_max)

    # Step 1: per-contact key.
    wp.launch(
        kernel=_contact_key_kernel,
        dim=rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            int(num_shapes),
        ],
        outputs=[scratch.keys],
        device=device,
    )

    # Step 2: graph-capture-safe RLE. Writes run_values into
    # scratch.run_values, run_lengths into scratch.pair_count, and
    # total run count into scratch.num_pairs.
    runlength_encode_variable_length(
        values=scratch.keys,
        active_length=contacts.rigid_contact_count,
        run_values=scratch.run_values,
        run_lengths=scratch.pair_count,
        run_count=scratch.num_pairs,
        sentinel=RLE_SENTINEL_INT32,
    )

    # Step 3a: unpack run_values -> pair_{shape_a, shape_b},
    # clear pair_columns tail.
    wp.launch(
        kernel=_pair_metadata_kernel,
        dim=rigid_contact_max,
        inputs=[scratch.run_values, scratch.num_pairs, int(num_shapes)],
        outputs=[scratch.pair_shape_a, scratch.pair_shape_b, scratch.pair_columns],
        device=device,
    )

    # Step 3b: pair_columns[p] = ceil(pair_count[p] / 6), or 0 for
    # pairs that the body-pair collision filter excludes. Filtered
    # pairs take zero output columns so the downstream scan + pack
    # never touches them.
    if filter_keys is None:
        raise ValueError(
            "ingest_contacts: filter_keys must be non-None (use a size-1 "
            "sentinel array when no filters are registered; the kernel "
            "signature has no null-pointer path)."
        )
    wp.launch(
        kernel=_pair_columns_from_count_kernel,
        dim=rigid_contact_max,
        inputs=[
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            scratch.num_pairs,
            shape_body,
            int(num_bodies),
            filter_keys,
            int(filter_count),
        ],
        outputs=[scratch.pair_columns],
        device=device,
    )

    # Step 3c: exclusive scan pair_count -> pair_first.
    wp.launch(
        kernel=_pair_first_kernel,
        dim=1,
        inputs=[scratch.pair_count, scratch.num_pairs],
        outputs=[scratch.pair_first],
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

    # Step 4: build the per-column -> pair map.
    wp.launch(
        kernel=_pair_source_idx_kernel,
        dim=1,
        inputs=[
            scratch.pair_col_offset,
            scratch.pair_columns,
            scratch.num_pairs,
            int(max_contact_columns),
        ],
        outputs=[scratch.pair_source_idx],
        device=device,
    )

    # Step 5: write the contact column headers + ranges. Launches
    # at max_contact_columns and gates internally on
    # num_contact_columns[0]. The materials / shape_material
    # sentinels keep the kernel signature constant across scenes
    # that do / don't wire the material system.
    if shape_material is None:
        shape_material = wp.array([-1], dtype=wp.int32, device=device)
    if materials is None:
        materials = wp.zeros(0, dtype=MaterialData, device=device)

    wp.launch(
        kernel=_contact_pack_columns_kernel,
        dim=max(1, max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_col_offset,
            scratch.pair_first,
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            shape_body,
            shape_material,
            materials,
            scratch.num_contact_columns,
            int(cid_base),
            float(default_friction),
        ],
        outputs=[constraints],
        device=device,
    )

    # NOTE: the MPR/GJK/SDF narrow phase can emit duplicate
    # manifold points within sub-mm distance of each other (~79%
    # of contact pairs on the m20 nut-bolt SDF scene in a
    # diagnostic scan). Duplicates double up each point's normal
    # impulse and friction budget, distorting physics. A naive
    # dedup pass (clear active_mask bits whose anchor points are
    # within 0.1 mm of an earlier slot) regressed the 20-cube
    # stack from 7 cm to 1.76 m slip even though the host-side
    # diagnostic reported zero duplicates at that threshold for
    # the stack scene -- root cause uncertain (likely a subtle
    # warm-start / effective-mass interaction when a PGS-active
    # slot gets turned off mid-column). A proper fix probably
    # belongs in Newton's narrow phase rather than our ingest,
    # and is deferred.


def stamp_forward_contact_map(
    rigid_contact_max: int,
    cid_base: int,
    scratch: IngestScratch,
    slot_of_contact: wp.array,
    cid_of_contact: wp.array,
    device: wp.DeviceLike = None,
) -> None:
    """Fill the forward (sorted-index -> (slot, cid)) lookup.

    Called after :func:`ingest_contacts`; the two output arrays are
    consumed by the *next* step's warm-start gather as the "prev"
    side of the match map. Graph-capture safe: resets the entire
    map (size ``rigid_contact_max``) on every call, then stamps
    active slots gated on ``scratch.num_contact_columns[0]``.
    """
    # Reset the full table to -1.
    wp.launch(
        kernel=_reset_forward_map_kernel,
        dim=rigid_contact_max,
        inputs=[],
        outputs=[slot_of_contact, cid_of_contact],
        device=device,
    )
    # Stamp the active slots.
    wp.launch(
        kernel=_stamp_slot_cid_of_contact_kernel,
        dim=max(1, scratch.max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_col_offset,
            scratch.pair_first,
            scratch.pair_count,
            scratch.num_contact_columns,
            int(cid_base),
        ],
        outputs=[slot_of_contact, cid_of_contact],
        device=device,
    )


def gather_contact_warmstart(
    cid_base: int,
    scratch: IngestScratch,
    rigid_contact_match_index: wp.array,
    prev_slot_of_contact: wp.array,
    prev_cid_of_contact: wp.array,
    bodies: BodyContainer,
    contacts: ContactViews,
    cc: ContactContainer,
    device: wp.DeviceLike = None,
) -> None:
    """Copy prev-frame state into ``cc`` for matched slots; initialise
    PhoenX-style for unmatched slots.

    Called after the pointer-swap (``cc.prev_lambdas`` now holds last
    step's persistent state; ``cc.lambdas`` is scratch) but before
    :func:`contact_prepare_for_iteration_at`. The kernel handles both
    the warm-start carry and the "new contact" initialise pass in one
    launch so the full per-slot frame
    (``lam_n, lam_t1, lam_t2, normal, tangent1, local_p0, local_p1``)
    is populated before the first prepare sees the column.
    """
    wp.launch(
        kernel=_contact_warmstart_gather_kernel,
        dim=max(1, scratch.max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_col_offset,
            scratch.pair_first,
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            rigid_contact_match_index,
            prev_slot_of_contact,
            prev_cid_of_contact,
            scratch.num_contact_columns,
            int(cid_base),
            bodies,
            contacts,
        ],
        outputs=[cc],
        device=device,
    )
