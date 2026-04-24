# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Convert Newton's sorted ``Contacts`` buffer into PhoenX contact columns.

Called once per :meth:`PhoenXWorld.step`. Newton's
:class:`~newton._src.sim.contacts.Contacts` is already sorted by
``(shape_a, shape_b)`` (contacts for one pair are contiguous), so
ingest segments that prefix via ``(shape_a * num_shapes + shape_b)``
run-length encoding and emits one :data:`CONSTRAINT_TYPE_CONTACT`
column per non-filtered shape pair with the pair's
``[contact_first, contact_first + contact_count)`` stamped into the
header.

Fully graph-capture-safe: ``num_contact_columns`` is a device-side
scalar, never host-read during the step; all kernels launch at fixed
sizes (``rigid_contact_max`` / ``max_contact_columns``) and gate
internally on device-held counters. ``num_shapes < 46340`` keeps the
packed int32 key safe.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactViews,
    contact_set_contact_count,
    contact_set_contact_first,
    contact_set_friction,
    contact_set_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_BODY1_OFFSET,
    CONSTRAINT_BODY2_OFFSET,
    CONSTRAINT_TYPE_CONTACT,
    CONSTRAINT_TYPE_OFFSET,
    ConstraintContainer,
    write_int,
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
from newton._src.solvers.phoenx.helpers.scan_and_sort import (
    RLE_SENTINEL_INT32,
    runlength_encode_variable_length,
)
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

        # Per-contact pair key.
        self.keys = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # RLE unique-values output (keeps ``keys`` intact so callers can
        # re-RLE / re-inspect if needed).
        self.run_values = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

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
    """Fill the per-contact key array for the RLE pass."""
    tid = wp.tid()
    count = rigid_contact_count[0]
    if tid < count:
        sa = rigid_contact_shape0[tid]
        sb = rigid_contact_shape1[tid]
        keys[tid] = sa * num_shapes + sb
    else:
        keys[tid] = wp.int32(0)


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
    """Unpack ``run_values`` into per-pair shapes + clear the column tail.

    ``pair_columns`` is cleared for threads past ``num_pairs[0]`` so the
    subsequent exclusive scan produces a stable total regardless of
    leftover data from a previous step.
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        pair_columns[tid] = wp.int32(0)
        return
    key = run_values[tid]
    sa = key // num_shapes
    sb = key - sa * num_shapes
    pair_shape_a[tid] = sa
    pair_shape_b[tid] = sb


@wp.kernel(enable_backward=False)
def _pair_columns_binary_kernel(
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
    """Emit exactly one contact column per non-filtered shape pair.

    Per-pair design: every non-filtered pair with at least one contact
    gets a single column covering its entire
    ``[pair_first, pair_first + pair_count)`` range. Body-pair-filtered
    pairs collapse to zero columns so the downstream exclusive scan
    treats them as invisible (no ingest, no warm-start, no dispatch).
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        return
    sa = pair_shape_a[tid]
    sb = pair_shape_b[tid]
    ba = shape_body[sa]
    bb = shape_body[sb]
    # Drop self-contacts: two shapes on the same body (compound
    # geometry) produce no constraint. Also covers the static-vs-static
    # degenerate case where both shapes resolve to slot 0 (world).
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
    # Zero-contact runs can't happen here (RLE produces positive
    # counts), but guard anyway for defensive robustness.
    if pair_count[tid] > wp.int32(0):
        pair_columns[tid] = wp.int32(1)
    else:
        pair_columns[tid] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _pair_first_kernel(
    run_lengths: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    # out
    pair_first: wp.array[wp.int32],
):
    """Exclusive prefix sum of ``run_lengths`` -> ``pair_first``.

    Single-thread O(num_pairs) scan. ``num_pairs`` is small in practice
    (100s, not millions); launching a proper device scan just for this
    would cost more in kernel overhead than the serial scan itself.
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

    With the per-pair design ``pair_columns[p]`` is either 0 (filtered)
    or 1, so the inner ``for k in range(cols)`` loop runs at most once
    per pair. Kept as a single-thread kernel because ``num_pairs`` is
    small (100s) and the launch overhead dominates the serial cost.
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
    """Materialise one contact column -- one thread per output column.

    Launched at ``dim == max_contact_columns`` and gates internally on
    ``num_contact_columns[0]`` so the launch size is captureable in a
    graph independently of the per-step column count.

    Each thread owns output column ``tid``. Looks up its source pair
    ``p`` via ``pair_source_idx`` and writes the header + range. The
    contact range is ``[pair_first[p], pair_first[p] + pair_count[p])``
    -- one column now covers an entire shape pair.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    cid = cid_base + tid

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

    write_int(constraints, CONSTRAINT_TYPE_OFFSET, cid, CONSTRAINT_TYPE_CONTACT)
    write_int(constraints, CONSTRAINT_BODY1_OFFSET, cid, b1)
    write_int(constraints, CONSTRAINT_BODY2_OFFSET, cid, b2)

    contact_set_friction(constraints, cid, mu_static)
    contact_set_friction_dynamic(constraints, cid, mu_dynamic)
    contact_set_contact_first(constraints, cid, first)
    contact_set_contact_count(constraints, cid, count)


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
    cid_base: wp.int32,
    bodies: BodyContainer,
    contacts: ContactViews,
    cc: ContactContainer,
):
    """Seed this frame's ``cc`` slots from the prev frame (PhoenX model).

    One thread per output column. For each contact ``k`` in the
    column's range:

      * ``prev_k = rigid_contact_match_index[k]`` is the index in the
        *previous* frame's sorted buffer that the matcher paired this
        contact with, or :data:`MATCH_NOT_FOUND` / :data:`MATCH_BROKEN`
        (``< 0``) for unmatched.
      * When matched, copy every persistent field (``lambdas, normal,
        tangent1, local_p0, local_p1``) across from ``cc.prev_*[prev_k]``.
        This is the fixed-frame warm-start: the scalar impulses stay
        physically meaningful because the ``(n, t1, t2)`` basis they
        were written in is the same basis used to interpret them.
      * When unmatched, run PhoenX's ``Initialize`` dance: pull
        ``normal`` + body-local anchors from the upstream buffer,
        derive ``tangent1`` from the tangential relative velocity,
        and zero the impulses.

    With the per-pair design, prev-frame data is keyed directly by
    the contact's sorted-buffer index ``prev_k`` -- we only need a
    ``prev_cid_of_contact`` gate to distinguish "prev contact was
    covered by an active column" from "stale entry" and avoid reading
    a prev slot that belonged to an already-overwritten frame.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    _ = cid_base

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
    sizes known at host time, and all step-varying counts are kept
    on-device in :attr:`IngestScratch.num_pairs` /
    :attr:`IngestScratch.num_contact_columns`.

    Args:
        contacts: Newton :class:`Contacts` buffer. Must have been built
            with a non-disabled ``contact_matching`` mode.
        shape_body: ``model.shape_body`` array.
        num_shapes: Total shape count; used to pack the ``(shape_a,
            shape_b)`` key into int32.
        constraints: Shared constraint storage. Contact columns are
            written at ``cid in [cid_base, cid_base +
            num_contact_columns)``.
        scratch: Reusable per-step scratch.
        cid_base: First cid reserved for contacts.
        max_contact_columns: Hard cap on the number of contact columns
            this step can emit. The cap is now the number of distinct
            shape pairs rather than ``ceil(contact_count / 6)``, so
            it sizes much more tightly than the old design.
        default_friction: Fallback friction coefficient.
        device: Warp device for the launches.
        num_bodies: Total body count. Used to pack the
            ``(min_body, max_body)`` pair into an int64 key for the
            body-pair filter.
        filter_keys: Sorted ``wp.int64`` array of packed canonical
            body-pair keys to ignore.
        filter_count: Number of valid entries in ``filter_keys``.
        shape_material: Optional per-shape material index.
        materials: Optional material table.
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

    # Step 2: RLE.
    runlength_encode_variable_length(
        values=scratch.keys,
        active_length=contacts.rigid_contact_count,
        run_values=scratch.run_values,
        run_lengths=scratch.pair_count,
        run_count=scratch.num_pairs,
        sentinel=RLE_SENTINEL_INT32,
    )

    # Step 3a: unpack run_values -> pair_{shape_a, shape_b}, clear
    # pair_columns tail.
    wp.launch(
        kernel=_pair_metadata_kernel,
        dim=rigid_contact_max,
        inputs=[scratch.run_values, scratch.num_pairs, int(num_shapes)],
        outputs=[scratch.pair_shape_a, scratch.pair_shape_b, scratch.pair_columns],
        device=device,
    )

    # Step 3b: pair_columns[p] = 1 for non-filtered pairs with
    # pair_count > 0, else 0.
    if filter_keys is None:
        raise ValueError(
            "ingest_contacts: filter_keys must be non-None (use a size-1 "
            "sentinel array when no filters are registered; the kernel "
            "signature has no null-pointer path)."
        )
    wp.launch(
        kernel=_pair_columns_binary_kernel,
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
            int(cid_base),
            float(default_friction),
        ],
        outputs=[constraints],
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
    cid_base: int,
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
    :func:`contact_prepare_for_iteration_at`.
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
            int(cid_base),
            bodies,
            contacts,
        ],
        outputs=[cc],
        device=device,
    )
