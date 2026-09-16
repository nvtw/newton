# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Repair duplicate fingerprint ownership without changing unique prior matches.

Distance and sort key remain the primary priority. Fresh body-local geometry
breaks duplicate-key ties deterministically; canonical rank is used only for
physically indistinguishable current contacts. This is an isolated experiment.
"""

import warp as wp

from newton._src.geometry import contact_match as cm


@wp.kernel(enable_backward=False)
def count_owners(data: cm._MatchData, snapshot: wp.array[int], count: wp.array[int]):
    k = wp.tid()
    if k < wp.min(data.new_count[0], snapshot.shape[0]):
        old = snapshot[k]
        if old >= 0:
            wp.atomic_add(count, old, 1)


@wp.func
def source_index(data: cm._MatchData, k: int):
    result = k
    if data.use_permutation != 0:
        result = data.canonical_to_source[k]
    return result


@wp.func
def claim(data: cm._MatchData, k: int, old: int):
    source = source_index(data, k)
    p0 = data.new_point0[source]
    p1 = data.new_point1[source]
    b0 = data.shape_body[data.new_shape0[source]]
    b1 = data.shape_body[data.new_shape1[source]]
    if b0 >= 0:
        p0 = wp.transform_point(data.body_q[b0], p0)
    if b1 >= 0:
        p1 = wp.transform_point(data.body_q[b1], p1)
    diff = wp.float32(0.5) * (p0 + p1) - data.prev_pos_world[old]
    dist = wp.dot(diff, diff)
    if data.sticky != 0:
        nd = wp.dot(diff, data.new_normal[source])
        dist = wp.max(dist - nd * nd, wp.float32(0))
    return cm._pack_claim(dist, data.new_keys[source])


@wp.func
def lex_before(data: cm._MatchData, a: int, b: int):
    """Compare complete source geometry without using unsorted allocation order."""
    sa = source_index(data, a)
    sb = source_index(data, b)
    ordering = int(0)
    for group in range(3):
        va = data.new_point0[sa]
        vb = data.new_point0[sb]
        if group == 1:
            va = data.new_point1[sa]
            vb = data.new_point1[sb]
        elif group == 2:
            va = data.new_normal[sa]
            vb = data.new_normal[sb]
        for axis in range(3):
            if ordering == 0:
                if va[axis] < vb[axis]:
                    ordering = -1
                elif va[axis] > vb[axis]:
                    ordering = 1
    if ordering == 0:
        if data.new_margin0[sa] < data.new_margin0[sb]:
            ordering = -1
        elif data.new_margin0[sa] > data.new_margin0[sb]:
            ordering = 1
    if ordering == 0:
        if data.new_margin1[sa] < data.new_margin1[sb]:
            ordering = -1
        elif data.new_margin1[sa] > data.new_margin1[sb]:
            ordering = 1
    return ordering < 0 or (ordering == 0 and a < b)


@wp.kernel(enable_backward=False)
def resolve_unique(data: cm._MatchData, snapshot: wp.array[int], count: wp.array[int], stats: wp.array[int]):
    k = wp.tid()
    n = wp.min(data.new_count[0], snapshot.shape[0])
    if k < n:
        old = snapshot[k]
        if old >= 0 and count[old] > 1:
            priority = claim(data, k, old)
            winner = wp.bool(True)
            for other in range(n):
                if other != k and snapshot[other] == old:
                    other_priority = claim(data, other, old)
                    if other_priority < priority or (other_priority == priority and lex_before(data, other, k)):
                        winner = False
            if not winner:
                data.match_index[k] = cm.MATCH_BROKEN
                wp.atomic_add(stats, 0, 1)


def match_data(matcher, kwargs):
    """Build the same geometry bundle as the actual matcher."""
    data = cm._MatchData()
    for field, attribute in {
        "prev_keys": "_prev_sorted_keys",
        "prev_pos_world": "_prev_pos_world",
        "prev_normal": "_prev_normal",
        "prev_count": "_prev_count",
        "reset_world_mask": "_reset_world_mask",
        "shape_world": "_shape_world",
        "world_count": "_world_count",
        "prev_claim": "_prev_claim",
        "pos_threshold_sq": "_pos_threshold_sq",
        "normal_dot_threshold": "_normal_dot_threshold",
        "pair_sub_key_mask": "_pair_sub_key_mask",
        "pair_key_stride": "_pair_key_stride",
    }.items():
        setattr(data, field, getattr(matcher, attribute))
    for field, argument in {
        "new_keys": "sort_keys",
        "new_point0": "point0",
        "new_point1": "point1",
        "new_shape0": "shape0",
        "new_shape1": "shape1",
        "new_normal": "normal",
        "new_margin0": "margin0",
        "new_margin1": "margin1",
        "new_count": "contact_count",
        "body_q": "body_q",
        "shape_body": "shape_body",
        "match_index": "match_index_out",
    }.items():
        setattr(data, field, kwargs[argument])
    permutation = kwargs.get("canonical_to_source")
    data.canonical_to_source = kwargs["match_index_out"] if permutation is None else permutation
    data.use_permutation = int(permutation is not None)
    data.sticky = int(matcher._sticky)
    return data


class UniqueClaims:
    """Allocate once and repair only multiply owned predecessor contacts."""

    def __init__(self, matcher):
        self.matcher = matcher
        self.device = matcher._prev_count.device
        self.snapshot = wp.zeros(matcher._capacity, dtype=int, device=self.device)
        self.count = wp.zeros(matcher._capacity, dtype=int, device=self.device)
        self.stats = wp.zeros(1, dtype=int, device=self.device)

    def apply_data(self, data):
        """Keep one deterministic claimant for each old contact."""
        wp.copy(self.snapshot, data.match_index)
        self.count.zero_()
        wp.launch(count_owners, self.matcher._capacity, inputs=[data, self.snapshot, self.count], device=self.device)
        wp.launch(
            resolve_unique,
            self.matcher._capacity,
            inputs=[data, self.snapshot, self.count, self.stats],
            device=self.device,
        )

    def apply(self, **kwargs):
        """Repair matches before any sticky geometry or solver history transfer."""
        self.apply_data(match_data(self.matcher, kwargs))
