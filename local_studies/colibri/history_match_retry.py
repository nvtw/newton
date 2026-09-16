# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Retry unclaimed eligible friction identities without changing sticky geometry.

Existing winners remain fixed. Each round uses the existing physical eligibility
and packed distance/key arbitration, excluding occupied predecessor contacts.
"""

import inspect
import os
import sys
import tempfile
import types
from pathlib import Path

import warp as wp

from newton._src.geometry import contact_match as cm


@wp.kernel
def initialize(
    data: cm._MatchData, occupied: wp.array[int], keep: wp.array[int], stats: wp.array[int], active: wp.array[int]
):
    k = wp.tid()
    if k == 0:
        active[0] = 1
    keep[k] = 0
    if k < data.new_count[0] and data.match_index[k] >= 0:
        old = data.match_index[k]
        wp.atomic_add(occupied, old, 1)
        keep[k] = 1
        wp.atomic_add(stats, 0, 1)
    elif k < data.new_count[0]:
        wp.atomic_add(stats, 1, 1)


@wp.kernel
def clear_round(data: cm._MatchData, owner: wp.array[int], active: wp.array[int], stats: wp.array[int]):
    k = wp.tid()
    data.prev_claim[k] = wp.int64(0x7FFFFFFFFFFFFFFF)
    owner[k] = wp.int32(0x7FFFFFFF)
    if k == 0:
        active[0] = 0
        stats[3] += 1


@wp.kernel
def elect(data: cm._MatchData, keep: wp.array[int], claim: wp.array[wp.int64], owner: wp.array[int]):
    k = wp.tid()
    if k < data.new_count[0] and keep[k] == 0:
        old = data.match_index[k]
        if old >= 0 and claim[k] == data.prev_claim[old]:
            winner = wp.bool(True)
            for other in range(wp.min(data.new_count[0], keep.shape[0])):
                if other != k and keep[other] == 0 and data.match_index[other] == old and claim[other] == claim[k]:
                    if cm._contact_geometry_precedes(data, other, k):
                        winner = False
            if winner:
                wp.atomic_min(owner, old, k)


@wp.kernel
def resolve(
    data: cm._MatchData,
    keep: wp.array[int],
    occupied: wp.array[int],
    owner: wp.array[int],
    active: wp.array[int],
    stats: wp.array[int],
):
    k = wp.tid()
    if k < data.new_count[0] and keep[k] == 0:
        old = data.match_index[k]
        if old >= 0:
            if owner[old] == k:
                keep[k] = 1
                occupied[old] = 1
                wp.atomic_add(active, 0, 1)
                wp.atomic_add(stats, 2, 1)
            else:
                data.match_index[k] = cm.MATCH_BROKEN


@wp.kernel
def validate(occupied: wp.array[int], stats: wp.array[int]):
    if occupied[wp.tid()] > 1:
        wp.atomic_add(stats, 4, 1)


def candidate_kernel(*, dormant):
    """Reuse actual eligibility source, changing only candidate occupancy."""
    source = inspect.getsource(cm._match_contacts_kernel.func)
    source = source.replace(
        "def _match_contacts_kernel(data: _MatchData):",
        "def retry_candidates(data: _MatchData, occupied: wp.array[int], keep: wp.array[int], claim: wp.array[wp.int64]):",
    )
    source = source.replace(
        "    n_new = data.new_count[0]", "    if keep[tid] != 0:\n        return\n    n_new = data.new_count[0]", 1
    )
    source = source.replace(
        "        old_pos = data.prev_pos_world[old_idx]",
        "        if occupied[old_idx] != 0:\n            continue\n        old_pos = data.prev_pos_world[old_idx]",
        1,
    )
    marker = "        wp.atomic_min(data.prev_claim, best_idx, _pack_claim(best_dist_sq, target_key))"
    assert source.count(marker) == 1
    source = source.replace(marker, "        claim[tid] = _pack_claim(best_dist_sq, target_key)\n" + marker)
    if dormant:
        marker = "fresh_gap > wp.float32(0.0)"
        assert source.count(marker) == 1
        source = source.replace(marker, "fresh_gap > wp.sqrt(data.pos_threshold_sq)")
    name = "colibri_history_retry_" + ("dormant" if dormant else "strict")
    module = types.ModuleType(name)
    module.__dict__.update(vars(cm))
    module.__name__ = name
    path = Path(tempfile.mkdtemp(prefix="colibri_history_retry_")) / (name + ".py")
    path.write_text(source)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.retry_candidates


class Retry:
    """Allocate once, then retry unmatched identities on CPU or captured CUDA."""

    def __init__(self, matcher, *, dormant=False, enabled=True):
        self.enabled = enabled
        self.matcher = matcher
        self.device = matcher._prev_count.device
        n = matcher._capacity
        self.occupied = wp.zeros(n, dtype=int, device=self.device)
        self.keep = wp.zeros(n, dtype=int, device=self.device)
        self.owner = wp.zeros(n, dtype=int, device=self.device)
        self.claim = wp.zeros(n, dtype=wp.int64, device=self.device)
        self.active = wp.zeros(1, dtype=int, device=self.device)
        self.stats = wp.zeros(5, dtype=int, device=self.device)
        self.original = wp.zeros(n, dtype=int, device=self.device)
        self.candidate = candidate_kernel(dormant=dormant)
        self.events = wp.zeros((128, 8), dtype=int, device=self.device)
        self.event_count = wp.zeros(1, dtype=int, device=self.device)
        self.snapshot = {}
        for name, dtype in {
            "prev_keys": wp.int64,
            "prev_pos_world": wp.vec3,
            "prev_normal": wp.vec3,
            "new_keys": wp.int64,
            "new_point0": wp.vec3,
            "new_point1": wp.vec3,
            "new_normal": wp.vec3,
            "new_shape0": int,
            "new_shape1": int,
            "canonical_to_source": int,
            "match_index": int,
        }.items():
            self.snapshot[name] = wp.empty(n, dtype=dtype, device=self.device)

    def apply_data(self, data):
        """Apply to the original actual matcher data; expose this for tests."""
        n = self.matcher._capacity
        wp.launch(begin_event, 1, inputs=[self.stats, self.events, self.event_count], device=self.device)
        wp.copy(self.original, data.match_index)
        self.occupied.zero_()
        wp.launch(initialize, n, inputs=[data, self.occupied, self.keep, self.stats, self.active], device=self.device)
        wp.launch(validate, n, inputs=[self.occupied, self.stats], device=self.device)

        def round_():
            wp.launch(clear_round, n, inputs=[data, self.owner, self.active, self.stats], device=self.device)
            wp.launch(self.candidate, n, inputs=[data, self.occupied, self.keep, self.claim], device=self.device)
            wp.launch(elect, n, inputs=[data, self.keep, self.claim, self.owner], device=self.device)
            wp.launch(
                resolve,
                n,
                inputs=[data, self.keep, self.occupied, self.owner, self.active, self.stats],
                device=self.device,
            )

        if self.enabled:
            wp.capture_while(self.active, round_)
        wp.launch(end_event, 1, inputs=[self.stats, self.events, self.event_count, data], device=self.device)
        for name, target in self.snapshot.items():
            wp.copy(target, getattr(data, name))

    def apply(self, **kwargs):
        """Build the same data bundle as ContactMatcher.match."""
        matcher = self.matcher
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
        self.apply_data(data)


def install(pipeline):
    """Install after the baseline dormant-history matcher; preserve geometry map."""
    matcher = pipeline._contact_matcher
    retry = Retry(matcher, dormant=True, enabled=os.environ.get("COLIBRI_HISTORY_RETRY", "1") == "1")
    original = matcher.match

    def match(**kwargs):
        original(**kwargs)
        retry.apply(**kwargs)

    matcher.match = match
    return retry


@wp.kernel
def begin_event(stats: wp.array[int], events: wp.array2d[int], count: wp.array[int]):
    slot = count[0] % events.shape[0]
    for j in range(5):
        events[slot, j] = -stats[j]


@wp.kernel
def end_event(stats: wp.array[int], events: wp.array2d[int], count: wp.array[int], data: cm._MatchData):
    slot = count[0] % events.shape[0]
    for j in range(5):
        events[slot, j] += stats[j]
    events[slot, 5] = data.prev_count[0]
    events[slot, 6] = data.new_count[0]
    events[slot, 7] = count[0]
    count[0] += 1
