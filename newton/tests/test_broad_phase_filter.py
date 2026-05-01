# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the broadphase ``filter_func`` callback.

Mirrors the ``contact_writer_warp_func`` pattern in NarrowPhase: a
``@wp.func`` is closed into the broadphase kernel at factory time, and
a runtime ``filter_data`` ``wp.struct`` is supplied via ``launch()``.
The filter runs after the AABB overlap accepts a pair and before
``write_pair`` -- returning ``0`` drops the pair.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP


# ``filter_data`` carrier consumed by every test below. The filter
# semantics: a pair is dropped when both shape indices are ``>=
# tri_threshold`` (i.e. simulating "both sides are cloth triangles")
# AND the second component (``aux``) of the per-shape lookup matches
# (i.e. simulating "share at least one node"). This keeps the test
# focused on the broadphase plumbing without dragging in the full
# phoenx tri-indices schema.
@wp.struct
class _TestFilterData:
    tri_threshold: wp.int32
    shape_aux: wp.array[wp.int32]


@wp.func
def _drop_paired_aux(pair: wp.vec2i, ud: _TestFilterData) -> wp.int32:
    a = pair[0]
    b = pair[1]
    if a >= ud.tri_threshold and b >= ud.tri_threshold:
        if ud.shape_aux[a] == ud.shape_aux[b]:
            return wp.int32(0)
    return wp.int32(1)


def _build_aabbs_overlap_all(n: int, device) -> tuple[wp.array, wp.array]:
    """N overlapping unit cubes centred at the origin."""
    lower = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    upper = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    return lower, upper


class TestBroadphaseFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()
        wp.init()

    def _expected_pairs_post_filter(self, n: int, tri_threshold: int, aux: list[int]) -> set[tuple[int, int]]:
        out = set()
        for i in range(n):
            for j in range(i + 1, n):
                if i >= tri_threshold and j >= tri_threshold and aux[i] == aux[j]:
                    continue
                out.add((i, j))
        return out

    def _read_pairs(self, candidate: wp.array, count: wp.array) -> set[tuple[int, int]]:
        n = int(count.numpy()[0])
        if n == 0:
            return set()
        arr = candidate.numpy()
        out = set()
        for k in range(n):
            a = int(arr[k][0])
            b = int(arr[k][1])
            if a > b:
                a, b = b, a
            out.add((a, b))
        return out

    def test_nxn_filter_drops_paired_aux(self) -> None:
        n = 5
        tri_threshold = 2
        aux_np = np.array([0, 0, 7, 7, 9], dtype=np.int32)  # shapes 2,3 share aux=7
        aux = wp.array(aux_np, dtype=wp.int32, device=self.device)

        shape_world = np.zeros(n, dtype=np.int32)
        bp = BroadPhaseAllPairs(shape_world, device=self.device, filter_func=_drop_paired_aux)

        lower, upper = _build_aabbs_overlap_all(n, self.device)
        # Positive collision groups so equal-group shapes interact.
        groups = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(n * (n - 1) // 2, dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)

        fdata = _TestFilterData()
        fdata.tri_threshold = tri_threshold
        fdata.shape_aux = aux

        bp.launch(
            shape_lower=lower,
            shape_upper=upper,
            shape_gap=None,
            shape_collision_group=groups,
            shape_world=worlds,
            shape_count=n,
            candidate_pair=candidate,
            candidate_pair_count=count,
            device=self.device,
            filter_data=fdata,
        )
        pairs = self._read_pairs(candidate, count)
        expected = self._expected_pairs_post_filter(n, tri_threshold, aux_np.tolist())
        self.assertEqual(pairs, expected)
        # Sanity: the unfiltered enumeration would emit all C(5,2)=10 pairs.
        self.assertEqual(len(pairs), 9)

    def test_sap_filter_drops_paired_aux(self) -> None:
        n = 5
        tri_threshold = 2
        aux_np = np.array([0, 0, 7, 7, 9], dtype=np.int32)
        aux = wp.array(aux_np, dtype=wp.int32, device=self.device)

        shape_world = np.zeros(n, dtype=np.int32)
        bp = BroadPhaseSAP(shape_world, device=self.device, filter_func=_drop_paired_aux)

        lower, upper = _build_aabbs_overlap_all(n, self.device)
        # Positive collision groups so equal-group shapes interact.
        groups = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(n * (n - 1) // 2, dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)

        fdata = _TestFilterData()
        fdata.tri_threshold = tri_threshold
        fdata.shape_aux = aux

        bp.launch(
            shape_lower=lower,
            shape_upper=upper,
            shape_gap=None,
            shape_collision_group=groups,
            shape_world=worlds,
            shape_count=n,
            candidate_pair=candidate,
            candidate_pair_count=count,
            device=self.device,
            filter_data=fdata,
        )
        pairs = self._read_pairs(candidate, count)
        expected = self._expected_pairs_post_filter(n, tri_threshold, aux_np.tolist())
        self.assertEqual(pairs, expected)

    def test_explicit_filter_drops_paired_aux(self) -> None:
        n = 5
        tri_threshold = 2
        aux_np = np.array([0, 0, 7, 7, 9], dtype=np.int32)
        aux = wp.array(aux_np, dtype=wp.int32, device=self.device)

        bp = BroadPhaseExplicit(filter_func=_drop_paired_aux)

        # Hand-built explicit pair list: every (i, j) with i < j.
        pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pairs_in = wp.array(np.array(pair_list, dtype=np.int32), dtype=wp.vec2i, device=self.device)

        lower, upper = _build_aabbs_overlap_all(n, self.device)
        candidate = wp.zeros(len(pair_list), dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)

        fdata = _TestFilterData()
        fdata.tri_threshold = tri_threshold
        fdata.shape_aux = aux

        bp.launch(
            shape_lower=lower,
            shape_upper=upper,
            shape_gap=None,
            shape_pairs=pairs_in,
            shape_pair_count=len(pair_list),
            candidate_pair=candidate,
            candidate_pair_count=count,
            device=self.device,
            filter_data=fdata,
        )
        pairs = self._read_pairs(candidate, count)
        expected = self._expected_pairs_post_filter(n, tri_threshold, aux_np.tolist())
        self.assertEqual(pairs, expected)

    def test_filter_data_required_when_filter_registered(self) -> None:
        shape_world = np.zeros(2, dtype=np.int32)
        bp = BroadPhaseAllPairs(shape_world, device=self.device, filter_func=_drop_paired_aux)
        lower, upper = _build_aabbs_overlap_all(2, self.device)
        groups = wp.array(np.full(2, -1, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(1, dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)
        with self.assertRaises(ValueError):
            bp.launch(
                shape_lower=lower,
                shape_upper=upper,
                shape_gap=None,
                shape_collision_group=groups,
                shape_world=worlds,
                shape_count=2,
                candidate_pair=candidate,
                candidate_pair_count=count,
                device=self.device,
            )


if __name__ == "__main__":
    unittest.main()
