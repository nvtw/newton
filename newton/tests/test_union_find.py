# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the GPU-parallel Union-Find module."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.union_find import UnionFind
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestUnionFind(unittest.TestCase):
    pass


def test_singleton_roots(test, device):
    """Every element in an untouched UF is its own root."""
    capacity = 64
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    expected = np.arange(capacity, dtype=np.int32)
    test.assertTrue(np.array_equal(roots, expected))


def test_chain_union(test, device):
    """Uniting a chain 0-1-2-3 gives all four the same root."""
    capacity = 16
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)

    pairs_np = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([len(pairs_np)], dtype=wp.int32, device=device)

    uf.unite_pairs(pairs, pair_count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    chain_root = roots[0]
    for i in range(4):
        test.assertEqual(roots[i], chain_root, f"Element {i} root mismatch")

    for i in range(4, capacity):
        test.assertEqual(roots[i], i, f"Element {i} should remain singleton")


def test_two_disjoint_groups(test, device):
    """Two separate groups stay disjoint."""
    capacity = 8
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)

    pairs_np = np.array([[0, 1], [1, 2], [4, 5], [5, 6]], dtype=np.int32)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([len(pairs_np)], dtype=wp.int32, device=device)

    uf.unite_pairs(pairs, pair_count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    test.assertEqual(roots[0], roots[1])
    test.assertEqual(roots[1], roots[2])
    test.assertEqual(roots[4], roots[5])
    test.assertEqual(roots[5], roots[6])
    test.assertNotEqual(roots[0], roots[4])
    test.assertEqual(roots[3], 3)
    test.assertEqual(roots[7], 7)


def test_merge_groups(test, device):
    """Two groups merged into one via a bridge pair."""
    capacity = 8
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)

    pairs_np = np.array([[0, 1], [2, 3], [1, 2]], dtype=np.int32)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([len(pairs_np)], dtype=wp.int32, device=device)

    uf.unite_pairs(pairs, pair_count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    test.assertEqual(roots[0], roots[1])
    test.assertEqual(roots[1], roots[2])
    test.assertEqual(roots[2], roots[3])


def test_idempotent_unite(test, device):
    """Uniting the same pair multiple times is harmless."""
    capacity = 4
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)

    pairs_np = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int32)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([len(pairs_np)], dtype=wp.int32, device=device)

    uf.unite_pairs(pairs, pair_count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    test.assertEqual(roots[0], roots[1])
    test.assertEqual(roots[2], 2)
    test.assertEqual(roots[3], 3)


def test_reinit(test, device):
    """Re-initialising resets all unions."""
    capacity = 8
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)
    pairs_np = np.array([[0, 1], [2, 3]], dtype=np.int32)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([len(pairs_np)], dtype=wp.int32, device=device)
    uf.unite_pairs(pairs, pair_count)

    uf.init(count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    expected = np.arange(capacity, dtype=np.int32)
    test.assertTrue(np.array_equal(roots, expected))


def test_large_scale(test, device):
    """Stress test: unite 10k elements into a single component."""
    capacity = 10_000
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)

    uf.init(count)

    n_pairs = capacity - 1
    pairs_np = np.stack([np.arange(n_pairs, dtype=np.int32), np.arange(1, capacity, dtype=np.int32)], axis=-1)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([n_pairs], dtype=wp.int32, device=device)

    uf.unite_pairs(pairs, pair_count)
    uf.find_roots(count)

    roots = uf.roots.numpy()
    test.assertEqual(len(np.unique(roots)), 1)


def test_graph_capture(test, device):
    """The full init/unite/find sequence is capturable as a CUDA graph."""
    if not wp.get_device(device).is_cuda:
        return

    capacity = 32
    uf = UnionFind(capacity, device=device)
    count = wp.array([capacity], dtype=wp.int32, device=device)
    pairs_np = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    pair_count = wp.array([len(pairs_np)], dtype=wp.int32, device=device)

    with wp.ScopedCapture(device=device) as capture:
        uf.init(count)
        uf.unite_pairs(pairs, pair_count)
        uf.find_roots(count)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    roots = uf.roots.numpy()
    test.assertEqual(roots[0], roots[1])
    test.assertEqual(roots[1], roots[2])
    test.assertEqual(roots[3], roots[4])
    test.assertNotEqual(roots[0], roots[3])


devices = get_test_devices()

add_function_test(TestUnionFind, "test_singleton_roots", test_singleton_roots, devices=devices)
add_function_test(TestUnionFind, "test_chain_union", test_chain_union, devices=devices)
add_function_test(TestUnionFind, "test_two_disjoint_groups", test_two_disjoint_groups, devices=devices)
add_function_test(TestUnionFind, "test_merge_groups", test_merge_groups, devices=devices)
add_function_test(TestUnionFind, "test_idempotent_unite", test_idempotent_unite, devices=devices)
add_function_test(TestUnionFind, "test_reinit", test_reinit, devices=devices)
add_function_test(TestUnionFind, "test_large_scale", test_large_scale, devices=devices)
add_function_test(TestUnionFind, "test_graph_capture", test_graph_capture, devices=devices)

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
