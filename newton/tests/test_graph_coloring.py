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

"""Tests for the Luby-MIS graph coloring module."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.maximal_independent_set import GraphColoring
from newton.tests.unittest_utils import add_function_test, get_test_devices

MAX_BODIES = 8


def _make_elements(rows, max_elements, device):
    """Build a (max_elements, 8) int32 array from a list of body-id lists."""
    arr = np.full((max_elements, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows):
        for j, b in enumerate(bodies):
            arr[i, j] = b
    return wp.array(arr, dtype=wp.int32, device=device).reshape((max_elements, MAX_BODIES))


def _validate_independence(test, gc, elements_np, n_elements):
    """Assert that each partition is an independent set (no shared body)."""
    pd = gc.partition_data.numpy()[:n_elements]
    pe = gc.partition_ends.numpy()
    np_val = gc.num_partitions.numpy()[0]
    ha = gc.has_additional.numpy()[0]
    n_parts = np_val + (1 if ha else 0)

    for p in range(n_parts):
        start = 0 if p == 0 else pe[p - 1]
        end = pe[p]
        bodies_seen = set()
        for idx in range(start, end):
            elem_id = pd[idx]
            row = elements_np[elem_id]
            for b in row:
                if b < 0:
                    break
                test.assertNotIn(b, bodies_seen, f"Partition {p}: body {b} duplicated (element {elem_id})")
                bodies_seen.add(b)


def _validate_coverage(test, gc, n_elements):
    """Assert that every element appears in exactly one partition."""
    pd = gc.partition_data.numpy()[:n_elements]
    test.assertEqual(sorted(pd), list(range(n_elements)))


class TestGraphColoring(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_disjoint_elements(test, device):
    """Elements with no shared bodies should all land in partition 0."""
    max_el, max_nd = 16, 16
    rows = [[0, 1], [2, 3], [4, 5], [6, 7]]
    n_el = len(rows)
    elements_np = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows):
        for j, b in enumerate(bodies):
            elements_np[i, j] = b
    elements = _make_elements(rows, max_el, device)

    gc = GraphColoring(max_el, max_nd, max_colors=8, device=device)
    gc.color(
        elements,
        wp.array([n_el], dtype=wp.int32, device=device),
        wp.array([8], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)

    _validate_independence(test, gc, elements_np, n_el)
    _validate_coverage(test, gc, n_el)
    # All disjoint => 1 partition holding all 4 elements
    test.assertEqual(gc.num_partitions.numpy()[0], 1)


def test_chain(test, device):
    """Linear chain: 0-1, 1-2, 2-3, 3-4 needs at least 2 colours."""
    max_el, max_nd = 16, 8
    rows = [[0, 1], [1, 2], [2, 3], [3, 4]]
    n_el = len(rows)
    elements_np = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows):
        for j, b in enumerate(bodies):
            elements_np[i, j] = b
    elements = _make_elements(rows, max_el, device)

    gc = GraphColoring(max_el, max_nd, max_colors=8, device=device)
    gc.color(
        elements,
        wp.array([n_el], dtype=wp.int32, device=device),
        wp.array([5], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)

    _validate_independence(test, gc, elements_np, n_el)
    _validate_coverage(test, gc, n_el)
    test.assertGreaterEqual(gc.num_partitions.numpy()[0], 2)


def test_clique(test, device):
    """Fully connected: 3 elements all sharing body 0 need 3 colours."""
    max_el, max_nd = 16, 8
    rows = [[0, 1], [0, 2], [0, 3]]
    n_el = len(rows)
    elements_np = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows):
        for j, b in enumerate(bodies):
            elements_np[i, j] = b
    elements = _make_elements(rows, max_el, device)

    gc = GraphColoring(max_el, max_nd, max_colors=8, device=device)
    gc.color(
        elements,
        wp.array([n_el], dtype=wp.int32, device=device),
        wp.array([4], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)

    _validate_independence(test, gc, elements_np, n_el)
    _validate_coverage(test, gc, n_el)
    test.assertEqual(gc.num_partitions.numpy()[0], 3)


def test_determinism(test, device):
    """Two runs with identical input produce identical output."""
    max_el, max_nd = 32, 16
    rng = np.random.RandomState(42)
    n_el = 20
    rows = []
    for _ in range(n_el):
        k = rng.randint(2, 5)
        rows.append(sorted(rng.choice(16, size=k, replace=False).tolist()))
    elements = _make_elements(rows, max_el, device)

    def run():
        gc = GraphColoring(max_el, max_nd, max_colors=8, device=device)
        gc.color(
            elements,
            wp.array([n_el], dtype=wp.int32, device=device),
            wp.array([max_nd], dtype=wp.int32, device=device),
        )
        wp.synchronize_device(device)
        return (
            gc.partition_data.numpy()[:n_el].copy(),
            gc.partition_ends.numpy().copy(),
            gc.num_partitions.numpy()[0],
        )

    pd1, pe1, np1 = run()
    pd2, pe2, np2 = run()
    test.assertTrue(np.array_equal(pd1, pd2))
    test.assertTrue(np.array_equal(pe1, pe2))
    test.assertEqual(np1, np2)


def test_full_coverage_random(test, device):
    """Random graph: every element is covered and partitions are independent."""
    max_el, max_nd = 64, 32
    rng = np.random.RandomState(123)
    n_el = 50
    rows = []
    for _ in range(n_el):
        k = rng.randint(2, 5)
        rows.append(sorted(rng.choice(32, size=k, replace=False).tolist()))
    elements_np = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows):
        for j, b in enumerate(bodies):
            elements_np[i, j] = b
    elements = _make_elements(rows, max_el, device)

    gc = GraphColoring(max_el, max_nd, max_colors=16, device=device)
    gc.color(
        elements,
        wp.array([n_el], dtype=wp.int32, device=device),
        wp.array([max_nd], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)

    _validate_independence(test, gc, elements_np, n_el)
    _validate_coverage(test, gc, n_el)


def test_single_element(test, device):
    """A single element should result in 1 partition."""
    max_el, max_nd = 8, 4
    rows = [[0, 1]]
    n_el = 1
    elements_np = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    elements_np[0, :2] = [0, 1]
    elements = _make_elements(rows, max_el, device)

    gc = GraphColoring(max_el, max_nd, max_colors=4, device=device)
    gc.color(
        elements,
        wp.array([n_el], dtype=wp.int32, device=device),
        wp.array([2], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)

    _validate_independence(test, gc, elements_np, n_el)
    _validate_coverage(test, gc, n_el)
    test.assertEqual(gc.num_partitions.numpy()[0], 1)


def test_reinit(test, device):
    """Re-running color() with new data produces correct results."""
    max_el, max_nd = 16, 8
    gc = GraphColoring(max_el, max_nd, max_colors=8, device=device)

    # Run 1: chain
    rows1 = [[0, 1], [1, 2], [2, 3]]
    elements_np1 = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows1):
        for j, b in enumerate(bodies):
            elements_np1[i, j] = b
    gc.color(
        _make_elements(rows1, max_el, device),
        wp.array([3], dtype=wp.int32, device=device),
        wp.array([4], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)
    _validate_independence(test, gc, elements_np1, 3)

    # Run 2: disjoint
    rows2 = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elements_np2 = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows2):
        for j, b in enumerate(bodies):
            elements_np2[i, j] = b
    gc.color(
        _make_elements(rows2, max_el, device),
        wp.array([4], dtype=wp.int32, device=device),
        wp.array([8], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)
    _validate_independence(test, gc, elements_np2, 4)
    _validate_coverage(test, gc, 4)
    test.assertEqual(gc.num_partitions.numpy()[0], 1)


def test_max8_bodies(test, device):
    """Element connecting all 8 bodies works correctly."""
    max_el, max_nd = 8, 16
    rows = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 8],
        [7, 9],
    ]
    n_el = 3
    elements_np = np.full((max_el, MAX_BODIES), -1, dtype=np.int32)
    for i, bodies in enumerate(rows):
        for j, b in enumerate(bodies):
            elements_np[i, j] = b
    elements = _make_elements(rows, max_el, device)

    gc = GraphColoring(max_el, max_nd, max_colors=8, device=device)
    gc.color(
        elements,
        wp.array([n_el], dtype=wp.int32, device=device),
        wp.array([10], dtype=wp.int32, device=device),
    )
    wp.synchronize_device(device)

    _validate_independence(test, gc, elements_np, n_el)
    _validate_coverage(test, gc, n_el)
    # Element 0 overlaps with both 1 (body 0) and 2 (body 7), so >= 2 colours
    test.assertGreaterEqual(gc.num_partitions.numpy()[0], 2)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestGraphColoring, "test_disjoint_elements", test_disjoint_elements, devices=devices)
add_function_test(TestGraphColoring, "test_chain", test_chain, devices=devices)
add_function_test(TestGraphColoring, "test_clique", test_clique, devices=devices)
add_function_test(TestGraphColoring, "test_determinism", test_determinism, devices=devices)
add_function_test(TestGraphColoring, "test_full_coverage_random", test_full_coverage_random, devices=devices)
add_function_test(TestGraphColoring, "test_single_element", test_single_element, devices=devices)
add_function_test(TestGraphColoring, "test_reinit", test_reinit, devices=devices)
add_function_test(TestGraphColoring, "test_max8_bodies", test_max8_bodies, devices=devices)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
