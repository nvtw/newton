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

"""Tests for pair-key encoding, binary search, and WarmStarter."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.warm_start import (
    WarmStarter,
    _compute_pair_keys_kernel,
    binary_search_int64,
    make_pair_key,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


# ---------------------------------------------------------------------------
# Helper kernels for testing device functions
# ---------------------------------------------------------------------------


@wp.kernel
def _test_make_pair_key_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    out[tid] = make_pair_key(a[tid], b[tid])


@wp.kernel
def _test_binary_search_kernel(
    keys: wp.array(dtype=wp.int64),
    targets: wp.array(dtype=wp.int64),
    count: int,
    out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    out[tid] = binary_search_int64(keys, count, targets[tid])


@wp.kernel
def _write_impulses_kernel(
    impulse_n: wp.array(dtype=wp.float32),
    impulse_t1: wp.array(dtype=wp.float32),
    impulse_t2: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < count[0]:
        impulse_n[tid] = float(tid + 1) * 10.0
        impulse_t1[tid] = float(tid + 1) * 1.0
        impulse_t2[tid] = float(tid + 1) * 0.1


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestWarmStart(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Pair key tests
# ---------------------------------------------------------------------------


def test_pair_key_symmetry(test, device):
    """make_pair_key(a, b) == make_pair_key(b, a)."""
    a = wp.array([0, 1, 5, 100], dtype=wp.int32, device=device)
    b = wp.array([1, 0, 100, 5], dtype=wp.int32, device=device)
    out_ab = wp.zeros(4, dtype=wp.int64, device=device)
    out_ba = wp.zeros(4, dtype=wp.int64, device=device)

    wp.launch(_test_make_pair_key_kernel, dim=4, inputs=[a, b, out_ab], device=device)
    wp.launch(_test_make_pair_key_kernel, dim=4, inputs=[b, a, out_ba], device=device)
    wp.synchronize_device(device)

    np.testing.assert_array_equal(out_ab.numpy(), out_ba.numpy())


def test_pair_key_unique(test, device):
    """Different pairs produce different keys."""
    a = wp.array([0, 0, 1], dtype=wp.int32, device=device)
    b = wp.array([1, 2, 2], dtype=wp.int32, device=device)
    out = wp.zeros(3, dtype=wp.int64, device=device)

    wp.launch(_test_make_pair_key_kernel, dim=3, inputs=[a, b, out], device=device)
    wp.synchronize_device(device)

    keys = out.numpy()
    test.assertEqual(len(set(keys.tolist())), 3)


# ---------------------------------------------------------------------------
# Binary search tests
# ---------------------------------------------------------------------------


def test_binary_search_found(test, device):
    """Binary search finds all present keys."""
    sorted_keys = wp.array([10, 20, 30, 40, 50], dtype=wp.int64, device=device)
    targets = wp.array([10, 30, 50], dtype=wp.int64, device=device)
    out = wp.zeros(3, dtype=wp.int32, device=device)

    wp.launch(_test_binary_search_kernel, dim=3, inputs=[sorted_keys, targets, 5, out], device=device)
    wp.synchronize_device(device)

    result = out.numpy()
    test.assertEqual(result[0], 0)
    test.assertEqual(result[1], 2)
    test.assertEqual(result[2], 4)


def test_binary_search_missing(test, device):
    """Binary search returns -1 for absent keys."""
    sorted_keys = wp.array([10, 20, 30], dtype=wp.int64, device=device)
    targets = wp.array([5, 15, 35], dtype=wp.int64, device=device)
    out = wp.zeros(3, dtype=wp.int32, device=device)

    wp.launch(_test_binary_search_kernel, dim=3, inputs=[sorted_keys, targets, 3, out], device=device)
    wp.synchronize_device(device)

    result = out.numpy()
    for i in range(3):
        test.assertEqual(result[i], -1)


def test_binary_search_single(test, device):
    """Binary search works with a single element."""
    sorted_keys = wp.array([42], dtype=wp.int64, device=device)
    targets = wp.array([42, 43], dtype=wp.int64, device=device)
    out = wp.zeros(2, dtype=wp.int32, device=device)

    wp.launch(_test_binary_search_kernel, dim=2, inputs=[sorted_keys, targets, 1, out], device=device)
    wp.synchronize_device(device)

    result = out.numpy()
    test.assertEqual(result[0], 0)
    test.assertEqual(result[1], -1)


# ---------------------------------------------------------------------------
# WarmStarter tests
# ---------------------------------------------------------------------------


def test_warm_starter_basic_transfer(test, device):
    """Impulses from frame 1 transfer to matching contacts in frame 2."""
    cap = 16
    ws = WarmStarter(cap, device=device)

    # --- Frame 1: 3 contacts with pairs (0,1), (2,3), (4,5) ---
    shape0_f1 = wp.array([0, 2, 4], dtype=wp.int32, device=device)
    shape1_f1 = wp.array([1, 3, 5], dtype=wp.int32, device=device)
    count_f1 = wp.array([3], dtype=wp.int32, device=device)

    ws.import_keys(shape0_f1, shape1_f1, count_f1)
    ws.sort()

    # Simulate solver writing impulses
    impulse_n = wp.zeros(cap, dtype=wp.float32, device=device)
    impulse_t1 = wp.zeros(cap, dtype=wp.float32, device=device)
    impulse_t2 = wp.zeros(cap, dtype=wp.float32, device=device)
    wp.launch(_write_impulses_kernel, dim=cap, inputs=[impulse_n, impulse_t1, impulse_t2, count_f1], device=device)

    ws.export_impulses(impulse_n, impulse_t1, impulse_t2)

    # --- Frame 2: 2 contacts with pairs (2,3) [same] and (6,7) [new] ---
    ws.begin_frame()

    shape0_f2 = wp.array([2, 6], dtype=wp.int32, device=device)
    shape1_f2 = wp.array([3, 7], dtype=wp.int32, device=device)
    count_f2 = wp.array([2], dtype=wp.int32, device=device)

    ws.import_keys(shape0_f2, shape1_f2, count_f2)
    ws.sort()

    out_n = wp.zeros(cap, dtype=wp.float32, device=device)
    out_t1 = wp.zeros(cap, dtype=wp.float32, device=device)
    out_t2 = wp.zeros(cap, dtype=wp.float32, device=device)
    ws.transfer_impulses(out_n, out_t1, out_t2)
    wp.synchronize_device(device)

    out_n_np = out_n.numpy()
    out_t1_np = out_t1.numpy()

    # One of the two contacts should have received a nonzero impulse
    # (the one matching pair (2,3)), the other (6,7) should be zero.
    transferred = sorted(out_n_np[:2].tolist(), reverse=True)
    test.assertGreater(transferred[0], 0.0)
    test.assertAlmostEqual(transferred[1], 0.0, places=5)


def test_warm_starter_no_previous(test, device):
    """First frame gets zero impulses (no previous data)."""
    cap = 8
    ws = WarmStarter(cap, device=device)

    shape0 = wp.array([0, 1], dtype=wp.int32, device=device)
    shape1 = wp.array([2, 3], dtype=wp.int32, device=device)
    count = wp.array([2], dtype=wp.int32, device=device)

    ws.import_keys(shape0, shape1, count)
    ws.sort()

    out_n = wp.zeros(cap, dtype=wp.float32, device=device)
    out_t1 = wp.zeros(cap, dtype=wp.float32, device=device)
    out_t2 = wp.zeros(cap, dtype=wp.float32, device=device)
    ws.transfer_impulses(out_n, out_t1, out_t2)
    wp.synchronize_device(device)

    test.assertTrue(np.all(out_n.numpy()[:2] == 0.0))
    test.assertTrue(np.all(out_t1.numpy()[:2] == 0.0))
    test.assertTrue(np.all(out_t2.numpy()[:2] == 0.0))


def test_warm_starter_reversed_pair_order(test, device):
    """Warm start matches even when shape order is flipped between frames."""
    cap = 8
    ws = WarmStarter(cap, device=device)

    # Frame 1: pair (0, 1)
    shape0_f1 = wp.array([0], dtype=wp.int32, device=device)
    shape1_f1 = wp.array([1], dtype=wp.int32, device=device)
    count_f1 = wp.array([1], dtype=wp.int32, device=device)

    ws.import_keys(shape0_f1, shape1_f1, count_f1)
    ws.sort()

    impulse_n = wp.zeros(cap, dtype=wp.float32, device=device)
    impulse_n_np = impulse_n.numpy()
    impulse_n_np[0] = 42.0
    impulse_n.assign(wp.array(impulse_n_np, dtype=wp.float32, device=device))
    impulse_t1 = wp.zeros(cap, dtype=wp.float32, device=device)
    impulse_t2 = wp.zeros(cap, dtype=wp.float32, device=device)
    ws.export_impulses(impulse_n, impulse_t1, impulse_t2)

    # Frame 2: pair (1, 0) -- reversed
    ws.begin_frame()
    shape0_f2 = wp.array([1], dtype=wp.int32, device=device)
    shape1_f2 = wp.array([0], dtype=wp.int32, device=device)
    count_f2 = wp.array([1], dtype=wp.int32, device=device)

    ws.import_keys(shape0_f2, shape1_f2, count_f2)
    ws.sort()

    out_n = wp.zeros(cap, dtype=wp.float32, device=device)
    out_t1 = wp.zeros(cap, dtype=wp.float32, device=device)
    out_t2 = wp.zeros(cap, dtype=wp.float32, device=device)
    ws.transfer_impulses(out_n, out_t1, out_t2)
    wp.synchronize_device(device)

    test.assertAlmostEqual(out_n.numpy()[0], 42.0, places=4)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestWarmStart, "test_pair_key_symmetry", test_pair_key_symmetry, devices=devices)
add_function_test(TestWarmStart, "test_pair_key_unique", test_pair_key_unique, devices=devices)
add_function_test(TestWarmStart, "test_binary_search_found", test_binary_search_found, devices=devices)
add_function_test(TestWarmStart, "test_binary_search_missing", test_binary_search_missing, devices=devices)
add_function_test(TestWarmStart, "test_binary_search_single", test_binary_search_single, devices=devices)
add_function_test(TestWarmStart, "test_warm_starter_basic_transfer", test_warm_starter_basic_transfer, devices=devices)
add_function_test(TestWarmStart, "test_warm_starter_no_previous", test_warm_starter_no_previous, devices=devices)
add_function_test(TestWarmStart, "test_warm_starter_reversed_pair_order", test_warm_starter_reversed_pair_order, devices=devices)

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
