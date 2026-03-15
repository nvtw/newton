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

"""Tests for the DataStore / HandleStore column-major storage system."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.data_base import DataStore, HandleStore, Schema
from newton.tests.unittest_utils import add_function_test, get_test_devices

# ---------------------------------------------------------------------------
# Test schema struct
# ---------------------------------------------------------------------------


@wp.struct
class SimpleSchema:
    position: wp.vec3
    velocity: wp.vec3
    mass: wp.float32
    flags: wp.int32


@wp.struct
class QuatSchema:
    orientation: wp.quat
    scale: wp.float32


# ---------------------------------------------------------------------------
# Helper kernels (defined at module scope so Warp can compile them)
# ---------------------------------------------------------------------------


@wp.kernel
def _write_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < count[0]:
        positions[tid] = wp.vec3(float(tid), float(tid * 10), float(tid * 100))


@wp.kernel
def _write_mass_kernel(
    mass: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < count[0]:
        mass[tid] = float(tid) + 0.5


@wp.kernel
def _write_quat_kernel(
    orientations: wp.array(dtype=wp.quat),
    count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < count[0]:
        orientations[tid] = wp.quat(float(tid), 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDataBase(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_schema_offsets(test, device):
    """Field offsets and total floats_per_row are computed correctly."""
    s = Schema(SimpleSchema)
    test.assertEqual(s.floats_per_row, 3 + 3 + 1 + 1)  # vec3 + vec3 + float + int
    test.assertEqual(s.fields["position"].col_offset, 0)
    test.assertEqual(s.fields["position"].float_width, 3)
    test.assertEqual(s.fields["velocity"].col_offset, 3)
    test.assertEqual(s.fields["velocity"].float_width, 3)
    test.assertEqual(s.fields["mass"].col_offset, 6)
    test.assertEqual(s.fields["mass"].float_width, 1)
    test.assertEqual(s.fields["flags"].col_offset, 7)
    test.assertEqual(s.fields["flags"].float_width, 1)


def test_schema_quat(test, device):
    """Quaternion fields have width 4."""
    s = Schema(QuatSchema)
    test.assertEqual(s.floats_per_row, 4 + 1)
    test.assertEqual(s.fields["orientation"].float_width, 4)
    test.assertEqual(s.fields["scale"].col_offset, 4)


# ---------------------------------------------------------------------------
# DataStore tests
# ---------------------------------------------------------------------------


def test_datastore_column_of_write_read(test, device):
    """Write via a column view and read back matching values."""
    capacity = 16
    ds = DataStore(SimpleSchema, capacity, device=device)

    n = 8
    ds.count.assign(wp.array([n], dtype=wp.int32, device=device))

    pos = ds.column_of("position")
    wp.launch(_write_positions_kernel, dim=capacity, inputs=[pos, ds.count], device=device)

    mass = ds.column_of("mass")
    wp.launch(_write_mass_kernel, dim=capacity, inputs=[mass, ds.count], device=device)
    wp.synchronize_device(device)

    pos_np = ds.column_of("position").numpy()
    for i in range(n):
        test.assertAlmostEqual(pos_np[i][0], float(i), places=5)
        test.assertAlmostEqual(pos_np[i][1], float(i * 10), places=5)
        test.assertAlmostEqual(pos_np[i][2], float(i * 100), places=5)

    mass_np = ds.column_of("mass").numpy()
    for i in range(n):
        test.assertAlmostEqual(mass_np[i], float(i) + 0.5, places=5)


def test_datastore_clear(test, device):
    """clear() zeros data and resets count."""
    ds = DataStore(SimpleSchema, 8, device=device)
    ds.count.assign(wp.array([5], dtype=wp.int32, device=device))
    mass = ds.column_of("mass")
    wp.launch(_write_mass_kernel, dim=8, inputs=[mass, ds.count], device=device)
    wp.synchronize_device(device)

    ds.clear()
    test.assertEqual(ds.count.numpy()[0], 0)
    test.assertTrue(np.all(ds.data.numpy() == 0.0))


def test_datastore_quat_column(test, device):
    """Quaternion column views work correctly."""
    capacity = 4
    ds = DataStore(QuatSchema, capacity, device=device)
    ds.count.assign(wp.array([capacity], dtype=wp.int32, device=device))

    ori = ds.column_of("orientation")
    wp.launch(_write_quat_kernel, dim=capacity, inputs=[ori, ds.count], device=device)
    wp.synchronize_device(device)

    ori_np = ds.column_of("orientation").numpy()
    for i in range(capacity):
        test.assertAlmostEqual(ori_np[i][0], float(i), places=5)
        test.assertAlmostEqual(ori_np[i][3], 1.0, places=5)


# ---------------------------------------------------------------------------
# HandleStore tests
# ---------------------------------------------------------------------------


def test_handlestore_allocate(test, device):
    """Allocating handles returns sequential IDs and sets count."""
    hs = HandleStore(SimpleSchema, 8, device=device)
    handles = [hs.allocate() for _ in range(4)]
    test.assertEqual(len(set(handles)), 4)
    for h in handles:
        test.assertGreaterEqual(h, 0)
    test.assertEqual(hs.count.numpy()[0], 4)


def test_handlestore_remove(test, device):
    """Removing a handle decrements count and invalidates mapping."""
    hs = HandleStore(SimpleSchema, 8, device=device)
    h0 = hs.allocate()
    h1 = hs.allocate()
    h2 = hs.allocate()

    hs.remove(h1)
    test.assertEqual(hs.count.numpy()[0], 2)

    h2i = hs.handle_to_index.numpy()
    test.assertEqual(h2i[h1], -1)
    test.assertGreaterEqual(h2i[h0], 0)
    test.assertGreaterEqual(h2i[h2], 0)


def test_handlestore_allocate_after_remove(test, device):
    """Freed handle IDs are recycled."""
    hs = HandleStore(SimpleSchema, 4, device=device)
    h0 = hs.allocate()
    h1 = hs.allocate()
    hs.remove(h0)

    h2 = hs.allocate()
    test.assertGreaterEqual(h2, 0)
    test.assertEqual(hs.count.numpy()[0], 2)


def test_handlestore_compact_basic(test, device):
    """Compact closes gaps and keeps handle mappings valid."""
    cap = 8
    hs = HandleStore(SimpleSchema, cap, device=device)

    handles = [hs.allocate() for _ in range(5)]

    # Write position data so we can verify reordering
    pos = hs.column_of("position")
    cnt = hs.count
    wp.launch(_write_positions_kernel, dim=cap, inputs=[pos, cnt], device=device)
    wp.synchronize_device(device)

    # Remove handles 1 and 3 (middle elements)
    hs.remove(handles[1])
    hs.remove(handles[3])

    hs.compact()
    wp.synchronize_device(device)

    test.assertEqual(hs.count.numpy()[0], 3)

    # Surviving handles still resolve to valid rows
    h2i = hs.handle_to_index.numpy()
    surviving = [handles[0], handles[2], handles[4]]
    rows = set()
    for h in surviving:
        row = h2i[h]
        test.assertGreaterEqual(row, 0)
        test.assertLess(row, 3)
        rows.add(row)
    test.assertEqual(len(rows), 3)  # all different

    # Removed handles are invalid
    test.assertEqual(h2i[handles[1]], -1)
    test.assertEqual(h2i[handles[3]], -1)


def test_handlestore_compact_data_integrity(test, device):
    """Position data is correctly reordered by compact."""
    cap = 8
    hs = HandleStore(SimpleSchema, cap, device=device)

    handles = [hs.allocate() for _ in range(4)]

    # Write known mass values: mass[row] = handle_id + 0.5
    mass = hs.column_of("mass")
    mass_np = mass.numpy()
    for h in handles:
        row = hs.handle_to_index.numpy()[h]
        mass_np[row] = float(h) + 0.5
    mass.assign(wp.array(mass_np, dtype=wp.float32, device=device))

    # Remove handle 1
    hs.remove(handles[1])
    hs.compact()
    wp.synchronize_device(device)

    # Verify surviving handles have correct mass
    mass_after = hs.column_of("mass").numpy()
    h2i = hs.handle_to_index.numpy()
    for h in [handles[0], handles[2], handles[3]]:
        row = h2i[h]
        test.assertAlmostEqual(mass_after[row], float(h) + 0.5, places=5)


def test_handlestore_full(test, device):
    """Allocating past capacity returns -1."""
    hs = HandleStore(SimpleSchema, 2, device=device)
    h0 = hs.allocate()
    h1 = hs.allocate()
    h2 = hs.allocate()
    test.assertGreaterEqual(h0, 0)
    test.assertGreaterEqual(h1, 0)
    test.assertEqual(h2, -1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestDataBase, "test_schema_offsets", test_schema_offsets, devices=devices)
add_function_test(TestDataBase, "test_schema_quat", test_schema_quat, devices=devices)
add_function_test(
    TestDataBase, "test_datastore_column_of_write_read", test_datastore_column_of_write_read, devices=devices
)
add_function_test(TestDataBase, "test_datastore_clear", test_datastore_clear, devices=devices)
add_function_test(TestDataBase, "test_datastore_quat_column", test_datastore_quat_column, devices=devices)
add_function_test(TestDataBase, "test_handlestore_allocate", test_handlestore_allocate, devices=devices)
add_function_test(TestDataBase, "test_handlestore_remove", test_handlestore_remove, devices=devices)
add_function_test(
    TestDataBase, "test_handlestore_allocate_after_remove", test_handlestore_allocate_after_remove, devices=devices
)
add_function_test(TestDataBase, "test_handlestore_compact_basic", test_handlestore_compact_basic, devices=devices)
add_function_test(
    TestDataBase, "test_handlestore_compact_data_integrity", test_handlestore_compact_data_integrity, devices=devices
)
add_function_test(TestDataBase, "test_handlestore_full", test_handlestore_full, devices=devices)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
