# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""Tests for the global contact reduction module."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    export_and_reduce_contact,
    make_contact_key,
    unpack_contact,
)


class TestGlobalContactReducer(unittest.TestCase):
    """Test cases for GlobalContactReducer."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp once for all tests."""
        wp.init()

    def test_basic_contact_storage(self):
        """Test basic contact storage and retrieval."""
        reducer = GlobalContactReducer(capacity=100, device="cpu")

        @wp.kernel
        def store_contact_kernel(
            position_depth: wp.array(dtype=wp.vec4),
            normal_feature: wp.array(dtype=wp.vec4),
            shape_pairs: wp.array(dtype=wp.vec2i),
            contact_count: wp.array(dtype=wp.int32),
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
            capacity: int,
        ):
            contact_id = export_and_reduce_contact(
                shape_a=0,
                shape_b=1,
                position=wp.vec3(1.0, 2.0, 3.0),
                normal=wp.vec3(0.0, 1.0, 0.0),
                depth=-0.01,
                feature=42,
                position_depth=position_depth,
                normal_feature=normal_feature,
                shape_pairs=shape_pairs,
                contact_count=contact_count,
                ht_keys=ht_keys,
                ht_values=ht_values,
                capacity=capacity,
            )

        wp.launch(
            store_contact_kernel,
            dim=1,
            inputs=[
                reducer.position_depth,
                reducer.normal_feature,
                reducer.shape_pairs,
                reducer.contact_count,
                reducer.hashtable.keys,
                reducer.hashtable.values,
                reducer.capacity,
            ],
            device="cpu",
        )

        self.assertEqual(reducer.get_contact_count(), 1)

        # Check stored data
        pd = reducer.position_depth.numpy()[0]
        self.assertAlmostEqual(pd[0], 1.0)
        self.assertAlmostEqual(pd[1], 2.0)
        self.assertAlmostEqual(pd[2], 3.0)
        self.assertAlmostEqual(pd[3], -0.01, places=5)

    def test_multiple_contacts_same_pair(self):
        """Test that multiple contacts for same shape pair get reduced."""
        reducer = GlobalContactReducer(capacity=100, device="cpu")

        @wp.kernel
        def store_multiple_contacts_kernel(
            position_depth: wp.array(dtype=wp.vec4),
            normal_feature: wp.array(dtype=wp.vec4),
            shape_pairs: wp.array(dtype=wp.vec2i),
            contact_count: wp.array(dtype=wp.int32),
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
            capacity: int,
        ):
            tid = wp.tid()
            # All contacts have same shape pair and similar normal (pointing up)
            # But different positions - reduction should pick spatial extremes
            x = float(tid) - 5.0  # Range from -5 to +4
            export_and_reduce_contact(
                shape_a=0,
                shape_b=1,
                position=wp.vec3(x, 0.0, 0.0),
                normal=wp.vec3(0.0, 1.0, 0.0),
                depth=-0.01,
                feature=tid,
                position_depth=position_depth,
                normal_feature=normal_feature,
                shape_pairs=shape_pairs,
                contact_count=contact_count,
                ht_keys=ht_keys,
                ht_values=ht_values,
                capacity=capacity,
            )

        wp.launch(
            store_multiple_contacts_kernel,
            dim=10,
            inputs=[
                reducer.position_depth,
                reducer.normal_feature,
                reducer.shape_pairs,
                reducer.contact_count,
                reducer.hashtable.keys,
                reducer.hashtable.values,
                reducer.capacity,
            ],
            device="cpu",
        )

        # All 10 contacts should be stored
        self.assertEqual(reducer.get_contact_count(), 10)

        # But only a few should win hashtable slots (spatial extremes)
        winners = reducer.get_winning_contacts()
        # Should have fewer winners than total contacts due to reduction
        self.assertLess(len(winners), 10)
        self.assertGreater(len(winners), 0)

    def test_different_shape_pairs(self):
        """Test that different shape pairs are tracked separately."""
        reducer = GlobalContactReducer(capacity=100, device="cpu")

        @wp.kernel
        def store_different_pairs_kernel(
            position_depth: wp.array(dtype=wp.vec4),
            normal_feature: wp.array(dtype=wp.vec4),
            shape_pairs: wp.array(dtype=wp.vec2i),
            contact_count: wp.array(dtype=wp.int32),
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
            capacity: int,
        ):
            tid = wp.tid()
            # Each thread represents a different shape pair
            export_and_reduce_contact(
                shape_a=tid,
                shape_b=tid + 100,
                position=wp.vec3(0.0, 0.0, 0.0),
                normal=wp.vec3(0.0, 1.0, 0.0),
                depth=-0.01,
                feature=tid,
                position_depth=position_depth,
                normal_feature=normal_feature,
                shape_pairs=shape_pairs,
                contact_count=contact_count,
                ht_keys=ht_keys,
                ht_values=ht_values,
                capacity=capacity,
            )

        wp.launch(
            store_different_pairs_kernel,
            dim=5,
            inputs=[
                reducer.position_depth,
                reducer.normal_feature,
                reducer.shape_pairs,
                reducer.contact_count,
                reducer.hashtable.keys,
                reducer.hashtable.values,
                reducer.capacity,
            ],
            device="cpu",
        )

        # All 5 contacts stored
        self.assertEqual(reducer.get_contact_count(), 5)

        # Each shape pair should have its own winners
        winners = reducer.get_winning_contacts()
        # All 5 should win (different pairs, no competition)
        self.assertEqual(len(winners), 5)

    def test_clear(self):
        """Test that clear resets the reducer."""
        reducer = GlobalContactReducer(capacity=100, device="cpu")

        @wp.kernel
        def store_one_contact_kernel(
            position_depth: wp.array(dtype=wp.vec4),
            normal_feature: wp.array(dtype=wp.vec4),
            shape_pairs: wp.array(dtype=wp.vec2i),
            contact_count: wp.array(dtype=wp.int32),
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
            capacity: int,
        ):
            export_and_reduce_contact(
                shape_a=0,
                shape_b=1,
                position=wp.vec3(0.0, 0.0, 0.0),
                normal=wp.vec3(0.0, 1.0, 0.0),
                depth=-0.01,
                feature=0,
                position_depth=position_depth,
                normal_feature=normal_feature,
                shape_pairs=shape_pairs,
                contact_count=contact_count,
                ht_keys=ht_keys,
                ht_values=ht_values,
                capacity=capacity,
            )

        wp.launch(
            store_one_contact_kernel,
            dim=1,
            inputs=[
                reducer.position_depth,
                reducer.normal_feature,
                reducer.shape_pairs,
                reducer.contact_count,
                reducer.hashtable.keys,
                reducer.hashtable.values,
                reducer.capacity,
            ],
            device="cpu",
        )

        self.assertEqual(reducer.get_contact_count(), 1)
        self.assertGreater(len(reducer.get_winning_contacts()), 0)

        reducer.clear()

        self.assertEqual(reducer.get_contact_count(), 0)
        self.assertEqual(len(reducer.get_winning_contacts()), 0)

    def test_stress_many_contacts(self):
        """Stress test with many contacts from many shape pairs."""
        reducer = GlobalContactReducer(capacity=10000, device="cpu")

        @wp.kernel
        def stress_kernel(
            position_depth: wp.array(dtype=wp.vec4),
            normal_feature: wp.array(dtype=wp.vec4),
            shape_pairs: wp.array(dtype=wp.vec2i),
            contact_count: wp.array(dtype=wp.int32),
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
            capacity: int,
        ):
            tid = wp.tid()
            # 100 shape pairs, 50 contacts each = 5000 total
            pair_id = tid // 50
            contact_in_pair = tid % 50

            shape_a = pair_id
            shape_b = pair_id + 1000

            # Vary positions within each pair
            x = float(contact_in_pair) - 25.0
            y = float(contact_in_pair % 10) - 5.0

            # Vary normals slightly
            nx = 0.1 * float(contact_in_pair % 3)
            ny = 1.0
            nz = 0.1 * float(contact_in_pair % 5)
            n_len = wp.sqrt(nx * nx + ny * ny + nz * nz)

            export_and_reduce_contact(
                shape_a=shape_a,
                shape_b=shape_b,
                position=wp.vec3(x, y, 0.0),
                normal=wp.vec3(nx / n_len, ny / n_len, nz / n_len),
                depth=-0.01,
                feature=tid,
                position_depth=position_depth,
                normal_feature=normal_feature,
                shape_pairs=shape_pairs,
                contact_count=contact_count,
                ht_keys=ht_keys,
                ht_values=ht_values,
                capacity=capacity,
            )

        wp.launch(
            stress_kernel,
            dim=5000,
            inputs=[
                reducer.position_depth,
                reducer.normal_feature,
                reducer.shape_pairs,
                reducer.contact_count,
                reducer.hashtable.keys,
                reducer.hashtable.values,
                reducer.capacity,
            ],
            device="cpu",
        )

        self.assertEqual(reducer.get_contact_count(), 5000)

        winners = reducer.get_winning_contacts()
        # Should have significant reduction
        self.assertLess(len(winners), 5000)
        # But at least some winners per pair (100 pairs * some contacts)
        self.assertGreater(len(winners), 100)

        print(f"Stress test: 5000 contacts reduced to {len(winners)} winners")


class TestKeyConstruction(unittest.TestCase):
    """Test the key construction function."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_key_uniqueness(self):
        """Test that different inputs produce different keys."""

        @wp.kernel
        def compute_keys_kernel(
            keys_out: wp.array(dtype=wp.uint64),
        ):
            # Test various combinations
            keys_out[0] = make_contact_key(0, 1, 0, 0)
            keys_out[1] = make_contact_key(1, 0, 0, 0)  # Swapped shapes
            keys_out[2] = make_contact_key(0, 1, 1, 0)  # Different bin
            keys_out[3] = make_contact_key(0, 1, 0, 1)  # Different direction
            keys_out[4] = make_contact_key(100, 200, 10, 3)  # Larger values

        keys = wp.zeros(5, dtype=wp.uint64, device="cpu")
        wp.launch(compute_keys_kernel, dim=1, inputs=[keys], device="cpu")

        keys_np = keys.numpy()
        # All keys should be unique
        self.assertEqual(len(set(keys_np)), 5)


if __name__ == "__main__":
    unittest.main()

