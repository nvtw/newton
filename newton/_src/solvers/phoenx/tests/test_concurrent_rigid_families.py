# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check that concurrent rigid-family lanes retain unique constraint ownership."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.simulation_kernels import _rigid_family_lane_allocation
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@wp.kernel
def _count_family_visits(
    count_joints: wp.int32,
    count_contacts: wp.int32,
    stride: wp.int32,
    joint_visits: wp.array[wp.int32],
    contact_visits: wp.array[wp.int32],
    allocation_out: wp.array[wp.int32],
):
    lane = wp.tid()
    allocation = _rigid_family_lane_allocation(count_joints, count_contacts, stride)
    joint_lanes = allocation[0]
    contact_lane_base = allocation[1]
    if lane == wp.int32(0):
        allocation_out[0] = joint_lanes
        allocation_out[1] = contact_lane_base

    for joint in range(lane / wp.int32(8), count_joints, joint_lanes / wp.int32(8)):
        if lane % wp.int32(8) == wp.int32(0):
            wp.atomic_add(joint_visits, joint, wp.int32(1))
    contact = lane - contact_lane_base
    while contact >= wp.int32(0) and contact < count_contacts:
        wp.atomic_add(contact_visits, contact, wp.int32(1))
        contact += stride - contact_lane_base


@unittest.skipUnless(_cuda_with_graph_capture(), "CUDA graph capture required")
class TestConcurrentRigidFamilies(unittest.TestCase):
    def test_each_joint_and_contact_has_one_lane(self):
        # Small colors share the launch; joint-heavy colors use the original
        # full-width assignment. The latter caught duplicate joint visits
        # when the concurrent lane budget was truncated.
        cases = [
            (0, 17, 32, (32, 0)),
            (1, 0, 32, (32, 0)),
            (1, 7, 256, (32, 32)),
            (2, 5, 256, (32, 32)),
            (3, 7, 256, (32, 32)),
            (4, 7, 32, (32, 0)),
            (39, 20, 256, (256, 0)),
            (3000, 20, 48128, (24000, 24000)),
            (7000, 20, 48128, (48128, 0)),
        ]
        for joints, contacts, stride, expected_allocation in cases:
            with self.subTest(joints=joints, contacts=contacts, stride=stride):
                joint_visits = wp.zeros(joints, dtype=wp.int32, device="cuda:0")
                contact_visits = wp.zeros(contacts, dtype=wp.int32, device="cuda:0")
                allocation = wp.zeros(2, dtype=wp.int32, device="cuda:0")
                wp.launch(
                    _count_family_visits,
                    dim=stride,
                    inputs=[joints, contacts, stride, joint_visits, contact_visits, allocation],
                    device="cuda:0",
                )
                np.testing.assert_array_equal(joint_visits.numpy(), np.ones(joints, dtype=np.int32))
                np.testing.assert_array_equal(contact_visits.numpy(), np.ones(contacts, dtype=np.int32))
                self.assertEqual(tuple(allocation.numpy()), expected_allocation)
