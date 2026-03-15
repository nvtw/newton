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

"""Tests for the PhoenX SolverState: body management, contact import, integration."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver_phoenx import SolverState
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestSolverPhoenX(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Body management
# ---------------------------------------------------------------------------


def test_add_body(test, device):
    """add_body allocates a handle and writes initial state."""
    ss = SolverState(body_capacity=8, contact_capacity=16, shape_count=4, device=device)
    h = ss.add_body(position=(1.0, 2.0, 3.0), inverse_mass=0.5)

    test.assertGreaterEqual(h, 0)
    test.assertEqual(ss.body_store.count.numpy()[0], 1)

    pos = ss.body_store.column_of("position").numpy()
    row = ss.body_store.handle_to_index.numpy()[h]
    np.testing.assert_allclose(pos[row], [1.0, 2.0, 3.0], atol=1e-6)

    inv_m = ss.body_store.column_of("inverse_mass").numpy()
    test.assertAlmostEqual(inv_m[row], 0.5, places=5)


def test_add_static_body(test, device):
    """Static bodies have zero inverse mass and the static flag set."""
    ss = SolverState(body_capacity=4, contact_capacity=4, shape_count=2, device=device)
    h = ss.add_body(is_static=True)

    row = ss.body_store.handle_to_index.numpy()[h]
    inv_m = ss.body_store.column_of("inverse_mass").numpy()
    test.assertAlmostEqual(inv_m[row], 0.0, places=5)

    flags = ss.body_store.column_of("flags").numpy()
    test.assertEqual(flags[row], 1)  # BODY_FLAG_STATIC


def test_set_shape_body(test, device):
    """set_shape_body correctly maps shape index to body row."""
    ss = SolverState(body_capacity=4, contact_capacity=4, shape_count=4, device=device)
    h0 = ss.add_body()
    h1 = ss.add_body()
    ss.set_shape_body(0, h0)
    ss.set_shape_body(1, h0)
    ss.set_shape_body(2, h1)

    sb = ss.shape_body.numpy()
    row0 = ss.body_store.handle_to_index.numpy()[h0]
    row1 = ss.body_store.handle_to_index.numpy()[h1]
    test.assertEqual(sb[0], row0)
    test.assertEqual(sb[1], row0)
    test.assertEqual(sb[2], row1)


# ---------------------------------------------------------------------------
# Contact import
# ---------------------------------------------------------------------------


def _make_synthetic_contacts(device, n, shape_pairs):
    """Create minimal Newton-like contact arrays for testing."""

    class FakeContacts:
        pass

    c = FakeContacts()
    s0 = [p[0] for p in shape_pairs]
    s1 = [p[1] for p in shape_pairs]
    c.rigid_contact_count = wp.array([n], dtype=wp.int32, device=device)
    c.rigid_contact_shape0 = wp.array(s0, dtype=wp.int32, device=device)
    c.rigid_contact_shape1 = wp.array(s1, dtype=wp.int32, device=device)
    c.rigid_contact_normal = wp.array([[0.0, 1.0, 0.0]] * n, dtype=wp.vec3, device=device)
    c.rigid_contact_offset0 = wp.zeros(n, dtype=wp.vec3, device=device)
    c.rigid_contact_offset1 = wp.zeros(n, dtype=wp.vec3, device=device)
    c.rigid_contact_margin0 = wp.array([0.01] * n, dtype=wp.float32, device=device)
    c.rigid_contact_margin1 = wp.array([0.01] * n, dtype=wp.float32, device=device)
    return c


def test_import_contacts(test, device):
    """import_contacts copies shape/body/normal data correctly."""
    ss = SolverState(body_capacity=4, contact_capacity=16, shape_count=4, device=device)
    h0 = ss.add_body()
    h1 = ss.add_body()
    ss.set_shape_body(0, h0)
    ss.set_shape_body(1, h1)

    contacts = _make_synthetic_contacts(device, 2, [(0, 1), (1, 0)])
    ss.import_contacts(contacts)
    wp.synchronize_device(device)

    cs = ss.contact_store
    test.assertEqual(cs.count.numpy()[0], 2)

    normals = cs.column_of("normal").numpy()
    np.testing.assert_allclose(normals[0], [0.0, 1.0, 0.0], atol=1e-6)

    body0 = cs.column_of("body0").numpy()
    body1 = cs.column_of("body1").numpy()
    row0 = ss.body_store.handle_to_index.numpy()[h0]
    row1 = ss.body_store.handle_to_index.numpy()[h1]
    test.assertEqual(body0[0], row0)
    test.assertEqual(body1[0], row1)


def test_import_contacts_warm_start(test, device):
    """Warm starting transfers impulses across two import_contacts calls."""
    ss = SolverState(body_capacity=4, contact_capacity=16, shape_count=4, device=device)
    h0 = ss.add_body()
    h1 = ss.add_body()
    ss.set_shape_body(0, h0)
    ss.set_shape_body(1, h1)
    ss.set_shape_body(2, h0)
    ss.set_shape_body(3, h1)

    # Frame 1: pair (0, 1)
    contacts_f1 = _make_synthetic_contacts(device, 1, [(0, 1)])
    ss.import_contacts(contacts_f1)

    # Simulate solver writing an impulse
    imp_col = ss.contact_store.column_of("accumulated_normal_impulse")
    imp_np = imp_col.numpy()
    imp_np[0] = 99.0
    imp_col.assign(wp.array(imp_np, dtype=wp.float32, device=device))

    ss.export_impulses()

    # Frame 2: same pair (0, 1) plus new pair (2, 3)
    ss.warm_starter.begin_frame()
    contacts_f2 = _make_synthetic_contacts(device, 2, [(0, 1), (2, 3)])
    ss.import_contacts(contacts_f2)
    wp.synchronize_device(device)

    imp_after = ss.contact_store.column_of("accumulated_normal_impulse").numpy()
    # One should be ~99.0 (transferred), the other 0.0 (new pair)
    vals = sorted(imp_after[:2].tolist(), reverse=True)
    test.assertAlmostEqual(vals[0], 99.0, places=3)
    test.assertAlmostEqual(vals[1], 0.0, places=5)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_free_fall_integration(test, device):
    """A dynamic body under gravity for 1 s reaches correct velocity and position."""
    ss = SolverState(body_capacity=4, contact_capacity=4, shape_count=1, device=device)
    h = ss.add_body(position=(0.0, 10.0, 0.0), linear_damping=1.0, angular_damping=1.0)

    dt = 0.01
    gravity = (0.0, -9.81, 0.0)
    steps = 100  # 1 second total

    for _ in range(steps):
        ss.step(dt, gravity)

    wp.synchronize_device(device)

    row = ss.body_store.handle_to_index.numpy()[h]
    vel = ss.body_store.column_of("velocity").numpy()[row]
    pos = ss.body_store.column_of("position").numpy()[row]

    test.assertAlmostEqual(vel[1], -9.81, delta=0.05)
    # y = 10 + 0.5 * (-9.81) * 1^2 = 10 - 4.905 = 5.095
    test.assertAlmostEqual(pos[1], 5.095, delta=0.1)


def test_static_body_not_integrated(test, device):
    """Static bodies are unaffected by gravity."""
    ss = SolverState(body_capacity=4, contact_capacity=4, shape_count=1, device=device)
    h = ss.add_body(position=(0.0, 0.0, 0.0), is_static=True)

    ss.step(0.01, (0.0, -9.81, 0.0))
    wp.synchronize_device(device)

    row = ss.body_store.handle_to_index.numpy()[h]
    vel = ss.body_store.column_of("velocity").numpy()[row]
    pos = ss.body_store.column_of("position").numpy()[row]

    np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(pos, [0.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestSolverPhoenX, "test_add_body", test_add_body, devices=devices)
add_function_test(TestSolverPhoenX, "test_add_static_body", test_add_static_body, devices=devices)
add_function_test(TestSolverPhoenX, "test_set_shape_body", test_set_shape_body, devices=devices)
add_function_test(TestSolverPhoenX, "test_import_contacts", test_import_contacts, devices=devices)
add_function_test(TestSolverPhoenX, "test_import_contacts_warm_start", test_import_contacts_warm_start, devices=devices)
add_function_test(TestSolverPhoenX, "test_free_fall_integration", test_free_fall_integration, devices=devices)
add_function_test(TestSolverPhoenX, "test_static_body_not_integrated", test_static_body_not_integrated, devices=devices)

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
