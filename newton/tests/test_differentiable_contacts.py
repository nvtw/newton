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

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_no_overhead_when_disabled(test, device):
    """Differentiable arrays are None when requires_grad=False."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=False)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        test.assertIsNone(contacts.rigid_contact_diff_distance)
        test.assertIsNone(contacts.rigid_contact_diff_normal)
        test.assertIsNone(contacts.rigid_contact_diff_point0_world)
        test.assertIsNone(contacts.rigid_contact_diff_point1_world)


def test_arrays_allocated_when_enabled(test, device):
    """Differentiable arrays are allocated when requires_grad=True."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        test.assertIsNotNone(contacts.rigid_contact_diff_distance)
        test.assertIsNotNone(contacts.rigid_contact_diff_normal)
        test.assertIsNotNone(contacts.rigid_contact_diff_point0_world)
        test.assertIsNotNone(contacts.rigid_contact_diff_point1_world)
        test.assertTrue(contacts.rigid_contact_diff_distance.requires_grad)


def test_sphere_on_plane_distance(test, device):
    """Sphere resting on ground plane produces correct differentiable distance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_height = 0.3
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()

        pipeline.collide(state, contacts)
        pipeline.eval_differentiable_contacts(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected at least one contact")

        diff_dist = contacts.rigid_contact_diff_distance.numpy()[:count]
        test.assertTrue(
            np.any(diff_dist < 0.0),
            f"Expected negative (penetrating) distance, got {diff_dist}",
        )


def test_gradient_flow_through_body_q(test, device):
    """Verify gradients flow from diff distance through body_q via wp.Tape."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_height = 0.3
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        pipeline.collide(state, contacts)

        with wp.Tape() as tape:
            pipeline.eval_differentiable_contacts(state, contacts)

        tape.backward(grads={contacts.rigid_contact_diff_distance: wp.ones(
            contacts.rigid_contact_max, dtype=float, device=device
        )})

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q, "body_q gradient should be recorded on tape")

        grad_np = grad_q.numpy()
        test.assertFalse(
            np.allclose(grad_np, 0.0),
            "body_q gradient should be non-zero for penetrating sphere",
        )


def test_gradient_direction(test, device):
    """Moving the sphere upward should increase (make less negative) the contact distance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        pipeline.collide(state, contacts)

        with wp.Tape() as tape:
            pipeline.eval_differentiable_contacts(state, contacts)

        tape.backward(grads={contacts.rigid_contact_diff_distance: wp.ones(
            contacts.rigid_contact_max, dtype=float, device=device
        )})

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)

        grad_np = grad_q.numpy()
        # body_q is a transform (p, q).  The translation gradient should point
        # upward (positive z) since moving the sphere up increases distance.
        # wp.transform stores (px, py, pz, qw, qx, qy, qz)
        dz = grad_np[0, 2]  # body 0, z-translation component
        test.assertGreater(
            dz, 0.0,
            f"Expected positive z-gradient (moving up increases distance), got dz={dz}",
        )


class TestDifferentiableContacts(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(TestDifferentiableContacts, "test_no_overhead_when_disabled", test_no_overhead_when_disabled, devices=devices)
add_function_test(TestDifferentiableContacts, "test_arrays_allocated_when_enabled", test_arrays_allocated_when_enabled, devices=devices)
add_function_test(TestDifferentiableContacts, "test_sphere_on_plane_distance", test_sphere_on_plane_distance, devices=devices)
add_function_test(TestDifferentiableContacts, "test_gradient_flow_through_body_q", test_gradient_flow_through_body_q, devices=devices)
add_function_test(TestDifferentiableContacts, "test_gradient_direction", test_gradient_direction, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
