# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare prepared friction rows with the paired scalar reference."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    BodyContainer,
    body_container_zeros,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_set_body1,
    contact_set_body2,
    contact_set_count1,
    contact_set_count2,
)
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, allocate_contact_tgs
from newton._src.solvers.phoenx.constraints.contact_tgs_friction import solve, solve_cached
from newton._src.solvers.phoenx.constraints.contact_tgs_prepare import friction_geometry


@wp.kernel
def setup(columns: ContactColumnContainer, scale0: int, scale1: int):
    contact_set_body1(columns, 0, 0)
    contact_set_body2(columns, 0, 1)
    contact_set_count1(columns, 0, scale0)
    contact_set_count2(columns, 0, scale1)


def kernel(cached, biased):
    @wp.kernel(module="unique", enable_backward=False)
    def run(state: ContactTGS, bodies: BodyContainer, velocity: wp.array[wp.vec3f]):
        v0 = velocity[0]
        v1 = velocity[1]
        w0 = velocity[2]
        w1 = velocity[3]
        pose0 = wp.transformf(bodies.position[0], bodies.orientation[0])
        pose1 = wp.transformf(bodies.position[1], bodies.orientation[1])
        for patch in range(2):
            if wp.static(cached):
                v0, v1, w0, w1 = solve_cached(
                    state.current,
                    state.anchors,
                    patch,
                    state.loads,
                    state.impulse,
                    pose0,
                    pose1,
                    v0,
                    v1,
                    w0,
                    w1,
                    bodies.inverse_mass[0],
                    bodies.inverse_mass[1],
                    mat33_from_sym6(bodies.inverse_inertia_world[0]),
                    mat33_from_sym6(bodies.inverse_inertia_world[1]),
                    0.5,
                    0.3,
                    1000.0,
                    state.gain,
                    wp.bool(wp.static(biased)),
                    state.friction_rows,
                )
            else:
                v0, v1, w0, w1 = solve(
                    state.current,
                    state.anchors,
                    patch,
                    state.loads,
                    state.impulse,
                    pose0,
                    pose1,
                    v0,
                    v1,
                    w0,
                    w1,
                    bodies.inverse_mass[0],
                    bodies.inverse_mass[1],
                    mat33_from_sym6(bodies.inverse_inertia_world[0]),
                    mat33_from_sym6(bodies.inverse_inertia_world[1]),
                    0.5,
                    0.3,
                    1000.0,
                    state.gain,
                    wp.bool(wp.static(biased)),
                )
        velocity[0] = v0
        velocity[1] = v1
        velocity[2] = w0
        velocity[3] = w1

    return run


class TestPreparedFriction(unittest.TestCase):
    def test_reference_and_momentum(self):
        """Match scalar friction and conserve momentum with prepared rows."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for biased in (False, True):
                with self.subTest(device=device, biased=biased):
                    self.check_case(device, biased)

    def check_case(self, device, biased):
        positions = np.array([[-0.397, 0.198, 0.701], [0.299, -0.096, -0.202]], dtype=np.float32)
        original = np.array([[-0.4, 0.2, 0.7], [0.3, -0.1, -0.2]], dtype=np.float32)
        inertia = np.array([[2.0, 4.0, 2.5], [3.0, 2.0, 5.0]])
        mass = np.array([2.0, 3.0])
        bodies = body_container_zeros(2, device)
        bodies.position.assign(positions)
        bodies.orientation.assign([[0, 0, 0, 1]] * 2)
        bodies.inverse_mass.assign(1.0 / mass)
        bodies.inverse_inertia_world.assign(np.c_[1.0 / inertia, np.zeros((2, 3))])
        columns = contact_column_container_zeros(1, device)
        wp.launch(setup, 1, [columns, 1, 1], device=device)
        state = allocate_contact_tgs(4, 2, 30, device)
        state.point_last.assign([0, 0, -1, -1])
        state.current.patch_normal.assign([[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        state.current.patch_first.assign([0, 1, -1, -1])
        state.current.point_next.assign([2, 3, -1, -1])
        state.anchors.count.assign([2, 2, 0, 0])
        points = np.zeros((4, 2, 3), dtype=np.float32)
        points[0] = [[-0.1, 0, 0], [0.1, 0, 0]]
        points[1] = [[0, -0.1, 0], [0, 0.1, 0]]
        state.anchors.local0.assign(points - original[0])
        state.anchors.local1.assign(points - original[1])
        state.loads.assign([0.1, 0.02, 0.2, 0.07])
        wp.launch(friction_geometry, (4, 2), [columns, state, bodies], device=device)
        initial = np.array([[0.2, -0.1, 0], [-0.3, 0.4, 0], [0.1, -0.2, 0.8], [-0.1, 0.2, -0.5]], dtype=np.float32)
        results = []
        for cached in (False, True):
            state.impulse.zero_()
            velocity = wp.array(initial, dtype=wp.vec3f, device=device)
            wp.launch(kernel(cached, biased), 1, [state, bodies, velocity], device=device)
            results.append((velocity.numpy(), state.impulse.numpy(), state.anchors.broken.numpy()))
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

        def momentum(v):
            linear = mass[:, None] * v[:2]
            angular = inertia * v[2:] + np.cross(positions, linear)
            return np.r_[linear.sum(0), angular.sum(0)]

        np.testing.assert_allclose(momentum(results[1][0]), momentum(initial), rtol=0, atol=3e-7)
        for static in (False, True):
            bodies.motion_type.assign([MOTION_DYNAMIC, MOTION_STATIC if static else MOTION_DYNAMIC])
            inverse_mass = 1.0 / mass
            inverse_inertia = 1.0 / inertia
            if static:
                inverse_mass[1] = 0
                inverse_inertia[1] = 0
            bodies.inverse_mass.assign(inverse_mass)
            bodies.inverse_inertia_world.assign(np.c_[inverse_inertia, np.zeros((2, 3))])
            wp.launch(setup, 1, [columns, 3, 2], device=device)
            wp.launch(friction_geometry, (4, 2), [columns, state, bodies], device=device)
            rows = state.friction_rows.numpy()
            scales = np.ones(2) if static else np.array([3, 2])
            for patch in range(2):
                for anchor in range(2):
                    row = rows[patch, anchor]
                    for axis in (0, 1):
                        tangent = row[f"t{axis}"]
                        angular0 = np.cross(row["r0"], tangent)
                        angular1 = np.cross(row["r1"], tangent)
                        expected = (
                            np.dot(scales, inverse_mass)
                            + scales[0] * np.dot(angular0 * inverse_inertia[0], angular0)
                            + scales[1] * np.dot(angular1 * inverse_inertia[1], angular1)
                        )
                        np.testing.assert_allclose(row[f"response{axis}"], expected, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
