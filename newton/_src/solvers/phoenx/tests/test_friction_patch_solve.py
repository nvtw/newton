# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""FP32 fixed-load proposal certificates; no live solver integration claims."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.friction_patch_solve import propose_friction_patch_stick


class PatchFixture:
    """Own preallocated proposal inputs and outputs for one CSR patch."""

    def __init__(self, device, points, capacity=None, velocity=None, mobility=None, basis=None, origin=None):
        self.device = device
        p = np.asarray(points, np.float32).reshape(-1, 3)
        n = len(p)
        basis = np.eye(3, dtype=np.float32) if basis is None else np.asarray(basis, np.float32)
        self.basis = basis
        self.origin = np.zeros(3, np.float32) if origin is None else np.asarray(origin, np.float32)
        self.points_host = p
        self.capacity_host = np.ones(n, np.float32) if capacity is None else np.asarray(capacity, np.float32)
        self.velocity_host = np.array([0.01, 0.02, 0.003] if velocity is None else velocity, np.float32)
        self.mobility_host = (
            np.diag([0.5, 0.7, 2.0]).astype(np.float32) if mobility is None else np.asarray(mobility, np.float32)
        )
        self.offsets = wp.array([0, n], dtype=wp.int32, device=device)
        self.members = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device)
        self.points = wp.array(p, dtype=wp.vec3f, device=device)
        self.normals = wp.array(np.tile(basis[2], (n, 1)), dtype=wp.vec3f, device=device)
        self.capacity = wp.array(self.capacity_host, dtype=wp.float32, device=device)
        self.origins = wp.array([self.origin], dtype=wp.vec3f, device=device)
        self.axes = [wp.array([axis], dtype=wp.vec3f, device=device) for axis in basis]
        self.mobility = wp.array([self.mobility_host], dtype=wp.mat33f, device=device)
        self.velocity = wp.array([self.velocity_host], dtype=wp.vec3f, device=device)
        self.impulses = wp.full(n, wp.vec3f(99.0), dtype=wp.vec3f, device=device)
        self.status = wp.zeros(1, dtype=wp.int32, device=device)
        self.wrench = wp.zeros(1, dtype=wp.vec3f, device=device)
        self.residuals = wp.zeros(1, dtype=wp.vec3f, device=device)

    def launch(self):
        """Use explicit roundoff tolerances, without changing capacities."""
        wp.launch(
            propose_friction_patch_stick,
            dim=1,
            inputs=[
                self.offsets,
                self.members,
                self.points,
                self.normals,
                self.capacity,
                self.origins,
                *self.axes,
                self.mobility,
                self.velocity,
                1e-7,
                2e-6,
                1e-7,
                2e-7,
                2e-7,
                2e-7,
                2e-7,
                self.impulses,
                self.status,
                self.wrench,
                self.residuals,
            ],
            device=self.device,
        )

    def certified(self, case):
        """Independently recompute original cones, resultant, response and work."""
        self.launch()
        case.assertEqual(int(self.status.numpy()[0]), 1)
        forces = self.impulses.numpy().astype(float)
        case.assertTrue(np.all(np.linalg.norm(forces, axis=1) <= self.capacity_host[self.members.numpy()] + 1e-7))
        np.testing.assert_allclose(forces @ self.basis[2], 0, atol=1e-7)
        total = forces.sum(axis=0)
        torque = np.cross(self.points_host[self.members.numpy()].astype(float) - self.origin, forces).sum(axis=0)
        actual = np.r_[self.basis[:2] @ total, self.basis[2] @ torque]
        expected = -np.linalg.solve(self.mobility_host.astype(float), self.velocity_host.astype(float))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
        np.testing.assert_allclose(self.mobility_host @ actual + self.velocity_host, 0, atol=2e-7)
        case.assertLessEqual(float(actual @ self.velocity_host + 0.5 * actual @ self.mobility_host @ actual), 0)
        return forces

    def unknown(self, case):
        """Rejected proposals never leak a partial scatter or stale impulse."""
        self.launch()
        case.assertEqual(int(self.status.numpy()[0]), 0)
        np.testing.assert_array_equal(self.impulses.numpy(), 0)


class TestFrictionPatchSolveCPU(unittest.TestCase):
    """Exercise the actual Warp kernel with independent physical certificates."""

    device = "cpu"

    def test_member_counts(self):
        """Support unbounded member loops and reject single-point rank loss."""
        for count in (1, 2, 4, 129, 513):
            with self.subTest(count=count):
                angles = np.arange(count) * (2 * np.pi / count)
                points = np.c_[np.cos(angles), np.sin(angles), np.zeros(count)]
                fixture = PatchFixture(self.device, points)
                if count == 1:
                    fixture.unknown(self)
                else:
                    fixture.certified(self)

    def test_last_member_beyond_warp_and_block_size(self):
        """Require the final contact beyond 128 entries to supply yaw support."""
        for count in (129, 513):
            points = np.zeros((count, 3), np.float32)
            points[0, 0], points[-1, 0] = -1, 1
            capacity = np.zeros(count, np.float32)
            capacity[0], capacity[-1] = 0.5, 0.5
            fixture = PatchFixture(self.device, points, capacity, velocity=[0, 0, -0.01])
            forces = fixture.certified(self)
            self.assertGreater(np.linalg.norm(forces[-1]), 0.001)
            np.testing.assert_array_equal(forces[1:-1], 0)

    def test_nonplanar_and_zero_capacity(self):
        """Reject nonplanar geometry even on zero-friction members."""
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0], [0, 0, 0.001]], [1, 1, 0])
        fixture.unknown(self)
        PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]], [0, 0]).unknown(self)
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0], [0, 1, 0]], [1, 1, 0])
        forces = fixture.certified(self)
        np.testing.assert_array_equal(forces[2], 0)

    def test_rank_invalid_and_empty(self):
        """Use UNKNOWN for singular physical mobility, coincident points and empty patches."""
        PatchFixture(self.device, [[0, 0, 0], [0, 0, 0]]).unknown(self)
        PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]], mobility=np.diag([1, 1, 0])).unknown(self)
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]])
        fixture.offsets.assign(np.array([0, 0], np.int32))
        fixture.launch()
        self.assertEqual(fixture.status.numpy()[0], 0)

    def test_false_rejection_is_unknown(self):
        """A feasible but non-minimum-norm distribution must not imply sliding."""
        points = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
        target = np.array([0, 0.8, 0.15])
        fixture = PatchFixture(self.device, points, [0.1, 1, 0.1], velocity=-target, mobility=np.eye(3))
        fixture.unknown(self)
        feasible = np.array([-0.05, 0.75, 0.1])
        self.assertTrue(np.all(abs(feasible) <= np.array([0.1, 1, 0.1])))
        np.testing.assert_allclose([feasible.sum(), -feasible[0] + feasible[2]], target[1:])

    def test_unequal_pressure_rejects_false_yaw(self):
        """Reject pooled-anchor torque exceeding actual per-point capacities."""
        fixture = PatchFixture(
            self.device, [[-0.1, 0, 0], [0.1, 0, 0]], [0.9, 0.1], velocity=[0, 0, -0.1], mobility=np.eye(3)
        )
        fixture.unknown(self)

    def test_common_point_pair_momentum_and_external_work(self):
        """Balance external force/torque and conserve paired spatial momentum."""
        points = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
        external = np.array([0.03, 0.02, 0.001])
        h = np.diag([0.5, 0.5, 2.0])
        fixture = PatchFixture(self.device, points, np.full(4, 0.05), velocity=h @ external, mobility=h)
        forces = fixture.certified(self)
        wrench = np.r_[forces.sum(axis=0)[:2], np.cross(points, forces).sum(axis=0)[2]]
        np.testing.assert_allclose(wrench + external, 0, atol=2e-7)
        work = wrench @ (h @ external) + 0.5 * wrench @ h @ wrench
        self.assertAlmostEqual(float(work + 0.5 * external @ h @ external), 0, places=10)
        positions = np.array([[-0.3, 0.2, 0.1], [0.4, -0.2, 0.3]])
        paired_force = np.array([-forces.sum(axis=0), forces.sum(axis=0)])
        paired_torque = np.array(
            [
                np.cross(points - positions[0], -forces).sum(axis=0),
                np.cross(points - positions[1], forces).sum(axis=0),
            ]
        )
        np.testing.assert_allclose(paired_force.sum(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose((paired_torque + np.cross(positions, paired_force)).sum(axis=0), 0, atol=1e-14)

    def test_rigid_frame_and_permutation(self):
        """Retain physical wrench covariance and use CSR membership indices."""
        points = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
        base = PatchFixture(self.device, points)
        original = base.certified(self)
        basis = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], np.float32)
        shift = np.array([0.5, -0.25, 0.125])
        moved = PatchFixture(self.device, points @ basis + shift, basis=basis, origin=shift)
        moved.members.assign(np.array([3, 1, 0, 2], np.int32))
        force = moved.certified(self)
        # The independent certificate consumes physical member order.
        np.testing.assert_allclose(force, (original @ basis)[[3, 1, 0, 2]], atol=2e-7)

    def test_invalid_physical_inputs(self):
        """Reject invalid capacities, normals, mobility and reversed basis."""
        for bad in (-1.0, np.nan, np.inf):
            with self.subTest(capacity=bad):
                PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]], [1, bad]).unknown(self)
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]])
        fixture.normals.assign(np.array([[0, 0, 1], [0, 1, 0]], np.float32))
        fixture.unknown(self)
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]])
        fixture.mobility.assign(np.array([[[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]], np.float32))
        fixture.unknown(self)
        PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]], basis=np.diag([1, 1, -1])).unknown(self)

    def test_nonfinite_frame_and_mobility(self):
        """Reject NaNs even where floating comparisons would otherwise pass."""
        for field in ("origins", "mobility", "axis"):
            with self.subTest(field=field):
                fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]])
                if field == "mobility":
                    value = fixture.mobility.numpy()
                    value[0, 0, 0] = np.nan
                    fixture.mobility.assign(value)
                elif field == "origins":
                    fixture.origins.assign(np.array([[np.nan, 0, 0]], np.float32))
                else:
                    fixture.axes[0].assign(np.array([[np.nan, 0, 0]], np.float32))
                fixture.unknown(self)

    def test_unequal_body_physical_response(self):
        """Verify work with actual unequal-body inertia and off-center impulses."""
        points = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
        positions = np.array([[-0.3, 0.2, 0.1], [0.4, -0.2, 0.3]])
        mass = np.array([2.0, 3.0])
        inertia = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        response = np.zeros((2, 6, 3))
        inverse_mass = np.zeros((2, 6))
        for body in range(2):
            response[body, :3, :2] = np.eye(3)[:, :2]
            response[body, 3:, :2] = np.cross(-positions[body], np.eye(3)[:2]).T
            response[body, 5, 2] = 1
            inverse_mass[body] = np.r_[np.full(3, 1 / mass[body]), 1 / inertia[body]]
        h = sum(b.T @ (w[:, None] * b) for b, w in zip(response, inverse_mass, strict=True))
        before = np.array([[0.01, -0.02, 0.03, 0.1, 0.02, -0.03], [-0.02, 0.04, 0.01, -0.03, 0.04, 0.05]])
        relative = response[1].T @ before[1] - response[0].T @ before[0]
        fixture = PatchFixture(self.device, points, velocity=relative, mobility=h)
        forces = fixture.certified(self)
        impulse = np.array(
            [
                np.r_[-forces.sum(axis=0), np.cross(points - positions[0], -forces).sum(axis=0)],
                np.r_[forces.sum(axis=0), np.cross(points - positions[1], forces).sum(axis=0)],
            ]
        )
        after = before + inverse_mass * impulse
        momentum = impulse[:, :3].sum(axis=0)
        angular = (impulse[:, 3:] + np.cross(positions, impulse[:, :3])).sum(axis=0)
        np.testing.assert_allclose(momentum, 0, atol=1e-14)
        np.testing.assert_allclose(angular, 0, atol=1e-14)
        work = np.sum(0.5 * impulse * (before + after))
        kinetic_change = np.sum(0.5 * (after**2 - before**2) / inverse_mass)
        self.assertAlmostEqual(float(work), float(kinetic_change), places=14)
        self.assertLess(work, 0)
        np.testing.assert_allclose(response[1].T @ after[1] - response[0].T @ after[0], 0, atol=2e-7)

    def test_rejection_clears_previous_proposal(self):
        """Clear accepted outputs when the next call loses its capacity."""
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]])
        fixture.certified(self)
        fixture.capacity.zero_()
        fixture.unknown(self)


class TestFrictionPatchSolveCUDA(TestFrictionPatchSolveCPU):
    """Run the same physical gates on CUDA and replay changing captured inputs."""

    device = "cuda:0"

    @classmethod
    def setUpClass(cls):
        """Skip only when no CUDA device exists."""
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required")

    def test_capture_capacity_change(self):
        """Preserve UNKNOWN semantics through captured repeated execution."""
        fixture = PatchFixture(self.device, [[-1, 0, 0], [1, 0, 0]])
        fixture.certified(self)
        with wp.ScopedCapture(device=self.device) as capture:
            fixture.launch()
        fixture.capacity.zero_()
        wp.capture_launch(capture.graph)
        self.assertEqual(fixture.status.numpy()[0], 0)
        np.testing.assert_array_equal(fixture.impulses.numpy(), 0)


if __name__ == "__main__":
    unittest.main()
