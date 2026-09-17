# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mixed-normal linked patches conserve momentum with independent load budgets."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_anchors import (
    PatchAnchors,
    prepare,
)
from newton._src.solvers.phoenx.constraints.contact_tgs_anchors import (
    allocate as allocate_anchors,
)
from newton._src.solvers.phoenx.constraints.contact_tgs_friction import solve
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import (
    NormalPatches,
    partition,
)
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import (
    allocate as allocate_partition,
)


@wp.kernel
def sweep(
    patches: NormalPatches,
    anchors: PatchAnchors,
    loads: wp.array[wp.float32],
    impulses: wp.array2d[wp.vec3f],
    poses: wp.array[wp.transformf],
    velocity: wp.array[wp.vec3f],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia: wp.array[wp.mat33f],
):
    v0 = wp.vec3f(velocity[0])
    v1 = wp.vec3f(velocity[1])
    w0 = wp.vec3f(velocity[2])
    w1 = wp.vec3f(velocity[3])
    patch = patches.group_first[0]
    while patch >= 0:
        v0, v1, w0, w1 = solve(
            patches,
            anchors,
            patch,
            loads,
            impulses,
            poses[0],
            poses[1],
            v0,
            v1,
            w0,
            w1,
            inverse_mass[0],
            inverse_mass[1],
            inverse_inertia[0],
            inverse_inertia[1],
            0.5,
            0.3,
            1000.0,
            0.36514837,
            False,
        )
        patch = patches.patch_next[patch]
    velocity[0] = v0
    velocity[1] = v1
    velocity[2] = w0
    velocity[3] = w1


class TestNormalPatchSolve(unittest.TestCase):
    def test_mixed_patch_momentum_and_loads(self):
        """Conserve both momenta while enforcing each patch friction budget."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                self._check_device(device)

    def _check_device(self, device):
        patches, empty = allocate_partition(4, 1, device), allocate_partition(4, 1, device)
        normals = wp.array([[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=wp.vec3f, device=device)
        first, count, active, absent = [wp.array([v], dtype=int, device=device) for v in (0, 4, 1, 0)]
        wp.launch(partition, 1, [normals, first, count, active, 0.999, patches], device=device)
        anchors, previous = allocate_anchors(4, device), allocate_anchors(4, device)
        poses_np = np.array([[-0.4, 0.2, 0.7, 0, 0, 0, 1], [0.3, -0.1, -0.2, 0, 0, 0, 1]], dtype=np.float32)
        poses = wp.array(poses_np, dtype=wp.transformf, device=device)
        keys = wp.array([[0, 0, 1, 7]], dtype=wp.vec4i, device=device)
        points = wp.array([[-0.1, 0, 0], [0, -0.1, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=wp.vec3f, device=device)
        gaps = wp.zeros(4, dtype=float, device=device)
        wp.launch(
            prepare,
            1,
            [
                patches,
                empty,
                keys,
                keys,
                active,
                absent,
                poses,
                points,
                gaps,
                0.00025,
                0.0004,
                previous,
                anchors,
            ],
            device=device,
        )
        np.testing.assert_array_equal(anchors.count.numpy()[[0, 1]], [2, 2])
        # Material points drift apart after preparation. Forces still
        # act in equal-and-opposite pairs at a common world point.
        poses_np[0, :3] += [0.003, -0.002, 0.001]
        poses_np[1, :3] += [-0.001, 0.004, -0.002]
        poses.assign(poses_np)
        mass = np.array([2.0, 3.0])
        inertia = np.array(
            [[[0.7, 0.1, 0], [0.1, 0.4, 0], [0, 0, 0.3]], [[1.2, 0, 0.1], [0, 0.8, 0], [0.1, 0, 0.6]]],
            dtype=np.float32,
        )
        initial = np.array([[0.2, -0.1, 0], [-0.3, 0.4, 0], [0.1, -0.2, 0.8], [-0.1, 0.2, -0.5]], dtype=np.float32)
        velocity = wp.array(initial, dtype=wp.vec3f, device=device)
        im = wp.array((1 / mass).astype(np.float32), dtype=float, device=device)
        ii = wp.array(np.linalg.inv(inertia), dtype=wp.mat33f, device=device)
        loads = wp.array([0.1, 0.02, 0.2, 0.07], dtype=float, device=device)
        impulses = wp.zeros((4, 2), dtype=wp.vec3f, device=device)
        wp.launch(sweep, 1, [patches, anchors, loads, impulses, poses, velocity, im, ii], device=device)
        final = velocity.numpy().astype(float)

        def totals(v):
            linear = mass[:, None] * v[:2]
            angular = np.cross(poses_np[:, :3], linear) + np.einsum("bij,bj->bi", inertia, v[2:])
            energy = 0.5 * (np.sum(linear * v[:2]) + np.sum(v[2:] * np.einsum("bij,bj->bi", inertia, v[2:])))
            return np.r_[linear.sum(0), angular.sum(0)], energy

        before, e0 = totals(initial.astype(float))
        after, e1 = totals(final)
        np.testing.assert_allclose(after, before, atol=3e-7, rtol=0)
        self.assertLessEqual(e1, e0 + 1e-8)
        applied = impulses.numpy().astype(float)
        for patch, load, normal in ((0, 0.3, [0, 0, 1]), (1, 0.09, [1, 0, 0])):
            self.assertLessEqual(float(np.linalg.norm(applied[patch], axis=1).max()), 0.5 * load / 2 + 1e-7)
            np.testing.assert_allclose(applied[patch] @ normal, 0, atol=1e-8)
        np.testing.assert_allclose(mass[1] * (final[1] - initial[1]), applied.sum(axis=(0, 1)), atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
