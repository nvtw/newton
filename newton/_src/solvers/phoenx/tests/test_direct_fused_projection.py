# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ragdoll gates for the fused one-panel D6 projection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.articulations.block_joint_system import MixedJointSystem
from newton._src.solvers.phoenx.mass_splitting.copy_state import copy_state_container_zeros
from newton._src.solvers.phoenx.mass_splitting.kernels import launch_average_rigid_velocity_into_bodies
from newton.examples.phoenx import example_phoenx_ragdoll_pile as scene
from newton.viewer import ViewerNull


def run(fused: bool, frames: int = 12):
    original_factory = scene.make_colored_solver

    def make_solver(*args, **kwargs):
        solver = original_factory(*args, **kwargs)
        direct = solver.world._direct_equality_system
        exact = direct.exact if isinstance(direct, MixedJointSystem) else direct
        assert exact._fused_projection is not None, "ragdolls should auto-select the fused projection"
        if not fused:
            exact._fused_projection = None
        return solver

    scene.make_colored_solver = make_solver
    viewer = ViewerNull()
    try:
        example = scene.Example(
            viewer,
            SimpleNamespace(grid=(4, 4, 4), seed=42, substeps=8, iterations=4, joint_solver="exact"),
        )
        world = example.solver.world
        snapshots = []
        for _ in range(frames):
            example.step()
            wp.synchronize()
            snapshots.append(
                (
                    example.state.body_q.numpy().copy(),
                    example.state.body_qd.numpy().copy(),
                    world._contact_container.impulses.numpy().copy(),
                )
            )
        return snapshots
    finally:
        scene.make_colored_solver = original_factory
        viewer.close()


class TestDirectFusedProjection(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_fused_projection_matches_separate_kernels_bitwise(self):
        """One fused projection reproduces RHS, solve and accumulation exactly, and body updates to rounding."""
        viewer = ViewerNull()
        try:
            example = scene.Example(
                viewer,
                SimpleNamespace(grid=(4, 4, 4), seed=42, substeps=8, iterations=4, joint_solver="exact"),
            )
            for _ in range(6):
                example.step()
            direct = example.solver.world._direct_equality_system
            exact = direct.exact if isinstance(direct, MixedJointSystem) else direct
            fused_kernel = exact._fused_projection
            self.assertIsNotNone(fused_kernel)
            bodies = exact.bodies
            state = (bodies.velocity, bodies.angular_velocity, exact.accumulated_impulse)
            initial = [array.numpy().copy() for array in state]
            # Perturb velocities so every mechanism has a correction to make.
            rng = np.random.default_rng(3)
            bodies.velocity.assign(initial[0] + rng.normal(scale=0.01, size=initial[0].shape).astype(np.float32))
            perturbed = [array.numpy().copy() for array in state]
            for use_bias in (True, False):
                results = []
                for fused in (False, True):
                    for array, values in zip(state, perturbed, strict=True):
                        array.assign(values)
                    exact._fused_projection = fused_kernel if fused else None
                    exact._fused_relax_tolerance = -1.0
                    exact.solve(use_bias=use_bias)
                    wp.synchronize()
                    if not fused and not use_bias:
                        # Biased solves always run; relaxation tests its residual.
                        self.assertEqual(int(exact.solve_active.numpy()[0]), 1)
                    results.append([array.numpy().copy() for array in state])
                for name, reference, candidate in zip(
                    ("velocity", "angular_velocity", "impulse"), *results, strict=True
                ):
                    if name == "impulse":
                        np.testing.assert_array_equal(candidate, reference, err_msg=f"{name}, bias={use_bias}")
                    else:
                        # The separate apply sums a body's rows with a warp tree reduction;
                        # the fused projection sums them serially, so only rounding differs.
                        np.testing.assert_allclose(
                            candidate, reference, rtol=1e-5, atol=1e-7, err_msg=f"{name}, bias={use_bias}"
                        )
            exact._fused_projection = fused_kernel
        finally:
            viewer.close()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_fused_copy_average_matches_standalone_kernel_bitwise(self):
        """Averaging overflow copies inside the projection equals the separate average pass."""
        viewer = ViewerNull()
        try:
            example = scene.Example(
                viewer,
                SimpleNamespace(grid=(4, 4, 4), seed=42, substeps=8, iterations=4, joint_solver="exact"),
            )
            example.step()
            world = example.solver.world
            direct = world._direct_equality_system
            exact = direct.exact if isinstance(direct, MixedJointSystem) else direct
            self.assertTrue(direct.fuses_copy_average)
            bodies = exact.bodies
            # Exercise both a partial eight-slot reduction and one spanning two chunks.
            counts = np.zeros(bodies.velocity.shape[0], dtype=np.int32)
            dynamic = np.flatnonzero(bodies.inverse_mass.numpy() > 0.0)
            counts[dynamic[:2]] = (3, 11)
            slots = int(counts.sum())
            copies = copy_state_container_zeros(slots, len(counts), bodies.velocity.device)
            copies.count_per_node.assign(counts)
            copies.section_end.assign(np.cumsum(counts, dtype=np.int32))
            copies.highest_index_in_use.fill_(slots)
            copies.access_mode.fill_(ACCESS_MODE_VELOCITY_LEVEL)
            rng = np.random.default_rng(5)
            for velocities in (copies.velocity, copies.angular_velocity):
                velocities.assign(rng.normal(scale=0.05, size=(slots, 3)).astype(np.float32))
            state = (bodies.velocity, bodies.angular_velocity, exact.accumulated_impulse)
            initial = [array.numpy().copy() for array in state]
            results = []
            for fused in (False, True):
                for array, values in zip(state, initial, strict=True):
                    array.assign(values)
                if not fused:
                    launch_average_rigid_velocity_into_bodies(copies, bodies)
                exact.solve(use_bias=True, copy_state=copies, average_copies=fused)
                wp.synchronize()
                results.append([array.numpy().copy() for array in state])
            self.assertFalse(np.array_equal(results[0][0], initial[0]))
            for name, reference, candidate in zip(("velocity", "angular_velocity", "impulse"), *results, strict=True):
                np.testing.assert_array_equal(candidate, reference, err_msg=name)
        finally:
            viewer.close()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_per_mechanism_relaxation_skip_is_negligible(self):
        """Skipping quiet mechanisms changes early frames only at rounding scale."""
        separate = run(False, frames=8)
        fused = run(True, frames=8)
        for frame, (before, after) in enumerate(zip(separate, fused, strict=True), start=1):
            np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=1.0e-6, err_msg=f"poses at frame {frame}")
            np.testing.assert_allclose(after[1], before[1], rtol=0.0, atol=1.0e-4, err_msg=f"velocity at frame {frame}")


if __name__ == "__main__":
    unittest.main()
