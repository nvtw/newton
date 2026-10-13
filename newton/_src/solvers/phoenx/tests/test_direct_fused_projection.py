# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bitwise ragdoll gate for the fused one-panel D6 projection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.block_joint_system import MixedJointSystem
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
    def test_fused_projection_replays_bitwise(self):
        """Fusing RHS, solve, accumulation and body updates keeps every captured frame."""
        separate = run(False)
        fused = run(True)
        for frame, (before, after) in enumerate(zip(separate, fused, strict=True), start=1):
            for field, reference, candidate in zip(("poses", "velocities", "impulses"), before, after, strict=True):
                np.testing.assert_array_equal(reference, candidate, err_msg=f"{field} at frame {frame}")


if __name__ == "__main__":
    unittest.main()
