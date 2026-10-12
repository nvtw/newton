# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bitwise ragdoll gate and construction guards for automatic D6 factor reuse."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.block_joint_system import MixedJointSystem
from newton._src.solvers.phoenx.articulations.direct_equality import DirectEqualitySystem
from newton._src.solvers.phoenx.solver import _can_reuse_relax_direct_factor
from newton.examples.phoenx import example_phoenx_ragdoll_pile as scene
from newton.viewer import ViewerNull


def _array(values):
    return SimpleNamespace(numpy=lambda: np.asarray(values))


def check_guards():
    exact = object.__new__(DirectEqualitySystem)
    exact.enabled = True
    exact.has_dynamic_rows = False
    exact.has_bounded_drives = False
    exact.has_direct_friction = False
    exact.topology = SimpleNamespace(joints=np.array([0], dtype=np.int32))
    mixed = object.__new__(MixedJointSystem)
    mixed.exact = exact
    mixed.blocks = SimpleNamespace(enabled=False)
    world = SimpleNamespace(
        step_layout="single_world",
        mass_splitting_unrolled=True,
        velocity_relaxation="each_substep",
        velocity_iterations=1,
        _sleeping_enabled=False,
    )
    model = SimpleNamespace(joint_parent=_array([0]), joint_child=_array([1]))
    bodies = SimpleNamespace(motion_type=_array([0, 2, 2]))

    def eligible():
        return _can_reuse_relax_direct_factor(mixed, world, model, bodies, "soft")

    assert eligible(), "static exact joints should reuse the factor"
    for owner, field, unsafe in (
        (exact, "has_dynamic_rows", True),
        (exact, "has_bounded_drives", True),
        (exact, "has_direct_friction", True),
        (mixed.blocks, "enabled", True),
        (world, "_sleeping_enabled", True),
        (world, "velocity_relaxation", "end_of_step"),
        (world, "velocity_iterations", 0),
        (world, "step_layout", "multi_world"),
        (world, "mass_splitting_unrolled", False),
    ):
        original = getattr(owner, field)
        setattr(owner, field, unsafe)
        assert not eligible(), field
        setattr(owner, field, original)
    bodies.motion_type = _array([0, 1, 2])
    assert not eligible(), "moving kinematic parent must disable reuse"
    bodies.motion_type = _array([0, 2, 1])
    assert not eligible(), "moving kinematic child must disable reuse"
    bodies.motion_type = _array([0, 2, 2])
    assert not _can_reuse_relax_direct_factor(mixed, world, model, bodies, "tgs")
    print("D6 factor reuse guards passed")


def run(reuse: bool, frames: int = 12):
    original_factory = scene.make_colored_solver
    factor_calls = 0

    def make_counted_solver(*args, **kwargs):
        nonlocal factor_calls
        solver = original_factory(*args, **kwargs)
        world = solver.world
        assert world._reuse_exact_relax_factor, "ragdoll should auto-select reuse"
        if not reuse:
            world._reuse_exact_relax_factor = False
        direct = world._direct_equality_system
        exact = direct.exact if isinstance(direct, MixedJointSystem) else direct
        original_prepare = exact.prepare_and_factor

        def count_prepare(*factor_args, **factor_kwargs):
            nonlocal factor_calls
            factor_calls += 1
            return original_prepare(*factor_args, **factor_kwargs)

        exact.prepare_and_factor = count_prepare
        return solver

    scene.make_colored_solver = make_counted_solver
    viewer = ViewerNull()
    try:
        example = scene.Example(
            viewer,
            SimpleNamespace(grid=(4, 4, 4), seed=42, substeps=8, iterations=4, joint_solver="exact"),
        )
        expected_calls = 9 if reuse else 16
        assert factor_calls == expected_calls, (reuse, factor_calls, expected_calls)
        world = example.solver.world
        snapshots = []
        for frame in range(frames):
            example.step()
            wp.synchronize()
            contacts = world._contact_container
            snapshots.append(
                (
                    example.state.body_q.numpy().copy(),
                    example.state.body_qd.numpy().copy(),
                    contacts.impulses.numpy().copy(),
                    contacts.lambdas.numpy().copy(),
                    example.contacts.rigid_contact_count.numpy().copy(),
                )
            )
            print(f"reuse={reuse} frame={frame + 1}", flush=True)
        return snapshots
    finally:
        scene.make_colored_solver = original_factory
        viewer.close()


class TestDirectFactorReuse(unittest.TestCase):
    def test_configuration_guards(self):
        check_guards()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_bitwise_replay_and_factor_count(self):
        baseline = run(False)
        reused = run(True)
        for frame, (before, after) in enumerate(zip(baseline, reused, strict=True), start=1):
            for field, reference, cached in zip(
                ("poses", "velocities", "impulses", "lambdas", "point_count"), before, after, strict=True
            ):
                np.testing.assert_array_equal(reference, cached, err_msg=f"{field} at frame {frame}")


if __name__ == "__main__":
    unittest.main()
