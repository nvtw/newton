# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`PhoenXWorld`'s global per-substep damping API."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.world_builder import WorldBuilder


def _free_cube(device, *, substeps: int = 1):
    """A single dynamic cube with non-zero linear + angular velocity in
    free space (no gravity, no contacts, no joints) -- isolates the
    damping kernel."""
    b = WorldBuilder()
    b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        velocity=(2.0, 4.0, 6.0),
        angular_velocity=(0.5, 0.0, 0.0),
        affected_by_gravity=False,
    )
    return b.finalize(
        substeps=substeps,
        solver_iterations=1,
        gravity=(0.0, 0.0, 0.0),
        device=device,
    )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA only")
class TestGlobalDampingAPI(unittest.TestCase):
    """Setter/getter behaviour and value validation."""

    def test_default_factors_are_zero(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device)
        self.assertEqual(world.get_global_linear_damping(), 0.0)
        self.assertEqual(world.get_global_angular_damping(), 0.0)

    def test_setters_are_independent(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device)
        world.set_global_linear_damping(0.3)
        world.set_global_angular_damping(0.7)
        self.assertAlmostEqual(world.get_global_linear_damping(), 0.3, places=5)
        self.assertAlmostEqual(world.get_global_angular_damping(), 0.7, places=5)
        # Re-setting one must not perturb the other.
        world.set_global_linear_damping(0.1)
        self.assertAlmostEqual(world.get_global_linear_damping(), 0.1, places=5)
        self.assertAlmostEqual(world.get_global_angular_damping(), 0.7, places=5)

    def test_rejects_out_of_range(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device)
        with self.assertRaises(ValueError):
            world.set_global_linear_damping(1.5)
        with self.assertRaises(ValueError):
            world.set_global_linear_damping(-0.1)
        with self.assertRaises(ValueError):
            world.set_global_angular_damping(1.5)
        with self.assertRaises(ValueError):
            world.set_global_angular_damping(-0.1)


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA only")
class TestGlobalDampingDynamics(unittest.TestCase):
    """Velocity decay rules per substep."""

    def test_default_no_op_preserves_velocity(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=4)
        world.step(0.01)
        v = world.bodies.velocity.numpy()[1]
        w = world.bodies.angular_velocity.numpy()[1]
        np.testing.assert_allclose(v, [2.0, 4.0, 6.0], atol=1e-5)
        np.testing.assert_allclose(w, [0.5, 0.0, 0.0], atol=1e-5)

    def test_factor_one_zeroes_velocity_in_one_substep(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=1)
        world.set_global_linear_damping(1.0)
        world.set_global_angular_damping(1.0)
        world.step(0.01)
        v = world.bodies.velocity.numpy()[1]
        w = world.bodies.angular_velocity.numpy()[1]
        np.testing.assert_allclose(v, 0.0, atol=1e-6)
        np.testing.assert_allclose(w, 0.0, atol=1e-6)

    def test_partial_damping_decays_per_substep(self) -> None:
        """``factor=0.5`` with ``substeps=4`` should decay velocity by
        ``0.5^4 = 0.0625`` after one ``step()`` (each substep multiplies
        by ``1 - 0.5 = 0.5``)."""
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=4)
        world.set_global_linear_damping(0.5)
        world.step(0.01)
        v = world.bodies.velocity.numpy()[1]
        np.testing.assert_allclose(v, np.array([2.0, 4.0, 6.0]) * 0.5**4, atol=1e-5)

    def test_linear_only_does_not_touch_angular(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=1)
        world.set_global_linear_damping(1.0)
        world.step(0.01)
        v = world.bodies.velocity.numpy()[1]
        w = world.bodies.angular_velocity.numpy()[1]
        np.testing.assert_allclose(v, 0.0, atol=1e-6)
        np.testing.assert_allclose(w, [0.5, 0.0, 0.0], atol=1e-5)

    def test_graph_replay_picks_up_factor_change(self) -> None:
        """Once the user has opted in (any setter called) and the
        backing array is allocated, a captured ``step()`` graph reads
        the damping value from the device slot at replay time, so a
        host-side setter call between replays takes effect on the next
        ``capture_launch`` without re-capture.

        Locking the kernel into the captured graph upfront with a
        ``set_*_damping(0.0)`` is the canonical pattern for users who
        want to enable damping later without re-capture."""
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=1)
        # Opt in upfront with a no-op factor so the captured graph
        # includes the damping kernel launch.
        world.set_global_linear_damping(0.0)
        # Warm-up + capture.
        world.step(0.01)
        with wp.ScopedCapture(device=device) as cap:
            world.step(0.01)
        graph = cap.graph

        # Replay with damping=0: velocity unchanged.
        wp.capture_launch(graph)
        v = world.bodies.velocity.numpy()[1]
        np.testing.assert_allclose(v, [2.0, 4.0, 6.0], atol=1e-5)

        # Flip to damping=1 and replay -> zero velocity.
        world.set_global_linear_damping(1.0)
        wp.capture_launch(graph)
        v = world.bodies.velocity.numpy()[1]
        np.testing.assert_allclose(v, 0.0, atol=1e-6)


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA only")
class TestGlobalDampingOptIn(unittest.TestCase):
    """Lazy-allocation behaviour: nothing is allocated and no kernel
    runs until the user calls a setter."""

    def test_no_alloc_until_setter_called(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=1)
        self.assertIsNone(world._global_damping)
        self.assertIsNone(world._global_damping_host)
        # A step should not allocate anything.
        world.step(0.01)
        self.assertIsNone(world._global_damping)

    def test_setter_allocates_once(self) -> None:
        device = wp.get_preferred_device()
        world = _free_cube(device, substeps=1)
        world.set_global_linear_damping(0.0)
        first = world._global_damping
        self.assertIsNotNone(first)
        # Subsequent setter calls must reuse the same allocation.
        world.set_global_linear_damping(0.5)
        world.set_global_angular_damping(0.5)
        self.assertIs(world._global_damping, first)


if __name__ == "__main__":
    wp.init()
    unittest.main()
