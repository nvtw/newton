# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated unit tests for the position-level state-update infra.

CUDA-only, fast (under a second total). Three tests:

* :class:`TestSnapshotSyncRoundTrip` -- snapshot then sync without
  any iterate work in between recovers ``v = 0`` (no motion ->
  no velocity), confirming the kernels' arithmetic.
* :class:`TestPureTranslationRecoversVelocity` -- snapshot, advance
  positions by a known offset, sync. Recovered velocity must match
  ``offset / dt`` exactly. Validates the ``(p - p_pre) * inv_dt``
  branch.
* :class:`TestPureRotationRecoversAngularVelocity` -- snapshot,
  rotate orientations by a known angle-axis, sync. Recovered
  angular velocity must match the true ``omega`` within float32
  noise. Validates the ``2 * inv_dt * Im(q * conj(q_pre))`` branch
  including the negative-w sign flip.

These three exercise every code path in
:func:`sync_position_to_velocity_kernel`. The
:class:`PositionPass.run` convenience wrapper is exercised by the
first test (it calls ``snapshot`` -> noop iterate -> ``sync``).
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.position_level import PositionPass


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX position-level infra requires CUDA")
class TestSnapshotSyncRoundTrip(unittest.TestCase):
    """No motion -> recovered velocity is zero."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_zero_iterate_produces_zero_velocity(self) -> None:
        n = 4
        body_position = wp.array(
            np.array([[float(i), 0.0, 0.0] for i in range(n)], dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        body_orientation = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
            dtype=wp.quatf,
            device=self.device,
        )
        # Pre-fill velocity with garbage so we can confirm sync
        # actually overwrites it.
        body_velocity = wp.array(
            np.full((n, 3), 99.0, dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        body_angular_velocity = wp.array(
            np.full((n, 3), 99.0, dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        pp = PositionPass(num_bodies=n, device=self.device)
        pp.run(
            body_position,
            body_orientation,
            body_velocity,
            body_angular_velocity,
            iterate=lambda i: None,  # zero iterations would do; this drives the loop
            num_iterations=0,
            inv_dt=1.0 / 0.01,
        )
        v = body_velocity.numpy()
        w = body_angular_velocity.numpy()
        np.testing.assert_allclose(v, np.zeros((n, 3), dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(w, np.zeros((n, 3), dtype=np.float32), atol=1e-6)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX position-level infra requires CUDA")
class TestPureTranslationRecoversVelocity(unittest.TestCase):
    """A pure translation between snapshot and sync recovers
    ``v = (p_new - p_pre) * inv_dt`` exactly."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_per_body_translation(self) -> None:
        n = 3
        # Each body gets a different translation to make sure the
        # per-thread bookkeeping is right.
        translations = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        p0 = np.zeros((n, 3), dtype=np.float32)
        body_position = wp.array(p0.copy(), dtype=wp.vec3f, device=self.device)
        body_orientation = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
            dtype=wp.quatf,
            device=self.device,
        )
        body_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        pp = PositionPass(num_bodies=n, device=self.device)
        pp.snapshot(body_position, body_orientation)
        # "Iterate" by overwriting positions with the new values --
        # this is what an XPBD projection would do, just hand-rolled
        # for the test.
        body_position.assign(p0 + translations)
        dt = 0.01
        pp.sync_to_velocity(
            body_position,
            body_orientation,
            body_velocity,
            body_angular_velocity,
            inv_dt=1.0 / dt,
        )
        v = body_velocity.numpy()
        w = body_angular_velocity.numpy()
        np.testing.assert_allclose(v, translations / dt, atol=1e-5)
        np.testing.assert_allclose(w, np.zeros((n, 3), dtype=np.float32), atol=1e-5)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX position-level infra requires CUDA")
class TestPureRotationRecoversAngularVelocity(unittest.TestCase):
    """A pure rotation between snapshot and sync recovers
    ``omega = 2 * inv_dt * Im(q * conj(q_pre))`` (with shortest-arc
    sign flip)."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def _quat_axis_angle(self, axis_xyz, angle):
        ax = np.array(axis_xyz, dtype=np.float64)
        ax = ax / np.linalg.norm(ax)
        s = math.sin(0.5 * angle)
        c = math.cos(0.5 * angle)
        return np.array([ax[0] * s, ax[1] * s, ax[2] * s, c], dtype=np.float32)

    def test_small_angle_rotation(self) -> None:
        n = 1
        dt = 1.0 / 60.0
        # 0.05 rad rotation about +Z over one substep -> omega = 0.05/dt rad/s
        # along Z.
        angle = 0.05
        q_pre = self._quat_axis_angle((0.0, 0.0, 1.0), 0.0)
        q_post = self._quat_axis_angle((0.0, 0.0, 1.0), angle)
        body_position = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_orientation = wp.array(q_pre[None, :].copy(), dtype=wp.quatf, device=self.device)
        body_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        pp = PositionPass(num_bodies=n, device=self.device)
        pp.snapshot(body_position, body_orientation)
        body_orientation.assign(q_post[None, :])
        pp.sync_to_velocity(
            body_position,
            body_orientation,
            body_velocity,
            body_angular_velocity,
            inv_dt=1.0 / dt,
        )
        w = body_angular_velocity.numpy()[0]
        # For small angles the log-map is approximately
        # ``omega = (angle / dt) * axis``. ``2 * Im(q_post * conj(q_pre))
        # / dt`` equals ``2 * sin(angle/2) / dt`` along the axis,
        # which differs from ``angle/dt`` by the sin small-angle
        # error -- ~1e-4 at 0.05 rad. Use that as the tolerance.
        expected_z = 2.0 * math.sin(0.5 * angle) / dt
        self.assertAlmostEqual(w[0], 0.0, places=4)
        self.assertAlmostEqual(w[1], 0.0, places=4)
        self.assertAlmostEqual(w[2], expected_z, places=4)

    def test_large_angle_uses_shortest_arc(self) -> None:
        # Rotation > pi: the naive q * conj(q_pre) gives a quaternion
        # with negative w, and the kernel must flip the sign so we
        # recover the *shortest-arc* angular velocity.
        n = 1
        dt = 1.0 / 60.0
        angle = math.pi + 0.1  # 180.1 degrees -- shortest arc is the *other* way
        q_pre = self._quat_axis_angle((0.0, 0.0, 1.0), 0.0)
        q_post = self._quat_axis_angle((0.0, 0.0, 1.0), angle)
        body_position = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_orientation = wp.array(q_pre[None, :].copy(), dtype=wp.quatf, device=self.device)
        body_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        pp = PositionPass(num_bodies=n, device=self.device)
        pp.snapshot(body_position, body_orientation)
        body_orientation.assign(q_post[None, :])
        pp.sync_to_velocity(
            body_position,
            body_orientation,
            body_velocity,
            body_angular_velocity,
            inv_dt=1.0 / dt,
        )
        w = body_angular_velocity.numpy()[0]
        # Shortest arc to a +pi+0.1 rotation about +Z is -(pi - 0.1)
        # about +Z. The kernel's sign flip must produce a negative
        # ``w_z``.
        self.assertLess(w[2], 0.0, msg=f"angular velocity {w} should be negative on Z (shortest arc)")


if __name__ == "__main__":
    unittest.main()
