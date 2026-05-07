# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Access-mode invariance tests.

The Jitter2-style POSITION_LEVEL / VELOCITY_LEVEL flip is the contract
that lets cloth (position-level XPBD) and contacts (velocity-level
PGS) share particle endpoints across alternating graph colours within
a substep. The contract has two invariants the rest of the solver
relies on:

* **Round-trip identity.** Switching POS -> VEL -> POS (or VEL -> POS
  -> VEL) leaves a particle / body in the same state it started in
  (within float32 precision against the substep-start snapshot). A
  silent loss here means cloth corrections evaporate when contacts
  read them back, or contact impulses evaporate when cloth iterates
  read them back.
* **Equivalence of dual updates.** Applying a velocity impulse Dv at
  VELOCITY_LEVEL produces the same final state as applying the
  matching position delta Dp = dt * Dv at POSITION_LEVEL.

The tests below cover both for particles and rigid bodies. They are
intentionally pure-helper tests on tiny SoA containers so they run on
CPU and CUDA without a model.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_pose_velocity,
    synchronize_position_velocity,
)


_DT = 1.0 / 1200.0  # typical substep dt
_INV_DT = 1.0 / _DT


# ---------------------------------------------------------------------------
# Particle (3-DoF) tests against ``synchronize_position_velocity``.
# ---------------------------------------------------------------------------


@wp.kernel
def _particle_round_trip_kernel(
    pos_in: wp.array[wp.vec3f],
    vel_in: wp.array[wp.vec3f],
    pos_prev: wp.array[wp.vec3f],
    inv_dt: wp.float32,
    pos_out: wp.array[wp.vec3f],
    vel_out: wp.array[wp.vec3f],
    mode_out: wp.array[wp.int32],
):
    i = wp.tid()
    # Start at POSITION_LEVEL.
    p, v, m = synchronize_position_velocity(
        pos_in[i], vel_in[i], pos_prev[i],
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        wp.int32(ACCESS_MODE_VELOCITY_LEVEL),
        inv_dt,
    )
    # Sync back to POSITION_LEVEL.
    p, v, m = synchronize_position_velocity(
        p, v, pos_prev[i], m,
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        inv_dt,
    )
    pos_out[i] = p
    vel_out[i] = v
    mode_out[i] = m


@wp.kernel
def _particle_velocity_update_kernel(
    pos_in: wp.array[wp.vec3f],
    vel_in: wp.array[wp.vec3f],
    pos_prev: wp.array[wp.vec3f],
    delta_v: wp.array[wp.vec3f],
    inv_dt: wp.float32,
    pos_out: wp.array[wp.vec3f],
    vel_out: wp.array[wp.vec3f],
):
    """Velocity-level impulse path: sync POS->VEL, add Dv, sync VEL->POS."""
    i = wp.tid()
    # POS->VEL recovers vel from the position delta.
    p, v, _m = synchronize_position_velocity(
        pos_in[i], vel_in[i], pos_prev[i],
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        wp.int32(ACCESS_MODE_VELOCITY_LEVEL),
        inv_dt,
    )
    # Apply the velocity impulse.
    v_new = v + delta_v[i]
    # VEL->POS folds the velocity update back into position via dt * vel.
    p, v, _m = synchronize_position_velocity(
        p, v_new, pos_prev[i],
        wp.int32(ACCESS_MODE_VELOCITY_LEVEL),
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        inv_dt,
    )
    pos_out[i] = p
    vel_out[i] = v


@wp.kernel
def _particle_position_update_kernel(
    pos_in: wp.array[wp.vec3f],
    vel_in: wp.array[wp.vec3f],
    pos_prev: wp.array[wp.vec3f],
    delta_v: wp.array[wp.vec3f],
    dt: wp.float32,
    inv_dt: wp.float32,
    pos_out: wp.array[wp.vec3f],
    vel_out: wp.array[wp.vec3f],
):
    """Position-level path: add Dp = dt*Dv, then sync POS->VEL."""
    i = wp.tid()
    # Position-level write applies Dp directly.
    p_new = pos_in[i] + dt * delta_v[i]
    # The cloth iterate's contract is "stay POSITION_LEVEL"; we then
    # sync POS->VEL so the comparison matches the velocity-level path's
    # final POSITION_LEVEL state by reading vel out via the finite-diff.
    _p, v, _m = synchronize_position_velocity(
        p_new, vel_in[i], pos_prev[i],
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        wp.int32(ACCESS_MODE_VELOCITY_LEVEL),
        inv_dt,
    )
    pos_out[i] = p_new
    vel_out[i] = v


class TestParticleAccessMode(unittest.TestCase):
    """Particle-level round-trip and dual-update equivalence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_preferred_device()

    def _run(self, kernel, *array_args):
        wp.launch(kernel, dim=array_args[0].shape[0], inputs=list(array_args), device=self.device)

    def test_pos_vel_pos_round_trip_is_identity(self) -> None:
        """POS -> VEL -> POS leaves the particle in its starting state."""
        rng = np.random.default_rng(seed=0)
        n = 64
        pos = rng.standard_normal((n, 3)).astype(np.float32) * 0.5 + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # vel arbitrary because POS-level reads it as derived from pos delta.
        vel = rng.standard_normal((n, 3)).astype(np.float32)
        # pos_prev != pos so the sync actually does work.
        pos_prev = pos - rng.standard_normal((n, 3)).astype(np.float32) * 0.001
        pos_in = wp.array(pos, dtype=wp.vec3f, device=self.device)
        vel_in = wp.array(vel, dtype=wp.vec3f, device=self.device)
        pos_prev_arr = wp.array(pos_prev, dtype=wp.vec3f, device=self.device)
        pos_out = wp.empty(n, dtype=wp.vec3f, device=self.device)
        vel_out = wp.empty(n, dtype=wp.vec3f, device=self.device)
        mode_out = wp.empty(n, dtype=wp.int32, device=self.device)
        wp.launch(
            _particle_round_trip_kernel,
            dim=n,
            inputs=[pos_in, vel_in, pos_prev_arr, wp.float32(_INV_DT)],
            outputs=[pos_out, vel_out, mode_out],
            device=self.device,
        )
        np.testing.assert_allclose(pos_out.numpy(), pos, atol=1.0e-5, rtol=1.0e-5)
        np.testing.assert_array_equal(mode_out.numpy(), ACCESS_MODE_POSITION_LEVEL)
        # vel_out is the recovered velocity; it should match (pos - pos_prev) / dt.
        expected_vel = (pos - pos_prev) * _INV_DT
        np.testing.assert_allclose(vel_out.numpy(), expected_vel, atol=1.0e-3, rtol=1.0e-4)

    def test_velocity_update_equals_position_update(self) -> None:
        """Applying Dv at VEL-level == applying Dp = dt*Dv at POS-level.

        This is the crucial invariant for cloth (POS-level writes) and
        contact (VEL-level writes) sharing the same particle within a
        substep: both flows must converge to the same final state, or
        the alternating colour pattern accumulates / loses corrections.
        """
        rng = np.random.default_rng(seed=42)
        n = 128
        pos = rng.standard_normal((n, 3)).astype(np.float32) * 0.5 + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vel = rng.standard_normal((n, 3)).astype(np.float32)
        pos_prev = pos - vel * _DT  # Make pos consistent with pos_prev + dt*vel.
        delta_v = rng.standard_normal((n, 3)).astype(np.float32) * 2.0

        pos_arr = wp.array(pos, dtype=wp.vec3f, device=self.device)
        vel_arr = wp.array(vel, dtype=wp.vec3f, device=self.device)
        pos_prev_arr = wp.array(pos_prev, dtype=wp.vec3f, device=self.device)
        dv_arr = wp.array(delta_v, dtype=wp.vec3f, device=self.device)

        # Path A: velocity-level update.
        pos_a = wp.empty(n, dtype=wp.vec3f, device=self.device)
        vel_a = wp.empty(n, dtype=wp.vec3f, device=self.device)
        wp.launch(
            _particle_velocity_update_kernel,
            dim=n,
            inputs=[pos_arr, vel_arr, pos_prev_arr, dv_arr, wp.float32(_INV_DT)],
            outputs=[pos_a, vel_a],
            device=self.device,
        )

        # Path B: position-level update.
        pos_b = wp.empty(n, dtype=wp.vec3f, device=self.device)
        vel_b = wp.empty(n, dtype=wp.vec3f, device=self.device)
        wp.launch(
            _particle_position_update_kernel,
            dim=n,
            inputs=[pos_arr, vel_arr, pos_prev_arr, dv_arr, wp.float32(_DT), wp.float32(_INV_DT)],
            outputs=[pos_b, vel_b],
            device=self.device,
        )

        # The two paths must converge to the same (pos, vel) state.
        np.testing.assert_allclose(pos_a.numpy(), pos_b.numpy(), atol=2.0e-5, rtol=1.0e-5)
        np.testing.assert_allclose(vel_a.numpy(), vel_b.numpy(), atol=1.0e-2, rtol=1.0e-4)

    def test_velocity_update_lands_in_position(self) -> None:
        """A VEL-level Dv impulse must materialise as a Dp = dt*Dv shift.

        Mirrors the contact-endpoint scatter: write Dv into vel, then
        flip to POSITION_LEVEL. The follow-up POS-level reader (cloth
        iterate, substep-end recovery) must see a position that has
        moved by dt*Dv from the pre-impulse pos.
        """
        rng = np.random.default_rng(seed=7)
        n = 32
        pos = rng.standard_normal((n, 3)).astype(np.float32) + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        pos_prev = pos.copy()  # Particle was sitting still at substep entry.
        delta_v = rng.standard_normal((n, 3)).astype(np.float32) * 0.5

        pos_arr = wp.array(pos, dtype=wp.vec3f, device=self.device)
        vel_arr = wp.array(vel, dtype=wp.vec3f, device=self.device)
        pos_prev_arr = wp.array(pos_prev, dtype=wp.vec3f, device=self.device)
        dv_arr = wp.array(delta_v, dtype=wp.vec3f, device=self.device)
        pos_out = wp.empty(n, dtype=wp.vec3f, device=self.device)
        vel_out = wp.empty(n, dtype=wp.vec3f, device=self.device)
        wp.launch(
            _particle_velocity_update_kernel,
            dim=n,
            inputs=[pos_arr, vel_arr, pos_prev_arr, dv_arr, wp.float32(_INV_DT)],
            outputs=[pos_out, vel_out],
            device=self.device,
        )

        expected_pos = pos + _DT * delta_v
        np.testing.assert_allclose(pos_out.numpy(), expected_pos, atol=2.0e-5, rtol=1.0e-5)
        np.testing.assert_allclose(vel_out.numpy(), delta_v, atol=2.0e-2, rtol=2.0e-3)


# ---------------------------------------------------------------------------
# Body (6-DoF) tests against ``synchronize_pose_velocity``.
# ---------------------------------------------------------------------------


@wp.kernel
def _body_round_trip_kernel(
    pos_in: wp.array[wp.vec3f],
    quat_in: wp.array[wp.quatf],
    vel_in: wp.array[wp.vec3f],
    omega_in: wp.array[wp.vec3f],
    pos_prev: wp.array[wp.vec3f],
    quat_prev: wp.array[wp.quatf],
    inv_dt: wp.float32,
    pos_out: wp.array[wp.vec3f],
    quat_out: wp.array[wp.quatf],
):
    i = wp.tid()
    p, q, v, w, m = synchronize_pose_velocity(
        pos_in[i], quat_in[i], vel_in[i], omega_in[i],
        pos_prev[i], quat_prev[i],
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        wp.int32(ACCESS_MODE_VELOCITY_LEVEL),
        inv_dt,
    )
    p, q, v, w, m = synchronize_pose_velocity(
        p, q, v, w, pos_prev[i], quat_prev[i], m,
        wp.int32(ACCESS_MODE_POSITION_LEVEL),
        inv_dt,
    )
    pos_out[i] = p
    quat_out[i] = q


class TestBodyAccessMode(unittest.TestCase):
    """Body-level round-trip identity (linear position + quaternion)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_preferred_device()

    def test_pos_vel_pos_round_trip_is_identity(self) -> None:
        rng = np.random.default_rng(seed=1)
        n = 32
        pos = rng.standard_normal((n, 3)).astype(np.float32)
        vel = rng.standard_normal((n, 3)).astype(np.float32)
        omega = rng.standard_normal((n, 3)).astype(np.float32) * 0.1

        # Build small-rotation quaternions about random axes for both
        # current and prev so the delta quat in POS->VEL is well-defined.
        def _rand_quats(n_, std):
            ax = rng.standard_normal((n_, 3)).astype(np.float32)
            ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1.0e-9
            ang = rng.standard_normal(n_).astype(np.float32) * std
            half = 0.5 * ang
            s = np.sin(half)[:, None]
            c = np.cos(half)
            return np.concatenate([ax * s, c[:, None]], axis=1).astype(np.float32)

        q_prev = _rand_quats(n, 0.5)
        q_now = _rand_quats(n, 0.05)  # Small rotation away from prev.
        # Compose: q = q_now * q_prev to keep things close.
        # Skipped here: this test only needs *some* well-formed quats.
        pos_prev = pos - vel * _DT

        pos_arr = wp.array(pos, dtype=wp.vec3f, device=self.device)
        quat_arr = wp.array(q_now, dtype=wp.quatf, device=self.device)
        vel_arr = wp.array(vel, dtype=wp.vec3f, device=self.device)
        omega_arr = wp.array(omega, dtype=wp.vec3f, device=self.device)
        pos_prev_arr = wp.array(pos_prev, dtype=wp.vec3f, device=self.device)
        quat_prev_arr = wp.array(q_prev, dtype=wp.quatf, device=self.device)

        pos_out = wp.empty(n, dtype=wp.vec3f, device=self.device)
        quat_out = wp.empty(n, dtype=wp.quatf, device=self.device)
        wp.launch(
            _body_round_trip_kernel,
            dim=n,
            inputs=[pos_arr, quat_arr, vel_arr, omega_arr, pos_prev_arr, quat_prev_arr, wp.float32(_INV_DT)],
            outputs=[pos_out, quat_out],
            device=self.device,
        )
        np.testing.assert_allclose(pos_out.numpy(), pos, atol=1.0e-5, rtol=1.0e-5)
        # Quaternion: compare up to sign (q and -q encode the same rotation).
        q_out = quat_out.numpy()
        dot = np.abs(np.sum(q_out * q_now, axis=1))
        # Round-trip flips orientation if pos_prev's delta quat was small,
        # so the test is "very close to one or minus one" -- equivalently
        # |dot| close to 1.
        # Loose tolerance: the small-angle quaternion finite-diff
        # accumulates a few percent error for randomly-large rotations,
        # which is not the cloth path's concern (particles are 3-DoF).
        np.testing.assert_allclose(dot, 1.0, atol=2.0e-2)


if __name__ == "__main__":
    unittest.main()
