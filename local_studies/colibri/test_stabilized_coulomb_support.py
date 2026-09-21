"""A coupled analytic support equilibrium must remain a native Coulomb fixed point."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer, contact_container_zeros
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_coupled_velocity_update_no_soft_pd
from newton._src.solvers.phoenx.helpers.math_helpers import apply_pair_velocity_impulse


@wp.kernel
def _native_support(
    cc: ContactContainer,
    state: wp.array[wp.vec3f],
    copies: wp.float32,
    reverse: wp.int32,
    impulses: wp.array[wp.vec3f],
):
    v = state[0]
    w = state[1]
    inertia = wp.mat33f(copies, 0.0, 0.0, 0.0, copies, 0.0, 0.0, 0.0, copies)
    for j in range(2):
        k = j
        if reverse != 0:
            k = 1 - j
        x = wp.float32(-0.1)
        if k == 1:
            x = wp.float32(0.1)
        r = wp.vec3f(x, 0.0, -0.1)
        relative = v + wp.cross(w, r)
        eff = wp.float32(1.0) / (copies * wp.float32(1.01))
        impulse = contact_project_coupled_velocity_update_no_soft_pd(
            cc,
            k,
            wp.vec3f(0.0, 0.0, 1.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            relative[2],
            relative[0],
            relative[1],
            eff,
            eff,
            wp.float32(1.0) / (copies * wp.float32(1.02)),
            -0.02,
            0.0,
            0.0,
            0.5,
            0.5,
            0.9417004,
            0.058299546,
            1.0,
            0.0,
            0.0,
            0.0,
            copies * x * wp.float32(0.1),
            0.0,
            0.0,
        )
        unused_v, v, unused_w, w = apply_pair_velocity_impulse(
            wp.vec3f(0.0), v, wp.vec3f(0.0), w, 0.0, copies, wp.mat33f(0.0), inertia, r, r, impulse
        )
        impulses[k] = impulse
    state[0] = v
    state[1] = w


class TestStabilizedCoulombSupport(unittest.TestCase):
    def test_coupled_support_fixed_point_and_physical_impulse_accounting(self):
        # Independent sagittal system [vx,vz,wy], two normal supports and one
        # shared tangent direction. Recovery enters the SAME physical velocity.
        J = np.array([[0, 1, 0.1], [0, 1, -0.1], [1, 0, -0.1]], dtype=float)
        free = np.array([0.0001, -0.001, 0.0])
        bias = np.array([-0.02, -0.02, 0.0])
        mc = float(np.float32(0.9417004))
        ic = float(np.float32(0.058299546))
        for count in (1, 3, 7):
            A = count * (J @ J.T)
            gamma = np.array([ic / mc * A[0, 0], ic / mc * A[1, 1], 0.0])
            lam = np.linalg.solve(A + np.diag(gamma), -J @ free - bias)
            solved = free + count * J.T @ lam
            np.testing.assert_allclose(J @ solved + bias + gamma * lam, 0.0, atol=1e-16)
            self.assertTrue(np.all(lam[:2] > 0))
            self.assertTrue(np.all(abs(lam[2] * 0.5) < 0.5 * lam[:2]))
            tangent_work = lam[2] * (J[2] @ (free + solved)) * 0.5
            self.assertLess(tangent_work, 0.0)
            for reverse in (0, 1):
                cc = contact_container_zeros(2, device="cpu")
                multipliers = np.zeros((3, 2), dtype=np.float32)
                multipliers[0] = lam[:2]
                multipliers[1] = lam[2] * 0.5
                cc.impulses.assign(multipliers)
                initial = np.array([[solved[0], 0, solved[1]], [0, solved[2], 0]], dtype=np.float32)
                state = wp.array(initial, dtype=wp.vec3f, device="cpu")
                impulse = wp.zeros(2, dtype=wp.vec3f, device="cpu")
                wp.launch(_native_support, 1, [cc, state, float(count), reverse, impulse], device="cpu")
                final = state.numpy().copy().astype(float)
                applied = impulse.numpy().copy().astype(float)
                # One updated mass-scaled copy, followed by the real1/N mean:
                # physical P/L changes equal the ACTUAL common-point impulses.
                delta = (final - initial.astype(float)) / count
                levers = np.array([[-0.1, 0, -0.1], [0.1, 0, -0.1]])
                np.testing.assert_allclose(delta[0], applied.sum(0), atol=2e-9, rtol=0)
                np.testing.assert_allclose(delta[1], np.cross(levers, applied).sum(0), atol=2e-9, rtol=0)
                # A converged sticking solution is invariant under either row
                # order. Diagonal recovery subtraction instead destroys load.
                np.testing.assert_allclose(final, initial, atol=2e-7, rtol=0)
                np.testing.assert_allclose(cc.impulses.numpy(), multipliers, atol=2e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
