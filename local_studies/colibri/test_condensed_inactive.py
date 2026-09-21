"""CPU boundary regression for the algebraically inactive condensed point update."""

import itertools
import unittest

import numpy as np
import warp as wp

from local_studies.colibri.check_d6_frozen_rows import contact_sweep
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric


@wp.kernel(enable_backward=False)
def skipped_contact_sweep(
    c: wp.array2d[wp.float64],
    wct: wp.array2d[wp.float64],
    h: wp.array2d[wp.float64],
    bias: wp.array[wp.float64],
    gamma: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    first: int,
    count: int,
    size: int,
):
    """Apply original-order normal and production metric tangent updates."""
    for p in range(first, count):
        row = 3 * p
        vn = bias[row]
        for j in range(size):
            vn += c[row, j] * velocity[j]
        if (
            vn >= wp.float64(0)
            and lam[row] == wp.float64(0)
            and lam[row + 1] == wp.float64(0)
            and lam[row + 2] == wp.float64(0)
        ):
            continue
        oldn = lam[row]
        newn = wp.max(wp.float64(0.0), oldn - (vn + gamma[p] * oldn) / (h[row, row] + gamma[p]))
        dn = newn - oldn
        lam[row] = newn
        for j in range(size):
            velocity[j] += wct[j, row] * dn
        vt1 = bias[row + 1]
        vt2 = bias[row + 2]
        for j in range(size):
            vt1 += c[row + 1, j] * velocity[j]
            vt2 += c[row + 2, j] * velocity[j]
        old1 = lam[row + 1]
        old2 = lam[row + 2]
        radius = wp.float32(mu[p] * newn)
        tangent = contact_project_friction_metric(
            wp.float32(h[row + 1, row + 1]),
            wp.float32(h[row + 1, row + 2]),
            wp.float32(h[row + 2, row + 2]),
            wp.float32(vt1),
            wp.float32(vt2),
            wp.float32(old1),
            wp.float32(old2),
            radius,
            radius,
        )
        lam[row + 1] = wp.float64(tangent[0])
        lam[row + 2] = wp.float64(tangent[1])
        for j in range(size):
            velocity[j] += wct[j, row + 1] * (lam[row + 1] - old1) + wct[j, row + 2] * (lam[row + 2] - old2)


class TestInactiveCondensedContact(unittest.TestCase):
    """Cover compliance, opening/closing and nonzero cached tangential impulses."""

    def test_native_update_equivalence(self):
        """The shortcut must retain every native impulse and velocity update."""
        c = np.eye(3, 12)
        mobility = np.eye(12)
        mobility[:3, :3] = [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.0]]
        g = mobility @ c.T
        h = c @ g
        cases = 0
        for gamma, mu, vn, old in itertools.product(
            (0.0, 0.1, 10.0),
            (0.0, 0.5),
            (-1.0, 0.0, 1.0),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.2, -0.1)),
        ):
            outputs = []
            for kernel in (contact_sweep, skipped_contact_sweep):
                v = np.zeros(12)
                v[:3] = (vn, 0.3, -0.4)
                arrays = [
                    wp.array(x, dtype=wp.float64, device="cpu")
                    for x in (c, g, h, np.zeros(3), np.array([gamma]), np.array([mu]), np.array(old), v)
                ]
                wp.launch(kernel, dim=1, inputs=[*arrays, 0, 1, 12], device="cpu")
                outputs.append((arrays[-2].numpy(), arrays[-1].numpy()))
            for original, skipped in zip(outputs[0], outputs[1], strict=True):
                np.testing.assert_array_equal(original, skipped)
            cases += 1
        self.assertEqual(cases, 54)


if __name__ == "__main__":
    unittest.main()
