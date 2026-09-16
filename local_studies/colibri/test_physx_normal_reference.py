"""Independent scalar examples for the literal PhysX hard-normal translation."""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.physx_normal_reference import normal_impulse_tgs


@wp.kernel
def evaluate(rows: wp.array2d[wp.float32], results: wp.array[wp.vec2f]):
    i = wp.tid()
    results[i] = normal_impulse_tgs(
        rows[i, 0],
        rows[i, 1],
        rows[i, 2],
        rows[i, 3],
        rows[i, 4],
        rows[i, 5],
        rows[i, 6],
        rows[i, 7],
        rows[i, 8],
        rows[i, 9],
        rows[i, 10],
        rows[i, 11],
        rows[i, 12],
    )


class TestPhysXNormalReference(unittest.TestCase):
    def test_unilateral_temporal_and_capacity_cases(self):
        # gap, delta linear, delta angular0/1, minPen, target, elapsed,
        # reciprocal response, bias coefficient, negative max recovery speed,
        # normal velocity, old impulse, upper impulse.
        rows = [
            [0, 0, 0, 0, -100, 0, 0, 2, -3600, -2, -1, 0, 100],
            [0.001, 0, 0, 0, -100, 0, 0, 2, -3600, -2, -1, 0, 100],
            [0.001, 0, 0, 0, -100, 0, 0, 2, -3600, -2, -4, 0, 100],
            [-0.001, 0, 0, 0, -100, 0, 0, 0.5, -2880, -2, 0, 0, 100],
            [0, 0, 0, 0, -100, 0, 0, 1, -3600, -2, 2, 3, 100],
            [0, 0, 0, 0, -100, 0, 0, 1, -3600, -2, 10, 3, 100],
            [0, 0, 0, 0, -100, 0, 0, 1, -3600, -2, -1, 0, 0.25],
            [0.001, -0.001, 0, 0, -100, 0, 0, 1, -3600, -2, -1, 0, 100],
            [0.0005, 0, 0, 0, -100, 0.5, 0.001, 1, -3600, -2, 0, 0, 100],
            [0.001, 0, -0.002, -0.001, -100, 0, 0, 1, -3600, -2, -1, 0, 100],
            # Velocity passes disable overlap recovery, retaining separation caps.
            [-0.001, 0, 0, 0, 0, 0, 0, 1, -3600, -2, 0, 0, 100],
            [0.001, 0, 0, 0, 0, 0, 0, 2, -3600, -2, -4, 0, 100],
        ]
        expected_new = np.array([2, 0, 0.8, 1, 1, 0, 0.25, 1, 0.5, 1, 0, 0.8])
        data = np.array(rows, dtype=np.float32)
        result = wp.zeros(len(rows), dtype=wp.vec2f, device="cpu")
        wp.launch(evaluate, dim=len(rows), inputs=[wp.array(data, device="cpu"), result], device="cpu")
        actual = result.numpy().copy()
        np.testing.assert_allclose(actual[:, 0], expected_new, atol=6e-7, rtol=0)
        np.testing.assert_allclose(actual[:, 1], expected_new - data[:, 11], atol=6e-7, rtol=0)
        self.assertTrue(np.all(actual[:, 0] >= 0))
        self.assertTrue(np.all(actual[:, 0] <= data[:, 12]))


if __name__ == "__main__":
    unittest.main()
