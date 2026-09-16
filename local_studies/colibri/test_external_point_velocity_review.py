"""Independent CPU audit of the proposed external point-velocity helper.

The boost regression intentionally fails for absolute-point-velocity subtraction.
No production source or tolerance is modified by this diagnostic.
"""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.external_point_velocity import external_relative_point_velocity
from local_studies.colibri.register_component_response import row_velocity_local


@wp.kernel(enable_backward=False)
def evaluate(values: wp.array2d[wp.vec3f], endpoints: wp.array2d[wp.int32], result: wp.array2d[wp.vec3f]):
    """Evaluate actual candidate and unchanged scalar joint-coordinate oracle."""
    i = wp.tid()
    root_v, root_w, child_w = values[i, 0], values[i, 1], values[i, 2]
    v0, w0, v1, w1 = values[i, 3], values[i, 4], values[i, 5], values[i, 6]
    linear, axis, shift = values[i, 7], values[i, 8], values[i, 9]
    r0, r1 = values[i, 10], values[i, 11]
    motion = wp.spatial_vectorf(linear[0], linear[1], linear[2], axis[0], axis[1], axis[2])
    b0, b1 = endpoints[i, 0], endpoints[i, 1]
    result[i, 0] = external_relative_point_velocity(
        root_v, root_w, child_w, v0, w0, v1, w1, 0, 1, motion, shift, b0, b1, r0, r1
    )
    original = wp.vec3f(0.0)
    for component in range(3):
        direction = wp.vec3f(0.0)
        direction[component] = 1.0
        original[component] = row_velocity_local(
            root_v, root_w, wp.vec3f(0.0), child_w, v0, w0, v1, w1, 0, 1, motion, shift, b0, b1, r0, r1, direction
        )
    result[i, 1] = original


def run_cases(values, endpoints):
    """Run only CPU kernels; return candidate and original row vectors."""
    output = wp.zeros((len(values), 2), dtype=wp.vec3f, device="cpu")
    wp.launch(
        evaluate,
        len(values),
        inputs=[
            wp.array(values, dtype=wp.vec3f, device="cpu"),
            wp.array(endpoints, dtype=wp.int32, device="cpu"),
            output,
        ],
        device="cpu",
    )
    return output.numpy()


class TestExternalPointVelocityReview(unittest.TestCase):
    """Check algebraic scope separately from cancellation sensitivity."""

    def test_root_child_and_reversed_moving_endpoint(self):
        """Preserve root/child signs and moving external endpoint contributions."""
        rng = np.random.default_rng(6184)
        values = rng.uniform(-0.01, 0.01, (128, 12, 3)).astype(np.float32)
        values[:, 8] = (0, 0, 1)
        endpoints = np.array([(i % 2, 2) if i % 4 < 2 else (2, i % 2) for i in range(128)], dtype=np.int32)
        results = run_cases(values, endpoints)
        np.testing.assert_allclose(results[:, 0], results[:, 1], atol=1e-8, rtol=0)

    def test_common_translation_preserves_resolved_relative_velocity(self):
        """Retain a resolved 0.1 micrometer/s row under a 16 m/s common boost."""
        values = np.zeros((2, 12, 3), dtype=np.float32)
        values[:, 1, 2] = 1.0
        values[:, 8, 2] = 1.0
        values[:, 10, 1] = 1e-7
        values[1, 0, 0] = 16.0
        values[1, 5, 0] = 16.0
        results = run_cases(values, np.array([[0, 2], [0, 2]], dtype=np.int32))
        print("EXTERNAL_BOOST_REVIEW", results.tolist(), flush=True)
        np.testing.assert_array_equal(results[0, 1], results[1, 1])
        np.testing.assert_allclose(results[:, 0], results[:, 1], atol=1e-8, rtol=0)

    def test_moving_kinematic_rotation_preserves_lever_difference(self):
        """Retain small rotational slip between nearly equal resolved levers."""
        values = np.zeros((1, 12, 3), dtype=np.float32)
        values[0, 1, 2] = 10.0
        values[0, 6, 2] = 10.0
        values[0, 8, 2] = 1.0
        values[0, 10, 1] = 1.0
        values[0, 11, 1] = np.nextafter(np.float32(1.0), np.float32(2.0))
        results = run_cases(values, np.array([[0, 2]], dtype=np.int32))
        print("EXTERNAL_ROTATION_REVIEW", results.tolist(), flush=True)
        np.testing.assert_allclose(results[:, 0], results[:, 1], atol=1e-8, rtol=0)

    def test_static_endpoint_near_instantaneous_center(self):
        """Preserve resolved rolling-contact slip against a stationary endpoint."""
        values = np.zeros((1, 12, 3), dtype=np.float32)
        values[0, 0, 0] = np.nextafter(np.float32(10.0), np.float32(11.0))
        values[0, 1, 2] = 10.0
        values[0, 8, 2] = 1.0
        values[0, 10, 1] = np.nextafter(np.float32(1.0), np.float32(2.0))
        results = run_cases(values, np.array([[0, 2]], dtype=np.int32))
        print("EXTERNAL_STATIC_ROLLING_REVIEW", results.tolist(), flush=True)
        np.testing.assert_allclose(results[:, 0], results[:, 1], atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
