# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default contact-coefficient specialization tests."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    _default_contact_solver_coefficients,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    soft_constraint_coefficients,
)


@wp.kernel
def _compare_default_contact_coefficients(
    idts: wp.array[wp.float32],
    specialized: wp.array2d[wp.float32],
    reference: wp.array2d[wp.float32],
):
    i = wp.tid()
    mass_coeff, impulse_coeff = _default_contact_solver_coefficients(idts[i])
    specialized[i, 0] = mass_coeff
    specialized[i, 1] = impulse_coeff

    dt = wp.float32(1.0) / idts[i]
    _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT,
        DEFAULT_DAMPING_RATIO,
        dt,
    )
    reference[i, 0] = mass_coeff
    reference[i, 1] = impulse_coeff


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact tests require CUDA")
class TestDefaultContactCoefficients(unittest.TestCase):
    def test_specialization_matches_reference_under_graph_capture(self):
        idts = wp.array(
            np.array([1.0, 60.0, 300.0, 360.0, 1.0e6, 1.0e9, 2.0e9, 2.1e9, 4.0e9], dtype=np.float32),
            device="cuda:0",
        )
        specialized = wp.empty((len(idts), 2), dtype=wp.float32, device="cuda:0")
        reference = wp.empty_like(specialized)

        wp.launch(
            _compare_default_contact_coefficients,
            dim=len(idts),
            inputs=[idts, specialized, reference],
            device="cuda:0",
        )
        with wp.ScopedCapture(device="cuda:0") as capture:
            wp.launch(
                _compare_default_contact_coefficients,
                dim=len(idts),
                inputs=[idts, specialized, reference],
                device="cuda:0",
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(specialized.numpy(), reference.numpy())


if __name__ == "__main__":
    unittest.main()
