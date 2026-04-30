# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Schema round-trip tests for the cloth triangle constraint row.

Commit 1 of the cloth-triangle plan -- the prepare / iterate
``@wp.func``s aren't there yet, so the only thing we can test
without reaching into the dispatcher is that:

* :func:`assert_constraint_header` accepts the schema (already
  asserted at import time, but we re-import in CI and verify
  there's no error path).
* The dword count is sane and fits inside the joint-side
  ``ConstraintContainer`` width.
* Every getter / setter pair round-trips a value through the
  dword storage.
* The ``constraint_type`` tag at dword 0 reads as
  :data:`CONSTRAINT_TYPE_CLOTH_TRIANGLE` after
  :func:`cloth_triangle_set_type`.

CUDA-only by Newton convention.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    CLOTH_TRIANGLE_DWORDS,
    cloth_triangle_get_alpha_lambda,
    cloth_triangle_get_alpha_mu,
    cloth_triangle_get_bias_lambda,
    cloth_triangle_get_bias_mu,
    cloth_triangle_get_body1,
    cloth_triangle_get_body2,
    cloth_triangle_get_body3,
    cloth_triangle_get_inv_mass_a,
    cloth_triangle_get_inv_mass_b,
    cloth_triangle_get_inv_mass_c,
    cloth_triangle_get_inv_rest,
    cloth_triangle_get_lambda_sum_lambda,
    cloth_triangle_get_lambda_sum_mu,
    cloth_triangle_get_rest_area,
    cloth_triangle_set_alpha_lambda,
    cloth_triangle_set_alpha_mu,
    cloth_triangle_set_bias_lambda,
    cloth_triangle_set_bias_mu,
    cloth_triangle_set_body1,
    cloth_triangle_set_body2,
    cloth_triangle_set_body3,
    cloth_triangle_set_inv_mass_a,
    cloth_triangle_set_inv_mass_b,
    cloth_triangle_set_inv_mass_c,
    cloth_triangle_set_inv_rest,
    cloth_triangle_set_lambda_sum_lambda,
    cloth_triangle_set_lambda_sum_mu,
    cloth_triangle_set_rest_area,
    cloth_triangle_set_type,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_TRIANGLE,
    ConstraintContainer,
    constraint_container_zeros,
    read_int,
)


class TestClothTriangleSchemaSize(unittest.TestCase):
    """Plain Python checks: dword count, header invariant."""

    def test_dword_count_in_reasonable_range(self) -> None:
        # Header (3) + body3 (1) + inv_rest (4) + rest_area (1) +
        # alpha_lambda/mu (2) + inv_mass_a/b/c (3) + bias_lambda/mu (2) +
        # lambda_sum_lambda/mu (2) = 18.
        self.assertEqual(CLOTH_TRIANGLE_DWORDS, 18)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-triangle tests require CUDA")
class TestClothTriangleSchemaRoundTrip(unittest.TestCase):
    """Each setter writes and the matching getter reads back the same
    value; type-tag stamp lands at dword 0."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        self.container = constraint_container_zeros(
            num_constraints=4,
            num_dwords=CLOTH_TRIANGLE_DWORDS,
            device=self.device,
        )

    def test_round_trip_all_fields(self) -> None:
        # Drive a single round-trip kernel that writes one
        # known-distinct value per field for every cid in the
        # container, then reads them back into output arrays for
        # comparison. Distinct per-cid values catch indexing bugs
        # that would otherwise pass with a single-cid check.
        n = 4
        out_type = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_body1 = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_body2 = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_body3 = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_inv_rest = wp.zeros(n, dtype=wp.mat22f, device=self.device)
        out_rest_area = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_alpha_lambda = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_alpha_mu = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_inv_mass_a = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_inv_mass_b = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_inv_mass_c = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_bias_lambda = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_bias_mu = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_lambda_sum_lambda = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_lambda_sum_mu = wp.zeros(n, dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=_round_trip_kernel,
            dim=n,
            inputs=[self.container],
            outputs=[
                out_type,
                out_body1,
                out_body2,
                out_body3,
                out_inv_rest,
                out_rest_area,
                out_alpha_lambda,
                out_alpha_mu,
                out_inv_mass_a,
                out_inv_mass_b,
                out_inv_mass_c,
                out_bias_lambda,
                out_bias_mu,
                out_lambda_sum_lambda,
                out_lambda_sum_mu,
            ],
            device=self.device,
        )

        type_arr = out_type.numpy()
        body1_arr = out_body1.numpy()
        body2_arr = out_body2.numpy()
        body3_arr = out_body3.numpy()
        inv_rest_arr = out_inv_rest.numpy()
        rest_area_arr = out_rest_area.numpy()
        alpha_lambda_arr = out_alpha_lambda.numpy()
        alpha_mu_arr = out_alpha_mu.numpy()
        inv_mass_a_arr = out_inv_mass_a.numpy()
        inv_mass_b_arr = out_inv_mass_b.numpy()
        inv_mass_c_arr = out_inv_mass_c.numpy()
        bias_lambda_arr = out_bias_lambda.numpy()
        bias_mu_arr = out_bias_mu.numpy()
        lambda_sum_lambda_arr = out_lambda_sum_lambda.numpy()
        lambda_sum_mu_arr = out_lambda_sum_mu.numpy()

        for cid in range(n):
            with self.subTest(cid=cid):
                # The kernel encodes ``cid`` into every value so we
                # can verify per-cid storage.
                self.assertEqual(type_arr[cid], int(CONSTRAINT_TYPE_CLOTH_TRIANGLE))
                self.assertEqual(body1_arr[cid], 100 + cid)
                self.assertEqual(body2_arr[cid], 200 + cid)
                self.assertEqual(body3_arr[cid], 300 + cid)
                np.testing.assert_allclose(
                    inv_rest_arr[cid],
                    [[1.0 + cid, 2.0 + cid], [3.0 + cid, 4.0 + cid]],
                    atol=1e-6,
                )
                self.assertAlmostEqual(rest_area_arr[cid], 0.5 + cid, places=6)
                self.assertAlmostEqual(alpha_lambda_arr[cid], 1e-3 * (cid + 1), places=8)
                self.assertAlmostEqual(alpha_mu_arr[cid], 2e-3 * (cid + 1), places=8)
                self.assertAlmostEqual(inv_mass_a_arr[cid], 0.1 + cid, places=6)
                self.assertAlmostEqual(inv_mass_b_arr[cid], 0.2 + cid, places=6)
                self.assertAlmostEqual(inv_mass_c_arr[cid], 0.3 + cid, places=6)
                self.assertAlmostEqual(bias_lambda_arr[cid], 10.0 + cid, places=5)
                self.assertAlmostEqual(bias_mu_arr[cid], 20.0 + cid, places=5)
                self.assertAlmostEqual(lambda_sum_lambda_arr[cid], -0.5 + cid, places=6)
                self.assertAlmostEqual(lambda_sum_mu_arr[cid], -1.5 + cid, places=6)

    def test_constraint_type_at_dword_zero(self) -> None:
        """``cloth_triangle_set_type`` writes
        :data:`CONSTRAINT_TYPE_CLOTH_TRIANGLE` at dword 0; reading
        offset 0 via the generic accessor must observe it."""
        n = 2
        out = wp.zeros(n, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_stamp_type_kernel,
            dim=n,
            inputs=[self.container],
            outputs=[out],
            device=self.device,
        )
        observed = out.numpy()
        for cid in range(n):
            self.assertEqual(observed[cid], int(CONSTRAINT_TYPE_CLOTH_TRIANGLE))


# ---------------------------------------------------------------------------
# Helper kernels (test-only).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _round_trip_kernel(
    c: ConstraintContainer,
    out_type: wp.array[wp.int32],
    out_body1: wp.array[wp.int32],
    out_body2: wp.array[wp.int32],
    out_body3: wp.array[wp.int32],
    out_inv_rest: wp.array[wp.mat22f],
    out_rest_area: wp.array[wp.float32],
    out_alpha_lambda: wp.array[wp.float32],
    out_alpha_mu: wp.array[wp.float32],
    out_inv_mass_a: wp.array[wp.float32],
    out_inv_mass_b: wp.array[wp.float32],
    out_inv_mass_c: wp.array[wp.float32],
    out_bias_lambda: wp.array[wp.float32],
    out_bias_mu: wp.array[wp.float32],
    out_lambda_sum_lambda: wp.array[wp.float32],
    out_lambda_sum_mu: wp.array[wp.float32],
):
    cid = wp.tid()

    # Distinct-per-cid encoding so an off-by-one in the cid->dword
    # mapping would surface immediately.
    cf = wp.float32(cid)

    cloth_triangle_set_type(c, cid)
    cloth_triangle_set_body1(c, cid, wp.int32(100) + cid)
    cloth_triangle_set_body2(c, cid, wp.int32(200) + cid)
    cloth_triangle_set_body3(c, cid, wp.int32(300) + cid)

    inv_rest = wp.mat22f(
        wp.float32(1.0) + cf,
        wp.float32(2.0) + cf,
        wp.float32(3.0) + cf,
        wp.float32(4.0) + cf,
    )
    cloth_triangle_set_inv_rest(c, cid, inv_rest)
    cloth_triangle_set_rest_area(c, cid, wp.float32(0.5) + cf)

    cloth_triangle_set_alpha_lambda(c, cid, wp.float32(1.0e-3) * (cf + wp.float32(1.0)))
    cloth_triangle_set_alpha_mu(c, cid, wp.float32(2.0e-3) * (cf + wp.float32(1.0)))

    cloth_triangle_set_inv_mass_a(c, cid, wp.float32(0.1) + cf)
    cloth_triangle_set_inv_mass_b(c, cid, wp.float32(0.2) + cf)
    cloth_triangle_set_inv_mass_c(c, cid, wp.float32(0.3) + cf)

    cloth_triangle_set_bias_lambda(c, cid, wp.float32(10.0) + cf)
    cloth_triangle_set_bias_mu(c, cid, wp.float32(20.0) + cf)

    cloth_triangle_set_lambda_sum_lambda(c, cid, wp.float32(-0.5) + cf)
    cloth_triangle_set_lambda_sum_mu(c, cid, wp.float32(-1.5) + cf)

    # Read everything back through the matching getters.
    out_type[cid] = read_int(c, wp.int32(0), cid)
    out_body1[cid] = cloth_triangle_get_body1(c, cid)
    out_body2[cid] = cloth_triangle_get_body2(c, cid)
    out_body3[cid] = cloth_triangle_get_body3(c, cid)
    out_inv_rest[cid] = cloth_triangle_get_inv_rest(c, cid)
    out_rest_area[cid] = cloth_triangle_get_rest_area(c, cid)
    out_alpha_lambda[cid] = cloth_triangle_get_alpha_lambda(c, cid)
    out_alpha_mu[cid] = cloth_triangle_get_alpha_mu(c, cid)
    out_inv_mass_a[cid] = cloth_triangle_get_inv_mass_a(c, cid)
    out_inv_mass_b[cid] = cloth_triangle_get_inv_mass_b(c, cid)
    out_inv_mass_c[cid] = cloth_triangle_get_inv_mass_c(c, cid)
    out_bias_lambda[cid] = cloth_triangle_get_bias_lambda(c, cid)
    out_bias_mu[cid] = cloth_triangle_get_bias_mu(c, cid)
    out_lambda_sum_lambda[cid] = cloth_triangle_get_lambda_sum_lambda(c, cid)
    out_lambda_sum_mu[cid] = cloth_triangle_get_lambda_sum_mu(c, cid)


@wp.kernel(enable_backward=False)
def _stamp_type_kernel(c: ConstraintContainer, out: wp.array[wp.int32]):
    cid = wp.tid()
    cloth_triangle_set_type(c, cid)
    out[cid] = read_int(c, wp.int32(0), cid)


if __name__ == "__main__":
    unittest.main()
