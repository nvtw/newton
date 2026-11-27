# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the four-row separation bound for temporal rigid contacts."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    _OFF_BODY2,
    _OFF_CONTACT_COUNT,
    _OFF_CONTACT_FIRST,
    ContactViews,
    contact_column_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_solve import _contact_group_is_separated
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    contact_container_zeros,
    contact_tile_rows_numpy,
    pack_contact_tiles,
)
from newton._src.solvers.phoenx.constraints.contact_warm_start import store_row_warm_start
from newton._src.solvers.phoenx.temporal_contacts import (
    TEMPORAL_SURFACE_GAP_LANES,
    temporal_contact_gap_kernel,
    temporal_surface_gap_kernel,
)


@wp.kernel
def _check_bound(
    bounds: wp.array[wp.vec4f], linear: wp.array[wp.vec3f], angular: wp.array[wp.vec3f], result: wp.array[int]
):
    i = wp.tid()
    result[i] = wp.int32(_contact_group_is_separated(bounds[i], linear[i], linear[i], wp.vec3f(0.0), angular[i]))


@wp.kernel
def _store_rows(
    cc: ContactContainer, linear: wp.array[wp.vec3f], angular0: wp.array[wp.vec3f], angular1: wp.array[wp.vec3f]
):
    store_row_warm_start(cc, wp.tid(), linear, angular0, angular1)


class TestTemporalContactGroups(unittest.TestCase):
    def test_shared_translation_preserves_active_normal(self):
        """Keep a row whose rounded point velocity closes the gap despite equal body translations."""
        velocity = np.float32(2049.2001953125)
        spin = np.float32(0.00013265592860989273)
        bias = np.float32(0.00023814711312297732)
        self.assertLess((velocity - spin) - velocity + bias, 0.0)
        with wp.ScopedDevice("cpu"):
            # The first contact closes after rounding. At rest the identical bound permits the skip.
            bound = bias * np.float32(0.99998) - np.float32(1.0e-4)
            bounds = wp.array([[bound, 0.0, 1.0, 0.0]] * 2, dtype=wp.vec4f)
            linear = wp.array([[velocity, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3f)
            angular = wp.array([[0.0, 0.0, spin]] * 2, dtype=wp.vec3f)
            result = wp.zeros(2, dtype=int)
            wp.launch(_check_bound, 2, inputs=[bounds, linear, angular, result])
            np.testing.assert_array_equal(result.numpy(), [0, 1])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA lane reductions required")
    def test_ragged_groups_preserve_gap_and_exclude_loaded_rows(self):
        """Refresh biases identically and reject groups with a tangent impulse or a penetrating row."""
        with wp.ScopedDevice("cuda:0"):
            columns = contact_column_container_zeros(3)
            data = columns.data.numpy()
            headers = data.view(np.int32)
            headers[int(_OFF_BODY2)] = 1
            headers[int(_OFF_CONTACT_FIRST)] = [0, 37, 40]
            headers[int(_OFF_CONTACT_COUNT)] = [37, 3, 1]
            columns.data.assign(data)
            columns.articulation_owner.assign(np.array([-1, -1, 0], dtype=np.int32))
            bodies = body_container_zeros(2)
            bodies.velocity.assign(np.array([[0, 0, 0], [0, 0, 0.04]], dtype=np.float32))
            cc = contact_container_zeros(41, track_gap=True, surface_velocity=True)
            basis = contact_tile_rows_numpy(cc, "lambdas")
            basis[2] = 1.0
            cc.lambdas.assign(pack_contact_tiles(basis))
            impulses = cc.impulses.numpy()
            impulses[2, 1] = 0.1
            cc.impulses.assign(impulses)
            gaps = np.full(41, 0.001, dtype=np.float32)
            gaps[36] = -0.002
            count = wp.array([3], dtype=int)
            inputs = [columns, cc, bodies, count, columns.articulation_owner]
            cc.gap.assign(gaps)
            empty = wp.empty(0, dtype=wp.vec3f)
            gap_inputs = [*inputs, 8, 0.001, 1000.0, 5.0, 2, ContactViews(), None, 0.002, 1, 2]
            wp.launch(temporal_contact_gap_kernel, (2, 8), inputs=[*gap_inputs, empty, empty, empty])
            reference_gaps, reference_biases = cc.gap.numpy(), contact_tile_rows_numpy(cc, "derived")[3].copy()
            # The same pass can store the rows' warm-start impulses; the gap update keeps them.
            columns.static_owner.assign(np.array([-1, 0, -1], dtype=np.int32))
            stored = [wp.full(41, wp.vec3(-7.0), dtype=wp.vec3f) for _ in range(3)]
            expected = [wp.full(41, wp.vec3(-7.0), dtype=wp.vec3f) for _ in range(3)]
            wp.launch(_store_rows, 37, inputs=[cc, *expected])
            cc.gap.assign(gaps)
            wp.launch(temporal_contact_gap_kernel, (2, 8), inputs=[*gap_inputs, *stored])
            np.testing.assert_array_equal(cc.gap.numpy(), reference_gaps)
            for result, reference in zip(stored, expected, strict=True):
                # Column 0 is refreshed; column 1 is static-owned and column 2 articulation-owned.
                np.testing.assert_array_equal(result.numpy(), reference.numpy())
            cc.gap.assign(gaps)
            cc.derived.zero_()
            cc.separation_bounds = wp.zeros(41, dtype=wp.vec4f)
            wp.launch(
                temporal_surface_gap_kernel, (2, TEMPORAL_SURFACE_GAP_LANES), inputs=[*inputs, 0.001, 1000.0, 5.0, 2]
            )
            np.testing.assert_array_equal(cc.gap.numpy(), reference_gaps)
            np.testing.assert_array_equal(contact_tile_rows_numpy(cc, "derived")[3], reference_biases)
            bounds = cc.separation_bounds.numpy()[:, 0]
            np.testing.assert_array_equal(np.flatnonzero(bounds > 0), [4, 8, 12, 16, 20, 24, 28, 32, 37])
            self.assertLess(bounds[0], 0.0)
            self.assertLess(bounds[36], 0.0)
            self.assertEqual(bounds[40], 0.0)
