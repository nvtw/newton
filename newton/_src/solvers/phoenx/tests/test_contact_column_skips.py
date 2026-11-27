# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check which columns and rows the moving-surface contact sweep skips.

The sweep tests the column owner, sleeping endpoints and the four-row
separation bounds with its own loads (see ``_make_contact_iterate_lean``);
these tests pin down which rows it solves.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    _OFF_BODY1,
    _OFF_BODY2,
    _OFF_CONTACT_COUNT,
    _OFF_CONTACT_FIRST,
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_solve import _make_contact_iterate_lean
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal_lambda,
    cc_set_bias,
    cc_set_eff_n,
    cc_set_normal,
    cc_set_tangent1,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer, copy_state_container_zeros
from newton._src.solvers.phoenx.particle import ParticleContainer, particle_container_zeros

_iterate = _make_contact_iterate_lean(
    has_sleeping=True,
    has_soft_contact_pd=False,
    tangent_mobility_layout=False,
    moving_surface_only=True,
    check_owner=True,
)


@wp.kernel
def _penetrating_rows(cc: ContactContainer):
    k = wp.tid()
    cc_set_normal(cc, k, wp.vec3f(0.0, 0.0, 1.0))
    cc_set_tangent1(cc, k, wp.vec3f(1.0, 0.0, 0.0))
    cc_set_eff_n(cc, k, wp.float32(1.0))
    cc_set_bias(cc, k, wp.float32(-0.1))


@wp.kernel
def _sweep(
    columns: ContactColumnContainer,
    num_columns: int,
    bodies: BodyContainer,
    particles: ParticleContainer,
    cc: ContactContainer,
    copy_state: CopyStateContainer,
    lambdas: wp.array[wp.float32],
):
    # One thread walks the columns in order; they share the static body.
    for cid in range(num_columns):
        _iterate(
            columns,
            cid,
            bodies,
            particles,
            bodies.inverse_mass.shape[0],
            wp.float32(360.0),
            cc,
            ContactViews(),
            True,
            copy_state,
            0,
            wp.float32(1.0),
        )
    for k in range(lambdas.shape[0]):
        lambdas[k] = cc_get_normal_lambda(cc, k)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX sweeps run captured on CUDA")
class TestContactColumnSkips(unittest.TestCase):
    def test_owned_frozen_and_separated_rows_are_skipped(self):
        """Solve exactly the rows of awake, unowned columns in groups that can touch."""
        with wp.ScopedDevice("cuda:0"):
            # Body 0 is static; bodies 1-4 fall onto it at 1 m/s, body 3 sleeps.
            bodies = body_container_zeros(5)
            motion = np.full(5, int(MOTION_DYNAMIC), dtype=np.int32)
            motion[0] = 0
            bodies.motion_type.assign(motion)
            bodies.island_root.assign(np.array([-1, -1, -1, 3, -1], dtype=np.int32))
            # Heavy bodies keep every solved row loaded in the Gauss-Seidel order.
            bodies.inverse_mass.assign(np.array([0, 1e-3, 1e-3, 1e-3, 1e-3], dtype=np.float32))
            velocity = np.zeros((5, 3), dtype=np.float32)
            velocity[1:, 2] = -1.0
            bodies.velocity.assign(velocity)

            # Column 0 holds 13 rows in four groups: 0-3 and 8-11 touch,
            # 4-7 and 12 are separated. Column 1 is articulation-owned,
            # column 2 joins the static body to the sleeping one, column 3
            # joins the sleeping body to an awake one, column 4 is
            # static-owned.
            first = np.array([0, 13, 14, 15, 16], dtype=np.int32)
            count = np.array([13, 1, 1, 1, 1], dtype=np.int32)
            body1 = np.array([0, 0, 0, 3, 0], dtype=np.int32)
            body2 = np.array([1, 2, 3, 4, 2], dtype=np.int32)
            columns = contact_column_container_zeros(5)
            data = columns.data.numpy()
            headers = data.view(np.int32)
            headers[int(_OFF_BODY1)] = body1
            headers[int(_OFF_BODY2)] = body2
            headers[int(_OFF_CONTACT_FIRST)] = first
            headers[int(_OFF_CONTACT_COUNT)] = count
            columns.data.assign(data)
            columns.articulation_owner.assign(np.array([-1, 0, -1, -1, -1], dtype=np.int32))
            columns.static_owner.assign(np.array([-1, -1, -1, -1, 0], dtype=np.int32))

            rows = 17
            cc = contact_container_zeros(rows, surface_velocity=True)
            bounds = np.zeros((rows, 4), dtype=np.float32)
            bounds[[0, 8, 13, 14, 15, 16], 0] = -1.0
            bounds[[4, 12], 0] = 10.0
            cc.separation_bounds = wp.array(bounds, dtype=wp.vec4f)
            wp.launch(_penetrating_rows, rows, inputs=[cc])

            particles = particle_container_zeros(1)
            copy_state = copy_state_container_zeros(1, 5)
            lambdas = wp.zeros(rows, dtype=wp.float32)
            inputs = [columns, 5, bodies, particles, cc, copy_state, lambdas]
            with wp.ScopedCapture() as capture:
                wp.launch(_sweep, 1, inputs=inputs)
            wp.capture_launch(capture.graph)

            solved = np.flatnonzero(lambdas.numpy() > 0.0)
            np.testing.assert_array_equal(solved, [0, 1, 2, 3, 8, 9, 10, 11, 15])
            v = bodies.velocity.numpy()[:, 2]
            # Solved columns push their bodies; skipped ones leave them untouched.
            self.assertGreater(v[1], -1.0)
            self.assertGreater(v[4], -1.0)
            self.assertEqual(v[2], -1.0)
            self.assertEqual(v[0], 0.0)


if __name__ == "__main__":
    unittest.main()
