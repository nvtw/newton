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
def _set_biases(cc: ContactContainer, biases: wp.array[wp.float32]):
    k = wp.tid()
    cc_set_bias(cc, k, biases[k])


@wp.kernel
def _sweep(
    columns: ContactColumnContainer,
    num_columns: int,
    bodies: BodyContainer,
    particles: ParticleContainer,
    cc: ContactContainer,
    copy_state: CopyStateContainer,
    use_bias: bool,
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
            use_bias,
            copy_state,
            0,
            wp.float32(1.0),
        )
    for k in range(lambdas.shape[0]):
        lambdas[k] = cc_get_normal_lambda(cc, k)


def _sweep_columns(columns, bodies, cc, use_bias):
    """Run one captured sweep over every column; return the rows' normal impulses."""
    lambdas = wp.zeros(cc.surface_target.shape[0], dtype=wp.float32)
    inputs = [
        columns,
        columns.articulation_owner.shape[0],
        bodies,
        particle_container_zeros(1),
        cc,
        copy_state_container_zeros(1, bodies.inverse_mass.shape[0]),
        use_bias,
        lambdas,
    ]
    with wp.ScopedCapture() as capture:
        wp.launch(_sweep, 1, inputs=inputs)
    wp.capture_launch(capture.graph)
    return lambdas.numpy()


def _scene(first, body1, body2, rows, articulation_owner=None, static_owner=None):
    """Columns of consecutive rows and five bodies: static body 0, bodies 1-4 falling at 1 m/s, body 3 asleep."""
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

    num_columns = len(first)
    columns = contact_column_container_zeros(num_columns)
    data = columns.data.numpy()
    headers = data.view(np.int32)
    headers[int(_OFF_BODY1)] = body1
    headers[int(_OFF_BODY2)] = body2
    headers[int(_OFF_CONTACT_FIRST)] = first
    headers[int(_OFF_CONTACT_COUNT)] = np.diff(np.append(first, rows))
    columns.data.assign(data)
    if articulation_owner is not None:
        columns.articulation_owner.assign(np.array(articulation_owner, dtype=np.int32))
    if static_owner is not None:
        columns.static_owner.assign(np.array(static_owner, dtype=np.int32))

    cc = contact_container_zeros(rows, surface_velocity=True)
    cc.separation_bounds = wp.zeros(rows, dtype=wp.vec4f)
    wp.launch(_penetrating_rows, rows, inputs=[cc])
    return columns, bodies, cc


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX sweeps run captured on CUDA")
class TestContactColumnSkips(unittest.TestCase):
    def test_owned_frozen_and_separated_rows_are_skipped(self):
        """Solve exactly the rows of awake, unowned columns in groups that can touch."""
        with wp.ScopedDevice("cuda:0"):
            # Column 0 holds 13 rows in four groups: 0-3 and 8-11 touch,
            # 4-7 and 12 are separated. Column 1 is articulation-owned,
            # column 2 joins the static body to the sleeping one, column 3
            # joins the sleeping body to an awake one, column 4 is
            # static-owned.
            columns, bodies, cc = _scene(
                first=[0, 13, 14, 15, 16],
                body1=[0, 0, 0, 3, 0],
                body2=[1, 2, 3, 4, 2],
                rows=17,
                articulation_owner=[-1, 0, -1, -1, -1],
                static_owner=[-1, -1, -1, -1, 0],
            )
            bounds = np.zeros((17, 4), dtype=np.float32)
            bounds[[0, 8, 13, 14, 15, 16], 0] = -1.0
            bounds[[4, 12], 0] = 10.0
            cc.separation_bounds.assign(bounds)

            lambdas = _sweep_columns(columns, bodies, cc, True)
            np.testing.assert_array_equal(np.flatnonzero(lambdas > 0.0), [0, 1, 2, 3, 8, 9, 10, 11, 15])
            v = bodies.velocity.numpy()[:, 2]
            # Solved columns push their bodies; skipped ones leave them untouched.
            self.assertGreater(v[1], -1.0)
            self.assertGreater(v[4], -1.0)
            self.assertEqual(v[2], -1.0)
            self.assertEqual(v[0], 0.0)

    def test_relaxation_skips_separated_rows(self):
        """Relax every row of a long column except those with a separating bias, before and past row 8."""
        with wp.ScopedDevice("cuda:0"):
            columns, bodies, cc = _scene(first=[0, 11], body1=[0, 0], body2=[1, 2], rows=12)
            separated = [2, 5, 8, 10, 11]
            biases = np.full(12, -0.1, dtype=np.float32)
            biases[separated] = 0.1
            wp.launch(_set_biases, 12, inputs=[cc, wp.array(biases, dtype=wp.float32)])

            lambdas = _sweep_columns(columns, bodies, cc, False)
            np.testing.assert_array_equal(np.flatnonzero(lambdas > 0.0), [0, 1, 3, 4, 6, 7, 9])
            # The column of one separated row leaves its body untouched.
            self.assertEqual(bodies.velocity.numpy()[2, 2], -1.0)


if __name__ == "__main__":
    unittest.main()
