# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare canonical gathered warm starts with the colored contact update."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros, inertia_sym6
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_owned_elsewhere,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_solve import contact_cached_warmstart_lean
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_normal,
    cc_set_normal_lambda,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
    contact_container_zeros,
    contact_tile_rows_numpy,
)
from newton._src.solvers.phoenx.constraints.contact_warm_start import (
    ContactBodyEntries,
    _refresh_cached_warm_start_kernel,
    store_row_warm_start,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.kernel
def _initialize_columns(columns: ContactColumnContainer, cc: ContactContainer, rows: int):
    cid = wp.tid()
    contact_set_body1(columns, cid, cid % 4)
    contact_set_body2(columns, cid, 1 + (cid + 1) % 4)
    contact_set_contact_first(columns, cid, rows * cid)
    contact_set_contact_count(columns, cid, rows)
    if cid == 4:
        columns.articulation_owner[cid] = 0
    if cid == 5:
        columns.static_owner[cid] = 1
    for j in range(rows):
        k = rows * cid + j
        cc_set_normal(cc, k, wp.vec3(0.0, 0.0, 1.0))
        cc_set_tangent1(cc, k, wp.vec3(1.0, 0.0, 0.0))
        cc_set_r0(cc, k, wp.vec3(0.1 * wp.float32(j + 1), -0.2, 0.3))
        cc_set_r1(cc, k, wp.vec3(-0.3, 0.2 * wp.float32(j + 1), -0.1))
        cc_set_normal_lambda(cc, k, 0.3 * wp.float32(k + 1))
        cc_set_tangent1_lambda(cc, k, -0.1 * wp.float32(k + 1))
        cc_set_tangent2_lambda(cc, k, 0.07 * wp.float32(k + 1))


@wp.kernel
def _refresh_reference(
    cc: ContactContainer,
    active: int,
    rows: int,
    linear: wp.array[wp.vec3],
    angular0: wp.array[wp.vec3],
    angular1: wp.array[wp.vec3],
):
    # The fixture's first three slots contain columns 2, a joint, and 0.
    # Its next five add the excluded owners, columns 1 and 3, and a joint.
    for cid in range(4):
        if active == 8 or (active == 3 and (cid == 0 or cid == 2)):
            for k in range(rows * cid, rows * (cid + 1)):
                store_row_warm_start(cc, k, linear, angular0, angular1)


@wp.kernel
def _colored_reference(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    ids: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    for slot in range(active[0]):
        cid = ids[slot] - 2
        if cid >= 0:
            if not contact_owned_elsewhere(columns, cid):
                contact_cached_warmstart_lean(
                    columns, cid, bodies, ParticleContainer(), 5, 120.0, cc, ContactViews(), CopyStateContainer(), 0
                )


class TestContactWarmStart(unittest.TestCase):
    def test_friction_anchor_preserves_local_witness_precision(self):
        """A common world translation leaves both tangential friction references unchanged."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            offset = (0.0234567, 0.0156789, 0.0)
            for z in (1.0, 1.099):
                body = builder.add_body(
                    xform=wp.transform((0, 0, z), wp.quat_identity()),
                    mass=1,
                    inertia=wp.mat33(np.eye(3)),
                    lock_inertia=True,
                )
                builder.add_shape_sphere(body, xform=wp.transform(offset, wp.quat_identity()), radius=0.05)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=8)
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                substeps=1,
                solver_iterations=1,
            )
            state, contacts = model.state(), pipeline.contacts()
            pipeline.collide(state, contacts)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
            poses = state.body_q.numpy()
            poses[:, :2] += (73.0, 43.0)
            state.body_q.assign(poses)
            solver.step(state, state, model.control(), contacts, 1 / 120)
            anchors = contact_tile_rows_numpy(solver.world._contact_container, "lambdas")[6:12, 0].reshape(2, 3)
            for anchor in anchors:
                np.testing.assert_allclose(anchor[:2], offset[:2], atol=1e-8, rtol=0)

    @staticmethod
    def _fixture(rows=2):
        columns = contact_column_container_zeros(7)
        cc = contact_container_zeros(7 * rows)
        wp.launch(_initialize_columns, 7, inputs=[columns, cc, rows])
        # Joint slots, shared bodies, two excluded owners, and one inactive column.
        ids = wp.array([4, 0, 2, 7, 3, 1, 6, 5, 8], dtype=wp.int32)
        active = wp.array([8], dtype=wp.int32)
        entries = ContactBodyEntries(9, 7 * rows, 5, wp.get_device())
        entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
        return columns, cc, ids, active, entries

    def test_refresh_rows_match_serial_reference(self):
        """Refresh each live row exactly once while preserving excluded rows."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with wp.ScopedDevice(device):
                for rows in (1, 2, 19):
                    columns, cc, ids, active, _ = self._fixture(rows)
                    for count in (8, 3, 0):
                        with self.subTest(device=device, rows=rows, active=count):
                            active.assign(np.array([count], dtype=np.int32))
                            actual = [wp.full(7 * rows, wp.vec3(-123.0), dtype=wp.vec3) for _ in range(3)]
                            expected = [wp.clone(array) for array in actual]
                            wp.launch(_refresh_reference, 1, inputs=[cc, count, rows, *expected])
                            # Two column lanes force multiple grid-stride passes;
                            # 19 rows exercise more than one eight-lane row pass.
                            lanes = 8 if wp.get_device().is_cuda else 1
                            wp.launch(
                                _refresh_cached_warm_start_kernel,
                                (2, lanes),
                                inputs=[columns, cc, ids, active, 2, *actual, 2, lanes],
                            )
                            for result, reference in zip(actual, expected, strict=True):
                                np.testing.assert_array_equal(result.numpy(), reference.numpy())

    def test_canonical_entries_preserve_order_and_reset(self):
        """Exclude owned and inactive columns while retaining endpoint order, also in a captured graph."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                columns, _, ids, active, entries = self._fixture()
                if wp.get_device().is_cuda:
                    with wp.ScopedCapture() as capture:
                        entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
                    entries.starts.zero_()
                    wp.capture_launch(capture.graph)
                starts, values = entries.starts.numpy(), entries.values.numpy()
                expected = ([0], [2, 7], [4, 1], [3, 6], [5])
                for body, codes in enumerate(expected):
                    np.testing.assert_array_equal(values[starts[body] : starts[body + 1]], codes)
                active.zero_()
                entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
                np.testing.assert_array_equal(entries.starts.numpy(), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA subgroup gather required")
    def test_cached_gather_matches_colored_impulses_cuda(self):
        """Replay normal and friction impulses without changing their values."""
        with wp.ScopedDevice("cuda:0"):
            columns, cc, ids, active, entries = self._fixture()
            bodies = body_container_zeros(5)
            bodies.inverse_mass.assign(np.array([0, 0.2, 3.0, 0.5, 0], dtype=np.float32))
            rng = np.random.default_rng(42)
            original_impulses = cc.impulses.numpy()
            for trial in range(4):
                inertia = rng.uniform(0.2, 3.0, (5, 6)).astype(np.float32)
                # Symmetric storage is xx, xy, xz, yy, yz, zz. Keep the
                # off-diagonal entries small enough for positive definiteness.
                inertia[:, [1, 2, 4]] *= 0.01
                inertia[[0, 4]] = 0
                wp.copy(bodies.inverse_inertia_world, wp.array(inertia, dtype=inertia_sym6))
                velocity = rng.normal(size=bodies.velocity.numpy().shape).astype(np.float32)
                angular = rng.normal(size=bodies.angular_velocity.numpy().shape).astype(np.float32)
                cc.impulses.assign(original_impulses * (trial + 1))
                before = cc.impulses.numpy()
                bodies.velocity.assign(velocity)
                bodies.angular_velocity.assign(angular)
                wp.launch(_colored_reference, 1, inputs=[columns, cc, bodies, ids, active])
                expected_v, expected_w = bodies.velocity.numpy(), bodies.angular_velocity.numpy()
                bodies.velocity.assign(velocity)
                bodies.angular_velocity.assign(angular)
                entries.apply_cached_warm_start(columns, bodies, cc, ids, active, 2)
                np.testing.assert_allclose(bodies.velocity.numpy(), expected_v, atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(bodies.angular_velocity.numpy(), expected_w, atol=2e-6, rtol=2e-6)
                np.testing.assert_array_equal(cc.impulses.numpy(), before)
            active.zero_()
            entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
            before_v, before_w = bodies.velocity.numpy(), bodies.angular_velocity.numpy()
            entries.apply_cached_warm_start(columns, bodies, cc, ids, active, 2)
            np.testing.assert_array_equal(bodies.velocity.numpy(), before_v)
            np.testing.assert_array_equal(bodies.angular_velocity.numpy(), before_w)


if __name__ == "__main__":
    unittest.main()
