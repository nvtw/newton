# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise persistent GPU preparation through the real column container."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, MOTION_STATIC, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
    contact_set_count1,
    contact_set_count2,
    contact_set_friction,
    contact_set_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_normal,
    cc_set_start_gap,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_tgs import (
    ContactTGS,
    advance_contact_tgs_generation,
    allocate_contact_tgs,
    snapshot_contact_tgs,
)
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import NormalPatches, allocate, partition_range
from newton._src.solvers.phoenx.constraints.contact_tgs_prepare import geometry, partition_groups, patches
from newton._src.solvers.phoenx.simulation import PhoenXWorld


@wp.kernel
def set_ranges(columns: ContactColumnContainer, first: wp.array[int], count: wp.array[int]):
    cid = wp.tid()
    contact_set_contact_first(columns, cid, first[cid])
    contact_set_contact_count(columns, cid, count[cid])
    contact_set_friction(columns, cid, 0.5)
    contact_set_friction_dynamic(columns, cid, 0.3)


@wp.kernel
def reference_partition(first: wp.array[int], count: wp.array[int], normals: wp.array[wp.vec3f], state: NormalPatches):
    cid = wp.tid()
    if count[cid] > 0:
        partition_range(first[cid], first[cid], count[cid], normals, 0.999, state)


@wp.kernel
def set_geometry_fixture(columns: ContactColumnContainer, cc: ContactContainer):
    cid = wp.tid()
    contact_set_body1(columns, cid, 0)
    contact_set_body2(columns, cid, cid + 1)
    contact_set_count1(columns, cid, 2)
    contact_set_count2(columns, cid, 3)
    contact_set_contact_first(columns, cid, cid)
    contact_set_contact_count(columns, cid, 1)
    cc_set_normal(cc, cid, wp.vec3f(0.0, 0.0, 1.0))


@wp.kernel
def set_friction_fixture(
    columns: ContactColumnContainer, cc: ContactContainer, state: ContactTGS, mu_s: float, mu_d: float
):
    contact_set_body1(columns, 0, 0)
    contact_set_body2(columns, 0, 1)
    contact_set_contact_first(columns, 0, 0)
    contact_set_contact_count(columns, 0, 2)
    contact_set_friction(columns, 0, mu_s)
    contact_set_friction_dynamic(columns, 0, mu_d)
    for k in range(2):
        cc_set_normal(cc, k, wp.vec3f(0.0, 0.0, 1.0))
        state.normals[k] = wp.vec3f(0.0, 0.0, 1.0)


@wp.kernel
def set_sparse_history_fixture(
    columns: ContactColumnContainer, cc: ContactContainer, state: ContactTGS, first: wp.array[int]
):
    cid = wp.tid()
    point = first[cid]
    contact_set_body1(columns, cid, 0)
    contact_set_body2(columns, cid, 1)
    contact_set_contact_first(columns, cid, point)
    contact_set_contact_count(columns, cid, 1)
    contact_set_friction(columns, cid, 0.5)
    contact_set_friction_dynamic(columns, cid, 0.3)
    cc_set_normal(cc, point, wp.vec3f(0.0, 0.0, 1.0))
    state.normals[point] = wp.vec3f(0.0, 0.0, 1.0)


@wp.kernel
def set_empty_patch_fixture(columns: ContactColumnContainer, cc: ContactContainer, state: ContactTGS, flip: int):
    contact_set_body1(columns, 0, 0)
    contact_set_body2(columns, 0, 1)
    contact_set_contact_first(columns, 0, 0)
    contact_set_contact_count(columns, 0, 3)
    contact_set_friction(columns, 0, 0.5)
    contact_set_friction_dynamic(columns, 0, 0.3)
    for k in range(3):
        normal = wp.vec3f(0.0, 0.0, 1.0)
        if k == 1:
            normal = wp.vec3f(1.0, 0.0, 0.0)
        elif k == 2:
            normal = wp.vec3f(0.0, 0.0, -1.0)
        cc_set_normal(cc, k, normal)
        state.normals[k] = normal
        cc_set_start_gap(cc, k, wp.float32((k + flip) % 2))


class TestContactTGSPrepare(unittest.TestCase):
    def test_empty_friction_patches_keep_complete_history(self):
        """Skip empty solve work while retaining every patch and normal row."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                bodies = body_container_zeros(2, device)
                bodies.orientation.assign([[0, 0, 0, 1], [0, 0, 0, 1]])
                columns = contact_column_container_zeros(1, device)
                cc = contact_container_zeros(3, device)
                cc.impulses.fill_(0.25)
                original_impulses = cc.impulses.numpy()
                state = allocate_contact_tgs(3, 2, 24, device)
                active = wp.array([1], dtype=int, device=device)
                for flip in (0, 1):
                    snapshot_contact_tgs(state)
                    wp.launch(advance_contact_tgs_generation, 1, [state], device=device)
                    state.previous_anchors.broken.fill_(1)
                    wp.launch(set_empty_patch_fixture, 1, [columns, cc, state, flip], device=device)
                    wp.launch(patches, 1, [columns, state, active, bodies, cc], device=device)
                    np.testing.assert_array_equal(state.current.point_patch.numpy(), [0, 1, 2])
                    np.testing.assert_array_equal(state.current.patch_next.numpy(), [1, 2, -1])
                    self.assertEqual(state.current.group_count.numpy()[0], 3)
                    self.assertEqual(state.solve_first.numpy()[0], flip)
                    if flip == 0:
                        self.assertEqual(state.solve_next.numpy()[0], 2)
                        self.assertEqual(state.solve_next.numpy()[2], -1)
                    else:
                        self.assertEqual(state.solve_next.numpy()[1], -1)
                    np.testing.assert_array_equal(state.anchors.broken.numpy(), 0)
                    np.testing.assert_array_equal(cc.impulses.numpy(), original_impulses)

    def test_compact_history_tracks_live_groups(self):
        """Keep sparse history indices once per generation and remove stale groups."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                bodies = body_container_zeros(2, device)
                bodies.orientation.assign([[0, 0, 0, 1], [0, 0, 0, 1]])
                columns = contact_column_container_zeros(2, device)
                cc = contact_container_zeros(32, device)
                state = allocate_contact_tgs(32, 2, 24, device)
                active = wp.array([2], dtype=int, device=device)
                first = wp.array([3, 17], dtype=int, device=device)
                wp.launch(set_sparse_history_fixture, 2, [columns, cc, state, first], device=device)
                for _ in range(2):
                    wp.launch(patches, 2, [columns, state, active, bodies, cc], device=device)
                self.assertEqual(state.current_group_count.numpy()[0], 2)
                np.testing.assert_array_equal(np.sort(state.current_groups.numpy()[:2]), [3, 17])
                snapshot_contact_tgs(state)
                wp.launch(advance_contact_tgs_generation, 1, [state], device=device)
                self.assertEqual(state.previous_active.numpy()[0], 2)
                np.testing.assert_array_equal(np.sort(state.previous_groups.numpy()[:2]), [3, 17])
                # Match the lowest original slot even when compact order is reversed.
                reversed_groups = np.zeros(32, dtype=np.int32)
                reversed_groups[:2] = [17, 3]
                state.previous_groups.assign(reversed_groups)
                self.assertEqual(state.current_group_count.numpy()[0], 0)
                active.assign([1])
                first.assign([8, 23])
                wp.launch(set_sparse_history_fixture, 2, [columns, cc, state, first], device=device)
                wp.launch(patches, 2, [columns, state, active, bodies, cc], device=device)
                self.assertEqual(state.current_group_count.numpy()[0], 1)
                self.assertEqual(state.current_groups.numpy()[0], 8)
                self.assertEqual(state.anchors.source.numpy()[8], 3)
                snapshot_contact_tgs(state)
                self.assertEqual(state.previous_active.numpy()[0], 1)
                self.assertEqual(state.previous_groups.numpy()[0], 8)
                snapshot_contact_tgs(state)
                self.assertEqual(state.previous_active.numpy()[0], 0)

    def test_frictionless_material_transition(self):
        """Omit zero-friction patches and rebuild history when friction returns."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                bodies = body_container_zeros(2, device)
                bodies.orientation.assign([[0, 0, 0, 1], [0, 0, 0, 1]])
                columns = contact_column_container_zeros(1, device)
                cc = contact_container_zeros(2, device)
                state = allocate_contact_tgs(2, 2, 24, device)
                active = wp.array([1], dtype=int, device=device)
                for mu_s, mu_d in ((0.5, 0.3), (0.0, 0.0), (0.5, 0.3), (0.0, 0.3), (0.5, 0.0)):
                    snapshot_contact_tgs(state)
                    wp.launch(advance_contact_tgs_generation, 1, [state], device=device)
                    wp.launch(set_friction_fixture, 1, [columns, cc, state, mu_s, mu_d], device=device)
                    if device != "cpu":
                        wp.launch(partition_groups, (64, 128), [columns, state, active], block_dim=128, device=device)
                    wp.launch(patches, 1, [columns, state, active, bodies, cc], device=device)
                    frictionless = mu_s == 0.0 and mu_d == 0.0
                    self.assertEqual(state.current.group_count.numpy()[0], 0 if frictionless else 1)
                    self.assertEqual(state.current.group_first.numpy()[0], -1 if frictionless else 0)
                    if not frictionless:
                        self.assertGreater(state.anchors.count.numpy()[0], 0)
                        self.assertEqual(state.anchors.source.numpy()[0], -1)
                    np.testing.assert_array_equal(cc.impulses.numpy(), 0)

    def test_removed_contacts_clear_temporal_support(self):
        """Clear removed support rows while advancing and preserving prior patch history."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                state = allocate_contact_tgs(1, 2, 3, device)
                state.keys.assign([[1, 2, 3, 4]])
                state.anchors.count.assign([1])
                state.impulse.fill_(wp.vec3f(1.0, 2.0, 3.0))
                cc = contact_container_zeros(1, device)
                cc.impulses.fill_(2.0)
                active = wp.array([1], dtype=int, device=device)
                world = SimpleNamespace(
                    _temporal_contact_state=state,
                    max_contact_columns=1,
                    _ingest_scratch=SimpleNamespace(num_contact_columns=active),
                    _num_active_constraints=wp.array([6], dtype=int, device=device),
                    _contact_offset=5,
                    _contact_container=cc,
                    device=device,
                )
                PhoenXWorld._ingest_and_warmstart_contacts(world, None, None)
                np.testing.assert_array_equal(state.generation.numpy(), [1])
                np.testing.assert_array_equal(state.previous_keys.numpy(), [[1, 2, 3, 4]])
                np.testing.assert_array_equal(state.previous_anchors.count.numpy(), [1])
                np.testing.assert_array_equal(state.keys.numpy(), -1)
                np.testing.assert_array_equal(state.impulse.numpy(), 0)
                np.testing.assert_array_equal(cc.impulses.numpy(), 0)
                np.testing.assert_array_equal(active.numpy(), 0)
                np.testing.assert_array_equal(world._num_active_constraints.numpy(), [5])
                self.assertIsNone(world._contact_views)

    def test_geometry_retains_impulses_and_physical_static_response(self):
        """Rebase common-point geometry without erasing or reapplying temporal impulses."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                bodies = body_container_zeros(3, device)
                bodies.orientation.assign(np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (3, 1)))
                bodies.inverse_mass.assign([2.0, 1.0, 0.0])
                positions = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)
                bodies.position.assign(positions)
                bodies.inverse_inertia_world.assign(
                    [[0.5, 0.5, 0.5, 0, 0, 0], [0.25, 0.25, 0.25, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
                )
                bodies.motion_type.assign([MOTION_DYNAMIC, MOTION_DYNAMIC, MOTION_STATIC])
                columns = contact_column_container_zeros(2, device)
                cc = contact_container_zeros(2, device)
                impulses = np.arange(cc.impulses.size, dtype=np.float32).reshape(cc.impulses.shape) + 1
                cc.impulses.assign(impulses)
                views = ContactViews()
                views.rigid_contact_point0 = wp.array([[1, 0, 0], [1, 0, 0]], dtype=wp.vec3f, device=device)
                views.rigid_contact_point1 = wp.array(
                    [[1, 0, -1 + 1 / 128], [1, 0, 1 - 1 / 1024]], dtype=wp.vec3f, device=device
                )
                views.rigid_contact_margin0 = wp.zeros(2, dtype=float, device=device)
                views.rigid_contact_margin1 = wp.zeros(2, dtype=float, device=device)
                state = allocate_contact_tgs(2, 3, 30, device)
                active = wp.array([2], dtype=int, device=device)
                wp.launch(set_geometry_fixture, 2, [columns, cc], device=device)
                before_v = bodies.velocity.numpy()
                before_w = bodies.angular_velocity.numpy()
                for _substep in range(2):
                    wp.launch(geometry, (2, 128), [columns, state, active, bodies, cc, views, 100.0], device=device)
                    rows = state.normal_rows.numpy()
                    np.testing.assert_allclose(rows["effective_mass"], [1.0 / 8.75, 0.4], rtol=1e-6)
                    np.testing.assert_allclose(rows["bias"], [100.0 / 128, -100.0 / 1024 * state.gain], rtol=1e-6)
                    np.testing.assert_allclose(rows["r0"], [[1, 0, 1 / 256], [1, 0, -1 / 2048]], atol=1e-8)
                    np.testing.assert_array_equal(rows["r0"] + positions[0], rows["r1"] + positions[1:])
                    np.testing.assert_array_equal(cc.impulses.numpy(), impulses)
                    np.testing.assert_array_equal(bodies.velocity.numpy(), before_v)
                    np.testing.assert_array_equal(bodies.angular_velocity.numpy(), before_w)

    def test_persistent_partition_generation(self):
        """Retain partitions within a generation and rebuild every active group afterward."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = "cuda:0"
        # More groups than persistent blocks, plus both sides of the shared-cache limit.
        counts = np.resize(np.array([1, 7, 0, 90], dtype=np.int32), 130)
        counts[63:66] = [1024, 1025, 513]
        first = np.cumsum(np.r_[0, counts[:-1]]).astype(np.int32)
        capacity = int(counts.sum())
        rng = np.random.default_rng(198)
        normals = rng.normal(size=(capacity, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals[::3] = [0, 0, 1]
        state = allocate_contact_tgs(capacity, 2, 30, device)
        state.normals.assign(normals)
        columns = contact_column_container_zeros(len(counts), device)
        first_gpu = wp.array(first, dtype=int, device=device)
        count_gpu = wp.array(counts, dtype=int, device=device)
        active = wp.array([len(counts)], dtype=int, device=device)
        reference = allocate(capacity, capacity, device)
        wp.launch(set_ranges, len(counts), [columns, first_gpu, count_gpu], device=device)
        fields = (
            "point_patch",
            "point_next",
            "patch_first",
            "patch_last",
            "patch_count",
            "patch_next",
            "patch_normal",
            "group_first",
            "group_count",
        )
        for _generation in range(2):
            wp.launch(advance_contact_tgs_generation, 1, [state], device=device)
            wp.launch(partition_groups, (64, 128), [columns, state, active], block_dim=128, device=device)
            wp.launch(reference_partition, len(counts), [first_gpu, count_gpu, state.normals, reference], device=device)
            for field in fields:
                np.testing.assert_array_equal(
                    getattr(state.current, field).numpy(), getattr(reference, field).numpy(), err_msg=field
                )
            np.testing.assert_array_equal(state.partition_last.numpy()[first[counts > 0]], state.generation.numpy()[0])
            saved = {field: getattr(state.current, field).numpy() for field in fields}
            # Changing input geometry within this generation must not re-partition history.
            state.normals.assign(np.tile(np.array([0, 1, 0], dtype=np.float32), (capacity, 1)))
            wp.launch(partition_groups, (64, 128), [columns, state, active], block_dim=128, device=device)
            for field in fields:
                np.testing.assert_array_equal(getattr(state.current, field).numpy(), saved[field], err_msg=field)


if __name__ == "__main__":
    unittest.main()
