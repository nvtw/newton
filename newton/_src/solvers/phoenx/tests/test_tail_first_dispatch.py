# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check tail-first control flow using real scheduler kernels and cloth dynamics."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx import solver_phoenx
from newton._src.solvers.phoenx import solver_phoenx_kernels as kernels
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.tests import test_cloth_mass_splitting as fixture


@wp.kernel(enable_backward=False)
def _count_dispatch(kind: int, counts: wp.array[int], cursor: wp.array[int], head_active: wp.array[int]):
    counts[kind] += 1
    # Turn an outer-loop deadlock into a bounded assertion failure.
    if counts[0] + counts[1] > 32:
        cursor[0] = 0
        head_active[0] = 0


@wp.func
def _trace_dispatch(
    constraints: kernels.ConstraintContainer,
    contact_cols: kernels.ContactColumnContainer,
    bodies: kernels.BodyContainer,
    particles: kernels.ParticleContainer,
    cc: kernels.ContactContainer,
    contacts: kernels.ContactViews,
    copy_state: kernels.CopyStateContainer,
    num_joints: wp.int32,
    joint_pgs_enabled: wp.array[wp.int32],
    num_cloth_triangles: wp.int32,
    num_cloth_bending: wp.int32,
    num_soft_tetrahedra: wp.int32,
    num_soft_hexahedra: wp.int32,
    num_bodies: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    cid: wp.int32,
    colored_slot: wp.int32,
    parallel_id: wp.int32,
):
    # The predecessor lies in an earlier color or the same serial overflow
    # batch. This makes missing/reordered work observable without floating-point
    # ambiguity or dependence on order inside an independent color.
    slot = cid + 1
    predecessor = wp.int32(bodies.velocity[slot][0])
    bodies.position[slot] = wp.vec3(
        bodies.position[predecessor][0] + 1.0, wp.float32(parallel_id), wp.float32(colored_slot)
    )
    bodies.inverse_mass[slot] += 1.0


def _legacy_head_first_sweep(self, head_kernel, tail_kernel, idt, contact_container=None):
    """Retain the previous dispatch order as an independent test reference."""
    assert not self.parallel_contact_prepare

    def round_sweep():
        wp.capture_while(
            self._head_active,
            self._capture_singleworld_sweep,
            kernel=head_kernel,
            idt=idt,
            contact_container=contact_container,
        )
        self._capture_singleworld_tail_sweep(kernel=tail_kernel, idt=idt, contact_container=contact_container)

    wp.capture_while(self._partitioner.color_cursor, round_sweep)


@contextmanager
def _candidate_sweep(enabled):
    # Both paths use the same canonical kernels. The tail's completion flag
    # differs from the old code only when the outer loop is already finished.
    if enabled:
        yield
    else:
        with patch.object(solver_phoenx.PhoenXWorld, "_singleworld_head_plus_tail_sweep", _legacy_head_first_sweep):
            yield


def _run_schedule(sizes, reverse, overflow, batch, candidate):
    with (
        patch.object(kernels, "_make_singleworld_dispatch_func", return_value=(_trace_dispatch, None)),
        _candidate_sweep(candidate),
    ):
        flags = {
            "phase": "iterate",
            "cloth_support": False,
            "has_joints": False,
            "has_contacts": False,
            "has_mass_splitting": True,
            "has_sleeping": False,
            "has_soft_contact_pd": False,
        }
        head = kernels._make_singleworld_persistent_kernel(**flags)
        tail = kernels._make_singleworld_fused_kernel(**flags)
        count = sum(sizes)
        bodies = body_container_zeros(count + 1, device="cuda:0")
        starts = np.r_[0, np.cumsum(sizes)].astype(np.int32)
        regular = len(sizes) if overflow < 0 else overflow
        order = list(range(regular))
        if reverse:
            order.reverse()
        if overflow >= 0:
            order.append(overflow)
        predecessor = np.zeros((count + 1, 3), dtype=np.float32)
        expected = np.zeros_like(predecessor)
        prior = 0
        for color in order:
            for offset, cid in enumerate(range(starts[color], starts[color + 1])):
                pred = cid if color == overflow and offset % batch else prior
                predecessor[cid + 1, 0] = pred
                expected[cid + 1] = (expected[pred, 0] + 1, offset // batch if color == overflow else 0, cid)
            if sizes[color]:
                prior = int(starts[color]) + 1
        bodies.velocity.assign(predecessor)
        cursor = wp.array([len(sizes)], dtype=int, device="cuda:0")
        head_active = wp.array([1], dtype=int, device="cuda:0")
        dispatch_counts = wp.zeros(2, dtype=int, device="cuda:0")
        common = [
            kernels.ConstraintContainer(),
            kernels.ContactColumnContainer(),
            bodies,
            kernels.ParticleContainer(),
            wp.float32(120),
            wp.float32(1),
            wp.array(np.arange(count, dtype=np.int32), dtype=int, device="cuda:0"),
            wp.array(starts, dtype=int, device="cuda:0"),
            wp.zeros(1, dtype=int, device="cuda:0"),
            wp.array([len(sizes)], dtype=int, device="cuda:0"),
            cursor,
            kernels.ContactContainer(),
            kernels.ContactViews(),
            wp.int32(0),
            wp.zeros(1, dtype=int, device="cuda:0"),
            wp.int32(0),
            wp.int32(0),
            wp.int32(0),
            wp.int32(0),
            wp.int32(count + 1),
        ]
        copy = kernels.CopyStateContainer()
        direction = wp.array([int(reverse)], dtype=int, device="cuda:0")

        def head_sweep(**_kwargs):
            wp.launch(_count_dispatch, dim=1, inputs=[0, dispatch_counts, cursor, head_active], device="cuda:0")
            for _ in range(8):
                wp.launch(
                    head,
                    dim=256,
                    block_dim=256,
                    inputs=[
                        *common,
                        wp.int32(256),
                        wp.int32(4),
                        head_active,
                        copy,
                        wp.int32(overflow),
                        wp.int32(batch),
                        direction,
                    ],
                    device="cuda:0",
                )

        def tail_sweep(**_kwargs):
            wp.launch(_count_dispatch, dim=1, inputs=[1, dispatch_counts, cursor, head_active], device="cuda:0")
            wp.launch_tiled(
                tail,
                dim=[1],
                block_dim=32,
                inputs=[*common, wp.int32(4), copy, wp.int32(overflow), wp.int32(batch), direction, head_active],
                device="cuda:0",
            )

        world = SimpleNamespace(
            _color_group_data=None,
            parallel_contact_prepare=False,
            _head_active=head_active,
            _partitioner=SimpleNamespace(color_cursor=cursor),
            _capture_singleworld_sweep=head_sweep,
            _capture_singleworld_tail_sweep=tail_sweep,
        )
        with wp.ScopedCapture(device="cuda:0") as capture:
            solver_phoenx.PhoenXWorld._singleworld_head_plus_tail_sweep(world, head, tail, wp.float32(120))
        wp.capture_launch(capture.graph)
        actual = bodies.position.numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(bodies.inverse_mass.numpy()[1:], np.ones(count))
        assert int(cursor.numpy()[0]) == 0
        counts = dispatch_counts.numpy()
        assert int(counts.sum()) <= 32, "Scheduler failed to make progress"
        return actual, counts


@unittest.skipUnless(wp.is_cuda_available(), "CUDA graph scheduler required")
class TestTailFirstDispatch(unittest.TestCase):
    def test_small_large_small_transitions_and_reverse(self):
        """Preserve dependent color order across both head-tail transitions."""
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                baseline, _ = _run_schedule([2, 8, 3], reverse, -1, 1, False)
                candidate, counts = _run_schedule([2, 8, 3], reverse, -1, 1, True)
                np.testing.assert_array_equal(candidate, baseline)
                self.assertGreater(counts[0], 0)
                self.assertGreater(counts[1], 1)

    def test_all_small_skips_head(self):
        """Avoid executing any head launch when the tail drains all colors."""
        for reverse in (False, True):
            baseline, old_counts = _run_schedule([2, 3, 1], reverse, -1, 1, False)
            candidate, counts = _run_schedule([2, 3, 1], reverse, -1, 1, True)
            np.testing.assert_array_equal(candidate, baseline)
            self.assertGreater(old_counts[0], 0)
            self.assertEqual(counts[0], 0)
            self.assertEqual(counts[1], 1)

    def test_mass_split_overflow_preserves_batch_order(self):
        """Keep overflow last and preserve serial operations inside each copy."""
        for sizes in ([2, 8, 7], [2, 3, 17]):
            for reverse in (False, True):
                with self.subTest(sizes=sizes, reverse=reverse):
                    baseline, _ = _run_schedule(sizes, reverse, 2, 3, False)
                    candidate, _ = _run_schedule(sizes, reverse, 2, 3, True)
                    np.testing.assert_array_equal(candidate, baseline)

    def test_hanging_cloth_contact_has_no_stall(self):
        """Preserve real cube-cloth dynamics when contacts change the color stack."""
        # Independent current-worktree cloth runs are not bitwise reproducible,
        # including two legacy-head runs. Exact dispatch/order coverage lives
        # in the synthetic cases above; this case guards real contact progress.
        for candidate in (False, True):
            with _candidate_sweep(candidate), patch.object(fixture, "CUBE_DROP_DZ", 0.21):
                world, model, state, _pipeline, contacts, _cube = fixture._build_scene(
                    mass_splitting=True, device="cuda:0"
                )
                # Force both paths on this compact version of the hanging-cloth
                # reproducer; the cube begins just above the cloth for prompt impact.
                world._fuse_threshold = 8
                rest_positions = model.particle_q.numpy().copy()
                pinned = model.particle_inv_mass.numpy() == 0
                # Match the fixture initialization: materialize lazy buffers
                # before capturing the repeatable step.
                fixture._step_once(world, model, state, contacts, "cuda:0")
                with wp.ScopedCapture(device="cuda:0") as capture:
                    fixture._step_once(world, model, state, contacts, "cuda:0")
                history = []
                contact_peak = 0
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                    contact_peak = max(contact_peak, int(contacts.rigid_contact_count.numpy()[0]))
                    history.append(
                        np.concatenate(
                            (
                                state.body_q.numpy().ravel(),
                                state.body_qd.numpy().ravel(),
                                state.particle_q.numpy().ravel(),
                                state.particle_qd.numpy().ravel(),
                            )
                        )
                    )
                    self.assertEqual(int(world._partitioner.color_cursor.numpy()[0]), 0)
                self.assertGreater(contact_peak, 0, "The cube never contacted the cloth")
                self.assertTrue(np.isfinite(history).all())
                positions = state.particle_q.numpy()
                pinned_drift = np.linalg.norm(positions[pinned] - rest_positions[pinned], axis=1)
                self.assertLess(float(pinned_drift.max()), 5.0e-3)
                ratios = fixture._edge_stretch_ratios(positions, rest_positions, model.tri_indices.numpy())
                self.assertGreater(float(ratios.min()), 0.3)
                self.assertLess(float(ratios.max()), 3.0)


if __name__ == "__main__":
    unittest.main()
