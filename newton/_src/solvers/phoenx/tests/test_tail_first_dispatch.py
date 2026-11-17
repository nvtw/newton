# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the single-world sweep's colour order and cross-block memory visibility."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx import simulation
from newton._src.solvers.phoenx import simulation_kernels as kernels
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.tests import test_cloth_mass_splitting as fixture

_DEVICE = "cuda:0"
_THRESHOLD = 4
_FACTORIES = (
    kernels._make_singleworld_head_color_func,
    kernels._make_singleworld_tail_color_func,
    kernels._make_singleworld_fused_kernel,
    kernels._make_singleworld_one_launch_kernel,
)


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
    # batch, usually written by another block on another SM. This makes
    # missing, reordered, or stale work observable without floating-point
    # ambiguity or dependence on order inside an independent color.
    slot = cid + 1
    predecessor = wp.int32(bodies.velocity[slot][0])
    bodies.position[slot] = wp.vec3(
        bodies.position[predecessor][0] + 1.0, wp.float32(parallel_id), wp.float32(colored_slot)
    )
    bodies.inverse_mass[slot] += 1.0


def _expected(sizes, reverse, overflow, batch):
    count = sum(sizes)
    starts = np.r_[0, np.cumsum(sizes)].astype(np.int32)
    order = list(range(len(sizes) if overflow < 0 else overflow))
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
    return starts, predecessor, expected


def _run_schedule(sizes, reverse, overflow, batch, phase, total_threads, blocks, repeats=1):
    """Run captured sweeps through ``PhoenXWorld._singleworld_head_plus_tail_sweep``.

    ``phase="relax"`` runs the whole sweep in one launch; ``phase="iterate"``
    alternates the fast-math one-launch kernel with the fused tail.
    """
    flags = {
        "phase": phase,
        "cloth_support": False,
        "has_joints": False,
        "has_contacts": False,
        "has_mass_splitting": True,
        "has_sleeping": False,
        "has_soft_contact_pd": False,
    }
    for factory in _FACTORIES:
        factory.cache_clear()
    try:
        with patch.object(kernels, "_make_singleworld_dispatch_func", return_value=(_trace_dispatch, None)):
            tail = kernels._make_singleworld_fused_kernel(**flags)
            axes = tuple(sorted(flags.items()))
            head = SimpleNamespace(phoenx_axes=axes)
            tail.phoenx_axes = axes
            count = sum(sizes)
            starts, predecessor, expected = _expected(sizes, reverse, overflow, batch)
            bodies = body_container_zeros(count + 1, device=_DEVICE)
            bodies.velocity.assign(predecessor)
            cursor = wp.array([len(sizes)], dtype=int, device=_DEVICE)
            barrier = wp.zeros(4, dtype=int, device=_DEVICE)
            launches = wp.zeros(2, dtype=int, device=_DEVICE)
            common = [
                kernels.ConstraintContainer(),
                kernels.ContactColumnContainer(),
                bodies,
                kernels.ParticleContainer(),
                wp.float32(120),
                wp.float32(1),
                wp.array(np.arange(count, dtype=np.int32), dtype=int, device=_DEVICE),
                wp.array(starts, dtype=int, device=_DEVICE),
                wp.zeros(1, dtype=int, device=_DEVICE),
                wp.zeros(1, dtype=int, device=_DEVICE),
                wp.array([len(sizes)], dtype=int, device=_DEVICE),
                cursor,
                kernels.ContactContainer(),
                kernels.ContactViews(),
                wp.int32(0),
                wp.zeros(1, dtype=int, device=_DEVICE),
                wp.int32(0),
                wp.int32(0),
                wp.int32(0),
                wp.int32(0),
                wp.int32(count + 1),
            ]
            ms = [kernels.CopyStateContainer(), wp.int32(overflow), wp.int32(batch)]
            direction = wp.array([int(reverse)], dtype=int, device=_DEVICE)

            def one_launch(kernel, _idt, _contact_container=None):
                wp.launch(_count_launch, dim=1, inputs=[0, launches, cursor], device=_DEVICE)
                wp.launch(
                    kernel,
                    dim=blocks * kernels.ONE_LAUNCH_BLOCK_DIM,
                    inputs=[
                        *common,
                        wp.int32(total_threads),
                        wp.int32(_THRESHOLD),
                        *ms,
                        direction,
                        barrier,
                        wp.int32(blocks),
                    ],
                    block_dim=kernels.ONE_LAUNCH_BLOCK_DIM,
                    device=_DEVICE,
                )

            def tail_sweep(kernel, **_kwargs):
                wp.launch(_count_launch, dim=1, inputs=[1, launches, cursor], device=_DEVICE)
                wp.launch_tiled(
                    kernel,
                    dim=[1],
                    block_dim=kernels.ONE_LAUNCH_BLOCK_DIM,
                    inputs=[*common, wp.int32(_THRESHOLD), *ms, direction],
                    device=_DEVICE,
                )

            world = SimpleNamespace(
                _color_group_data=None,
                parallel_contact_prepare=False,
                device=wp.get_device(_DEVICE),
                _partitioner=SimpleNamespace(color_cursor=cursor),
                _launch_one_launch_sweep=one_launch,
                _capture_singleworld_tail_sweep=tail_sweep,
                _singleworld_kernels=tuple,
            )
            with wp.ScopedCapture(device=_DEVICE) as capture:
                simulation.PhoenXWorld._singleworld_head_plus_tail_sweep(world, head, tail, wp.float32(120))
            results = []
            for _ in range(repeats):
                bodies.position.zero_()
                bodies.inverse_mass.zero_()
                launches.zero_()
                cursor.fill_(len(sizes))
                wp.capture_launch(capture.graph)
                results.append(
                    (bodies.position.numpy().copy(), bodies.inverse_mass.numpy()[1:].copy(), launches.numpy().copy())
                )
                assert int(cursor.numpy()[0]) == 0
            # The barrier and exit counters return to zero for the next sweep.
            assert int(barrier.numpy()[0]) == 0 and int(barrier.numpy()[2]) == 0
            return results, expected
    finally:
        for factory in _FACTORIES:
            factory.cache_clear()


@wp.kernel(enable_backward=False)
def _count_launch(kind: int, counts: wp.array[int], cursor: wp.array[int]):
    counts[kind] += 1
    # Turn a scheduling deadlock into a bounded assertion failure.
    if counts[0] + counts[1] > 64:
        cursor[0] = 0


@unittest.skipUnless(wp.is_cuda_available(), "CUDA graph scheduler required")
class TestTailFirstDispatch(unittest.TestCase):
    def _check(self, sizes, reverse=False, overflow=-1, batch=1, phases=("relax", "iterate"), repeats=1):
        sm = wp.get_device(_DEVICE).sm_count
        for phase in phases:
            for total_threads, blocks in ((32, 1), (256, 3), (4 * sm * 32, sm), (12 * sm * 32, sm)):
                with self.subTest(sizes=sizes[:6], reverse=reverse, phase=phase, blocks=blocks, threads=total_threads):
                    results, expected = _run_schedule(
                        sizes, reverse, overflow, batch, phase, total_threads, blocks, repeats
                    )
                    for position, inverse_mass, launches in results:
                        np.testing.assert_array_equal(position, expected)
                        np.testing.assert_array_equal(inverse_mass, np.ones(sum(sizes)))
                        self.assertLessEqual(int(launches.sum()), 2 * len(sizes) + 3)
                        if phase == "relax":
                            # Same compile options: the whole sweep is one launch.
                            self.assertEqual(launches.tolist(), [1, 0])
                    yield phase, results

    def test_small_large_small_transitions_and_reverse(self):
        """Preserve dependent colour order across head-tail transitions."""
        for reverse in (False, True):
            for phase, results in self._check([2, 8, 3], reverse):
                if phase == "iterate":
                    # Leading tail, one-launch head, trailing tail; no loop round.
                    self.assertEqual(results[0][2].tolist(), [1, 2])
            list(self._check([8, 9, 1, 2, 3, 9, 2], reverse))

    def test_all_small_and_all_large(self):
        for reverse in (False, True):
            list(self._check([2, 3, 1], reverse))
            list(self._check([8] * 15 + [2, 3, 8, 2], reverse))

    def test_mass_split_overflow_preserves_batch_order(self):
        """Keep overflow last and preserve serial operations inside each copy."""
        for sizes in ([2, 8, 7], [2, 3, 17], [40, 3, 2, 300]):
            for reverse in (False, True):
                list(self._check(sizes, reverse, overflow=len(sizes) - 1, batch=3))

    def test_large_colours_cross_block_visibility(self):
        """Every element reads a predecessor that another SM wrote in an earlier colour."""
        rng = np.random.default_rng(3)
        sizes = [int(s) for s in rng.choice([1, 2, 3, 4, 700, 5000, 30000], size=40)]
        list(self._check(sizes, repeats=3))

    def test_hanging_cloth_contact_has_no_stall(self):
        """Preserve real cube-cloth dynamics when contacts change the color stack."""
        with patch.object(fixture, "CUBE_DROP_DZ", 0.21):
            world, model, state, _pipeline, contacts, _cube = fixture._build_scene(mass_splitting=True, device=_DEVICE)
            # Force both small and large colours on this compact version of the
            # hanging-cloth reproducer; the cube begins just above the cloth.
            world._fuse_threshold = 8
            rest_positions = model.particle_q.numpy().copy()
            pinned = model.particle_inv_mass.numpy() == 0
            # Match the fixture initialization: materialize lazy buffers
            # before capturing the repeatable step.
            fixture._step_once(world, model, state, contacts, _DEVICE)
            with wp.ScopedCapture(device=_DEVICE) as capture:
                fixture._step_once(world, model, state, contacts, _DEVICE)
            history = []
            contact_peak = 0
            for _ in range(3):
                wp.capture_launch(capture.graph)
                # Cube-cloth contacts arrive through the deformable contact arrays.
                contact_peak = max(
                    contact_peak,
                    int(contacts.rigid_contact_count.numpy()[0]) + int(contacts.soft_contact_count.numpy()[0]),
                )
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


if __name__ == "__main__":
    unittest.main()
