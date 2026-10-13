# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row-parallel contact preparation with per-body warm start matches the colored sweep."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.simulation import PhoenXWorld
from newton.examples.phoenx import example_phoenx_ragdoll_pile as scene
from newton.viewer import ViewerNull


def _make_example(gathered: bool):
    original = PhoenXWorld._gathered_warm_start_supported
    if not gathered:
        PhoenXWorld._gathered_warm_start_supported = lambda self: False
    try:
        return scene.Example(
            ViewerNull(),
            SimpleNamespace(grid=(4, 4, 4), seed=42, substeps=8, iterations=4, joint_solver="exact"),
        )
    finally:
        PhoenXWorld._gathered_warm_start_supported = original


class TestGatheredContactWarmStart(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_captured_trajectory_matches_colored_prepare(self):
        """Overflow headers, copies and warm starts compose into the same captured steps."""
        colored = _make_example(False)
        gathered = _make_example(True)
        try:
            self.assertIsNone(colored.solver.world._contact_body_entries)
            self.assertTrue(gathered.solver.world._use_gathered_warm_start())
            for frame in range(1, 9):
                colored.step()
                gathered.step()
                wp.synchronize()
                np.testing.assert_allclose(
                    gathered.state.body_qd.numpy(),
                    colored.state.body_qd.numpy(),
                    rtol=1.0e-5,
                    atol=1.0e-6,
                    err_msg=f"velocities at frame {frame}",
                )
        finally:
            colored.viewer.close()
            gathered.viewer.close()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_geometry_bitwise_and_warm_start_equivalent(self):
        """Prepared rows are identical; summed warm-start impulses agree to rounding."""
        viewer = ViewerNull()
        try:
            example = scene.Example(
                viewer,
                SimpleNamespace(grid=(4, 4, 4), seed=42, substeps=8, iterations=4, joint_solver="exact"),
            )
            for _ in range(12):
                example.step()
            world = example.solver.world
            self.assertTrue(world._use_gathered_warm_start())
            # Outside a step the packed headers hold canonical row offsets;
            # the in-step gather switches them to color order.
            world._gather_colored_contact_rows()
            idt = wp.float32(1.0 / world.substep_dt)
            cc = world._contact_container_solve
            total = int(world._contact_row_total.numpy()[0])
            self.assertGreater(total, 0)
            bodies = world.bodies
            state = (bodies.velocity, bodies.angular_velocity, cc.impulses, cc.lambdas, cc.derived)
            snapshot = [array.numpy().copy() for array in state]
            copies = (world._copy_state.velocity, world._copy_state.angular_velocity)
            copy_snapshot = [array.numpy().copy() for array in copies]
            results = []
            for gathered in (False, True):
                for array, values in zip(state, snapshot, strict=True):
                    array.assign(values)
                for array, values in zip(copies, copy_snapshot, strict=True):
                    array.assign(values)
                if gathered:
                    world._contact_body_entries.build(
                        world._contact_cols_packed,
                        world._partitioner.element_ids_by_color,
                        world._num_active_constraints,
                        world._contact_offset,
                        world._contact_cols.articulation_owner,
                    )
                    world._prepare_contacts_gathered(idt)
                else:
                    prepare_head = world._singleworld_kernels()[0]
                    world._partitioner.begin_sweep()
                    world._dispatcher._unrolled_sweep(prepare_head, idt, cc, average=True)
                wp.synchronize()
                results.append([array.numpy().copy() for array in state])
            (reference_v, reference_w, _, _, reference_derived), (v, w, _, _, derived) = results
            np.testing.assert_array_equal(derived[:, :total], reference_derived[:, :total])
            self.assertFalse(np.array_equal(reference_v, snapshot[0]))
            np.testing.assert_allclose(v, reference_v, rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(w, reference_w, rtol=1.0e-5, atol=1.0e-5)
            world._scatter_colored_contact_rows()
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
