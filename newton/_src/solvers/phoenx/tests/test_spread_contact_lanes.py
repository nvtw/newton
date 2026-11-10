# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Spreading the contact columns of a color over warps must not change the solve."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx import simulation as phoenx_simulation
from newton._src.solvers.phoenx.simulation_kernels import _spread_lane
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


@wp.kernel(enable_backward=False)
def _spread_lane_kernel(lanes: wp.int32, output: wp.array[wp.int32]):
    tid = wp.tid()
    output[tid] = _spread_lane(tid, lanes)


class TestSpreadLane(unittest.TestCase):
    def test_work_lanes_are_a_permutation_across_warps(self):
        """Every work lane has one physical lane and neighbours sit in different warps."""
        for lanes in (32, 96, 1024, 24064):
            with self.subTest(lanes=lanes):
                output = wp.zeros(lanes, dtype=wp.int32, device="cpu")
                wp.launch(_spread_lane_kernel, dim=lanes, inputs=[lanes], outputs=[output], device="cpu")
                work = output.numpy()
                np.testing.assert_array_equal(np.sort(work), np.arange(lanes))
                warp_of_work = np.empty(lanes, dtype=np.int64)
                warp_of_work[work] = np.arange(lanes) // 32
                warps = lanes // 32
                # The first items of a color take one lane in every warp
                # before any warp receives a second one.
                np.testing.assert_array_equal(warp_of_work[:warps], np.arange(warps))
                np.testing.assert_array_equal(warp_of_work[warps : 2 * warps], np.arange(warps))


def _build_box_field(spread: bool | None) -> _PhoenXScene:
    """Two layers of boxes: many ground columns in one color plus box-box columns."""
    # A 32-warp grid makes the larger colors wrap onto several lanes per warp.
    scene = _PhoenXScene(
        fps=120,
        substeps=4,
        solver_iterations=4,
        velocity_iterations=1,
        step_layout="single_world",
        max_thread_blocks=32,
    )
    scene.add_ground_plane()
    for x in range(12):
        for y in range(12):
            scene.add_box(position=(0.25 * x, 0.25 * y, 0.1), half_extents=(0.1, 0.1, 0.1))
            if (x + y) % 2 == 0:
                # Offset upper boxes rest on two lower boxes and tip over.
                scene.add_box(position=(0.25 * x + 0.12, 0.25 * y, 0.31), half_extents=(0.1, 0.1, 0.1))
    scene.finalize()
    # Route every color through the persistent grid instead of the fused tail.
    scene.world._fuse_threshold = 0
    original = phoenx_simulation.get_singleworld_kernel
    kernels = []

    def get_kernel(*args, **kwargs):
        if kwargs.get("fused", False):
            return original(*args, **kwargs)
        if spread is not None:
            kwargs["spread_contact_lanes"] = spread
        kernels.append(original(*args, **kwargs))
        return kernels[-1]

    with patch.object(phoenx_simulation, "get_singleworld_kernel", get_kernel):
        for _ in range(40):
            scene.step()
    scene.kernels = kernels
    return scene


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping")
class TestSpreadContactLanesParity(unittest.TestCase):
    def test_unpacked_colors_solve_bitwise_equal(self):
        dense = _build_box_field(spread=False)
        spread = _build_box_field(spread=True)
        default = _build_box_field(spread=None)
        self.assertFalse(dense.world._colored_contact_headers)
        # The switch selects different kernels, and unpacked columns spread by default.
        self.assertFalse(set(dense.kernels) & set(spread.kernels))
        self.assertEqual(set(default.kernels), set(spread.kernels))
        # More columns than warps: work lanes wrap onto a second lane per warp.
        partitioner = dense.world._partitioner
        sizes = np.diff(partitioner.color_starts.numpy()[: int(partitioner.num_colors.numpy()[0]) + 1])
        self.assertGreater(int(sizes.max()), dense.world._singleworld_total_threads // 32)
        moved = np.linalg.norm(dense.bodies.velocity.numpy()[1:], axis=1)
        self.assertGreater(float(moved.max()), 1.0e-3, "the scene must still be moving")
        for name in ("position", "orientation", "velocity", "angular_velocity"):
            with self.subTest(name=name):
                reference = getattr(dense.bodies, name).numpy()
                np.testing.assert_array_equal(getattr(spread.bodies, name).numpy(), reference)
                np.testing.assert_array_equal(getattr(default.bodies, name).numpy(), reference)


if __name__ == "__main__":
    unittest.main()
