# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare independent island blocks to the original color-ordered CUDA sweep."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.islands.iteration_schedule import (
    _MERGED_THREADS,
    _index_colors,
    _key_island_order,
    _merged_grid_sync,
    _sync_island_warps,
)
from newton._src.solvers.phoenx.simulation_kernels import ONE_LAUNCH_BLOCK_DIM
from newton._src.solvers.phoenx.tests.test_parallel_prepare_sleeping import _build_model


@wp.kernel(enable_backward=False, grid_stride=False, module="unique")
def _split_block_barriers(state: wp.array[wp.int32], steps: wp.array2d[wp.int32], num_blocks: int):
    block, lane = wp.tid()
    if lane < _MERGED_THREADS:
        for step in range(8):
            if lane == 0:
                steps[block, step] = wp.atomic_add(state, 2, 1) / num_blocks
            _merged_grid_sync(state, wp.int32(num_blocks))
        return
    for _step in range(8):
        _sync_island_warps()


class TestIslandIterationSchedule(unittest.TestCase):
    def test_color_index_covers_the_entire_island(self):
        """Index every live slot with its color and the end of that color's run."""
        device = "cpu"
        keys = wp.array([0, 0, 0, 0, 1, 2], dtype=int, device=device)
        slots = wp.array([0, 1, 3, 4, 7, 8], dtype=int, device=device)
        island_ends = wp.array([4, 5], dtype=int, device=device)
        starts = wp.array([0, 2, 3, 5, 8], dtype=int, device=device)
        num_colors = wp.array([4], dtype=int, device=device)
        families = wp.empty(0, dtype=int, device=device)
        colors = wp.full(6, -1, dtype=int, device=device)
        ends = wp.full_like(colors, -1)
        wp.launch(
            _index_colors,
            dim=len(keys),
            inputs=[keys, slots, island_ends, 2, starts, num_colors, families, False, colors, ends, families],
            device=device,
        )
        np.testing.assert_array_equal(colors.numpy(), [0, 0, 2, 2, 3, -1])
        np.testing.assert_array_equal(ends.numpy(), [2, -1, 4, -1, 5, -1])

    def test_islands_are_ordered_longest_first(self):
        """Order every island once by its color span, longest first, ties in index order.

        Blocks deal the ordered islands out in alternating directions, so the
        longest serial color chains are spread over different blocks.
        """
        device = "cpu"
        roots = wp.array([3, 1, 0, 4, -1, -1], dtype=int, device=device)
        count = wp.array([4], dtype=int, device=device)
        first_color = wp.array([0, 2, 0, 5, 1], dtype=int, device=device)
        last_color = wp.array([1, 9, 0, 6, 9], dtype=int, device=device)
        keys = wp.empty(2 * len(roots), dtype=int, device=device)
        order = wp.empty(2 * len(roots), dtype=int, device=device)
        wp.launch(
            _key_island_order,
            dim=len(roots),
            inputs=[roots, count, first_color, last_color, keys, order],
            device=device,
        )
        wp.utils.radix_sort_pairs(keys, order, len(roots), end_bit=17)
        # Spans: island 4 -> 9, 1 -> 8, 3 -> 2, 0 -> 2 (ties keep their index order).
        np.testing.assert_array_equal(order.numpy()[:4], [4, 1, 3, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "Island iteration scheduling requires CUDA")
    def test_split_block_barriers_allow_several_blocks_per_sm(self):
        """Pass the merged warps' grid barrier with four blocks per SM, as small kernels launch.

        A register barrier id would reserve every hardware barrier of the SM
        for one block, so the grid would not be co-resident and would hang.
        """
        with wp.ScopedDevice("cuda:0"):
            blocks = 4 * wp.get_device().sm_count
            state = wp.zeros(3, dtype=wp.int32)
            steps = wp.full((blocks, 8), -1, dtype=wp.int32)
            with wp.ScopedCapture() as capture:
                wp.launch(
                    _split_block_barriers,
                    (blocks, ONE_LAUNCH_BLOCK_DIM),
                    [state, steps, blocks],
                    block_dim=ONE_LAUNCH_BLOCK_DIM,
                )
            wp.capture_launch(capture.graph)
            # Every block takes its ticket of a step only after all took those of the previous step.
            np.testing.assert_array_equal(steps.numpy(), np.broadcast_to(np.arange(8), (blocks, 8)))

    @unittest.skipUnless(wp.is_cuda_available(), "Island iteration scheduling requires CUDA")
    def test_mixed_islands_in_captured_sweeps(self):
        """Preserve poses and impulses through multiple warm starts on the same island schedule."""
        # The stack and anchored pendulum share the static world body. Their
        # contacts and joints must form independent writable components.
        with wp.ScopedDevice("cuda:0"):
            model = _build_model()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")

            def run(independent, giant=False, temporal=False, merge_width=None):
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    step_layout="single_world",
                    joint_mode="maximal_pgs",
                    mass_splitting=False,
                    substeps=2,
                    parallel_contact_prepare=True,
                    solver_iterations=3,
                    island_parallel_iterations=independent,
                    temporal_iterations=temporal,
                )
                world = solver.world
                # Island blocks solve every color with the head kernel. Without a
                # fused tail, the color-ordered sweep uses the same arithmetic.
                world._fuse_threshold = 0
                if merge_width is not None:
                    # Only the island with wider colors joins the grid-swept merged
                    # island; the other one is walked meanwhile by the warps
                    # each block keeps for small islands.
                    world._dispatcher._island_schedule.merge_width = merge_width
                if giant:
                    # Every island is solved by the whole grid, one color at a time.
                    schedule = world._dispatcher._island_schedule
                    schedule.merge_width = 0
                    # Block 0 walks the narrow colors, the grid the wide ones.
                    schedule.fuse_columns = 1
                state, control, contacts = model.state(), model.control(), pipeline.contacts()

                def step():
                    pipeline.collide(state, contacts)
                    solver.step(state, state, control, contacts, 1 / 240)

                step()
                with wp.ScopedCapture() as capture:
                    step()
                history = []
                for _ in range(8):
                    wp.capture_launch(capture.graph)
                    history.append(
                        (state.body_q.numpy(), state.body_qd.numpy(), world._contact_container.impulses.numpy())
                    )
                if independent and not giant:
                    schedule = world._dispatcher._island_schedule
                    self.assertGreaterEqual(int(schedule.count.numpy()[0]), 2)
                    self.assertGreater(int(world._partitioner.num_colors.numpy()[0]), 1)
                    merged = world.num_bodies
                    merged_columns = int(schedule.ends.numpy()[merged] - schedule.starts.numpy()[merged])
                    self.assertEqual(merged_columns > 0, merge_width is not None)
                return history

            for temporal in (False, True):
                reference = run(False, temporal=temporal)
                for actual in (
                    run(True, temporal=temporal),
                    run(True, giant=True, temporal=temporal),
                    run(True, temporal=temporal, merge_width=1),
                ):
                    for expected_step, actual_step in zip(reference, actual, strict=True):
                        for expected, result in zip(expected_step, actual_step, strict=True):
                            np.testing.assert_array_equal(result.view(np.uint8), expected.view(np.uint8))

    @unittest.skipUnless(wp.is_cuda_available(), "Island iteration scheduling requires CUDA")
    def test_islands_sweep_their_requested_iterations(self):
        """Sweep a stack's island five times and a pendulum's twice in the same captured step.

        Islands are independent sets, so each one must follow the trajectory of
        a scene that sweeps every island as often as it requested, bit for bit.
        A fixed contact frequency keeps the contact spring independent of the
        global count.
        """
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            newton.solvers.SolverPhoenX.register_custom_attributes(builder)
            builder.add_ground_plane()
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.6)
            stack = []
            for i in range(3):
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.1 + 0.201 * i), q=wp.quat_identity()),
                    custom_attributes={"phoenx:solver_iterations": 5},
                )
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                stack.append(body)
            parent = -1
            joints = []
            for i in range(2):
                link = builder.add_link(
                    xform=wp.transform(p=wp.vec3(2.0, 0.0, 2.0 - 0.4 * i), q=wp.quat_rpy(0.0, 0.3, 0.0))
                )
                builder.add_shape_box(
                    link, hx=0.05, hy=0.05, hz=0.15, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
                )
                joints.append(
                    builder.add_joint_revolute(
                        parent,
                        link,
                        parent_xform=wp.transform(p=wp.vec3(2.0, 0.0, 2.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.2)),
                        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.2)),
                        axis=newton.Axis.Y,
                    )
                )
                parent = link
            builder.add_articulation(joints)
            model = builder.finalize()
            requested = model.phoenx.solver_iterations.numpy()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")

            def run(iterations, per_body, merge_width=None):
                model.phoenx.solver_iterations.assign(requested if per_body else np.zeros_like(requested))
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    step_layout="single_world",
                    joint_mode="maximal_pgs",
                    mass_splitting=False,
                    substeps=2,
                    solver_iterations=iterations,
                    contact_hertz=20.0,
                )
                schedule = solver.world._dispatcher._island_schedule
                if merge_width is not None:
                    # The stack's island, with the wider colors, is swept by the grid.
                    schedule.merge_width = merge_width
                state = model.state()
                # Drop the stack so that its contacts are still converging.
                state.body_qd.assign(np.tile([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], (model.body_count, 1)))
                control, contacts = model.control(), pipeline.contacts()

                def step():
                    pipeline.collide(state, contacts)
                    solver.step(state, state, control, contacts, 1 / 120)

                step()
                with wp.ScopedCapture() as capture:
                    step()
                for _ in range(10):
                    wp.capture_launch(capture.graph)
                if per_body:
                    self.assertEqual({2, 5} - set(schedule.island_iterations.numpy().tolist()), set())
                return state.body_q.numpy()

            light, heavy = run(2, False), run(5, False)
            self.assertFalse(np.array_equal(light[stack], heavy[stack]))
            pendulum = np.setdiff1d(np.arange(model.body_count), stack)
            for merge_width in (None, 1):
                mixed = run(2, True, merge_width)
                np.testing.assert_array_equal(mixed[stack].view(np.uint8), heavy[stack].view(np.uint8))
                np.testing.assert_array_equal(mixed[pendulum].view(np.uint8), light[pendulum].view(np.uint8))

    def test_requests_require_the_island_schedule(self):
        """Reject sweep requests above ``solver_iterations`` that no island schedule could honor."""
        builder = newton.ModelBuilder()
        newton.solvers.SolverPhoenX.register_custom_attributes(builder)
        body = builder.add_body(custom_attributes={"phoenx:solver_iterations": 4})
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "island schedule"):
            newton.solvers.SolverPhoenX(model, step_layout="single_world", solver_iterations=2)


if __name__ == "__main__":
    unittest.main()
