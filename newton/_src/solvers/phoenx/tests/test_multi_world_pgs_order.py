# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for multi-world PGS color ordering."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx import solver_phoenx, solver_phoenx_kernels
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mini.benchmark import _make_stack_model
from newton._src.solvers.phoenx.solver_phoenx import _choose_fast_tail_solve_schedule
from newton._src.solvers.phoenx.tests.test_robot_policy_parity import (
    _g1_29dof_yaml,
    _g1_robot_model,
    _run_with_contacts,
)


def _target_from_g1_standing_pose() -> np.ndarray:
    model = _g1_robot_model()
    target = np.zeros(int(model.joint_dof_count), dtype=np.float32)
    target[6:] = model.joint_q.numpy()[7:]
    return target


def _phoenx_factory(step_layout: str, multi_world_scheduler: str = "auto"):
    def make(model: newton.Model) -> newton.solvers.SolverPhoenX:
        return newton.solvers.SolverPhoenX(
            model,
            substeps=4,
            solver_iterations=16,
            velocity_iterations=1,
            step_layout=step_layout,
            multi_world_scheduler=multi_world_scheduler,
            prepare_refresh_stride=1,
        )

    return make


def _wp_int32_arg_expr(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "wp"
        and node.func.attr == "int32"
        and len(node.args) == 1
    ):
        return None
    return ast.unparse(node)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX world bucketing tests run on CUDA only.",
)
class TestMultiWorldStableBucketing(unittest.TestCase):
    def test_monotone_run_merge_matches_stable_world_sort(self) -> None:
        device = wp.get_preferred_device()
        world_ids = np.array([2, 0, 1, 1, -1, 3, 0, 2, 1, 0, 3, -1, 2, 2, 0], dtype=np.int32)
        element_count = len(world_ids)
        element_dtype = np.dtype({"names": ["bodies"], "formats": ["8i4"], "offsets": [0], "itemsize": 32})
        host_elements = np.zeros(element_count, dtype=element_dtype)
        host_elements["bodies"][:] = -1
        for cid, world_id in enumerate(world_ids):
            if world_id >= 0:
                host_elements["bodies"][cid, 0] = cid

        elements = wp.array(
            host_elements,
            dtype=ElementInteractionData,
            device=device,
        )
        num_elements = wp.array([element_count], dtype=wp.int32, device=device)
        bodies = body_container_zeros(element_count, device=device)
        bodies.world_id.assign(np.maximum(world_ids, 0))
        particle_world = wp.zeros(1, dtype=wp.int32, device=device)
        counts = wp.zeros(4, dtype=wp.int32, device=device)
        shifted = wp.zeros(5, dtype=wp.int32, device=device)
        offsets = wp.zeros(5, dtype=wp.int32, device=device)
        run_flags = wp.zeros(element_count, dtype=wp.int32, device=device)
        run_ids = wp.zeros(element_count, dtype=wp.int32, device=device)
        run_starts = wp.zeros(element_count, dtype=wp.int32, device=device)
        num_runs = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.full(element_count, -1, dtype=wp.int32, device=device)

        wp.launch(
            solver_phoenx_kernels._count_and_mark_world_runs_kernel,
            dim=element_count,
            inputs=[
                elements,
                num_elements,
                bodies,
                particle_world,
                wp.int32(element_count),
            ],
            outputs=[counts, shifted, run_flags],
            device=device,
        )
        wp.utils.array_scan(shifted, offsets, inclusive=True)
        wp.utils.array_scan(run_flags, run_ids, inclusive=True)
        wp.launch(
            solver_phoenx_kernels._scatter_monotone_world_run_starts_kernel,
            dim=element_count,
            inputs=[num_elements, run_flags, run_ids],
            outputs=[run_starts, num_runs],
            device=device,
        )
        wp.launch(
            solver_phoenx_kernels._merge_monotone_world_runs_kernel,
            dim=4,
            inputs=[
                elements,
                bodies,
                particle_world,
                wp.int32(element_count),
                wp.int32(4),
                offsets,
                num_elements,
                run_starts,
                num_runs,
            ],
            outputs=[output],
            device=device,
        )

        expected = [
            cid for world_id in range(4) for cid, element_world in enumerate(world_ids) if element_world == world_id
        ]
        np.testing.assert_array_equal(offsets.numpy(), np.array([0, 4, 7, 11, 13], dtype=np.int32))
        np.testing.assert_array_equal(output.numpy()[: len(expected)], np.asarray(expected, dtype=np.int32))
        self.assertEqual(int(num_runs.numpy()[0]), 8)


class TestMultiWorldColoringContract(unittest.TestCase):
    def test_per_world_greedy_overflow_flag_is_cleared_before_build(self) -> None:
        source = inspect.getsource(solver_phoenx.PhoenXWorld._finish_per_world_coloring)
        clear_idx = source.index("self._per_world_greedy_overflow.zero_()")
        launch_idx = source.index("get_per_world_greedy_coloring_kernel")
        self.assertLess(clear_idx, launch_idx)

    def test_direct_greedy_path_skips_adjacency_but_fallback_rebuilds_it(self) -> None:
        rebuild_source = inspect.getsource(solver_phoenx.PhoenXWorld._rebuild_partition)
        self.assertIn(
            'if self.step_layout == "single_world" or not self._use_greedy_coloring:',
            rebuild_source,
        )
        fallback_source = inspect.getsource(solver_phoenx.PhoenXWorld._maybe_fallback_from_per_world_greedy_overflow)
        reset_idx = fallback_source.index("self._partitioner.reset")
        fallback_idx = fallback_source.index("_per_world_jp_coloring_kernel")
        self.assertLess(reset_idx, fallback_idx)


class TestMultiWorldFastTailSolveContract(unittest.TestCase):
    def test_solve_schedule_traverses_all_colors_per_iteration(self) -> None:
        self.assertEqual(_choose_fast_tail_solve_schedule(substeps=80), (1, 1, 1))
        self.assertEqual(_choose_fast_tail_solve_schedule(substeps=20), (1, 1, 1))

    def test_relax_schedule_traverses_all_colors_per_iteration(self) -> None:
        source = inspect.getsource(solver_phoenx_kernels._make_fast_tail_relax_kernel)
        self.assertIn("relax_iterations = num_iterations", source)
        self.assertNotIn("sweeps_per_dispatch = num_iterations", source)
        self.assertEqual(solver_phoenx_kernels._BLOCK_WORLD_SOLVE_INNER_SWEEPS, 1)

    def test_multi_world_kernels_alternate_color_direction(self) -> None:
        factories = (
            solver_phoenx_kernels._make_fast_tail_prepare_plus_iterate_kernel,
            solver_phoenx_kernels._make_fast_tail_relax_kernel,
            solver_phoenx_kernels._make_block_world_prepare_plus_iterate_kernel,
            solver_phoenx_kernels._make_block_world_relax_kernel,
        )
        reverse_index = "color = n_colors - wp.int32(1) - c"
        for factory in factories:
            with self.subTest(factory=factory.__name__):
                self.assertIn(reverse_index, inspect.getsource(factory))

    def test_single_world_forward_color_order_is_default(self) -> None:
        parameter = inspect.signature(solver_phoenx.PhoenXWorld).parameters["symmetric_color_sweep"]
        self.assertIs(parameter.default, False)

    def test_ordered_solve_dispatch_uses_selected_inner_sweeps(self) -> None:
        source = textwrap.dedent(inspect.getsource(solver_phoenx_kernels._make_fast_tail_prepare_plus_iterate_kernel))
        tree = ast.parse(source)
        dispatches: list[tuple[str, str | None]] = []
        outer_chunk_exprs: list[str | None] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "solve_outer_chunk":
                        outer_chunk_exprs.append(_wp_int32_arg_expr(node.value))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {
                    "_dispatch_iterate_joint",
                    "_dispatch_iterate_contact",
                    "_dispatch_iterate_cid",
                }:
                    dispatches.append((node.func.id, _wp_int32_arg_expr(node.args[-1])))

        self.assertIn("wp.int32(solve_outer_iteration_chunk)", outer_chunk_exprs)
        self.assertGreaterEqual(len([name for name, _ in dispatches if name == "_dispatch_iterate_joint"]), 1)
        self.assertGreaterEqual(len([name for name, _ in dispatches if name == "_dispatch_iterate_contact"]), 1)
        for name, inner_sweeps in dispatches:
            if name == "_dispatch_iterate_joint":
                self.assertEqual(inner_sweeps, "wp.int32(solve_joint_inner_sweeps)")
            elif name == "_dispatch_iterate_contact":
                self.assertEqual(inner_sweeps, "wp.int32(solve_contact_inner_sweeps)")
            else:
                self.fail("fast-tail solve should split joint/contact dispatches explicitly")


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX multi-world ordering tests run on CUDA only.",
)
class TestMultiWorldPgsOrder(unittest.TestCase):
    """Multi-world and single-world must mean the same PGS iterations."""

    def test_stable_rigid_coloring_matches_forced_rebuild(self) -> None:
        """Cached coloring must preserve an evolving deterministic solve."""
        device = wp.get_preferred_device()

        def make_sim(reuse_coloring: bool):
            model = _make_stack_model(16, 4, str(device))
            pipeline = newton.CollisionPipeline(
                model,
                rigid_contact_max=16 * 32,
                contact_matching="sticky",
                deterministic=True,
            )
            contacts = pipeline.contacts()
            state_0 = model.state()
            state_1 = model.state()
            solver = newton.solvers.SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=4,
                velocity_iterations=0,
                contact_friction_model="point",
                step_layout="multi_world",
                articulation_mode="maximal",
            )
            solver.world._reuse_rigid_coloring = reuse_coloring
            return model, pipeline, contacts, state_0, state_1, solver, model.control()

        def step(sim) -> None:
            _model, pipeline, contacts, state_0, state_1, solver, control = sim
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        cached = make_sim(True)
        rebuilt = make_sim(False)
        step(cached)
        step(rebuilt)
        graphs = []
        for sim in (cached, rebuilt):
            with wp.ScopedCapture(device=device) as capture:
                step(sim)
            graphs.append(capture.graph)

        # Force the first replay down the rebuild branch; later unchanged
        # frames exercise the cached branch in the same captured graph.
        cached[-2].world._previous_topology_count.assign([-1])
        dirty_frames = []
        for _ in range(12):
            for graph in graphs:
                wp.capture_launch(graph)

            dirty_frames.append(int(cached[-2].world._topology_rebuild.numpy()[0]))
            np.testing.assert_array_equal(cached[3].body_q.numpy(), rebuilt[3].body_q.numpy())
            np.testing.assert_array_equal(cached[3].body_qd.numpy(), rebuilt[3].body_qd.numpy())

        self.assertIn(1, dirty_frames)
        self.assertIn(0, dirty_frames)

    def test_g1_contact_drive_matches_single_world(self) -> None:
        """Catch color-local multi-sweeps in contact/joint scenes."""
        cfg = _g1_29dof_yaml()
        ankle_l_dof = cfg["mjw_joint_names"].index("left_ankle_pitch_joint")
        ankle_r_dof = cfg["mjw_joint_names"].index("right_ankle_pitch_joint")
        target = _target_from_g1_standing_pose()

        _, _, jq_single = _run_with_contacts(
            _phoenx_factory("single_world"),
            _g1_robot_model,
            80,
            1.0 / 200.0,
            target_pos=target,
            record_joint_q=True,
        )

        variants = (
            ("fast_tail", _phoenx_factory("multi_world")),
            ("block_world_64", _phoenx_factory("multi_world", "block_world_64")),
            # block_world_32 is the config the auto-heuristic selects for robot
            # RL fleets (e.g. Anymal); it runs the inner-sweep register-cached
            # solve, so guard it against single-world drift explicitly.
            ("block_world_32", _phoenx_factory("multi_world", "block_world_32")),
        )
        window = slice(-20, None)
        tol = math.radians(1.0)
        for variant, factory in variants:
            with self.subTest(variant=variant):
                _, _, jq_multi = _run_with_contacts(
                    factory,
                    _g1_robot_model,
                    80,
                    1.0 / 200.0,
                    target_pos=target,
                    record_joint_q=True,
                )
                for dof, label in (
                    (ankle_l_dof, "left ankle pitch"),
                    (ankle_r_dof, "right ankle pitch"),
                ):
                    q_multi = float(jq_multi[window, 7 + dof].mean())
                    q_single = float(jq_single[window, 7 + dof].mean())
                    self.assertAlmostEqual(
                        q_multi,
                        q_single,
                        delta=tol,
                        msg=(
                            f"{label}: {variant}={math.degrees(q_multi):+.2f} deg "
                            f"single_world={math.degrees(q_single):+.2f} deg"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
