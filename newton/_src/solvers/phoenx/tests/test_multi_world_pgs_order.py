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
from newton._src.solvers.phoenx import solver_phoenx_kernels
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
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


class TestMultiWorldFastTailSolveContract(unittest.TestCase):
    def test_solve_schedule_keeps_high_substep_total_work_constant(self) -> None:
        solver_iterations = 8
        joint_sweeps, contact_sweeps, outer_chunk = PhoenXWorld._choose_fast_tail_solve_schedule(substeps=80)
        outer_iterations = (solver_iterations + outer_chunk - 1) // outer_chunk

        self.assertEqual((joint_sweeps, contact_sweeps, outer_chunk), (2, 2, 2))
        self.assertEqual(outer_iterations * joint_sweeps, solver_iterations)
        self.assertEqual(outer_iterations * contact_sweeps, solver_iterations)
        self.assertEqual(PhoenXWorld._choose_fast_tail_solve_schedule(substeps=20), (3, 3, 1))

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
