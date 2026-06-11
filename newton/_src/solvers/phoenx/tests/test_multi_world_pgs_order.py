# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for multi-world PGS color ordering."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
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
