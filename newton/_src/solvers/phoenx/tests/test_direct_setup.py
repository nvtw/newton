# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for direct-solver host setup policies."""

import unittest
from types import SimpleNamespace

import numpy as np

from newton._src.sim import JointTargetMode, JointType
from newton._src.solvers.phoenx.articulations.direct_equality import (
    _active_dynamic_dofs,
    _drive_dof_masks,
    _dynamic_joint_masks,
    _effective_joint_axes,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_FIXED,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
)


def _host_array(values, dtype):
    """Wrap a NumPy value with the subset of the Warp array API under test."""
    array = np.asarray(values, dtype=dtype)
    return SimpleNamespace(numpy=lambda: array)


class TestDirectSetup(unittest.TestCase):
    """Verify vectorized direct-solver setup decisions."""

    def test_drive_masks_preserve_modes_and_unlimited_effort(self) -> None:
        """Classify driven and finite-effort DoFs without scalar setup loops."""
        model = SimpleNamespace(
            joint_target_mode=_host_array(
                (
                    JointTargetMode.POSITION,
                    JointTargetMode.POSITION_VELOCITY,
                    JointTargetMode.VELOCITY,
                    JointTargetMode.NONE,
                ),
                np.int32,
            ),
            joint_target_ke=_host_array((1.0, 1.0, 0.0, 1.0), np.float32),
            joint_target_kd=_host_array((0.0, 0.0, 1.0, 1.0), np.float32),
            joint_effort_limit=_host_array((10.0, 10.0, np.inf, 10.0), np.float32),
            joint_gear=_host_array((1.0, 0.0, 1.0, 1.0), np.float32),
        )
        active, bounded = _drive_dof_masks(model)
        np.testing.assert_array_equal(active, (True, True, True, False))
        np.testing.assert_array_equal(bounded, (True, False, False, False))

    def test_axial_axes_are_gathered_and_normalized(self) -> None:
        """Gather common axial joint axes and retain the fixed-joint default."""
        model = SimpleNamespace(
            joint_count=3,
            joint_axis=_host_array(((0.0, 3.0, 0.0), (0.0, 0.0, -2.0)), np.float32),
            joint_qd_start=_host_array((0, 1, 2), np.int32),
            joint_dof_dim=_host_array(((0, 1), (1, 0), (0, 0)), np.int32),
            joint_limit_lower=_host_array((-np.inf, -np.inf), np.float32),
            joint_limit_upper=_host_array((np.inf, np.inf), np.float32),
        )
        modes = np.asarray((JOINT_MODE_REVOLUTE, JOINT_MODE_PRISMATIC, JOINT_MODE_FIXED), dtype=np.int32)
        axes = _effective_joint_axes(model, modes, np.asarray((0, 1, 2), dtype=np.int32))
        np.testing.assert_allclose(axes, ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))

    def test_dynamic_rows_reuse_drive_masks(self) -> None:
        """Reuse precomputed drive masks while classifying axial dynamic rows."""
        model = SimpleNamespace(
            joint_count=2,
            joint_type=_host_array((JointType.REVOLUTE, JointType.PRISMATIC), np.int32),
            joint_qd_start=_host_array((0, 1), np.int32),
            joint_dof_dim=_host_array(((0, 1), (1, 0)), np.int32),
            joint_limit_lower=_host_array((-np.inf, -np.inf), np.float32),
            joint_limit_upper=_host_array((np.inf, np.inf), np.float32),
            joint_armature=_host_array((0.0, 1.0), np.float32),
            joint_damping=_host_array((0.0, 0.0), np.float32),
        )
        modes = np.asarray((JOINT_MODE_REVOLUTE, JOINT_MODE_PRISMATIC), dtype=np.int32)
        dof_start = np.asarray((0, 1), dtype=np.int32)
        excluded = np.zeros(2, dtype=bool)
        drive = np.asarray((True, False))
        bounded = np.asarray((True, False))

        dynamic, direct_drive, bounded_drive = _dynamic_joint_masks(model, modes, dof_start, excluded, drive, bounded)
        dynamic_dofs = _active_dynamic_dofs(model, modes, dof_start, excluded, drive)
        np.testing.assert_array_equal(dynamic, (True, True))
        np.testing.assert_array_equal(direct_drive, (True, False))
        np.testing.assert_array_equal(bounded_drive, (True, False))
        self.assertEqual(dynamic_dofs, ((0,), (1,)))


if __name__ == "__main__":
    unittest.main()
