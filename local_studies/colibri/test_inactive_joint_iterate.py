# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Preserve active block-joint transitions when skipping unused axial metrics."""

import unittest

import numpy as np

import newton
from local_studies.colibri.inactive_joint_iterate import iterate, original
from newton._src.solvers.phoenx import solver_phoenx_kernels as kernels
from newton._src.solvers.phoenx.constraints import constraint_joint as joint
from newton._src.solvers.phoenx.dispatch import color_groups
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum
from newton._src.solvers.phoenx.tests.test_direct_drive import _make_prismatic


def run(fast, prismatic, split):
    """Exercise active and inactive rows through actual solver dispatch."""
    kernels.joint_constraint_iterate_inequality = iterate if fast else original
    for value in vars(kernels).values():
        if hasattr(value, "cache_clear"):
            value.cache_clear()
    color_groups.get_sweep_kernel.cache_clear()
    model = (
        _make_prismatic(mass=1.0, armature=0.0, passive_damping=0.0, kp=0.0, kd=0.0) if prismatic else make_model(0.0)
    )
    model.joint_limit_lower.assign(np.array([-0.02], np.float32))
    model.joint_limit_upper.assign(np.array([0.02], np.float32))
    solver = make_solver(model, mass_splitting=split, max_colored_partitions=0)
    state = model.state()
    velocity = state.body_qd.numpy()
    velocity[0, 2 if prismatic else 5] = -2.0
    velocity[1, 2 if prismatic else 5] = 2.0
    state.body_qd.assign(velocity)
    control = model.control()
    control.joint_target_q.assign(np.array([0.5], np.float32))
    snapshots = []
    axial_masses = []
    momentum = _total_momentum(model, state)
    for frame in range(10):
        if frame == 3:
            model.joint_limit_lower.assign(np.array([1.0], np.float32))
            model.joint_limit_upper.assign(np.array([-1.0], np.float32))
        if frame == 4:
            model.joint_friction.assign(np.array([0.2], np.float32))
        if frame == 5:
            model.joint_friction.assign(np.array([0.0], np.float32))
            model.joint_velocity_limit.assign(np.array([0.1], np.float32))
        if frame == 6:
            model.joint_velocity_limit.assign(np.array([float("inf")], np.float32))
            model.joint_target_ke.assign(np.array([40.0], np.float32))
            model.joint_effort_limit.assign(np.array([0.2], np.float32))
        if frame >= 3:
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        if frame == 3:
            multipliers = solver.world.constraints.multipliers.numpy()
            for offset in (int(joint._MUL_ACC_LIMIT), int(joint._MUL_ACC_FRICTION)):
                multipliers[offset // 4, 0, offset % 4] = 0.125
            solver.world.constraints.multipliers.assign(multipliers)
        state.clear_forces()
        solver.step(state, state, control, None, 0.01)
        np.testing.assert_allclose(_total_momentum(model, state), momentum, atol=2e-6, rtol=0)
        data = solver.world.constraints.data.numpy()
        axial_masses.append(float(data[int(joint._OFF_EFF_INV_AXIAL), 0]))
        fields = [int(joint._OFF_CLAMP)]
        for offset in (joint._OFF_R1_B1, joint._OFF_R1_B2, joint._OFF_AXIS_WORLD):
            fields.extend(range(int(offset), int(offset) + 3))
        if not prismatic:
            fields.extend([int(joint._OFF_REVOLUTION_COUNTER), int(joint._OFF_PREVIOUS_QUATERNION_ANGLE)])
        snapshots.append(
            (
                state.body_q.numpy(),
                state.body_qd.numpy(),
                data[fields],
                solver.world.constraints.multipliers.numpy(),
                solver.world.constraints.bilateral.accumulated.numpy(),
            )
        )
        if frame == 0:
            assert data.view(np.int32)[int(joint._OFF_CLAMP), 0] == 0
        if frame == 1:
            assert data.view(np.int32)[int(joint._OFF_CLAMP), 0] != 0
    return snapshots, axial_masses


class TestInactivePrepare(unittest.TestCase):
    def test_limit_crossing_and_property_transitions(self):
        """Retain finite crossings, stale releases, friction, speed limits and drives."""
        try:
            for prismatic in (False, True):
                for split in (False, True):
                    expected, original_mass = run(False, prismatic, split)
                    actual, skipped_mass = run(True, prismatic, split)
                    self.assertEqual(original_mass[0], skipped_mass[0])
                    self.assertEqual(original_mass[1], skipped_mass[1])
                    for frame, (before, after) in enumerate(zip(expected, actual, strict=True)):
                        for a, b in zip(before, after, strict=True):
                            np.testing.assert_array_equal(
                                a.view(np.uint32), b.view(np.uint32), err_msg=f"{prismatic=}, {split=}, {frame=}"
                            )
        finally:
            kernels.joint_constraint_iterate_inequality = original


if __name__ == "__main__":
    unittest.main()
