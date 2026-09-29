# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Hoberman spring policy acts only at the middle cross pivots."""

import unittest

import newton
from newton._src.solvers.phoenx.examples import hoberman_sphere_data
from newton._src.solvers.phoenx.examples.example_hoberman_sphere import make_hoberman_builder


class TestHobermanMidSprings(unittest.TestCase):
    def test_spring_selection_and_rest_angles(self):
        stiffness = 100.0
        damping = 10.0
        builder = make_hoberman_builder(mid_spring_stiffness=stiffness, mid_spring_damping=damping)
        selected = 0
        for joint, (kind, label, _parent, _child, _parent_frame, _child_frame, _axes, coordinates) in enumerate(
            hoberman_sphere_data.JOINTS
        ):
            if kind == "FREE":
                continue
            dof = builder.joint_qd_start[joint]
            target = builder.joint_q_start[joint]
            if "_mid" in label:
                selected += 1
                self.assertEqual(builder.joint_target_mode[dof], newton.JointTargetMode.POSITION)
                self.assertEqual(builder.joint_target_ke[dof], stiffness)
                self.assertEqual(builder.joint_target_kd[dof], damping)
                self.assertEqual(builder.joint_target_q[target], coordinates[0])
            else:
                self.assertEqual(builder.joint_target_ke[dof], 0.0)
                self.assertEqual(builder.joint_target_kd[dof], 0.0)
        self.assertEqual(selected, 120)


if __name__ == "__main__":
    unittest.main()
