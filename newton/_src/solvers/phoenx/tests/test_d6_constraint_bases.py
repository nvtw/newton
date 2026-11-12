# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Locked-axis bases of the direct-equality joints, which joints with equal free axes share."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.direct_equality import _generic_d6_constraint_bases


def _free_axes(model: newton.Model, joint: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    start = int(model.joint_qd_start.numpy()[joint])
    linear, angular = (int(n) for n in model.joint_dof_dim.numpy()[joint])
    axes = model.joint_axis.numpy()
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    free = [axes[d] / np.linalg.norm(axes[d]) for d in range(start, start + linear + angular) if lower[d] <= upper[d]]
    free_linear = [axes[d] for d in range(start, start + linear) if lower[d] <= upper[d]]
    return free_linear, free[len(free_linear) :]


class TestD6ConstraintBases(unittest.TestCase):
    def test_bases_lock_exactly_the_other_axes(self) -> None:
        rng = np.random.default_rng(3)
        builder = newton.ModelBuilder()
        axes = [tuple(float(x) for x in a / np.linalg.norm(a)) for a in rng.normal(size=(3, 3))]
        for i in range(60):
            child = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, float(i)), wp.quat_identity()), mass=1.0)
            axis = axes[i % 3]
            kind = i % 5
            if kind == 0:
                joint = builder.add_joint_revolute(-1, child, axis=axis)
            elif kind == 1:
                joint = builder.add_joint_prismatic(-1, child, axis=axis)
            elif kind == 2:
                joint = builder.add_joint_ball(-1, child)
            elif kind == 3:
                joint = builder.add_joint_fixed(-1, child)
            else:
                cfg = newton.ModelBuilder.JointDofConfig
                joint = builder.add_joint_d6(
                    -1,
                    child,
                    linear_axes=[cfg(axis=axis)],
                    angular_axes=[
                        cfg(axis=(1.0, 0.0, 0.0)),
                        cfg(axis=(0.0, 1.0, 0.0), limit_lower=0.5, limit_upper=-0.5),
                    ],
                )
            builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        linear_axes, angular_axes, linear_count, angular_count, _structural = _generic_d6_constraint_bases(model)
        seen: dict[bytes, tuple] = {}
        for joint in range(model.joint_count):
            free_linear, free_angular = _free_axes(model, joint)
            self.assertEqual(linear_count[joint], 3 - len(free_linear))
            self.assertEqual(angular_count[joint], 3 - len(free_angular))
            for locked, count, free in (
                (linear_axes[joint], linear_count[joint], free_linear),
                (angular_axes[joint], angular_count[joint], free_angular),
            ):
                basis = locked[:count]
                np.testing.assert_allclose(basis @ basis.T, np.eye(count), atol=1.0e-5)
                for axis in free:
                    np.testing.assert_allclose(basis @ (axis / np.linalg.norm(axis)), 0.0, atol=1.0e-5)
            key = b"|".join(
                b"".join(np.asarray(a, dtype=np.float32).tobytes() for a in group)
                for group in (free_linear, free_angular)
            )
            result = (linear_axes[joint].tobytes(), angular_axes[joint].tobytes())
            self.assertEqual(seen.setdefault(key, result), result, f"joint {joint} differs from an equal joint")

    def test_model_without_dofs(self) -> None:
        builder = newton.ModelBuilder()
        child = builder.add_link(mass=1.0)
        builder.add_articulation([builder.add_joint_fixed(-1, child)])
        _linear, _angular, linear_count, angular_count, structural = _generic_d6_constraint_bases(
            builder.finalize(device="cpu")
        )
        self.assertEqual((int(linear_count[0]), int(angular_count[0]), int(structural[0])), (3, 3, 6))


if __name__ == "__main__":
    unittest.main()
