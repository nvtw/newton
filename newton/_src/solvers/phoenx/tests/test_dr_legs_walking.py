# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Walking DR Legs regression for ``SolverPhoenX``.

This pins the old single-world DR Legs settings that are known to walk:
80 substeps, 8 solver iterations, 0.001 joint armature, and the bundled
open-loop animation. The checks intentionally look at pelvis trajectory
and uprightness, not just finite state, so soft/limp drive regressions
fail even if the simulation remains numerically stable.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs

_PELVIS_LABEL = "/DR_Legs/RigidBodies/pelvis"
_SAMPLE_FRAMES = (0, 120, 240, 360, 480, 600)
_REFERENCE_PELVIS_XY = np.array(
    [
        [0.00210191, -0.00000001],
        [0.09024552, -0.07226246],
        [0.17129117, 0.05888342],
        [0.23876992, -0.01924408],
        [0.34276560, 0.02577228],
        [0.40250567, 0.01641181],
    ],
    dtype=np.float32,
)


def _quat_up_z_xyzw(q: np.ndarray) -> float:
    """World z component of local +z rotated by quaternion ``(x, y, z, w)``."""
    x, y, _z, _w = (float(v) for v in q)
    return 1.0 - 2.0 * (x * x + y * y)


def _recover_model_state(handle) -> tuple[newton.Model, newton.State]:
    model = None
    state = None
    for cell in handle.simulate_one_frame.__closure__ or ():
        obj = cell.cell_contents
        if isinstance(obj, newton.Model):
            model = obj
        elif isinstance(obj, newton.State):
            state = obj
    if model is None or state is None:
        raise RuntimeError("failed to recover DR Legs model/state from scenario closure")
    return model, state


@unittest.skipUnless(wp.is_cuda_available(), "DR Legs PhoenX walking regression requires CUDA")
class TestDrLegsWalking(unittest.TestCase):
    """The restored DR Legs settings should walk and stay roughly upright."""

    def test_single_dr_legs_walks_with_old_settings(self) -> None:
        try:
            handle = dr_legs.build(
                num_worlds=1,
                solver_name="phoenx",
                substeps=80,
                solver_iterations=8,
                velocity_iterations=1,
                animation=True,
                armature=0.001,
                step_layout="multi_world",
                prepare_refresh_stride="auto",
            )
        except ImportError as e:
            raise unittest.SkipTest(f"DR Legs USD dependencies unavailable: {e}") from e

        model, state = _recover_model_state(handle)
        pelvis = list(model.body_label).index(_PELVIS_LABEL)

        samples_xy = []
        min_up_z = 1.0
        min_body_z = float("inf")
        max_abs_body_speed = 0.0
        max_abs_joint_speed = 0.0

        for frame in range(_SAMPLE_FRAMES[-1] + 1):
            if frame > 0:
                handle.simulate_one_frame()

            body_q = state.body_q.numpy()
            body_qd = state.body_qd.numpy()
            joint_qd = state.joint_qd.numpy()

            self.assertTrue(np.isfinite(body_q).all(), f"body_q became non-finite at frame {frame}")
            self.assertTrue(np.isfinite(body_qd).all(), f"body_qd became non-finite at frame {frame}")
            self.assertTrue(np.isfinite(joint_qd).all(), f"joint_qd became non-finite at frame {frame}")

            pelvis_q = body_q[pelvis]
            min_up_z = min(min_up_z, _quat_up_z_xyzw(pelvis_q[3:7]))
            min_body_z = min(min_body_z, float(np.min(body_q[:, 2])))
            max_abs_body_speed = max(max_abs_body_speed, float(np.max(np.abs(body_qd[:, :3]))))
            max_abs_joint_speed = max(max_abs_joint_speed, float(np.max(np.abs(joint_qd))))

            if frame in _SAMPLE_FRAMES:
                samples_xy.append(pelvis_q[:2].copy())

        samples_xy_np = np.asarray(samples_xy, dtype=np.float32)
        xy_error = np.linalg.norm(samples_xy_np - _REFERENCE_PELVIS_XY, axis=1)
        displacement = samples_xy_np[-1] - samples_xy_np[0]

        self.assertGreater(float(displacement[0]), 0.32, "DR Legs did not walk forward far enough")
        self.assertLess(abs(float(displacement[1])), 0.20, "DR Legs lateral drifted too far while walking")
        self.assertLess(float(np.max(xy_error)), 0.12, "DR Legs pelvis trajectory drifted from the walking reference")
        self.assertGreater(min_up_z, 0.85, "DR Legs pelvis tipped too far from upright")
        self.assertGreater(min_body_z, -0.02, "DR Legs body penetrated/fell below the ground tolerance")
        self.assertLess(max_abs_body_speed, 2.0, "DR Legs developed excessive body linear speed")
        self.assertLess(max_abs_joint_speed, 20.0, "DR Legs developed excessive joint speed")


if __name__ == "__main__":
    unittest.main()
