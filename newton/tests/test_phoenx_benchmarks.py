# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PhoenX comparison benchmark validation."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.runner import SceneHandle, run_one
from newton._src.solvers.phoenx.benchmarks.scenarios.feather_pgs_common import _validate_simulation_state


class _FakeArray:
    def __init__(self, values):
        self._values = np.asarray(values)

    def numpy(self):
        return self._values.copy()


class _FakeState:
    def __init__(self, *, body_q, body_qd):
        self.joint_q = _FakeArray([0.0])
        self.joint_qd = _FakeArray([0.0])
        self.body_q = _FakeArray(body_q)
        self.body_qd = _FakeArray(body_qd)


class _FakeContacts:
    def __init__(self, count):
        self.rigid_contact_count = _FakeArray([count])


class TestPhoenXBenchmarkValidation(unittest.TestCase):
    def test_validate_simulation_state_accepts_useful_work(self):
        """Accept finite motion with normalized rotations and contacts."""
        initial_body_q = np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
        state = _FakeState(
            body_q=[[0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0]],
            body_qd=[[0.0, 0.0, 0.0, 0.1, 0.0, -0.1]],
        )

        metrics = _validate_simulation_state(state, initial_body_q, _FakeContacts(2))

        self.assertGreater(metrics["validation_max_translation_m"], 0.0)
        self.assertEqual(metrics["validation_rigid_contact_count"], 2)

    def test_validate_simulation_state_rejects_no_motion(self):
        """Reject a benchmark whose rigid bodies did not advance."""
        initial_body_q = np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
        state = _FakeState(
            body_q=initial_body_q,
            body_qd=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )

        with self.assertRaisesRegex(RuntimeError, "did not move"):
            _validate_simulation_state(state, initial_body_q, _FakeContacts(1))

    def test_run_one_calls_validation_after_measurement(self):
        """Run the validation callback once after all measured frames."""
        events = []

        def simulate():
            events.append("simulate")

        def validate():
            events.append("validate")
            return {"validation_marker": 1}

        handle = SceneHandle(
            name="test",
            solver_name="test",
            num_worlds=1,
            substeps=1,
            solver_iterations=1,
            simulate_one_frame=simulate,
            validate=validate,
        )
        with wp.ScopedDevice("cpu"):
            result = run_one(handle, warmup_frames=1, measure_frames=2)

        self.assertEqual(events, ["simulate", "simulate", "simulate", "validate"])
        self.assertEqual(result["validation_marker"], 1)


if __name__ == "__main__":
    unittest.main()
