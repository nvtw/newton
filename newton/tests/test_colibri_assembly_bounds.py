# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check Colibri escape diagnostics independently of global assembly motion."""

import unittest
from types import SimpleNamespace

import numpy as np

from newton.examples.kamino.example_kamino_colibri import Example
from newton.examples.phoenx.example_phoenx_colibri import Example as PhoenxExample


def _array(value):
    return SimpleNamespace(numpy=lambda: np.asarray(value))


def _example():
    example = Example.__new__(Example)
    example.fix_base = False
    example.initial_q = np.array(
        [[0, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]], dtype=np.float32
    )
    example.model = SimpleNamespace(body_label=["FrameGround", "Frame", "Flower"])
    for name in ("joint_parent", "joint_child", "joint_X_p", "joint_X_c", "joint_type", "joint_axis", "joint_qd_start"):
        setattr(example.model, name, _array([]))
    example.state_0 = SimpleNamespace(body_q=_array(example.initial_q.copy()), body_qd=_array(np.zeros((3, 6))))
    return example


class TestColibriAssemblyBounds(unittest.TestCase):
    def test_phoenx_admission_default(self):
        """Colibri defaults to geometric admission and retains the legacy control."""
        parser = PhoenxExample.create_parser()
        self.assertTrue(parser.parse_args([]).geometric_candidates)
        self.assertFalse(parser.parse_args(["--velocity-filtered-candidates"]).geometric_candidates)

    def test_free_assembly_rigid_motion(self):
        """Global translation and rotation preserve a free assembly's escape result."""
        example = _example()
        q = example.initial_q.copy()
        # Translate two meters and rotate the assembly 90 degrees around Z.
        q[:2, :3] = np.array([[2, -1, 0.4], [2, -0.8, 0.4]])
        q[:2, 3:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        example.state_0.body_q = _array(q)
        example.test_post_step()

    def test_detached_body_rejected(self):
        """Relative body escape is rejected even if the base itself moves."""
        example = _example()
        q = example.initial_q.copy()
        q[:2, 0] += 2
        q[1, 1] += 0.6
        example.state_0.body_q = _array(q)
        with self.assertRaisesRegex(AssertionError, "escaped"):
            example.test_post_step()

    def test_fixed_base_motion_rejected(self):
        """Fixed-base diagnostics still reject movement smaller than the escape bound."""
        example = _example()
        example.fix_base = True
        q = example.initial_q.copy()
        q[0, 0] = 0.01
        example.state_0.body_q = _array(q)
        with self.assertRaises(AssertionError):
            example.test_post_step()


if __name__ == "__main__":
    unittest.main()
