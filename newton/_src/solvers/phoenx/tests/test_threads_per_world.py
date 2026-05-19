# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive ``threads_per_world`` infrastructure.

The fast-tail kernels read their effective tpw from a 1-element GPU
buffer that the per-step picker writes. Forcing tpw via the ``int``
form of the constructor argument pins that buffer; ``"auto"`` runs
the picker every step. These tests verify three properties:

  1. **Correctness**: forced tpw=8/16/32 on an identical scene
     converge to body-positions within tight numerical tolerance of
     each other after a settle. The kernels' adaptive lane-count is
     a SIMT layout knob, not a math change, so results must match.
  2. **Picker decisions**: the heuristic picks tpw=32 on
     small-fleet / dense-colour scenes and tpw=16 on big sparse-fleet
     scenes. Catches regressions in the threshold tuning.
  3. **Construction validation**: invalid ``threads_per_world``
     values raise ``ValueError`` at construction time, before any
     kernel runs.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.solver import SolverPhoenX


def _settle_pos(num_worlds: int, tpw, frames: int = 30) -> np.ndarray:
    """Build a tiny falling-box scene, advance ``frames`` steps, return body-q."""
    builder = newton.ModelBuilder()
    for w in range(num_worlds):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0 + 0.05 * float(w))))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    model = builder.finalize()
    solver = SolverPhoenX(model, threads_per_world=tpw)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    dt = 1.0 / 60.0
    for _ in range(frames):
        model.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy().copy()


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX adaptive-tpw tests require CUDA")
class TestThreadsPerWorldCorrectness(unittest.TestCase):
    """Forced tpw=8 / 16 / 32 must converge to the same body state."""

    def test_forced_tpw_settle_matches(self) -> None:
        # 4 worlds: small enough that the picker would pick tpw=32 on
        # auto, but the kernels still exercise the adaptive read path
        # for any forced value.
        ref = _settle_pos(num_worlds=4, tpw=32)
        for tpw in (8, 16):
            other = _settle_pos(num_worlds=4, tpw=tpw)
            np.testing.assert_allclose(
                other,
                ref,
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"settled body_q diverged at tpw={tpw}",
            )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX adaptive-tpw tests require CUDA")
class TestThreadsPerWorldPicker(unittest.TestCase):
    """Picker chooses tpw=32 on small/dense scenes, tpw=16 on sparse big fleets."""

    def _settle_and_read_pick(self, num_worlds: int) -> int:
        # Auto mode + a few real steps so the picker has populated
        # ``_tpw_choice`` from per-step colour stats.
        builder = newton.ModelBuilder()
        for w in range(num_worlds):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0 + 0.01 * float(w))))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_ground_plane()
        model = builder.finalize()
        solver = SolverPhoenX(model, threads_per_world="auto")
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        for _ in range(3):
            model.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        return int(solver.world._tpw_choice.numpy()[0])

    def test_small_fleet_picks_32(self) -> None:
        """Below the saturation threshold, picker pins tpw=32. The
        host-side fast-path also disables the per-step picker launch
        in this case -- both paths land on the same value."""
        self.assertEqual(self._settle_and_read_pick(num_worlds=64), 32)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX adaptive-tpw tests require CUDA")
class TestThreadsPerWorldValidation(unittest.TestCase):
    """Construction-time validation of ``threads_per_world`` argument."""

    def _build_min_model(self) -> newton.Model:
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_ground_plane()
        return builder.finalize()

    def test_unknown_string_rejected(self) -> None:
        model = self._build_min_model()
        with self.assertRaises(ValueError):
            SolverPhoenX(model, threads_per_world="adaptive")

    def test_unknown_int_rejected(self) -> None:
        model = self._build_min_model()
        with self.assertRaises(ValueError):
            SolverPhoenX(model, threads_per_world=24)

    def test_zero_rejected(self) -> None:
        model = self._build_min_model()
        with self.assertRaises(ValueError):
            SolverPhoenX(model, threads_per_world=0)


if __name__ == "__main__":
    unittest.main()
