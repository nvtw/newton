# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scale-independent hard-limit tests for direct prismatic structure."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81
LIMIT_LOWER = -0.05
LIMIT_UPPER = 0.05
FPS = 120
SUBSTEPS = 5
SETTLE_FRAMES = 240
INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _settle_slider(*, mass: float, limit_ke: float) -> tuple[float, float, newton.solvers.SolverPhoenX]:
    """Return a gravity-loaded slider coordinate, velocity, and solver."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY), up_axis=newton.Axis.Z)
    body = builder.add_link(xform=wp.transform_identity(), mass=mass, inertia=INERTIA)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        limit_lower=LIMIT_LOWER,
        limit_upper=LIMIT_UPPER,
        limit_ke=limit_ke,
        limit_kd=2.0,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=wp.get_preferred_device())
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=2,
        velocity_iterations=1,
        articulation_mode="maximal",
    )
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(SETTLE_FRAMES):
        wp.capture_launch(capture.graph)
    q = wp.zeros_like(model.joint_q)
    qd = wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, state, q, qd)
    return float(q.numpy()[0]), float(qd.numpy()[0]), solver


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX scale tests require CUDA graphs.")
class TestScaleLimitContract(unittest.TestCase):
    """Check Newton limit gains preserve the public rigid-stop contract."""

    def test_gravity_loaded_slider_stops_at_lower_limit(self) -> None:
        """Hold several masses and limit gains at the exact lower stop."""
        for mass, ke in ((1.0, 200.0), (1.0, 1000.0), (2.0, 5000.0)):
            with self.subTest(mass=mass, stiffness=ke):
                q, qd, solver = _settle_slider(mass=mass, limit_ke=ke)
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (5,))
                np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [1])
                self.assertLess(abs(qd), 1.0e-3)
                self.assertAlmostEqual(q, LIMIT_LOWER, delta=5.0e-4)

    def test_limit_gain_does_not_soften_public_stop(self) -> None:
        """Keep the public stop rigid for different XPBD gain metadata."""
        soft_q, _, _ = _settle_slider(mass=1.0, limit_ke=200.0)
        stiff_q, _, _ = _settle_slider(mass=1.0, limit_ke=5000.0)
        self.assertAlmostEqual(soft_q, stiff_q, delta=2.0e-4)
        self.assertAlmostEqual(stiff_q, LIMIT_LOWER, delta=5.0e-4)


if __name__ == "__main__":
    unittest.main()
