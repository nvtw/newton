# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression for high-friction humanoid contact relax stability."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX humanoid contact test requires CUDA")
class TestPhoenXHumanoidContactRelax(unittest.TestCase):
    def test_speculative_contact_relax_does_not_launch_humanoid(self) -> None:
        builder_humanoid = newton.ModelBuilder()
        builder_humanoid.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=["floor", "ground"],
            up_axis="Z",
            parse_sites=False,
            enable_self_collisions=False,
        )
        builder_humanoid.joint_q[:7] = [0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0]

        builder = newton.ModelBuilder()
        builder.add_world(builder_humanoid)
        builder.add_ground_plane()
        model = builder.finalize()

        state = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=50,
            solver_iterations=50,
            velocity_iterations=4,
            prepare_refresh_stride="auto",
        )
        contacts = model.contacts()

        for frame in range(80):
            model.collide(state, contacts)
            state.clear_forces()
            solver.step(state, state, control, contacts, 1.0 / 60.0)

            q = state.body_q.numpy()
            qd = state.body_qd.numpy()
            finite = np.isfinite(q).all() and np.isfinite(qd).all()
            max_qd = float(np.nanmax(np.abs(qd)))
            self.assertTrue(finite, f"non-finite humanoid state at frame {frame}")
            self.assertLess(max_qd, 200.0, f"humanoid launched at frame {frame}: max |qd|={max_qd:.3f}")


if __name__ == "__main__":
    unittest.main()
