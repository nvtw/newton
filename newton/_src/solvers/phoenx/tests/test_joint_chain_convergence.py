# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the unified D6 joint's coupled positional solve.

Both scenes are horizontal cantilever chains: ``NUM_LINKS`` unit cubes in
a column along world -y with gravity along -z, anchored at the root to the
static world body. The joint axes are chosen perpendicular to the gravity
bending plane, so gravity exerts no torque/force on any joint's *free*
DoF -- the chain is held straight purely by each joint's *locked* rows:

* REVOLUTE: hinge axis +z; the chain hangs from the anchor-2 swing
  (axis-alignment) lock.
* PRISMATIC: slide axis +x; the chain hangs from the locked transverse +
  rotational rows.

An ideally-converged solver keeps every link at ``z = 0``; the free-end
-z droop is therefore a direct measure of the joint lock's convergence.

These guard the coupled anchor-block Schur formulation. The throughput
experiment that replaced it with decoupled block-Gauss-Seidel (anchor-1
solved independently of anchor-2) collapses these chains by metres at the
low iteration counts used here -- this test fails loudly if that
decoupling ever returns.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests._test_helpers import STEP_LAYOUTS, run_settle_loop
from newton._src.solvers.phoenx.world_builder import JointMode, WorldBuilder

GRAVITY = 9.81
FPS = 120
SUBSTEPS = 8
# Deliberately low -- the coupled Schur holds the cantilever even here,
# while the decoupled block-Gauss-Seidel formulation collapses it.
SOLVER_ITERATIONS = 4
SETTLE_FRAMES = 120  # 1 s @ 120 fps
NUM_LINKS = 16
HALF_EXTENT = 0.5
PITCH = 2.0 * HALF_EXTENT  # links touch corner-to-corner along -y
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
# A converged cantilever barely droops (<2% of its length). The decoupled
# solver drooped it by tens of percent / metres; guard inside that gap.
_MAX_TIP_SAG = 0.02 * (NUM_LINKS * PITCH)


def _build_cantilever(device, mode: JointMode, *, step_layout: str = "multi_world"):
    """Cantilever chain of ``NUM_LINKS`` cubes along -y, gravity along -z.

    Joint ``k`` connects link ``k-1`` (or the static world body for the
    root) to link ``k`` at their shared boundary. The two anchors are
    offset along the joint axis: +z for REVOLUTE, +x for PRISMATIC -- both
    perpendicular to the gravity bending plane.
    """
    b = WorldBuilder()
    bodies = [b.world_body]
    for k in range(NUM_LINKS):
        bodies.append(
            b.add_dynamic_body(
                position=(0.0, -(k + 0.5) * PITCH, 0.0),
                inverse_mass=1.0,
                inverse_inertia=_INV_INERTIA,
            )
        )

    for k in range(NUM_LINKS):
        y = -k * PITCH  # boundary between link k-1 and link k
        if mode is JointMode.REVOLUTE:
            anchor1 = (0.0, y, -HALF_EXTENT)
            anchor2 = (0.0, y, HALF_EXTENT)
        else:  # PRISMATIC
            anchor1 = (-HALF_EXTENT, y, 0.0)
            anchor2 = (HALF_EXTENT, y, 0.0)
        b.add_joint(
            body1=bodies[k],
            body2=bodies[k + 1],
            anchor1=anchor1,
            anchor2=anchor2,
            mode=mode,
        )

    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, -GRAVITY),
        step_layout=step_layout,
        device=device,
    )


def _tip_sag(world) -> float:
    """Downward (-z) droop of the free-end link, in metres."""
    z = world.bodies.position.numpy()[1:, 2]
    return float(-z[-1])


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestJointChainConvergence(unittest.TestCase):
    """The coupled positional solve must hold a cantilever chain straight."""

    def test_revolute_cantilever_holds(self):
        device = wp.get_preferred_device()
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                world = _build_cantilever(device, JointMode.REVOLUTE, step_layout=layout)
                run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)
                positions = world.bodies.position.numpy()[1:]
                self.assertTrue(np.isfinite(positions).all(), "non-finite body position")
                sag = _tip_sag(world)
                self.assertLess(
                    sag,
                    _MAX_TIP_SAG,
                    msg=f"revolute cantilever drooped {sag * 1e3:.1f} mm "
                    f"(limit {_MAX_TIP_SAG * 1e3:.1f} mm) -- swing lock under-converged",
                )

    def test_prismatic_cantilever_holds(self):
        device = wp.get_preferred_device()
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                world = _build_cantilever(device, JointMode.PRISMATIC, step_layout=layout)
                run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)
                positions = world.bodies.position.numpy()[1:]
                self.assertTrue(np.isfinite(positions).all(), "non-finite body position")
                sag = _tip_sag(world)
                self.assertLess(
                    sag,
                    _MAX_TIP_SAG,
                    msg=f"prismatic cantilever drooped {sag * 1e3:.1f} mm "
                    f"(limit {_MAX_TIP_SAG * 1e3:.1f} mm) -- slider lock under-converged",
                )


if __name__ == "__main__":
    wp.init()
    unittest.main()
