# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end correctness test for ``World.gather_constraint_wrenches``.

Reuses the *equilibrium* hanging-chain scene from
:mod:`example_body_chain` (10 unit cubes rotated 45° about z so the
chain hangs straight down from the world anchor along -y) and lets the
solver settle for enough frames that the PGS warm-start has pumped the
per-substep impulse up to the value that exactly cancels gravity. Once
settled, every ball-socket force is known analytically: socket *k*
(joining cube ``k-1`` to cube ``k``, with cube -1 == world anchor) must
hold up cubes ``k..NUM_CUBES-1``, so its world-frame +y reaction force
on body2 is::

    fy(k) = (NUM_CUBES - k) * |gravity|

Lateral (x, z) forces and the moment about the COMs should both be ~0
because the joint anchors sit on the y axis through every cube's centre
of mass.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests._test_helpers import run_settle_loop
from newton._src.solvers.jitter.examples.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import WorldBuilder

# ---------------------------------------------------------------------------
# Scene constants -- mirror example_body_chain.py exactly.
# ---------------------------------------------------------------------------

NUM_CUBES = 10
HALF_EXTENT = 0.5
_DIAGONAL_HALF = HALF_EXTENT * math.sqrt(2.0)
_HALF_ANGLE = math.pi / 8.0  # half of 45 degrees
_DIAGONAL_QUAT = (0.0, 0.0, math.sin(_HALF_ANGLE), math.cos(_HALF_ANGLE))
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 20  # matches the smoke test that achieved <0.02 N error
SETTLE_FRAMES = 150  # ~2.5 s of sim, plenty for PGS warm-start to converge


def _build_equilibrium_chain(device):
    """Construct the same scene example_body_chain builds when its
    ``START_AT_EQUILIBRIUM`` toggle is on. Returns the assembled
    :class:`World`."""
    b = WorldBuilder()
    world_body = b.world_body  # static anchor at origin

    cube_ids: list[int] = []
    for j in range(NUM_CUBES):
        cube_ids.append(
            b.add_dynamic_body(
                position=(0.0, -(2 * j + 1) * _DIAGONAL_HALF, 0.0),
                orientation=_DIAGONAL_QUAT,
                inverse_mass=1.0,
                inverse_inertia=_INV_INERTIA,
            )
        )

    # Joints in cid order: k=0 anchors cube 0 to the world; k>=1 connects
    # cube k-1 to cube k. Anchor sits on the y axis at -k * 2 * h*sqrt(2),
    # which is exactly the meeting corner between consecutive 45°-rotated
    # cubes.
    for k in range(NUM_CUBES):
        body_a = world_body if k == 0 else cube_ids[k - 1]
        body_b = cube_ids[k]
        anchor = (0.0, -k * 2.0 * _DIAGONAL_HALF, 0.0)
        b.add_ball_socket(body_a, body_b, anchor)

    return b.finalize(
        enable_all_constraints=True,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
    )


@scene(
    "Momentum: hanging chain (10 cubes, ball sockets)",
    description=(
        "Static equilibrium scene from test_momentum_conservation: "
        "10 unit cubes rotated 45° about z, hanging straight down "
        "from a world anchor through corner-to-corner ball sockets."
    ),
    tags=("momentum", "ball_socket"),
)
def build_equilibrium_chain_scene(device) -> Scene:
    """Visualizer-facing wrapper that returns a :class:`Scene`.

    Reuses :func:`_build_equilibrium_chain` so the test and the
    visualizer always render identical geometry; if the test scene is
    ever tweaked the visualization follows along automatically.
    """
    world = _build_equilibrium_chain(device)
    # Body 0 is the static world anchor (kept invisible / non-pickable
    # via zero half-extents); bodies 1..NUM_CUBES are unit cubes.
    half_extents = np.zeros((NUM_CUBES + 1, 3), dtype=np.float32)
    half_extents[1:] = HALF_EXTENT
    return Scene(
        world=world,
        body_half_extents=half_extents,
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
        description="Hanging chain of 10 cubes -- analytic-equilibrium pose.",
    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestMomentumConservation(unittest.TestCase):
    """Validate that ``gather_constraint_wrenches`` reports the analytic
    equilibrium reaction forces for the body-chain example."""

    def test_hanging_chain_reaction_forces(self):
        device = wp.get_preferred_device()
        world = _build_equilibrium_chain(device)

        dt = 1.0 / FPS
        run_settle_loop(world, SETTLE_FRAMES, dt=dt)

        out = wp.zeros(NUM_CUBES, dtype=wp.spatial_vector, device=device)
        world.gather_constraint_wrenches(out)
        wrenches = out.numpy()

        # Loose tolerance: 0.5% of the largest expected force, which keeps
        # the test robust to small PGS residuals while still catching real
        # regressions (e.g. wrong dt, wrong sign convention).
        f_tol = 0.005 * NUM_CUBES * GRAVITY  # ~0.49 N
        # Lateral forces and torques should be at machine zero modulo
        # solver noise; allow a generous absolute slack for FP error.
        lateral_tol = 1.0  # N
        torque_tol = 1.0  # N·m

        for cid in range(NUM_CUBES):
            fx, fy, fz, tx, ty, tz = wrenches[cid]
            expected_fy = (NUM_CUBES - cid) * GRAVITY

            self.assertTrue(
                np.isfinite(wrenches[cid]).all(),
                msg=f"cid {cid} produced non-finite wrench: {wrenches[cid]}",
            )
            self.assertAlmostEqual(
                fy,
                expected_fy,
                delta=f_tol,
                msg=(
                    f"cid {cid}: fy={fy:.4f} N, expected ~{expected_fy:.4f} N "
                    f"(weight of {NUM_CUBES - cid} cubes below)"
                ),
            )
            self.assertLess(
                abs(fx),
                lateral_tol,
                msg=f"cid {cid}: |fx|={abs(fx):.4f} N too large for vertical chain",
            )
            self.assertLess(
                abs(fz),
                lateral_tol,
                msg=f"cid {cid}: |fz|={abs(fz):.4f} N too large for vertical chain",
            )
            self.assertLess(
                abs(tx),
                torque_tol,
                msg=f"cid {cid}: |tx|={abs(tx):.4f} N·m -- joint should pass through COM",
            )
            self.assertLess(
                abs(ty),
                torque_tol,
                msg=f"cid {cid}: |ty|={abs(ty):.4f} N·m -- joint should pass through COM",
            )
            self.assertLess(
                abs(tz),
                torque_tol,
                msg=f"cid {cid}: |tz|={abs(tz):.4f} N·m -- joint should pass through COM",
            )


if __name__ == "__main__":
    wp.init()
    unittest.main()
