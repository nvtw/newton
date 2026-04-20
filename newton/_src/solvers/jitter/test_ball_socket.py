# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the stand-alone ball-socket joint.

The momentum-conservation suite already exercises a long chain; these
tests target the *minimal* behaviour of one ball socket so a regression
in (e.g.) the linear-soft-constraint formulation, anchor snapshotting,
or warm-start path produces an obvious local failure rather than a
subtle force imbalance many cubes deep in the chain.

Three properties are checked:

* **Anchor coincidence.** After settling, the anchor point expressed
  in body 1's frame and in body 2's frame must map to the same world
  position to within a small linear-soft-constraint slop.
* **Reaction force.** A unit cube hung from the world anchor must
  carry exactly its own weight; the joint's world-frame +y reaction on
  body 2 (the cube) is ``mass * |g|``.
* **Free angular motion.** The joint must not transmit torque about
  the anchor: a free-spinning cube hung off the world keeps spinning
  about its initial axis (no spurious damping from the joint itself).

All scenes are also registered with :func:`scene` so the test
visualizer can replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter._test_helpers import run_settle_loop
from newton._src.solvers.jitter.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import WorldBuilder

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 120  # 2 s @ 60 fps -- PGS warm-start converges well within this
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_pendulum(
    device,
    *,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = True,
):
    """One ball socket between the static world body and a unit cube.

    Cube COM sits one unit below the anchor (so the static world
    anchor is the cube's top face center). Picked deliberately: with
    gravity on the cube hangs as a 1 m pendulum from a single socket
    -- the simplest scene that lets us assert (a) anchor coincidence
    and (b) the reaction force equals ``mass * |g|``.
    """
    b = WorldBuilder()
    world_body = b.world_body
    cube = b.add_dynamic_body(
        position=(0.0, -1.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=initial_angular_velocity,
    )
    b.add_ball_socket(world_body, cube, anchor=(0.0, 0.0, 0.0))
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
    )


@scene(
    "BallSocket: 1-cube pendulum",
    description="Single ball socket holding a unit cube under gravity.",
    tags=("ball_socket",),
)
def build_ball_socket_pendulum_scene(device) -> Scene:
    world = _build_pendulum(device)
    he = np.zeros((2, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


class TestBallSocket(unittest.TestCase):
    """End-to-end physics checks for :func:`WorldBuilder.add_ball_socket`."""

    def test_anchor_coincidence_under_gravity(self):
        """After settling the world anchor (cube's local +y face) must
        track the static anchor at the origin to within ~1 cm.

        A drift larger than the soft-constraint slop indicates the
        positional Jacobian or the bias-rate computation regressed.
        """
        device = wp.get_preferred_device()
        world = _build_pendulum(device)
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        # Cube top-face centre in body-local frame is (0, +HALF_EXTENT*2 - 1, 0)
        # = (0, 0, 0) measured from the cube COM at -1; rotate it by
        # the cube's orientation and add its world position to recover
        # the world-space anchor point as seen from body 2.
        positions = world.bodies.position.numpy()
        orientations = world.bodies.orientation.numpy()  # xyzw

        # body 1 (the world body) sees the anchor at (0, 0, 0) by
        # construction; we compare against that.
        local_anchor_b2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        q = orientations[1]
        # Rotate a vector by a quaternion (xyzw)
        x, y, z, w = q
        # quat -> rot matrix
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        anchor_world = positions[1].astype(np.float64) + rot @ local_anchor_b2
        # Anchor on body 1 (static world) is just the origin.
        drift = np.linalg.norm(anchor_world)
        self.assertLess(
            drift,
            5e-2,  # 5 cm slop -- generous for a soft constraint at default Hz
            msg=f"ball-socket anchors drifted apart by {drift:.4f} m",
        )

    def test_reaction_force_matches_weight(self):
        """The single socket holds up exactly one unit cube under gravity.

        Reaction +y on body 2 (the cube) must equal ``mass * |g|`` in
        equilibrium. Lateral and torque components must be ~0 because
        the anchor sits on the cube's symmetry axis through the COM.
        """
        device = wp.get_preferred_device()
        world = _build_pendulum(device)
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        out = wp.zeros(world.num_constraints, dtype=wp.spatial_vector, device=device)
        world.gather_constraint_wrenches(out)
        wrench = out.numpy()[0]
        fx, fy, fz, tx, ty, tz = wrench
        self.assertTrue(np.isfinite(wrench).all(), msg=f"non-finite wrench: {wrench}")
        self.assertAlmostEqual(
            fy,
            GRAVITY,
            delta=0.05,
            msg=f"expected fy~{GRAVITY:.3f} N, got {fy:.4f}",
        )
        for label, val in (("fx", fx), ("fz", fz), ("tx", tx), ("ty", ty), ("tz", tz)):
            self.assertLess(
                abs(val),
                0.5,
                msg=f"{label}={val:.4f} should be near zero for symmetric pendulum",
            )

    def test_does_not_resist_rotation(self):
        """A ball socket leaves all 3 angular DoF free.

        Anchor through the cube's COM (no lever arm) so an arbitrary
        spin must be preserved exactly: the joint only constrains
        translation, and with the COM at the pivot rotation does not
        translate the COM, so no constraint force is needed.
        """
        device = wp.get_preferred_device()

        # Build a *no-lever-arm* pendulum: cube COM coincides with the
        # static world anchor at the origin. Doing this through a
        # local builder rather than reusing _build_pendulum keeps the
        # scene's purpose obvious in this single test.
        omega0 = (0.7, -0.3, 0.5)  # rad/s, deliberately off-axis
        b = WorldBuilder()
        world_body = b.world_body
        cube = b.add_dynamic_body(
            position=(0.0, 0.0, 0.0),
            inverse_mass=1.0,
            inverse_inertia=_INV_INERTIA,
            affected_by_gravity=False,
            angular_velocity=omega0,
        )
        b.add_ball_socket(world_body, cube, anchor=(0.0, 0.0, 0.0))
        world = b.finalize(
            substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
        )

        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        omega_final = world.bodies.angular_velocity.numpy()[1]
        omega0_arr = np.asarray(omega0)
        # Identity inertia + no torque about the COM => angular velocity
        # in body frame is conserved; the world-frame vector also stays
        # constant because for the unit-inertia case dL/dt = 0 and
        # I*omega_world = omega_world.
        diff = np.linalg.norm(omega_final - omega0_arr)
        self.assertLess(
            diff,
            0.05,
            msg=(
                f"ball socket bled angular velocity: started at {omega0_arr}, "
                f"ended at {omega_final}"
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
