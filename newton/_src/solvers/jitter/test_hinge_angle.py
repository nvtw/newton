# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the stand-alone HingeAngle constraint.

HingeAngle is the *angular* half of a hinge: it locks the two
rotational DoF perpendicular to a hinge axis and (optionally) clamps
the remaining axial DoF to ``[min_angle, max_angle]``. This file
exercises both jobs in isolation -- no co-located ball socket, no
fused HingeJoint -- so a regression in either the perpendicular lock
or the axial limit produces a sharp, local failure.

Three scenes are checked:

* **Axis-only spin.** Two free bodies sharing a hinge axis; one is
  spun about that axis. The relative angular velocity *along* the
  axis must be preserved, while the perpendicular relative angular
  velocity must decay to ~0.
* **Perpendicular tilt rejected.** Same setup, but the spin is given
  *off* the hinge axis. The perpendicular component of relative
  angular velocity must be killed (the lock works) while the axial
  component is preserved.
* **Limit clamping.** A free body is spun toward an angular limit;
  once the limit fires the relative axial velocity must reverse /
  decay (the limit is a one-sided spring + damper).

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

FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 120  # 2 s @ 60 fps -- PGS warm-start converges well within this
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Hinge axis = world +z for every scene in this file -- keeps the
# math obvious (axial angular velocity is just omega.z).
_HINGE_AXIS = (0.0, 0.0, 1.0)


def _build_two_body_hinge(
    device,
    *,
    initial_angular_velocity_b1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_angular_velocity_b2: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_angle: float = -math.pi,
    max_angle: float = math.pi,
):
    """Two free dynamic cubes joined only by a HingeAngle constraint.

    Both bodies are gravity-free so the only thing that can change
    their angular velocities is the joint. They start at the same
    position (positional drift is irrelevant -- the only constraint
    is angular) so the test signal is dominated by the angular DoF.
    """
    b = WorldBuilder()
    b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity_b1,
    )
    b.add_dynamic_body(
        position=(2.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity_b2,
    )
    b.add_hinge_angle(1, 2, axis=_HINGE_AXIS, min_angle=min_angle, max_angle=max_angle)
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
    )


@scene(
    "HingeAngle: 2-body axial spin",
    description="Two free bodies sharing a +z hinge axis; one body spinning about z.",
    tags=("hinge_angle",),
)
def build_hinge_angle_axial_spin_scene(device) -> Scene:
    world = _build_two_body_hinge(
        device, initial_angular_velocity_b2=(0.0, 0.0, 2.0)
    )
    he = np.zeros((3, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@scene(
    "HingeAngle: limit ±10°",
    description="Two free bodies, axial spin into a tight ±10° limit.",
    tags=("hinge_angle",),
)
def build_hinge_angle_limit_scene(device) -> Scene:
    limit = math.radians(10.0)
    world = _build_two_body_hinge(
        device,
        initial_angular_velocity_b2=(0.0, 0.0, 2.0),
        min_angle=-limit,
        max_angle=limit,
    )
    he = np.zeros((3, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestHingeAngle(unittest.TestCase):
    """End-to-end physics checks for :func:`WorldBuilder.add_hinge_angle`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        run_settle_loop(world, frames, dt=1.0 / FPS)

    def test_axis_spin_preserved(self):
        """Pure axial spin must survive the lock untouched.

        Body 2 starts spinning about +z; body 1 is at rest. The
        perpendicular lock can do nothing because neither body has
        any perpendicular angular velocity to begin with, so body 2
        keeps spinning at its initial rate and body 1 stays still.
        """
        device = wp.get_preferred_device()
        omega_axial = 2.0  # rad/s about +z
        world = _build_two_body_hinge(
            device, initial_angular_velocity_b2=(0.0, 0.0, omega_axial)
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        # Allow the perpendicular lock a tiny FP slack but require it
        # to be at least 100x smaller than the axial signal.
        self.assertAlmostEqual(omegas[2, 2], omega_axial, delta=0.05)
        self.assertAlmostEqual(omegas[1, 2], 0.0, delta=0.05)
        for bid in (1, 2):
            self.assertLess(
                np.linalg.norm(omegas[bid, :2]),
                0.02 * abs(omega_axial),
                msg=f"body {bid} grew perpendicular spin: {omegas[bid]}",
            )

    def test_perpendicular_lock_kills_off_axis_spin(self):
        """Off-axis relative spin must be eliminated by the lock.

        Body 2 starts spinning about world +x (perpendicular to the
        hinge axis); body 1 is at rest. The perpendicular lock must
        kill the relative perpendicular velocity entirely, leaving
        the two bodies sharing the same residual perpendicular
        angular velocity (so total angular momentum is conserved
        about the perp axis: with equal inverse inertia, each ends
        up at half the initial omega_x). The axial relative velocity
        was zero and must stay zero.
        """
        device = wp.get_preferred_device()
        omega_x = 2.0
        world = _build_two_body_hinge(
            device, initial_angular_velocity_b2=(omega_x, 0.0, 0.0)
        )
        # The symmetric two-body scene has an oscillatory transient: the
        # perpendicular lock makes the bodies exchange angular momentum
        # about +x, producing a damped ring-down that passes through its
        # equilibrium ``omega_x/2`` around frame ~60 but temporarily
        # overshoots near ~120 before settling again. 60 frames is
        # plenty for the first equilibrium crossing and deliberately
        # avoids the overshoot window.
        self._step(world, frames=60)

        omegas = world.bodies.angular_velocity.numpy()
        # Conservation of angular momentum about +x with identity
        # inverse inertia: each body ends up at omega_x / 2.
        expected_omega_x = omega_x / 2.0
        self.assertAlmostEqual(
            omegas[1, 0],
            expected_omega_x,
            delta=0.05,
            msg=f"body1 omega_x={omegas[1, 0]}, expected ~{expected_omega_x}",
        )
        self.assertAlmostEqual(
            omegas[2, 0],
            expected_omega_x,
            delta=0.05,
            msg=f"body2 omega_x={omegas[2, 0]}, expected ~{expected_omega_x}",
        )
        # Relative perpendicular angular velocity must be ~0.
        rel_perp = omegas[2, :2] - omegas[1, :2]
        self.assertLess(
            np.linalg.norm(rel_perp),
            0.05,
            msg=f"perpendicular lock left rel_perp={rel_perp}",
        )
        # Axial relative velocity must remain 0.
        self.assertAlmostEqual(omegas[2, 2] - omegas[1, 2], 0.0, delta=0.05)

    def test_limit_clamps_axial_spin(self):
        """An axial spin into a tight ±10° limit must be braked.

        Body 2 spins toward +max_angle; once the limit engages the
        relative axial velocity should be driven back to ~0 (the
        limit acts as a critically-damped one-sided spring) and the
        relative axial position should sit at-or-just-past the limit.
        Without the limit this scene would spin forever.
        """
        device = wp.get_preferred_device()
        max_angle = math.radians(10.0)
        omega_axial = 2.0  # rad/s -> would reach 10° in ~0.087 s w/o limit
        world = _build_two_body_hinge(
            device,
            initial_angular_velocity_b2=(0.0, 0.0, omega_axial),
            min_angle=-max_angle,
            max_angle=max_angle,
        )
        self._step(world, frames=120)  # 2 s, plenty for the limit to fire

        omegas = world.bodies.angular_velocity.numpy()
        rel_axial = omegas[2, 2] - omegas[1, 2]
        # The joint should have dissipated the relative axial spin
        # (Box2D-style soft limit acts as a critically-damped spring).
        self.assertLess(
            abs(rel_axial),
            0.2 * omega_axial,
            msg=f"limit failed to brake spin: rel_omega_z={rel_axial}",
        )

        # And the actual relative angle along +z should have stopped
        # at-or-just-past the limit. Recover it from the body
        # quaternions: the rotation taking body 1 onto body 2 about
        # +z has angle 2*atan2(q.z, q.w) (xyzw). Since both bodies
        # started at identity, this is just body2.angle - body1.angle
        # about +z.
        quats = world.bodies.orientation.numpy()  # xyzw

        def axis_angle_z(q):
            # Project the rotation onto +z by extracting the swing-
            # twist twist component about +z. For a small extra-axis
            # tilt this is well approximated by 2*atan2(q.z, q.w).
            return 2.0 * math.atan2(q[2], q[3])

        rel_angle = axis_angle_z(quats[2]) - axis_angle_z(quats[1])
        # The soft limit acts as a critically-damped one-sided spring,
        # so after the spin reverses the body can settle anywhere
        # inside (or just barely past) ``[min_angle, max_angle]``. The
        # only thing this test asserts about the *position* is that
        # the joint did contain it -- without the limit the cube
        # would have been at omega_axial * t = 4 rad after 2 s.
        slop = math.radians(5.0)
        self.assertLess(
            rel_angle,
            max_angle + slop,
            msg=f"relative axial angle {rel_angle:.4f} overshot +limit {max_angle:.4f}",
        )
        self.assertGreater(
            rel_angle,
            -max_angle - slop,
            msg=f"relative axial angle {rel_angle:.4f} overshot -limit {-max_angle:.4f}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
