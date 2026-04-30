# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Energy conservation tests.

A frictionless pendulum with no drive is a closed Hamiltonian system:
total mechanical energy ``KE + PE`` is conserved. PGS solvers are
known to drift slowly (typically 1-3% per period), but a broken
solver dissipates much faster (warmup-bias leak, over-damped
position correction, mis-tuned soft constraints).

This test runs a single-pendulum swing for ~10 s and asserts the
total mechanical energy stays within +/-5% of its initial value.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.scene_registry import Scene, scene
from newton._src.solvers.phoenx.tests._test_helpers import STEP_LAYOUTS
from newton._src.solvers.phoenx.world_builder import (
    JointMode,
    WorldBuilder,
)

GRAVITY = 9.81
LEVER = 1.0
MASS = 1.0
INV_MASS = 1.0 / MASS
# Inertia about COM for a unit cube of half-extent 0.05; small relative
# to the m*L^2 = 1 kg*m^2 lever-arm contribution, so the pendulum is
# essentially a point mass on a string -- T = 2*pi*sqrt(L/g).
HE = 0.05
_I_COM = MASS / 6.0 * (2 * HE) * (2 * HE)
INV_INERTIA = ((1.0 / _I_COM, 0.0, 0.0), (0.0, 1.0 / _I_COM, 0.0), (0.0, 0.0, 1.0 / _I_COM))
INITIAL_ANGLE = 0.3  # rad, ~17 deg from straight-down
FPS = 240
SUBSTEPS = 8
SOLVER_ITERATIONS = 32  # generous so we observe genuine energy drift, not PGS noise


def _build_swinging_pendulum(device, *, step_layout: str = "multi_world"):
    """Frictionless pendulum: world anchor + 1 m revolute joint about
    +z axis, with the bob at ``(LEVER * sin(angle), -LEVER * cos(angle), 0)``."""
    b = WorldBuilder()
    anchor = b.world_body
    bx = LEVER * math.sin(INITIAL_ANGLE)
    by = -LEVER * math.cos(INITIAL_ANGLE)
    bob = b.add_dynamic_body(
        position=(bx, by, 0.0),
        inverse_mass=INV_MASS,
        inverse_inertia=INV_INERTIA,
        affected_by_gravity=True,
    )
    b.add_joint(
        body1=anchor,
        body2=bob,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 1.0),  # +z hinge axis
        mode=JointMode.REVOLUTE,
    )
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, -GRAVITY, 0.0),
        step_layout=step_layout,
        device=device,
    )


@scene(
    "Energy: frictionless pendulum",
    description=(
        "1 m frictionless pendulum, no drive, initial 0.3 rad deflection. "
        "Total mechanical energy KE+PE should stay constant within solver slop."
    ),
    tags=("energy", "conservation"),
)
def build_swinging_pendulum_scene(device) -> Scene:
    world = _build_swinging_pendulum(device)
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HE
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestEnergyConservation(unittest.TestCase):
    """Total mechanical energy of an undamped pendulum must stay
    constant within solver-tolerance bounds over many oscillations.
    """

    def test_pendulum_energy_drifts_under_5_percent(self) -> None:
        device = wp.get_preferred_device()
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                world = _build_swinging_pendulum(device, step_layout=layout)
                dt = 1.0 / FPS
                # ~10 s simulation at 240 Hz -- many periods of a 1 m
                # pendulum (T ~= 2 s).
                n_frames = int(10.0 * FPS)

                # Initial PE: m*g*h, h measured up from bob's lowest
                # point (straight-down, y = -LEVER).
                positions = world.bodies.position.numpy()
                velocities = world.bodies.velocity.numpy()
                omegas = world.bodies.angular_velocity.numpy()
                bob_pos_0 = positions[1]
                h0 = float(bob_pos_0[1]) - (-LEVER)  # height above lowest point
                v_lin = velocities[1]
                v_ang = omegas[1]
                ke0 = 0.5 * MASS * float(np.dot(v_lin, v_lin))
                ke0 += 0.5 * _I_COM * float(np.dot(v_ang, v_ang))
                pe0 = MASS * GRAVITY * h0
                e0 = ke0 + pe0

                # Capture a single step into a CUDA graph after a
                # warm-up; replay it for the remaining frames.
                world.step(dt)
                with wp.ScopedCapture(device=device) as cap:
                    world.step(dt)
                graph = cap.graph

                # Frame 0 was the warm-up step; frame 1 was the captured
                # step; replay frames 2..n_frames-1.
                for _ in range(n_frames - 2):
                    wp.capture_launch(graph)
                positions = world.bodies.position.numpy()
                velocities = world.bodies.velocity.numpy()
                omegas = world.bodies.angular_velocity.numpy()

                bob_pos_f = positions[1]
                h_f = float(bob_pos_f[1]) - (-LEVER)
                v_lin_f = velocities[1]
                v_ang_f = omegas[1]
                ke_f = 0.5 * MASS * float(np.dot(v_lin_f, v_lin_f))
                ke_f += 0.5 * _I_COM * float(np.dot(v_ang_f, v_ang_f))
                pe_f = MASS * GRAVITY * h_f
                e_f = ke_f + pe_f

                # No NaN.
                self.assertTrue(math.isfinite(e_f), msg=f"final energy non-finite: {e_f}")

                # Energy drift bounded -- 5% max deviation over ~10 s
                # (~5 periods). PGS without explicit projection drifts
                # slowly; we expect a few % drop typical, +-5% catches
                # outright dissipation regression.
                rel_err = abs(e_f - e0) / e0
                self.assertLess(
                    rel_err,
                    0.05,
                    msg=f"E_initial={e0:.4f} J, E_final={e_f:.4f} J, drift {rel_err * 100:.2f}% > 5%",
                )

                # Also: at the end of run the pendulum should still be
                # swinging (not damped to zero). Linear speed > 5% of
                # initial peak (which was sqrt(2*g*h0)).
                peak_v = math.sqrt(2.0 * GRAVITY * h0)
                v_speed = float(np.linalg.norm(v_lin_f))
                # At an arbitrary final-frame time the bob may be near a
                # turning point (small v), so use the angular velocity
                # (which doesn't have a phase equivalent to v_lin).
                v_ang_speed = float(np.linalg.norm(v_ang_f))
                # Peak omega = peak_v / LEVER.
                peak_omega = peak_v / LEVER
                # Sum-of-squares speed (phase invariant) should still
                # carry most of the swing energy.
                speed2 = (v_speed / LEVER) ** 2 + v_ang_speed**2
                rms_omega = math.sqrt(speed2 / 2.0)  # rms across linear and angular contributions
                # If pendulum lost all its swing energy this drops to 0.
                self.assertGreater(
                    rms_omega,
                    0.3 * peak_omega,
                    msg=f"end-of-run pendulum almost stopped: rms_omega={rms_omega:.4f}, "
                    f"peak_omega={peak_omega:.4f} -- spurious damping",
                )


if __name__ == "__main__":
    wp.init()
    unittest.main()
