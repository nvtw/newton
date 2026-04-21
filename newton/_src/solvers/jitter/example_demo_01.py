# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 01 -- Constraint Bridge
#
# Adapted from ``JitterDemo.Demos.Demo01`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo01.cs``). The C#
# original is split into a constraint bridge *and* a constraint-built
# car; our port reproduces only the bridge portion. The car uses a
# compound chassis (``TransformedShape``) plus ``ConstraintCar`` helper
# wheels built from multi-shape bodies -- both of which are outside
# the self-imposed "one shape per body, no compound shapes" scope of
# the Newton Jitter demo port.
#
# Topology (30-plank hinge bridge, identical to the C# source):
#
#   world <-- hinge --> plank 0 <-- hinge --> plank 1 <-- ... --> plank 29 <-- hinge --> world
#
# Each plank is a thin box (0.7 x 0.1 x 4 in Jitter's +Y-up axes, which
# becomes half-extents (0.35, 2.0, 0.05) in Newton's +Z-up axes). The
# hinge axis is Jitter's +Z, i.e. Newton's +Y -- perpendicular to the
# bridge's long span so the planks can sag under gravity. Both end
# hinges attach to :data:`WORLD_BODY` (the static Jitter world anchor).
#
# The C# demo deletes individual hinges whose impulse grows past
# ``0.5`` (the bridge breaks under extreme load); we leave that out
# because our ``WorldBuilder`` has no analogue of
# ``HingeJoint.Remove``. The resulting bridge is therefore
# indestructible, which is what we want for a headless regression
# test anyway.
#
# Run:  python -m newton._src.solvers.jitter.example_demo_01
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    WORLD_BODY,
    DemoConfig,
    DemoExample,
)
from newton._src.solvers.jitter.world_builder import DriveMode, JointMode

# Plank dimensions -- C# BoxShape(0.7, 0.1, 4) in Jitter's +Y-up axes,
# remapped to Newton's +Z-up half-extents (x, z, y) -> (HX, HY, HZ) =
# (0.35, 2.0, 0.05).
PLANK_HX = 0.35
PLANK_HY = 2.0
PLANK_HZ = 0.05

NUM_PLANKS = 30
# Centre-to-centre step between adjacent plank positions.
PLANK_STEP = 0.8
# Half-gap between plank edge and the hinge centre along +X:
# the C# demo uses 0.7 m for the end hinges (plank tip) and 0.1 m for
# the between-plank hinges (plank edge - 0.1 overlap).
PLANK_EDGE_END = 0.7
PLANK_EDGE_INNER = 0.1

# Start position (C# ``(-10, 8, -20)`` with +Y<->+Z swap).
START_X = -10.0
START_Y = -20.0
START_Z = 8.0

# Hinge axial damping. An implicit PD velocity-drive with
# ``target_velocity = 0`` and a small torque cap acts as a soft damper
# on the free axial rotation -- Box2D v3 / Solver2D style: damping
# lives inside the soft constraint, not as a per-body velocity decay.
# The torque cap is sized to bleed the swing over ~5 s without fighting
# the (much larger) gravity-driven lever torque on the bridge.
HINGE_AXIAL_DAMPING_TORQUE = 0.1  # [N*m]
# PD damping gain for the hinge velocity servo [N*m*s/rad]. Large
# enough that the effective time constant ``I_hinge / gain`` is far
# below the solver substep so the servo saturates against
# ``HINGE_AXIAL_DAMPING_TORQUE`` at any perceptible relative spin;
# the torque cap is the thing that actually bounds steady-state
# energy bleed, just like in the rigid-velocity-motor formulation
# this replaced.
HINGE_AXIAL_DAMPING_GAIN = 10.0  # [N*m*s/rad]


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Constraint Bridge",
            camera_pos=(6.0, -10.0, 14.0),
            camera_pitch=-20.0,
            camera_yaw=-90.0,
            fps=60,
            # C# Demo01 uses SubstepCount=4 and SolverIterations=(2,2);
            # our solver benefits from a bit more work on the 30-plank
            # hinge chain (which is long and rope-like) so we bump the
            # iteration count to 6 while keeping 4 substeps.
            substeps=4,
            solver_iterations=6,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # Centre each plank at START + (i * PLANK_STEP, 0, 0). The C#
        # code uses i * 0.8 unchanged; we just mirror that offset.
        self._planks: list[int] = []
        for i in range(NUM_PLANKS):
            cx = START_X + i * PLANK_STEP
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(cx, START_Y, START_Z), q=wp.quat_identity()
                ),
                mass=1.0,
            )
            mb.add_shape_box(body, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
            self.register_body_extent(body, (PLANK_HX, PLANK_HY, PLANK_HZ))
            self._planks.append(body)

        # Hinges between adjacent planks. Each plank is a long thin
        # box with its long axis along +Y (half-extent ``PLANK_HY``),
        # so the hinge line at ``X = ax`` runs through both planks from
        # ``(ax, START_Y - PLANK_HY, START_Z)`` to
        # ``(ax, START_Y + PLANK_HY, START_Z)``. We place the unified
        # joint's two ball-socket anchors at those two endpoints
        # instead of picking an arbitrary 1 m axis-direction offset:
        # anchor1 and anchor2 then sit at real "pin" locations on the
        # shared +Y edge of the two planks. Geometrically the hinge is
        # the same (the free DoF is still rotation about the line),
        # but the rank-5 Schur complement sees two positional locks
        # separated by the full plank length (2 * PLANK_HY = 4 m),
        # which gives the solver more informative lever arms than a
        # 1 m step would.
        #
        # X-anchor placement matches the C# source: the inner hinges
        # sit ``PLANK_EDGE_INNER`` inside plank ``i``'s -X edge
        # (giving a 0.1 m overlap with plank ``i-1``); the two end
        # hinges sit ``PLANK_EDGE_END`` past each plank's +-X tip,
        # anchored to :data:`WORLD_BODY`.
        hinge_y_lo = START_Y - PLANK_HY
        hinge_y_hi = START_Y + PLANK_HY
        # All hinges share the same axial velocity-drive damper
        # (target 0, small torque cap) so energy is dissipated inside
        # the revolute DoF rather than via per-body velocity decay
        # (Box2D v3 / Solver2D philosophy). Applied to both inter-plank
        # hinges *and* the two world-anchored end hinges: without
        # damping at the ends the whole chain's pendulum-like
        # rigid-body swing mode is unopposed.
        # PD velocity servo (``stiffness_drive = 0`` kills the spring;
        # ``damping_drive`` is the per-rad/s torque gain). The torque
        # cap bounds the steady-state impulse at large velocity
        # errors, matching the old rigid-velocity-motor behaviour that
        # this API used to fall through to.
        damper_kwargs = {
            "drive_mode": DriveMode.VELOCITY,
            "target_velocity": 0.0,
            "max_force_drive": HINGE_AXIAL_DAMPING_TORQUE,
            "stiffness_drive": 0.0,
            "damping_drive": HINGE_AXIAL_DAMPING_GAIN,
        }
        for i in range(1, NUM_PLANKS):
            ax = START_X + i * PLANK_STEP - PLANK_EDGE_INNER
            self.add_joint(
                body1=self._planks[i - 1],
                body2=self._planks[i],
                anchor1=(ax, hinge_y_lo, START_Z),
                anchor2=(ax, hinge_y_hi, START_Z),
                mode=JointMode.REVOLUTE,
                **damper_kwargs,
            )

        # End hinge at plank 0's -X tip.
        ax0 = START_X - PLANK_EDGE_END
        self.add_joint(
            body1=WORLD_BODY,
            body2=self._planks[0],
            anchor1=(ax0, hinge_y_lo, START_Z),
            anchor2=(ax0, hinge_y_hi, START_Z),
            mode=JointMode.REVOLUTE,
            **damper_kwargs,
        )
        # End hinge at plank N-1's +X tip.
        axN = START_X + (NUM_PLANKS - 1) * PLANK_STEP + PLANK_EDGE_END
        self.add_joint(
            body1=self._planks[-1],
            body2=WORLD_BODY,
            anchor1=(axN, hinge_y_lo, START_Z),
            anchor2=(axN, hinge_y_hi, START_Z),
            mode=JointMode.REVOLUTE,
            **damper_kwargs,
        )

    def test_final(self) -> None:
        """The bridge must stay suspended: both endpoints are pinned,
        and every plank should remain within a generous Z envelope of
        its starting height. Under gravity the middle planks sag, but
        they must not escape above the start or drop below a plausible
        sag depth (< 10 m for a 30-plank chain).
        """
        min_z = START_Z - 10.0
        max_z = START_Z + 0.5  # planks can't rise above the hinge line
        for i, body in enumerate(self._planks):
            pos, vel = self.jitter_body_state(body)
            assert np.isfinite(pos).all() and np.isfinite(vel).all(), (
                f"plank {i} non-finite state {pos=} {vel=}"
            )
            assert min_z < float(pos[2]) < max_z, (
                f"plank {i} escaped Z envelope: z={pos[2]:.2f} "
                f"(expected in [{min_z}, {max_z}])"
            )
            # X must stay within the bridge span plus a generous
            # sideways slop for the hinge sag.
            x_span_min = START_X - 2.0
            x_span_max = START_X + NUM_PLANKS * PLANK_STEP + 2.0
            assert x_span_min < float(pos[0]) < x_span_max, (
                f"plank {i} escaped X span: x={pos[0]:.2f}"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
