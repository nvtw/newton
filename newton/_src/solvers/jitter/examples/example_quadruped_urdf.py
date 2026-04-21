# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Quadruped URDF
#
# Jitter-side reproduction of ``newton.examples.basic_urdf``. Parses the
# same ``quadruped.urdf``, keeps the same inertia augmentation, PD drive
# targets and floating-base height, then hands the articulation off to
# the Jitter solver instead of ``SolverXPBD`` / ``SolverVBD``.
#
# The URDF has a floating base plus 12 revolute hip/knee hinges (all
# axis-X-local). :func:`build_jitter_joints_from_model` mirrors them
# into the Jitter :class:`WorldBuilder` as :attr:`JointMode.REVOLUTE`
# entries with position drives pointing at ``joint_target_pos`` -- i.e.
# the canonical stance that ``basic_urdf`` commands. The free joint
# binding the base to world is silently skipped (Jitter has no free-
# joint constraint; the child body is already a free dynamic rigid
# body once mirrored through :func:`build_jitter_world_from_model`).
#
# Use ``--world-count`` to replicate the quadruped. Defaults to a
# modest number (4) -- basic_urdf's default of 100 is easy with XPBD
# but Jitter's contact budget scales linearly with bodies, so start
# small and tweak via the CLI once you know your hardware.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_quadruped_urdf
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
    build_jitter_joints_from_model,
)
from newton._src.solvers.jitter.world_builder import DriveMode, WorldBuilder

# Starting floating-base height [m]. Matches basic_urdf.
BASE_Z = 0.7

# Per-body inertia augmentation [kg*m^2] added to the diagonal of every
# link's inertia tensor for numerical stability. Matches basic_urdf.
BODY_ARMATURE = 0.01

# Joint armature added to ``default_joint_cfg`` before URDF import;
# matches basic_urdf's XPBD / VBD path.
JOINT_ARMATURE = 0.01

# PD drive gains and torque cap for the hinge motors. basic_urdf runs
# XPBD with ``target_ke=2000`` / ``target_kd=1``; those values feed a
# soft position constraint, not a torque cap, so there isn't a direct
# transfer function to Jitter's ``max_force_drive``. 50 Nm is plenty
# for this small quadruped and keeps the drives from fighting the
# impulse limits during the initial drop.
#
# Jitter uses the same ``kp`` / ``kd`` convention as XPBD's ``target_ke``
# / ``target_kd`` (the Jitter2 AngularMotor PD: ``tau = kp*(theta -
# theta*) + kd*theta_dot``), so we pass the basic_urdf gains through
# directly. ``max_force_drive`` then clamps the per-substep impulse.
DRIVE_STIFFNESS = 2000.0
DRIVE_DAMPING = 1.0
DRIVE_MAX_TORQUE = 50.0

# Twelve stance targets, one per actuated joint (hip-abduct, hip-flex,
# knee-flex for each of the four legs). Copied verbatim from
# basic_urdf's ``joint_target_pos[-12:]`` so the PD drives settle to
# the same canonical pose.
STANCE_JOINT_Q: list[float] = [
    0.2,
    0.4,
    -0.6,
    -0.2,
    -0.4,
    0.6,
    -0.2,
    0.4,
    -0.6,
    0.2,
    -0.4,
    0.6,
]

# Layout knobs for replicating the quadruped.
WORLD_SPACING_X = 2.5
WORLD_SPACING_Y = 2.5


def _world_xform(grid_x: int, grid_y: int) -> wp.transform:
    """Return the spawn transform for world ``(grid_x, grid_y)`` on a
    compact square lattice centred on the origin.
    """
    # Lay worlds out on a roughly square grid. Separate rows along +X
    # and columns along +Y so the ground plane stays open in front of
    # the camera.
    return wp.transform(
        p=wp.vec3(
            float(grid_x) * WORLD_SPACING_X,
            float(grid_y) * WORLD_SPACING_Y,
            BASE_Z,
        ),
        q=wp.quat_identity(),
    )


class Example(DemoExample):
    def __init__(self, viewer, args):
        self.world_count = int(getattr(args, "world_count", 1) or 1)

        cfg = DemoConfig(
            title="Quadruped URDF",
            camera_pos=(3.5, -3.5, 1.8),
            camera_pitch=-10.0,
            camera_yaw=-45.0,
            # basic_urdf uses fps=100, substeps=10; Jitter's PGS runs
            # many more active constraints per step than XPBD, so a
            # 60 Hz render with 10 substeps (600 Hz physics) keeps
            # contacts tight without hammering the CUDA graph.
            fps=60,
            substeps=10,
            solver_iterations=12,
        )
        super().__init__(viewer, args, cfg)
        self._root_bodies: list[int] = []
        self.finish_setup()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def build_scene(self) -> None:
        mb = self.model_builder

        # One quadruped builder that we replicate -- matches basic_urdf's
        # sub-builder pattern. Replicated quadrupeds share the same
        # shape defaults and joint defaults so the Jitter-side inertia
        # mirror is consistent across worlds.
        quadruped = newton.ModelBuilder()
        quadruped.default_joint_cfg.armature = JOINT_ARMATURE
        # ``target_ke`` / ``target_kd`` are XPBD-specific. We don't use
        # them in Jitter (``max_force_drive`` + ``stiffness_drive`` /
        # ``damping_drive`` cover the same role -- a Jitter2
        # LinearMotor / AngularMotor PD clamped by a force cap), but
        # we keep the friction coefficient since Newton's
        # CollisionPipeline does read it.
        quadruped.default_shape_cfg.mu = 1.0

        quadruped.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            ignore_inertial_definitions=True,
        )

        # basic_urdf bumps every link's inertia diagonal by BODY_ARMATURE
        # to stabilise the articulation. Jitter uses the same
        # inverse-inertia tensor, so we mirror the tweak here.
        for body in range(quadruped.body_count):
            inertia_np = np.asarray(
                quadruped.body_inertia[body], dtype=np.float32
            ).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * BODY_ARMATURE
            quadruped.body_inertia[body] = wp.mat33(inertia_np)

        # Stance: the last 12 joint coordinates drive the hip/knee
        # hinges. Set both ``joint_q`` (initial pose) and
        # ``joint_target_pos`` (PD setpoint) so ``eval_fk`` spawns the
        # legs in the canonical pose and the Jitter drive holds it.
        quadruped.joint_q[-12:] = list(STANCE_JOINT_Q)
        quadruped.joint_target_pos[-12:] = list(STANCE_JOINT_Q)

        # Replicate into the scene on a square lattice.
        # basic_urdf uses ``scene.replicate`` which stacks worlds at
        # the origin; Jitter mirrors every body to a Jitter rigid body
        # so co-located copies would collide on frame 0. We pass a
        # per-world spawn ``xform`` to space them out instead.
        self._root_bodies = []
        side = max(1, int(np.ceil(np.sqrt(self.world_count))))
        spawned = 0
        for gy in range(side):
            for gx in range(side):
                if spawned >= self.world_count:
                    break
                start_body = mb.body_count
                mb.add_builder(quadruped, xform=_world_xform(gx, gy))
                # First body of each replicated builder is the
                # floating base; remember it for ``test_final``.
                self._root_bodies.append(start_body)
                spawned += 1

        # Ground plane (Newton convention: normal along +Z, origin at
        # z = 0). Matches ``scene.add_ground_plane`` in basic_urdf.
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

    # ------------------------------------------------------------------
    # Mirror Newton-side URDF joints into the Jitter builder
    # ------------------------------------------------------------------

    def on_jitter_builder_ready(
        self,
        builder: WorldBuilder,
        newton_to_jitter: dict[int, int],
    ) -> None:
        """Translate every URDF-sourced revolute joint (12 per
        quadruped, times ``world_count``) into a Jitter
        :attr:`JointMode.REVOLUTE` with a PD drive pointing at
        ``joint_target_pos``. The floating base's ``FREE`` joint is
        skipped by the helper.
        """
        build_jitter_joints_from_model(
            self.model,
            self.state,
            builder,
            newton_to_jitter,
            drive_mode=DriveMode.POSITION,
            max_force_drive=DRIVE_MAX_TORQUE,
            stiffness_drive=DRIVE_STIFFNESS,
            damping_drive=DRIVE_DAMPING,
            apply_joint_targets=True,
        )

    # ------------------------------------------------------------------
    # Asserts
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """After a few seconds of settling every quadruped should have
        its base within a generous height band around the terminal
        stance and a bounded velocity. Tolerances are loose because
        the Jitter PD drives settle slightly higher than XPBD's
        ``SolverXPBD`` reference (``z ~ 0.46`` in basic_urdf).
        """
        for idx, root in enumerate(self._root_bodies):
            pos, vel = self.jitter_body_state(root)
            assert np.isfinite(pos).all() and np.isfinite(vel).all(), (
                f"quadruped {idx} non-finite state {pos=} {vel=}"
            )
            z = float(pos[2])
            assert 0.15 < z < BASE_Z + 0.2, (
                f"quadruped {idx} base out of range: z={z:.3f} "
                f"(expected 0.15 < z < {BASE_Z + 0.2:.3f})"
            )
            speed = float(np.linalg.norm(vel))
            assert speed < 2.0, (
                f"quadruped {idx} base still moving fast: |v|={speed:.3f}"
            )

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def default_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
