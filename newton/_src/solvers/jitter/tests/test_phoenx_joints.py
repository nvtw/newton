# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PhoenX solver's actuated double-ball-socket joint.

The unified joint covers three modes (see
:class:`~newton._src.solvers.jitter.world_builder.JointMode`):

* **REVOLUTE**: hinge about the ``anchor1 -> anchor2`` axis. Locks
  3 translational + 2 rotational DoF.
* **BALL_SOCKET**: pins ``body2`` at ``anchor1`` in ``body1``'s
  frame. Locks 3 translational DoF; all 3 rotational DoF are free.
* **PRISMATIC**: slider along the ``anchor1 -> anchor2`` axis.
  Locks 3 rotational + 2 translational DoF.

The runtime dispatcher (fast-path tail kernels in
:mod:`solver_jitter_kernels`) already routes
:data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET`; these tests
validate the PhoenX wiring around it:

* :class:`PhoenXWorld` correctly carves the cid range so contacts
  stay at ``[num_joints, num_joints + max_contact_columns)`` and
  joints occupy ``[0, num_joints)``,
* :meth:`PhoenXWorld.initialize_actuated_double_ball_socket_joints`
  seeds the per-joint anchor and soft-constraint state,
* the graph-colouring partitioner picks up joint + contact
  elements in the same CSR and the fast-path dispatcher sweeps
  both with a single launch per substep,
* dynamics qualitatively match the physical expectations for each
  joint mode.

Runs on CUDA only -- same rationale as the other PhoenX tests.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    body_container_zeros,
)
from newton._src.solvers.jitter.constraints.contact_matching_config import (
    JITTER_CONTACT_MATCHING,
)
from newton._src.solvers.jitter.solver_phoenx import PhoenXWorld
from newton._src.solvers.jitter.tests.test_phoenx_stacking import (
    _init_phoenx_bodies_kernel,
    _newton_to_phoenx_kernel,
    _phoenx_to_newton_kernel,
)
from newton._src.solvers.jitter.world_builder import DriveMode, JointMode

_G = 9.81


# ---------------------------------------------------------------------------
# Minimal two-body scene: one static anchor + one free dynamic body.
# ---------------------------------------------------------------------------


class _PendulumScene:
    """Anchor body (static) + pendulum body (dynamic) + one joint.

    The anchor sits at ``anchor_world``. The pendulum spawns at
    ``pendulum_world`` and may hang / slide / rotate depending on
    the joint mode. No ground plane -- we want contacts isolated
    from joint dynamics, so the joint is the only constraint.
    """

    def __init__(
        self,
        *,
        anchor_world: tuple[float, float, float],
        pendulum_world: tuple[float, float, float],
        pendulum_mass: float = 1.0,
        pendulum_half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1),
        fps: int = 120,
        substeps: int = 8,
        solver_iterations: int = 16,
        gravity: tuple[float, float, float] = (0.0, 0.0, -_G),
    ) -> None:
        self.device = wp.get_device("cuda:0")
        self.fps = int(fps)
        self.frame_dt = 1.0 / self.fps
        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.anchor_world = anchor_world
        self.pendulum_world = pendulum_world
        self.pendulum_mass = float(pendulum_mass)
        self.gravity = gravity

        # ---- Newton side ---------------------------------------------
        mb = newton.ModelBuilder()
        # Anchor body: dynamic per Newton but we'll flip it STATIC on
        # the PhoenX side. Newton's builder requires at least one
        # shape, so we attach a tiny sphere and make it kinematic-
        # equivalent by zeroing its inverse mass in the PhoenX body
        # container below.
        self._anchor_newton = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(*anchor_world), q=wp.quat_identity()
            ),
            # ``mass = 0`` on Newton side means "static body" -- the
            # ingest path then produces inv_mass = 0 and the PhoenX
            # init kernel sets motion_type = STATIC for us.
            mass=0.0,
        )
        mb.add_shape_sphere(
            self._anchor_newton,
            radius=1.0e-3,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        # Pendulum body: explicit mass + analytic inertia so the
        # analytical hang / torque checks are clean (same trick
        # _PhoenXScene.add_box uses to dodge the density=0 zero-
        # inertia-validator bug).
        hx, hy, hz = pendulum_half_extents
        ixx = self.pendulum_mass / 3.0 * (hy * hy + hz * hz)
        iyy = self.pendulum_mass / 3.0 * (hx * hx + hz * hz)
        izz = self.pendulum_mass / 3.0 * (hx * hx + hy * hy)
        self._pendulum_newton = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(*pendulum_world), q=wp.quat_identity()
            ),
            mass=self.pendulum_mass,
            inertia=((ixx, 0.0, 0.0), (0.0, iyy, 0.0), (0.0, 0.0, izz)),
        )
        mb.add_shape_box(
            self._pendulum_newton,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        self.model = mb.finalize()
        self.state = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state
        )
        self.model.body_q.assign(self.state.body_q)

        # ---- PhoenX body container (anchor + pendulum) ---------------
        num_bodies = self.model.body_count + 1  # slot 0 = world anchor
        bodies = body_container_zeros(num_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        wp.launch(
            _init_phoenx_bodies_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
            ],
            device=self.device,
        )
        self.bodies = bodies

        # ---- Collision pipeline (joint-only scene -> no contacts) ----
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=JITTER_CONTACT_MATCHING
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])
        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_phoenx, dtype=wp.int32, device=self.device
        )

        # ---- Constraint container with 1 joint + contact capacity ----
        self.num_joints = 1
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=self.num_joints,
            max_contact_columns=max_contact_columns,
            device=self.device,
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=1,  # helps joint stiffness converge
            gravity=self.gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(self.model.shape_count),
            num_joints=self.num_joints,
            device=self.device,
        )

    def init_joint(
        self,
        *,
        mode: JointMode,
        anchor1: tuple[float, float, float] | None = None,
        anchor2: tuple[float, float, float] | None = None,
        hertz: float = 60.0,
        damping_ratio: float = 1.0,
        drive_mode: DriveMode = DriveMode.OFF,
        target: float = 0.0,
        target_velocity: float = 0.0,
        max_force_drive: float = 0.0,
        stiffness_drive: float = 0.0,
        damping_drive: float = 0.0,
        min_value: float = 1.0,
        max_value: float = -1.0,
        hertz_limit: float = 60.0,
        damping_ratio_limit: float = 1.0,
        stiffness_limit: float = 0.0,
        damping_limit: float = 0.0,
    ) -> None:
        """Initialise the single joint between anchor body (slot 1)
        and pendulum body (slot 2).

        Default anchors: ``anchor1`` at :attr:`anchor_world`,
        ``anchor2`` one unit below it along +X. The caller may
        override to pick an explicit hinge / slide axis.
        """
        if anchor1 is None:
            anchor1 = self.anchor_world
        if anchor2 is None:
            anchor2 = (
                self.anchor_world[0] + 1.0,
                self.anchor_world[1],
                self.anchor_world[2],
            )

        anchor_phoenx = self._anchor_newton + 1
        pend_phoenx = self._pendulum_newton + 1
        d = self.device

        def _f(xs: list[float]) -> wp.array:
            return wp.array(np.asarray(xs, dtype=np.float32), dtype=wp.float32, device=d)

        def _i(xs: list[int]) -> wp.array:
            return wp.array(np.asarray(xs, dtype=np.int32), dtype=wp.int32, device=d)

        def _v(xs: list[tuple[float, float, float]]) -> wp.array:
            return wp.array(np.asarray(xs, dtype=np.float32), dtype=wp.vec3f, device=d)

        self.world.initialize_actuated_double_ball_socket_joints(
            body1=_i([anchor_phoenx]),
            body2=_i([pend_phoenx]),
            anchor1=_v([anchor1]),
            anchor2=_v([anchor2]),
            hertz=_f([hertz]),
            damping_ratio=_f([damping_ratio]),
            joint_mode=_i([int(mode)]),
            drive_mode=_i([int(drive_mode)]),
            target=_f([target]),
            target_velocity=_f([target_velocity]),
            max_force_drive=_f([max_force_drive]),
            stiffness_drive=_f([stiffness_drive]),
            damping_drive=_f([damping_drive]),
            min_value=_f([min_value]),
            max_value=_f([max_value]),
            hertz_limit=_f([hertz_limit]),
            damping_ratio_limit=_f([damping_ratio_limit]),
            stiffness_limit=_f([stiffness_limit]),
            damping_limit=_f([damping_limit]),
        )

    def step(self) -> None:
        # Pure Python per-frame path (no graph capture; these tests
        # run for at most a few hundred frames and the kernel-launch
        # overhead is negligible compared with the warm-up compile
        # cost).
        wp.launch(
            _newton_to_phoenx_kernel,
            dim=self.model.body_count,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + self.model.body_count],
                self.bodies.orientation[1 : 1 + self.model.body_count],
                self.bodies.velocity[1 : 1 + self.model.body_count],
                self.bodies.angular_velocity[1 : 1 + self.model.body_count],
            ],
            device=self.device,
        )
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )
        wp.launch(
            _phoenx_to_newton_kernel,
            dim=self.model.body_count,
            inputs=[
                self.bodies.position[1 : 1 + self.model.body_count],
                self.bodies.orientation[1 : 1 + self.model.body_count],
                self.bodies.velocity[1 : 1 + self.model.body_count],
                self.bodies.angular_velocity[1 : 1 + self.model.body_count],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def pendulum_position(self) -> np.ndarray:
        return self.bodies.position.numpy()[self._pendulum_newton + 1].copy()

    def pendulum_velocity(self) -> np.ndarray:
        return self.bodies.velocity.numpy()[self._pendulum_newton + 1].copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX joint tests require CUDA"
)
class TestPhoenXActuatedDoubleBallSocket(unittest.TestCase):
    """Unified-joint smoke + dynamics tests for :class:`PhoenXWorld`."""

    def test_ball_socket_anchor_holds(self) -> None:
        """Ball-socket pinned at the anchor: the pendulum body
        cannot drift translationally past the joint stiffness even
        under gravity, but is free to rotate.

        We release the pendulum 0.5 m directly below the anchor so
        the anchor point is ``pendulum + (0, 0, 0.5)`` (above it)
        and watch that the anchor-to-pendulum offset stays near
        the initial 0.5 m across several frames of simulation.
        """
        scene = _PendulumScene(
            anchor_world=(0.0, 0.0, 1.0),
            pendulum_world=(0.0, 0.0, 0.5),
        )
        # Ball-socket: anchor1 = pendulum's initial position (the
        # "attach point" on the pendulum is at anchor1 in world
        # frame). Drive / limit inputs must stay at defaults.
        scene.init_joint(
            mode=JointMode.BALL_SOCKET,
            anchor1=(0.0, 0.0, 0.5),
            anchor2=(0.0, 0.0, 0.5),  # ignored for ball-socket
        )

        initial = scene.pendulum_position()
        for _ in range(120):  # 1 s
            scene.step()
        final = scene.pendulum_position()

        # The ball-socket attach point was world (0,0,0.5), which
        # coincides with the pendulum's COM at t=0, so the pendulum
        # stays anchored to that same world point for the whole sim
        # (no moment arm -> no pendulum swing). Position drift must
        # stay well below the sphere radius.
        drift = float(np.linalg.norm(final - initial))
        self.assertLess(
            drift,
            0.05,
            f"ball-socket pendulum drifted {drift:.4f} m from its anchor",
        )

    def test_revolute_hinge_hangs(self) -> None:
        """Revolute joint: pendulum released sideways must fall
        under gravity but stay on the swing arc.

        Setup: anchor at ``(0, 0, 1)``, pendulum at ``(0.5, 0, 1)``
        (to the +X of the anchor, at the same height). The hinge
        axis runs along +Y, so the pendulum swings in the XZ plane.
        The pendulum attach point (anchor1 in world frame) is the
        anchor itself, so the pendulum swings with a 0.5 m arm --
        classic gravity pendulum.

        After a quarter-period the pendulum must have dropped from
        its release height and its radial distance from the anchor
        must stay near 0.5 m (within joint soft-constraint slack).
        """
        anchor = (0.0, 0.0, 1.0)
        scene = _PendulumScene(
            anchor_world=anchor,
            pendulum_world=(0.5, 0.0, 1.0),
        )
        # Anchor2 = anchor1 + (0, 1, 0) -> hinge axis along +Y.
        scene.init_joint(
            mode=JointMode.REVOLUTE,
            anchor1=anchor,
            anchor2=(anchor[0], anchor[1] + 1.0, anchor[2]),
            hertz=60.0,
            damping_ratio=1.0,
        )

        # Run long enough that the pendulum drops appreciably but
        # not a full period. ``T = 2 * pi * sqrt(L/g) ~ 1.42 s`` for
        # L=0.5 m; quarter period ~ 0.36 s.
        initial = scene.pendulum_position()
        arm_len = 0.5
        for _ in range(40):  # ~0.33 s
            scene.step()
        final = scene.pendulum_position()

        # Must have dropped under gravity.
        self.assertLess(
            float(final[2]),
            float(initial[2]) - 0.05,
            f"revolute pendulum did not drop under gravity "
            f"(z_initial={initial[2]:.3f} -> z_final={final[2]:.3f})",
        )
        # Arm length preserved within soft-constraint slack
        # (``hertz=60`` with critical damping is tight). Allow 5 %
        # -- any bigger drift points at a lock-row bug.
        r = float(np.linalg.norm(final - np.array(anchor)))
        self.assertLess(
            abs(r - arm_len),
            0.05,
            f"revolute arm length drifted: |d|={r:.4f} m vs L={arm_len} m",
        )

    def test_prismatic_slide_with_drive(self) -> None:
        """Prismatic joint with a position drive: the pendulum
        slides along the axis and settles at the drive target.

        Setup: anchor body at origin; pendulum at ``(0.5, 0, 0)``;
        ``anchor1 = (0, 0, 0)``; slide axis along +X (``anchor2 =
        (1, 0, 0)``). The slide coordinate ``x`` is
        ``n_hat . (p2 - p1)`` where ``p1`` is the anchor1 point
        projected onto body 1's frame (origin, static) and ``p2``
        is the anchor1 point projected onto body 2's frame. At
        init ``p2 = anchor1 = (0, 0, 0)`` (in world), so the slide
        starts at 0. A drive target of ``x = 2.0`` therefore
        drags the pendulum's attach point to ``(2, 0, 0)``,
        which means the pendulum's *COM* ends at
        ``(2.5, 0, 0)`` (pendulum COM = attach point +
        initial_COM_to_attach_offset = (2, 0, 0) + (0.5, 0, 0)).

        Gravity is disabled so the drive's spring/damper is the
        only force.
        """
        scene = _PendulumScene(
            anchor_world=(0.0, 0.0, 0.0),
            pendulum_world=(0.5, 0.0, 0.0),
            gravity=(0.0, 0.0, 0.0),  # isolate the drive's effect
        )
        scene.init_joint(
            mode=JointMode.PRISMATIC,
            anchor1=(0.0, 0.0, 0.0),
            anchor2=(1.0, 0.0, 0.0),  # slide axis along +X, rest length 1
            drive_mode=DriveMode.POSITION,
            target=2.0,
            max_force_drive=1.0e4,
            stiffness_drive=200.0,
            damping_drive=40.0,
        )

        # 2 s is plenty for a critically-damped PD drive at kp=200
        # on a 1 kg body.
        for _ in range(240):
            scene.step()
        final = scene.pendulum_position()

        # Expected pendulum COM x = target_slide + pendulum_init_x
        # = 2.0 + 0.5 = 2.5.
        expected_x = 2.5
        x = float(final[0])
        self.assertAlmostEqual(
            x,
            expected_x,
            delta=0.1,
            msg=f"prismatic drive did not converge: x={x:.3f} vs expected "
            f"{expected_x:.3f} (slide target 2.0 + initial COM offset 0.5)",
        )
        # Perpendicular offsets must stay zero -- the prismatic
        # 2+2+1 lock pins them.
        perp = math.hypot(float(final[1]), float(final[2]))
        self.assertLess(
            perp,
            0.05,
            f"prismatic perpendicular drift {perp:.4f} m -- lock row leaked",
        )

    def test_revolute_limit_holds(self) -> None:
        """Revolute with a tight angle window: the pendulum must
        not swing past the window even under full gravitational
        torque.

        Geometry identical to :meth:`test_revolute_hinge_hangs`,
        but a narrow limit ``[-0.1, 0.1] rad`` stops the pendulum
        near the release angle. The arm angle stays inside the
        window within soft-limit slack.
        """
        anchor = (0.0, 0.0, 1.0)
        scene = _PendulumScene(
            anchor_world=anchor,
            pendulum_world=(0.5, 0.0, 1.0),
        )
        scene.init_joint(
            mode=JointMode.REVOLUTE,
            anchor1=anchor,
            anchor2=(anchor[0], anchor[1] + 1.0, anchor[2]),
            min_value=-0.1,
            max_value=0.1,
            hertz_limit=120.0,
            damping_ratio_limit=1.0,
        )

        for _ in range(120):  # 1 s
            scene.step()
        final = scene.pendulum_position()

        # Pendulum should still be near the +X side (hasn't
        # swung past +0.1 rad limit). With arm=0.5, angle 0.1 rad
        # means the pendulum is at roughly (0.5 cos(0.1), 0, 1 -
        # 0.5 sin(0.1)) = (0.4975, 0, 0.9501). Allow 5 % window
        # slack for the soft limit.
        arm = 0.5
        limit_rad = 0.1
        expected_x_min = arm * math.cos(limit_rad + 0.1)  # worst case past limit
        self.assertGreater(
            float(final[0]),
            expected_x_min,
            f"revolute limit overshot: pendulum at x={final[0]:.3f} "
            f"(limit expects > {expected_x_min:.3f})",
        )


if __name__ == "__main__":
    unittest.main()
