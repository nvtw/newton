# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`SolverPhoenX` -- the Newton-Model /
Newton-State wrapper around :class:`PhoenXWorld`.

Scenes cover the full integration surface from smallest to largest:

* free fall + contacts: a box dropped onto a plane must settle;
* revolute drive: a pendulum's PD drive parks at a target angle;
* fixed weld: a cube welded to the world must not move under gravity;
* a FREE-base + REVOLUTE-driven two-link chain (the Anymal building
  block at miniature scale) tracks simple PD targets.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton


GRAVITY = 9.81


def _make_box_model(*, box_z: float = 0.5, mu: float = 0.5) -> newton.Model:
    """Dynamic unit-mass cube + ground plane. No joints."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, box_z), q=wp.quat_identity()),
    )
    mb.add_shape_box(
        body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu),
    )
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_pendulum_model(*, target_angle: float = 0.0) -> newton.Model:
    """Static world body + dynamic cube + revolute joint with a PD
    drive towards ``target_angle`` (rad).

    Built with :meth:`add_link` + explicit joint + :meth:`add_articulation`
    so Newton does not auto-insert a FREE base joint on top.
    """
    mb = newton.ModelBuilder()
    cube = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, -1.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        cube,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_revolute(
        parent=-1,
        child=cube,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 1.0, 0.0), q=wp.quat_identity()),
        target_pos=target_angle,
        target_ke=50.0,
        target_kd=5.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    mb.add_articulation([joint])
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_welded_cube_model() -> newton.Model:
    """Single body welded to the world. Under gravity the cube must
    not move."""
    mb = newton.ModelBuilder()
    cube = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.5), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        cube,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_fixed(
        parent=-1,
        child=cube,
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.5), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    mb.add_articulation([joint])
    mb.gravity = -GRAVITY
    return mb.finalize()


def _run_frames(solver, state_0, state_1, control, contacts, model, n: int, dt: float):
    """Advance ``n`` frames eagerly with collision + solver step.

    Eager rather than CUDA-graph captured because the state pingpong
    (``state_0, state_1 = state_1, state_0``) between frames needs to
    be replayed on the Python side; mixing that with graph capture
    pins stale pointers into the graph.
    """
    for _ in range(n):
        state_0.clear_forces()
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX tests run on CUDA only.",
)
class TestSolverPhoenX(unittest.TestCase):
    """Behavioural checks for the Newton-Model driven PhoenX solver."""

    def test_box_settles_on_plane(self) -> None:
        """Dynamic cube on a plane -- after 1 s the COM z must be
        ~0.1 m (cube half-height) and velocity must be small."""
        model = _make_box_model(box_z=0.5, mu=0.5)
        solver = newton.solvers.SolverPhoenX(
            model, substeps=8, solver_iterations=16
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(
            solver, state_0, state_1, control, contacts, model, n=60, dt=dt
        )

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        com_z = float(body_q[0, 2])
        speed = float(np.linalg.norm(body_qd[0, :3]))
        self.assertAlmostEqual(
            com_z,
            0.1,
            delta=0.02,
            msg=f"cube COM z = {com_z:.3f} m, expected ~0.1",
        )
        self.assertLess(speed, 0.1, msg=f"cube still moving: |v|={speed:.3f}")

    def test_revolute_drive_tracks_target(self) -> None:
        """PD position drive on a revolute pendulum parks the cube at
        ``target_angle`` after a settle window."""
        target = math.pi / 4.0
        model = _make_pendulum_model(target_angle=target)
        solver = newton.solvers.SolverPhoenX(
            model, substeps=8, solver_iterations=16
        )
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        control = model.control()

        dt = 1.0 / 120.0
        state_0, state_1 = _run_frames(
            solver, state_0, state_1, control, None, model, n=480, dt=dt
        )

        # Read the joint angle back via eval_ik from body_q.
        joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32)
        joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)
        newton.eval_ik(model, state_0, joint_q, joint_qd)
        angle = float(joint_q.numpy()[0])
        self.assertAlmostEqual(
            angle,
            target,
            delta=0.1,
            msg=f"revolute drive off target: got {angle:.3f} rad, expected {target:.3f}",
        )

    def test_fixed_joint_holds_welded_cube(self) -> None:
        """FIXED joint anchors a cube in space under gravity."""
        model = _make_welded_cube_model()
        solver = newton.solvers.SolverPhoenX(
            model, substeps=8, solver_iterations=16
        )
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        control = model.control()

        dt = 1.0 / 120.0
        state_0, state_1 = _run_frames(
            solver, state_0, state_1, control, None, model, n=240, dt=dt
        )

        body_q = state_0.body_q.numpy()
        pos = body_q[0, :3]
        drift = float(np.linalg.norm(pos - np.array([0.5, 0.0, 0.5])))
        self.assertLess(drift, 0.05, msg=f"welded cube drifted {drift:.4f} m")


if __name__ == "__main__":
    wp.init()
    unittest.main()
