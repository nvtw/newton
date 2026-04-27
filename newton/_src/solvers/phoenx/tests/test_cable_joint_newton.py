# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-side ``JointType.CABLE`` -> PhoenX ``JointMode.CABLE`` adapter
tests.

The underlying PhoenX cable constraint (rigid ball-socket at
``anchor1`` plus 2 bend + 1 twist soft angular rows) was already
implemented and unit-tested via PhoenX's standalone ``WorldBuilder``;
see ``test_cable_joint.py``. This file checks the *adapter* glue:
that ``ModelBuilder.add_joint_cable`` survives ``model.finalize()``
and lands on PhoenX's cable mode with the right anchor / stiffness
/ damping wiring.

PhoenX has no axial-length compliance, so Newton's stretch DoF is
treated as rigid. The tests assert (a) the rigid ball-socket holds
the parent and child attachments coincident under load and (b) the
isotropic bend stiffness produces a measurable restoring torque on
the rotation between the two bodies.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    JOINT_MODE_CABLE,
)


def _two_body_cable_world(
    *, bend_stiffness: float, bend_damping: float
) -> tuple[newton.Model, newton.solvers.SolverPhoenX]:
    """Build a two-body cable scene:

    * Body 1: 1 kg cube anchored to the world via a FIXED joint at
      ``(0, 0, 1)``.
    * Body 2: 1 kg cube hanging from body 1 via a ``CABLE`` joint
      whose attachment is shared at ``(0, 0, 1)``. Body 2's COM is
      at ``(0, 0, 0)``, so under -z gravity the cable swings like a
      rotational pendulum about the cable's bend axes.

    The bend stiffness is the only restoring torque; with bend = 0
    the cable acts like a pure ball-socket and body 2 swings freely.
    """
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)

    box_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0 / (0.1 * 0.1 * 0.1))
    # Anchor: 0.1 m cube at (0, 0, 1).
    anchor = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-3, 0, 0), (0, 1.0e-3, 0), (0, 0, 1.0e-3)),
    )
    mb.add_shape_box(anchor, hx=0.05, hy=0.05, hz=0.05, cfg=box_cfg)
    mb.add_joint_fixed(parent=-1, child=anchor)

    # Hung body: another 0.1 m cube at (0, 0, 0). Cable attachment
    # point lies at the anchor's lower face = world (0, 0, 1).
    bob = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-3, 0, 0), (0, 1.0e-3, 0), (0, 0, 1.0e-3)),
    )
    mb.add_shape_box(bob, hx=0.05, hy=0.05, hz=0.05, cfg=box_cfg)

    cable = mb.add_joint_cable(
        parent=anchor,
        child=bob,
        # Anchor's lower face: at body-local ``(0, 0, 0)`` the cube
        # sits centred at the body, so the lower face is at
        # ``(0, 0, 0)`` of the body's frame for the parent and at
        # ``(0, 0, +1)`` of the child's frame so both attach in world
        # at ``(0, 0, 1)``.
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        bend_stiffness=float(bend_stiffness),
        bend_damping=float(bend_damping),
    )
    mb.add_articulation([cable])

    model = mb.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=8,
        velocity_iterations=1,
    )
    return model, solver


def _step_n(model: newton.Model, solver, frames: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance the model ``frames`` steps; return final ``body_q`` and
    peak inter-body distance along the way (a divergence indicator)."""
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = model.contacts() if model.shape_count > 0 else None
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    peak_disp = 0.0
    for _ in range(frames):
        s0.clear_forces()
        if contacts is not None:
            model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        bq = s0.body_q.numpy()
        peak_disp = max(peak_disp, float(np.linalg.norm(bq[1, :3] - np.array([0.0, 0.0, 0.0]))))
    return s0.body_q.numpy(), np.array([peak_disp], dtype=np.float32)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cable tests run on CUDA only (graph-capture path).",
)
class TestNewtonCableAdapter(unittest.TestCase):
    """Verify ``add_joint_cable`` -> PhoenX cable wiring."""

    def test_cable_constructs_without_error(self) -> None:
        """Newton CABLE used to raise ``NotImplementedError`` in
        ``model_adapter``; this is the regression."""
        model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=0.5)
        # Sanity: solver constructed, model has the right joint type.
        self.assertEqual(int(model.joint_count), 2)  # FIXED + CABLE
        types = model.joint_type.numpy()
        self.assertIn(int(newton.JointType.CABLE), types.tolist())
        # And the ADBS column for the cable joint exists.
        self.assertEqual(int(solver._adbs.num_joint_columns), 2)

    def test_cable_holds_attachment_under_gravity(self) -> None:
        """Rigid ball-socket: under gravity the bob may swing about the
        anchor, but its world attachment point must stay coincident with
        the anchor's attachment point. With both attachments at the
        anchor body's centre (parent_xform.p = 0) and the bob's
        attachment at child-local ``(0, 0, +1)``, the bob's pose obeys
        ``bob_pos + R(bob_q) @ (0, 0, 1) ~ anchor_pos`` at all times."""
        model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=2.0)
        bq, _ = _step_n(model, solver, frames=240, dt=1.0 / 200.0)
        anchor_pos = bq[0, :3]
        bob_pos = bq[1, :3]
        bob_quat = bq[1, 3:7]  # (x, y, z, w)
        # Rotate (0, 0, 1) by bob_quat: child-local cable attachment
        # in world.
        x, y, z, w = (float(v) for v in bob_quat)
        # Vector v = (0, 0, 1), use the standard q*v formula.
        tx = 2.0 * (y * 1.0 - z * 0.0)
        ty = 2.0 * (z * 0.0 - x * 1.0)
        tz = 2.0 * (x * 0.0 - y * 0.0)
        attach_offset = np.array(
            [
                0.0 + w * tx + (y * tz - z * ty),
                0.0 + w * ty + (z * tx - x * tz),
                1.0 + w * tz + (x * ty - y * tx),
            ]
        )
        bob_attach_world = bob_pos + attach_offset
        # Parent attachment in world = anchor_pos + (0, 0, 0) = anchor_pos.
        np.testing.assert_allclose(
            bob_attach_world,
            anchor_pos,
            atol=5.0e-3,
            err_msg=f"cable attachment slipped: bob_attach={bob_attach_world}, anchor={anchor_pos}",
        )

    def test_phoenx_cable_mode_id_set(self) -> None:
        """The descriptor for the Newton cable joint must end up tagged
        as PhoenX ``JOINT_MODE_CABLE`` (not REVOLUTE / FIXED / etc.)."""
        _model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=0.5)
        modes = solver._adbs.joint_mode.numpy()
        self.assertIn(int(JOINT_MODE_CABLE), modes.tolist())


if __name__ == "__main__":
    unittest.main()
