# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-side ``JointType.CABLE`` -> PhoenX ``JointMode.CABLE`` adapter
tests.

This module exercises the PhoenX cable constraint analytically and checks
the adapter glue: that
:meth:`ModelBuilder.add_joint_cable` survives ``model.finalize()`` and
lands on PhoenX's cable mode with the right anchor / stiffness /
damping wiring.

PhoenX has no axial-length compliance, so Newton's stretch DoF is
treated as rigid. The tests assert (a) the rigid ball-socket holds
the parent and child attachments coincident under load and (b) the
user-supplied isotropic bend stiffness produces a measurable
restoring torque on the rotation between the two bodies, scaling
correctly with the bend gain.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_CABLE,
)


def _two_body_cable_world(
    *,
    bend_stiffness: float,
    bend_damping: float,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    rest_bend: float = 0.0,
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

    box_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    anchor = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-3, 0, 0), (0, 1.0e-3, 0), (0, 0, 1.0e-3)),
    )
    mb.add_shape_box(anchor, hx=0.05, hy=0.05, hz=0.05, cfg=box_cfg)
    mb.add_joint_fixed(
        parent=-1,
        child=anchor,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )

    bob_at_anchor = rest_bend != 0.0
    bob = mb.add_link(
        xform=wp.transform(
            p=wp.vec3(0.0, 0.0, 1.0) if bob_at_anchor else wp.vec3(0.0, 0.0, 0.0),
            q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), rest_bend),
        ),
        mass=1.0,
        inertia=((1.0e-3, 0, 0), (0, 1.0e-3, 0), (0, 0, 1.0e-3)),
    )
    mb.add_shape_box(bob, hx=0.05, hy=0.05, hz=0.05, cfg=box_cfg)

    cable = mb.add_joint_cable(
        parent=anchor,
        child=bob,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=(
            wp.transform_identity() if bob_at_anchor else wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())
        ),
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        bend_stiffness=float(bend_stiffness),
        bend_damping=float(bend_damping),
    )
    mb.add_articulation([cable])

    model = mb.finalize()
    model.set_gravity(gravity)

    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=5,
        solver_iterations=1,
        velocity_iterations=0,
        articulation_mode="maximal",
    )
    return model, solver


def _step_n(model: newton.Model, solver: newton.solvers.SolverPhoenX, frames: int, dt: float) -> np.ndarray:
    """Advance a cable model through one captured five-substep frame."""
    state = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, dt)
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    return state.body_q.numpy()


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cable tests run on CUDA only (graph-capture path).",
)
class TestNewtonCableAdapter(unittest.TestCase):
    """Verify ``add_joint_cable`` -> PhoenX cable wiring."""

    def test_cable_constructs_without_error(self) -> None:
        """``add_joint_cable`` must build a valid solver column."""
        model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=0.5)
        self.assertEqual(int(model.joint_count), 2)  # FIXED + CABLE
        types = model.joint_type.numpy()
        self.assertIn(int(newton.JointType.CABLE), types.tolist())
        self.assertEqual(int(solver._joint_constraints.num_joint_columns), 2)
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (12,))
        np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [0, 0])

    def test_phoenx_cable_mode_id_set(self) -> None:
        """The descriptor for the Newton cable joint must end up
        tagged as PhoenX :data:`JOINT_MODE_CABLE`."""
        _model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=0.5)
        modes = solver._joint_constraints.joint_mode.numpy()
        self.assertIn(int(JOINT_MODE_CABLE), modes.tolist())

    def test_cable_holds_attachment_under_gravity(self) -> None:
        """Rigid ball-socket: under gravity the bob may swing about
        the anchor, but its world attachment point must stay
        coincident with the anchor's attachment point. With both
        attachments at the anchor body's centre and the bob's
        attachment at child-local ``(0, 0, +1)``, the bob's pose
        obeys ``bob_pos + R(bob_q) @ (0, 0, 1) ~ anchor_pos`` at all
        times."""
        model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=2.0)
        bq = _step_n(model, solver, frames=240, dt=1.0 / 200.0)
        anchor_pos = bq[0, :3]
        bob_pos = bq[1, :3]
        bob_quat = bq[1, 3:7]
        x, y, z, w = (float(v) for v in bob_quat)
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
        np.testing.assert_allclose(
            bob_attach_world,
            anchor_pos,
            atol=5.0e-3,
            err_msg=f"cable attachment slipped: bob_attach={bob_attach_world}, anchor={anchor_pos}",
        )

    def test_cable_bend_equilibrium_matches_hookes_law(self) -> None:
        """Balance transverse gravity with the direct cable bend spring."""
        stiffness = 20.0
        model, solver = _two_body_cable_world(
            bend_stiffness=stiffness,
            bend_damping=9.0,
            gravity=(-9.81, 0.0, 0.0),
        )
        body_q = _step_n(model, solver, frames=480, dt=1.0 / 240.0)
        measured = 2.0 * math.acos(min(abs(float(body_q[1, 6])), 1.0))
        expected = 0.0
        for _ in range(40):
            expected = 9.81 * math.cos(expected) / stiffness
        self.assertAlmostEqual(measured, expected, delta=2.0e-3)

    def test_cable_preserves_authored_curved_rest_state(self) -> None:
        """Preserve a non-straight cable rest orientation without external load."""
        rest_bend = 0.45
        model, solver = _two_body_cable_world(
            bend_stiffness=40.0,
            bend_damping=4.0,
            gravity=(0.0, 0.0, 0.0),
            rest_bend=rest_bend,
        )
        body_q = _step_n(model, solver, frames=120, dt=1.0 / 240.0)
        measured = 2.0 * math.acos(min(abs(float(body_q[1, 6])), 1.0))
        self.assertAlmostEqual(measured, rest_bend, delta=2.0e-3)

    def test_cable_refreshes_authored_curved_rest_state(self) -> None:
        """Refresh cable rest bend after an authored body-pose edit."""
        model, solver = _two_body_cable_world(
            bend_stiffness=40.0,
            bend_damping=4.0,
            gravity=(0.0, 0.0, 0.0),
            rest_bend=0.45,
        )
        rest_bend = 0.25
        body_q = model.body_q.numpy()
        body_q[1, 3:7] = (0.0, math.sin(0.5 * rest_bend), 0.0, math.cos(0.5 * rest_bend))
        model.body_q.assign(body_q)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

        body_q = _step_n(model, solver, frames=120, dt=1.0 / 240.0)
        measured = 2.0 * math.acos(min(abs(float(body_q[1, 6])), 1.0))
        self.assertAlmostEqual(measured, rest_bend, delta=2.0e-3)


if __name__ == "__main__":
    unittest.main()
