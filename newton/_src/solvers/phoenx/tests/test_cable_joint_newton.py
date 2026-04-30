# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton-side ``JointType.CABLE`` -> PhoenX ``JointMode.CABLE`` adapter
tests.

The PhoenX cable constraint itself is exercised analytically in
``test_cable_joint.py`` -- this file checks the *adapter glue*: that
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
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    JOINT_MODE_CABLE,
)


def _two_body_cable_world(
    *,
    bend_stiffness: float,
    bend_damping: float,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
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
    mb.add_joint_fixed(parent=-1, child=anchor)

    bob = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-3, 0, 0), (0, 1.0e-3, 0), (0, 0, 1.0e-3)),
    )
    mb.add_shape_box(bob, hx=0.05, hy=0.05, hz=0.05, cfg=box_cfg)

    cable = mb.add_joint_cable(
        parent=anchor,
        child=bob,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
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
        substeps=4,
        solver_iterations=8,
        velocity_iterations=1,
    )
    return model, solver


def _step_n(model: newton.Model, solver, frames: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance the model ``frames`` steps using CUDA graph capture
    (CUDA only); return final ``body_q`` and peak inter-body distance
    along the way (a divergence indicator).

    Captures a *two-step* graph (one ``s0->s1`` + one ``s1->s0``
    substep) so the s0 / s1 alternation that ``solver.step`` requires
    lands at a fixed-point under graph replay. After capture,
    ``s0`` references the same buffer it did pre-capture (the
    even-step output), so ``s0.body_q`` is always the post-replay
    state.
    """
    device = wp.get_device()
    if not device.is_cuda:
        raise unittest.SkipTest("graph-capture path requires CUDA")
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = model.contacts() if model.shape_count > 0 else None
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    def _eager_pair() -> None:
        # Run *two* eager steps so s0 / s1 return to their starting
        # roles (matches the captured graph's post-replay invariant).
        for s_in, s_out in ((s0, s1), (s1, s0)):
            s_in.clear_forces()
            if contacts is not None:
                model.collide(s_in, contacts)
            solver.step(s_in, s_out, control, contacts, dt)

    if frames < 4:
        for _ in range(frames // 2):
            _eager_pair()
        return s0.body_q.numpy(), np.array([float(np.linalg.norm(s0.body_q.numpy()[1, :3]))], dtype=np.float32)

    # Warm-up: two paired steps to JIT the kernels and absorb lazy
    # allocations before capture (4 frames consumed).
    _eager_pair()

    with wp.ScopedCapture(device=device) as capture:
        for s_in, s_out in ((s0, s1), (s1, s0)):
            s_in.clear_forces()
            if contacts is not None:
                model.collide(s_in, contacts)
            solver.step(s_in, s_out, control, contacts, dt)
    graph = capture.graph

    # Already advanced 4 frames (2 warm-up + 2 capture). Replay the
    # remainder.
    consumed = 4
    remaining = max(0, frames - consumed)
    n_pairs = remaining // 2
    odd = remaining % 2

    peak_disp = 0.0
    for _ in range(n_pairs):
        wp.capture_launch(graph)
        bq = s0.body_q.numpy()
        peak_disp = max(peak_disp, float(np.linalg.norm(bq[1, :3] - np.array([0.0, 0.0, 0.0]))))
    for _ in range(odd):
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
        """``add_joint_cable`` must build a valid solver column."""
        model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=0.5)
        self.assertEqual(int(model.joint_count), 2)  # FIXED + CABLE
        types = model.joint_type.numpy()
        self.assertIn(int(newton.JointType.CABLE), types.tolist())
        self.assertEqual(int(solver._adbs.num_joint_columns), 2)

    def test_phoenx_cable_mode_id_set(self) -> None:
        """The descriptor for the Newton cable joint must end up
        tagged as PhoenX :data:`JOINT_MODE_CABLE`."""
        _model, solver = _two_body_cable_world(bend_stiffness=10.0, bend_damping=0.5)
        modes = solver._adbs.joint_mode.numpy()
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
        bq, _ = _step_n(model, solver, frames=240, dt=1.0 / 200.0)
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

    def test_cable_bend_gain_resists_gravity(self) -> None:
        """Bend gain wiring smoke test: the cable bend spring must
        actually be loaded from ``model.joint_target_ke`` and produce
        a *measurable* restoring torque -- a missing-gain bug
        (``stiff_drive`` written as 0) would let the cable swing
        freely like a ball-socket pendulum to the gravity-equilibrium
        angle, while a working gain holds the cable nearly upright.

        Setup: same two-body cable as the other tests, but with
        gravity rotated to ``-X`` so it produces a torque about the
        cable bend axes (with default ``-Z`` gravity the cable axis is
        aligned with gravity and there is no lever arm). After 4 s
        of settling, compare the cable tilt under a stiff bend
        (``k=2000 N*m/rad``) with a soft bend (``k=1 N*m/rad``):
        stiff must produce a *significantly smaller* tilt.
        """
        g = 9.81
        n_frames = int(4.0 * 240)
        dt = 1.0 / 240.0

        def measure_tilt(k: float) -> float:
            model, solver = _two_body_cable_world(
                bend_stiffness=k,
                bend_damping=8.0,
                # Rotate gravity to -X so it produces a moment about
                # the cable's bend axes (default -Z gravity is parallel
                # to the cable axis and produces no lever arm).
                gravity=(-g, 0.0, 0.0),
            )
            bq, _ = _step_n(model, solver, frames=n_frames, dt=dt)
            w = abs(float(bq[1, 6]))
            return 2.0 * math.acos(min(w, 1.0))

        tilt_stiff = measure_tilt(2000.0)
        tilt_soft = measure_tilt(1.0)

        # Stiff spring should hold the cable upright (-X gravity is
        # ~1 N*m torque -> theta_eq ~= 1/2000 rad ~= 0.03 deg in the
        # small-angle limit), soft spring should barely resist gravity
        # (theta_eq saturates near pi/2 when k is too small to balance
        # m*g*r). 5x gap is generous against numerical noise.
        self.assertLess(
            tilt_stiff,
            0.2 * tilt_soft,
            msg=(
                f"cable bend gain wiring: stiff k=2000 tilt={math.degrees(tilt_stiff):.3f} deg, "
                f"soft k=1 tilt={math.degrees(tilt_soft):.3f} deg; "
                f"stiff should be much smaller (catches stiff_drive=0 / missing-gain bugs)."
            ),
        )


if __name__ == "__main__":
    unittest.main()
