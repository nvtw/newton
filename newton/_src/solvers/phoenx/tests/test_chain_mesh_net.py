# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Chain-mesh net regression: a sphere dropped onto a net of interlocked
chain rings must not fall through.

Each chain ring is one rigid body whose collision geometry is six
capsules arranged around a circle (a compound body). Adjacent rings
in a chain are rotated 90 degrees about the chain axis so they
interlock; parallel chains are linked by transverse rings, forming a
net. Anchors at the four corners hold the net up. A heavy sphere is
released above the centre and must come to rest on the net rather
than punching through to the ground plane.

The scene reproduces (a scaled-down variant of) ``example_chain_mesh``.
The test runs it under both PhoenX step layouts (``single_world`` and
``multi_world``) and asserts the sphere stays above the ground plane.
The two layouts are physically equivalent for a single-world scene;
divergence between them indicates a bug in the multi-world coloring or
PGS dispatch path.

Runs on CUDA only -- the PhoenX path is GPU-only by design.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

# Topology matches ``example_chain_mesh`` (31 x 15) so the scene
# stresses the same multi-world coloring path the user reported as
# divergent. Each chain ring is one rigid body with 6 capsule shapes
# (compound). Connection rings between adjacent chains turn the
# linear chains into a 2D net.
_RING_RADIUS = 0.05  # m
_TUBE_RADIUS = 0.01  # m
_SEGMENTS_PER_RING = 6
# 15 x 10 keeps the total body count (~210) and substep cost under
# the 16 GB Blackwell budget while preserving the interlocking-net
# topology. 20 x 12 reproduces the user-reported divergence more
# strongly but is too slow for a routine unit test.
_N_RINGS = 15
_N_CHAINS = 10
_SEGMENT_HALF_HEIGHT = _RING_RADIUS * math.sin(math.pi / _SEGMENTS_PER_RING)
_RING_SPACING = 2.0 * (_RING_RADIUS + _TUBE_RADIUS) + 0.005 - 0.05
_DROP_HEIGHT = _TUBE_RADIUS + 0.01
# Ground placed far below the net so "the sphere fell through" can be
# detected by checking the sphere's z at the end of the settle window:
# if it's anywhere near the ground plane, the net failed.
_GROUND_Z = -50.0

_FPS = 120
_SUBSTEPS = 15
_SOLVER_ITERATIONS = 5
_VELOCITY_ITERATIONS = 1
_FRICTION = 0.4
_GRAVITY = 9.81

# Two physics steps per render frame, like the example.
_STEPS_PER_FRAME = 2
# 0.5 s of simulated time -- enough for the sphere to settle into the
# net under gravity (with the example's substep/iteration budget).
_SETTLE_FRAMES = 60


def _add_chain_ring(builder: newton.ModelBuilder, body: int, *, static: bool = False) -> None:
    """Attach 6 capsules around a circle of radius ``_RING_RADIUS`` in
    the body-local x-y plane."""
    cfg = None
    if static:
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
            gap=builder.default_shape_cfg.gap,
        )

    for i in range(_SEGMENTS_PER_RING):
        angle = (2.0 * math.pi * i) / _SEGMENTS_PER_RING
        cx = _RING_RADIUS * math.cos(angle)
        cy = _RING_RADIUS * math.sin(angle)
        qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)
        q = qz * qx
        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(cx, cy, 0.0), q=q),
            radius=_TUBE_RADIUS,
            half_height=_SEGMENT_HALF_HEIGHT,
            cfg=cfg,
        )


def _build_chain_mesh_model() -> tuple[newton.Model, int]:
    """Build the chain-mesh scene. Returns ``(model, sphere_body_idx)``."""
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=1000.0,
        mu=_FRICTION,
        restitution=0.0,
        gap=0.01,
    )

    mb.add_ground_plane(height=_GROUND_Z)

    q_rot_x_90 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)

    # Main chains.
    for c in range(_N_CHAINS):
        y = c * (2.0 * _RING_SPACING)
        for i in range(_N_RINGS):
            x = i * _RING_SPACING
            q = q_rot_x_90 if (i % 2 == 1) else wp.quat_identity()
            is_anchor = (c in (0, _N_CHAINS - 1)) and (i in (0, _N_RINGS - 1))
            xform = wp.transform(p=wp.vec3(x, y, _DROP_HEIGHT), q=q)
            if is_anchor:
                body = mb.add_link(xform=xform)
            else:
                body = mb.add_body(xform=xform)
            _add_chain_ring(mb, body, static=is_anchor)

    # Big sphere falling onto the centre of the net.
    sphere_radius = 0.45
    net_cx = 0.5 * (_N_RINGS - 1) * _RING_SPACING
    net_cy = 0.5 * (_N_CHAINS - 1) * 2.0 * _RING_SPACING
    sphere_body = mb.add_body(
        xform=wp.transform(
            p=wp.vec3(net_cx, net_cy, _DROP_HEIGHT + sphere_radius + _TUBE_RADIUS),
            q=wp.quat_identity(),
        ),
    )
    sphere_cfg = newton.ModelBuilder.ShapeConfig(
        density=100.0,
        mu=_FRICTION,
        restitution=0.0,
        gap=mb.default_shape_cfg.gap,
    )
    mb.add_shape_sphere(sphere_body, radius=sphere_radius, cfg=sphere_cfg)

    # Connection rings between adjacent chains.
    q_rot_y_90 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
    for c in range(_N_CHAINS - 1):
        y_mid = (c + 0.5) * (2.0 * _RING_SPACING)
        for i in range(0, _N_RINGS, 2):
            x = i * _RING_SPACING
            body = mb.add_body(
                xform=wp.transform(p=wp.vec3(x, y_mid, _DROP_HEIGHT), q=q_rot_y_90),
            )
            _add_chain_ring(mb, body)

    model = mb.finalize()
    model.set_gravity((0.0, 0.0, -_GRAVITY))
    return model, sphere_body


def _make_solver(model: newton.Model, step_layout: str):
    return newton.solvers.SolverPhoenX(
        model,
        substeps=_SUBSTEPS,
        solver_iterations=_SOLVER_ITERATIONS,
        velocity_iterations=_VELOCITY_ITERATIONS,
        default_friction=_FRICTION,
        step_layout=step_layout,
    )


def _step_n(model: newton.Model, solver, n_render_frames: int, dt: float):
    """Advance ``n_render_frames`` frames, with ``_STEPS_PER_FRAME``
    physics steps per frame (matches the example)."""
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(n_render_frames):
        for _ in range(_STEPS_PER_FRAME):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
    return s0


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX chain-mesh test requires CUDA.",
)
class TestChainMeshNetCatchesSphere(unittest.TestCase):
    """The sphere must come to rest on the net rather than punch
    through to the ground plane.

    Failure modes this guards against:
      * Multi-world coloring miscolours adjacent chain rings, letting
        the PGS sweep miss correlated impulses and the sphere leaks
        through.
      * Compound-body grouping breaks the ring's 6-capsule clusters
        so the contact column doesn't see all per-shape contacts.
      * Warm-start gather permutation regression: loose match indices
        re-seed lambdas at the wrong contact slot.
    """

    #: Min sphere z at the end of the settle window. The net anchors
    #: hold the chains at ``z = 0.01``; a settled sphere should remain
    #: well above ``z = -1.0``. Anything below means the sphere has
    #: started its terminal-velocity descent toward the ground (placed
    #: far below at ``_GROUND_Z = -50 m``).
    MIN_SPHERE_Z = -1.0

    def _run_layout(self, step_layout: str) -> float:
        """Build the scene, step it, return the final sphere z."""
        dt = 1.0 / _FPS
        model, sphere_idx = _build_chain_mesh_model()
        solver = _make_solver(model, step_layout)
        s = _step_n(model, solver, n_render_frames=_SETTLE_FRAMES, dt=dt)
        body_q = s.body_q.numpy()
        self.assertTrue(np.isfinite(body_q).all(), msg=f"non-finite body_q under {step_layout}")
        return float(body_q[sphere_idx, 2])

    def test_single_world_catches_sphere(self) -> None:
        """Single-world step layout: sphere stays above the ground."""
        z = self._run_layout("single_world")
        self.assertGreater(
            z,
            self.MIN_SPHERE_Z,
            msg=(
                f"single_world: sphere fell through the net to z = {z:.3f} "
                f"(expected > {self.MIN_SPHERE_Z:.3f}; ground at {_GROUND_Z:.1f})"
            ),
        )

    def test_multi_world_catches_sphere(self) -> None:
        """Multi-world step layout: sphere stays above the ground."""
        z = self._run_layout("multi_world")
        self.assertGreater(
            z,
            self.MIN_SPHERE_Z,
            msg=(
                f"multi_world: sphere fell through the net to z = {z:.3f} "
                f"(expected > {self.MIN_SPHERE_Z:.3f}; ground at {_GROUND_Z:.1f})"
            ),
        )

    def test_single_and_multi_world_agree_on_sphere_z(self) -> None:
        """Both step layouts model the same physics; for a single-world
        scene they must produce equivalent sphere trajectories.

        Expected divergence under the user-reported multi-world
        coloring bug: the sphere settles at noticeably different z in
        the two layouts. Tolerance is generous (0.20 m) so PGS
        ordering noise alone won't trip the test, but a coloring bug
        that lets the sphere leak through the net will exceed it.
        """
        z_single = self._run_layout("single_world")
        z_multi = self._run_layout("multi_world")
        diff = abs(z_single - z_multi)
        self.assertLess(
            diff,
            0.20,
            msg=(
                f"single_world settled sphere at z={z_single:.3f}, "
                f"multi_world at z={z_multi:.3f} -- |diff|={diff:.3f} m "
                f"exceeds the 0.20 m tolerance. The two layouts model "
                "the same physics for a 1-world scene; a divergence of "
                "this size suggests a bug in the multi-world coloring "
                "or PGS dispatch path."
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
