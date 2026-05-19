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
# Topology matches ``example_chain_mesh`` (31 x 15) so the scene
# stresses the multi-world coloring path on the exact graph shape the
# user reported as divergent. Each chain ring is one rigid body with
# 6 capsule shapes (compound). The SAP broad-phase keeps memory
# bounded at this body count -- the default NxN broad-phase blows up
# quadratically.
_N_RINGS = 31
_N_CHAINS = 15
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

# 300 physics steps at dt = 1/120 = 2.5 s simulated time. PhoenX
# handles ``_SUBSTEPS = 15`` internally per step, matching
# ``example_chain_mesh.py``. With graph capture this is a few hundred
# milliseconds wallclock on Blackwell.
_TOTAL_STEPS = 300


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


def _make_solver_and_pipeline(model: newton.Model, step_layout: str):
    """Attach an explicit SAP-broad-phase ``CollisionPipeline`` to the
    model BEFORE constructing the solver. Otherwise ``SolverPhoenX``
    builds its own pipeline internally without ``broad_phase="sap"``,
    falling back to the O(N^2) NxN broad-phase that OOMs on the
    full-resolution chain mesh."""
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="sticky",
        broad_phase="sap",
    )
    model._collision_pipeline = pipeline
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=_SUBSTEPS,
        solver_iterations=_SOLVER_ITERATIONS,
        velocity_iterations=_VELOCITY_ITERATIONS,
        default_friction=_FRICTION,
        step_layout=step_layout,
    )
    return solver, pipeline, contacts


def _step_n(model: newton.Model, solver, pipeline, contacts, total_steps: int, dt: float):
    """Advance ``total_steps`` physics steps using CUDA graph capture.

    PhoenX runs ``_SUBSTEPS = 15`` substeps inside each ``solver.step``
    call (configured at solver construction). The captured graph
    therefore covers one full physics step including its substep
    loop -- replay is a single ``capture_launch`` per step.

    Captures a *two-step pair* (s0 -> s1 then s1 -> s0) so the
    in-place state swap lands at a fixed point after each replay;
    ``s0`` is therefore always the post-replay output."""
    device = wp.get_device()
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    def _eager_pair() -> None:
        # Two paired steps so s0 / s1 return to their starting roles.
        for s_in, s_out in ((s0, s1), (s1, s0)):
            s_in.clear_forces()
            model.collide(s_in, contacts=contacts, collision_pipeline=pipeline)
            solver.step(s_in, s_out, control, contacts, dt)

    if not device.is_cuda or total_steps < 4:
        for _ in range(total_steps // 2):
            _eager_pair()
        return s0

    # Warm-up: one paired step (2 physics steps) to JIT every kernel
    # before capture; otherwise the first capture would record JIT-time
    # allocations into the graph and replay would crash.
    _eager_pair()

    with wp.ScopedCapture(device=device) as capture:
        _eager_pair()
    graph = capture.graph

    # 4 steps already consumed (2 warm-up + 2 capture). Replay the
    # remainder in pairs.
    consumed = 4
    remaining = max(0, total_steps - consumed)
    for _ in range(remaining // 2):
        wp.capture_launch(graph)
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
        solver, pipeline, contacts = _make_solver_and_pipeline(model, step_layout)
        s = _step_n(model, solver, pipeline, contacts, total_steps=_TOTAL_STEPS, dt=dt)
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
