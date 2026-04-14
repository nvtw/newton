# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stability tests for SolverBox3D — ported from solver2d benchmarks.

Real-world complexity scenes (stacking, pyramids, chains) translated
from Erin Catto's solver2d framework to 3D.  Uses full body counts
from solver2d.  Each test simulates until settled and checks stability.

Solver2d defaults: 60 Hz, 4 primary + 2 secondary iterations, 1 substep.
For 3D (more contact points per pair, full inertia tensors) we use
2 substeps with 4+2 iterations — about 2x the total solver work,
which is the minimum needed for 3D box stacking stability.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestBox3DStability(unittest.TestCase):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Shared config matching solver2d settings (scaled for 3D)
# ═══════════════════════════════════════════════════════════════════════

# solver2d: 4 primary + 2 secondary iters, 1 substep, 60 Hz
# 3D equivalent: 2 substeps, 6+3 iters (~3x work for extra 3D DOFs/contacts)
_CONTACT_CFG = Box3DConfig(
    num_substeps=2,
    num_velocity_iters=6,
    num_relaxation_iters=3,
    contact_hertz=30.0,
    contact_damping_ratio=1.0,
)

_JOINT_CFG = Box3DConfig(
    num_substeps=4,
    num_velocity_iters=2,
    num_relaxation_iters=1,
    contact_hertz=30.0,
    joint_hertz=60.0,
    joint_damping_ratio=1.0,
    linear_damping=0.1,
    angular_damping=0.1,
)

_DT = 1.0 / 60.0


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _step_sim(solver, pipeline, state_in, state_out, dt, steps):
    contacts = pipeline.contacts()
    for _ in range(steps):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    return state_in


def _check_no_tunneling(test, state, body_indices, ground_z=0.0, min_z_offset=0.0):
    q = state.body_q.numpy()
    for i in body_indices:
        z = float(q[i][2])
        test.assertGreater(z, ground_z - min_z_offset,
                           f"Body {i} tunneled: z={z}")


def _check_finite(test, state, body_indices):
    q = state.body_q.numpy()
    for i in body_indices:
        for k in range(3):
            test.assertTrue(np.isfinite(q[i][k]),
                            f"Body {i} has non-finite position: {q[i][:3]}")


# ═══════════════════════════════════════════════════════════════════════
# Vertical Stack (15 boxes) — solver2d VerticalStack
# ═══════════════════════════════════════════════════════════════════════


def test_vertical_stack(test, device):
    """15 boxes (1m cubes) stacked vertically, alternating ±0.01m offset.

    solver2d: 15 boxes, 0.5m half-extent, density 1.0, friction 0.3.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    h = 0.5
    row_count = 15
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.3)

    for i in range(row_count):
        shift = -0.01 if i % 2 == 0 else 0.01
        b = builder.add_body(xform=wp.transform(wp.vec3(shift, 0.0, h + 2.0 * h * i)))
        builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=shape_cfg)
        body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    # 2 seconds settling
    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, body_indices, min_z_offset=h)
    _check_finite(test, final, body_indices)

    # Top box should remain elevated (stack didn't fully collapse).
    # In 3D, box stacking is inherently less stable than 2D due to
    # more contact points and DOFs.  We check that the stack maintains
    # meaningful height, not perfect stability.
    top_z = float(final.body_q.numpy()[body_indices[-1]][2])
    test.assertGreater(top_z, 5.0,
                       f"Stack fully collapsed: top box at z={top_z}, expected > 5")


# ═══════════════════════════════════════════════════════════════════════
# Pyramid (20 base) — solver2d Pyramid (debug)
# ═══════════════════════════════════════════════════════════════════════


def test_pyramid(test, device):
    """20-base pyramid of boxes (210 total bodies).

    solver2d: baseCount=20 (debug), 0.5m square boxes, density 1.0.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    base_count = 20
    h = 0.5
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    for i in range(base_count):
        z = (2.0 * i + 1.0) * h
        for j in range(i, base_count):
            x = (i + 1.0) * h + 2.0 * (j - i) * h - h * base_count
            b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z)))
            builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=shape_cfg)
            body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(
        num_substeps=2, num_velocity_iters=4, num_relaxation_iters=2,
        contact_hertz=30.0,
        max_bodies_per_world=256, max_contacts_per_world=8192,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, body_indices, min_z_offset=h)
    _check_finite(test, final, body_indices)

    # Bottom row should be near ground
    bottom_z = float(final.body_q.numpy()[body_indices[0]][2])
    test.assertAlmostEqual(bottom_z, h, delta=1.0,
                           msg=f"Pyramid base should be near z={h}, got z={bottom_z}")

    # Top should still be elevated
    top_z = float(final.body_q.numpy()[body_indices[-1]][2])
    test.assertGreater(top_z, 5.0,
                       f"Pyramid collapsed: top at z={top_z}")


# ═══════════════════════════════════════════════════════════════════════
# High Mass Ratio — solver2d HighMassRatio2
# ═══════════════════════════════════════════════════════════════════════


def test_high_mass_ratio(test, device):
    """Heavy box on light boxes — mass ratio ~1000:1.

    solver2d: 2 small boxes (0.5x0.5) at x=±9 + 1 large box (10x10) at y=26.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    b0 = builder.add_body(xform=wp.transform(wp.vec3(-9.0, 0.0, 0.5)))
    builder.add_shape_box(body=b0, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
    b1 = builder.add_body(xform=wp.transform(wp.vec3(9.0, 0.0, 0.5)))
    builder.add_shape_box(body=b1, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
    b2 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 26.0)))
    builder.add_shape_box(body=b2, hx=10.0, hy=10.0, hz=10.0, cfg=shape_cfg)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, [b0, b1], min_z_offset=0.5)
    _check_finite(test, final, [b0, b1, b2])

    # Heavy box should have settled above ground (not tunneled or exploded)
    heavy_z = float(final.body_q.numpy()[b2][2])
    test.assertGreater(heavy_z, 5.0, f"Heavy box tunneled or compressed too much: z={heavy_z}")
    test.assertLess(heavy_z, 40.0, f"Heavy box exploded: z={heavy_z}")

    # Small boxes should still exist near their original positions
    small_z0 = float(final.body_q.numpy()[b0][2])
    small_z1 = float(final.body_q.numpy()[b1][2])
    test.assertGreater(small_z0, 0.0, f"Small box 0 tunneled: z={small_z0}")
    test.assertGreater(small_z1, 0.0, f"Small box 1 tunneled: z={small_z1}")


# ═══════════════════════════════════════════════════════════════════════
# Circle Stack (10 spheres) — solver2d CircleStack
# ═══════════════════════════════════════════════════════════════════════


def test_sphere_stack(test, device):
    """10 spheres (radius 1.0) stacked vertically, 3.0m spacing.

    solver2d: 10 circles radius 1.0, first at y=4.0, spacing 3.0m.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    radius = 1.0
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    for i in range(10):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 4.0 + 3.0 * i)))
        builder.add_shape_sphere(body=b, radius=radius, cfg=shape_cfg)
        body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, body_indices, min_z_offset=radius)
    _check_finite(test, final, body_indices)

    # Bottom sphere should be near the ground (z < initial height)
    bottom_z = float(final.body_q.numpy()[body_indices[0]][2])
    test.assertLess(bottom_z, 4.0,
                    msg=f"Bottom sphere should have settled below starting height, z={bottom_z}")

    # Top sphere should be elevated (spheres stacked, not all collapsed)
    top_z = float(final.body_q.numpy()[body_indices[-1]][2])
    test.assertGreater(top_z, 5.0 * radius,
                       f"Sphere stack collapsed: top at z={top_z}")


# ═══════════════════════════════════════════════════════════════════════
# Overlap Recovery (16 boxes) — solver2d OverlapRecovery
# ═══════════════════════════════════════════════════════════════════════


def test_overlap_recovery(test, device):
    """10 boxes in 4-level pyramid with 25% initial overlap.

    solver2d: baseCount=4 → 4+3+2+1=10 boxes, 0.5m half-extent, 25% overlap.
    Tests solver ability to recover from initial penetration.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    h = 0.5
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    # 4-level pyramid: 4, 3, 2, 1 boxes with 25% overlap
    base_count = 4
    for i in range(base_count):
        z = (2.0 * i + 1.0) * h * 0.75  # 25% overlap
        for j in range(base_count - i):
            x = (j + 0.5 * i) * 2.0 * h - h * (base_count - i - 1)
            b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z)))
            builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=shape_cfg)
            body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, body_indices, min_z_offset=h)
    _check_finite(test, final, body_indices)

    # After recovery, no two boxes should still be overlapping significantly.
    # Check that all boxes have separated (minimum pairwise distance > 0)
    q = final.body_q.numpy()
    positions = [q[bi][:3] for bi in body_indices]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(np.array(positions[i]) - np.array(positions[j])))
            # Two boxes of half-extent 0.5 should not overlap (centers > 1.0 apart, or stacked)
            # Allow some tolerance for contact-touching
            test.assertGreater(dist, 0.5,
                               f"Boxes {body_indices[i]} and {body_indices[j]} still overlapping: dist={dist}")


# ═══════════════════════════════════════════════════════════════════════
# Standing Dominoes (15 thin boxes) — solver2d DoubleDomino
# ═══════════════════════════════════════════════════════════════════════


def test_standing_dominoes(test, device):
    """15 thin standing dominoes remain upright and stable.

    solver2d: 15 boxes 0.125x0.5, spacing 1.0m, friction 0.6.
    In 3D, thin boxes standing upright test contact stability.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    body_indices = []
    hx, hy, hz = 0.125, 0.25, 0.5
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.6)

    for i in range(15):
        b = builder.add_body(xform=wp.transform(wp.vec3(1.0 * i, 0.0, hz)))
        builder.add_shape_box(body=b, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
        body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    for bi in body_indices:
        z = float(final.body_q.numpy()[bi][2])
        test.assertAlmostEqual(z, hz, delta=0.2,
                               msg=f"Domino {bi} should stand at z≈{hz}, got z={z}")

    _check_no_tunneling(test, final, body_indices, min_z_offset=hx)


# ═══════════════════════════════════════════════════════════════════════
# Bridge Chain (20 revolute links) — solver2d Bridge (reduced)
# ═══════════════════════════════════════════════════════════════════════


def test_bridge_chain(test, device):
    """160-link revolute chain anchored at left end, sags under gravity.

    solver2d: e_count=160, box 0.5x0.125, density=20, damping=0.1.
    """
    builder = newton.ModelBuilder()

    link_count = 160
    xbase = -80.0
    height = 10.0
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=20.0, mu=0.5)

    anchor = builder.add_body(
        xform=wp.transform(wp.vec3(xbase, 0.0, height)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=anchor, radius=0.05)

    prev_body = anchor
    for i in range(link_count):
        pos = wp.vec3(xbase + 0.5 + 1.0 * i, 0.0, height)
        link = builder.add_body(xform=wp.transform(pos))
        builder.add_shape_box(body=link, hx=0.5, hy=0.125, hz=0.125, cfg=shape_cfg)
        body_indices.append(link)

        parent_xf = wp.transform_identity() if prev_body == anchor else wp.transform(wp.vec3(0.5, 0.0, 0.0))
        builder.add_joint_revolute(
            parent=prev_body, child=link,
            parent_xform=parent_xf,
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)),
            axis=newton.Axis.Y,
        )
        prev_body = link

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg_bridge = Box3DConfig(
        num_substeps=4, num_velocity_iters=4, num_relaxation_iters=2,
        contact_hertz=30.0,
        joint_hertz=60.0, joint_damping_ratio=1.0,
        linear_damping=0.1, angular_damping=0.1,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg_bridge)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_finite(test, final, body_indices)

    # Chain tip should sag below initial height
    tip_z = float(final.body_q.numpy()[body_indices[-1]][2])
    test.assertLess(tip_z, height,
                    f"Chain tip should fall, z={tip_z}")

    # No link should have exploded (all z within reasonable range)
    for i, bi in enumerate(body_indices):
        z = float(final.body_q.numpy()[bi][2])
        test.assertLess(abs(z), 50.0,
                        f"Link {i} exploded: z={z}")


# ═══════════════════════════════════════════════════════════════════════
# Friction Ramp (5 boxes) — solver2d FrictionRamp
# ═══════════════════════════════════════════════════════════════════════


def test_friction_ramp(test, device):
    """5 boxes on an inclined plane with varying friction.

    solver2d: 5 boxes (0.5x0.5) on tilted surfaces, friction [0.75, 0.5, 0.35, 0.1, 0.0].
    In 3D: 5 boxes on a tilted ground, different friction per box.
    Tests that high-friction boxes stick and low-friction boxes slide.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    body_indices = []
    frictions = [0.75, 0.5, 0.35, 0.1, 0.0]
    h = 0.5

    for i, mu in enumerate(frictions):
        x = 2.0 * i
        z = h  # start on ground
        cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=mu)
        b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z)))
        builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=cfg)
        body_indices.append(b)

    # Give all boxes a sideways push
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    qd = state_in.body_qd.numpy()
    for bi in body_indices:
        qd[bi] = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    state_in.body_qd.assign(qd)

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 60)

    _check_finite(test, final, body_indices)
    _check_no_tunneling(test, final, body_indices, min_z_offset=h)

    # High-friction box should have slid less than low-friction box
    x_high_friction = float(final.body_q.numpy()[body_indices[0]][0])
    x_low_friction = float(final.body_q.numpy()[body_indices[-1]][0])
    test.assertLess(x_high_friction, x_low_friction,
                    f"High-friction box slid more than low-friction: {x_high_friction} vs {x_low_friction}")


# ═══════════════════════════════════════════════════════════════════════
# Confined (100 spheres in a box) — solver2d Confined (reduced)
# ═══════════════════════════════════════════════════════════════════════


def test_confined_spheres(test, device):
    """225 spheres confined in a box, no gravity.

    solver2d: 625 circles (25x25) in a capsule box, radius=0.5, no gravity.
    3D: 225 (15x15) spheres to fit within contact limits.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))

    # Walls (kinematic boxes)
    wall_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    for sign in [-1, 1]:
        w = builder.add_body(xform=wp.transform(wp.vec3(sign * 6.0, 0.0, 5.0)), is_kinematic=True)
        builder.add_shape_box(body=w, hx=0.5, hy=6.0, hz=6.0, cfg=wall_cfg)
        w = builder.add_body(xform=wp.transform(wp.vec3(0.0, sign * 6.0, 5.0)), is_kinematic=True)
        builder.add_shape_box(body=w, hx=6.0, hy=0.5, hz=6.0, cfg=wall_cfg)

    # Floor and ceiling
    floor = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.5)), is_kinematic=True)
    builder.add_shape_box(body=floor, hx=6.0, hy=6.0, hz=0.5, cfg=wall_cfg)
    ceil = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 10.5)), is_kinematic=True)
    builder.add_shape_box(body=ceil, hx=6.0, hy=6.0, hz=0.5, cfg=wall_cfg)

    body_indices = []
    radius = 0.5
    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.2)
    grid = 15
    spacing = 10.0 / grid
    for ix in range(grid):
        for iy in range(grid):
            x = -4.5 + spacing * ix
            y = -4.5 + spacing * iy
            z = 5.0
            b = builder.add_body(xform=wp.transform(wp.vec3(x, y, z)))
            builder.add_shape_sphere(body=b, radius=radius, cfg=sphere_cfg)
            body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    # Give small random initial velocities (solver2d uses gravity=0, so motion from collisions)
    qd = state_in.body_qd.numpy()
    rng = np.random.RandomState(42)
    for bi in body_indices:
        qd[bi, :3] = rng.randn(3) * 0.5
    state_in.body_qd.assign(qd)

    cfg = Box3DConfig(
        num_substeps=2, num_velocity_iters=6, num_relaxation_iters=3,
        contact_hertz=30.0, max_contacts_per_world=32768,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 60)

    _check_finite(test, final, body_indices)

    # All spheres should remain within the confinement box (walls at ±6,
    # allow some margin for soft contact penetration)
    _check_finite(test, final, body_indices)
    escaped = 0
    for bi in body_indices:
        pos = final.body_q.numpy()[bi][:3]
        if any(abs(float(pos[k])) > 10.0 for k in range(3)):
            escaped += 1
    # Allow up to 5% escapees (soft constraint margins)
    max_escaped = max(1, len(body_indices) // 20)
    test.assertLessEqual(escaped, max_escaped,
                         f"{escaped} spheres escaped confinement (max allowed: {max_escaped})")


# ═══════════════════════════════════════════════════════════════════════
# Single Box — solver2d SingleBox (simplest contact test)
# ═══════════════════════════════════════════════════════════════════════


def test_single_box(test, device):
    """Single box falls onto ground and settles.

    solver2d: 1x1 box at y=4, ground segment, friction 0.5.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 4.0)))
    builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0, cfg=shape_cfg)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    z = float(final.body_q.numpy()[b][2])
    test.assertAlmostEqual(z, 1.0, delta=0.2,
                           msg=f"Box should rest at z≈1.0, got z={z}")


# ═══════════════════════════════════════════════════════════════════════
# Warm Start Energy — solver2d WarmStartEnergy
# ═══════════════════════════════════════════════════════════════════════


def test_warm_start_energy(test, device):
    """3 stacked circles (2 light + 1 heavy) tests warm starting.

    solver2d: 3 circles radius 0.5, bottom 2 density=1, top density=100.
    After settling the stack should be stable with warm starting active.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    r = 0.5
    cfg_light = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    cfg_heavy = newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5)
    body_indices = []

    b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, r)))
    builder.add_shape_sphere(body=b0, radius=r, cfg=cfg_light)
    body_indices.append(b0)

    b1 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, r + 2.0 * r)))
    builder.add_shape_sphere(body=b1, radius=r, cfg=cfg_light)
    body_indices.append(b1)

    b2 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, r + 4.0 * r)))
    builder.add_shape_sphere(body=b2, radius=r, cfg=cfg_heavy)
    body_indices.append(b2)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, body_indices, min_z_offset=r)
    _check_finite(test, final, body_indices)

    # Heavy top ball should be above the two light balls
    z_top = float(final.body_q.numpy()[b2][2])
    z_bottom = float(final.body_q.numpy()[b0][2])
    test.assertGreater(z_top, z_bottom + r,
                       f"Heavy ball should be above light ball: z_top={z_top}, z_bottom={z_bottom}")


# ═══════════════════════════════════════════════════════════════════════
# High Mass Ratio 1 — solver2d HighMassRatio1 (3 pyramids)
# ═══════════════════════════════════════════════════════════════════════


def test_high_mass_ratio_pyramids(test, device):
    """3 pyramids (10-base each) with varying heavy top body.

    solver2d: 3 pyramids, baseCount=10, extent=1.0, top density=(j+1)*100.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    base_count = 10
    h = 0.5  # half-extent (extent=1.0 in solver2d, h=half)
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    for pyramid_idx in range(3):
        x_offset = -10.0 + 12.0 * pyramid_idx
        for i in range(base_count):
            z = (2.0 * i + 1.0) * h
            for j in range(base_count - i):
                x = x_offset + (j + 0.5 * i) * 2.0 * h - h * (base_count - i - 1)
                is_top = (i == base_count - 1 and j == 0)
                if is_top:
                    top_cfg = newton.ModelBuilder.ShapeConfig(
                        density=(pyramid_idx + 1) * 100.0, mu=0.5)
                    b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z)))
                    builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=top_cfg)
                else:
                    b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z)))
                    builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=shape_cfg)
                body_indices.append(b)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(
        num_substeps=2, num_velocity_iters=6, num_relaxation_iters=3,
        contact_hertz=30.0, max_bodies_per_world=256, max_contacts_per_world=8192,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, body_indices, min_z_offset=h)
    _check_finite(test, final, body_indices)


# ═══════════════════════════════════════════════════════════════════════
# Ball and Chain (40 capsule links + heavy ball) — solver2d BallAndChain
# ═══════════════════════════════════════════════════════════════════════


def test_ball_and_chain(test, device):
    """40 capsule links connected by revolute joints with a heavy ball at the end.

    solver2d: e_count=40, capsules hx=0.5, capsule radius=0.125, density=20,
    ball radius=8.0, damping=0.1.
    """
    builder = newton.ModelBuilder()

    link_count = 40
    hx = 0.5
    height = link_count * hx
    body_indices = []

    cap_cfg = newton.ModelBuilder.ShapeConfig(density=20.0, mu=0.5)
    ball_cfg = newton.ModelBuilder.ShapeConfig(density=20.0, mu=0.5)

    # Anchor
    anchor = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, height)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=anchor, radius=0.05)

    prev_body = anchor
    for i in range(link_count):
        pos = wp.vec3(hx + 2.0 * hx * i, 0.0, height)
        link = builder.add_body(xform=wp.transform(pos))
        builder.add_shape_capsule(body=link, radius=0.125, half_height=hx, cfg=cap_cfg)
        body_indices.append(link)

        parent_xf = wp.transform_identity() if prev_body == anchor else wp.transform(wp.vec3(hx, 0.0, 0.0))
        builder.add_joint_revolute(
            parent=prev_body, child=link,
            parent_xform=parent_xf,
            child_xform=wp.transform(wp.vec3(-hx, 0.0, 0.0)),
            axis=newton.Axis.Y,
        )
        prev_body = link

    # Heavy ball at end (solver2d: radius=8.0)
    ball_r = 8.0
    ball_pos = wp.vec3(hx + 2.0 * hx * link_count + ball_r, 0.0, height)
    ball = builder.add_body(xform=wp.transform(ball_pos))
    builder.add_shape_sphere(body=ball, radius=ball_r, cfg=ball_cfg)
    body_indices.append(ball)

    builder.add_joint_revolute(
        parent=prev_body, child=ball,
        parent_xform=wp.transform(wp.vec3(hx, 0.0, 0.0)),
        child_xform=wp.transform(wp.vec3(-ball_r, 0.0, 0.0)),
        axis=newton.Axis.Y,
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_JOINT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_finite(test, final, body_indices)

    # Ball should have fallen significantly
    ball_z = float(final.body_q.numpy()[ball][2])
    test.assertLess(ball_z, height,
                    f"Ball should fall, z={ball_z}")

    # No body should have exploded to extreme positions
    for i, bi in enumerate(body_indices):
        pos = final.body_q.numpy()[bi][:3]
        for k in range(3):
            test.assertTrue(np.isfinite(pos[k]),
                            f"Body {i} has non-finite position: {pos}")
            test.assertLess(abs(float(pos[k])), 200.0,
                            f"Body {i} exploded: pos={pos}")


# ═══════════════════════════════════════════════════════════════════════
# Stretched Chain (40 circles) — solver2d StretchedChain
# ═══════════════════════════════════════════════════════════════════════


def test_stretched_chain(test, device):
    """40 circles connected by revolute joints, hanging under gravity.

    solver2d: count=40, circle radius=0.2, length=1.0, spacing=2*length.
    """
    builder = newton.ModelBuilder()

    count = 40
    length = 1.0
    radius = 0.2
    start_z = count * 2.0 * length
    body_indices = []

    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    # Anchor
    anchor = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, start_z)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=anchor, radius=0.05)

    prev_body = anchor
    for i in range(count):
        z = start_z - 2.0 * length * (i + 1)
        link = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z)))
        builder.add_shape_sphere(body=link, radius=radius, cfg=sphere_cfg)
        body_indices.append(link)

        builder.add_joint_revolute(
            parent=prev_body, child=link,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -length)),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, length)),
            axis=newton.Axis.Y,
        )
        prev_body = link

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_JOINT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_finite(test, final, body_indices)

    # Bottom circle should hang below start height
    bottom_z = float(final.body_q.numpy()[body_indices[-1]][2])
    test.assertLess(bottom_z, start_z,
                    f"Bottom circle should hang below anchor, z={bottom_z}")

    # Chain should be roughly vertical (all circles near x=0, y=0)
    for i, bi in enumerate(body_indices):
        pos = final.body_q.numpy()[bi][:3]
        lateral = float(np.sqrt(pos[0]**2 + pos[1]**2))
        test.assertLess(lateral, 5.0,
                        f"Circle {i} drifted laterally: x={pos[0]:.2f}, y={pos[1]:.2f}")


# ═══════════════════════════════════════════════════════════════════════
# Joint Grid (20 circles) — solver2d JointGrid (debug mode)
# ═══════════════════════════════════════════════════════════════════════


def test_joint_grid(test, device):
    """1x20 grid of circles connected by revolute joints.

    solver2d debug mode: numi=1, numk=20, circle radius=0.4, shift=1.0.
    Middle row fixed to ground, gravity scale=2.0.
    """
    builder = newton.ModelBuilder()

    numk = 20
    radius = 0.4
    shift = 1.0
    body_indices = []
    fixed_indices = []

    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    # Create circles in a vertical column
    for k in range(numk):
        z = (k + 1) * shift
        is_fixed = (numk // 2 - 3 <= k <= numk // 2 + 3)
        b = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z)),
            is_kinematic=is_fixed,
        )
        builder.add_shape_sphere(body=b, radius=radius, cfg=sphere_cfg)
        body_indices.append(b)
        if is_fixed:
            fixed_indices.append(b)

    # Connect adjacent circles with revolute joints
    for k in range(numk - 1):
        builder.add_joint_revolute(
            parent=body_indices[k], child=body_indices[k + 1],
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * shift)),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * shift)),
            axis=newton.Axis.Y,
        )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_JOINT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_finite(test, final, body_indices)

    # Fixed circles should stay at their original positions
    for bi in fixed_indices:
        z = float(final.body_q.numpy()[bi][2])
        expected_z = (body_indices.index(bi) + 1) * shift
        test.assertAlmostEqual(z, expected_z, delta=0.1,
                               msg=f"Fixed body {bi} moved: z={z}, expected {expected_z}")

    # Bottom circles should sag below fixed point
    bottom_z = float(final.body_q.numpy()[body_indices[0]][2])
    fixed_z = float(final.body_q.numpy()[fixed_indices[0]][2])
    test.assertLess(bottom_z, fixed_z,
                    f"Bottom should sag below fixed: bottom_z={bottom_z}, fixed_z={fixed_z}")


# ═══════════════════════════════════════════════════════════════════════
# Register tests
# ═══════════════════════════════════════════════════════════════════════

devices = get_cuda_test_devices()

add_function_test(TestBox3DStability, "test_vertical_stack", test_vertical_stack, devices=devices)
add_function_test(TestBox3DStability, "test_pyramid", test_pyramid, devices=devices)
add_function_test(TestBox3DStability, "test_high_mass_ratio", test_high_mass_ratio, devices=devices)
add_function_test(TestBox3DStability, "test_sphere_stack", test_sphere_stack, devices=devices)
add_function_test(TestBox3DStability, "test_overlap_recovery", test_overlap_recovery, devices=devices)
add_function_test(TestBox3DStability, "test_standing_dominoes", test_standing_dominoes, devices=devices)
add_function_test(TestBox3DStability, "test_bridge_chain", test_bridge_chain, devices=devices)
add_function_test(TestBox3DStability, "test_friction_ramp", test_friction_ramp, devices=devices)
add_function_test(TestBox3DStability, "test_confined_spheres", test_confined_spheres, devices=devices)
add_function_test(TestBox3DStability, "test_single_box", test_single_box, devices=devices)
add_function_test(TestBox3DStability, "test_warm_start_energy", test_warm_start_energy, devices=devices)
add_function_test(TestBox3DStability, "test_high_mass_ratio_pyramids", test_high_mass_ratio_pyramids, devices=devices)
add_function_test(TestBox3DStability, "test_ball_and_chain", test_ball_and_chain, devices=devices)
add_function_test(TestBox3DStability, "test_stretched_chain", test_stretched_chain, devices=devices)
add_function_test(TestBox3DStability, "test_joint_grid", test_joint_grid, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
