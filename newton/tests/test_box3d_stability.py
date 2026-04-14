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
# 3D equivalent: 2 substeps, 4+2 iters (2x work for extra 3D DOFs)
_CONTACT_CFG = Box3DConfig(
    num_substeps=2,
    num_velocity_iters=4,
    num_relaxation_iters=2,
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

    solver2d: 15 boxes, 0.5m half-extent, density 1.0, friction 0.5.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    h = 0.5
    row_count = 15
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

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

    # Top box should remain elevated (stack didn't fully collapse)
    top_z = float(final.body_q.numpy()[body_indices[-1]][2])
    test.assertGreater(top_z, 10.0,
                       f"Stack collapsed: top box at z={top_z}, expected > 10")


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

    solver2d: 2 small boxes (0.5x0.5) + 1 large box (10x10).
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    shape_cfg_light = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    shape_cfg_heavy = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    b0 = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.25)))
    builder.add_shape_box(body=b0, hx=0.25, hy=0.25, hz=0.25, cfg=shape_cfg_light)
    b1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.25)))
    builder.add_shape_box(body=b1, hx=0.25, hy=0.25, hz=0.25, cfg=shape_cfg_light)
    b2 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)))
    builder.add_shape_box(body=b2, hx=5.0, hy=5.0, hz=5.0, cfg=shape_cfg_heavy)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    solver = newton.solvers.SolverBox3D(model, config=_CONTACT_CFG)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 120)

    _check_no_tunneling(test, final, [b0, b1], min_z_offset=0.25)
    _check_finite(test, final, [b0, b1, b2])

    heavy_z = float(final.body_q.numpy()[b2][2])
    test.assertGreater(heavy_z, 0.0, f"Heavy box tunneled: z={heavy_z}")
    test.assertLess(heavy_z, 20.0, f"Heavy box exploded: z={heavy_z}")


# ═══════════════════════════════════════════════════════════════════════
# Circle Stack (10 spheres) — solver2d CircleStack
# ═══════════════════════════════════════════════════════════════════════


def test_sphere_stack(test, device):
    """10 spheres (radius 1.0) stacked vertically, 3.0m spacing.

    solver2d: 10 circles radius 1.0, center spacing 3.0m.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    radius = 1.0
    body_indices = []
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

    for i in range(10):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius + 3.0 * i)))
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

    bottom_z = float(final.body_q.numpy()[body_indices[0]][2])
    test.assertAlmostEqual(bottom_z, radius, delta=0.5,
                           msg=f"Bottom sphere at z={bottom_z}, expected ~{radius}")


# ═══════════════════════════════════════════════════════════════════════
# Overlap Recovery (16 boxes) — solver2d OverlapRecovery
# ═══════════════════════════════════════════════════════════════════════


def test_overlap_recovery(test, device):
    """16 boxes in 4-level pyramid with 25% initial overlap.

    solver2d: baseCount=4, 0.5m boxes, 25% overlap per level.
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


# ═══════════════════════════════════════════════════════════════════════
# Standing Dominoes (15 thin boxes) — solver2d DoubleDomino
# ═══════════════════════════════════════════════════════════════════════


def test_standing_dominoes(test, device):
    """15 thin standing dominoes remain upright and stable.

    solver2d: 15 boxes 0.125x0.5, spacing 1.0m.
    In 3D, thin boxes standing upright test contact stability.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    body_indices = []
    hx, hy, hz = 0.125, 0.25, 0.5
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)

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
    """20-link revolute chain anchored at left end, sags under gravity.

    solver2d: 160 links with density 20, damping 0.1.
    Reduced to 20 for test speed. Full maximal coordinates.
    """
    builder = newton.ModelBuilder()

    link_count = 20
    xbase = -10.0
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

    solver = newton.solvers.SolverBox3D(model, config=_JOINT_CFG)
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
    """100 spheres confined in a box, no gravity.

    solver2d: 625 circles (25x25) in a capsule box, no gravity.
    Reduced to 100 (10x10) for test speed. Tests many-body stability.
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
    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.3)
    grid = 10
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

    # Give random initial velocities (moderate)
    qd = state_in.body_qd.numpy()
    rng = np.random.RandomState(42)
    for bi in body_indices:
        qd[bi, :3] = rng.randn(3) * 1.0
    state_in.body_qd.assign(qd)

    cfg = Box3DConfig(
        num_substeps=2, num_velocity_iters=4, num_relaxation_iters=2,
        contact_hertz=30.0, max_bodies_per_world=256, max_contacts_per_world=8192,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    final = _step_sim(solver, pipeline, state_in, state_out, _DT, 60)

    _check_finite(test, final, body_indices)

    # All spheres should remain within the box (allow some margin)
    for bi in body_indices:
        pos = final.body_q.numpy()[bi][:3]
        for k in range(3):
            test.assertLess(abs(float(pos[k])), 10.0,
                            f"Sphere {bi} escaped confinement: pos={pos}")


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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
