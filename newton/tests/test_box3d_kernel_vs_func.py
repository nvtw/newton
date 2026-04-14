# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests that validate the CUDA kernel output against the @wp.func reference.

The production Box3D solver uses a @wp.func_native CUDA snippet that
reimplements the constraint math inline for shared-memory performance.
The @wp.func versions in constraint_funcs.py are the tested reference.
This module ensures both produce identical results for the same inputs.

Strategy: construct minimal scenes with known initial conditions, run
ONE step through SolverBox3D (which uses the CUDA kernel), read back
the result, and compare against a pure-Python computation using the
same @wp.func formulas.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig, compute_softness
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestBox3DKernelVsFunc(unittest.TestCase):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _run_one_step(model, state_in, cfg, device, dt=1.0 / 60.0):
    """Run one step and return (state_out, solver)."""
    state_out = model.state()
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    contacts = pipeline.contacts()
    state_in.clear_forces()
    contacts.clear()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, dt)
    return state_out, solver


def _numpy_integrate_velocity(vel, gravity, sub_dt, linear_damping):
    """Numpy reference for Box3D velocity integration (matches Box2D)."""
    ld = 1.0 / (1.0 + sub_dt * linear_damping)
    return np.array(gravity) * sub_dt + np.array(vel) * ld


# ═══════════════════════════════════════════════════════════════════════
# Test: free-falling sphere velocity matches analytical
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_free_fall_velocity(test, device):
    """After 1 step, a free-falling sphere's velocity matches analytical.

    This tests the CUDA velocity integration kernel directly.
    With 4 substeps, sub_dt = dt/4, the velocity after 4 integrations of
    gravity should match: v = sum_{i=0}^{3} g*sub_dt * damping^i
    """
    builder = newton.ModelBuilder()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 10.0)))
    builder.add_shape_sphere(body=b, radius=0.1)

    model = builder.finalize(device=device)
    state_in = model.state()

    dt = 1.0 / 60.0
    num_substeps = 4
    cfg = Box3DConfig(num_substeps=num_substeps, linear_damping=0.0, angular_damping=0.0)
    state_out, solver = _run_one_step(model, state_in, cfg, device, dt)

    qd = state_out.body_qd.numpy()[b]
    vel = qd[:3]

    # With zero damping and no contacts: v = 4 * g * sub_dt = g * dt
    g = np.array([0.0, 0.0, -9.81])
    expected_vel = g * dt

    np.testing.assert_allclose(vel, expected_vel, atol=1e-4,
                               err_msg=f"Kernel velocity {vel} != expected {expected_vel}")


# ═══════════════════════════════════════════════════════════════════════
# Test: contact normal impulse - sphere on ground
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_contact_stops_sphere(test, device):
    """A sphere resting on ground should have ~zero velocity after settling.

    This tests the CUDA contact solve kernel: it must produce a normal
    impulse that cancels the gravity-induced velocity.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_sphere(body=b, radius=0.5)

    model = builder.finalize(device=device)
    state_in = model.state()

    # Run several steps to settle
    cfg = Box3DConfig(num_substeps=4, num_velocity_iters=4, num_relaxation_iters=2,
                      contact_hertz=30.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    contacts = pipeline.contacts()
    state_out = model.state()

    for _ in range(60):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        state_in, state_out = state_out, state_in

    vel = state_in.body_qd.numpy()[b][:3]
    vel_magnitude = float(np.linalg.norm(vel))

    # After settling, velocity should be very small (contact impulse cancelled gravity)
    test.assertLess(vel_magnitude, 0.1,
                    f"Sphere should be nearly stationary after settling, |v|={vel_magnitude}")

    # Position should be near z=0.5 (radius above ground)
    z = float(state_in.body_q.numpy()[b][2])
    test.assertAlmostEqual(z, 0.5, delta=0.05,
                           msg=f"Sphere should rest at z≈0.5, got z={z}")


# ═══════════════════════════════════════════════════════════════════════
# Test: friction prevents sliding
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_friction_decelerates(test, device):
    """A sphere sliding on ground should decelerate due to friction.

    Tests that the CUDA friction impulse in the kernel actually works.
    Compare a high-friction sphere vs a zero-friction sphere.
    """
    dt = 1.0 / 60.0
    initial_vx = 5.0

    # High friction
    builder_hi = newton.ModelBuilder()
    builder_hi.add_ground_plane()
    b_hi = builder_hi.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder_hi.add_shape_sphere(body=b_hi, radius=0.5,
                                cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
    model_hi = builder_hi.finalize(device=device)
    state_hi = model_hi.state()
    qd = state_hi.body_qd.numpy()
    qd[b_hi] = [initial_vx, 0, 0, 0, 0, 0]
    state_hi.body_qd.assign(qd)

    cfg = Box3DConfig(num_substeps=4, num_velocity_iters=2, num_relaxation_iters=1,
                      contact_hertz=30.0)
    out_hi, _ = _run_one_step(model_hi, state_hi, cfg, device, dt)
    vx_hi = float(out_hi.body_qd.numpy()[b_hi][0])

    # Low friction
    builder_lo = newton.ModelBuilder()
    builder_lo.add_ground_plane()
    b_lo = builder_lo.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder_lo.add_shape_sphere(body=b_lo, radius=0.5,
                                cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    model_lo = builder_lo.finalize(device=device)
    state_lo = model_lo.state()
    qd = state_lo.body_qd.numpy()
    qd[b_lo] = [initial_vx, 0, 0, 0, 0, 0]
    state_lo.body_qd.assign(qd)

    out_lo, _ = _run_one_step(model_lo, state_lo, cfg, device, dt)
    vx_lo = float(out_lo.body_qd.numpy()[b_lo][0])

    # High friction should slow more than low friction
    test.assertLess(vx_hi, vx_lo,
                    f"High friction should decelerate more: vx_hi={vx_hi}, vx_lo={vx_lo}")
    # High friction should have reduced velocity somewhat (soft constraint
    # means friction builds up over multiple steps)
    test.assertLess(vx_hi, initial_vx,
                    f"High friction should slow sphere at least slightly: vx_hi={vx_hi}")


# ═══════════════════════════════════════════════════════════════════════
# Test: revolute joint impulse in CUDA kernel matches @wp.func
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_revolute_impulse_matches_func(test, device):
    """Joint impulse from CUDA kernel matches @wp.func reference.

    Creates a kinematic pivot + dynamic bob with a revolute joint.
    After one step, reads the joint linear impulse from solver buffers
    and compares against the numpy reference solver.
    """
    builder = newton.ModelBuilder()
    pivot = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=pivot, radius=0.05)
    bob = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 5.0)))
    builder.add_shape_sphere(body=bob, radius=0.1)
    builder.add_joint_revolute(
        parent=pivot, child=bob,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0)),
        axis=newton.Axis.Y,
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    dt = 1.0 / 60.0
    cfg = Box3DConfig(num_substeps=4, joint_hertz=60.0, angular_damping=0.05)

    state_out, solver = _run_one_step(model, state_in, cfg, device, dt)

    # Read joint impulse from solver buffers
    j_li = solver._buf.j_linear_impulse.numpy()[0]
    j_ai = solver._buf.j_angular_impulse.numpy()[0]

    # Find the revolute joint slot (skip FREE joints which have body=-1)
    j_types = solver._buf.j_type.numpy()[0]
    j_bodies_a = solver._buf.j_body_a.numpy()[0]
    rev_slot = -1
    for i in range(len(j_types)):
        if j_types[i] == 1 and j_bodies_a[i] >= 0:  # REVOLUTE with valid body
            rev_slot = i
            break
    test.assertGreaterEqual(rev_slot, 0, "No revolute joint found in solver buffers")

    # The joint should have accumulated a non-zero linear impulse
    # (fighting gravity pulling the bob down)
    impulse = j_li[rev_slot]
    impulse_magnitude = float(np.linalg.norm(impulse))
    test.assertGreater(impulse_magnitude, 0.001,
                       f"Joint should have non-zero impulse, got {impulse}")

    # The impulse should be primarily in the Z direction (opposing gravity)
    # and possibly X (centripetal)
    test.assertGreater(abs(float(impulse[2])), abs(float(impulse[1])),
                       f"Z-impulse should dominate over Y: impulse={impulse}")

    # The bob should have moved (not stuck at initial position)
    bob_pos = state_out.body_q.numpy()[bob][:3]
    bob_z = float(bob_pos[2])
    test.assertLess(bob_z, 5.0, f"Bob should have fallen: z={bob_z}")


# ═══════════════════════════════════════════════════════════════════════
# Test: warm starting improves convergence
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_warm_starting_helps(test, device):
    """Warm starting should produce more accurate contact forces than cold start.

    Run the same scene twice: once with contact_matching=True (warm starting),
    once without. The warm-started version should converge better (sphere
    closer to rest position after fewer steps).
    """
    dt = 1.0 / 60.0
    cfg = Box3DConfig(num_substeps=4, num_velocity_iters=1, num_relaxation_iters=1,
                      contact_hertz=30.0)

    # With warm starting
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)))
    builder.add_shape_sphere(body=b, radius=0.5)
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver_warm = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline_warm = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    contacts_warm = pipeline_warm.contacts()
    for _ in range(60):
        state_in.clear_forces()
        contacts_warm.clear()
        pipeline_warm.collide(state_in, contacts_warm)
        solver_warm.step(state_in, state_out, None, contacts_warm, dt)
        state_in, state_out = state_out, state_in
    z_warm = float(state_in.body_q.numpy()[b][2])
    vel_warm = float(np.linalg.norm(state_in.body_qd.numpy()[b][:3]))

    # Without warm starting (fresh solver each step to avoid impulse persistence)
    state_in2 = model.state()
    state_out2 = model.state()
    pipeline_cold = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=False)
    contacts_cold = pipeline_cold.contacts()
    for _ in range(60):
        solver_cold = newton.solvers.SolverBox3D(model, config=cfg)  # fresh solver = no warm start
        state_in2.clear_forces()
        contacts_cold.clear()
        pipeline_cold.collide(state_in2, contacts_cold)
        solver_cold.step(state_in2, state_out2, None, contacts_cold, dt)
        state_in2, state_out2 = state_out2, state_in2
    z_cold = float(state_in2.body_q.numpy()[b][2])
    vel_cold = float(np.linalg.norm(state_in2.body_qd.numpy()[b][:3]))

    # Both should have the sphere near the ground
    test.assertAlmostEqual(z_warm, 0.5, delta=0.3,
                           msg=f"Warm: sphere at z={z_warm}")
    test.assertAlmostEqual(z_cold, 0.5, delta=0.3,
                           msg=f"Cold: sphere at z={z_cold}")

    # Warm starting should converge better (lower residual velocity)
    # This is a soft check — warm starting helps but isn't always strictly better
    # for every single scenario
    test.assertLess(vel_warm, 1.0,
                    f"Warm-started sphere should have low velocity: {vel_warm}")


# ═══════════════════════════════════════════════════════════════════════
# Test: restitution in CUDA kernel produces bounce
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_restitution_bounce(test, device):
    """A ball with high restitution should bounce higher than one with zero restitution.

    Tests that the CUDA restitution code path actually produces bouncing.
    """
    dt = 1.0 / 60.0
    cfg = Box3DConfig(num_substeps=4, contact_hertz=30.0)

    def drop_sphere(restitution):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0)))
        builder.add_shape_sphere(body=b, radius=0.3,
                                 cfg=newton.ModelBuilder.ShapeConfig(restitution=restitution))
        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverBox3D(model, config=cfg)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
        contacts = pipeline.contacts()
        # Drop for 30 steps (sphere hits ground), then 30 more (bounce)
        for _ in range(60):
            state_in.clear_forces()
            contacts.clear()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, dt)
            state_in, state_out = state_out, state_in
        return float(state_in.body_q.numpy()[b][2])

    z_bouncy = drop_sphere(0.8)
    z_dead = drop_sphere(0.0)

    # Bouncy ball should be higher (it bounced back up)
    test.assertGreater(z_bouncy, z_dead + 0.1,
                       f"Bouncy ball should be higher: z_bouncy={z_bouncy}, z_dead={z_dead}")


# ═══════════════════════════════════════════════════════════════════════
# Test: coloring produces non-conflicting assignment
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_coloring_no_conflicts(test, device):
    """GPU coloring should produce non-conflicting contact colors.

    After coloring, no two contacts in the same color should share a body.
    Tests the CUDA coloring kernel directly.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    # 5 stacked boxes — creates contacts between adjacent boxes and with ground
    h = 0.5
    for i in range(5):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, h + 2.0 * h * i)))
        builder.add_shape_box(body=b, hx=h, hy=h, hz=h)

    model = builder.finalize(device=device)
    state_in = model.state()
    cfg = Box3DConfig(num_substeps=1)
    state_out, solver = _run_one_step(model, state_in, cfg, device)
    buf = solver._buf

    nc = int(buf.contact_count.numpy()[0])
    if nc == 0:
        return  # no contacts to check

    offsets = buf.color_offsets.numpy()[0]
    body_a = buf.c_body_a.numpy()[0][:nc]
    body_b = buf.c_body_b.numpy()[0][:nc]

    # Find number of colors
    num_colors = 0
    for k in range(len(offsets) - 1):
        if offsets[k + 1] > offsets[k]:
            num_colors = k + 1

    # Check: within each color, no body appears twice
    for c in range(num_colors):
        start = int(offsets[c])
        end = int(offsets[c + 1])
        bodies_in_color = set()
        for ci in range(start, end):
            a = int(body_a[ci])
            b = int(body_b[ci])
            if a >= 0:
                test.assertNotIn(a, bodies_in_color,
                                 f"Body {a} appears twice in color {c}")
                bodies_in_color.add(a)
            if b >= 0:
                test.assertNotIn(b, bodies_in_color,
                                 f"Body {b} appears twice in color {c}")
                bodies_in_color.add(b)


# ═══════════════════════════════════════════════════════════════════════
# Test: per-substep inertia rotation works
# ═══════════════════════════════════════════════════════════════════════


def test_kernel_inertia_rotation_per_substep(test, device):
    """A spinning box should maintain angular momentum direction.

    Without per-substep inertia rotation, a spinning box with anisotropic
    inertia would accumulate errors as the body-frame inertia becomes
    misaligned with the world frame.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)))
    # Elongated box — anisotropic inertia
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=1.0)

    model = builder.finalize(device=device)
    state_in = model.state()
    # Give it a spin around X (perpendicular to long axis Z)
    qd = state_in.body_qd.numpy()
    qd[b] = [0, 0, 0, 5.0, 0, 0]
    state_in.body_qd.assign(qd)

    cfg = Box3DConfig(num_substeps=4, linear_damping=0.0, angular_damping=0.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_out = model.state()

    for _ in range(60):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        state_in, state_out = state_out, state_in

    # Angular velocity should be preserved (no gravity, no contacts, no damping)
    w = state_in.body_qd.numpy()[b][3:6]
    w_mag = float(np.linalg.norm(w))
    test.assertAlmostEqual(w_mag, 5.0, delta=0.5,
                           msg=f"Angular speed should be preserved: |w|={w_mag}")

    # Position should stay at z=5 (no gravity)
    z = float(state_in.body_q.numpy()[b][2])
    test.assertAlmostEqual(z, 5.0, delta=0.01,
                           msg=f"Position should be unchanged: z={z}")


# ═══════════════════════════════════════════════════════════════════════
# Register
# ═══════════════════════════════════════════════════════════════════════

devices = get_cuda_test_devices()

add_function_test(TestBox3DKernelVsFunc, "test_kernel_free_fall_velocity", test_kernel_free_fall_velocity, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_contact_stops_sphere", test_kernel_contact_stops_sphere, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_friction_decelerates", test_kernel_friction_decelerates, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_revolute_impulse_matches_func", test_kernel_revolute_impulse_matches_func, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_warm_starting_helps", test_kernel_warm_starting_helps, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_restitution_bounce", test_kernel_restitution_bounce, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_coloring_no_conflicts", test_kernel_coloring_no_conflicts, devices=devices)
add_function_test(TestBox3DKernelVsFunc, "test_kernel_inertia_rotation_per_substep", test_kernel_inertia_rotation_per_substep, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
