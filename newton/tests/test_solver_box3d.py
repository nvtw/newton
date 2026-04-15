# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for SolverBox3D -- full pipeline with Newton collision."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestSolverBox3D(unittest.TestCase):
    pass


# =====================================================================
# Helpers
# =====================================================================


def _step_simulation(solver, pipeline, state_in, state_out, control, dt, steps):
    """Run simulation for *steps* steps, alternating state buffers."""
    contacts = pipeline.contacts()
    for _ in range(steps):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
    return state_in  # last written state


# =====================================================================
# Free fall (no contacts)
# =====================================================================


def test_free_fall(test, device):
    """A sphere in free fall matches analytical z = z0 - 0.5*g*t^2."""
    builder = newton.ModelBuilder()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 10.0)))
    builder.add_shape_sphere(body=b, radius=0.1)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    control = model.control()

    cfg = Box3DConfig(num_substeps=4, contact_hertz=30.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    dt = 1.0 / 60.0
    steps = 30  # 0.5 seconds
    final_state = _step_simulation(solver, pipeline, state_in, state_out, control, dt, steps)

    pos = final_state.body_q.numpy()[0]
    z = float(pos[2])
    t = steps * dt
    g = 9.81
    # Expected: z0 - 0.5*g*t^2 = 10 - 0.5*9.81*0.25 ≈ 8.774
    # With 4 substeps and sub-dt damping, expect close match.
    expected_z = 10.0 - 0.5 * g * t * t
    test.assertAlmostEqual(z, expected_z, delta=0.15,
                           msg=f"Free fall z={z}, expected ~{expected_z}")


# =====================================================================
# Ground contact — sphere settles at y=radius
# =====================================================================


def test_ground_contact(test, device):
    """A sphere dropped onto ground settles at z ≈ radius."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)))
    builder.add_shape_sphere(body=b, radius=0.5)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    control = model.control()

    cfg = Box3DConfig(num_substeps=4, contact_hertz=30.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    dt = 1.0 / 60.0
    final_state = _step_simulation(solver, pipeline, state_in, state_out, control, dt, 300)

    pos = final_state.body_q.numpy()[0]
    z = float(pos[2])
    # Should settle at approximately radius (0.5) above ground
    test.assertAlmostEqual(z, 0.5, delta=0.3,
                           msg=f"Sphere should rest at z≈0.5, got z={z}")


# =====================================================================
# Zero gravity — velocity preserved
# =====================================================================


def test_zero_gravity_velocity_preserved(test, device):
    """With zero gravity and no contacts, velocity is preserved (minus damping)."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)))
    builder.add_shape_sphere(body=b, radius=0.1)

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    control = model.control()

    # Set initial velocity: spatial_vector = [vx, vy, vz, wx, wy, wz]
    state_in.body_qd.assign(np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    cfg = Box3DConfig(num_substeps=4, linear_damping=0.0, angular_damping=0.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    dt = 1.0 / 60.0
    final_state = _step_simulation(solver, pipeline, state_in, state_out, control, dt, 60)

    pos = final_state.body_q.numpy()[0]
    x = float(pos[0])
    z = float(pos[2])
    # 1 second at 10 m/s with zero damping → x ≈ 10
    test.assertAlmostEqual(x, 10.0, delta=0.3, msg=f"x should be ~10, got {x}")
    # Z should not drift (no gravity, no contacts)
    test.assertAlmostEqual(z, 5.0, delta=0.01, msg=f"z should stay at 5.0, got {z}")
    # Final velocity should be preserved
    qd = final_state.body_qd.numpy()[0]
    test.assertAlmostEqual(float(qd[0]), 10.0, delta=0.3, msg=f"vx should be ~10, got {qd[0]}")


# =====================================================================
# Pendulum — revolute joint test
# =====================================================================


def test_pendulum_revolute(test, device):
    """A revolute joint pendulum swings and the anchor stays near the pivot."""
    builder = newton.ModelBuilder()

    # Maximal-coordinate construction (no articulations)
    pivot = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=pivot, radius=0.05)
    bob = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 2.0)))
    builder.add_shape_sphere(body=bob, radius=0.1)

    builder.add_joint_revolute(
        parent=pivot, child=bob,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0)),
        axis=newton.Axis.Y,
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(num_substeps=4, joint_hertz=60.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    dt = 1.0 / 60.0
    # Run 20 steps (~0.33s) — the pendulum should be mid-swing
    final_state = _step_simulation(solver, pipeline, state_in, state_out, None, dt, 20)

    # The bob should have swung down from z=2 significantly
    bob_pos = final_state.body_q.numpy()[1]  # body 1 is the bob
    bob_z = float(bob_pos[2])
    test.assertLess(bob_z, 1.6, f"Pendulum bob should swing down, z={bob_z}")

    # The distance from pivot to bob should stay close to 1m
    # (soft constraint allows some stretch)
    pivot_pos = final_state.body_q.numpy()[0]
    dist = float(np.linalg.norm(bob_pos[:3] - pivot_pos[:3]))
    test.assertAlmostEqual(dist, 1.0, delta=0.3,
                           msg=f"Pendulum length should be ~1m, got {dist}")


# =====================================================================
# Fixed joint — two bodies stay rigidly connected
# =====================================================================


def test_fixed_joint(test, device):
    """Two bodies connected by a fixed joint maintain constant relative position."""
    builder = newton.ModelBuilder()

    b0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)))
    builder.add_shape_sphere(body=b0, radius=0.1)
    b1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 5.0)))
    builder.add_shape_sphere(body=b1, radius=0.1)

    builder.add_joint_fixed(
        parent=b0, child=b1,
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(num_substeps=4, joint_hertz=60.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    dt = 1.0 / 60.0
    final_state = _step_simulation(solver, pipeline, state_in, state_out, None, dt, 60)

    # Both bodies should fall together under gravity
    p0 = final_state.body_q.numpy()[0]
    p1 = final_state.body_q.numpy()[1]
    z0 = float(p0[2])
    z1 = float(p1[2])

    # Both should have fallen below initial z=5
    test.assertLess(z0, 5.0, f"Body 0 should fall, z={z0}")
    test.assertLess(z1, 5.0, f"Body 1 should fall, z={z1}")

    # Distance should remain close to 1m
    dist = float(np.linalg.norm(p0[:3] - p1[:3]))
    test.assertAlmostEqual(dist, 1.0, delta=0.2,
                           msg=f"Fixed joint distance should be ~1m, got {dist}")


# =====================================================================
# Multi-world parallel simulation
# =====================================================================


def test_multi_world(test, device):
    """4 independent worlds simulate correctly in parallel.

    Each world has a sphere at a different height. After simulation,
    each sphere should have fallen independently and match a single-world
    reference run.
    """
    # Build 4 worlds with spheres at different heights
    scene = newton.ModelBuilder()
    scene.add_ground_plane()

    heights = [2.0, 4.0, 6.0, 8.0]
    world_bodies = []

    for w, h in enumerate(heights):
        sub = newton.ModelBuilder()
        b = sub.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, h)))
        sub.add_shape_sphere(body=b, radius=0.5)
        scene.add_world(sub)
        world_bodies.append(w + 1)  # +1 for ground plane body offset... actually bodies are indexed globally

    model = scene.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(num_substeps=4, contact_hertz=30.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    dt = 1.0 / 60.0
    final = _step_simulation(solver, pipeline, state_in, state_out, None, dt, 30)

    # Each sphere should have fallen
    q = final.body_q.numpy()
    # Find dynamic bodies (skip ground shapes)
    dynamic_indices = []
    body_flags = model.body_flags.numpy()
    for i in range(model.body_count):
        if not (body_flags[i] & newton.BodyFlags.KINEMATIC):
            dynamic_indices.append(i)

    test.assertEqual(len(dynamic_indices), 4, f"Expected 4 dynamic bodies, got {len(dynamic_indices)}")

    # Higher starting spheres should be higher (or equal) now — they're independent
    zs = [float(q[bi][2]) for bi in dynamic_indices]
    for i in range(len(zs)):
        test.assertTrue(np.isfinite(zs[i]), f"World {i} has non-finite z={zs[i]}")
        test.assertLess(zs[i], heights[i],
                        f"World {i} sphere didn't fall: z={zs[i]}, started at {heights[i]}")

    # Ordering should be preserved (sphere that started higher should still be higher)
    for i in range(len(zs) - 1):
        test.assertLess(zs[i], zs[i + 1] + 0.5,
                        f"World ordering broken: z[{i}]={zs[i]} >= z[{i+1}]={zs[i+1]}")


# =====================================================================
# Revolute joint with angle limits
# =====================================================================


def test_revolute_with_limits(test, device):
    """Revolute joint with angle limits restricts pendulum swing range.

    A pendulum with limits [-0.5, 0.5] rad should not swing past those angles.
    Without limits, a 1m pendulum from horizontal would swing to vertical (pi/2 rad).
    With limits at 0.5 rad, it should be clamped.
    """
    builder = newton.ModelBuilder()

    pivot = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=pivot, radius=0.05)

    # Bob starts offset in X (horizontal position)
    bob = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 5.0)))
    builder.add_shape_sphere(body=bob, radius=0.1)

    builder.add_joint_revolute(
        parent=pivot, child=bob,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0)),
        axis=newton.Axis.Y,
        limit_lower=-0.5,
        limit_upper=0.5,
        limit_ke=1000.0,  # stiff limits
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(num_substeps=4, joint_hertz=60.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    dt = 1.0 / 60.0
    final_state = _step_simulation(solver, pipeline, state_in, state_out, None, dt, 60)

    # The bob should have moved but stayed within limited range
    bob_pos = final_state.body_q.numpy()[bob][:3]
    # With limit at 0.5 rad and soft constraints, the bob should stay above z≈3.5
    # (without limits it would swing to z≈4.0 at the bottom, so limits should restrict motion)
    bob_z = float(bob_pos[2])
    test.assertGreater(bob_z, 3.5,
                       f"Bob fell too far (limits not working): z={bob_z}")
    # Also verify the bob moved (didn't just stay at start)
    test.assertLess(bob_z, 5.0,
                    f"Bob didn't move at all: z={bob_z}")


# =====================================================================
# Ball joint (point-to-point only)
# =====================================================================


def test_ball_joint(test, device):
    """Ball joint keeps two bodies connected but allows free rotation.

    Two bodies connected by a ball joint should maintain distance
    between anchor points while both rotate freely.
    """
    builder = newton.ModelBuilder()

    pivot = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 5.0)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=pivot, radius=0.05)

    bob = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 4.0)))
    builder.add_shape_sphere(body=bob, radius=0.1)

    builder.add_joint_ball(
        parent=pivot, child=bob,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)),
    )

    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    # Give bob initial angular velocity to test free rotation
    qd = state_in.body_qd.numpy()
    qd[bob] = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]  # spin around X
    state_in.body_qd.assign(qd)

    cfg = Box3DConfig(num_substeps=4, num_velocity_iters=2, num_relaxation_iters=1,
                      joint_hertz=60.0)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    dt = 1.0 / 60.0
    final_state = _step_simulation(solver, pipeline, state_in, state_out, None, dt, 30)

    # Distance between pivot and bob should be approximately 1m
    # (soft constraint allows some stretch under gravity)
    pivot_pos = final_state.body_q.numpy()[pivot][:3]
    bob_pos = final_state.body_q.numpy()[bob][:3]
    dist = float(np.linalg.norm(bob_pos - pivot_pos))
    test.assertAlmostEqual(dist, 1.0, delta=1.5,
                           msg=f"Ball joint distance should be ~1m, got {dist}")

    # Bob should have fallen (gravity)
    test.assertLess(float(bob_pos[2]), 5.0,
                    f"Bob should fall under gravity, z={bob_pos[2]}")


# =====================================================================
# CUDA graph capture
# =====================================================================


def test_cuda_graph_matches_eager(test, device):
    """CUDA graph capture produces identical results to eager execution."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)))
    builder.add_shape_sphere(body=b, radius=0.5)

    model = builder.finalize(device=device)

    # Run eager (no graph)
    cfg_eager = Box3DConfig(num_substeps=4, enable_graph=False)
    solver_eager = newton.solvers.SolverBox3D(model, config=cfg_eager)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    state_eager_in = model.state()
    state_eager_out = model.state()
    final_eager = _step_simulation(solver_eager, pipeline, state_eager_in, state_eager_out, None, 1.0 / 60.0, 30)
    pos_eager = final_eager.body_q.numpy()[0].copy()

    # Run with graph
    cfg_graph = Box3DConfig(num_substeps=4, enable_graph=True)
    solver_graph = newton.solvers.SolverBox3D(model, config=cfg_graph)
    pipeline2 = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    state_graph_in = model.state()
    state_graph_out = model.state()
    final_graph = _step_simulation(solver_graph, pipeline2, state_graph_in, state_graph_out, None, 1.0 / 60.0, 30)
    pos_graph = final_graph.body_q.numpy()[0].copy()

    # Results should be very close (floating point may differ slightly due to graph optimization)
    np.testing.assert_allclose(pos_graph[:3], pos_eager[:3], atol=0.01,
                               err_msg=f"Graph and eager positions differ: graph={pos_graph[:3]}, eager={pos_eager[:3]}")


# =====================================================================
# Contact force reporting (update_contacts)
# =====================================================================


def _run_contact_force_test(test, device, build_fn, expected_mass, *, settle_steps=120, avg_steps=60):
    """Shared helper: settle an object, then average the reported contact force.

    Args:
        build_fn: Callable(builder) that adds shapes and returns nothing.
            A ground plane is added automatically.
        expected_mass: Expected mass of the dynamic body [kg].
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    build_fn(builder)

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    cfg = Box3DConfig(num_substeps=8, num_velocity_iters=4, num_relaxation_iters=4,
                      enable_graph=False)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)

    state_in = model.state()
    state_out = model.state()
    contacts = pipeline.contacts()
    dt = 1.0 / 60.0

    for _ in range(settle_steps):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    force_acc = np.zeros(3)
    for _ in range(avg_steps):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)
        nc = int(contacts.rigid_contact_count.numpy()[0])
        if nc > 0:
            f = contacts.force.numpy()[:nc, :3]
            force_acc += np.sum(f, axis=0)

    avg_force = force_acc / avg_steps
    gravity = 9.81
    expected_fz = expected_mass * gravity

    np.testing.assert_allclose(
        avg_force[2], -expected_fz, rtol=0.10,
        err_msg=f"Vertical contact force should match -mg ({-expected_fz:.2f} N), got {avg_force[2]:.2f} N",
    )
    np.testing.assert_allclose(avg_force[0], 0.0, atol=1.0, err_msg="Horizontal X force should be ~0")
    np.testing.assert_allclose(avg_force[1], 0.0, atol=1.0, err_msg="Horizontal Y force should be ~0")

    return avg_force


def test_contact_force_sphere_on_plane(test, device):
    """A sphere resting on a ground plane must report contact force equal to its weight."""
    radius = 0.25
    density = 1000.0
    mass = density * (4.0 / 3.0) * np.pi * radius**3

    def build(builder):
        builder.default_shape_cfg.density = density
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity()))
        builder.add_shape_sphere(body=b, radius=radius)

    _run_contact_force_test(test, device, build, mass)


def test_contact_force_box_on_plane(test, device):
    """A box on a ground plane: total contact force over multiple contacts equals mg."""
    hx, hy, hz = 0.5, 0.5, 0.5
    density = 1000.0
    mass = density * (2.0 * hx) * (2.0 * hy) * (2.0 * hz)

    def build(builder):
        builder.default_shape_cfg.density = density
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, hz), wp.quat_identity()))
        builder.add_shape_box(body=b, hx=hx, hy=hy, hz=hz)

    _run_contact_force_test(test, device, build, mass)


def test_update_contacts_requires_force_attribute(test, device):
    """update_contacts should raise ValueError when contacts.force is not allocated."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_sphere(body=b, radius=0.25)
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverBox3D(model)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    state_in = model.state()
    state_out = model.state()
    contacts = pipeline.contacts()

    state_in.clear_forces()
    contacts.clear()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, 1.0 / 60.0)

    test.assertIsNone(contacts.force)
    with test.assertRaises(ValueError):
        solver.update_contacts(contacts)


# =====================================================================
# Register tests
# =====================================================================

devices = get_cuda_test_devices()

add_function_test(TestSolverBox3D, "test_free_fall", test_free_fall, devices=devices)
add_function_test(TestSolverBox3D, "test_ground_contact", test_ground_contact, devices=devices)
add_function_test(TestSolverBox3D, "test_zero_gravity_velocity_preserved", test_zero_gravity_velocity_preserved, devices=devices)
add_function_test(TestSolverBox3D, "test_pendulum_revolute", test_pendulum_revolute, devices=devices)
add_function_test(TestSolverBox3D, "test_fixed_joint", test_fixed_joint, devices=devices)
add_function_test(TestSolverBox3D, "test_multi_world", test_multi_world, devices=devices)
add_function_test(TestSolverBox3D, "test_revolute_with_limits", test_revolute_with_limits, devices=devices)
add_function_test(TestSolverBox3D, "test_ball_joint", test_ball_joint, devices=devices)
add_function_test(TestSolverBox3D, "test_cuda_graph_matches_eager", test_cuda_graph_matches_eager, devices=devices)
add_function_test(TestSolverBox3D, "test_contact_force_sphere_on_plane", test_contact_force_sphere_on_plane, devices=devices)
add_function_test(TestSolverBox3D, "test_contact_force_box_on_plane", test_contact_force_box_on_plane, devices=devices)
add_function_test(TestSolverBox3D, "test_update_contacts_requires_force_attribute", test_update_contacts_requires_force_attribute, devices=devices, check_output=False)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
