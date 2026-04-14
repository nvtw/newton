# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for SolverBox3D — full pipeline with Newton collision."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestSolverBox3D(unittest.TestCase):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
# Free fall (no contacts)
# ═══════════════════════════════════════════════════════════════════════


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
    # Expected: z0 - 0.5*g*t^2 = 10 - 0.5*9.81*0.25 = 10 - 1.226 ≈ 8.774
    expected_z = 10.0 - 0.5 * g * t * t
    test.assertAlmostEqual(z, expected_z, delta=0.5,
                           msg=f"Free fall z={z}, expected ~{expected_z}")


# ═══════════════════════════════════════════════════════════════════════
# Ground contact — sphere settles at y=radius
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
# Zero gravity — velocity preserved
# ═══════════════════════════════════════════════════════════════════════


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
    # 1 second at 10 m/s → x ≈ 10
    test.assertGreater(x, 8.0, f"x should be ~10, got {x}")


# ═══════════════════════════════════════════════════════════════════════
# Register tests
# ═══════════════════════════════════════════════════════════════════════

devices = get_cuda_test_devices()

add_function_test(TestSolverBox3D, "test_free_fall", test_free_fall, devices=devices)
add_function_test(TestSolverBox3D, "test_ground_contact", test_ground_contact, devices=devices)
add_function_test(TestSolverBox3D, "test_zero_gravity_velocity_preserved", test_zero_gravity_velocity_preserved, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
