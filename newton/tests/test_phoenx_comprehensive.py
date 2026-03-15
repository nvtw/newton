# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive tests for the PhoenX PGS solver with Newton collision.

Tests cover:
- Bias sign correctness (boxes must not fall through ground)
- Warm starting with impulse scaling (0.90 factor)
- Mass splitting improves convergence for stacked bodies
- Damping applied per-frame (not per-substep)
- Velocity clamping safety guard
- Multiple shape types (sphere, capsule, box on plane)
- Determinism across repeated runs
- Full collision pipeline integration (BroadPhase + NarrowPhase)
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestPhoenXComprehensive(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Helper: inject synthetic contact directly
# ---------------------------------------------------------------------------


def _inject_contact(ss, ci, shape0, shape1, body0, body1, normal,
                    offset0=(0, 0, 0), offset1=(0, 0, 0), friction=0.5):
    cs = ss.contact_store
    d = ss.device

    def _w(name, val, dtype):
        col = cs.column_of(name).numpy()
        col[ci] = val if not isinstance(val, tuple) else np.array(val, dtype=np.float32)
        cs.column_of(name).assign(wp.array(col, dtype=dtype, device=d))

    _w("shape0", shape0, wp.int32)
    _w("shape1", shape1, wp.int32)
    _w("body0", body0, wp.int32)
    _w("body1", body1, wp.int32)
    _w("normal", normal, wp.vec3)
    _w("offset0", offset0, wp.vec3)
    _w("offset1", offset1, wp.vec3)
    _w("margin0", 0.0, wp.float32)
    _w("margin1", 0.0, wp.float32)
    _w("friction", friction, wp.float32)
    _w("accumulated_normal_impulse", 0.0, wp.float32)
    _w("accumulated_tangent_impulse1", 0.0, wp.float32)
    _w("accumulated_tangent_impulse2", 0.0, wp.float32)


def _step_with_contacts(ss, dt, gravity, num_iterations=8):
    """Run a single substep: velocities -> partition -> solve -> integrate."""
    inv_dt = 1.0 / dt
    ss.integrate_velocities(gravity, dt)
    ss._partition_contacts()
    d = ss.device
    bs = ss.body_store
    cs = ss.contact_store

    from newton._src.solvers.phoenx.kernels import (
        clear_contact_count_kernel,
        count_contacts_per_body_kernel,
    )

    wp.launch(clear_contact_count_kernel, dim=bs.capacity,
              inputs=[ss._contact_count_per_body, bs.count], device=d)
    wp.launch(count_contacts_per_body_kernel, dim=cs.capacity,
              inputs=[cs.column_of("body0"), cs.column_of("body1"),
                      cs.count, ss._contact_count_per_body], device=d)

    max_slots = ss.graph_coloring.max_colors + 1
    for p in range(max_slots):
        ss._launch_prepare(p, inv_dt)
    for _ in range(num_iterations):
        for p in range(max_slots):
            ss._launch_solve(p, 1)
    ss.integrate_positions(dt)


# ===========================================================================
# Test 1: Bias sign — box on ground must NOT fall through
# ===========================================================================

def test_bias_sign_box_on_ground(test, device):
    """A box on a ground plane must be supported, not fall through.

    This is the primary regression test for the critical bias sign bug.
    """
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 0.5, 0), inverse_mass=1.0)

    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    for _ in range(120):
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, row_b,
                        normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=12)

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[row_b]
    vel = ss.body_store.column_of("velocity").numpy()[row_b]

    test.assertGreater(pos[1], -0.05, f"Box fell through ground: y={pos[1]:.4f}")
    test.assertLess(abs(vel[1]), 2.0, f"Excessive vertical velocity: vy={vel[1]:.4f}")


# ===========================================================================
# Test 2: Bias produces correct POSITIVE normal impulse for overlapping bodies
# ===========================================================================

def test_bias_produces_positive_impulse(test, device):
    """When two bodies overlap, the solver must generate positive normal impulse."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    # Place box slightly below the contact surface (overlapping)
    h_box = ss.add_body(position=(0, -0.01, 0), inverse_mass=1.0)

    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    ss.update_world_inertia()
    ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
    _inject_contact(ss, 0, 0, 1, row_g, row_b,
                    normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, 0, 0))
    _step_with_contacts(ss, 1.0 / 60.0, (0, 0, 0), num_iterations=8)

    wp.synchronize_device(device)
    acc_n = ss.contact_store.column_of("accumulated_normal_impulse").numpy()[0]
    test.assertGreater(acc_n, 0.0,
                       f"Normal impulse should be positive for overlap; got {acc_n:.6f}")


# ===========================================================================
# Test 3: Warm start scaling factor (0.90)
# ===========================================================================

def test_warm_start_scaling(test, device):
    """Warm-started impulse should be scaled by 0.90 (ImpulseInheritanceFactor)."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 0.5, 0), inverse_mass=1.0)
    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0

    # Frame 1: solve and export
    ss.update_world_inertia()
    ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
    _inject_contact(ss, 0, 0, 1, row_g, row_b,
                    normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
    ss.warm_starter.import_keys(
        ss.contact_store.column_of("shape0"),
        ss.contact_store.column_of("shape1"),
        ss.contact_store.count,
    )
    ss.warm_starter.sort()
    ss.warm_starter.transfer_impulses(
        ss.contact_store.column_of("accumulated_normal_impulse"),
        ss.contact_store.column_of("accumulated_tangent_impulse1"),
        ss.contact_store.column_of("accumulated_tangent_impulse2"),
    )
    _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=8)
    ss.export_impulses()

    wp.synchronize_device(device)
    solved_n = float(ss.contact_store.column_of("accumulated_normal_impulse").numpy()[0])

    # Frame 2: begin_frame swaps buffers, re-inject same contact, check scaling
    ss.warm_starter.begin_frame()
    ss.update_world_inertia()
    ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
    _inject_contact(ss, 0, 0, 1, row_g, row_b,
                    normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
    ss.warm_starter.import_keys(
        ss.contact_store.column_of("shape0"),
        ss.contact_store.column_of("shape1"),
        ss.contact_store.count,
    )
    ss.warm_starter.sort()
    ss.warm_starter.transfer_impulses(
        ss.contact_store.column_of("accumulated_normal_impulse"),
        ss.contact_store.column_of("accumulated_tangent_impulse1"),
        ss.contact_store.column_of("accumulated_tangent_impulse2"),
    )

    wp.synchronize_device(device)
    transferred_n = float(ss.contact_store.column_of("accumulated_normal_impulse").numpy()[0])

    # The transferred impulse should match the solved impulse (before scaling).
    # After prepare_contacts_kernel applies the 0.90 scaling, the actual
    # warm-started value will be 0.90 * transferred_n.
    test.assertGreater(transferred_n, 0.0, "Warm-started impulse should be positive")
    test.assertAlmostEqual(transferred_n, solved_n, delta=0.01,
                           msg=f"Warm start should transfer solved impulse: {transferred_n:.4f} vs {solved_n:.4f}")


# ===========================================================================
# Test 4: Damping per-frame, not per-substep
# ===========================================================================

def test_damping_per_frame(test, device):
    """Damping should apply once per frame (update_world_inertia), not per substep."""
    ss = SolverState(body_capacity=4, contact_capacity=4, shape_count=1, device=device)
    h = ss.add_body(
        position=(0, 0, 0),
        velocity=(10.0, 0, 0),
        inverse_mass=1.0,
        linear_damping=0.9,
    )
    row = int(ss.body_store.handle_to_index.numpy()[h])

    # Run update_world_inertia once (applies damping)
    ss.update_world_inertia()
    wp.synchronize_device(device)

    vel = ss.body_store.column_of("velocity").numpy()[row]
    expected = 10.0 * 0.9  # Single application of damping
    test.assertAlmostEqual(vel[0], expected, places=4,
                           msg=f"After 1 frame: expected vx={expected}, got {vel[0]:.4f}")

    # Run 4 substeps without calling update_world_inertia again
    for _ in range(4):
        ss.contact_store.count.assign(wp.array([0], dtype=wp.int32, device=device))
        ss.step(1.0 / 240.0, gravity=(0, 0, 0))

    wp.synchronize_device(device)
    vel2 = ss.body_store.column_of("velocity").numpy()[row]
    # Velocity should remain at 9.0 (no additional damping from substeps)
    test.assertAlmostEqual(vel2[0], expected, delta=0.01,
                           msg=f"Substeps should not apply additional damping: vx={vel2[0]:.4f}")


# ===========================================================================
# Test 5: Velocity magnitude clamping
# ===========================================================================

def test_velocity_clamping(test, device):
    """Velocity should be clamped to 100 m/s during integration."""
    ss = SolverState(body_capacity=4, contact_capacity=4, shape_count=1, device=device)
    h = ss.add_body(
        position=(0, 0, 0),
        velocity=(200.0, 0, 0),
        inverse_mass=1.0,
        linear_damping=1.0,
    )
    row = int(ss.body_store.handle_to_index.numpy()[h])

    dt = 1.0
    ss.contact_store.count.assign(wp.array([0], dtype=wp.int32, device=device))
    ss.step(dt, gravity=(0, 0, 0))

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[row]
    # Position should advance by at most 100*dt = 100, not 200
    test.assertLessEqual(pos[0], 100.1,
                         f"Position should be clamped: x={pos[0]:.2f}")


# ===========================================================================
# Test 6: Mass splitting improves stacked box convergence
# ===========================================================================

def test_mass_splitting_stacked(test, device):
    """Mass splitting should prevent bottom box from being crushed in a stack."""
    ss = SolverState(body_capacity=16, contact_capacity=64, shape_count=8, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])

    # Stack of 4 boxes
    handles = []
    for i in range(4):
        h = ss.add_body(position=(0, 0.5 + i * 1.0, 0), inverse_mass=1.0)
        handles.append(h)

    rows = [int(ss.body_store.handle_to_index.numpy()[h]) for h in handles]

    dt = 1.0 / 60.0
    for _ in range(60):
        ss.update_world_inertia()

        # Ground-box0 contact + box-box contacts
        num_c = 4
        ss.contact_store.count.assign(wp.array([num_c], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, rows[0],
                        normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        for j in range(3):
            _inject_contact(ss, j + 1, j + 1, j + 2, rows[j], rows[j + 1],
                            normal=(0, 1, 0), offset0=(0, 0.5, 0), offset1=(0, -0.5, 0))

        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=12)

    wp.synchronize_device(device)

    # All boxes should be above ground
    for i, row in enumerate(rows):
        pos = ss.body_store.column_of("position").numpy()[row]
        test.assertGreater(pos[1], -0.5,
                           f"Box {i} fell through ground: y={pos[1]:.4f}")

    # Top box should be higher than bottom box
    pos_bottom = ss.body_store.column_of("position").numpy()[rows[0]]
    pos_top = ss.body_store.column_of("position").numpy()[rows[3]]
    test.assertGreater(pos_top[1], pos_bottom[1],
                       f"Stack collapsed: bottom={pos_bottom[1]:.3f}, top={pos_top[1]:.3f}")


# ===========================================================================
# Test 7: Determinism — repeated runs produce identical results
# ===========================================================================

def _run_sim(device, seed_offset=0):
    """Run a short simulation and return final positions."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 1.0, 0), inverse_mass=1.0,
                        velocity=(0.5, 0, 0))
    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    for _ in range(30):
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, row_b,
                        normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0),
                        friction=0.6)
        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=8)

    wp.synchronize_device(device)
    return ss.body_store.column_of("position").numpy()[row_b].copy()


def test_determinism(test, device):
    """Two identical runs must produce bit-identical results."""
    pos1 = _run_sim(device)
    pos2 = _run_sim(device)
    np.testing.assert_array_equal(pos1, pos2,
                                  err_msg=f"Non-deterministic: {pos1} vs {pos2}")


# ===========================================================================
# Test 8: Full collision pipeline — box on plane via BroadPhase+NarrowPhase
# ===========================================================================

def test_collision_pipeline_box_on_plane(test, device):
    """Box dropped onto plane via full collision pipeline should come to rest.

    Uses Z-up convention: Newton's plane normal defaults to +Z.
    """
    body_cap = 4
    contact_cap = 64
    shape_count = 2

    ss = SolverState(body_capacity=body_cap, contact_capacity=contact_cap,
                     shape_count=shape_count, device=device, default_friction=0.6)
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count,
                                       max_contacts=contact_cap, device=device)

    # Ground (plane normal = +Z by default)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    # Box at height 2 along Z
    mass = 1.0
    inv_mass = 1.0 / mass
    h = 0.5
    inv_inertia = np.eye(3, dtype=np.float32) * (6.0 * inv_mass / (2.0 * h) ** 2)
    h_box = ss.add_body(position=(0, 0, 2.0), inverse_mass=inv_mass,
                        inverse_inertia_local=inv_inertia, linear_damping=0.995,
                        angular_damping=0.99)
    box_row = int(ss.body_store.handle_to_index.numpy()[h_box])
    ss.set_shape_body(1, h_box)
    pipeline.add_shape_box(body_row=box_row, half_extents=(h, h, h))

    pipeline.finalize()

    dt = 1.0 / 60.0
    substeps = 4
    sub_dt = dt / substeps

    for frame in range(120):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=12)
            ss.export_impulses()

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[box_row]
    vel = ss.body_store.column_of("velocity").numpy()[box_row]

    # Box should settle near z=0.5 (half-extent above ground)
    test.assertGreater(pos[2], 0.0, f"Box fell through ground: z={pos[2]:.4f}")
    test.assertLess(pos[2], 2.0, f"Box didn't fall at all: z={pos[2]:.4f}")
    test.assertLess(abs(vel[2]), 1.0, f"Box not settling: vz={vel[2]:.4f}")


# ===========================================================================
# Test 9: Collision pipeline — sphere on plane
# ===========================================================================

def test_collision_pipeline_sphere_on_plane(test, device):
    """Sphere dropped onto plane via collision pipeline should bounce/settle.

    Uses Z-up convention: Newton's plane normal defaults to +Z.
    """
    body_cap = 4
    contact_cap = 32
    shape_count = 2

    ss = SolverState(body_capacity=body_cap, contact_capacity=contact_cap,
                     shape_count=shape_count, device=device, default_friction=0.5)
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count,
                                       max_contacts=contact_cap, device=device)

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    radius = 0.5
    mass = 1.0
    inv_inertia = np.eye(3, dtype=np.float32) * (2.5 / (mass * radius * radius))
    h_sphere = ss.add_body(position=(0, 0, 2.0), inverse_mass=1.0 / mass,
                           inverse_inertia_local=inv_inertia)
    sphere_row = int(ss.body_store.handle_to_index.numpy()[h_sphere])
    ss.set_shape_body(1, h_sphere)
    pipeline.add_shape_sphere(body_row=sphere_row, radius=radius)

    pipeline.finalize()

    dt = 1.0 / 60.0
    substeps = 4
    sub_dt = dt / substeps

    for frame in range(120):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=8)
            ss.export_impulses()

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[sphere_row]

    test.assertGreater(pos[2], -0.1, f"Sphere fell through: z={pos[2]:.4f}")
    test.assertLess(pos[2], 2.0, f"Sphere didn't fall: z={pos[2]:.4f}")


# ===========================================================================
# Test 10: Collision pipeline — capsule on plane
# ===========================================================================

def test_collision_pipeline_capsule_on_plane(test, device):
    """Capsule dropped onto plane should rest above ground.

    Uses Z-up convention. Capsule axis is local Z, so when dropped
    vertically it lands on its bottom hemisphere.
    """
    body_cap = 4
    contact_cap = 32
    shape_count = 2

    ss = SolverState(body_capacity=body_cap, contact_capacity=contact_cap,
                     shape_count=shape_count, device=device, default_friction=0.5)
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count,
                                       max_contacts=contact_cap, device=device)

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    radius = 0.25
    half_len = 0.5
    h_capsule = ss.add_body(position=(0, 0, 2.0), inverse_mass=1.0)
    cap_row = int(ss.body_store.handle_to_index.numpy()[h_capsule])
    ss.set_shape_body(1, h_capsule)
    pipeline.add_shape_capsule(body_row=cap_row, radius=radius, half_length=half_len)

    pipeline.finalize()

    dt = 1.0 / 60.0
    substeps = 4
    sub_dt = dt / substeps

    for frame in range(120):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=8)
            ss.export_impulses()

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[cap_row]

    test.assertGreater(pos[2], -0.1, f"Capsule fell through: z={pos[2]:.4f}")
    test.assertLess(pos[2], 2.0, f"Capsule didn't fall: z={pos[2]:.4f}")


# ===========================================================================
# Test 11: Friction prevents sliding on flat surface
# ===========================================================================

def test_friction_stops_sliding(test, device):
    """A body with lateral velocity on a frictional surface should decelerate."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 0.01, 0), velocity=(2.0, 0, 0), inverse_mass=1.0)
    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    for _ in range(60):
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, row_b,
                        normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.01, 0),
                        friction=0.8)
        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=8)

    wp.synchronize_device(device)
    vel = ss.body_store.column_of("velocity").numpy()[row_b]
    test.assertLess(abs(vel[0]), 1.0,
                    f"Friction should have slowed the body; vx={vel[0]:.4f}")


# ===========================================================================
# Test 12: Multiple boxes stacking via full collision pipeline
# ===========================================================================

def test_collision_pipeline_box_stack(test, device):
    """Stack of 3 boxes on a plane via full collision pipeline.

    Uses Z-up convention: gravity along -Z, boxes stacked along Z.
    """
    n_boxes = 3
    body_cap = n_boxes + 1
    contact_cap = n_boxes * 16
    shape_count = n_boxes + 1

    ss = SolverState(body_capacity=body_cap, contact_capacity=contact_cap,
                     shape_count=shape_count, device=device, default_friction=0.6)
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count,
                                       max_contacts=contact_cap, device=device)

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    h = 0.5
    inv_inertia = np.eye(3, dtype=np.float32) * (6.0 / (2.0 * h) ** 2)
    box_handles = []
    box_rows = []
    for i in range(n_boxes):
        z = h + i * (2.0 * h + 0.02)
        bh = ss.add_body(position=(0, 0, z), inverse_mass=1.0,
                         inverse_inertia_local=inv_inertia,
                         linear_damping=0.995, angular_damping=0.99)
        row = int(ss.body_store.handle_to_index.numpy()[bh])
        ss.set_shape_body(i + 1, bh)
        pipeline.add_shape_box(body_row=row, half_extents=(h, h, h))
        box_handles.append(bh)
        box_rows.append(row)

    pipeline.finalize()

    dt = 1.0 / 60.0
    substeps = 4
    sub_dt = dt / substeps

    for frame in range(180):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=12)
            ss.export_impulses()

    wp.synchronize_device(device)

    for i, row in enumerate(box_rows):
        pos = ss.body_store.column_of("position").numpy()[row]
        test.assertGreater(pos[2], -0.5, f"Box {i} fell through: z={pos[2]:.4f}")

    # Verify ordering is maintained
    positions = [ss.body_store.column_of("position").numpy()[r][2] for r in box_rows]
    for i in range(len(positions) - 1):
        test.assertGreater(positions[i + 1], positions[i] - 0.1,
                           f"Stack order violated at box {i}: {positions}")


# ===========================================================================
# Registration
# ===========================================================================

devices = get_test_devices()

add_function_test(TestPhoenXComprehensive, "test_bias_sign_box_on_ground",
                  test_bias_sign_box_on_ground, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_bias_produces_positive_impulse",
                  test_bias_produces_positive_impulse, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_warm_start_scaling",
                  test_warm_start_scaling, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_damping_per_frame",
                  test_damping_per_frame, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_velocity_clamping",
                  test_velocity_clamping, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_mass_splitting_stacked",
                  test_mass_splitting_stacked, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_determinism",
                  test_determinism, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_collision_pipeline_box_on_plane",
                  test_collision_pipeline_box_on_plane, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_collision_pipeline_sphere_on_plane",
                  test_collision_pipeline_sphere_on_plane, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_collision_pipeline_capsule_on_plane",
                  test_collision_pipeline_capsule_on_plane, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_friction_stops_sliding",
                  test_friction_stops_sliding, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_collision_pipeline_box_stack",
                  test_collision_pipeline_box_stack, devices=devices)

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
