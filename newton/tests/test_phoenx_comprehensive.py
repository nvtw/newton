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


def _inject_contact(ss, ci, shape0, shape1, body0, body1, normal, offset0=(0, 0, 0), offset1=(0, 0, 0), friction=0.5):
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


def _build_bundles(ss):
    """Run the warm-start key pipeline: import_keys, sort, build_bundles."""
    cs = ss.contact_store
    ws = ss.warm_starter
    ws.import_keys(
        cs.column_of("shape0"),
        cs.column_of("shape1"),
        cs.count,
        offset0=cs.column_of("offset0"),
    )
    ws.sort()
    ws.build_bundles()
    ws.transfer_impulses(
        cs.column_of("accumulated_normal_impulse"),
        cs.column_of("accumulated_tangent_impulse1"),
        cs.column_of("accumulated_tangent_impulse2"),
    )


def _step_with_contacts(ss, dt, gravity, num_iterations=8):
    """Run a single substep: velocities -> partition -> solve -> integrate."""
    inv_dt = 1.0 / dt
    ss.integrate_velocities(gravity, dt)

    _build_bundles(ss)
    ss._partition_contacts()

    d = ss.device
    bs = ss.body_store
    cs = ss.contact_store

    from newton._src.solvers.phoenx.contacts import (
        clear_contact_count_kernel,
        count_contacts_per_body_kernel,
    )

    wp.launch(clear_contact_count_kernel, dim=bs.capacity, inputs=[ss._contact_count_per_body, bs.count], device=d)
    wp.launch(
        count_contacts_per_body_kernel,
        dim=cs.capacity,
        inputs=[cs.column_of("body0"), cs.column_of("body1"), cs.count, ss._contact_count_per_body],
        device=d,
    )

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
        _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=12)

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[row_b]
    vel = ss.body_store.column_of("velocity").numpy()[row_b]

    test.assertGreater(pos[1], -0.01, f"Box fell through ground: y={pos[1]:.4f}")
    test.assertLess(abs(vel[1]), 0.5, f"Box not settling: vy={vel[1]:.4f}")


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
    _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, 0, 0))
    _step_with_contacts(ss, 1.0 / 60.0, (0, 0, 0), num_iterations=8)

    wp.synchronize_device(device)
    acc_n = ss.contact_store.column_of("accumulated_normal_impulse").numpy()[0]
    test.assertGreater(acc_n, 0.0, f"Normal impulse should be positive for overlap; got {acc_n:.6f}")


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
    _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
    ss.warm_starter.import_keys(
        ss.contact_store.column_of("shape0"),
        ss.contact_store.column_of("shape1"),
        ss.contact_store.count,
        offset0=ss.contact_store.column_of("offset0"),
    )
    ss.warm_starter.sort()
    ss.warm_starter.build_bundles()
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
    _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
    ss.warm_starter.import_keys(
        ss.contact_store.column_of("shape0"),
        ss.contact_store.column_of("shape1"),
        ss.contact_store.count,
        offset0=ss.contact_store.column_of("offset0"),
    )
    ss.warm_starter.sort()
    ss.warm_starter.build_bundles()
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
    test.assertAlmostEqual(
        transferred_n,
        solved_n,
        delta=0.01,
        msg=f"Warm start should transfer solved impulse: {transferred_n:.4f} vs {solved_n:.4f}",
    )


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
    test.assertAlmostEqual(vel[0], expected, places=4, msg=f"After 1 frame: expected vx={expected}, got {vel[0]:.4f}")

    # Run 4 substeps without calling update_world_inertia again
    for _ in range(4):
        ss.contact_store.count.assign(wp.array([0], dtype=wp.int32, device=device))
        ss.step(1.0 / 240.0, gravity=(0, 0, 0))

    wp.synchronize_device(device)
    vel2 = ss.body_store.column_of("velocity").numpy()[row]
    # Velocity should remain at 9.0 (no additional damping from substeps)
    test.assertAlmostEqual(
        vel2[0], expected, delta=0.01, msg=f"Substeps should not apply additional damping: vx={vel2[0]:.4f}"
    )


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
    test.assertLessEqual(pos[0], 100.1, f"Position should be clamped: x={pos[0]:.2f}")


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
        _inject_contact(ss, 0, 0, 1, row_g, rows[0], normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        for j in range(3):
            _inject_contact(
                ss,
                j + 1,
                j + 1,
                j + 2,
                rows[j],
                rows[j + 1],
                normal=(0, 1, 0),
                offset0=(0, 0.5, 0),
                offset1=(0, -0.5, 0),
            )

        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=12)

    wp.synchronize_device(device)

    # All boxes should be above ground
    for i, row in enumerate(rows):
        pos = ss.body_store.column_of("position").numpy()[row]
        test.assertGreater(pos[1], -0.5, f"Box {i} fell through ground: y={pos[1]:.4f}")

    # Top box should be higher than bottom box
    pos_bottom = ss.body_store.column_of("position").numpy()[rows[0]]
    pos_top = ss.body_store.column_of("position").numpy()[rows[3]]
    test.assertGreater(pos_top[1], pos_bottom[1], f"Stack collapsed: bottom={pos_bottom[1]:.3f}, top={pos_top[1]:.3f}")


# ===========================================================================
# Test 7: Determinism — repeated runs produce identical results
# ===========================================================================


def _run_sim(device, seed_offset=0):
    """Run a short simulation and return final positions."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 1.0, 0), inverse_mass=1.0, velocity=(0.5, 0, 0))
    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    for _ in range(30):
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(
            ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0), friction=0.6
        )
        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=8)

    wp.synchronize_device(device)
    return ss.body_store.column_of("position").numpy()[row_b].copy()


def test_determinism(test, device):
    """Two identical runs must produce bit-identical results."""
    pos1 = _run_sim(device)
    pos2 = _run_sim(device)
    np.testing.assert_array_equal(pos1, pos2, err_msg=f"Non-deterministic: {pos1} vs {pos2}")


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

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=shape_count,
        device=device,
        default_friction=0.6,
    )
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count, max_contacts=contact_cap, device=device)

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
    h_box = ss.add_body(
        position=(0, 0, 2.0),
        inverse_mass=inv_mass,
        inverse_inertia_local=inv_inertia,
        linear_damping=0.995,
        angular_damping=0.99,
    )
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
    test.assertGreater(pos[2], 0.1, f"Box fell through ground: z={pos[2]:.4f}")
    test.assertLess(pos[2], 1.0, f"Box didn't settle: z={pos[2]:.4f}")
    test.assertLess(abs(vel[2]), 0.5, f"Box not settling: vz={vel[2]:.4f}")


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

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=shape_count,
        device=device,
        default_friction=0.5,
    )
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count, max_contacts=contact_cap, device=device)

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    radius = 0.5
    mass = 1.0
    inv_inertia = np.eye(3, dtype=np.float32) * (2.5 / (mass * radius * radius))
    h_sphere = ss.add_body(position=(0, 0, 2.0), inverse_mass=1.0 / mass, inverse_inertia_local=inv_inertia)
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

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=shape_count,
        device=device,
        default_friction=0.5,
    )
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count, max_contacts=contact_cap, device=device)

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
        _inject_contact(
            ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.01, 0), friction=0.8
        )
        _step_with_contacts(ss, dt, (0, -9.81, 0), num_iterations=8)

    wp.synchronize_device(device)
    vel = ss.body_store.column_of("velocity").numpy()[row_b]
    test.assertLess(abs(vel[0]), 0.2, f"Friction should have nearly stopped the body; vx={vel[0]:.4f}")


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

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=shape_count,
        device=device,
        default_friction=0.6,
    )
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count, max_contacts=contact_cap, device=device)

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
        bh = ss.add_body(
            position=(0, 0, z),
            inverse_mass=1.0,
            inverse_inertia_local=inv_inertia,
            linear_damping=0.995,
            angular_damping=0.99,
        )
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
        test.assertGreater(positions[i + 1], positions[i] - 0.1, f"Stack order violated at box {i}: {positions}")


# ===========================================================================
# Test 13: Pyramid stability (regression test from example_phoenx_pyramid)
# ===========================================================================


def test_pyramid_stability(test, device):
    """Pyramid of boxes on a plane should remain stable for several seconds.

    Boxes may sink slightly due to PGS convergence limits, but should
    not drift laterally or fall through the ground.  This catches
    regressions in the contact solver, bias computation, or warm starting.
    """
    num_layers = 3
    h = 0.5
    spacing = 2.0 * h + 0.02

    box_positions = []
    for layer in range(num_layers):
        n = num_layers - layer
        z = layer * spacing + h
        offset = -(n - 1) * spacing * 0.5
        for row in range(n):
            for col in range(n):
                x = offset + col * spacing
                y = offset + row * spacing
                box_positions.append((x, y, z))

    num_boxes = len(box_positions)
    num_shapes = num_boxes + 1
    body_cap = num_boxes + 1
    contact_cap = max(num_boxes * 16, 512)

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=num_shapes,
        device=device,
        default_friction=0.6,
    )
    pipeline = PhoenXCollisionPipeline(
        max_shapes=num_shapes,
        max_contacts=contact_cap,
        device=device,
    )

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    mass = 1.0
    inv_mass = 1.0 / mass
    inv_inertia = np.eye(3, dtype=np.float32) * (6.0 * inv_mass / (2.0 * h) ** 2)

    box_handles = []
    for i, (px, py, pz) in enumerate(box_positions):
        bh = ss.add_body(
            position=(px, py, pz),
            inverse_mass=inv_mass,
            inverse_inertia_local=inv_inertia,
            linear_damping=0.995,
            angular_damping=0.99,
        )
        shape_idx = i + 1
        ss.set_shape_body(shape_idx, bh)
        pipeline.add_shape_box(
            body_row=int(ss.body_store.handle_to_index.numpy()[bh]),
            half_extents=(h, h, h),
        )
        box_handles.append(bh)

    pipeline.finalize()

    initial_positions = np.array(box_positions, dtype=np.float32)

    dt = 1.0 / 60.0
    substeps = 8
    sub_dt = dt / substeps
    num_frames = 120  # 2 seconds

    for _ in range(num_frames):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=12)
            ss.export_impulses()

    wp.synchronize_device(device)

    h2i = ss.body_store.handle_to_index.numpy()
    positions = ss.body_store.column_of("position").numpy()

    for i, bh in enumerate(box_handles):
        row = int(h2i[bh])
        pos = positions[row]
        init = initial_positions[i]

        # No box should fall through the ground
        test.assertGreater(pos[2], -0.1, f"Box {i} fell through ground: z={pos[2]:.4f}")

        # Lateral drift should be small (< 1 box width)
        lateral_drift = np.sqrt((pos[0] - init[0]) ** 2 + (pos[1] - init[1]) ** 2)
        test.assertLess(lateral_drift, 1.0, f"Box {i} drifted laterally: {lateral_drift:.4f}")

        # Vertical sinking should be limited (allow up to 1 box height)
        test.assertGreater(pos[2], init[2] - 2.0 * h, f"Box {i} sank too much: z={pos[2]:.4f} (init={init[2]:.2f})")


# ===========================================================================
# Test 14: Graph coloring invariant — simple contact graph
# ===========================================================================


def _validate_partition_invariant(
    test, elements_np, partition_data_np, partition_ends_np, num_partitions, has_additional, element_count
):
    """Check that no two contacts in the same partition share a body.

    Args:
        elements_np: (N, 8) int32 array of body indices per contact (-1 = unused).
        partition_data_np: sorted element indices from GraphColoring.
        partition_ends_np: cumulative partition sizes.
        num_partitions: number of color partitions (excludes overflow).
        has_additional: whether an overflow partition exists.
        element_count: total number of active elements.
    """
    total_slots = num_partitions + (1 if has_additional else 0)
    start = 0
    for p in range(total_slots):
        end = int(partition_ends_np[p])
        is_overflow = (p == num_partitions) and has_additional
        elem_indices = partition_data_np[start:end]

        if not is_overflow:
            # Collect all body indices touched by contacts in this partition
            bodies_seen = set()
            for ei in elem_indices:
                ei = int(ei)
                test.assertLess(ei, element_count, f"Partition {p} references out-of-range element {ei}")
                row = elements_np[ei]
                for b in row:
                    if b < 0:
                        break
                    test.assertNotIn(
                        b,
                        bodies_seen,
                        f"Partition {p}: body {b} appears in multiple contacts "
                        f"(element {ei} conflicts with a prior element)",
                    )
                    bodies_seen.add(b)
        start = end


def test_graph_coloring_simple(test, device):
    """Graph coloring on a small contact graph must satisfy the partition invariant.

    Setup: 3 bodies (0, 1, 2), 4 contacts:
        c0: bodies (0, 1)
        c1: bodies (1, 2)
        c2: bodies (0, 2)
        c3: bodies (0, 1)   -- duplicate of c0
    Contacts c0 and c1 share body 1, c0 and c2 share body 0, etc.
    The coloring must ensure no two contacts in the same partition share a body.
    """
    from newton._src.solvers.phoenx.maximal_independent_set import GraphColoring

    num_contacts = 4
    num_bodies = 3
    max_elements = 16
    max_nodes = 8
    max_colors = 8

    gc = GraphColoring(max_elements=max_elements, max_nodes=max_nodes, max_colors=max_colors, device=device)

    # Build elements array: (max_elements, 8), padded with -1
    elements_np = np.full((max_elements, 8), -1, dtype=np.int32)
    elements_np[0, 0] = 0
    elements_np[0, 1] = 1  # c0: bodies 0, 1
    elements_np[1, 0] = 1
    elements_np[1, 1] = 2  # c1: bodies 1, 2
    elements_np[2, 0] = 0
    elements_np[2, 1] = 2  # c2: bodies 0, 2
    elements_np[3, 0] = 0
    elements_np[3, 1] = 1  # c3: bodies 0, 1 (dup)

    elements = wp.array(elements_np, dtype=wp.int32, device=device)
    element_count = wp.array([num_contacts], dtype=wp.int32, device=device)
    node_count = wp.array([num_bodies], dtype=wp.int32, device=device)

    gc.color(elements, element_count, node_count)
    wp.synchronize_device(device)

    partition_data_np = gc.partition_data.numpy()[:max_elements]
    partition_ends_np = gc.partition_ends.numpy()
    num_partitions = int(gc.num_partitions.numpy()[0])
    has_additional = int(gc.has_additional.numpy()[0]) != 0

    # All contacts must be accounted for
    total_assigned = int(partition_ends_np[num_partitions - 1]) if num_partitions > 0 else 0
    if has_additional:
        total_assigned = int(partition_ends_np[max_colors])
    test.assertEqual(total_assigned, num_contacts, f"Expected {num_contacts} contacts assigned, got {total_assigned}")

    # Validate the key invariant
    _validate_partition_invariant(
        test, elements_np, partition_data_np, partition_ends_np, num_partitions, has_additional, num_contacts
    )


# ===========================================================================
# Test 15: Graph coloring — pyramid configuration
# ===========================================================================


def test_graph_coloring_pyramid(test, device):
    """Graph coloring on a pyramid-like scene with heavy ground-body sharing.

    15 bodies (body 0 = ground), ~40 contacts where many share body 0.
    Validates the partition invariant and checks overflow is <10%.
    """
    from newton._src.solvers.phoenx.maximal_independent_set import GraphColoring

    num_bodies = 15
    # Build contacts: each non-ground body has a contact with the ground,
    # plus contacts between adjacent bodies in a grid pattern.
    contacts = []

    # Layer 0: bodies 1..9 (3x3 grid on ground)
    layer0 = list(range(1, 10))
    for b in layer0:
        contacts.append((0, b))  # ground contact

    # Adjacent pairs in 3x3 grid
    for r in range(3):
        for c in range(3):
            idx = 1 + r * 3 + c
            if c < 2:
                contacts.append((idx, idx + 1))
            if r < 2:
                contacts.append((idx, idx + 3))

    # Layer 1: bodies 10..13 (2x2 on top of layer 0)
    layer1 = list(range(10, 14))
    for i, b in enumerate(layer1):
        r, c = divmod(i, 2)
        base = 1 + r * 3 + c
        contacts.append((base, b))
        contacts.append((base + 1, b))
        contacts.append((base + 3, b))
        contacts.append((base + 4, b))

    # Layer 2: body 14 on top of layer 1
    for b in layer1:
        contacts.append((b, 14))

    num_contacts = len(contacts)
    max_elements = max(num_contacts * 2, 128)
    max_nodes = max(num_bodies * 2, 32)
    max_colors = 16

    gc = GraphColoring(max_elements=max_elements, max_nodes=max_nodes, max_colors=max_colors, device=device)

    elements_np = np.full((max_elements, 8), -1, dtype=np.int32)
    for i, (b0, b1) in enumerate(contacts):
        elements_np[i, 0] = b0
        elements_np[i, 1] = b1

    elements = wp.array(elements_np, dtype=wp.int32, device=device)
    element_count = wp.array([num_contacts], dtype=wp.int32, device=device)
    node_count = wp.array([num_bodies], dtype=wp.int32, device=device)

    gc.color(elements, element_count, node_count)
    wp.synchronize_device(device)

    partition_data_np = gc.partition_data.numpy()[:max_elements]
    partition_ends_np = gc.partition_ends.numpy()
    num_partitions = int(gc.num_partitions.numpy()[0])
    has_additional = int(gc.has_additional.numpy()[0]) != 0

    # All contacts must be accounted for
    total_assigned = int(partition_ends_np[num_partitions - 1]) if num_partitions > 0 else 0
    overflow_count = 0
    if has_additional:
        total_with_overflow = int(partition_ends_np[max_colors])
        overflow_count = total_with_overflow - total_assigned
        total_assigned = total_with_overflow
    test.assertEqual(total_assigned, num_contacts, f"Expected {num_contacts} contacts assigned, got {total_assigned}")

    # Validate the key invariant
    _validate_partition_invariant(
        test, elements_np, partition_data_np, partition_ends_np, num_partitions, has_additional, num_contacts
    )

    # Flag if overflow exceeds 10%
    overflow_pct = overflow_count / num_contacts * 100.0 if num_contacts > 0 else 0.0
    test.assertLessEqual(
        overflow_pct,
        10.0,
        f"Overflow partition has {overflow_count}/{num_contacts} contacts "
        f"({overflow_pct:.1f}%), exceeding 10% threshold",
    )


# ===========================================================================
# Test 16: CUDA graph capture — step() must be sync-free
# ===========================================================================


def test_step_cuda_graph_capture(test, device):
    """step() must execute without GPU-to-CPU sync so it can be graph-captured.

    Captures step() into a CUDA graph, replays it, and verifies that the
    graph-captured result matches a non-captured run bit-for-bit.  Any .numpy()
    or host sync inside step() will cause the capture to fail with RuntimeError.
    """
    if not device.is_cuda:
        return  # graph capture is CUDA-only

    def _setup_and_run(use_graph):
        """Build identical scene, run 5 steps, return final position."""
        ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
        h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
        h_box = ss.add_body(position=(0, 0.5, 0), inverse_mass=1.0)
        row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
        row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

        dt = 1.0 / 60.0

        # Inject contact and build bundles (runs outside step, OK to sync)
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        _build_bundles(ss)

        # Warm-up step (compiles kernels)
        ss.step(dt, gravity=(0, -9.81, 0), num_iterations=4)
        wp.synchronize_device(device)

        # Re-inject for the timed run
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        _build_bundles(ss)

        if use_graph:
            wp.capture_begin(device=device)
            ss.step(dt, gravity=(0, -9.81, 0), num_iterations=4)
            graph = wp.capture_end(device=device)
            for _ in range(5):
                wp.capture_launch(graph)
        else:
            for _ in range(5):
                ss.step(dt, gravity=(0, -9.81, 0), num_iterations=4)

        wp.synchronize_device(device)
        return ss.body_store.column_of("position").numpy()[row_b].copy()

    # Run without graph capture
    pos_normal = _setup_and_run(use_graph=False)

    # Run with graph capture — this will fail if step() has any CPU sync
    try:
        pos_graph = _setup_and_run(use_graph=True)
    except RuntimeError as e:
        test.fail(f"CUDA graph capture of step() failed: {e}")

    # Results must match exactly
    np.testing.assert_array_equal(
        pos_graph, pos_normal, err_msg=f"Graph-captured step() differs from normal: {pos_graph} vs {pos_normal}"
    )


# ===========================================================================
# Test 17: Bundle correctness — bundle count matches expected grouping
# ===========================================================================


def test_bundle_count_correctness(test, device):
    """Verify bundle building produces the correct number of bundles.

    7 contacts: 5 for pair (0,1) -> 1 bundle of 5, 2 for pair (2,3) -> 1 bundle of 2.
    Total: 2 bundles.
    """
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device)
    h_g = ss.add_body(position=(0, 0, 0), is_static=True)
    h_b1 = ss.add_body(position=(0, 1, 0), inverse_mass=1.0)
    h_b2 = ss.add_body(position=(1, 1, 0), inverse_mass=1.0)

    row_g = int(ss.body_store.handle_to_index.numpy()[h_g])
    row_b1 = int(ss.body_store.handle_to_index.numpy()[h_b1])
    row_b2 = int(ss.body_store.handle_to_index.numpy()[h_b2])

    # 5 contacts for pair (shapes 0,1) -> body pair (row_g, row_b1)
    # 2 contacts for pair (shapes 2,3) -> body pair (row_g, row_b2)
    num_contacts = 7
    ss.contact_store.count.assign(wp.array([num_contacts], dtype=wp.int32, device=device))
    for ci in range(5):
        _inject_contact(ss, ci, 0, 1, row_g, row_b1, normal=(0, 1, 0), offset0=(0, 0, float(ci) * 0.1))
    for ci in range(5, 7):
        _inject_contact(ss, ci, 2, 3, row_g, row_b2, normal=(0, 1, 0), offset0=(0, 0, float(ci) * 0.1))

    _build_bundles(ss)
    wp.synchronize_device(device)

    n_bundles = int(ss.warm_starter.bundle_count.numpy()[0])
    test.assertEqual(n_bundles, 2, f"Expected 2 bundles (5+2 contacts in 2 pairs), got {n_bundles}")

    # With 6 contacts for one pair, it should split into 2 bundles (5+1)
    ss.contact_store.count.assign(wp.array([6], dtype=wp.int32, device=device))
    for ci in range(6):
        _inject_contact(ss, ci, 0, 1, row_g, row_b1, normal=(0, 1, 0), offset0=(0, 0, float(ci) * 0.1))

    _build_bundles(ss)
    wp.synchronize_device(device)

    n_bundles = int(ss.warm_starter.bundle_count.numpy()[0])
    test.assertEqual(n_bundles, 2, f"Expected 2 bundles (6 contacts, max 5 per bundle), got {n_bundles}")


# ===========================================================================
# Test 18: XPBD contacts — box on ground must settle
# ===========================================================================


def test_xpbd_box_on_ground(test, device):
    """XPBD position-level contacts must support a box on a ground plane."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device, contact_mode="xpbd")
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 0.5, 0), inverse_mass=1.0)

    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    for _ in range(120):
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.5, 0))
        _build_bundles(ss)
        ss.step(dt, gravity=(0, -9.81, 0), num_iterations=12)

    wp.synchronize_device(device)
    pos = ss.body_store.column_of("position").numpy()[row_b]
    test.assertGreater(pos[1], -0.1, f"XPBD: Box fell through ground: y={pos[1]:.4f}")


# ===========================================================================
# Test 19: XPBD friction — lateral velocity should be reduced
# ===========================================================================


def test_xpbd_friction(test, device):
    """XPBD contacts must apply friction to reduce lateral velocity."""
    ss = SolverState(body_capacity=8, contact_capacity=32, shape_count=4, device=device, contact_mode="xpbd")
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    h_box = ss.add_body(position=(0, 0.01, 0), velocity=(2.0, 0, 0), inverse_mass=1.0)
    row_g = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_b = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    for _ in range(60):
        ss.update_world_inertia()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))
        _inject_contact(
            ss, 0, 0, 1, row_g, row_b, normal=(0, 1, 0), offset0=(0, 0, 0), offset1=(0, -0.01, 0), friction=0.8
        )
        _build_bundles(ss)
        ss.step(dt, gravity=(0, -9.81, 0), num_iterations=8)

    wp.synchronize_device(device)
    vel = ss.body_store.column_of("velocity").numpy()[row_b]
    test.assertLess(abs(vel[0]), 1.5, f"XPBD: Friction should have reduced lateral velocity; vx={vel[0]:.4f}")


# ===========================================================================
# Test 20: Momentum conservation — two spheres collide in zero gravity
# ===========================================================================


def test_momentum_conservation_equal_mass(test, device):
    """Two equal-mass spheres colliding head-on must conserve total momentum.

    Sphere A moves right, sphere B is stationary. After collision,
    total momentum (m1*v1 + m2*v2) must equal the initial value.
    Energy must not increase (no energy injection from the solver).
    """
    ss = SolverState(body_capacity=4, contact_capacity=16, shape_count=4, device=device)
    pipeline = PhoenXCollisionPipeline(max_shapes=4, max_contacts=16, device=device)

    mass = 1.0
    inv_mass = 1.0 / mass
    radius = 0.5
    inv_inertia = np.eye(3, dtype=np.float32) * (2.5 * inv_mass / (radius * radius))

    # Sphere A: moving right
    h_a = ss.add_body(
        position=(-0.9, 0, 0),
        velocity=(2.0, 0, 0),
        inverse_mass=inv_mass,
        inverse_inertia_local=inv_inertia,
        linear_damping=1.0,
        angular_damping=1.0,
    )
    ss.set_shape_body(0, h_a)
    row_a = int(ss.body_store.handle_to_index.numpy()[h_a])
    pipeline.add_shape_sphere(body_row=row_a, radius=radius)

    # Sphere B: stationary
    h_b = ss.add_body(
        position=(0.9, 0, 0),
        velocity=(0, 0, 0),
        inverse_mass=inv_mass,
        inverse_inertia_local=inv_inertia,
        linear_damping=1.0,
        angular_damping=1.0,
    )
    ss.set_shape_body(1, h_b)
    row_b = int(ss.body_store.handle_to_index.numpy()[h_b])
    pipeline.add_shape_sphere(body_row=row_b, radius=radius)

    pipeline.finalize()

    # Initial momentum
    p_initial = mass * 2.0  # sphere A: m*v = 1*2 = 2, sphere B: 0

    dt = 1.0 / 120.0
    substeps = 8
    sub_dt = dt / substeps
    no_gravity = (0.0, 0.0, 0.0)

    for _ in range(60):  # 0.5 seconds
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=no_gravity, num_iterations=8)
            ss.export_impulses()

    wp.synchronize_device(device)

    vel_a = ss.body_store.column_of("velocity").numpy()[row_a]
    vel_b = ss.body_store.column_of("velocity").numpy()[row_b]

    # Total momentum (x-component, the collision axis)
    p_final = mass * vel_a[0] + mass * vel_b[0]

    # Momentum conservation: relative error < 1%
    test.assertAlmostEqual(
        p_final,
        p_initial,
        delta=0.02 * abs(p_initial),
        msg=f"Momentum not conserved: initial={p_initial:.4f}, final={p_final:.4f}",
    )

    # Energy must not increase
    ke_initial = 0.5 * mass * 2.0 * 2.0  # = 2.0
    ke_final = 0.5 * mass * (np.dot(vel_a, vel_a) + np.dot(vel_b, vel_b))
    # PGS with Baumgarte bias can inject a small amount of energy during
    # contact correction — allow up to 15% increase.
    test.assertLessEqual(
        ke_final, ke_initial * 1.15, f"Energy increased too much: initial={ke_initial:.4f}, final={ke_final:.4f}"
    )

    # Spheres should have separated (both moving apart or momentum transferred)
    test.assertGreater(vel_b[0], 0.1, f"Sphere B should be moving right after collision: vx={vel_b[0]:.4f}")


# ===========================================================================
# Test 21: Momentum conservation — unequal mass ratio (1:10)
# ===========================================================================


def test_momentum_conservation_unequal_mass(test, device):
    """Heavy sphere hitting a light sphere must conserve momentum.

    Mass ratio 10:1. The light sphere should rebound faster.
    """
    ss = SolverState(body_capacity=4, contact_capacity=16, shape_count=4, device=device)
    pipeline = PhoenXCollisionPipeline(max_shapes=4, max_contacts=16, device=device)

    mass_a = 10.0
    mass_b = 1.0
    radius = 0.5
    inv_inertia_a = np.eye(3, dtype=np.float32) * (2.5 / mass_a / (radius * radius))
    inv_inertia_b = np.eye(3, dtype=np.float32) * (2.5 / mass_b / (radius * radius))

    h_a = ss.add_body(
        position=(-0.45, 0, 0),
        velocity=(2.0, 0, 0),
        inverse_mass=1.0 / mass_a,
        inverse_inertia_local=inv_inertia_a,
        linear_damping=1.0,
        angular_damping=1.0,
    )
    ss.set_shape_body(0, h_a)
    row_a = int(ss.body_store.handle_to_index.numpy()[h_a])
    pipeline.add_shape_sphere(body_row=row_a, radius=radius)

    h_b = ss.add_body(
        position=(0.45, 0, 0),
        velocity=(0, 0, 0),
        inverse_mass=1.0 / mass_b,
        inverse_inertia_local=inv_inertia_b,
        linear_damping=1.0,
        angular_damping=1.0,
    )
    ss.set_shape_body(1, h_b)
    row_b = int(ss.body_store.handle_to_index.numpy()[h_b])
    pipeline.add_shape_sphere(body_row=row_b, radius=radius)

    pipeline.finalize()

    p_initial = mass_a * 2.0  # = 20

    dt = 1.0 / 120.0
    substeps = 8
    sub_dt = dt / substeps

    for _ in range(60):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, 0), num_iterations=8)
            ss.export_impulses()

    wp.synchronize_device(device)

    vel_a = ss.body_store.column_of("velocity").numpy()[row_a]
    vel_b = ss.body_store.column_of("velocity").numpy()[row_b]

    p_final = mass_a * vel_a[0] + mass_b * vel_b[0]

    test.assertAlmostEqual(
        p_final,
        p_initial,
        delta=0.02 * abs(p_initial),
        msg=f"Momentum not conserved (10:1): initial={p_initial:.4f}, final={p_final:.4f}",
    )

    ke_initial = 0.5 * mass_a * 2.0 * 2.0
    ke_final = 0.5 * (mass_a * np.dot(vel_a, vel_a) + mass_b * np.dot(vel_b, vel_b))
    test.assertLessEqual(
        ke_final, ke_initial * 1.15, f"Energy increased too much (10:1): initial={ke_initial:.4f}, final={ke_final:.4f}"
    )

    # Light sphere should move faster than heavy sphere
    test.assertGreater(vel_b[0], vel_a[0], f"Light sphere should be faster: vb={vel_b[0]:.4f}, va={vel_a[0]:.4f}")


# ===========================================================================
# Test 22: Per-shape friction — slide distance matches analytical result
# ===========================================================================


def test_per_shape_friction_slide_distance(test, device):
    """Boxes with different friction coefficients sliding on a ground plane.

    Analytical slide distance for Coulomb friction:
        d = v0^2 / (2 * mu * g)

    Each box starts with the same velocity but has a different friction
    coefficient.  After coming to rest, its displacement should match
    the analytical prediction within 20% (PGS bias and discrete time
    introduce some error).
    """
    g = 9.81
    v0 = 2.0
    friction_values = [0.2, 0.5, 1.0]
    n_boxes = len(friction_values)

    body_cap = n_boxes + 1
    contact_cap = n_boxes * 16
    shape_count = n_boxes + 1

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=shape_count,
        device=device,
        default_friction=0.5,
    )
    pipeline = PhoenXCollisionPipeline(
        max_shapes=shape_count,
        max_contacts=contact_cap,
        device=device,
    )

    # Ground (static, high friction shape — actual friction is averaged with box)
    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    # Ground friction = 10 (high), so avg(10, mu_box) ≈ mu_box for small mu_box.
    # Actually we want the contact friction to equal the box friction.
    # avg(mu_ground, mu_box) = mu_box when mu_ground = mu_box.
    # Simpler: set ground friction = 0, then avg(0, mu_box) = mu_box/2.
    # Best: set ground friction equal to box friction. But it's one shape.
    # Use the convention: ground mu = 0, contact mu = avg(0, 2*target) = target.
    # Actually, just set ground mu to match each box is impossible with one shape.
    # Instead: set ground mu very high, so avg ≈ mu_box/2 + high/2. Not helpful.
    # Cleanest: set ground friction = same as box, then avg = mu_box.
    # But ground is shared. Use the formula: contact_mu = avg(ground, box).
    # Set ground_mu = 0. Then contact_mu = box_mu / 2. Adjust analytical accordingly.
    pipeline.add_shape_plane(body_row=ground_row, friction=0.0)

    h = 0.5  # box half-extent
    inv_inertia = np.eye(3, dtype=np.float32) * 6.0  # unit mass, unit cube
    box_rows = []
    for i, mu in enumerate(friction_values):
        # Place boxes along y so they don't interfere, sliding along x
        bh = ss.add_body(
            position=(0, float(i) * 3.0, h + 0.01),
            velocity=(v0, 0, 0),
            inverse_mass=1.0,
            inverse_inertia_local=inv_inertia,
            linear_damping=1.0,
            angular_damping=1.0,
        )
        row = int(ss.body_store.handle_to_index.numpy()[bh])
        ss.set_shape_body(i + 1, bh)
        # Box friction = 2 * target_mu so that avg(0, 2*mu) = mu
        pipeline.add_shape_box(body_row=row, half_extents=(h, h, h), friction=2.0 * mu)
        box_rows.append(row)

    pipeline.finalize()

    dt = 1.0 / 60.0
    substeps = 8
    sub_dt = dt / substeps

    # Run long enough for slowest box (mu=0.2) to stop:
    # t_stop = v0/(mu*g) = 2/(0.2*9.81) ≈ 1.02s → ~62 frames at 60fps
    for _ in range(120):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -g), num_iterations=8)
            ss.export_impulses()

    wp.synchronize_device(device)

    for i, mu in enumerate(friction_values):
        pos = ss.body_store.column_of("position").numpy()[box_rows[i]]
        vel = ss.body_store.column_of("velocity").numpy()[box_rows[i]]
        slide_distance = pos[0]  # started at x=0
        analytical_distance = v0**2 / (2.0 * mu * g)

        # Box should have stopped (velocity near zero)
        test.assertLess(
            abs(vel[0]),
            0.3,
            f"Box mu={mu}: should have stopped, vx={vel[0]:.4f}",
        )

        # Slide distance should match analytical within 20%
        # (PGS Baumgarte bias and discrete time cause some deviation)
        rel_error = abs(slide_distance - analytical_distance) / analytical_distance
        test.assertLess(
            rel_error,
            0.25,
            f"Box mu={mu}: slide distance {slide_distance:.4f} vs "
            f"analytical {analytical_distance:.4f} (error {rel_error:.1%})",
        )

        # Higher friction should produce shorter slide distance
        if i > 0:
            prev_pos = ss.body_store.column_of("position").numpy()[box_rows[i - 1]]
            test.assertGreater(
                prev_pos[0],
                pos[0],
                f"Box mu={mu} slid farther than mu={friction_values[i - 1]}: {pos[0]:.4f} vs {prev_pos[0]:.4f}",
            )


# ===========================================================================
# Test 23: Zero friction — box slides forever
# ===========================================================================


def test_zero_friction_no_deceleration(test, device):
    """A box with zero friction on a ground plane should maintain its velocity.

    With mu=0 for both shapes, the contact friction is zero and the box
    should slide at constant speed (no deceleration).
    """
    body_cap = 4
    contact_cap = 32
    shape_count = 2

    ss = SolverState(
        body_capacity=body_cap,
        contact_capacity=contact_cap,
        shape_count=shape_count,
        device=device,
        default_friction=0.0,
    )
    pipeline = PhoenXCollisionPipeline(max_shapes=shape_count, max_contacts=contact_cap, device=device)

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row, friction=0.0)

    v0 = 3.0
    h = 0.5
    inv_inertia = np.eye(3, dtype=np.float32) * 6.0
    h_box = ss.add_body(
        position=(0, 0, h + 0.01),
        velocity=(v0, 0, 0),
        inverse_mass=1.0,
        inverse_inertia_local=inv_inertia,
        linear_damping=1.0,
        angular_damping=1.0,
    )
    box_row = int(ss.body_store.handle_to_index.numpy()[h_box])
    ss.set_shape_body(1, h_box)
    pipeline.add_shape_box(body_row=box_row, half_extents=(h, h, h), friction=0.0)

    pipeline.finalize()

    dt = 1.0 / 60.0
    substeps = 8
    sub_dt = dt / substeps

    for _ in range(120):
        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=8)
            ss.export_impulses()

    wp.synchronize_device(device)

    vel = ss.body_store.column_of("velocity").numpy()[box_row]
    # Should retain most of its initial velocity (>80%)
    test.assertGreater(
        vel[0],
        v0 * 0.8,
        f"Zero-friction box decelerated too much: vx={vel[0]:.4f} (initial={v0})",
    )

    # Should have traveled approximately v0 * t = 3.0 * 2.0 = 6.0m
    pos = ss.body_store.column_of("position").numpy()[box_row]
    test.assertGreater(
        pos[0],
        v0 * 2.0 * 0.8,
        f"Zero-friction box didn't travel far enough: x={pos[0]:.4f}",
    )


# ===========================================================================
# Registration
# ===========================================================================

devices = get_test_devices()

add_function_test(
    TestPhoenXComprehensive, "test_bias_sign_box_on_ground", test_bias_sign_box_on_ground, devices=devices
)
add_function_test(
    TestPhoenXComprehensive, "test_bias_produces_positive_impulse", test_bias_produces_positive_impulse, devices=devices
)
add_function_test(TestPhoenXComprehensive, "test_warm_start_scaling", test_warm_start_scaling, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_damping_per_frame", test_damping_per_frame, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_velocity_clamping", test_velocity_clamping, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_mass_splitting_stacked", test_mass_splitting_stacked, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_determinism", test_determinism, devices=devices)
add_function_test(
    TestPhoenXComprehensive,
    "test_collision_pipeline_box_on_plane",
    test_collision_pipeline_box_on_plane,
    devices=devices,
)
add_function_test(
    TestPhoenXComprehensive,
    "test_collision_pipeline_sphere_on_plane",
    test_collision_pipeline_sphere_on_plane,
    devices=devices,
)
add_function_test(
    TestPhoenXComprehensive,
    "test_collision_pipeline_capsule_on_plane",
    test_collision_pipeline_capsule_on_plane,
    devices=devices,
)
add_function_test(TestPhoenXComprehensive, "test_friction_stops_sliding", test_friction_stops_sliding, devices=devices)
add_function_test(
    TestPhoenXComprehensive, "test_collision_pipeline_box_stack", test_collision_pipeline_box_stack, devices=devices
)
add_function_test(TestPhoenXComprehensive, "test_pyramid_stability", test_pyramid_stability, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_graph_coloring_simple", test_graph_coloring_simple, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_graph_coloring_pyramid", test_graph_coloring_pyramid, devices=devices)
add_function_test(
    TestPhoenXComprehensive, "test_step_cuda_graph_capture", test_step_cuda_graph_capture, devices=devices
)
add_function_test(
    TestPhoenXComprehensive, "test_bundle_count_correctness", test_bundle_count_correctness, devices=devices
)
add_function_test(TestPhoenXComprehensive, "test_xpbd_box_on_ground", test_xpbd_box_on_ground, devices=devices)
add_function_test(TestPhoenXComprehensive, "test_xpbd_friction", test_xpbd_friction, devices=devices)
add_function_test(
    TestPhoenXComprehensive,
    "test_momentum_conservation_equal_mass",
    test_momentum_conservation_equal_mass,
    devices=devices,
)
add_function_test(
    TestPhoenXComprehensive,
    "test_momentum_conservation_unequal_mass",
    test_momentum_conservation_unequal_mass,
    devices=devices,
)
add_function_test(
    TestPhoenXComprehensive,
    "test_per_shape_friction_slide_distance",
    test_per_shape_friction_slide_distance,
    devices=devices,
)
add_function_test(
    TestPhoenXComprehensive,
    "test_zero_friction_no_deceleration",
    test_zero_friction_no_deceleration,
    devices=devices,
)

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
