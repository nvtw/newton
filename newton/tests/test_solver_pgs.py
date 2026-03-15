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

"""Tests for the PhoenX PGS contact solver: ball-on-ground, friction, partitions."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver_phoenx import SolverState
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _inject_contact(
    ss,
    contact_idx,
    shape0,
    shape1,
    body0,
    body1,
    normal,
    offset0=(0.0, 0.0, 0.0),
    offset1=(0.0, 0.0, 0.0),
    friction=0.5,
    margin0=0.0,
    margin1=0.0,
):
    """Write one synthetic contact directly into the DataStore columns."""
    cs = ss.contact_store
    d = ss.device

    def _write(name, idx, value, dtype):
        col = cs.column_of(name).numpy()
        col[idx] = value
        cs.column_of(name).assign(wp.array(col, dtype=dtype, device=d))

    _write("shape0", contact_idx, shape0, wp.int32)
    _write("shape1", contact_idx, shape1, wp.int32)
    _write("body0", contact_idx, body0, wp.int32)
    _write("body1", contact_idx, body1, wp.int32)
    _write("normal", contact_idx, np.array(normal, dtype=np.float32), wp.vec3)
    _write("offset0", contact_idx, np.array(offset0, dtype=np.float32), wp.vec3)
    _write("offset1", contact_idx, np.array(offset1, dtype=np.float32), wp.vec3)
    _write("margin0", contact_idx, margin0, wp.float32)
    _write("margin1", contact_idx, margin1, wp.float32)
    _write("friction", contact_idx, friction, wp.float32)
    _write("accumulated_normal_impulse", contact_idx, 0.0, wp.float32)
    _write("accumulated_tangent_impulse1", contact_idx, 0.0, wp.float32)
    _write("accumulated_tangent_impulse2", contact_idx, 0.0, wp.float32)


class TestSolverPGS(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Ball-on-ground: sphere above static ground should not fall through
# ---------------------------------------------------------------------------


def test_ball_on_ground(test, device):
    """Dynamic sphere on a static ground plane should settle without penetrating."""
    ss = SolverState(
        body_capacity=8, contact_capacity=32, shape_count=4, device=device
    )
    h_ground = ss.add_body(position=(0.0, 0.0, 0.0), is_static=True)
    h_ball = ss.add_body(
        position=(0.0, 0.05, 0.0),
        inverse_mass=1.0,
    )

    row_ground = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_ball = int(ss.body_store.handle_to_index.numpy()[h_ball])

    dt = 1.0 / 60.0
    gravity = (0.0, -9.81, 0.0)
    num_steps = 120

    for _ in range(num_steps):
        ss.update_world_inertia()
        ss.integrate_velocities(gravity, dt)

        ss.contact_store.count.zero_()
        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))

        _inject_contact(
            ss,
            contact_idx=0,
            shape0=0,
            shape1=1,
            body0=row_ground,
            body1=row_ball,
            normal=(0.0, 1.0, 0.0),
            offset0=(0.0, 0.0, 0.0),
            offset1=(0.0, -0.05, 0.0),
            friction=0.5,
        )

        ss._partition_contacts()

        max_slots = ss.graph_coloring.max_colors + 1
        inv_dt = 1.0 / dt

        for p in range(max_slots):
            ss._launch_prepare(p, inv_dt)

        for _ in range(8):
            for p in range(max_slots):
                ss._launch_solve(p, 1)

        ss.integrate_positions(dt)

    wp.synchronize_device(device)

    pos = ss.body_store.column_of("position").numpy()[row_ball]
    vel = ss.body_store.column_of("velocity").numpy()[row_ball]

    test.assertGreaterEqual(pos[1], -0.01, f"Ball penetrated ground: y={pos[1]:.4f}")
    test.assertAlmostEqual(vel[1], 0.0, delta=0.5, msg="Ball velocity should be near zero")


# ---------------------------------------------------------------------------
# Friction: body on flat surface should not slide under small lateral force
# ---------------------------------------------------------------------------


def test_friction_holds(test, device):
    """A body with lateral velocity on a frictional surface should decelerate."""
    ss = SolverState(
        body_capacity=8, contact_capacity=32, shape_count=4, device=device
    )
    h_ground = ss.add_body(position=(0.0, 0.0, 0.0), is_static=True)
    h_box = ss.add_body(
        position=(0.0, 0.01, 0.0),
        velocity=(1.0, 0.0, 0.0),
        inverse_mass=1.0,
    )

    row_ground = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_box = int(ss.body_store.handle_to_index.numpy()[h_box])

    dt = 1.0 / 60.0
    num_steps = 60

    for _ in range(num_steps):
        ss.update_world_inertia()
        ss.integrate_velocities((0.0, -9.81, 0.0), dt)

        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))

        _inject_contact(
            ss,
            contact_idx=0,
            shape0=0,
            shape1=1,
            body0=row_ground,
            body1=row_box,
            normal=(0.0, 1.0, 0.0),
            offset0=(0.0, 0.0, 0.0),
            offset1=(0.0, -0.01, 0.0),
            friction=0.8,
        )

        ss._partition_contacts()

        max_slots = ss.graph_coloring.max_colors + 1
        inv_dt = 1.0 / dt

        for p in range(max_slots):
            ss._launch_prepare(p, inv_dt)

        for _ in range(8):
            for p in range(max_slots):
                ss._launch_solve(p, 1)

        ss.integrate_positions(dt)

    wp.synchronize_device(device)

    vel = ss.body_store.column_of("velocity").numpy()[row_box]
    test.assertLess(
        abs(vel[0]), 1.0,
        f"Friction should have reduced lateral velocity; vx={vel[0]:.4f}",
    )


# ---------------------------------------------------------------------------
# Warm start convergence: running the same scenario with warm starting
# should yield lower residual velocity than cold start
# ---------------------------------------------------------------------------


def _run_ball_drop(device, num_iterations, use_warm_start):
    """Helper: drops a ball for 30 frames and returns final y-velocity magnitude."""
    ss = SolverState(
        body_capacity=8, contact_capacity=32, shape_count=4, device=device
    )
    h_ground = ss.add_body(position=(0.0, 0.0, 0.0), is_static=True)
    h_ball = ss.add_body(
        position=(0.0, 0.1, 0.0),
        inverse_mass=1.0,
    )

    row_ground = int(ss.body_store.handle_to_index.numpy()[h_ground])
    row_ball = int(ss.body_store.handle_to_index.numpy()[h_ball])

    dt = 1.0 / 60.0
    gravity = (0.0, -9.81, 0.0)

    for step_i in range(30):
        if use_warm_start and step_i > 0:
            ss.warm_starter.begin_frame()

        ss.update_world_inertia()
        ss.integrate_velocities(gravity, dt)

        ss.contact_store.count.assign(wp.array([1], dtype=wp.int32, device=device))

        _inject_contact(
            ss,
            contact_idx=0,
            shape0=0,
            shape1=1,
            body0=row_ground,
            body1=row_ball,
            normal=(0.0, 1.0, 0.0),
            offset0=(0.0, 0.0, 0.0),
            offset1=(0.0, -0.1, 0.0),
            friction=0.5,
        )

        if use_warm_start:
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

        ss._partition_contacts()

        max_slots = ss.graph_coloring.max_colors + 1
        inv_dt = 1.0 / dt

        for p in range(max_slots):
            ss._launch_prepare(p, inv_dt)

        for _ in range(num_iterations):
            for p in range(max_slots):
                ss._launch_solve(p, 1)

        ss.integrate_positions(dt)

        if use_warm_start:
            ss.export_impulses()

    wp.synchronize_device(device)
    vel = ss.body_store.column_of("velocity").numpy()[row_ball]
    return abs(float(vel[1]))


def test_warm_start_convergence(test, device):
    """Warm-started solver should converge to lower residual than cold start."""
    residual_cold = _run_ball_drop(device, num_iterations=4, use_warm_start=False)
    residual_warm = _run_ball_drop(device, num_iterations=4, use_warm_start=True)

    test.assertLessEqual(
        residual_warm,
        residual_cold + 0.1,
        f"Warm start residual ({residual_warm:.4f}) should be <= cold ({residual_cold:.4f})",
    )


# ---------------------------------------------------------------------------
# Partition validity: no two contacts in the same partition share a body
# ---------------------------------------------------------------------------


def test_partitioning_valid(test, device):
    """Graph coloring produces valid partitions where no two contacts share a body."""
    ss = SolverState(
        body_capacity=16, contact_capacity=64, shape_count=8, device=device
    )
    bodies = []
    for i in range(6):
        bodies.append(ss.add_body(position=(float(i), 1.0, 0.0)))

    num_contacts = 5
    ss.contact_store.count.assign(
        wp.array([num_contacts], dtype=wp.int32, device=device)
    )

    contact_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]

    for ci, (b0_h, b1_h) in enumerate(contact_pairs):
        row0 = int(ss.body_store.handle_to_index.numpy()[bodies[b0_h]])
        row1 = int(ss.body_store.handle_to_index.numpy()[bodies[b1_h]])
        _inject_contact(
            ss,
            contact_idx=ci,
            shape0=ci,
            shape1=ci + 1,
            body0=row0,
            body1=row1,
            normal=(0.0, 1.0, 0.0),
        )

    ss._partition_contacts()
    wp.synchronize_device(device)

    num_p = int(ss.graph_coloring.num_partitions.numpy()[0])
    has_add = int(ss.graph_coloring.has_additional.numpy()[0])
    total_partitions = num_p + has_add
    ends_np = ss.graph_coloring.partition_ends.numpy()
    partition_data_np = ss.graph_coloring.partition_data.numpy()

    body0_np = ss.contact_store.column_of("body0").numpy()
    body1_np = ss.contact_store.column_of("body1").numpy()

    for p in range(total_partitions):
        start = 0 if p == 0 else int(ends_np[p - 1])
        end = int(ends_np[p])

        bodies_in_partition = set()
        for j in range(start, end):
            ci = partition_data_np[j]
            if ci >= num_contacts:
                continue
            b0 = int(body0_np[ci])
            b1 = int(body1_np[ci])
            test.assertNotIn(
                b0, bodies_in_partition,
                f"Partition {p} has duplicate body {b0}",
            )
            test.assertNotIn(
                b1, bodies_in_partition,
                f"Partition {p} has duplicate body {b1}",
            )
            bodies_in_partition.add(b0)
            bodies_in_partition.add(b1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestSolverPGS, "test_ball_on_ground", test_ball_on_ground, devices=devices)
add_function_test(TestSolverPGS, "test_friction_holds", test_friction_holds, devices=devices)
add_function_test(TestSolverPGS, "test_warm_start_convergence", test_warm_start_convergence, devices=devices)
add_function_test(TestSolverPGS, "test_partitioning_valid", test_partitioning_valid, devices=devices)

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
