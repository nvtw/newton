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

"""Tests for PhoenX constraint system (all joint types + drives)."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.constraints import (
    JointSchema,
    col_base,
    ds_load_float,
    ds_load_int,
    ds_load_mat33,
    ds_load_quat,
    ds_load_vec3,
    ds_store_float,
    ds_store_int,
    ds_store_mat33,
    ds_store_vec3,
)
from newton._src.solvers.phoenx.data_base import DataStore
from newton._src.solvers.phoenx.solver_phoenx import SolverState
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestPhoenXConstraints(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate(ss, frames=60, substeps=8, sub_dt=1.0 / 480.0, gravity=(0, 0, -9.81), iters=12):
    ss.update_world_inertia()
    for _ in range(frames):
        for _ in range(substeps):
            ss.step(sub_dt, gravity=gravity, num_iterations=iters)


def _body_pos(ss, handle):
    h2i = ss.body_store.handle_to_index.numpy()
    return ss.body_store.column_of("position").numpy()[int(h2i[handle])].copy()


def _body_vel(ss, handle):
    h2i = ss.body_store.handle_to_index.numpy()
    return ss.body_store.column_of("velocity").numpy()[int(h2i[handle])].copy()


def _body_angvel(ss, handle):
    h2i = ss.body_store.handle_to_index.numpy()
    return ss.body_store.column_of("angular_velocity").numpy()[int(h2i[handle])].copy()


def _body_orient(ss, handle):
    h2i = ss.body_store.handle_to_index.numpy()
    return ss.body_store.column_of("orientation").numpy()[int(h2i[handle])].copy()


# ---------------------------------------------------------------------------
# Ball-socket joint tests
# ---------------------------------------------------------------------------


def test_ball_socket_pendulum(test, device):
    """Ball-socket pendulum: body swings down, anchor distance preserved."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(2, 0, 5), inverse_mass=1.0)
    ss.add_joint_ball_socket(h_a, h_b, anchor_world=(0, 0, 5))

    _simulate(ss, frames=30, substeps=8, iters=16)

    pa = _body_pos(ss, h_a)
    pb = _body_pos(ss, h_b)
    dist = np.linalg.norm(pb - pa)

    test.assertAlmostEqual(dist, 2.0, delta=0.05,
                           msg=f"Distance drift: {dist:.4f}")
    test.assertLess(pb[2], 4.95, msg=f"Body should swing down: z={pb[2]:.4f}")


def test_ball_socket_double_pendulum(test, device):
    """Double pendulum: two links, both anchors preserved."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b0 = ss.add_body(position=(1, 0, 5), inverse_mass=1.0)
    h_b1 = ss.add_body(position=(2, 0, 5), inverse_mass=1.0)
    ss.add_joint_ball_socket(h_a, h_b0, anchor_world=(0, 0, 5))
    ss.add_joint_ball_socket(h_b0, h_b1, anchor_world=(1, 0, 5))

    _simulate(ss, frames=30, substeps=8, iters=16)

    pa = _body_pos(ss, h_a)
    p0 = _body_pos(ss, h_b0)
    p1 = _body_pos(ss, h_b1)

    d0 = np.linalg.norm(p0 - pa)
    d1 = np.linalg.norm(p1 - p0)

    test.assertAlmostEqual(d0, 1.0, delta=0.1, msg=f"First link drift: {d0:.4f}")
    test.assertAlmostEqual(d1, 1.0, delta=0.1, msg=f"Second link drift: {d1:.4f}")
    test.assertLess(p0[2], 4.95)
    test.assertLess(p1[2], 4.95)


# ---------------------------------------------------------------------------
# Revolute joint tests
# ---------------------------------------------------------------------------


def test_revolute_hinge_axis(test, device):
    """Revolute joint constrains motion to the hinge plane."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(2, 0, 5), inverse_mass=1.0)
    ss.add_joint_revolute(h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 1, 0))

    # More frames and fewer iterations for faster convergence of swing motion
    _simulate(ss, frames=240, substeps=4, sub_dt=1.0 / 240.0, iters=8)

    pb = _body_pos(ss, h_b)
    test.assertAlmostEqual(pb[1], 0.0, delta=0.1,
                           msg=f"Out-of-plane drift: y={pb[1]:.4f}")
    test.assertLess(pb[2], 4.95, msg=f"Body should swing: z={pb[2]:.4f}")


def test_revolute_anchor_preserved(test, device):
    """Revolute joint: distance to anchor stays constant."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(2, 0, 5), inverse_mass=1.0)
    ss.add_joint_revolute(h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 1, 0))

    _simulate(ss, frames=60, substeps=8, iters=16)

    pa = _body_pos(ss, h_a)
    pb = _body_pos(ss, h_b)
    dist = np.linalg.norm(pb - pa)
    test.assertAlmostEqual(dist, 2.0, delta=0.1,
                           msg=f"Anchor distance drift: {dist:.4f}")


# ---------------------------------------------------------------------------
# Fixed joint tests
# ---------------------------------------------------------------------------


def test_fixed_joint(test, device):
    """Fixed joint: body remains at initial relative position."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(1, 0, 5), inverse_mass=1.0)
    ss.add_joint_fixed(h_a, h_b, anchor_world=(0.5, 0, 5))

    _simulate(ss, frames=60, substeps=8, iters=16)

    pb = _body_pos(ss, h_b)
    pa = _body_pos(ss, h_a)
    rel = pb - pa
    test.assertAlmostEqual(rel[0], 1.0, delta=0.2,
                           msg=f"X drift: {rel[0]:.4f}")
    test.assertAlmostEqual(rel[2], 0.0, delta=0.2,
                           msg=f"Z drift: {rel[2]:.4f}")


# ---------------------------------------------------------------------------
# Constraint with contacts
# ---------------------------------------------------------------------------


def test_ball_socket_with_contacts(test, device):
    """Ball-socket joint works alongside the contact solver."""
    ss = SolverState(
        body_capacity=10, contact_capacity=64, shape_count=4,
        joint_capacity=4, device=device, default_friction=0.5,
    )
    pipeline = PhoenXCollisionPipeline(max_shapes=4, max_contacts=64, device=device)

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    h = 0.5
    inv_mass = 1.0
    inv_inertia = np.eye(3, dtype=np.float32) * 6.0 * inv_mass / (2.0 * h) ** 2

    h_b0 = ss.add_body(
        position=(0, 0, h), inverse_mass=inv_mass,
        inverse_inertia_local=inv_inertia,
    )
    ss.set_shape_body(1, h_b0)
    pipeline.add_shape_box(
        body_row=int(ss.body_store.handle_to_index.numpy()[h_b0]),
        half_extents=(h, h, h),
    )

    h_b1 = ss.add_body(
        position=(2, 0, h), inverse_mass=inv_mass,
        inverse_inertia_local=inv_inertia,
    )
    ss.set_shape_body(2, h_b1)
    pipeline.add_shape_box(
        body_row=int(ss.body_store.handle_to_index.numpy()[h_b1]),
        half_extents=(h, h, h),
    )
    pipeline.finalize()

    ss.add_joint_ball_socket(h_b0, h_b1, anchor_world=(1, 0, h))

    ss.update_world_inertia()
    for _ in range(120):
        for _ in range(4):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(1.0 / 240.0, gravity=(0, 0, -9.81), num_iterations=12)
            ss.export_impulses()

    p0 = _body_pos(ss, h_b0)
    p1 = _body_pos(ss, h_b1)

    test.assertGreater(p0[2], 0.3, msg=f"Box 0 fell through: z={p0[2]:.4f}")
    test.assertGreater(p1[2], 0.3, msg=f"Box 1 fell through: z={p1[2]:.4f}")

    dist = np.linalg.norm(p1 - p0)
    test.assertAlmostEqual(dist, 2.0, delta=0.3,
                           msg=f"Joint distance drift: {dist:.4f}")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_constraint_determinism(test, device):
    """Two identical runs produce bit-identical results."""
    results = []
    for _ in range(2):
        ss = SolverState(
            body_capacity=10, contact_capacity=32, shape_count=2,
            joint_capacity=4, device=device,
        )
        h_a = ss.add_body(position=(0, 0, 5), is_static=True)
        h_b = ss.add_body(position=(2, 0, 5), inverse_mass=1.0)
        ss.add_joint_ball_socket(h_a, h_b, anchor_world=(0, 0, 5))
        _simulate(ss, frames=30, substeps=8, iters=16)
        results.append(_body_pos(ss, h_b))

    np.testing.assert_array_equal(results[0], results[1])


# ---------------------------------------------------------------------------
# DataStore flat-array accessor tests
# ---------------------------------------------------------------------------


@wp.kernel
def _ds_accessor_test_kernel(
    data: wp.array(dtype=wp.float32),
    base_int: int,
    base_float: int,
    base_vec3: int,
    base_quat: int,
    base_mat33: int,
    out_int: wp.array(dtype=wp.int32),
    out_float: wp.array(dtype=wp.float32),
    out_vec3: wp.array(dtype=wp.vec3),
    out_quat: wp.array(dtype=wp.quat),
    out_mat33: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    # Write via ds_store, read back via ds_load
    ds_store_int(data, base_int, tid, tid + 100)
    ds_store_float(data, base_float, tid, float(tid) * 1.5)
    ds_store_vec3(data, base_vec3, tid, wp.vec3(float(tid), float(tid) + 0.5, float(tid) + 1.0))
    m = wp.mat33(
        float(tid), 0.0, 0.0,
        0.0, float(tid) + 1.0, 0.0,
        0.0, 0.0, float(tid) + 2.0,
    )
    ds_store_mat33(data, base_mat33, tid, m)

    out_int[tid] = ds_load_int(data, base_int, tid)
    out_float[tid] = ds_load_float(data, base_float, tid)
    out_vec3[tid] = ds_load_vec3(data, base_vec3, tid)
    out_quat[tid] = ds_load_quat(data, base_quat, tid)
    out_mat33[tid] = ds_load_mat33(data, base_mat33, tid)


def test_ds_accessor_roundtrip(test, device):
    """DataStore flat-array accessors read back exactly what was written."""
    cap = 8
    store = DataStore(JointSchema, cap, device=device)

    # Write known quat values via column_of view
    q_col = store.column_of("inv_initial_orientation").numpy()
    for i in range(cap):
        q_col[i] = [float(i) * 0.1, 0.0, 0.0, 1.0]
    store.column_of("inv_initial_orientation").assign(
        wp.array(q_col, dtype=wp.quat, device=device)
    )

    out_int = wp.zeros(cap, dtype=wp.int32, device=device)
    out_float = wp.zeros(cap, dtype=wp.float32, device=device)
    out_vec3 = wp.zeros(cap, dtype=wp.vec3, device=device)
    out_quat = wp.zeros(cap, dtype=wp.quat, device=device)
    out_mat33 = wp.zeros(cap, dtype=wp.mat33, device=device)

    wp.launch(
        _ds_accessor_test_kernel,
        dim=cap,
        inputs=[
            store.data,
            col_base(store, "joint_type"),
            col_base(store, "hinge_lambda_x"),
            col_base(store, "local_anchor0"),
            col_base(store, "inv_initial_orientation"),
            col_base(store, "point_eff_mass"),
            out_int, out_float, out_vec3, out_quat, out_mat33,
        ],
        device=device,
    )
    wp.synchronize()

    ints = out_int.numpy()
    floats = out_float.numpy()
    vecs = out_vec3.numpy()
    quats = out_quat.numpy()
    mats = out_mat33.numpy()

    for i in range(cap):
        test.assertEqual(ints[i], i + 100, msg=f"int roundtrip failed at {i}")
        test.assertAlmostEqual(floats[i], i * 1.5, places=5, msg=f"float roundtrip at {i}")
        np.testing.assert_allclose(vecs[i], [i, i + 0.5, i + 1.0], atol=1e-6)
        np.testing.assert_allclose(quats[i], [i * 0.1, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(mats[i].diagonal(), [i, i + 1.0, i + 2.0], atol=1e-6)


def test_ds_accessor_matches_column_of(test, device):
    """Values written via ds_store are visible through column_of, and vice versa."""
    cap = 4
    store = DataStore(JointSchema, cap, device=device)

    # Write via column_of
    body0_col = store.column_of("body0").numpy()
    body0_col[:] = [10, 20, 30, 40]
    store.column_of("body0").assign(wp.array(body0_col, dtype=wp.int32, device=device))

    anchor_col = store.column_of("local_anchor0").numpy()
    anchor_col[0] = [1.0, 2.0, 3.0]
    anchor_col[1] = [4.0, 5.0, 6.0]
    store.column_of("local_anchor0").assign(wp.array(anchor_col, dtype=wp.vec3, device=device))

    # Read back via ds_load in a kernel
    out_int = wp.zeros(cap, dtype=wp.int32, device=device)
    out_vec = wp.zeros(cap, dtype=wp.vec3, device=device)
    b0_base = col_base(store, "body0")
    a0_base = col_base(store, "local_anchor0")

    @wp.kernel
    def _read_kernel(
        data: wp.array(dtype=wp.float32),
        oi: wp.array(dtype=wp.int32),
        ov: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        oi[tid] = ds_load_int(data, wp.static(b0_base), tid)
        ov[tid] = ds_load_vec3(data, wp.static(a0_base), tid)

    wp.launch(_read_kernel, dim=cap, inputs=[store.data, out_int, out_vec], device=device)
    wp.synchronize()

    np.testing.assert_array_equal(out_int.numpy(), [10, 20, 30, 40])
    np.testing.assert_allclose(out_vec.numpy()[0], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(out_vec.numpy()[1], [4.0, 5.0, 6.0], atol=1e-6)

    # Write via ds_store, read back via column_of
    @wp.kernel
    def _write_kernel(data: wp.array(dtype=wp.float32)):
        ds_store_int(data, wp.static(b0_base), 0, 99)
        ds_store_vec3(data, wp.static(a0_base), 0, wp.vec3(7.0, 8.0, 9.0))

    wp.launch(_write_kernel, dim=1, inputs=[store.data], device=device)
    wp.synchronize()

    test.assertEqual(store.column_of("body0").numpy()[0], 99)
    np.testing.assert_allclose(store.column_of("local_anchor0").numpy()[0], [7.0, 8.0, 9.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Revolute angle limits
# ---------------------------------------------------------------------------


def test_revolute_angle_limits(test, device):
    """Revolute joint with tight angle limits: body stays within bounds."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(2, 0, 5), inverse_mass=1.0)
    # Hinge around Y axis, limit swing to +/- 30 degrees
    limit_rad = np.radians(30.0)
    ss.add_joint_revolute(
        h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 1, 0),
        angle_min=-limit_rad, angle_max=limit_rad,
    )

    _simulate(ss, frames=120, substeps=8, iters=16)

    pa = _body_pos(ss, h_a)
    pb = _body_pos(ss, h_b)
    r = pb - pa
    # The angle from initial position: atan2(z_drop, x_extent)
    # With 30-degree limit, the body can swing at most ~30 deg from horizontal
    # so z should not drop below 2*sin(30) ≈ 1.0 from anchor
    angle = np.arctan2(-(r[2]), r[0])
    test.assertLess(abs(angle), limit_rad + 0.15,
                    msg=f"Angle exceeds limit: {np.degrees(angle):.1f} deg")
    # Also verify anchor distance preserved
    dist = np.linalg.norm(r)
    test.assertAlmostEqual(dist, 2.0, delta=0.15,
                           msg=f"Anchor distance drift: {dist:.4f}")


def test_revolute_energy_conservation(test, device):
    """Revolute pendulum: KE + PE roughly conserved (no damping)."""
    L = 2.0
    mass = 1.0
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(
        position=(L, 0, 5), inverse_mass=1.0 / mass,
        linear_damping=1.0, angular_damping=1.0,  # no damping
    )
    ss.add_joint_revolute(h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 1, 0))

    g = 9.81
    z0 = 5.0  # initial height = anchor height, so PE_initial = 0
    # Total energy should be ~0 (starts at rest at anchor height)
    initial_energy = mass * g * (z0 - z0)  # = 0

    _simulate(ss, frames=30, substeps=8, iters=16)

    pb = _body_pos(ss, h_b)
    vb = _body_vel(ss, h_b)
    ke = 0.5 * mass * np.dot(vb, vb)
    pe = mass * g * (pb[2] - z0)
    total_e = ke + pe

    # Should be close to 0 (conservation). PGS is dissipative so allow some loss.
    test.assertAlmostEqual(total_e, initial_energy, delta=2.0,
                           msg=f"Energy not conserved: KE={ke:.3f}, PE={pe:.3f}, total={total_e:.3f}")


# ---------------------------------------------------------------------------
# Prismatic joint tests
# ---------------------------------------------------------------------------


def test_prismatic_slide_axis(test, device):
    """Prismatic joint: body slides along axis under gravity."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(0, 0, 5), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    # Slide along Z axis (vertical), no limits
    ss.add_joint_prismatic(h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 0, 1))

    _simulate(ss, frames=30, substeps=8, iters=16)

    pb = _body_pos(ss, h_b)
    # Body should fall straight down along Z
    test.assertAlmostEqual(pb[0], 0.0, delta=0.05,
                           msg=f"X drift: {pb[0]:.4f}")
    test.assertAlmostEqual(pb[1], 0.0, delta=0.05,
                           msg=f"Y drift: {pb[1]:.4f}")
    test.assertLess(pb[2], 4.5, msg=f"Body should fall: z={pb[2]:.4f}")


def test_prismatic_no_rotation(test, device):
    """Prismatic joint: body should not rotate."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(0, 0, 4), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ss.add_joint_prismatic(h_a, h_b, anchor_world=(0, 0, 4.5), axis_world=(0, 0, 1))

    _simulate(ss, frames=60, substeps=8, iters=16)

    q = _body_orient(ss, h_b)
    # Should stay near identity quaternion (0,0,0,1)
    # Measure rotation angle: 2*acos(|w|)
    angle = 2.0 * np.arccos(np.clip(abs(q[3]), 0.0, 1.0))
    test.assertLess(angle, 0.1, msg=f"Rotation angle: {np.degrees(angle):.2f} deg")


def test_prismatic_lateral_constraint(test, device):
    """Prismatic joint: lateral velocity is constrained to zero."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    # Start body on-axis but with lateral initial velocity
    h_b = ss.add_body(position=(0, 0, 5), inverse_mass=1.0,
                      velocity=(3.0, 0.0, 0.0),  # lateral kick
                      linear_damping=1.0, angular_damping=1.0)
    ss.add_joint_prismatic(h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 0, 1))

    _simulate(ss, frames=60, substeps=8, iters=16)

    pb = _body_pos(ss, h_b)
    # Lateral X should remain near 0 (perpendicular constraint prevents drift)
    test.assertAlmostEqual(pb[0], 0.0, delta=0.2,
                           msg=f"Lateral X drift: {pb[0]:.4f}")
    test.assertAlmostEqual(pb[1], 0.0, delta=0.2,
                           msg=f"Lateral Y drift: {pb[1]:.4f}")


def test_prismatic_slide_limits(test, device):
    """Prismatic joint with slide limits: body stops at limit."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(0, 0, 5), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    # Slide along Z, limit to [-1, 0] — can only fall 1m
    ss.add_joint_prismatic(
        h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 0, 1),
        slide_min=-1.0, slide_max=0.0,
    )

    _simulate(ss, frames=120, substeps=8, iters=16)

    pb = _body_pos(ss, h_b)
    # Body should stop around z=4.0 (fell 1m from z=5)
    test.assertGreater(pb[2], 3.5, msg=f"Fell past limit: z={pb[2]:.4f}")
    test.assertLess(pb[2], 5.1, msg=f"Body went up: z={pb[2]:.4f}")


def test_prismatic_free_fall_analytical(test, device):
    """Prismatic (no limits): body in free-fall along axis matches s = 0.5*g*t^2."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    z0 = 10.0
    h_a = ss.add_body(position=(0, 0, z0), is_static=True)
    h_b = ss.add_body(position=(0, 0, z0), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ss.add_joint_prismatic(h_a, h_b, anchor_world=(0, 0, z0), axis_world=(0, 0, 1))

    g = 9.81
    frames = 30
    substeps = 8
    sub_dt = 1.0 / 480.0
    t = frames * substeps * sub_dt

    _simulate(ss, frames=frames, substeps=substeps, sub_dt=sub_dt, iters=16)

    pb = _body_pos(ss, h_b)
    expected_z = z0 - 0.5 * g * t * t
    # PGS constraint solving + discrete integration: allow ~5% error
    test.assertAlmostEqual(pb[2], expected_z, delta=abs(expected_z - z0) * 0.10 + 0.05,
                           msg=f"z={pb[2]:.4f} vs expected {expected_z:.4f}")


# ---------------------------------------------------------------------------
# Revolute drive tests
# ---------------------------------------------------------------------------


def test_revolute_position_drive(test, device):
    """Revolute position drive: pendulum driven to target angle."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(2, 0, 5), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ji = ss.add_joint_revolute(
        h_a, h_b, anchor_world=(0, 0, 5), axis_world=(0, 1, 0),
        angle_min=-3.14, angle_max=3.14,
    )
    # Drive to 45 degrees with stiff PD
    target_angle = np.radians(45.0)
    ss.set_joint_drive(ji, mode=SolverState.DRIVE_POSITION, target=target_angle,
                       stiffness=500.0, damping=50.0, max_force=1.0e6)

    # Run long enough for the drive to settle
    _simulate(ss, frames=300, substeps=8, iters=16)

    pa = _body_pos(ss, h_a)
    pb = _body_pos(ss, h_b)
    r = pb - pa
    # Angle from horizontal in XZ plane (hinge around Y)
    actual_angle = np.arctan2(-r[2], r[0])

    # The drive fights gravity; check it's in the right ballpark
    # With stiff drive (500) the body should be near the target
    test.assertAlmostEqual(actual_angle, target_angle, delta=0.5,
                           msg=f"Drive angle: {np.degrees(actual_angle):.1f} vs "
                               f"target {np.degrees(target_angle):.1f}")


def test_revolute_velocity_drive(test, device):
    """Revolute velocity drive: body spins at target angular velocity."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 0), is_static=True)
    h_b = ss.add_body(position=(1, 0, 0), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ji = ss.add_joint_revolute(
        h_a, h_b, anchor_world=(0, 0, 0), axis_world=(0, 0, 1),
    )
    # Drive at 2 rad/s around Z axis (no gravity effect since axis is vertical)
    target_vel = 2.0
    ss.set_joint_drive(ji, mode=SolverState.DRIVE_VELOCITY, target=target_vel,
                       stiffness=0.0, damping=100.0, max_force=1.0e6)

    # No gravity so the drive purely controls velocity
    _simulate(ss, frames=120, substeps=8, gravity=(0, 0, 0), iters=16)

    # Check angular velocity of body along hinge axis
    w = _body_angvel(ss, h_b)
    # The hinge axis is Z, so w[2] should approach target_vel
    # body1 angular vel relative to body0 = w1 - w0 along axis
    w0 = _body_angvel(ss, h_a)
    rel_w = w[2] - w0[2]
    test.assertAlmostEqual(rel_w, target_vel, delta=0.5,
                           msg=f"Angular vel: {rel_w:.3f} vs target {target_vel:.3f}")


# ---------------------------------------------------------------------------
# Prismatic drive tests
# ---------------------------------------------------------------------------


def test_prismatic_position_drive(test, device):
    """Prismatic position drive: body driven to target displacement."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(0, 0, 5), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ji = ss.add_joint_prismatic(
        h_a, h_b, anchor_world=(0, 0, 5), axis_world=(1, 0, 0),
    )
    # Drive to +1.5 m along X with stiff PD
    target_pos = 1.5
    ss.set_joint_drive(ji, mode=SolverState.DRIVE_POSITION, target=target_pos,
                       stiffness=500.0, damping=50.0, max_force=1.0e6)

    # Use zero gravity to isolate drive behavior
    _simulate(ss, frames=300, substeps=8, gravity=(0, 0, 0), iters=16)

    pb = _body_pos(ss, h_b)
    pa = _body_pos(ss, h_a)
    displacement = pb[0] - pa[0]
    test.assertAlmostEqual(displacement, target_pos, delta=0.3,
                           msg=f"Prismatic drive: disp={displacement:.3f} vs target {target_pos}")


def test_prismatic_velocity_drive(test, device):
    """Prismatic velocity drive: body moves at target linear velocity."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(0, 0, 5), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ji = ss.add_joint_prismatic(
        h_a, h_b, anchor_world=(0, 0, 5), axis_world=(1, 0, 0),
    )
    # Drive at 1 m/s along X
    target_vel = 1.0
    ss.set_joint_drive(ji, mode=SolverState.DRIVE_VELOCITY, target=target_vel,
                       stiffness=0.0, damping=100.0, max_force=1.0e6)

    _simulate(ss, frames=120, substeps=8, gravity=(0, 0, 0), iters=16)

    vb = _body_vel(ss, h_b)
    test.assertAlmostEqual(vb[0], target_vel, delta=0.3,
                           msg=f"Prismatic vel drive: vx={vb[0]:.3f} vs target {target_vel}")


# ---------------------------------------------------------------------------
# Fixed joint: rotation lock test
# ---------------------------------------------------------------------------


def test_fixed_joint_no_rotation(test, device):
    """Fixed joint: relative orientation stays constant."""
    ss = SolverState(
        body_capacity=10, contact_capacity=32, shape_count=2,
        joint_capacity=4, device=device,
    )
    h_a = ss.add_body(position=(0, 0, 5), is_static=True)
    h_b = ss.add_body(position=(1, 0, 5), inverse_mass=1.0,
                      linear_damping=1.0, angular_damping=1.0)
    ss.add_joint_fixed(h_a, h_b, anchor_world=(0.5, 0, 5))

    _simulate(ss, frames=120, substeps=8, iters=16)

    q = _body_orient(ss, h_b)
    angle = 2.0 * np.arccos(np.clip(abs(q[3]), 0.0, 1.0))
    test.assertLess(angle, 0.15,
                    msg=f"Fixed joint rotation: {np.degrees(angle):.2f} deg")


# ---------------------------------------------------------------------------
# Schema union tests
# ---------------------------------------------------------------------------


def test_schema_union_offsets(test, device):
    """Per-type schemas share common header offsets and drive offsets."""
    from newton._src.solvers.phoenx.constraints import (
        BallSocketJointData, FixedJointData, PrismaticJointData,
        RevoluteJointData, schema_col_base,
    )
    cap = 16  # arbitrary capacity

    # Common header fields must have identical offsets across all types
    header_fields = [
        "joint_type", "body0", "body1",
        "local_anchor0", "local_anchor1",
        "local_axis0", "local_axis1",
        "inv_initial_orientation", "rw0", "rw1",
    ]
    for name in header_fields:
        offsets = set()
        for st in [BallSocketJointData, FixedJointData, RevoluteJointData, PrismaticJointData]:
            offsets.add(schema_col_base(st, cap, name))
        test.assertEqual(len(offsets), 1,
                         msg=f"Header field '{name}' has inconsistent offsets: {offsets}")

    # Hinge fields must match across Revolute, Fixed, Prismatic
    hinge_fields = [
        "hinge_lambda_x", "hinge_lambda_y",
        "hinge_b2xa1", "hinge_c2xa1",
        "hinge_eff_mass_00", "hinge_eff_mass_01",
        "hinge_eff_mass_10", "hinge_eff_mass_11",
    ]
    for name in hinge_fields:
        offsets = set()
        for st in [RevoluteJointData, FixedJointData, PrismaticJointData]:
            offsets.add(schema_col_base(st, cap, name))
        test.assertEqual(len(offsets), 1,
                         msg=f"Hinge field '{name}' has inconsistent offsets: {offsets}")

    # Drive fields must match between Revolute and Prismatic
    drive_fields = [
        "drive_mode", "drive_target", "drive_stiffness",
        "drive_damping", "drive_max_force", "drive_lambda", "drive_eff_mass",
    ]
    for name in drive_fields:
        r_off = schema_col_base(RevoluteJointData, cap, name)
        p_off = schema_col_base(PrismaticJointData, cap, name)
        test.assertEqual(r_off, p_off,
                         msg=f"Drive field '{name}' mismatch: revolute={r_off}, prismatic={p_off}")


def test_access_mode_roundtrip(test, device):
    """vel→pos→vel and pos→vel→pos roundtrips recover original state."""
    from newton._src.solvers.phoenx.constraints import sync_pos_to_vel, sync_vel_to_pos

    dt = 1.0 / 240.0
    inv_dt = 1.0 / dt

    # Reference state: body at some position with a non-trivial orientation
    ref_pos = wp.vec3(1.0, 2.0, 3.0)
    ref_orient = wp.normalize(wp.quat(0.1, 0.2, 0.3, 0.9))

    # --- vel → pos → vel roundtrip ---
    vel_in = wp.vec3(0.5, -1.0, 2.0)
    angvel_in = wp.vec3(0.3, -0.4, 0.1)

    @wp.kernel
    def _vel_pos_vel_kernel(
        result_vel: wp.array(dtype=wp.vec3),
        result_angvel: wp.array(dtype=wp.vec3),
    ):
        pos, orient = sync_vel_to_pos(ref_pos, ref_orient, vel_in, angvel_in, dt)
        vel_out, angvel_out = sync_pos_to_vel(ref_pos, ref_orient, pos, orient, inv_dt)
        result_vel[0] = vel_out
        result_angvel[0] = angvel_out

    rv = wp.zeros(1, dtype=wp.vec3, device=device)
    ra = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_vel_pos_vel_kernel, dim=1, inputs=[], outputs=[rv, ra], device=device)

    vel_out = rv.numpy()[0]
    angvel_out = ra.numpy()[0]
    np.testing.assert_allclose(vel_out, [0.5, -1.0, 2.0], atol=1e-4,
                               err_msg="vel→pos→vel linear velocity mismatch")
    np.testing.assert_allclose(angvel_out, [0.3, -0.4, 0.1], atol=1e-3,
                               err_msg="vel→pos→vel angular velocity mismatch")

    # --- pos → vel → pos roundtrip ---
    cur_pos_in = wp.vec3(1.01, 2.02, 2.98)
    cur_orient_in = wp.normalize(wp.quat(0.11, 0.19, 0.31, 0.89))

    @wp.kernel
    def _pos_vel_pos_kernel(
        result_pos: wp.array(dtype=wp.vec3),
        result_orient: wp.array(dtype=wp.vec4),
    ):
        vel, angvel = sync_pos_to_vel(ref_pos, ref_orient, cur_pos_in, cur_orient_in, inv_dt)
        pos_out, orient_out = sync_vel_to_pos(ref_pos, ref_orient, vel, angvel, dt)
        result_pos[0] = pos_out
        result_orient[0] = wp.vec4(orient_out[0], orient_out[1], orient_out[2], orient_out[3])

    rp = wp.zeros(1, dtype=wp.vec3, device=device)
    ro = wp.zeros(1, dtype=wp.vec4, device=device)
    wp.launch(_pos_vel_pos_kernel, dim=1, inputs=[], outputs=[rp, ro], device=device)

    pos_out = rp.numpy()[0]
    orient_out = ro.numpy()[0]
    np.testing.assert_allclose(pos_out, [1.01, 2.02, 2.98], atol=1e-4,
                               err_msg="pos→vel→pos position mismatch")
    cur_orient_np = np.array([0.11, 0.19, 0.31, 0.89])
    cur_orient_np /= np.linalg.norm(cur_orient_np)
    np.testing.assert_allclose(orient_out, cur_orient_np, atol=1e-3,
                               err_msg="pos→vel→pos orientation mismatch")


# ---------------------------------------------------------------------------
# Register tests
# ---------------------------------------------------------------------------

devices = get_test_devices()

add_function_test(TestPhoenXConstraints, "test_ball_socket_pendulum", test_ball_socket_pendulum, devices=devices)
add_function_test(TestPhoenXConstraints, "test_ball_socket_double_pendulum", test_ball_socket_double_pendulum, devices=devices)
add_function_test(TestPhoenXConstraints, "test_revolute_hinge_axis", test_revolute_hinge_axis, devices=devices)
add_function_test(TestPhoenXConstraints, "test_revolute_anchor_preserved", test_revolute_anchor_preserved, devices=devices)
add_function_test(TestPhoenXConstraints, "test_fixed_joint", test_fixed_joint, devices=devices)
add_function_test(TestPhoenXConstraints, "test_ball_socket_with_contacts", test_ball_socket_with_contacts, devices=devices)
add_function_test(TestPhoenXConstraints, "test_constraint_determinism", test_constraint_determinism, devices=devices)
add_function_test(TestPhoenXConstraints, "test_ds_accessor_roundtrip", test_ds_accessor_roundtrip, devices=devices)
add_function_test(TestPhoenXConstraints, "test_ds_accessor_matches_column_of", test_ds_accessor_matches_column_of, devices=devices)
add_function_test(TestPhoenXConstraints, "test_revolute_angle_limits", test_revolute_angle_limits, devices=devices)
add_function_test(TestPhoenXConstraints, "test_revolute_energy_conservation", test_revolute_energy_conservation, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_slide_axis", test_prismatic_slide_axis, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_no_rotation", test_prismatic_no_rotation, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_lateral_constraint", test_prismatic_lateral_constraint, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_slide_limits", test_prismatic_slide_limits, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_free_fall_analytical", test_prismatic_free_fall_analytical, devices=devices)
add_function_test(TestPhoenXConstraints, "test_revolute_position_drive", test_revolute_position_drive, devices=devices)
add_function_test(TestPhoenXConstraints, "test_revolute_velocity_drive", test_revolute_velocity_drive, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_position_drive", test_prismatic_position_drive, devices=devices)
add_function_test(TestPhoenXConstraints, "test_prismatic_velocity_drive", test_prismatic_velocity_drive, devices=devices)
add_function_test(TestPhoenXConstraints, "test_fixed_joint_no_rotation", test_fixed_joint_no_rotation, devices=devices)
add_function_test(TestPhoenXConstraints, "test_schema_union_offsets", test_schema_union_offsets, devices=devices)
add_function_test(TestPhoenXConstraints, "test_access_mode_roundtrip", test_access_mode_roundtrip, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
