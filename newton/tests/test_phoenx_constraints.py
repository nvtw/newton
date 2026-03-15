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

"""Tests for PhoenX constraint system (ball-socket, revolute, fixed joints)."""

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

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
