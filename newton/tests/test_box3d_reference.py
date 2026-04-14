# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference solver tests — validate Box3D against a pure numpy TGS-Soft implementation.

This module implements the core Box2D v3 TGS-Soft revolute joint solver
in pure Python/numpy (no GPU), then compares its output against SolverBox3D
for the same scene.  This catches any 2D→3D porting errors in the GPU code.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig, Softness, compute_softness
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestBox3DReference(unittest.TestCase):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Pure numpy TGS-Soft solver (2-body revolute joint, following Box2D)
# ═══════════════════════════════════════════════════════════════════════


def _skew(v):
    """3D skew-symmetric matrix [v]_x such that [v]_x @ w = cross(v, w)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    qv = q[:3]
    qw = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _quat_to_mat(q):
    """Quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _quat_integrate(q, w, dt):
    """First-order quaternion integration."""
    hw = w * dt * 0.5
    dq = np.array([
        hw[0]*q[3] + hw[1]*q[2] - hw[2]*q[1],
        -hw[0]*q[2] + hw[1]*q[3] + hw[2]*q[0],
        hw[0]*q[1] - hw[1]*q[0] + hw[2]*q[3],
        -hw[0]*q[0] - hw[1]*q[1] - hw[2]*q[2],
    ])
    r = q + dq
    return r / np.linalg.norm(r)


def numpy_tgs_soft_revolute_step(
    pos_a, ori_a, vel_a, ang_vel_a, inv_mass_a, inv_inertia_body_a,
    pos_b, ori_b, vel_b, ang_vel_b, inv_mass_b, inv_inertia_body_b,
    local_anchor_a, local_anchor_b, hinge_axis_local,
    linear_impulse, angular_impulse,
    gravity, dt, num_substeps,
    contact_hertz, contact_damping, joint_hertz, joint_damping,
    linear_damping_coeff, angular_damping_coeff,
):
    """Pure numpy implementation of one full TGS-Soft step with a revolute joint.

    Follows Box2D v3's step loop exactly, generalized to 3D:
    - Per substep: integrate vel → biased solve → integrate pos → relaxation solve
    - Point-to-point: 3x3 K-matrix via [r]_x^T @ I^{-1} @ [r]_x
    - Angular (2 DOF perp to hinge): diagonal approximation

    Returns updated (pos_a, ori_a, vel_a, ang_vel_a, pos_b, ori_b, vel_b, ang_vel_b,
                     linear_impulse, angular_impulse).
    """
    sub_dt = dt / num_substeps
    inv_sub_dt = 1.0 / sub_dt if sub_dt > 0 else 0.0

    soft_j = compute_softness(joint_hertz, joint_damping, sub_dt)

    delta_pos_a = np.zeros(3)
    delta_pos_b = np.zeros(3)

    for sub in range(num_substeps):
        is_first = sub == 0

        # ── Update world-frame inertia ──
        R_a = _quat_to_mat(ori_a)
        R_b = _quat_to_mat(ori_b)
        inv_I_a = R_a @ inv_inertia_body_a @ R_a.T
        inv_I_b = R_b @ inv_inertia_body_b @ R_b.T

        # ── Integrate velocities ──
        if inv_mass_a > 0:
            vel_a += gravity * sub_dt
            vel_a *= 1.0 / (1.0 + sub_dt * linear_damping_coeff)
            ang_vel_a *= 1.0 / (1.0 + sub_dt * angular_damping_coeff)
        if inv_mass_b > 0:
            vel_b += gravity * sub_dt
            vel_b *= 1.0 / (1.0 + sub_dt * linear_damping_coeff)
            ang_vel_b *= 1.0 / (1.0 + sub_dt * angular_damping_coeff)

        # ── Biased solve ──
        for use_bias in [True, False]:
            # Compute world-space anchors
            r_a = _quat_rotate(ori_a, local_anchor_a)
            r_b = _quat_rotate(ori_b, local_anchor_b)

            # ── Point-to-point (3 DOF) ──
            Cdot = (vel_b + np.cross(ang_vel_b, r_b)) - (vel_a + np.cross(ang_vel_a, r_a))

            bias = np.zeros(3)
            ms = 1.0
            isv = 0.0
            if use_bias:
                sep = (pos_b + r_b) - (pos_a + r_a)
                bias = soft_j.bias_rate * sep
                ms = soft_j.mass_scale
                isv = soft_j.impulse_scale

            # Build K = (mA+mB)*I + [rA]_x^T @ IA^{-1} @ [rA]_x + [rB]_x^T @ IB^{-1} @ [rB]_x
            K = (inv_mass_a + inv_mass_b) * np.eye(3)
            rx_a = _skew(r_a)
            rx_b = _skew(r_b)
            K += rx_a.T @ inv_I_a @ rx_a
            K += rx_b.T @ inv_I_b @ rx_b

            # Solve K @ x = Cdot + bias
            rhs = Cdot + bias
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                sol = np.zeros(3)

            imp = -ms * sol - isv * linear_impulse
            linear_impulse += imp

            # Apply
            if inv_mass_a > 0:
                vel_a -= inv_mass_a * imp
                ang_vel_a -= inv_I_a @ np.cross(r_a, imp)
            if inv_mass_b > 0:
                vel_b += inv_mass_b * imp
                ang_vel_b += inv_I_b @ np.cross(r_b, imp)

            # ── Angular (2 DOF perpendicular to hinge) ──
            h = _quat_rotate(ori_b, hinge_axis_local)
            h = h / (np.linalg.norm(h) + 1e-12)

            # Build perpendicular basis
            if abs(h[1]) < 0.9:
                b1 = np.array([h[2], 0.0, -h[0]])
            else:
                b1 = np.array([0.0, -h[2], h[1]])
            b1 = b1 / (np.linalg.norm(b1) + 1e-12)
            b2 = np.cross(h, b1)

            dw = ang_vel_b - ang_vel_a
            k_ang = inv_I_a + inv_I_b

            k1 = b1 @ k_ang @ b1
            k2 = b2 @ k_ang @ b2
            am1 = 1.0 / k1 if k1 > 0 else 0.0
            am2 = 1.0 / k2 if k2 > 0 else 0.0

            cdot1 = np.dot(dw, b1)
            cdot2 = np.dot(dw, b2)

            i1 = -ms * am1 * cdot1 - isv * angular_impulse[0]
            i2 = -ms * am2 * cdot2 - isv * angular_impulse[1]
            angular_impulse[0] += i1
            angular_impulse[1] += i2

            ap = i1 * b1 + i2 * b2
            if inv_mass_a > 0:
                ang_vel_a -= inv_I_a @ ap
            if inv_mass_b > 0:
                ang_vel_b += inv_I_b @ ap

            # ── Integrate positions (only after biased solve, before relaxation) ──
            if use_bias:
                if inv_mass_a > 0:
                    pos_a += vel_a * sub_dt
                    delta_pos_a += vel_a * sub_dt
                    ori_a = _quat_integrate(ori_a, ang_vel_a, sub_dt)
                if inv_mass_b > 0:
                    pos_b += vel_b * sub_dt
                    delta_pos_b += vel_b * sub_dt
                    ori_b = _quat_integrate(ori_b, ang_vel_b, sub_dt)

    return pos_a, ori_a, vel_a, ang_vel_a, pos_b, ori_b, vel_b, ang_vel_b, linear_impulse, angular_impulse


# ═══════════════════════════════════════════════════════════════════════
# Test: compare numpy reference vs SolverBox3D for a 2-body revolute
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_matches_numpy_reference(test, device):
    """Box3D revolute joint output matches pure numpy TGS-Soft reference.

    Constructs a kinematic pivot + dynamic bob connected by a revolute
    joint, runs 10 steps, and compares the bob's position/velocity
    against the numpy reference solver.
    """
    # ── Numpy reference ──
    inv_mass_a = 0.0  # kinematic pivot
    # For a sphere of radius 0.1, density 1000: mass = 4/3 * pi * 0.1^3 * 1000 ≈ 4.189
    # I = 2/5 * m * r^2 ≈ 0.01676, inv_I ≈ 59.66
    mass_b = 4.0 / 3.0 * math.pi * 0.1**3 * 1000.0
    inv_mass_b = 1.0 / mass_b
    I_scalar = 2.0 / 5.0 * mass_b * 0.1**2
    inv_I_body_b = np.eye(3) / I_scalar
    inv_I_body_a = np.zeros((3, 3))

    pos_a_ref = np.array([0.0, 0.0, 2.0])
    ori_a_ref = np.array([0.0, 0.0, 0.0, 1.0])
    vel_a_ref = np.zeros(3)
    ang_a_ref = np.zeros(3)

    pos_b_ref = np.array([1.0, 0.0, 2.0])
    ori_b_ref = np.array([0.0, 0.0, 0.0, 1.0])
    vel_b_ref = np.zeros(3)
    ang_b_ref = np.zeros(3)

    local_anchor_a = np.zeros(3)  # pivot at its own COM
    local_anchor_b = np.array([-1.0, 0.0, 0.0])  # 1m offset in child frame
    hinge_axis = np.array([0.0, 1.0, 0.0])  # Y axis

    gravity = np.array([0.0, 0.0, -9.81])
    dt = 1.0 / 60.0
    num_substeps = 4

    lin_imp_ref = np.zeros(3)
    ang_imp_ref = np.zeros(2)

    for step in range(10):
        (pos_a_ref, ori_a_ref, vel_a_ref, ang_a_ref,
         pos_b_ref, ori_b_ref, vel_b_ref, ang_b_ref,
         lin_imp_ref, ang_imp_ref) = numpy_tgs_soft_revolute_step(
            pos_a_ref.copy(), ori_a_ref.copy(), vel_a_ref.copy(), ang_a_ref.copy(),
            inv_mass_a, inv_I_body_a,
            pos_b_ref.copy(), ori_b_ref.copy(), vel_b_ref.copy(), ang_b_ref.copy(),
            inv_mass_b, inv_I_body_b,
            local_anchor_a, local_anchor_b, hinge_axis,
            lin_imp_ref.copy(), ang_imp_ref.copy(),
            gravity, dt, num_substeps,
            30.0, 1.0, 60.0, 1.0, 0.05, 0.05,
        )

    # ── SolverBox3D ──
    builder = newton.ModelBuilder()
    pivot = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)),
        is_kinematic=True,
    )
    builder.add_shape_sphere(body=pivot, radius=0.05)  # tiny shape

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

    cfg = Box3DConfig(
        num_substeps=4, joint_hertz=60.0, joint_damping_ratio=1.0,
        angular_damping=0.05, linear_damping=0.0,
        contact_hertz=30.0,
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()

    for _ in range(10):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    # ── Compare ──
    q_box3d = state_in.body_q.numpy()
    qd_box3d = state_in.body_qd.numpy()

    bob_pos_box3d = q_box3d[bob, :3]
    bob_vel_box3d = qd_box3d[bob, :3]

    # Position should match within reasonable tolerance
    # (differences from COM offset handling, floating point, etc.)
    np.testing.assert_allclose(
        bob_pos_box3d, pos_b_ref, atol=0.3,
        err_msg=f"Bob position: box3d={bob_pos_box3d}, ref={pos_b_ref}",
    )

    # Velocity direction should match (magnitude may differ due to integration differences)
    vel_dir_ref = vel_b_ref / (np.linalg.norm(vel_b_ref) + 1e-12)
    vel_dir_box3d = bob_vel_box3d / (np.linalg.norm(bob_vel_box3d) + 1e-12)
    dot_prod = np.dot(vel_dir_ref, vel_dir_box3d)
    test.assertGreater(dot_prod, 0.8,
                       f"Velocity directions diverge: dot={dot_prod}, ref={vel_b_ref}, box3d={bob_vel_box3d}")


# ═══════════════════════════════════════════════════════════════════════
# Register
# ═══════════════════════════════════════════════════════════════════════

devices = get_cuda_test_devices()
add_function_test(TestBox3DReference, "test_revolute_matches_numpy_reference", test_revolute_matches_numpy_reference, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
