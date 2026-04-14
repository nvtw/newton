# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Box3D constraint and joint @wp.func building blocks.

Each test wraps a pure @wp.func in a trivial @wp.kernel, launches it
with known inputs, and compares outputs against hand-computed expected
values.  This validates the core physics math independently of the
fused GPU kernel, shared memory, or graph coloring.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.box3d.constraint_funcs import (
    apply_body_angular_impulse,
    apply_body_impulse,
    compute_contact_effective_mass,
    compute_relative_velocity,
    compute_softness,
    compute_tangent_basis,
    solve_contact_friction,
    solve_contact_normal,
    solve_restitution,
)
from newton._src.solvers.box3d.coloring import color_joints_cpu
from newton._src.solvers.box3d.joint_funcs import (
    _skew_outer_diag,
    _solve_symmetric_3x3,
    solve_fixed_angular,
    solve_revolute_angular,
    solve_revolute_limit,
    solve_revolute_motor,
    solve_revolute_point_to_point,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


# ═══════════════════════════════════════════════════════════════════════
# Test wrapper kernels — thin shims calling the @wp.func under test
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def _test_softness_kernel(
    hertz: float,
    damping_ratio: float,
    h: float,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = compute_softness(hertz, damping_ratio, h)


@wp.kernel
def _test_tangent_basis_kernel(
    normal: wp.vec3,
    out_row0: wp.array[wp.vec3],
    out_row1: wp.array[wp.vec3],
    out_row2: wp.array[wp.vec3],
):
    tid = wp.tid()
    m = compute_tangent_basis(normal)
    out_row0[tid] = wp.vec3(m[0, 0], m[0, 1], m[0, 2])
    out_row1[tid] = wp.vec3(m[1, 0], m[1, 1], m[1, 2])
    out_row2[tid] = wp.vec3(m[2, 0], m[2, 1], m[2, 2])


@wp.kernel
def _test_effective_mass_kernel(
    inv_mass_a: float,
    inv_inertia_a: wp.mat33,
    inv_mass_b: float,
    inv_inertia_b: wp.mat33,
    r_a: wp.vec3,
    r_b: wp.vec3,
    direction: wp.vec3,
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = compute_contact_effective_mass(
        inv_mass_a, inv_inertia_a, inv_mass_b, inv_inertia_b, r_a, r_b, direction
    )


@wp.kernel
def _test_relative_velocity_kernel(
    vel_a: wp.vec3,
    ang_vel_a: wp.vec3,
    r_a: wp.vec3,
    vel_b: wp.vec3,
    ang_vel_b: wp.vec3,
    r_b: wp.vec3,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = compute_relative_velocity(vel_a, ang_vel_a, r_a, vel_b, ang_vel_b, r_b)


@wp.kernel
def _test_normal_impulse_kernel(
    vn: float,
    separation: float,
    lambda_n: float,
    normal_mass: float,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    inv_sub_dt: float,
    contact_speed: float,
    use_bias: int,
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = solve_contact_normal(
        vn, separation, lambda_n, normal_mass,
        bias_rate, mass_scale, impulse_scale,
        inv_sub_dt, contact_speed, use_bias,
    )


@wp.kernel
def _test_friction_impulse_kernel(
    vt: float,
    lambda_f: float,
    max_friction: float,
    tangent_mass: float,
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = solve_contact_friction(vt, lambda_f, max_friction, tangent_mass)


@wp.kernel
def _test_apply_impulse_kernel(
    vel: wp.vec3,
    ang_vel: wp.vec3,
    inv_mass: float,
    inv_inertia: wp.mat33,
    r: wp.vec3,
    impulse: wp.vec3,
    sign: float,
    out_vel: wp.array[wp.vec3],
    out_ang_vel: wp.array[wp.vec3],
):
    tid = wp.tid()
    out_vel[tid] = apply_body_impulse(vel, ang_vel, inv_mass, inv_inertia, r, impulse, sign)
    out_ang_vel[tid] = apply_body_angular_impulse(ang_vel, inv_inertia, r, impulse, sign)


@wp.kernel
def _test_restitution_kernel(
    vn: float,
    pre_vel: float,
    lambda_n: float,
    total_ni: float,
    normal_mass: float,
    restitution: float,
    threshold: float,
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = solve_restitution(vn, pre_vel, lambda_n, total_ni, normal_mass, restitution, threshold)


@wp.kernel
def _test_solve_3x3_kernel(
    K00: float, K01: float, K02: float,
    K11: float, K12: float, K22: float,
    rhs_x: float, rhs_y: float, rhs_z: float,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = _solve_symmetric_3x3(K00, K01, K02, K11, K12, K22, rhs_x, rhs_y, rhs_z)


@wp.kernel
def _test_revolute_p2p_kernel(
    vel_a: wp.vec3, ang_vel_a: wp.vec3,
    vel_b: wp.vec3, ang_vel_b: wp.vec3,
    r_a: wp.vec3, r_b: wp.vec3,
    separation: wp.vec3,
    inv_mass_a: float, inv_inertia_a: wp.mat33,
    inv_mass_b: float, inv_inertia_b: wp.mat33,
    linear_impulse: wp.vec3,
    bias_rate: float, mass_scale: float, impulse_scale: float,
    use_bias: int,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = solve_revolute_point_to_point(
        vel_a, ang_vel_a, vel_b, ang_vel_b,
        r_a, r_b, separation,
        inv_mass_a, inv_inertia_a, inv_mass_b, inv_inertia_b,
        linear_impulse, bias_rate, mass_scale, impulse_scale, use_bias,
    )


@wp.kernel
def _test_revolute_angular_kernel(
    ang_vel_a: wp.vec3, ang_vel_b: wp.vec3,
    hinge_axis: wp.vec3,
    inv_inertia_a: wp.mat33, inv_inertia_b: wp.mat33,
    angular_impulse: wp.vec2,
    bias_rate: float, mass_scale: float, impulse_scale: float,
    use_bias: int,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = solve_revolute_angular(
        ang_vel_a, ang_vel_b, hinge_axis,
        inv_inertia_a, inv_inertia_b,
        angular_impulse, bias_rate, mass_scale, impulse_scale, use_bias,
    )


@wp.kernel
def _test_revolute_motor_kernel(
    ang_vel_a: wp.vec3, ang_vel_b: wp.vec3,
    hinge_axis: wp.vec3,
    inv_inertia_a: wp.mat33, inv_inertia_b: wp.mat33,
    motor_impulse: float, motor_speed: float,
    max_motor_torque: float, sub_dt: float,
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = solve_revolute_motor(
        ang_vel_a, ang_vel_b, hinge_axis,
        inv_inertia_a, inv_inertia_b,
        motor_impulse, motor_speed, max_motor_torque, sub_dt,
    )


@wp.kernel
def _test_revolute_limit_kernel(
    ang_vel_a: wp.vec3, ang_vel_b: wp.vec3,
    hinge_axis: wp.vec3,
    inv_inertia_a: wp.mat33, inv_inertia_b: wp.mat33,
    limit_impulse: float, angle_error: float,
    bias_rate: float, mass_scale: float, impulse_scale: float,
    inv_sub_dt: float, use_bias: int, is_upper: int,
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = solve_revolute_limit(
        ang_vel_a, ang_vel_b, hinge_axis,
        inv_inertia_a, inv_inertia_b,
        limit_impulse, angle_error,
        bias_rate, mass_scale, impulse_scale,
        inv_sub_dt, use_bias, is_upper,
    )


@wp.kernel
def _test_fixed_angular_kernel(
    ang_vel_a: wp.vec3, ang_vel_b: wp.vec3,
    inv_inertia_a: wp.mat33, inv_inertia_b: wp.mat33,
    angular_impulse: wp.vec3,
    bias_rate: float, mass_scale: float, impulse_scale: float,
    use_bias: int,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = solve_fixed_angular(
        ang_vel_a, ang_vel_b,
        inv_inertia_a, inv_inertia_b,
        angular_impulse, bias_rate, mass_scale, impulse_scale, use_bias,
    )


# ═══════════════════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════════════════


class TestBox3DConstraintFuncs(unittest.TestCase):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Softness tests
# ═══════════════════════════════════════════════════════════════════════


def test_softness_critical_damping(test, device):
    """Critical damping (zeta=1) with known hertz produces correct softness params."""
    hertz = 30.0
    zeta = 1.0
    h = 1.0 / 60.0 / 4.0  # substep at 60 Hz with 4 substeps
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_test_softness_kernel, dim=1, inputs=[hertz, zeta, h], outputs=[out], device=device)
    result = out.numpy()[0]

    omega = 2.0 * math.pi * hertz
    a1 = 2.0 * zeta + h * omega
    a2 = h * omega * a1
    a3 = 1.0 / (1.0 + a2)
    expected = np.array([omega / a1, a2 * a3, a3])

    np.testing.assert_allclose(result, expected, rtol=1e-5)
    # Key invariant: mass_scale + impulse_scale == 1
    test.assertAlmostEqual(float(result[1] + result[2]), 1.0, places=6)


def test_softness_zero_hertz(test, device):
    """Zero hertz produces all-zero softness (fully rigid constraint)."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_test_softness_kernel, dim=1, inputs=[0.0, 1.0, 0.01], outputs=[out], device=device)
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# Tangent basis tests
# ═══════════════════════════════════════════════════════════════════════


def test_tangent_basis_y_up(test, device):
    """Tangent basis from Y-up normal is orthonormal."""
    normal = wp.vec3(0.0, 1.0, 0.0)
    r0 = wp.zeros(1, dtype=wp.vec3, device=device)
    r1 = wp.zeros(1, dtype=wp.vec3, device=device)
    r2 = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_test_tangent_basis_kernel, dim=1, inputs=[normal], outputs=[r0, r1, r2], device=device)
    n = r0.numpy()[0]
    t1 = r1.numpy()[0]
    t2 = r2.numpy()[0]

    # Row 0 should match the normal
    np.testing.assert_allclose(n, [0, 1, 0], atol=1e-6)
    # All rows should be unit length
    test.assertAlmostEqual(float(np.linalg.norm(t1)), 1.0, places=5)
    test.assertAlmostEqual(float(np.linalg.norm(t2)), 1.0, places=5)
    # All pairs orthogonal
    test.assertAlmostEqual(float(np.dot(n, t1)), 0.0, places=5)
    test.assertAlmostEqual(float(np.dot(n, t2)), 0.0, places=5)
    test.assertAlmostEqual(float(np.dot(t1, t2)), 0.0, places=5)


def test_tangent_basis_arbitrary_normal(test, device):
    """Tangent basis from an arbitrary normal is orthonormal."""
    n_raw = np.array([0.3, -0.7, 0.5])
    n_raw /= np.linalg.norm(n_raw)
    normal = wp.vec3(*n_raw.tolist())
    r0 = wp.zeros(1, dtype=wp.vec3, device=device)
    r1 = wp.zeros(1, dtype=wp.vec3, device=device)
    r2 = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_test_tangent_basis_kernel, dim=1, inputs=[normal], outputs=[r0, r1, r2], device=device)
    rows = [r0.numpy()[0], r1.numpy()[0], r2.numpy()[0]]
    for i in range(3):
        test.assertAlmostEqual(float(np.linalg.norm(rows[i])), 1.0, places=5)
    for i in range(3):
        for j in range(i + 1, 3):
            test.assertAlmostEqual(float(np.dot(rows[i], rows[j])), 0.0, places=5)


# ═══════════════════════════════════════════════════════════════════════
# Effective mass tests
# ═══════════════════════════════════════════════════════════════════════


def test_effective_mass_equal_masses(test, device):
    """Two equal-mass point bodies with zero lever arms give m_eff = m/2."""
    inv_mass = 1.0  # mass = 1 kg
    inv_I = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # point mass
    r = wp.vec3(0.0, 0.0, 0.0)
    d = wp.vec3(0.0, 1.0, 0.0)
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_effective_mass_kernel, dim=1,
        inputs=[inv_mass, inv_I, inv_mass, inv_I, r, r, d],
        outputs=[out], device=device,
    )
    # K = inv_m_a + inv_m_b = 2, effective mass = 1/K = 0.5
    test.assertAlmostEqual(float(out.numpy()[0]), 0.5, places=5)


def test_effective_mass_static_body(test, device):
    """One static body (inv_mass=0): effective mass equals dynamic body mass."""
    inv_I_dyn = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inv_I_zero = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r = wp.vec3(0.0, 0.0, 0.0)
    d = wp.vec3(0.0, 1.0, 0.0)
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_effective_mass_kernel, dim=1,
        inputs=[1.0, inv_I_dyn, 0.0, inv_I_zero, r, r, d],
        outputs=[out], device=device,
    )
    # K = 1 + 0 = 1, effective mass = 1
    test.assertAlmostEqual(float(out.numpy()[0]), 1.0, places=5)


def test_effective_mass_with_lever_arm(test, device):
    """Non-zero lever arm increases K (reduces effective mass)."""
    inv_mass = 1.0
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inv_I_zero = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r_a = wp.vec3(0.0, 0.0, 1.0)  # lever arm along z
    r_b = wp.vec3(0.0, 0.0, 0.0)
    d = wp.vec3(0.0, 1.0, 0.0)  # normal along y
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_effective_mass_kernel, dim=1,
        inputs=[inv_mass, inv_I, 0.0, inv_I_zero, r_a, r_b, d],
        outputs=[out], device=device,
    )
    result = float(out.numpy()[0])
    # K = inv_mass_a + dot(d, cross(inv_I * cross(r_a, d), r_a))
    # cross(r_a, d) = cross((0,0,1), (0,1,0)) = (-1,0,0)
    # inv_I * (-1,0,0) = (-1,0,0)
    # cross((-1,0,0), (0,0,1)) = (0,1,0)
    # dot((0,1,0), (0,1,0)) = 1
    # K = 1 + 1 = 2, effective_mass = 0.5
    test.assertAlmostEqual(result, 0.5, places=5)


# ═══════════════════════════════════════════════════════════════════════
# Normal impulse tests
# ═══════════════════════════════════════════════════════════════════════


def test_normal_impulse_penetrating(test, device):
    """Penetrating contact with bias produces exact impulse from Box2D formula."""
    out = wp.zeros(1, dtype=float, device=device)
    # Bodies approaching at 1 m/s, penetrating 0.01 m
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            -1.0,   # vn: approaching
            -0.01,  # separation: penetrating
            0.0,    # lambda_n: no accumulated
            1.0,    # normal_mass
            10.0,   # bias_rate
            0.8,    # mass_scale
            0.2,    # impulse_scale
            240.0,  # inv_sub_dt
            10.0,   # contact_speed
            1,      # use_bias
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Hand computation:
    # sep = -0.01 <= 0, so penetrating path:
    # velocity_bias = max(0.8 * 10.0 * (-0.01), -10.0) = max(-0.08, -10) = -0.08
    # ms = 0.8, isv = 0.2
    # impulse = -1.0 * (0.8 * (-1.0) + (-0.08)) - 0.2 * 0.0 = -1.0 * (-0.88) = 0.88
    # new_lambda = max(0 + 0.88, 0) = 0.88, delta = 0.88
    test.assertAlmostEqual(delta, 0.88, places=4)


def test_normal_impulse_separating(test, device):
    """Bodies moving apart with zero penetration produce zero impulse."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            1.0,    # vn: separating
            0.0,    # separation: just touching
            0.0,    # lambda_n
            1.0,    # normal_mass
            10.0, 0.8, 0.2, 240.0, 10.0, 1,
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Impulse = -normal_mass * (ms * vn + 0) = -0.8*1.0 = -0.8
    # new_lambda = max(0 + (-0.8), 0) = 0
    # delta = 0 - 0 = 0
    test.assertAlmostEqual(delta, 0.0, places=6)


def test_normal_impulse_clamp_nonnegative(test, device):
    """Accumulated normal impulse never goes negative."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            5.0,    # vn: strongly separating
            0.0,    # separation
            0.5,    # lambda_n: has accumulated impulse
            1.0,    # normal_mass
            10.0, 0.8, 0.2, 240.0, 10.0, 1,
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Delta must bring lambda down to zero but not below
    test.assertAlmostEqual(delta, -0.5, places=5)


def test_normal_impulse_relaxation_no_bias(test, device):
    """Relaxation pass (use_bias=0) has no position correction."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            -1.0,   # vn: approaching
            -0.05,  # separation: deeply penetrating
            0.0,    # lambda_n
            1.0,    # normal_mass
            10.0, 0.8, 0.2, 240.0, 10.0,
            0,      # use_bias = false
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # ms=1, isv=0, bias=0: impulse = -1.0 * (1.0*(-1.0) + 0) = 1.0
    test.assertAlmostEqual(delta, 1.0, places=5)


def test_normal_impulse_speculative(test, device):
    """Positive separation (speculative contact) uses sep * inv_dt as bias."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            -2.0,   # vn: approaching fast
            0.01,   # separation: not yet touching (speculative)
            0.0,    # lambda_n
            1.0,    # normal_mass
            10.0, 0.8, 0.2,
            240.0,  # inv_sub_dt
            10.0, 1,
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Speculative: bias = 0.01 * 240.0 = 2.4, ms=1, isv=0
    # impulse = -1.0 * (1.0 * (-2.0) + 2.4) = -0.4
    # new_lambda = max(0 + (-0.4), 0) = 0.0
    test.assertAlmostEqual(delta, 0.0, places=5)


# ═══════════════════════════════════════════════════════════════════════
# Friction tests
# ═══════════════════════════════════════════════════════════════════════


def test_friction_within_cone(test, device):
    """Small tangential velocity stays within Coulomb cone."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_friction_impulse_kernel, dim=1,
        inputs=[0.1, 0.0, 10.0, 1.0],  # vt, lambda_f, max_friction, tangent_mass
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # impulse = -1.0 * 0.1 = -0.1
    # clamp(0 + (-0.1), -10, 10) = -0.1
    test.assertAlmostEqual(delta, -0.1, places=5)


def test_friction_clamped_to_cone(test, device):
    """Large tangential velocity is clamped by Coulomb friction limit."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_friction_impulse_kernel, dim=1,
        inputs=[100.0, 0.0, 0.5, 1.0],  # vt, lambda_f, max_friction, tangent_mass
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # impulse = -100.0, clamp(0 + (-100), -0.5, 0.5) = -0.5, delta = -0.5
    test.assertAlmostEqual(delta, -0.5, places=5)


def test_friction_accumulated_clamp(test, device):
    """Accumulated friction impulse respects the cone even with prior accumulation."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_friction_impulse_kernel, dim=1,
        inputs=[1.0, 0.4, 0.5, 1.0],  # vt, lambda_f=0.4, max=0.5, mass=1
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # impulse = -1.0, clamp(0.4 + (-1.0), -0.5, 0.5) = -0.5, delta = -0.5 - 0.4 = -0.9
    test.assertAlmostEqual(delta, -0.9, places=5)


# ═══════════════════════════════════════════════════════════════════════
# Relative velocity tests
# ═══════════════════════════════════════════════════════════════════════


def test_relative_velocity_pure_linear(test, device):
    """Pure linear velocity, no rotation — v_rel = vB - vA."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _test_relative_velocity_kernel, dim=1,
        inputs=[
            wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(3.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [2.0, 0.0, 0.0], atol=1e-6)


def test_relative_velocity_with_rotation(test, device):
    """Angular velocity contributes to contact-point velocity."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    # Body A: no linear vel, rotating around z at 1 rad/s, contact at r=(1,0,0)
    # w x r = (0,0,1) x (1,0,0) = (0,1,0)
    wp.launch(
        _test_relative_velocity_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    # v_rel = (0) - (0 + (0,1,0)) = (0, -1, 0)
    np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Apply impulse tests
# ═══════════════════════════════════════════════════════════════════════


def test_apply_impulse_linear(test, device):
    """Linear impulse changes velocity by inv_mass * impulse."""
    out_v = wp.zeros(1, dtype=wp.vec3, device=device)
    out_w = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_apply_impulse_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),  # vel
            wp.vec3(0.0, 0.0, 0.0),  # ang_vel
            2.0,                       # inv_mass
            inv_I,
            wp.vec3(0.0, 0.0, 0.0),  # r = 0 (no torque)
            wp.vec3(1.0, 0.0, 0.0),  # impulse
            1.0,                       # sign (body B)
        ],
        outputs=[out_v, out_w], device=device,
    )
    v = out_v.numpy()[0]
    w = out_w.numpy()[0]
    np.testing.assert_allclose(v, [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(w, [0.0, 0.0, 0.0], atol=1e-6)


def test_apply_impulse_with_torque(test, device):
    """Non-zero lever arm produces angular velocity change."""
    out_v = wp.zeros(1, dtype=wp.vec3, device=device)
    out_w = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_apply_impulse_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            1.0,
            inv_I,
            wp.vec3(1.0, 0.0, 0.0),  # r along x
            wp.vec3(0.0, 1.0, 0.0),  # impulse along y
            -1.0,                      # sign (body A)
        ],
        outputs=[out_v, out_w], device=device,
    )
    w = out_w.numpy()[0]
    # cross(r, impulse) = cross((1,0,0), (0,1,0)) = (0,0,1)
    # ang_vel += -1 * I^-1 * (0,0,1) = (0,0,-1)
    np.testing.assert_allclose(w, [0.0, 0.0, -1.0], atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Restitution tests
# ═══════════════════════════════════════════════════════════════════════


def test_restitution_bouncing(test, device):
    """Fast approach + nonzero restitution produces bounce impulse."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_restitution_kernel, dim=1,
        inputs=[
            0.0,   # vn: currently zero (after solve)
            -5.0,  # pre_solve: was approaching at 5 m/s
            0.0,   # lambda_n
            1.0,   # total_ni: contact was active
            1.0,   # normal_mass
            0.8,   # restitution
            1.0,   # threshold
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # impulse = -1.0 * (0.0 + 0.8 * (-5.0)) = 4.0
    test.assertAlmostEqual(delta, 4.0, places=5)


def test_restitution_slow_approach_no_bounce(test, device):
    """Approach speed below threshold produces no bounce."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_restitution_kernel, dim=1,
        inputs=[0.0, -0.5, 0.0, 1.0, 1.0, 0.8, 1.0],
        outputs=[out], device=device,
    )
    test.assertAlmostEqual(float(out.numpy()[0]), 0.0, places=6)


def test_restitution_zero_coeff(test, device):
    """Zero restitution coefficient produces no bounce."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_restitution_kernel, dim=1,
        inputs=[0.0, -5.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        outputs=[out], device=device,
    )
    test.assertAlmostEqual(float(out.numpy()[0]), 0.0, places=6)


# ═══════════════════════════════════════════════════════════════════════
# 3x3 solve tests
# ═══════════════════════════════════════════════════════════════════════


def test_solve_3x3_identity(test, device):
    """Identity matrix solve: K=I, rhs=v → x=v."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _test_solve_3x3_kernel, dim=1,
        inputs=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0], atol=1e-5)


def test_solve_3x3_diagonal(test, device):
    """Diagonal matrix solve: K=diag(2,3,4) → x = rhs / diag."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _test_solve_3x3_kernel, dim=1,
        inputs=[2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 6.0, 9.0, 8.0],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [3.0, 3.0, 2.0], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Revolute joint point-to-point tests
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_p2p_coincident_relaxation(test, device):
    """Coincident anchors, zero velocity → zero impulse (nothing to correct)."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_p2p_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),  # vel/angvel A
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),  # vel/angvel B
            wp.vec3(0.0, 0.5, 0.0), wp.vec3(0.0, 0.5, 0.0),  # r_a, r_b
            wp.vec3(0.0, 0.0, 0.0),  # separation = 0
            1.0, inv_I, 1.0, inv_I,  # inv_mass, inv_inertia
            wp.vec3(0.0, 0.0, 0.0),  # accumulated impulse
            10.0, 0.8, 0.2, 0,       # softness params, use_bias=0
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_revolute_p2p_velocity_correction(test, device):
    """Bodies drifting apart produce corrective impulse toward each other."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_p2p_kernel, dim=1,
        inputs=[
            wp.vec3(-1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),  # A moving left
            wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),   # B moving right
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),   # r_a, r_b = 0
            wp.vec3(0.0, 0.0, 0.0),  # separation = 0
            1.0, inv_I, 1.0, inv_I,
            wp.vec3(0.0, 0.0, 0.0),
            10.0, 1.0, 0.0, 0,  # ms=1, is=0 (relaxation)
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    # Cdot = (1,0,0) - (-1,0,0) = (2,0,0)
    # K = (1+1)I = 2I, K^-1 = 0.5I
    # impulse = -1.0 * 0.5 * (2,0,0) = (-1,0,0)
    np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Revolute angular tests
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_angular_aligned(test, device):
    """Zero relative angular velocity perpendicular to hinge → zero impulse."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_angular_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),  # hinge along Y
            inv_I, inv_I,
            wp.vec2(0.0, 0.0),
            10.0, 1.0, 0.0, 0,
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_revolute_angular_correction(test, device):
    """Relative angular velocity perpendicular to hinge is corrected with exact magnitude."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # Body B rotating around X at 2 rad/s, hinge along Y
    wp.launch(
        _test_revolute_angular_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(2.0, 0.0, 0.0),  # B rotating around X (perp to Y hinge)
            wp.vec3(0.0, 1.0, 0.0),  # hinge along Y
            inv_I, inv_I,
            wp.vec2(0.0, 0.0),
            10.0, 1.0, 0.0, 0,  # relaxation: ms=1, is=0
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    # Hand computation:
    # h = (0,1,0). abs(h[1])=1 >= 0.9, so b1 = (0, -h[2], h[1]) = (0,0,1)
    # b2 = cross(h, b1) = cross((0,1,0),(0,0,1)) = (1,0,0)
    # dw = (2,0,0), k_ang = 2*I
    # k1 = dot(b1, 2*b1) = 2, am1 = 0.5
    # k2 = dot(b2, 2*b2) = 2, am2 = 0.5
    # cdot1 = dot((2,0,0), (0,0,1)) = 0 → i1 = 0
    # cdot2 = dot((2,0,0), (1,0,0)) = 2 → i2 = -1.0 * 0.5 * 2 = -1.0
    # result = 0*(0,0,1) + (-1)*(1,0,0) = (-1, 0, 0)
    np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Revolute motor tests
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_motor_speed_tracking(test, device):
    """Motor produces impulse to reach target speed."""
    out = wp.zeros(1, dtype=float, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_motor_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),  # hinge along Y
            inv_I, inv_I,
            0.0,   # motor_impulse accumulated
            5.0,   # target motor speed
            100.0, # max torque
            0.004, # sub_dt
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # cdot = 0 - 5 = -5, am = 1/(1+1) = 0.5
    # impulse = -0.5 * (-5) = 2.5
    # clamp(0 + 2.5, -0.4, 0.4) = 0.4
    test.assertAlmostEqual(delta, 0.4, places=5)


def test_revolute_motor_torque_clamp(test, device):
    """Motor impulse is clamped by max_torque * sub_dt."""
    out = wp.zeros(1, dtype=float, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_motor_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            inv_I, inv_I,
            0.0, 100.0,  # large target speed
            1.0, 0.004,  # small max torque
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # max_impulse = 1.0 * 0.004 = 0.004
    test.assertAlmostEqual(delta, 0.004, places=5)


# ═══════════════════════════════════════════════════════════════════════
# Revolute limit tests
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_limit_within_range(test, device):
    """Limit not reached (angle_error > 0, speculative) — limited correction."""
    out = wp.zeros(1, dtype=float, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_limit_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            inv_I, inv_I,
            0.0,     # limit_impulse
            0.1,     # angle_error > 0 (within limit)
            10.0, 0.8, 0.2,
            240.0, 1, 0,  # use_bias=1, lower limit
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Speculative: bias = 0.1 * 240 = 24
    # cdot = 0, am = 0.5
    # impulse = -0.5 * (0 + 24) = -12
    # new = max(0 + (-12), 0) = 0
    test.assertAlmostEqual(delta, 0.0, places=5)


def test_revolute_limit_violated(test, device):
    """Limit violated (angle_error < 0) produces exact corrective impulse."""
    out = wp.zeros(1, dtype=float, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_limit_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            inv_I, inv_I,
            0.0,       # limit_impulse
            -0.1,      # angle_error < 0 (violated)
            10.0, 0.8, 0.2,
            240.0, 1, 0,
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Hand computation:
    # angle_error = -0.1 <= 0, so bias path:
    # bias = mass_scale * bias_rate * angle_error = 0.8 * 10.0 * (-0.1) = -0.8
    # ms = 0.8, isv = 0.2
    # k = dot(h, (inv_I_a + inv_I_b) * h) = dot((0,1,0), 2*(0,1,0)) = 2
    # am = 1/2 = 0.5
    # cdot = dot((0,0,0)-(0,0,0), (0,1,0)) = 0 (lower limit, is_upper=0)
    # impulse = -0.8 * 0.5 * (0 + (-0.8)) - 0.2 * 0 = 0.32
    # new = max(0 + 0.32, 0) = 0.32
    test.assertAlmostEqual(delta, 0.32, places=4)


# ═══════════════════════════════════════════════════════════════════════
# Fixed joint angular tests
# ═══════════════════════════════════════════════════════════════════════


def test_fixed_angular_zero_velocity(test, device):
    """Zero relative angular velocity → zero angular impulse."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_fixed_angular_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),
            inv_I, inv_I,
            wp.vec3(0.0, 0.0, 0.0),
            10.0, 1.0, 0.0, 0,
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_fixed_angular_correction(test, device):
    """Relative angular velocity is fully corrected by fixed joint."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_fixed_angular_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(2.0, 0.0, 0.0),  # B rotating around X
            inv_I, inv_I,
            wp.vec3(0.0, 0.0, 0.0),
            10.0, 1.0, 0.0, 0,  # relaxation: ms=1, is=0
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    # dw = (2,0,0), k_ang = 2*I, k_ang^-1 = 0.5*I
    # impulse = -0.5 * (2,0,0) = (-1,0,0)
    np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Softness: hand-computed (not mirror)
# ═══════════════════════════════════════════════════════════════════════


def test_softness_hand_computed(test, device):
    """Softness with hertz=30, zeta=1.0, h=1/240: verify against manual calculation."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    h = 1.0 / 240.0  # 4 substeps at 60Hz
    wp.launch(_test_softness_kernel, dim=1, inputs=[30.0, 1.0, h], outputs=[out], device=device)
    result = out.numpy()[0]
    # omega = 2*pi*30 = 188.4956; a1 = 2 + h*omega = 2.78540; a2 = h*omega*a1 = 2.18712
    # a3 = 1/3.18712 = 0.31376; bias_rate = omega/a1 = 67.672; mass_scale = a2*a3 = 0.68624
    test.assertAlmostEqual(float(result[0]), 67.672, delta=0.01)  # bias_rate
    test.assertAlmostEqual(float(result[1]), 0.6862, delta=0.001)  # mass_scale
    test.assertAlmostEqual(float(result[2]), 0.3138, delta=0.001)  # impulse_scale


# ═══════════════════════════════════════════════════════════════════════
# Tangent basis: verify specific output vectors
# ═══════════════════════════════════════════════════════════════════════


def test_tangent_basis_z_up_exact(test, device):
    """Tangent basis for Z-up normal should produce X and Y as tangents."""
    normal = wp.vec3(0.0, 0.0, 1.0)
    r0 = wp.zeros(1, dtype=wp.vec3, device=device)
    r1 = wp.zeros(1, dtype=wp.vec3, device=device)
    r2 = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(_test_tangent_basis_kernel, dim=1, inputs=[normal], outputs=[r0, r1, r2], device=device)
    n = r0.numpy()[0]
    t1 = r1.numpy()[0]
    t2 = r2.numpy()[0]
    np.testing.assert_allclose(n, [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(t1, [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(t2, [0, 1, 0], atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Fixed angular with off-diagonal inertia and multi-axis dw
# ═══════════════════════════════════════════════════════════════════════


def test_fixed_angular_off_diagonal_inertia(test, device):
    """Fixed angular correction with non-diagonal inertia, verified against numpy."""
    # Non-diagonal symmetric positive definite inv_inertia
    inv_I_np = np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.5]])
    inv_I = wp.mat33(*inv_I_np.flatten().tolist())
    dw = np.array([1.0, -0.5, 0.3])

    # k_ang = inv_I_a + inv_I_b = 2 * inv_I_np (same body inertia)
    k_ang = 2.0 * inv_I_np
    sol = np.linalg.solve(k_ang, dw)  # ms=1, isv=0 for relaxation
    expected = -sol  # impulse = -K^-1 @ dw

    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _test_fixed_angular_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(*dw.tolist()),
            inv_I, inv_I,
            wp.vec3(0.0, 0.0, 0.0),
            10.0, 1.0, 0.0, 0,  # relaxation
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Revolute P2P with both non-zero lever arms in different directions
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_p2p_asymmetric_levers(test, device):
    """P2P with asymmetric lever arms, verified against numpy K-matrix solve."""
    inv_I_a = np.eye(3) * 2.0
    inv_I_b = np.eye(3) * 1.0
    r_a = np.array([0.3, 0.0, 0.5])
    r_b = np.array([-0.2, 0.4, 0.0])
    inv_mass_a, inv_mass_b = 1.0, 0.5
    vel_a = np.array([0.5, -0.3, 0.0])
    vel_b = np.array([-0.2, 0.1, 0.4])
    ang_a = np.array([0.0, 0.1, -0.1])
    ang_b = np.array([0.2, 0.0, 0.3])

    # Build K in numpy
    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    K = (inv_mass_a + inv_mass_b) * np.eye(3)
    K += skew(r_a).T @ inv_I_a @ skew(r_a)
    K += skew(r_b).T @ inv_I_b @ skew(r_b)

    Cdot = (vel_b + np.cross(ang_b, r_b)) - (vel_a + np.cross(ang_a, r_a))
    sol = np.linalg.solve(K, Cdot)
    expected = -sol  # ms=1, isv=0 for relaxation

    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _test_revolute_p2p_kernel, dim=1,
        inputs=[
            wp.vec3(*vel_a.tolist()), wp.vec3(*ang_a.tolist()),
            wp.vec3(*vel_b.tolist()), wp.vec3(*ang_b.tolist()),
            wp.vec3(*r_a.tolist()), wp.vec3(*r_b.tolist()),
            wp.vec3(0.0, 0.0, 0.0),  # separation=0 (relaxation)
            inv_mass_a, wp.mat33(*inv_I_a.flatten().tolist()),
            inv_mass_b, wp.mat33(*inv_I_b.flatten().tolist()),
            wp.vec3(0.0, 0.0, 0.0),  # accumulated
            10.0, 1.0, 0.0, 0,  # relaxation
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# Restitution with accumulated lambda_n > 0
# ═══════════════════════════════════════════════════════════════════════


def test_restitution_with_accumulated_impulse(test, device):
    """Restitution with pre-existing accumulated impulse, verify clamp behavior."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_restitution_kernel, dim=1,
        inputs=[
            -2.0,  # vn: still approaching (post-solve residual)
            -8.0,  # pre_vel: was approaching fast
            3.0,   # lambda_n: already has accumulated impulse
            5.0,   # total_ni: active contact
            1.0,   # normal_mass
            0.5,   # restitution
            1.0,   # threshold
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # impulse = -1.0 * (-2.0 + 0.5 * (-8.0)) = -1.0 * (-6.0) = 6.0
    # new_lambda = max(3.0 + 6.0, 0) = 9.0
    # delta = 9.0 - 3.0 = 6.0
    test.assertAlmostEqual(delta, 6.0, places=4)


# ═══════════════════════════════════════════════════════════════════════
# Normal impulse: speculative during relaxation (Box2D fix)
# ═══════════════════════════════════════════════════════════════════════


def test_normal_impulse_speculative_relaxation(test, device):
    """Speculative contacts apply bias even during relaxation (use_bias=0).

    This matches Box2D where speculative bias is always active.
    """
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            -5.0,   # vn: approaching fast
            0.02,   # separation: speculative gap
            0.0,    # lambda_n
            1.0,    # normal_mass
            10.0, 0.8, 0.2,
            240.0,  # inv_sub_dt
            10.0,
            0,      # use_bias = 0 (relaxation!)
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Speculative: bias = 0.02 * 240 = 4.8, ms=1, isv=0 (even in relaxation)
    # impulse = -1.0 * (1.0 * (-5.0) + 4.8) = 0.2
    # new_lambda = max(0 + 0.2, 0) = 0.2
    test.assertAlmostEqual(delta, 0.2, places=4)


# ═══════════════════════════════════════════════════════════════════════
# Skew outer diagonal (direct test of _skew_outer_diag)
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def _test_skew_outer_kernel(
    r: wp.vec3, inv_I: wp.mat33,
    out: wp.array[wp.mat33],
):
    tid = wp.tid()
    out[tid] = _skew_outer_diag(r, inv_I)


def test_skew_outer_diag_identity_inertia(test, device):
    """_skew_outer_diag with identity inertia matches numpy [r]_x^T @ I^-1 @ [r]_x."""
    r_val = np.array([1.0, 2.0, 3.0])
    inv_I_val = np.eye(3)

    # Compute reference with numpy
    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rx = skew(r_val)
    expected = rx.T @ inv_I_val @ rx

    out = wp.zeros(1, dtype=wp.mat33, device=device)
    wp.launch(
        _test_skew_outer_kernel, dim=1,
        inputs=[wp.vec3(*r_val.tolist()), wp.mat33(*inv_I_val.flatten().tolist())],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_skew_outer_diag_non_diagonal_inertia(test, device):
    """_skew_outer_diag with non-diagonal inertia matches numpy reference."""
    r_val = np.array([0.5, -0.3, 0.8])
    # Symmetric positive-definite inverse inertia
    inv_I_val = np.array([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]])

    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rx = skew(r_val)
    expected = rx.T @ inv_I_val @ rx

    out = wp.zeros(1, dtype=wp.mat33, device=device)
    wp.launch(
        _test_skew_outer_kernel, dim=1,
        inputs=[wp.vec3(*r_val.tolist()), wp.mat33(*inv_I_val.flatten().tolist())],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Revolute P2P with lever arms + bias (exercises full K-matrix + bias)
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_p2p_with_lever_arms_biased(test, device):
    """P2P with non-zero lever arms and bias exercises _skew_outer_diag and softness."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # Bodies separated: A at origin, B at (0,0,0) but anchors create 1m gap along x
    # r_a = (0.5, 0, 0), r_b = (-0.5, 0, 0)
    # separation = (0.5 + (-0.5)) - (0 + 0.5) = ... need actual positions
    # Let's use: pos_A=(0,0,0), pos_B=(2,0,0), r_a=(0.5,0,0), r_b=(-0.5,0,0)
    # anchor_A_world = (0.5, 0, 0), anchor_B_world = (1.5, 0, 0)
    # separation = (1.5,0,0) - (0.5,0,0) = (1,0,0)
    wp.launch(
        _test_revolute_p2p_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),  # vel/angvel A
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0),  # vel/angvel B
            wp.vec3(0.5, 0.0, 0.0), wp.vec3(-0.5, 0.0, 0.0),  # r_a, r_b
            wp.vec3(1.0, 0.0, 0.0),  # separation (anchor gap)
            1.0, inv_I, 1.0, inv_I,
            wp.vec3(0.0, 0.0, 0.0),  # accumulated impulse
            10.0, 0.8, 0.2, 1,       # softness, use_bias=1
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]

    # Hand computation:
    # Cdot = (0,0,0), bias = 10.0 * (1,0,0) = (10,0,0), ms=0.8, isv=0.2
    # K = 2*I + [r_a]^T I^-1 [r_a] + [r_b]^T I^-1 [r_b]
    # r_a = (0.5,0,0): [r_a]^T [r_a] = diag(0, 0.25, 0.25)
    # r_b = (-0.5,0,0): [r_b]^T [r_b] = diag(0, 0.25, 0.25)
    # K = diag(2, 2.5, 2.5)
    # K^-1 rhs = K^-1 (0 + (10,0,0)) = (5, 0, 0)
    # impulse = -0.8 * (5,0,0) - 0.2 * (0,0,0) = (-4, 0, 0)
    np.testing.assert_allclose(result, [-4.0, 0.0, 0.0], atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Revolute angular with use_bias=1
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_angular_biased(test, device):
    """Revolute angular with use_bias=1 tests softness application."""
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wp.launch(
        _test_revolute_angular_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(2.0, 0.0, 0.0),  # B rotating around X
            wp.vec3(0.0, 1.0, 0.0),  # hinge Y
            inv_I, inv_I,
            wp.vec2(1.0, 0.0),  # accumulated impulse (1.0 on b1)
            10.0, 0.8, 0.2, 1,  # use_bias=1
        ],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    # h=(0,1,0), b1=(0,0,1), b2=(1,0,0) (same basis as before)
    # dw=(2,0,0), k1=k2=2, am1=am2=0.5
    # cdot1=dot((2,0,0),(0,0,1))=0 → i1 = -0.8*0.5*0 - 0.2*1.0 = -0.2
    # cdot2=dot((2,0,0),(1,0,0))=2 → i2 = -0.8*0.5*2 - 0.2*0.0 = -0.8
    # result = -0.2*(0,0,1) + -0.8*(1,0,0) = (-0.8, 0, -0.2)
    np.testing.assert_allclose(result, [-0.8, 0.0, -0.2], atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Revolute limit with is_upper=1
# ═══════════════════════════════════════════════════════════════════════


def test_revolute_limit_upper_violated(test, device):
    """Upper limit (is_upper=1) with violation flips velocity sign correctly."""
    out = wp.zeros(1, dtype=float, device=device)
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # Body B rotating at 1 rad/s along hinge, upper limit violated
    wp.launch(
        _test_revolute_limit_kernel, dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),  # B rotating around Y (hinge)
            wp.vec3(0.0, 1.0, 0.0),  # hinge Y
            inv_I, inv_I,
            0.0,       # limit_impulse
            -0.1,      # angle_error < 0 (violated)
            10.0, 0.8, 0.2,
            240.0, 1, 1,  # use_bias=1, is_upper=1
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # k = dot((0,1,0), 2*(0,1,0)) = 2, am = 0.5
    # cdot = dot(B-A, h) = dot((0,1,0)-(0,0,0), (0,1,0)) = 1
    # is_upper=1: cdot flipped → cdot = -1
    # bias = 0.8 * 10 * (-0.1) = -0.8, ms=0.8, isv=0.2
    # impulse = -0.8 * 0.5 * (-1 + (-0.8)) - 0.2*0 = -0.8*0.5*(-1.8) = 0.72
    # new = max(0 + 0.72, 0) = 0.72
    test.assertAlmostEqual(delta, 0.72, places=3)


# ═══════════════════════════════════════════════════════════════════════
# Effective mass with non-diagonal inertia
# ═══════════════════════════════════════════════════════════════════════


def test_effective_mass_non_diagonal_inertia(test, device):
    """Non-diagonal inertia tensor changes effective mass vs diagonal case."""
    # Body A: inv_inertia with off-diagonal terms
    inv_I_a = wp.mat33(2.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 1.0)
    inv_I_zero = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r_a = wp.vec3(0.0, 0.0, 1.0)  # lever arm along z
    r_b = wp.vec3(0.0, 0.0, 0.0)
    d = wp.vec3(0.0, 1.0, 0.0)  # normal along y
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_effective_mass_kernel, dim=1,
        inputs=[1.0, inv_I_a, 0.0, inv_I_zero, r_a, r_b, d],
        outputs=[out], device=device,
    )
    result = float(out.numpy()[0])

    # numpy reference:
    # rn_a = cross((0,0,1), (0,1,0)) = (-1,0,0)
    # inv_I * (-1,0,0) = (-2, -0.5, 0)
    # cross((-2,-0.5,0), (0,0,1)) = (-0.5, 2, 0)  -- no, let me recompute
    # Actually: cross(inv_I * rn_a, r_a) where rn_a = cross(r_a, d)
    # rn_a = cross((0,0,1), (0,1,0)) = (-1, 0, 0)
    # inv_I * rn_a = inv_I * (-1,0,0) = (-2, -0.5, 0)
    # cross((-2,-0.5,0), (0,0,1)) = (-0.5, 2, 0)  -- wait:
    # cross(a, b) = (a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)
    # cross((-2,-0.5,0), (0,0,1)) = (-0.5*1 - 0*0, 0*0 - (-2)*1, (-2)*0 - (-0.5)*0) = (-0.5, 2, 0)
    # dot((0,1,0), (-0.5, 2, 0)) = 2.0
    # K = 1.0 + 2.0 = 3.0, eff_mass = 1/3
    expected = 1.0 / 3.0
    test.assertAlmostEqual(result, expected, places=4)


# ═══════════════════════════════════════════════════════════════════════
# Color joints CPU test
# ═══════════════════════════════════════════════════════════════════════


def test_color_joints_cpu_basic(test, device):
    """CPU joint coloring produces valid non-conflicting colors."""
    # 4 joints forming a chain: 0-1, 1-2, 2-3, 3-0 (cycle)
    body_a = np.array([0, 1, 2, 3], dtype=np.int32)
    body_b = np.array([1, 2, 3, 0], dtype=np.int32)
    order, offsets, num_colors = color_joints_cpu(body_a, body_b, 4, 4)

    # A 4-cycle needs at least 2 colors
    test.assertGreaterEqual(num_colors, 2)
    test.assertEqual(len(order), 4)
    test.assertEqual(int(offsets[-1]), 4)

    # Verify no two joints in the same color share a body
    for c in range(num_colors):
        start = int(offsets[c])
        end = int(offsets[c + 1])
        bodies_in_color = set()
        for ji in range(start, end):
            orig_j = int(order[ji])
            a, b = int(body_a[orig_j]), int(body_b[orig_j])
            test.assertNotIn(a, bodies_in_color, f"Body {a} appears twice in color {c}")
            test.assertNotIn(b, bodies_in_color, f"Body {b} appears twice in color {c}")
            bodies_in_color.add(a)
            bodies_in_color.add(b)


def test_color_joints_cpu_ground(test, device):
    """Ground body (-1) doesn't cause conflicts between joints."""
    # Two joints both connected to ground — should be same color
    body_a = np.array([-1, -1], dtype=np.int32)
    body_b = np.array([0, 1], dtype=np.int32)
    order, offsets, num_colors = color_joints_cpu(body_a, body_b, 2, 2)

    # Both joints connect to ground + different dynamic bodies → 1 color
    test.assertEqual(num_colors, 1)


# ═══════════════════════════════════════════════════════════════════════
# Additional tests for coverage gaps
# ═══════════════════════════════════════════════════════════════════════


def test_solve_3x3_off_diagonal(test, device):
    """3x3 solve with off-diagonal entries tests cofactor computation."""
    # K = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]  (symmetric positive definite)
    # rhs = (1, 2, 3)
    # Solve via numpy for reference:
    K = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=np.float64)
    rhs = np.array([1, 2, 3], dtype=np.float64)
    expected = np.linalg.solve(K, rhs)

    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _test_solve_3x3_kernel, dim=1,
        inputs=[4.0, 2.0, 1.0, 5.0, 3.0, 6.0, 1.0, 2.0, 3.0],
        outputs=[out], device=device,
    )
    result = out.numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_normal_impulse_deep_penetration_clamp(test, device):
    """Deep penetration: velocity_bias clamped by contact_speed."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_normal_impulse_kernel, dim=1,
        inputs=[
            0.0,      # vn: zero relative velocity
            -10.0,    # separation: deeply penetrating
            0.0,      # lambda_n
            1.0,      # normal_mass
            10.0,     # bias_rate
            0.8,      # mass_scale
            0.2,      # impulse_scale
            240.0,    # inv_sub_dt
            2.0,      # contact_speed = 2.0 (the clamp)
            1,        # use_bias
        ],
        outputs=[out], device=device,
    )
    delta = float(out.numpy()[0])
    # Hand computation:
    # sep = -10.0, unclamped bias = 0.8 * 10.0 * (-10.0) = -80.0
    # clamped: max(-80.0, -2.0) = -2.0  ← contact_speed clamp is binding
    # ms = 0.8, isv = 0.2
    # impulse = -1.0 * (0.8 * 0.0 + (-2.0)) - 0.2 * 0.0 = 2.0
    test.assertAlmostEqual(delta, 2.0, places=4)


def test_effective_mass_two_lever_arms(test, device):
    """Both bodies have non-zero lever arms — tests full K formula."""
    inv_mass = 1.0
    inv_I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    r_a = wp.vec3(0.0, 0.0, 1.0)
    r_b = wp.vec3(0.0, 0.0, -1.0)
    d = wp.vec3(0.0, 1.0, 0.0)
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _test_effective_mass_kernel, dim=1,
        inputs=[inv_mass, inv_I, inv_mass, inv_I, r_a, r_b, d],
        outputs=[out], device=device,
    )
    result = float(out.numpy()[0])
    # cross(r_a, d) = cross((0,0,1),(0,1,0)) = (-1,0,0)
    # inv_I * (-1,0,0) = (-1,0,0)
    # cross((-1,0,0), (0,0,1)) = (0,1,0)
    # dot((0,1,0), (0,1,0)) = 1.0  (body A contribution)
    # cross(r_b, d) = cross((0,0,-1),(0,1,0)) = (1,0,0)
    # inv_I * (1,0,0) = (1,0,0)
    # cross((1,0,0), (0,0,-1)) = (0,1,0)
    # dot((0,1,0), (0,1,0)) = 1.0  (body B contribution)
    # K = 1 + 1 + 1 + 1 = 4, effective_mass = 0.25
    test.assertAlmostEqual(result, 0.25, places=5)


# ═══════════════════════════════════════════════════════════════════════
# Register all tests
# ═══════════════════════════════════════════════════════════════════════

devices = get_test_devices()

# Softness
add_function_test(TestBox3DConstraintFuncs, "test_softness_critical_damping", test_softness_critical_damping, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_softness_zero_hertz", test_softness_zero_hertz, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_softness_hand_computed", test_softness_hand_computed, devices=devices)

# Tangent basis
add_function_test(TestBox3DConstraintFuncs, "test_tangent_basis_y_up", test_tangent_basis_y_up, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_tangent_basis_arbitrary_normal", test_tangent_basis_arbitrary_normal, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_tangent_basis_z_up_exact", test_tangent_basis_z_up_exact, devices=devices)

# Effective mass
add_function_test(TestBox3DConstraintFuncs, "test_effective_mass_equal_masses", test_effective_mass_equal_masses, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_effective_mass_static_body", test_effective_mass_static_body, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_effective_mass_with_lever_arm", test_effective_mass_with_lever_arm, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_effective_mass_two_lever_arms", test_effective_mass_two_lever_arms, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_effective_mass_non_diagonal_inertia", test_effective_mass_non_diagonal_inertia, devices=devices)

# Normal impulse
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_penetrating", test_normal_impulse_penetrating, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_separating", test_normal_impulse_separating, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_clamp_nonnegative", test_normal_impulse_clamp_nonnegative, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_relaxation_no_bias", test_normal_impulse_relaxation_no_bias, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_speculative", test_normal_impulse_speculative, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_deep_penetration_clamp", test_normal_impulse_deep_penetration_clamp, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_normal_impulse_speculative_relaxation", test_normal_impulse_speculative_relaxation, devices=devices)

# Friction
add_function_test(TestBox3DConstraintFuncs, "test_friction_within_cone", test_friction_within_cone, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_friction_clamped_to_cone", test_friction_clamped_to_cone, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_friction_accumulated_clamp", test_friction_accumulated_clamp, devices=devices)

# Relative velocity
add_function_test(TestBox3DConstraintFuncs, "test_relative_velocity_pure_linear", test_relative_velocity_pure_linear, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_relative_velocity_with_rotation", test_relative_velocity_with_rotation, devices=devices)

# Apply impulse
add_function_test(TestBox3DConstraintFuncs, "test_apply_impulse_linear", test_apply_impulse_linear, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_apply_impulse_with_torque", test_apply_impulse_with_torque, devices=devices)

# Restitution
add_function_test(TestBox3DConstraintFuncs, "test_restitution_bouncing", test_restitution_bouncing, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_restitution_slow_approach_no_bounce", test_restitution_slow_approach_no_bounce, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_restitution_zero_coeff", test_restitution_zero_coeff, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_restitution_with_accumulated_impulse", test_restitution_with_accumulated_impulse, devices=devices)

# 3x3 solve
add_function_test(TestBox3DConstraintFuncs, "test_solve_3x3_identity", test_solve_3x3_identity, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_solve_3x3_diagonal", test_solve_3x3_diagonal, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_solve_3x3_off_diagonal", test_solve_3x3_off_diagonal, devices=devices)

# Skew outer diag
add_function_test(TestBox3DConstraintFuncs, "test_skew_outer_diag_identity_inertia", test_skew_outer_diag_identity_inertia, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_skew_outer_diag_non_diagonal_inertia", test_skew_outer_diag_non_diagonal_inertia, devices=devices)

# Revolute point-to-point
add_function_test(TestBox3DConstraintFuncs, "test_revolute_p2p_coincident_relaxation", test_revolute_p2p_coincident_relaxation, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_p2p_velocity_correction", test_revolute_p2p_velocity_correction, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_p2p_with_lever_arms_biased", test_revolute_p2p_with_lever_arms_biased, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_p2p_asymmetric_levers", test_revolute_p2p_asymmetric_levers, devices=devices)

# Revolute angular
add_function_test(TestBox3DConstraintFuncs, "test_revolute_angular_aligned", test_revolute_angular_aligned, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_angular_correction", test_revolute_angular_correction, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_angular_biased", test_revolute_angular_biased, devices=devices)

# Revolute motor
add_function_test(TestBox3DConstraintFuncs, "test_revolute_motor_speed_tracking", test_revolute_motor_speed_tracking, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_motor_torque_clamp", test_revolute_motor_torque_clamp, devices=devices)

# Revolute limits
add_function_test(TestBox3DConstraintFuncs, "test_revolute_limit_within_range", test_revolute_limit_within_range, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_limit_violated", test_revolute_limit_violated, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_revolute_limit_upper_violated", test_revolute_limit_upper_violated, devices=devices)

# Fixed joint
add_function_test(TestBox3DConstraintFuncs, "test_fixed_angular_zero_velocity", test_fixed_angular_zero_velocity, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_fixed_angular_correction", test_fixed_angular_correction, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_fixed_angular_off_diagonal_inertia", test_fixed_angular_off_diagonal_inertia, devices=devices)

# Color joints CPU (no device needed, but run in both for consistency)
add_function_test(TestBox3DConstraintFuncs, "test_color_joints_cpu_basic", test_color_joints_cpu_basic, devices=devices)
add_function_test(TestBox3DConstraintFuncs, "test_color_joints_cpu_ground", test_color_joints_cpu_ground, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
