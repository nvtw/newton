"""Local native integration clone with an exact-zero angular-velocity guard."""

import warp as wp
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    MOTION_DYNAMIC,
    BodyContainer,
    _rotation_quaternion,
    mat33_from_sym6,
    rotate_inertia,
    sym6_from_mat33,
)


@wp.kernel(enable_backward=False)
def exact_zero_integrate(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Advance pose for dynamic bodies only. Kinematic bodies advance via
    lerp/slerp in :func:`_kinematic_interpolate_substep_kernel`."""
    i = wp.tid()
    mt = bodies.motion_type[i]
    if mt != MOTION_DYNAMIC:
        return
    # Sleeping bodies must not drift. They may carry a small residual
    # velocity (anywhere below the per-island sleep threshold) at the
    # moment ``island_root`` is stamped; integrating that for many
    # substeps would slide the whole sleeping island visibly.
    if bodies.island_root[i] >= wp.int32(0):
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt

    # Integrate torque-free rotation with an implicit midpoint update. Contact
    # and external impulses have already changed omega, so preserve their
    # resulting world angular momentum while the anisotropic inertia rotates.
    omega = bodies.angular_velocity[i]
    if omega[0] == wp.float32(0.0) and omega[1] == wp.float32(0.0) and omega[2] == wp.float32(0.0):
        return
    q0 = bodies.orientation[i]
    inv_inertia0 = mat33_from_sym6(bodies.inverse_inertia_world[i])
    angular_momentum = wp.inverse(inv_inertia0) * bodies.angular_velocity[i]
    omega_mid = bodies.angular_velocity[i]
    for _ in range(3):
        q_half = wp.normalize(_rotation_quaternion(omega_mid, dt * wp.float32(0.5)) * q0)
        r_half = wp.quat_to_matrix(q_half)
        inv_inertia_half = rotate_inertia(r_half, bodies.inverse_inertia[i])
        omega_mid = inv_inertia_half * angular_momentum

    q1 = wp.normalize(_rotation_quaternion(omega_mid, dt) * q0)
    r1 = wp.quat_to_matrix(q1)
    inv_inertia1 = rotate_inertia(r1, bodies.inverse_inertia[i])
    bodies.orientation[i] = q1
    bodies.inverse_inertia_world[i] = sym6_from_mat33(inv_inertia1)
    bodies.angular_velocity[i] = inv_inertia1 * angular_momentum
