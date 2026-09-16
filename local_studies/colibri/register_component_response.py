"""Register-only two-body native response and row-velocity helpers.

Uses the existing native arithmetic and precision, including its double row-
velocity evaluation. No new contact law or precision policy is introduced.
"""

import warp as wp

from newton._src.solvers.phoenx.articulations.maximal_projector import _make_spatial_shift_transform


@wp.func
def serial_pair_local(
    root_mobility: wp.spatial_matrixf,
    child_inertia: wp.spatial_matrixf,
    motion: wp.spatial_vectorf,
    shift: wp.vec3f,
    inverse_d: wp.float32,
    root_impulse: wp.spatial_vectorf,
    child_impulse: wp.spatial_vectorf,
):
    u = child_inertia @ motion
    projected_rhs = child_impulse - inverse_d * wp.dot(motion, child_impulse) * u
    transform = _make_spatial_shift_transform(shift)
    parent_rhs = wp.transpose(transform) @ projected_rhs
    root_rhs = root_impulse
    root_rhs += parent_rhs
    root_delta = root_mobility @ root_rhs
    base = transform @ root_delta
    joint_delta = inverse_d * wp.dot(motion, child_impulse - child_inertia @ base)
    child_delta = base + joint_delta * motion
    return root_delta, child_delta, joint_delta


@wp.func
def vec_double(v: wp.vec3f):
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def row_velocity_local(
    root_v: wp.vec3f,
    root_w: wp.vec3f,
    child_v: wp.vec3f,
    child_w: wp.vec3f,
    external_v0: wp.vec3f,
    external_w0: wp.vec3f,
    external_v1: wp.vec3f,
    external_w1: wp.vec3f,
    root_body: wp.int32,
    child_body: wp.int32,
    motion: wp.spatial_vectorf,
    shift: wp.vec3f,
    body0: wp.int32,
    body1: wp.int32,
    r0: wp.vec3f,
    r1: wp.vec3f,
    direction: wp.vec3f,
):
    # Only external-contact rows reach this function. The same-component
    # cancellation-aware original routine remains the internal fallback.
    force0 = -vec_double(direction)
    force1 = vec_double(direction)
    torque0 = wp.cross(vec_double(r0), force0)
    torque1 = wp.cross(vec_double(r1), force1)
    axis = wp.vec3d(wp.float64(motion[3]), wp.float64(motion[4]), wp.float64(motion[5]))
    linear = wp.vec3d(wp.float64(motion[0]), wp.float64(motion[1]), wp.float64(motion[2]))
    velocity = wp.float64(0.0)
    source0 = body0
    source1 = body1
    if body0 == child_body:
        speed = wp.dot(axis, vec_double(child_w) - vec_double(root_w))
        velocity += (wp.dot(linear, force0) + wp.dot(axis, torque0)) * speed
        torque0 -= wp.cross(vec_double(shift), force0)
        source0 = root_body
    if body1 == child_body:
        speed = wp.dot(axis, vec_double(child_w) - vec_double(root_w))
        velocity += (wp.dot(linear, force1) + wp.dot(axis, torque1)) * speed
        torque1 -= wp.cross(vec_double(shift), force1)
        source1 = root_body
    v0 = external_v0
    w0 = external_w0
    v1 = external_v1
    w1 = external_w1
    if source0 == root_body:
        v0 = root_v
        w0 = root_w
    if source1 == root_body:
        v1 = root_v
        w1 = root_w
    velocity += wp.dot(force0, vec_double(v0)) + wp.dot(torque0, vec_double(w0))
    velocity += wp.dot(force1, vec_double(v1)) + wp.dot(torque1, vec_double(w1))
    return wp.float32(velocity)
