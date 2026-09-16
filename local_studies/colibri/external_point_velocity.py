"""FP32 joint-coordinate point velocity for an external two-body contact.

Root/child motion is assembled before projecting contact directions. This helper
is never used for an internal pair, whose common-ancestor cancellation requires
the original native path. No change to impulse response or contact laws.
"""

import warp as wp


@wp.func
def external_relative_point_velocity(
    root_v: wp.vec3f,
    root_w: wp.vec3f,
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
):
    axis = wp.spatial_bottom(motion)
    linear = wp.spatial_top(motion)
    joint_speed = wp.dot(axis, child_w - root_w)
    translation0 = external_v0
    translation1 = external_v1
    rotation0 = wp.cross(external_w0, r0)
    rotation1 = wp.cross(external_w1, r1)
    if body0 == root_body:
        translation0 = root_v
        rotation0 = wp.cross(root_w, r0)
    if body0 == child_body:
        translation0 = root_v
        rotation0 = wp.cross(root_w, r0 - shift) + joint_speed * (linear + wp.cross(axis, r0))
    if body1 == root_body:
        translation1 = root_v
        rotation1 = wp.cross(root_w, r1)
    if body1 == child_body:
        translation1 = root_v
        rotation1 = wp.cross(root_w, r1 - shift) + joint_speed * (linear + wp.cross(axis, r1))
    # Preserve small contact motion under a common translational boost.
    return (translation1 - translation0) + (rotation1 - rotation0)
