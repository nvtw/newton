"""FP32 high/low joint-coordinate point velocity and direction projection.

Products retain their FMA residual, and sums retain their rounding residual.
No double arithmetic, deadzone, rank cutoff, or change to physical response.
"""

import warp as wp


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return fmaf(a, b, c);
#endif
""")
def fused32(a: wp.float32, b: wp.float32, c: wp.float32) -> wp.float32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    volatile float rounded = a * b;
    return rounded;
#endif
""")
def product32(a: wp.float32, b: wp.float32) -> wp.float32: ...


@wp.func
def pair_sum(a: wp.float32, b: wp.float32):
    high = a + b
    carried = high - a
    low = (a - (high - carried)) + (b - carried)
    return wp.vec2f(high, low)


@wp.func
def pair_product(a: wp.float32, b: wp.float32):
    high = product32(a, b)
    low = fused32(a, b, -high)
    return wp.vec2f(high, low)


@wp.func
def pair_add(a: wp.vec2f, b: wp.vec2f):
    high = pair_sum(a[0], b[0])
    return pair_sum(high[0], high[1] + (a[1] + b[1]))


@wp.func
def pair_multiply(a: wp.vec2f, b: wp.vec2f):
    high = pair_product(a[0], b[0])
    low = fused32(a[0], b[1], a[1] * b[0])
    low = fused32(a[1], b[1], low)
    return pair_add(high, wp.vec2f(low, 0.0))


@wp.func
def compensated_relative_point_velocity(
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
    speed = wp.vec2f(0.0)
    for axis in range(3):
        difference = pair_sum(child_w[axis], -root_w[axis])
        speed = pair_add(speed, pair_multiply(wp.vec2f(motion[3 + axis], 0.0), difference))
    translation0 = external_v0
    translation1 = external_v1
    omega0 = external_w0
    omega1 = external_w1
    if body0 == root_body or body0 == child_body:
        translation0 = root_v
        omega0 = root_w
    if body1 == root_body or body1 == child_body:
        translation1 = root_v
        omega1 = root_w
    result = wp.spatial_vectorf(0.0)
    for axis in range(3):
        j = (axis + 1) % 3
        k = (axis + 2) % 3
        lever0j = wp.vec2f(r0[j], 0.0)
        lever0k = wp.vec2f(r0[k], 0.0)
        lever1j = wp.vec2f(r1[j], 0.0)
        lever1k = wp.vec2f(r1[k], 0.0)
        if body0 == child_body:
            lever0j = pair_sum(r0[j], -shift[j])
            lever0k = pair_sum(r0[k], -shift[k])
        if body1 == child_body:
            lever1j = pair_sum(r1[j], -shift[j])
            lever1k = pair_sum(r1[k], -shift[k])
        rotation0 = pair_add(
            pair_multiply(wp.vec2f(omega0[j], 0.0), lever0k), -pair_multiply(wp.vec2f(omega0[k], 0.0), lever0j)
        )
        rotation1 = pair_add(
            pair_multiply(wp.vec2f(omega1[j], 0.0), lever1k), -pair_multiply(wp.vec2f(omega1[k], 0.0), lever1j)
        )
        if body0 == child_body:
            joint_motion = pair_add(
                wp.vec2f(motion[axis], 0.0),
                pair_add(pair_product(motion[3 + j], r0[k]), -pair_product(motion[3 + k], r0[j])),
            )
            rotation0 = pair_add(rotation0, pair_multiply(speed, joint_motion))
        if body1 == child_body:
            joint_motion = pair_add(
                wp.vec2f(motion[axis], 0.0),
                pair_add(pair_product(motion[3 + j], r1[k]), -pair_product(motion[3 + k], r1[j])),
            )
            rotation1 = pair_add(rotation1, pair_multiply(speed, joint_motion))
        value = pair_add(pair_sum(translation1[axis], -translation0[axis]), pair_add(rotation1, -rotation0))
        result[axis] = value[0]
        result[axis + 3] = value[1]
    return result


@wp.func
def project_compensated_point_velocity(velocity: wp.spatial_vectorf, direction: wp.vec3f):
    value = wp.vec2f(0.0)
    for axis in range(3):
        component = wp.vec2f(velocity[axis], velocity[axis + 3])
        value = pair_add(value, pair_multiply(component, wp.vec2f(direction[axis], 0.0)))
    return value[0] + value[1]
