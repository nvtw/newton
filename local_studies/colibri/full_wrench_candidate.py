"""Local FP32 full-wrench sufficient proposal. No scatter or history writes."""

import warp as wp


@wp.func
def solve6(a: wp.spatial_matrixf, rhs: wp.spatial_vectorf):
    scale = wp.spatial_vectorf(0.0)
    valid = wp.int32(1)
    for i in range(6):
        diagonal = a[i, i]
        if diagonal <= 0.0 or not wp.isfinite(diagonal):
            valid = 0
            diagonal = 1.0
        scale[i] = 1.0 / wp.sqrt(diagonal)
    l = wp.spatial_matrixf(0.0)
    for i in range(6):
        for k in range(i + 1):
            value = scale[i] * a[i, k] * scale[k]
            for j in range(k):
                value -= l[i, j] * l[k, j]
            if i == k:
                if value <= 0.0 or not wp.isfinite(value):
                    valid = 0
                    value = 1.0
                l[i, k] = wp.sqrt(value)
            else:
                l[i, k] = value / l[k, k]
    y = wp.spatial_vectorf(0.0)
    for i in range(6):
        value = scale[i] * rhs[i]
        for j in range(i):
            value -= l[i, j] * y[j]
        y[i] = value / l[i, i]
    z = wp.spatial_vectorf(0.0)
    for rev in range(6):
        i = 5 - rev
        value = y[i]
        for j in range(i + 1, 6):
            value -= l[j, i] * z[j]
        z[i] = value / l[i, i]
    result = wp.spatial_vectorf(0.0)
    for i in range(6):
        result[i] = scale[i] * z[i]
    return result, valid


@wp.func
def dot6(a: wp.spatial_vectorf, b: wp.spatial_vectorf):
    result = wp.float32(0.0)
    error = wp.float32(0.0)
    for i in range(6):
        add = a[i] * b[i] - error
        new = result + add
        error = (new - result) - add
        result = new
    return result


@wp.kernel(enable_backward=False)
def propose(
    mobility: wp.array[wp.spatial_matrixf],
    velocity: wp.array[wp.spatial_vectorf],
    maps: wp.array2d[wp.spatial_vectorf],
    old: wp.array2d[float],
    coefficients: wp.array[float],
    eligible: wp.array[int],
    interval: wp.array[int],
    candidate: wp.array2d[float],
    status: wp.array[int],
    requested_out: wp.array[wp.spatial_vectorf],
):
    first = interval[0]
    count = interval[1]
    status[0] = -1
    maximum = wp.float32(0.0)
    for k in range(first, first + count):
        for r in range(3):
            candidate[r, k] = old[r, k]
        if eligible[k] != 0:
            maximum = wp.max(maximum, old[0, k])
    if maximum > 0.0:
        gram = wp.spatial_matrixf(0.0)
        ge = wp.spatial_matrixf(0.0)
        force = wp.spatial_vectorf(0.0)
        fe = wp.spatial_vectorf(0.0)
        for k in range(first, first + count):
            if eligible[k] != 0:
                weight = wp.max(old[0, k], wp.float32(0.0)) / maximum
                for r in range(3):
                    column = maps[k, r]
                    add = column * old[r, k] - fe
                    new = force + add
                    fe = (new - force) - add
                    force = new
                    term = wp.spatial_matrixf(0.0)
                    for i in range(6):
                        for j in range(6):
                            term[i, j] = weight * column[i] * column[j]
                    ga = term - ge
                    gn = gram + ga
                    ge = (gn - gram) - ga
                    gram = gn
        correction, valid0 = solve6(mobility[0], velocity[0])
        requested = force - correction
        dual, valid1 = solve6(gram, requested)
        valid = valid0 * valid1
        for k in range(first, first + count):
            if eligible[k] != 0:
                weight = wp.max(old[0, k], wp.float32(0.0)) / maximum
                n = weight * dot6(maps[k, 0], dual)
                t0 = weight * dot6(maps[k, 1], dual)
                t1 = weight * dot6(maps[k, 2], dual)
                radius = coefficients[k] * n
                if (
                    not wp.isfinite(coefficients[k])
                    or coefficients[k] < 0.0
                    or not wp.isfinite(n)
                    or not wp.isfinite(t0)
                    or not wp.isfinite(t1)
                    or not wp.isfinite(radius)
                    or not wp.isfinite(t0 * t0 + t1 * t1)
                    or not wp.isfinite(radius * radius)
                    or n < 0.0
                    or t0 * t0 + t1 * t1 > radius * radius
                ):
                    valid = 0
                candidate[0, k] = n
                candidate[1, k] = t0
                candidate[2, k] = t1
        requested_out[0] = requested
        status[0] = 1
        if valid == 0:
            status[0] = -3
        if valid0 == 0 or valid1 == 0:
            status[0] = -2
