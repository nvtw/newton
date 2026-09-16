# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Conservative FP32 planar sticking proposals under fixed point normal loads.

This module does not scatter impulses, change history, or solve normal/joint
coupling. A caller must validate those equations before applying a proposal.
UNKNOWN (status 0) includes feasible cases missed by the distribution heuristic.
"""

import warp as wp


@wp.func
def _solve_spd(a: wp.mat33f, b: wp.vec3f):
    # No diagonal regularization or rank truncation: unresolved pivots fail.
    result = wp.vec3f(0.0)
    valid = False
    if a[0, 0] > 0.0:
        l00 = wp.sqrt(a[0, 0])
        l10 = a[1, 0] / l00
        l20 = a[2, 0] / l00
        d11 = a[1, 1] - l10 * l10
        if d11 > 0.0:
            l11 = wp.sqrt(d11)
            l21 = (a[2, 1] - l20 * l10) / l11
            d22 = a[2, 2] - l20 * l20 - l21 * l21
            if d22 > 0.0:
                l22 = wp.sqrt(d22)
                y0 = b[0] / l00
                y1 = (b[1] - l10 * y0) / l11
                y2 = (b[2] - l20 * y0 - l21 * y1) / l22
                x2 = y2 / l22
                x1 = (y1 - l21 * x2) / l11
                x0 = (y0 - l10 * x1 - l20 * x2) / l00
                result = wp.vec3f(x0, x1, x2)
                valid = wp.isfinite(x0) and wp.isfinite(x1) and wp.isfinite(x2)
    return result, valid


@wp.kernel(enable_backward=False)
def propose_friction_patch_stick(
    offsets: wp.array[wp.int32],
    members: wp.array[wp.int32],
    points: wp.array[wp.vec3f],
    normals: wp.array[wp.vec3f],
    capacities: wp.array[wp.float32],
    origins: wp.array[wp.vec3f],
    tangent0: wp.array[wp.vec3f],
    tangent1: wp.array[wp.vec3f],
    patch_normals: wp.array[wp.vec3f],
    mobility: wp.array[wp.mat33f],
    velocity: wp.array[wp.vec3f],
    plane_tolerance: wp.float32,
    basis_tolerance: wp.float32,
    cone_tolerance: wp.float32,
    force_tolerance: wp.float32,
    torque_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    spin_tolerance: wp.float32,
    impulses: wp.array[wp.vec3f],
    status: wp.array[wp.int32],
    requested_wrench: wp.array[wp.vec3f],
    residuals: wp.array[wp.vec3f],
):
    """Propose total point tangent impulses without modifying physical state.

    Launch one thread per patch. Inputs use CSR offsets/members; each member
    names an original point. Outputs impulses are indexed by CSR entry, not
    original point. The caller guarantees valid, disjoint CSR ranges and
    unique members within each patch. There is no member-count limit.

    Capacities are original mu_static * normal_impulse [N s], held fixed.
    Origins and points are common spatial force application points [m].
    The right-handed orthonormal basis defines two planar translations and
    normal-axis spin. Mobility maps (N s, N s, N m s) into (m/s, m/s, rad/s);
    velocity is the free relative velocity in that basis. Supply the free
    state BEFORE these proposed total tangent impulses, not an already
    tangentially corrected state. Existing impulses must be accounted for
    by the caller when converting the proposal into an increment.

    Tolerances bound FP32 roundoff in the explicitly named SI quantities;
    basis_tolerance is dimensionless and also bounds relative asymmetry of H.
    No physical mode is regularized or removed. Nonplanarity, invalid inputs,
    unresolved rank, or failed certificates return UNKNOWN (0), zero impulses.
    Status 1 certifies only the fixed-load planar proposal, not the coupled
    contact problem. Residuals hold force, torque and cone errors.
    """
    patch = wp.tid()
    start = offsets[patch]
    end = offsets[patch + 1]
    status[patch] = 0
    requested_wrench[patch] = wp.vec3f(0.0)
    residuals[patch] = wp.vec3f(0.0)
    for entry in range(start, end):
        impulses[entry] = wp.vec3f(0.0)
    if start == end:
        return
    if not wp.isfinite(
        plane_tolerance
        + basis_tolerance
        + cone_tolerance
        + force_tolerance
        + torque_tolerance
        + velocity_tolerance
        + spin_tolerance
    ):
        return
    if (
        plane_tolerance < 0.0
        or basis_tolerance < 0.0
        or cone_tolerance < 0.0
        or force_tolerance < 0.0
        or torque_tolerance < 0.0
        or velocity_tolerance < 0.0
        or spin_tolerance < 0.0
    ):
        return

    origin = origins[patch]
    t0 = tangent0[patch]
    t1 = tangent1[patch]
    normal = patch_normals[patch]
    if (
        wp.abs(wp.dot(t0, t0) - 1.0) > basis_tolerance
        or wp.abs(wp.dot(t1, t1) - 1.0) > basis_tolerance
        or wp.abs(wp.dot(normal, normal) - 1.0) > basis_tolerance
        or wp.length(wp.cross(t0, t1) - normal) > basis_tolerance
        or wp.abs(wp.dot(t0, t1)) > basis_tolerance
    ):
        return

    h = mobility[patch]
    v = velocity[patch]
    scale = float(0.0)
    asymmetry = float(0.0)
    for i in range(3):
        if not wp.isfinite(v[i] + origin[i] + t0[i] + t1[i] + normal[i]):
            return
        for j in range(3):
            if not wp.isfinite(h[i, j]):
                return
            scale = wp.max(scale, wp.abs(h[i, j]))
            asymmetry = wp.max(asymmetry, wp.abs(h[i, j] - h[j, i]))
    if asymmetry > basis_tolerance * scale:
        return
    wrench, valid = _solve_spd(h, -v)
    if not valid:
        return
    stopped = v + h * wrench
    if (
        wp.abs(stopped[0]) > velocity_tolerance
        or wp.abs(stopped[1]) > velocity_tolerance
        or wp.abs(stopped[2]) > spin_tolerance
    ):
        return

    csum = float(0.0)
    cx = float(0.0)
    cy = float(0.0)
    cr2 = float(0.0)
    first_loaded = wp.vec2f(0.0)
    have_loaded = bool(False)
    distinct_loaded = bool(False)
    for entry in range(start, end):
        point = members[entry]
        r = points[point] - origin
        c = capacities[point]
        if not wp.isfinite(c) or c < 0.0:
            return
        for i in range(3):
            if not wp.isfinite(r[i]) or not wp.isfinite(normals[point][i]):
                return
        if wp.abs(wp.dot(r, normal)) > plane_tolerance:
            return
        if wp.length(normals[point] - normal) > basis_tolerance:
            return
        x = wp.dot(r, t0)
        y = wp.dot(r, t1)
        if c > 0.0:
            if not have_loaded:
                first_loaded = wp.vec2f(x, y)
                have_loaded = True
            elif x != first_loaded[0] or y != first_loaded[1]:
                distinct_loaded = True
        csum += c
        cx += c * x
        cy += c * y
        cr2 += c * (x * x + y * y)
    # Coincident support has exactly no torsional degree of freedom; do not
    # let cancellation in Cholesky manufacture a small positive pivot.
    if not distinct_loaded:
        return
    gram = wp.mat33f(csum, 0.0, -cy, 0.0, csum, cx, -cy, cx, cr2)
    dual, valid = _solve_spd(gram, wrench)
    if not valid:
        return

    force = wp.vec3f(0.0)
    torque = wp.vec3f(0.0)
    cone_error = float(0.0)
    for entry in range(start, end):
        point = members[entry]
        r = points[point] - origin
        x = wp.dot(r, t0)
        y = wp.dot(r, t1)
        f = capacities[point] * ((dual[0] - y * dual[2]) * t0 + (dual[1] + x * dual[2]) * t1)
        if not wp.isfinite(wp.length(f)):
            return
        cone_error = wp.max(cone_error, wp.length(f) - capacities[point])
        # Check the original tangent plane as well as the disk.
        cone_error = wp.max(cone_error, wp.abs(wp.dot(f, normals[point])))
        force += f
        torque += wp.cross(r, f)
    force_error = wp.length(force - wrench[0] * t0 - wrench[1] * t1)
    torque_error = wp.length(torque - wrench[2] * normal)
    residuals[patch] = wp.vec3f(force_error, torque_error, cone_error)
    if (
        not wp.isfinite(force_error + torque_error + cone_error)
        or force_error > force_tolerance
        or torque_error > torque_tolerance
        or cone_error > cone_tolerance
    ):
        return
    # Certify the actual accumulated wrench, not merely the requested solve.
    actual = wp.vec3f(wp.dot(force, t0), wp.dot(force, t1), wp.dot(torque, normal))
    stopped = v + h * actual
    if (
        wp.abs(stopped[0]) > velocity_tolerance
        or wp.abs(stopped[1]) > velocity_tolerance
        or wp.abs(stopped[2]) > spin_tolerance
    ):
        return
    if wp.dot(actual, v) + 0.5 * wp.dot(actual, h * actual) > 0.0:
        return

    for entry in range(start, end):
        point = members[entry]
        r = points[point] - origin
        x = wp.dot(r, t0)
        y = wp.dot(r, t1)
        impulses[entry] = capacities[point] * ((dual[0] - y * dual[2]) * t0 + (dual[1] + x * dual[2]) * t1)
    requested_wrench[patch] = wrench
    status[patch] = 1
