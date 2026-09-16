"""GPU storage/preparation for an explicitly hybrid two-body patch experiment."""

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_r0,
    cc_get_start_gap,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_normal,
    cc_set_normal_lambda,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)


@wp.struct
class PatchState:
    count: wp.array[int]
    broken: wp.array[int]
    claims: wp.array[int]
    refresh: wp.array[int]
    error: wp.array[int]
    local0: wp.array2d[wp.vec3f]
    local1: wp.array2d[wp.vec3f]
    r0: wp.array2d[wp.vec3f]
    r1: wp.array2d[wp.vec3f]
    normal0: wp.array[wp.vec3f]
    normal1: wp.array[wp.vec3f]
    normal: wp.array[wp.vec3f]
    tangent: wp.array[wp.vec3f]
    initial: wp.array2d[wp.vec2f]
    linear: wp.array[wp.vec3f]
    angular: wp.array[wp.vec3f]


@wp.kernel
def mark_refresh(state: PatchState):
    state.refresh[0] = 1


@wp.kernel
def begin_refresh(state: PatchState):
    b = wp.tid()
    if state.refresh[0] != 0:
        state.claims[b] = 0
        state.linear[b] = wp.vec3f(0.0)
        state.angular[b] = wp.vec3f(0.0)


@wp.kernel
def finish_refresh(state: PatchState):
    if state.refresh[0] != 0:
        for key in range(state.count.shape[0]):
            if state.claims[key] == 0:
                state.count[key] = 0
                state.broken[key] = 0
    state.refresh[0] = 0


@wp.kernel
def accumulate_motion(state: PatchState, bodies: BodyContainer, dt: float):
    b = wp.tid()
    state.linear[b] += bodies.velocity[b] * dt
    state.angular[b] += bodies.angular_velocity[b] * dt


@wp.kernel
def refresh_patches(
    state: PatchState,
    bodies: BodyContainer,
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    number: wp.array[int],
    anchors: ContactContainer,
    friction_offset: float,
    correlation: float,
):
    col = wp.tid()
    if col < number[0] and state.refresh[0] != 0:
        b0 = contact_get_body1(columns, col)
        b1 = contact_get_body2(columns, col)
        key = wp.max(b0, b1)
        if wp.min(b0, b1) != 0 or key > 2:
            state.error[0] = 1
            return
        if wp.atomic_add(state.claims, key, 1) != 0:
            state.error[0] = 2
            return
        first = contact_get_contact_first(columns, col)
        count = contact_get_contact_count(columns, col)
        if count <= 0:
            state.count[key] = 0
            return
        p0 = bodies.position[b0]
        p1 = bodies.position[b1]
        q0 = bodies.orientation[b0]
        q1 = bodies.orientation[b1]
        normal = cc_get_normal(contacts, first)
        n = state.count[key]
        keep = n > 0 and n <= count and state.broken[key] == 0
        local_normal0 = wp.quat_rotate_inv(q0, normal)
        if wp.dot(local_normal0, state.normal0[key]) <= 0.999:
            keep = False
        if wp.dot(wp.quat_rotate(q0, state.normal0[key]), wp.quat_rotate(q1, state.normal1[key])) <= 0.999:
            keep = False
        for a in range(n):
            d = (p0 + wp.quat_rotate(q0, state.local0[key, a])) - (p1 + wp.quat_rotate(q1, state.local1[key, a]))
            if wp.abs(wp.dot(d, wp.quat_rotate(q0, state.normal0[key]))) >= correlation:
                keep = False
        low = wp.vec3f(1.0e20)
        high = wp.vec3f(-1.0e20)
        for offset in range(count):
            k = first + offset
            point = p0 + cc_get_r0(contacts, k)
            low = wp.min(low, point)
            high = wp.max(high, point)
            if wp.dot(normal, cc_get_normal(contacts, k)) <= 0.999:
                state.error[0] = 3
        if keep and n == 2:
            span = state.local0[key, 0] - state.local0[key, 1]
            diagonal = high - low
            if 4.0 * wp.dot(span, span) < wp.dot(diagonal, diagonal):
                keep = False
        if not keep:
            n = 0
        old_count = n
        a0 = wp.vec3f(0.0)
        a1 = wp.vec3f(0.0)
        distance = float(0.0)
        if n > 0:
            a0 = p0 + wp.quat_rotate(q0, state.local0[key, 0])
        if n == 2:
            a1 = p0 + wp.quat_rotate(q0, state.local0[key, 1])
        if n < 2:
            for offset in range(count):
                k = first + offset
                if cc_get_start_gap(contacts, k) < friction_offset:
                    point = p0 + cc_get_r0(contacts, k)
                    if n == 0:
                        a0 = point
                        n = 1
                    elif n == 1:
                        distance = wp.length_sq(point - a0)
                        if distance > 1.0e-8:
                            a1 = point
                            n = 2
                    else:
                        d0 = wp.length_sq(point - a0)
                        d1 = wp.length_sq(point - a1)
                        if d0 > d1:
                            if d0 > distance:
                                a1 = point
                                distance = d0
                        elif d1 > distance:
                            a0 = point
                            distance = d1
            if n > 0 and old_count == 0:
                state.local0[key, 0] = wp.quat_rotate_inv(q0, a0 - p0)
                state.local1[key, 0] = wp.quat_rotate_inv(q1, a0 - p1)
            if n == 2:
                state.local0[key, 1] = wp.quat_rotate_inv(q0, a1 - p0)
                state.local1[key, 1] = wp.quat_rotate_inv(q1, a1 - p1)
        state.count[key] = n
        state.broken[key] = 0
        state.normal0[key] = local_normal0
        state.normal1[key] = wp.quat_rotate_inv(q1, normal)
        state.normal[key] = normal
        vrel = bodies.velocity[b1] - bodies.velocity[b0]
        tangent = vrel - normal * wp.dot(vrel, normal)
        if wp.length_sq(tangent) <= 0.0001:
            tangent = wp.vec3f(0.0, -normal[2], normal[1])
            if wp.abs(normal[0]) >= 0.70710678:
                tangent = wp.vec3f(-normal[1], normal[0], 0.0)
        tangent = wp.normalize(tangent)
        state.tangent[key] = tangent
        t1 = wp.cross(normal, tangent)
        for a in range(n):
            r0 = wp.quat_rotate(q0, state.local0[key, a])
            r1 = wp.quat_rotate(q1, state.local1[key, a])
            state.r0[key, a] = r0
            state.r1[key, a] = r1
            error = (p1 + r1) - (p0 + r0)
            state.initial[key, a] = wp.vec2f(wp.dot(error, tangent), wp.dot(error, t1))
            k = 2 * key + a
            cc_set_tangent1_lambda(anchors, k, 0.0)
            cc_set_tangent2_lambda(anchors, k, 0.0)


@wp.kernel
def prepare_anchors(
    state: PatchState,
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    number: wp.array[int],
    anchor_columns: ContactColumnContainer,
    anchors: ContactContainer,
    inverse_dt: float,
):
    col = wp.tid()
    if col < number[0]:
        b0 = contact_get_body1(columns, col)
        b1 = contact_get_body2(columns, col)
        key = wp.max(b0, b1)
        for row in range(32):
            anchor_columns.data[row, col] = columns.data[row, col]
        anchor_columns.articulation_owner[col] = columns.articulation_owner[col]
        count = state.count[key]
        contact_set_contact_first(anchor_columns, col, 2 * key)
        contact_set_contact_count(anchor_columns, col, count)
        first = contact_get_contact_first(columns, col)
        normal_sum = float(0.0)
        for offset in range(contact_get_contact_count(columns, col)):
            normal_sum += cc_get_normal_lambda(contacts, first + offset)
        normal = state.normal[key]
        t0 = state.tangent[key]
        t1 = wp.cross(normal, t0)
        for a in range(count):
            k = 2 * key + a
            r0 = state.r0[key, a]
            r1 = state.r1[key, a]
            delta = (
                state.linear[b1] - state.linear[b0] + wp.cross(state.angular[b1], r1) - wp.cross(state.angular[b0], r0)
            )
            error = state.initial[key, a] + wp.vec2f(wp.dot(delta, t0), wp.dot(delta, t1))
            cc_set_normal(anchors, k, normal)
            cc_set_tangent1(anchors, k, t0)
            cc_set_r0(anchors, k, r0)
            cc_set_r1(anchors, k, r1)
            cc_set_bias_t1(anchors, k, error[0] * inverse_dt)
            cc_set_bias_t2(anchors, k, error[1] * inverse_dt)
            cc_set_normal_lambda(anchors, k, normal_sum / wp.float32(count))


@wp.kernel
def writeback_broken(state: PatchState, anchors: ContactContainer):
    key = wp.tid()
    broken = int(0)
    for a in range(state.count[key]):
        if anchors.lambdas[12, 2 * key + a] != 0.0:
            broken = 1
    state.broken[key] = broken


@wp.kernel
def record_before(
    bodies: BodyContainer, history: wp.array3d[float], counter: wp.array[int], ledger: wp.array[wp.spatial_vector]
):
    b = wp.tid()
    slot = counter[0] % 128
    for d in range(3):
        history[slot, b, d] = bodies.velocity[b][d]
        history[slot, b, 3 + d] = bodies.angular_velocity[b][d]
        history[slot, b, 6 + d] = bodies.position[b][d]
    for d in range(4):
        history[slot, b, 9 + d] = bodies.orientation[b][d]
    ledger[b] = wp.spatial_vectorf(0.0)


@wp.kernel
def record_after(
    bodies: BodyContainer, history: wp.array3d[float], counter: wp.array[int], ledger: wp.array[wp.spatial_vector]
):
    b = wp.tid()
    slot = counter[0] % 128
    for d in range(3):
        history[slot, b, 13 + d] = bodies.velocity[b][d]
        history[slot, b, 16 + d] = bodies.angular_velocity[b][d]
    for d in range(6):
        history[slot, b, 19 + d] = ledger[b][d]


@wp.kernel
def advance_record(counter: wp.array[int]):
    counter[0] += 1


@wp.kernel
def record_joint(
    counter: wp.array[int],
    history: wp.array3d[float],
    accumulated: wp.array[float],
    wrench0: wp.array2d[wp.spatial_vector],
    wrench1: wp.array2d[wp.spatial_vector],
    dynamic: wp.array[bool],
    after: bool,
):
    row = wp.tid()
    slot = counter[0] % 128
    if after:
        history[slot, row, 1] = accumulated[row]
    else:
        history[slot, row, 0] = accumulated[row]
        for d in range(6):
            history[slot, row, 2 + d] = wrench0[0, row][d]
            history[slot, row, 8 + d] = wrench1[0, row][d]
        history[slot, row, 14] = wp.float32(dynamic[row])


@wp.kernel
def record_endpoints(
    state: PatchState,
    bodies: BodyContainer,
    columns: ContactColumnContainer,
    anchors: ContactContainer,
    number: wp.array[int],
    counter: wp.array[int],
    history: wp.array3d[float],
):
    col = wp.tid()
    slot = counter[0] % 128
    if col < number[0]:
        b0 = contact_get_body1(columns, col)
        b1 = contact_get_body2(columns, col)
        key = wp.max(b0, b1)
        for a in range(state.count[key]):
            k = 2 * key + a
            p0 = bodies.position[b0] + state.r0[key, a]
            p1 = bodies.position[b1] + state.r1[key, a]
            for d in range(3):
                history[slot, k, d] = p0[d]
                history[slot, k, 3 + d] = p1[d]
            history[slot, k, 6] = cc_get_normal_lambda(anchors, k)
            history[slot, k, 7] = wp.float32(state.count[key])
