"""Local small-component response cache. No contact law or production edits.

The cache is built by applying the existing native factor to coordinate basis
wrenches. It stores every body response and compliant generalized response;
there is no eigenvalue truncation, symmetrization, or added regularization.
"""

import warp as wp

from newton._src.solvers.phoenx.articulations.maximal_contact_response import (
    MaximalContactResponseData,
    apply_maximal_contact_impulse_thread,
)
from newton._src.solvers.phoenx.articulations.maximal_projector import (
    MaximalTreeProjectorData,
    _make_spatial_shift_transform,
    _sync_tree,
)
from newton._src.solvers.phoenx.body import BodyContainer


@wp.struct
class CachedComponentData:
    velocity: wp.array3d[wp.spatial_vectorf]
    joint: wp.array3d[wp.float32]
    valid: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def build_cache(tree: MaximalTreeProjectorData, response: MaximalContactResponseData, cache: CachedComponentData):
    tid = wp.tid()
    articulation = tid // 64
    lane = tid % 64
    count = tree.body_count[articulation]
    if lane == 0:
        cache.valid[articulation] = wp.int32(count == 2)
    _sync_tree()
    if count == 2:
        for coordinate in range(count * 6):
            if lane < count:
                impulse = wp.spatial_vectorf(0.0)
                if lane == coordinate // 6:
                    impulse[coordinate % 6] = 1.0
                response.impulse[articulation, lane] = impulse
            _sync_tree()
            apply_maximal_contact_impulse_thread(articulation, lane, tree, response)
            if lane < count:
                cache.velocity[articulation, coordinate, lane] = response.velocity[articulation, lane]
                cache.joint[articulation, coordinate, lane] = response.joint_velocity[articulation, lane]
            _sync_tree()


@wp.func
def generalized_cached_delta(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    cache: CachedComponentData,
    articulation: wp.int32,
    target: wp.int32,
):
    root_delta = wp.spatial_vectorf(0.0)
    joint_delta = wp.float32(0.0)
    count = tree.body_count[articulation]
    for source in range(count):
        impulse = response.impulse[articulation, source]
        for axis in range(6):
            coefficient = impulse[axis]
            root_delta += coefficient * cache.velocity[articulation, source * 6 + axis, 0]
            joint_delta += coefficient * cache.joint[articulation, source * 6 + axis, target]
    delta = root_delta
    if target > 0:
        transform = _make_spatial_shift_transform(tree.shift[articulation, target])
        delta = transform @ root_delta + joint_delta * tree.motion[articulation, target]
    return delta, joint_delta


@wp.func
def apply_cached(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    cache: CachedComponentData,
    bodies: BodyContainer,
    accumulated: wp.array[wp.float32],
    articulation: wp.int32,
    lane: wp.int32,
):
    # The independent dense and generalized coefficient caches failed the
    # momentum gate. Exact native local elimination is the accepted control:
    # same operations, no per-point shared scratch or block synchronization.
    root_delta, child_delta, joint_delta = serial_native_pair(tree, response, articulation)
    for target in range(2):
        delta = root_delta
        if target == 1:
            delta = child_delta
        body = tree.body_slot[articulation, target]
        bodies.velocity[body] += wp.spatial_top(delta)
        bodies.angular_velocity[body] += wp.spatial_bottom(delta)
        row = tree.dynamic_row[articulation, target]
        if row >= 0:
            accumulated[row] -= tree.generalized_mass[articulation, target] * joint_delta


@wp.func
def cached_cross(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    cache: CachedComponentData,
    body0: wp.int32,
    body1: wp.int32,
    r0: wp.vec3f,
    r1: wp.vec3f,
    direction0: wp.vec3f,
    direction1: wp.vec3f,
):
    articulation = response.body_articulation[body0]
    if articulation < 0:
        articulation = response.body_articulation[body1]
    result = wp.float32(0.0)
    for endpoint in range(2):
        target_body = body0
        target_r = r0
        target_sign = wp.float32(-1.0)
        if endpoint == 1:
            target_body = body1
            target_r = r1
            target_sign = wp.float32(1.0)
        if response.body_articulation[target_body] == articulation:
            target = response.body_lane[target_body]
            tf = target_sign * direction0
            tt = wp.cross(target_r, tf)
            test = wp.spatial_vectorf(tf[0], tf[1], tf[2], tt[0], tt[1], tt[2])
            for source_endpoint in range(2):
                source_body = body0
                source_r = r0
                source_sign = wp.float32(-1.0)
                if source_endpoint == 1:
                    source_body = body1
                    source_r = r1
                    source_sign = wp.float32(1.0)
                if response.body_articulation[source_body] == articulation:
                    source = response.body_lane[source_body]
                    sf = source_sign * direction1
                    st = wp.cross(source_r, sf)
                    impulse = wp.spatial_vectorf(sf[0], sf[1], sf[2], st[0], st[1], st[2])
                    for axis in range(6):
                        result += impulse[axis] * wp.dot(test, cache.velocity[articulation, source * 6 + axis, target])
    return result


def allocate(component_count, device):
    cache = CachedComponentData()
    cache.velocity = wp.zeros((component_count, 12, 2), dtype=wp.spatial_vectorf, device=device)
    cache.joint = wp.zeros((component_count, 12, 2), dtype=float, device=device)
    cache.valid = wp.zeros(component_count, dtype=int, device=device)
    return cache


@wp.kernel
def compare_vectors(a: wp.array[wp.vec3f], b: wp.array[wp.vec3f], errors: wp.array[float], slot: int):
    i = wp.tid()
    for d in range(3):
        wp.atomic_max(errors, slot, wp.abs(a[i][d] - b[i][d]))
        wp.atomic_max(errors, slot + 8, wp.abs(b[i][d]))


@wp.kernel
def compare_scalars(a: wp.array[float], b: wp.array[float], errors: wp.array[float], slot: int):
    i = wp.tid()
    wp.atomic_max(errors, slot, wp.abs(a[i] - b[i]))
    wp.atomic_max(errors, slot + 8, wp.abs(b[i]))


@wp.kernel
def compare_rows(a: wp.array2d[float], b: wp.array2d[float], errors: wp.array[float], slot: int):
    i, j = wp.tid()
    wp.atomic_max(errors, slot, wp.abs(a[i, j] - b[i, j]))
    wp.atomic_max(errors, slot + 8, wp.abs(b[i, j]))


@wp.func
def serial_native_pair(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    articulation: wp.int32,
):
    """Two-body native elimination in local variables, preserving operation order.

    Precision control for the generalized cache. No barriers, array scratch, or
    independent body response summation. Requires exactly root plus one child.
    """
    motion = tree.motion[articulation, 1]
    child_rhs = response.impulse[articulation, 1]
    u = tree.articulated[articulation, 1] @ motion
    projected_rhs = child_rhs - tree.inverse_d[articulation, 1] * wp.dot(motion, child_rhs) * u
    transform = _make_spatial_shift_transform(tree.shift[articulation, 1])
    parent_rhs = wp.transpose(transform) @ projected_rhs
    root_rhs = response.impulse[articulation, 0]
    root_rhs += parent_rhs
    root_delta = response.mobility[articulation, 0] @ root_rhs
    base = transform @ root_delta
    joint_delta = tree.inverse_d[articulation, 1] * wp.dot(motion, child_rhs - tree.articulated[articulation, 1] @ base)
    child_delta = base + joint_delta * motion
    return root_delta, child_delta, joint_delta
