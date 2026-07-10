"""Rigid-body SoA storage for :class:`PhoenXWorld`.

One ``wp.array`` per field, indexed by body id. All fields are float32.
"""

import os

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_NONE,
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_pose_velocity,
)

__all__ = [
    "MOTION_ARTICULATED",
    "MOTION_DYNAMIC",
    "MOTION_KINEMATIC",
    "MOTION_STATIC",
    "WIDE_LOADS_QUAT",
    "WIDE_LOADS_SYM6",
    "WIDE_LOADS_VW",
    "BodyContainer",
    "ReducedArticulationData",
    "body_alloc_velocity_storage",
    "body_attach_wide_aliases",
    "body_container_zeros",
    "body_load_inv_inertia_sym6",
    "body_load_orientation",
    "body_load_vw",
    "body_set_access_mode",
    "body_store_vw",
    "inertia_sym6",
    "inertia_sym6_pack_np",
    "inertia_sym6_unpack_np",
    "mat33_from_sym6",
    "sym6_from_mat33",
]


# ---------------------------------------------------------------------------
# Wide (multi-word) loads for the per-cid gather-style body reads in the
# multi-world fast-tail / block-world constraint kernels. Warp emits only
# scalar 4-byte loads for vector types; aliasing the underlying buffers as
# 8/16-byte words and loading via minimal ``wp.func_native`` snippets cuts
# the load-request count 2-4x while reading the exact same bytes
# (bit-identical results). Opt-in via ``PHOENX_BODY_WIDE_LOADS`` so A/B
# comparisons run the untouched baseline by default:
#
# * ``vw``   -- velocity + angular_velocity interleaved per body into one
#               24 B slot (three 8-byte words). This is the only field that
#               needs a storage-layout change (two separate 12 B vec3 arrays
#               cannot be 8-byte aligned per element); ``velocity`` /
#               ``angular_velocity`` stay valid as strided vec3 views, so
#               every non-wide reader / writer is unchanged.
# * ``quat`` -- orientation as one 16-byte load (pure alias, no layout change).
# * ``sym6`` -- inverse_inertia_world as three 8-byte words (pure alias).
#
# ``PHOENX_BODY_WIDE_LOADS=1`` (or ``all``) enables everything; a comma list
# (e.g. ``vw,quat``) enables a subset.
def _parse_wide_load_fields() -> frozenset[str]:
    raw = os.environ.get("PHOENX_BODY_WIDE_LOADS", "").strip().lower()
    if raw in ("", "0"):
        return frozenset()
    if raw in ("1", "all"):
        return frozenset({"vw", "quat", "sym6"})
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


_WIDE_LOAD_FIELDS = _parse_wide_load_fields()
#: Debug: interleaved (v, w) layout without the wide kernel loads.
_WIDE_VW_LAYOUT_ONLY: bool = "vwlayout" in _WIDE_LOAD_FIELDS
WIDE_LOADS_VW: bool = "vw" in _WIDE_LOAD_FIELDS
WIDE_LOADS_QUAT: bool = "quat" in _WIDE_LOAD_FIELDS
WIDE_LOADS_SYM6: bool = "sym6" in _WIDE_LOAD_FIELDS


# Motion types (mirrors Jitter2 ``MotionType``).
MOTION_STATIC = wp.constant(0)
MOTION_KINEMATIC = wp.constant(1)
MOTION_DYNAMIC = wp.constant(2)
MOTION_ARTICULATED = wp.constant(3)


# Symmetric 3x3 packed as 6 floats: [I00, I11, I22, I01, I02, I12].
# ``inverse_inertia_world`` (= R I^-1 R^T) is always symmetric, so storing the
# 6 unique entries instead of a full ``mat33f`` cuts its per-body footprint
# 36 B -> 24 B. The hot multi-world iterate reads it per-cid, so the narrower
# load reduces memory traffic on the bandwidth-bound solve; the mat33
# reconstruction is a few register moves.
inertia_sym6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def mat33_from_sym6(s: inertia_sym6) -> wp.mat33f:
    """Expand a packed symmetric 3x3 (see :data:`inertia_sym6`) to ``mat33f``."""
    return wp.mat33f(s[0], s[3], s[4], s[3], s[1], s[5], s[4], s[5], s[2])


@wp.func
def sym6_from_mat33(m: wp.mat33f) -> inertia_sym6:
    """Pack the upper triangle of a (symmetric) ``mat33f`` into :data:`inertia_sym6`."""
    return inertia_sym6(m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2])


def inertia_sym6_pack_np(m: np.ndarray) -> np.ndarray:
    """Host-side pack of a ``(..., 3, 3)`` symmetric array into ``(..., 6)``
    (:data:`inertia_sym6` order). Inverse of :func:`inertia_sym6_unpack_np`."""
    m = np.asarray(m, dtype=np.float32)
    return np.stack(
        [m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], m[..., 0, 1], m[..., 0, 2], m[..., 1, 2]],
        axis=-1,
    )


def inertia_sym6_unpack_np(s: np.ndarray) -> np.ndarray:
    """Host-side expand of a ``(..., 6)`` packed symmetric array (see
    :data:`inertia_sym6`) back to ``(..., 3, 3)``."""
    s = np.asarray(s, dtype=np.float32)
    out = np.zeros((*s.shape[:-1], 3, 3), dtype=np.float32)
    out[..., 0, 0] = s[..., 0]
    out[..., 1, 1] = s[..., 1]
    out[..., 2, 2] = s[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = s[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = s[..., 4]
    out[..., 1, 2] = out[..., 2, 1] = s[..., 5]
    return out


# --- Wide-load native primitives (see PHOENX_BODY_WIDE_LOADS above) --------
# Layout contract for ``_load_vw_wide`` / ``_store_vw_wide``: word ``3*b``
# holds (vx, vy), ``3*b+1`` holds (vz, wx), ``3*b+2`` holds (wy, wz).

_LOAD_VW_WIDE_SNIPPET = """
    const unsigned long long* p = &arr.data[3 * b];
    unsigned long long a0 = p[0];
    unsigned long long a1 = p[1];
    unsigned long long a2 = p[2];
    union { unsigned long long u; float f[2]; } c;
    wp::spatial_vectorf out;
    c.u = a0; out[0] = c.f[0]; out[1] = c.f[1];
    c.u = a1; out[2] = c.f[0]; out[3] = c.f[1];
    c.u = a2; out[4] = c.f[0]; out[5] = c.f[1];
    return out;
"""


@wp.func_native(_LOAD_VW_WIDE_SNIPPET)
def _load_vw_wide(arr: wp.array[wp.uint64], b: wp.int32) -> wp.spatial_vector:
    """Load body ``b``'s (velocity, angular_velocity) as three 8-byte words."""
    ...


_STORE_VW_WIDE_SNIPPET = """
    unsigned long long* p = &arr.data[3 * b];
    union { unsigned long long u; float f[2]; } c;
    c.f[0] = v[0]; c.f[1] = v[1]; p[0] = c.u;
    c.f[0] = v[2]; c.f[1] = w[0]; p[1] = c.u;
    c.f[0] = w[1]; c.f[1] = w[2]; p[2] = c.u;
"""


@wp.func_native(_STORE_VW_WIDE_SNIPPET)
def _store_vw_wide(arr: wp.array[wp.uint64], b: wp.int32, v: wp.vec3f, w: wp.vec3f):
    """Store body ``b``'s (velocity, angular_velocity) as three 8-byte words."""
    ...


_LOAD_QUAT_WIDE_SNIPPET = """
    const float4 q = *reinterpret_cast<const float4*>(&arr.data[2 * b]);
    return wp::quatf(q.x, q.y, q.z, q.w);
"""


@wp.func_native(_LOAD_QUAT_WIDE_SNIPPET)
def _load_quat_wide(arr: wp.array[wp.uint64], b: wp.int32) -> wp.quatf:
    """Load body ``b``'s orientation as one 16-byte word."""
    ...


_LOAD_SYM6_WIDE_SNIPPET = """
    const unsigned long long* p = &arr.data[3 * b];
    unsigned long long a0 = p[0];
    unsigned long long a1 = p[1];
    unsigned long long a2 = p[2];
    union { unsigned long long u; float f[2]; } c;
    wp::vec_t<6, float> out;
    c.u = a0; out[0] = c.f[0]; out[1] = c.f[1];
    c.u = a1; out[2] = c.f[0]; out[3] = c.f[1];
    c.u = a2; out[4] = c.f[0]; out[5] = c.f[1];
    return out;
"""


@wp.func_native(_LOAD_SYM6_WIDE_SNIPPET)
def _load_sym6_wide(arr: wp.array[wp.uint64], b: wp.int32) -> inertia_sym6:
    """Load body ``b``'s packed symmetric inertia as three 8-byte words."""
    ...


@wp.struct
class ReducedArticulationData:
    """Device view of a cached reduced-coordinate inverse-mass operator."""

    enabled: wp.array[wp.int32]
    body_articulation: wp.array[wp.int32]
    articulation_origin: wp.array[wp.vec3]
    body_q_com: wp.array[wp.transform]
    articulation_start: wp.array[wp.int32]
    articulation_end: wp.array[wp.int32]
    joint_parent: wp.array[wp.int32]
    joint_child: wp.array[wp.int32]
    joint_qd_start: wp.array[wp.int32]
    joint_s: wp.array[wp.spatial_vector]
    joint_u: wp.array[wp.spatial_vector]
    joint_d_inv: wp.array2d[wp.float32]
    joint_qd: wp.array[wp.float32]
    body_work: wp.array[wp.spatial_vector]
    joint_work: wp.array[wp.float32]
    body_acceleration: wp.array[wp.spatial_vector]
    generalized_response: wp.array[wp.float32]
    impulse_response: wp.array2d[wp.spatial_vector]
    deferred_wrench: wp.array[wp.spatial_vector]
    body_joint: wp.array[wp.int32]
    body_path_start: wp.array[wp.int32]
    body_path_joint: wp.array[wp.int32]


def reduced_articulation_data_zeros(device: wp.DeviceLike = None) -> ReducedArticulationData:
    """Allocate a disabled placeholder accepted by graph-specialized kernels."""
    data = ReducedArticulationData()
    data.enabled = wp.zeros(1, dtype=wp.int32, device=device)
    data.body_articulation = wp.full(1, value=-1, dtype=wp.int32, device=device)
    data.articulation_origin = wp.zeros(1, dtype=wp.vec3, device=device)
    data.body_q_com = wp.zeros(1, dtype=wp.transform, device=device)
    data.articulation_start = wp.zeros(1, dtype=wp.int32, device=device)
    data.articulation_end = wp.zeros(1, dtype=wp.int32, device=device)
    data.joint_parent = wp.full(1, value=-1, dtype=wp.int32, device=device)
    data.joint_child = wp.zeros(1, dtype=wp.int32, device=device)
    data.joint_qd_start = wp.zeros(2, dtype=wp.int32, device=device)
    data.joint_s = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    data.joint_u = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    data.joint_d_inv = wp.zeros((1, 6), dtype=wp.float32, device=device)
    data.joint_qd = wp.zeros(1, dtype=wp.float32, device=device)
    data.body_work = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    data.joint_work = wp.zeros(1, dtype=wp.float32, device=device)
    data.body_acceleration = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    data.generalized_response = wp.zeros(1, dtype=wp.float32, device=device)
    data.impulse_response = wp.zeros((1, 6), dtype=wp.spatial_vector, device=device)
    data.deferred_wrench = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    data.body_joint = wp.full(1, value=-1, dtype=wp.int32, device=device)
    data.body_path_start = wp.zeros(2, dtype=wp.int32, device=device)
    data.body_path_joint = wp.zeros(1, dtype=wp.int32, device=device)
    return data


@wp.struct
class BodyContainer:
    """SoA storage for a batch of rigid bodies. All arrays length ``num_bodies``."""

    position: wp.array[wp.vec3f]
    velocity: wp.array[wp.vec3f]
    angular_velocity: wp.array[wp.vec3f]

    orientation: wp.array[wp.quatf]

    #: Local-frame offset from body-origin (Newton contact-anchor frame) to COM.
    #: :attr:`position` tracks the COM; contact anchors must subtract ``body_com``
    #: before being used as lever arms about the COM.
    body_com: wp.array[wp.vec3f]

    #: Symmetric inverse inertia in world frame, packed as 6 floats
    #: (:data:`inertia_sym6`). Read via :func:`mat33_from_sym6`.
    inverse_inertia_world: wp.array[inertia_sym6]
    inverse_inertia: wp.array[wp.mat33f]

    inverse_mass: wp.array[wp.float32]

    force: wp.array[wp.vec3f]
    torque: wp.array[wp.vec3f]

    linear_damping: wp.array[wp.float32]
    angular_damping: wp.array[wp.float32]

    affected_by_gravity: wp.array[wp.int32]
    motion_type: wp.array[wp.int32]
    world_id: wp.array[wp.int32]
    #: Graph-coloring ownership node. Articulation links share their root
    #: node so no two constraints can update one generalized velocity
    #: vector concurrently.
    constraint_node: wp.array[wp.int32]
    reduced: ReducedArticulationData

    # Kinematic-body pose scripting: prev = lerp/slerp origin, target = endpoint.
    # If kinematic_target_valid == 0 the prepare kernel synthesises target from
    # constant-velocity fallthrough.
    position_prev: wp.array[wp.vec3f]
    orientation_prev: wp.array[wp.quatf]
    kinematic_target_pos: wp.array[wp.vec3f]
    kinematic_target_orient: wp.array[wp.quatf]
    kinematic_target_valid: wp.array[wp.int32]

    #: Pose snapshot at substep entry. Read by
    #: :mod:`newton._src.solvers.phoenx.access_mode` synchronize helpers
    #: as the finite-diff anchor when a constraint flips this body's
    #: :attr:`access_mode` between velocity- and position-level. Written
    #: once per substep by ``_phoenx_apply_forces_and_gravity_kernel``;
    #: not read in the rigid PGS hot path.
    position_prev_substep: wp.array[wp.vec3f]
    orientation_prev_substep: wp.array[wp.quatf]
    #: Per-body access-mode tag (see
    #: :mod:`newton._src.solvers.phoenx.access_mode`). Set at substep
    #: entry to ``VELOCITY_LEVEL`` for dynamic bodies and ``STATIC`` for
    #: anchored / kinematic / pinned bodies. Constraint kernels that
    #: want to read or write at a specific level call
    #: ``body_set_access_mode`` to flip lazily.
    access_mode: wp.array[wp.int32]

    #: Single-element scene-wide flag. ``1`` if the scene contains any
    #: constraint type that can write ``ACCESS_MODE_POSITION_LEVEL``
    #: (cloth triangle, cloth bending, soft-tet XPBD) -- ``0`` otherwise.
    #: Read as the first line of :func:`body_set_access_mode` /
    #: :func:`particle_set_access_mode`: a warp-uniform broadcast load
    #: that short-circuits the whole flip in rigid-only scenes without
    #: touching the scattered :attr:`access_mode` array.
    has_position_level_writers: wp.array[wp.int32]

    #: Per-body sleep label. ``-1`` = awake; any non-negative value is
    #: the *root body id* (the lowest body id in the island this body
    #: was in at the moment it fell asleep). Acts as both the sleeping
    #: flag and the persistent island identifier consumed by
    #: :meth:`PhoenXWorld.wake_on_external_input` -- a force on any
    #: sleeping body lifts every body sharing the same ``island_root``.
    #: Only written when :attr:`PhoenXWorld.sleeping_velocity_threshold`
    #: > 0; stays at ``-1`` (awake) when sleeping is disabled. Sleeping
    #: bodies skip gravity / force application, are collapsed to -1 in
    #: the partitioner's element view (so their constraints drop out
    #: of the coloring), and are excluded from the per-step union-find
    #: island build (so the live ``set_nr`` covers awake bodies only).
    island_root: wp.array[wp.int32]

    #: Aliased wide views for the per-cid gather loads in the constraint hot
    #: path (see the ``PHOENX_BODY_WIDE_LOADS`` note at the top of this
    #: module). Never read unless the matching flag is enabled at import
    #: time; small dummies otherwise.
    #:
    #: ``velocity_pair_u64``: 8-byte word view of the interleaved
    #: (velocity, angular_velocity) buffer -- only meaningful when
    #: :data:`WIDE_LOADS_VW` picked the interleaved layout at allocation.
    velocity_pair_u64: wp.array[wp.uint64]
    #: 8-byte word view aliasing :attr:`orientation` (2 words per quat).
    orientation_u64: wp.array[wp.uint64]
    #: 8-byte word view aliasing :attr:`inverse_inertia_world` (3 words each).
    inverse_inertia_world_u64: wp.array[wp.uint64]

    #: Per-body counter: number of consecutive frames the body's island
    #: has had its max-velocity score below
    #: :attr:`PhoenXWorld.sleeping_velocity_threshold`. Reset to 0 the
    #: moment any body in the island exceeds the threshold (the
    #: island-level max governs every body in the island). Saturates
    #: at :attr:`PhoenXWorld.sleeping_frames_required` so it never
    #: overflows. :attr:`island_root` is stamped with the island's
    #: lowest body id once this counter reaches the required frame
    #: count; with ``sleeping_frames_required == 0`` the body sleeps
    #: on the first below-threshold frame.
    frames_below_threshold: wp.array[wp.int32]


def _alias_u64(parent: wp.array, count: int) -> wp.array:
    """8-byte word view over ``parent``'s buffer (keeps ``parent`` alive)."""
    alias = wp.array(ptr=parent.ptr, dtype=wp.uint64, shape=(count,), device=parent.device)
    alias._ref = parent
    return alias


def body_alloc_velocity_storage(
    num_bodies: int,
    device: wp.DeviceLike = None,
    velocities: np.ndarray | None = None,
    angular_velocities: np.ndarray | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Allocate ``(velocity, angular_velocity, velocity_pair_u64)``.

    Default layout: two contiguous vec3 arrays plus a dummy pair view.
    With :data:`WIDE_LOADS_VW` the pair is interleaved per body into one
    2N-element vec3 buffer (24 B, 8-byte aligned slots); ``velocity`` and
    ``angular_velocity`` become strided views into it so all existing
    readers / writers observe identical values, and ``velocity_pair_u64``
    aliases the buffer for the wide load / store primitives.

    Both :class:`BodyContainer` construction sites (this module and
    ``world_builder``) must allocate through this helper.
    """
    if (not WIDE_LOADS_VW and not _WIDE_VW_LAYOUT_ONLY) or num_bodies == 0:
        if velocities is None:
            vel = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
            ang = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
        else:
            vel = wp.array(velocities, dtype=wp.vec3f, device=device)
            ang = wp.array(angular_velocities, dtype=wp.vec3f, device=device)
        return vel, ang, wp.zeros(3, dtype=wp.uint64, device=device)
    if velocities is None:
        buf = wp.zeros(2 * num_bodies, dtype=wp.vec3f, device=device)
    else:
        interleaved = np.empty((2 * num_bodies, 3), dtype=np.float32)
        interleaved[0::2] = np.asarray(velocities, dtype=np.float32)
        interleaved[1::2] = np.asarray(angular_velocities, dtype=np.float32)
        buf = wp.array(interleaved, dtype=wp.vec3f, device=device)
    itemsize = 12  # vec3f
    vel = wp.array(ptr=buf.ptr, dtype=wp.vec3f, shape=(num_bodies,), strides=(2 * itemsize,), device=buf.device)
    vel._ref = buf
    ang = wp.array(
        ptr=buf.ptr + itemsize, dtype=wp.vec3f, shape=(num_bodies,), strides=(2 * itemsize,), device=buf.device
    )
    ang._ref = buf
    return vel, ang, _alias_u64(buf, 3 * num_bodies)


def body_attach_wide_aliases(c: BodyContainer, num_bodies: int, device: wp.DeviceLike = None) -> None:
    """Set the ``orientation_u64`` / ``inverse_inertia_world_u64`` aliases.

    Pure views over the already-allocated contiguous buffers (16 B quat ->
    2 words, 24 B sym6 -> 3 words); dummies when the container is empty.
    Call after ``orientation`` and ``inverse_inertia_world`` are set.
    """
    if num_bodies == 0:
        c.orientation_u64 = wp.zeros(2, dtype=wp.uint64, device=device)
        c.inverse_inertia_world_u64 = wp.zeros(3, dtype=wp.uint64, device=device)
        return
    c.orientation_u64 = _alias_u64(c.orientation, 2 * num_bodies)
    c.inverse_inertia_world_u64 = _alias_u64(c.inverse_inertia_world, 3 * num_bodies)


@wp.func
def body_load_vw(bodies: BodyContainer, b: wp.int32):
    """Load ``(velocity, angular_velocity)`` for body ``b`` (hot per-cid path).

    Wide path (:data:`WIDE_LOADS_VW`): three 8-byte loads from the
    interleaved pair buffer -- bit-identical to the scalar path.
    """
    if wp.static(WIDE_LOADS_VW):
        vw = _load_vw_wide(bodies.velocity_pair_u64, b)
        return wp.vec3f(vw[0], vw[1], vw[2]), wp.vec3f(vw[3], vw[4], vw[5])
    return bodies.velocity[b], bodies.angular_velocity[b]


@wp.func
def body_store_vw(bodies: BodyContainer, b: wp.int32, v: wp.vec3f, w: wp.vec3f):
    """Store ``(velocity, angular_velocity)`` for body ``b``; pairs :func:`body_load_vw`."""
    if wp.static(WIDE_LOADS_VW):
        _store_vw_wide(bodies.velocity_pair_u64, b, v, w)
        return
    bodies.velocity[b] = v
    bodies.angular_velocity[b] = w


@wp.func
def body_load_orientation(bodies: BodyContainer, b: wp.int32) -> wp.quatf:
    """Load body ``b``'s orientation (one 16-byte load under :data:`WIDE_LOADS_QUAT`)."""
    if wp.static(WIDE_LOADS_QUAT):
        return _load_quat_wide(bodies.orientation_u64, b)
    return bodies.orientation[b]


@wp.func
def body_load_inv_inertia_sym6(bodies: BodyContainer, b: wp.int32) -> inertia_sym6:
    """Load body ``b``'s packed world inverse inertia (three 8-byte loads under :data:`WIDE_LOADS_SYM6`)."""
    if wp.static(WIDE_LOADS_SYM6):
        return _load_sym6_wide(bodies.inverse_inertia_world_u64, b)
    return bodies.inverse_inertia_world[b]


def body_container_zeros(num_bodies: int, device: wp.DeviceLike = None) -> BodyContainer:
    """Allocate a zero-initialized :class:`BodyContainer` on ``device``.

    Defaults: damping = 1.0 (none), affected_by_gravity = 1, motion_type = STATIC,
    world_id = 0. Quaternions are zero — callers must set identity if needed.
    """
    c = BodyContainer()
    c.position = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.velocity, c.angular_velocity, c.velocity_pair_u64 = body_alloc_velocity_storage(num_bodies, device)
    c.orientation = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.body_com = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.inverse_inertia_world = wp.zeros(num_bodies, dtype=inertia_sym6, device=device)
    c.inverse_inertia = wp.zeros(num_bodies, dtype=wp.mat33f, device=device)
    c.inverse_mass = wp.zeros(num_bodies, dtype=wp.float32, device=device)
    c.force = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.torque = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.linear_damping = wp.full(num_bodies, value=1.0, dtype=wp.float32, device=device)
    c.angular_damping = wp.full(num_bodies, value=1.0, dtype=wp.float32, device=device)
    c.affected_by_gravity = wp.full(num_bodies, value=1, dtype=wp.int32, device=device)
    c.motion_type = wp.full(num_bodies, value=int(MOTION_STATIC), dtype=wp.int32, device=device)
    c.world_id = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    c.constraint_node = wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    c.reduced = reduced_articulation_data_zeros(device)
    c.position_prev = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation_prev = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.kinematic_target_pos = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.kinematic_target_orient = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.kinematic_target_valid = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    c.position_prev_substep = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation_prev_substep = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.access_mode = wp.full(num_bodies, value=int(ACCESS_MODE_VELOCITY_LEVEL), dtype=wp.int32, device=device)
    c.has_position_level_writers = wp.zeros(1, dtype=wp.int32, device=device)
    c.island_root = wp.full(num_bodies, value=-1, dtype=wp.int32, device=device)
    c.frames_below_threshold = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    body_attach_wide_aliases(c, num_bodies, device)
    return c


@wp.func
def body_set_access_mode(
    bodies: BodyContainer,
    b: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Lazy SoA wrapper around :func:`synchronize_pose_velocity`.

    First check is the warp-uniform scene-wide gate
    :attr:`BodyContainer.has_position_level_writers` -- when ``0``
    (no cloth / cloth-bend / soft-tet in the scene) the entire flip
    is provably a no-op, and the broadcast load avoids touching the
    scattered :attr:`access_mode` array. Second check is the per-body
    same-mode fast path.

    Mirrors Jitter2 ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
    (``MassSplitting/TinyRigidState.cs:62-65``).
    """
    if bodies.has_position_level_writers[0] == 0:
        return
    current = bodies.access_mode[b]
    # Early-out paths -- match the no-op short-circuits in
    # :func:`synchronize_pose_velocity`. Skipping the function call
    # avoids reading the six pose / velocity arrays on a no-op flip,
    # which is the common case for rigid-only scenes (every body
    # stays in VELOCITY_LEVEL after the substep-entry kernel sets it).
    if current == new_access_mode:
        return
    if current == ACCESS_MODE_STATIC:
        return
    if current == ACCESS_MODE_NONE:
        bodies.access_mode[b] = new_access_mode
        return
    p_new, q_new, v_new, w_new, mode_new = synchronize_pose_velocity(
        bodies.position[b],
        bodies.orientation[b],
        bodies.velocity[b],
        bodies.angular_velocity[b],
        bodies.position_prev_substep[b],
        bodies.orientation_prev_substep[b],
        current,
        new_access_mode,
        inv_dt,
    )
    bodies.position[b] = p_new
    bodies.orientation[b] = q_new
    bodies.velocity[b] = v_new
    bodies.angular_velocity[b] = w_new
    bodies.access_mode[b] = mode_new
