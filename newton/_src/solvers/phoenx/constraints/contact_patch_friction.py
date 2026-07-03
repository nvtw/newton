# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent state for convex contact-patch friction."""

from __future__ import annotations

import warp as wp

from newton._src.geometry.types import GeoType
from newton._src.solvers.phoenx.constraints.constraint_block import (
    BlockVector2Update,
    block_project_friction_delta_sor_2,
)


@wp.struct
class ContactPatchFriction:
    """Per-contact-column state for a coupled two-dimensional friction row."""

    eligible: wp.array[wp.int32]
    impulse_world: wp.array[wp.vec3]
    prev_impulse_world: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    tangent1: wp.array[wp.vec3]
    r0: wp.array[wp.vec3]
    r1: wp.array[wp.vec3]
    effective_mass: wp.array[wp.mat22]
    bias: wp.array[wp.vec2]


@wp.func
def contact_patch_effective_mass(
    tangent1: wp.vec3f,
    tangent2: wp.vec3f,
    r0: wp.vec3f,
    r1: wp.vec3f,
    inverse_mass0: wp.float32,
    inverse_mass1: wp.float32,
    inverse_inertia0: wp.mat33f,
    inverse_inertia1: wp.mat33f,
) -> wp.mat22f:
    """Return the inverse full 2-by-2 tangent response at one point."""
    linear = inverse_mass0 + inverse_mass1
    r0_t1 = wp.cross(r0, tangent1)
    r0_t2 = wp.cross(r0, tangent2)
    r1_t1 = wp.cross(r1, tangent1)
    r1_t2 = wp.cross(r1, tangent2)
    k00 = linear + wp.dot(r0_t1, inverse_inertia0 @ r0_t1) + wp.dot(r1_t1, inverse_inertia1 @ r1_t1)
    k01 = wp.dot(r0_t1, inverse_inertia0 @ r0_t2) + wp.dot(r1_t1, inverse_inertia1 @ r1_t2)
    k11 = linear + wp.dot(r0_t2, inverse_inertia0 @ r0_t2) + wp.dot(r1_t2, inverse_inertia1 @ r1_t2)
    determinant = k00 * k11 - k01 * k01
    if determinant > wp.float32(1.0e-20):
        inverse_determinant = wp.float32(1.0) / determinant
        return wp.mat22f(
            k11 * inverse_determinant,
            -k01 * inverse_determinant,
            -k01 * inverse_determinant,
            k00 * inverse_determinant,
        )
    return wp.mat22f(0.0, 0.0, 0.0, 0.0)


@wp.func
def contact_patch_project_velocity_update(
    lambda_old: wp.vec2f,
    relative_velocity: wp.vec2f,
    bias: wp.vec2f,
    effective_mass: wp.mat22f,
    normal_load: wp.float32,
    friction_static: wp.float32,
    friction_kinetic: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector2Update:
    """Solve one coupled tangent block and project it onto a Coulomb disk.

    effective_mass is the inverse of the full 2-by-2 tangent response,
    including the off-diagonal coupling caused by rotational inertia.
    """
    rhs = relative_velocity + bias
    delta = -(effective_mass @ rhs)
    return block_project_friction_delta_sor_2(
        lambda_old[0],
        lambda_old[1],
        delta[0],
        delta[1],
        sor_boost,
        friction_static * normal_load,
        friction_kinetic * normal_load,
    )


@wp.func
def _shape_is_convex_for_contact_patch(shape_type: wp.int32) -> wp.bool:
    """Return whether collision reduction preserves one convex patch."""
    return (
        shape_type == wp.int32(GeoType.PLANE)
        or shape_type == wp.int32(GeoType.SPHERE)
        or shape_type == wp.int32(GeoType.CAPSULE)
        or shape_type == wp.int32(GeoType.ELLIPSOID)
        or shape_type == wp.int32(GeoType.CYLINDER)
        or shape_type == wp.int32(GeoType.BOX)
        or shape_type == wp.int32(GeoType.CONE)
        or shape_type == wp.int32(GeoType.CONVEX_MESH)
        or shape_type == wp.int32(GeoType.TRIANGLE)
        or shape_type == wp.int32(GeoType.TETRAHEDRON)
    )


@wp.kernel(enable_backward=False)
def _classify_contact_patch_columns_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    allow_shape_pair_columns: wp.int32,
    patch: ContactPatchFriction,
):
    column = wp.tid()
    if column >= num_contact_columns[0]:
        return

    eligible = wp.int32(0)
    if allow_shape_pair_columns != wp.int32(0):
        pair = pair_source_idx[column]
        shape_a = pair_shape_a[pair]
        shape_b = pair_shape_b[pair]
        if _shape_is_convex_for_contact_patch(shape_type[shape_a]) and _shape_is_convex_for_contact_patch(
            shape_type[shape_b]
        ):
            eligible = wp.int32(1)
    patch.eligible[column] = eligible


@wp.kernel(enable_backward=False)
def _gather_contact_patch_warmstart_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    allow_shape_pair_columns: wp.int32,
    rigid_contact_match_index: wp.array[wp.int32],
    prev_cid_of_contact: wp.array[wp.int32],
    reuse_contact_indices: wp.array[wp.int32],
    previous_cid_base: wp.int32,
    patch: ContactPatchFriction,
):
    column = wp.tid()
    if column >= num_contact_columns[0]:
        return

    pair = pair_source_idx[column]
    eligible = wp.int32(0)
    if allow_shape_pair_columns != wp.int32(0):
        shape_a = pair_shape_a[pair]
        shape_b = pair_shape_b[pair]
        if _shape_is_convex_for_contact_patch(shape_type[shape_a]) and _shape_is_convex_for_contact_patch(
            shape_type[shape_b]
        ):
            eligible = wp.int32(1)
    patch.eligible[column] = eligible

    impulse = wp.vec3f(0.0, 0.0, 0.0)
    if reuse_contact_indices[0] != wp.int32(0):
        first = pair_first[pair]
        count = pair_count[pair]
        for i in range(count):
            match = rigid_contact_match_index[first + i]
            if match >= wp.int32(0):
                previous_cid = prev_cid_of_contact[match]
                previous_column = previous_cid - previous_cid_base
                if previous_column >= wp.int32(0) and previous_column < patch.prev_impulse_world.shape[0]:
                    impulse = patch.prev_impulse_world[previous_column]
                    break
    patch.impulse_world[column] = impulse


@wp.kernel(enable_backward=False)
def _copy_contact_patch_impulses_kernel(
    num_contact_columns: wp.array[wp.int32],
    patch: ContactPatchFriction,
):
    column = wp.tid()
    if column < num_contact_columns[0]:
        patch.prev_impulse_world[column] = patch.impulse_world[column]


def contact_patch_friction_zeros(
    max_contact_columns: int,
    device: wp.DeviceLike = None,
) -> ContactPatchFriction:
    """Allocate zero-initialized per-column patch-friction state."""
    capacity = max(1, int(max_contact_columns))
    patch = ContactPatchFriction()
    patch.eligible = wp.zeros(capacity, dtype=wp.int32, device=device)
    patch.impulse_world = wp.zeros(capacity, dtype=wp.vec3, device=device)
    patch.prev_impulse_world = wp.zeros(capacity, dtype=wp.vec3, device=device)
    patch.normal = wp.zeros(capacity, dtype=wp.vec3, device=device)
    patch.tangent1 = wp.zeros(capacity, dtype=wp.vec3, device=device)
    patch.r0 = wp.zeros(capacity, dtype=wp.vec3, device=device)
    patch.r1 = wp.zeros(capacity, dtype=wp.vec3, device=device)
    patch.effective_mass = wp.zeros(capacity, dtype=wp.mat22, device=device)
    patch.bias = wp.zeros(capacity, dtype=wp.vec2, device=device)
    return patch


def classify_contact_patch_columns(
    patch: ContactPatchFriction,
    scratch,
    shape_type: wp.array[wp.int32],
    *,
    enable_body_pair_grouping: bool,
    device: wp.DeviceLike = None,
) -> None:
    """Classify current contact columns without host readback.

    Body-pair grouping may combine contacts from several shape pairs, so those
    columns conservatively retain point friction until collision reduction
    supplies explicit patch identifiers.
    """
    wp.launch(
        _classify_contact_patch_columns_kernel,
        dim=int(patch.eligible.shape[0]),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            scratch.num_contact_columns,
            shape_type,
            int(not enable_body_pair_grouping),
            patch,
        ],
        device=device,
    )


def gather_contact_patch_warmstart(
    patch: ContactPatchFriction,
    scratch,
    rigid_contact_match_index: wp.array[wp.int32],
    prev_cid_of_contact: wp.array[wp.int32],
    reuse_contact_indices: wp.array[wp.int32],
    previous_cid_base: int,
    shape_type: wp.array[wp.int32],
    *,
    enable_body_pair_grouping: bool,
    device: wp.DeviceLike = None,
) -> None:
    """Gather each current patch impulse through persistent contact matches.

    The first matched contact in packed order identifies the previous column.
    This remains correct when column order or the first contact in a manifold
    changes between frames.
    """
    wp.launch(
        _gather_contact_patch_warmstart_kernel,
        dim=int(patch.eligible.shape[0]),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_first,
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            scratch.num_contact_columns,
            shape_type,
            int(not enable_body_pair_grouping),
            rigid_contact_match_index,
            prev_cid_of_contact,
            reuse_contact_indices,
            previous_cid_base,
            patch,
        ],
        device=device,
    )


def copy_contact_patch_impulses(
    patch: ContactPatchFriction,
    num_contact_columns: wp.array[wp.int32],
    device: wp.DeviceLike = None,
) -> None:
    """Copy live current patch impulses into stable previous-frame storage."""
    wp.launch(
        _copy_contact_patch_impulses_kernel,
        dim=int(patch.eligible.shape[0]),
        inputs=[num_contact_columns, patch],
        device=device,
    )
