# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton ↔ Box3D format conversion kernels.

Newton stores bodies in flat 1-D arrays (all worlds concatenated) with
``wp.transform`` for pose and ``wp.spatial_vector`` for velocity.
Box3D needs 2-D ``[world, local_body]`` arrays with separate position,
orientation, linear velocity, and angular velocity.

These kernels run once per step at the boundary between Newton's
collision pipeline and the Box3D solver.
"""

from __future__ import annotations

import warp as wp

from ...sim import BodyFlags


# ═══════════════════════════════════════════════════════════════════════
# Newton → Box3D  (bodies)
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def convert_bodies_to_box3d(
    # Newton flat arrays
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    body_world_start: wp.array[wp.int32],
    num_worlds: int,
    # Box3D 2-D outputs
    out_pos: wp.array2d(dtype=wp.vec3),
    out_ori: wp.array2d(dtype=wp.quat),
    out_vel: wp.array2d(dtype=wp.vec3),
    out_ang_vel: wp.array2d(dtype=wp.vec3),
    out_inv_mass: wp.array2d(dtype=float),
    out_inv_inertia: wp.array2d(dtype=wp.mat33),
    out_com: wp.array2d(dtype=wp.vec3),
    out_delta_pos: wp.array2d(dtype=wp.vec3),
    out_inv_inertia_body: wp.array2d(dtype=wp.mat33),
    out_bodies_per_world: wp.array[wp.int32],
):
    """Convert Newton body state to Box3D 2-D layout.

    Launched with ``dim = total_body_count``.  Each thread converts one
    body, determining its world and local index from *body_world_start*.
    """
    global_idx = wp.tid()

    # Determine world and local index.
    # Bodies with world == -1 (global) are mapped to world 0.
    bw = body_world_start[0]  # unused, just to reference the array
    w_raw = body_world[global_idx]
    world = w_raw
    if world < 0:
        world = 0
    local_idx = global_idx - body_world_start[world]
    # For global bodies (before world 0 start), local_idx = global_idx
    if w_raw < 0:
        local_idx = global_idx

    q = body_q[global_idx]
    pos = wp.transform_get_translation(q)
    ori = wp.transform_get_rotation(q)
    com = body_com[global_idx]

    # Store COM world position
    out_pos[world, local_idx] = pos + wp.quat_rotate(ori, com)
    out_ori[world, local_idx] = ori
    out_com[world, local_idx] = com

    qd = body_qd[global_idx]
    out_vel[world, local_idx] = wp.spatial_top(qd)
    out_ang_vel[world, local_idx] = wp.spatial_bottom(qd)

    # Kinematic bodies → zero inverse mass/inertia
    flags = body_flags[global_idx]
    if (flags & BodyFlags.KINEMATIC) != 0:
        out_inv_mass[world, local_idx] = 0.0
        out_inv_inertia[world, local_idx] = wp.mat33(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
    else:
        out_inv_mass[world, local_idx] = body_inv_mass[global_idx]
        # Store body-frame inverse inertia (constant)
        I_body_inv = body_inv_inertia[global_idx]
        out_inv_inertia_body[world, local_idx] = I_body_inv
        # Rotate to world frame for initial substep:
        # I_world^{-1} = R * I_body^{-1} * R^T
        R = wp.quat_to_matrix(ori)
        out_inv_inertia[world, local_idx] = R * I_body_inv * wp.transpose(R)

    # Zero delta-pos for substep tracking
    out_delta_pos[world, local_idx] = wp.vec3(0.0, 0.0, 0.0)

    # Atomically count bodies per world
    wp.atomic_add(out_bodies_per_world, world, 1)


# ═══════════════════════════════════════════════════════════════════════
# Newton → Box3D  (contacts)
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def convert_contacts_to_box3d(
    # Newton contact arrays (1-D, up to rigid_contact_count)
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    rigid_contact_match_index: wp.array[wp.int32],
    # Model arrays
    shape_body: wp.array[wp.int32],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    body_world: wp.array[wp.int32],
    body_world_start: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    # Previous-frame impulses (2-D, indexed by match_index)
    prev_ni: wp.array2d(dtype=float),
    prev_fi1: wp.array2d(dtype=float),
    prev_fi2: wp.array2d(dtype=float),
    prev_count: wp.array[wp.int32],
    # Max contacts per world (for bounds checking)
    max_contacts: int,
    # Box3D raw contact outputs (2-D: [world, contact])
    out_body_a: wp.array2d(dtype=wp.int32),
    out_body_b: wp.array2d(dtype=wp.int32),
    out_normal: wp.array2d(dtype=wp.vec3),
    out_r_a: wp.array2d(dtype=wp.vec3),
    out_r_b: wp.array2d(dtype=wp.vec3),
    out_base_sep: wp.array2d(dtype=float),
    out_friction: wp.array2d(dtype=float),
    out_restitution: wp.array2d(dtype=float),
    out_ni: wp.array2d(dtype=float),
    out_fi1: wp.array2d(dtype=float),
    out_fi2: wp.array2d(dtype=float),
    out_contact_count: wp.array[wp.int32],
):
    """Convert Newton contacts to Box3D raw-contact format.

    Launched with ``dim = rigid_contact_max``.  Each thread checks if its
    index is below the contact count and converts one contact.

    Contact matching: if ``rigid_contact_match_index[i] >= 0``, warm-start
    impulses are loaded from the previous frame's buffers.
    """
    ci = wp.tid()
    count = rigid_contact_count[0]
    if ci >= count:
        return

    s0 = rigid_contact_shape0[ci]
    s1 = rigid_contact_shape1[ci]
    if s0 < 0 or s1 < 0:
        return

    b0_global = shape_body[s0]
    b1_global = shape_body[s1]

    # Determine world (use body 0, or body 1 if body 0 is ground -1).
    # Bodies with world == -1 (global) are mapped to world 0.
    world = int(0)
    if b0_global >= 0:
        w_raw = body_world[b0_global]
        if w_raw >= 0:
            world = w_raw
    elif b1_global >= 0:
        w_raw = body_world[b1_global]
        if w_raw >= 0:
            world = w_raw

    # Local body indices within the world.
    # Ground or unassigned bodies (global_idx < 0) remain -1.
    # Global bodies (world == -1) use global_idx as their local index.
    local_a = int(-1)
    local_b = int(-1)
    if b0_global >= 0:
        bw0 = body_world[b0_global]
        if bw0 >= 0:
            local_a = b0_global - body_world_start[bw0]
        else:
            local_a = b0_global  # global body: local_idx = global_idx
    if b1_global >= 0:
        bw1 = body_world[b1_global]
        if bw1 >= 0:
            local_b = b1_global - body_world_start[bw1]
        else:
            local_b = b1_global

    # Atomic increment to get the per-world contact slot
    slot = wp.atomic_add(out_contact_count, world, 1)
    if slot >= max_contacts:
        return

    out_body_a[world, slot] = local_a
    out_body_b[world, slot] = local_b

    normal = rigid_contact_normal[ci]
    out_normal[world, slot] = normal

    # Compute world-space anchor offsets (r_a, r_b) from body COM.
    # Newton stores contact points in body frame.
    # r = quat_rotate(body_ori, body_frame_point + offset)
    # Then r_a is from COM world pos to the world-space contact point.
    p0 = rigid_contact_point0[ci]
    p1 = rigid_contact_point1[ci]
    off0 = rigid_contact_offset0[ci]
    off1 = rigid_contact_offset1[ci]

    r_a = wp.vec3(0.0, 0.0, 0.0)
    r_b = wp.vec3(0.0, 0.0, 0.0)
    # Friction anchor world positions (using offset for friction anchor)
    anchor_world_a = wp.vec3(0.0, 0.0, 0.0)
    anchor_world_b = wp.vec3(0.0, 0.0, 0.0)
    # Surface contact world positions (using point WITHOUT offset, for separation)
    surface_world_a = wp.vec3(0.0, 0.0, 0.0)
    surface_world_b = wp.vec3(0.0, 0.0, 0.0)

    if b0_global >= 0:
        q0 = body_q[b0_global]
        ori0 = wp.transform_get_rotation(q0)
        pos0 = wp.transform_get_translation(q0)
        com0 = body_com[b0_global]
        com_world0 = pos0 + wp.quat_rotate(ori0, com0)
        anchor_world_a = pos0 + wp.quat_rotate(ori0, p0 + off0)
        surface_world_a = pos0 + wp.quat_rotate(ori0, p0)
        r_a = anchor_world_a - com_world0
    else:
        anchor_world_a = p0 + off0
        surface_world_a = p0

    if b1_global >= 0:
        q1 = body_q[b1_global]
        ori1 = wp.transform_get_rotation(q1)
        pos1 = wp.transform_get_translation(q1)
        com1 = body_com[b1_global]
        com_world1 = pos1 + wp.quat_rotate(ori1, com1)
        anchor_world_b = pos1 + wp.quat_rotate(ori1, p1 + off1)
        surface_world_b = pos1 + wp.quat_rotate(ori1, p1)
        r_b = anchor_world_b - com_world1
    else:
        anchor_world_b = p1 + off1
        surface_world_b = p1

    out_r_a[world, slot] = r_a
    out_r_b[world, slot] = r_b

    # Base separation: signed distance between surfaces (negative = penetrating).
    # Uses body-frame contact points (WITHOUT friction offsets) for geometry,
    # then subtracts margins (effective radii) to get surface-to-surface distance.
    margin0 = rigid_contact_margin0[ci]
    margin1 = rigid_contact_margin1[ci]
    sep = wp.dot(surface_world_b - surface_world_a, normal) - (margin0 + margin1)
    out_base_sep[world, slot] = sep

    # Material: geometric mean friction, max restitution
    mu0 = shape_material_mu[s0]
    mu1 = shape_material_mu[s1]
    out_friction[world, slot] = wp.sqrt(mu0 * mu1)
    rest0 = shape_material_restitution[s0]
    rest1 = shape_material_restitution[s1]
    out_restitution[world, slot] = wp.max(rest0, rest1)

    # Warm starting via contact matching
    ni = 0.0
    fi1 = 0.0
    fi2 = 0.0
    match_idx = rigid_contact_match_index[ci]
    if match_idx >= 0:
        pc = prev_count[world]
        if match_idx < pc:
            ni = prev_ni[world, match_idx]
            fi1 = prev_fi1[world, match_idx]
            fi2 = prev_fi2[world, match_idx]
    out_ni[world, slot] = ni
    out_fi1[world, slot] = fi1
    out_fi2[world, slot] = fi2


# ═══════════════════════════════════════════════════════════════════════
# Box3D → Newton  (bodies)
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def convert_bodies_from_box3d(
    # Box3D 2-D arrays
    box_pos: wp.array2d(dtype=wp.vec3),
    box_ori: wp.array2d(dtype=wp.quat),
    box_vel: wp.array2d(dtype=wp.vec3),
    box_ang_vel: wp.array2d(dtype=wp.vec3),
    box_com: wp.array2d(dtype=wp.vec3),
    # Newton layout info
    body_world_arr: wp.array[wp.int32],
    body_world_start: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    num_worlds: int,
    # Newton flat outputs
    out_body_q: wp.array[wp.transform],
    out_body_qd: wp.array[wp.spatial_vector],
):
    """Convert Box3D body state back to Newton flat format.

    Launched with ``dim = total_body_count``.
    """
    global_idx = wp.tid()

    # Skip kinematic bodies — they were not modified
    flags = body_flags[global_idx]
    if (flags & BodyFlags.KINEMATIC) != 0:
        return

    # Find world + local index
    w_raw = body_world_arr[global_idx]
    world = w_raw
    if world < 0:
        world = 0
    local_idx = global_idx - body_world_start[world]
    if w_raw < 0:
        local_idx = global_idx

    com_world = box_pos[world, local_idx]
    ori = box_ori[world, local_idx]
    com_local = body_com[global_idx]

    # Newton transform origin = com_world - quat_rotate(ori, com_local)
    pos = com_world - wp.quat_rotate(ori, com_local)
    out_body_q[global_idx] = wp.transform(pos, ori)

    vel = box_vel[world, local_idx]
    ang_vel = box_ang_vel[world, local_idx]
    out_body_qd[global_idx] = wp.spatial_vector(vel, ang_vel)


# ═══════════════════════════════════════════════════════════════════════
# Newton → Box3D  (joints)
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def convert_joints_to_box3d(
    # Newton flat arrays
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_world_start: wp.array[wp.int32],
    body_world_start: wp.array[wp.int32],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    num_worlds: int,
    # Box3D 2-D outputs (in color order, mapping provided)
    color_order: wp.array2d(dtype=wp.int32),
    out_body_a: wp.array2d(dtype=wp.int32),
    out_body_b: wp.array2d(dtype=wp.int32),
    out_type: wp.array2d(dtype=wp.int32),
    out_local_anchor_a: wp.array2d(dtype=wp.vec3),
    out_local_anchor_b: wp.array2d(dtype=wp.vec3),
    out_hinge_axis: wp.array2d(dtype=wp.vec3),
    out_limit_lower: wp.array2d(dtype=float),
    out_limit_upper: wp.array2d(dtype=float),
    out_limit_enabled: wp.array2d(dtype=wp.int32),
):
    """Convert Newton joints to Box3D 2-D format with color ordering.

    Launched with ``dim = total_joint_count``.
    """
    global_idx = wp.tid()

    # Find world + local index
    world = int(0)
    for w in range(num_worlds):
        if global_idx >= joint_world_start[w]:
            world = w
    local_idx = global_idx - joint_world_start[world]

    # Get color-order destination slot
    slot = color_order[world, local_idx]

    jtype = joint_type[global_idx]
    out_type[world, slot] = jtype

    # Parent/child body → local indices
    parent_global = joint_parent[global_idx]
    child_global = joint_child[global_idx]
    bws = body_world_start[world]

    local_parent = -1
    local_child = -1
    if parent_global >= 0:
        local_parent = parent_global - bws
    if child_global >= 0:
        local_child = child_global - bws

    out_body_a[world, slot] = local_parent
    out_body_b[world, slot] = local_child

    # Joint anchors in body-local frame
    X_p = joint_X_p[global_idx]
    X_c = joint_X_c[global_idx]
    out_local_anchor_a[world, slot] = wp.transform_get_translation(X_p)
    out_local_anchor_b[world, slot] = wp.transform_get_translation(X_c)

    # Hinge axis (in child body frame for revolute joints)
    out_hinge_axis[world, slot] = joint_axis[global_idx]

    # Limits
    lo = joint_limit_lower[global_idx]
    hi = joint_limit_upper[global_idx]
    out_limit_lower[world, slot] = lo
    out_limit_upper[world, slot] = hi
    # Limits enabled if ke > 0 and lo < hi
    ke = joint_limit_ke[global_idx]
    enabled = 0
    if ke > 0.0 and lo < hi:
        enabled = 1
    out_limit_enabled[world, slot] = enabled
