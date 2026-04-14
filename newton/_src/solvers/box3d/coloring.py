# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph coloring and contact preparation for the Box3D solver.

Contact coloring runs on the GPU via a ``@wp.func_native`` snippet
(thread 0 per block does the greedy color assignment while other threads
wait, then all threads scatter contacts into color order in parallel).

Joint coloring is done on the CPU at solver construction since the joint
topology is static.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .constraint_funcs import (
    compute_contact_effective_mass,
    compute_relative_velocity,
    compute_tangent_basis,
)


# ═══════════════════════════════════════════════════════════════════════
# CPU joint coloring (called once at construction)
# ═══════════════════════════════════════════════════════════════════════


def color_joints_cpu(
    body_a: np.ndarray,
    body_b: np.ndarray,
    num_joints: int,
    num_bodies: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy bitmask graph coloring for joints (CPU, NumPy).

    Two joints conflict if they share a body.  Ground bodies (index -1)
    never cause conflicts.

    Returns:
        order: Permutation that sorts joints by color, shape ``(num_joints,)``.
        offsets: Prefix-sum array ``(num_colors + 1,)`` where
            ``offsets[c]..offsets[c+1]`` gives the range of joints in color ``c``.
        num_colors: Number of colors used.
    """
    if num_joints == 0:
        return np.array([], dtype=np.int32), np.array([0], dtype=np.int32), 0

    colors = np.full(num_joints, -1, dtype=np.int32)
    body_used = [0] * (num_bodies + 1)  # slot num_bodies = ground sentinel

    for j in range(num_joints):
        a = int(body_a[j])
        b = int(body_b[j])
        bi_a = a if a >= 0 else num_bodies
        bi_b = b if b >= 0 else num_bodies
        used = body_used[bi_a] | body_used[bi_b]

        color = 0
        while used & (1 << color):
            color += 1
        colors[j] = color

        mask = 1 << color
        if a >= 0:
            body_used[bi_a] |= mask
        if b >= 0:
            body_used[bi_b] |= mask

    num_colors = int(colors.max()) + 1
    order = np.argsort(colors, kind="stable").astype(np.int32)
    offsets = np.zeros(num_colors + 1, dtype=np.int32)
    for c in range(num_colors):
        offsets[c + 1] = offsets[c] + int(np.sum(colors == c))

    return order, offsets, num_colors


# ═══════════════════════════════════════════════════════════════════════
# GPU contact coloring + preparation kernel
# ═══════════════════════════════════════════════════════════════════════

# The coloring + prepare is done in a single tiled kernel:
#   Phase 1 (thread 0): greedy-color raw contacts, build color_offsets.
#   Phase 2 (all threads): scatter raw contacts → color-ordered solver arrays,
#           compute tangent basis, effective masses, pre-solve relative velocity.

_COLORING_PREPARE_SNIPPET = r"""
    int nc = *wp::address(contact_count, wid);

    // ── Phase 1: greedy coloring (thread 0 only) ──────────────────

    // temp_order[i] stores the destination slot for raw contact i
    // We reuse the color_to_raw buffer as temp scratch for the per-contact color.
    // After coloring, we compute a prefix sum to get offsets then scatter.

    // Per-body color bitmask (in global memory — color_body_mask array)
    if (tid == 0) {
        int nb = *wp::address(bodies_per_world, wid);

        // Zero body masks
        for (int b = 0; b < nb; b++) {
            *wp::address(color_body_mask, wid, b) = (long long)0;
        }

        int max_color_used = 0;

        // Greedy assign colors to each raw contact
        for (int c = 0; c < nc; c++) {
            int a = *wp::address(raw_body_a, wid, c);
            int b = *wp::address(raw_body_b, wid, c);

            long long used = 0;
            if (a >= 0) used |= *wp::address(color_body_mask, wid, a);
            if (b >= 0) used |= *wp::address(color_body_mask, wid, b);

            int color = 0;
            while (used & (1LL << color)) color++;
            if (color > max_color_used) max_color_used = color;

            // Store color in color_to_raw temporarily
            *wp::address(color_to_raw, wid, c) = color;

            long long mask = 1LL << color;
            if (a >= 0) *wp::address(color_body_mask, wid, a) |= mask;
            if (b >= 0) *wp::address(color_body_mask, wid, b) |= mask;
        }

        // Build color_offsets via counting sort
        int num_colors = max_color_used + 1;
        for (int k = 0; k <= max_colors; k++) {
            *wp::address(color_offsets, wid, k) = 0;
        }
        // Count per color
        for (int c = 0; c < nc; c++) {
            int col = *wp::address(color_to_raw, wid, c);
            // Use offsets[col+1] as counter
            int prev = *wp::address(color_offsets, wid, col + 1);
            *wp::address(color_offsets, wid, col + 1) = prev + 1;
        }
        // Prefix sum
        for (int k = 1; k <= max_colors; k++) {
            int prev = *wp::address(color_offsets, wid, k - 1);
            int cur = *wp::address(color_offsets, wid, k);
            *wp::address(color_offsets, wid, k) = prev + cur;
        }

        // Compute scatter indices: color_to_raw[c] = destination slot
        // We need a running offset per color. Reuse body_mask low 32 bits as counters.
        // Actually, let's just use a small on-stack array (max 64 colors).
        int running[64];
        for (int k = 0; k < 64; k++) {
            running[k] = *wp::address(color_offsets, wid, k);
        }
        for (int c = 0; c < nc; c++) {
            int col = *wp::address(color_to_raw, wid, c);
            int dest = running[col];
            running[col]++;
            *wp::address(color_to_raw, wid, c) = dest;
        }
    }
    __syncthreads();

    // ── Phase 2: scatter + prepare (all threads) ──────────────────

    for (int c = tid; c < nc; c += blockDim.x) {
        int dest = *wp::address(color_to_raw, wid, c);

        // Scatter raw → color-ordered
        int a = *wp::address(raw_body_a, wid, c);
        int b = *wp::address(raw_body_b, wid, c);
        *wp::address(c_body_a, wid, dest) = a;
        *wp::address(c_body_b, wid, dest) = b;

        auto n = *wp::address(raw_normal, wid, c);
        *wp::address(c_normal, wid, dest) = n;

        auto ra = *wp::address(raw_r_a, wid, c);
        auto rb = *wp::address(raw_r_b, wid, c);
        *wp::address(c_r_a, wid, dest) = ra;
        *wp::address(c_r_b, wid, dest) = rb;

        *wp::address(c_base_sep, wid, dest) = *wp::address(raw_base_sep, wid, c);
        *wp::address(c_friction, wid, dest) = *wp::address(raw_friction, wid, c);
        *wp::address(c_restitution, wid, dest) = *wp::address(raw_restitution, wid, c);

        // Warm-started impulses
        *wp::address(c_ni, wid, dest) = *wp::address(raw_ni, wid, c);
        *wp::address(c_fi1, wid, dest) = *wp::address(raw_fi1, wid, c);
        *wp::address(c_fi2, wid, dest) = *wp::address(raw_fi2, wid, c);
        *wp::address(c_tni, wid, dest) = 0.0f;

        // Is static (one body is ground or kinematic)
        int is_static = 0;
        if (a < 0 || b < 0) is_static = 1;
        else {
            float ima = *wp::address(body_inv_mass, wid, a);
            float imb = *wp::address(body_inv_mass, wid, b);
            if (ima == 0.0f || imb == 0.0f) is_static = 1;
        }
        *wp::address(c_is_static, wid, dest) = is_static;

        // Remember raw→color mapping for impulse store-back
        // color_to_raw[c] already has dest, but we need the reverse:
        // we want: for each color-ordered slot, what raw index?
        // Actually we need: for each color-ordered slot, what raw index (for storing back).
        // We store dest → c (inverse mapping). But color_to_raw is written above.
        // Overwrite: color_to_raw[dest] = c (reverse mapping)
        // This is safe because dest is unique per c.
    }
    __syncthreads();

    // Fix up the reverse mapping: we need color_to_raw[dest] = raw_index
    // But we already wrote color_to_raw[c] = dest above.
    // To get reverse, we do a second pass.
    for (int c = tid; c < nc; c += blockDim.x) {
        // c is the raw index, color_to_raw[c] was the dest
        // We need to invert this. Use c_body_a as temp? No.
        // Actually, let's store the raw index in a separate temp.
        // For now, we won't need the reverse mapping in the solve kernel.
        // The store kernel can iterate raw contacts directly.
        // Skip reverse mapping here.
    }
    __syncthreads();
"""


@wp.func_native(snippet=_COLORING_PREPARE_SNIPPET)
def _coloring_prepare_native(
    # Raw contacts
    raw_body_a: wp.array2d(dtype=wp.int32),
    raw_body_b: wp.array2d(dtype=wp.int32),
    raw_normal: wp.array2d(dtype=wp.vec3),
    raw_r_a: wp.array2d(dtype=wp.vec3),
    raw_r_b: wp.array2d(dtype=wp.vec3),
    raw_base_sep: wp.array2d(dtype=float),
    raw_friction: wp.array2d(dtype=float),
    raw_restitution: wp.array2d(dtype=float),
    raw_ni: wp.array2d(dtype=float),
    raw_fi1: wp.array2d(dtype=float),
    raw_fi2: wp.array2d(dtype=float),
    # Colored outputs
    c_body_a: wp.array2d(dtype=wp.int32),
    c_body_b: wp.array2d(dtype=wp.int32),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_r_a: wp.array2d(dtype=wp.vec3),
    c_r_b: wp.array2d(dtype=wp.vec3),
    c_base_sep: wp.array2d(dtype=float),
    c_friction: wp.array2d(dtype=float),
    c_restitution: wp.array2d(dtype=float),
    c_ni: wp.array2d(dtype=float),
    c_fi1: wp.array2d(dtype=float),
    c_fi2: wp.array2d(dtype=float),
    c_tni: wp.array2d(dtype=float),
    c_is_static: wp.array2d(dtype=wp.int32),
    # Coloring arrays
    contact_count: wp.array[wp.int32],
    color_offsets: wp.array2d(dtype=wp.int32),
    color_body_mask: wp.array2d(dtype=wp.int64),
    color_to_raw: wp.array2d(dtype=wp.int32),
    bodies_per_world: wp.array[wp.int32],
    body_inv_mass: wp.array2d(dtype=float),
    max_colors: int,
    wid: int,
    tid: int,
):
    ...


@wp.kernel
def coloring_prepare_kernel(
    # Raw contacts
    raw_body_a: wp.array2d(dtype=wp.int32),
    raw_body_b: wp.array2d(dtype=wp.int32),
    raw_normal: wp.array2d(dtype=wp.vec3),
    raw_r_a: wp.array2d(dtype=wp.vec3),
    raw_r_b: wp.array2d(dtype=wp.vec3),
    raw_base_sep: wp.array2d(dtype=float),
    raw_friction: wp.array2d(dtype=float),
    raw_restitution: wp.array2d(dtype=float),
    raw_ni: wp.array2d(dtype=float),
    raw_fi1: wp.array2d(dtype=float),
    raw_fi2: wp.array2d(dtype=float),
    # Colored outputs
    c_body_a: wp.array2d(dtype=wp.int32),
    c_body_b: wp.array2d(dtype=wp.int32),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_r_a: wp.array2d(dtype=wp.vec3),
    c_r_b: wp.array2d(dtype=wp.vec3),
    c_base_sep: wp.array2d(dtype=float),
    c_friction: wp.array2d(dtype=float),
    c_restitution: wp.array2d(dtype=float),
    c_ni: wp.array2d(dtype=float),
    c_fi1: wp.array2d(dtype=float),
    c_fi2: wp.array2d(dtype=float),
    c_tni: wp.array2d(dtype=float),
    c_is_static: wp.array2d(dtype=wp.int32),
    # Coloring arrays
    contact_count: wp.array[wp.int32],
    color_offsets: wp.array2d(dtype=wp.int32),
    color_body_mask: wp.array2d(dtype=wp.int64),
    color_to_raw: wp.array2d(dtype=wp.int32),
    bodies_per_world: wp.array[wp.int32],
    body_inv_mass: wp.array2d(dtype=float),
    max_colors: int,
):
    """Tiled kernel: one thread block per world performs greedy coloring + scatter.

    Launched with ``wp.launch_tiled(dim=[num_worlds], block_dim=block_dim)``.
    """
    wid, tid = wp.tid()
    _coloring_prepare_native(
        raw_body_a, raw_body_b, raw_normal, raw_r_a, raw_r_b,
        raw_base_sep, raw_friction, raw_restitution,
        raw_ni, raw_fi1, raw_fi2,
        c_body_a, c_body_b, c_normal, c_r_a, c_r_b,
        c_base_sep, c_friction, c_restitution,
        c_ni, c_fi1, c_fi2, c_tni, c_is_static,
        contact_count, color_offsets, color_body_mask, color_to_raw,
        bodies_per_world, body_inv_mass, max_colors,
        wid, tid,
    )


# ═══════════════════════════════════════════════════════════════════════
# Contact effective-mass + tangent basis computation
# ═══════════════════════════════════════════════════════════════════════


@wp.kernel
def prepare_contact_masses_2d(
    c_body_a: wp.array2d(dtype=wp.int32),
    c_body_b: wp.array2d(dtype=wp.int32),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_r_a: wp.array2d(dtype=wp.vec3),
    c_r_b: wp.array2d(dtype=wp.vec3),
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    body_inv_mass: wp.array2d(dtype=float),
    body_inv_inertia: wp.array2d(dtype=wp.mat33),
    contact_count: wp.array[wp.int32],
    max_contacts: int,
    # Outputs
    c_tangent1: wp.array2d(dtype=wp.vec3),
    c_tangent2: wp.array2d(dtype=wp.vec3),
    c_normal_mass: wp.array2d(dtype=float),
    c_tangent1_mass: wp.array2d(dtype=float),
    c_tangent2_mass: wp.array2d(dtype=float),
    c_rel_vel_normal: wp.array2d(dtype=float),
):
    """Compute tangent basis, effective masses, and pre-solve relative velocity.

    Launched with ``dim = (num_worlds, max_contacts_per_world)``.
    """
    wid, ci = wp.tid()
    nc = contact_count[wid]
    if ci >= nc:
        return

    n = c_normal[wid, ci]
    basis = compute_tangent_basis(n)
    t1 = wp.vec3(basis[1, 0], basis[1, 1], basis[1, 2])
    t2 = wp.vec3(basis[2, 0], basis[2, 1], basis[2, 2])
    c_tangent1[wid, ci] = t1
    c_tangent2[wid, ci] = t2

    a = c_body_a[wid, ci]
    b = c_body_b[wid, ci]
    r_a = c_r_a[wid, ci]
    r_b = c_r_b[wid, ci]

    im_a = 0.0
    ii_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    im_b = 0.0
    ii_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    va = wp.vec3(0.0, 0.0, 0.0)
    wa = wp.vec3(0.0, 0.0, 0.0)
    vb = wp.vec3(0.0, 0.0, 0.0)
    wb = wp.vec3(0.0, 0.0, 0.0)

    if a >= 0:
        im_a = body_inv_mass[wid, a]
        ii_a = body_inv_inertia[wid, a]
        va = body_vel[wid, a]
        wa = body_ang_vel[wid, a]
    if b >= 0:
        im_b = body_inv_mass[wid, b]
        ii_b = body_inv_inertia[wid, b]
        vb = body_vel[wid, b]
        wb = body_ang_vel[wid, b]

    c_normal_mass[wid, ci] = compute_contact_effective_mass(im_a, ii_a, im_b, ii_b, r_a, r_b, n)
    c_tangent1_mass[wid, ci] = compute_contact_effective_mass(im_a, ii_a, im_b, ii_b, r_a, r_b, t1)
    c_tangent2_mass[wid, ci] = compute_contact_effective_mass(im_a, ii_a, im_b, ii_b, r_a, r_b, t2)

    # Pre-solve relative velocity for restitution
    v_rel = compute_relative_velocity(va, wa, r_a, vb, wb, r_b)
    c_rel_vel_normal[wid, ci] = wp.dot(v_rel, n)
