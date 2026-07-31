# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Oracle benchmark: one thread group per world, articulated contact solve on chip.

The production reduced contact phase publishes generalized rows through global
memory in separate phases (gather -> packed row build -> optional transpose ->
tile solve, repeated per 32-point page), and is bandwidth-bound at large world
counts. This oracle fuses row build, biased Gauss-Seidel, relax Gauss-Seidel,
and the generalized apply into ONE kernel per world (1, 2, or 4 warps; all
variants are timed): the contact Jacobian (path-compressed with on-chip DOF
indices), the dense articulated response rows, row velocities, effective
masses, and per-point contact data live in shared-memory tiles; the
generalized velocity delta lives in registers mirrored into a small shared
vector; and nothing row-shaped ever round-trips through global memory. In
particular the production build kernel's ``joint_work``/``body_response``
global scratch (hundreds of MB of traffic per pass at 8192 worlds) is replaced
by a per-lane DFS stack. Inputs are read once per world (joint factors
``joint_s`` / ``joint_u`` / ``joint_d_inv``, gathered contact points, body
velocities); outputs are the final generalized velocity delta, updated
joint/body velocities, and accumulated contact impulses.

Validation runs the real production contact phase (``block.solve`` with
``prepare=True`` biased pass and the relax pass, including its page loop,
transpose, and fused apply) and the oracle from identical state snapshots on
the live standing-G1 scene from ``bench_g1_shared_physics`` and compares the
applied generalized velocity deltas. Because a post-step snapshot is a
degenerate contact problem (warmstart lambdas relax to exactly zero), the
snapshot state is first perturbed in generalized coordinates through the
production apply kernel so the solve does real work.

Measured on RTX PRO 6000 Blackwell (2026-07): the oracle validates to ~1e-6
relative deviation and WINS at partial GPU occupancy (~1.6x ex-gather at 512
worlds, parity at 2048 worlds) but LOSES ~13% at 8192 worlds: ~40 KB of
on-chip rows per world caps residency at ~3 worlds/SM, and the fused kernel
serializes build + 3 sweeps within a world while the production phase kernels
overlap that work across worlds at high occupancy.

Honest caveats:

* The gather kernel (contact point enumeration + ``reduced_contact_prepare``)
  is NOT replaced: the oracle consumes the same gathered point arrays, while
  the production bracket re-gathers inside its pass. A hand-rolled two-page
  gather graph is timed separately so both a gather-inclusive and a
  gather-exclusive comparison can be read off.
* Buckets: <= 2 pages (64 contact points) per world, <= 64 articulation DOFs,
  bounded tree depth and root-to-body path length/DOFs. Deferred and fallback
  contacts are out of scope; the scene is asserted to not need them.
* The oracle solves pages in production order (page-sequential warmstart and
  sweeps) but computes row velocities as ``rv0 + J * delta`` carried on chip
  instead of re-gathering/refreshing them from applied body velocities. This
  is algebraically identical (rigid kinematics is linear in the velocities)
  but not bit-identical to the production refresh.
* The per-lane DFS body-response stack and path scratch are dynamically
  indexed, so the compiler places them in local memory (576-2304 B/thread);
  register/shared usage is reported so this is visible.
* The lambda read-modify-write runs on lane 0 with a collective broadcast of
  the impulse (multi-warp blocks race otherwise); explicit block barriers
  order the shared-tile element writes between warps, since Warp only fences
  full-tile operations.

Example::

    uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.experimental.bench_warp_world_contact_oracle
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _BLOCK_DIM,
    _CACHED_PAGE_COUNT,
    _POINTS_PER_PAGE,
    _advance_reduced_contact_page_cursor_kernel,
    _apply_generalized_contact_delta_kernel,
    _reset_reduced_contact_page_cursor_kernel,
    _sync_contact_block,
)
from newton._src.solvers.phoenx.benchmarks.bench_g1_shared_physics import (
    _collision_only_mjcf,
    _mujoco_free_qpos_to_newton,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_friction,
    contact_get_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_project_velocity_update_no_soft_pd,
)

_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_LANES = 32  # one warp per world


@functools.cache
def _make_oracle_kernel(
    rows_pad: int,
    nv_stride: int,
    page_bucket: int,
    depth_bucket: int,
    path_joint_bucket: int,
    path_dof_bucket: int,
    lanes: int,
):
    """Build the fused threads-per-world contact kernel for the given buckets."""
    rows_pad = ((rows_pad + lanes - 1) // lanes) * lanes
    waves = rows_pad // lanes
    resp_elems = rows_pad * nv_stride
    # The Jacobian row is nonzero only on the source-body path, so it is
    # stored compressed (path-local DOF order) with one padding slot for
    # masked off-path writes.
    jac_stride = path_dof_bucket + 1
    jac_elems = rows_pad * jac_stride
    dofs_per_lane = (nv_stride + lanes - 1) // lanes
    delta_slots = lanes * dofs_per_lane
    points_cap = page_bucket * _POINTS_PER_PAGE
    point_float_stride = 12  # n(3) t0(3) bias(3) mu(2) pad(1)
    point_float_elems = points_cap * point_float_stride
    _projvec = wp.types.vector(length=path_joint_bucket * 6, dtype=wp.float32)
    _deltavec = wp.types.vector(length=dofs_per_lane, dtype=wp.float32)
    module = (
        f"oracle_world_contact_r{rows_pad}_n{nv_stride}_g{page_bucket}"
        f"_d{depth_bucket}_j{path_joint_bucket}_p{path_dof_bucket}_l{lanes}"
    )

    @wp.kernel(enable_backward=False, module=module)
    def oracle_kernel(
        columns: ContactColumnContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        cc: ContactContainer,
        iterations_biased: wp.int32,
        iterations_relax: wp.int32,
        enabled: wp.array[wp.int32],
        total_point_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_velocity: wp.array2d[wp.float32],
        dfs_joint: wp.array[wp.int32],
        dfs_depth: wp.array[wp.int32],
        bodies: BodyContainer,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        generalized_delta_out: wp.array2d[wp.float32],
        body_delta: wp.array2d[wp.spatial_vector],
    ):
        articulation, lane = wp.tid()
        if enabled[articulation] == wp.int32(0):
            return
        data = bodies.reduced
        start = data.articulation_start[articulation]
        end = data.articulation_end[articulation]
        dof_start_articulation = data.joint_qd_start[start]
        active_point_count = total_point_count[articulation]
        row_count = wp.int32(3) * active_point_count

        # Whole per-world row state lives on chip.
        jac = wp.tile_zeros(shape=(jac_elems,), dtype=wp.float32, storage="shared")
        jac_dof = wp.tile_zeros(shape=(jac_elems,), dtype=wp.int32, storage="shared")
        jac_count = wp.tile_zeros(shape=(rows_pad,), dtype=wp.int32, storage="shared")
        resp = wp.tile_zeros(shape=(resp_elems,), dtype=wp.float32, storage="shared")
        rv = wp.tile_zeros(shape=(rows_pad,), dtype=wp.float32, storage="shared")
        eff = wp.tile_zeros(shape=(rows_pad,), dtype=wp.float32, storage="shared")
        point_data = wp.tile_zeros(shape=(point_float_elems,), dtype=wp.float32, storage="shared")
        point_id = wp.tile_zeros(shape=(points_cap,), dtype=wp.int32, storage="shared")

        # ---------------- build phase: lane = row ----------------
        for wave in range(waves):
            row = wp.int32(wave * lanes) + lane
            active = row < row_count
            point = row // wp.int32(3)
            axis = row - wp.int32(3) * point
            page = point // wp.int32(_POINTS_PER_PAGE)
            local_point = point - page * wp.int32(_POINTS_PER_PAGE)
            local_row = wp.int32(3) * local_point + axis
            packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + wp.min(
                page, wp.int32(_CACHED_PAGE_COUNT - 1)
            )
            # Inactive lanes run the same uniform sweep on a safe dummy body;
            # their writes land in padding rows that the solve never reads.
            source_body = data.joint_child[start]
            if active:
                candidate = row_body[packed_articulation, local_row]
                if candidate >= wp.int32(0):
                    source_body = candidate
            source_wrench = row_wrench[packed_articulation, local_row]
            path_start = data.body_path_start[source_body]
            path_end = data.body_path_start[source_body + wp.int32(1)]

            # Down-sweep along the source path (divergent length, registers only).
            proj = _projvec(wp.float32(0.0))
            propagated = source_wrench
            for reverse in range(path_end - path_start):
                path_index = path_end - wp.int32(1) - reverse
                joint = data.body_path_joint[path_index]
                dof_start = data.joint_qd_start[joint]
                dof_count = data.joint_qd_start[joint + wp.int32(1)] - dof_start
                local_path_joint = path_index - path_start
                projected = _vec6(0.0)
                reduced = _vec6(0.0)
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        dof = dof_start + wp.int32(dof_row)
                        projected[dof_row] = wp.dot(data.joint_s[dof], propagated)
                        proj[local_path_joint * wp.int32(6) + wp.int32(dof_row)] = projected[dof_row]
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        for dof_column in range(6):
                            if wp.int32(dof_column) < dof_count:
                                reduced[dof_row] += (
                                    data.joint_d_inv[dof_start + wp.int32(dof_row), dof_column] * projected[dof_column]
                                )
                        propagated -= data.joint_u[dof_start + wp.int32(dof_row)] * reduced[dof_row]

            # Forward DFS sweep. Trip count and per-joint DOF counts are
            # identical across lanes of one world, so control flow is uniform
            # and the shared-tile element writes are safe.
            stack = wp.matrix(shape=(depth_bucket, 6), dtype=wp.float32)
            path_cursor = path_start
            next_path_joint = wp.int32(-1)
            if path_cursor < path_end:
                next_path_joint = data.body_path_joint[path_cursor]
            inverse_mass = wp.float32(0.0)
            path_dof_cursor = wp.int32(0)
            for order in range(end - start):
                slot = start + order
                joint = dfs_joint[slot]
                depth = dfs_depth[slot]
                parent_delta = wp.spatial_vector()
                if depth > wp.int32(0):
                    for component in range(6):
                        parent_delta[component] = stack[depth - wp.int32(1), component]
                on_path = joint == next_path_joint
                local_path_joint = path_cursor - path_start
                dof_start = data.joint_qd_start[joint]
                dof_count = data.joint_qd_start[joint + wp.int32(1)] - dof_start
                rhs = _vec6(0.0)
                generalized = _vec6(0.0)
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        dof = dof_start + wp.int32(dof_row)
                        rhs[dof_row] = -wp.dot(data.joint_u[dof], parent_delta)
                        if on_path:
                            rhs[dof_row] += proj[local_path_joint * wp.int32(6) + wp.int32(dof_row)]
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        for dof_column in range(6):
                            if wp.int32(dof_column) < dof_count:
                                generalized[dof_row] += (
                                    data.joint_d_inv[dof_start + wp.int32(dof_row), dof_column] * rhs[dof_column]
                                )
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        dof = dof_start + wp.int32(dof_row)
                        local_dof = dof - dof_start_articulation
                        jac_value = wp.float32(0.0)
                        jac_slot = wp.int32(path_dof_bucket)  # masked padding slot
                        if on_path:
                            jac_value = wp.dot(data.joint_s[dof], source_wrench)
                            jac_slot = wp.min(path_dof_cursor + wp.int32(dof_row), wp.int32(path_dof_bucket))
                        response_value = generalized[dof_row]
                        jac[row * wp.int32(jac_stride) + jac_slot] = jac_value
                        jac_dof[row * wp.int32(jac_stride) + jac_slot] = local_dof
                        resp[row * wp.int32(nv_stride) + local_dof] = response_value
                        if on_path:
                            inverse_mass += jac_value * response_value
                        parent_delta += data.joint_s[dof] * response_value
                for component in range(6):
                    stack[depth, component] = parent_delta[component]
                if on_path:
                    path_dof_cursor += dof_count
                    path_cursor += wp.int32(1)
                    next_path_joint = wp.int32(-1)
                    if path_cursor < path_end:
                        next_path_joint = data.body_path_joint[path_cursor]

            rv_value = wp.float32(0.0)
            eff_value = wp.float32(0.0)
            if active:
                rv_value = row_velocity[packed_articulation, local_row]
                if inverse_mass > wp.float32(1.0e-12):
                    eff_value = wp.float32(1.0) / inverse_mass
                else:
                    # Production keeps the previous effective mass in this case.
                    contact = point_contact[packed_articulation, local_point]
                    if axis == wp.int32(0):
                        eff_value = cc_get_eff_n(cc, contact)
                    elif axis == wp.int32(1):
                        eff_value = cc_get_eff_t1(cc, contact)
                    else:
                        eff_value = cc_get_eff_t2(cc, contact)
            rv[row] = rv_value
            eff[row] = eff_value
            jac_count[row] = wp.min(path_dof_cursor, wp.int32(path_dof_bucket))

        # Make build-phase tile writes visible across the block's warps.
        _sync_contact_block()

        # ---------------- per-point contact cache ----------------
        # Cache the solve-constant per-point data so the Gauss-Seidel loop only
        # touches global memory for the lambda state (which production also
        # keeps in global memory).
        for point_base in range(0, points_cap, lanes):
            cache_point = wp.min(wp.int32(point_base) + lane, wp.int32(points_cap - 1))
            cache_page = cache_point // wp.int32(_POINTS_PER_PAGE)
            cache_local = cache_point - cache_page * wp.int32(_POINTS_PER_PAGE)
            cache_packed = articulation * wp.int32(_CACHED_PAGE_COUNT) + wp.min(
                cache_page, wp.int32(_CACHED_PAGE_COUNT - 1)
            )
            cache_contact = wp.int32(0)
            cache_n = wp.vec3()
            cache_t0 = wp.vec3()
            cache_bias = wp.float32(0.0)
            cache_bias_t0 = wp.float32(0.0)
            cache_bias_t1 = wp.float32(0.0)
            cache_mu_s = wp.float32(0.0)
            cache_mu_k = wp.float32(0.0)
            if cache_point < active_point_count:
                cache_contact = point_contact[cache_packed, cache_local]
                cache_column = point_column[cache_packed, cache_local]
                cache_n = normal[cache_packed, cache_local]
                cache_t0 = tangent0[cache_packed, cache_local]
                cache_bias = cc_get_bias(cc, cache_contact)
                cache_bias_t0 = cc_get_bias_t1(cc, cache_contact)
                cache_bias_t1 = cc_get_bias_t2(cc, cache_contact)
                cache_mu_s = contact_get_friction(columns, cache_column)
                cache_mu_k = contact_get_friction_dynamic(columns, cache_column)
            slot_base = cache_point * wp.int32(point_float_stride)
            point_data[slot_base] = cache_n[0]
            point_data[slot_base + wp.int32(1)] = cache_n[1]
            point_data[slot_base + wp.int32(2)] = cache_n[2]
            point_data[slot_base + wp.int32(3)] = cache_t0[0]
            point_data[slot_base + wp.int32(4)] = cache_t0[1]
            point_data[slot_base + wp.int32(5)] = cache_t0[2]
            point_data[slot_base + wp.int32(6)] = cache_bias
            point_data[slot_base + wp.int32(7)] = cache_bias_t0
            point_data[slot_base + wp.int32(8)] = cache_bias_t1
            point_data[slot_base + wp.int32(9)] = cache_mu_s
            point_data[slot_base + wp.int32(10)] = cache_mu_k
            point_id[cache_point] = cache_contact
        _sync_contact_block()

        # ---------------- solve phase: lanes = DOFs ----------------
        # The generalized delta lives in registers (owned DOFs per lane) and is
        # mirrored into a small shared vector so every lane can compute J*delta
        # by broadcast reads (shared-tile element reads do not synchronize).
        delta_lane = _deltavec(wp.float32(0.0))
        delta_sh = wp.tile_zeros(shape=(delta_slots,), dtype=wp.float32, storage="shared")
        _bias_rate, biased_mass_coeff, biased_impulse_coeff = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
        )
        for phase in range(2):
            use_bias = phase == 0
            iterations = iterations_biased
            mass_coeff = biased_mass_coeff
            impulse_coeff = biased_impulse_coeff
            if phase == 1:
                iterations = iterations_relax
                mass_coeff = wp.float32(1.0)
                impulse_coeff = wp.float32(0.0)
            for page in range(page_bucket):
                page_point_start = wp.int32(page * _POINTS_PER_PAGE)
                page_point_end = wp.min(active_point_count, page_point_start + wp.int32(_POINTS_PER_PAGE))
                page_points = page_point_end - page_point_start
                if page_points <= wp.int32(0):
                    continue
                if use_bias:
                    # Warm start this page from previous lambdas (as production).
                    for point_offset in range(page_points):
                        point = page_point_start + wp.int32(point_offset)
                        contact = point_id[point]
                        lambda0 = cc_get_normal_lambda(cc, contact)
                        lambda1 = cc_get_tangent1_lambda(cc, contact)
                        lambda2 = cc_get_tangent2_lambda(cc, contact)
                        row = wp.int32(3) * point
                        for part in range(dofs_per_lane):
                            dof_lane = lane + wp.int32(part * lanes)
                            index = wp.min(dof_lane, wp.int32(nv_stride - 1))
                            mask = wp.where(dof_lane < wp.int32(nv_stride), wp.float32(1.0), wp.float32(0.0))
                            delta_lane[part] += mask * (
                                lambda0 * resp[row * wp.int32(nv_stride) + index]
                                + lambda1 * resp[(row + wp.int32(1)) * wp.int32(nv_stride) + index]
                                + lambda2 * resp[(row + wp.int32(2)) * wp.int32(nv_stride) + index]
                            )
                    for part in range(dofs_per_lane):
                        delta_sh[lane + wp.int32(part * lanes)] = delta_lane[part]
                    _sync_contact_block()
                for iteration in range(iterations):
                    for point_offset in range(page_points):
                        local_point = wp.int32(point_offset)
                        if (iteration & wp.int32(1)) != wp.int32(0):
                            local_point = page_points - wp.int32(1) - wp.int32(point_offset)
                        point = page_point_start + local_point
                        row = wp.int32(3) * point
                        jv0 = rv[row]
                        jv1 = rv[row + wp.int32(1)]
                        jv2 = rv[row + wp.int32(2)]
                        # J is stored compressed along the source-body path with
                        # on-chip DOF indices; all inner-loop reads are shared
                        # memory broadcasts.
                        base = row * wp.int32(jac_stride)
                        for slot in range(jac_count[row]):
                            delta_value = delta_sh[jac_dof[base + slot]]
                            jv0 += jac[base + slot] * delta_value
                            jv1 += jac[base + wp.int32(jac_stride) + slot] * delta_value
                            jv2 += jac[base + wp.int32(2 * jac_stride) + slot] * delta_value
                        eff0 = eff[row]
                        eff1 = eff[row + wp.int32(1)]
                        eff2 = eff[row + wp.int32(2)]
                        delta0 = wp.float32(0.0)
                        delta1 = wp.float32(0.0)
                        delta2 = wp.float32(0.0)
                        contact = point_id[point]
                        slot_base = point * wp.int32(point_float_stride)
                        n = wp.vec3(
                            point_data[slot_base],
                            point_data[slot_base + wp.int32(1)],
                            point_data[slot_base + wp.int32(2)],
                        )
                        t0 = wp.vec3(
                            point_data[slot_base + wp.int32(3)],
                            point_data[slot_base + wp.int32(4)],
                            point_data[slot_base + wp.int32(5)],
                        )
                        t1 = wp.cross(n, t0)
                        bias = point_data[slot_base + wp.int32(6)]
                        speculative = bias > wp.float32(0.0)
                        # Lane 0 owns the lambda read-modify-write (multi-warp
                        # blocks would otherwise race on the cc state); the
                        # resulting impulse is broadcast collectively below.
                        if lane == wp.int32(0) and not (speculative and not use_bias):
                            bias_t0 = wp.float32(0.0)
                            bias_t1 = wp.float32(0.0)
                            if use_bias:
                                bias_t0 = point_data[slot_base + wp.int32(7)]
                                bias_t1 = point_data[slot_base + wp.int32(8)]
                            else:
                                bias = wp.float32(0.0)
                            row_mass_coeff = mass_coeff
                            row_impulse_coeff = impulse_coeff
                            mu_static = point_data[slot_base + wp.int32(9)]
                            mu_dynamic = point_data[slot_base + wp.int32(10)]
                            if speculative:
                                row_mass_coeff = wp.float32(1.0)
                                row_impulse_coeff = wp.float32(0.0)
                                if bias > idt * wp.float32(0.002):
                                    mu_static = wp.float32(0.0)
                                    mu_dynamic = wp.float32(0.0)
                            impulse = contact_project_velocity_update_no_soft_pd(
                                cc,
                                contact,
                                n,
                                t0,
                                t1,
                                jv0,
                                jv1,
                                jv2,
                                eff0,
                                eff1,
                                eff2,
                                bias,
                                bias_t0,
                                bias_t1,
                                mu_static,
                                mu_dynamic,
                                row_mass_coeff,
                                row_impulse_coeff,
                                sor_boost,
                                wp.float32(0.0),
                                wp.float32(0.0),
                                wp.float32(0.0),
                            )
                            delta0 = wp.dot(impulse, n)
                            delta1 = wp.dot(impulse, t0)
                            delta2 = wp.dot(impulse, t1)
                        broadcast = wp.tile_from_thread(
                            shape=1, value=wp.vec3(delta0, delta1, delta2), thread_idx=0, storage="shared"
                        )
                        deltas = wp.tile_extract(broadcast, 0)
                        delta0 = deltas[0]
                        delta1 = deltas[1]
                        delta2 = deltas[2]
                        for part in range(dofs_per_lane):
                            dof_lane = lane + wp.int32(part * lanes)
                            index = wp.min(dof_lane, wp.int32(nv_stride - 1))
                            mask = wp.where(dof_lane < wp.int32(nv_stride), wp.float32(1.0), wp.float32(0.0))
                            delta_lane[part] += mask * (
                                delta0 * resp[row * wp.int32(nv_stride) + index]
                                + delta1 * resp[(row + wp.int32(1)) * wp.int32(nv_stride) + index]
                                + delta2 * resp[(row + wp.int32(2)) * wp.int32(nv_stride) + index]
                            )
                            delta_sh[dof_lane] = delta_lane[part]
                        _sync_contact_block()

        # ---------------- write back + fused apply ----------------
        for part in range(dofs_per_lane):
            dof_lane = lane + wp.int32(part * lanes)
            if dof_lane < wp.int32(nv_stride):
                generalized_delta_out[articulation, dof_lane] = delta_lane[part]
        _sync_contact_block()
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                local_joint = joint - start
                parent = data.joint_parent[joint]
                delta = wp.spatial_vector()
                if parent >= wp.int32(0):
                    delta = body_delta[articulation, data.body_joint[parent] - start]
                for dof in range(data.joint_qd_start[joint], data.joint_qd_start[joint + wp.int32(1)]):
                    dof_delta = generalized_delta_out[articulation, dof - dof_start_articulation]
                    data.joint_qd[dof] += dof_delta
                    delta += data.joint_s[dof] * dof_delta
                body_delta[articulation, local_joint] = delta
                child = data.joint_child[joint]
                slot = child + wp.int32(1)
                delta_omega = wp.spatial_bottom(delta)
                bodies.angular_velocity[slot] += delta_omega
                local_com_position = wp.transform_get_translation(data.body_q_com[child])
                bodies.velocity[slot] += wp.spatial_top(delta) + wp.cross(delta_omega, local_com_position)
                index += wp.int32(lanes)
            _sync_contact_block()

    return oracle_kernel


def _compute_dfs_order(reduced) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per-articulation DFS joint order + depth (generic, from topology)."""
    articulation_start = np.asarray(reduced.articulation_start.numpy())
    articulation_end = np.asarray(reduced.articulation_end.numpy())
    joint_parent = reduced.joint_parent.numpy()  # joint -> parent body
    body_joint = reduced.body_joint.numpy()  # body -> joint
    joint_count = joint_parent.shape[0]
    dfs_joint = np.zeros(joint_count, dtype=np.int32)
    dfs_depth = np.zeros(joint_count, dtype=np.int32)
    max_depth = 0

    parent_joint = np.where(joint_parent >= 0, body_joint[np.maximum(joint_parent, 0)], -1)
    articulation_count = min(articulation_start.shape[0], articulation_end.shape[0])
    for articulation in range(articulation_count):
        start = int(articulation_start[articulation])
        end = int(articulation_end[articulation])
        children: dict[int, list[int]] = {}
        roots: list[int] = []
        for joint in range(start, end):
            parent = int(parent_joint[joint])
            if start <= parent < end:
                children.setdefault(parent, []).append(joint)
            else:
                roots.append(joint)
        cursor = start
        stack = [(joint, 0) for joint in reversed(roots)]
        while stack:
            joint, depth = stack.pop()
            dfs_joint[cursor] = joint
            dfs_depth[cursor] = depth
            cursor += 1
            max_depth = max(max_depth, depth)
            for child in reversed(children.get(joint, ())):
                stack.append((child, depth + 1))
        if cursor != end:
            raise RuntimeError(f"articulation {articulation} joint tree is not connected")

    body_path_start = reduced.body_path_start.numpy()
    max_path_joints = int(np.max(np.diff(body_path_start))) if body_path_start.shape[0] > 1 else 1
    body_path_joint = reduced.body_path_joint.numpy()
    joint_qd_start = reduced.joint_qd_start.numpy()
    joint_dofs = np.diff(joint_qd_start)
    max_path_dofs = 1
    for body in range(body_path_start.shape[0] - 1):
        path = body_path_joint[body_path_start[body] : body_path_start[body + 1]]
        max_path_dofs = max(max_path_dofs, int(joint_dofs[path].sum()))
    return dfs_joint, dfs_depth, max_depth, max_path_joints, max_path_dofs


def _kernel_resources(kernel, device: wp.context.Device, block_dim: int) -> dict:
    """Best-effort register/shared-memory query through the driver API."""
    try:
        module_exec = kernel.module.load(device, block_dim)
        hooks = module_exec.get_kernel_hooks(kernel)
        cuda = ctypes.CDLL("libcuda.so")
        result = {}
        for name, attribute in (
            ("registers_per_thread", 4),
            ("static_shared_bytes", 1),
            ("local_bytes_per_thread", 3),
        ):
            value = ctypes.c_int(0)
            status = cuda.cuFuncGetAttribute(
                ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_void_p(hooks.forward)
            )
            if status != 0:
                status = cuda.cuKernelGetAttribute(
                    ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_void_p(hooks.forward), ctypes.c_int(0)
                )
            result[name] = int(value.value) if status == 0 else None
        return result
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": str(exc)}


def _build_scene(world_count: int, device: wp.context.Device):
    """Standing G1 scene, built exactly like bench_g1_shared_physics."""
    asset_dir = Path(newton.__file__).parent / "examples" / "assets" / "mjwarp_benchmarks" / "unitree_g1"
    replay = np.load(asset_dir / "shuffle_dance.npz")
    qpos = _mujoco_free_qpos_to_newton(replay["qpos"][0])
    ctrl = np.asarray(replay["ctrl"][0], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="newton_g1_oracle_") as temp_dir:
        scene_path = _collision_only_mjcf(asset_dir, Path(temp_dir))
        template = newton.ModelBuilder(up_axis=newton.Axis.Z)
        template.add_mjcf(
            str(scene_path),
            ignore_names=("floor",),
            parse_visuals=False,
            parse_meshes=False,
            enable_self_collisions=False,
        )
    for index, value in enumerate(qpos):
        template.joint_q[index] = float(value)
        template.joint_target_q[index] = float(value)
    for channel, value in enumerate(ctrl):
        template.joint_target_q[7 + channel] = float(value)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count)
    builder.default_shape_cfg.mu = 0.6
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverPhoenX(
        model,
        articulation_mode="reduced",
        contact_friction_model="point",
        substeps=1,
        solver_iterations=2,
        velocity_iterations=1,
    )
    state = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = model.contacts()
    return model, solver, state, control, contacts


def _time_graph_batches(graph, replays: int, batches: int, device) -> list[float]:
    times_us = []
    wp.capture_launch(graph)
    wp.synchronize_device(device)
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(replays):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        times_us.append((time.perf_counter() - start) * 1.0e6 / replays)
    return times_us


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument("--replays", type=int, default=300)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--step-replays", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("the one-warp-per-world contact oracle requires CUDA")

    model, solver, state, control, contacts = _build_scene(args.world_count, device)

    def step() -> None:
        model.collide(state, contacts)
        state.clear_forces()
        solver.step(state, state, control, contacts, args.dt)

    for _ in range(args.settle_steps):
        step()
    with wp.ScopedCapture(device=device) as step_capture:
        step()
    for _ in range(10):
        wp.capture_launch(step_capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(args.step_replays):
        wp.capture_launch(step_capture.graph)
    wp.synchronize_device(device)
    step_us = (time.perf_counter() - start) * 1.0e6 / args.step_replays

    world = solver.world
    reduced_system = solver._reduced_articulation
    if reduced_system is None:
        raise RuntimeError("reduced articulation was not initialized")
    block = reduced_system.contact_block_system
    if block.packed_jacobian is None or block.packed_response is None:
        raise RuntimeError("reduced contact buffers were not initialized")
    bodies = world.bodies
    reduced = bodies.reduced
    cc = world._contact_container
    columns = world._contact_cols
    contact_views = world._active_contact_views()
    idt = wp.float32(1.0 / args.dt)
    sor_boost = float(world.sor_boost)
    iterations_biased = 2
    iterations_relax = 1

    # ---- scene assertions: the oracle bucket must cover the live state ----
    max_pages = int(block.max_page_count.numpy()[0])
    if max_pages > _CACHED_PAGE_COUNT:
        raise RuntimeError("oracle covers cached pages only (<=64 contact points per world)")
    if int(block.deferred_active.numpy()[0]) != 0:
        raise RuntimeError("deferred contacts are active; oracle scene assumption violated")
    if block.fallback_count is not None and int(block.fallback_count.numpy()[0]) != 0:
        raise RuntimeError("fallback contacts are active; oracle scene assumption violated")
    transpose_active = int(block.transpose_active.numpy()[0])
    nv_stride = int(block.contact_dof_width)
    if nv_stride > 2 * _LANES:
        raise RuntimeError(f"articulation DOF width {nv_stride} exceeds oracle bucket {2 * _LANES}")
    point_counts = block.total_point_count.numpy()
    max_points = int(np.max(point_counts))
    rows_pad = ((3 * max_points + _LANES - 1) // _LANES) * _LANES
    page_bucket = max_pages

    dfs_joint_np, dfs_depth_np, max_tree_depth, max_path_joints, max_path_dofs = _compute_dfs_order(reduced)
    depth_bucket = next(bucket for bucket in (4, 8, 12, 16, 24, 32) if bucket >= max_tree_depth + 1)
    path_joint_bucket = next(bucket for bucket in (4, 8, 12, 16) if bucket >= max_path_joints)
    path_dof_bucket = next(bucket for bucket in (8, 12, 16, 24, 32) if bucket >= max_path_dofs)
    dfs_joint = wp.array(dfs_joint_np, dtype=wp.int32, device=device)
    dfs_depth = wp.array(dfs_depth_np, dtype=wp.int32, device=device)

    articulation_count = block.articulation_count
    oracle_variants = {
        f"oracle_tpw{lanes}": (
            _make_oracle_kernel(
                rows_pad, nv_stride, page_bucket, depth_bucket, path_joint_bucket, path_dof_bucket, lanes
            ),
            lanes,
        )
        for lanes in (32, 64, 128)
    }
    oracle_delta = wp.zeros((articulation_count, nv_stride), dtype=wp.float32, device=device)
    oracle_body_delta = wp.zeros_like(block.generalized_body_delta)

    # ---- consistent two-page gather from one body state (see caveats) ----
    def launch_gather_pages() -> None:
        wp.launch(
            _reset_reduced_contact_page_cursor_kernel,
            dim=1,
            inputs=[block.max_page_count],
            outputs=[block.page_cursor, block.page_index],
            device=device,
        )
        for _page in range(max_pages):
            wp.launch(
                block.gather_kernel,
                dim=(articulation_count, block.gather_tile_width),
                block_dim=_BLOCK_DIM,
                inputs=[
                    block.schedule_section_end,
                    block.schedule_columns,
                    columns,
                    bodies,
                    idt,
                    wp.bool(True),
                    cc,
                    contact_views,
                    block.enabled,
                    block.total_point_count,
                    block.page_index,
                ],
                outputs=[
                    block.point_count,
                    block.point_contact,
                    block.point_column,
                    block.point0,
                    block.point1,
                    block.normal,
                    block.tangent0,
                    block.row_body,
                    block.row_wrench,
                    block.row_velocity,
                ],
                device=device,
            )
            wp.launch(
                _advance_reduced_contact_page_cursor_kernel,
                dim=1,
                inputs=[block.page_cursor, block.page_index],
                device=device,
            )
        wp.launch(
            _reset_reduced_contact_page_cursor_kernel,
            dim=1,
            inputs=[block.max_page_count],
            outputs=[block.page_cursor, block.page_index],
            device=device,
        )

    # A post-step snapshot is a degenerate contact problem (warmstart lambdas
    # relax to exactly zero and the phase is a no-op). Perturb the state in
    # generalized space through the production apply kernel so joint_qd and
    # body velocities stay consistent and the contact solve does real work.
    rng = np.random.default_rng(11)
    perturbation = (args.perturbation * rng.standard_normal((articulation_count, nv_stride))).astype(np.float32)
    block.generalized_delta.assign(perturbation)
    block.generalized_body_delta.zero_()
    wp.launch(
        _apply_generalized_contact_delta_kernel,
        dim=articulation_count,
        inputs=[bodies, block.enabled, block.generalized_delta],
        outputs=[block.generalized_body_delta],
        device=device,
    )

    launch_gather_pages()
    wp.synchronize_device(device)

    # ---- snapshot every array either bracket mutates ----
    snapshot_sources = {
        "velocity": bodies.velocity,
        "angular_velocity": bodies.angular_velocity,
        "joint_qd": reduced.joint_qd,
        "cc_impulses": cc.impulses,
        "cc_prev_impulses": cc.prev_impulses,
        "cc_lambdas": cc.lambdas,
        "cc_prev_lambdas": cc.prev_lambdas,
        "cc_derived": cc.derived,
        "packed_jacobian": block.packed_jacobian,
        "packed_response": block.packed_response,
        "packed_previous_row_body": block.packed_previous_row_body,
        "point_count": block.point_count,
        "point_contact": block.point_contact,
        "point_column": block.point_column,
        "point0": block.point0,
        "point1": block.point1,
        "normal": block.normal,
        "tangent0": block.tangent0,
        "row_body": block.row_body,
        "row_wrench": block.row_wrench,
        "row_velocity": block.row_velocity,
        "page_index": block.page_index,
        "page_cursor": block.page_cursor,
        "generalized_delta": block.generalized_delta,
        "generalized_body_delta": block.generalized_body_delta,
    }
    snapshots = {name: wp.clone(array) for name, array in snapshot_sources.items()}

    def restore() -> None:
        for name, array in snapshot_sources.items():
            wp.copy(array, snapshots[name])
        oracle_delta.zero_()
        oracle_body_delta.zero_()

    # ---- production bracket: the real block.solve biased + relax passes ----
    def launch_production_biased() -> None:
        block.solve(
            columns,
            bodies,
            idt,
            sor_boost,
            cc,
            contact_views,
            iterations_biased,
            use_bias=True,
            prepare=True,
        )

    def launch_production_relax() -> None:
        block.solve(
            columns,
            bodies,
            idt,
            sor_boost,
            cc,
            contact_views,
            iterations_relax,
            use_bias=False,
            prepare=False,
        )

    def launch_production() -> None:
        launch_production_biased()
        launch_production_relax()

    def launch_oracle(kernel, lanes: int, biased: int, relax: int) -> None:
        wp.launch_tiled(
            kernel,
            dim=[articulation_count],
            block_dim=lanes,
            inputs=[
                columns,
                idt,
                wp.float32(sor_boost),
                cc,
                wp.int32(biased),
                wp.int32(relax),
                block.enabled,
                block.total_point_count,
                block.point_contact,
                block.point_column,
                block.normal,
                block.tangent0,
                block.row_body,
                block.row_wrench,
                block.row_velocity,
                dfs_joint,
                dfs_depth,
                bodies,
                wp.int32(block.max_depth),
                block.articulation_depth_start,
                block.articulation_depth_joint,
            ],
            outputs=[oracle_delta, oracle_body_delta],
            device=device,
        )

    def make_oracle_launch(kernel, lanes: int, biased: int, relax: int):
        return lambda: launch_oracle(kernel, lanes, biased, relax)

    # ---- capture graphs (block.solve uses conditional nodes internally) ----
    launches = {}
    for variant, (kernel, lanes) in oracle_variants.items():
        launches[variant] = make_oracle_launch(kernel, lanes, iterations_biased, iterations_relax)
        # Diagnostic: row build + warmstart + apply only (no GS sweeps).
        launches[variant + "_build_only"] = make_oracle_launch(kernel, lanes, 0, 0)
    launches["production_contact_phase"] = launch_production
    launches["production_biased"] = launch_production_biased
    launches["production_relax"] = launch_production_relax
    launches["production_gather"] = launch_gather_pages

    graphs = {}
    for name, launch in launches.items():
        restore()
        with wp.ScopedCapture(device=device) as capture:
            launch()
        graphs[name] = capture.graph

    # ---- validation from identical snapshots ----
    qd_before = snapshots["joint_qd"].numpy()
    v_before = snapshots["velocity"].numpy()

    restore()
    wp.capture_launch(graphs["production_contact_phase"])
    wp.synchronize_device(device)
    production_qd_delta = reduced.joint_qd.numpy() - qd_before
    production_v_delta = bodies.velocity.numpy() - v_before
    production_lambdas = cc.lambdas.numpy().copy()

    qd_scale = max(1.0e-6, float(np.max(np.abs(production_qd_delta))))
    v_scale = max(1.0e-6, float(np.max(np.abs(production_v_delta))))
    lambda_scale = max(1.0e-6, float(np.max(np.abs(production_lambdas))))
    validation = {}
    for variant in oracle_variants:
        restore()
        wp.capture_launch(graphs[variant])
        wp.synchronize_device(device)
        oracle_qd_delta = reduced.joint_qd.numpy() - qd_before
        oracle_v_delta = bodies.velocity.numpy() - v_before
        oracle_lambdas = cc.lambdas.numpy().copy()
        if not np.isfinite(oracle_qd_delta).all():
            raise RuntimeError(f"{variant} produced non-finite generalized velocity deltas")
        validation[variant] = {
            "qd_delta_max_abs_deviation": float(np.max(np.abs(oracle_qd_delta - production_qd_delta))),
            "qd_delta_max_rel_deviation": float(np.max(np.abs(oracle_qd_delta - production_qd_delta))) / qd_scale,
            "body_velocity_delta_max_rel_deviation": float(np.max(np.abs(oracle_v_delta - production_v_delta)))
            / v_scale,
            "lambda_max_rel_deviation": float(np.max(np.abs(oracle_lambdas - production_lambdas))) / lambda_scale,
            "passed_rtol_1e3": bool(
                np.allclose(oracle_qd_delta, production_qd_delta, rtol=1.0e-3, atol=1.0e-3 * qd_scale)
            ),
        }

    # ---- graph-captured timing, reversed-order brackets ----
    order = list(graphs)
    times: dict[str, list[float]] = {name: [] for name in order}
    for direction in (order, list(reversed(order)), order, list(reversed(order))):
        for name in direction:
            restore()
            batch = _time_graph_batches(graphs[name], args.replays, args.batches, device)
            times[name].extend(batch)
    median_us = {name: float(np.median(values)) for name, values in times.items()}

    best_variant = min(oracle_variants, key=lambda name: median_us[name])
    oracle_us = median_us[best_variant]
    production_us = median_us["production_contact_phase"]
    gather_us = median_us["production_gather"]
    production_ex_gather_us = max(0.0, production_us - gather_us)
    implied_step_us = step_us - production_ex_gather_us + oracle_us
    resources = {
        variant: _kernel_resources(kernel, device, lanes) for variant, (kernel, lanes) in oracle_variants.items()
    }

    payload = {
        "schema": "phoenx_warp_world_contact_oracle_v2",
        "device": device.name,
        "world_count": args.world_count,
        "dt": args.dt,
        "iterations_biased": iterations_biased,
        "iterations_relax": iterations_relax,
        "contact_points_per_world_max": max_points,
        "contact_points_per_world_mean": float(np.mean(point_counts)),
        "pages": max_pages,
        "rows_pad": rows_pad,
        "nv_stride": nv_stride,
        "depth_bucket": depth_bucket,
        "path_joint_bucket": path_joint_bucket,
        "path_dof_bucket": path_dof_bucket,
        "transpose_active": transpose_active,
        "production_qd_delta_max_abs": qd_scale,
        "validation": validation,
        "oracle_best_variant": best_variant,
        "oracle_us": oracle_us,
        "oracle_variant_us": {variant: median_us[variant] for variant in oracle_variants},
        "oracle_build_only_us": {variant: median_us[variant + "_build_only"] for variant in oracle_variants},
        "production_contact_phase_us": production_us,
        "production_contact_phase_ex_gather_us": production_ex_gather_us,
        "production_biased_us": median_us["production_biased"],
        "production_relax_us": median_us["production_relax"],
        "production_gather_us": gather_us,
        "contact_phase_speedup_vs_full": production_us / oracle_us if oracle_us > 0.0 else None,
        "contact_phase_speedup_ex_gather": production_ex_gather_us / oracle_us if oracle_us > 0.0 else None,
        "full_step_us": step_us,
        "implied_step_us_with_oracle": implied_step_us,
        "implied_end_to_end_speedup": step_us / implied_step_us if implied_step_us > 0.0 else None,
        "oracle_kernel_resources": resources,
        "oracle_smem_bytes_analytic": (
            rows_pad * nv_stride
            + 2 * rows_pad * (path_dof_bucket + 1)
            + 3 * rows_pad
            + max_pages * _POINTS_PER_PAGE * 13
        )
        * 4,
        "timing_batches_us": dict(times),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
