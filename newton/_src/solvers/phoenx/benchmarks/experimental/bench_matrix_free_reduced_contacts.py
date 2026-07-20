# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free reduced-coordinate contact solve: no per-row articulated responses.

The production reduced contact phase materializes ``packed_jacobian`` AND
``packed_response`` (each rows x nv floats) and runs tile Gauss-Seidel over
them; the on-chip oracle (``bench_warp_world_contact_oracle``) proved that
locality alone loses at 8192 worlds because O(rows x nv) response construction
is the floor of both designs. This benchmark removes the response rows
entirely with a projected-Jacobi (proximal) iteration; per sweep it touches
only

1. ``w = J v``: each contact row dots its path-compressed Jacobian
   (~path_dofs entries, NOT nv) with the on-chip generalized velocity delta,
2. an impulse update with the production EXACT Coulomb cone projection
   (``contact_project_velocity_update_no_soft_pd``: disc projection of the
   tangent impulse to the ``mu * lambda_n`` circle) using per-row effective
   masses ``1 / (J M^-1 J^T)_rr`` computed once at build along the body path,
3. ``dv = M^-1 J^T dlambda`` through the EXISTING linear-time factorization
   (``joint_s`` / ``joint_u`` / ``joint_d_inv``): a J^T scatter over the
   path-sparse entries followed by one O(joints) articulated backward/forward
   tree pass per sweep (the ``_solve_articulated_system_kernel`` recurrence),
   instead of accumulating rows x nv dense response rows.

Per iteration the data touched is rows x path_dofs for J plus O(nv) tree
passes -- strictly less than rows x nv -- and the shared-memory footprint per
world drops by the whole response tile (rows_pad x nv floats), roughly
doubling residency against the oracle. Jacobi is naturally order-free, so the
solve is deterministic without coloring; under-relaxation (``omega`` scaling
the SOR boost, the same lever as PhoenX's "Jacobi simple" flavor) restores
convergence for coupled same-body contacts.

How this differs from prior in-tree art:

* "Prototype matrix-free contact Jacobians" (030be01c, rejected cross-scene
  at 26456a8c) removed only the global Jacobian MATRIX by re-deriving J from
  path masks, but still materialized and streamed the dense per-row RESPONSE
  matrix every sweep. Here the response matrix never exists: each sweep pays
  one O(depth) tree pass instead of rows x nv response reads.
* "Adaptive response bases" and "unit-wrench compressed contacts" both kept
  per-row responses in compressed forms. This design has no per-row response
  storage at all, at the cost of changing the iteration from Gauss-Seidel to
  under-relaxed Jacobi (quality compared explicitly below).

Validation and quality methodology (all from identical state snapshots on the
live perturbed standing-G1 scene, reusing the oracle's snapshot machinery):

* Machinery check: the 0-sweep variant applies exactly the warm-start
  ``M^-1 J^T lambda``; production with ``iterations=0`` applies the same
  quantity through its response rows. Their generalized velocity deltas must
  match to ~1e-5 relative.
* Quality: production with (16 biased, 8 relax) iterations is the converged
  reference. Production (2, 1) and every Jacobi config are scored on
  (a) relative L2 / Linf distance of the generalized velocity delta to the
  reference, (b) residual penetration velocity along contact normals
  (non-speculative points), (c) exact-cone violations, all computed from the
  production ``packed_jacobian`` of the same snapshot.
* Timing: graph-captured 300-replay reversed brackets against the production
  contact phase (biased + relax) and the standalone gather, exactly as the
  oracle benchmark.

Measured on RTX PRO 6000 Blackwell (2026-07, standing G1, 33 contact points
per world, nv=35): the machinery is exact (warm start matches production to
2.4e-7 relative) and the iteration REACHES production quality -- 12 Jacobi
sweeps at omega 0.5, or the same 2+1 sweep counts as production with strided
4-point groups -- but it LOSES on time at every scale: matched quality costs
6874 us vs 2490 us production at 8192 worlds (0.36x) and 1806 us vs 863 us at
2048 worlds (0.48x). The specific bottleneck is the per-world serial M^-1
tree pass: one application costs ~478 us at 8192 worlds (about one full
production dense-response GS sweep over ALL rows), because a single warp per
world walks a ~30-joint dependency chain through shared memory while
production overlaps rows x nv streaming across the whole GPU at full
occupancy. Quality parity needs >= 12 tree passes vs production's 3 dense
sweeps, and the build+warm-start+apply floor alone (1222 us) is already half
of the entire production phase. Even a hypothetical 4x-faster depth-parallel
tree pass would only reach parity, so this direction is rejected: O(rows x
nv) response STREAMING is cheap at high occupancy; it is the per-sweep
LATENCY chain of the factorized solve that is expensive.

Honest caveats:

* The gather kernel is NOT replaced (same as the oracle); gather-inclusive
  and -exclusive comparisons can both be read off.
* The J^T scatter and the tree passes run redundantly on all lanes of the
  world's warp (identical values, benign same-value shared writes); this is
  simple and race-free but serializes O(points x path_dofs + joints) work
  per sweep within a world.
* Buckets: <= 2 gathered pages per world, <= contact_dof_width DOFs, bounded
  path length; deferred/fallback contacts are asserted inactive.
* Row velocities are carried as ``rv0 + J delta`` on chip (algebraically
  identical to the production refresh, not bit-identical).

Example::

    uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.experimental.bench_matrix_free_reduced_contacts
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _BLOCK_DIM,
    _CACHED_PAGE_COUNT,
    _MAX_ROWS,
    _POINTS_PER_PAGE,
    _advance_reduced_contact_page_cursor_kernel,
    _apply_generalized_contact_delta_kernel,
    _reset_reduced_contact_page_cursor_kernel,
    _sync_contact_block,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_warp_world_contact_oracle import (
    _build_scene,
    _compute_dfs_order,
    _kernel_resources,
    _time_graph_batches,
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
_LANES = 32

# Timing/breakdown modes for the sweep loop.
_MODE_FULL = 0  # full solve
_MODE_NO_PROJECTION = 1  # skip cone projection (dlambda = 0): J*v + scatter + tree cost only
_MODE_NO_TREE = 2  # skip J^T scatter + tree pass: projection cost only (physics invalid)


@functools.cache
def _make_matrix_free_kernel(
    rows_pad: int,
    nv_stride: int,
    page_bucket: int,
    joint_bucket: int,
    path_joint_bucket: int,
    path_dof_bucket: int,
    lanes: int,
):
    """Fused one-block-per-world matrix-free contact kernel for the given buckets."""
    rows_pad = ((rows_pad + lanes - 1) // lanes) * lanes
    waves = rows_pad // lanes
    jac_stride = path_dof_bucket + 1  # one padding slot for masked off-path writes
    jac_elems = rows_pad * jac_stride
    points_cap = page_bucket * _POINTS_PER_PAGE
    point_waves = (points_cap + lanes - 1) // lanes
    point_float_stride = 12  # n(3) t0(3) bias(3) mu(2) pad(1)
    point_float_elems = points_cap * point_float_stride
    nv_waves = (nv_stride + lanes - 1) // lanes
    joint_waves = (joint_bucket + lanes - 1) // lanes
    dl_slots = rows_pad + 3  # 3 padding slots for inactive point lanes
    factor_elems = nv_stride * 6
    work_elems = joint_bucket * 6
    qdstart_slots = joint_bucket + 1
    _projvec = wp.types.vector(length=path_joint_bucket * 6, dtype=wp.float32)
    _jacvec = wp.types.vector(length=jac_stride, dtype=wp.float32)
    _jacdofvec = wp.types.vector(length=jac_stride, dtype=wp.int32)
    module = (
        f"matrix_free_contact_r{rows_pad}_n{nv_stride}_g{page_bucket}"
        f"_k{joint_bucket}_j{path_joint_bucket}_p{path_dof_bucket}_l{lanes}"
    )

    @wp.kernel(enable_backward=False, module=module)
    def matrix_free_kernel(
        columns: ContactColumnContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        omega: wp.float32,
        cc: ContactContainer,
        iterations_biased: wp.int32,
        iterations_relax: wp.int32,
        group_size: wp.int32,
        strided: wp.int32,
        mode: wp.int32,
        enabled: wp.array[wp.int32],
        total_point_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_velocity: wp.array2d[wp.float32],
        joint_parent_local: wp.array[wp.int32],
        bodies: BodyContainer,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        debug: wp.int32,
        debug_jac: wp.array3d[wp.float32],
        debug_jac_dof: wp.array3d[wp.int32],
        debug_jac_count: wp.array2d[wp.int32],
        debug_eff: wp.array2d[wp.float32],
        debug_f: wp.array2d[wp.float32],
        generalized_delta_out: wp.array2d[wp.float32],
        body_delta: wp.array2d[wp.spatial_vector],
    ):
        articulation, lane = wp.tid()
        if enabled[articulation] == wp.int32(0):
            return
        data = bodies.reduced
        start = data.articulation_start[articulation]
        end = data.articulation_end[articulation]
        joint_count = end - start
        dof_start_articulation = data.joint_qd_start[start]
        nv = data.joint_qd_start[end] - dof_start_articulation
        active_point_count = total_point_count[articulation]
        row_count = wp.int32(3) * active_point_count

        # ---------------- on-chip state (NO response tile) ----------------
        jac = wp.tile_zeros(shape=(jac_elems,), dtype=wp.float32, storage="shared")
        jac_dof = wp.tile_zeros(shape=(jac_elems,), dtype=wp.int32, storage="shared")
        jac_count = wp.tile_zeros(shape=(rows_pad,), dtype=wp.int32, storage="shared")
        rv = wp.tile_zeros(shape=(rows_pad,), dtype=wp.float32, storage="shared")
        eff = wp.tile_zeros(shape=(rows_pad,), dtype=wp.float32, storage="shared")
        point_data = wp.tile_zeros(shape=(point_float_elems,), dtype=wp.float32, storage="shared")
        point_id = wp.tile_zeros(shape=(points_cap,), dtype=wp.int32, storage="shared")
        # Cached articulated factors (local DOF indexing): joint_s / joint_u rows
        # and the d_inv rows, so the per-sweep tree pass never touches global.
        s_sh = wp.tile_zeros(shape=(factor_elems,), dtype=wp.float32, storage="shared")
        u_sh = wp.tile_zeros(shape=(factor_elems,), dtype=wp.float32, storage="shared")
        dinv_sh = wp.tile_zeros(shape=(factor_elems,), dtype=wp.float32, storage="shared")
        parent_sh = wp.tile_zeros(shape=(joint_bucket,), dtype=wp.int32, storage="shared")
        qdstart_sh = wp.tile_zeros(shape=(qdstart_slots,), dtype=wp.int32, storage="shared")
        # Per-sweep scratch: generalized delta, generalized force, per-row
        # impulse deltas, tree-pass body work and joint work.
        delta_sh = wp.tile_zeros(shape=(nv_stride,), dtype=wp.float32, storage="shared")
        f_sh = wp.tile_zeros(shape=(nv_stride,), dtype=wp.float32, storage="shared")
        dl_sh = wp.tile_zeros(shape=(dl_slots,), dtype=wp.float32, storage="shared")
        work_sh = wp.tile_zeros(shape=(work_elems,), dtype=wp.float32, storage="shared")
        jw_sh = wp.tile_zeros(shape=(nv_stride,), dtype=wp.float32, storage="shared")

        # ---------------- build phase A: factor + topology caches ----------------
        for wave in range(nv_waves):
            dof_local = wp.int32(wave * lanes) + lane
            source_dof = dof_start_articulation + wp.min(dof_local, nv - wp.int32(1))
            s_value = data.joint_s[source_dof]
            u_value = data.joint_u[source_dof]
            slot = wp.min(dof_local, wp.int32(nv_stride - 1))
            for component in range(6):
                s_sh[slot * wp.int32(6) + wp.int32(component)] = s_value[component]
                u_sh[slot * wp.int32(6) + wp.int32(component)] = u_value[component]
                dinv_sh[slot * wp.int32(6) + wp.int32(component)] = data.joint_d_inv[source_dof, component]
        for wave in range(joint_waves):
            joint_local = wp.int32(wave * lanes) + lane
            source_joint = start + wp.min(joint_local, joint_count - wp.int32(1))
            parent_value = joint_parent_local[source_joint]
            if joint_local >= joint_count:
                parent_value = wp.int32(-1)
            slot = wp.min(joint_local, wp.int32(joint_bucket - 1))
            parent_sh[slot] = parent_value
            qdstart_sh[slot] = data.joint_qd_start[source_joint] - dof_start_articulation
        qdstart_sh[wp.min(joint_count, wp.int32(joint_bucket))] = nv
        _sync_contact_block()

        # ---------------- build phase B: path-sparse J + effective masses ----------------
        # Per row (lane = row): backward sweep along the source-body path
        # (registers only), then a forward sweep along the SAME path that yields
        # both the compressed Jacobian entries and the exact effective mass
        # (J M^-1 J^T)_rr. Off-path response entries are never computed: they do
        # not contribute to the row's effective mass and the per-sweep dv comes
        # from the tree pass instead.
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
            source_body = data.joint_child[start]
            if active:
                candidate = row_body[packed_articulation, local_row]
                if candidate >= wp.int32(0):
                    source_body = candidate
            source_wrench = row_wrench[packed_articulation, local_row]
            path_start = data.body_path_start[source_body]
            path_end = data.body_path_start[source_body + wp.int32(1)]
            path_length = path_end - path_start

            # Backward sweep (leaf -> root) along the path: registers only.
            proj = _projvec(wp.float32(0.0))
            propagated = source_wrench
            for reverse in range(path_length):
                path_index = path_end - wp.int32(1) - reverse
                joint = data.body_path_joint[path_index]
                dof_local = data.joint_qd_start[joint] - dof_start_articulation
                dof_count = data.joint_qd_start[joint + wp.int32(1)] - data.joint_qd_start[joint]
                local_path_joint = path_index - path_start
                projected = _vec6(0.0)
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                        value = wp.float32(0.0)
                        for component in range(6):
                            value += s_sh[base + wp.int32(component)] * propagated[component]
                        projected[dof_row] = value
                        proj[local_path_joint * wp.int32(6) + wp.int32(dof_row)] = value
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        reduced_value = wp.float32(0.0)
                        for dof_column in range(6):
                            if wp.int32(dof_column) < dof_count:
                                reduced_value += (
                                    dinv_sh[(dof_local + wp.int32(dof_row)) * wp.int32(6) + wp.int32(dof_column)]
                                    * projected[dof_column]
                                )
                        base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                        for component in range(6):
                            propagated[component] -= u_sh[base + wp.int32(component)] * reduced_value

            # Forward sweep (root -> leaf) along the path: J entries + eff mass.
            jac_values = _jacvec(wp.float32(0.0))
            jac_dofs = _jacdofvec(wp.int32(0))
            parent_delta = wp.spatial_vector()
            inverse_mass = wp.float32(0.0)
            path_dof_cursor = wp.int32(0)
            for path_offset in range(path_length):
                path_index = path_start + path_offset
                joint = data.body_path_joint[path_index]
                dof_local = data.joint_qd_start[joint] - dof_start_articulation
                dof_count = data.joint_qd_start[joint + wp.int32(1)] - data.joint_qd_start[joint]
                rhs = _vec6(0.0)
                generalized = _vec6(0.0)
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                        value = proj[wp.int32(path_offset * 6) + wp.int32(dof_row)]
                        for component in range(6):
                            value -= u_sh[base + wp.int32(component)] * parent_delta[component]
                        rhs[dof_row] = value
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        for dof_column in range(6):
                            if wp.int32(dof_column) < dof_count:
                                generalized[dof_row] += (
                                    dinv_sh[(dof_local + wp.int32(dof_row)) * wp.int32(6) + wp.int32(dof_column)]
                                    * rhs[dof_column]
                                )
                for dof_row in range(6):
                    if wp.int32(dof_row) < dof_count:
                        base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                        jac_value = wp.float32(0.0)
                        for component in range(6):
                            jac_value += s_sh[base + wp.int32(component)] * source_wrench[component]
                        slot = wp.min(path_dof_cursor + wp.int32(dof_row), wp.int32(path_dof_bucket))
                        jac_values[slot] = jac_value
                        jac_dofs[slot] = dof_local + wp.int32(dof_row)
                        inverse_mass += jac_value * generalized[dof_row]
                        for component in range(6):
                            parent_delta[component] += s_sh[base + wp.int32(component)] * generalized[dof_row]
                path_dof_cursor += dof_count

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
            row_base = row * wp.int32(jac_stride)
            for slot in range(jac_stride):
                jac[row_base + wp.int32(slot)] = jac_values[slot]
                jac_dof[row_base + wp.int32(slot)] = jac_dofs[slot]
        _sync_contact_block()

        # ---------------- per-point contact cache (as the oracle) ----------------
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

        if debug != wp.int32(0):
            for wave in range(waves):
                row = wp.int32(wave * lanes) + lane
                debug_jac_count[articulation, row] = jac_count[row]
                debug_eff[articulation, row] = eff[row]
                for slot in range(jac_stride):
                    debug_jac[articulation, row, slot] = jac[row * wp.int32(jac_stride) + wp.int32(slot)]
                    debug_jac_dof[articulation, row, slot] = jac_dof[row * wp.int32(jac_stride) + wp.int32(slot)]

        # ---------------- warm start: dl = lambda, then one tree pass ----------------
        for wave in range(point_waves):
            point = wp.int32(wave * lanes) + lane
            active = point < active_point_count
            lambda0 = wp.float32(0.0)
            lambda1 = wp.float32(0.0)
            lambda2 = wp.float32(0.0)
            if active:
                contact = point_id[point]
                lambda0 = cc_get_normal_lambda(cc, contact)
                lambda1 = cc_get_tangent1_lambda(cc, contact)
                lambda2 = cc_get_tangent2_lambda(cc, contact)
            slot = wp.int32(rows_pad)
            if active:
                slot = wp.int32(3) * point
            dl_sh[slot] = lambda0
            dl_sh[slot + wp.int32(1)] = lambda1
            dl_sh[slot + wp.int32(2)] = lambda2
        _sync_contact_block()

        # J^T scatter + M^-1 tree pass, executed redundantly on ALL lanes
        # (identical values -> benign same-value shared writes, no barriers in
        # the serial section). Total sweeps: 1 warm start + biased + relax.
        _bias_rate, biased_mass_coeff, biased_impulse_coeff = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
        )
        total_sweeps = wp.int32(1) + iterations_biased + iterations_relax
        sweep = wp.int32(0)
        while sweep < total_sweeps:
            # The warm-start sweep applies all lambdas in one group; regular
            # sweeps iterate fixed groups of points: Jacobi within a group
            # (one tree pass per group), Gauss-Seidel across groups.
            # group_size == point count -> pure Jacobi; == 1 -> exact GS.
            step_size = group_size
            if sweep == wp.int32(0) or step_size <= wp.int32(0):
                step_size = wp.max(active_point_count, wp.int32(1))
            group_count = (active_point_count + step_size - wp.int32(1)) // step_size
            group_index = wp.int32(0)
            while group_index < group_count:
                # Contiguous groups: [start, end). Strided groups: points with
                # point % group_count == group_index (mixes contact clusters).
                group_start = group_index * step_size
                group_end = wp.min(group_start + step_size, active_point_count)
                if sweep > wp.int32(0):
                    use_bias = sweep <= iterations_biased
                    mass_coeff = wp.float32(1.0)
                    impulse_coeff = wp.float32(0.0)
                    if use_bias:
                        mass_coeff = biased_mass_coeff
                        impulse_coeff = biased_impulse_coeff
                    for wave in range(point_waves):
                        point = wp.int32(wave * lanes) + lane
                        if strided != wp.int32(0):
                            active = point < active_point_count and (point % group_count) == group_index
                        else:
                            active = point >= group_start and point < group_end
                        delta0 = wp.float32(0.0)
                        delta1 = wp.float32(0.0)
                        delta2 = wp.float32(0.0)
                        if active and mode != wp.int32(_MODE_NO_PROJECTION):
                            row = wp.int32(3) * point
                            # (1) w = J v: path-compressed row dot the shared delta.
                            jv0 = rv[row]
                            jv1 = rv[row + wp.int32(1)]
                            jv2 = rv[row + wp.int32(2)]
                            base = row * wp.int32(jac_stride)
                            for slot in range(jac_count[row]):
                                delta_value = delta_sh[jac_dof[base + slot]]
                                jv0 += jac[base + slot] * delta_value
                                jv1 += jac[base + wp.int32(jac_stride) + slot] * delta_value
                                jv2 += jac[base + wp.int32(2 * jac_stride) + slot] * delta_value
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
                            if not (speculative and not use_bias):
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
                                    point_id[point],
                                    n,
                                    t0,
                                    t1,
                                    jv0,
                                    jv1,
                                    jv2,
                                    eff[row],
                                    eff[row + wp.int32(1)],
                                    eff[row + wp.int32(2)],
                                    bias,
                                    bias_t0,
                                    bias_t1,
                                    mu_static,
                                    mu_dynamic,
                                    row_mass_coeff,
                                    row_impulse_coeff,
                                    sor_boost * omega,
                                    wp.float32(0.0),
                                    wp.float32(0.0),
                                    wp.float32(0.0),
                                )
                                delta0 = wp.dot(impulse, n)
                                delta1 = wp.dot(impulse, t0)
                                delta2 = wp.dot(impulse, t1)
                        slot = wp.int32(rows_pad)
                        if active:
                            slot = wp.int32(3) * point
                        dl_sh[slot] = delta0
                        dl_sh[slot + wp.int32(1)] = delta1
                        dl_sh[slot + wp.int32(2)] = delta2
                    _sync_contact_block()

                # ---- (3) dv = M^-1 J^T dl: path-sparse scatter + O(joints) tree pass ----
                if mode != wp.int32(_MODE_NO_TREE):
                    for wave in range(nv_waves):
                        dof_local = wp.int32(wave * lanes) + lane
                        slot = wp.min(dof_local, wp.int32(nv_stride - 1))
                        f_sh[slot] = wp.float32(0.0)
                    for wave in range(joint_waves):
                        joint_local = wp.int32(wave * lanes) + lane
                        slot = wp.min(joint_local, wp.int32(joint_bucket - 1))
                        for component in range(6):
                            work_sh[slot * wp.int32(6) + wp.int32(component)] = wp.float32(0.0)
                    _sync_contact_block()
                    # J^T dl scatter, redundant-serial (all lanes identical values).
                    group_points = group_end - group_start
                    if strided != wp.int32(0):
                        group_points = (active_point_count - group_index + group_count - wp.int32(1)) // group_count
                    for member in range(group_points):
                        point = group_start + wp.int32(member)
                        if strided != wp.int32(0):
                            point = group_index + wp.int32(member) * group_count
                        row = wp.int32(3) * point
                        count = jac_count[row]
                        base = row * wp.int32(jac_stride)
                        dl0 = dl_sh[row]
                        dl1 = dl_sh[row + wp.int32(1)]
                        dl2 = dl_sh[row + wp.int32(2)]
                        for slot in range(count):
                            dof = jac_dof[base + slot]
                            f_value = f_sh[dof] + (
                                jac[base + slot] * dl0
                                + jac[base + wp.int32(jac_stride) + slot] * dl1
                                + jac[base + wp.int32(2 * jac_stride) + slot] * dl2
                            )
                            f_sh[dof] = f_value
                    if debug != wp.int32(0) and sweep == wp.int32(0):
                        for wave in range(nv_waves):
                            dof_local = wp.int32(wave * lanes) + lane
                            if dof_local < wp.int32(nv_stride):
                                debug_f[articulation, dof_local] = f_sh[wp.min(dof_local, wp.int32(nv_stride - 1))]
                    # Backward tree pass (leaf -> root, storage order is topological).
                    for reverse in range(joint_count):
                        joint_local = joint_count - wp.int32(1) - reverse
                        dof_local = qdstart_sh[joint_local]
                        dof_count = qdstart_sh[joint_local + wp.int32(1)] - dof_local
                        p = _vec6(0.0)
                        for component in range(6):
                            p[component] = work_sh[joint_local * wp.int32(6) + wp.int32(component)]
                        reduced_force = _vec6(0.0)
                        for dof_row in range(6):
                            if wp.int32(dof_row) < dof_count:
                                base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                                value = f_sh[dof_local + wp.int32(dof_row)]
                                for component in range(6):
                                    value -= s_sh[base + wp.int32(component)] * p[component]
                                reduced_force[dof_row] = value
                                jw_sh[dof_local + wp.int32(dof_row)] = value
                        d_inv_u = _vec6(0.0)
                        for dof_row in range(6):
                            if wp.int32(dof_row) < dof_count:
                                for dof_column in range(6):
                                    if wp.int32(dof_column) < dof_count:
                                        d_inv_u[dof_row] += (
                                            dinv_sh[
                                                (dof_local + wp.int32(dof_row)) * wp.int32(6) + wp.int32(dof_column)
                                            ]
                                            * reduced_force[dof_column]
                                        )
                        for dof_row in range(6):
                            if wp.int32(dof_row) < dof_count:
                                base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                                for component in range(6):
                                    p[component] += u_sh[base + wp.int32(component)] * d_inv_u[dof_row]
                        parent = parent_sh[joint_local]
                        if parent >= wp.int32(0):
                            for component in range(6):
                                work_value = work_sh[parent * wp.int32(6) + wp.int32(component)] + p[component]
                                work_sh[parent * wp.int32(6) + wp.int32(component)] = work_value
                    # Forward tree pass (root -> leaf); work_sh is reused for the
                    # body acceleration (parents are processed before children).
                    for joint_local_index in range(joint_count):
                        joint_local = wp.int32(joint_local_index)
                        dof_local = qdstart_sh[joint_local]
                        dof_count = qdstart_sh[joint_local + wp.int32(1)] - dof_local
                        parent = parent_sh[joint_local]
                        parent_acceleration = _vec6(0.0)
                        if parent >= wp.int32(0):
                            for component in range(6):
                                parent_acceleration[component] = work_sh[parent * wp.int32(6) + wp.int32(component)]
                        rhs = _vec6(0.0)
                        qdd = _vec6(0.0)
                        for dof_row in range(6):
                            if wp.int32(dof_row) < dof_count:
                                base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                                value = jw_sh[dof_local + wp.int32(dof_row)]
                                for component in range(6):
                                    value -= u_sh[base + wp.int32(component)] * parent_acceleration[component]
                                rhs[dof_row] = value
                        for dof_row in range(6):
                            if wp.int32(dof_row) < dof_count:
                                for dof_column in range(6):
                                    if wp.int32(dof_column) < dof_count:
                                        qdd[dof_row] += (
                                            dinv_sh[
                                                (dof_local + wp.int32(dof_row)) * wp.int32(6) + wp.int32(dof_column)
                                            ]
                                            * rhs[dof_column]
                                        )
                        acceleration = parent_acceleration
                        for dof_row in range(6):
                            if wp.int32(dof_row) < dof_count:
                                base = (dof_local + wp.int32(dof_row)) * wp.int32(6)
                                delta_value = delta_sh[dof_local + wp.int32(dof_row)] + qdd[dof_row]
                                delta_sh[dof_local + wp.int32(dof_row)] = delta_value
                                for component in range(6):
                                    acceleration[component] += s_sh[base + wp.int32(component)] * qdd[dof_row]
                        for component in range(6):
                            work_sh[joint_local * wp.int32(6) + wp.int32(component)] = acceleration[component]
                    _sync_contact_block()
                group_index += wp.int32(1)
            sweep += wp.int32(1)

        # ---------------- write back + fused apply (as the oracle) ----------------
        for wave in range(nv_waves):
            dof_local = wp.int32(wave * lanes) + lane
            if dof_local < wp.int32(nv_stride):
                generalized_delta_out[articulation, dof_local] = delta_sh[wp.min(dof_local, wp.int32(nv_stride - 1))]
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

    return matrix_free_kernel


@wp.kernel(enable_backward=False)
def _dump_point_state_kernel(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    enabled: wp.array[wp.int32],
    total_point_count: wp.array[wp.int32],
    point_contact: wp.array2d[wp.int32],
    point_column: wp.array2d[wp.int32],
    lambda_out: wp.array2d[wp.vec3],
    bias_out: wp.array2d[wp.float32],
    mu_out: wp.array2d[wp.vec2],
    active_out: wp.array2d[wp.int32],
):
    """Dump per-point lambdas / bias / friction for host-side residual metrics."""
    articulation, point = wp.tid()
    if enabled[articulation] == wp.int32(0) or point >= total_point_count[articulation]:
        active_out[articulation, point] = wp.int32(0)
        return
    page = point // wp.int32(_POINTS_PER_PAGE)
    local_point = point - page * wp.int32(_POINTS_PER_PAGE)
    packed = articulation * wp.int32(_CACHED_PAGE_COUNT) + wp.min(page, wp.int32(_CACHED_PAGE_COUNT - 1))
    contact = point_contact[packed, local_point]
    column = point_column[packed, local_point]
    lambda_out[articulation, point] = wp.vec3(
        cc_get_normal_lambda(cc, contact),
        cc_get_tangent1_lambda(cc, contact),
        cc_get_tangent2_lambda(cc, contact),
    )
    bias_out[articulation, point] = cc_get_bias(cc, contact)
    mu_out[articulation, point] = wp.vec2(
        contact_get_friction(columns, column),
        contact_get_friction_dynamic(columns, column),
    )
    active_out[articulation, point] = wp.int32(1)


def _joint_parent_local(reduced) -> np.ndarray:
    """Local (within-articulation) parent joint index per joint, -1 for roots."""
    joint_parent = reduced.joint_parent.numpy()
    body_joint = reduced.body_joint.numpy()
    articulation_start = np.asarray(reduced.articulation_start.numpy())
    articulation_end = np.asarray(reduced.articulation_end.numpy())
    parent_joint = np.where(joint_parent >= 0, body_joint[np.maximum(joint_parent, 0)], -1).astype(np.int32)
    result = np.full(joint_parent.shape[0], -1, dtype=np.int32)
    count = min(articulation_start.shape[0], articulation_end.shape[0])
    for articulation in range(count):
        start = int(articulation_start[articulation])
        end = int(articulation_end[articulation])
        local = parent_joint[start:end].copy()
        inside = (local >= start) & (local < end)
        result[start:end] = np.where(inside, local - start, -1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Matrix-free reduced contact solve benchmark")
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument("--replays", type=int, default=300)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--step-replays", type=int, default=100)
    parser.add_argument("--omegas", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[1, 4, 8, 0],
        help="points per Jacobi group (GS across groups); 0 = all points (pure Jacobi)",
    )
    parser.add_argument(
        "--sweep-configs",
        type=str,
        nargs="+",
        default=["2:1", "4:2", "8:4"],
        help="biased:relax Jacobi sweep configurations for the convergence scan",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("the matrix-free contact benchmark requires CUDA")

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
    start_time = time.perf_counter()
    for _ in range(args.step_replays):
        wp.capture_launch(step_capture.graph)
    wp.synchronize_device(device)
    step_us = (time.perf_counter() - start_time) * 1.0e6 / args.step_replays

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
    reference_biased = 16
    reference_relax = 8

    # ---- scene assertions: buckets must cover the live state ----
    max_pages = int(block.max_page_count.numpy()[0])
    if max_pages > _CACHED_PAGE_COUNT:
        raise RuntimeError("benchmark covers cached pages only (<=64 contact points per world)")
    if int(block.deferred_active.numpy()[0]) != 0:
        raise RuntimeError("deferred contacts are active; scene assumption violated")
    if block.fallback_count is not None and int(block.fallback_count.numpy()[0]) != 0:
        raise RuntimeError("fallback contacts are active; scene assumption violated")
    nv_stride = int(block.contact_dof_width)
    point_counts = block.total_point_count.numpy()
    max_points = int(np.max(point_counts))
    rows_pad = ((3 * max_points + _LANES - 1) // _LANES) * _LANES
    page_bucket = max_pages

    _dfs_joint, _dfs_depth, _max_tree_depth, max_path_joints, max_path_dofs = _compute_dfs_order(reduced)
    path_joint_bucket = next(bucket for bucket in (4, 8, 12, 16) if bucket >= max_path_joints)
    path_dof_bucket = next(bucket for bucket in (8, 12, 16, 24, 32) if bucket >= max_path_dofs)
    articulation_start_np = np.asarray(reduced.articulation_start.numpy())
    articulation_end_np = np.asarray(reduced.articulation_end.numpy())
    articulation_count = block.articulation_count
    joint_counts = articulation_end_np[:articulation_count] - articulation_start_np[:articulation_count]
    joint_bucket = next(bucket for bucket in (8, 16, 24, 32, 48, 64) if bucket >= int(np.max(joint_counts)))
    joint_parent_local = wp.array(_joint_parent_local(reduced), dtype=wp.int32, device=device)

    # DOF bookkeeping for host-side residual metrics (uniform replicated worlds).
    joint_qd_start_np = reduced.joint_qd_start.numpy()
    dof_starts = joint_qd_start_np[articulation_start_np[:articulation_count]]
    dof_ends = joint_qd_start_np[articulation_end_np[:articulation_count]]
    nv_per_articulation = dof_ends - dof_starts
    if int(np.min(nv_per_articulation)) != int(np.max(nv_per_articulation)):
        raise RuntimeError("residual metrics assume uniform articulations")
    nv = int(nv_per_articulation[0])

    kernel = _make_matrix_free_kernel(
        rows_pad, nv_stride, page_bucket, joint_bucket, path_joint_bucket, path_dof_bucket, _LANES
    )
    matrix_free_delta = wp.zeros((articulation_count, nv_stride), dtype=wp.float32, device=device)
    matrix_free_body_delta = wp.zeros_like(block.generalized_body_delta)

    points_cap = page_bucket * _POINTS_PER_PAGE
    lambda_dump = wp.zeros((articulation_count, points_cap), dtype=wp.vec3, device=device)
    bias_dump = wp.zeros((articulation_count, points_cap), dtype=wp.float32, device=device)
    mu_dump = wp.zeros((articulation_count, points_cap), dtype=wp.vec2, device=device)
    active_dump = wp.zeros((articulation_count, points_cap), dtype=wp.int32, device=device)

    # ---- perturb the post-step snapshot so the solve does real work ----
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

    # ---- consistent gather from the perturbed body state (as the oracle) ----
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

    launch_gather_pages()
    wp.synchronize_device(device)

    # ---- production bracket launchers ----
    def launch_production(biased: int, relax: int) -> None:
        block.solve(
            columns,
            bodies,
            idt,
            sor_boost,
            cc,
            contact_views,
            biased,
            use_bias=True,
            prepare=True,
        )
        block.solve(
            columns,
            bodies,
            idt,
            sor_boost,
            cc,
            contact_views,
            relax,
            use_bias=False,
            prepare=False,
        )

    # Snapshot BEFORE any solve; jac_ref for the residual metrics is produced
    # by one production run from the snapshot (its prepare pass re-gathers the
    # same rows from the identical restored state) and the state is restored.
    snapshot_sources = {
        "velocity": bodies.velocity,
        "angular_velocity": bodies.angular_velocity,
        "joint_qd": reduced.joint_qd,
        "cc_impulses": cc.impulses,
        "cc_prev_impulses": cc.prev_impulses,
        "cc_lambdas": cc.lambdas,
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
        matrix_free_delta.zero_()
        matrix_free_body_delta.zero_()

    # Exact contact Jacobian of this snapshot for host-side residual metrics.
    launch_production(iterations_biased, iterations_relax)
    wp.synchronize_device(device)
    jac_ref = wp.clone(block.packed_jacobian)
    restore()

    debug_jac = wp.zeros((articulation_count, rows_pad, path_dof_bucket + 1), dtype=wp.float32, device=device)
    debug_jac_dof = wp.zeros((articulation_count, rows_pad, path_dof_bucket + 1), dtype=wp.int32, device=device)
    debug_jac_count = wp.zeros((articulation_count, rows_pad), dtype=wp.int32, device=device)
    debug_eff = wp.zeros((articulation_count, rows_pad), dtype=wp.float32, device=device)
    debug_f = wp.zeros((articulation_count, nv_stride), dtype=wp.float32, device=device)

    def launch_matrix_free(
        biased: int, relax: int, group: int, strided: int, omega: float, mode: int, debug: int = 0
    ) -> None:
        wp.launch_tiled(
            kernel,
            dim=[articulation_count],
            block_dim=_LANES,
            inputs=[
                columns,
                idt,
                wp.float32(sor_boost),
                wp.float32(omega),
                cc,
                wp.int32(biased),
                wp.int32(relax),
                wp.int32(group),
                wp.int32(strided),
                wp.int32(mode),
                block.enabled,
                block.total_point_count,
                block.point_contact,
                block.point_column,
                block.normal,
                block.tangent0,
                block.row_body,
                block.row_wrench,
                block.row_velocity,
                joint_parent_local,
                bodies,
                wp.int32(block.max_depth),
                block.articulation_depth_start,
                block.articulation_depth_joint,
                wp.int32(debug),
                debug_jac,
                debug_jac_dof,
                debug_jac_count,
                debug_eff,
                debug_f,
            ],
            outputs=[matrix_free_delta, matrix_free_body_delta],
            device=device,
        )

    # ---- graphs ----
    sweep_configs = []
    for token in args.sweep_configs:
        biased_text, relax_text = token.split(":")
        sweep_configs.append((int(biased_text), int(relax_text)))

    graphs: dict[str, wp.Graph] = {}

    def capture(name: str, launch) -> None:
        restore()
        with wp.ScopedCapture(device=device) as capture_scope:
            launch()
        graphs[name] = capture_scope.graph

    capture("production_contact_phase", lambda: launch_production(iterations_biased, iterations_relax))
    capture("production_reference", lambda: launch_production(reference_biased, reference_relax))
    capture("production_warmstart_only", lambda: launch_production(0, 0))
    capture("production_gather", launch_gather_pages)
    capture("matrix_free_warmstart_only", lambda: launch_matrix_free(0, 0, 0, 0, 1.0, _MODE_FULL))
    for biased, relax in sweep_configs:
        for group in args.groups:
            strided_options = (0, 1) if 1 < group < max_points else (0,)
            for strided in strided_options:
                for omega in args.omegas:
                    tag = "s" if strided else "g"
                    capture(
                        f"matrix_free_b{biased}_r{relax}_{tag}{group}_w{omega:g}",
                        lambda b=biased, r=relax, g=group, st=strided, w=omega: launch_matrix_free(
                            b, r, g, st, w, _MODE_FULL
                        ),
                    )

    # ---- run + measure quality for every configuration ----
    qd_before = snapshots["joint_qd"].numpy()
    rv0 = snapshots["row_velocity"].numpy().reshape(articulation_count, _CACHED_PAGE_COUNT, _MAX_ROWS)
    jacobian = jac_ref.numpy().reshape(articulation_count, _CACHED_PAGE_COUNT, _MAX_ROWS, nv_stride)[..., :nv]

    def dump_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wp.launch(
            _dump_point_state_kernel,
            dim=(articulation_count, points_cap),
            inputs=[
                columns,
                cc,
                block.enabled,
                block.total_point_count,
                block.point_contact,
                block.point_column,
            ],
            outputs=[lambda_dump, bias_dump, mu_dump, active_dump],
            device=device,
        )
        return lambda_dump.numpy(), bias_dump.numpy(), mu_dump.numpy(), active_dump.numpy()

    def run_and_measure(name: str) -> dict:
        restore()
        wp.capture_launch(graphs[name])
        wp.synchronize_device(device)
        qd_delta = reduced.joint_qd.numpy() - qd_before
        if not np.isfinite(qd_delta).all():
            return {"finite": False}
        lambdas, biases, mus, actives = dump_points()
        delta_gen = qd_delta.reshape(articulation_count, nv)

        # Row velocities after the solve: w = rv0 + J * delta.
        w = rv0 + np.einsum("apri,ai->apr", jacobian, delta_gen)
        w_points = w.reshape(articulation_count, _CACHED_PAGE_COUNT * _POINTS_PER_PAGE, 3)[:, :points_cap]
        # Map (page, local) row blocks back to point order (identical layout).
        active = actives.astype(bool)
        non_speculative = active & (biases <= 0.0)
        w_n = w_points[..., 0]
        penetration = np.maximum(0.0, -w_n)[non_speculative]
        lambda_n = lambdas[..., 0]
        lambda_t = np.linalg.norm(lambdas[..., 1:3], axis=-1)
        cone_violation = np.maximum(0.0, lambda_t - mus[..., 0] * lambda_n)[active]
        complementarity = np.abs(lambda_n * w_n)[non_speculative]
        return {
            "finite": True,
            "penetration_velocity_max": float(np.max(penetration)) if penetration.size else 0.0,
            "penetration_velocity_mean": float(np.mean(penetration)) if penetration.size else 0.0,
            "penetration_velocity_rms": float(np.sqrt(np.mean(penetration**2))) if penetration.size else 0.0,
            "cone_violation_max": float(np.max(cone_violation)) if cone_violation.size else 0.0,
            "cone_violation_mean": float(np.mean(cone_violation)) if cone_violation.size else 0.0,
            "complementarity_mean": float(np.mean(complementarity)) if complementarity.size else 0.0,
            "_qd_delta": qd_delta,
        }

    quality: dict[str, dict] = {}
    raw_deltas: dict[str, np.ndarray] = {}
    for name in graphs:
        if name == "production_gather":
            continue
        entry = run_and_measure(name)
        if entry.get("finite", False):
            raw_deltas[name] = entry.pop("_qd_delta")
        quality[name] = entry

    reference_delta = raw_deltas["production_reference"]
    reference_norm = max(1.0e-9, float(np.linalg.norm(reference_delta)))
    reference_scale = max(1.0e-9, float(np.max(np.abs(reference_delta))))
    for name, qd_delta in raw_deltas.items():
        entry = quality[name]
        entry["distance_to_reference_rel_l2"] = float(np.linalg.norm(qd_delta - reference_delta)) / reference_norm
        entry["distance_to_reference_rel_linf"] = float(np.max(np.abs(qd_delta - reference_delta))) / reference_scale

    # Machinery validation: the 0-sweep variant applies exactly the warm-start
    # M^-1 J^T lambda that production applies through its response rows.
    warmstart_validation = None
    if "matrix_free_warmstart_only" in raw_deltas and "production_warmstart_only" in raw_deltas:
        ws_matrix_free = raw_deltas["matrix_free_warmstart_only"]
        ws_production = raw_deltas["production_warmstart_only"]
        scale = max(1.0e-9, float(np.max(np.abs(ws_production))))
        warmstart_validation = {
            "max_abs_deviation": float(np.max(np.abs(ws_matrix_free - ws_production))),
            "max_rel_deviation": float(np.max(np.abs(ws_matrix_free - ws_production))) / scale,
            "passed_rtol_1e3": bool(np.allclose(ws_matrix_free, ws_production, rtol=1.0e-3, atol=1.0e-3 * scale)),
        }

    production_entry = quality["production_contact_phase"]

    def beats_production(entry: dict) -> bool:
        if not entry.get("finite", False):
            return False
        # Scale-aware: cone violations at ~1e-9 are float noise on O(1) lambdas.
        gates = (
            ("penetration_velocity_rms", 1.05, 1.0e-4),
            ("cone_violation_mean", 1.05, 1.0e-6),
            ("distance_to_reference_rel_l2", 1.02, 0.01),
        )
        return all(entry[key] <= production_entry[key] * factor + atol for key, factor, atol in gates)

    # Cheapest matrix-free config matching quality: order by tree passes per
    # solve (sweeps x groups per sweep), tie-broken toward larger groups.
    match_name = None
    match_config = None
    scored = []
    for biased, relax in sweep_configs:
        for group in args.groups:
            group_count = -(-max_points // group) if group > 0 else 1
            cost = (biased + relax) * group_count
            strided_options = (0, 1) if 1 < group < max_points else (0,)
            for strided in strided_options:
                tag = "s" if strided else "g"
                for omega in args.omegas:
                    name = f"matrix_free_b{biased}_r{relax}_{tag}{group}_w{omega:g}"
                    scored.append(
                        (cost, -group if group > 0 else -1000000, name, (biased, relax, group, strided, omega))
                    )
    for _cost, _tie, name, config in sorted(scored):
        if beats_production(quality[name]):
            match_name = name
            match_config = config
            break

    # ---- timing brackets (graph-captured, reversed orders) ----
    timing_names = ["production_contact_phase", "production_reference", "production_gather"]
    fallback_omega = args.omegas[min(1, len(args.omegas) - 1)]
    if match_name is not None:
        breakdown_config = match_config
    else:
        breakdown_config = (
            *sweep_configs[len(sweep_configs) // 2],
            args.groups[min(1, len(args.groups) - 1)],
            0,
            fallback_omega,
        )
    b_b, b_r, b_g, b_s, b_w = breakdown_config
    b_tag = "s" if b_s else "g"
    capture("matrix_free_build_only", lambda: launch_matrix_free(0, 0, b_g, b_s, b_w, _MODE_FULL))
    capture(
        f"matrix_free_noproj_b{b_b}_r{b_r}_{b_tag}{b_g}",
        lambda: launch_matrix_free(b_b, b_r, b_g, b_s, b_w, _MODE_NO_PROJECTION),
    )
    capture(
        f"matrix_free_notree_b{b_b}_r{b_r}_{b_tag}{b_g}",
        lambda: launch_matrix_free(b_b, b_r, b_g, b_s, b_w, _MODE_NO_TREE),
    )
    timing_names += [
        "matrix_free_build_only",
        f"matrix_free_noproj_b{b_b}_r{b_r}_{b_tag}{b_g}",
        f"matrix_free_notree_b{b_b}_r{b_r}_{b_tag}{b_g}",
    ]
    if match_name is not None:
        timing_names.append(match_name)
    # Extra sweep counts at the breakdown group/omega for the per-sweep slope.
    for biased, relax in sweep_configs:
        name = f"matrix_free_b{biased}_r{relax}_{b_tag}{b_g}_w{b_w:g}"
        if name in graphs and name not in timing_names:
            timing_names.append(name)

    times: dict[str, list[float]] = {name: [] for name in timing_names}
    for direction in (timing_names, list(reversed(timing_names)), timing_names, list(reversed(timing_names))):
        for name in direction:
            restore()
            times[name].extend(_time_graph_batches(graphs[name], args.replays, args.batches, device))
    median_us = {name: float(np.median(values)) for name, values in times.items()}

    production_us = median_us["production_contact_phase"]
    build_us = median_us["matrix_free_build_only"]
    sweep_times = {name: median_us[name] for name in timing_names if name.startswith("matrix_free_b") and "_w" in name}
    # Per-sweep slope from the two largest timed sweep counts.
    slope_entries = sorted(
        (
            (int(name.split("_b")[1].split("_")[0]) + int(name.split("_r")[1].split("_")[0]), median_us[name])
            for name in sweep_times
        ),
    )
    per_sweep_us = None
    if len(slope_entries) >= 2:
        (sweeps_low, time_low), (sweeps_high, time_high) = slope_entries[0], slope_entries[-1]
        if sweeps_high > sweeps_low:
            per_sweep_us = (time_high - time_low) / (sweeps_high - sweeps_low)

    resources = _kernel_resources(kernel, device, _LANES)
    matched_us = median_us.get(match_name) if match_name is not None else None
    payload = {
        "schema": "phoenx_matrix_free_reduced_contacts_v1",
        "device": device.name,
        "world_count": args.world_count,
        "dt": args.dt,
        "production_iterations": [iterations_biased, iterations_relax],
        "reference_iterations": [reference_biased, reference_relax],
        "contact_points_per_world_max": max_points,
        "contact_points_per_world_mean": float(np.mean(point_counts)),
        "rows_pad": rows_pad,
        "nv": nv,
        "nv_stride": nv_stride,
        "joint_bucket": joint_bucket,
        "path_dof_bucket": path_dof_bucket,
        "warmstart_validation": warmstart_validation,
        "quality": dict(quality),
        "matched_config": {
            "name": match_name,
            "biased": match_config[0],
            "relax": match_config[1],
            "group": match_config[2],
            "strided": match_config[3],
            "omega": match_config[4],
            "time_us": matched_us,
        }
        if match_name is not None
        else None,
        "median_us": median_us,
        "per_sweep_us": per_sweep_us,
        "build_warmstart_apply_us": build_us,
        "production_contact_phase_us": production_us,
        "production_gather_us": median_us["production_gather"],
        "production_contact_phase_ex_gather_us": max(0.0, production_us - median_us["production_gather"]),
        "speedup_at_matched_quality": (production_us / matched_us) if matched_us else None,
        "speedup_at_matched_quality_gather_inclusive": (production_us / (matched_us + median_us["production_gather"]))
        if matched_us
        else None,
        "full_step_us": step_us,
        "kernel_resources": resources,
        "smem_bytes_analytic": 4
        * (
            2 * rows_pad * (path_dof_bucket + 1)
            + 3 * rows_pad
            + 3
            + points_cap * 13
            + 3 * nv_stride * 6
            + 2 * (joint_bucket + 1)
            + 3 * nv_stride
            + joint_bucket * 6
        ),
        "timing_batches_us": dict(times),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
