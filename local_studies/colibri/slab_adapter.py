# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Eager-only slab adapter using the actual rigid Phoenx callbacks."""

import functools
import inspect
import linecache

import numpy as np
import warp as wp

from local_studies.colibri import slab_gpu
from local_studies.colibri.slab_schedule import build_schedule
from newton._src.solvers.phoenx.mass_splitting import slot_cache
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import build_interaction_graph
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    BodyContainer,
    ConstraintContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
    _make_singleworld_dispatch_func,
    _sync_threads,
)


@functools.cache
def _slot_kernel():
    # Preserve the existing endpoint/type-specific cache code verbatim; only
    # replace its overflow-position-derived pid with the explicit slab id.
    source = inspect.getsource(slot_cache.build_slot_cache_kernel.func)
    source = source.replace(
        "def build_slot_cache_kernel(", "def build_slab_slot_cache_kernel(\n    row_slab: wp.array[wp.int32],"
    )
    begin = source.index("    # parallel_id discovery.")
    end = source.index("    _cache_slots_for_partition(", begin)
    source = source[:begin] + "    parallel_id = row_slab[cid]\n\n" + source[end:]
    filename = __file__ + ".generated_slots"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(slot_cache))
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    return namespace["build_slab_slot_cache_kernel"]


@functools.cache
def _kernel(phase, soft_pd):
    dispatch, _ = _make_singleworld_dispatch_func(
        cloth_support=False,
        soft_tet_neohookean=False,
        enable_column_timers=False,
        has_joints=True,
        skip_joint_pgs=False,
        has_mass_splitting=True,
        packed_contact_headers=False,
        has_sleeping=False,
        has_soft_contact_pd=soft_pd,
        is_prepare=phase == "prepare",
        is_cached_prepare=phase == "cached_prepare",
        use_bias=phase == "iterate",
        patch_friction=False,
        bilateral_joint_blocks=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        constraints: ConstraintContainer,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
        num_joints: wp.int32,
        enabled: wp.array[wp.int32],
        num_bodies: wp.int32,
        idt: wp.float32,
        ids: wp.array[wp.int32],
        starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        slab_width: wp.int32,
    ):
        block, lane = wp.tid()
        for slab in range(block, (num_colors[0] + slab_width - 1) / slab_width, 32):
            for local_color in range(slab_width):
                color = slab * slab_width + local_color
                if color < num_colors[0]:
                    for index in range(starts[color] + lane, starts[color + 1], 32):
                        cid = ids[index]
                        dispatch(
                            constraints,
                            columns,
                            bodies,
                            particles,
                            cc,
                            contacts,
                            copies,
                            num_joints,
                            enabled,
                            0,
                            0,
                            0,
                            0,
                            num_bodies,
                            idt,
                            1.0,
                            cid,
                            index,
                            slab,
                        )
                _sync_threads()

    return sweep


def install(solver, colors_per_slab=8, gpu=False):
    """Install a deliberately eager CPU graph build and real callback sweep."""
    world = solver.world
    if (
        not world.mass_splitting_enabled
        or world.step_layout != "single_world"
        or world._colored_contact_headers
        or world._colored_contact_rows
        or world._contact_patch_enabled
        or world.num_cloth_triangles
        or world.num_cloth_bending
        or world.num_soft_tetrahedra
        or world.num_soft_hexahedra
    ):
        raise ValueError("Slab reference supports eager unsorted rigid mass-split single-world only")
    phase_heads = dict(zip(world._singleworld_kernels()[::2], ("prepare", "iterate", "relax"), strict=True))
    cache_kernel = _slot_kernel()
    data = slab_gpu.allocate(max(world._constraint_capacity, 1), world.num_bodies, world.device)

    def gpu_rebuild():
        slab_gpu.build(data, world._elements, world._num_active_constraints, colors_per_slab, world.device)
        wp.launch(
            slab_gpu.emit_slab_pairs,
            world._constraint_capacity,
            [world._elements, world._num_active_constraints, data["row_slab"], world._interaction_graph_scratch],
            device=world.device,
        )
        build_interaction_graph(world._interaction_graph_scratch, world._copy_state)
        wp.launch(
            cache_kernel,
            world._constraint_capacity,
            [
                data["row_slab"],
                data["ids"],
                data["starts"],
                world._num_active_constraints,
                world._copy_state,
                world.constraints,
                world._contact_cols,
                world._contact_offset,
                -1,
                1,
            ],
            device=world.device,
        )

    def rebuild():
        if world.device.is_capturing:
            raise ValueError("CPU slab reference cannot run in CUDA graph capture")
        n = int(world._num_active_constraints.numpy()[0])
        endpoints = world._elements.numpy()["bodies"][:n]
        schedule = build_schedule(endpoints, world.num_bodies, colors_per_slab)
        copies = world._copy_state
        counts = np.array([len(s) for s in schedule.body_slabs], np.int32)
        ends = np.cumsum(counts, dtype=np.int32)
        keys = np.full(copies.partition_list.shape[0], -1, np.int32)
        flat = [slab for slabs in schedule.body_slabs for slab in slabs]
        if len(flat) > len(keys):
            raise ValueError("Slab copies exceed allocated capacity")
        keys[: len(flat)] = flat
        pid0 = np.full(world.num_bodies, -1, np.int32)
        for body, slabs in enumerate(schedule.body_slabs):
            if slabs and slabs[0] == 0:
                pid0[body] = ends[body] - counts[body]
        copies.section_end.assign(ends)
        copies.count_per_node.assign(counts)
        copies.partition_list.assign(keys)
        copies.slot_for_pid0.assign(pid0)
        copies.highest_index_in_use.assign(np.array([len(flat)], np.int32))
        data["schedule"] = schedule
        data["num_colors"].assign(np.array([len(schedule.colors)], np.int32))
        data["ids"] = wp.array([r for c in schedule.colors for r in c], dtype=wp.int32, device=world.device)
        data["starts"] = wp.array(
            np.r_[0, np.cumsum([len(c) for c in schedule.colors])], dtype=wp.int32, device=world.device
        )
        data["row_slab"] = wp.array(schedule.row_slab, dtype=wp.int32, device=world.device)
        if n:
            wp.launch(
                cache_kernel,
                n,
                [
                    data["row_slab"],
                    data["ids"],
                    data["starts"],
                    world._num_active_constraints,
                    copies,
                    world.constraints,
                    world._contact_cols,
                    world._contact_offset,
                    -1,
                    1,
                ],
                device=world.device,
            )

    def sweep(head_kernel, tail_kernel, idt, contact_container=None):
        phase = phase_heads[head_kernel]
        soft_pd = bool(world._dispatch_specialization_flags()["has_soft_contact_pd"])
        wp.launch(
            _kernel(phase, soft_pd),
            (32, 32),
            [
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                world._contact_container if contact_container is None else contact_container,
                world._active_contact_views(),
                world._copy_state,
                world.num_joints,
                world._joint_pgs_enabled,
                world.num_bodies,
                idt,
                data["ids"],
                data["starts"],
                data["num_colors"],
                colors_per_slab,
            ],
            block_dim=32,
            device=world.device,
        )

    world._rebuild_mass_splitting_graph = gpu_rebuild if gpu else rebuild
    world._singleworld_head_plus_tail_sweep = sweep
    return data
