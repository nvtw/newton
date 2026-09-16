# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reuse canonical point geometry before the existing ordered slab warm start."""

import warp as wp

from local_studies.colibri.slab_adapter import _kernel
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import _get_parallel_contact_prepare_kernel


def install(solver):
    """Wrap an installed slab topology; leave iteration and relaxation untouched."""
    world = solver.world
    data = solver._slab_reference
    original_sweep = world._singleworld_head_plus_tail_sweep
    original_prepare = world._singleworld_kernels()[0]
    if not world.mass_splitting_enabled or world._colored_contact_headers or world._contact_patch_enabled:
        raise ValueError("This diagnostic requires unpacked rigid point contacts and slab copies.")

    def sweep(head_kernel, tail_kernel, idt, contact_container=None):
        if head_kernel is not original_prepare:
            return original_sweep(head_kernel, tail_kernel, idt, contact_container)
        cc = world._contact_container if contact_container is None else contact_container
        soft_pd = bool(world._dispatch_specialization_flags()["has_soft_contact_pd"])
        wp.launch(
            _get_parallel_contact_prepare_kernel(True, soft_pd),
            dim=(world.max_contact_columns, min(128, world.contact_chunk_size or 128)),
            inputs=[
                world._contact_cols,
                world._ingest_scratch.num_contact_columns,
                world.bodies,
                world._particles_or_sentinel(),
                wp.int32(world.num_bodies),
                idt,
                cc,
                world._active_contact_views(),
                world._copy_state,
            ],
            device=world.device,
        )
        wp.launch(
            _kernel("cached_prepare", soft_pd),
            (32, 32),
            [
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                cc,
                world._active_contact_views(),
                world._copy_state,
                world.num_joints,
                world._joint_pgs_enabled,
                world.num_bodies,
                idt,
                data["ids"],
                data["starts"],
                data["num_colors"],
                8,
            ],
            block_dim=32,
            device=world.device,
        )

    world._singleworld_head_plus_tail_sweep = sweep
