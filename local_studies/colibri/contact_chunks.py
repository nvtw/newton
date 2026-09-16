# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Opt-in contact range splitting; every point and history index is retained."""

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_set_contact_count,
    contact_set_contact_first,
)


@wp.kernel
def _count_chunks(
    columns: ContactColumnContainer,
    active: wp.array[wp.int32],
    chunk_size: int,
    counts: wp.array[wp.int32],
):
    i = wp.tid()
    value = int(0)
    if i < active[0]:
        value = (contact_get_contact_count(columns, i) + chunk_size - 1) // chunk_size
    counts[i] = value


@wp.kernel
def _split_columns(
    source: ContactColumnContainer,
    target: ContactColumnContainer,
    active: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    chunk_size: int,
    source_pair: wp.array[wp.int32],
    target_pair: wp.array[wp.int32],
    cid_base: int,
    cid_of_contact: wp.array[wp.int32],
):
    i = wp.tid()
    if i >= active[0]:
        return
    first = contact_get_contact_first(source, i)
    count = contact_get_contact_count(source, i)
    for chunk in range(counts[i]):
        out = offsets[i] + chunk
        # Capacity is at least the total contact-point capacity, so every
        # positive-length chunk fits without dropping physical rows.
        for row in range(source.data.shape[0]):
            target.data[row, out] = source.data[row, i]
        target.articulation_owner[out] = source.articulation_owner[i]
        target_pair[out] = source_pair[i]
        chunk_first = first + chunk * chunk_size
        chunk_count = wp.min(chunk_size, count - chunk * chunk_size)
        contact_set_contact_first(target, out, chunk_first)
        contact_set_contact_count(target, out, chunk_count)
        for point in range(chunk_count):
            cid_of_contact[chunk_first + point] = cid_base + out


@wp.kernel
def _publish_chunks(
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    active: wp.array[wp.int32],
    cid_base: int,
    total_active: wp.array[wp.int32],
):
    last = counts.shape[0] - 1
    total = offsets[last] + counts[last]
    active[0] = total
    total_active[0] = cid_base + total


@wp.kernel
def _restamp_chunk_owners(
    columns: ContactColumnContainer,
    active: wp.array[wp.int32],
    cid_base: int,
    cid_of_contact: wp.array[wp.int32],
):
    column = wp.tid()
    if column >= active[0]:
        return
    first = contact_get_contact_first(columns, column)
    count = contact_get_contact_count(columns, column)
    for point in range(first, first + count):
        cid_of_contact[point] = cid_base + column


def reserve_capacity():
    """Reserve one column per possible point before constructing the solver."""
    from newton._src.solvers.phoenx import solver as solver_module

    solver_module._estimate_contact_column_max_phoenx = lambda model, rigid_contact_max: int(rigid_contact_max)


def install(solver, chunk_size=6):
    """Split ingested contact ranges before warm-start gather and coloring."""
    from newton._src.solvers.phoenx import solver_phoenx as world_module

    world = solver.world
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if world.max_contact_columns < world.rigid_contact_max:
        raise ValueError("Call reserve_capacity() before constructing the solver")
    if world._contact_patch_enabled:
        raise ValueError("The chunk prototype does not support patch friction")
    capacity = world.max_contact_columns
    source = contact_column_container_zeros(
        capacity, device=world.device, data_dwords=world._contact_cols.data.shape[0]
    )
    source_pair = wp.zeros(capacity, dtype=wp.int32, device=world.device)
    counts = wp.zeros(capacity, dtype=wp.int32, device=world.device)
    offsets = wp.zeros_like(counts)
    original = world_module.ingest_contacts

    def ingest_chunks(*args, **kwargs):
        original(*args, **kwargs)
        if kwargs.get("contact_cols") is not world._contact_cols:
            return
        active = world._ingest_scratch.num_contact_columns
        wp.copy(source_pair, world._ingest_scratch.pair_source_idx)
        wp.copy(source.data, world._contact_cols.data)
        wp.copy(source.articulation_owner, world._contact_cols.articulation_owner)
        wp.launch(_count_chunks, capacity, [source, active, chunk_size, counts], device=world.device)
        wp.utils.array_scan(counts, offsets, inclusive=False)
        wp.launch(
            _split_columns,
            capacity,
            [
                source,
                world._contact_cols,
                active,
                counts,
                offsets,
                chunk_size,
                source_pair,
                world._ingest_scratch.pair_source_idx,
                world._contact_offset,
                world._cid_of_contact_cur,
            ],
            device=world.device,
        )
        wp.launch(
            _publish_chunks,
            1,
            [counts, offsets, active, world._contact_offset, world._num_active_constraints],
            device=world.device,
        )

    world_module.ingest_contacts = ingest_chunks
    original_ingest_and_warmstart = world._ingest_and_warmstart_contacts

    def ingest_and_restamp(*args, **kwargs):
        original_ingest_and_warmstart(*args, **kwargs)
        if world._ingest_scratch is None:
            return
        # The original warm-start gather stamps one owner per unsplit pair.
        # Publish the final chunk owners after gather and endpoint overlays.
        wp.launch(
            _restamp_chunk_owners,
            capacity,
            [
                world._contact_cols,
                world._ingest_scratch.num_contact_columns,
                world._contact_offset,
                world._cid_of_contact_cur,
            ],
            device=world.device,
        )

    world._ingest_and_warmstart_contacts = ingest_and_restamp
    return {"source": source, "counts": counts, "offsets": offsets, "original_ingest": original}
