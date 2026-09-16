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


class ContactChunkScratch:
    """Per-world-owned scratch for lossless point-contact range splitting."""

    def __init__(self, capacity: int, columns: ContactColumnContainer, device):
        self.capacity = capacity
        self.source = contact_column_container_zeros(capacity, device=device, data_dwords=columns.data.shape[0])
        self.source_pair = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.counts = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.offsets = wp.zeros_like(self.counts)


def split_contact_columns(
    scratch: ContactChunkScratch,
    columns: ContactColumnContainer,
    active: wp.array,
    source_pair: wp.array,
    chunk_size: int,
    cid_base: int,
    cid_of_contact: wp.array,
    total_active: wp.array,
    *,
    device,
) -> None:
    """Split columns without changing point order, history indices or endpoints."""
    wp.copy(scratch.source_pair, source_pair)
    wp.copy(scratch.source.data, columns.data)
    wp.copy(scratch.source.articulation_owner, columns.articulation_owner)
    wp.launch(_count_chunks, scratch.capacity, [scratch.source, active, chunk_size, scratch.counts], device=device)
    wp.utils.array_scan(scratch.counts, scratch.offsets, inclusive=False)
    wp.launch(
        _split_columns,
        scratch.capacity,
        [
            scratch.source,
            columns,
            active,
            scratch.counts,
            scratch.offsets,
            chunk_size,
            scratch.source_pair,
            source_pair,
            cid_base,
            cid_of_contact,
        ],
        device=device,
    )
    wp.launch(_publish_chunks, 1, [scratch.counts, scratch.offsets, active, cid_base, total_active], device=device)


def restamp_contact_chunk_owners(
    columns: ContactColumnContainer,
    active: wp.array,
    cid_base: int,
    cid_of_contact: wp.array,
    *,
    device,
) -> None:
    """Restore current chunk ownership after point-history gather."""
    wp.launch(_restamp_chunk_owners, columns.data.shape[1], [columns, active, cid_base, cid_of_contact], device=device)
