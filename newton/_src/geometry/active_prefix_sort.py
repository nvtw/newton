# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Radix sort over the active prefix of a capacity-sized buffer inside a captured graph."""

from __future__ import annotations

import warp as wp

# Active-prefix sort: CUB sorts a host-sized range, but the active count is
# device state. Sorting the whole capacity spends most passes on sentinels
# when buffers are sized for worst-case contact counts. A short ladder of
# conditional graph branches sorts the smallest power-of-two prefix that
# strictly contains the active entries. Each sort costs one predicate per
# level, which is negligible next to the skipped radix passes.

#: Smallest prefix worth a separate branch; below this one launch dominates.
_ACTIVE_PREFIX_MIN_ITEMS = 1 << 16
#: Largest capacity reduction tried, as a power of two.
_ACTIVE_PREFIX_MAX_SHIFT = 5


def active_prefix_sort_levels(count: int) -> int:
    """Return the number of prefix sizes ``count >> k`` used for ``count`` keys."""
    levels = 1
    while levels <= _ACTIVE_PREFIX_MAX_SHIFT and (count >> levels) >= _ACTIVE_PREFIX_MIN_ITEMS:
        levels += 1
    return levels


@wp.kernel(enable_backward=False)
def _select_active_prefix_kernel(active_length: wp.array[int], count: int, flags: wp.array[wp.int32]):
    level = wp.tid()
    n = active_length[0]
    chosen = int(0)
    for shift in range(1, flags.shape[0]):
        if n < (count >> shift):
            chosen = shift
    flags[level] = wp.int32(1) if level == chosen else wp.int32(0)


@wp.kernel(enable_backward=False)
def _restore_sort_scratch_kernel(keys: wp.array[wp.int64], values: wp.array[int], prefix: int):
    # The prefix sort used [prefix, 2 * prefix) as ping-pong storage. Its
    # last sorted key is a sentinel because the active entries end earlier.
    tid = wp.tid()
    keys[prefix + tid] = keys[prefix - 1]
    values[prefix + tid] = values[prefix - 1]


@wp.kernel(enable_backward=False)
def _restore_sort_scratch_int_kernel(keys: wp.array[int], values: wp.array[int], prefix: int):
    tid = wp.tid()
    keys[prefix + tid] = keys[prefix - 1]
    values[prefix + tid] = values[prefix - 1]


def radix_sort_active_prefix(
    keys: wp.array,
    values: wp.array[int],
    active_length: wp.array[int],
    flags: wp.array[wp.int32],
    end_bit: int | None = None,
) -> None:
    """Sort int32 or int64 ``keys[:count]`` whose entries past ``active_length`` hold one sentinel.

    ``count = keys.shape[0] // 2`` and ``flags`` must contain
    :func:`active_prefix_sort_levels` entries. The sentinel must sort after
    every active key. Afterwards all of ``keys[:count]`` is sorted and its
    tail again holds the sentinel; tail values are unspecified.
    """
    count = keys.shape[0] // 2
    levels = flags.shape[0]
    if levels != active_prefix_sort_levels(count):
        raise ValueError(f"flags must contain {active_prefix_sort_levels(count)} entries, got {levels}")
    if levels == 1 or not keys.device.is_cuda:
        wp.utils.radix_sort_pairs(keys, values, count, end_bit=end_bit)
        return
    wp.launch(_select_active_prefix_kernel, dim=levels, inputs=[active_length, count, flags], device=keys.device)
    for shift in range(levels):
        prefix = count >> shift

        def sort_prefix(prefix: int = prefix) -> None:
            wp.utils.radix_sort_pairs(keys, values, prefix, end_bit=end_bit)
            if prefix < count:
                wp.launch(
                    _restore_sort_scratch_kernel if keys.dtype == wp.int64 else _restore_sort_scratch_int_kernel,
                    dim=min(prefix, count - prefix),
                    inputs=[keys, values, prefix],
                    device=keys.device,
                )

        wp.capture_if(flags[shift : shift + 1], on_true=sort_prefix)


__all__ = ["active_prefix_sort_levels", "radix_sort_active_prefix"]
