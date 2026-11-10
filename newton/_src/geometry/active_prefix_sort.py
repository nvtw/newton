# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Radix sort over the active prefix of a capacity-sized buffer inside a captured graph."""

from __future__ import annotations

import warp as wp

# Active-prefix sort: CUB sorts a host-sized range, but the active count is
# device state. Sorting the whole capacity spends most passes on sentinels
# when buffers are sized for worst-case contact counts. A short ladder of
# conditional graph branches sorts the smallest power-of-two prefix that
# strictly contains the active entries. Each sort descends a search tree over
# the levels, a few predicates that cost little next to the skipped radix passes.

#: Smallest prefix worth a separate branch; below this one launch dominates.
_ACTIVE_PREFIX_MIN_ITEMS = 1 << 16
#: Largest capacity reduction tried, as a power of two.
_ACTIVE_PREFIX_MAX_SHIFT = 12


def active_prefix_sort_levels(count: int) -> int:
    """Return the number of prefix sizes ``count >> k`` used for ``count`` keys."""
    levels = 1
    while levels <= _ACTIVE_PREFIX_MAX_SHIFT and (count >> levels) >= _ACTIVE_PREFIX_MIN_ITEMS:
        levels += 1
    return levels


@wp.kernel(enable_backward=False)
def _select_active_prefix_kernel(
    active_length: wp.array[int], count: int, pivots: wp.array[wp.int32], flags: wp.array[wp.int32]
):
    # One flag per node of the search tree over the prefix sizes: whether the
    # chosen size ``count >> chosen`` lies at or before the node's pivot.
    node = wp.tid()
    n = active_length[0]
    chosen = int(0)
    for shift in range(1, flags.shape[0]):
        if n < (count >> shift):
            chosen = shift
    flags[node] = wp.int32(1) if chosen <= pivots[node] else wp.int32(0)


_PIVOTS: dict[tuple[str, int], wp.array] = {}


def _search_pivots(levels: int, device) -> wp.array:
    """Return the pivots of a balanced search tree over ``levels`` leaves, in preorder."""
    key = (str(device), levels)
    if key not in _PIVOTS:
        pivots: list[int] = []

        def visit(low: int, high: int) -> None:
            if low < high:
                middle = (low + high) // 2
                pivots.append(middle)
                visit(low, middle)
                visit(middle + 1, high)

        visit(0, levels - 1)
        _PIVOTS[key] = wp.array(pivots, dtype=wp.int32, device=device)
    return _PIVOTS[key]


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
    wp.launch(
        _select_active_prefix_kernel,
        dim=levels - 1,
        inputs=[active_length, count, _search_pivots(levels, keys.device)],
        outputs=[flags],
        device=keys.device,
    )

    def sort_prefix(shift: int) -> None:
        prefix = count >> shift
        wp.utils.radix_sort_pairs(keys, values, prefix, end_bit=end_bit)
        if prefix < count:
            wp.launch(
                _restore_sort_scratch_kernel if keys.dtype == wp.int64 else _restore_sort_scratch_int_kernel,
                dim=min(prefix, count - prefix),
                inputs=[keys, values, prefix],
                device=keys.device,
            )

    # A conditional graph node costs a few microseconds: descend a search tree over the
    # prefix sizes instead of testing each in turn. Nodes are numbered in preorder.
    node = 0

    def descend(low: int, high: int) -> None:
        nonlocal node
        if low == high:
            sort_prefix(low)
            return
        flag = flags[node : node + 1]
        node += 1
        middle = (low + high) // 2
        left_node = node

        def left() -> None:
            nonlocal node
            node = left_node
            descend(low, middle)

        # The right subtree's nodes follow the left subtree's in preorder.
        right_node = left_node + (middle - low)

        def right() -> None:
            nonlocal node
            node = right_node
            descend(middle + 1, high)

        wp.capture_if(flag, on_true=left, on_false=right)

    descend(0, levels - 1)


__all__ = ["active_prefix_sort_levels", "radix_sort_active_prefix"]
