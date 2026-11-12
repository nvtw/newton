# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Radix sort over the active prefix of a capacity-sized buffer inside a captured graph."""

from __future__ import annotations

from collections.abc import Callable

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
def _select_active_prefix_kernel(active_length: wp.array[int], count: int, flags: wp.array[wp.int32]):
    # One flag per node of the search tree over the prefix sizes: whether the
    # chosen size ``count >> chosen`` lies at or before the node's pivot.
    node = wp.tid()
    n = active_length[0]
    chosen = int(0)
    for shift in range(1, flags.shape[0]):
        if n < (count >> shift):
            chosen = shift
    # The pivot of a node of the balanced tree over the sizes, nodes in preorder. A table of
    # them would be uploaded while a graph is captured, and uploaded again from memory that
    # is long gone on every replay.
    low = int(0)
    high = flags.shape[0] - 1
    index = int(0)
    pivot = (low + high) // 2
    while index != node:
        # The left subtree holds one node less than its sizes.
        if node <= index + (pivot - low):
            high = pivot
            index += 1
        else:
            index += 1 + (pivot - low)
            low = pivot + 1
        pivot = (low + high) // 2
    flags[node] = wp.int32(1) if chosen <= pivot else wp.int32(0)


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


#: Capacity from which a group of kernels is worth launching through :func:`dispatch_active_prefix`:
#: the choice costs about 10 us, a kernel over two million idle threads about as much.
ACTIVE_DISPATCH_MIN_ITEMS = 1 << 21

#: Active entries up to which one pass that ranks every key against all others beats the
#: radix sort, whose eight passes over 64-bit keys each cost a launch however few keys there are.
_RANK_SORT_MAX_ITEMS = 1024


@wp.kernel(enable_backward=False)
def _select_rank_sort_kernel(active_length: wp.array[int], flag: wp.array[wp.int32]):
    flag[0] = wp.int32(1) if active_length[0] <= wp.static(_RANK_SORT_MAX_ITEMS) else wp.int32(0)


@wp.kernel(enable_backward=False)
def _rank_sort_scatter_kernel(
    keys: wp.array[wp.int64], values: wp.array[int], active_length: wp.array[int], count: int
):
    # Stable: equal keys keep their order. The sorted entries land in the ping-pong half.
    i = wp.tid()
    n = active_length[0]
    if i >= n:
        return
    key = keys[i]
    rank = int(0)
    for j in range(n):
        other = keys[j]
        if other < key or (other == key and j < i):
            rank += 1
    keys[count + rank] = key
    values[count + rank] = values[i]


@wp.kernel(enable_backward=False)
def _rank_sort_scatter_int_kernel(keys: wp.array[int], values: wp.array[int], active_length: wp.array[int], count: int):
    i = wp.tid()
    n = active_length[0]
    if i >= n:
        return
    key = keys[i]
    rank = int(0)
    for j in range(n):
        other = keys[j]
        if other < key or (other == key and j < i):
            rank += 1
    keys[count + rank] = key
    values[count + rank] = values[i]


@wp.kernel(enable_backward=False)
def _rank_sort_gather_kernel(keys: wp.array[wp.int64], values: wp.array[int], active_length: wp.array[int], count: int):
    i = wp.tid()
    if i < active_length[0]:
        keys[i] = keys[count + i]
        values[i] = values[count + i]


@wp.kernel(enable_backward=False)
def _rank_sort_gather_int_kernel(keys: wp.array[int], values: wp.array[int], active_length: wp.array[int], count: int):
    i = wp.tid()
    if i < active_length[0]:
        keys[i] = keys[count + i]
        values[i] = values[count + i]


def dispatch_active_prefix(
    active_length: wp.array[int],
    count: int,
    flags: wp.array[wp.int32],
    body: Callable[[int], None],
) -> None:
    """Call ``body(prefix)`` for the smallest size ``count >> k`` that strictly contains the active entries.

    ``body`` launches its kernels over ``prefix`` instead of ``count`` threads. Inside a
    captured graph the size is chosen on the device in every replay: ``body`` is captured
    once per size, and a search tree of conditional nodes picks one. ``flags`` must contain
    :func:`active_prefix_sort_levels` entries for ``count``. ``body`` must not allocate.
    """
    levels = flags.shape[0]
    if levels != active_prefix_sort_levels(count):
        raise ValueError(f"flags must contain {active_prefix_sort_levels(count)} entries, got {levels}")
    if levels == 1 or not flags.device.is_cuda:
        body(count)
        return
    wp.launch(
        _select_active_prefix_kernel,
        dim=levels - 1,
        inputs=[active_length, count],
        outputs=[flags],
        device=flags.device,
    )

    # A conditional graph node costs a few microseconds: descend a search tree over the
    # prefix sizes instead of testing each in turn. Nodes are numbered in preorder.
    node = 0

    def descend(low: int, high: int) -> None:
        nonlocal node
        if low == high:
            body(count >> low)
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


def radix_sort_prefix(keys: wp.array, values: wp.array[int], prefix: int, end_bit: int | None = None) -> None:
    """Sort the first ``prefix`` of ``keys.shape[0] // 2`` int32 or int64 keys with their values.

    The sort uses the entries after the prefix as scratch; they get the sentinel back, which
    the last entry of the sorted prefix must hold.
    """
    count = keys.shape[0] // 2
    wp.utils.radix_sort_pairs(keys, values, prefix, end_bit=end_bit)
    if prefix < count:
        wp.launch(
            _restore_sort_scratch_kernel if keys.dtype == wp.int64 else _restore_sort_scratch_int_kernel,
            dim=min(prefix, count - prefix),
            inputs=[keys, values, prefix],
            device=keys.device,
        )


def radix_sort_active_prefix(
    keys: wp.array,
    values: wp.array[int],
    active_length: wp.array[int],
    flags: wp.array[wp.int32],
    end_bit: int | None = None,
    prefix: int | None = None,
) -> None:
    """Sort int32 or int64 ``keys[:count]`` whose entries past ``active_length`` hold one sentinel.

    ``count = keys.shape[0] // 2`` and ``flags`` must contain
    :func:`active_prefix_sort_levels` entries. The sentinel must sort after
    every active key. Afterwards all of ``keys[:count]`` is sorted and its
    tail again holds the sentinel; tail values are unspecified. Few active
    entries are ranked against each other instead, in two launches.

    A caller inside :func:`dispatch_active_prefix` passes its ``prefix`` and the flags of
    that dispatch: only ``keys[:prefix]`` is sorted and needs the sentinel behind the
    active entries.
    """
    count = keys.shape[0] // 2

    def radix() -> None:
        if prefix is not None:
            radix_sort_prefix(keys, values, prefix, end_bit)
            return
        dispatch_active_prefix(active_length, count, flags, lambda size: radix_sort_prefix(keys, values, size, end_bit))

    if not keys.device.is_cuda or count <= _RANK_SORT_MAX_ITEMS:
        radix()
        return
    # The last flag is free: the search tree over the prefix sizes uses one fewer.
    few = flags[flags.shape[0] - 1 :]
    wp.launch(_select_rank_sort_kernel, dim=1, inputs=[active_length], outputs=[few], device=keys.device)

    def rank() -> None:
        wide = keys.dtype == wp.int64
        for kernel in (
            _rank_sort_scatter_kernel if wide else _rank_sort_scatter_int_kernel,
            _rank_sort_gather_kernel if wide else _rank_sort_gather_int_kernel,
        ):
            wp.launch(kernel, dim=_RANK_SORT_MAX_ITEMS, inputs=[keys, values, active_length, count], device=keys.device)

    wp.capture_if(few, on_true=rank, on_false=radix)


__all__ = [
    "ACTIVE_DISPATCH_MIN_ITEMS",
    "active_prefix_sort_levels",
    "dispatch_active_prefix",
    "radix_sort_active_prefix",
    "radix_sort_prefix",
]
