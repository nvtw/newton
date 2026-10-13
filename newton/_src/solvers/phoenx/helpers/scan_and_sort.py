# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

# Masking kernels: set entries in [active_length[0], count) to the type's max
# so a full-range sort pushes them to the end. Needed because active_length
# lives on the device.


@wp.kernel
def _mask_tail_int(array: wp.array[int], active_length: wp.array[int]):
    tid = wp.tid()
    if tid >= active_length[0]:
        array[tid] = 2147483647  # INT32_MAX


@wp.kernel
def _mask_tail_int64(array: wp.array[wp.int64], active_length: wp.array[int]):
    tid = wp.tid()
    if tid >= active_length[0]:
        array[tid] = wp.int64(9223372036854775807)  # INT64_MAX


# Scan: full-array; values past active_length are don't-care.


def scan_variable_length(array: wp.array, active_length: wp.array[int], inclusive: bool = False) -> None:
    wp.utils.array_scan(array, array, inclusive=inclusive)


# Sort: emulates a keys-only sort via radix_sort_pairs. Arrays must have
# size >= 2 * count (ping-pong buffer). Tail is masked to push inactive
# entries to the end.


def sort_variable_length_int(
    keys: wp.array[int], values: wp.array[int], active_length: wp.array[int], end_bit: int | None = None
) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_int, dim=count, inputs=[keys, active_length])
    # ``end_bit`` lets callers cap the radix passes when the key range is known
    # to be small (e.g. a world id). The inactive-tail sentinel is INT32_MAX, so
    # its low ``end_bit`` bits (2**end_bit - 1) still exceed any in-range key and
    # sort to the end -- the result is identical to a full-width sort.
    wp.utils.radix_sort_pairs(keys, values, count, end_bit=end_bit)


def sort_variable_length_int64(
    keys: wp.array[wp.int64],
    values: wp.array[int],
    active_length: wp.array[int],
    end_bit: int | None = None,
    prefix_flags: wp.array[wp.int32] | None = None,
) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_int64, dim=count, inputs=[keys, active_length])
    # As for ``sort_variable_length_int``: callers whose nonnegative keys stay
    # below ``2**end_bit - 1`` get the full-width order with fewer passes.
    if prefix_flags is None:
        wp.utils.radix_sort_pairs(keys, values, count, end_bit=end_bit)
    else:
        radix_sort_active_prefix(keys, values, active_length, prefix_flags, end_bit=end_bit)


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


# Variable-length RLE: stamp the tail with a sentinel so the full-size RLE
# folds the inactive tail into one run, then drop that run on-device.
# Graph-capture safe (fixed launch size).


#: Sentinel for :func:`runlength_encode_variable_length`. INT32_MAX is never
#: a legal key (keys are shape_a * num_shapes + shape_b, bounded by world build).
RLE_SENTINEL_INT32: int = 2147483647


@wp.kernel
def _rle_mask_tail_int(
    values: wp.array[wp.int32],
    active_length: wp.array[wp.int32],
    sentinel: wp.int32,
):
    tid = wp.tid()
    if tid >= active_length[0]:
        values[tid] = sentinel


@wp.kernel
def _rle_drop_sentinel_tail_kernel(
    run_values: wp.array[wp.int32],
    run_count: wp.array[wp.int32],
    sentinel: wp.int32,
):
    """If the last RLE run is the sentinel tail, shave it off in-place."""
    tid = wp.tid()
    if tid != 0:
        return
    n = run_count[0]
    if n <= 0:
        return
    if run_values[n - 1] == sentinel:
        run_count[0] = n - 1


def runlength_encode_variable_length(
    values: wp.array,
    active_length: wp.array,
    run_values: wp.array,
    run_lengths: wp.array,
    run_count: wp.array,
    sentinel: int = RLE_SENTINEL_INT32,
) -> None:
    """Graph-capture-safe RLE where ``active_length`` lives on device.
    ``values`` is **modified in place** (tail stamped with sentinel)."""
    n = values.shape[0]
    wp.launch(
        _rle_mask_tail_int,
        dim=n,
        inputs=[values, active_length, int(sentinel)],
        device=values.device,
    )
    wp.utils.runlength_encode(
        values,
        run_values=run_values,
        run_lengths=run_lengths,
        run_count=run_count,
        value_count=n,
    )
    wp.launch(
        _rle_drop_sentinel_tail_kernel,
        dim=1,
        inputs=[run_values, run_count, int(sentinel)],
        device=values.device,
    )
