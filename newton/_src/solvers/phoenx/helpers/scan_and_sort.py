# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry.active_prefix_sort import radix_sort_active_prefix

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
    wp.launch(_mask_tail_int, dim=count, inputs=[keys, active_length], device=keys.device)
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
    tail_masked: bool = False,
) -> None:
    """Sort ``keys[:active_length]`` with ``values``; inactive entries sort to the end.

    ``tail_masked`` skips the sentinel fill for callers that already wrote
    ``INT64_MAX`` to every key past ``active_length``.
    """
    count = keys.shape[0] // 2
    if not tail_masked:
        wp.launch(_mask_tail_int64, dim=count, inputs=[keys, active_length], device=keys.device)
    # As for ``sort_variable_length_int``: callers whose nonnegative keys stay
    # below ``2**end_bit - 1`` get the full-width order with fewer passes.
    if prefix_flags is None:
        wp.utils.radix_sort_pairs(keys, values, count, end_bit=end_bit)
    else:
        radix_sort_active_prefix(keys, values, active_length, prefix_flags, end_bit=end_bit)
