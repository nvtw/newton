import warp as wp


# ---------------------------------------------------------------------------
# Masking kernels: set entries in [active_length[0], count) to the type's max
# value so that a subsequent full-range sort pushes them to the end. Required
# because `active_length` lives on the device and we cannot cheaply read it
# host-side to pass as the sort's `count` argument.
# ---------------------------------------------------------------------------


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


@wp.kernel
def _mask_tail_float(array: wp.array[float], active_length: wp.array[int]):
    tid = wp.tid()
    if tid >= active_length[0]:
        array[tid] = 3.4028235e38  # FLT_MAX


# ---------------------------------------------------------------------------
# Scan (prefix sum) wrappers.
# Warp's array_scan always operates on the full array size. Callers only
# read results within [0, active_length); values written past that are
# don't-care.
# ---------------------------------------------------------------------------


def scan_variable_length(array: wp.array, active_length: wp.array[int], inclusive: bool = False) -> None:
    wp.utils.array_scan(array, array, inclusive=inclusive)


# ---------------------------------------------------------------------------
# Sort wrappers.
# Warp only provides key-value radix sort (`radix_sort_pairs`). The C# code
# uses a keys-only sort; we emulate it by requiring the caller to pass a
# values scratch array. Both arrays must have size >= 2 * count (ping-pong
# buffer required by Warp's radix sort).
#
# Masking: we first set all entries in [active_length[0], count) to the
# largest representable value so they sort to the end of the active region
# and leave the head (the real data) correctly ordered.
# ---------------------------------------------------------------------------


def sort_variable_length_int(keys: wp.array[int], values: wp.array[int], active_length: wp.array[int]) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_int, dim=count, inputs=[keys, active_length])
    wp.utils.radix_sort_pairs(keys, values, count)


def sort_variable_length_int64(
    keys: wp.array[wp.int64], values: wp.array[int], active_length: wp.array[int]
) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_int64, dim=count, inputs=[keys, active_length])
    wp.utils.radix_sort_pairs(keys, values, count)


def sort_variable_length_float(keys: wp.array[float], values: wp.array[int], active_length: wp.array[int]) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_float, dim=count, inputs=[keys, active_length])
    wp.utils.radix_sort_pairs(keys, values, count)
