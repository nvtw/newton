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


def sort_variable_length_int64(keys: wp.array[wp.int64], values: wp.array[int], active_length: wp.array[int]) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_int64, dim=count, inputs=[keys, active_length])
    wp.utils.radix_sort_pairs(keys, values, count)


def sort_variable_length_float(keys: wp.array[float], values: wp.array[int], active_length: wp.array[int]) -> None:
    count = keys.shape[0] // 2
    wp.launch(_mask_tail_float, dim=count, inputs=[keys, active_length])
    wp.utils.radix_sort_pairs(keys, values, count)


# ---------------------------------------------------------------------------
# Variable-length run-length encoding.
# Wrapper around ``wp.utils.runlength_encode`` that stays graph-capture
# compatible when the number of "active" input values is only known on
# the device. Same trick as the sort wrappers: stamp the tail with a
# sentinel so the full-size RLE call folds the inactive tail into a
# single trailing run, then drop that run on-device by decrementing
# ``run_count`` if its last ``run_value`` equals the sentinel.
#
# All kernels here run at fixed launch size determined host-side at
# wrapper-call time, making the whole sequence safe to record once and
# replay any number of times inside a captured CUDA graph.
# ---------------------------------------------------------------------------


#: Sentinel used by :func:`runlength_encode_variable_length` to mark
#: tail entries. ``INT32_MAX`` is never a legal key in any current
#: caller (keys are ``shape_a * num_shapes + shape_b`` with
#: ``num_shapes * num_shapes < INT32_MAX`` enforced at world build).
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
    """If the last RLE run is the sentinel tail, shave it off in-place.

    Single-thread kernel -- we only need to inspect one element of
    ``run_count`` / ``run_values``. The body is branch-heavy but the
    per-step cost is negligible (one thread, one global load, one
    store) compared to the RLE itself.
    """
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

    ``values.shape[0]`` is the fixed launch size. Tail entries past
    ``active_length[0]`` are overwritten with ``sentinel`` so they
    collapse into a single trailing run that we drop in-place --
    equivalent to ``wp.utils.runlength_encode`` with
    ``value_count = active_length[0]`` (unavailable because CUB's
    ``value_count`` is a host int).

    After the call, ``run_values[0:run_count[0]]`` /
    ``run_lengths[0:run_count[0]]`` hold the unique values and matching
    lengths of ``values[0:active_length[0]]`` in first-appearance order.

    Args:
        values: ``int32`` input array, **modified in place** (tail
            stamped with ``sentinel``).
        active_length: 1-elem device scalar, valid prefix length.
        run_values, run_lengths: Outputs sized ``>= values.shape[0]``.
        run_count: 1-elem output, final run count (sentinel dropped).
        sentinel: Must not appear in legitimate input.
    """
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
