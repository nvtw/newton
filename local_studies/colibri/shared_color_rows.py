"""Local exact rigid first-fit coloring with a bounded shared mask cache."""

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups

_original_build = color_groups.build


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ unsigned int cached[128 * 8];
    const int lane = threadIdx.x;
    const int bodies = masks.shape[0];
    const int words = masks.shape[1];
    const int cached_words = words < 8 ? words : 8;
    for (int i = lane; i < bodies * cached_words; i += blockDim.x)
        cached[i] = masks.data[(i / cached_words) * words + i % cached_words];
    __syncthreads();
    if (lane == 0) {
        int colors = 0;
        for (int row = 0; row < active.data[0]; ++row) {
            int chosen = -1, word = 0;
            while (chosen < 0) {
                unsigned int occupied = 0;
                for (int endpoint = 0; endpoint < 2; ++endpoint) {
                    int body = elements.data[row].bodies[endpoint];
                    if (body >= 0)
                        occupied |= word < cached_words ? cached[body * cached_words + word]
                                                       : masks.data[body * words + word];
                }
                if (occupied != 0xffffffffu) {
                    unsigned int available = ~occupied;
                    int bit = 0;
                    if ((available & 0xffffu) == 0) { available >>= 16; bit += 16; }
                    if ((available & 0xffu) == 0) { available >>= 8; bit += 8; }
                    if ((available & 0xfu) == 0) { available >>= 4; bit += 4; }
                    if ((available & 0x3u) == 0) { available >>= 2; bit += 2; }
                    if ((available & 0x1u) == 0) ++bit;
                    chosen = word * 32 + bit;
                } else ++word;
            }
            for (int endpoint = 0; endpoint < 2; ++endpoint) {
                int body = elements.data[row].bodies[endpoint];
                if (body >= 0) {
                    unsigned int bit = 1u << (chosen % 32);
                    if (word < cached_words) cached[body * cached_words + word] |= bit;
                    else masks.data[body * words + word] |= bit;
                }
            }
            row_color.data[row] = chosen;
            row_partition.data[row] = chosen / width;
            ++counts.data[chosen];
            colors = colors > chosen + 1 ? colors : chosen + 1;
        }
        num_colors.data[0] = colors;
        starts.data[0] = 0;
        for (int color = 0; color < colors; ++color) {
            starts.data[color + 1] = starts.data[color] + counts.data[color];
            cursors.data[color] = starts.data[color];
        }
        for (int row = 0; row < active.data[0]; ++row) {
            int color = row_color.data[row];
            ids.data[cursors.data[color]++] = row;
        }
    }
    __syncthreads();
    for (int i = lane; i < bodies * cached_words; i += blockDim.x)
        masks.data[(i / cached_words) * words + i % cached_words] = cached[i];
#endif
""")
def _color(
    elements: wp.array(dtype=ElementInteractionData),
    active: wp.array(dtype=wp.int32),
    width: wp.int32,
    masks: wp.array2d(dtype=wp.uint32),
    row_color: wp.array(dtype=wp.int32),
    row_partition: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    cursors: wp.array(dtype=wp.int32),
    ids: wp.array(dtype=wp.int32),
    num_colors: wp.array(dtype=wp.int32),
): ...


@wp.kernel(enable_backward=False)
def color_shared(
    elements: wp.array(dtype=ElementInteractionData),
    active: wp.array(dtype=wp.int32),
    width: wp.int32,
    masks: wp.array2d(dtype=wp.uint32),
    row_color: wp.array(dtype=wp.int32),
    row_partition: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    cursors: wp.array(dtype=wp.int32),
    ids: wp.array(dtype=wp.int32),
    num_colors: wp.array(dtype=wp.int32),
):
    _color(elements, active, width, masks, row_color, row_partition, counts, starts, cursors, ids, num_colors)


def build(data, elements, active, width, device, *, rigid_only=False):
    """Use original storage for generic, CPU, or larger-body graphs."""
    if width < 1:
        raise ValueError("A color group must contain at least one color")
    if not rigid_only or not wp.get_device(device).is_cuda or data["masks"].shape[0] > 128:
        return _original_build(data, elements, active, width, device, rigid_only=rigid_only)
    if not all(array.is_contiguous for array in [elements, active, *data.values()]):
        raise ValueError("Local native prototype requires contiguous allocated arrays")
    data["masks"].zero_()
    data["counts"].zero_()
    wp.launch(
        color_shared,
        dim=32,
        inputs=[
            elements,
            active,
            width,
            *[
                data[name]
                for name in ("masks", "row_color", "row_partition", "counts", "starts", "cursors", "ids", "num_colors")
            ],
        ],
        device=device,
        block_dim=32,
    )
