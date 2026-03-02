# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Texture-based (tex3d) sparse SDF construction and sampling.

This module provides a GPU-accelerated sparse SDF implementation using 3D CUDA textures.
Construction mirrors the NanoVDB sparse-volume pattern in ``sdf_utils.py``:

1. Check subgrid occupancy by querying mesh SDF at subgrid centers
2. Build background/coarse SDF by querying mesh at subgrid corner positions
3. Populate only occupied subgrid textures by querying mesh at each texel

The format uses:
- A coarse 3D texture for background/far-field sampling
- A packed subgrid 3D texture for narrow-band high-resolution sampling
- An indirection array mapping coarse cells to subgrid blocks

Sampling uses analytical trilinear gradient computation from 8 corner texel reads,
providing exact accuracy with only 8 texture reads (vs 56 for finite differences).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .sdf_utils import get_distance_to_mesh

# ============================================================================
# Texture SDF Data Structure
# ============================================================================


class QuantizationMode:
    """Quantization modes for subgrid SDF data."""

    FLOAT32 = 4  # No quantization, full precision
    UINT16 = 2  # 16-bit quantization
    UINT8 = 1  # 8-bit quantization


@wp.struct
class TextureSDFData:
    """Sparse SDF stored in 3D CUDA textures with indirection array.

    Uses a two-level structure:
    - A coarse 3D texture for background/far-field sampling
    - A packed subgrid 3D texture for narrow-band high-resolution sampling
    - An indirection array mapping coarse cells to subgrid texture blocks
    """

    # Textures and indirection
    coarse_texture: wp.Texture3D
    subgrid_texture: wp.Texture3D
    subgrid_start_slots: wp.array(dtype=wp.uint32)

    # Grid parameters
    sdf_box_lower: wp.vec3
    sdf_box_upper: wp.vec3
    inv_sdf_dx: float
    coarse_size_x: int
    coarse_size_y: int
    coarse_size_z: int
    subgrid_size: int
    subgrid_size_f: float  # float(subgrid_size) - avoids int->float conversion
    subgrid_samples_f: float  # float(subgrid_size + 1) - samples per subgrid dimension
    fine_to_coarse: float

    # Quantization parameters for subgrid values
    subgrids_min_sdf_value: float
    subgrids_sdf_value_range: float  # max - min

    # Whether shape_scale was baked into the SDF (mirrors SDFData.scale_baked)
    scale_baked: wp.bool


# ============================================================================
# Sparse SDF Construction Kernels
# ============================================================================


@wp.func
def _idx3d(x: int, y: int, z: int, size_x: int, size_y: int) -> int:
    """Convert 3D coordinates to linear index."""
    return z * size_x * size_y + y * size_x + x


@wp.func
def _id_to_xyz(idx: int, size_x: int, size_y: int) -> wp.vec3i:
    """Convert linear index to 3D coordinates."""
    z = idx // (size_x * size_y)
    rem = idx - z * size_x * size_y
    y = rem // size_x
    x = rem - y * size_x
    return wp.vec3i(x, y, z)


@wp.kernel
def _check_subgrid_occupied_kernel(
    mesh: wp.uint64,
    subgrid_centers: wp.array(dtype=wp.vec3),
    threshold: wp.vec2f,
    winding_threshold: float,
    subgrid_required: wp.array(dtype=wp.int32),
):
    """Mark subgrids that overlap the narrow band by checking mesh SDF at center."""
    tid = wp.tid()
    sample_pos = subgrid_centers[tid]

    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]

    if is_occupied:
        subgrid_required[tid] = 1
    else:
        subgrid_required[tid] = 0


@wp.kernel
def _build_coarse_sdf_from_mesh_kernel(
    mesh: wp.uint64,
    background_sdf: wp.array(dtype=float),
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    cells_per_subgrid: int,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
    winding_threshold: float,
):
    """Populate background SDF by querying mesh at subgrid corner positions."""
    tid = wp.tid()

    total_bg = bg_size_x * bg_size_y * bg_size_z
    if tid >= total_bg:
        return

    coords = _id_to_xyz(tid, bg_size_x, bg_size_y)
    x_block = coords[0]
    y_block = coords[1]
    z_block = coords[2]

    pos = min_corner + wp.vec3(
        float(x_block * cells_per_subgrid) * cell_size[0],
        float(y_block * cells_per_subgrid) * cell_size[1],
        float(z_block * cells_per_subgrid) * cell_size[2],
    )

    background_sdf[tid] = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)


@wp.kernel
def _populate_subgrid_texture_float32_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    subgrid_texture: wp.array(dtype=float),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
):
    """Populate subgrid texture by querying mesh SDF (float32 version)."""
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = _id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = _id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = _id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[subgrid_idx] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    tex_idx = _idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = sdf_val


@wp.kernel
def _populate_subgrid_texture_uint16_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    subgrid_texture: wp.array(dtype=wp.uint16),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    """Populate subgrid texture by querying mesh SDF (uint16 quantized version)."""
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = _id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = _id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = _id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[subgrid_idx] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint16(v_normalized * 65535.0)

    tex_idx = _idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


@wp.kernel
def _populate_subgrid_texture_uint8_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32),
    subgrid_texture: wp.array(dtype=wp.uint8),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    """Populate subgrid texture by querying mesh SDF (uint8 quantized version)."""
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = _id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = _id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = _id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[subgrid_idx] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint8(v_normalized * 255.0)

    tex_idx = _idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


# ============================================================================
# Texture Sampling Functions (wp.func, used by collision kernels)
# ============================================================================


@wp.func
def apply_subgrid_start(start_slot: wp.uint32, local_f: wp.vec3, subgrid_samples_f: float) -> wp.vec3:
    """Apply subgrid block offset to local coordinates."""
    block_x = float(start_slot & wp.uint32(0x3FF))
    block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
    block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))

    return wp.vec3(
        local_f[0] + block_x * subgrid_samples_f,
        local_f[1] + block_y * subgrid_samples_f,
        local_f[2] + block_z * subgrid_samples_f,
    )


@wp.func
def apply_subgrid_sdf_scale(raw_value: float, min_value: float, value_range: float) -> float:
    """Apply quantization scale to convert normalized [0,1] value back to SDF distance."""
    return raw_value * value_range + min_value


@wp.func
def _sample_texture_at_cell(
    sdf: TextureSDFData,
    start_slot: wp.uint32,
    x_base: int,
    y_base: int,
    z_base: int,
    f: wp.vec3,
) -> float:
    """Sample SDF at a texture cell (coarse or subgrid)."""
    if start_slot == wp.uint32(0xFFFFFFFF):
        coarse_f = f * sdf.fine_to_coarse
        return wp.texture_sample(
            sdf.coarse_texture,
            wp.vec3f(coarse_f[0] + 0.5, coarse_f[1] + 0.5, coarse_f[2] + 0.5),
            dtype=float,
        )
    else:
        fx_base = float(x_base)
        fy_base = float(y_base)
        fz_base = float(z_base)
        local_x = wp.clamp(f[0] - fx_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)
        local_y = wp.clamp(f[1] - fy_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)
        local_z = wp.clamp(f[2] - fz_base * sdf.subgrid_size_f, 0.0, sdf.subgrid_samples_f)

        local_f = wp.vec3(local_x, local_y, local_z)
        tex_coords = apply_subgrid_start(start_slot, local_f, sdf.subgrid_samples_f)

        raw_val = wp.texture_sample(
            sdf.subgrid_texture,
            wp.vec3f(tex_coords[0] + 0.5, tex_coords[1] + 0.5, tex_coords[2] + 0.5),
            dtype=float,
        )
        return apply_subgrid_sdf_scale(raw_val, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)


@wp.func
def texture_sample_sdf(
    sdf: TextureSDFData,
    local_pos: wp.vec3,
) -> float:
    """Sample SDF value from texture with extrapolation for out-of-bounds points.

    Args:
        sdf: texture SDF data
        local_pos: query position in local SDF space [m]

    Returns:
        Signed distance value [m].
    """
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff_mag = wp.length(local_pos - clamped)

    f = (clamped - sdf.sdf_box_lower) * sdf.inv_sdf_dx

    x_base = wp.clamp(int(f[0] * sdf.fine_to_coarse), 0, sdf.coarse_size_x - 1)
    y_base = wp.clamp(int(f[1] * sdf.fine_to_coarse), 0, sdf.coarse_size_y - 1)
    z_base = wp.clamp(int(f[2] * sdf.fine_to_coarse), 0, sdf.coarse_size_z - 1)

    slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
    start_slot = sdf.subgrid_start_slots[slot_idx]

    sdf_val = _sample_texture_at_cell(sdf, start_slot, x_base, y_base, z_base, f)
    return sdf_val + diff_mag


@wp.func
def texture_sample_sdf_grad(
    sdf: TextureSDFData,
    local_pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    """Sample SDF value and gradient using analytical trilinear from 8 corner texels.

    For subgrid cells: reads 8 texels from the packed subgrid texture.
    For coarse cells: reads 8 texels from the coarse texture.
    Gradient is computed analytically from the trilinear partial derivatives,
    giving exact accuracy (no finite difference approximation).

    Args:
        sdf: texture SDF data
        local_pos: query position in local SDF space [m]

    Returns:
        Tuple of (distance [m], gradient [unitless]).
    """
    # Clamp to SDF box
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff = local_pos - clamped
    diff_mag = wp.length(diff)

    # Convert to fine grid coordinates
    f = (clamped - sdf.sdf_box_lower) * sdf.inv_sdf_dx

    # Fine grid vertex count per dimension (vertex-centered: subgrid_size+1 per coarse cell)
    fine_verts_x = float(sdf.coarse_size_x) * sdf.subgrid_size_f
    fine_verts_y = float(sdf.coarse_size_y) * sdf.subgrid_size_f
    fine_verts_z = float(sdf.coarse_size_z) * sdf.subgrid_size_f

    # Clamp to valid vertex range [0, fine_verts]
    fx = wp.clamp(f[0], 0.0, fine_verts_x)
    fy = wp.clamp(f[1], 0.0, fine_verts_y)
    fz = wp.clamp(f[2], 0.0, fine_verts_z)

    # Integer cell indices and fractional parts
    # Cell index must be in [0, num_fine_cells - 1]
    num_fine_cells_x = int(fine_verts_x)
    num_fine_cells_y = int(fine_verts_y)
    num_fine_cells_z = int(fine_verts_z)
    ix = wp.clamp(int(wp.floor(fx)), 0, num_fine_cells_x - 1)
    iy = wp.clamp(int(wp.floor(fy)), 0, num_fine_cells_y - 1)
    iz = wp.clamp(int(wp.floor(fz)), 0, num_fine_cells_z - 1)
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)

    # Coarse cell containing this fine cell
    x_base = wp.clamp(int(float(ix) * sdf.fine_to_coarse), 0, sdf.coarse_size_x - 1)
    y_base = wp.clamp(int(float(iy) * sdf.fine_to_coarse), 0, sdf.coarse_size_y - 1)
    z_base = wp.clamp(int(float(iz) * sdf.fine_to_coarse), 0, sdf.coarse_size_z - 1)

    # Look up indirection slot
    slot_idx = (z_base * sdf.coarse_size_y + y_base) * sdf.coarse_size_x + x_base
    start_slot = sdf.subgrid_start_slots[slot_idx]

    # -- Sample 8 corner texels --
    v000 = float(0.0)
    v100 = float(0.0)
    v010 = float(0.0)
    v110 = float(0.0)
    v001 = float(0.0)
    v101 = float(0.0)
    v011 = float(0.0)
    v111 = float(0.0)

    if start_slot == wp.uint32(0xFFFFFFFF):
        # Coarse texture: sample 8 corners at coarse cell indices
        cx = float(x_base)
        cy = float(y_base)
        cz = float(z_base)
        # Recompute fractional within coarse cell
        coarse_f = wp.vec3(fx, fy, fz) * sdf.fine_to_coarse
        tx = coarse_f[0] - cx
        ty = coarse_f[1] - cy
        tz = coarse_f[2] - cz
        # Sample at texel centers (coord + 0.5 with LINEAR = point sample)
        v000 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 0.5), dtype=float)
        v100 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 0.5), dtype=float)
        v010 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 0.5), dtype=float)
        v110 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 0.5), dtype=float)
        v001 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 1.5), dtype=float)
        v101 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 1.5), dtype=float)
        v011 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 1.5), dtype=float)
        v111 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 1.5), dtype=float)
    else:
        # Subgrid texture: sample 8 corners from packed subgrid block
        block_x = float(start_slot & wp.uint32(0x3FF))
        block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
        block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))
        tex_ox = block_x * sdf.subgrid_samples_f
        tex_oy = block_y * sdf.subgrid_samples_f
        tex_oz = block_z * sdf.subgrid_samples_f
        lx = float(ix) - float(x_base) * sdf.subgrid_size_f
        ly = float(iy) - float(y_base) * sdf.subgrid_size_f
        lz = float(iz) - float(z_base) * sdf.subgrid_size_f
        ox = tex_ox + lx + 0.5
        oy = tex_oy + ly + 0.5
        oz = tex_oz + lz + 0.5
        v000 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz), dtype=float)
        v100 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz), dtype=float)
        v010 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz), dtype=float)
        v110 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz), dtype=float)
        v001 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz + 1.0), dtype=float)
        v101 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz + 1.0), dtype=float)
        v011 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz + 1.0), dtype=float)
        v111 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz + 1.0), dtype=float)
        # Apply quantization scale (for uint16/uint8 modes)
        v000 = apply_subgrid_sdf_scale(v000, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v100 = apply_subgrid_sdf_scale(v100, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v010 = apply_subgrid_sdf_scale(v010, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v110 = apply_subgrid_sdf_scale(v110, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v001 = apply_subgrid_sdf_scale(v001, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v101 = apply_subgrid_sdf_scale(v101, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v011 = apply_subgrid_sdf_scale(v011, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v111 = apply_subgrid_sdf_scale(v111, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)

    # -- Trilinear interpolation --
    c00 = v000 + (v100 - v000) * tx
    c10 = v010 + (v110 - v010) * tx
    c01 = v001 + (v101 - v001) * tx
    c11 = v011 + (v111 - v011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    sdf_val = c0 + (c1 - c0) * tz

    # -- Analytical gradient (partial derivatives of trilinear) --
    omtx = 1.0 - tx
    omty = 1.0 - ty
    omtz = 1.0 - tz

    gx = omty * omtz * (v100 - v000) + ty * omtz * (v110 - v010) + omty * tz * (v101 - v001) + ty * tz * (v111 - v011)
    gy = omtx * omtz * (v010 - v000) + tx * omtz * (v110 - v100) + omtx * tz * (v011 - v001) + tx * tz * (v111 - v101)
    gz = omtx * omty * (v001 - v000) + tx * omty * (v101 - v100) + omtx * ty * (v011 - v010) + tx * ty * (v111 - v110)

    # Gradient is in grid coordinates; convert to world coordinates
    grad = wp.vec3(gx, gy, gz) * sdf.inv_sdf_dx

    # Handle extrapolation for points outside the SDF box
    if diff_mag > 0.0:
        sdf_val = sdf_val + diff_mag
        grad = diff / diff_mag

    return sdf_val, grad


# ============================================================================
# Host-side Construction Functions
# ============================================================================


def build_sparse_sdf_from_mesh(
    mesh: wp.Mesh,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    cell_size: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    subgrid_size: int = 8,
    narrow_band_thickness: float = 0.1,
    quantization_mode: int = QuantizationMode.FLOAT32,
    winding_threshold: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Build sparse SDF texture representation by querying mesh directly.

    Mirrors the NanoVDB sparse-volume construction pattern: check subgrid
    occupancy at centers, then populate only occupied subgrids.

    Args:
        mesh: Warp mesh with ``support_winding_number=True``.
        grid_size_x: fine grid X dimension [sample].
        grid_size_y: fine grid Y dimension [sample].
        grid_size_z: fine grid Z dimension [sample].
        cell_size: fine grid cell size per axis [m].
        min_corner: lower corner of domain [m].
        max_corner: upper corner of domain [m].
        subgrid_size: cells per subgrid.
        narrow_band_thickness: distance threshold for subgrids [m].
        quantization_mode: :class:`QuantizationMode` value.
        winding_threshold: winding number threshold for inside/outside.
        device: Warp device string.

    Returns:
        Dictionary with all sparse SDF data.
    """
    w = (grid_size_x - 1) // subgrid_size
    h = (grid_size_y - 1) // subgrid_size
    d = (grid_size_z - 1) // subgrid_size
    total_subgrids = w * h * d

    min_corner_wp = wp.vec3(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
    cell_size_wp = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))

    # Build background SDF (coarse grid) by querying mesh at subgrid corners
    bg_size_x = w + 1
    bg_size_y = h + 1
    bg_size_z = d + 1
    total_bg = bg_size_x * bg_size_y * bg_size_z

    background_sdf = wp.zeros(total_bg, dtype=float, device=device)

    wp.launch(
        _build_coarse_sdf_from_mesh_kernel,
        dim=total_bg,
        inputs=[
            mesh.id,
            background_sdf,
            min_corner_wp,
            cell_size_wp,
            subgrid_size,
            bg_size_x,
            bg_size_y,
            bg_size_z,
            winding_threshold,
        ],
        device=device,
    )

    # Check subgrid occupancy by querying mesh SDF at subgrid centers
    subgrid_centers = np.empty((total_subgrids, 3), dtype=np.float32)
    for idx in range(total_subgrids):
        bz = idx // (w * h)
        rem = idx - bz * w * h
        by = rem // w
        bx = rem - by * w
        subgrid_centers[idx, 0] = (bx * subgrid_size + subgrid_size * 0.5) * cell_size[0] + min_corner[0]
        subgrid_centers[idx, 1] = (by * subgrid_size + subgrid_size * 0.5) * cell_size[1] + min_corner[1]
        subgrid_centers[idx, 2] = (bz * subgrid_size + subgrid_size * 0.5) * cell_size[2] + min_corner[2]

    subgrid_centers_gpu = wp.array(subgrid_centers, dtype=wp.vec3, device=device)

    # Expand threshold by subgrid half-diagonal (same pattern as NanoVDB tiles)
    half_subgrid = subgrid_size * 0.5 * cell_size
    subgrid_radius = float(np.linalg.norm(half_subgrid))
    threshold = wp.vec2f(-narrow_band_thickness - subgrid_radius, narrow_band_thickness + subgrid_radius)

    subgrid_required = wp.zeros(total_subgrids, dtype=wp.int32, device=device)

    wp.launch(
        _check_subgrid_occupied_kernel,
        dim=total_subgrids,
        inputs=[mesh.id, subgrid_centers_gpu, threshold, winding_threshold, subgrid_required],
        device=device,
    )
    wp.synchronize()

    # Exclusive scan to assign sequential addresses to required subgrids
    subgrid_addresses = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    wp._src.utils.array_scan(subgrid_required, subgrid_addresses, inclusive=False)
    wp.synchronize()

    required_np = subgrid_required.numpy()
    num_required = int(np.sum(required_np))

    # Conservative quantization bounds from narrow band range
    global_sdf_min = -narrow_band_thickness - subgrid_radius
    global_sdf_max = narrow_band_thickness + subgrid_radius
    sdf_range = global_sdf_max - global_sdf_min
    if sdf_range < 1e-10:
        sdf_range = 1.0

    if num_required == 0:
        subgrid_start_slots = np.full(total_subgrids, 0xFFFFFFFF, dtype=np.uint32)
        subgrid_texture_data = np.zeros((1, 1, 1), dtype=np.float32)
        tex_size = 1
        final_sdf_min = 0.0
        final_sdf_range = 1.0
    else:
        cubic_root = num_required ** (1.0 / 3.0)
        tex_blocks_per_dim = max(1, int(np.ceil(cubic_root)))
        while tex_blocks_per_dim**3 < num_required:
            tex_blocks_per_dim += 1

        samples_per_dim = subgrid_size + 1
        tex_size = tex_blocks_per_dim * samples_per_dim

        subgrid_start_slots = np.full(total_subgrids, 0xFFFFFFFF, dtype=np.uint32)
        subgrid_start_slots_gpu = wp.array(subgrid_start_slots, dtype=wp.uint32, device=device)

        total_tex_samples = tex_size * tex_size * tex_size
        samples_per_subgrid = samples_per_dim**3
        total_work = total_subgrids * samples_per_subgrid

        sdf_range_inv = 1.0 / sdf_range

        if quantization_mode == QuantizationMode.FLOAT32:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=float, device=device)
            wp.launch(
                _populate_subgrid_texture_float32_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                ],
                device=device,
            )
            final_sdf_min = 0.0
            final_sdf_range = 1.0
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))
            subgrid_texture_data = subgrid_texture_data.astype(np.float32)

        elif quantization_mode == QuantizationMode.UINT16:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint16, device=device)
            wp.launch(
                _populate_subgrid_texture_uint16_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            uint16_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))
            subgrid_texture_data = uint16_data.astype(np.float32) / 65535.0

        elif quantization_mode == QuantizationMode.UINT8:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint8, device=device)
            wp.launch(
                _populate_subgrid_texture_uint8_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            uint8_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))
            subgrid_texture_data = uint8_data.astype(np.float32) / 255.0

        else:
            raise ValueError(f"Unknown quantization mode: {quantization_mode}")

        wp.synchronize()
        subgrid_start_slots = subgrid_start_slots_gpu.numpy()

    background_sdf_np = background_sdf.numpy().reshape((bg_size_z, bg_size_y, bg_size_x))

    return {
        "coarse_sdf": background_sdf_np.astype(np.float32),
        "subgrid_data": subgrid_texture_data.astype(np.float32),
        "subgrid_start_slots": subgrid_start_slots,
        "coarse_dims": (w, h, d),
        "subgrid_tex_size": tex_size,
        "num_subgrids": num_required,
        "min_extents": min_corner,
        "max_extents": max_corner,
        "cell_size": cell_size,
        "subgrid_size": subgrid_size,
        "quantization_mode": quantization_mode,
        "subgrids_min_sdf_value": final_sdf_min,
        "subgrids_sdf_value_range": final_sdf_range,
    }


def create_sparse_sdf_textures(
    sparse_data: dict,
    device: str = "cuda",
) -> tuple[TextureSDFData, wp.Texture3D, wp.Texture3D]:
    """Create TextureSDFData struct with GPU textures from sparse data.

    Args:
        sparse_data: dictionary from :func:`build_sparse_sdf_from_mesh`.
        device: Warp device string.

    Returns:
        Tuple of ``(texture_sdf, coarse_texture, subgrid_texture)``.
        Caller must keep texture references alive to prevent GC.
    """
    coarse_tex = wp.Texture3D(
        sparse_data["coarse_sdf"],
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    subgrid_tex = wp.Texture3D(
        sparse_data["subgrid_data"],
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    subgrid_slots = wp.array(sparse_data["subgrid_start_slots"], dtype=wp.uint32, device=device)

    avg_spacing = float(np.mean(sparse_data["cell_size"]))
    coarse_x = sparse_data["coarse_dims"][0]
    coarse_y = sparse_data["coarse_dims"][1]
    coarse_z = sparse_data["coarse_dims"][2]

    sdf_params = TextureSDFData()
    sdf_params.coarse_texture = coarse_tex
    sdf_params.subgrid_texture = subgrid_tex
    sdf_params.subgrid_start_slots = subgrid_slots
    sdf_params.sdf_box_lower = wp.vec3(
        float(sparse_data["min_extents"][0]),
        float(sparse_data["min_extents"][1]),
        float(sparse_data["min_extents"][2]),
    )
    sdf_params.sdf_box_upper = wp.vec3(
        float(sparse_data["max_extents"][0]),
        float(sparse_data["max_extents"][1]),
        float(sparse_data["max_extents"][2]),
    )
    sdf_params.inv_sdf_dx = 1.0 / avg_spacing
    sdf_params.coarse_size_x = coarse_x
    sdf_params.coarse_size_y = coarse_y
    sdf_params.coarse_size_z = coarse_z
    sdf_params.subgrid_size = sparse_data["subgrid_size"]
    sdf_params.subgrid_size_f = float(sparse_data["subgrid_size"])
    sdf_params.subgrid_samples_f = float(sparse_data["subgrid_size"] + 1)
    sdf_params.fine_to_coarse = 1.0 / sparse_data["subgrid_size"]
    sdf_params.subgrids_min_sdf_value = sparse_data["subgrids_min_sdf_value"]
    sdf_params.subgrids_sdf_value_range = sparse_data["subgrids_sdf_value_range"]
    sdf_params.scale_baked = False

    return sdf_params, coarse_tex, subgrid_tex


def create_texture_sdf_from_mesh(
    mesh: wp.Mesh,
    *,
    margin: float = 0.05,
    narrow_band_range: tuple[float, float] = (-0.1, 0.1),
    max_resolution: int = 64,
    subgrid_size: int = 8,
    quantization_mode: int = QuantizationMode.FLOAT32,
    winding_threshold: float = 0.5,
    scale_baked: bool = False,
    device: str | None = None,
) -> tuple[TextureSDFData, wp.Texture3D, wp.Texture3D]:
    """Create texture SDF from a Warp mesh.

    This is the main entry point for texture SDF construction. It mirrors the
    parameters of :func:`~newton._src.geometry.sdf_utils._compute_sdf_from_shape_impl`.

    Args:
        mesh: Warp mesh with ``support_winding_number=True``.
        margin: extra AABB padding [m].
        narrow_band_range: signed narrow-band distance range [m] as ``(inner, outer)``.
        max_resolution: maximum grid dimension [voxel].
        subgrid_size: cells per subgrid.
        quantization_mode: :class:`QuantizationMode` value.
        winding_threshold: winding number threshold for inside/outside classification.
        scale_baked: whether shape scale was baked into the mesh vertices.
        device: Warp device string. ``None`` uses the mesh's device.

    Returns:
        Tuple of ``(texture_sdf, coarse_texture, subgrid_texture)``.
        Caller must keep texture references alive to prevent GC.
    """
    if device is None:
        device = str(mesh.device)

    points_np = mesh.points.numpy()
    mesh_min = np.min(points_np, axis=0)
    mesh_max = np.max(points_np, axis=0)

    min_ext = mesh_min - margin
    max_ext = mesh_max + margin

    # Compute grid dimensions (same math as the former build_dense_sdf)
    ext = max_ext - min_ext
    max_ext_scalar = np.max(ext)
    cell_size_scalar = max_ext_scalar / max_resolution
    dims = np.ceil(ext / cell_size_scalar).astype(int) + 1
    grid_x, grid_y, grid_z = int(dims[0]), int(dims[1]), int(dims[2])
    cell_size = ext / (dims - 1)

    narrow_band_thickness = max(abs(narrow_band_range[0]), abs(narrow_band_range[1]))

    sparse_data = build_sparse_sdf_from_mesh(
        mesh,
        grid_x,
        grid_y,
        grid_z,
        cell_size,
        min_ext,
        max_ext,
        subgrid_size=subgrid_size,
        narrow_band_thickness=narrow_band_thickness,
        quantization_mode=quantization_mode,
        winding_threshold=winding_threshold,
        device=device,
    )

    sdf_params, coarse_tex, subgrid_tex = create_sparse_sdf_textures(sparse_data, device)
    sdf_params.scale_baked = scale_baked

    return sdf_params, coarse_tex, subgrid_tex


def create_empty_texture_sdf_data() -> TextureSDFData:
    """Return an empty TextureSDFData struct for shapes without texture SDF.

    An empty struct has ``coarse_size_x == 0``, which collision kernels
    use to detect the absence of a texture SDF and fall back to BVH.

    Returns:
        A zeroed-out :class:`TextureSDFData` struct.
    """
    sdf = TextureSDFData()
    # Zero-size signals "no texture SDF available"
    sdf.coarse_size_x = 0
    sdf.coarse_size_y = 0
    sdf.coarse_size_z = 0
    sdf.subgrid_size = 0
    sdf.subgrid_size_f = 0.0
    sdf.subgrid_samples_f = 0.0
    sdf.fine_to_coarse = 0.0
    sdf.inv_sdf_dx = 0.0
    sdf.sdf_box_lower = wp.vec3(0.0, 0.0, 0.0)
    sdf.sdf_box_upper = wp.vec3(0.0, 0.0, 0.0)
    sdf.subgrids_min_sdf_value = 0.0
    sdf.subgrids_sdf_value_range = 1.0
    sdf.scale_baked = False
    return sdf
