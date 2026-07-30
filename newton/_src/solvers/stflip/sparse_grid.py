# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capacity-bounded packed sparse grid with no halo-cell storage.

The topology follows the PhysX particle-isosurface sparse-grid organization:
occupied subgrids use sorted, collision-free keys and each subgrid stores a
3x3x3 neighbor table. See ``newton/licenses/physx-sparse-grid-LICENSE.txt``.
Unlike the optional PhysX halo layout, every field in this module stores only
core cells. Cross-subgrid stencils resolve their owner through the neighbor
table.
"""

from __future__ import annotations

import warp as wp

from ...core.types import Devicelike

_KEY_BITS = wp.constant(21)
_KEY_SHIFT_X = wp.constant(wp.int64(42))
_KEY_SHIFT_Y = wp.constant(wp.int64(21))
_KEY_BIAS = wp.constant(1 << 20)
_KEY_MASK = wp.constant(wp.int64((1 << 21) - 1))
_KEY_MIN_COORD = wp.constant(-(1 << 20))
_KEY_MAX_COORD = wp.constant((1 << 20) - 2)
_KEY_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))

_STATUS_SUCCESS = 0
_STATUS_TILE_CAPACITY_EXCEEDED = 1 << 0
_STATUS_COORDINATE_OUT_OF_RANGE = 1 << 1


@wp.func
def _floor_div(value: int, divisor: int) -> int:
    """Return mathematical floor division for a positive divisor."""
    if value >= 0:
        return value // divisor
    return -((-value + divisor - 1) // divisor)


@wp.func
def _tile_key(coords: wp.vec3i) -> wp.int64:
    """Pack one signed tile coordinate into a sortable 63-bit key."""
    x = wp.int64(coords[0] + _KEY_BIAS)
    y = wp.int64(coords[1] + _KEY_BIAS)
    z = wp.int64(coords[2] + _KEY_BIAS)
    return (x << _KEY_SHIFT_X) | (y << _KEY_SHIFT_Y) | z


@wp.func
def _tile_coords(key: wp.int64) -> wp.vec3i:
    """Unpack one sparse-grid tile key."""
    x = int((key >> _KEY_SHIFT_X) & _KEY_MASK) - _KEY_BIAS
    y = int((key >> _KEY_SHIFT_Y) & _KEY_MASK) - _KEY_BIAS
    z = int(key & _KEY_MASK) - _KEY_BIAS
    return wp.vec3i(x, y, z)


@wp.func
def _neighbor_slot(x: int, y: int, z: int) -> int:
    """Return the x-major slot for an offset in ``[-1, 1]^3``."""
    return (x + 1) + 3 * ((y + 1) + 3 * (z + 1))


@wp.func
def _find_key(keys: wp.array[wp.int64], count: int, key: wp.int64) -> int:
    """Find a key in a sorted prefix, returning -1 when absent."""
    lo = int(0)
    hi = int(count)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        value = keys[mid]
        if value < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < count and keys[lo] == key:
        return lo
    return -1


@wp.struct
class SparseGridData:
    """Device view of a packed sparse-grid topology."""

    tile_keys: wp.array[wp.int64]
    tile_neighbors: wp.array[wp.int32]
    tile_count: wp.array[wp.int32]
    tile_size: int


@wp.func
def sparse_grid_find_tile(data: SparseGridData, tile_coords: wp.vec3i) -> int:
    """Find the packed index of a tile coordinate."""
    if (
        tile_coords[0] < _KEY_MIN_COORD
        or tile_coords[0] > _KEY_MAX_COORD
        or tile_coords[1] < _KEY_MIN_COORD
        or tile_coords[1] > _KEY_MAX_COORD
        or tile_coords[2] < _KEY_MIN_COORD
        or tile_coords[2] > _KEY_MAX_COORD
    ):
        return -1
    return _find_key(data.tile_keys, data.tile_count[0], _tile_key(tile_coords))


@wp.func
def sparse_grid_index_from_cell(data: SparseGridData, cell: wp.vec3i) -> int:
    """Resolve a global integer cell coordinate to its packed core index."""
    tile_size = data.tile_size
    tile_coords = wp.vec3i(
        _floor_div(cell[0], tile_size),
        _floor_div(cell[1], tile_size),
        _floor_div(cell[2], tile_size),
    )
    tile = sparse_grid_find_tile(data, tile_coords)
    if tile < 0:
        return -1
    x = cell[0] - tile_coords[0] * tile_size
    y = cell[1] - tile_coords[1] * tile_size
    z = cell[2] - tile_coords[2] * tile_size
    return x + tile_size * (y + tile_size * (z + tile_size * tile))


@wp.func
def sparse_grid_cell_coord(data: SparseGridData, index: int) -> wp.vec3i:
    """Return the global integer coordinate of a packed core cell."""
    tile_volume = data.tile_size * data.tile_size * data.tile_size
    tile = index // tile_volume
    local = index - tile * tile_volume
    x = local % data.tile_size
    y = (local // data.tile_size) % data.tile_size
    z = local // (data.tile_size * data.tile_size)
    tile_coords = _tile_coords(data.tile_keys[tile])
    return wp.vec3i(
        tile_coords[0] * data.tile_size + x,
        tile_coords[1] * data.tile_size + y,
        tile_coords[2] * data.tile_size + z,
    )


@wp.func
def sparse_grid_cell_index(data: SparseGridData, tile: int, x: int, y: int, z: int) -> int:
    """Resolve a core-cell index, including a one-tile neighbor crossing."""
    tile_size = data.tile_size
    tile_dx = _floor_div(x, tile_size)
    tile_dy = _floor_div(y, tile_size)
    tile_dz = _floor_div(z, tile_size)
    if tile_dx < -1 or tile_dx > 1 or tile_dy < -1 or tile_dy > 1 or tile_dz < -1 or tile_dz > 1:
        return -1

    owner = tile
    if tile_dx != 0 or tile_dy != 0 or tile_dz != 0:
        owner = data.tile_neighbors[27 * tile + _neighbor_slot(tile_dx, tile_dy, tile_dz)]
    if owner < 0:
        return -1

    local_x = x - tile_dx * tile_size
    local_y = y - tile_dy * tile_size
    local_z = z - tile_dz * tile_size
    return local_x + tile_size * (local_y + tile_size * (local_z + tile_size * owner))


@wp.kernel(enable_backward=False)
def _prepare_point_keys(
    positions: wp.array[wp.vec3],
    active: wp.array[wp.int32],
    origin: wp.vec3,
    inv_cell_size: float,
    tile_size: int,
    keys: wp.array[wp.int64],
    indices: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    point = wp.tid()
    indices[point] = point
    if active.shape[0] > 0 and active[point] == 0:
        keys[point] = _KEY_SENTINEL
        return

    p = (positions[point] - origin) * inv_cell_size
    cell = wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))
    tile = wp.vec3i(
        _floor_div(cell[0], tile_size),
        _floor_div(cell[1], tile_size),
        _floor_div(cell[2], tile_size),
    )
    if (
        tile[0] < _KEY_MIN_COORD
        or tile[0] > _KEY_MAX_COORD
        or tile[1] < _KEY_MIN_COORD
        or tile[1] > _KEY_MAX_COORD
        or tile[2] < _KEY_MIN_COORD
        or tile[2] > _KEY_MAX_COORD
    ):
        keys[point] = _KEY_SENTINEL
        wp.atomic_or(status, 0, _STATUS_COORDINATE_OUT_OF_RANGE)
        return
    keys[point] = _tile_key(tile)


@wp.kernel(enable_backward=False)
def _mark_unique_keys(
    sorted_keys: wp.array[wp.int64],
    unique_flags: wp.array[wp.int32],
):
    index = wp.tid()
    key = sorted_keys[index]
    unique_flags[index] = int(key != _KEY_SENTINEL and (index == 0 or key != sorted_keys[index - 1]))


@wp.kernel(enable_backward=False)
def _compact_unique_keys(
    sorted_keys: wp.array[wp.int64],
    unique_flags: wp.array[wp.int32],
    unique_offsets: wp.array[wp.int32],
    point_capacity: int,
    tile_capacity: int,
    tile_keys: wp.array[wp.int64],
    tile_count: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    index = wp.tid()
    if unique_flags[index] != 0:
        output = unique_offsets[index]
        if output < tile_capacity:
            tile_keys[output] = sorted_keys[index]
    if index == point_capacity - 1:
        count = unique_offsets[index] + unique_flags[index]
        if count > tile_capacity:
            wp.atomic_or(status, 0, _STATUS_TILE_CAPACITY_EXCEEDED)
        tile_count[0] = wp.min(count, tile_capacity)


@wp.kernel(enable_backward=False)
def _clear_inactive_tile_keys(
    tile_keys: wp.array[wp.int64],
    tile_count: wp.array[wp.int32],
):
    index = wp.tid()
    if index >= tile_count[0]:
        tile_keys[index] = _KEY_SENTINEL


@wp.kernel(enable_backward=False)
def _prepare_neighbor_candidates(
    tile_keys: wp.array[wp.int64],
    tile_count: wp.array[wp.int32],
    candidate_keys: wp.array[wp.int64],
    candidate_indices: wp.array[wp.int32],
):
    index = wp.tid()
    candidate_indices[index] = index
    tile = index // 27
    slot = index - tile * 27
    if tile >= tile_count[0]:
        candidate_keys[index] = _KEY_SENTINEL
        return

    dz = slot // 9 - 1
    rem = slot - (dz + 1) * 9
    dy = rem // 3 - 1
    dx = rem - (dy + 1) * 3 - 1
    coords = _tile_coords(tile_keys[tile]) + wp.vec3i(dx, dy, dz)
    if (
        coords[0] < _KEY_MIN_COORD
        or coords[0] > _KEY_MAX_COORD
        or coords[1] < _KEY_MIN_COORD
        or coords[1] > _KEY_MAX_COORD
        or coords[2] < _KEY_MIN_COORD
        or coords[2] > _KEY_MAX_COORD
    ):
        candidate_keys[index] = _KEY_SENTINEL
    else:
        candidate_keys[index] = _tile_key(coords)


@wp.kernel(enable_backward=False)
def _build_neighbors(
    tile_keys: wp.array[wp.int64],
    tile_count: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
):
    index = wp.tid()
    tile = index // 27
    slot = index - tile * 27
    count = tile_count[0]
    if tile >= count:
        neighbors[index] = -1
        return

    dz = slot // 9 - 1
    rem = slot - (dz + 1) * 9
    dy = rem // 3 - 1
    dx = rem - (dy + 1) * 3 - 1
    coords = _tile_coords(tile_keys[tile]) + wp.vec3i(dx, dy, dz)
    if (
        coords[0] < _KEY_MIN_COORD
        or coords[0] > _KEY_MAX_COORD
        or coords[1] < _KEY_MIN_COORD
        or coords[1] > _KEY_MAX_COORD
        or coords[2] < _KEY_MIN_COORD
        or coords[2] > _KEY_MAX_COORD
    ):
        neighbors[index] = -1
    else:
        neighbors[index] = _find_key(tile_keys, count, _tile_key(coords))


@wp.kernel(enable_backward=False)
def _map_sorted_points_to_tiles(
    sorted_keys: wp.array[wp.int64],
    tile_keys: wp.array[wp.int64],
    tile_count: wp.array[wp.int32],
    point_tile: wp.array[wp.int32],
):
    point = wp.tid()
    key = sorted_keys[point]
    if key == _KEY_SENTINEL:
        point_tile[point] = -1
    else:
        point_tile[point] = _find_key(tile_keys, tile_count[0], key)


class SparseGrid:
    """Build and store a capacity-bounded packed sparse-grid topology.

    The first ``tile_count[0]`` entries of :attr:`tile_keys` are sorted active
    keys. Every active tile owns exactly ``tile_size**3`` field entries.
    Neighbor slots use x-major offsets over ``[-1, 1]^3``; slot 13 is the tile
    itself.
    """

    STATUS_SUCCESS = _STATUS_SUCCESS
    STATUS_TILE_CAPACITY_EXCEEDED = _STATUS_TILE_CAPACITY_EXCEEDED
    STATUS_COORDINATE_OUT_OF_RANGE = _STATUS_COORDINATE_OUT_OF_RANGE

    def __init__(
        self,
        *,
        point_capacity: int,
        tile_capacity: int,
        tile_size: int = 16,
        cell_size: float = 1.0,
        padding_tiles: int = 0,
        origin: wp.vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: Devicelike = None,
    ):
        if point_capacity <= 0:
            raise ValueError("point_capacity must be positive")
        if tile_capacity <= 0:
            raise ValueError("tile_capacity must be positive")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if padding_tiles < 0:
            raise ValueError("padding_tiles must be non-negative")

        self.point_capacity = int(point_capacity)
        self.tile_capacity = int(tile_capacity)
        self.tile_size = int(tile_size)
        self.cell_size = float(cell_size)
        self.padding_tiles = int(padding_tiles)
        self.origin = wp.vec3(origin)
        self.device = wp.get_device(device)

        with wp.ScopedDevice(self.device):
            # Warp radix sort uses the second half of each array as scratch.
            self.sorted_point_keys = wp.full(
                2 * self.point_capacity,
                value=_KEY_SENTINEL,
                dtype=wp.int64,
            )
            self.sorted_point_indices = wp.zeros(2 * self.point_capacity, dtype=wp.int32)
            self.unique_flags = wp.zeros(self.point_capacity, dtype=wp.int32)
            self.unique_offsets = wp.zeros(self.point_capacity, dtype=wp.int32)
            if self.padding_tiles > 0:
                self.candidate_capacity = 27 * self.tile_capacity
                self.candidate_keys = wp.full(
                    2 * self.candidate_capacity,
                    value=_KEY_SENTINEL,
                    dtype=wp.int64,
                )
                self.candidate_indices = wp.zeros(2 * self.candidate_capacity, dtype=wp.int32)
                self.candidate_unique_flags = wp.zeros(self.candidate_capacity, dtype=wp.int32)
                self.candidate_unique_offsets = wp.zeros(self.candidate_capacity, dtype=wp.int32)
            else:
                self.candidate_capacity = 0
                self.candidate_keys = wp.empty(0, dtype=wp.int64)
                self.candidate_indices = wp.empty(0, dtype=wp.int32)
                self.candidate_unique_flags = wp.empty(0, dtype=wp.int32)
                self.candidate_unique_offsets = wp.empty(0, dtype=wp.int32)

            self.tile_keys = wp.full(self.tile_capacity, value=_KEY_SENTINEL, dtype=wp.int64)
            self.tile_neighbors = wp.full(27 * self.tile_capacity, value=-1, dtype=wp.int32)
            self.tile_count = wp.zeros(1, dtype=wp.int32)
            self.sorted_point_tiles = wp.full(self.point_capacity, value=-1, dtype=wp.int32)
            self.status = wp.zeros(1, dtype=wp.int32)
            self._empty_active = wp.empty(0, dtype=wp.int32)

        self.data = SparseGridData()
        self.data.tile_keys = self.tile_keys
        self.data.tile_neighbors = self.tile_neighbors
        self.data.tile_count = self.tile_count
        self.data.tile_size = self.tile_size

    @property
    def cell_capacity(self) -> int:
        """Maximum number of stored core cells."""
        return self.tile_capacity * self.tile_size**3

    def build(self, positions: wp.array[wp.vec3], active: wp.array[wp.int32] | None = None) -> None:
        """Rebuild the occupied-tile topology from fixed-capacity points."""
        if positions.device != self.device:
            raise ValueError(f"positions are on {positions.device}, expected {self.device}")
        if positions.shape[0] != self.point_capacity:
            raise ValueError(f"positions contain {positions.shape[0]} points, expected {self.point_capacity}")
        if active is None:
            active = self._empty_active
        elif active.device != self.device or active.shape[0] != self.point_capacity:
            raise ValueError("active must match positions shape and device")

        self.status.zero_()
        wp.launch(
            _prepare_point_keys,
            dim=self.point_capacity,
            inputs=[
                positions,
                active,
                self.origin,
                1.0 / self.cell_size,
                self.tile_size,
                self.sorted_point_keys,
                self.sorted_point_indices,
                self.status,
            ],
            device=self.device,
        )
        wp.utils.radix_sort_pairs(
            keys=self.sorted_point_keys,
            values=self.sorted_point_indices,
            count=self.point_capacity,
        )
        wp.launch(
            _mark_unique_keys,
            dim=self.point_capacity,
            inputs=[self.sorted_point_keys, self.unique_flags],
            device=self.device,
        )
        wp.utils.array_scan(self.unique_flags, self.unique_offsets, inclusive=False)
        wp.launch(
            _compact_unique_keys,
            dim=self.point_capacity,
            inputs=[
                self.sorted_point_keys,
                self.unique_flags,
                self.unique_offsets,
                self.point_capacity,
                self.tile_capacity,
                self.tile_keys,
                self.tile_count,
                self.status,
            ],
            device=self.device,
        )
        wp.launch(
            _clear_inactive_tile_keys,
            dim=self.tile_capacity,
            inputs=[self.tile_keys, self.tile_count],
            device=self.device,
        )
        for _ in range(self.padding_tiles):
            wp.launch(
                _prepare_neighbor_candidates,
                dim=self.candidate_capacity,
                inputs=[
                    self.tile_keys,
                    self.tile_count,
                    self.candidate_keys,
                    self.candidate_indices,
                ],
                device=self.device,
            )
            wp.utils.radix_sort_pairs(
                keys=self.candidate_keys,
                values=self.candidate_indices,
                count=self.candidate_capacity,
            )
            wp.launch(
                _mark_unique_keys,
                dim=self.candidate_capacity,
                inputs=[self.candidate_keys, self.candidate_unique_flags],
                device=self.device,
            )
            wp.utils.array_scan(
                self.candidate_unique_flags,
                self.candidate_unique_offsets,
                inclusive=False,
            )
            wp.launch(
                _compact_unique_keys,
                dim=self.candidate_capacity,
                inputs=[
                    self.candidate_keys,
                    self.candidate_unique_flags,
                    self.candidate_unique_offsets,
                    self.candidate_capacity,
                    self.tile_capacity,
                    self.tile_keys,
                    self.tile_count,
                    self.status,
                ],
                device=self.device,
            )
            wp.launch(
                _clear_inactive_tile_keys,
                dim=self.tile_capacity,
                inputs=[self.tile_keys, self.tile_count],
                device=self.device,
            )
        wp.launch(
            _build_neighbors,
            dim=27 * self.tile_capacity,
            inputs=[self.tile_keys, self.tile_count, self.tile_neighbors],
            device=self.device,
        )
        wp.launch(
            _map_sorted_points_to_tiles,
            dim=self.point_capacity,
            inputs=[
                self.sorted_point_keys,
                self.tile_keys,
                self.tile_count,
                self.sorted_point_tiles,
            ],
            device=self.device,
        )

    def check_status(self) -> None:
        """Raise after synchronizing if the latest topology build failed."""
        status = int(self.status.numpy()[0])
        if status & _STATUS_COORDINATE_OUT_OF_RANGE:
            raise RuntimeError("Sparse-grid tile coordinate exceeded the supported signed 21-bit range")
        if status & _STATUS_TILE_CAPACITY_EXCEEDED:
            raise RuntimeError(f"Sparse-grid tile capacity {self.tile_capacity} was exceeded; increase tile_capacity")
