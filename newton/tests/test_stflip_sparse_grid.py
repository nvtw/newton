# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.stflip.sparse_grid import (
    SparseGrid,
    SparseGridData,
    sparse_grid_cell_index,
    sparse_grid_index_from_cell,
)


@wp.kernel
def _resolve_test_cells(
    grid: SparseGridData,
    tile: int,
    coordinates: wp.array[wp.vec3i],
    result: wp.array[wp.int32],
):
    index = wp.tid()
    cell = coordinates[index]
    result[index] = sparse_grid_cell_index(grid, tile, cell[0], cell[1], cell[2])


@wp.kernel
def _resolve_global_test_cells(
    grid: SparseGridData,
    coordinates: wp.array[wp.vec3i],
    result: wp.array[wp.int32],
):
    index = wp.tid()
    result[index] = sparse_grid_index_from_cell(grid, coordinates[index])


def _decode_tile_keys(grid):
    keys = grid.tile_keys.numpy()[: int(grid.tile_count.numpy()[0])].astype(np.int64)
    mask = (1 << 21) - 1
    bias = 1 << 20
    return np.column_stack(
        (
            ((keys >> 42) & mask) - bias,
            ((keys >> 21) & mask) - bias,
            (keys & mask) - bias,
        )
    ).astype(np.int32)


def _reference_tiles(positions, active, cell_size, tile_size, padding_tiles):
    occupied = {tuple(v) for v in np.floor(positions / (cell_size * tile_size)).astype(np.int32)[active != 0]}
    result = occupied
    for _ in range(padding_tiles):
        result = {
            (x + dx, y + dy, z + dz)
            for x, y, z in result
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        }
    return sorted(result)


class TestSparseGrid(unittest.TestCase):
    def test_build_sorted_tiles_and_neighbors(self):
        """Build deterministic tiles and connect adjacent neighbors."""
        positions = wp.array(
            [
                (-0.1, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (4.1, 0.0, 0.0),
                (4.2, 0.0, 0.0),
            ],
            dtype=wp.vec3,
            device="cpu",
        )
        grid = SparseGrid(
            point_capacity=4,
            tile_capacity=4,
            tile_size=4,
            cell_size=1.0,
            device="cpu",
        )
        grid.build(positions)
        grid.check_status()

        self.assertEqual(int(grid.tile_count.numpy()[0]), 3)
        neighbors = grid.tile_neighbors.numpy().reshape(grid.tile_capacity, 27)
        self.assertEqual(neighbors[0, 14], 1)
        self.assertEqual(neighbors[1, 12], 0)
        self.assertEqual(neighbors[1, 14], 2)
        self.assertEqual(neighbors[2, 12], 1)
        self.assertTrue(np.all(neighbors[:3, 13] == np.arange(3)))

    def test_resolve_cross_tile_core_cells(self):
        """Resolve canonical core cells without allocating halo entries."""
        positions = wp.array([(0.1, 0.0, 0.0), (4.1, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        grid = SparseGrid(
            point_capacity=2,
            tile_capacity=2,
            tile_size=4,
            cell_size=1.0,
            device="cpu",
        )
        grid.build(positions)

        coordinates = wp.array([(3, 0, 0), (4, 0, 0), (-1, 0, 0), (8, 0, 0)], dtype=wp.vec3i, device="cpu")
        result = wp.empty(4, dtype=wp.int32, device="cpu")
        wp.launch(_resolve_test_cells, dim=4, inputs=[grid.data, 0, coordinates, result], device="cpu")

        values = result.numpy()
        self.assertEqual(values[0], 3)
        self.assertEqual(values[1], 4**3)
        self.assertEqual(values[2], -1)
        self.assertEqual(values[3], -1)
        self.assertEqual(grid.cell_capacity, 2 * 4**3)

    def test_filter_inactive_points(self):
        """Exclude inactive particles from sparse topology construction."""
        positions = wp.array([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        active = wp.array([1, 0], dtype=wp.int32, device="cpu")
        grid = SparseGrid(point_capacity=2, tile_capacity=2, tile_size=4, device="cpu")
        grid.build(positions, active)
        grid.check_status()

        self.assertEqual(int(grid.tile_count.numpy()[0]), 1)

    def test_filter_flag_array_with_active_mask(self):
        """Select active points from a packed integer flag array."""
        active_flag = int(newton.ParticleFlags.ACTIVE)
        positions = wp.array([(0.0, 0.0, 0.0), (8.0, 0.0, 0.0), (16.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        flags = wp.array([active_flag, 2, active_flag | 2], dtype=wp.int32, device="cpu")
        grid = SparseGrid(point_capacity=3, tile_capacity=3, tile_size=4, device="cpu")
        grid.build(positions, flags, active_mask=active_flag)
        grid.check_status()

        self.assertEqual(int(grid.tile_count.numpy()[0]), 2)

    def test_activate_padding_without_halo_cells(self):
        """Activate neighboring core tiles without allocating halo cells."""
        positions = wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        grid = SparseGrid(
            point_capacity=1,
            tile_capacity=27,
            tile_size=4,
            padding_tiles=1,
            device="cpu",
        )
        grid.build(positions)
        grid.check_status()

        self.assertEqual(int(grid.tile_count.numpy()[0]), 27)
        self.assertEqual(grid.cell_capacity, 27 * 4**3)
        neighbors = grid.tile_neighbors.numpy().reshape(27, 27)
        self.assertTrue(np.all(neighbors[:, 13] == np.arange(27)))

    def test_report_tile_capacity_overflow(self):
        """Report topology overflow without writing beyond reserved storage."""
        positions = wp.array([(0.0, 0.0, 0.0), (8.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        grid = SparseGrid(point_capacity=2, tile_capacity=1, tile_size=4, device="cpu")
        grid.build(positions)

        with self.assertRaisesRegex(RuntimeError, "tile capacity"):
            grid.check_status()

    def test_randomized_topology_matches_reference(self):
        """Match randomized signed topology and neighbors to a NumPy reference."""
        rng = np.random.default_rng(1234)
        positions_np = rng.uniform(-19.0, 19.0, size=(257, 3)).astype(np.float32)
        active_np = (rng.random(257) > 0.2).astype(np.int32)
        cell_size = 0.37
        tile_size = 4
        expected = _reference_tiles(positions_np, active_np, cell_size, tile_size, 1)
        grid = SparseGrid(
            point_capacity=len(positions_np),
            tile_capacity=len(expected),
            tile_size=tile_size,
            cell_size=cell_size,
            padding_tiles=1,
            device="cpu",
        )
        grid.build(
            wp.array(positions_np, dtype=wp.vec3, device="cpu"),
            wp.array(active_np, dtype=wp.int32, device="cpu"),
        )
        grid.check_status()

        actual = [tuple(v) for v in _decode_tile_keys(grid)]
        self.assertEqual(actual, expected)
        lookup = {coords: index for index, coords in enumerate(expected)}
        neighbors = grid.tile_neighbors.numpy().reshape(grid.tile_capacity, 27)
        for tile, (x, y, z) in enumerate(expected):
            expected_neighbors = [
                lookup.get((x + dx, y + dy, z + dz), -1) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            ]
            np.testing.assert_array_equal(neighbors[tile], expected_neighbors)

    def test_randomized_global_lookup_matches_reference(self):
        """Resolve randomized global cells to canonical packed indices."""
        rng = np.random.default_rng(4321)
        positions_np = rng.uniform(-8.0, 8.0, size=(80, 3)).astype(np.float32)
        expected_tiles = _reference_tiles(positions_np, np.ones(80), 0.5, 4, 0)
        grid = SparseGrid(
            point_capacity=80,
            tile_capacity=len(expected_tiles),
            tile_size=4,
            cell_size=0.5,
            device="cpu",
        )
        grid.build(wp.array(positions_np, dtype=wp.vec3, device="cpu"))

        cells = rng.integers(-24, 24, size=(500, 3), dtype=np.int32)
        result = wp.empty(len(cells), dtype=wp.int32, device="cpu")
        wp.launch(
            _resolve_global_test_cells,
            dim=len(cells),
            inputs=[grid.data, wp.array(cells, dtype=wp.vec3i, device="cpu"), result],
            device="cpu",
        )
        tile_lookup = {coords: index for index, coords in enumerate(expected_tiles)}
        expected_indices = []
        for cell in cells:
            tile_coords = tuple(np.floor_divide(cell, 4))
            tile = tile_lookup.get(tile_coords, -1)
            if tile < 0:
                expected_indices.append(-1)
            else:
                local = cell - 4 * np.asarray(tile_coords)
                expected_indices.append(int(local[0] + 4 * (local[1] + 4 * (local[2] + 4 * tile))))
        np.testing.assert_array_equal(result.numpy(), expected_indices)

    def test_exact_signed_tile_boundaries(self):
        """Assign exact and adjacent signed boundaries using floor semantics."""
        positions_np = np.array(
            [
                [-4.0, 0.0, 0.0],
                [np.nextafter(np.float32(-4.0), np.float32(-np.inf)), 0.0, 0.0],
                [np.nextafter(np.float32(-4.0), np.float32(np.inf)), 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [np.nextafter(np.float32(4.0), np.float32(-np.inf)), 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        grid = SparseGrid(point_capacity=6, tile_capacity=4, tile_size=4, cell_size=1.0, device="cpu")
        grid.build(wp.array(positions_np, dtype=wp.vec3, device="cpu"))
        grid.check_status()

        self.assertEqual([tuple(v) for v in _decode_tile_keys(grid)], [(-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0)])

    def test_two_padding_layers_activate_chebyshev_cube(self):
        """Activate exactly two no-halo tile layers around one particle."""
        grid = SparseGrid(
            point_capacity=1,
            tile_capacity=125,
            tile_size=4,
            padding_tiles=2,
            device="cpu",
        )
        grid.build(wp.array([(0.1, 0.1, 0.1)], dtype=wp.vec3, device="cpu"))
        grid.check_status()

        self.assertEqual(int(grid.tile_count.numpy()[0]), 125)
        self.assertEqual(
            {tuple(v) for v in _decode_tile_keys(grid)},
            set(_reference_tiles(np.array([[0.1] * 3]), np.ones(1), 1.0, 4, 2)),
        )

    def test_report_coordinate_range_overflow(self):
        """Report positions outside the collision-free signed key range."""
        grid = SparseGrid(point_capacity=1, tile_capacity=1, tile_size=4, cell_size=1.0, device="cpu")
        grid.build(wp.array([(5.0e6, 0.0, 0.0)], dtype=wp.vec3, device="cpu"))

        with self.assertRaisesRegex(RuntimeError, "coordinate"):
            grid.check_status()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_cpu_cuda_randomized_parity(self):
        """Build bit-identical randomized topology on CPU and CUDA."""
        rng = np.random.default_rng(99)
        positions_np = rng.uniform(-12.0, 12.0, size=(511, 3)).astype(np.float32)
        active_np = (rng.random(511) > 0.3).astype(np.int32)
        expected = _reference_tiles(positions_np, active_np, 0.25, 8, 1)
        grids = []
        for device in ("cpu", "cuda:0"):
            grid = SparseGrid(
                point_capacity=511,
                tile_capacity=len(expected),
                tile_size=8,
                cell_size=0.25,
                padding_tiles=1,
                device=device,
            )
            grid.build(
                wp.array(positions_np, dtype=wp.vec3, device=device),
                wp.array(active_np, dtype=wp.int32, device=device),
            )
            grid.check_status()
            grids.append(grid)

        np.testing.assert_array_equal(_decode_tile_keys(grids[0]), _decode_tile_keys(grids[1]))
        np.testing.assert_array_equal(grids[0].tile_neighbors.numpy(), grids[1].tile_neighbors.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_rebuild_inside_cuda_graph(self):
        """Rebuild changing sparse topology during CUDA graph replay."""
        device = wp.get_device("cuda:0")
        positions = wp.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        grid = SparseGrid(
            point_capacity=2,
            tile_capacity=54,
            tile_size=4,
            padding_tiles=1,
            device=device,
        )
        grid.build(positions)
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as capture:
            grid.build(positions)
        positions.assign(np.array([(0.0, 0.0, 0.0), (12.0, 0.0, 0.0)], dtype=np.float32))
        wp.capture_launch(capture.graph)
        grid.check_status()

        self.assertEqual(int(grid.tile_count.numpy()[0]), 54)


if __name__ == "__main__":
    unittest.main()
