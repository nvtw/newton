# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.fluid import VolumeBrickGrid


@wp.kernel(enable_backward=False)
def _lookup(volume: wp.uint64, coords: wp.array[wp.vec3i], values: wp.array[float]):
    i = wp.tid()
    p = coords[i]
    values[i] = wp.volume_lookup_f(volume, p[0], p[1], p[2])


class TestVolumeBrickGrid(unittest.TestCase):
    def test_round_trip(self):
        """Round-trip sparse bricks without changing dense values."""
        dense = np.zeros((11, 9, 7, 2), dtype=np.float16)
        dense[2:5, 3:7, 1:6, 0] = 0.5
        dense[8:10, 1:3, 4:6, 1] = 1.0
        grid = VolumeBrickGrid.from_dense(
            dense,
            voxel_size=(0.1, 0.2, 0.3),
            channels=("density", "temperature"),
        )
        with TemporaryDirectory() as directory:
            path = f"{directory}/field.pvol"
            grid.save(path)
            loaded = VolumeBrickGrid.load(path)
        np.testing.assert_array_equal(loaded.dense(), dense)

    def test_nanovdb_contains_scalar_payload(self):
        """Store scalar values in a writable NanoVDB grid."""
        dense = np.zeros((10, 9, 8), dtype=np.float32)
        dense[2, 3, 4] = 0.75
        dense[9, 8, 7] = 1.25
        grid = VolumeBrickGrid.from_dense(dense, voxel_size=(0.1, 0.2, 0.3))
        with TemporaryDirectory() as directory:
            volume = grid.save_nanovdb(f"{directory}/field.nvdb", device="cpu")
        self.assertFalse(volume.is_index)
        coords = wp.array(((2, 3, 4), (9, 8, 7)), dtype=wp.vec3i, device="cpu")
        values = wp.zeros(2, device="cpu")
        wp.launch(_lookup, 2, inputs=[volume.id, coords, values], device="cpu")
        np.testing.assert_allclose(values.numpy(), (0.75, 1.25))


if __name__ == "__main__":
    unittest.main()
