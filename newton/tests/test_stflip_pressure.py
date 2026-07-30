# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.stflip.kernels import (
    apply_pressure,
    build_pressure_system,
    enforce_grid_domain,
    pressure_chebyshev,
    pressure_jacobi,
)
from newton._src.solvers.stflip.sparse_grid import (
    SparseGrid,
    SparseGridData,
    sparse_grid_cell_coord,
    sparse_grid_index_from_cell,
)


@wp.func
def _inside_liquid_block(cell: wp.vec3i) -> bool:
    return cell[0] >= 0 and cell[0] < 4 and cell[1] >= 0 and cell[1] < 4 and cell[2] >= 0 and cell[2] < 4


@wp.kernel(enable_backward=False)
def _initialize_pressure_case(
    grid: SparseGridData,
    velocity_mode: int,
    cell_mass: wp.array[float],
    face_mass: wp.array[float],
    face_valid: wp.array[int],
    face_velocity: wp.array[float],
):
    index = wp.tid()
    cell = sparse_grid_cell_coord(grid, index)
    cell_mass[index] = float(_inside_liquid_block(cell))
    for axis in range(3):
        direction = wp.vec3i(0)
        direction[axis] = 1
        face = 3 * index + axis
        if _inside_liquid_block(cell - direction) or _inside_liquid_block(cell):
            face_mass[face] = 1.0
            face_valid[face] = 1
            if velocity_mode == 0:
                face_velocity[face] = float(cell[axis])
            else:
                face_velocity[face] = float(axis + 1) * 0.25


@wp.kernel(enable_backward=False)
def _read_pressure_cells(
    grid: SparseGridData,
    cells: wp.array[wp.vec3i],
    diagonal: wp.array[float],
    face_velocity: wp.array[float],
    values: wp.array[wp.vec2],
):
    sample = wp.tid()
    index = sparse_grid_index_from_cell(grid, cells[sample])
    values[sample] = wp.vec2(diagonal[index], face_velocity[3 * index])


class TestSTFLIPPressure(unittest.TestCase):
    def _devices(self):
        """Return every device used for pressure validation."""
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    def _create_case(self, device, velocity_mode):
        """Create one sparse four-cell-cube projection case."""
        point = wp.array([[1.5, 1.5, 1.5]], dtype=wp.vec3, device=device)
        grid = SparseGrid(
            point_capacity=1,
            tile_capacity=27,
            tile_size=4,
            cell_size=1.0,
            padding_tiles=1,
            device=device,
        )
        grid.build(point)
        grid.check_status()
        capacity = grid.cell_capacity
        arrays = {
            "cell_mass": wp.zeros(capacity, dtype=wp.float32, device=device),
            "face_mass": wp.zeros(3 * capacity, dtype=wp.float32, device=device),
            "face_valid": wp.zeros(3 * capacity, dtype=wp.int32, device=device),
            "face_velocity": wp.zeros(3 * capacity, dtype=wp.float32, device=device),
            "rhs": wp.zeros(capacity, dtype=wp.float32, device=device),
            "diag": wp.zeros(capacity, dtype=wp.float32, device=device),
            "pressure": wp.zeros(capacity, dtype=wp.float32, device=device),
            "scratch": wp.zeros(capacity, dtype=wp.float32, device=device),
            "direction": wp.zeros(capacity, dtype=wp.float32, device=device),
            "direction_scratch": wp.zeros(capacity, dtype=wp.float32, device=device),
        }
        wp.launch(
            _initialize_pressure_case,
            dim=capacity,
            inputs=[
                grid.data,
                velocity_mode,
                arrays["cell_mass"],
                arrays["face_mass"],
                arrays["face_valid"],
                arrays["face_velocity"],
            ],
            device=device,
        )
        return grid, arrays

    def _divergence(self, grid, arrays, device):
        """Measure liquid-cell divergence through the production assembly."""
        wp.launch(
            build_pressure_system,
            dim=grid.cell_capacity,
            inputs=[
                grid.data,
                arrays["cell_mass"],
                arrays["face_velocity"],
                0.5,
                1.0,
                False,
                wp.vec3i(0),
                wp.vec3i(0),
                arrays["rhs"],
                arrays["diag"],
            ],
            device=device,
        )
        mass = arrays["cell_mass"].numpy()
        return arrays["rhs"].numpy()[mass > 0.5]

    def _project(self, device, iterations, accelerated=False):
        """Project the expanding manufactured velocity field."""
        grid, arrays = self._create_case(device, velocity_mode=0)
        before = self._divergence(grid, arrays, device)
        pressure_in = arrays["pressure"]
        pressure_out = arrays["scratch"]
        direction_in = arrays["direction"]
        direction_out = arrays["direction_scratch"]
        center = 1.01
        radius = 0.99
        alpha = 1.0 / center
        for iteration in range(iterations):
            if accelerated:
                beta = 0.0
                if iteration:
                    beta = (0.5 * radius * alpha) ** 2
                    alpha = 1.0 / (center - beta / alpha)
                wp.launch(
                    pressure_chebyshev,
                    dim=grid.cell_capacity,
                    inputs=[
                        grid.data,
                        arrays["cell_mass"],
                        0.5,
                        arrays["rhs"],
                        arrays["diag"],
                        alpha,
                        beta,
                        pressure_in,
                        direction_in,
                        pressure_out,
                        direction_out,
                    ],
                    device=device,
                )
                direction_in, direction_out = direction_out, direction_in
            else:
                wp.launch(
                    pressure_jacobi,
                    dim=grid.cell_capacity,
                    inputs=[
                        grid.data,
                        arrays["cell_mass"],
                        0.5,
                        arrays["rhs"],
                        arrays["diag"],
                        pressure_in,
                        pressure_out,
                    ],
                    device=device,
                )
            pressure_in, pressure_out = pressure_out, pressure_in
        wp.launch(
            apply_pressure,
            dim=grid.cell_capacity,
            inputs=[
                grid.data,
                arrays["cell_mass"],
                0.5,
                pressure_in,
                1.0,
                False,
                wp.vec3i(0),
                wp.vec3i(0),
                arrays["face_valid"],
                arrays["face_velocity"],
            ],
            device=device,
        )
        after = self._divergence(grid, arrays, device)
        return before, after

    def test_chebyshev_acceleration_reduces_iteration_count(self):
        """Converge faster than unaccelerated Jacobi pressure iterations."""
        for device in self._devices():
            with self.subTest(device=device):
                _before, accelerated = self._project(device, iterations=20, accelerated=True)
                _before, jacobi = self._project(device, iterations=20)
                self.assertLess(np.linalg.norm(accelerated), np.linalg.norm(jacobi))

    def test_projection_reduces_manufactured_divergence(self):
        """Reduce a manufactured expanding field to near-zero divergence."""
        for device in self._devices():
            with self.subTest(device=device):
                before, after = self._project(device, iterations=160)
                self.assertEqual(len(before), 64)
                np.testing.assert_allclose(before, 3.0, rtol=0.0, atol=0.0)
                self.assertLess(np.linalg.norm(after), 2.0e-4 * np.linalg.norm(before))

    def test_projection_converges_with_iteration_count(self):
        """Decrease divergence as Jacobi iteration count increases."""
        for device in self._devices():
            with self.subTest(device=device):
                norms = []
                for iterations in (4, 16, 64):
                    _before, after = self._project(device, iterations)
                    norms.append(float(np.linalg.norm(after)))
                self.assertGreater(norms[0], norms[1])
                self.assertGreater(norms[1], norms[2])
                self.assertLess(norms[2], 0.02 * norms[0])

    def test_projection_preserves_divergence_free_constant_field(self):
        """Preserve a constant MAC field with exactly zero divergence."""
        for device in self._devices():
            with self.subTest(device=device):
                grid, arrays = self._create_case(device, velocity_mode=1)
                velocity_before = arrays["face_velocity"].numpy()
                divergence = self._divergence(grid, arrays, device)
                np.testing.assert_array_equal(divergence, np.zeros(64, dtype=np.float32))

                pressure_in = arrays["pressure"]
                pressure_out = arrays["scratch"]
                for _ in range(32):
                    wp.launch(
                        pressure_jacobi,
                        dim=grid.cell_capacity,
                        inputs=[
                            grid.data,
                            arrays["cell_mass"],
                            0.5,
                            arrays["rhs"],
                            arrays["diag"],
                            pressure_in,
                            pressure_out,
                        ],
                        device=device,
                    )
                    pressure_in, pressure_out = pressure_out, pressure_in
                wp.launch(
                    apply_pressure,
                    dim=grid.cell_capacity,
                    inputs=[
                        grid.data,
                        arrays["cell_mass"],
                        0.5,
                        pressure_in,
                        1.0,
                        False,
                        wp.vec3i(0),
                        wp.vec3i(0),
                        arrays["face_valid"],
                        arrays["face_velocity"],
                    ],
                    device=device,
                )
                np.testing.assert_array_equal(arrays["face_velocity"].numpy(), velocity_before)

    def test_closed_domain_enforces_neumann_wall_faces(self):
        """Exclude solid-wall neighbors and zero boundary-normal velocity."""
        for device in self._devices():
            with self.subTest(device=device):
                grid, arrays = self._create_case(device, velocity_mode=1)
                lower = wp.vec3i(0)
                upper = wp.vec3i(4)
                wp.launch(
                    enforce_grid_domain,
                    dim=grid.cell_capacity,
                    inputs=[
                        grid.data,
                        lower,
                        upper,
                        arrays["face_velocity"],
                        arrays["face_velocity"],
                        arrays["face_valid"],
                    ],
                    device=device,
                )
                wp.launch(
                    build_pressure_system,
                    dim=grid.cell_capacity,
                    inputs=[
                        grid.data,
                        arrays["cell_mass"],
                        arrays["face_velocity"],
                        0.5,
                        1.0,
                        True,
                        lower,
                        upper,
                        arrays["rhs"],
                        arrays["diag"],
                    ],
                    device=device,
                )
                cells = wp.array([(0, 1, 1), (1, 1, 1)], dtype=wp.vec3i, device=device)
                values = wp.empty(2, dtype=wp.vec2, device=device)
                wp.launch(
                    _read_pressure_cells,
                    dim=2,
                    inputs=[grid.data, cells, arrays["diag"], arrays["face_velocity"], values],
                    device=device,
                )
                np.testing.assert_allclose(
                    values.numpy(),
                    np.array([[5.0, 0.0], [6.0, 0.25]], dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
