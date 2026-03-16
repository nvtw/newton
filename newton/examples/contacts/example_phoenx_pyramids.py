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

###########################################################################
# Example PhoenX Pyramids
#
# Multiple box pyramids placed side by side. Each pyramid is a classic
# stacking test with unit cubes. Translated from C# PhoenX Demo07
# "Many Pyramids".
#
# Command: python -m newton.examples phoenx_pyramids
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

PYRAMID_SIZE = 15
NUM_ROWS = 2
NUM_COLS = 4
CUBE_HALF = 0.5
SPACING = 1.01

PGS_ITERATIONS = 4
SIM_SUBSTEPS = 4
FPS = 60
GRAVITY = (0.0, 0.0, -9.81)


def _count_pyramid_cubes(size):
    return sum(size - i for i in range(size))


NUM_CUBES_PER_PYRAMID = _count_pyramid_cubes(PYRAMID_SIZE)
NUM_PYRAMIDS = NUM_ROWS * NUM_COLS
NUM_CUBES = NUM_CUBES_PER_PYRAMID * NUM_PYRAMIDS


def _box_inv_inertia(inv_mass: float, half: float) -> np.ndarray:
    val = 6.0 * inv_mass / (2.0 * half) ** 2
    return np.eye(3, dtype=np.float32) * val


@wp.kernel
def _build_xforms_kernel(
    handle_rows: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    xforms_out: wp.array(dtype=wp.transform),
    count: int,
):
    tid = wp.tid()
    if tid >= count:
        return
    row = handle_rows[tid]
    xforms_out[tid] = wp.transform(positions[row], orientations[row])


class Example:
    """Multiple box pyramids arranged in a grid.

    Tests many-body contact stability and solver convergence for pyramid
    stacking, a classical rigid-body benchmark.
    """

    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / FPS
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)

        device = wp.get_preferred_device()
        self.device = device

        num_bodies = 1 + NUM_CUBES
        num_shapes = num_bodies
        contact_cap = max(NUM_CUBES * 8, 2048)

        self.ss = SolverState(
            body_capacity=num_bodies,
            contact_capacity=contact_cap,
            shape_count=num_shapes,
            device=device,
            default_friction=0.5,
            max_colors=12,
        )
        ss = self.ss

        self.pipeline = PhoenXCollisionPipeline(
            max_shapes=num_shapes,
            max_contacts=contact_cap,
            device=device,
        )

        # Ground
        h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
        row_ground = int(ss.body_store.handle_to_index.numpy()[h_ground])
        ss.set_shape_body(0, h_ground)
        self.pipeline.add_shape_plane(body_row=row_ground)

        # Pyramids
        cube_mass = 1.0
        cube_inv_mass = 1.0 / cube_mass
        cube_inv_inertia = _box_inv_inertia(cube_inv_mass, CUBE_HALF)

        self.cube_handles = []
        self.cube_rows = []
        shape_idx = 1

        pyramid_spacing_x = PYRAMID_SIZE * SPACING + 3.0
        pyramid_spacing_y = PYRAMID_SIZE * SPACING + 3.0

        for row_idx in range(NUM_ROWS):
            for col_idx in range(NUM_COLS):
                base_x = (col_idx - (NUM_COLS - 1) / 2.0) * pyramid_spacing_x
                base_y = (row_idx - (NUM_ROWS - 1) / 2.0) * pyramid_spacing_y

                for i in range(PYRAMID_SIZE):
                    for e in range(i, PYRAMID_SIZE):
                        x = base_x + (e - i * 0.5) * SPACING
                        y = base_y
                        z = CUBE_HALF + i * (2.0 * CUBE_HALF)

                        handle = ss.add_body(
                            position=(x, y, z),
                            inverse_mass=cube_inv_mass,
                            inverse_inertia_local=cube_inv_inertia,
                            linear_damping=0.999,
                            angular_damping=0.999,
                        )
                        r = int(ss.body_store.handle_to_index.numpy()[handle])
                        ss.set_shape_body(shape_idx, handle)
                        self.pipeline.add_shape_box(
                            body_row=r,
                            half_extents=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
                        )
                        self.cube_handles.append(handle)
                        self.cube_rows.append(r)
                        shape_idx += 1

        self.pipeline.finalize()

        # Rendering
        self._cube_rows_gpu = wp.array(self.cube_rows, dtype=wp.int32, device=device)
        self.cube_xforms = wp.zeros(NUM_CUBES, dtype=wp.transform, device=device)
        self.cube_colors = wp.array(
            [wp.vec3(0.6, 0.3, 0.15)] * NUM_CUBES, dtype=wp.vec3, device=device
        )
        self.cube_materials = wp.array(
            [wp.vec4(0.5, 0.3, 0.0, 0.0)] * NUM_CUBES, dtype=wp.vec4, device=device
        )
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        self.viewer.set_camera(
            pos=wp.vec3(25.0, -20.0, 15.0),
            pitch=-25.0,
            yaw=135.0,
        )

        self.graph = None
        self.simulate()
        try:
            self.capture()
        except Exception:
            pass

    def simulate(self):
        self.ss.update_world_inertia()
        self.ss.warm_starter.begin_frame()
        self.pipeline.collide(self.ss)
        for _ in range(self.sim_substeps):
            self.ss.step(self.sim_dt, gravity=GRAVITY, num_iterations=PGS_ITERATIONS)
        self.ss.export_impulses()

    def capture(self):
        if not self.device.is_cuda:
            return
        self.simulate()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(self.device) as capture:
            self.simulate()
        self.graph = capture.graph

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)

        bs = self.ss.body_store
        d = self.device

        wp.launch(
            _build_xforms_kernel,
            dim=NUM_CUBES,
            inputs=[
                self._cube_rows_gpu,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.cube_xforms,
                NUM_CUBES,
            ],
            device=d,
        )

        self.viewer.log_shapes(
            "/cubes",
            newton.GeoType.BOX,
            (CUBE_HALF, CUBE_HALF, CUBE_HALF),
            self.cube_xforms,
            self.cube_colors,
            self.cube_materials,
        )
        self.viewer.log_shapes(
            "/ground",
            newton.GeoType.PLANE,
            (100.0, 100.0),
            self.ground_xform,
            self.ground_color,
            self.ground_material,
        )
        self.viewer.end_frame()

    def test_final(self):
        wp.synchronize_device(self.device)
        positions = self.ss.body_store.column_of("position").numpy()
        h2i = self.ss.body_store.handle_to_index.numpy()
        for i, h in enumerate(self.cube_handles):
            row = int(h2i[h])
            z = positions[row][2]
            assert z > -0.5, f"Cube {i} fell through ground: z={z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
