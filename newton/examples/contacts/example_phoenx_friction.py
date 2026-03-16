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
# Example PhoenX Friction
#
# A row of cubes launched across a ground plane, each with different
# friction coefficients (1.0 down to 0.0).  The high-friction cubes
# should stop quickly while the low-friction cubes slide far.
# Translated from C# PhoenX Demo09 "Restitution and Friction".
#
# Command: python -m newton.examples phoenx_friction
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

NUM_CUBES = 11
CUBE_HALF = 0.25
LAUNCH_SPEED = 10.0

PGS_ITERATIONS = 12
SIM_SUBSTEPS = 4
FPS = 60
GRAVITY = (0.0, 0.0, -9.81)


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
    """Friction test: 11 cubes with friction from 1.0 to 0.0.

    Each cube is launched with the same initial velocity along the Y axis.
    High-friction cubes stop quickly; low-friction cubes slide further.
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
        contact_cap = NUM_CUBES * 8

        self.ss = SolverState(
            body_capacity=num_bodies,
            contact_capacity=contact_cap,
            shape_count=num_shapes,
            device=device,
            default_friction=0.5,
            max_colors=8,
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
        self.pipeline.add_shape_plane(body_row=row_ground, friction=0.0)

        # Cubes with varying friction
        cube_mass = 1.0
        cube_inv_mass = 1.0 / cube_mass
        cube_inv_inertia = _box_inv_inertia(cube_inv_mass, CUBE_HALF)

        self.cube_handles = []
        self.cube_rows = []
        self.cube_frictions = []

        for i in range(NUM_CUBES):
            friction = 1.0 - i * 0.1
            self.cube_frictions.append(friction)

            handle = ss.add_body(
                position=(2.0 + i * 1.0, 0.0, CUBE_HALF),
                velocity=(0.0, -LAUNCH_SPEED, 0.0),
                inverse_mass=cube_inv_mass,
                inverse_inertia_local=cube_inv_inertia,
                linear_damping=0.999,
                angular_damping=0.999,
            )
            row = int(ss.body_store.handle_to_index.numpy()[handle])
            ss.set_shape_body(1 + i, handle)
            self.pipeline.add_shape_box(
                body_row=row,
                half_extents=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
                friction=friction,
            )
            self.cube_handles.append(handle)
            self.cube_rows.append(row)

        self.pipeline.finalize()

        # Rendering
        self._cube_rows_gpu = wp.array(self.cube_rows, dtype=wp.int32, device=device)
        self.cube_xforms = wp.zeros(NUM_CUBES, dtype=wp.transform, device=device)

        colors = []
        for i in range(NUM_CUBES):
            t = i / (NUM_CUBES - 1)
            colors.append(wp.vec3(0.2 + 0.7 * t, 0.8 - 0.6 * t, 0.2))
        self.cube_colors = wp.array(colors, dtype=wp.vec3, device=device)
        self.cube_materials = wp.array(
            [wp.vec4(0.5, 0.3, 0.0, 0.0)] * NUM_CUBES, dtype=wp.vec4, device=device
        )
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        self.viewer.set_camera(
            pos=wp.vec3(8.0, 8.0, 6.0),
            pitch=-25.0,
            yaw=-135.0,
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
            (50.0, 50.0),
            self.ground_xform,
            self.ground_color,
            self.ground_material,
        )
        self.viewer.end_frame()

    def test_final(self):
        """High-friction cubes should have traveled less than low-friction cubes."""
        wp.synchronize_device(self.device)
        positions = self.ss.body_store.column_of("position").numpy()
        h2i = self.ss.body_store.handle_to_index.numpy()

        distances = []
        for i, h in enumerate(self.cube_handles):
            row = int(h2i[h])
            y = positions[row][1]
            distances.append(-y)

        assert distances[-1] > distances[0], (
            f"Low-friction cube (dist={distances[-1]:.2f}) did not slide "
            f"further than high-friction cube (dist={distances[0]:.2f})"
        )

        for i, h in enumerate(self.cube_handles):
            row = int(h2i[h])
            z = positions[row][2]
            assert z > -0.5, f"Cube {i} fell through ground: z={z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
