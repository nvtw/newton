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
# Example PhoenX Coin Piles
#
# Tall piles of thin cylinder discs ("coins") stacked in columns.
# Demonstrates cylinder-cylinder and cylinder-plane contact stability.
# Translated from C# PhoenX Demo06 "Shapes".
#
# Command: python -m newton.examples phoenx_coin_piles
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

NUM_PILES = 3
PILE_HEIGHT = 20
PILE_WIDTH = 6
COIN_RADIUS = 0.5
COIN_HALF_THICKNESS = 0.05

PGS_ITERATIONS = 8
SIM_SUBSTEPS = 4
FPS = 60
GRAVITY = (0.0, 0.0, -9.81)


def _count_coins():
    total = 0
    for k in range(NUM_PILES):
        for i in range(PILE_HEIGHT):
            row_size = PILE_WIDTH - i % 2
            total += row_size
    return total


NUM_COINS = _count_coins()


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
    """Piles of cylinder coins stacked in columns.

    Tests cylinder shape collision and many-body stacking stability.
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

        num_bodies = 1 + NUM_COINS
        num_shapes = num_bodies
        contact_cap = max(NUM_COINS * 24, 2048)

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

        # Coins
        coin_mass = 1.0
        coin_inv_mass = 1.0 / coin_mass
        r, h = COIN_RADIUS, COIN_HALF_THICKNESS
        inv_ixx = 12.0 * coin_inv_mass / (3.0 * r * r + 4.0 * h * h)
        inv_izz = 2.0 * coin_inv_mass / (r * r)
        coin_inv_inertia = np.diag(
            np.array([inv_ixx, inv_ixx, inv_izz], dtype=np.float32)
        )

        self.coin_handles = []
        self.coin_rows = []
        shape_idx = 1
        coin_margin = -0.001

        for k in range(NUM_PILES):
            for i in range(PILE_HEIGHT):
                row_size = PILE_WIDTH - i % 2
                z_height = (0.5 + i) * (2.0 * COIN_HALF_THICKNESS + coin_margin * 2)
                for j in range(row_size):
                    shift = (2.0 * j - row_size) * (COIN_RADIUS + coin_margin)
                    x = shift
                    y = k * COIN_RADIUS * 4

                    handle = ss.add_body(
                        position=(x, y, z_height),
                        inverse_mass=coin_inv_mass,
                        inverse_inertia_local=coin_inv_inertia,
                        linear_damping=0.999,
                        angular_damping=0.99,
                    )
                    row = int(ss.body_store.handle_to_index.numpy()[handle])
                    ss.set_shape_body(shape_idx, handle)
                    self.pipeline.add_shape_cylinder(
                        body_row=row,
                        radius=COIN_RADIUS,
                        half_height=COIN_HALF_THICKNESS,
                    )
                    self.coin_handles.append(handle)
                    self.coin_rows.append(row)
                    shape_idx += 1

        self.pipeline.finalize()

        # Rendering
        self._coin_rows_gpu = wp.array(self.coin_rows, dtype=wp.int32, device=device)
        self.coin_xforms = wp.zeros(NUM_COINS, dtype=wp.transform, device=device)
        self.coin_colors = wp.array(
            [wp.vec3(0.8, 0.7, 0.2)] * NUM_COINS, dtype=wp.vec3, device=device
        )
        self.coin_materials = wp.array(
            [wp.vec4(0.7, 0.4, 0.0, 0.0)] * NUM_COINS, dtype=wp.vec4, device=device
        )
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        self.viewer.set_camera(
            pos=wp.vec3(10.0, -6.0, 5.0),
            pitch=-20.0,
            yaw=150.0,
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
            dim=NUM_COINS,
            inputs=[
                self._coin_rows_gpu,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.coin_xforms,
                NUM_COINS,
            ],
            device=d,
        )

        self.viewer.log_shapes(
            "/coins",
            newton.GeoType.CYLINDER,
            (COIN_RADIUS, COIN_HALF_THICKNESS),
            self.coin_xforms,
            self.coin_colors,
            self.coin_materials,
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
        wp.synchronize_device(self.device)
        positions = self.ss.body_store.column_of("position").numpy()
        h2i = self.ss.body_store.handle_to_index.numpy()
        for i, h in enumerate(self.coin_handles):
            row = int(h2i[h])
            z = positions[row][2]
            assert z > -0.5, f"Coin {i} fell through ground: z={z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
