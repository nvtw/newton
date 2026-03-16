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
# Example PhoenX Tower
#
# A cylindrical tower built from small box bricks arranged in rings.
# 32 bricks per ring, each ring rotated by a half-step, stacking to a
# configurable height.  Translated from C# PhoenX Demo02 "Tower of Jitter".
#
# Command: python -m newton.examples phoenx_tower
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

TOWER_HEIGHT = 30
BRICKS_PER_RING = 32
TOWER_RADIUS = 3.0
BRICK_HALF = (0.15, 0.05, 0.45)

PGS_ITERATIONS = 3
SIM_SUBSTEPS = 12
FPS = 60
GRAVITY = (0.0, 0.0, -9.81)

NUM_BRICKS = TOWER_HEIGHT * BRICKS_PER_RING


def _box_inv_inertia(inv_mass: float, hx: float, hy: float, hz: float) -> np.ndarray:
    return np.diag(
        np.array(
            [
                3.0 * inv_mass / (hy * hy + hz * hz),
                3.0 * inv_mass / (hx * hx + hz * hz),
                3.0 * inv_mass / (hx * hx + hy * hy),
            ],
            dtype=np.float32,
        )
    )


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
    """Cylindrical tower of box bricks.

    32 bricks per ring, rotated by a half-step each level. Tests many-body
    stacking stability with the PGS velocity-level contact solver.
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

        num_bodies = 1 + NUM_BRICKS  # ground + bricks
        num_shapes = num_bodies
        contact_cap = max(NUM_BRICKS * 8, 1024)

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

        # Bricks
        hx, hy, hz = BRICK_HALF
        brick_mass = 1.0
        brick_inv_mass = 1.0 / brick_mass
        brick_inv_inertia = _box_inv_inertia(brick_inv_mass, hx, hy, hz)

        half_rot_step = 2.0 * math.pi / (2.0 * BRICKS_PER_RING)
        full_rot_step = 2.0 * half_rot_step

        self.brick_handles = []
        self.brick_rows = []
        shape_idx = 1

        angle = 0.0
        for level in range(TOWER_HEIGHT):
            angle += half_rot_step
            level_z = (0.5 + level) * (2.0 * hy + 0.001)
            for _ in range(BRICKS_PER_RING):
                bx = TOWER_RADIUS * math.cos(angle)
                by = TOWER_RADIUS * math.sin(angle)

                ca, sa = math.cos(angle), math.sin(angle)
                qx, qy, qz, qw = 0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)

                h = ss.add_body(
                    position=(bx, by, level_z),
                    orientation=(qx, qy, qz, qw),
                    inverse_mass=brick_inv_mass,
                    inverse_inertia_local=brick_inv_inertia,
                    linear_damping=0.998,
                    angular_damping=0.998,
                )
                r = int(ss.body_store.handle_to_index.numpy()[h])
                ss.set_shape_body(shape_idx, h)
                self.pipeline.add_shape_box(body_row=r, half_extents=BRICK_HALF)
                self.brick_handles.append(h)
                self.brick_rows.append(r)
                shape_idx += 1
                angle += full_rot_step

        self.pipeline.finalize()

        # Rendering
        self._brick_rows_gpu = wp.array(self.brick_rows, dtype=wp.int32, device=device)
        self.brick_xforms = wp.zeros(NUM_BRICKS, dtype=wp.transform, device=device)
        self.brick_colors = wp.array(
            [wp.vec3(0.85, 0.55, 0.25)] * NUM_BRICKS, dtype=wp.vec3, device=device
        )
        self.brick_materials = wp.array(
            [wp.vec4(0.5, 0.3, 0.0, 0.0)] * NUM_BRICKS, dtype=wp.vec4, device=device
        )
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        self.viewer.set_camera(
            pos=wp.vec3(8.0, -8.0, 4.0),
            pitch=-20.0,
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
            dim=NUM_BRICKS,
            inputs=[
                self._brick_rows_gpu,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.brick_xforms,
                NUM_BRICKS,
            ],
            device=d,
        )

        self.viewer.log_shapes(
            "/bricks",
            newton.GeoType.BOX,
            BRICK_HALF,
            self.brick_xforms,
            self.brick_colors,
            self.brick_materials,
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
        for i, h in enumerate(self.brick_handles):
            row = int(h2i[h])
            z = positions[row][2]
            assert z > -0.5, f"Brick {i} fell through ground: z={z:.4f}"
        top_z = max(positions[int(h2i[h])][2] for h in self.brick_handles)
        assert top_z > 0.5, f"Tower collapsed completely: max z={top_z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
