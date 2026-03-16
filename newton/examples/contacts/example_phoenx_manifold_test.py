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
# Example PhoenX Contact Manifold Test
#
# A thin static box rail with a frictionless cylinder resting on top.
# Demonstrates edge-edge contact manifold generation between a cylinder
# and a narrow box. Translated from C# PhoenX Demo08 "Contact Manifold
# Test".
#
# Command: python -m newton.examples phoenx_manifold_test
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

RAIL_HALF = (2.5, 0.25, 0.25)
CYLINDER_RADIUS = 3.0
CYLINDER_HALF_HEIGHT = 0.25

PGS_ITERATIONS = 8
SIM_SUBSTEPS = 4
FPS = 60
GRAVITY = (0.0, 0.0, -9.81)


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
    """Contact manifold test: frictionless cylinder on a thin box rail.

    The cylinder should balance on the rail edge without friction, testing
    the contact manifold generation for edge-edge contact pairs.
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

        num_bodies = 3  # ground + rail + cylinder
        num_shapes = 3
        contact_cap = 64

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
        self.pipeline.add_shape_plane(body_row=row_ground)

        # Static rail box
        hx, hy, hz = RAIL_HALF
        h_rail = ss.add_body(position=(0, 0, 1.0), is_static=True)
        self.row_rail = int(ss.body_store.handle_to_index.numpy()[h_rail])
        ss.set_shape_body(1, h_rail)
        self.pipeline.add_shape_box(body_row=self.row_rail, half_extents=RAIL_HALF)

        # Dynamic cylinder (frictionless)
        cyl_mass = 5.0
        cyl_inv_mass = 1.0 / cyl_mass
        r, hh = CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT
        inv_ixx = 12.0 * cyl_inv_mass / (3.0 * r * r + 4.0 * hh * hh)
        inv_izz = 2.0 * cyl_inv_mass / (r * r)
        cyl_inv_inertia = np.diag(
            np.array([inv_ixx, inv_ixx, inv_izz], dtype=np.float32)
        )

        h_cyl = ss.add_body(
            position=(0, 0, 1.0 + RAIL_HALF[2] + CYLINDER_RADIUS + 0.5),
            inverse_mass=cyl_inv_mass,
            inverse_inertia_local=cyl_inv_inertia,
            linear_damping=0.999,
            angular_damping=0.99,
        )
        self.h_cyl = h_cyl
        self.row_cyl = int(ss.body_store.handle_to_index.numpy()[h_cyl])
        ss.set_shape_body(2, h_cyl)
        self.pipeline.add_shape_cylinder(
            body_row=self.row_cyl,
            radius=CYLINDER_RADIUS,
            half_height=CYLINDER_HALF_HEIGHT,
            friction=0.0,
        )

        self.pipeline.finalize()

        # Rendering arrays
        self._rail_row = wp.array([self.row_rail], dtype=wp.int32, device=device)
        self.rail_xform = wp.zeros(1, dtype=wp.transform, device=device)
        self.rail_color = wp.array([wp.vec3(0.3, 0.3, 0.7)], dtype=wp.vec3, device=device)
        self.rail_material = wp.array([wp.vec4(0.5, 0.3, 0.0, 0.0)], dtype=wp.vec4, device=device)

        self._cyl_row = wp.array([self.row_cyl], dtype=wp.int32, device=device)
        self.cyl_xform = wp.zeros(1, dtype=wp.transform, device=device)
        self.cyl_color = wp.array([wp.vec3(0.9, 0.4, 0.2)], dtype=wp.vec3, device=device)
        self.cyl_material = wp.array([wp.vec4(0.5, 0.3, 0.0, 0.0)], dtype=wp.vec4, device=device)

        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        self.viewer.set_camera(
            pos=wp.vec3(6.0, -6.0, 5.0),
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
            dim=1,
            inputs=[self._rail_row, bs.column_of("position"), bs.column_of("orientation"), self.rail_xform, 1],
            device=d,
        )
        wp.launch(
            _build_xforms_kernel,
            dim=1,
            inputs=[self._cyl_row, bs.column_of("position"), bs.column_of("orientation"), self.cyl_xform, 1],
            device=d,
        )

        self.viewer.log_shapes(
            "/rail", newton.GeoType.BOX, RAIL_HALF, self.rail_xform, self.rail_color, self.rail_material
        )
        self.viewer.log_shapes(
            "/cylinder",
            newton.GeoType.CYLINDER,
            (CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT),
            self.cyl_xform,
            self.cyl_color,
            self.cyl_material,
        )
        self.viewer.log_shapes(
            "/ground", newton.GeoType.PLANE, (20.0, 20.0), self.ground_xform, self.ground_color, self.ground_material
        )
        self.viewer.end_frame()

    def test_final(self):
        wp.synchronize_device(self.device)
        pos = self.ss.body_store.column_of("position").numpy()[self.row_cyl]
        assert pos[2] > 0.5, f"Cylinder fell through rail: z={pos[2]:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
