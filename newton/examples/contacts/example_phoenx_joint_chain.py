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
# Example PhoenX Joint Chain
#
# A chain of box links connected by revolute joints, hanging from a
# static anchor.  The chain swings freely under gravity, testing
# constraint stability and joint solve. Translated from C# PhoenX
# Demo12 "Joint Chain".
#
# Command: python -m newton.examples phoenx_joint_chain
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

NUM_LINKS = 10
LINK_HALF = (1.0, 0.05, 0.05)
LINK_GAP = 0.2

PGS_ITERATIONS = 3
SIM_SUBSTEPS = 15
FPS = 60
GRAVITY = (0.0, 0.0, -9.81)


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
    """Chain of box links connected by revolute joints.

    A static anchor holds the first link; the rest swing freely.
    Tests constraint solver convergence with many sequential joints.
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

        num_bodies = 2 + NUM_LINKS  # ground + anchor + links
        num_shapes = num_bodies
        num_joints = NUM_LINKS
        contact_cap = max(NUM_LINKS * 4, 64)

        self.ss = SolverState(
            body_capacity=num_bodies,
            contact_capacity=contact_cap,
            shape_count=num_shapes,
            device=device,
            default_friction=0.5,
            max_colors=12,
            joint_capacity=num_joints,
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

        # Static anchor body (the bar the chain hangs from)
        hx, hy, hz = LINK_HALF
        anchor_x = 0.0
        anchor_z = 6.0

        h_anchor = ss.add_body(
            position=(anchor_x, 0, anchor_z),
            is_static=True,
        )
        row_anchor = int(ss.body_store.handle_to_index.numpy()[h_anchor])
        ss.set_shape_body(1, h_anchor)
        self.pipeline.add_shape_box(body_row=row_anchor, half_extents=LINK_HALF)

        # Chain links
        link_mass = 1.0
        link_inv_mass = 1.0 / link_mass
        link_inv_inertia = _box_inv_inertia(link_inv_mass, hx, hy, hz)

        self.link_handles = []
        self.link_rows = []
        prev_handle = h_anchor
        link_stride = 2.0 * hx + LINK_GAP

        for i in range(NUM_LINKS):
            link_x = anchor_x + (i + 1) * link_stride
            nudge_y = 0.01 if (i % 2 == 0) else 0.0

            handle = ss.add_body(
                position=(link_x, nudge_y, anchor_z),
                inverse_mass=link_inv_mass,
                inverse_inertia_local=link_inv_inertia,
                linear_damping=0.999,
                angular_damping=0.99,
            )
            row = int(ss.body_store.handle_to_index.numpy()[handle])
            ss.set_shape_body(2 + i, handle)
            self.pipeline.add_shape_box(body_row=row, half_extents=LINK_HALF)
            self.link_handles.append(handle)
            self.link_rows.append(row)

            anchor_pt = (link_x - hx - LINK_GAP * 0.5, 0, anchor_z)
            ss.add_joint_revolute(
                body_handle0=prev_handle,
                body_handle1=handle,
                anchor_world=anchor_pt,
                axis_world=(0.0, 1.0, 0.0),
            )
            prev_handle = handle

        self.pipeline.finalize()

        # Rendering
        all_body_rows = [row_anchor] + self.link_rows
        num_render = 1 + NUM_LINKS
        self._all_rows_gpu = wp.array(all_body_rows, dtype=wp.int32, device=device)
        self.all_xforms = wp.zeros(num_render, dtype=wp.transform, device=device)

        colors = [wp.vec3(0.4, 0.4, 0.7)]
        for i in range(NUM_LINKS):
            t = i / max(NUM_LINKS - 1, 1)
            colors.append(wp.vec3(0.9 - 0.4 * t, 0.3 + 0.4 * t, 0.2))
        self.all_colors = wp.array(colors, dtype=wp.vec3, device=device)
        self.all_materials = wp.array(
            [wp.vec4(0.5, 0.3, 0.0, 0.0)] * num_render, dtype=wp.vec4, device=device
        )
        self.num_render = num_render

        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        self.viewer.set_camera(
            pos=wp.vec3(10.0, -10.0, 8.0),
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
            dim=self.num_render,
            inputs=[
                self._all_rows_gpu,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.all_xforms,
                self.num_render,
            ],
            device=d,
        )

        self.viewer.log_shapes(
            "/chain",
            newton.GeoType.BOX,
            LINK_HALF,
            self.all_xforms,
            self.all_colors,
            self.all_materials,
        )
        self.viewer.log_shapes(
            "/ground",
            newton.GeoType.PLANE,
            (30.0, 30.0),
            self.ground_xform,
            self.ground_color,
            self.ground_material,
        )
        self.viewer.end_frame()

    def test_final(self):
        """The chain should hang below the anchor, not explode or pass through ground."""
        wp.synchronize_device(self.device)
        positions = self.ss.body_store.column_of("position").numpy()
        h2i = self.ss.body_store.handle_to_index.numpy()

        for i, h in enumerate(self.link_handles):
            row = int(h2i[h])
            z = positions[row][2]
            assert z > -0.5, f"Link {i} fell through ground: z={z:.4f}"

        last_row = int(h2i[self.link_handles[-1]])
        last_z = positions[last_row][2]
        assert last_z < 6.0, f"Last link did not drop below anchor: z={last_z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
