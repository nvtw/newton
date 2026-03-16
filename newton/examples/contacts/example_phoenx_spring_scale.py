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
# Example PhoenX Spring Scale
#
# A flat platform on a damped spring with small cubes dropped on top.
# Demonstrates the distance-limit spring constraint and provides a
# momentum-conservation test: the platform should settle at the
# equilibrium displacement d = total_weight / stiffness.
#
# Command: python -m newton.examples phoenx_spring_scale
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

# Scene parameters
PLATFORM_HALF = wp.vec3(1.0, 1.0, 0.1)  # wide flat platform
CUBE_HALF = 0.15  # small cubes
NUM_CUBES = 6
DROP_HEIGHT = 1.5  # height above platform to drop cubes [m]

# Spring parameters
SPRING_REST_HEIGHT = 1.0  # rest height of platform centre above ground [m]
SPRING_STIFFNESS = 500.0  # [N/m]
SPRING_DAMPING = 50.0  # [N s/m]

# Solver parameters
PGS_ITERATIONS = 12
SIM_SUBSTEPS = 8
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
    """Compose body positions/orientations into viewer transforms on GPU."""
    tid = wp.tid()
    if tid >= count:
        return
    row = handle_rows[tid]
    xforms_out[tid] = wp.transform(positions[row], orientations[row])


class Example:
    """Spring scale: a platform on a spring with cubes dropped on top.

    The platform is attached to a static ground anchor via a distance-limit
    spring constraint.  Small cubes are placed above and dropped.  After
    settling, the platform should be at ``z = rest_height - total_mass * g / k``.
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

        # 1 ground + 1 platform + NUM_CUBES
        num_bodies = 2 + NUM_CUBES
        num_shapes = num_bodies
        contact_cap = max(NUM_CUBES * 16, 128)

        self.ss = SolverState(
            body_capacity=num_bodies,
            contact_capacity=contact_cap,
            shape_count=num_shapes,
            device=device,
            default_friction=0.6,
            max_colors=12,
            joint_capacity=1,  # one spring joint
        )
        ss = self.ss

        self.pipeline = PhoenXCollisionPipeline(
            max_shapes=num_shapes, max_contacts=contact_cap, device=device,
        )

        # --- Ground (static, shape 0) ---
        h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
        self.row_ground = int(ss.body_store.handle_to_index.numpy()[h_ground])
        ss.set_shape_body(0, h_ground)
        self.pipeline.add_shape_plane(body_row=self.row_ground)

        # --- Platform (dynamic, shape 1) ---
        platform_mass = 5.0
        inv_mass = 1.0 / platform_mass
        hx, hy, hz = float(PLATFORM_HALF[0]), float(PLATFORM_HALF[1]), float(PLATFORM_HALF[2])
        inv_inertia = np.diag(np.array([
            12.0 * inv_mass / (4.0 * (hy**2 + hz**2)),
            12.0 * inv_mass / (4.0 * (hx**2 + hz**2)),
            12.0 * inv_mass / (4.0 * (hx**2 + hy**2)),
        ], dtype=np.float32))

        h_platform = ss.add_body(
            position=(0, 0, SPRING_REST_HEIGHT),
            inverse_mass=inv_mass,
            inverse_inertia_local=inv_inertia,
            linear_damping=0.999,
            angular_damping=0.99,
        )
        self.h_platform = h_platform
        self.row_platform = int(ss.body_store.handle_to_index.numpy()[h_platform])
        ss.set_shape_body(1, h_platform)
        self.pipeline.add_shape_box(
            body_row=self.row_platform,
            half_extents=(hx, hy, hz),
        )

        # --- Spring constraint: ground anchor -> platform ---
        # Ground anchor is at (0, 0, SPRING_REST_HEIGHT), platform anchor at its centre.
        # Rest distance = 0 (anchors coincide at start).
        # Limits: allow the platform to move ±0.8m from rest.
        ss.add_joint_distance_limit(
            body_handle0=h_ground,
            body_handle1=h_platform,
            anchor0_world=(0, 0, SPRING_REST_HEIGHT),
            anchor1_world=(0, 0, SPRING_REST_HEIGHT),
            limit_min=-0.8,
            limit_max=0.8,
            stiffness=SPRING_STIFFNESS,
            damping=SPRING_DAMPING,
        )

        # --- Small cubes (shapes 2..2+NUM_CUBES) ---
        cube_mass = 1.0
        cube_inv_mass = 1.0 / cube_mass
        cube_inv_inertia = np.eye(3, dtype=np.float32) * (6.0 * cube_inv_mass / (2.0 * CUBE_HALF) ** 2)

        self.cube_handles = []
        self.cube_rows = []
        for i in range(NUM_CUBES):
            # Place cubes in a grid above the platform
            col = i % 3
            row = i // 3
            x = (col - 1) * (CUBE_HALF * 3)
            y = (row - 0.5) * (CUBE_HALF * 3)
            z = SPRING_REST_HEIGHT + float(PLATFORM_HALF[2]) + DROP_HEIGHT + i * (2 * CUBE_HALF + 0.05)

            h = ss.add_body(
                position=(x, y, z),
                inverse_mass=cube_inv_mass,
                inverse_inertia_local=cube_inv_inertia,
                linear_damping=0.999,
                angular_damping=0.99,
            )
            r = int(ss.body_store.handle_to_index.numpy()[h])
            ss.set_shape_body(2 + i, h)
            self.pipeline.add_shape_box(
                body_row=r,
                half_extents=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
            )
            self.cube_handles.append(h)
            self.cube_rows.append(r)

        self.pipeline.finalize()

        # --- Track total mass for test_final ---
        self.platform_mass = platform_mass
        self.cube_mass = cube_mass
        self.total_mass = platform_mass + NUM_CUBES * cube_mass

        # --- Rendering arrays ---
        all_handles = [h_platform] + self.cube_handles
        self.num_render = len(all_handles)
        h2i = ss.body_store.handle_to_index.numpy()
        rows_np = np.array([int(h2i[h]) for h in all_handles], dtype=np.int32)
        self.handle_rows = wp.array(rows_np, dtype=wp.int32, device=device)
        self.xforms = wp.zeros(self.num_render, dtype=wp.transform, device=device)

        # Shapes for viewer
        self.render_shapes = []
        # Platform
        self.render_shapes.append({
            "body": 0,
            "shape": newton.Cuboid(hx, hy, hz),
            "color": (0.4, 0.6, 0.9),
        })
        # Cubes
        for i in range(NUM_CUBES):
            self.render_shapes.append({
                "body": 1 + i,
                "shape": newton.Cuboid(CUBE_HALF, CUBE_HALF, CUBE_HALF),
                "color": (0.9, 0.5, 0.2),
            })

        # CUDA graph capture
        self.graph = None
        try:
            self.capture()
        except Exception:
            pass

    def simulate(self):
        """Run one frame of simulation (substeps)."""
        self.ss.update_world_inertia()
        for _ in range(self.sim_substeps):
            self.ss.warm_starter.begin_frame()
            self.pipeline.collide(self.ss)
            self.ss.step(
                self.sim_dt,
                gravity=GRAVITY,
                num_iterations=PGS_ITERATIONS,
            )
            self.ss.export_impulses()

    def capture(self):
        """Capture simulate() into a CUDA graph."""
        if not self.device.is_cuda:
            return
        # Warm up
        self.simulate()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(self.device) as capture:
            self.simulate()
        self.graph = capture.graph

    def step(self):
        """Advance one frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the scene."""
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)

        bs = self.ss.body_store
        wp.launch(
            _build_xforms_kernel,
            dim=self.num_render,
            inputs=[
                self.handle_rows,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.xforms,
                self.num_render,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        xforms_np = self.xforms.numpy()

        for i, info in enumerate(self.render_shapes):
            p = xforms_np[i][:3]
            q = xforms_np[i][3:]
            self.viewer.log_shapes(
                name=f"body_{i}",
                shape=info["shape"],
                pos=p,
                rot=q,
                color=info["color"],
            )

        self.viewer.end_frame()

    def test_final(self):
        """Verify the spring scale reaches equilibrium.

        At equilibrium, spring force balances weight:
            k * d = total_mass * g
            d = total_mass * g / k

        The platform's z-position should be approximately:
            z_eq = rest_height - d
        """
        wp.synchronize_device(self.device)
        bs = self.ss.body_store
        pos_platform = bs.column_of("position").numpy()[self.row_platform]
        vel_platform = bs.column_of("velocity").numpy()[self.row_platform]

        g = abs(GRAVITY[2])
        expected_displacement = self.total_mass * g / SPRING_STIFFNESS
        expected_z = SPRING_REST_HEIGHT - expected_displacement

        # Platform should have settled near equilibrium
        assert abs(pos_platform[2] - expected_z) < 0.3, (
            f"Platform z={pos_platform[2]:.3f}, expected ~{expected_z:.3f} "
            f"(displacement={expected_displacement:.3f}m)"
        )

        # Velocity should be near zero (settled)
        speed = np.linalg.norm(vel_platform)
        assert speed < 1.0, f"Platform not settled: speed={speed:.3f} m/s"

        # All cubes should be above ground
        h2i = bs.handle_to_index.numpy()
        positions = bs.column_of("position").numpy()
        for i, h in enumerate(self.cube_handles):
            row = int(h2i[h])
            z = positions[row][2]
            assert z > -0.5, f"Cube {i} fell through ground: z={z:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
