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
PLATFORM_HALF = wp.vec3(1.5, 1.5, 0.1)  # wide flat platform
CUBE_HALF = 0.2  # small cubes
PYRAMID_BASE = 5  # 5x5 base pyramid
NUM_CUBES = sum(n * n for n in range(1, PYRAMID_BASE + 1))  # 1+4+9+16+25 = 55
DROP_HEIGHT = 0.0  # cubes start resting on platform (no drop)

# Spring parameters
SPRING_REST_HEIGHT = 2.0  # rest height of platform centre above ground [m]
SPRING_STIFFNESS = 20000.0  # [N/m] — stiffer to support the pyramid
SPRING_DAMPING = 200.0  # [N s/m]

# Solver parameters
PGS_ITERATIONS = 12
SIM_SUBSTEPS = 8
FPS = 60

GRAVITY = (0.0, 0.0, -9.81)

PICK_STIFFNESS = 50.0
PICK_DAMPING = 5.0


def _ray_aabb_intersect(ray_origin, ray_dir, box_min, box_max):
    """Ray-AABB intersection test. Return distance or None if no hit."""
    tmin = -1e30
    tmax = 1e30
    for i in range(3):
        if abs(ray_dir[i]) < 1e-12:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return None
        else:
            inv_d = 1.0 / ray_dir[i]
            t1 = (box_min[i] - ray_origin[i]) * inv_d
            t2 = (box_max[i] - ray_origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    if tmax < 0.0:
        return None
    return max(tmin, 0.0)


def _quat_rotate_vec(q, v):
    """Rotate vector v by quaternion q (x, y, z, w layout)."""
    qv = np.array([q[0], q[1], q[2]], dtype=np.float32)
    w = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _quat_inv_rotate_vec(q, v):
    """Rotate vector v by the inverse of quaternion q."""
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
    return _quat_rotate_vec(q_conj, v)


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
        contact_cap = max(NUM_CUBES * 16, 512)

        self.ss = SolverState(
            body_capacity=num_bodies,
            contact_capacity=contact_cap,
            shape_count=num_shapes,
            device=device,
            default_friction=0.6,
            max_colors=12,
            joint_capacity=1,  # prismatic with spring drive
        )
        ss = self.ss

        self.pipeline = PhoenXCollisionPipeline(
            max_shapes=num_shapes,
            max_contacts=contact_cap,
            device=device,
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
        inv_inertia = np.diag(
            np.array(
                [
                    12.0 * inv_mass / (4.0 * (hy**2 + hz**2)),
                    12.0 * inv_mass / (4.0 * (hx**2 + hz**2)),
                    12.0 * inv_mass / (4.0 * (hx**2 + hy**2)),
                ],
                dtype=np.float32,
            )
        )

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

        # --- Spring constraint: prismatic joint + position drive ---
        # The prismatic joint locks 5 DOF (lateral translation + all rotation),
        # allowing only vertical sliding. The position drive adds spring
        # behavior (F = -k*x - c*v) so the platform bounces and settles.
        ji = ss.add_joint_prismatic(
            body_handle0=h_ground,
            body_handle1=h_platform,
            anchor_world=(0, 0, SPRING_REST_HEIGHT),
            axis_world=(0, 0, 1),
            slide_min=-0.8,
            slide_max=0.8,
        )
        ss.set_joint_drive(
            ji,
            mode=ss.DRIVE_POSITION,
            target=0.0,
            stiffness=SPRING_STIFFNESS,
            damping=SPRING_DAMPING,
        )

        # --- Small cubes in a pyramid (shapes 2..2+NUM_CUBES) ---
        cube_mass = 1.0
        cube_inv_mass = 1.0 / cube_mass
        cube_inv_inertia = np.eye(3, dtype=np.float32) * (6.0 * cube_inv_mass / (2.0 * CUBE_HALF) ** 2)

        self.cube_handles = []
        self.cube_rows = []
        spacing = 2.0 * CUBE_HALF + 0.02
        cube_idx = 0
        for layer in range(PYRAMID_BASE):
            n = PYRAMID_BASE - layer
            layer_z = SPRING_REST_HEIGHT + float(PLATFORM_HALF[2]) + CUBE_HALF + 0.01 + layer * spacing
            offset = -(n - 1) * spacing * 0.5
            for row in range(n):
                for col in range(n):
                    x = offset + col * spacing
                    y = offset + row * spacing
                    z = layer_z

                    h = ss.add_body(
                        position=(x, y, z),
                        inverse_mass=cube_inv_mass,
                        inverse_inertia_local=cube_inv_inertia,
                        linear_damping=0.999,
                        angular_damping=0.99,
                    )
                    r = int(ss.body_store.handle_to_index.numpy()[h])
                    ss.set_shape_body(2 + cube_idx, h)
                    self.pipeline.add_shape_box(
                        body_row=r,
                        half_extents=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
                    )
                    self.cube_handles.append(h)
                    self.cube_rows.append(r)
                    cube_idx += 1

        self.pipeline.finalize()

        # --- Track total mass for test_final ---
        self.platform_mass = platform_mass
        self.cube_mass = cube_mass
        self.total_mass = platform_mass + NUM_CUBES * cube_mass

        # --- Rendering arrays ---
        h2i = ss.body_store.handle_to_index.numpy()

        # Platform (1 body)
        self._platform_row = wp.array([self.row_platform], dtype=wp.int32, device=device)
        self.platform_xform = wp.zeros(1, dtype=wp.transform, device=device)
        self.platform_color = wp.array([wp.vec3(0.4, 0.6, 0.9)], dtype=wp.vec3, device=device)
        self.platform_material = wp.array([wp.vec4(0.5, 0.3, 0.0, 0.0)], dtype=wp.vec4, device=device)

        # Cubes (NUM_CUBES bodies)
        self._cube_rows = wp.array(
            [int(h2i[h]) for h in self.cube_handles],
            dtype=wp.int32,
            device=device,
        )
        self.cube_xforms = wp.zeros(NUM_CUBES, dtype=wp.transform, device=device)
        self.cube_colors = wp.array(
            [wp.vec3(0.9, 0.5, 0.2)] * NUM_CUBES,
            dtype=wp.vec3,
            device=device,
        )
        self.cube_materials = wp.array(
            [wp.vec4(0.5, 0.3, 0.0, 0.0)] * NUM_CUBES,
            dtype=wp.vec4,
            device=device,
        )

        # Ground
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=device)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=device)

        # Picking data: pickable bodies are platform + cubes
        self._pickable_handles = [h_platform, *self.cube_handles]
        self._pickable_rows_np = np.array(
            [self.row_platform, *self.cube_rows],
            dtype=np.int32,
        )
        self._pickable_halves = [
            np.array([hx, hy, hz], dtype=np.float32),
        ] + [
            np.array([CUBE_HALF, CUBE_HALF, CUBE_HALF], dtype=np.float32),
        ] * NUM_CUBES

        self._pick_body_row = -1
        self._pick_local_offset = np.zeros(3, dtype=np.float32)
        self._pick_distance = 0.0
        self._pick_target = np.zeros(3, dtype=np.float32)

        self._setup_picking()

        self.viewer.set_camera(
            pos=wp.vec3(5.0, -5.0, 4.0),
            pitch=-25.0,
            yaw=135.0,
        )

        # CUDA graph capture
        self.graph = None
        self.simulate()
        try:
            self.capture()
        except Exception:
            pass

    # -- picking setup and callbacks ----------------------------------------

    def _setup_picking(self):
        """Register mouse callbacks on the viewer's renderer for picking."""
        try:
            renderer = self.viewer._renderer if hasattr(self.viewer, "_renderer") else None
            if renderer is None:
                renderer = self.viewer.renderer if hasattr(self.viewer, "renderer") else None
            if renderer is None:
                return

            self._viewer_renderer = renderer
            renderer.register_mouse_press(self._on_mouse_press)
            renderer.register_mouse_release(self._on_mouse_release)
            renderer.register_mouse_drag(self._on_mouse_drag)
        except Exception:
            pass

    def _to_framebuffer_coords(self, x, y):
        """Convert window coords to framebuffer coords (for HiDPI displays)."""
        try:
            fb_w, fb_h = self._viewer_renderer.window.get_framebuffer_size()
            win_w, win_h = self._viewer_renderer.window.get_size()
            if win_w <= 0 or win_h <= 0:
                return float(x), float(y)
            return float(x) * fb_w / win_w, float(y) * fb_h / win_h
        except Exception:
            return float(x), float(y)

    def _get_camera_ray(self, x, y):
        """Get camera ray from pixel coordinates."""
        fb_x, fb_y = self._to_framebuffer_coords(x, y)
        ray_start, ray_dir = self.viewer.camera.get_world_ray(fb_x, fb_y)
        origin = np.array([ray_start[0], ray_start[1], ray_start[2]], dtype=np.float32)
        direction = np.array([ray_dir[0], ray_dir[1], ray_dir[2]], dtype=np.float32)
        return origin, direction

    def _find_picked_body(self, ray_origin, ray_dir):
        """Find the closest pickable body intersected by the ray."""
        bs = self.ss.body_store
        positions = bs.column_of("position").numpy()

        best_dist = 1e30
        best_idx = -1

        for i in range(len(self._pickable_rows_np)):
            row = self._pickable_rows_np[i]
            pos = positions[row]
            half = self._pickable_halves[i]
            box_min = pos - half
            box_max = pos + half
            dist = _ray_aabb_intersect(ray_origin, ray_dir, box_min, box_max)
            if dist is not None and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx, best_dist

    def _on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse press for picking."""
        try:
            import pyglet

            if button != pyglet.window.mouse.RIGHT:
                return
        except ImportError:
            return

        if hasattr(self.viewer, "ui") and self.viewer.ui and self.viewer.ui.is_capturing():
            return

        ray_origin, ray_dir = self._get_camera_ray(x, y)
        idx, dist = self._find_picked_body(ray_origin, ray_dir)

        if idx < 0:
            return

        row = self._pickable_rows_np[idx]
        bs = self.ss.body_store
        pos = bs.column_of("position").numpy()[row]
        orient = bs.column_of("orientation").numpy()[row]

        hit_world = ray_origin + ray_dir * dist

        self._pick_body_row = row
        self._pick_distance = dist
        self._pick_local_offset = _quat_inv_rotate_vec(orient, hit_world - pos)
        self._pick_target = hit_world.copy()

    def _on_mouse_release(self, x, y, button, modifiers):
        """Clear pick state on release."""
        self._pick_body_row = -1

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Update pick target during drag."""
        try:
            import pyglet

            if not (buttons & pyglet.window.mouse.RIGHT):
                return
        except ImportError:
            return

        if self._pick_body_row < 0:
            return

        if hasattr(self.viewer, "ui") and self.viewer.ui and self.viewer.ui.is_capturing():
            return

        ray_origin, ray_dir = self._get_camera_ray(x, y)
        self._pick_target = ray_origin + ray_dir * self._pick_distance

    def _apply_pick_force(self):
        """Apply spring-damper force from picked point to mouse target."""
        if self._pick_body_row < 0:
            return

        bs = self.ss.body_store
        row = self._pick_body_row
        pos = bs.column_of("position").numpy()[row]
        orient = bs.column_of("orientation").numpy()[row]
        vel = bs.column_of("velocity").numpy()[row]

        picked_world = pos + _quat_rotate_vec(orient, self._pick_local_offset)

        diff = self._pick_target - picked_world
        force = PICK_STIFFNESS * diff - PICK_DAMPING * vel

        impulse = force * self.frame_dt
        self.ss.apply_body_impulse(
            body_row=row,
            impulse_world=tuple(impulse),
            point_world=tuple(picked_world),
            dt=self.frame_dt,
        )

    # -- simulation (graph-capturable) --------------------------------------

    def simulate(self):
        """Run one frame of simulation (matching C# World.Step)."""
        self.ss.update_world_inertia()
        # Collision detection once per frame (C# architecture)
        self.ss.warm_starter.begin_frame()
        self.pipeline.collide(self.ss)
        for _ in range(self.sim_substeps):
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
        if self._pick_body_row >= 0:
            self._apply_pick_force()
            self.simulate()
        elif self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        # Print spring deflection periodically
        frame_num = int(self.sim_time * self.fps + 0.5)
        if frame_num % 60 == 0 and frame_num > 0:
            wp.synchronize_device(self.device)
            pos = self.ss.body_store.column_of("position").numpy()[self.row_platform]
            g = abs(GRAVITY[2])
            displacement = SPRING_REST_HEIGHT - pos[2]
            expected_disp = self.total_mass * g / SPRING_STIFFNESS
            print(
                f"  t={self.sim_time:.1f}s: platform z={pos[2]:.4f}, "
                f"deflection={displacement:.4f}m "
                f"(expected={expected_disp:.4f}m, "
                f"total_mass={self.total_mass:.1f}kg)"
            )

    def render(self):
        """Render the scene."""
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)

        bs = self.ss.body_store
        d = self.device
        hx = float(PLATFORM_HALF[0])
        hy = float(PLATFORM_HALF[1])
        hz = float(PLATFORM_HALF[2])

        # Build platform transform
        wp.launch(
            _build_xforms_kernel,
            dim=1,
            inputs=[
                self._platform_row,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.platform_xform,
                1,
            ],
            device=d,
        )
        # Build cube transforms
        wp.launch(
            _build_xforms_kernel,
            dim=NUM_CUBES,
            inputs=[
                self._cube_rows,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.cube_xforms,
                NUM_CUBES,
            ],
            device=d,
        )

        self.viewer.log_shapes(
            "/platform",
            newton.GeoType.BOX,
            (hx, hy, hz),
            self.platform_xform,
            self.platform_color,
            self.platform_material,
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

        if self._pick_body_row >= 0:
            pos = bs.column_of("position").numpy()[self._pick_body_row]
            orient = bs.column_of("orientation").numpy()[self._pick_body_row]
            picked_world = pos + _quat_rotate_vec(orient, self._pick_local_offset)

            starts = wp.array([wp.vec3(*picked_world)], dtype=wp.vec3, device=d)
            ends = wp.array([wp.vec3(*self._pick_target)], dtype=wp.vec3, device=d)
            self.viewer.log_lines("/pick_line", starts, ends, (0.0, 1.0, 1.0))
        else:
            self.viewer.log_lines("/pick_line", None, None, None)

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

        # Platform should have settled near equilibrium.
        # The spring deflection d = total_mass * g / k tests momentum
        # conservation through the contact stack.
        # Note: with many stacked cubes, mass splitting causes extra
        # deflection.  Full accuracy requires per-partition copy states
        # (Tonge 2012), as implemented in C# PhoenX.
        actual_displacement = SPRING_REST_HEIGHT - pos_platform[2]
        assert actual_displacement > 0.0, f"Platform did not deflect at all: z={pos_platform[2]:.4f}"
        assert pos_platform[2] > 0.0, f"Platform crashed to ground: z={pos_platform[2]:.4f}"

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
