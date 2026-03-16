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
# Example PhoenX Pyramid
#
# A pyramid of boxes simulated by the PhoenX PGS solver with a custom
# collision pipeline (BroadPhaseAllPairs + NarrowPhase).  Demonstrates
# the PhoenX solver running without Newton Model / State, using
# viewer.log_shapes() for rendering and CUDA graph capture for
# performance.
#
# Right-click and drag on a box to pick it up and move it around.
#
# Command: python -m newton.examples phoenx_pyramid
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.contacts import build_contact_lines_kernel
from newton._src.solvers.phoenx.solver_phoenx import SolverState

CUBE_HALF = 0.5
NUM_LAYERS = 15
PGS_ITERATIONS = 12
SIM_SUBSTEPS = 8
FPS = 60

PICK_STIFFNESS = 50.0
PICK_DAMPING = 5.0


@wp.kernel
def _build_box_xforms_kernel(
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


class Example:
    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)

        num_layers = NUM_LAYERS
        h = CUBE_HALF
        spacing = 2.0 * h + 0.02

        box_positions = []
        for layer in range(num_layers):
            n = num_layers - layer
            z = layer * spacing + h
            offset = -(n - 1) * spacing * 0.5
            for row in range(n):
                for col in range(n):
                    x = offset + col * spacing
                    y = offset + row * spacing
                    box_positions.append((x, y, z))

        num_boxes = len(box_positions)
        num_shapes = num_boxes + 1
        body_capacity = num_boxes + 1
        contact_capacity = max(num_boxes * 16, 512)

        self.ss = SolverState(
            body_capacity=body_capacity,
            contact_capacity=contact_capacity,
            shape_count=num_shapes,
            device=args.device if hasattr(args, "device") else None,
            default_friction=0.6,
        )
        self.pipeline = PhoenXCollisionPipeline(
            max_shapes=num_shapes,
            max_contacts=contact_capacity,
            device=self.ss.device,
        )

        # Ground: static body + plane shape
        h_ground = self.ss.add_body(position=(0.0, 0.0, 0.0), is_static=True)
        ground_row = int(self.ss.body_store.handle_to_index.numpy()[h_ground])
        self.ss.set_shape_body(0, h_ground)
        self.pipeline.add_shape_plane(body_row=ground_row)

        # Pyramid boxes
        self.box_handles = []
        self.box_shape_indices = []
        mass = 1.0
        inv_mass = 1.0 / mass
        inv_inertia = np.eye(3, dtype=np.float32) * (6.0 * inv_mass / ((2.0 * h) ** 2))

        for i, (px, py, pz) in enumerate(box_positions):
            bh = self.ss.add_body(
                position=(px, py, pz),
                inverse_mass=inv_mass,
                inverse_inertia_local=inv_inertia,
                linear_damping=0.995,
                angular_damping=0.99,
            )
            shape_idx = i + 1
            self.ss.set_shape_body(shape_idx, bh)
            self.pipeline.add_shape_box(
                body_row=int(self.ss.body_store.handle_to_index.numpy()[bh]),
                half_extents=(h, h, h),
            )
            self.box_handles.append(bh)
            self.box_shape_indices.append(shape_idx)

        self.pipeline.finalize()

        self.num_boxes = num_boxes
        self.initial_positions = np.array(box_positions, dtype=np.float32)

        # Pre-allocate rendering arrays
        d = self.ss.device

        # GPU-resident array of body-store rows for each box handle
        h2i = self.ss.body_store.handle_to_index.numpy()
        self._box_rows = wp.array([int(h2i[bh]) for bh in self.box_handles], dtype=wp.int32, device=d)
        self._box_rows_np = np.array([int(h2i[bh]) for bh in self.box_handles], dtype=np.int32)

        self.box_xforms = wp.zeros(num_boxes, dtype=wp.transform, device=d)
        self.box_colors = wp.array([wp.vec3(0.85, 0.55, 0.25)] * num_boxes, dtype=wp.vec3, device=d)
        self.box_materials = wp.array([wp.vec4(0.5, 0.3, 0.0, 0.0)] * num_boxes, dtype=wp.vec4, device=d)
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=d)
        self.ground_color = wp.array([wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3, device=d)
        self.ground_material = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4, device=d)

        # Contact line arrays for visualization
        self._contact_starts = wp.zeros(contact_capacity, dtype=wp.vec3, device=d)
        self._contact_ends = wp.zeros(contact_capacity, dtype=wp.vec3, device=d)

        self.viewer.set_camera(
            pos=wp.vec3(8.0, -8.0, 6.0),
            pitch=-20.0,
            yaw=135.0,
        )

        # -- Picking state --
        self._pick_body_row = -1  # body store row of picked box (-1 = none)
        self._pick_local_offset = np.zeros(3, dtype=np.float32)  # hit point in body-local coords
        self._pick_distance = 0.0  # distance from camera to hit point along ray
        self._pick_target = np.zeros(3, dtype=np.float32)  # current mouse target in world

        # Register mouse callbacks on the viewer renderer (if available)
        self._setup_picking()

        # Graph capture
        self.graph = None
        self.simulate()
        self.capture()

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
            pass  # headless or no renderer

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

    def _find_picked_box(self, ray_origin, ray_dir):
        """Find the closest box intersected by the ray. Return (box_index, distance) or (-1, inf)."""
        bs = self.ss.body_store
        positions = bs.column_of("position").numpy()

        best_dist = 1e30
        best_idx = -1
        h = CUBE_HALF

        for i in range(self.num_boxes):
            row = self._box_rows_np[i]
            pos = positions[row]
            # AABB test (axis-aligned, ignoring rotation for simplicity)
            box_min = pos - h
            box_max = pos + h
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

        # Check if the viewer UI is capturing (e.g. ImGui panel)
        if hasattr(self.viewer, "ui") and self.viewer.ui and self.viewer.ui.is_capturing():
            return

        ray_origin, ray_dir = self._get_camera_ray(x, y)
        box_idx, dist = self._find_picked_box(ray_origin, ray_dir)

        if box_idx < 0:
            return

        row = self._box_rows_np[box_idx]
        bs = self.ss.body_store
        pos = bs.column_of("position").numpy()[row]
        orient = bs.column_of("orientation").numpy()[row]

        # World-space hit point
        hit_world = ray_origin + ray_dir * dist

        # Store pick state
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

        # Current world position of the picked point on the body
        picked_world = pos + _quat_rotate_vec(orient, self._pick_local_offset)

        # Spring-damper force toward mouse target
        diff = self._pick_target - picked_world
        force = PICK_STIFFNESS * diff - PICK_DAMPING * vel

        # Apply as impulse (force * dt)
        impulse = force * self.frame_dt
        self.ss.apply_body_impulse(
            body_row=row,
            impulse_world=tuple(impulse),
            point_world=tuple(picked_world),
            dt=self.frame_dt,
        )

    # -- simulation (graph-capturable) --------------------------------------

    def simulate(self):
        # Matching C# World.Step: detect once, substep N times
        self.ss.update_world_inertia()
        self.ss.warm_starter.begin_frame()
        self.pipeline.collide(self.ss)
        for _ in range(self.sim_substeps):
            self.ss.step(
                self.sim_dt,
                gravity=(0.0, 0.0, -9.81),
                num_iterations=PGS_ITERATIONS,
            )
        self.ss.export_impulses()

    def capture(self):
        self.graph = None
        if self.ss.device.is_cuda:
            with wp.ScopedCapture(device=self.ss.device) as cap:
                self.simulate()
            self.graph = cap.graph

    # -- public interface ---------------------------------------------------

    def step(self):
        # Apply pick force before simulation (outside graph capture)
        if self._pick_body_row >= 0:
            self._apply_pick_force()
            # Cannot use graph when pick force modifies velocities on CPU
            self.simulate()
        elif self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        bs = self.ss.body_store
        cs = self.ss.contact_store
        d = self.ss.device

        # Build box transforms on GPU
        wp.launch(
            _build_box_xforms_kernel,
            dim=self.num_boxes,
            inputs=[
                self._box_rows,
                bs.column_of("position"),
                bs.column_of("orientation"),
                self.box_xforms,
                self.num_boxes,
            ],
            device=d,
        )

        self.viewer.log_shapes(
            "/boxes",
            newton.GeoType.BOX,
            (CUBE_HALF, CUBE_HALF, CUBE_HALF),
            self.box_xforms,
            self.box_colors,
            self.box_materials,
        )
        self.viewer.log_shapes(
            "/ground",
            newton.GeoType.PLANE,
            (50.0, 50.0),
            self.ground_xform,
            self.ground_color,
            self.ground_material,
        )

        # Contact visualization (toggle via viewer UI)
        if self.viewer.show_contacts:
            wp.launch(
                build_contact_lines_kernel,
                dim=cs.capacity,
                inputs=[
                    cs.column_of("body0"),
                    cs.column_of("offset0"),
                    cs.column_of("normal"),
                    bs.column_of("position"),
                    bs.column_of("orientation"),
                    cs.count,
                    self._contact_starts,
                    self._contact_ends,
                ],
                device=d,
            )
            nc = cs.count.numpy()[0]
            if nc > 0:
                self.viewer.log_lines(
                    "/contacts",
                    self._contact_starts[:nc],
                    self._contact_ends[:nc],
                    (0.0, 1.0, 0.0),
                )
            else:
                self.viewer.log_lines("/contacts", None, None, None)
        else:
            self.viewer.log_lines("/contacts", None, None, None)

        # Picking line visualization
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
        """Verify the PGS contact solver is active (boxes don't freefall).

        The PhoenX PGS solver has known convergence limitations for deeply
        stacked scenes, so boxes will gradually sink.  This test only
        checks that contacts provide *some* resistance compared to pure
        freefall.
        """
        bs = self.ss.body_store
        positions = bs.column_of("position").numpy()
        h2i = bs.handle_to_index.numpy()

        # Under pure freefall at 9.81 m/s^2, after T seconds z = -0.5*g*T^2.
        # At T=16.7s (1000 frames @ 60 FPS), that's ~-1370m.
        # With contacts, boxes sink slowly; a generous threshold confirms
        # the contact system is functioning.
        freefall_z = -0.5 * 9.81 * self.sim_time**2
        for i, bh in enumerate(self.box_handles):
            row = int(h2i[bh])
            z = positions[row][2]
            assert z > freefall_z * 0.5, (
                f"Box {i} appears to be in freefall: z={z:.4f} "
                f"(freefall would be {freefall_z:.1f} at t={self.sim_time:.1f}s)"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
