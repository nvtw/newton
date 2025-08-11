# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

import warp as wp
from warp.render.imgui_manager import ImGuiManager


class ReplayImGuiManager(ImGuiManager):
    """An ImGui manager for controlling replay playback with recorded data."""

    def __init__(self, renderer, recorder, example, window_pos=(10, 10), window_size=(400, 150)):
        super().__init__(renderer)
        if not self.is_available:
            return

        self.window_pos = window_pos
        self.window_size = window_size
        self.recorder = recorder
        self.example = example
        self.selected_frame = 0
        self.num_point_clouds_rendered = 0
        self.is_playing = False
        self.playback_speed = 1.0

    def _clear_contact_points(self):
        """Clears all rendered contact points."""
        for i in range(self.num_point_clouds_rendered):
            # use size 1 as size 0 seems to do nothing
            self.renderer.render_points(f"contact_points{i}", wp.empty(1, dtype=wp.vec3), radius=1e-2)
        self.num_point_clouds_rendered = 0

    def _update_frame(self, frame_id):
        """Update the selected frame and renderer transforms."""
        total_frames = len(self.recorder.transforms_history)
        if frame_id < 0 or frame_id >= total_frames:
            return

        self.selected_frame = frame_id
        transforms, point_clouds = self.recorder.playback(self.selected_frame)

        # Clear previous objects
        self.renderer.clear_objects()
        self.renderer.render_ground()

        # Display transforms as spheres
        if transforms:
            transforms_np = transforms.numpy()
            for i, transform in enumerate(transforms_np):
                pos = transform[:3]
                quat = transform[3:7]
                
                # Render a small sphere at each transform location
                self.renderer.render_sphere(
                    f"body_{i}",
                    pos=pos,
                    rot=quat,
                    radius=0.05,
                    color=(0.8, 0.3, 0.2)
                )

        # Display point clouds
        self._clear_contact_points()
        if point_clouds:
            for i, pc in enumerate(point_clouds):
                if pc is not None and pc.size > 0:
                    self.renderer.render_points(
                        f"contact_points{i}", pc, radius=1e-2, colors=self.renderer.get_new_color(i)
                    )
            self.num_point_clouds_rendered = len(point_clouds)

    def _update_playback(self):
        """Update automatic playback if playing."""
        if not self.is_playing:
            return

        total_frames = len(self.recorder.transforms_history)
        if total_frames == 0:
            return

        # Simple frame advance based on playback speed
        import time
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = time.time()
            return

        current_time = time.time()
        dt = current_time - self._last_update_time
        frame_dt = (1.0 / 60.0) / self.playback_speed  # Base 60 FPS

        if dt >= frame_dt:
            next_frame = (self.selected_frame + 1) % total_frames  # Loop back to start
            self._update_frame(next_frame)
            self._last_update_time = current_time

    def draw_ui(self):
        """Draw the replay controls UI."""
        total_frames = len(self.recorder.transforms_history)

        self.imgui.set_next_window_size(self.window_size[0], self.window_size[1], self.imgui.ONCE)
        self.imgui.set_next_window_position(self.window_pos[0], self.window_pos[1], self.imgui.ONCE)

        self.imgui.begin("Replay Controls")

        # Play/Pause button
        if self.is_playing:
            if self.imgui.button("Pause"):
                self.is_playing = False
        else:
            if self.imgui.button("Play"):
                self.is_playing = True

        self.imgui.same_line()

        # Step buttons
        if self.imgui.button("<<"):
            self._update_frame(0)
            self.is_playing = False

        self.imgui.same_line()
        if self.imgui.button("<"):
            self._update_frame(max(0, self.selected_frame - 1))
            self.is_playing = False

        self.imgui.same_line()
        if self.imgui.button(">"):
            self._update_frame(min(total_frames - 1, self.selected_frame + 1))
            self.is_playing = False

        self.imgui.same_line()
        if self.imgui.button(">>"):
            if total_frames > 0:
                self._update_frame(total_frames - 1)
            self.is_playing = False

        # Frame slider
        if total_frames > 0:
            changed, new_frame = self.imgui.slider_int("Timeline", self.selected_frame, 0, total_frames - 1)
            if changed:
                self._update_frame(new_frame)
                self.is_playing = False

        # Frame info
        frame_time = self.selected_frame * self.example.frame_dt if total_frames > 0 else 0.0
        self.imgui.text(f"Frame: {self.selected_frame}/{total_frames - 1 if total_frames > 0 else 0} ({frame_time:.2f}s)")

        # Speed control
        changed, new_speed = self.imgui.slider_float("Speed", self.playback_speed, 0.1, 5.0)
        if changed:
            self.playback_speed = new_speed

        self.imgui.separator()

        # Load button
        if self.imgui.button("Load Recording"):
            file_path = self.open_load_file_dialog(
                filetypes=[
                    ("Numpy Archives", "*.npz"),
                    ("Pickle files", "*.pkl;*.pickle"),
                    ("All files", "*.*")
                ],
                title="Load Recording",
            )
            if file_path:
                try:
                    self.recorder.load_from_file(file_path, device=wp.get_device())
                    self.selected_frame = 0
                    self.is_playing = False
                    if len(self.recorder.transforms_history) > 0:
                        self._update_frame(self.selected_frame)
                    print(f"✓ Loaded: {file_path}")
                except Exception as e:
                    print(f"✗ Error loading recording: {e}")

        # Recording info
        if total_frames > 0:
            self.imgui.same_line()
            self.imgui.text(f"Frames: {total_frames}")

        self.imgui.end()

        # Update playback
        self._update_playback()

    def render_frame(self):
        """Render frame callback for the renderer."""
        if self.is_available:
            self.draw_ui()