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

import os

from warp.render.imgui_manager import ImGuiManager

from newton.utils.recorder import ModelAndStateRecorder


class ReplayViewerManager(ImGuiManager):
    """A replay viewer manager for controlling playback with recorded data."""

    def __init__(self, renderer, recorder, example, window_pos=(10, 10), window_size=(400, 200)):
        super().__init__(renderer)
        if not self.is_available:
            return

        self.window_pos = window_pos
        self.window_size = window_size
        self.renderer = renderer
        self.recorder = recorder
        self.example = example
        
        # Playback state
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.loaded_file_path = None
        self.recording_type = None  # 'basic' or 'model_state'
        
        # Data containers for different recording types
        self.model_recorder = None

    def load_recording(self, file_path: str):
        """Load a recording file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Recording file not found: {file_path}")

        self.loaded_file_path = file_path

        try:
            # Try to load as BasicRecorder format first (.npz)
            if file_path.lower().endswith('.npz'):
                self._load_basic_recording(file_path)
            elif file_path.lower().endswith(('.pkl', '.pickle')):
                self._load_model_state_recording(file_path)
            else:
                # Try to auto-detect format
                try:
                    self._load_model_state_recording(file_path)
                except Exception:
                    self._load_basic_recording(file_path)

        except Exception as e:
            raise ValueError(f"Failed to load recording: {e}") from e

        # Reset playback state
        self.current_frame = 0
        self.is_playing = False

        # Update display
        self._update_frame_display()

    def _load_basic_recording(self, file_path: str):
        """Load BasicRecorder format (.npz)."""
        # Clear the existing recorder and load new data
        self.recorder.transforms_history.clear()
        self.recorder.point_clouds_history.clear()
        self.recorder.load_from_file(file_path, device="cpu")
        self.total_frames = len(self.recorder.transforms_history)
        self.recording_type = 'basic'
        print(f"Loaded BasicRecorder format: {self.total_frames} frames")

    def _load_model_state_recording(self, file_path: str):
        """Load ModelAndStateRecorder format (.pkl)."""
        self.model_recorder = ModelAndStateRecorder()
        self.model_recorder.load_from_file(file_path)
        self.total_frames = len(self.model_recorder.history)
        self.recording_type = 'model_state'
        print(f"Loaded ModelAndStateRecorder format: {self.total_frames} frames")

    def set_frame(self, frame_index: int):
        """Set the current frame for display."""
        if self.total_frames == 0:
            return

        self.current_frame = max(0, min(frame_index, self.total_frames - 1))
        self._update_frame_display()

    def _update_frame_display(self):
        """Update the renderer with the current frame data."""
        if self.total_frames == 0:
            return

        if self.recording_type == 'basic':
            self._display_basic_frame()
        elif self.recording_type == 'model_state':
            self._display_model_state_frame()

    def _display_basic_frame(self):
        """Display a frame from BasicRecorder data."""
        transforms, point_clouds = self.recorder.playback(self.current_frame)

        if transforms is not None:
            # Clear previous objects
            self.renderer.clear_objects()
            self.renderer.render_ground()

            # Render transforms as coordinate frames
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

        # Render point clouds if available
        if point_clouds:
            for i, pc in enumerate(point_clouds):
                if pc is not None and pc.size > 0:
                    self.renderer.render_points(
                        f"points_{i}",
                        pc.numpy(),
                        radius=0.01,
                        colors=(0.0, 0.7, 1.0)
                    )

    def _display_model_state_frame(self):
        """Display a frame from ModelAndStateRecorder data."""
        if not self.model_recorder or self.current_frame >= len(self.model_recorder.history):
            return

        # Get the current frame data
        frame_data = self.model_recorder.history[self.current_frame]

        # Clear previous objects
        self.renderer.clear_objects()
        self.renderer.render_ground()

        # If the frame has body_q data, visualize body positions
        if 'body_q' in frame_data:
            body_transforms = frame_data['body_q']
            for i, transform in enumerate(body_transforms):
                # Extract position (first 3 elements) and rotation (next 4 elements)
                pos = transform[:3]
                quat = transform[3:7] if len(transform) >= 7 else [0, 0, 0, 1]
                
                # Render a small sphere at each body location
                self.renderer.render_sphere(
                    f"model_body_{i}",
                    pos=pos,
                    rot=quat,
                    radius=0.1,
                    color=(0.2, 0.8, 0.3)
                )

    def render_frame(self):
        """Render frame callback for the renderer."""
        if self.is_available:
            self.draw_ui()
            
        # Update playback automatically if playing
        if self.is_playing and self.total_frames > 0:
            import time
            current_time = time.time()
            if not hasattr(self, '_last_update_time'):
                self._last_update_time = current_time
            
            dt = current_time - self._last_update_time
            frame_dt = 1.0 / (60.0 * self.playback_speed)  # 60 FPS base
            
            if dt >= frame_dt:
                next_frame = self.current_frame + 1
                if next_frame >= self.total_frames:
                    next_frame = 0  # Loop back to start
                self.set_frame(next_frame)
                self._last_update_time = current_time

    def draw_ui(self):
        """Draw the GUI interface."""
        if not self.is_available:
            return

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
            self.set_frame(0)

        self.imgui.same_line()
        if self.imgui.button("<"):
            if self.current_frame > 0:
                self.set_frame(self.current_frame - 1)

        self.imgui.same_line()
        if self.imgui.button(">"):
            if self.current_frame < self.total_frames - 1:
                self.set_frame(self.current_frame + 1)

        self.imgui.same_line()
        if self.imgui.button(">>"):
            if self.total_frames > 0:
                self.set_frame(self.total_frames - 1)

        # Frame slider
        if self.total_frames > 0:
            changed, new_frame = self.imgui.slider_int(
                "Frame",
                self.current_frame,
                0,
                self.total_frames - 1
            )
            if changed:
                self.set_frame(new_frame)

        # Frame info
        self.imgui.text(f"Frame: {self.current_frame + 1} / {self.total_frames}")

        # Speed control
        changed, new_speed = self.imgui.slider_float(
            "Speed",
            self.playback_speed,
            0.1,
            5.0
        )
        if changed:
            self.playback_speed = max(0.1, min(new_speed, 10.0))

        self.imgui.separator()

        # File operations
        if self.imgui.button("Load Recording"):
            file_path = self.open_load_file_dialog(
                title="Load Recording",
                filetypes=[
                    ("All supported", "*.npz;*.pkl;*.pickle"),
                    ("BasicRecorder (NPZ)", "*.npz"),
                    ("ModelAndStateRecorder (PKL)", "*.pkl;*.pickle"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                try:
                    self.load_recording(file_path)
                    print(f"✓ Loaded: {file_path}")
                except Exception as e:
                    print(f"✗ Error loading recording: {e}")

        # Recording info
        if self.loaded_file_path:
            self.imgui.separator()
            filename = os.path.basename(self.loaded_file_path)
            self.imgui.text(f"File: {filename}")
            self.imgui.text(f"Type: {self.recording_type}")

        self.imgui.end()