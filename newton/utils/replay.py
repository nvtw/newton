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

from warp.render.imgui_manager import ImGuiManager
from newton.utils.recorder import ModelAndStateRecorder


class ReplayImGuiManager(ImGuiManager):
    """An ImGui manager for controlling replay playback with ModelAndStateRecorder data."""

    def __init__(self, renderer, recorder, example, window_pos=(10, 10), window_size=(400, 150)):
        super().__init__(renderer)
        if not self.is_available:
            return

        self.window_pos = window_pos
        self.window_size = window_size
        self.recorder = recorder  # ModelAndStateRecorder instance
        self.example = example
        self.selected_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0

    def _update_frame(self, frame_id):
        """Update the selected frame using ModelAndStateRecorder."""
        if not self.recorder or not hasattr(self.recorder, "history"):
            return
        total_frames = len(self.recorder.history)
        if frame_id < 0 or frame_id >= total_frames:
            return

        self.selected_frame = frame_id
        # Update the example's state with the selected frame
        if hasattr(self.example, "state") and self.example.state is not None:
            self.recorder.playback(self.example.state, self.selected_frame)

    def _update_playback(self):
        """Update automatic playback if playing."""
        if not self.is_playing or not self.recorder:
            return

        total_frames = len(self.recorder.history)
        if total_frames == 0:
            return

        # Simple frame advance based on playback speed
        import time

        if not hasattr(self, "_last_update_time"):
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
        if not self.recorder:
            return
        total_frames = len(self.recorder.history)

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
        self.imgui.text(
            f"Frame: {self.selected_frame}/{total_frames - 1 if total_frames > 0 else 0} ({frame_time:.2f}s)"
        )

        # Speed control
        changed, new_speed = self.imgui.slider_float("Speed", self.playback_speed, 0.1, 5.0)
        if changed:
            self.playback_speed = new_speed

        self.imgui.separator()

        # Load button
        if self.imgui.button("Load Recording"):
            file_path = self.open_load_file_dialog(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Recording",
            )
            if file_path:
                try:
                    # Create new recorder and load the file
                    new_recorder = ModelAndStateRecorder()
                    new_recorder.load_from_file(file_path)

                    # Update example with loaded data
                    if hasattr(self.example, "load_raw_simulation"):
                        success = self.example.load_raw_simulation(file_path)
                        if success:
                            self.recorder = self.example.model_recorder
                            self.selected_frame = 0
                            self.is_playing = False
                            if len(self.recorder.history) > 0:
                                self._update_frame(self.selected_frame)
                            print(f"✓ Loaded: {file_path}")
                        else:
                            print(f"✗ Failed to load: {file_path}")
                    else:
                        print("✗ Example does not support loading recordings")
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
