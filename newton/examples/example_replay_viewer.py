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

###########################################################################
# Example Replay Viewer
#
# A simple ImGui-based replay viewer that loads JSON files recorded with
# ModelAndStateRecorder and allows scrubbing through frames with a slider.
#
###########################################################################

import argparse
import os

import warp as wp
from warp.render.imgui_manager import ImGuiManager

import newton
import newton.utils
from newton.sim.model import Model
from newton.sim.state import State
from newton.utils.recorder import ModelAndStateRecorder


class ReplayViewerGUI(ImGuiManager):
    """Simple ImGui interface for the replay viewer."""

    def __init__(self, renderer, example, window_pos=(10, 10), window_size=(300, 200)):
        super().__init__(renderer)
        if not self.is_available:
            return
        
        self.window_pos = window_pos
        self.window_size = window_size
        self.example = example
        self.current_frame = 0
        self.selected_file = ""

    def draw_ui(self):
        """Draw the ImGui interface."""
        self.imgui.set_next_window_size(self.window_size[0], self.window_size[1], self.imgui.ONCE)
        self.imgui.set_next_window_position(self.window_pos[0], self.window_pos[1], self.imgui.ONCE)

        self.imgui.begin("Replay Controls")

        # File selection
        self.imgui.text("JSON File:")
        self.imgui.same_line()
        self.imgui.text(self.selected_file if self.selected_file else "No file selected")

        if self.imgui.button("Browse..."):
            self._browse_file()

        self.imgui.separator()

        # Frame controls (only show if recording is loaded)
        if self.example.model_recorder and len(self.example.model_recorder.history) > 0:
            total_frames = len(self.example.model_recorder.history)
            self.imgui.text(f"Total frames: {total_frames}")

            # Frame slider
            changed, new_frame = self.imgui.slider_int(
                "Frame",
                self.current_frame,
                0,
                total_frames - 1
            )

            if changed:
                self.current_frame = new_frame
                self.example.load_frame(self.current_frame)

            # Playback controls
            self.imgui.separator()
            if self.imgui.button("First Frame"):
                self.current_frame = 0
                self.example.load_frame(self.current_frame)

            self.imgui.same_line()
            if self.imgui.button("Previous") and self.current_frame > 0:
                self.current_frame -= 1
                self.example.load_frame(self.current_frame)

            self.imgui.same_line()
            if self.imgui.button("Next") and self.current_frame < total_frames - 1:
                self.current_frame += 1
                self.example.load_frame(self.current_frame)

            self.imgui.same_line()
            if self.imgui.button("Last Frame"):
                self.current_frame = total_frames - 1
                self.example.load_frame(self.current_frame)

        else:
            self.imgui.text("Load a JSON file to begin playback")

        self.imgui.end()

    def _browse_file(self):
        """Open file browser to select JSON file."""
        file_path = self.open_load_file_dialog(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Select Recording File"
        )
        
        if file_path:
            self.selected_file = os.path.basename(file_path)
            success = self.example.load_recording(file_path)
            if success:
                self.current_frame = 0


class Example:
    def __init__(self, stage_path="example_replay_viewer.usd"):
        # Initialize empty model and state - will be loaded from recording
        self.model = None
        self.state = State()
        self.model_recorder = None
        
        # Initialize simulation parameters (copied from quadruped)
        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        if stage_path:
            # Start with empty model initially
            self.renderer = newton.utils.SimRendererOpenGL(self.model, path=stage_path)
            self.gui = ReplayViewerGUI(self.renderer, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
        else:
            self.renderer = None
            self.gui = None

        # Start in paused mode since we're viewing recordings
        if self.renderer:
            self.renderer.paused = True

    @property
    def paused(self):
        if self.renderer:
            return self.renderer.paused
        return True

    @paused.setter
    def paused(self, value):
        if self.renderer:
            self.renderer.paused = value

    def load_recording(self, file_path):
        """Load a JSON recording file."""
        print(f"Loading recording from: {file_path}")

        # Create a ModelAndStateRecorder instance
        self.model_recorder = ModelAndStateRecorder()

        # Load the JSON file
        try:
            self.model_recorder.load_from_file(file_path)
            print(f"Successfully loaded JSON file with {len(self.model_recorder.history)} frames")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return False

        # Create new model and state objects
        self.model = Model()
        self.state = State()

        # Restore the model from the recording
        self.model_recorder.playback_model(self.model)
        print(f"Model restored with {self.model.body_count} bodies")

        # Update renderer with new model
        if self.renderer:
            self.renderer.model = self.model

        # Restore the first frame's state
        if len(self.model_recorder.history) > 0:
            self.model_recorder.playback(self.state, 0)
            print("State restored from first frame")

        return True

    def load_frame(self, frame_id):
        """Load a specific frame from the recorded data."""
        if self.model_recorder and 0 <= frame_id < len(self.model_recorder.history):
            self.model_recorder.playback(self.state, frame_id)
            return True
        return False

    def step(self):
        """Step function - no simulation since we're just viewing recordings."""
        if self.paused:
            return
        # In replay mode, we don't simulate - just update time
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            if not self.paused and self.model is not None and self.state is not None:
                self.renderer.render(self.state)
            else:
                # in paused mode, the GUI will handle rendering
                if self.model is not None and self.state is not None:
                    self.renderer.render(self.state)
            self.renderer.end_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_replay_viewer.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Recording file to load on startup (.json)"
    )

    args = parser.parse_known_args()[0]

    print("Newton Physics Replay Viewer")
    print("Use the GUI to load JSON recordings and scrub through frames.")

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        # # Load file if specified via command line
        # if args.file:
        #     print(f"Loading file: {args.file}")
        #     example.load_recording(args.file)

        if example.renderer:
            while example.renderer.is_running():
                example.step()
                example.render()