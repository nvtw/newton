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
# Shows how to use the Newton replay viewer to visualize previously
# recorded simulation data from ModelAndStateRecorder (.json or .bin) files.
#
# Use the GUI to load recordings and scrub through frames.
#
###########################################################################

import argparse
import os

import warp as wp

import newton
import newton.utils


class ReplayViewerGL(newton.viewer.ViewerGL):
    """Custom ViewerGL with integrated replay controls."""

    def __init__(self, width=1920, height=1080, vsync=False, headless=False):
        super().__init__(width=width, height=height, vsync=vsync, headless=headless)

        # Replay-specific state
        self.model_recorder = None
        self.current_frame = 0
        self.selected_file = ""
        self.num_envs = 1

        # Set window title
        self.renderer.set_title("Newton Replay Viewer")

        # Start paused
        self._paused = True

    def _render_ui(self):
        """Override the UI rendering to include replay controls."""
        if not self.ui.is_available:
            return

        # Render the standard ViewerGL UI
        self._render_left_panel()
        self._render_stats_overlay()

        # Add our replay controls panel
        self._render_replay_controls()

    def _render_replay_controls(self):
        """Render the replay controls panel."""
        imgui = self.ui.imgui
        io = self.ui.io

        # Position the replay controls window on the right side
        window_width = 350
        window_height = 300
        imgui.set_next_window_position(io.display_size[0] - window_width - 10, 10)
        imgui.set_next_window_size(window_width, window_height)

        flags = imgui.WINDOW_NO_RESIZE

        if imgui.begin("Replay Controls", flags=flags):
            imgui.separator()

            # File selection
            imgui.text("Recording File:")
            imgui.text(self.selected_file if self.selected_file else "No file selected")

            # Disable browse button if a file is already loaded
            file_loaded = self.model_recorder and len(self.model_recorder.history) > 0

            if file_loaded:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.5, 0.5, 0.5, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.5, 0.5, 1.0)

            button_clicked = imgui.button("Browse..." if not file_loaded else "Browse... (disabled)")

            if file_loaded:
                imgui.pop_style_color(3)

            if button_clicked and not file_loaded:
                self._browse_file()

            imgui.separator()

            # Frame controls (only show if recording is loaded)
            if self.model_recorder and len(self.model_recorder.history) > 0:
                total_frames = len(self.model_recorder.history)
                imgui.text(f"Total frames: {total_frames}")

                # Frame slider
                changed, new_frame = imgui.slider_int("Frame", self.current_frame, 0, total_frames - 1)

                if changed:
                    self.current_frame = new_frame
                    self.load_frame(self.current_frame)

                # Playback controls
                imgui.separator()
                if imgui.button("First Frame"):
                    self.current_frame = 0
                    self.load_frame(self.current_frame)

                imgui.same_line()
                if imgui.button("Previous") and self.current_frame > 0:
                    self.current_frame -= 1
                    self.load_frame(self.current_frame)

                imgui.same_line()
                if imgui.button("Next") and self.current_frame < total_frames - 1:
                    self.current_frame += 1
                    self.load_frame(self.current_frame)

                imgui.same_line()
                if imgui.button("Last Frame"):
                    self.current_frame = total_frames - 1
                    self.load_frame(self.current_frame)

            else:
                imgui.text("Load a recording file (.json or .bin)")
                imgui.text("to begin playback")

        imgui.end()

    def _browse_file(self):
        """Open file browser to select recording file."""
        file_path = self.ui.open_load_file_dialog(
            filetypes=[
                ("Recording files", ("*.json", "*.bin")),
                ("JSON files", "*.json"),
                ("Binary files", "*.bin"),
                ("All files", "*.*"),
            ],
            title="Select Recording File",
        )
        if file_path:
            self.selected_file = os.path.basename(file_path)
            success = self.load_recording(file_path)
            if success:
                self.current_frame = 0

    def load_recording(self, file_path):
        """Load a recording file (.json or .bin) and set up the complete rendering pipeline."""
        print(f"Loading recording from: {file_path}")

        # Create a ModelAndStateRecorder instance
        self.model_recorder = newton.utils.ModelAndStateRecorder()

        # Load the recording file (format auto-detected from extension)
        try:
            self.model_recorder.load_from_file(file_path)
            print(f"Successfully loaded recording file with {len(self.model_recorder.history)} frames")
        except Exception as e:
            print(f"Error loading recording file: {e}")
            return False

        # Extract shape_source from the model data (for debugging)
        if self.model_recorder.deserialized_model and "shape_source" in self.model_recorder.deserialized_model:
            print("Found shape_source in recording")
            shape_source = self.model_recorder.deserialized_model["shape_source"]
            print(f"Shape source contains {len(shape_source)} entries")
        else:
            print("Warning: No shape_source found in recording")

        # Create new model and state objects
        model = newton.Model()
        state = newton.State()

        # Restore the model from the recording
        self.model_recorder.playback_model(model)
        print(f"Model restored with {model.body_count} bodies")

        # Set the model in the viewer (this will trigger setup)
        self.set_model(model)

        # Restore the first frame's state
        if len(self.model_recorder.history) > 0:
            self.model_recorder.playback(state, 0)
            print("State restored from first frame")

            # Log the initial state to the viewer
            self.log_state(state)

        # Unpause the viewer
        self._paused = False

        return True

    def load_frame(self, frame_id):
        """Load a specific frame from the recorded data."""
        if self.model_recorder and 0 <= frame_id < len(self.model_recorder.history):
            state = newton.State()
            self.model_recorder.playback(state, frame_id)
            self.log_state(state)
            print(f"Loaded frame {frame_id}")
            return True
        return False


def main():
    """Main entry point for the replay viewer example."""
    parser = argparse.ArgumentParser(
        description="Newton Physics Replay Viewer - Visualize recorded simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--window-title", type=str, default="Newton Replay Viewer", help="Window title")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Recording file to load on startup (.json or .bin for ModelAndStateRecorder)",
    )

    args = parser.parse_args()

    print("Newton Physics Replay Viewer")
    print("Use the GUI to load recordings and explore your data.")
    print("Note: Supports both JSON (.json) and binary (.bin) files from ModelAndStateRecorder.")
    if args.file:
        print(f"Loading: {args.file}")

    with wp.ScopedDevice(args.device):
        # Create the replay viewer
        viewer = ReplayViewerGL()

        # Set window title if provided
        if args.window_title != "Newton Replay Viewer":
            viewer.renderer.set_title(args.window_title)

        # Load file if specified via command line
        if args.file:
            success = viewer.load_recording(args.file)
            if not success:
                print(f"Failed to load recording: {args.file}")
                return

        # Main loop using ViewerGL's approach
        while viewer.is_running():
            viewer.begin_frame(0.0)
            viewer.end_frame()


if __name__ == "__main__":
    main()
