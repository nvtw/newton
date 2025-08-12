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
# recorded simulation data from ModelAndStateRecorder (.json) files.
#
# Use the GUI to load recordings and scrub through frames.
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
    def __init__(self, stage_path="Newton_Replay_Viewer.usd"):
        # Model and state will be set when loading recordings
        self.model = None
        self.state = State()
        self.solver = None
        self.model_recorder = None
        self.num_envs = 1  # Default number of environments

        # Set up renderer and replay components
        if stage_path:
            # Create SimRendererOpenGL without a model initially
            self.renderer = newton.utils.SimRendererOpenGL(model=None, path=stage_path)
            # GUI will be set up when loading recordings
            self.gui = ReplayViewerGUI(self.renderer, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
        else:
            self.renderer = None
            self.gui = None

        # Start in paused mode
        if self.renderer:
            self.renderer.paused = True

        # Frame timing for GUI
        self.frame_dt = 1.0 / 60.0  # 60 FPS

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

    def load_raw_simulation(self, file_path):
        """
        Raw testing method to load a simulation JSON file and set up model/state.

        Args:
            file_path (str): Path to the JSON file (e.g., "C:/tmp/my_simulation.json")
        """
        print(f"Loading simulation from: {file_path}")

        # Create a ModelAndStateRecorder instance
        self.model_recorder = ModelAndStateRecorder()

        # Load the JSON file
        try:
            self.model_recorder.load_from_file(file_path)
            print(f"Successfully loaded JSON file with {len(self.model_recorder.history)} frames")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return False

        # Extract shape_source from the model data
        if self.model_recorder.deserialized_model and "shape_source" in self.model_recorder.deserialized_model:
            print("Found shape_source in recording")
            shape_source = self.model_recorder.deserialized_model["shape_source"]
            print(f"Shape source contains {len(shape_source)} entries")
        else:
            print("Warning: No shape_source found in recording")
            shape_source = []

        # Create empty model and state objects
        self.model = Model()
        self.state = State()

        # Use playback_model to restore the model
        self.model_recorder.playback_model(self.model)
        print(f"Model restored with {self.model.body_count} bodies")

        # Use playback to restore the first frame's state
        if len(self.model_recorder.history) > 0:
            self.model_recorder.playback(self.state, 0)
            print("State restored from first frame")

        # Set up the renderer with the loaded model
        self._setup_renderer_with_model()

        return True

    def _setup_renderer_with_model(self):
        """Set up the renderer with the loaded model using the provided pattern."""
        if not self.renderer or not self.model:
            return

        print("Setting up renderer with model...")

        # Update renderer model
        self.renderer.model = self.model

        # Setup body names and environments
        if self.model.body_count:
            bodies_per_env = self.model.body_count // self.num_envs
            self.renderer.body_env = []
            self.renderer.body_names = self.renderer.populate_bodies(
                self.model.body_key, bodies_per_env, self.renderer.body_env
            )
            print(f"Set up {len(self.renderer.body_names)} bodies for rendering")

        # print("SHAPE SOURCE: ", self.model.shape_source)

        # Setup shapes if available
        if self.model.shape_count:
            self.renderer.geo_shape = {}
            self.renderer_instance_count = self.renderer.populate_shapes(
                self.renderer.body_names,
                self.renderer.geo_shape,
                self.model.shape_body.numpy(),
                self.model.shape_source,
                self.model.shape_type.numpy(),
                self.model.shape_scale.numpy(),
                self.model.shape_thickness.numpy(),
                self.model.shape_is_solid.numpy(),
                self.model.shape_transform.numpy(),
                self.model.shape_flags.numpy(),
                self.model.shape_key,
            )
            print(f"Set up {self.model.shape_count} shapes for rendering")

            # Render ground plane if present
            if hasattr(self.model, "ground") and self.model.ground:
                self.renderer.render_ground(plane=self.model.ground_plane_params)
                print("Ground plane rendered")

        # Complete setup if method exists
        if hasattr(self.renderer, "complete_setup"):
            self.renderer.complete_setup()
            print("Renderer setup completed")

    def load_frame(self, frame_id):
        """Load a specific frame from the recorded data."""
        if self.model_recorder and 0 <= frame_id < len(self.model_recorder.history):
            self.model_recorder.playback(self.state, frame_id)            
            print(f"Loaded frame {frame_id}")
            return True
        return False


    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(0.0)
            # If we have a model and state, render the state (for ModelAndStateRecorder)
            if self.model is not None and self.state is not None:
                self.renderer.render(self.state)
                print("RENDERED")
            else:
                if(self.model is None):
                    print("MODEL IS NONE")
                if(self.state is None):
                    print("STATE IS NONE")
            # Otherwise, let the replay manager handle display
            self.renderer.end_frame()


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
        help="Recording file to load on startup (.json for ModelAndStateRecorder)",
    )

    args = parser.parse_args()

    print("Newton Physics Replay Viewer")
    print("Use the GUI to load recordings and explore your data.")
    print("Note: Only JSON files (.json) from ModelAndStateRecorder are supported.")
    if args.file:
        print(f"Loading: {args.file}")

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.window_title)

        # RAW TESTING: Load the specific JSON file
        raw_json_path = r"C:\tmp\test.json"
        success = example.load_raw_simulation(raw_json_path)

        if not success:
            print("Failed to load simulation. Exiting...")
            return

        # Load file if specified via command line (this will override the raw testing)
        if args.file:
            print(f"Command line file loading not implemented yet: {args.file}")
            # example.load_recording(args.file)  # TODO: Implement this method if needed

        # Main loop following example_quadruped pattern
        if example.renderer:
            while example.renderer.is_running():
                example.render()
                print("renderloop")


if __name__ == "__main__":
    main()
