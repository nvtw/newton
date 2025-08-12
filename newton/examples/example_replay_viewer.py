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
# recorded simulation data from ModelAndStateRecorder (.pkl) files.
#
# Use the GUI to load recordings and scrub through frames.
#
###########################################################################

import argparse

import warp as wp

import newton
import newton.utils
from newton.sim.model import Model
from newton.sim.state import State
from newton.utils.recorder import ModelAndStateRecorder
from newton.utils.replay import ReplayImGuiManager


class Example:
    def __init__(self, stage_path="Newton Replay Viewer"):
        # Model and state will be set when loading recordings
        self.model = None
        self.state = None
        self.solver = None
        self.model_recorder = None
        self.num_envs = 1  # Default number of environments

        # Set up renderer and replay components
        if stage_path:
            # Create SimRendererOpenGL without a model initially
            self.renderer = newton.utils.SimRendererOpenGL(model=None, path=stage_path)
            # GUI will be set up when loading recordings
            self.gui = ReplayImGuiManager(self.renderer, None, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
        else:
            self.renderer = None
            self.gui = None

        # Start in paused mode
        if self.renderer:
            self.renderer.paused = True

        # Frame timing for GUI
        self.frame_dt = 1.0 / 60.0  # 60 FPS

    def load_raw_simulation(self, pickle_path):
        """
        Raw testing method to load a simulation pickle file and set up model/state.

        Args:
            pickle_path (str): Path to the pickle file (e.g., "C:/tmp/my_simulation.pkl")
        """
        print(f"Loading simulation from: {pickle_path}")

        # Create a ModelAndStateRecorder instance
        self.model_recorder = ModelAndStateRecorder()

        # Load the pickle file
        try:
            self.model_recorder.load_from_file(pickle_path)
            print(f"Successfully loaded pickle file with {len(self.model_recorder.history)} frames")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return False
        
        # Extract shape_source from the model data
        if "shape_source" in self.model_recorder.model_data:
            print("Found shape_source in recording")
            shape_source = self.model_recorder.model_data["shape_source"]
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

        print("SHAPE SOURCE: ", self.model.shape_source)

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
            if hasattr(self.model, 'ground') and self.model.ground:
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

    def step(self):
        # For testing, cycle through frames automatically
        if hasattr(self, '_current_frame'):
            self._current_frame = (self._current_frame + 1) % len(self.model_recorder.history)
        else:
            self._current_frame = 0
        
        # Load every 60th frame (1 second at 60 FPS)
        if hasattr(self, '_frame_counter'):
            self._frame_counter += 1
        else:
            self._frame_counter = 0
            
        if self._frame_counter % 60 == 0:  # Change frame every second
            self.load_frame(self._current_frame)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(0.0)
            # If we have a model and state, render the state (for ModelAndStateRecorder)
            if self.model is not None and self.state is not None:
                self.renderer.render(self.state)
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
        help="Recording file to load on startup (.pkl for ModelAndStateRecorder)",
    )

    args = parser.parse_args()

    print("Newton Physics Replay Viewer")
    print("Use the GUI to load recordings and explore your data.")
    print("Note: Only pickle files (.pkl) from ModelAndStateRecorder are supported.")
    if args.file:
        print(f"Loading: {args.file}")

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.window_title)

        # RAW TESTING: Load the specific pickle file
        raw_pickle_path = r"C:\tmp\test.pkl"
        success = example.load_raw_simulation(raw_pickle_path)

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
                example.step()
                example.render()


if __name__ == "__main__":
    main()
