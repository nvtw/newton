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
# recorded simulation data from BasicRecorder (.npz) or
# ModelAndStateRecorder (.pkl) files.
#
# Use the GUI to load recordings and scrub through frames.
#
###########################################################################

import argparse

import warp as wp

import newton
import newton.utils
from newton.utils.recorder import BasicRecorder
from newton.utils.recorder_gui import RecorderImGuiManager


class Example:
    def __init__(self, stage_path="Newton Replay Viewer"):
        # No model needed for replay
        self.model = None
        self.solver = None
        
        # Set up renderer and replay components
        if stage_path:
            # Create SimRendererOpenGL without a model
            self.renderer = newton.utils.SimRendererOpenGL(model=None, path=stage_path)
            self.recorder = BasicRecorder()
            self.gui = RecorderImGuiManager(self.renderer, self.recorder, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
        else:
            self.renderer = None
            self.recorder = None
            self.gui = None
        
        # No state needed since we have no model
        self.state = None
        
        # Start in paused mode
        if self.renderer:
            self.renderer.paused = True

        # Frame timing for GUI
        self.frame_dt = 1.0 / 60.0  # 60 FPS

    @property
    def paused(self):
        if self.renderer:
            return self.renderer.paused
        return False

    @paused.setter
    def paused(self, value):
        if self.renderer:
            if self.renderer.paused == value:
                return
            self.renderer.paused = value

    def step(self):
        # No simulation stepping needed for replay viewer
        pass

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(0.0)
            # Don't render the minimal model - just let the replay manager handle display
            self.renderer.end_frame()


def main():
    """Main entry point for the replay viewer example."""
    parser = argparse.ArgumentParser(
        description="Newton Physics Replay Viewer - Visualize recorded simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--window-title",
        type=str,
        default="Newton Replay Viewer",
        help="Window title"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Recording file to load on startup (.npz for BasicRecorder, .pkl for ModelAndStateRecorder)",
    )

    args = parser.parse_args()

    print("Newton Physics Replay Viewer")
    print("Use the GUI to load recordings and explore your data.")
    if args.file:
        print(f"Loading: {args.file}")

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.window_title)

        # Load file if specified
        if args.file and example.recorder:
            try:
                example.recorder.load_from_file(args.file, device=wp.get_device())
                print(f"✓ Loaded: {args.file}")
            except Exception as e:
                print(f"✗ Error loading recording: {e}")

        # Main loop following example_quadruped pattern
        if example.renderer:
            while example.renderer.is_running():
                example.step()
                example.render()


if __name__ == "__main__":
    main()