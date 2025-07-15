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

import numpy as np
import warp as wp

from newton.utils.render import SimRendererOpenGL


class BodyTransformRecorder:
    """A class to record and playback simulation body transforms."""

    def __init__(self, renderer: SimRendererOpenGL):
        """
        Initializes the Recorder.

        Args:
            renderer: The simulation renderer instance.
        """
        self.renderer = renderer
        self.transforms_history: list[wp.array] = []

    def record(self, body_transforms: wp.array):
        """
        Records a snapshot of body transforms.

        Args:
            body_transforms (wp.array): A warp array representing the body transforms.
                This is typically retrieved from `state.body_q`.
        """
        self.transforms_history.append(wp.clone(body_transforms))

    def playback(self, frame_id: int):
        """
        Plays back a recorded frame by updating the renderer with the stored body transforms.

        Args:
            frame_id (int): The integer index of the frame to be played back.
        """
        if not (0 <= frame_id < len(self.transforms_history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return

        body_transforms = self.transforms_history[frame_id]
        self.renderer.update_body_transforms(body_transforms)

    def save_to_file(self, file_path: str):
        """
        Saves the recorded transforms history to a file.

        Args:
            file_path (str): The full path to the file where the transforms will be saved.
        """
        history_np = {f"frame_{i}": t.numpy() for i, t in enumerate(self.transforms_history)}
        np.savez_compressed(file_path, **history_np)

    def load_from_file(self, file_path: str, device=None):
        """
        Loads recorded transforms from a file, replacing the current history.

        Args:
            file_path (str): The full path to the file from which to load the transforms.
            device: The device to load the transforms onto. If None, uses CPU.
        """
        self.transforms_history.clear()
        with np.load(file_path) as data:
            frame_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
            for key in frame_keys:
                transform_np = data[key]
                transform_wp = wp.array(transform_np, dtype=wp.transform, device=device)
                self.transforms_history.append(transform_wp)