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

from typing import List

import warp as wp

from newton.utils.render import SimRendererOpenGL


class Recorder:
    """A class to record and playback simulation body transforms."""

    def __init__(self, renderer: SimRendererOpenGL):
        """
        Initializes the Recorder.

        Args:
            renderer: The simulation renderer instance.
        """
        self.renderer = renderer
        self.transforms_history: List[wp.array] = []

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
