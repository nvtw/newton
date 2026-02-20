# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Shared color conversion helpers for OptiX viewer modules."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(channel: float) -> float:
    """Convert a single sRGB channel value to linear color space."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def srgb_to_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert an array of RGB values from sRGB to linear space."""
    threshold = 0.04045
    return np.where(rgb <= threshold, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
