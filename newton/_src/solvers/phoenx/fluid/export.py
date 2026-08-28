# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .resample import resample_state_3d
from .volume import VolumeBrickGrid


def export_state_3d(solver, *, threshold: float = 0.0) -> VolumeBrickGrid:
    """Convert a KPM-FR state to the sparse rendering interchange."""
    config = solver.config
    order = config.order
    state = resample_state_3d(solver.state.numpy(), config.resolution, solver.points)
    rho = state[..., 0]
    velocity = state[..., 1:4] / rho[..., None]
    farfield = np.array([config.reference_velocity, 0.0, 0.0])
    velocity -= farfield
    density = np.abs(rho - 1.0)
    speed = np.linalg.norm(velocity, axis=-1)
    values = np.concatenate((density[..., None], velocity, speed[..., None]), axis=-1).astype(np.float16)
    voxel_size = tuple(config.size[axis] / (config.resolution[axis] * order) for axis in range(3))
    return VolumeBrickGrid.from_dense(
        values,
        voxel_size=voxel_size,
        channels=("density", "velocity_x", "velocity_y", "velocity_z", "speed"),
        threshold=threshold,
    )
