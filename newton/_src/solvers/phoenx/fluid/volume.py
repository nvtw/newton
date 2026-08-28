# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _store_volume(volume: wp.uint64, coords: wp.array[wp.vec3i], values: wp.array[wp.float32]):
    tid = wp.tid()
    p = coords[tid]
    wp.volume_store_f(volume, p[0], p[1], p[2], values[tid])


@dataclass
class VolumeBrickGrid:
    """Sparse dense-brick volume interchange."""

    brick_coords: np.ndarray
    values: np.ndarray
    resolution: tuple[int, int, int]
    voxel_size: tuple[float, float, float]
    origin: tuple[float, float, float]
    channels: tuple[str, ...]
    brick_size: int = 8

    @classmethod
    def from_dense(
        cls,
        values: np.ndarray,
        *,
        voxel_size: tuple[float, float, float],
        origin=(0.0, 0.0, 0.0),
        channels=("density",),
        threshold=0.0,
        brick_size=8,
    ):
        """Pack active dense regions into fixed-size bricks."""
        dense = np.asarray(values)
        if dense.ndim == 3:
            dense = dense[..., None]
        if dense.ndim != 4 or dense.shape[3] != len(channels):
            raise ValueError("values must have shape (x, y, z, channels)")
        resolution = dense.shape[:3]
        padded = tuple((n + brick_size - 1) // brick_size * brick_size for n in resolution)
        field = np.zeros((*padded, dense.shape[3]), dtype=dense.dtype)
        field[: resolution[0], : resolution[1], : resolution[2]] = dense
        grid = field.reshape(
            padded[0] // brick_size,
            brick_size,
            padded[1] // brick_size,
            brick_size,
            padded[2] // brick_size,
            brick_size,
            dense.shape[3],
        ).transpose(0, 2, 4, 1, 3, 5, 6)
        active = np.max(np.abs(grid), axis=(3, 4, 5, 6)) > threshold
        return cls(
            np.argwhere(active).astype(np.int32),
            np.ascontiguousarray(grid[active]),
            resolution,
            voxel_size,
            origin,
            channels,
            brick_size,
        )

    def dense(self) -> np.ndarray:
        """Expand the active bricks into a dense array."""
        b = self.brick_size
        padded = tuple((n + b - 1) // b * b for n in self.resolution)
        dense = np.zeros((*padded, len(self.channels)), dtype=self.values.dtype)
        for coord, brick in zip(self.brick_coords, self.values, strict=True):
            lo = coord * b
            dense[lo[0] : lo[0] + b, lo[1] : lo[1] + b, lo[2] : lo[2] + b] = brick
        return dense[: self.resolution[0], : self.resolution[1], : self.resolution[2]]

    def save(self, path: str | Path):
        """Write the sparse brick stream."""
        metadata = json.dumps(
            {
                "resolution": self.resolution,
                "voxel_size": self.voxel_size,
                "origin": self.origin,
                "channels": self.channels,
                "brick_size": self.brick_size,
            },
            separators=(",", ":"),
        )
        with Path(path).open("wb") as stream:
            np.savez(stream, metadata=metadata, brick_coords=self.brick_coords, values=self.values)

    @classmethod
    def load(cls, path: str | Path):
        """Read a sparse brick stream."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            return cls(
                archive["brick_coords"],
                archive["values"],
                tuple(metadata["resolution"]),
                tuple(metadata["voxel_size"]),
                tuple(metadata["origin"]),
                tuple(metadata["channels"]),
                metadata["brick_size"],
            )

    def save_nanovdb(self, path: str | Path, channel="density", device: wp.DeviceLike = None):
        """Convert one channel to NanoVDB."""
        dense = self.dense()[..., self.channels.index(channel)]
        coords = np.argwhere(dense != 0.0).astype(np.int32)
        values = dense[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
        device = wp.get_device(device)
        coords_wp = wp.array(coords, dtype=wp.vec3i, device=device)
        values_wp = wp.array(values, device=device)
        tile_coords = np.unique(coords // 8, axis=0).astype(np.int32) * 8
        if len(tile_coords) == 0:
            tile_coords = np.zeros((1, 3), dtype=np.int32)
        volume = wp.Volume.allocate_by_tiles(
            wp.array(tile_coords, dtype=wp.vec3i, device=device),
            voxel_size=self.voxel_size,
            bg_value=0.0,
            translation=self.origin,
            device=device,
        )
        wp.launch(_store_volume, dim=len(coords), inputs=[volume.id, coords_wp, values_wp], device=device)
        volume.save_to_nvdb(str(path))
        return volume
