from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-8:
        return v.copy()
    return v / n


@dataclass
class FreeCamera:
    """Simple free camera inspired by MiniRenderer.FreeCamera."""

    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float = 60.0

    @classmethod
    def create_default(cls) -> FreeCamera:
        return cls(
            position=np.array([0.0, 0.0, -5.0], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            fov_y_degrees=60.0,
        )

    def set_pose(self, position, target, up) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.up = np.asarray(up, dtype=np.float32)

    def get_forward(self) -> np.ndarray:
        return _normalize(self.target.astype(np.float32) - self.position.astype(np.float32))

    def get_right(self) -> np.ndarray:
        return _normalize(np.cross(self.get_forward(), _normalize(self.up.astype(np.float32))))

    def get_up_orthonormal(self) -> np.ndarray:
        return _normalize(np.cross(self.get_right(), self.get_forward()))

    def move_local(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> None:
        f = self.get_forward()
        r = self.get_right()
        u = self.get_up_orthonormal()
        delta = np.float32(forward) * f + np.float32(right) * r + np.float32(up) * u
        self.position = (self.position + delta).astype(np.float32)
        self.target = (self.target + delta).astype(np.float32)

    def orbit_target(self, yaw_radians: float = 0.0, pitch_radians: float = 0.0) -> None:
        offset = self.position.astype(np.float32) - self.target.astype(np.float32)
        radius = float(np.linalg.norm(offset))
        if radius < 1.0e-6:
            return
        x, y, z = [float(v) for v in offset]
        yaw = float(np.arctan2(x, z))
        pitch = float(np.arctan2(y, max(np.sqrt(x * x + z * z), 1.0e-8)))
        yaw += float(yaw_radians)
        pitch += float(pitch_radians)
        pitch = float(np.clip(pitch, -1.4835299, 1.4835299))  # +/-85 degrees
        cp = float(np.cos(pitch))
        sp = float(np.sin(pitch))
        cy = float(np.sin(yaw))
        cz = float(np.cos(yaw))
        new_offset = np.array([radius * cp * cy, radius * sp, radius * cp * cz], dtype=np.float32)
        self.position = (self.target.astype(np.float32) + new_offset).astype(np.float32)

    def dolly(self, amount: float) -> None:
        dir_to_target = self.target.astype(np.float32) - self.position.astype(np.float32)
        dist = float(np.linalg.norm(dir_to_target))
        if dist < 1.0e-6:
            return
        f = dir_to_target / np.float32(dist)
        step = np.float32(amount)
        # Keep a minimum distance to avoid crossing through target.
        if amount > 0.0 and dist - float(amount) < 0.2:
            step = np.float32(max(dist - 0.2, 0.0))
        self.position = (self.position.astype(np.float32) + step * f).astype(np.float32)

    def get_basis(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return camera basis vectors (pos, u, v, w) for raygen.

        Direction is `normalize(w + fx*u + fy*v)`.
        """

        pos = self.position.astype(np.float32)
        forward = _normalize(self.target.astype(np.float32) - pos)
        right = _normalize(np.cross(forward, _normalize(self.up.astype(np.float32))))
        up_vec = _normalize(np.cross(right, forward))

        aspect = float(width) / max(float(height), 1.0)
        tan_half_fov = np.tan(np.deg2rad(float(self.fov_y_degrees)) * 0.5)
        u = right * np.float32(aspect * tan_half_fov)
        v = up_vec * np.float32(tan_half_fov)
        w = forward
        return pos, u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)
