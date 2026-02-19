from __future__ import annotations

import numpy as np


def build_launch_params_dtype() -> np.dtype:
    """Build launch-params dtype with explicit CUDA-compatible offsets.

    Layout (bytes):
    - image (u64)      @ 0
    - width (u32)      @ 8
    - height (u32)     @ 12
    - time_sec (f32)   @ 16
    - trav_handle(u64) @ 24
    """

    return np.dtype(
        {
            "names": ["image", "width", "height", "time_sec", "trav_handle"],
            "formats": ["u8", "u4", "u4", "f4", "u8"],
            "offsets": [0, 8, 12, 16, 24],
            "itemsize": 32,
        }
    )


def pack_launch_params_bytes(
    image_ptr: int, width: int, height: int, time_sec: float, trav_handle: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create one-element launch-params struct and raw bytes view."""

    dtype = build_launch_params_dtype()
    params_host = np.zeros(1, dtype=dtype)
    params_host["image"][0] = np.uint64(image_ptr)
    params_host["width"][0] = np.uint32(width)
    params_host["height"][0] = np.uint32(height)
    params_host["time_sec"][0] = np.float32(time_sec)
    params_host["trav_handle"][0] = np.uint64(trav_handle)
    return params_host, params_host.view(np.uint8).reshape(-1)
