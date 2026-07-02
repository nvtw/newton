# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark packed tile synthesis for reduced G1 contact-response bases.

The common G1 contact page has eight points on two feet: 24 contact rows can
therefore be reconstructed from twelve exact unit-wrench ABA responses.  This
benchmark isolates the reconstruction step before changing the production ABA
path.  Jacobian and response columns are concatenated so one GEMM produces both.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import warp as wp

_ROWS = 24
_BASIS = 12
_DOFS = 48
_PACKED_COLUMNS = 2 * _DOFS
_COLUMN_TILE = 32


@wp.kernel(enable_backward=False)
def _synthesize_scalar_kernel(
    coefficients: wp.array3d[wp.float32],
    basis: wp.array3d[wp.float32],
    output: wp.array3d[wp.float32],
):
    articulation, row, column = wp.tid()
    value = wp.float32(0.0)
    for basis_index in range(_BASIS):
        value += coefficients[articulation, row, basis_index] * basis[articulation, basis_index, column]
    output[articulation, row, column] = value


@wp.kernel(enable_backward=False)
def _synthesize_tile_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    output: wp.array2d[wp.float32],
):
    articulation, column_tile = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(_ROWS, _BASIS),
        offset=(articulation * _ROWS, 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _COLUMN_TILE),
        offset=(articulation * _BASIS, column_tile * _COLUMN_TILE),
        storage="shared",
    )
    result = wp.tile_matmul(coefficients_tile, basis_tile)
    wp.tile_store(
        output,
        result,
        offset=(articulation * _ROWS, column_tile * _COLUMN_TILE),
    )


def _time_graph(device: wp.context.Device, launch: Callable[[], None], *, replays: int) -> float:
    with wp.ScopedCapture(device=device) as capture:
        launch()
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    return 1.0e6 * (time.perf_counter() - start) / float(replays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.world_count <= 0 or args.replays <= 0:
        raise ValueError("world-count and replays must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("response-basis synthesis benchmark requires CUDA")

    rng = np.random.default_rng(42)
    coefficients_np = rng.standard_normal((args.world_count, _ROWS, _BASIS), dtype=np.float32)
    basis_np = rng.standard_normal((args.world_count, _BASIS, _PACKED_COLUMNS), dtype=np.float32)
    coefficients_3d = wp.array(coefficients_np, dtype=wp.float32, device=device)
    basis_3d = wp.array(basis_np, dtype=wp.float32, device=device)
    scalar_output = wp.empty((args.world_count, _ROWS, _PACKED_COLUMNS), dtype=wp.float32, device=device)
    tile_output = wp.empty((args.world_count * _ROWS, _PACKED_COLUMNS), dtype=wp.float32, device=device)
    coefficients_2d = wp.array(coefficients_np.reshape(-1, _BASIS), dtype=wp.float32, device=device)
    basis_2d = wp.array(basis_np.reshape(-1, _PACKED_COLUMNS), dtype=wp.float32, device=device)

    def launch_scalar() -> None:
        wp.launch(
            _synthesize_scalar_kernel,
            dim=(args.world_count, _ROWS, _PACKED_COLUMNS),
            inputs=[coefficients_3d, basis_3d],
            outputs=[scalar_output],
            device=device,
        )

    def launch_tile() -> None:
        wp.launch_tiled(
            _synthesize_tile_kernel,
            dim=[args.world_count, _PACKED_COLUMNS // _COLUMN_TILE],
            block_dim=int(args.block_dim),
            inputs=[coefficients_2d, basis_2d],
            outputs=[tile_output],
            device=device,
        )

    launch_scalar()
    launch_tile()
    wp.synchronize_device(device)
    scalar_np = scalar_output.numpy().reshape(args.world_count * _ROWS, _PACKED_COLUMNS)
    tile_np = tile_output.numpy()
    max_abs_error = float(np.max(np.abs(scalar_np - tile_np)))
    scalar_us = _time_graph(device, launch_scalar, replays=int(args.replays))
    tile_us = _time_graph(device, launch_tile, replays=int(args.replays))
    print(
        json.dumps(
            {
                "block_dim": int(args.block_dim),
                "dimensions": [_ROWS, _BASIS, _PACKED_COLUMNS],
                "max_abs_error": max_abs_error,
                "scalar_us": scalar_us,
                "speedup": scalar_us / tile_us,
                "tile_us": tile_us,
                "world_count": int(args.world_count),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
