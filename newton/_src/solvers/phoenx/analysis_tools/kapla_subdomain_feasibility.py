# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Surface-to-volume gate for resident-subdomain block Gauss-Seidel.

Colored GS touches every body once per color, so body-state reuse exists
only across colors and iterations. Capturing it needs one block to own
every constraint incident to a set of bodies, i.e. spatial partitioning.
A contact straddling two parts stays on the global mass-splitting path,
so the interior fraction decides whether the design can pay.

Reports interior contact fraction, bodies per part, and shared-memory
footprint at 52 B/body for each part count ``P``. See PERF_NOTES.md.
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

# 52 B/body: vec3 velocity + vec3 angular_velocity + float inv_mass +
# 6-float symmetric inverse world inertia.
_BYTES_PER_BODY = 52


def _morton3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Interleave three 21-bit integer coordinate arrays into 63-bit keys."""

    def spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        return (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)

    return spread(x) | (spread(y) << np.uint64(1)) | (spread(z) << np.uint64(2))


def _morton_order(positions: np.ndarray) -> np.ndarray:
    """Return body indices sorted by Morton code of their position."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    grid = ((positions - lo) / span * ((1 << 21) - 1)).astype(np.uint64)
    keys = _morton3(grid[:, 0], grid[:, 1], grid[:, 2])
    return np.argsort(keys, kind="stable")


def _partition_stats(
    body1: np.ndarray,
    body2: np.ndarray,
    part_of_body: np.ndarray,
    num_parts: int,
) -> dict[str, float]:
    """Interior/boundary split and per-part load for one partitioning."""
    p1 = part_of_body[body1]
    p2 = part_of_body[body2]
    interior = p1 == p2
    counts = np.bincount(part_of_body, minlength=num_parts)
    return {
        "interior_frac": float(interior.mean()),
        "bodies_max": int(counts.max()),
        "bodies_mean": float(counts.mean()),
        "smem_max_kb": counts.max() * _BYTES_PER_BODY / 1024.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settle", type=int, default=60, help="Warmup frames before sampling.")
    parser.add_argument("--grid", type=int, default=1, help="Tower grid edge length.")
    parser.add_argument(
        "--parts",
        type=int,
        nargs="+",
        default=None,
        help="Part counts to evaluate (default: 0.5x, 1x, 2x, 4x SM count).",
    )
    args = parser.parse_args()

    from newton._src.solvers.phoenx.constraints.constraint_contact import (  # noqa: PLC0415
        _OFF_BODY1,
        _OFF_BODY2,
        _OFF_CONTACT_COUNT,
    )
    from newton._src.solvers.phoenx.examples import example_kapla_tower as ek  # noqa: PLC0415

    ek.TOWER_GRID_DIMS = (args.grid, args.grid)
    ek.ENABLE_MASS_SPLITTING = True

    class _Viewer:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    class _Args:
        pass

    ex = ek.Example(_Viewer(), _Args())
    for _ in range(args.settle):
        ex.step()

    positions = ex.bodies.position.numpy()
    motion_type = ex.bodies.motion_type.numpy()
    num_bodies = positions.shape[0]

    # Contact column headers: body1/body2 are int-reinterpreted floats at
    # fixed dword offsets of the column-major ``data`` array.
    cols = ex.world._contact_cols.data.numpy()
    body1 = cols[int(_OFF_BODY1)].view(np.int32)
    body2 = cols[int(_OFF_BODY2)].view(np.int32)

    # Column storage is allocated to capacity; only columns carrying at
    # least one contact point are live work for the solver.
    point_count = cols[int(_OFF_CONTACT_COUNT)].view(np.int32)
    valid = (point_count > 0) & (body1 >= 0) & (body1 < num_bodies) & (body2 >= 0) & (body2 < num_bodies)
    body1 = body1[valid]
    body2 = body2[valid]

    sm_count = ex.device.sm_count
    part_counts = args.parts or [sm_count // 2, sm_count, sm_count * 2, sm_count * 4]

    order = _morton_order(positions)
    rank = np.empty(num_bodies, dtype=np.int64)
    rank[order] = np.arange(num_bodies)

    print(f"bodies={num_bodies} (dynamic={int((motion_type >= 2).sum())})  contacts={body1.size}  SMs={sm_count}")
    print(f"{'P':>6} {'interior%':>10} {'bodies/part':>12} {'max/part':>9} {'smem KB':>9}")
    for num_parts in part_counts:
        part_of_body = (rank * num_parts // num_bodies).astype(np.int64)
        stats = _partition_stats(body1, body2, part_of_body, num_parts)
        print(
            f"{num_parts:>6} {stats['interior_frac'] * 100:>9.1f}% "
            f"{stats['bodies_mean']:>12.1f} {stats['bodies_max']:>9d} {stats['smem_max_kb']:>9.1f}"
        )

    return 0


if __name__ == "__main__":
    wp.init()
    raise SystemExit(main())
