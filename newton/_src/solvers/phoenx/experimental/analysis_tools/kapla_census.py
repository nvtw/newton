# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Structural census of the settled Kapla tower.

Prices solver architecture changes before any kernel is written.

``subdomain`` -- colored GS touches every body once per color, so
body-state reuse exists only across colors and iterations. Capturing it
needs one block to own every constraint incident to a set of bodies,
i.e. spatial partitioning. A contact straddling two parts stays on the
global mass-splitting path, so the interior fraction decides whether the
design can pay.

``dormancy`` -- reports how much solver work is skippable at rest, at
both granularities: per body (a contact is dead iff both endpoints are
at rest) and per contact-graph island (what the existing
``sleeping_velocity_threshold`` machinery can actually reach).

See PERF_NOTES.md.
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


def _islands(body1: np.ndarray, body2: np.ndarray, dynamic: np.ndarray) -> np.ndarray:
    """Union-find contact-graph roots. Static bodies are excluded so the
    ground plane does not fuse otherwise-separate islands."""
    parent = np.arange(dynamic.size)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    both_dynamic = dynamic[body1] & dynamic[body2]
    for x, y in zip(body1[both_dynamic], body2[both_dynamic], strict=True):
        rx, ry = find(int(x)), find(int(y))
        if rx != ry:
            parent[rx] = ry
    return np.array([find(i) for i in range(dynamic.size)])


def _report_dormancy(
    body1: np.ndarray,
    body2: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    dynamic: np.ndarray,
) -> None:
    """Skippable-work census at body and island granularity."""
    speed = np.linalg.norm(velocity, axis=1)
    spin = np.linalg.norm(angular_velocity, axis=1)

    print(f"{'v,w threshold':>16} {'bodies at rest':>15} {'contacts dead':>14}")
    for lin, ang in ((0.01, 0.05), (0.02, 0.1)):
        # Static bodies never move, so they are permanently at rest.
        at_rest = ((speed < lin) & (spin < ang)) | ~dynamic
        dead = at_rest[body1] & at_rest[body2]
        print(f"{f'{lin}, {ang}':>16} {at_rest[dynamic].mean() * 100:>14.1f}% {dead.mean() * 100:>13.1f}%")

    roots = _islands(body1, body2, dynamic)[dynamic]
    uniq, counts = np.unique(roots, return_counts=True)
    print(f"\nislands={uniq.size}  largest={counts.max()} ({counts.max() / dynamic.sum() * 100:.1f}% of bodies)")

    # The shipped mechanism sleeps a whole island only when its fastest
    # member is below the threshold.
    score = np.maximum(speed, spin)[dynamic]
    index = {int(r): i for i, r in enumerate(uniq)}
    island_max = np.zeros(uniq.size)
    np.maximum.at(island_max, [index[int(r)] for r in roots], score)
    print(f"{'island threshold':>16} {'islands asleep':>15} {'bodies asleep':>14}")
    for threshold in (0.05, 0.15, 0.3):
        asleep = island_max < threshold
        frac = counts[asleep].sum() / dynamic.sum()
        print(f"{threshold:>16} {asleep.mean() * 100:>14.1f}% {frac * 100:>13.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("subdomain", "dormancy"), default="subdomain")
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

    print(f"bodies={num_bodies} (dynamic={int((motion_type >= 2).sum())})  contacts={body1.size}")

    if args.mode == "dormancy":
        _report_dormancy(
            body1,
            body2,
            ex.bodies.velocity.numpy(),
            ex.bodies.angular_velocity.numpy(),
            motion_type >= 2,
        )
        return 0

    sm_count = ex.device.sm_count
    part_counts = args.parts or [sm_count // 2, sm_count, sm_count * 2, sm_count * 4]

    order = _morton_order(positions)
    rank = np.empty(num_bodies, dtype=np.int64)
    rank[order] = np.arange(num_bodies)

    print(f"SMs={sm_count}")
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
