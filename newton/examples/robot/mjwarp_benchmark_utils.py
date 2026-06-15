# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections.abc import Callable
from pathlib import Path

import warp as wp

import newton.examples

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def benchmark_asset_dir(name: str) -> Path:
    path = Path(newton.examples.get_asset("mjwarp_benchmarks")) / name
    if not path.is_dir():
        raise FileNotFoundError(f"MuJoCo Warp benchmark asset directory not found: {path}")
    return path


def download_menagerie_folder(folder_path: str) -> Path:
    return Path(newton.examples.download_external_git_folder(MENAGERIE_URL, folder_path))


def make_path_resolver(*roots: Path | str) -> Callable[[str | None, str], str]:
    search_roots = tuple(Path(root) for root in roots if root is not None)

    def resolve(base_dir: str | None, file_path: str) -> str:
        path = Path(file_path)
        candidates: list[Path] = []

        if path.is_absolute():
            candidates.append(path)
        if base_dir is not None:
            candidates.append(Path(base_dir) / path)

        for root in search_roots:
            candidates.append(root / path)
            candidates.append(root / path.name)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(candidates[0] if candidates else path)

    return resolve


def parse_prepare_refresh_stride(value: str) -> int | str:
    if value.strip().lower() == "auto":
        return "auto"
    stride = int(value)
    if stride < 1:
        raise argparse.ArgumentTypeError("prepare refresh stride must be >= 1 or 'auto'")
    return stride


@wp.kernel(enable_backward=False)
def scatter_replay_targets(
    sim_time: wp.array[wp.float32],
    replay_speed: wp.float32,
    replay_fps: wp.float32,
    n_frames: wp.int32,
    loop: wp.int32,
    replay_data: wp.array2d[wp.float32],
    replay_indices: wp.array2d[wp.int32],
    joint_target_q: wp.array[wp.float32],
):
    world_idx, channel = wp.tid()
    frame = wp.int32(sim_time[0] * replay_speed * replay_fps)
    if frame >= n_frames:
        if loop != 0:
            frame = frame % n_frames
        else:
            frame = n_frames - 1
    if frame < 0:
        frame = 0

    target_idx = replay_indices[world_idx, channel]
    joint_target_q[target_idx] = replay_data[frame, channel]


@wp.kernel(enable_backward=False)
def advance_sim_time(sim_time: wp.array[wp.float32], dt: wp.float32):
    sim_time[0] = sim_time[0] + dt
