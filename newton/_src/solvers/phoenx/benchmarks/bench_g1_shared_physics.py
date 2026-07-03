# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark PhoenX on the bundled MuJoCo Warp G1 collision scene.

The benchmark removes visual-only mesh declarations without changing bodies,
inertias, joints, actuators, or collision geometry. This makes the checked-in
MJCF self-contained and provides a shared-model physics throughput comparison
with mujoco_warp/testspeed.py.

Example:
    uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.bench_g1_shared_physics
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton


def _collision_only_mjcf(asset_dir: Path, output_dir: Path) -> Path:
    model = (asset_dir / "unitree_g1_mjlab.xml").read_text()
    model = re.sub(r"^\s*<mesh\b[^>]*/>\s*$", "", model, flags=re.MULTILINE)
    model = re.sub(
        r'^\s*<geom\b[^>]*class="visual"[^>]*/>\s*$',
        "",
        model,
        flags=re.MULTILINE,
    )
    (output_dir / "unitree_g1_mjlab.xml").write_text(model)
    scene_path = output_dir / "scene_flat.xml"
    scene_path.write_text((asset_dir / "scene_flat.xml").read_text())
    return scene_path


def _load_cudart() -> ctypes.CDLL:
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise RuntimeError("CUDA runtime library not found")


def _mujoco_free_qpos_to_newton(qpos: np.ndarray) -> np.ndarray:
    converted = np.asarray(qpos, dtype=np.float32).copy()
    converted[3:7] = converted[[4, 5, 6, 3]]
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--measure-replays", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument(
        "--contact-friction-model",
        choices=("point", "patch"),
        default="point",
    )
    parser.add_argument(
        "--articulation-mode",
        choices=("maximal", "hybrid", "reduced"),
        default="reduced",
    )
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket only measured graph replays with the CUDA profiler API.",
    )
    parser.add_argument(
        "--multi-world-scheduler",
        choices=("auto", "fast_tail", "block_world"),
        default="auto",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.world_count <= 0:
        raise ValueError("world_count must be positive")
    if args.warmup_replays < 0 or args.measure_replays <= 0:
        raise ValueError("replay counts must be non-negative, with at least one measured replay")

    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("shared G1 physics benchmark requires CUDA with Warp mempool enabled")

    asset_dir = Path(newton.__file__).parent / "examples" / "assets" / "mjwarp_benchmarks" / "unitree_g1"
    replay = np.load(asset_dir / "shuffle_dance.npz")
    qpos = _mujoco_free_qpos_to_newton(replay["qpos"][0])
    ctrl = np.asarray(replay["ctrl"][0], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="newton_g1_collision_") as temp_dir:
        scene_path = _collision_only_mjcf(asset_dir, Path(temp_dir))
        template = newton.ModelBuilder(up_axis=newton.Axis.Z)
        template.add_mjcf(
            str(scene_path),
            ignore_names=("floor",),
            parse_visuals=False,
            parse_meshes=False,
            enable_self_collisions=False,
        )

    for index, value in enumerate(qpos):
        template.joint_q[index] = float(value)
        template.joint_target_q[index] = float(value)
    for channel, value in enumerate(ctrl):
        template.joint_target_q[7 + channel] = float(value)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, args.world_count)
    builder.default_shape_cfg.mu = 0.6
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverPhoenX(
        model,
        articulation_mode=args.articulation_mode,
        contact_friction_model=args.contact_friction_model,
        substeps=1,
        solver_iterations=args.solver_iterations,
        velocity_iterations=args.velocity_iterations,
        multi_world_scheduler=args.multi_world_scheduler,
    )
    state = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = model.contacts()

    def step() -> None:
        model.collide(state, contacts)
        state.clear_forces()
        solver.step(state, state, control, contacts, args.dt)

    for _ in range(4):
        step()
    with wp.ScopedCapture(device=device) as capture:
        step()
    for _ in range(args.warmup_replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    cudart = _load_cudart() if args.cuda_profiler_api else None
    if cudart is not None and cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    start = time.perf_counter()
    for _ in range(args.measure_replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start
    if cudart is not None and cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")
    physics_steps = args.world_count * args.measure_replays
    gpu_used_gb = float(device.total_memory - device.free_memory) / float(1024**3)

    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
        raise RuntimeError("shared G1 benchmark produced non-finite body state")

    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    reduced_contact_points_mean = None
    reduced_contact_points_max = None
    if solver._reduced_articulation is not None:
        point_counts = solver._reduced_articulation.contact_block_system.total_point_count.numpy()
        reduced_contact_points_mean = float(np.mean(point_counts))
        reduced_contact_points_max = int(np.max(point_counts))

    print(
        json.dumps(
            {
                "articulation_mode": args.articulation_mode,
                "contact_friction_model": args.contact_friction_model,
                "active_contact_count": contact_count,
                "articulation_contact_points_max": reduced_contact_points_max,
                "articulation_contact_points_mean": reduced_contact_points_mean,
                "body_count_per_world": int(model.body_count) // args.world_count,
                "contact_capacity": int(contacts.rigid_contact_max),
                "dt": args.dt,
                "elapsed_s": elapsed,
                "engine": "phoenx",
                "gpu_used_gb": gpu_used_gb,
                "measure_replays": args.measure_replays,
                "multi_world_scheduler": args.multi_world_scheduler,
                "physics_steps": physics_steps,
                "physics_steps_per_s": physics_steps / elapsed,
                "shape_count_per_world": (int(model.shape_count) - 1) // args.world_count,
                "solver_iterations": args.solver_iterations,
                "velocity_iterations": args.velocity_iterations,
                "world_count": args.world_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
