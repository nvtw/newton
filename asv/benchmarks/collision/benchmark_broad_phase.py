# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ASV benchmarks for broad-phase collision detection.

The nightly classes cover each broad-phase backend where the input size is
appropriate:

* ``nxn``: all-pairs AABB checks, representative for small scenes.
* ``sap_segmented``: sweep-and-prune with segmented sort, representative for
  larger dynamic scenes.
* ``sap_tile``: sweep-and-prune with tile sort, kept at tile-sized inputs.
* ``explicit``: AABB checks over a precomputed pair set.

The stress benchmark is derived from the broad-phase edge-case tests: it mixes
touching AABBs, gaps, global/world-specific shapes, group filters, overlapping
clusters, and isolated rejects.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from newton.geometry import BroadPhaseAllPairs, BroadPhaseExplicit, BroadPhaseSAP

BROAD_PHASE_TYPES = ("nxn", "sap_segmented", "sap_tile", "explicit")


@dataclass(frozen=True)
class _BroadPhaseScene:
    shape_aabb_lower: wp.array[wp.vec3]
    shape_aabb_upper: wp.array[wp.vec3]
    shape_gap: wp.array[float]
    shape_collision_group: wp.array[int]
    shape_world: wp.array[int]
    explicit_shape_pairs: wp.array[wp.vec2i]

    @property
    def shape_count(self) -> int:
        return self.shape_aabb_lower.shape[0]

    @property
    def explicit_pair_count(self) -> int:
        return self.explicit_shape_pairs.shape[0]


def _groups_collide(group_a: int, group_b: int) -> bool:
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    return group_a != group_b


def _worlds_and_groups_collide(world_a: int, world_b: int, group_a: int, group_b: int) -> bool:
    if world_a != -1 and world_b != -1 and world_a != world_b:
        return False
    return _groups_collide(group_a, group_b)


def _prefilter_shape_pairs(shape_world: np.ndarray, shape_collision_group: np.ndarray) -> np.ndarray:
    shape_count = shape_world.shape[0]
    pairs = []
    for i in range(shape_count):
        for j in range(i + 1, shape_count):
            if _worlds_and_groups_collide(
                int(shape_world[i]),
                int(shape_world[j]),
                int(shape_collision_group[i]),
                int(shape_collision_group[j]),
            ):
                pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int32)


def _build_sphere_grid_scene(grid_size: int, spacing_factor: float = 0.95) -> _BroadPhaseScene:
    """Build a deterministic sphere grid for broad-phase scaling."""
    sphere_radius = 0.5
    spacing = 2.0 * sphere_radius * spacing_factor
    shape_count = grid_size**3
    centers = np.empty((shape_count, 3), dtype=np.float32)
    cursor = 0

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                centers[cursor] = (i * spacing, j * spacing, k * spacing + sphere_radius)
                cursor += 1

    device = wp.get_device()
    lower = centers - sphere_radius
    upper = centers + sphere_radius
    shape_world = np.zeros(shape_count, dtype=np.int32)
    shape_collision_group = np.ones(shape_count, dtype=np.int32)

    return _BroadPhaseScene(
        shape_aabb_lower=wp.array(lower, dtype=wp.vec3, device=device),
        shape_aabb_upper=wp.array(upper, dtype=wp.vec3, device=device),
        shape_gap=wp.empty(0, dtype=wp.float32, device=device),
        shape_collision_group=wp.array(shape_collision_group, dtype=wp.int32, device=device),
        shape_world=wp.array(shape_world, dtype=wp.int32, device=device),
        explicit_shape_pairs=wp.array(
            _prefilter_shape_pairs(shape_world, shape_collision_group), dtype=wp.vec2i, device=device
        ),
    )


def _append_box(
    lowers: list[np.ndarray],
    uppers: list[np.ndarray],
    gaps: list[float],
    groups: list[int],
    worlds: list[int],
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    *,
    gap: float = 0.0,
    group: int = 1,
    world: int = 0,
) -> None:
    lowers.append(np.array(lower, dtype=np.float32))
    uppers.append(np.array(upper, dtype=np.float32))
    gaps.append(gap)
    groups.append(group)
    worlds.append(world)


def _build_edge_cluster_scene() -> _BroadPhaseScene:
    """Build a fixed edge-case scene inspired by ``test_broad_phase``."""
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    gaps: list[float] = []
    groups: list[int] = []
    worlds: list[int] = []

    _append_box(lowers, uppers, gaps, groups, worlds, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    _append_box(lowers, uppers, gaps, groups, worlds, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    _append_box(lowers, uppers, gaps, groups, worlds, (2.15, 0.0, 0.0), (3.0, 1.0, 1.0), group=2)
    _append_box(lowers, uppers, gaps, groups, worlds, (3.25, 0.0, 0.0), (4.0, 1.0, 1.0), gap=0.15, group=2)

    for x in (5.0, 6.0, 7.0, 8.0):
        _append_box(lowers, uppers, gaps, groups, worlds, (x, 0.0, 0.0), (x + 1.5, 1.0, 1.0))

    for inset in (0.0, 0.1, 0.2, 0.3):
        _append_box(
            lowers,
            uppers,
            gaps,
            groups,
            worlds,
            (10.0 + inset, inset, inset),
            (11.0 - inset, 1.0 - inset, 1.0 - inset),
        )

    _append_box(lowers, uppers, gaps, groups, worlds, (15.0, 0.0, 0.0), (16.0, 1.0, 1.0), group=-1, world=-1)
    _append_box(lowers, uppers, gaps, groups, worlds, (15.2, 0.2, 0.2), (15.8, 0.8, 0.8), group=-2, world=-1)
    _append_box(lowers, uppers, gaps, groups, worlds, (18.0, 0.0, 0.0), (19.0, 1.0, 1.0), group=-1, world=-1)
    _append_box(lowers, uppers, gaps, groups, worlds, (18.2, 0.2, 0.2), (18.8, 0.8, 0.8), group=1, world=1)
    _append_box(lowers, uppers, gaps, groups, worlds, (18.4, 0.4, 0.4), (18.6, 0.6, 0.6), group=2, world=2)

    for group in (1, 2, -1, -2):
        offset = 0.05 * len(groups)
        _append_box(
            lowers,
            uppers,
            gaps,
            groups,
            worlds,
            (22.0 + offset, offset, offset),
            (23.0 - offset, 1.0 - offset, 1.0 - offset),
            group=group,
        )

    _append_box(lowers, uppers, gaps, groups, worlds, (26.0, 0.0, 0.0), (27.0, 1.0, 1.0), world=0)
    _append_box(lowers, uppers, gaps, groups, worlds, (26.2, 0.2, 0.2), (26.8, 0.8, 0.8), world=1)
    _append_box(lowers, uppers, gaps, groups, worlds, (30.0, 0.0, 0.0), (31.0, 1.0, 1.0))
    _append_box(lowers, uppers, gaps, groups, worlds, (29.0, 0.0, 0.0), (30.5, 1.0, 1.0))
    _append_box(lowers, uppers, gaps, groups, worlds, (33.0, 0.0, 0.0), (34.0, 1.0, 1.0), group=0)
    _append_box(lowers, uppers, gaps, groups, worlds, (33.2, 0.2, 0.2), (33.8, 0.8, 0.8), group=0)

    rng = np.random.default_rng(888)
    for cluster_id in range(8):
        center = np.array([50.0 + cluster_id * 6.0, rng.random() * 8.0, rng.random() * 8.0], dtype=np.float32)
        cluster_world = -1 if cluster_id < 3 else cluster_id % 4
        cluster_group = (cluster_id % 4) + 1
        for i in range(8):
            offset = rng.random(3, dtype=np.float32) * 0.6
            lower = center - 0.6 + offset
            upper = center + 0.6 + offset
            _append_box(
                lowers,
                uppers,
                gaps,
                groups,
                worlds,
                tuple(lower),
                tuple(upper),
                gap=0.1 if i % 3 == 0 else 0.0,
                group=-1 if i % 5 == 0 else cluster_group,
                world=cluster_world,
            )

    for i in range(16):
        center = np.array([140.0 + i * 6.0, rng.random() * 5.0, rng.random() * 5.0], dtype=np.float32)
        _append_box(
            lowers,
            uppers,
            gaps,
            groups,
            worlds,
            tuple(center - 0.25),
            tuple(center + 0.25),
            group=(i % 6) + 1,
            world=i % 4,
        )

    device = wp.get_device()
    lower_np = np.asarray(lowers, dtype=np.float32)
    upper_np = np.asarray(uppers, dtype=np.float32)

    shape_world = np.asarray(worlds, dtype=np.int32)
    shape_collision_group = np.asarray(groups, dtype=np.int32)

    return _BroadPhaseScene(
        shape_aabb_lower=wp.array(lower_np, dtype=wp.vec3, device=device),
        shape_aabb_upper=wp.array(upper_np, dtype=wp.vec3, device=device),
        shape_gap=wp.array(np.asarray(gaps, dtype=np.float32), dtype=wp.float32, device=device),
        shape_collision_group=wp.array(shape_collision_group, dtype=wp.int32, device=device),
        shape_world=wp.array(shape_world, dtype=wp.int32, device=device),
        explicit_shape_pairs=wp.array(
            _prefilter_shape_pairs(shape_world, shape_collision_group), dtype=wp.vec2i, device=device
        ),
    )


class _BroadPhaseBenchmark:
    """Shared broad-phase benchmark harness."""

    repeat = 5
    number = 1
    rounds = 2
    timeout = 300
    warmup_iterations = 5
    timed_iterations = 20
    scene_kind = "grid"
    grid_size = 8
    spacing_factor = 0.95

    def setup(self, broad_phase_type: str) -> None:
        self.broad_phase_type = broad_phase_type
        self._skip = False

        if self.scene_kind == "grid":
            self.scene = _build_sphere_grid_scene(self.grid_size, self.spacing_factor)
        elif self.scene_kind == "edge_cluster":
            self.scene = _build_edge_cluster_scene()
        else:
            raise ValueError(f"Unknown broad-phase scene kind: {self.scene_kind!r}")

        self.device = self.scene.shape_aabb_lower.device
        self.shape_count = self.scene.shape_count
        self.max_candidate_pairs = (
            self.scene.explicit_pair_count
            if broad_phase_type == "explicit"
            else (self.shape_count * (self.shape_count - 1)) // 2
        )
        self.candidate_pairs = wp.zeros(self.max_candidate_pairs, dtype=wp.vec2i, device=self.device)
        self.num_candidate_pairs = wp.zeros(1, dtype=wp.int32, device=self.device)

        if broad_phase_type == "nxn":
            self.broad_phase = BroadPhaseAllPairs(
                self.scene.shape_world,
                device=self.device,
            )
            self._launch_func = self._launch_nxn
        elif broad_phase_type == "sap_segmented":
            self.broad_phase = BroadPhaseSAP(
                self.scene.shape_world,
                sort_type="segmented",
                device=self.device,
            )
            self._launch_func = self._launch_sap
        elif broad_phase_type == "sap_tile":
            if self.shape_count > 512:
                self._skip = True
                return
            self.broad_phase = BroadPhaseSAP(
                self.scene.shape_world,
                sort_type="tile",
                device=self.device,
            )
            self._launch_func = self._launch_sap
        elif broad_phase_type == "explicit":
            self.broad_phase = BroadPhaseExplicit()
            self._launch_func = self._launch_explicit
        else:
            raise ValueError(f"Unknown broad-phase type: {broad_phase_type!r}")

        self._launch_func()
        wp.synchronize_device()

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._launch_func()
            self.graph = capture.graph

        for _ in range(self.warmup_iterations):
            self._launch()
        wp.synchronize_device()

    def _launch_nxn(self) -> None:
        self.broad_phase.launch(
            self.scene.shape_aabb_lower,
            self.scene.shape_aabb_upper,
            self.scene.shape_gap,
            self.scene.shape_collision_group,
            self.scene.shape_world,
            self.shape_count,
            self.candidate_pairs,
            self.num_candidate_pairs,
            device=self.device,
        )

    def _launch_sap(self) -> None:
        self.broad_phase.launch(
            self.scene.shape_aabb_lower,
            self.scene.shape_aabb_upper,
            self.scene.shape_gap,
            self.scene.shape_collision_group,
            self.scene.shape_world,
            self.shape_count,
            self.candidate_pairs,
            self.num_candidate_pairs,
            device=self.device,
        )

    def _launch_explicit(self) -> None:
        self.broad_phase.launch(
            self.scene.shape_aabb_lower,
            self.scene.shape_aabb_upper,
            self.scene.shape_gap,
            self.scene.explicit_shape_pairs,
            self.scene.explicit_pair_count,
            self.candidate_pairs,
            self.num_candidate_pairs,
            device=self.device,
        )

    def _launch(self) -> None:
        if self.graph is None:
            self._launch_func()
        else:
            wp.capture_launch(self.graph)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_broad_phase(self, broad_phase_type: str) -> None:
        if self._skip:
            return
        for _ in range(self.timed_iterations):
            self._launch()
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_median_launch_ms(self, broad_phase_type: str) -> float:
        if self._skip:
            return float("nan")

        samples = []
        for _ in range(self.timed_iterations):
            with wp.ScopedTimer("broad_phase", synchronize=True, print=False) as timer:
                self._launch()
            samples.append(timer.elapsed)
        return statistics.median(samples)

    track_median_launch_ms.unit = "ms"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_candidate_pairs(self, broad_phase_type: str) -> int | float:
        if self._skip:
            return float("nan")
        self._launch_func()
        wp.synchronize_device()
        return int(self.num_candidate_pairs.numpy()[0])

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_shape_count(self, broad_phase_type: str) -> int:
        return self.shape_count

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_pair_capacity(self, broad_phase_type: str) -> int:
        return self.max_candidate_pairs


class FastBroadPhaseGrid(_BroadPhaseBenchmark):
    """Small broad-phase smoke benchmark for pull-request ASV runs."""

    repeat = 3
    warmup_iterations = 3
    timed_iterations = 5
    grid_size = 5
    params = (BROAD_PHASE_TYPES,)
    param_names = ["broad_phase_type"]


class BroadPhaseGrid(_BroadPhaseBenchmark):
    """Nightly broad-phase benchmark on a regular sphere grid."""

    grid_size = 8
    params = (BROAD_PHASE_TYPES,)
    param_names = ["broad_phase_type"]


class BroadPhaseStress(_BroadPhaseBenchmark):
    """Nightly broad-phase benchmark on edge cases and overlap clusters."""

    scene_kind = "edge_cluster"
    params = (BROAD_PHASE_TYPES,)
    param_names = ["broad_phase_type"]


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastBroadPhaseGrid": FastBroadPhaseGrid,
        "BroadPhaseGrid": BroadPhaseGrid,
        "BroadPhaseStress": BroadPhaseStress,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple.",
    )
    args = parser.parse_known_args()[0]

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    for key in benchmarks:
        run_benchmark(benchmark_list[key])
