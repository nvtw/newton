# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import importlib

import numpy as np

import newton.examples
from newton.viewer import ViewerNull

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
ROCK_PILE_WORLD_COUNT = 1_024
ROCK_PILE_VERTEX_COUNTS = (10, 14, 18, 26)
_ROCK_PILE_POSITIONS = (
    (-0.42, -0.30, 0.48),
    (0.12, -0.32, 0.46),
    (0.48, 0.08, 0.50),
    (-0.34, 0.28, 0.52),
    (0.16, 0.30, 0.49),
    (-0.18, -0.12, 1.00),
    (0.32, 0.04, 1.02),
    (-0.30, 0.30, 1.04),
    (0.08, 0.22, 1.48),
    (-0.04, -0.08, 1.90),
)
ROCK_PILE_ROCK_COUNT = len(_ROCK_PILE_POSITIONS)

try:
    from newton.examples import download_external_git_folder as _download_external_git_folder
except ImportError:
    from newton._src.utils.download_assets import download_git_folder as _download_external_git_folder


def _import_example_class(module_names: list[str]):
    """Import and return the ``Example`` class from candidate modules.

    Args:
        module_names: Ordered module names to try importing.

    Returns:
        The first successfully imported module's ``Example`` class.

    Raises:
        SkipNotImplemented: If none of the module names can be imported.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        return module.Example

    raise SkipNotImplemented


def _make_irregular_rock(vertex_count: int, seed: int) -> newton.Mesh:
    """Create a closed irregular convex bipyramid for the rock-pile benchmark."""
    ring_count = vertex_count - 2
    rng = np.random.default_rng(seed)
    vertices = []
    for index in range(ring_count):
        angle = 2.0 * np.pi * index / ring_count
        radius = 0.48 * rng.uniform(0.82, 1.18)
        vertices.append([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(-0.12, 0.12)])

    vertices.extend(
        [
            [0.04, -0.03, rng.uniform(0.48, 0.60)],
            [-0.03, 0.04, -rng.uniform(0.48, 0.60)],
        ]
    )
    top = ring_count
    bottom = ring_count + 1
    indices = []
    for index in range(ring_count):
        next_index = (index + 1) % ring_count
        indices.extend([top, index, next_index])
        indices.extend([bottom, next_index, index])

    return newton.Mesh(np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32))


def _build_rock_pile_scene() -> newton.Model:
    """Build replicated compact piles of varied irregular convex rocks."""
    rocks = [_make_irregular_rock(count, 100 + index) for index, count in enumerate(ROCK_PILE_VERTEX_COUNTS)]

    world_builder = newton.ModelBuilder()
    shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.01, margin=0.0)
    axis = wp.normalize(wp.vec3(0.3, 0.2, 1.0))
    for index, position in enumerate(_ROCK_PILE_POSITIONS):
        body = world_builder.add_body(
            xform=wp.transform(
                wp.vec3(*position),
                wp.quat_from_axis_angle(axis, 0.37 * index),
            )
        )
        world_builder.add_shape_convex_hull(body, mesh=rocks[index % len(rocks)], cfg=shape_cfg)

    builder = newton.ModelBuilder()
    builder.replicate(world_builder, world_count=ROCK_PILE_WORLD_COUNT)
    return builder.finalize()


class FastExampleContactSdfDefaults:
    """Benchmark the SDF nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_sdf",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=100,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactHydroWorkingDefaults:
    """Benchmark the hydroelastic nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_hydro",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=20,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactPyramidDefaults:
    """Benchmark the box pyramid example with default configuration."""

    repeat = 2
    number = 1

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_pyramid",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                solver="xpbd",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastRockPileCollision:
    """Benchmark irregular convex-rock collision across parallel environments."""

    repeat = 3
    number = 1

    def setup(self):
        device = wp.get_device()
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            raise SkipNotImplemented

        self.model = _build_rock_pile_scene()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=self.model.shape_count * 8,
            verify_buffers=False,
        )
        self.contacts = self.collision_pipeline.contacts()

        for _ in range(3):
            self.collision_pipeline.collide(self.state, self.contacts)
        if int(self.collision_pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]) == 0:
            raise RuntimeError("rock-pile benchmark produced no GJK candidate pairs")

        with wp.ScopedCapture(device=device) as capture:
            self.collision_pipeline.collide(self.state, self.contacts)
        self.graph = capture.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self):
        for _ in range(20):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleContactSdfDefaults": FastExampleContactSdfDefaults,
        "FastExampleContactHydroWorkingDefaults": FastExampleContactHydroWorkingDefaults,
        "FastExampleContactPyramidDefaults": FastExampleContactPyramidDefaults,
        "FastRockPileCollision": FastRockPileCollision,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
