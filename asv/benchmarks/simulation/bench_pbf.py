# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-based fluid benchmarks.

Tracks total simulation cost across particle counts, and provides a per-kernel
breakdown (``--profile``) so that changes to the PBF inner loop -- hash grid
rebuild frequency, neighbour list caching, boundary density -- are individually
attributable rather than showing up as a single opaque number.
"""

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.log_level = wp.LOG_WARNING

import newton.examples
from newton.examples.pbf.example_pbf_dam_break import Example as ExamplePBFDamBreak
from newton.viewer import ViewerNull


def _args(dim, substeps=8, iterations=4):
    parser = ExamplePBFDamBreak.create_parser()
    argv = [
        "--dim-x", str(dim[0]),
        "--dim-y", str(dim[1]),
        "--dim-z", str(dim[2]),
        "--substeps", str(substeps),
        "--iterations", str(iterations),
    ]
    args = parser.parse_known_args(argv)[0]
    args.num_frames = getattr(args, "num_frames", None)
    return args


class FastExamplePBFDamBreak:
    """Standard dam break, matching the shipped example's defaults."""

    timeout = 600
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 30
        self.example = ExamplePBFDamBreak(ViewerNull(num_frames=self.num_frames), _args((46, 46, 46)))

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class PBFScaling:
    """Cost vs particle count, to separate per-particle work from fixed overhead."""

    timeout = 900
    repeat = 3
    number = 1
    params = [(20, 20, 20), (32, 32, 32), (46, 46, 46)]
    param_names = ["grid_dim"]

    def setup(self, grid_dim):
        self.num_frames = 20
        self.example = ExamplePBFDamBreak(ViewerNull(num_frames=self.num_frames), _args(grid_dim))

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self, grid_dim):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


def profile(dim=(46, 46, 46), frames=10, substeps=8, iterations=4):
    """Per-kernel GPU time breakdown for one frame of the dam break.

    Wraps ``wp.launch`` and synchronizes around every launch, so absolute times
    are inflated by synchronization overhead; the *relative* attribution across
    kernels is what this is for.
    """
    import collections
    import time

    example = ExamplePBFDamBreak(ViewerNull(num_frames=frames), _args(dim, substeps, iterations))

    totals = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    real_launch = wp.launch
    real_build = type(example.model.particle_grid).build

    def timed_launch(kernel, *a, **kw):
        wp.synchronize_device()
        t0 = time.perf_counter()
        real_launch(kernel, *a, **kw)
        wp.synchronize_device()
        name = getattr(kernel, "key", getattr(kernel, "__name__", str(kernel)))
        totals[name] += time.perf_counter() - t0
        counts[name] += 1

    def timed_build(self, *a, **kw):
        wp.synchronize_device()
        t0 = time.perf_counter()
        real_build(self, *a, **kw)
        wp.synchronize_device()
        totals["<hash grid build>"] += time.perf_counter() - t0
        counts["<hash grid build>"] += 1

    wp.launch = timed_launch
    type(example.model.particle_grid).build = timed_build
    try:
        example.step()  # warm up / compile
        totals.clear()
        counts.clear()
        t_start = time.perf_counter()
        for _ in range(frames):
            example.step()
        wall = time.perf_counter() - t_start
    finally:
        wp.launch = real_launch
        type(example.model.particle_grid).build = real_build

    print(f"\nPBF profile: dim={dim} particles={example.particle_count} "
          f"substeps={substeps} iterations={iterations} frames={frames}")
    print(f"{'kernel':<46}{'ms/frame':>10}{'launches/frame':>16}{'% total':>9}")
    print("-" * 81)
    total = sum(totals.values())
    for name, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{name[:46]:<46}{1000 * t / frames:10.3f}{counts[name] / frames:16.1f}{100 * t / total:9.1f}")
    print("-" * 81)
    print(f"{'sum of measured':<46}{1000 * total / frames:10.3f}{'':>16}{100.0:9.1f}")
    print(f"{'wall clock (incl. sync overhead)':<46}{1000 * wall / frames:10.3f}")


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExamplePBFDamBreak": FastExamplePBFDamBreak,
        "PBFScaling": PBFScaling,
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
    parser.add_argument("--profile", action="store_true", help="Print a per-kernel time breakdown and exit.")
    parser.add_argument("--dim", type=int, nargs=3, default=[46, 46, 46], help="Fluid grid dimensions")
    parser.add_argument("--frames", type=int, default=10, help="Frames to profile")
    args = parser.parse_known_args()[0]

    if args.profile:
        profile(dim=tuple(args.dim), frames=args.frames)
    else:
        if args.bench is None:
            for name, benchmark in benchmark_list.items():
                run_benchmark(benchmark)
        else:
            for name in args.bench:
                run_benchmark(benchmark_list[name])
