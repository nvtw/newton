# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
from newton._src.solvers.stflip.sparse_grid import SparseGrid
from newton.solvers import SolverSTFLIP


class STFLIPStepScaling:
    """Track complete sparse-fluid step scaling with particle count."""

    params = ([8, 16, 24, 47],)
    param_names = ["particle_dimension"]
    number = 1
    repeat = 5
    rounds = 2

    def setup(self, particle_dimension):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverSTFLIP.register_custom_attributes(builder)
        spacing = 0.04
        builder.add_particle_grid(
            pos=wp.vec3(0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_dimension,
            dim_y=particle_dimension,
            dim_z=particle_dimension,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=1000.0 * spacing**3,
            jitter=0.0,
            radius_mean=0.015,
        )
        self.model = builder.finalize(device=device)
        tile_capacity = {8: 32, 16: 64, 24: 80, 47: 160}[particle_dimension]
        domain_extent = max(1.2, 0.04 * particle_dimension + 0.2)
        self.solver = SolverSTFLIP(
            self.model,
            SolverSTFLIP.Config(
                cell_size=0.08,
                tile_size=8,
                max_active_tile_count=tile_capacity,
                padding_tiles=1,
                pressure_iterations=40,
                transfer_scheme="apic",
                domain_lower=(0.0, 0.0, 0.0),
                domain_upper=(domain_extent, domain_extent, domain_extent),
            ),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.dt = 1.0 / 240.0
        self.solver.step(self.state_0, self.state_1, None, None, self.dt)
        self.solver.step(self.state_1, self.state_0, None, None, self.dt)
        wp.synchronize_device(device)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_five_steps(self, particle_dimension):
        del particle_dimension
        for _ in range(5):
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device()


class SparseGridRebuildScaling:
    """Track packed sparse-grid rebuild scaling with point count."""

    params = ([1024, 8192, 65536, 131072],)
    param_names = ["point_count"]
    number = 1
    repeat = 5
    rounds = 2

    def setup(self, point_count):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        rng = np.random.default_rng(1234)
        positions = rng.uniform(-1.0, 1.0, size=(point_count, 3)).astype(np.float32)
        self.positions = wp.array(positions, dtype=wp.vec3, device=device)
        self.active = wp.ones(point_count, dtype=wp.int32, device=device)
        self.grid = SparseGrid(
            point_capacity=point_count,
            tile_capacity=512,
            tile_size=8,
            cell_size=0.08,
            padding_tiles=1,
            device=device,
        )
        self.grid.build(self.positions, self.active)
        wp.synchronize_device(device)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_ten_rebuilds(self, point_count):
        del point_count
        for _ in range(10):
            self.grid.build(self.positions, self.active)
        wp.synchronize_device()


class STFLIPGraphReplay:
    """Compare eager stepping with steady-state CUDA graph replay."""

    params = ([False, True],)
    param_names = ["cuda_graph"]
    number = 1
    repeat = 5
    rounds = 2

    def setup(self, cuda_graph):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverSTFLIP.register_custom_attributes(builder)
        spacing = 0.04
        builder.add_particle_grid(
            pos=wp.vec3(0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=47,
            dim_y=47,
            dim_z=47,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=1000.0 * spacing**3,
            jitter=0.0,
            radius_mean=0.015,
        )
        self.model = builder.finalize(device=device)
        self.solver = SolverSTFLIP(
            self.model,
            SolverSTFLIP.Config(
                cell_size=0.08,
                tile_size=8,
                max_active_tile_count=160,
                padding_tiles=1,
                pressure_iterations=60,
                transfer_scheme="apic",
                domain_lower=(0.0, 0.0, 0.0),
                domain_upper=(2.2, 2.2, 2.2),
            ),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.dt = 1.0 / 240.0
        self.solver.step(self.state_0, self.state_1, None, None, self.dt)
        self.solver.step(self.state_1, self.state_0, None, None, self.dt)
        self.graph = None
        if cuda_graph:
            with wp.ScopedCapture(device=device) as capture:
                self.solver.step(self.state_0, self.state_1, None, None, self.dt)
                self.solver.step(self.state_1, self.state_0, None, None, self.dt)
            self.graph = capture.graph
        wp.synchronize_device(device)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_twenty_steps(self, cuda_graph):
        if cuda_graph:
            for _ in range(10):
                wp.capture_launch(self.graph)
        else:
            for _ in range(20):
                self.solver.step(self.state_0, self.state_1, None, None, self.dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(STFLIPStepScaling)
    run_benchmark(SparseGridRebuildScaling)
    run_benchmark(STFLIPGraphReplay)
