# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark SDF sampling throughput for NanoVDB and texture-backed SDFs.

Measures distance and gradient evaluation at random query points near the
surface for SDF resolutions from 32 to 512.
"""

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

from newton import Mesh
from newton._src.geometry.sdf_utils import sample_sdf_extrapolated, sample_sdf_grad_extrapolated
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_grad
from newton._src.geometry.sdf_utils import SDF, SDFData


# ---------------------------------------------------------------------------
# Sampling kernels
# ---------------------------------------------------------------------------
@wp.kernel
def _bench_nanovdb_distance(
    sdf_data: SDFData,
    positions: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = sample_sdf_extrapolated(sdf_data, positions[tid])


@wp.kernel
def _bench_nanovdb_grad(
    sdf_data: SDFData,
    positions: wp.array(dtype=wp.vec3),
    out_dist: wp.array(dtype=float),
    out_grad: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    d, g = sample_sdf_grad_extrapolated(sdf_data, positions[tid])
    out_dist[tid] = d
    out_grad[tid] = g


@wp.kernel
def _bench_texture_distance(
    tex_data: TextureSDFData,
    positions: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = texture_sample_sdf(tex_data, positions[tid])


@wp.kernel
def _bench_texture_grad(
    tex_data: TextureSDFData,
    positions: wp.array(dtype=wp.vec3),
    out_dist: wp.array(dtype=float),
    out_grad: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    d, g = texture_sample_sdf_grad(tex_data, positions[tid])
    out_dist[tid] = d
    out_grad[tid] = g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_box_mesh(half_extents=(0.5, 0.5, 0.5)):
    hx, hy, hz = half_extents
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


def _random_surface_points(n, half_ext=0.5, spread=0.15, seed=42):
    """Generate random query points near the surface of a box."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-half_ext - spread, half_ext + spread, (n, 3)).astype(np.float32)
    return pts


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------
NUM_QUERY_POINTS = 100_000
NUM_LAUNCHES = 50


class FastSDFSampling:
    """Benchmark NanoVDB vs texture SDF sampling at varying resolutions."""

    repeat = 3
    number = 1
    timeout = 600

    params = ([32, 64, 128, 256, 512],)
    param_names = ["resolution"]

    def setup(self, resolution):
        device = "cuda:0"
        mesh = _create_box_mesh()
        sdf_obj = SDF.create_from_mesh(
            mesh,
            narrow_band_range=(-0.1, 0.1),
            max_resolution=resolution,
            margin=0.05,
        )

        # Keep references alive
        self._sdf_obj = sdf_obj

        self.sdf_data = sdf_obj.to_kernel_data()
        self.tex_data = sdf_obj.to_texture_kernel_data()

        pts_np = _random_surface_points(NUM_QUERY_POINTS)
        self.positions = wp.array(pts_np, dtype=wp.vec3, device=device)
        self.out_dist = wp.zeros(NUM_QUERY_POINTS, dtype=float, device=device)
        self.out_grad = wp.zeros(NUM_QUERY_POINTS, dtype=wp.vec3, device=device)

        # Warm-up: compile all four kernels
        wp.launch(_bench_nanovdb_distance, dim=NUM_QUERY_POINTS, inputs=[self.sdf_data, self.positions, self.out_dist], device=device)
        wp.launch(_bench_nanovdb_grad, dim=NUM_QUERY_POINTS, inputs=[self.sdf_data, self.positions, self.out_dist, self.out_grad], device=device)
        wp.launch(_bench_texture_distance, dim=NUM_QUERY_POINTS, inputs=[self.tex_data, self.positions, self.out_dist], device=device)
        wp.launch(_bench_texture_grad, dim=NUM_QUERY_POINTS, inputs=[self.tex_data, self.positions, self.out_dist, self.out_grad], device=device)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_nanovdb_distance(self, resolution):
        for _ in range(NUM_LAUNCHES):
            wp.launch(
                _bench_nanovdb_distance,
                dim=NUM_QUERY_POINTS,
                inputs=[self.sdf_data, self.positions, self.out_dist],
                device="cuda:0",
            )
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_nanovdb_grad(self, resolution):
        for _ in range(NUM_LAUNCHES):
            wp.launch(
                _bench_nanovdb_grad,
                dim=NUM_QUERY_POINTS,
                inputs=[self.sdf_data, self.positions, self.out_dist, self.out_grad],
                device="cuda:0",
            )
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_texture_distance(self, resolution):
        for _ in range(NUM_LAUNCHES):
            wp.launch(
                _bench_texture_distance,
                dim=NUM_QUERY_POINTS,
                inputs=[self.tex_data, self.positions, self.out_dist],
                device="cuda:0",
            )
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_texture_grad(self, resolution):
        for _ in range(NUM_LAUNCHES):
            wp.launch(
                _bench_texture_grad,
                dim=NUM_QUERY_POINTS,
                inputs=[self.tex_data, self.positions, self.out_dist, self.out_grad],
                device="cuda:0",
            )
        wp.synchronize_device()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(FastSDFSampling)
