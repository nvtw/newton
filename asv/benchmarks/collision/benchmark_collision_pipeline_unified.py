# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""ASV benchmarks for CollisionPipelineUnified performance.

Measures the performance of the unified collision pipeline with different:
1. Broad phase modes (NXN, SAP)
2. Scene sizes (number of dynamic shapes)

The benchmark scene consists of a 3D grid of mixed primitive shapes (spheres, boxes,
capsules, cylinders, cones) falling onto a flat 100x100 triangle mesh ground plane.
"""

import statistics

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

import newton


def create_flat_mesh_grid(grid_x: int, grid_y: int, cell_size: float = 1.0):
    """Create a flat triangle mesh grid.

    Args:
        grid_x: Number of quads along the X axis.
        grid_y: Number of quads along the Y axis.
        cell_size: Size of each cell in meters.

    Returns:
        Tuple of (vertices, indices) as numpy arrays.
    """
    # Create vertices for a (grid_x+1) x (grid_y+1) grid
    num_verts_x = grid_x + 1
    num_verts_y = grid_y + 1

    vertices = []
    for j in range(num_verts_y):
        for i in range(num_verts_x):
            x = i * cell_size
            y = j * cell_size
            z = 0.0
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)

    # Create triangle indices (2 triangles per quad)
    indices = []
    for j in range(grid_y):
        for i in range(grid_x):
            # Vertex indices for this quad
            v00 = j * num_verts_x + i
            v10 = j * num_verts_x + (i + 1)
            v01 = (j + 1) * num_verts_x + i
            v11 = (j + 1) * num_verts_x + (i + 1)

            # First triangle (lower-left)
            indices.extend([v00, v10, v01])
            # Second triangle (upper-right)
            indices.extend([v10, v11, v01])

    indices = np.array(indices, dtype=np.int32)

    return vertices, indices


def build_collision_scene(grid_size: int, broad_phase_mode: newton.BroadPhaseMode):
    """Build a scene with a 3D grid of shapes and a flat mesh ground.

    Args:
        grid_size: Number of shapes along each axis (total shapes = grid_size^3).
        broad_phase_mode: Broad phase mode for collision detection.

    Returns:
        Tuple of (model, collision_pipeline, state, solver).
    """
    builder = newton.ModelBuilder()

    # Create flat mesh ground (100x100 quads = 20000 triangles)
    ground_vertices, ground_indices = create_flat_mesh_grid(100, 100, cell_size=1.0)
    ground_mesh = newton.Mesh(ground_vertices, ground_indices)

    # Center the ground under the grid of shapes
    terrain_offset = wp.transform(p=wp.vec3(-50.0, -50.0, -0.5), q=wp.quat_identity())
    builder.add_shape_mesh(
        body=-1,  # Static body
        mesh=ground_mesh,
        xform=terrain_offset,
    )

    # Shape types to cycle through
    shape_types = ["sphere", "box", "capsule", "cylinder", "cone"]
    shape_index = 0

    # Grid parameters
    grid_spacing = 1.5
    grid_offset = wp.vec3(-grid_size * grid_spacing / 2, -grid_size * grid_spacing / 2, 2.0)
    rng = np.random.default_rng(42)
    position_randomness = 0.2

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Calculate position with random offset
                base_x = grid_offset[0] + i * grid_spacing
                base_y = grid_offset[1] + j * grid_spacing
                base_z = grid_offset[2] + k * grid_spacing

                random_offset_x = (rng.random() - 0.5) * 2 * position_randomness
                random_offset_y = (rng.random() - 0.5) * 2 * position_randomness
                random_offset_z = (rng.random() - 0.5) * 2 * position_randomness

                pos = wp.vec3(
                    base_x + random_offset_x,
                    base_y + random_offset_y,
                    base_z + random_offset_z,
                )

                # Cycle through different shape types
                shape_type = shape_types[shape_index % len(shape_types)]
                shape_index += 1

                # Create body
                body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))

                # Add shape based on type
                if shape_type == "sphere":
                    builder.add_shape_sphere(body, radius=0.3)
                elif shape_type == "box":
                    builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3)
                elif shape_type == "capsule":
                    builder.add_shape_capsule(body, radius=0.2, half_height=0.4)
                elif shape_type == "cylinder":
                    builder.add_shape_cylinder(body, radius=0.25, half_height=0.35)
                elif shape_type == "cone":
                    builder.add_shape_cone(body, radius=0.3, half_height=0.4)

                # Add free joint for articulation
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])

    # Finalize model
    model = builder.finalize()

    # Create collision pipeline
    collision_pipeline = newton.CollisionPipelineUnified.from_model(
        model,
        rigid_contact_max_per_pair=10,
        broad_phase_mode=broad_phase_mode,
    )

    # Create solver and state
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    return model, collision_pipeline, state, solver


class FastCollisionPipelineUnifiedCollide:
    """Benchmark collision detection (model.collide) with CollisionPipelineUnified.

    Uses CUDA graph capture for reliable timing measurements.
    Runs multiple iterations internally with warmup for stable results.
    """

    repeat = 5
    number = 1
    timeout = 300
    warmup_iterations = 5
    timed_iterations = 20
    params = [
        [4, 6, 8],  # grid_size (64, 216, 512 shapes)
        ["NXN", "SAP"],  # broad_phase_mode
    ]
    param_names = ["grid_size", "broad_phase_mode"]

    def setup(self, grid_size, broad_phase_mode_str):
        broad_phase_map = {
            "NXN": newton.BroadPhaseMode.NXN,
            "SAP": newton.BroadPhaseMode.SAP,
        }
        broad_phase_mode = broad_phase_map[broad_phase_mode_str]

        self.model, self.collision_pipeline, self.state, self.solver = build_collision_scene(
            grid_size, broad_phase_mode
        )
        self.control = self.model.control()

        # Warm up (required before graph capture)
        self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
        wp.synchronize()

        # Capture CUDA graph for collision detection
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
            self.graph = capture.graph

        # Warmup the captured graph
        for _ in range(self.warmup_iterations):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self, grid_size, broad_phase_mode_str):
        """Time the collision detection phase only (using CUDA graph).

        Runs multiple iterations internally and returns the median time.
        Uses wp.ScopedTimer for accurate GPU timing via CUDA events.
        """
        samples = []
        for _ in range(self.timed_iterations):
            with wp.ScopedTimer("collide", synchronize=True, print=False, cuda_filter=wp.TIMING_ALL) as timer:
                if self.graph is not None:
                    wp.capture_launch(self.graph)
                else:
                    self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
            samples.append(timer.elapsed)

        return statistics.median(samples)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_contact_count(self, grid_size, broad_phase_mode_str):
        """Track the number of contacts generated."""
        # Run without graph to get actual contact count
        self.contacts = self.model.collide(self.state, collision_pipeline=self.collision_pipeline)
        wp.synchronize()
        return self.contacts.rigid_contact_count.numpy()[0]


class FastCollisionPipelineUnifiedStep:
    """Benchmark full simulation step including collision detection.

    Uses CUDA graph capture for reliable timing measurements.
    Runs multiple iterations internally with warmup for stable results.
    """

    repeat = 5
    number = 1
    timeout = 300
    warmup_iterations = 5
    timed_iterations = 20
    params = [
        [4, 6, 8],  # grid_size (64, 216, 512 shapes)
        ["NXN", "SAP"],  # broad_phase_mode
    ]
    param_names = ["grid_size", "broad_phase_mode"]

    def _do_step(self):
        """Perform one simulation step (collide + solve)."""
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def setup(self, grid_size, broad_phase_mode_str):
        broad_phase_map = {
            "NXN": newton.BroadPhaseMode.NXN,
            "SAP": newton.BroadPhaseMode.SAP,
        }
        broad_phase_mode = broad_phase_map[broad_phase_mode_str]

        self.model, self.collision_pipeline, self.state_0, self.solver = build_collision_scene(
            grid_size, broad_phase_mode
        )
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sim_dt = 1.0 / 600.0  # 10 substeps at 60fps

        # Warm up (required before graph capture)
        self._do_step()
        wp.synchronize()

        # Capture CUDA graph for full step
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._do_step()
            self.graph = capture.graph

        # Warmup the captured graph
        for _ in range(self.warmup_iterations):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._do_step()
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, grid_size, broad_phase_mode_str):
        """Time a single simulation substep (collide + solve) using CUDA graph.

        Runs multiple iterations internally and returns the median time.
        Uses wp.ScopedTimer for accurate GPU timing via CUDA events.
        """
        samples = []
        for _ in range(self.timed_iterations):
            with wp.ScopedTimer("step", synchronize=True, print=False, cuda_filter=wp.TIMING_ALL) as timer:
                if self.graph is not None:
                    wp.capture_launch(self.graph)
                else:
                    self._do_step()
            samples.append(timer.elapsed)

        return statistics.median(samples)


class FastCollisionPipelineUnifiedFrame:
    """Benchmark a full frame (multiple substeps) of simulation.

    Uses CUDA graph capture for reliable timing measurements.
    Runs multiple iterations internally with warmup for stable results.
    """

    repeat = 3
    number = 1
    timeout = 600
    warmup_iterations = 3
    timed_iterations = 10
    params = [
        [4, 6],  # grid_size (64, 216 shapes) - smaller for frame benchmark
        ["NXN", "SAP"],  # broad_phase_mode
    ]
    param_names = ["grid_size", "broad_phase_mode"]

    def _do_frame(self):
        """Perform one full frame (multiple substeps)."""
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def setup(self, grid_size, broad_phase_mode_str):
        broad_phase_map = {
            "NXN": newton.BroadPhaseMode.NXN,
            "SAP": newton.BroadPhaseMode.SAP,
        }
        broad_phase_mode = broad_phase_map[broad_phase_mode_str]

        self.model, self.collision_pipeline, self.state_0, self.solver = build_collision_scene(
            grid_size, broad_phase_mode
        )
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.fps = 60
        self.substeps = 10
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.substeps

        # Warm up with one frame (required before graph capture)
        self._do_frame()
        wp.synchronize()

        # Capture CUDA graph for full frame
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._do_frame()
            self.graph = capture.graph

        # Warmup the captured graph
        for _ in range(self.warmup_iterations):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._do_frame()
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_frame(self, grid_size, broad_phase_mode_str):
        """Time a full frame (10 substeps at 60fps) using CUDA graph.

        Runs multiple iterations internally and returns the median time.
        Uses wp.ScopedTimer for accurate GPU timing via CUDA events.
        """
        samples = []
        for _ in range(self.timed_iterations):
            with wp.ScopedTimer("frame", synchronize=True, print=False, cuda_filter=wp.TIMING_ALL) as timer:
                if self.graph is not None:
                    wp.capture_launch(self.graph)
                else:
                    self._do_frame()
            samples.append(timer.elapsed)

        return statistics.median(samples)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastCollisionPipelineUnifiedCollide": FastCollisionPipelineUnifiedCollide,
        "FastCollisionPipelineUnifiedStep": FastCollisionPipelineUnifiedStep,
        "FastCollisionPipelineUnifiedFrame": FastCollisionPipelineUnifiedFrame,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
