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

"""ASV benchmarks for broad phase collision detection algorithms.

This benchmark compares the performance of different broad phase algorithms:
- NxN: All-pairs with AABB checks (O(N²) pairs, good for small scenes)
- SAP: Sweep and Prune with segmented radix sort (O(N log N))
- SAP_TILE: Sweep and Prune with tile-based sort (shared memory, faster for some sizes)
- EXPLICIT: Precomputed pairs with AABB checks (fastest when pairs are known)

The test scene is a 3D grid of spheres with configurable density.
At "tight" spacing, many AABBs overlap, producing many candidate pairs.
At "loose" spacing, fewer AABBs overlap, testing rejection efficiency.

This isolates broad phase performance from narrow phase overhead.
"""

import statistics

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

import newton
from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP, SAPSortType
from newton._src.sim.collide_unified import compute_shape_aabbs


def build_sphere_grid_scene(grid_size: int, spacing_factor: float = 2.5):
    """Build a scene with a 3D grid of spheres for broad phase testing.

    Args:
        grid_size: Number of spheres along each axis (total = grid_size^3).
        spacing_factor: Multiplier for sphere spacing relative to diameter.
            - 1.0: Spheres touch (maximum overlap)
            - 2.0: Spheres separated by one diameter (some overlap from margin)
            - 3.0+: Spheres well separated (minimal overlap)

    Returns:
        Tuple of (model, state, shape_aabb_lower, shape_aabb_upper).
    """
    builder = newton.ModelBuilder()

    sphere_radius = 0.5
    sphere_diameter = 2.0 * sphere_radius
    spacing = sphere_diameter * spacing_factor

    # Create a 3D grid of spheres
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = i * spacing
                y = j * spacing
                z = k * spacing + sphere_radius  # Offset so spheres are above z=0

                body = builder.add_body(xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()))
                builder.add_shape_sphere(body, radius=sphere_radius)
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])

    # Finalize model (builds shape_contact_pairs automatically)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # Pre-compute AABBs (these are needed by broad phase)
    device = model.device
    shape_count = model.shape_count

    shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

    # Compute AABBs using the unified pipeline's kernel
    wp.launch(
        kernel=compute_shape_aabbs,
        dim=shape_count,
        inputs=[
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_type,
            model.shape_scale,
            model.shape_collision_radius,
            model.shape_source_ptr,
            model.shape_contact_margin,
        ],
        outputs=[
            shape_aabb_lower,
            shape_aabb_upper,
        ],
        device=device,
    )

    return model, state, shape_aabb_lower, shape_aabb_upper


class BroadPhaseComparison:
    """Benchmark comparing all broad phase algorithms.

    Compares NxN, SAP (segmented sort), SAP (tile sort), and EXPLICIT.
    Uses CUDA graph capture for reliable timing measurements.
    """

    repeat = 5
    number = 1
    timeout = 300
    warmup_iterations = 5
    timed_iterations = 20
    params = [
        [5, 8, 10, 12, 15, 18],  # grid_size (125, 512, 1000, 1728, 3375, 5832 shapes)
        [2.5],  # spacing_factor (medium - representative case)
        ["nxn", "sap_segmented", "sap_tile", "explicit"],  # broad_phase_type
    ]
    param_names = ["grid_size", "spacing_factor", "broad_phase_type"]

    def setup(self, grid_size, spacing_factor, broad_phase_type):
        # SAP tile has shared memory limits - skip at larger sizes
        # (0xc000 = 48KB max shared memory, tile sort needs ~shape_count * 12 bytes)
        self._skip = broad_phase_type == "sap_tile" and grid_size > 12
        if self._skip:
            self.shape_count = grid_size**3
            return

        self.model, self.state, self.shape_aabb_lower, self.shape_aabb_upper = build_sphere_grid_scene(
            grid_size, spacing_factor
        )
        self.device = self.model.device
        self.shape_count = self.model.shape_count
        self.broad_phase_type = broad_phase_type

        # Calculate max pairs for output buffer
        max_pairs = (self.shape_count * (self.shape_count - 1)) // 2

        # Allocate output arrays
        self.candidate_pairs = wp.zeros(max_pairs, dtype=wp.vec2i, device=self.device)
        self.num_candidate_pairs = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Create broad phase based on type
        if broad_phase_type == "nxn":
            self.broad_phase = BroadPhaseAllPairs(
                self.model.shape_world,
                shape_flags=self.model.shape_flags,
                device=self.device,
            )
            self._launch_func = self._launch_nxn
        elif broad_phase_type == "sap_segmented":
            self.broad_phase = BroadPhaseSAP(
                self.model.shape_world,
                shape_flags=self.model.shape_flags,
                sort_type=SAPSortType.SEGMENTED,
                device=self.device,
            )
            self._launch_func = self._launch_sap
        elif broad_phase_type == "sap_tile":
            self.broad_phase = BroadPhaseSAP(
                self.model.shape_world,
                shape_flags=self.model.shape_flags,
                sort_type=SAPSortType.TILE,
                device=self.device,
            )
            self._launch_func = self._launch_sap
        else:  # explicit
            self.broad_phase = BroadPhaseExplicit()
            self.shape_pairs_filtered = self.model.shape_contact_pairs
            self.num_precomputed_pairs = len(self.shape_pairs_filtered)
            self._launch_func = self._launch_explicit

        # Warmup (required before graph capture)
        self._launch_func()
        wp.synchronize()

        # Capture CUDA graph
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._launch_func()
            self.graph = capture.graph

        # Warmup the captured graph
        for _ in range(self.warmup_iterations):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._launch_func()
        wp.synchronize()

    def _launch_nxn(self):
        """Launch NxN broad phase."""
        self.broad_phase.launch(
            self.shape_aabb_lower,
            self.shape_aabb_upper,
            None,  # AABBs are pre-expanded
            self.model.shape_collision_group,
            self.model.shape_world,
            self.shape_count,
            self.candidate_pairs,
            self.num_candidate_pairs,
            device=self.device,
        )

    def _launch_sap(self):
        """Launch SAP broad phase."""
        self.broad_phase.launch(
            self.shape_aabb_lower,
            self.shape_aabb_upper,
            None,  # AABBs are pre-expanded
            self.model.shape_collision_group,
            self.model.shape_world,
            self.shape_count,
            self.candidate_pairs,
            self.num_candidate_pairs,
            device=self.device,
        )

    def _launch_explicit(self):
        """Launch EXPLICIT broad phase."""
        self.broad_phase.launch(
            self.shape_aabb_lower,
            self.shape_aabb_upper,
            None,  # AABBs are pre-expanded
            self.shape_pairs_filtered,
            self.num_precomputed_pairs,
            self.candidate_pairs,
            self.num_candidate_pairs,
            device=self.device,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_broad_phase(self, grid_size, spacing_factor, broad_phase_type):
        """Time the broad phase execution."""
        if getattr(self, "_skip", False):
            return float("nan")  # SAP tile exceeds shared memory at this size
        samples = []
        for _ in range(self.timed_iterations):
            with wp.ScopedTimer("broad_phase", synchronize=True, print=False) as timer:
                if self.graph is not None:
                    wp.capture_launch(self.graph)
                else:
                    self._launch_func()
            samples.append(timer.elapsed)
        return statistics.median(samples)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_candidate_pairs(self, grid_size, spacing_factor, broad_phase_type):
        """Track the number of candidate pairs found."""
        if self._skip:
            return float("nan")
        self._launch_func()
        wp.synchronize()
        return self.num_candidate_pairs.numpy()[0]

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_shape_count(self, grid_size, spacing_factor, broad_phase_type):
        """Track the number of shapes."""
        return self.shape_count

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_all_possible_pairs(self, grid_size, spacing_factor, broad_phase_type):
        """Track the total number of possible pairs (N*(N-1)/2)."""
        return (self.shape_count * (self.shape_count - 1)) // 2
