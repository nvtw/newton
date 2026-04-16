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

###########################################################################
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
#
###########################################################################

import time
from collections import defaultdict

import numpy as np
import warp as wp

import newton
import newton.examples

USE_MESH_CUBES = True
USE_MESH_GROUND = True

# Use HybridViewer for ray-traced rendering with DLSS
USE_HYBRID_VIEWER = False

# wp.config.mode = "debug"
# wp.config.verify_cuda = True

# Solver Selection
# =================
# Choose which solver to use for rigid body contact simulation:
SOLVER_TYPE = "XPBD"  # Options: "XPBD", "MUJOCO_NEWTON", "MUJOCO_NATIVE", "FEATHERSTONE"

# Solver descriptions:
# - "XPBD": Newton's native XPBD solver (fast, stable, good for general use)
# - "MUJOCO_NEWTON": MuJoCo Warp solver using Newton contacts (best of both worlds)
# - "MUJOCO_NATIVE": MuJoCo Warp solver using MuJoCo contacts (MuJoCo's native contact handling)
# - "FEATHERSTONE": Featherstone reduced-coordinate solver (good for articulated systems)

# Contact Detection Frequency
# ============================
# Choose when to compute contacts:
# - "PER_TIMESTEP": Compute contacts once per frame (faster, works well with MuJoCo)
# - "PER_SUBSTEP": Compute contacts every substep (more accurate for fast-moving objects)
# - "AUTO": Automatically choose based on solver (PER_TIMESTEP for MuJoCo, PER_SUBSTEP for others)
CONTACT_FREQUENCY = "AUTO"  # Options: "PER_TIMESTEP", "PER_SUBSTEP", "AUTO"

# XPBD Contact Response Tuning
# =============================
# These parameters control how "violent" or "soft" contact responses are in XPBD
XPBD_ITERATIONS = 2  # Number of constraint solver iterations (2-5 typical)
XPBD_RIGID_CONTACT_RELAXATION = 0.8  # Contact stiffness: 0.3=soft, 0.8=medium, 1.0=stiff
XPBD_ANGULAR_DAMPING = 0.0  # Damping for rotation: 0.0=none, 0.05=moderate, 0.1=heavy

# CUDA Graph Capture
# ==================
# Enable CUDA graph capture for better performance (CUDA devices only)
USE_CUDA_GRAPH = True  # Set to True to enable CUDA graph capture

# Broad Phase Mode
# ================
# Choose broad phase collision detection mode:
# - "nxn": All-pairs AABB (O(N²), good for small scenes)
# - "sap": Sweep-and-prune AABB (O(N log N), better for larger scenes)
# - "explicit": Use precomputed shape pairs (most efficient)
BROAD_PHASE_MODE = "nxn"

# Contact Reduction
# =================
# Enable contact reduction to reduce the number of contacts per shape pair
USE_CONTACT_REDUCTION = True


class Example:
    def __init__(self, viewer, args=None):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        viewer._paused = False

        print("Example Basic Shapes")

        builder = newton.ModelBuilder()

        if USE_MESH_GROUND:
            # Generate procedural terrain mesh
            from newton._src.geometry import create_mesh_terrain

            terrain_vertices, terrain_indices = create_mesh_terrain(
                grid_size=(6, 6),  # 6x6 grid of terrain blocks
                block_size=(5.0, 5.0),  # Each block is 5x5 meters
                terrain_types=["flat", "wave", "random_grid", "pyramid_stairs"],
                terrain_params={
                    "pyramid_stairs": {"step_width": 0.4, "step_height": 0.05, "platform_width": 0.8},
                    "random_grid": {"grid_width": 0.4, "grid_height_range": (0, 0.1)},
                    "wave": {"wave_amplitude": 0.2, "wave_frequency": 1.5},
                },
                seed=42,
            )
            terrain_mesh = newton.Mesh(terrain_vertices, terrain_indices)
            terrain_mesh.build_sdf(max_resolution=512)
            terrain_offset = wp.transform(p=wp.vec3(-15.0, -15.0, -0.5), q=wp.quat_identity())
            builder.add_shape_mesh(
                body=-1,
                mesh=terrain_mesh,
                xform=terrain_offset,
            )
        else:
            builder.add_ground_plane()

        # ICOSAHEDRON MESH (for use in grid)
        # Create an icosahedron using the golden ratio
        # Each face gets its own duplicated vertices with per-face normals for flat shading
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
        ico_radius = 0.35  # Radius to match other shapes (spheres ~0.3, boxes ~0.3)

        # Original icosahedron vertices (12 vertices) - normalized to unit sphere then scaled
        ico_base_vertices = np.array(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ],
            dtype=np.float32,
        )
        # Normalize each vertex to unit length, then scale to desired radius
        for i in range(len(ico_base_vertices)):
            ico_base_vertices[i] = ico_base_vertices[i] / np.linalg.norm(ico_base_vertices[i]) * ico_radius

        # Icosahedron face indices (20 triangular faces, referencing base vertices)
        ico_face_indices = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        # Duplicate vertices per face and compute per-face normals for flat shading
        ico_vertices = []
        ico_normals = []
        ico_indices = []
        for face_idx, face in enumerate(ico_face_indices):
            # Get the 3 vertices of this face
            v0 = ico_base_vertices[face[0]]
            v1 = ico_base_vertices[face[1]]
            v2 = ico_base_vertices[face[2]]

            # Compute face normal (cross product of two edges)
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)

            # Add vertices and assign the same face normal to all 3 vertices
            ico_vertices.extend([v0, v1, v2])
            ico_normals.extend([normal, normal, normal])

            base = face_idx * 3
            ico_indices.extend([base, base + 1, base + 2])

        ico_vertices = np.array(ico_vertices, dtype=np.float32)
        ico_normals = np.array(ico_normals, dtype=np.float32)
        ico_indices = np.array(ico_indices, dtype=np.int32)

        ico_mesh = newton.Mesh(ico_vertices, ico_indices, normals=ico_normals)

        # CUBE MESH (triangle mesh version of box)
        hs = 0.3  # half-size, same as the box shape
        cube_verts = np.array(
            [
                [-hs, -hs, -hs],
                [hs, -hs, -hs],
                [hs, hs, -hs],
                [-hs, hs, -hs],
                [-hs, -hs, hs],
                [hs, -hs, hs],
                [hs, hs, hs],
                [-hs, hs, hs],
            ],
            dtype=np.float32,
        )
        # CCW winding when viewed from outside
        cube_tris = np.array(
            [
                0, 3, 2, 0, 2, 1,  # -Z face
                4, 5, 6, 4, 6, 7,  # +Z face
                0, 1, 5, 0, 5, 4,  # -Y face
                2, 3, 7, 2, 7, 6,  # +Y face
                0, 4, 7, 0, 7, 3,  # -X face
                1, 2, 6, 1, 6, 5,  # +X face
            ],
            dtype=np.int32,
        )
        cube_mesh = newton.Mesh(cube_verts, cube_tris)
        cube_mesh.build_sdf(max_resolution=64)

        # 3D GRID OF SHAPES
        # Configuration
        grid_size_x = 10  # Number of shapes along X axis
        grid_size_y = 10  # Number of shapes along Y axis
        grid_size_z = 15  # Number of shapes along Z axis
        grid_spacing = 1.5  # Space between shape centers
        grid_offset = wp.vec3(-10.0, -10.0, 2.0)  # Starting position of the grid
        position_randomness = 0.2  # Amount of random offset to add to each position (in units)

        # Shape types to cycle through (every second cube-like shape is a mesh when USE_MESH_CUBES is True)
        if USE_MESH_CUBES:
            shape_types = ["sphere", "box", "capsule", "mesh_cube", "cylinder", "cone", "icosahedron"]
        else:
            shape_types = ["sphere", "box", "capsule", "box", "cylinder", "cone", "icosahedron"]
        shape_index = 0

        print(
            f"Creating 3D grid: {grid_size_x}x{grid_size_y}x{grid_size_z} = {grid_size_x * grid_size_y * grid_size_z} shapes"
        )

        # Set random seed for reproducibility (optional - comment out for different results each run)
        rng = np.random.default_rng(42)

        for ix in range(grid_size_x):
            for iy in range(grid_size_y):
                for iz in range(grid_size_z):
                    # Calculate base position
                    base_x = grid_offset[0] + ix * grid_spacing
                    base_y = grid_offset[1] + iy * grid_spacing
                    base_z = grid_offset[2] + iz * grid_spacing

                    # Add random offset to position
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
                    elif shape_type == "mesh_cube":
                        builder.add_shape_mesh(body, mesh=cube_mesh)
                    elif shape_type == "icosahedron":
                        builder.add_shape_convex_hull(body, mesh=ico_mesh)

                    # Add free joint for MuJoCo compatibility
                    joint = builder.add_joint_free(body)
                    # Each free body is its own articulation
                    builder.add_articulation([joint])

        # finalize model
        self.model = builder.finalize()

        # Create contact reduction (optional)
        contact_reduction = None
        if USE_CONTACT_REDUCTION:
            from newton._src.geometry.contact_reduction_global import GlobalContactReducer

            print("Creating GlobalContactReducer instance")
            contact_reduction = GlobalContactReducer(
                capacity=1000000,
            )

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        # Can override with: --collision-pipeline unified|standard --broad-phase-mode nxn|sap|explicit
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)

        # Determine contact detection frequency
        if CONTACT_FREQUENCY == "AUTO":
            # MuJoCo solvers work well with once-per-timestep contacts
            self.contacts_per_timestep = SOLVER_TYPE in ["MUJOCO_NEWTON", "MUJOCO_NATIVE"]
        else:
            self.contacts_per_timestep = CONTACT_FREQUENCY == "PER_TIMESTEP"

        contact_freq_str = "once per timestep" if self.contacts_per_timestep else "every substep"
        print(f"Contact detection: {contact_freq_str}")

        # Initialize solver based on the selected type
        if SOLVER_TYPE == "XPBD":
            print("Using XPBD solver")
            print(f"  Iterations: {XPBD_ITERATIONS}")
            print(f"  Contact relaxation: {XPBD_RIGID_CONTACT_RELAXATION}")
            print(f"  Angular damping: {XPBD_ANGULAR_DAMPING}")
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=XPBD_ITERATIONS,
                rigid_contact_relaxation=XPBD_RIGID_CONTACT_RELAXATION,
                angular_damping=XPBD_ANGULAR_DAMPING,
            )
        elif SOLVER_TYPE == "MUJOCO_NEWTON":
            print("Using MuJoCo Warp solver with Newton contacts")
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,  # Use Newton contacts instead of MuJoCo contacts
                iterations=20,
                ls_iterations=10,
                integrator="euler",
                solver="cg",
            )
        elif SOLVER_TYPE == "MUJOCO_NATIVE":
            print("Using MuJoCo Warp solver with MuJoCo contacts")
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=True,  # Use MuJoCo's native contact handling
                iterations=20,
                ls_iterations=10,
                integrator="euler",
                solver="cg",
            )
        elif SOLVER_TYPE == "FEATHERSTONE":
            print("Using Featherstone reduced-coordinate solver")
            self.solver = newton.solvers.SolverFeatherstone(self.model, angular_damping=0.05, friction_smoothing=1.0)
        else:
            raise ValueError(
                f"Unknown solver type: {SOLVER_TYPE}. Choose from: XPBD, MUJOCO_NEWTON, MUJOCO_NATIVE, FEATHERSTONE"
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Kernel timing setup
        self.enable_kernel_timing = True
        self.step_count = 0
        self.max_sim_steps = 200
        self._terminate_requested = False
        self._exit_report_printed = False
        self._timing_totals: dict[str, list[float]] = defaultdict(list)
        self._timing_frame_count = 0
        self.perf_log_period = 60
        self.last_perf_step = 0
        self.last_perf_time = time.perf_counter()

        # CUDA graph launches do not provide per-kernel timings.
        if self.enable_kernel_timing:
            self.graph = None
        else:
            self.capture()

    def _print_exit_benchmark_report(self):
        """Print kernel timing summary and peak memory usage."""
        if self._exit_report_printed:
            return
        self._exit_report_printed = True

        if self._timing_frame_count > 0:
            frame_count = self._timing_frame_count
            width = 110
            kernel_width = width - 30
            print(f"\n{'=' * width}")
            print(f"  Kernel timing report ({frame_count} frames)")
            print(f"{'=' * width}")
            print(f"{'Kernel':<{kernel_width}} {'Total ms':>10} {'Avg ms':>10} {'Count':>7}")
            print(f"{'-' * kernel_width} {'-' * 10} {'-' * 10} {'-' * 7}")

            grand_total = 0.0
            rows = []
            for name, times in self._timing_totals.items():
                total = float(sum(times))
                grand_total += total
                rows.append((total, name, total / len(times), len(times)))
            rows.sort(key=lambda row: row[0], reverse=True)

            for total, name, avg, count in rows:
                label = name if len(name) <= kernel_width else name[: kernel_width - 3] + "..."
                print(f"{label:<{kernel_width}} {total:>10.3f} {avg:>10.4f} {count:>7}")

            print(f"{'-' * kernel_width} {'-' * 10}")
            print(f"{'TOTAL':<{kernel_width}} {grand_total:>10.3f}")
            print(f"{'Per-frame average':<{kernel_width}} {grand_total / frame_count:>10.3f}")
            print()

        device = wp.get_device()
        if device.is_cuda and wp.is_mempool_enabled(device):
            peak_bytes = wp.get_mempool_used_mem_high(device)
            print(f"Warp mempool peak usage: {peak_bytes / (1024 * 1024):.2f} MiB")
        else:
            print("Warp mempool peak usage: unavailable (non-CUDA or mempool disabled)")

        # Machine-readable metric for autoresearch (per-frame average ms, lower is better)
        if self._timing_frame_count > 0:
            grand_total = sum(sum(t) for t in self._timing_totals.values())
            avg = grand_total / self._timing_frame_count
            print(f"METRIC: {avg:.4f}")

    def _report_steps_per_second(self):
        """Report simulation throughput every perf_log_period steps."""
        if self.step_count % self.perf_log_period != 0:
            return
        now = time.perf_counter()
        elapsed = now - self.last_perf_time
        steps_elapsed = self.step_count - self.last_perf_step
        if elapsed > 0.0 and steps_elapsed > 0:
            steps_per_second = steps_elapsed / elapsed
            print(f"[step {self.step_count}] throughput={steps_per_second:.2f} steps/s")
        self.last_perf_step = self.step_count
        self.last_perf_time = now

    def capture(self):
        if USE_CUDA_GRAPH and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        # Compute contacts once per timestep if configured (MuJoCo works well with this)
        if self.contacts_per_timestep:
            if SOLVER_TYPE in ["XPBD", "MUJOCO_NEWTON", "FEATHERSTONE"]:
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            else:
                self.contacts = None  # MuJoCo native contacts don't need Newton contacts

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Compute contacts every substep if configured (more accurate for fast-moving objects)
            if not self.contacts_per_timestep:
                if SOLVER_TYPE in ["XPBD", "MUJOCO_NEWTON", "FEATHERSTONE"]:
                    self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
                else:
                    self.contacts = None  # MuJoCo native contacts don't need Newton contacts

            # Step solver - all solvers use the same interface
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.enable_kernel_timing:
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN)
            self.simulate()
            timing_results = wp.timing_end()
            for result in timing_results:
                self._timing_totals[result.name].append(result.elapsed)
            self._timing_frame_count += 1
        elif self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.step_count += 1
        self.sim_time += self.frame_dt
        self._report_steps_per_second()

        if self.step_count >= self.max_sim_steps and not self._terminate_requested:
            self._terminate_requested = True
            print(f"[step {self.step_count}] Reached stop at {self.max_sim_steps} simulation steps.")

    def test(self):
        pass

    def render(self):
        if self._terminate_requested:
            self._print_exit_benchmark_report()
            self.viewer.close()
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    if USE_HYBRID_VIEWER:
        # Use DLSS Ray Reconstruction hybrid viewer
        import pyglet
        from minimal_dlssrr import HybridViewer

        viewer = HybridViewer(width=1920, height=1080)

        # Initialize window and bridge before creating example (which calls set_model)
        viewer._init_window()
        viewer._init_bridge()

        example = Example(viewer)

        # Set up shared texture for zero-copy GPU interop
        viewer._setup_shared_texture()

        # Request TLAS update after scene is built
        if viewer._bridge is not None:
            viewer._bridge.request_tlas_update()

        def update(dt):
            # Update camera from keyboard input
            viewer._update_camera_from_input(dt)

            if not viewer.is_paused():
                example.step()
            example.render()

        pyglet.clock.schedule_interval(update, 1.0 / example.fps)
        pyglet.app.run()
    else:
        # Standard newton viewer
        viewer, args = newton.examples.init()
        example = Example(viewer, args)
        newton.examples.run(example, args)
        example._print_exit_benchmark_report()
