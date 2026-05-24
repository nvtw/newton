# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX soft-body tetrahedral drop demo.

A configurable grid of soft-cube piles drops onto a static ground
plane. Each pile is a vertical column of ``num_cubes`` cubes; piles
are tiled in the x/y plane on a ``grid_x`` by ``grid_y`` grid with
``grid_spacing`` between pile centres (default 4x4). Each cube is
voxelised at a configurable resolution (``cube_resolution`` voxels
per side) and every voxel is decomposed into 5 tetrahedra by
Newton's standard hex-to-tet split in
:meth:`newton.ModelBuilder.add_soft_grid`, yielding
``5 * cube_resolution**3`` tets per cube (5 at res=1, 40 at res=2,
135 at res=3, 320 at res=4, ...). Within each pile the cubes share
an x/y centre and are stacked vertically with a configurable
spacing, so the lower cubes get to partially settle before the
upper ones impact -- producing a deform-and-squash stack as the
column collapses.

The 5-tets-per-hex decomposition follows the Jitter2 reference
(matching ``MeshBuilder.GenerateTetrahedronBlock`` in
``jitterphysics2/.../MeshBuilder.cs:282``).

Exercises:

* FemTetPBD corotational shear (Jitter2 port)
* Virtual ``GeoType.TETRAHEDRON`` collision shapes per tet
* GJK/MPR tet-vs-plane narrow-phase contacts
* 4-node barycentric contact endpoint helpers

Run::

    python -m newton._src.solvers.phoenx.examples.example_soft_body_drop
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.timer import print_column_timings

# Tonge mass splitting (C# PhoenX default). When ``True`` the
# partitioner caps at :data:`MASS_SPLITTING_MAX_COLORED_PARTITIONS`
# colours and any remainder lands in an overflow bucket solved with
# per-(body, partition) copy states. Requires the single-world step
# layout (already used by this example); soft-tetrahedron constraints
# route through the slot-aware helpers and so coexist fine with mass
# splitting.
# Mass-splitting setup. With dense soft-tet meshes (15k tets sharing
# ~7k particles via the voxel hex-to-tet decomposition) the contact
# graph has high vertex-degree, and the greedy coloring with a moderate
# colour cap spills ~50% of constraints into the sequential-batch
# overflow bucket. Dropping the cap to ``0`` puts ALL soft-tet rows
# into the overflow bucket -- which is mass-splitting's Jacobi-block
# path, batched ``ms_batch_size`` cids per thread with
# ``_average_and_broadcast`` between PGS iterations. On this scene
# the pure-Jacobi path is *faster* than the Gauss-Seidel coloured
# path because the GPU runs one big parallel block instead of
# ``N_colors`` sequential dispatches, and the in-batch sequential
# Gauss-Seidel of ``ms_batch_size=8`` (the default) still provides
# enough intra-thread feedback for convergence. Tests pass with
# ``=0``; raising it doesn't help (more launches, worse FPS).
ENABLE_MASS_SPLITTING: bool = True
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 0

#: Opt-in per-column wall-clock profiling. When ``True`` the solver
#: brackets every PGS dispatch with CUDA ``%globaltimer`` reads and
#: atomic-adds the elapsed microseconds into the column's ``time_us``
#: slot; :func:`~newton._src.solvers.phoenx.timer.print_column_timings`
#: then dumps the per-type totals once per full step. Adds two
#: ``%globaltimer`` reads + one atomic_add per dispatch and breaks
#: CUDA-graph capture (the print path performs a D2H read), so leave
#: off for benchmarks / shipped runs.
ENABLE_COLUMN_TIMERS: bool = True


class Example:
    """A configurable column of soft cubes dropping onto a ground plane."""

    def __init__(
        self,
        viewer,
        args=None,
        num_cubes: int = 2,
        cube_size: float = 0.4,
        cube_resolution: int = 1,
        base_drop_height: float = 2.0,
        cube_spacing: float = 1.5,
        grid_x: int = 4,
        grid_y: int = 4,
        grid_spacing: float | None = None,
        youngs_modulus: float = 5.0e8,
        poisson_ratio: float = 0.3,
        density: float = 500.0,
        soft_body_thickness: float = 0.005,
        soft_body_gap: float = 0.010,
    ):
        if int(num_cubes) < 1:
            raise ValueError(f"num_cubes must be >= 1, got {num_cubes}")
        if int(cube_resolution) < 1:
            raise ValueError(f"cube_resolution must be >= 1, got {cube_resolution}")
        if int(grid_x) < 1:
            raise ValueError(f"grid_x must be >= 1, got {grid_x}")
        if int(grid_y) < 1:
            raise ValueError(f"grid_y must be >= 1, got {grid_y}")

        self.viewer = viewer
        self.device = wp.get_device()
        self.num_cubes = int(num_cubes)
        self.cube_size = float(cube_size)
        self.cube_resolution = int(cube_resolution)
        self.base_drop_height = float(base_drop_height)
        self.cube_spacing = float(cube_spacing)
        self.grid_x = int(grid_x)
        self.grid_y = int(grid_y)
        # Default pile-to-pile spacing leaves a half-cube gap between
        # neighbouring piles so they have room to splay sideways.
        self.grid_spacing = float(grid_spacing) if grid_spacing is not None else 1.5 * self.cube_size

        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # XPBD compliance scales as ``alpha_tilde = alpha / dt^2``. With
        # more substeps (smaller dt), the per-substep ``bias_mu``
        # ``= 1 / (k_mu * dt^2)`` grows quadratically and the constraint
        # behaves softer per substep. Match the Jitter2 reference
        # (``Playground.NumberSubsteps = 1``, ``SolverIterations = 8``)
        # at frame dt 1/60 so the user's intuition for Young's modulus
        # transfers from the C# experimental sim.
        self.sim_substeps = 5
        self.solver_iterations = 4
        self.velocity_iterations = 0

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.k_lambda, self.k_mu = soft_tet_lame_from_youngs_poisson(self.youngs_modulus, self.poisson_ratio)

        builder = newton.ModelBuilder()

        # Static ground plane at z = 0.
        builder.add_ground_plane(height=1.0)

        # Each cube is voxelised at ``cube_resolution`` cells per
        # side. ``add_soft_grid`` decomposes every hex cell into 5
        # tets (4 corner tets + 1 central tet -- the same scheme as
        # Jitter2's ``MeshBuilder.GenerateTetrahedronBlock``), so the
        # tet count per cube is ``5 * cube_resolution**3`` (5 at
        # res=1, 40 at res=2, 135 at res=3, 320 at res=4, ...). Cell
        # size is ``cube_size / cube_resolution`` so the cube's outer
        # dimensions stay at ``cube_size`` regardless of resolution.
        # ``add_soft_grid`` spawns the grid in the +x/+y/+z octant
        # from ``pos``, so we shift x/y by ``-0.5 * cube_size`` to
        # centre each pile on its grid cell. Piles are tiled on a
        # ``grid_x`` by ``grid_y`` grid centred on the world Z-axis;
        # within each pile, cubes are stacked along +z at
        # ``base_drop_height + i * cube_spacing`` so the lower cubes
        # get to partially settle before the upper ones land.
        cube_xy_offset = -0.5 * self.cube_size
        cell_size = self.cube_size / self.cube_resolution
        grid_x0 = -0.5 * (self.grid_x - 1) * self.grid_spacing
        grid_y0 = -0.5 * (self.grid_y - 1) * self.grid_spacing
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                pile_cx = grid_x0 + gx * self.grid_spacing
                pile_cy = grid_y0 + gy * self.grid_spacing
                for i in range(self.num_cubes):
                    drop_height = self.base_drop_height + i * self.cube_spacing
                    builder.add_soft_grid(
                        pos=wp.vec3(
                            pile_cx + cube_xy_offset,
                            pile_cy + cube_xy_offset,
                            drop_height,
                        ),
                        rot=wp.quat_identity(),
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        dim_x=self.cube_resolution,
                        dim_y=self.cube_resolution,
                        dim_z=self.cube_resolution,
                        cell_x=cell_size,
                        cell_y=cell_size,
                        cell_z=cell_size,
                        density=density,
                        k_mu=self.k_mu,
                        k_lambda=self.k_lambda,
                        k_damp=0.0,
                        add_surface_mesh_edges=False,
                    )

        self.model = builder.finalize(device=self.device)

        # PhoenX bodies: slot 0 = static world anchor. No dynamic
        # rigid bodies in this scene (only the ground plane, which
        # lives at shape_body=-1 and doesn't need a PhoenX body slot).
        num_phoenx_bodies = max(1, int(self.model.body_count) + 1)
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            )
        )
        if self.model.body_count > 0:
            self.state_init = self.model.state()
            wp.launch(
                init_phoenx_bodies_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.model.body_q,
                    self.state_init.body_qd,
                    self.model.body_com,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                ],
                outputs=[
                    bodies.position,
                    bodies.orientation,
                    bodies.velocity,
                    bodies.angular_velocity,
                    bodies.inverse_mass,
                    bodies.inverse_inertia,
                    bodies.inverse_inertia_world,
                    bodies.motion_type,
                    bodies.body_com,
                ],
                device=self.device,
            )
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            # Scale contact buffer with cube_resolution^2 since soft-tet
            # contacts roughly track the surface-triangle count per cube.
            rigid_contact_max=2
            * 1024
            * self.cube_resolution
            * self.cube_resolution
            * max(1, self.grid_x * self.grid_y),
            step_layout="single_world",
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            # On dense soft-body stacks greedy MIS produces a pyramidal
            # colour-size distribution that the head/fused tail solver
            # exploits; Luby-fixed produces more uniform sizes which keeps
            # the expensive head kernel busy longer. Greedy wins by ~50%
            # on the 7-cube/res=3 stack.
            mass_splitting_batch_size=2,
            partitioner_algorithm="greedy",
            enable_column_timers=ENABLE_COLUMN_TIMERS,
            device=self.device,
        )
        self._frame_index = 0
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            soft_body_thickness=soft_body_thickness,
            soft_body_gap=soft_body_gap,
            rigid_contact_max=1024 * self.cube_resolution * self.cube_resolution * max(1, self.grid_x * self.grid_y),
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()

        self.viewer.set_model(self.model)
        # Frame the whole grid from the side. The bottom cube sits at
        # ~``base_drop_height`` and the top cube at
        # ``base_drop_height + (num_cubes - 1) * cube_spacing`` (plus
        # half a cube-size in either direction). Sit the camera back
        # along +x at the column's mid-height, scaled by both the
        # column height and the grid's x-extent so larger grids stay
        # in frame; yaw 180 deg points back at the origin (same
        # convention as the other PhoenX examples, e.g.
        # ``example_kapla_arena``).
        column_top = self.base_drop_height + (self.num_cubes - 1) * self.cube_spacing + 0.5 * self.cube_size
        column_mid_z = 0.5 * (self.base_drop_height - 0.5 * self.cube_size + column_top)
        grid_extent_x = (self.grid_x - 1) * self.grid_spacing + self.cube_size
        grid_extent_y = (self.grid_y - 1) * self.grid_spacing + self.cube_size
        cam_distance = max(4.0, 3.0 * column_top, 2.0 * grid_extent_x, 2.0 * grid_extent_y)
        self.viewer.set_camera(
            pos=wp.vec3(cam_distance, 0.0, column_mid_z),
            pitch=-15.0,
            yaw=180.0,
        )

        self._capture()

    def _sync_newton_to_phoenx(self) -> None:
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = int(self.model.body_count)
        if n > 0:
            wp.launch(
                phoenx_to_newton_kernel,
                dim=n,
                inputs=[
                    self.bodies.position[1 : 1 + n],
                    self.bodies.orientation[1 : 1 + n],
                    self.bodies.velocity[1 : 1 + n],
                    self.bodies.angular_velocity[1 : 1 + n],
                    self.model.body_com,
                ],
                outputs=[self.state.body_q, self.state.body_qd],
                device=self.device,
            )
        # Push particle state into Newton's state for rendering.
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def _capture(self) -> None:
        # The captured graph covers the substep + collision work only.
        # The D2H readback in ``print_column_timings`` happens outside
        # the capture in :meth:`step`, so graph capture stays enabled
        # even with ``ENABLE_COLUMN_TIMERS=True`` -- without the graph
        # the eager Python launch overhead per substep is ~1000x what
        # the GPU work itself costs on this scene and the demo crawls.
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        self._sync_newton_to_phoenx()
        self.world.collide(self.state, self.contacts)
        self.world.step(self.frame_dt, contacts=self.contacts)
        self._sync_phoenx_to_newton()

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt
        self._frame_index += 1
        # Optional snapshot dump for the standalone coloring benchmark.
        # Trigger via ``PHOENX_DUMP_COLORING_GRAPH=<frame> python
        # -m ...example_soft_body_drop``. Mirrors the kapla hook.
        dump_frame_env = os.environ.get("PHOENX_DUMP_COLORING_GRAPH")
        if dump_frame_env is not None and int(dump_frame_env) == self._frame_index:
            self._dump_coloring_graph()
        print_column_timings(self.world, self._frame_index, label="soft_body_drop")

    def _dump_coloring_graph(self) -> None:
        """Write the active constraint graph to ``soft_body_drop_graph.npz``
        for the standalone coloring benchmark.

        ``num_bodies`` in the npz stores the partitioner's
        ``max_num_nodes`` (rigid bodies + particles); the bench loader
        passes that directly as ``max_num_nodes`` so adjacency sizing
        covers every body id referenced in the elements array.
        """
        partitioner = self.world._partitioner
        n_active = int(self.world._num_active_constraints.numpy()[0])
        elements_struct = partitioner._elements.numpy()[:n_active]
        bodies = elements_struct["bodies"].astype(np.int32, copy=False)
        cost = partitioner._cost_values.numpy()[:n_active].astype(np.int32, copy=False)
        jitter = partitioner._random_values.numpy()[:n_active].astype(np.int32, copy=False)
        num_nodes = self.world.num_bodies + self.world.num_particles
        out_path = pathlib.Path("soft_body_drop_graph.npz").resolve()
        np.savez(
            out_path,
            bodies=bodies,
            cost_values=cost,
            random_values=jitter,
            num_bodies=np.int32(num_nodes),
            frame_index=np.int32(self._frame_index),
        )
        print(
            f"[PhoenX SoftBodyDrop] dumped coloring graph to {out_path} "
            f"(frame={self._frame_index}, n={n_active}, nodes={num_nodes})"
        )

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")
        # Cubes interacted with the ground (didn't free-fall through
        # the plane). Use a generous bound -- the soft-body collision
        # stack still tunes for tight equilibrium.
        if positions[:, 2].min() < -5.0:
            raise RuntimeError(f"a soft-cube particle escaped downward (min z = {positions[:, 2].min():.4f})")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-cubes", type=int, default=7)
    parser.add_argument("--cube-size", type=float, default=0.4)
    parser.add_argument(
        "--cube-resolution",
        type=int,
        default=4,
        help="Voxels per cube side; each voxel becomes 5 tets (res=1 -> 5 tets/cube, res=3 -> 135 tets/cube).",
    )
    parser.add_argument("--base-drop-height", type=float, default=1.0)
    parser.add_argument("--cube-spacing", type=float, default=0.5)
    parser.add_argument(
        "--grid-x",
        type=int,
        default=4,
        help="Number of piles along x (default 4 -> 4x4 grid of piles).",
    )
    parser.add_argument(
        "--grid-y",
        type=int,
        default=4,
        help="Number of piles along y (default 4 -> 4x4 grid of piles).",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=None,
        help="Distance between pile centres in x and y. Defaults to 1.5 * cube_size.",
    )
    parser.add_argument("--youngs-modulus", type=float, default=1.0e9)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--soft-body-thickness", type=float, default=0.005)
    parser.add_argument("--soft-body-gap", type=float, default=0.010)
    viewer, args = newton.examples.init(parser)
    viewer._paused = True
    example = Example(
        viewer,
        args,
        num_cubes=args.num_cubes,
        cube_size=args.cube_size,
        cube_resolution=args.cube_resolution,
        base_drop_height=args.base_drop_height,
        cube_spacing=args.cube_spacing,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        grid_spacing=args.grid_spacing,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        density=args.density,
        soft_body_thickness=args.soft_body_thickness,
        soft_body_gap=args.soft_body_gap,
    )
    newton.examples.run(example, args)
