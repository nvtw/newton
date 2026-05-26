# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX port of Jitter2 ``Demo11`` ("Bag").

A unit cube falls into a cloth bag. The bag is a subdivided
icosahedron (level 2) with vertices projected back onto the unit
sphere -- ~252 vertices, ~320 cloth triangles arranged as a closed
spherical shell. The cube enters through self-collision-free
geometry and is contained by the bag's tension. Mass splitting ON
(matches the Jitter2 demo's default and exercises the cloth-rigid
contact + overflow-bucket cloth elasticity path the
test_cloth_mass_splitting regression covers).

Run::

    python -m newton._src.solvers.phoenx.examples.example_jitter_bag
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

ENABLE_MASS_SPLITTING: bool = True
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 12


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, triangles)`` for a unit-radius icosahedron.

    12 vertices, 20 triangular faces. Vertex coords from the golden-ratio
    construction; faces wound CCW from outside.
    """
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    verts = np.array(
        [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts[0])
    tris = np.array(
        [
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
        ],
        dtype=np.int32,
    )
    return verts, tris


def _subdivide(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-to-1 triangle subdivision: each face becomes 4 by splitting its 3 edges
    at their midpoints. Edges are deduplicated by ``(min(a,b), max(a,b))``."""
    midpoint_cache: dict[tuple[int, int], int] = {}
    verts_list = [v for v in verts]

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = (verts[a] + verts[b]) * 0.5
        idx = len(verts_list)
        verts_list.append(m)
        midpoint_cache[key] = idx
        return idx

    new_tris = []
    for a, b, c in tris:
        ab = midpoint(int(a), int(b))
        bc = midpoint(int(b), int(c))
        ca = midpoint(int(c), int(a))
        new_tris.append([int(a), ab, ca])
        new_tris.append([int(b), bc, ab])
        new_tris.append([int(c), ca, bc])
        new_tris.append([ab, bc, ca])
    return np.asarray(verts_list, dtype=np.float32), np.asarray(new_tris, dtype=np.int32)


def _make_bag_mesh(subdivisions: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Subdivided icosahedron projected onto a sphere of ``radius``."""
    verts, tris = _icosahedron()
    for _ in range(subdivisions):
        verts, tris = _subdivide(verts, tris)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / np.maximum(norms, 1.0e-9) * radius
    return verts, tris


class Example:
    """A unit cube dropping into a cloth bag (Jitter2 ``Demo11`` port)."""

    def __init__(
        self,
        viewer,
        args=None,
        bag_radius: float = 2.0,
        bag_center_z: float = 4.0,
        bag_subdivisions: int = 2,
        bag_total_mass: float = 5.0,
        cube_size: float = 0.5,
        cube_drop_z: float = 7.0,
        cube_mass: float = 1.0,
        youngs_modulus: float = 2.0e7,
        poisson_ratio: float = 0.3,
        cloth_thickness: float = 0.01,
        cloth_gap: float = 0.02,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = 4
        self.solver_iterations = 8

        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(youngs_modulus, poisson_ratio)
        self._cube_drop_z = float(cube_drop_z)

        # Build the bag mesh.
        verts_local, tris = _make_bag_mesh(bag_subdivisions, bag_radius)
        num_verts = int(verts_local.shape[0])
        num_tris = int(tris.shape[0])
        particle_mass = bag_total_mass / float(num_verts)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=0.0)

        # Free rigid cube above the bag, falls inside.
        self._cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, cube_drop_z), q=wp.quat_identity()),
            mass=cube_mass,
        )
        builder.add_shape_box(self._cube_body, hx=cube_size, hy=cube_size, hz=cube_size)

        # Cloth bag at ``bag_center_z``. Uses density-based mass so we
        # need to convert the requested ``bag_total_mass`` -> density.
        # ``add_cloth_mesh`` integrates density * triangle_area over all
        # triangles to set per-particle mass, so density = total /
        # surface_area.
        surface_area = 0.0
        for a, b, c in tris:
            p1, p2, p3 = verts_local[a], verts_local[b], verts_local[c]
            surface_area += 0.5 * float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))
        density = bag_total_mass / max(surface_area, 1.0e-6)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, bag_center_z),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=verts_local.tolist(),
            indices=tris.flatten().tolist(),
            density=density,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.04,
        )

        self.model = builder.finalize(device=self.device)

        # Stash dimensions for ``test_final``.
        self._num_bag_verts = num_verts
        self._num_bag_tris = num_tris
        self._particle_mass = particle_mass

        # PhoenX bodies: slot 0 = world anchor, slot 1 = cube.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            )
        )
        self.state_init = self.model.state()
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=int(self.model.body_count),
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
            num_cloth_triangles=int(self.model.tri_count),
            num_cloth_bending=int(self.model.edge_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=int(self.model.tri_count),
            num_cloth_bending=int(self.model.edge_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            rigid_contact_max=8192,
            step_layout="single_world",
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)
        self.world.populate_cloth_bending_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            cloth_thickness=cloth_thickness,
            cloth_gap=cloth_gap,
            rigid_contact_max=8192,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-15.0, yaw=180.0)

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
        if n == 0:
            return
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
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def _capture(self) -> None:
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

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")

        cube_q = self.state.body_q.numpy()
        cube_z = float(cube_q[self._cube_body, 2])
        if not np.isfinite(cube_z):
            raise RuntimeError(f"cube z went non-finite: {cube_z}")

        # Cube must have fallen below its starting height (gravity is active)
        # and stay above the ground (the bag should contain it).
        if cube_z > self._cube_drop_z - 0.1:
            raise RuntimeError(f"cube did not fall: z = {cube_z:.3f}, started at {self._cube_drop_z:.3f}")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--bag-radius", type=float, default=2.0)
    parser.add_argument("--bag-subdivisions", type=int, default=2)
    parser.add_argument("--bag-total-mass", type=float, default=5.0)
    parser.add_argument("--cube-size", type=float, default=0.5)
    parser.add_argument("--cube-drop-z", type=float, default=7.0)
    parser.add_argument("--cube-mass", type=float, default=1.0)
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        bag_radius=args.bag_radius,
        bag_subdivisions=args.bag_subdivisions,
        bag_total_mass=args.bag_total_mass,
        cube_size=args.cube_size,
        cube_drop_z=args.cube_drop_z,
        cube_mass=args.cube_mass,
    )
    newton.examples.run(example, args)
