# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for rigid ground contacts in a cloth scene."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactConstraintData
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cloth contacts run on CUDA only.",
)
class TestClothHangingGroundContact(unittest.TestCase):
    def test_cloth_self_collision_defaults_enabled(self):
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            scene = _GroundCubeClothScene(device)
            self.assertIsNotNone(scene.graph, "expected CUDA graph-captured stepping")

            groups = scene.pipeline.unified_shape_collision_group.numpy()
            rigid_count = int(scene.model.shape_count)
            tri_count = int(scene.model.tri_count)
            cloth_groups = groups[rigid_count : rigid_count + tri_count]

            self.assertTrue(np.all(cloth_groups == 1))

    def test_soft_body_self_collision_defaults_enabled(self):
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=1,
                dim_y=1,
                dim_z=1,
                cell_x=0.1,
                cell_y=0.1,
                cell_z=0.1,
                density=1000.0,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=0.0,
                add_surface_mesh_edges=False,
                particle_radius=0.0,
            )
            model = builder.finalize(device=device)
            bodies = body_container_zeros(1, device=device)
            constraints = PhoenXWorld.make_constraint_container(
                num_joints=0,
                num_cloth_triangles=0,
                num_soft_tetrahedra=int(model.tet_count),
                device=device,
            )
            world = PhoenXWorld(
                bodies=bodies,
                constraints=constraints,
                num_joints=0,
                num_particles=int(model.particle_count),
                num_cloth_triangles=0,
                num_soft_tetrahedra=int(model.tet_count),
                num_worlds=1,
                step_layout="single_world",
                device=device,
            )
            pipeline = world.setup_cloth_collision_pipeline(model)

            groups = pipeline.unified_shape_collision_group.numpy()
            rigid_count = int(model.shape_count)
            tet_count = int(model.tet_count)
            tet_groups = groups[rigid_count : rigid_count + tet_count]

            self.assertTrue(np.all(tet_groups == 1))

    def test_cloth_self_collision_can_be_disabled(self):
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            scene = _GroundCubeClothScene(device, cloth_self_collision=False)
            self.assertIsNotNone(scene.graph, "expected CUDA graph-captured stepping")

            groups = scene.pipeline.unified_shape_collision_group.numpy()
            rigid_count = int(scene.model.shape_count)
            tri_count = int(scene.model.tri_count)
            cloth_groups = groups[rigid_count : rigid_count + tri_count]

            self.assertTrue(np.all(cloth_groups == -2))

    def test_cube_stays_on_ground_with_cloth_pipeline(self):
        device = wp.get_preferred_device()
        with wp.ScopedDevice(device):
            scene = _GroundCubeClothScene(device)
            self.assertIsNotNone(scene.graph, "expected CUDA graph-captured stepping")

            for _ in range(70):
                scene.step()

            cube_z = float(scene.state.body_q.numpy()[scene.cube_body, 2])
            cube_vz = float(scene.bodies.velocity.numpy()[scene.cube_body + 1, 2])
            floor_z = scene.ground_height + scene.cube_half_extent

            self.assertTrue(np.isfinite(cube_z), f"cube z went non-finite: {cube_z}")
            self.assertGreater(
                cube_z,
                floor_z - 0.03,
                f"cube fell through ground (z={cube_z:.4f}, floor={floor_z:.4f})",
            )
            self.assertLess(
                abs(cube_vz),
                0.1,
                f"cube did not settle on the ground (vz={cube_vz:.4f})",
            )

            contact_count = int(scene.contacts.rigid_contact_count.numpy()[0])
            shape0 = scene.contacts.rigid_contact_shape0.numpy()[:contact_count]
            shape1 = scene.contacts.rigid_contact_shape1.numpy()[:contact_count]
            self.assertTrue(
                bool(np.any((shape0 == 0) & (shape1 == 1))),
                "expected an active ground-cube contact",
            )

            data_i = scene.world._contact_cols.data.numpy().view(np.int32)
            body1_off = dword_offset_of(ContactConstraintData, "body1")
            body2_off = dword_offset_of(ContactConstraintData, "body2")
            self.assertEqual(int(data_i[body1_off, 0]), 0)
            self.assertEqual(int(data_i[body2_off, 0]), scene.cube_body + 1)


class _GroundCubeClothScene:
    def __init__(self, device: wp.Device, *, cloth_self_collision: bool = True):
        self.device = device
        self.ground_height = 0.0
        self.cube_half_extent = 0.4

        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=self.ground_height)
        self.cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.2), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(
            self.cube_body,
            hx=self.cube_half_extent,
            hy=self.cube_half_extent,
            hz=self.cube_half_extent,
        )
        builder.add_cloth_grid(
            pos=wp.vec3(2.0, -0.4, 8.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.05,
            fix_left=True,
            fix_right=True,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.0,
        )
        self.model = builder.finalize(device=device)

        num_phoenx_bodies = int(self.model.body_count) + 1
        self.bodies = body_container_zeros(num_phoenx_bodies, device=device)
        self.bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=device,
            )
        )
        state_init = self.model.state()
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=int(self.model.body_count),
            inputs=[
                self.model.body_q,
                state_init.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[
                self.bodies.position,
                self.bodies.orientation,
                self.bodies.velocity,
                self.bodies.angular_velocity,
                self.bodies.inverse_mass,
                self.bodies.inverse_inertia,
                self.bodies.inverse_inertia_world,
                self.bodies.motion_type,
                self.bodies.body_com,
            ],
            device=device,
        )

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(self.model.tri_count),
            device=device,
        )
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=int(self.model.tri_count),
            num_worlds=1,
            substeps=4,
            solver_iterations=4,
            rigid_contact_max=1024,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=12,
            device=device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)
        self.pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            cloth_thickness=0.005,
            cloth_gap=0.010,
            cloth_self_collision=cloth_self_collision,
            rigid_contact_max=1024,
        )
        self.contacts = self.pipeline.contacts()
        self.state = self.model.state()

        self._simulate_one_frame()
        with wp.ScopedCapture(device=device) as cap:
            self._simulate_one_frame()
        self.graph = cap.graph

    def _sync_newton_to_phoenx(self) -> None:
        n = int(self.model.body_count)
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

    def _simulate_one_frame(self) -> None:
        self._sync_newton_to_phoenx()
        self.world.collide(self.state, self.contacts)
        self.world.step(1.0 / 60.0, contacts=self.contacts)
        self._sync_phoenx_to_newton()

    def step(self) -> None:
        wp.capture_launch(self.graph)


if __name__ == "__main__":
    unittest.main()
