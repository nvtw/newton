# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph regressions for convex contact-patch state."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.types import GeoType
from newton._src.solvers.phoenx.constraints.contact_ingest import IngestScratch
from newton._src.solvers.phoenx.constraints.contact_patch_friction import (
    classify_contact_patch_columns,
    contact_patch_friction_zeros,
    contact_patch_project_velocity_update,
    copy_contact_patch_impulses,
    gather_contact_patch_warmstart,
)
from newton._src.solvers.phoenx.materials import COMBINE_AVERAGE, Material, material_table_from_list
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture
from newton._src.solvers.phoenx.tests.test_determinism import _build_mesh_sdf_pyramid_scene
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


@wp.kernel(enable_backward=False)
def _project_patch_kernel(
    lambda_old: wp.array[wp.vec2],
    relative_velocity: wp.array[wp.vec2],
    effective_mass: wp.array[wp.mat22],
    normal_load: wp.array[wp.float32],
    friction_static: wp.float32,
    friction_kinetic: wp.float32,
    output_lambda: wp.array[wp.vec2],
    output_delta: wp.array[wp.vec2],
):
    update = contact_patch_project_velocity_update(
        lambda_old[0],
        relative_velocity[0],
        wp.vec2f(0.0, 0.0),
        effective_mass[0],
        normal_load[0],
        friction_static,
        friction_kinetic,
        wp.float32(1.0),
    )
    output_lambda[0] = update.lambda_new
    output_delta[0] = update.delta


class TestContactPatchFriction(unittest.TestCase):
    """Patch state is conservative, persistent, and capture-safe."""

    def test_convex_only_eligibility_under_graph_capture(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-patch tests")
        scratch = IngestScratch(8, 4, device=device)
        scratch.pair_source_idx.assign(np.arange(4, dtype=np.int32))
        scratch.pair_shape_a.assign(np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32))
        scratch.pair_shape_b.assign(np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int32))
        scratch.num_contact_columns.assign(np.array([4], dtype=np.int32))
        shape_type = wp.array(
            [GeoType.BOX, GeoType.PLANE, GeoType.CONVEX_MESH, GeoType.MESH, GeoType.HFIELD],
            dtype=wp.int32,
            device=device,
        )
        patch = contact_patch_friction_zeros(4, device=device)

        with wp.ScopedCapture(device=device) as capture:
            classify_contact_patch_columns(
                patch,
                scratch,
                shape_type,
                enable_body_pair_grouping=False,
                device=device,
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(patch.eligible.numpy(), np.array([1, 1, 0, 0], dtype=np.int32))

    def test_compound_columns_conservatively_use_point_friction(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-patch tests")
        scratch = IngestScratch(2, 1, device=device, enable_body_pair_grouping=True)
        scratch.pair_source_idx.assign(np.array([0], dtype=np.int32))
        scratch.pair_shape_a.assign(np.array([0, 0], dtype=np.int32))
        scratch.pair_shape_b.assign(np.array([1, 0], dtype=np.int32))
        scratch.num_contact_columns.assign(np.array([1], dtype=np.int32))
        shape_type = wp.array([GeoType.BOX, GeoType.BOX], dtype=wp.int32, device=device)
        patch = contact_patch_friction_zeros(1, device=device)
        patch.eligible.assign(np.array([7], dtype=np.int32))

        with wp.ScopedCapture(device=device) as capture:
            classify_contact_patch_columns(
                patch,
                scratch,
                shape_type,
                enable_body_pair_grouping=True,
                device=device,
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(patch.eligible.numpy(), np.zeros(1, dtype=np.int32))

    def test_impulse_history_copy_is_scoped_to_live_columns(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-patch tests")
        patch = contact_patch_friction_zeros(5, device=device)
        current = np.arange(15, dtype=np.float32).reshape(5, 3)
        patch.impulse_world.assign(current)
        patch.prev_impulse_world.assign(np.full((5, 3), -7.0, dtype=np.float32))
        live_count = wp.array([3], dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            copy_contact_patch_impulses(patch, live_count, device=device)
        wp.capture_launch(capture.graph)

        previous = patch.prev_impulse_world.numpy()
        np.testing.assert_array_equal(previous[:3], current[:3])
        np.testing.assert_array_equal(previous[3:], np.full((2, 3), -7.0, dtype=np.float32))

    def test_warmstart_follows_matches_when_columns_reorder(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-patch tests")
        scratch = IngestScratch(4, 2, device=device)
        scratch.pair_source_idx.assign(np.array([0, 1], dtype=np.int32))
        scratch.pair_first.assign(np.array([0, 1, 0, 0], dtype=np.int32))
        scratch.pair_count.assign(np.array([1, 2, 0, 0], dtype=np.int32))
        scratch.num_contact_columns.assign(np.array([2], dtype=np.int32))
        matches = wp.array([1, -1, 0, -1], dtype=wp.int32, device=device)
        previous_contact_to_cid = wp.array([10, 11, -1, -1], dtype=wp.int32, device=device)
        reuse = wp.array([1], dtype=wp.int32, device=device)
        patch = contact_patch_friction_zeros(2, device=device)
        patch.prev_impulse_world.assign(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32))

        with wp.ScopedCapture(device=device) as capture:
            gather_contact_patch_warmstart(
                patch,
                scratch,
                matches,
                previous_contact_to_cid,
                reuse,
                10,
                wp.array([GeoType.BOX, GeoType.BOX], dtype=wp.int32, device=device),
                enable_body_pair_grouping=False,
                device=device,
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(
            patch.impulse_world.numpy(),
            np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_patch_box_rests_on_plane_under_graph_capture(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")
        scene = _PhoenXScene(
            substeps=10,
            solver_iterations=3,
            velocity_iterations=1,
            contact_friction_model="patch",
        )
        scene.add_ground_plane()
        body = scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        scene.set_body_velocity(body, (1.0, 0.0, 0.0))
        scene.step()
        self.assertGreater(int(scene.world._contact_cols.patch.eligible.numpy().max()), 0)
        self.assertGreater(
            float(np.linalg.norm(scene.world._contact_cols.patch.impulse_world.numpy(), axis=1).max()), 0.0
        )
        for _ in range(119):
            scene.step()

        self.assertAlmostEqual(float(scene.body_position(body)[2]), 0.5, delta=0.03)
        self.assertLess(abs(float(scene.body_velocity(body)[2])), 0.05)
        self.assertLess(abs(float(scene.body_velocity(body)[0])), 0.05)

    def test_raw_mesh_contacts_retain_point_friction(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")
        scene = _build_mesh_sdf_pyramid_scene(2, contact_friction_model="patch")
        for _ in range(30):
            scene.step()

        self.assertEqual(int(scene.world._contact_cols.patch.eligible.numpy().max()), 0)
        self.assertTrue(np.isfinite(scene.world.bodies.position.numpy()).all())
        self.assertTrue(np.isfinite(scene.world.bodies.velocity.numpy()).all())

    def test_patch_solver_is_bit_deterministic(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")

        def build_scene() -> _PhoenXScene:
            scene = _PhoenXScene(
                substeps=4,
                solver_iterations=3,
                contact_friction_model="patch",
            )
            scene.add_ground_plane()
            scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
            scene.add_box(position=(0.0, 0.0, 1.5), half_extents=(0.5, 0.5, 0.5))
            scene.finalize()
            return scene

        scene0 = build_scene()
        scene1 = build_scene()
        for _ in range(30):
            scene0.step()
            scene1.step()

        np.testing.assert_array_equal(scene0.world.bodies.position.numpy(), scene1.world.bodies.position.numpy())
        np.testing.assert_array_equal(scene0.world.bodies.orientation.numpy(), scene1.world.bodies.orientation.numpy())
        np.testing.assert_array_equal(scene0.world.bodies.velocity.numpy(), scene1.world.bodies.velocity.numpy())
        np.testing.assert_array_equal(
            scene0.world.bodies.angular_velocity.numpy(), scene1.world.bodies.angular_velocity.numpy()
        )

    def test_patch_static_friction_holds_above_kinetic_budget(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")
        scene = _PhoenXScene(
            substeps=4,
            solver_iterations=8,
            friction=0.0,
            contact_friction_model="patch",
        )
        scene.add_ground_plane()
        half_extent = 0.5
        body = scene.add_box(
            position=(0.0, 0.0, half_extent + 1.0e-3),
            half_extents=(half_extent, half_extent, half_extent),
            density=1000.0,
        )
        scene.finalize()
        material = Material(
            static_friction=0.8,
            dynamic_friction=0.2,
            friction_combine_mode=COMBINE_AVERAGE,
        )
        scene.install_materials(
            material_table_from_list([Material(), material], device=scene.device),
            wp.array([1, 1], dtype=wp.int32, device=scene.device),
        )
        for _ in range(10):
            scene.step()
        mass = 1000.0 * (2.0 * half_extent) ** 3
        for _ in range(120):
            scene.apply_body_force(body, force=(0.4 * mass * 9.81, 0.0, 0.0))
            scene.step()

        self.assertLess(abs(float(scene.body_position(body)[0])), 0.05)

    def test_patch_pair_conserves_linear_and_angular_momentum(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")
        scene = _PhoenXScene(
            fps=1000,
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
            contact_friction_model="patch",
        )
        body0 = scene.add_box(position=(-0.5, 0.0, 1.0), half_extents=(0.5, 0.5, 0.5), mass=2.0)
        body1 = scene.add_box(position=(0.5, 0.0, 1.0), half_extents=(0.5, 0.5, 0.5), mass=2.0)
        scene.finalize()
        scene.set_body_velocity(body0, (1.0, 1.0, 0.0))
        scene.set_body_velocity(body1, (-1.0, -1.0, 0.0))
        scene._sync_newton_to_phoenx()

        def momentum_xy_lz(position_override: np.ndarray | None = None) -> tuple[np.ndarray, float]:
            indices = np.array([body0 + 1, body1 + 1], dtype=np.int32)
            position = scene.bodies.position.numpy()[indices] if position_override is None else position_override
            velocity = scene.bodies.velocity.numpy()[indices]
            angular_velocity = scene.bodies.angular_velocity.numpy()[indices]
            inverse_mass = scene.bodies.inverse_mass.numpy()[indices]
            mass = 1.0 / inverse_mass
            linear = (mass[:, None] * velocity).sum(axis=0)
            cube_inertia = np.array(
                [scene.model.body_inertia.numpy()[body0, 2, 2], scene.model.body_inertia.numpy()[body1, 2, 2]]
            )
            angular_z = np.sum(
                mass * (position[:, 0] * velocity[:, 1] - position[:, 1] * velocity[:, 0])
                + cube_inertia * angular_velocity[:, 2]
            )
            return linear[:2], float(angular_z)

        reference_position = scene.bodies.position.numpy()[[body0 + 1, body1 + 1]].copy()
        linear_before, angular_before = momentum_xy_lz(reference_position)
        scene.step()
        linear_after, angular_after = momentum_xy_lz(reference_position)

        self.assertGreater(float(np.linalg.norm(scene.world._contact_cols.patch.impulse_world.numpy())), 0.0)
        np.testing.assert_allclose(linear_after, linear_before, atol=2.0e-5)
        self.assertAlmostEqual(angular_after, angular_before, delta=2.0e-5)

    def test_patch_block_world_runs_under_graph_capture(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")
        scene = _PhoenXScene(
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            contact_friction_model="patch",
        )
        scene.add_ground_plane()
        body = scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        scene.world._configure_multi_world_scheduler("block_world_32")
        self.assertEqual(scene.world._multi_world_scheduler, "block_world")
        scene.set_body_velocity(body, (0.5, 0.0, 0.0))
        for _ in range(30):
            scene.step()

        self.assertIsNotNone(scene._graph)
        self.assertGreater(int(scene.world._contact_cols.patch.eligible.numpy().max()), 0)
        self.assertTrue(np.isfinite(scene.body_position(body)).all())
        self.assertTrue(np.isfinite(scene.body_velocity(body)).all())

    def test_patch_single_world_runs_under_graph_capture(self) -> None:
        require_cuda_graph_capture("PhoenX contact-patch tests")
        scene = _PhoenXScene(
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            step_layout="single_world",
            contact_friction_model="patch",
        )
        scene.add_ground_plane()
        body = scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        scene.set_body_velocity(body, (0.5, 0.0, 0.0))
        for _ in range(30):
            scene.step()

        self.assertGreater(int(scene.world._contact_cols.patch.eligible.numpy().max()), 0)
        self.assertTrue(np.isfinite(scene.body_position(body)).all())
        self.assertTrue(np.isfinite(scene.body_velocity(body)).all())

    def test_coupled_projection_uses_full_effective_mass(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-patch tests")
        lambda_old = wp.array([wp.vec2(0.1, -0.2)], dtype=wp.vec2, device=device)
        relative_velocity = wp.array([wp.vec2(1.0, -2.0)], dtype=wp.vec2, device=device)
        effective_mass = wp.array([wp.mat22(2.0, 0.5, 0.5, 1.0)], dtype=wp.mat22, device=device)
        normal_load = wp.array([100.0], dtype=wp.float32, device=device)
        output_lambda = wp.zeros(1, dtype=wp.vec2, device=device)
        output_delta = wp.zeros(1, dtype=wp.vec2, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _project_patch_kernel,
                dim=1,
                inputs=[lambda_old, relative_velocity, effective_mass, normal_load, 1.0, 1.0],
                outputs=[output_lambda, output_delta],
                device=device,
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(output_delta.numpy()[0], [-1.0, 1.5], atol=1.0e-6)
        np.testing.assert_allclose(output_lambda.numpy()[0], [-0.9, 1.3], atol=1.0e-6)

    def test_sliding_projection_uses_kinetic_disk(self) -> None:
        device = require_cuda_graph_capture("PhoenX contact-patch tests")
        lambda_old = wp.zeros(1, dtype=wp.vec2, device=device)
        relative_velocity = wp.array([wp.vec2(3.0, 4.0)], dtype=wp.vec2, device=device)
        effective_mass = wp.array([wp.mat22(1.0, 0.0, 0.0, 1.0)], dtype=wp.mat22, device=device)
        normal_load = wp.array([2.0], dtype=wp.float32, device=device)
        output_lambda = wp.zeros(1, dtype=wp.vec2, device=device)
        output_delta = wp.zeros(1, dtype=wp.vec2, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _project_patch_kernel,
                dim=1,
                inputs=[lambda_old, relative_velocity, effective_mass, normal_load, 0.8, 0.5],
                outputs=[output_lambda, output_delta],
                device=device,
            )
        wp.capture_launch(capture.graph)

        actual = output_lambda.numpy()[0]
        np.testing.assert_allclose(actual, [-0.6, -0.8], atol=1.0e-6)
        self.assertAlmostEqual(float(np.linalg.norm(actual)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
