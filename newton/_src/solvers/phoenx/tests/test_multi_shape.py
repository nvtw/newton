# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for multi-shape rigid bodies in PhoenX.

Covers three pieces of the stack:

1. **Mass / inertia accumulation** in :class:`WorldBuilder`: spheres,
   boxes, capsules, and compound assemblies match closed-form
   textbook formulas (parallel-axis theorem for compound bodies).
2. **Ambiguity rejection**: combining an explicit body-level
   ``inverse_mass`` with shape-provided density / mass raises a
   ``ValueError`` at finalize.
3. **Contact-ingest self-filter**: two shapes on the same dynamic
   body never produce a constraint column; the graph coloring is
   never asked to handle a self-edge.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_ingest import (
    _pair_columns_binary_kernel,
)
from newton._src.solvers.phoenx.world_builder import (
    ShapeType,
    WorldBuilder,
)


class TestCompoundMass(unittest.TestCase):
    """Mass / inertia accumulation reproduces textbook values."""

    def test_sphere_mass_and_inertia(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body(position=(0, 0, 0))
        b.add_shape_sphere(body, radius=0.5, density=1000.0)
        b._accumulate_mass_inertia_from_shapes()
        desc = b._bodies[body]

        expected_mass = 1000.0 * (4.0 / 3.0) * math.pi * 0.5**3
        expected_i = 0.4 * expected_mass * 0.5**2
        self.assertAlmostEqual(desc.inverse_mass, 1.0 / expected_mass, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[0][0], 1.0 / expected_i, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[1][1], 1.0 / expected_i, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[2][2], 1.0 / expected_i, places=6)

    def test_box_mass_and_inertia(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body(position=(0, 0, 0))
        b.add_shape_box(body, half_extents=(0.5, 0.5, 0.5), density=1000.0)
        b._accumulate_mass_inertia_from_shapes()
        desc = b._bodies[body]

        expected_mass = 1000.0  # 1 m^3 cube
        expected_i = expected_mass / 6.0  # I_cube = m * (h^2 + h^2)/3 with h=0.5
        self.assertAlmostEqual(desc.inverse_mass, 1.0 / expected_mass, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[0][0], 1.0 / expected_i, places=6)

    def test_explicit_mass_overrides_density(self) -> None:
        """Passing ``mass`` on a shape instead of ``density`` must
        produce exactly that total mass."""
        b = WorldBuilder()
        body = b.add_dynamic_body()
        b.add_shape_sphere(body, radius=0.5, mass=5.0)
        b._accumulate_mass_inertia_from_shapes()
        self.assertAlmostEqual(b._bodies[body].inverse_mass, 1.0 / 5.0, places=6)

    def test_compound_dumbbell_parallel_axis(self) -> None:
        """Two spheres offset along ``+x`` and ``-x`` at distance 1 m:
        ``I_xx`` is just the sum of their own inertias (offset is
        along x), ``I_yy = I_zz`` additionally pick up ``2 * m *
        d^2`` from the parallel-axis theorem.
        """
        b = WorldBuilder()
        body = b.add_dynamic_body()
        b.add_shape_sphere(body, radius=0.5, local_pos=(+1.0, 0, 0), density=1000.0)
        b.add_shape_sphere(body, radius=0.5, local_pos=(-1.0, 0, 0), density=1000.0)
        b._accumulate_mass_inertia_from_shapes()
        desc = b._bodies[body]

        m_sph = 1000.0 * (4.0 / 3.0) * math.pi * 0.5**3
        i_sph = 0.4 * m_sph * 0.5**2
        expected_mass = 2.0 * m_sph
        expected_ixx = 2.0 * i_sph
        expected_iyy = 2.0 * (i_sph + m_sph * 1.0 * 1.0)
        self.assertAlmostEqual(desc.inverse_mass, 1.0 / expected_mass, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[0][0], 1.0 / expected_ixx, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[1][1], 1.0 / expected_iyy, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[2][2], 1.0 / expected_iyy, places=6)

    def test_collision_only_shapes_dont_change_mass(self) -> None:
        """Shapes without ``density`` / ``mass`` are pure collision
        geometry and leave the body's descriptor-set inverse_mass
        alone."""
        b = WorldBuilder()
        body = b.add_dynamic_body(inverse_mass=2.0, inverse_inertia=((3, 0, 0), (0, 3, 0), (0, 0, 3)))
        b.add_shape_sphere(body, radius=0.5)  # no density, no mass
        b._accumulate_mass_inertia_from_shapes()
        desc = b._bodies[body]
        self.assertAlmostEqual(desc.inverse_mass, 2.0, places=6)
        self.assertAlmostEqual(desc.inverse_inertia[0][0], 3.0, places=6)


class TestMassAmbiguityRejection(unittest.TestCase):
    """Users must declare mass source *once* -- either on the body or
    on its shapes -- not both. Mixing raises at finalize."""

    def test_rejects_explicit_mass_with_density_shape(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body(inverse_mass=0.5)
        b.add_shape_sphere(body, radius=0.5, density=1000.0)
        with self.assertRaisesRegex(ValueError, "declared both on the body"):
            b._accumulate_mass_inertia_from_shapes()

    def test_rejects_explicit_inertia_with_density_shape(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body(inverse_inertia=((2, 0, 0), (0, 2, 0), (0, 0, 2)))
        b.add_shape_sphere(body, radius=0.5, density=1000.0)
        with self.assertRaisesRegex(ValueError, "declared both on the body"):
            b._accumulate_mass_inertia_from_shapes()

    def test_rejects_density_shape_on_static_body(self) -> None:
        b = WorldBuilder()
        ground = b.add_static_body()
        with self.assertRaisesRegex(ValueError, "mass-providing shapes are only meaningful"):
            b.add_shape_box(ground, half_extents=(10, 0.1, 10), density=1000.0)
            b._accumulate_mass_inertia_from_shapes()

    def test_rejects_both_density_and_mass_on_one_shape(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body()
        with self.assertRaisesRegex(ValueError, "set exactly one"):
            b.add_shape_sphere(body, radius=0.5, density=1000.0, mass=5.0)

    def test_rejects_plane_on_dynamic_body(self) -> None:
        b = WorldBuilder()
        dynamic = b.add_dynamic_body()
        with self.assertRaisesRegex(ValueError, "may only be attached to static bodies"):
            b.add_shape_plane(dynamic)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX contact ingest runs on CUDA only.",
)
class TestSelfContactFilter(unittest.TestCase):
    """The contact-ingest kernel must drop any pair whose two shapes
    resolve to the same body id. Verified via a scene with two
    overlapping shapes on a single dynamic body; after stepping, no
    contact column should fire for that body pair."""

    def test_overlapping_shapes_on_same_body_produce_no_column(self) -> None:
        # We don't run the full Newton collision pipeline here; instead
        # we synthesise a minimal rigid-contacts buffer with one
        # self-pair and verify ingest drops it. The contact-ingest
        # code is Warp-kernel-driven and exercised end-to-end via
        # Newton-Model-integrated tests; this test isolates the
        # self-filter branch.
        device = wp.get_preferred_device()

        # Shape 0 and shape 1 both belong to body 1 (body 0 is the
        # static anchor). Shape 2 is on body 2.
        shape_body_np = np.array([0, 1, 1, 2], dtype=np.int32)
        shape_body = wp.array(shape_body_np, dtype=wp.int32, device=device)

        # Two pairs: (1, 2) shape->(body 1, body 1) is self-contact;
        # (1, 3) shape->(body 1, body 2) is legitimate.
        pair_shape_a = wp.array([1, 1], dtype=wp.int32, device=device)
        pair_shape_b = wp.array([2, 3], dtype=wp.int32, device=device)
        pair_count = wp.array([5, 3], dtype=wp.int32, device=device)
        num_pairs = wp.array([2], dtype=wp.int32, device=device)
        filter_keys = wp.array([0], dtype=wp.int64, device=device)
        pair_columns = wp.zeros(2, dtype=wp.int32, device=device)

        wp.launch(
            _pair_columns_binary_kernel,
            dim=2,
            inputs=[
                pair_count,
                pair_shape_a,
                pair_shape_b,
                num_pairs,
                shape_body,
                wp.int32(4),
                filter_keys,
                wp.int32(0),
                pair_columns,
            ],
            device=device,
        )
        wp.synchronize_device()

        cols = pair_columns.numpy()
        # Pair 0 (self-contact on body 1) must be dropped; pair 1
        # (body 1 vs body 2) must survive as a single column.
        self.assertEqual(int(cols[0]), 0, msg=f"self-contact not filtered: cols={cols}")
        self.assertEqual(int(cols[1]), 1, msg=f"valid pair dropped: cols={cols}")

    def test_two_statics_both_slot_zero_filtered(self) -> None:
        """Two shapes attached to the world anchor (slot 0) also
        resolve to the same body and must be filtered -- otherwise
        the ground-vs-ground pair would produce a dummy column with
        a singular effective mass."""
        device = wp.get_preferred_device()
        shape_body = wp.array([0, 0, 1], dtype=wp.int32, device=device)
        pair_shape_a = wp.array([0], dtype=wp.int32, device=device)
        pair_shape_b = wp.array([1], dtype=wp.int32, device=device)
        pair_count = wp.array([2], dtype=wp.int32, device=device)
        num_pairs = wp.array([1], dtype=wp.int32, device=device)
        filter_keys = wp.array([0], dtype=wp.int64, device=device)
        pair_columns = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch(
            _pair_columns_binary_kernel,
            dim=1,
            inputs=[
                pair_count,
                pair_shape_a,
                pair_shape_b,
                num_pairs,
                shape_body,
                wp.int32(3),
                filter_keys,
                wp.int32(0),
                pair_columns,
            ],
            device=device,
        )
        wp.synchronize_device()
        self.assertEqual(int(pair_columns.numpy()[0]), 0)


class TestShapeApiBasics(unittest.TestCase):
    """Error paths and the ``ShapeType`` enum wiring."""

    def test_shape_types_defined(self) -> None:
        self.assertIn("SPHERE", ShapeType.__members__)
        self.assertIn("BOX", ShapeType.__members__)
        self.assertIn("CAPSULE", ShapeType.__members__)
        self.assertIn("PLANE", ShapeType.__members__)

    def test_invalid_sphere_radius(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body()
        with self.assertRaises(ValueError):
            b.add_shape_sphere(body, radius=0.0, density=1.0)
        with self.assertRaises(ValueError):
            b.add_shape_sphere(body, radius=-1.0, density=1.0)

    def test_invalid_box_half_extents(self) -> None:
        b = WorldBuilder()
        body = b.add_dynamic_body()
        with self.assertRaises(ValueError):
            b.add_shape_box(body, half_extents=(0.5, -0.1, 0.5), density=1.0)

    def test_capsule_zero_height_is_sphere(self) -> None:
        """A zero-height capsule collapses to a sphere; mass/I should
        equal a bare sphere of the same radius and density."""
        b = WorldBuilder()
        body = b.add_dynamic_body()
        b.add_shape_capsule(body, radius=0.5, half_height=0.0, density=1000.0)
        b._accumulate_mass_inertia_from_shapes()
        cap_inv_mass = b._bodies[body].inverse_mass

        b2 = WorldBuilder()
        body2 = b2.add_dynamic_body()
        b2.add_shape_sphere(body2, radius=0.5, density=1000.0)
        b2._accumulate_mass_inertia_from_shapes()
        sph_inv_mass = b2._bodies[body2].inverse_mass

        self.assertAlmostEqual(cap_inv_mass, sph_inv_mass, places=5)

    def test_finalize_installs_shape_body(self) -> None:
        """After finalize, the :class:`PhoenXWorld` carries an
        internally-stored ``shape_body`` so ``step(contacts=...)``
        doesn't need the caller to thread it through."""
        device = wp.get_preferred_device() if wp.is_cuda_available() else "cpu"
        b = WorldBuilder()
        ground = b.add_static_body()
        ball = b.add_dynamic_body(position=(0, 0, 1))
        b.add_shape_plane(ground)
        b.add_shape_sphere(ball, radius=0.25, density=1000.0)
        world = b.finalize(substeps=1, solver_iterations=4, gravity=(0, 0, -9.81), device=device)

        shape_body = world._shape_body_internal
        self.assertIsNotNone(shape_body)
        sb = shape_body.numpy()
        self.assertEqual(int(sb[0]), ground)
        self.assertEqual(int(sb[1]), ball)


if __name__ == "__main__":
    wp.init()
    unittest.main()
