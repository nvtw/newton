# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dihedral-angle edge filter used by SDF-mesh contact generation.

`Mesh._filter_edges_by_dihedral_angle` drops internal edges whose two adjacent
triangle face normals are within an angle threshold (near-coplanar). Boundary
edges and non-manifold edges are always kept. The filter is applied at
`ModelBuilder.finalize()` time via `ModelBuilder.mesh_edge_lower_angle_threshold_rad`.
"""

import math
import unittest

import numpy as np

import newton


def _flat_quad_mesh() -> newton.Mesh:
    """Two coplanar triangles sharing one internal edge (in the XY plane)."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _single_triangle_mesh() -> newton.Mesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _non_manifold_mesh() -> newton.Mesh:
    """Three triangles sharing the edge (v0, v1)."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


class TestMeshEdgeAngleFilter(unittest.TestCase):
    def test_threshold_zero_returns_full_edges(self):
        mesh = _flat_quad_mesh()
        full = mesh.edges
        filtered = mesh._filter_edges_by_dihedral_angle(0.0)
        np.testing.assert_array_equal(filtered, full)

    def test_negative_threshold_returns_full_edges(self):
        mesh = _flat_quad_mesh()
        full = mesh.edges
        filtered = mesh._filter_edges_by_dihedral_angle(-1.0)
        np.testing.assert_array_equal(filtered, full)

    def test_flat_quad_drops_internal_edge(self):
        mesh = _flat_quad_mesh()
        full = mesh.edges
        # 4 boundary edges + 1 internal coplanar edge.
        self.assertEqual(len(full), 5)

        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(1.0))
        self.assertEqual(len(filtered), 4)

        kept = _edge_set(filtered)
        expected_boundary = {(0, 1), (1, 2), (2, 3), (0, 3)}
        self.assertEqual(kept, expected_boundary)

    def test_cube_drops_face_diagonals(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        full = mesh.edges
        # 12 cube edges + 6 face diagonals = 18 unique geometric edges.
        self.assertEqual(len(full), 18)

        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(1.0))
        # 6 face diagonals are coplanar and should be filtered out.
        self.assertEqual(len(filtered), 12)

        # The 12 silhouette edges all have length equal to one box edge (1.0).
        verts = np.asarray(mesh.vertices)
        for a, b in filtered:
            length = float(np.linalg.norm(verts[a] - verts[b]))
            self.assertAlmostEqual(length, 1.0, places=5)

    def test_open_mesh_keeps_all_boundary_edges(self):
        mesh = _single_triangle_mesh()
        for threshold in (0.0, math.radians(1.0), math.radians(179.0)):
            filtered = mesh._filter_edges_by_dihedral_angle(threshold)
            self.assertEqual(len(filtered), 3, msg=f"threshold={threshold}")

    def test_non_manifold_edge_always_kept(self):
        mesh = _non_manifold_mesh()
        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(179.0))
        kept = _edge_set(filtered)
        self.assertIn((0, 1), kept)

    def test_high_threshold_drops_low_angle_edges(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        # 90 degree dihedral on all silhouette edges; threshold above that drops them too.
        filtered = mesh._filter_edges_by_dihedral_angle(math.radians(91.0))
        self.assertEqual(len(filtered), 0)

    def test_diagnostics_shapes_and_subset(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        edges, angles, normals = mesh._filter_edges_by_dihedral_angle(math.radians(1.0), return_diagnostics=True)
        self.assertEqual(angles.shape, (len(edges),))
        self.assertEqual(normals.shape, (len(edges), 3))
        # Cube silhouette edges are 90 degree dihedrals between two valid triangles.
        np.testing.assert_allclose(angles, math.radians(90.0), atol=1e-5)
        finite = np.isfinite(normals).all(axis=1)
        self.assertTrue(bool(finite.all()))

    def test_diagnostics_nan_for_boundary_edges(self):
        mesh = _single_triangle_mesh()
        edges, angles, normals = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        self.assertEqual(len(edges), 3)
        self.assertTrue(bool(np.all(np.isnan(angles))))
        self.assertTrue(bool(np.all(np.isnan(normals))))

    def test_diagnostics_nan_for_non_manifold_edges(self):
        mesh = _non_manifold_mesh()
        edges, angles, normals = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        # Locate the non-manifold (0, 1) edge in the returned set.
        rows = [tuple(sorted((int(a), int(b)))) for a, b in edges]
        nm = rows.index((0, 1))
        self.assertTrue(math.isnan(float(angles[nm])))
        self.assertTrue(bool(np.all(np.isnan(normals[nm]))))

    def test_diagnostics_flat_quad_zero_angle(self):
        mesh = _flat_quad_mesh()
        edges, angles, normals = mesh._filter_edges_by_dihedral_angle(-1.0, return_diagnostics=True)
        # The internal diagonal (0, 2) is shared by exactly two coplanar triangles
        # whose normals are both +Z, so the dihedral angle is 0 and the average
        # normal is +Z. Boundary edges remain NaN.
        rows = [tuple(sorted((int(a), int(b)))) for a, b in edges]
        diag = rows.index((0, 2))
        self.assertAlmostEqual(float(angles[diag]), 0.0, places=5)
        np.testing.assert_allclose(normals[diag], [0.0, 0.0, 1.0], atol=1e-5)
        boundary_mask = np.array([row != (0, 2) for row in rows])
        self.assertTrue(bool(np.all(np.isnan(angles[boundary_mask]))))

    def test_filter_preserves_edges_subset_and_order(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        full_rows = [tuple(row) for row in mesh.edges.tolist()]
        full_index = {row: i for i, row in enumerate(full_rows)}

        filtered_rows = [tuple(row) for row in mesh._filter_edges_by_dihedral_angle(math.radians(1.0)).tolist()]
        # Subset.
        for row in filtered_rows:
            self.assertIn(row, full_index)
        # First-occurrence order preserved.
        positions = [full_index[row] for row in filtered_rows]
        self.assertEqual(positions, sorted(positions))


class TestModelBuilderEdgeAngleThreshold(unittest.TestCase):
    def test_default_is_point_one_degree(self):
        builder = newton.ModelBuilder()
        self.assertAlmostEqual(builder.mesh_edge_lower_angle_threshold_rad, math.radians(0.1))

    def test_finalize_packs_filtered_edges_per_threshold(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)

        # Threshold 0 -> 18 unique edges packed.
        builder0 = newton.ModelBuilder()
        builder0.mesh_edge_lower_angle_threshold_rad = 0.0
        body0 = builder0.add_body()
        builder0.add_shape_mesh(body=body0, mesh=mesh)
        model0 = builder0.finalize()
        ranges0 = model0.shape_edge_range.numpy()
        # First (and only) mesh shape should have count == 18.
        self.assertEqual(int(ranges0[0][1]), 18)
        self.assertEqual(int(model0.mesh_edge_indices.shape[0]), 18)

        # Default threshold (1 degree) -> 12 silhouette edges packed.
        builder1 = newton.ModelBuilder()
        body1 = builder1.add_body()
        builder1.add_shape_mesh(body=body1, mesh=mesh)
        model1 = builder1.finalize()
        ranges1 = model1.shape_edge_range.numpy()
        self.assertEqual(int(ranges1[0][1]), 12)
        self.assertEqual(int(model1.mesh_edge_indices.shape[0]), 12)

    def test_finalize_shares_edges_across_shapes_referencing_same_mesh(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        builder = newton.ModelBuilder()
        body_a = builder.add_body()
        body_b = builder.add_body()
        builder.add_shape_mesh(body=body_a, mesh=mesh)
        builder.add_shape_mesh(body=body_b, mesh=mesh)
        model = builder.finalize()

        ranges = model.shape_edge_range.numpy()
        # Two mesh shapes, both referencing the same Mesh -> identical (start, count) slice.
        mesh_ranges = [tuple(int(x) for x in r) for r in ranges if int(r[1]) > 0]
        self.assertEqual(len(mesh_ranges), 2)
        self.assertEqual(mesh_ranges[0], mesh_ranges[1])
        # Packed array stores only one copy.
        self.assertEqual(int(model.mesh_edge_indices.shape[0]), mesh_ranges[0][1])


if __name__ == "__main__":
    unittest.main()
