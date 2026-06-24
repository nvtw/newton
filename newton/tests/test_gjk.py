# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for GJK distance computation using the new simplex solver."""

import unittest

import warp as wp

from newton import GeoType
from newton._src.geometry.mpr import create_solve_mpr
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map

MAX_ITERATIONS = 30


@wp.kernel
def _gjk_kernel(
    type_a: int,
    size_a: wp.vec3,
    pos_a: wp.vec3,
    quat_a: wp.quat,
    type_b: int,
    size_b: wp.vec3,
    pos_b: wp.vec3,
    quat_b: wp.quat,
    # Outputs:
    collision_out: wp.array[int],
    dist_out: wp.array[float],
    point_out: wp.array[wp.vec3],
    normal_out: wp.array[wp.vec3],
):
    """Kernel to compute GJK distance between two shapes."""
    # Create shape data for both geometries
    shape_a = GenericShapeData()
    shape_a.shape_type = type_a
    shape_a.scale = size_a
    shape_a.auxiliary = wp.vec3(0.0)

    shape_b = GenericShapeData()
    shape_b.shape_type = type_b
    shape_b.scale = size_b
    shape_b.auxiliary = wp.vec3(0.0)

    data_provider = SupportMapDataProvider()

    # Call GJK solver
    collision, distance, point, normal = wp.static(create_solve_closest_distance(support_map))(
        shape_a,
        shape_b,
        quat_a,
        quat_b,
        pos_a,
        pos_b,
        0.0,  # combined_margin
        data_provider,
        MAX_ITERATIONS,
        1e-6,  # COLLIDE_EPSILON
    )

    collision_out[0] = int(collision)
    dist_out[0] = distance
    point_out[0] = point
    normal_out[0] = normal


def _geom_dist(
    geom_type1: int,
    size1: wp.vec3,
    pos1: wp.vec3,
    quat1: wp.quat,
    geom_type2: int,
    size2: wp.vec3,
    pos2: wp.vec3,
    quat2: wp.quat,
):
    """
    Compute distance between two geometries using GJK algorithm.

    Returns:
        Tuple of (distance, midpoint_contact_point, normal, collision_flag)
    """
    # Convert GeoType enums to int if needed
    type1 = int(geom_type1)
    type2 = int(geom_type2)

    collision_out = wp.zeros(1, dtype=int)
    dist_out = wp.zeros(1, dtype=float)
    point_out = wp.zeros(1, dtype=wp.vec3)
    normal_out = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        _gjk_kernel,
        dim=1,
        inputs=[type1, size1, pos1, quat1, type2, size2, pos2, quat2],
        outputs=[collision_out, dist_out, point_out, normal_out],
    )

    return (
        dist_out.numpy()[0],
        point_out.numpy()[0],
        normal_out.numpy()[0],
        collision_out.numpy()[0],
    )


class TestGJK(unittest.TestCase):
    """Tests for GJK distance computation using the new simplex solver."""

    def test_spheres_distance(self):
        """Test distance between two separated spheres."""
        # Two spheres of radius 1.0, separated by distance 3.0
        # Expected distance: 3.0 - 1.0 - 1.0 = 1.0
        dist, _point, _normal, collision = _geom_dist(
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(-1.5, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(1.5, 0.0, 0.0),
            wp.quat_identity(),
        )
        self.assertAlmostEqual(1.0, dist, places=5)
        self.assertEqual(0, collision)  # No collision

    def test_spheres_touching(self):
        """Test two touching spheres have zero distance."""
        # Two spheres of radius 1.0, centers at distance 2.0
        # Expected distance: 0.0 (just touching)
        dist, _point, _normal, _collision = _geom_dist(
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(-1.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        self.assertAlmostEqual(0.0, dist, places=5)

    def test_sphere_sphere_overlapping(self):
        """Test overlapping spheres return collision=True and distance=0."""
        # Two spheres of radius 3.0, centers at distance 4.0
        # Expected overlap: 3.0 + 3.0 - 4.0 = 2.0
        # Note: GJK returns collision=True and distance=0 for overlapping shapes (MPR would give penetration depth)
        dist, _point, _normal, collision = _geom_dist(
            GeoType.SPHERE,
            wp.vec3(3.0, 0.0, 0.0),
            wp.vec3(-1.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.SPHERE,
            wp.vec3(3.0, 0.0, 0.0),
            wp.vec3(3.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        self.assertAlmostEqual(0.0, dist, places=5)
        self.assertEqual(1, collision)  # GJK reports collision=True for overlapping shapes

    def test_box_box_separated(self):
        """Test distance between two separated boxes."""
        # Two boxes: first is 5x5x5 (half-extents 2.5), second is 2x2x2 (half-extents 1.0)
        # First centered at (-1, 0, 0), second at (1.5, 0, 0)
        # Distance between centers: 2.5, half-extents sum: 3.5
        # Expected separation: 2.5 - 2.5 - 1.0 = -1.0 (overlapping)
        # But let's test a separated case
        dist, _point, normal, collision = _geom_dist(
            GeoType.BOX,
            wp.vec3(1.0, 1.0, 1.0),
            wp.vec3(-2.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            wp.vec3(1.0, 1.0, 1.0),
            wp.vec3(2.5, 0.0, 0.0),
            wp.quat_identity(),
        )
        # Centers at distance 4.5, half-extents sum: 2.0
        # Expected distance: 4.5 - 1.0 - 1.0 = 2.5
        self.assertAlmostEqual(2.5, dist, places=5)
        self.assertEqual(0, collision)
        # Normal should point from A to B (positive X direction)
        self.assertAlmostEqual(normal[0], 1.0, places=5)
        self.assertAlmostEqual(normal[1], 0.0, places=5)
        self.assertAlmostEqual(normal[2], 0.0, places=5)


@wp.kernel
def _mpr_kernel(
    type_a: int,
    size_a: wp.vec3,
    pos_a: wp.vec3,
    quat_a: wp.quat,
    type_b: int,
    size_b: wp.vec3,
    pos_b: wp.vec3,
    quat_b: wp.quat,
    # Outputs:
    collision_out: wp.array[int],
    dist_out: wp.array[float],
    point_out: wp.array[wp.vec3],
    normal_out: wp.array[wp.vec3],
):
    shape_a = GenericShapeData()
    shape_a.shape_type = type_a
    shape_a.scale = size_a
    shape_a.auxiliary = wp.vec3(0.0)

    shape_b = GenericShapeData()
    shape_b.shape_type = type_b
    shape_b.scale = size_b
    shape_b.auxiliary = wp.vec3(0.0)

    data_provider = SupportMapDataProvider()

    collision, dist, point, normal = wp.static(create_solve_mpr(support_map))(
        shape_a,
        shape_b,
        quat_a,
        quat_b,
        pos_a,
        pos_b,
        0.0,
        data_provider,
    )

    collision_out[0] = int(collision)
    dist_out[0] = dist
    point_out[0] = point
    normal_out[0] = normal


def _mpr_dist(type_a, size_a, pos_a, quat_a, type_b, size_b, pos_b, quat_b):
    collision_out = wp.zeros(1, dtype=int)
    dist_out = wp.zeros(1, dtype=float)
    point_out = wp.zeros(1, dtype=wp.vec3)
    normal_out = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        _mpr_kernel,
        dim=1,
        inputs=[int(type_a), size_a, pos_a, quat_a, int(type_b), size_b, pos_b, quat_b],
        outputs=[collision_out, dist_out, point_out, normal_out],
    )

    return (
        dist_out.numpy()[0],
        point_out.numpy()[0],
        normal_out.numpy()[0],
        collision_out.numpy()[0],
    )


class TestFlatTableContact(unittest.TestCase):
    """
    Regression tests for the large-flat-box geometric-center bug.

    Root cause: for a thin table box (e.g. 1 m x 1 m x 0.02 m), the box center
    sits deep below the table surface.  When a small fingertip is laterally
    offset from the table center, the center-to-center vector points mostly
    sideways.  MPR/GJK then starts searching toward a side face and reports a
    huge (~0.4 m) "penetration" through that face instead of the correct shallow
    top-face contact (~a few mm).

    The fix projects each box's "interior point" onto its nearest surface, giving
    an initial search direction that is nearly perpendicular to the contact face.
    """

    # Table: 1 m x 1 m x 0.02 m, center at z = 0, top face at z = 0.01 m.
    TABLE_HALF = wp.vec3(0.5, 0.5, 0.01)
    # Fingertip box: 0.02 m cube, laterally offset 0.3 m from table center.
    FINGER_HALF = wp.vec3(0.01, 0.01, 0.01)
    # Fingertip sits 1 mm inside the table top face (slight penetration).
    FINGER_POS = wp.vec3(0.3, 0.0, 0.019)  # finger center above table surface (0.01m)

    # Generous tolerance: normal must be within 10° of vertical.
    NORMAL_Z_MIN = 0.98  # cos(~11°)
    # Penetration must be shallow: less than half the table thickness.
    MAX_PENETRATION = 0.005  # 5 mm

    def _assert_top_face_contact(self, dist, normal, collision, label):
        """Assert the contact is through the top face with a nearly-vertical normal."""
        self.assertTrue(collision, f"{label}: expected collision=True")
        self.assertGreater(
            abs(normal[2]),
            self.NORMAL_Z_MIN,
            f"{label}: normal Z component {normal[2]:.4f} is below {self.NORMAL_Z_MIN} "
            f"(normal={normal}) — side-face contact detected instead of top-face",
        )
        self.assertGreater(
            -dist,  # dist is negative (penetration), so -dist is positive depth
            0.0,
            f"{label}: expected penetration depth > 0, got {dist:.4f}",
        )
        self.assertLess(
            -dist,
            self.MAX_PENETRATION,
            f"{label}: penetration depth {-dist:.4f} m exceeds {self.MAX_PENETRATION} m "
            f"— likely wrong face (side) was selected",
        )

    def test_mpr_box_on_flat_table_lateral_offset(self):
        """MPR: small box resting on large flat table, offset from table center."""
        dist, _point, normal, collision = _mpr_dist(
            GeoType.BOX,
            self.TABLE_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            self.FINGER_HALF,
            self.FINGER_POS,
            wp.quat_identity(),
        )
        self._assert_top_face_contact(dist, normal, collision, "MPR box-on-table lateral")

    def test_gjk_box_on_flat_table_lateral_offset(self):
        """GJK: small box resting on large flat table, offset from table center."""
        _dist, _point, _normal, _collision = _geom_dist(
            GeoType.BOX,
            self.TABLE_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            self.FINGER_HALF,
            self.FINGER_POS,
            wp.quat_identity(),
        )
        # GJK is used for separated shapes; place finger just above table.
        finger_pos_separated = wp.vec3(0.3, 0.0, 0.021)
        _dist2, _point2, normal2, _coll2 = _geom_dist(
            GeoType.BOX,
            self.TABLE_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            self.FINGER_HALF,
            finger_pos_separated,
            wp.quat_identity(),
        )
        self.assertGreater(
            abs(normal2[2]),
            self.NORMAL_Z_MIN,
            f"GJK box-on-table lateral: normal Z {normal2[2]:.4f} — wrong face",
        )

    def test_mpr_box_on_flat_table_lateral_offset_swapped(self):
        """MPR: same geometry as above but A=finger, B=table (shape order swapped)."""
        dist, _point, normal, collision = _mpr_dist(
            GeoType.BOX,
            self.FINGER_HALF,
            self.FINGER_POS,
            wp.quat_identity(),
            GeoType.BOX,
            self.TABLE_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        self._assert_top_face_contact(dist, normal, collision, "MPR box-on-table lateral (swapped)")

    def test_mpr_capsule_on_flat_table_lateral_offset(self):
        """MPR: thin capsule (Minkowski-expanded line segment) on large flat table, lateral offset.

        Simulates how Newton treats a CAPSULE in GJK/MPR: the radius is shrunk
        to ``small_radius = 1e-4`` and the Minkowski offset is added in post-process.
        """
        # CAPSULE scale: x=radius, y=half_height (axis +Z). half_height=0.02 extends
        # the bottom endpoint to z = center_z - 0.02. Place center at z=0.029 so
        # the bottom endpoint sits 1 mm inside the table top (z=0.01).
        capsule_half = wp.vec3(1e-4, 0.02, 0.0)  # tiny radius, 20 mm half-height along Z
        capsule_pos = wp.vec3(0.3, 0.0, 0.029)  # bottom endpoint at z=0.009 (1 mm inside table)

        dist, _point, normal, collision = _mpr_dist(
            GeoType.BOX,
            self.TABLE_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.CAPSULE,
            capsule_half,
            capsule_pos,
            wp.quat_identity(),
        )
        self._assert_top_face_contact(dist, normal, collision, "MPR capsule-on-table lateral")


class TestFlatFlatContact(unittest.TestCase):
    """Two flat-box (slab) shapes in contact — exercises both geom_a and geom_b BOX fixes."""

    SLAB_HALF = wp.vec3(0.5, 0.5, 0.01)  # 1 m x 1 m x 0.02 m slab
    NORMAL_Z_MIN = 0.98

    def test_mpr_two_flat_boxes_stacked(self):
        """Two identical slabs stacked directly on top of each other, 1 mm penetration."""
        # Slab A: z in [-0.01, 0.01]. Slab B center at z=0.019, so B bottom at z=0.009 (1 mm inside A).
        _dist, _point, normal, collision = _mpr_dist(
            GeoType.BOX,
            self.SLAB_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            self.SLAB_HALF,
            wp.vec3(0.0, 0.0, 0.019),
            wp.quat_identity(),
        )
        self.assertTrue(collision, "two stacked flat boxes: expected collision=True")
        self.assertGreater(abs(normal[2]), self.NORMAL_Z_MIN, f"normal Z {normal[2]:.4f} — wrong face")

    def test_mpr_two_flat_boxes_stacked_lateral_offset(self):
        """Two identical slabs with 0.3 m lateral offset and 1 mm penetration."""
        _dist, _point, normal, collision = _mpr_dist(
            GeoType.BOX,
            self.SLAB_HALF,
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            self.SLAB_HALF,
            wp.vec3(0.3, 0.0, 0.019),
            wp.quat_identity(),
        )
        self.assertTrue(collision, "two laterally-offset flat boxes: expected collision=True")
        self.assertGreater(abs(normal[2]), self.NORMAL_Z_MIN, f"normal Z {normal[2]:.4f} — wrong face")


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
