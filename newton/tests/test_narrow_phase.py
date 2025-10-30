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

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.types import GeoType


class TestNarrowPhase(unittest.TestCase):
    """Test NarrowPhase collision detection API with various primitive pairs."""

    def setUp(self):
        """Set up narrow phase instance for tests."""
        self.narrow_phase = NarrowPhase()
        self.contact_margin = 0.01

    def _create_geometry_arrays(self, geom_list):
        """Create geometry arrays from a list of geometry descriptions.

        Each geometry is a dict with:
            - type: GeoType value
            - transform: (position, quaternion) tuple
            - data: scale/size as vec3, thickness as float
            - source: mesh pointer (default 0)
            - cutoff: cutoff distance (default 0.0)

        Returns:
            Tuple of (geom_types, geom_data, geom_transform, geom_source, geom_cutoff)
        """
        n = len(geom_list)

        geom_types = np.zeros(n, dtype=np.int32)
        geom_data = np.zeros(n, dtype=wp.vec4)
        geom_transforms = []
        geom_source = np.zeros(n, dtype=np.uint64)
        geom_cutoff = np.zeros(n, dtype=np.float32)

        for i, geom in enumerate(geom_list):
            geom_types[i] = int(geom["type"])

            # Data: (scale_x, scale_y, scale_z, thickness)
            data = geom.get("data", ([1.0, 1.0, 1.0], 0.0))
            if isinstance(data, tuple):
                scale, thickness = data
            else:
                scale = data
                thickness = 0.0
            geom_data[i] = wp.vec4(scale[0], scale[1], scale[2], thickness)

            # Transform: position and quaternion
            pos, quat = geom.get("transform", ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]))
            geom_transforms.append(
                wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat(quat[0], quat[1], quat[2], quat[3]))
            )

            geom_source[i] = geom.get("source", 0)
            geom_cutoff[i] = geom.get("cutoff", 0.0)

        return (
            wp.array(geom_types, dtype=wp.int32),
            wp.array(geom_data, dtype=wp.vec4),
            wp.array(geom_transforms, dtype=wp.transform),
            wp.array(geom_source, dtype=wp.uint64),
            wp.array(geom_cutoff, dtype=wp.float32),
        )

    def _run_narrow_phase(self, geom_list, pairs):
        """Run narrow phase on given geometry and pairs.

        Args:
            geom_list: List of geometry descriptions
            pairs: List of (i, j) tuples indicating which geometries to test

        Returns:
            Tuple of (contact_count, contact_pairs, positions, normals, penetrations, tangents)
        """
        geom_types, geom_data, geom_transform, geom_source, geom_cutoff = self._create_geometry_arrays(geom_list)

        # Create candidate pairs
        candidate_pair = wp.array(np.array(pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i)
        num_candidate_pair = wp.array([len(pairs)], dtype=wp.int32)

        # Allocate output arrays
        max_contacts = len(pairs) * 10  # Allow multiple contacts per pair
        contact_pair = wp.zeros(max_contacts, dtype=wp.vec2i)
        contact_position = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_normal = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_penetration = wp.zeros(max_contacts, dtype=float)
        contact_tangent = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_count = wp.zeros(1, dtype=int)

        # Launch narrow phase
        self.narrow_phase.launch(
            candidate_pair=candidate_pair,
            num_candidate_pair=num_candidate_pair,
            geom_types=geom_types,
            geom_data=geom_data,
            geom_transform=geom_transform,
            geom_source=geom_source,
            geom_cutoff=geom_cutoff,
            contact_pair=contact_pair,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_tangent=contact_tangent,
            contact_count=contact_count,
            rigid_contact_margin=self.contact_margin,
        )

        wp.synchronize()

        # Return numpy arrays
        count = contact_count.numpy()[0]
        return (
            count,
            contact_pair.numpy()[:count],
            contact_position.numpy()[:count],
            contact_normal.numpy()[:count],
            contact_penetration.numpy()[:count],
            contact_tangent.numpy()[:count],
        )

    def test_sphere_sphere_separated(self):
        """Test sphere-sphere collision when separated."""
        # Two spheres separated by distance 1.5
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([3.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Separated spheres should produce no contacts (or contacts with positive separation)
        if count > 0:
            # If contact is generated, penetration should be positive (separation)
            self.assertGreater(penetrations[0], 0.0, "Separated spheres should have positive penetration (separation)")

    def test_sphere_sphere_touching(self):
        """Test sphere-sphere collision when exactly touching."""
        # Two unit spheres exactly touching at x=2.0
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Should generate contact at touching point
        self.assertGreater(count, 0, "Touching spheres should generate contact")
        self.assertAlmostEqual(penetrations[0], 0.0, places=2, msg="Touching spheres should have near-zero penetration")

        # Normal should point from sphere 0 to sphere 1 (along +X)
        self.assertAlmostEqual(normals[0][0], 1.0, places=2, msg="Normal should point along +X")
        self.assertAlmostEqual(normals[0][1], 0.0, places=2, msg="Normal Y should be 0")
        self.assertAlmostEqual(normals[0][2], 0.0, places=2, msg="Normal Z should be 0")

    def test_sphere_sphere_penetrating(self):
        """Test sphere-sphere collision with penetration."""
        test_cases = [
            # (separation, expected_penetration)
            (1.8, -0.2),  # Small penetration
            (1.5, -0.5),  # Medium penetration
            (1.2, -0.8),  # Large penetration
        ]

        for separation, expected_penetration in test_cases:
            with self.subTest(separation=separation):
                geom_list = [
                    {
                        "type": GeoType.SPHERE,
                        "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                        "data": ([1.0, 1.0, 1.0], 0.0),
                    },
                    {
                        "type": GeoType.SPHERE,
                        "transform": ([separation, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                        "data": ([1.0, 1.0, 1.0], 0.0),
                    },
                ]

                count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

                self.assertGreater(count, 0, "Penetrating spheres should generate contact")
                self.assertAlmostEqual(
                    penetrations[0],
                    expected_penetration,
                    places=2,
                    msg=f"Expected penetration {expected_penetration}, got {penetrations[0]}",
                )

                # Normal should be unit length
                normal_length = np.linalg.norm(normals[0])
                self.assertAlmostEqual(normal_length, 1.0, places=2, msg="Normal should be unit length")

                # Normal should point from sphere 0 toward sphere 1 (+X direction)
                self.assertGreater(normals[0][0], 0.9, msg="Normal should point primarily in +X direction")

    def test_sphere_sphere_different_radii(self):
        """Test sphere-sphere collision with different radii."""
        # Sphere at origin with radius 0.5, sphere at x=1.5 with radius 1.0
        # Distance between centers = 1.5
        # Sum of radii = 1.5
        # Expected penetration = 0.0 (just touching)
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Touching spheres should generate contact")
        self.assertAlmostEqual(penetrations[0], 0.0, places=2, msg="Should be just touching")

    def test_sphere_box_penetrating(self):
        """Test sphere-box collision with penetration."""
        # Unit sphere at origin (radius 1.0), box at (1.999, 0, 0) with half-size 1.0
        # Sphere surface at x=1.0, box left surface at x=0.999
        # Expected penetration = 0.001
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {"type": GeoType.BOX, "transform": ([1.999, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([1.0, 1.0, 1.0], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Should generate contact
        self.assertGreater(count, 0, "Sphere-box should generate contact")

        # Normal should point approximately along +X axis
        self.assertGreater(abs(normals[0][0]), 0.9, msg="Normal should be primarily along X axis")

    def test_sphere_box_corner_collision(self):
        """Test sphere-box collision at box corner."""
        # Sphere approaching box corner
        offset = 1.5  # Distance to corner
        corner_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # Unit vector toward corner
        sphere_pos = corner_dir * offset

        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": (sphere_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {"type": GeoType.BOX, "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([1.0, 1.0, 1.0], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # May or may not have contact depending on exact distance
        if count > 0:
            # Normal should point approximately along corner direction
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=2, msg="Normal should be unit length")

    def test_box_box_face_collision(self):
        """Test box-box collision with face contact."""
        # Two unit boxes, one at origin, one offset by 1.8 along X
        # Box surfaces at x=1.0 and x=0.8, overlap = 0.2
        geom_list = [
            {"type": GeoType.BOX, "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([1.0, 1.0, 1.0], 0.0)},
            {"type": GeoType.BOX, "transform": ([1.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([1.0, 1.0, 1.0], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Penetrating boxes should generate contact(s)")

        # Check that at least one contact has normal along X axis
        has_x_normal = False
        for i in range(count):
            if abs(normals[i][0]) > 0.9:
                has_x_normal = True
                break
        self.assertTrue(has_x_normal, "At least one contact should have normal along X axis")

    def test_box_box_edge_collision(self):
        """Test box-box collision with edge contact."""
        # Two boxes, one rotated 45 degrees around Z axis
        # This creates an edge-edge contact scenario
        angle = np.pi / 4.0  # 45 degrees
        quat = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]

        geom_list = [
            {"type": GeoType.BOX, "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([0.5, 0.5, 0.5], 0.0)},
            {"type": GeoType.BOX, "transform": ([1.2, 0.0, 0.0], quat), "data": ([0.5, 0.5, 0.5], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Edge-edge collision should generate contact
        self.assertGreater(count, 0, "Edge-edge collision should generate contact")

    def test_sphere_capsule_cylinder_side(self):
        """Test sphere collision with capsule cylinder side."""
        # Capsule along Z axis, sphere approaching from +Y side
        # Capsule: radius=0.5, half_length=1.0 (extends from z=-1 to z=1)
        # Sphere: radius=0.5, at (0, 1.5, 0)
        # Distance = 0.5 (separation)
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # May not generate contact if separated beyond margin
        if count > 0:
            # Normal should point primarily along Y axis
            self.assertGreater(abs(normals[0][1]), 0.9, msg="Normal should be along Y axis for cylinder side collision")

    def test_sphere_capsule_cap(self):
        """Test sphere collision with capsule hemispherical cap."""
        # Capsule along Z axis, sphere approaching from above
        # Capsule: radius=0.5, half_length=1.0
        # Sphere: radius=0.5, at (0, 0, 2.2)
        # Top cap center at z=1.0, combined radii = 1.0, distance = 1.2
        # Expected separation = 0.2
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 2.2], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        if count > 0:
            # Normal should point primarily along Z axis
            self.assertGreater(abs(normals[0][2]), 0.9, msg="Normal should be along Z axis for cap collision")

    def test_capsule_capsule_parallel(self):
        """Test capsule-capsule collision when parallel."""
        # Two capsules parallel along Z axis, offset in Y direction
        geom_list = [
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Capsules with combined radius 1.0 and separation 1.5 should be separated
        if count > 0:
            self.assertGreater(penetrations[0], 0.0, "Separated capsules should have positive penetration")

    def test_capsule_capsule_crossed(self):
        """Test capsule-capsule collision when crossed (perpendicular)."""
        # Two capsules perpendicular: one along Z, one along X
        # Rotate second capsule 90 degrees around Y axis
        angle = np.pi / 2.0
        quat = [0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)]

        geom_list = [
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
            {"type": GeoType.CAPSULE, "transform": ([0.0, 0.0, 0.0], quat), "data": ([0.5, 1.0, 0.0], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Crossed capsules with radius 0.5 each should be penetrating (distance = 0, combined radii = 1.0)
        self.assertGreater(count, 0, "Crossed capsules should generate contact")

    def test_plane_sphere_above(self):
        """Test plane-sphere collision when sphere is above plane."""
        # Infinite plane at z=0, normal pointing up (+Z)
        # Sphere radius 1.0 at z=2.0 (center)
        # Distance from center to plane = 2.0, minus radius = 1.0 separation
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Separated - may not generate contact
        if count > 0:
            self.assertGreater(penetrations[0], 0.0, "Sphere above plane should have positive penetration")

    def test_plane_sphere_touching(self):
        """Test plane-sphere collision when sphere is nearly touching plane."""
        # Infinite plane at z=0, sphere radius 1.0 at z=0.999 (very slight penetration)
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.999], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Nearly touching sphere-plane should generate contact")
        self.assertAlmostEqual(penetrations[0], 0.0, places=2, msg="Nearly touching should have near-zero penetration")

    def test_plane_sphere_penetrating(self):
        """Test plane-sphere collision when sphere penetrates plane."""
        # Infinite plane at z=0, sphere radius 1.0 at z=0.5
        # Penetration depth = radius - distance = 1.0 - 0.5 = 0.5
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Penetrating sphere-plane should generate contact")
        self.assertLess(penetrations[0], 0.0, "Penetration should be negative")

        # Normal should point up (+Z)
        self.assertGreater(normals[0][2], 0.9, msg="Normal should point in +Z direction")

    def test_plane_box_resting(self):
        """Test plane-box collision when box is resting on plane."""
        # Infinite plane at z=0, box with size 1.0 at z=0.999 (very slightly penetrating)
        # Box bottom face at z=-0.001, top at z=1.999, so penetration depth ~0.001
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {"type": GeoType.BOX, "transform": ([0.0, 0.0, 0.999], [0.0, 0.0, 0.0, 1.0]), "data": ([1.0, 1.0, 1.0], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Box resting on plane should generate contact(s)
        self.assertGreater(count, 0, "Box on plane should generate contact")

        # All contacts should have normals pointing up
        for i in range(count):
            self.assertGreater(normals[i][2], 0.5, msg=f"Contact {i} normal should point upward")

    def test_plane_capsule_resting(self):
        """Test plane-capsule collision when capsule is resting on plane."""
        # Infinite plane at z=0, capsule radius 0.5 at z=0.5 (bottom touching)
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Capsule on plane should generate contact")

        # Normal should point up
        self.assertGreater(normals[0][2], 0.9, msg="Normal should point in +Z direction")

    def test_multiple_pairs(self):
        """Test narrow phase with multiple collision pairs."""
        # Create 3 spheres in a line, test all pairs
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([3.6, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        # Test pairs (0,1), (1,2), and (0,2)
        pairs = [(0, 1), (1, 2), (0, 2)]
        count, contact_pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, pairs)

        # Should get contacts for (0,1) and (1,2) which are penetrating
        # Pair (0,2) is separated so may not generate contact
        self.assertGreaterEqual(count, 2, "Should have at least 2 contacts for penetrating pairs")

        # Verify pairs are correct
        pair_set = {tuple(p) for p in contact_pairs}
        self.assertIn((0, 1), pair_set, "Should have contact for pair (0, 1)")
        self.assertIn((1, 2), pair_set, "Should have contact for pair (1, 2)")

    def test_cylinder_sphere(self):
        """Test cylinder-sphere collision."""
        # Cylinder along Z axis, sphere approaching from side
        geom_list = [
            {
                "type": GeoType.CYLINDER,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Cylinder radius 0.5 + sphere radius 0.5 = 1.0, distance = 1.5, so separation = 0.5
        if count > 0:
            # If contact generated, should have positive penetration (separation)
            self.assertGreater(penetrations[0], 0.0, "Separated should have positive penetration")

    def test_no_self_collision(self):
        """Test that narrow phase doesn't generate self-collisions."""
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        # Try to test sphere against itself
        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 0)])

        # Should not generate any contacts for self-collision
        self.assertEqual(count, 0, "Self-collision should not generate contacts")

    def test_contact_normal_unit_length(self):
        """Test that all contact normals are unit length."""
        # Create various collision scenarios
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {"type": GeoType.BOX, "transform": ([0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([0.5, 0.5, 0.5], 0.0)},
            {
                "type": GeoType.CAPSULE,
                "transform": ([3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        pairs = [(0, 1), (0, 2), (1, 3)]
        count, contact_pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, pairs)

        # Check all normals are unit length
        for i in range(count):
            normal_length = np.linalg.norm(normals[i])
            self.assertAlmostEqual(
                normal_length, 1.0, places=2, msg=f"Contact {i} normal should be unit length, got {normal_length}"
            )

    def test_contact_tangent_perpendicular(self):
        """Test that contact tangents are perpendicular to normals."""
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        for i in range(count):
            # Tangent should be perpendicular to normal (dot product ~ 0)
            dot_product = np.dot(normals[i], tangents[i])
            self.assertAlmostEqual(
                dot_product,
                0.0,
                places=2,
                msg=f"Contact {i} tangent should be perpendicular to normal, dot product = {dot_product}",
            )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)