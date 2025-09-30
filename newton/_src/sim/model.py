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

"""Implementation of the Newton model class."""
# isort: skip_file

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.types import Devicelike
from ..geometry.collision_convex import create_solve_convex_multi_contact
from ..geometry.collision_primitive import collide_plane_box, collide_plane_sphere, collide_sphere_box, collide_box_box
from ..geometry.support_function import (
    center_func as center_map,
    support_map as support_map_func,
    SupportMapDataProvider,
    GenericShapeData,
)
from ..geometry.types import GeoType
from .contacts import Contacts
from .control import Control
from .state import State


# Pre-create the convex multi-contact solver (usable inside kernels)
solve_convex_multi_contact = create_solve_convex_multi_contact(support_map_func, center_map)



@wp.func
def write_contact(
    contact_point_center: wp.vec3,
    contact_normal_a_to_b: wp.vec3,
    contact_distance: float,
    radius_eff_a: float,
    radius_eff_b: float,
    thickness_a: float,
    thickness_b: float,
    shape_a: int,
    shape_b: int,
    X_bw_a: wp.transform,
    X_bw_b: wp.transform,
    tid: int,
    rigid_contact_margin: float,
    contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    """
    Write a contact to the output arrays.

    Args:
        contact_point_center: Center point of contact in world space
        contact_normal_a_to_b: Contact normal pointing from shape A to B
        contact_distance: Distance between contact points
        radius_eff_a: Effective radius of shape A (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        radius_eff_b: Effective radius of shape B (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        thickness_a: Contact thickness for shape A (similar to contact offset)
        thickness_b: Contact thickness for shape B (similar to contact offset)
        shape_a: Shape A index
        shape_b: Shape B index
        X_bw_a: Transform from world to body A
        X_bw_b: Transform from world to body B
        tid: Thread ID
        rigid_contact_margin: Contact margin for rigid bodies
        contact_max: Maximum number of contacts
        contact_count: Array to track contact count
        out_shape0: Output array for shape A indices
        out_shape1: Output array for shape B indices
        out_point0: Output array for contact points on shape A
        out_point1: Output array for contact points on shape B
        out_offset0: Output array for offsets on shape A
        out_offset1: Output array for offsets on shape B
        out_normal: Output array for contact normals
        out_thickness0: Output array for thickness values for shape A
        out_thickness1: Output array for thickness values for shape B
        out_tids: Output array for thread IDs
    """

    total_separation_needed = radius_eff_a + radius_eff_b + thickness_a + thickness_b

    offset_mag_a = radius_eff_a + thickness_a
    offset_mag_b = radius_eff_b + thickness_b

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_normal_a_to_b)

    a_contact_world = contact_point_center - contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_a)
    b_contact_world = contact_point_center + contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_b)

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed
    if d < rigid_contact_margin:
        index = wp.atomic_add(contact_count, 0, 1)
        if index >= contact_max:
            # Reached buffer limit
            return

        out_shape0[index] = shape_a
        out_shape1[index] = shape_b

        # Contact points are stored in body frames
        out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
        out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

        # Match kernels.py convention: normal should point from box to plane (downward)
        contact_normal = -contact_normal_a_to_b

        # Offsets in body frames
        out_offset0[index] = wp.transform_vector(X_bw_a, -offset_mag_a * contact_normal)
        out_offset1[index] = wp.transform_vector(X_bw_b, offset_mag_b * contact_normal)

        out_normal[index] = contact_normal
        out_thickness0[index] = offset_mag_a
        out_thickness1[index] = offset_mag_b
        out_tids[index] = tid

        
@wp.kernel(enable_backward=False)
def build_contacts_kernel_gjk_mpr(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    shape_collision_radius: wp.array(dtype=float),
    shape_pairs: wp.array(dtype=wp.vec2i),
    sum_of_contact_offsets: float,
    rigid_contact_margin: float,
    contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid == 0:
        wp.printf("build_contacts_kernel_gjk_mpr\n")

    # Early bounds check - following convert_newton_contacts_to_mjwarp_kernel pattern
    if tid >= shape_pairs.shape[0]:
        return

    pair = shape_pairs[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    # Safety: ignore self-collision pairs
    if shape_a == shape_b:
        return

    # Validate shape indices - following convert_newton_contacts_to_mjwarp_kernel pattern
    if shape_a < 0 or shape_b < 0:
        return

    # Get shape types early for validation
    type_a = shape_type[shape_a]
    type_b = shape_type[shape_b]

    # Sort shapes by type to ensure consistent collision handling order
    if type_a > type_b:
        # Swap shapes to maintain consistent ordering
        shape_a, shape_b = shape_b, shape_a
        type_a, type_b = type_b, type_a

    # Get body indices - following convert_newton_contacts_to_mjwarp_kernel pattern
    rigid_a = shape_body[shape_a]
    rigid_b = shape_body[shape_b]

    # Compute world transforms
    X_ws_a = (
        shape_transform[shape_a] if rigid_a == -1 else wp.transform_multiply(body_q[rigid_a], shape_transform[shape_a])
    )
    X_ws_b = (
        shape_transform[shape_b] if rigid_b == -1 else wp.transform_multiply(body_q[rigid_b], shape_transform[shape_b])
    )

    pos_a = wp.transform_get_translation(X_ws_a)
    pos_b = wp.transform_get_translation(X_ws_b)

    # Early bounding sphere check using precomputed collision radii - inflate by rigid_contact_margin
    radius_a = shape_collision_radius[shape_a] + rigid_contact_margin
    radius_b = shape_collision_radius[shape_b] + rigid_contact_margin
    center_distance = wp.length(pos_b - pos_a)

    # Early out if bounding spheres don't overlap
    if center_distance > (radius_a + radius_b):
        return

    rot_a = wp.transform_get_rotation(X_ws_a)
    rot_b = wp.transform_get_rotation(X_ws_b)

    # World->body transforms for writing contact points
    X_bw_a = wp.transform_identity() if rigid_a == -1 else wp.transform_inverse(body_q[rigid_a])
    X_bw_b = wp.transform_identity() if rigid_b == -1 else wp.transform_inverse(body_q[rigid_b])

    # Create geometry data structures for convex collision detection
    geom_a = GenericShapeData()
    geom_a.shape_type = type_a
    geom_a.scale = shape_scale[shape_a]

    geom_b = GenericShapeData()
    geom_b.shape_type = type_b
    geom_b.scale = shape_scale[shape_b]

    data_provider = SupportMapDataProvider()

    count, normal, penetrations, points_a, points_b, features = wp.static(solve_convex_multi_contact)(
        geom_a,
        geom_b,
        rot_a,
        rot_b,
        pos_a,
        pos_b,
        0.0,  # sum_of_contact_offsets
        data_provider,
    )

    for id in range(4):
        write_contact(
            0.5 * (points_a[id] + points_b[id]),
            normal,
            # wp.dot(normal, points_b[id] - points_a[id]),
            penetrations[id],
            0.0,
            0.0,
            shape_thickness[shape_a],
            shape_thickness[shape_b],
            shape_a,
            shape_b,
            X_bw_a,
            X_bw_b,
            tid,
            rigid_contact_margin,
            contact_max,
            contact_count,
            out_shape0,
            out_shape1,
            out_point0,
            out_point1,
            out_offset0,
            out_offset1,
            out_normal,
            out_thickness0,
            out_thickness1,
            out_tids,
        )

    return

    if type_a == int(GeoType.PLANE) and type_b == int(GeoType.BOX):
        plane_normal_world = wp.transform_vector(X_ws_a, wp.vec3(0.0, 0.0, 1.0))
        box_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_b))

        # Call collide_plane_box to get contact information
        contact_distances, contact_positions, contact_normal = collide_plane_box(
            plane_normal_world,  # plane_normal
            pos_a,  # plane_pos (plane position in world space)
            pos_b,  # box_pos (box position in world space)
            box_rot_mat,  # box_rot (box rotation matrix)
            shape_scale[shape_b],  # box_size (box half-extents)
        )

        for id in range(4):
            dist = contact_distances[id]
            if dist < wp.inf:
                write_contact(
                    contact_positions[id],
                    contact_normal,
                    contact_distances[id],
                    0.0,
                    0.0,
                    shape_thickness[shape_a],
                    shape_thickness[shape_b],
                    shape_a,
                    shape_b,
                    X_bw_a,
                    X_bw_b,
                    tid,
                    rigid_contact_margin,
                    contact_max,
                    contact_count,
                    out_shape0,
                    out_shape1,
                    out_point0,
                    out_point1,
                    out_offset0,
                    out_offset1,
                    out_normal,
                    out_thickness0,
                    out_thickness1,
                    out_tids,
                )

    # Implement Plane vs Sphere contacts (type order enforced above)
    elif type_a == int(GeoType.PLANE) and type_b == int(GeoType.SPHERE):
        # Plane frame helpers
        plane_normal_world = wp.transform_vector(X_ws_a, wp.vec3(0.0, 0.0, 1.0))

        # Sphere radius
        sphere_radius = shape_scale[shape_b][0]  # Sphere radius is stored in x component

        # Call collide_plane_sphere to get contact information
        contact_distance, contact_position = collide_plane_sphere(
            plane_normal_world,  # plane_normal
            pos_a,  # plane_pos (plane position in world space)
            pos_b,  # sphere_pos (sphere position in world space)
            sphere_radius,  # sphere_radius
        )

        # Use the write_contact function to write the contact
        write_contact(
            contact_position,
            plane_normal_world,  # contact normal from plane to sphere
            contact_distance,
            0.0,
            sphere_radius,
            shape_thickness[shape_a],
            shape_thickness[shape_b],
            shape_a,
            shape_b,
            X_bw_a,
            X_bw_b,
            tid,
            rigid_contact_margin,
            contact_max,
            contact_count,
            out_shape0,
            out_shape1,
            out_point0,
            out_point1,
            out_offset0,
            out_offset1,
            out_normal,
            out_thickness0,
            out_thickness1,
            out_tids,
        )

    # Implement Sphere vs Box contacts (type order enforced above)
    elif type_a == int(GeoType.SPHERE) and type_b == int(GeoType.BOX):
        # Sphere radius
        sphere_radius = shape_scale[shape_a][0]  # Sphere radius is stored in x component

        # Box half extents
        box_half_extents = shape_scale[shape_b]  # Box half-extents (hx, hy, hz)

        # Get box rotation matrix from quaternion
        box_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_b))

        # Call collide_sphere_box to get contact information
        contact_distance, contact_position, contact_normal = collide_sphere_box(
            pos_a,  # sphere_pos (sphere position in world space)
            sphere_radius,  # sphere_radius
            pos_b,  # box_pos (box position in world space)
            box_rot_mat,  # box_rot (box rotation matrix)
            box_half_extents,  # box_size (box half-extents)
        )

        # Print contact information in the requested format
        wp.printf(
            "point_a: (%f,%f,%f), point_b: (%f,%f,%f), normal: (%f,%f,%f), dist: %f\n",
            contact_position[0] + contact_normal[0] * sphere_radius,  # point_a
            contact_position[1] + contact_normal[1] * sphere_radius,
            contact_position[2] + contact_normal[2] * sphere_radius,
            contact_position[0],  # point_b 
            contact_position[1],
            contact_position[2],
            contact_normal[0],    # normal
            contact_normal[1], 
            contact_normal[2],
            contact_distance      # distance
        )


        # Use the write_contact function to write the contact
        write_contact(
            contact_position,
            contact_normal,  # contact normal from sphere to box
            contact_distance,
            sphere_radius,  # sphere's effective radius
            0.0,  # box has no effective radius
            shape_thickness[shape_a],
            shape_thickness[shape_b],
            shape_a,
            shape_b,
            X_bw_a,
            X_bw_b,
            tid,
            rigid_contact_margin,
            contact_max,
            contact_count,
            out_shape0,
            out_shape1,
            out_point0,
            out_point1,
            out_offset0,
            out_offset1,
            out_normal,
            out_thickness0,
            out_thickness1,
            out_tids,
        )

    # Implement Box vs Box contacts (type order enforced above)
    elif type_a == int(GeoType.BOX) and type_b == int(GeoType.BOX):
        # Box half extents for both boxes
        box1_half_extents = shape_scale[shape_a]  # Box A half-extents (hx, hy, hz)
        box2_half_extents = shape_scale[shape_b]  # Box B half-extents (hx, hy, hz)

        # Get box rotation matrices from quaternions
        box1_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_a))
        box2_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_b))

        # Call collide_box_box to get contact information
        contact_distances8, contact_positions8, contact_normals8 = collide_box_box(
            pos_a,              # box1_pos (first box position in world space)
            box1_rot_mat,       # box1_rot (first box rotation matrix)
            box1_half_extents,  # box1_size (first box half-extents)
            pos_b,              # box2_pos (second box position in world space)
            box2_rot_mat,       # box2_rot (second box rotation matrix)
            box2_half_extents,  # box2_size (second box half-extents)
        )

        # Process up to 8 contact points (box-box can generate multiple contacts)
        for id in range(8):
            dist = contact_distances8[id]
            if dist < wp.inf:
                # Use the write_contact function to write the contact
                write_contact(
                    contact_positions8[id],
                    contact_normals8[id],  # contact normal for this specific contact
                    dist,
                    0.0,                  # box A has no effective radius
                    0.0,                  # box B has no effective radius
                    shape_thickness[shape_a],
                    shape_thickness[shape_b],
                    shape_a,
                    shape_b,
                    X_bw_a,
                    X_bw_b,
                    tid,
                    rigid_contact_margin,
                    contact_max,
                    contact_count,
                    out_shape0,
                    out_shape1,
                    out_point0,
                    out_point1,
                    out_offset0,
                    out_offset1,
                    out_normal,
                    out_thickness0,
                    out_thickness1,
                    out_tids,
                )


class Model:
    """
    Represents the static (non-time-varying) definition of a simulation model in Newton.

    The Model class encapsulates all geometry, constraints, and parameters that describe a physical system
    for simulation. It is designed to be constructed via the ModelBuilder, which handles the correct
    initialization and population of all fields.

    Key Features:
        - Stores all static data for simulation: particles, rigid bodies, joints, shapes, soft/rigid elements, etc.
        - Supports grouping of entities by environment using group indices (e.g., `particle_group`, `body_group`, etc.).
          - Group index -1: global entities shared across all environments.
          - Group indices 0, 1, 2, ...: environment-specific entities.
        - Grouping enables:
          - Collision detection optimization (e.g., separating environments)
          - Visualization (e.g., spatially separating environments)
          - Parallel processing of independent environments

    Note:
        It is strongly recommended to use the :class:`ModelBuilder` to construct a Model.
        Direct instantiation and manual population of Model fields is possible but discouraged.
    """

    def __init__(self, device: Devicelike | None = None):
        """
        Initialize a Model object.

        Args:
            device (wp.Device, optional): Device on which the Model's data will be allocated.
        """
        self.requires_grad = False
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        self.num_envs = 0
        """Number of articulation environments added to the ModelBuilder via `add_builder`."""

        self.particle_q = None
        """Particle positions, shape [particle_count, 3], float."""
        self.particle_qd = None
        """Particle velocities, shape [particle_count, 3], float."""
        self.particle_mass = None
        """Particle mass, shape [particle_count], float."""
        self.particle_inv_mass = None
        """Particle inverse mass, shape [particle_count], float."""
        self.particle_radius = None
        """Particle radius, shape [particle_count], float."""
        self.particle_max_radius = 0.0
        """Maximum particle radius (useful for HashGrid construction)."""
        self.particle_ke = 1.0e3
        """Particle normal contact stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kd = 1.0e2
        """Particle normal contact damping (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kf = 1.0e2
        """Particle friction force stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_mu = 0.5
        """Particle friction coefficient."""
        self.particle_cohesion = 0.0
        """Particle cohesion strength."""
        self.particle_adhesion = 0.0
        """Particle adhesion strength."""
        self.particle_grid = None
        """HashGrid instance for accelerated simulation of particle interactions."""
        self.particle_flags = None
        """Particle enabled state, shape [particle_count], int."""
        self.particle_max_velocity = 1e5
        """Maximum particle velocity (to prevent instability)."""
        self.particle_group = None
        """Environment group index for each particle, shape [particle_count], int. -1 for global."""

        self.shape_key = []
        """List of keys for each shape."""
        self.shape_transform = None
        """Rigid shape transforms, shape [shape_count, 7], float."""
        self.shape_body = None
        """Rigid shape body index, shape [shape_count], int."""
        self.shape_flags = None
        """Rigid shape flags, shape [shape_count], int."""
        self.body_shapes = {}
        """Mapping from body index to list of attached shape indices."""

        # Shape material properties
        self.shape_material_ke = None
        """Shape contact elastic stiffness, shape [shape_count], float."""
        self.shape_material_kd = None
        """Shape contact damping stiffness, shape [shape_count], float."""
        self.shape_material_kf = None
        """Shape contact friction stiffness, shape [shape_count], float."""
        self.shape_material_ka = None
        """Shape contact adhesion distance, shape [shape_count], float."""
        self.shape_material_mu = None
        """Shape coefficient of friction, shape [shape_count], float."""
        self.shape_material_restitution = None
        """Shape coefficient of restitution, shape [shape_count], float."""

        # Shape geometry properties
        self.shape_type = None
        """Shape geometry type, shape [shape_count], int32."""
        self.shape_is_solid = None
        """Whether shape is solid or hollow, shape [shape_count], bool."""
        self.shape_thickness = None
        """Shape thickness, shape [shape_count], float."""
        self.shape_source = []
        """List of source geometry objects (e.g., :class:`~newton.Mesh`, :class:`~newton.SDF`) used for rendering and broadphase, shape [shape_count]."""
        self.shape_source_ptr = None
        """Geometry source pointer to be used inside the Warp kernels which can be generated by finalizing the geometry objects, see for example :meth:`newton.Mesh.finalize`, shape [shape_count], uint64."""
        self.shape_scale = None
        """Shape 3D scale, shape [shape_count, 3], float."""
        self.shape_filter = None
        """Shape filter group, shape [shape_count], int."""

        self.shape_collision_group = []
        """Collision group of each shape, shape [shape_count], int."""
        self.shape_collision_filter_pairs = set()
        """Pairs of shape indices that should not collide."""
        self.shape_collision_radius = None
        """Collision radius for bounding sphere broadphase, shape [shape_count], float."""
        self.shape_contact_pairs = None
        """Pairs of shape indices that may collide, shape [contact_pair_count, 2], int."""
        self.shape_contact_pair_count = 0
        """Number of shape contact pairs."""
        self.shape_group = None
        """Environment group index for each shape, shape [shape_count], int. -1 for global."""

        self.spring_indices = None
        """Particle spring indices, shape [spring_count*2], int."""
        self.spring_rest_length = None
        """Particle spring rest length, shape [spring_count], float."""
        self.spring_stiffness = None
        """Particle spring stiffness, shape [spring_count], float."""
        self.spring_damping = None
        """Particle spring damping, shape [spring_count], float."""
        self.spring_control = None
        """Particle spring activation, shape [spring_count], float."""
        self.spring_constraint_lambdas = None
        """Lagrange multipliers for spring constraints (internal use)."""

        self.tri_indices = None
        """Triangle element indices, shape [tri_count*3], int."""
        self.tri_poses = None
        """Triangle element rest pose, shape [tri_count, 2, 2], float."""
        self.tri_activations = None
        """Triangle element activations, shape [tri_count], float."""
        self.tri_materials = None
        """Triangle element materials, shape [tri_count, 5], float."""
        self.tri_areas = None
        """Triangle element rest areas, shape [tri_count], float."""

        self.edge_indices = None
        """Bending edge indices, shape [edge_count*4], int, each row is [o0, o1, v1, v2], where v1, v2 are on the edge."""
        self.edge_rest_angle = None
        """Bending edge rest angle, shape [edge_count], float."""
        self.edge_rest_length = None
        """Bending edge rest length, shape [edge_count], float."""
        self.edge_bending_properties = None
        """Bending edge stiffness and damping, shape [edge_count, 2], float."""
        self.edge_constraint_lambdas = None
        """Lagrange multipliers for edge constraints (internal use)."""

        self.tet_indices = None
        """Tetrahedral element indices, shape [tet_count*4], int."""
        self.tet_poses = None
        """Tetrahedral rest poses, shape [tet_count, 3, 3], float."""
        self.tet_activations = None
        """Tetrahedral volumetric activations, shape [tet_count], float."""
        self.tet_materials = None
        """Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]."""

        self.muscle_start = None
        """Start index of the first muscle point per muscle, shape [muscle_count], int."""
        self.muscle_params = None
        """Muscle parameters, shape [muscle_count, 5], float."""
        self.muscle_bodies = None
        """Body indices of the muscle waypoints, int."""
        self.muscle_points = None
        """Local body offset of the muscle waypoints, float."""
        self.muscle_activations = None
        """Muscle activations, shape [muscle_count], float."""

        self.body_q = None
        """Rigid body poses for state initialization, shape [body_count, 7], float."""
        self.body_qd = None
        """Rigid body velocities for state initialization, shape [body_count, 6], float."""
        self.body_com = None
        """Rigid body center of mass (in local frame), shape [body_count, 3], float."""
        self.body_inertia = None
        """Rigid body inertia tensor (relative to COM), shape [body_count, 3, 3], float."""
        self.body_inv_inertia = None
        """Rigid body inverse inertia tensor (relative to COM), shape [body_count, 3, 3], float."""
        self.body_mass = None
        """Rigid body mass, shape [body_count], float."""
        self.body_inv_mass = None
        """Rigid body inverse mass, shape [body_count], float."""
        self.body_key = []
        """Rigid body keys, shape [body_count], str."""
        self.body_group = None
        """Environment group index for each body, shape [body_count], int. Global entities have group index -1."""

        self.joint_q = None
        """Generalized joint positions for state initialization, shape [joint_coord_count], float."""
        self.joint_qd = None
        """Generalized joint velocities for state initialization, shape [joint_dof_count], float."""
        self.joint_f = None
        """Generalized joint forces for state initialization, shape [joint_dof_count], float."""
        self.joint_target = None
        """Generalized joint target inputs, shape [joint_dof_count], float."""
        self.joint_type = None
        """Joint type, shape [joint_count], int."""
        self.joint_parent = None
        """Joint parent body indices, shape [joint_count], int."""
        self.joint_child = None
        """Joint child body indices, shape [joint_count], int."""
        self.joint_ancestor = None
        """Maps from joint index to the index of the joint that has the current joint parent body as child (-1 if no such joint ancestor exists), shape [joint_count], int."""
        self.joint_X_p = None
        """Joint transform in parent frame, shape [joint_count, 7], float."""
        self.joint_X_c = None
        """Joint mass frame in child frame, shape [joint_count, 7], float."""
        self.joint_axis = None
        """Joint axis in child frame, shape [joint_dof_count, 3], float."""
        self.joint_armature = None
        """Armature for each joint axis (used by :class:`~newton.solvers.SolverMuJoCo` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_target_ke = None
        """Joint stiffness, shape [joint_dof_count], float."""
        self.joint_target_kd = None
        """Joint damping, shape [joint_dof_count], float."""
        self.joint_effort_limit = None
        """Joint effort (force/torque) limits, shape [joint_dof_count], float."""
        self.joint_velocity_limit = None
        """Joint velocity limits, shape [joint_dof_count], float."""
        self.joint_friction = None
        """Joint friction coefficient, shape [joint_dof_count], float."""
        self.joint_dof_dim = None
        """Number of linear and angular dofs per joint, shape [joint_count, 2], int."""
        self.joint_dof_mode = None
        """Control mode for each joint dof, shape [joint_dof_count], int."""
        self.joint_enabled = None
        """Controls which joint is simulated (bodies become disconnected if False), shape [joint_count], int."""
        self.joint_limit_lower = None
        """Joint lower position limits, shape [joint_dof_count], float."""
        self.joint_limit_upper = None
        """Joint upper position limits, shape [joint_dof_count], float."""
        self.joint_limit_ke = None
        """Joint position limit stiffness (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_limit_kd = None
        """Joint position limit damping (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_twist_lower = None
        """Joint lower twist limit, shape [joint_count], float."""
        self.joint_twist_upper = None
        """Joint upper twist limit, shape [joint_count], float."""
        self.joint_q_start = None
        """Start index of the first position coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_qd_start = None
        """Start index of the first velocity coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_key = []
        """Joint keys, shape [joint_count], str."""
        self.joint_group = None
        """Environment group index for each joint, shape [joint_count], int. -1 for global."""
        self.articulation_start = None
        """Articulation start index, shape [articulation_count], int."""
        self.articulation_key = []
        """Articulation keys, shape [articulation_count], str."""
        self.articulation_group = None
        """Environment group index for each articulation, shape [articulation_count], int. -1 for global."""
        self.max_joints_per_articulation = 0
        """Maximum number of joints in any articulation (used for IK kernel dimensioning)."""

        self.soft_contact_ke = 1.0e3
        """Stiffness of soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kd = 10.0
        """Damping of soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kf = 1.0e3
        """Stiffness of friction force in soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_mu = 0.5
        """Friction coefficient of soft contacts."""
        self.soft_contact_restitution = 0.0
        """Restitution coefficient of soft contacts (used by :class:`SolverXPBD`)."""

        self.rigid_contact_max = 0
        """Number of potential contact points between rigid bodies."""
        self.rigid_contact_torsional_friction = 0.0
        """Torsional friction coefficient for rigid body contacts (used by :class:`SolverXPBD`)."""
        self.rigid_contact_rolling_friction = 0.0
        """Rolling friction coefficient for rigid body contacts (used by :class:`SolverXPBD`)."""

        self.up_vector = np.array((0.0, 0.0, 1.0))
        """Up vector of the world, shape [3], float."""
        self.up_axis = 2
        """Up axis: 0 for x, 1 for y, 2 for z."""
        self.gravity = None
        """Gravity vector, shape [1], dtype vec3."""

        self.equality_constraint_type = None
        """Type of equality constraint, shape [equality_constraint_count], int."""
        self.equality_constraint_body1 = None
        """First body index, shape [equality_constraint_count], int."""
        self.equality_constraint_body2 = None
        """Second body index, shape [equality_constraint_count], int."""
        self.equality_constraint_anchor = None
        """Anchor point on first body, shape [equality_constraint_count, 3], float."""
        self.equality_constraint_torquescale = None
        """Torque scale, shape [equality_constraint_count], float."""
        self.equality_constraint_relpose = None
        """Relative pose, shape [equality_constraint_count, 7], float."""
        self.equality_constraint_joint1 = None
        """First joint index, shape [equality_constraint_count], int."""
        self.equality_constraint_joint2 = None
        """Second joint index, shape [equality_constraint_count], int."""
        self.equality_constraint_polycoef = None
        """Polynomial coefficients, shape [equality_constraint_count, 2], float."""
        self.equality_constraint_key = []
        """Constraint name/key, shape [equality_constraint_count], str."""
        self.equality_constraint_enabled = None
        """Whether constraint is active, shape [equality_constraint_count], bool."""

        self.particle_count = 0
        """Total number of particles in the system."""
        self.body_count = 0
        """Total number of bodies in the system."""
        self.shape_count = 0
        """Total number of shapes in the system."""
        self.joint_count = 0
        """Total number of joints in the system."""
        self.tri_count = 0
        """Total number of triangles in the system."""
        self.tet_count = 0
        """Total number of tetrahedra in the system."""
        self.edge_count = 0
        """Total number of edges in the system."""
        self.spring_count = 0
        """Total number of springs in the system."""
        self.muscle_count = 0
        """Total number of muscles in the system."""
        self.articulation_count = 0
        """Total number of articulations in the system."""
        self.joint_dof_count = 0
        """Total number of velocity degrees of freedom of all joints. Equals the number of joint axes."""
        self.joint_coord_count = 0
        """Total number of position degrees of freedom of all joints."""
        self.equality_constraint_count = 0
        """Total number of equality constraints in the system."""

        # indices of particles sharing the same color
        self.particle_color_groups = []
        """Coloring of all particles for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of particles sharing the same color."""
        self.particle_colors = None
        """Color assignment for every particle."""

        self.device = wp.get_device(device)
        """Device on which the Model was allocated."""

        self.attribute_frequency = {}
        """Classifies each attribute as per body, per joint, per DOF, etc."""

        # attributes per body
        self.attribute_frequency["body_q"] = "body"
        self.attribute_frequency["body_qd"] = "body"
        self.attribute_frequency["body_com"] = "body"
        self.attribute_frequency["body_inertia"] = "body"
        self.attribute_frequency["body_inv_inertia"] = "body"
        self.attribute_frequency["body_mass"] = "body"
        self.attribute_frequency["body_inv_mass"] = "body"
        self.attribute_frequency["body_f"] = "body"

        # attributes per joint
        self.attribute_frequency["joint_type"] = "joint"
        self.attribute_frequency["joint_parent"] = "joint"
        self.attribute_frequency["joint_child"] = "joint"
        self.attribute_frequency["joint_ancestor"] = "joint"
        self.attribute_frequency["joint_X_p"] = "joint"
        self.attribute_frequency["joint_X_c"] = "joint"
        self.attribute_frequency["joint_dof_dim"] = "joint"
        self.attribute_frequency["joint_enabled"] = "joint"
        self.attribute_frequency["joint_twist_lower"] = "joint"
        self.attribute_frequency["joint_twist_upper"] = "joint"

        # attributes per joint coord
        self.attribute_frequency["joint_q"] = "joint_coord"

        # attributes per joint dof
        self.attribute_frequency["joint_qd"] = "joint_dof"
        self.attribute_frequency["joint_f"] = "joint_dof"
        self.attribute_frequency["joint_armature"] = "joint_dof"
        self.attribute_frequency["joint_target"] = "joint_dof"
        self.attribute_frequency["joint_axis"] = "joint_dof"
        self.attribute_frequency["joint_target_ke"] = "joint_dof"
        self.attribute_frequency["joint_target_kd"] = "joint_dof"
        self.attribute_frequency["joint_dof_mode"] = "joint_dof"
        self.attribute_frequency["joint_limit_lower"] = "joint_dof"
        self.attribute_frequency["joint_limit_upper"] = "joint_dof"
        self.attribute_frequency["joint_limit_ke"] = "joint_dof"
        self.attribute_frequency["joint_limit_kd"] = "joint_dof"
        self.attribute_frequency["joint_effort_limit"] = "joint_dof"
        self.attribute_frequency["joint_friction"] = "joint_dof"
        self.attribute_frequency["joint_velocity_limit"] = "joint_dof"

        # attributes per shape
        self.attribute_frequency["shape_transform"] = "shape"
        self.attribute_frequency["shape_body"] = "shape"
        self.attribute_frequency["shape_flags"] = "shape"
        self.attribute_frequency["shape_material_ke"] = "shape"
        self.attribute_frequency["shape_material_kd"] = "shape"
        self.attribute_frequency["shape_material_kf"] = "shape"
        self.attribute_frequency["shape_material_ka"] = "shape"
        self.attribute_frequency["shape_material_mu"] = "shape"
        self.attribute_frequency["shape_material_restitution"] = "shape"
        self.attribute_frequency["shape_type"] = "shape"
        self.attribute_frequency["shape_is_solid"] = "shape"
        self.attribute_frequency["shape_thickness"] = "shape"
        self.attribute_frequency["shape_source_ptr"] = "shape"
        self.attribute_frequency["shape_scale"] = "shape"
        self.attribute_frequency["shape_filter"] = "shape"

    def state(self, requires_grad: bool | None = None) -> State:
        """
        Create and return a new :class:`State` object for this model.

        The returned state is initialized with the initial configuration from the model description.

        Args:
            requires_grad (bool, optional): Whether the state variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.

        Returns:
            State: The state object
        """
        s = State()
        if requires_grad is None:
            requires_grad = self.requires_grad

        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        # rigid bodies
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # joints
        if self.joint_count:
            s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
            s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        return s

    def control(self, requires_grad: bool | None = None, clone_variables: bool = True) -> Control:
        """
        Create and return a new :class:`Control` object for this model.

        The returned control object is initialized with the control inputs from the model description.

        Args:
            requires_grad (bool, optional): Whether the control variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.
            clone_variables (bool): If True, clone the control input arrays; if False, use references.

        Returns:
            Control: The initialized control object.
        """
        c = Control()
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            if self.joint_count:
                c.joint_target = wp.clone(self.joint_target, requires_grad=requires_grad)
                c.joint_f = wp.clone(self.joint_f, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_target = self.joint_target
            c.joint_f = self.joint_f
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        return c

    def collide_pure_gjk_mpr_multicontact(self: Model, state: State):
        # print("collide_pure_gjk_mpr_multicontact")

        # Allocate a Contacts buffer compatible with the default pipeline
        rigid_contact_max = (
            self.rigid_contact_max if self.rigid_contact_max > 0 else max(1, self.shape_contact_pair_count * 4)
        )
        contacts = Contacts(
            rigid_contact_max=rigid_contact_max,
            soft_contact_max=0,
            requires_grad=self.requires_grad,
            device=self.device,
        )
        contacts.clear()
        contacts.rigid_contact_count.zero_()

        # Launch kernel across all shape pairs
        if self.shape_contact_pair_count and self.shape_contact_pairs is not None:
            wp.launch(
                kernel=build_contacts_kernel_gjk_mpr,
                dim=self.shape_contact_pair_count,
                inputs=[
                    state.body_q,
                    self.shape_transform,
                    self.shape_body,
                    self.shape_type,
                    self.shape_scale,
                    self.shape_thickness,
                    self.shape_collision_radius,
                    self.shape_contact_pairs,
                    0.0,  # sum_of_contact_offsets
                    self._collision_pipeline.rigid_contact_margin
                    if hasattr(self, "_collision_pipeline") and self._collision_pipeline is not None
                    else 0.1,
                    contacts.rigid_contact_max,
                ],
                outputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_thickness0,
                    contacts.rigid_contact_thickness1,
                    contacts.rigid_contact_tids,
                ],
                device=contacts.device,
            )

        return contacts

    def set_gravity(self, gravity: tuple[float, float, float] | list[float] | wp.vec3) -> None:
        """
        Set gravity for runtime modification.

        Args:
            gravity: Gravity vector as a tuple, list, or wp.vec3.
                    Common values: (0, 0, -9.81) for Z-up, (0, -9.81, 0) for Y-up.

        Note:
            After calling this method, you should notify solvers via
            `solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)`.
        """
        if self.gravity is None:
            raise RuntimeError(
                "Model gravity not initialized. Ensure the model was created via ModelBuilder.finalize()"
            )

        if isinstance(gravity, tuple | list):
            self.gravity.assign([wp.vec3(gravity[0], gravity[1], gravity[2])])
        else:
            self.gravity.assign([gravity])

    def collide(
        self: Model,
        state: State,
        collision_pipeline: CollisionPipeline | None = None,
        rigid_contact_max_per_pair: int | None = None,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool | None = None,
    ) -> Contacts:
        """
        Generate contact points for the particles and rigid bodies in the model.

        This method produces a :class:`Contacts` object containing collision/contact information
        for use in contact-dynamics kernels.

        Args:
            state (State): The current state of the model.
            collision_pipeline (CollisionPipeline, optional): Collision pipeline to use for contact generation.
                If not provided, a new one will be created if it hasn't been constructed before for this model.
            rigid_contact_max_per_pair (int, optional): Maximum number of rigid contacts per shape pair.
                If None, a kernel is launched to count the number of possible contacts.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Default is 0.01.
            soft_contact_max (int, optional): Maximum number of soft contacts.
                If None, a kernel is launched to count the number of possible contacts.
            soft_contact_margin (float, optional): Margin for soft contact generation. Default is 0.01.
            edge_sdf_iter (int, optional): Number of search iterations for finding closest contact points between edges and SDF. Default is 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate over all vertices of a mesh for contact generation (used for capsule/box <> mesh collision). Default is True.
            requires_grad (bool, optional): Whether to duplicate contact arrays for gradient computation. If None, uses :attr:`Model.requires_grad`.

        Returns:
            Contacts: The contact object containing collision information.
        """
        return self.collide_pure_gjk_mpr_multicontact(state)

        from .collide import CollisionPipeline

        if requires_grad is None:
            requires_grad = self.requires_grad

        if collision_pipeline is not None:
            self._collision_pipeline = collision_pipeline
        elif not hasattr(self, "_collision_pipeline"):
            self._collision_pipeline = CollisionPipeline.from_model(
                self,
                rigid_contact_max_per_pair,
                rigid_contact_margin,
                soft_contact_max,
                soft_contact_margin,
                edge_sdf_iter,
                iterate_mesh_vertices,
                requires_grad,
            )

        # update any additional parameters
        self._collision_pipeline.rigid_contact_margin = rigid_contact_margin
        self._collision_pipeline.soft_contact_margin = soft_contact_margin
        self._collision_pipeline.edge_sdf_iter = edge_sdf_iter
        self._collision_pipeline.iterate_mesh_vertices = iterate_mesh_vertices

        return self._collision_pipeline.collide(self, state)

    def add_attribute(self, name: str, attrib: wp.array, frequency: str):
        """
        Add a custom attribute to the model.

        Args:
            name (str): Name of the attribute.
            attrib (wp.array): The array to add as an attribute.
            frequency (str): The frequency of the attribute. Must be one of:
                - "body": per body
                - "joint": per joint
                - "joint_coord": per joint coordinate
                - "joint_dof": per joint degree of freedom
                - "shape": per shape

        Raises:
            AttributeError: If the attribute already exists, is not a wp.array, or is on the wrong device.
        """
        if hasattr(self, name):
            raise AttributeError(f"Attribute '{name}' already exists")

        if not isinstance(attrib, wp.array):
            raise AttributeError(f"Attribute '{name}' must be an array, got {type(attrib)}")
        if attrib.device != self.device:
            raise AttributeError(
                f"Attribute '{name}' must be on the same device as the Model, expected {self.device}, got {attrib.device}"
            )

        setattr(self, name, attrib)

        self.attribute_frequency[name] = frequency

    def get_attribute_frequency(self, name):
        """
        Get the frequency of an attribute.

        Possible frequencies are:
            - "body": per body
            - "joint": per joint
            - "joint_coord": per joint coordinate
            - "joint_dof": per joint degree of freedom
            - "shape": per shape

        Args:
            name (str): Name of the attribute.

        Returns:
            str: The frequency of the attribute.

        Raises:
            AttributeError: If the attribute frequency is not known.
        """
        frequency = self.attribute_frequency.get(name)
        if frequency is None:
            raise AttributeError(f"Attribute frequency of '{name}' is not known")
        return frequency
