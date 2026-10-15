.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.geometry
===============

.. py:module:: newton.geometry
.. currentmodule:: newton.geometry

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   BroadPhaseAllPairs
   BroadPhaseExplicit
   BroadPhaseImplicitTree
   BroadPhaseSAP
   HydroelasticSDF
   ImplicitBvh
   ImplicitBvhData
   ImplicitBvhMesh
   ImplicitBvhMeshData
   NarrowPhase
   ParticleSurface
   SDFCachedSampling
   TriMeshCollisionInfo

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   collide_box_box
   collide_capsule_box
   collide_capsule_capsule
   collide_plane_box
   collide_plane_capsule
   collide_plane_cylinder
   collide_plane_ellipsoid
   collide_plane_sphere
   collide_sphere_box
   collide_sphere_capsule
   collide_sphere_cylinder
   collide_sphere_sphere
   compute_inertia_shape
   compute_offset_mesh
   create_empty_sdf_data
   extract_particle_surface
   implicit_bvh_make_query_pairs
   implicit_bvh_make_query_point
   implicit_bvh_make_query_ray
   implicit_bvh_make_query_ray_foreach
   implicit_bvh_mesh_query_point
   implicit_bvh_mesh_query_ray
   implicit_bvh_query_aabb
   implicit_bvh_query_aabb_next
   implicit_bvh_query_point
   implicit_bvh_query_ray
   sdf_box
   sdf_capsule
   sdf_cone
   sdf_cylinder
   sdf_mesh
   sdf_plane
   sdf_sphere
   transform_inertia
