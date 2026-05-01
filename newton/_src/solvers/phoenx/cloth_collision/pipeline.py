# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX-aware :class:`~newton._src.sim.collide.CollisionPipeline` subclass.

Adds cloth-triangle "virtual shapes" to the standard rigid pipeline
without forking it. Triangles occupy slots ``[S, S+T)`` in the
pipeline's per-shape arrays (``S = model.shape_count``,
``T = num_cloth_triangles``); rigid shapes keep their existing slots
``[0, S)``. The pipeline reuses the standard broadphase + narrowphase
+ contact-matching machinery; only the per-step "stamp the cloth
slots" work is phoenx-specific and lives in
:meth:`PhoenxCollisionPipeline._pre_broadphase_hook`.

Cloth triangles flow through the GJK/MPR narrow phase as
:data:`~newton._src.geometry.support_function.GeoTypeEx.TRIANGLE`
shapes (vertex A as the world-frame origin, B/C as offsets in
``shape_data.xyz`` / ``shape_auxiliary``). Tri-tri pairs that share a
node are dropped at broadphase time via the
:func:`~newton._src.solvers.phoenx.cloth_collision.broadphase_filter.phoenx_broadphase_filter`
``filter_func`` callback so the narrow phase doesn't waste time on
adjacent triangles already coupled by the cloth elasticity rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.support_function import GeoTypeEx
from newton._src.geometry.types import GeoType
from newton._src.sim.collide import CollisionPipeline, write_contact
from newton._src.solvers.phoenx.cloth_collision.broadphase_filter import (
    PhoenxBroadphaseFilterData,
    phoenx_broadphase_filter,
)
from newton._src.solvers.phoenx.cloth_collision.triangle_stamp import launch_cloth_triangle_stamp
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer

__all__ = ["PhoenxCollisionPipeline"]


def _extend_array(
    src: wp.array,
    extra_count: int,
    fill_value: Any,
    *,
    dtype: Any,
    device: wp.DeviceLike,
) -> wp.array:
    """Allocate ``len(src) + extra_count`` of ``dtype`` with the rigid
    prefix copied from ``src`` and the tail filled with
    ``fill_value``.

    Used at finalize-time to build the extended per-shape arrays the
    phoenx pipeline overrides on
    :class:`~newton._src.sim.collide.CollisionPipeline`. Cheap (O(N+T)
    on host, copies the prefix to device once).
    """
    s = int(src.shape[0]) if src is not None else 0
    out_np = np.empty(s + extra_count, dtype=src.numpy().dtype if src is not None else np.int32)
    if s > 0:
        out_np[:s] = src.numpy()
    if extra_count > 0:
        out_np[s:] = fill_value
    return wp.array(out_np, dtype=dtype, device=device)


def _zero_extend(
    src: wp.array,
    extra_count: int,
    *,
    dtype: Any,
    device: wp.DeviceLike,
) -> wp.array:
    """Like :func:`_extend_array` but the tail is zeroed rather than
    filled with a Python scalar (for ``vec3`` / ``transform`` /
    ``vec4`` arrays where a Python scalar can't be broadcast)."""
    s = int(src.shape[0]) if src is not None else 0
    total = s + extra_count
    out = wp.zeros(total, dtype=dtype, device=device)
    if s > 0:
        wp.copy(out, src, dest_offset=0, src_offset=0, count=s)
    return out


class PhoenxCollisionPipeline(CollisionPipeline):
    """Cloth-aware :class:`CollisionPipeline`.

    Reuses every standard pipeline feature -- contact matching,
    deterministic sort, hydroelastic, soft contacts -- and adds
    cloth triangles as ``GeoTypeEx.TRIANGLE`` virtual shapes in
    the slots ``[S, S+T)``.

    Args:
        model: Newton :class:`~newton.Model` for the rigid side.
            ``model.shape_count`` rigid shapes occupy slots ``[0, S)``.
        num_cloth_triangles: Cloth triangle count ``T`` (= ``model.tri_count``
            for typical Newton cloth meshes).
        tri_indices: ``(T, 3)`` particle indices per cloth triangle.
        particle_q: World-space particle positions, ``(num_particles, 3)``.
            Read every step by the cloth-stamping hook.
        particle_radius: Per-particle cloth half-thickness; the
            triangle's effective contact radius is the max of its
            three vertex radii.
        cloth_world: World index assigned to every cloth triangle
            shape. ``-1`` (global, collide with all worlds) is the
            default for single-world phoenx scenes; pass an explicit
            world id for multi-world setups where cloth lives in a
            specific world.
        cloth_collision_group: Collision group for cloth shapes.
            Newton's :func:`test_group_pair` rules: positive groups
            collide with same-group + negatives, negatives collide
            with everything except their counterpart, ``0`` doesn't
            collide. Defaults to ``1`` so cloth shapes collide with
            each other (modulo the shared-node filter) and with
            rigids in group ``1`` or any negative group.
        cloth_extra_margin: Extra contact-detection margin added to
            cloth-triangle AABBs each step on top of the
            ``max(vertex radius)`` expansion. Equivalent to the
            rigid-shape ``shape_gap``; used by the broadphase
            overlap test.
        cloth_shape_data_margin: Margin written into ``shape_data.w``
            for cloth shape slots; consumed by ``extract_shape_data``
            as the per-shape margin offset.
        shape_pairs_max: See :class:`CollisionPipeline`. The default
            here scales with ``S + T`` instead of just ``S``.
        kwargs: Forwarded to :class:`CollisionPipeline`.
    """

    def __init__(
        self,
        model,
        *,
        num_cloth_triangles: int,
        constraints: ConstraintContainer,
        cloth_cid_offset: int,
        num_bodies: int,
        particle_q: wp.array,
        particle_radius: wp.array,
        cloth_world: int = -1,
        cloth_collision_group: int = 1,
        cloth_extra_margin: float = 0.0,
        cloth_shape_data_margin: float = 0.0,
        shape_pairs_max: int | None = None,
        broad_phase: Any = "nxn",
        narrow_phase: NarrowPhase | None = None,
        **kwargs,
    ) -> None:
        if num_cloth_triangles < 0:
            raise ValueError(f"num_cloth_triangles must be >= 0 (got {num_cloth_triangles})")

        S = int(model.shape_count)
        T = int(num_cloth_triangles)
        device = model.device
        total = S + T

        # ---- Build extended per-shape arrays --------------------------
        # Static defaults for the cloth-shape slots: every cloth slot
        # gets the same fill (no per-tri customisation). Per-step
        # values (transform, shape_data, shape_auxiliary, AABBs) are
        # filled by the pre-broadphase hook each call.
        cloth_flags = int(ShapeFlags.COLLIDE_SHAPES)
        per_shape_int_defaults: dict[str, tuple[wp.array, int]] = {
            "shape_type": (model.shape_type, int(GeoTypeEx.TRIANGLE)),
            "shape_body": (model.shape_body, -1),
            "shape_world": (model.shape_world, int(cloth_world)),
            "shape_collision_group": (model.shape_collision_group, int(cloth_collision_group)),
            "shape_flags": (model.shape_flags, cloth_flags),
            "shape_sdf_index": (model.shape_sdf_index, -1),
            "shape_heightfield_index": (model.shape_heightfield_index, -1),
        }
        per_shape_float_defaults: dict[str, tuple[wp.array, float]] = {
            "shape_collision_radius": (model.shape_collision_radius, float(cloth_extra_margin)),
            "shape_gap": (model.shape_gap, float(cloth_extra_margin)),
        }
        ext_int = {
            k: _extend_array(src, T, fill, dtype=wp.int32, device=device)
            for k, (src, fill) in per_shape_int_defaults.items()
        }
        ext_float = {
            k: _extend_array(src, T, fill, dtype=wp.float32, device=device)
            for k, (src, fill) in per_shape_float_defaults.items()
        }
        ext_shape_source_ptr = _extend_array(model.shape_source_ptr, T, np.uint64(0), dtype=wp.uint64, device=device)
        # vec3 / transform extensions: zero-fill the cloth tail; the
        # hook overwrites those slots every step.
        ext_shape_transform = _zero_extend(model.shape_transform, T, dtype=wp.transform, device=device)
        ext_shape_collision_aabb_lower = _zero_extend(model.shape_collision_aabb_lower, T, dtype=wp.vec3, device=device)
        ext_shape_collision_aabb_upper = _zero_extend(model.shape_collision_aabb_upper, T, dtype=wp.vec3, device=device)
        # ``shape_auxiliary`` (TRIANGLE C-A offset) and the broadphase
        # AABB arrays own all S+T slots from scratch -- nothing on
        # the rigid side feeds them, the cloth-stamping hook fills
        # the cloth tail, and the parent's ``compute_shape_aabbs``
        # writes the rigid prefix.
        ext_shape_auxiliary = wp.zeros(total, dtype=wp.vec3, device=device)
        ext_aabb_lower = wp.zeros(total, dtype=wp.vec3, device=device)
        ext_aabb_upper = wp.zeros(total, dtype=wp.vec3, device=device)

        # ---- Build broadphase with phoenx filter ----------------------
        if not isinstance(broad_phase, str):
            raise TypeError(
                "PhoenxCollisionPipeline builds its own broadphase to register the cloth "
                "filter callback; pass a string mode ('nxn' or 'sap') instead of a "
                "prebuilt instance."
            )
        if broad_phase == "nxn":
            bp_cls = BroadPhaseAllPairs
        elif broad_phase == "sap":
            bp_cls = BroadPhaseSAP
        else:
            raise ValueError(f"PhoenxCollisionPipeline broad_phase must be 'nxn' or 'sap'; got {broad_phase!r}")
        bp = bp_cls(
            ext_int["shape_world"],
            shape_flags=ext_int["shape_flags"],
            device=device,
            filter_func=phoenx_broadphase_filter,
        )

        # ---- Build narrowphase with extended AABB arrays --------------
        if shape_pairs_max is None:
            # NXN worst case across S+T shapes.  The phoenx filter
            # discards cloth-cloth pairs sharing a node so the actual
            # candidate count is much smaller, but the broadphase
            # buffer must still bound the unfiltered count.
            shape_pairs_max = max(1, total * (total - 1) // 2)

        if narrow_phase is None:
            # Mesh / heightfield support: derive from the rigid model's
            # shape types. Cloth shapes are TRIANGLE only, never
            # MESH/HFIELD, so they don't influence the flag.
            shape_types_np = model.shape_type.numpy()
            has_meshes = bool((shape_types_np == int(GeoType.MESH)).any())
            has_heightfields = bool((shape_types_np == int(GeoType.HFIELD)).any())
            np_kwargs = {
                "max_candidate_pairs": shape_pairs_max,
                "device": device,
                "shape_aabb_lower": ext_aabb_lower,
                "shape_aabb_upper": ext_aabb_upper,
                "contact_writer_warp_func": write_contact,
                "shape_voxel_resolution": getattr(model, "_shape_voxel_resolution", None),
                "has_meshes": has_meshes,
                "has_heightfields": has_heightfields,
                "deterministic": kwargs.get("deterministic", False)
                or kwargs.get("contact_matching", "disabled") != "disabled",
                "contact_max": kwargs.get("rigid_contact_max"),
                "verify_buffers": kwargs.get("verify_buffers", True),
            }
            narrow_phase = NarrowPhase(**np_kwargs)

        # ---- Filter data carrier --------------------------------------
        filter_data = PhoenxBroadphaseFilterData()
        filter_data.num_rigid_shapes = wp.int32(S)
        filter_data.constraints = constraints
        filter_data.cloth_cid_offset = wp.int32(cloth_cid_offset)

        # ---- Hand off to the standard pipeline ------------------------
        super().__init__(
            model,
            broad_phase=bp,
            narrow_phase=narrow_phase,
            shape_pairs_max=shape_pairs_max,
            extra_shape_count=T,
            **kwargs,
        )

        # ---- Override per-shape arrays with extended versions ---------
        # The base class read ``model.shape_*`` into ``self.shape_*``;
        # swap in the extended copies so the rest of collide() picks
        # them up.
        for k, v in ext_int.items():
            setattr(self, k, v)
        for k, v in ext_float.items():
            setattr(self, k, v)
        self.shape_transform = ext_shape_transform
        self.shape_source_ptr = ext_shape_source_ptr
        self.shape_collision_aabb_lower = ext_shape_collision_aabb_lower
        self.shape_collision_aabb_upper = ext_shape_collision_aabb_upper
        self.shape_auxiliary = ext_shape_auxiliary
        self._broadphase_filter_data = filter_data

        # ---- Cloth-side state for the per-step hook -------------------
        self._cloth_S = S
        self._cloth_T = T
        self._cloth_constraints = constraints
        self._cloth_cid_offset = int(cloth_cid_offset)
        self._cloth_num_bodies = int(num_bodies)
        self._cloth_particle_q = particle_q
        self._cloth_particle_radius = particle_radius
        self._cloth_extra_margin = float(cloth_extra_margin)
        self._cloth_shape_data_margin = float(cloth_shape_data_margin)

    def _pre_broadphase_hook(self, state, contacts) -> None:
        if self._cloth_T == 0:
            return
        # Single fused per-cloth-triangle pass that fills both the
        # shape descriptor (shape_type / shape_transform / shape_data
        # / shape_auxiliary) and the broadphase AABB slots in one
        # kernel launch.  The rigid AABB prefix [0, S) was already
        # populated by ``compute_shape_aabbs`` upstream; we only
        # touch slots [S, S+T).
        launch_cloth_triangle_stamp(
            particle_q=self._cloth_particle_q,
            particle_radius=self._cloth_particle_radius,
            constraints=self._cloth_constraints,
            cloth_cid_offset=self._cloth_cid_offset,
            num_bodies=self._cloth_num_bodies,
            num_cloth_triangles=self._cloth_T,
            base_offset=self._cloth_S,
            aabb_extra_margin=self._cloth_extra_margin,
            shape_data_margin=self._cloth_shape_data_margin,
            shape_type=self.shape_type,
            shape_transform=self.geom_transform,
            shape_data=self.geom_data,
            shape_auxiliary=self.shape_auxiliary,
            aabb_lower=self.narrow_phase.shape_aabb_lower,
            aabb_upper=self.narrow_phase.shape_aabb_upper,
            device=self.device,
        )
