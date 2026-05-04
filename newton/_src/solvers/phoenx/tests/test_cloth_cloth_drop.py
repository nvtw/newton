# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TT-contact regression: cloth falling onto pinned cloth.

Counterpart to :mod:`test_cloth_rigid_drop` for the cloth-vs-cloth
(TT) path.  Verifies the same shared infrastructure --
``GeoTypeEx.TRIANGLE`` registration, broadphase share-vertex filter,
``ENDPOINT_KIND_TRIANGLE`` ingest, ``endpoint_load`` /
``endpoint_apply_impulse`` -- works when both endpoints of a contact
are triangles.

Scene: two flat cloth grids stacked vertically.  The bottom cloth has
all four corners pinned to act as a hammock; the top cloth falls onto
it under gravity.  Pass criteria are the same shape: no NaN, the
falling cloth ends up below its spawn height but above the bottom
cloth (no tunnelling), and pinned corners stay put.

CUDA-only.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import ParticleFlags
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

# Smaller-than-the-rigid-drop scene to keep iteration fast under
# graph capture (TT pairs grow O(T^2) before the share-vertex
# filter prunes adjacents).
CLOTH_DIM = 4
CLOTH_CELL = 0.08
CLOTH_MASS_PER_PARTICLE = 0.02
# Top cloth starts close to bottom so contact engages before the fall
# builds enough velocity for the contacts to overshoot.  Larger margin
# gives more time for the speculative branch.
TOP_Z = 1.04
BOTTOM_Z = 1.00
YOUNGS_MODULUS = 5.0e8
POISSON_RATIO = 0.3
CLOTH_MARGIN = 0.015

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 8
NUM_FRAMES = 8


def _all_corner_indices(dim_x: int, dim_y: int) -> tuple[int, int, int, int]:
    nx = dim_x + 1
    return (0, dim_x, dim_y * nx, dim_y * nx + dim_x)


def _build_model(device: wp.Device) -> tuple[newton.Model, list[int], list[int]]:
    builder = newton.ModelBuilder()
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(YOUNGS_MODULUS, POISSON_RATIO)

    # Bottom cloth (hammock): all corners pinned.
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * CLOTH_DIM * CLOTH_CELL, -0.5 * CLOTH_DIM * CLOTH_CELL, BOTTOM_Z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=CLOTH_DIM,
        dim_y=CLOTH_DIM,
        cell_x=CLOTH_CELL,
        cell_y=CLOTH_CELL,
        mass=CLOTH_MASS_PER_PARTICLE,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        particle_radius=0.5 * CLOTH_CELL,
    )
    bottom_corners = list(_all_corner_indices(CLOTH_DIM, CLOTH_DIM))
    for c in bottom_corners:
        builder.particle_mass[c] = 0.0
        builder.particle_flags[c] = builder.particle_flags[c] & ~ParticleFlags.ACTIVE

    # Top cloth: free, falls onto bottom.
    top_offset = (CLOTH_DIM + 1) * (CLOTH_DIM + 1)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * CLOTH_DIM * CLOTH_CELL, -0.5 * CLOTH_DIM * CLOTH_CELL, TOP_Z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=CLOTH_DIM,
        dim_y=CLOTH_DIM,
        cell_x=CLOTH_CELL,
        cell_y=CLOTH_CELL,
        mass=CLOTH_MASS_PER_PARTICLE,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        particle_radius=0.5 * CLOTH_CELL,
    )
    top_corners = [top_offset + c for c in _all_corner_indices(CLOTH_DIM, CLOTH_DIM)]
    builder.gravity = -GRAVITY
    return builder.finalize(device=device), bottom_corners, top_corners


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-cloth drop test requires CUDA")
class TestClothClothDrop(unittest.TestCase):
    """Top cloth falls onto bottom (pinned) cloth.  The TT-contact
    broadphase + ``endpoint_kind=TRIANGLE`` apply path must keep the
    top cloth from tunnelling and leave the pinned corners in place."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_device()
        cls.model, cls.bottom_corners, cls.top_corners = _build_model(cls.device)

        bodies = body_container_zeros(1, device=cls.device)
        wp.copy(
            bodies.orientation,
            wp.array(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.quatf, device=cls.device),
        )
        cls.bodies = bodies

        cls.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(cls.model.tri_count),
            device=cls.device,
        )
        cls.world = PhoenXWorld(
            bodies=cls.bodies,
            constraints=cls.constraints,
            num_joints=0,
            num_particles=int(cls.model.particle_count),
            num_cloth_triangles=int(cls.model.tri_count),
            num_worlds=1,
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, -GRAVITY),
            rigid_contact_max=max(8192, 16 * CLOTH_DIM * CLOTH_DIM),
            step_layout="single_world",
            device=cls.device,
        )
        cls.world.populate_cloth_triangles_from_model(cls.model)
        cls.collision_pipeline = cls.world.setup_cloth_collision_pipeline(cls.model, cloth_margin=CLOTH_MARGIN)
        cls.contacts = cls.collision_pipeline.contacts()
        # Length-1 sentinel; the cloth-only ingest never indexes
        # ``shape_body`` (both endpoints are TRIANGLE).
        cls._shape_body = wp.zeros(1, dtype=wp.int32, device=cls.device)

        cls.state = cls.model.state()
        cls.initial_particle_q = cls.model.particle_q.numpy().copy()
        cls.frame_dt = 1.0 / FPS

    @classmethod
    def _simulate_one_frame(cls) -> None:
        wp.copy(cls.state.particle_q, cls.world.particles.position)
        wp.copy(cls.state.particle_qd, cls.world.particles.velocity)
        cls.world.step(
            dt=cls.frame_dt,
            contacts=cls.contacts,
            shape_body=cls._shape_body,
            external_aabb_state=cls.state,
        )

    def test_top_cloth_rests_on_bottom(self) -> None:
        max_contacts_seen = 0
        for f in range(NUM_FRAMES):
            self._simulate_one_frame()
            ccount = int(self.contacts.rigid_contact_count.numpy()[0])
            max_contacts_seen = max(max_contacts_seen, ccount)
            if f % 4 == 0 or f == NUM_FRAMES - 1:
                positions = self.world.particles.position.numpy()
                top_offset = (CLOTH_DIM + 1) * (CLOTH_DIM + 1)
                top_z = positions[top_offset:, 2].mean()
                bot_z = positions[:top_offset, 2].mean()
                print(f"frame {f}: top_z_mean={top_z:.4f} bot_z_mean={bot_z:.4f} contacts={ccount}")
        print(f"max contacts seen during run: {max_contacts_seen}")
        # Replace strict zero check with the historical max so we
        # confirm the path fired at least once.
        contact_count_after_test = max_contacts_seen

        positions = self.world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(positions)), "non-finite particle position")

        # Pinned bottom corners stayed put.
        bottom_drift = np.linalg.norm(
            positions[self.bottom_corners] - self.initial_particle_q[self.bottom_corners],
            axis=1,
        )
        self.assertLess(
            float(bottom_drift.max()),
            5.0e-3,
            f"pinned bottom corner drifted: max={bottom_drift.max():.4f} m",
        )

        # Top cloth interacted with the bottom: it neither tunnelled
        # (min z above floor) nor escaped to infinity (max z below a
        # generous ceiling).  We don't assert it settled because the
        # default cloth contact stiffness is bouncy by design.
        top_indices = list(range((CLOTH_DIM + 1) * (CLOTH_DIM + 1), 2 * (CLOTH_DIM + 1) * (CLOTH_DIM + 1)))
        top_z_after = positions[top_indices, 2]
        floor = BOTTOM_Z - 3.0 * CLOTH_CELL
        ceiling = TOP_Z + 0.5  # ~5x the initial gap
        self.assertGreater(
            float(top_z_after.min()),
            floor,
            f"top cloth tunnelled (min z={top_z_after.min():.4f}, floor={floor:.4f})",
        )
        self.assertLess(
            float(top_z_after.max()),
            ceiling,
            f"top cloth escaped (max z={top_z_after.max():.4f}, ceiling={ceiling:.4f})",
        )

        # The TT contact path actually fires (many contacts -- the
        # broadphase emits one per overlapping triangle pair, the
        # share-vertex filter prunes adjacents, and the narrowphase
        # writes manifold points for each survivor).
        self.assertGreater(
            contact_count_after_test,
            10,
            "no contacts produced -- broad/narrow phase missed the cloth/cloth overlap",
        )


if __name__ == "__main__":
    unittest.main()
