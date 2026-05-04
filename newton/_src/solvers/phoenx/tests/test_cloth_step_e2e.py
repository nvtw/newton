# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end cloth integration via :meth:`PhoenXWorld.step`.

Validates the per-substep particle access-mode pattern:

* Substep entry transition (Velocity-level -> Position-level): the
  predict kernel applies gravity to particle velocity, snapshots
  the pre-predict position into ``position_prev_substep``, and
  advances ``position`` by ``velocity * dt``. The cloth iterate
  consumes the predicted positions.
* Substep exit transition (Position-level -> Velocity-level): the
  recover kernel writes
  ``velocity = (position - position_prev_substep) * inv_dt`` so
  the next substep starts from a consistent velocity-level state.

Mirrors the C# ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
pattern from ``BodyTypes.cs``. The cloth iterate itself doesn't
know about access modes; the position / velocity duality is
managed at the substep boundaries.

CUDA-only.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-step E2E requires CUDA")
class TestClothStepE2E(unittest.TestCase):
    """A pinned cloth strip falls under gravity through
    :meth:`PhoenXWorld.step`. Pinned corners stay put;
    free corners drop; nothing NaNs."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        self.dim_x = 4
        self.dim_y = 4
        self.cell = 0.25
        self.mass = 0.05
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.mass,
            fix_left=True,  # pin the entire left edge
            tri_ke=1.0e6,
            tri_ka=1.0e6,
        )
        self.model = builder.finalize(device=self.device)
        self.particle_count = (self.dim_x + 1) * (self.dim_y + 1)
        self.tri_count = self.dim_x * self.dim_y * 2

    def _make_world(self) -> PhoenXWorld:
        bodies = body_container_zeros(1, device=self.device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=self.tri_count,
            device=self.device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=self.particle_count,
            num_cloth_triangles=self.tri_count,
            num_worlds=1,
            step_layout="single_world",
            substeps=4,
            solver_iterations=8,
        )
        # Earth gravity; PhoenX uses -z up convention here.
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        return world

    def test_pinned_strip_drops_under_gravity(self) -> None:
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)

        positions_before = world.particles.position.numpy().copy()
        # Pinned indices = left edge (x == 0), one row per y.
        pinned_indices = [j * (self.dim_x + 1) for j in range(self.dim_y + 1)]
        # Free corner = top-right (x == dim_x, y == dim_y).
        free_corner = self.dim_y * (self.dim_x + 1) + self.dim_x

        # Step several frames at 60 Hz.
        for _ in range(20):
            world.step(1.0 / 60.0, contacts=None)

        positions_after = world.particles.position.numpy()

        # No NaN / Inf anywhere.
        self.assertTrue(np.all(np.isfinite(positions_after)), "non-finite particle position after step")

        # Pinned corners haven't moved (their inv_mass==0 -- the
        # predict kernel's early-return covers this; the iterate
        # also leaves them alone via the inv_mass weighting).
        np.testing.assert_allclose(positions_after[pinned_indices], positions_before[pinned_indices], atol=1e-4)

        # Free corner has dropped meaningfully under gravity.
        # A perfectly inextensible strip pinned on one side and
        # released will swing; we just check (a) it dropped (z went
        # down), (b) it didn't fall arbitrarily far (the cloth held
        # together).
        z_drop = positions_before[free_corner, 2] - positions_after[free_corner, 2]
        self.assertGreater(z_drop, 0.05, f"free corner barely dropped (z_drop={z_drop:.4f})")
        self.assertLess(z_drop, 1.0, f"free corner fell too far (z_drop={z_drop:.4f}); cloth came apart")


if __name__ == "__main__":
    unittest.main()
