# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end RT-contact regression: corner-pinned cloth catching a
falling rigid cube.

This is the integration counterpart to
:mod:`newton._src.solvers.phoenx.tests.test_unified_contact` (which
exercises RT/TT warm-start at the kernel level) and
:mod:`newton._src.solvers.phoenx.tests.test_collision_pipeline_external_aabbs`
(which checks the unified-array wiring).  Here we drive the full
:class:`PhoenXWorld` step path under graph capture and verify the
emergent behaviour: gravity pulls the cube into the cloth, the
cloth deflects, the cube settles instead of tunnelling, and the
pinned corners stay put.

The test exercises the *clean internal* PhoenX API (no
:class:`SolverPhoenX` adapter) so it directly covers
:meth:`PhoenXWorld.setup_cloth_collision_pipeline` and the
``external_aabb_state`` auto-collide path inside
:meth:`PhoenXWorld.step`.

CUDA-only by Newton convention; PhoenX tests rely on graph capture
to keep launch overhead bearable.
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
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

# Test scene parameters (smaller / faster than the example).
CLOTH_DIM = 8
CLOTH_CELL = 0.05
CLOTH_MASS_PER_PARTICLE = 0.01
CLOTH_Z = 1.0
YOUNGS_MODULUS = 5.0e8
POISSON_RATIO = 0.3
CLOTH_MARGIN = 0.005

CUBE_HE = 0.06
CUBE_DENSITY = 60.0
CUBE_SPAWN_Z = 1.25
CUBE_FRICTION = 0.6

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 8
SOLVER_ITERATIONS = 12
NUM_FRAMES = 15


def _corner_indices(dim_x: int, dim_y: int) -> tuple[int, int, int, int]:
    nx = dim_x + 1
    return (0, dim_x, dim_y * nx, dim_y * nx + dim_x)


def _build_model(device: wp.Device) -> tuple[newton.Model, tuple[int, ...], int]:
    builder = newton.ModelBuilder()
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(YOUNGS_MODULUS, POISSON_RATIO)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * CLOTH_DIM * CLOTH_CELL, -0.5 * CLOTH_DIM * CLOTH_CELL, CLOTH_Z),
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
    corners = _corner_indices(CLOTH_DIM, CLOTH_DIM)
    for c in corners:
        builder.particle_mass[c] = 0.0
        builder.particle_flags[c] = builder.particle_flags[c] & ~ParticleFlags.ACTIVE

    cube = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, CUBE_SPAWN_Z), q=wp.quat_identity()),
    )
    builder.add_shape_box(
        cube,
        hx=CUBE_HE,
        hy=CUBE_HE,
        hz=CUBE_HE,
        cfg=newton.ModelBuilder.ShapeConfig(density=CUBE_DENSITY, mu=CUBE_FRICTION),
    )
    builder.gravity = -GRAVITY
    return builder.finalize(device=device), corners, cube


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth + rigid drop test requires CUDA")
class TestClothRigidDrop(unittest.TestCase):
    """Corner-pinned cloth catches a small falling cube; verify the
    cube settles, the cloth doesn't explode, and the pinned corners
    don't drift. Drives :class:`PhoenXWorld` directly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_device()
        cls.model, cls.corners, cls.cube_body = _build_model(cls.device)

        # FK once so ``model.body_q`` reflects the spawn pose queued
        # via ``add_body``'s free joint.
        if int(cls.model.body_count) > 0 and int(cls.model.joint_count) > 0:
            tmp_state = cls.model.state()
            newton.eval_fk(cls.model, cls.model.joint_q, cls.model.joint_qd, tmp_state)
            cls.model.body_q.assign(tmp_state.body_q)
            cls.model.body_qd.assign(tmp_state.body_qd)

        # ---- Build PhoenX body container ------------------------------
        num_phoenx_bodies = int(cls.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=cls.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=cls.device,
            ),
        )
        wp.launch(
            _init_phoenx_bodies_kernel,
            dim=cls.model.body_count,
            inputs=[
                cls.model.body_q,
                cls.model.body_qd,
                cls.model.body_com,
                cls.model.body_inv_mass,
                cls.model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=cls.device,
        )
        cls.bodies = bodies

        # ---- Build PhoenXWorld ---------------------------------------
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
            rigid_contact_max=max(1024, 4 * CLOTH_DIM * CLOTH_DIM),
            step_layout="single_world",
            device=cls.device,
        )
        cls.world.populate_cloth_triangles_from_model(cls.model)

        # ---- Cloth-aware collision pipeline ---------------------------
        cls.collision_pipeline = cls.world.setup_cloth_collision_pipeline(
            cls.model, cloth_margin=CLOTH_MARGIN
        )
        cls.contacts = cls.collision_pipeline.contacts()
        shape_body_np = cls.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        cls._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=cls.device)

        # ---- Newton state for the auto-collide hand-off ---------------
        cls.state = cls.model.state()
        cls.initial_particle_q = cls.model.particle_q.numpy().copy()

        cls.frame_dt = 1.0 / FPS

        # Warm-up + graph capture.
        cls._simulate_one_frame_uncaptured()
        with wp.ScopedCapture(device=cls.device) as cap:
            cls._simulate_one_frame_uncaptured()
        cls.graph = cap.graph

    @classmethod
    def _sync_phoenx_to_newton(cls) -> None:
        n = cls.model.body_count
        if n == 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                cls.bodies.position[1 : 1 + n],
                cls.bodies.orientation[1 : 1 + n],
                cls.bodies.velocity[1 : 1 + n],
                cls.bodies.angular_velocity[1 : 1 + n],
                cls.model.body_com,
            ],
            outputs=[cls.state.body_q, cls.state.body_qd],
            device=cls.device,
        )

    @classmethod
    def _simulate_one_frame_uncaptured(cls) -> None:
        cls._sync_phoenx_to_newton()
        wp.copy(cls.state.particle_q, cls.world.particles.position)
        wp.copy(cls.state.particle_qd, cls.world.particles.velocity)
        cls.world.step(
            dt=cls.frame_dt,
            contacts=cls.contacts,
            shape_body=cls._shape_body,
            external_aabb_state=cls.state,
        )

    def test_cube_rests_on_cloth(self) -> None:
        """Drive the captured graph for ``NUM_FRAMES`` frames; verify
        the cube is below its spawn height but above a tunnel-through
        floor, the pinned corners haven't moved, and the contact
        buffer holds a positive count."""
        # NUM_FRAMES - 2 because setUpClass already ran two warm-up
        # frames (the uncaptured warm-up plus the captured one).
        for _ in range(NUM_FRAMES - 2):
            wp.capture_launch(self.graph)

        # Bring final PhoenX body / particle state back into the
        # Newton ``State`` for assertions. The captured graph already
        # syncs at the start of the next frame; we re-run the sync
        # explicitly so the final state is what we just simulated.
        self._sync_phoenx_to_newton()
        wp.copy(self.state.particle_q, self.world.particles.position)

        positions = self.state.particle_q.numpy()
        body_q = self.state.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(positions)), "non-finite particle position")
        self.assertTrue(np.all(np.isfinite(body_q)), "non-finite body transform")

        # Pinned corners stayed put.
        pinned_drift = np.linalg.norm(
            positions[list(self.corners)] - self.initial_particle_q[list(self.corners)],
            axis=1,
        )
        self.assertLess(
            float(pinned_drift.max()),
            1.0e-3,
            f"pinned corner drifted: max={pinned_drift.max():.4f} m",
        )

        # Cube has fallen.
        cube_z = float(body_q[self.cube_body, 2])
        self.assertLess(cube_z, CUBE_SPAWN_Z - 1.0e-3, f"cube did not fall (z={cube_z:.4f})")

        # Cube hasn't tunnelled. Allow up to ~3 cube half-extents
        # below the pinned cloth height to account for sag under load.
        floor = CLOTH_Z - 3.0 * CUBE_HE
        self.assertGreater(cube_z, floor, f"cube fell through cloth (z={cube_z:.4f}, floor={floor:.4f})")

        # The cloth produced rigid contacts at some point during the
        # run -- the contact buffer should hold non-zero rigid contact
        # pairs after the latest step.
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(
            contact_count,
            0,
            "no contacts produced -- broad/narrow phase missed the cube/cloth overlap",
        )


if __name__ == "__main__":
    unittest.main()
