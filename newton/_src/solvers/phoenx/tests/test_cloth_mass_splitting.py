# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression: cloth + mass splitting must not crumple on first contact.

Drops a free rigid cube onto a pinned-edge cloth grid sitting just
above a static ground plane. Same scene run twice -- once with
``mass_splitting=False`` (baseline that already works) and once with
``mass_splitting=True`` (the broken case).

Pre-fix behaviour: with mass splitting enabled, cloth iterate writes
PBD position corrections into per-partition copy-state slots but the
copy-state writeback (``_copy_state_into_rigids_kernel``) only writes
slot[0].velocity back to ``particles.velocity``; positions are
dropped. ``cloth_recover_kernel`` then reads the stale
``particles.position`` and computes ``v = (pos - pos_prev) / dt ~= 0``.
On first cube-cloth contact the cloth crumples and goes non-finite.

Post-fix expectation: both variants converge to a comparable stable
end state. The asserts below characterise "stable" as:

* No NaN/inf in cloth positions
* Pinned particles haven't drifted
* Max edge stretch (current / rest length) stays bounded -- the
  crumple mode pushes this above ~5x on at least one edge
* Cloth bounding box doesn't collapse along any axis

CUDA-only. Graph-captured. Two ~5 s tests.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

CLOTH_DIM_X = 16  # 16x8 cloth -> 17x9 = 153 particles, 256 tris
CLOTH_DIM_Y = 8
CELL = 0.1  # 0.1 m per cell -> 1.6 x 0.8 m cloth
CLOTH_Z = 2.0
CUBE_DROP_DZ = 1.0  # cube starts 1 m above cloth (impact speed ~4.4 m/s)
CUBE_HALF_SIDE = 0.2
PARTICLE_MASS = 0.05
CUBE_MASS = 1.0
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
N_FRAMES = 60  # 1 s @ 60 fps -- impact happens around frame ~30
DT = 1.0 / 60.0


def _build_scene(*, mass_splitting: bool, device):
    """Build the cube-on-pinned-cloth scene; return (world, model, state, pipeline, contacts, cube_body)."""
    youngs, poisson = 5.0e8, 0.3
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(youngs, poisson)

    builder = newton.ModelBuilder()
    # Static ground plane at z=0 (well below the cloth).
    builder.add_ground_plane(height=0.0)
    # Free rigid cube just above the cloth.
    cube_body = builder.add_body(
        xform=wp.transform(
            p=wp.vec3(0.0, 0.0, CLOTH_Z + CUBE_DROP_DZ),
            q=wp.quat_identity(),
        ),
        mass=CUBE_MASS,
    )
    builder.add_shape_box(
        cube_body,
        hx=CUBE_HALF_SIDE,
        hy=CUBE_HALF_SIDE,
        hz=CUBE_HALF_SIDE,
    )
    # Cloth lying flat in XY at z = CLOTH_Z. Pinned on the two short
    # edges only (fix_left + fix_right) so it can sag freely in the
    # middle where the cube lands -- mirroring the example_cloth_hanging
    # scene but with a more compact horizontal layout.
    cloth_origin = wp.vec3(-CLOTH_DIM_X * CELL * 0.5, -CLOTH_DIM_Y * CELL * 0.5, CLOTH_Z)
    builder.add_cloth_grid(
        pos=cloth_origin,
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=CLOTH_DIM_X,
        dim_y=CLOTH_DIM_Y,
        cell_x=CELL,
        cell_y=CELL,
        mass=PARTICLE_MASS,
        fix_left=True,
        fix_right=True,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        particle_radius=0.04,
    )
    model = builder.finalize(device=device)

    num_phoenx_bodies = int(model.body_count) + 1
    bodies = body_container_zeros(num_phoenx_bodies, device=device)
    bodies.orientation.assign(
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
            dtype=wp.quatf,
            device=device,
        )
    )
    state_init = model.state()
    if int(model.body_count) > 0:
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=int(model.body_count),
            inputs=[
                model.body_q,
                state_init.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
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
            device=device,
        )

    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0,
        num_cloth_triangles=int(model.tri_count),
        device=device,
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=int(model.particle_count),
        num_cloth_triangles=int(model.tri_count),
        num_worlds=1,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        rigid_contact_max=4096,
        step_layout="single_world",
        mass_splitting=mass_splitting,
        max_colored_partitions=12,
        device=device,
    )
    world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
    world.populate_cloth_triangles_from_model(model)
    pipeline = world.setup_cloth_collision_pipeline(
        model, cloth_thickness=0.005, cloth_gap=0.010, rigid_contact_max=4096
    )

    state = model.state()
    contacts = pipeline.contacts()
    return world, model, state, pipeline, contacts, cube_body


def _sync_newton_to_phoenx(world, model, state, device):
    n = int(model.body_count)
    if n == 0:
        return
    wp.launch(
        newton_to_phoenx_kernel,
        dim=n,
        inputs=[state.body_q, state.body_qd, model.body_com],
        outputs=[
            world.bodies.position[1 : 1 + n],
            world.bodies.orientation[1 : 1 + n],
            world.bodies.velocity[1 : 1 + n],
            world.bodies.angular_velocity[1 : 1 + n],
        ],
        device=device,
    )


def _sync_phoenx_to_newton(world, model, state, device):
    n = int(model.body_count)
    if n == 0:
        return
    wp.launch(
        phoenx_to_newton_kernel,
        dim=n,
        inputs=[
            world.bodies.position[1 : 1 + n],
            world.bodies.orientation[1 : 1 + n],
            world.bodies.velocity[1 : 1 + n],
            world.bodies.angular_velocity[1 : 1 + n],
            model.body_com,
        ],
        outputs=[state.body_q, state.body_qd],
        device=device,
    )
    wp.copy(state.particle_q, world.particles.position)
    wp.copy(state.particle_qd, world.particles.velocity)


def _step_once(world, model, state, contacts, device):
    _sync_newton_to_phoenx(world, model, state, device)
    world.collide(state, contacts)
    world.step(DT, contacts=contacts)
    _sync_phoenx_to_newton(world, model, state, device)


def _edge_stretch_ratios(positions: np.ndarray, rest_positions: np.ndarray, tri_indices: np.ndarray) -> np.ndarray:
    """For every triangle edge, return current length / rest length."""
    ratios = []
    for tri in tri_indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for i, j in ((a, b), (b, c), (c, a)):
            rest = np.linalg.norm(rest_positions[i] - rest_positions[j])
            curr = np.linalg.norm(positions[i] - positions[j])
            if rest > 1e-9:
                ratios.append(curr / rest)
    return np.asarray(ratios, dtype=np.float64)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cloth + mass splitting test is CUDA-only.",
)
class TestClothMassSplitting(unittest.TestCase):
    """Drive a falling rigid cube onto a pinned cloth strip; assert that
    cloth stays well-behaved with and without mass splitting."""

    def _run(self, *, mass_splitting: bool):
        device = wp.get_preferred_device()
        world, model, state, pipeline, contacts, cube_body = _build_scene(mass_splitting=mass_splitting, device=device)
        rest_positions = model.particle_q.numpy().copy()

        # fix_left + fix_right pins the i=0 and i=dim_x columns of the
        # (dim_x+1) x (dim_y+1) grid.
        nx1 = CLOTH_DIM_X + 1
        ny1 = CLOTH_DIM_Y + 1
        pinned = []
        for j in range(ny1):
            pinned.append(j * nx1 + 0)
            pinned.append(j * nx1 + (nx1 - 1))
        pinned_indices = np.asarray(pinned, dtype=np.int64)

        # Warm-up: 1 un-captured step builds buffers + caches kernels.
        _step_once(world, model, state, contacts, device)

        with wp.ScopedCapture(device=device) as cap:
            _step_once(world, model, state, contacts, device)
        graph = cap.graph

        for _ in range(N_FRAMES - 1):
            wp.capture_launch(graph)
        wp.synchronize_device(device)

        positions = state.particle_q.numpy()
        cube_q = state.body_q.numpy()

        import sys

        bbox = positions.max(axis=0) - positions.min(axis=0)
        sys.stderr.write(
            f"\n[diag mass_splitting={mass_splitting}] cube_z={float(cube_q[cube_body, 2]):.4f} "
            f"cloth_z=[{float(positions[:, 2].min()):.4f},{float(positions[:, 2].mean()):.4f},{float(positions[:, 2].max()):.4f}] "
            f"bbox_z={float(bbox[2]):.4f}\n"
        )

        # 1. No NaN / inf anywhere -- this catches the crumple-then-explode mode.
        self.assertTrue(
            np.all(np.isfinite(positions)),
            f"non-finite particle position with mass_splitting={mass_splitting}: "
            f"first non-finite index = {int(np.argmax(~np.isfinite(positions).all(axis=1)))}",
        )
        self.assertTrue(
            np.all(np.isfinite(cube_q)),
            f"non-finite cube transform with mass_splitting={mass_splitting}",
        )

        # 2. Pinned particles barely drifted. PBD has zero compliance at
        # the pin so this should be < a few millimetres even with
        # contact perturbations.
        pinned_drift = np.linalg.norm(positions[pinned_indices] - rest_positions[pinned_indices], axis=1)
        self.assertLess(
            float(pinned_drift.max()),
            5.0e-3,
            f"pinned particle drifted with mass_splitting={mass_splitting}: max drift = {pinned_drift.max():.4f} m",
        )

        # 3. Cloth bounding box is sane: every axis between 1 cm and a
        # few times the cloth side length. A crumpled cloth collapses
        # to ~zero width on one axis; an exploded cloth expands to
        # >> cloth_side.
        cloth_side = max(CLOTH_DIM_X, CLOTH_DIM_Y) * CELL
        bbox = positions.max(axis=0) - positions.min(axis=0)
        for axis, label in enumerate("xyz"):
            self.assertGreater(
                float(bbox[axis]),
                0.01,
                f"cloth collapsed on {label}-axis with mass_splitting={mass_splitting}: bbox = {bbox.tolist()}",
            )
            self.assertLess(
                float(bbox[axis]),
                cloth_side * 5.0,
                f"cloth blew up on {label}-axis with mass_splitting={mass_splitting}: bbox = {bbox.tolist()}",
            )

        # 4. No edge stretched / compressed wildly. Rest ratio = 1.0;
        # PBD with high Young's modulus should keep this within ~+/- 50%
        # under contact loads. Crumple shows up here first.
        tri_indices = model.tri_indices.numpy()
        ratios = _edge_stretch_ratios(positions, rest_positions, tri_indices)
        self.assertGreater(
            float(ratios.min()),
            0.3,
            f"cloth edge compressed to {ratios.min():.3f}x rest with "
            f"mass_splitting={mass_splitting} (crumple signature)",
        )
        self.assertLess(
            float(ratios.max()),
            3.0,
            f"cloth edge stretched to {ratios.max():.3f}x rest with mass_splitting={mass_splitting}",
        )

        # 5. Cube must rest ON the cloth, not fall through. With the
        # cloth pinned on all four corners and the cube dropped from
        # 20 cm above, the cube should be above the cloth's mean z
        # (with some sag). If mass splitting drops the cloth's PBD
        # position writeback, the cube falls straight through the
        # cloth and lands on / near the ground.
        cube_z = float(cube_q[cube_body, 2])
        cloth_mean_z = float(positions[:, 2].mean())
        self.assertGreater(
            cube_z,
            cloth_mean_z - CUBE_HALF_SIDE,
            f"cube fell through the cloth with mass_splitting={mass_splitting}: "
            f"cube_z = {cube_z:.4f}, cloth mean z = {cloth_mean_z:.4f}",
        )
        self.assertGreater(
            cube_z,
            CUBE_HALF_SIDE + 0.05,
            f"cube ended up on / near the ground with mass_splitting={mass_splitting}: cube_z = {cube_z:.4f}",
        )

    def test_mass_splitting_off_baseline(self):
        self._run(mass_splitting=False)

    def test_mass_splitting_on_regression(self):
        self._run(mass_splitting=True)


if __name__ == "__main__":
    unittest.main()
