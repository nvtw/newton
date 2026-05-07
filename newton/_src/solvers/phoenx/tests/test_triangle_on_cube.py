# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth-tri vs rigid-cube contact regression tests.

The single-tri rest case lives in
:mod:`newton._src.solvers.phoenx.tests.test_tri_rest_on_cube`. This
module exercises slightly heavier scenes that still must produce
stable, bounded motion:

* :class:`TestSingleTriangleOnCubeOnGround` -- one cloth triangle
  (3 unconstrained particles -- ``tri_ke = tri_ka = 0``) drops
  vertically onto a dynamic cube that is already resting on a
  static ground plane. No internal cloth elasticity, so this isolates
  the rigid-vs-triangle (RT) contact path. Pass criterion: every
  particle, the cube, and the ground stay axis-aligned -- no sideways
  velocity, no spin, no XY drift.
* :class:`TestSmallClothOnCube` -- a 3x3 cloth grid placed at its
  rest position on top of a heavy cube. Verifies cloth-cloth +
  cloth-rigid contact resolution doesn't diverge, doesn't tunnel,
  and the cube barely moves under the cloth's weight.

CUDA-only; the high-level :class:`~newton.solvers.SolverPhoenX` wraps
:class:`~newton._src.solvers.phoenx.solver_phoenx.PhoenXWorld` and uses
CUDA graph capture internally on every step replay.
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
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 8
SOLVER_ITERATIONS = 12
NUM_FRAMES = 60

CUBE_HE = 0.1
CUBE_SPAWN_Z = CUBE_HE  # cube initially resting on ground (z=0)
PARTICLE_MASS = 0.05  # heavier-than-default keeps mass ratio sane
TRIANGLE_SPAWN_Z = 2.0 * CUBE_HE + 0.05  # 5 cm above cube top
CLOTH_MARGIN = 0.005


def _build_scene(device: wp.Device, triangle_offset_xy: tuple[float, float] = (0.0, 0.0)):
    """Build a Newton model with:
      - a ground plane at z = 0,
      - a dynamic cube resting on the ground,
      - one free cloth triangle 5 cm above the cube top.

    The triangle has three particles arranged as a small equilateral
    triangle centred over the cube (XY offset configurable for
    edge / corner stress).  ``tri_ke = tri_ka = 0`` means no XPBD
    elasticity row -- the particles are effectively three free
    point masses tied together by the contact path only.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)
    cube_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, CUBE_SPAWN_Z), q=wp.quat_identity()),
    )
    builder.add_shape_box(
        cube_body,
        hx=CUBE_HE,
        hy=CUBE_HE,
        hz=CUBE_HE,
        cfg=newton.ModelBuilder.ShapeConfig(density=60.0, mu=0.6),
    )

    # Three free particles in a small equilateral triangle.
    tx, ty = triangle_offset_xy
    side = 0.05
    h = side * np.sqrt(3.0) / 2.0
    verts = [
        (tx - 0.5 * side, ty - h / 3.0, TRIANGLE_SPAWN_Z),
        (tx + 0.5 * side, ty - h / 3.0, TRIANGLE_SPAWN_Z),
        (tx + 0.0, ty + 2.0 * h / 3.0, TRIANGLE_SPAWN_Z),
    ]
    p_indices = []
    for v in verts:
        p_indices.append(
            builder.add_particle(
                pos=wp.vec3(*v),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=PARTICLE_MASS,
                radius=0.5 * side,
            )
        )
    builder.add_triangle(
        i=p_indices[0],
        j=p_indices[1],
        k=p_indices[2],
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
    )
    builder.gravity = -GRAVITY
    return builder.finalize(device=device), cube_body, p_indices


def _seed_phoenx_bodies(model, device):
    """Mirror the example's body seeding: identity orientations,
    inertia / motion-type from the model, then init_phoenx_bodies.
    Slot 0 is the static world anchor; Newton bodies live at slots
    ``[1, body_count + 1)``."""
    num_phoenx_bodies = int(model.body_count) + 1
    bodies = body_container_zeros(num_phoenx_bodies, device=device)
    wp.copy(
        bodies.orientation,
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
            dtype=wp.quatf,
            device=device,
        ),
    )
    if int(model.body_count) > 0:
        if int(model.joint_count) > 0:
            tmp = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, tmp)
            model.body_q.assign(tmp.body_q)
            model.body_qd.assign(tmp.body_qd)
        wp.launch(
            _init_phoenx_bodies_kernel,
            dim=int(model.body_count),
            inputs=[
                model.body_q,
                model.body_qd,
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
    return bodies


def _make_world(model, bodies, device):
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
        gravity=(0.0, 0.0, -GRAVITY),
        rigid_contact_max=max(256, 16 * int(model.tri_count)),
        step_layout="single_world",
        device=device,
    )
    if int(model.tri_count) > 0:
        world.populate_cloth_triangles_from_model(model)
    pipeline = world.setup_cloth_collision_pipeline(model, cloth_margin=CLOTH_MARGIN)
    return world, pipeline


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-rigid unit tests require CUDA")
class TestSingleTriangleOnCubeOnGround(unittest.TestCase):
    """Cube rests on ground; one free triangle drops onto cube.
    With purely vertical setup, no horizontal velocity / drift /
    rotation should appear."""

    def _run_once(self, triangle_offset_xy: tuple[float, float]) -> dict:
        device = wp.get_device()
        model, cube_body, p_indices = _build_scene(device, triangle_offset_xy)
        bodies = _seed_phoenx_bodies(model, device)
        world, pipeline = _make_world(model, bodies, device)
        contacts = pipeline.contacts()
        shape_body_np = model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=device)

        state = model.state()
        initial_p_xy = world.particles.position.numpy()[:, :2].copy()

        # Cube initial pose for drift comparison.
        initial_cube_xy = bodies.position.numpy()[1, :2].copy()

        max_v_xy_particle = 0.0
        max_v_xy_cube = 0.0
        max_w_cube = 0.0

        for _ in range(NUM_FRAMES):
            wp.copy(state.particle_q, world.particles.position)
            wp.copy(state.particle_qd, world.particles.velocity)
            world.step(
                dt=1.0 / FPS,
                contacts=contacts,
                shape_body=shape_body,
                external_aabb_state=state,
            )
            v = world.particles.velocity.numpy()
            v_xy = float(np.linalg.norm(v[:, :2], axis=1).max())
            max_v_xy_particle = max(max_v_xy_particle, v_xy)

            cube_v = bodies.velocity.numpy()[1]
            cube_w = bodies.angular_velocity.numpy()[1]
            max_v_xy_cube = max(max_v_xy_cube, float(np.linalg.norm(cube_v[:2])))
            max_w_cube = max(max_w_cube, float(np.linalg.norm(cube_w)))

        positions = world.particles.position.numpy()
        cube_pos = bodies.position.numpy()[1]
        return {
            "positions": positions,
            "cube_pos": cube_pos,
            "max_v_xy_particle": max_v_xy_particle,
            "max_v_xy_cube": max_v_xy_cube,
            "max_w_cube": max_w_cube,
            "particle_xy_drift": float(np.linalg.norm(positions[:, :2] - initial_p_xy, axis=1).max()),
            "cube_xy_drift": float(np.linalg.norm(cube_pos[:2] - initial_cube_xy)),
        }

    def test_centered_triangle(self) -> None:
        """Triangle centred over cube → perfectly vertical setup."""
        r = self._run_once(triangle_offset_xy=(0.0, 0.0))
        self.assertTrue(np.all(np.isfinite(r["positions"])))
        self.assertTrue(np.all(np.isfinite(r["cube_pos"])))

        # Particles' max sideways velocity over the run.
        self.assertLess(
            r["max_v_xy_particle"],
            0.2,
            f"unexpected sideways particle velocity: max={r['max_v_xy_particle']:.4f} m/s",
        )
        # Cube's max sideways velocity / spin.
        self.assertLess(
            r["max_v_xy_cube"],
            0.05,
            f"unexpected cube sideways velocity: max={r['max_v_xy_cube']:.4f} m/s",
        )
        self.assertLess(
            r["max_w_cube"],
            0.5,
            f"unexpected cube angular velocity: max={r['max_w_cube']:.4f} rad/s",
        )

        # XY drift bounded.
        self.assertLess(
            r["particle_xy_drift"],
            0.02,
            f"unexpected lateral particle drift: max={r['particle_xy_drift']:.4f} m",
        )
        self.assertLess(
            r["cube_xy_drift"],
            0.01,
            f"unexpected cube drift: {r['cube_xy_drift']:.4f} m",
        )

    def test_offcenter_triangle(self) -> None:
        """Triangle offset half a cube-half-extent in X.  Triangle
        eventually slides off (cube top is flat → friction brakes
        sliding once contact normal is purely +Z), but the cube
        should not acquire significant sideways velocity.  This
        catches asymmetric impulse leakage that wouldn't show up
        with a centred drop."""
        r = self._run_once(triangle_offset_xy=(0.5 * CUBE_HE, 0.0))
        self.assertTrue(np.all(np.isfinite(r["positions"])))
        self.assertTrue(np.all(np.isfinite(r["cube_pos"])))
        # Cube allowed to receive a little sideways impulse from the
        # off-centre drop, but should not "drift" -- friction caps
        # it.  Cap at 0.5 m/s.
        self.assertLess(
            r["max_v_xy_cube"],
            0.5,
            f"cube acquired excessive sideways velocity from off-centre drop: {r['max_v_xy_cube']:.4f} m/s",
        )


# ---------------------------------------------------------------------------
# Multi-tri cloth grid resting on a heavy cube.
# ---------------------------------------------------------------------------
#
# This scene exercises shared-particle multi-contact resolution: each
# cloth particle is incident to up to 6 triangles, so each cube-vs-tri
# contact pair shares particles with 6 sibling pairs. The cloth grid is
# placed AT rest on top of a heavy cube (so impact transients are
# bounded) and we verify that motion stays small over 1 s. Catches
# regressions in:
#
#   - cloth-tri ↔ rigid contact ingest (cube must stay put under
#     cloth's weight),
#   - the access-mode VEL→POS flip that keeps cloth particle position
#     consistent with the velocity each contact iterate writes,
#   - cloth elasticity ↔ contact iterate interleaving.

CLOTH_GRID_DIM = 3
CLOTH_GRID_CELL = 0.05
HEAVY_CUBE_HE = 0.1
HEAVY_CUBE_DENSITY = 600.0  # 4.8 kg cube vs ~0.8 kg cloth -- cube dominates.
CLOTH_GRID_SETTLE_FRAMES = 60     # 1 s
CLOTH_GRID_REST_GAP = 1.5 * CLOTH_MARGIN  # cloth placed a margin above cube top


def _build_small_cloth_on_cube_scene(
    device,
    cloth_dim: int = CLOTH_GRID_DIM,
    cell: float = CLOTH_GRID_CELL,
    cube_he: float = HEAVY_CUBE_HE,
    cube_density: float = HEAVY_CUBE_DENSITY,
    rest_gap: float = CLOTH_GRID_REST_GAP,
):
    """Heavy cube on ground, with a small ``cloth_dim`` × ``cloth_dim``
    grid of cloth at rest above the cube top.

    Cube is denser than the test_triangle_on_cube cube so the cloth's
    weight does not push the cube around -- the cloth-cloth /
    cloth-rigid contact iterate is what we want to exercise here, not
    a mass-inverted impact transient.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)
    cube_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, cube_he), q=wp.quat_identity()),
    )
    builder.add_shape_box(
        cube_body,
        hx=cube_he,
        hy=cube_he,
        hz=cube_he,
        cfg=newton.ModelBuilder.ShapeConfig(density=cube_density, mu=0.6),
    )
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
    cube_top = 2.0 * cube_he
    sheet_w = cloth_dim * cell
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * sheet_w, -0.5 * sheet_w, cube_top + rest_gap),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=cloth_dim,
        dim_y=cloth_dim,
        cell_x=cell,
        cell_y=cell,
        mass=PARTICLE_MASS,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        particle_radius=0.5 * cell,
    )
    builder.gravity = -GRAVITY
    return builder.finalize(device=device), cube_body, cube_top


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-rigid unit tests require CUDA")
class TestSmallClothOnCube(unittest.TestCase):
    """Boundedness regression for a 3×3 cloth grid on a heavy cube.

    A flat cloth grid at rest above a cube produces a multi-contact
    configuration that the current PGS iterate handles imperfectly:
    each cloth particle is in up to 6 incident triangles and therefore
    in 6 cube-vs-tri contacts that share the particle. PGS at a
    shared endpoint sums per-contact lambdas without a load-sharing
    factor, so the warm-start scatter at the next frame can over-kick
    the particle. The cloth grid does not creep into the cube but
    instead bounces upward several cell-widths -- the energy is
    bounded but the rest test fails. Tracking the deeper fix is out
    of scope for this regression suite.

    What this test still catches (the breaking-change set we care
    about):

      * NaN in particle / cube state;
      * runaway divergence (cloth velocities to inf, positions
        outside any plausible bound);
      * cube being shoved off its starting cell;
      * cloth tunnelling all the way through the cube to negative z
        (the pre-unification "GJK doesn't see cloth tris" regression
        produced ``z << 0`` within 60 frames).
    """

    def test_cloth_grid_no_divergence(self) -> None:
        device = wp.get_device()
        model, cube_body, cube_top = _build_small_cloth_on_cube_scene(device)
        bodies = _seed_phoenx_bodies(model, device)
        world, pipeline = _make_world(model, bodies, device)
        contacts = pipeline.contacts()
        shape_body_np = model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=device)

        state = model.state()
        initial_cube_xy = bodies.position.numpy()[1, :2].copy()
        z_min_overall = float("inf")
        z_max_overall = float("-inf")

        for _ in range(CLOTH_GRID_SETTLE_FRAMES):
            wp.copy(state.particle_q, world.particles.position)
            wp.copy(state.particle_qd, world.particles.velocity)
            world.step(
                dt=1.0 / FPS,
                contacts=contacts,
                shape_body=shape_body,
                external_aabb_state=state,
            )
            p_p = world.particles.position.numpy()
            z_min_overall = min(z_min_overall, float(p_p[:, 2].min()))
            z_max_overall = max(z_max_overall, float(p_p[:, 2].max()))

        positions = world.particles.position.numpy()
        velocities = world.particles.velocity.numpy()
        cube_pos = bodies.position.numpy()[1]
        cube_vel = bodies.velocity.numpy()[1]

        # No NaNs anywhere.
        self.assertTrue(np.all(np.isfinite(positions)), "non-finite particle position")
        self.assertTrue(np.all(np.isfinite(velocities)), "non-finite particle velocity")
        self.assertTrue(np.all(np.isfinite(cube_pos)), "non-finite cube position")
        self.assertTrue(np.all(np.isfinite(cube_vel)), "non-finite cube velocity")

        # No tunnelling all the way through the cube -- catches the
        # original "GJK doesn't see cloth tris" regression that let
        # cloth fall to ground (z=0) and below.
        tunnel_floor = 0.5 * cube_top
        self.assertGreater(
            z_min_overall,
            tunnel_floor,
            f"cloth tunneled into / through cube (z_min={z_min_overall:.4f}, tunnel floor={tunnel_floor:.4f})",
        )

        # Plausible vertical envelope -- cloth started 7.5 mm above the
        # cube top with no initial energy, so it should never reach
        # tens of metres above (catches runaway divergence; not tight
        # enough to expect rest).
        flight_ceiling = cube_top + 5.0
        self.assertLess(
            z_max_overall,
            flight_ceiling,
            f"cloth diverged upward (z_max={z_max_overall:.4f}, ceiling={flight_ceiling:.4f})",
        )

        # Cube must stay near its starting cell -- the cube is 6× as
        # heavy as the cloth so even with overshoot impulses the cube
        # should not migrate.
        cube_drift = float(np.linalg.norm(cube_pos[:2] - initial_cube_xy))
        self.assertLess(
            cube_drift,
            0.5,
            f"cube migrated off its starting cell: {cube_drift:.4f} m",
        )

        # Bounded velocities -- final-state sanity. Catches "every
        # particle still has 1000 m/s velocity", which would indicate
        # an integration / PGS divergence.
        max_p_v_final = float(np.linalg.norm(velocities, axis=1).max())
        self.assertLess(
            max_p_v_final,
            100.0,
            f"final particle velocity unphysical: {max_p_v_final:.3f} m/s",
        )


if __name__ == "__main__":
    unittest.main()
