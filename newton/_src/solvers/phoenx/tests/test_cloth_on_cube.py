# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Boundedness regression for the cloth-on-cube example.

Direct unit-test port of the high-frequency configuration from
:mod:`newton._src.solvers.phoenx.examples.example_cloth_on_cube`:
ground plane + cube resting on it + a small cloth grid on top of
the cube.

The single-triangle rest case lives in
:mod:`newton._src.solvers.phoenx.tests.test_tri_rest_on_cube` and
asserts < 0.1 mm drift over 2 s. This file deals with the multi-tri
grid configuration where each cloth particle is shared by up to 6
incident triangles. PhoenX's PGS iterate sums per-contact lambdas
without a load-sharing factor at shared endpoints, so the grid-on-cube
configuration does not settle to a tight rest -- the energy is
bounded but the cloth oscillates / drifts upward by a few cell-widths
after first contact. Tracking that fix is out of scope here.

What this regression catches (the breaking-change set we care about):

  * NaN in any state field;
  * runaway divergence (positions / velocities to inf);
  * cloth tunnelling all the way through the cube (the pre-fix
    "GJK doesn't see cloth tris" regression sent z_min to 0 within a
    second);
  * cube migrating off its starting cell.

CUDA-only; PhoenX uses CUDA graph capture internally for the substep
iterate, so this test exercises that path.
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


# Mirrors example_cloth_on_cube.py defaults.
FPS = 60
SUBSTEPS = 8
SOLVER_ITERATIONS = 8
PARTICLE_MASS = 0.1
CLOTH_THICKNESS = 0.02
CUBE_GAP = 0.01
YOUNGS_MODULUS = 5.0e7
POISSON_RATIO = 0.3
CUBE_DENSITY = 200.0
CUBE_FRICTION = 0.6

NUM_FRAMES = 120  # 2 s


def _build_scene(device, cloth_dim: int = 4, cell: float = 0.1):
    sheet_w = cloth_dim * cell
    cube_he = 0.65 * sheet_w
    cube_center_z = cube_he
    cube_top_z = 2.0 * cube_he

    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)
    cube_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, cube_center_z), q=wp.quat_identity()),
    )
    builder.add_shape_box(
        cube_body,
        hx=cube_he,
        hy=cube_he,
        hz=cube_he,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=CUBE_DENSITY, mu=CUBE_FRICTION, gap=CUBE_GAP
        ),
    )

    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(YOUNGS_MODULUS, POISSON_RATIO)
    sheet_origin = (-0.5 * sheet_w, -0.5 * sheet_w)
    # Place cloth just above the cube top so we exercise the
    # speculative-to-penetrating contact transition without the
    # full free-fall transient.
    sheet_z = cube_top_z + 1.5 * CLOTH_THICKNESS
    builder.add_cloth_grid(
        pos=wp.vec3(sheet_origin[0], sheet_origin[1], sheet_z),
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
    builder.gravity = -9.81

    model = builder.finalize(device=device)
    if int(model.body_count) > 0 and int(model.joint_count) > 0:
        tmp = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, tmp)
        model.body_q.assign(tmp.body_q)
        model.body_qd.assign(tmp.body_qd)
    return model, cube_body, cube_he, cube_top_z


def _seed_phoenx_bodies(model, device):
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


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-on-cube test requires CUDA")
class TestClothOnCubeBounded(unittest.TestCase):
    """Drop a cloth onto a cube and verify the simulation stays
    bounded, doesn't NaN, and doesn't tunnel through the cube."""

    def _run(self, cloth_dim: int = 4, cell: float = 0.1) -> dict:
        device = wp.get_device()
        model, cube_body, cube_he, cube_top_z = _build_scene(device, cloth_dim, cell)
        bodies = _seed_phoenx_bodies(model, device)

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
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=max(4096, 16 * int(model.tri_count)),
            step_layout="single_world",
            device=device,
        )
        world.populate_cloth_triangles_from_model(model)
        pipeline = world.setup_cloth_collision_pipeline(model, cloth_margin=CLOTH_THICKNESS)
        contacts = pipeline.contacts()

        shape_body_np = model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=device)

        state = model.state()
        cube_initial_xy = bodies.position.numpy()[1, :2].copy()

        z_min_overall = float("inf")
        z_max_overall = float("-inf")

        for _ in range(NUM_FRAMES):
            wp.copy(state.particle_q, world.particles.position)
            wp.copy(state.particle_qd, world.particles.velocity)
            world.step(
                dt=1.0 / FPS,
                contacts=contacts,
                shape_body=shape_body,
                external_aabb_state=state,
            )
            tp = world.particles.position.numpy()
            z_min_overall = min(z_min_overall, float(tp[:, 2].min()))
            z_max_overall = max(z_max_overall, float(tp[:, 2].max()))

        positions = world.particles.position.numpy()
        velocities = world.particles.velocity.numpy()
        cube_pos = bodies.position.numpy()[1]
        cube_vel = bodies.velocity.numpy()[1]
        return {
            "positions": positions,
            "velocities": velocities,
            "cube_pos": cube_pos,
            "cube_vel": cube_vel,
            "cube_drift_xy": float(np.linalg.norm(cube_pos[:2] - cube_initial_xy)),
            "z_min_overall": z_min_overall,
            "z_max_overall": z_max_overall,
            "cube_top_z": cube_top_z,
        }

    def test_low_resolution_cloth_does_not_diverge(self) -> None:
        """4×4 cloth on cube. Catches NaN, infinite divergence, deep
        tunnelling, and large cube migration."""
        r = self._run(cloth_dim=4, cell=0.1)

        # No NaNs.
        self.assertTrue(np.all(np.isfinite(r["positions"])), "non-finite particle position")
        self.assertTrue(np.all(np.isfinite(r["velocities"])), "non-finite particle velocity")
        self.assertTrue(np.all(np.isfinite(r["cube_pos"])), "non-finite cube position")
        self.assertTrue(np.all(np.isfinite(r["cube_vel"])), "non-finite cube velocity")

        # Cloth must not tunnel all the way through the cube. The
        # pre-unification regression dropped z_min to ~0 within ~30
        # frames; the bound here is generous enough to allow the
        # current solver's transient elastic vibration but tight
        # enough to catch a "cloth fell through" regression.
        self.assertGreater(
            r["z_min_overall"],
            0.5 * r["cube_top_z"],
            f"cloth tunneled deep into cube (z_min={r['z_min_overall']:.4f}, cube top={r['cube_top_z']:.4f})",
        )

        # Cloth must not fly off into orbit -- catches the
        # speculative-storm regression that produced 100+ m/s velocity
        # kicks. Generous ceiling: 5 m above cube top.
        self.assertLess(
            r["z_max_overall"],
            r["cube_top_z"] + 5.0,
            f"cloth flew off into orbit (z_max={r['z_max_overall']:.4f}, cube top={r['cube_top_z']:.4f})",
        )

        # Cube stays near its starting cell.
        self.assertLess(
            r["cube_drift_xy"],
            0.5,
            f"cube drifted off its starting cell: {r['cube_drift_xy']:.4f} m",
        )

        # Final-state velocities bounded -- catches "everything has
        # 1000 m/s velocity" integration divergence.
        max_p_v = float(np.linalg.norm(r["velocities"], axis=1).max())
        self.assertLess(
            max_p_v,
            100.0,
            f"particle velocities unphysical: max={max_p_v:.3f} m/s",
        )
        max_cube_v = float(np.linalg.norm(r["cube_vel"]))
        self.assertLess(
            max_cube_v,
            10.0,
            f"cube velocity unphysical: {max_cube_v:.3f} m/s",
        )


if __name__ == "__main__":
    unittest.main()
