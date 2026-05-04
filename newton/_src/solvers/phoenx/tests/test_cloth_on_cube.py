# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resting-state regression for the cloth-on-cube scene.

Direct unit-test port of
:mod:`newton._src.solvers.phoenx.examples.example_cloth_on_cube`:
ground plane + cube resting on it + a small cloth grid dropped onto
the cube.  After enough frames, the cloth must come to rest on the
cube's top face: small velocities, small drift, no NaNs, no
tunneling.

The tolerances are tight on purpose -- the user sees the rendered
example continue to twitch after settling, and that twitch is what
this test catches in CI.
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

# Drop short and let the simulation settle -- 4 s is plenty for a
# stiff cloth on a heavy cube.
NUM_FRAMES = 240          # 4 s
SETTLE_FRAME = 60         # frames before measuring "rest"

# Tight rest-state tolerances (the user wants very small slack).
PARTICLE_REST_V_MAX = 0.05      # m/s
CUBE_REST_V_MAX = 0.005         # m/s
CUBE_REST_W_MAX = 0.05          # rad/s
CUBE_REST_DRIFT_MAX = 5.0e-3    # m


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
    sheet_z = cube_top_z + 0.1
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
class TestClothOnCubeRestState(unittest.TestCase):
    """Drop a cloth onto a cube and verify both bodies come to rest
    with very low residual velocities.  Tight tolerances catch the
    "doesn't settle" twitch the user observed in the example."""

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

        peak_particle_v = 0.0
        peak_cube_v = 0.0
        peak_cube_w = 0.0
        rest_particle_v = 0.0
        rest_cube_v = 0.0
        rest_cube_w = 0.0

        for f in range(NUM_FRAMES):
            wp.copy(state.particle_q, world.particles.position)
            wp.copy(state.particle_qd, world.particles.velocity)
            world.step(
                dt=1.0 / FPS,
                contacts=contacts,
                shape_body=shape_body,
                external_aabb_state=state,
            )
            v = world.particles.velocity.numpy()
            cube_v = bodies.velocity.numpy()[1]
            cube_w = bodies.angular_velocity.numpy()[1]

            v_mag = float(np.linalg.norm(v, axis=1).max())
            cv = float(np.linalg.norm(cube_v))
            cw = float(np.linalg.norm(cube_w))
            peak_particle_v = max(peak_particle_v, v_mag)
            peak_cube_v = max(peak_cube_v, cv)
            peak_cube_w = max(peak_cube_w, cw)
            if f >= SETTLE_FRAME:
                rest_particle_v = max(rest_particle_v, v_mag)
                rest_cube_v = max(rest_cube_v, cv)
                rest_cube_w = max(rest_cube_w, cw)

        positions = world.particles.position.numpy()
        cube_pos = bodies.position.numpy()[1]
        return {
            "positions": positions,
            "cube_pos": cube_pos,
            "cube_drift_xy": float(np.linalg.norm(cube_pos[:2] - cube_initial_xy)),
            "peak_particle_v": peak_particle_v,
            "peak_cube_v": peak_cube_v,
            "peak_cube_w": peak_cube_w,
            "rest_particle_v": rest_particle_v,
            "rest_cube_v": rest_cube_v,
            "rest_cube_w": rest_cube_w,
            "cube_top_z": cube_top_z,
        }

    def test_low_resolution_cloth_settles_calmly(self) -> None:
        """4x4 cloth on cube.  After ~1 s the system must be at rest
        with very small residual velocities."""
        r = self._run(cloth_dim=4, cell=0.1)

        # No NaNs.
        self.assertTrue(np.all(np.isfinite(r["positions"])), "non-finite particle position")
        self.assertTrue(np.all(np.isfinite(r["cube_pos"])), "non-finite cube position")

        # Cloth doesn't tunnel through the cube top.  Allow a bit of
        # margin penetration but no full tunnel.
        z_min = float(r["positions"][:, 2].min())
        self.assertGreater(
            z_min,
            r["cube_top_z"] - 5.0 * CLOTH_THICKNESS,
            f"cloth tunneled into cube top (z_min={z_min:.4f}, cube top={r['cube_top_z']:.4f})",
        )

        # Cube hasn't drifted off centre.
        self.assertLess(
            r["cube_drift_xy"],
            CUBE_REST_DRIFT_MAX,
            f"cube drifted: xy={r['cube_drift_xy']:.4f} m",
        )

        # Resting velocities tight.
        self.assertLess(
            r["rest_particle_v"],
            PARTICLE_REST_V_MAX,
            f"particles still moving at rest: max v={r['rest_particle_v']:.4f} m/s "
            f"(peak during impact={r['peak_particle_v']:.4f})",
        )
        self.assertLess(
            r["rest_cube_v"],
            CUBE_REST_V_MAX,
            f"cube still drifting at rest: max v={r['rest_cube_v']:.4f} m/s "
            f"(peak during impact={r['peak_cube_v']:.4f})",
        )
        self.assertLess(
            r["rest_cube_w"],
            CUBE_REST_W_MAX,
            f"cube still spinning at rest: max w={r['rest_cube_w']:.4f} rad/s "
            f"(peak during impact={r['peak_cube_w']:.4f})",
        )


if __name__ == "__main__":
    unittest.main()
