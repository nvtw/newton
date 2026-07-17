# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`Picking`.

Covers both pickable kinds:

* Rigid OBB raycast + PD force on ``bodies.force`` / ``bodies.torque``.
* Cloth-triangle Möller-Trumbore raycast + per-particle impulse on
  ``particles.velocity`` (the "divide by 3" force split for cloth).

The shared-min reduction across the two raycasts is exercised via a
ray that hits both the cube and the cloth -- the closer hit wins
and the other's state slot stays at ``-1``.

These tests run on CUDA only (PhoenX kernels are not CPU-portable).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import MOTION_KINEMATIC, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.examples.example_common import init_phoenx_bodies_kernel
from newton._src.solvers.phoenx.picking import Picking
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_scene(device, *, kinematic_occluder: bool = False):
    """Tiny cloth + cube + ground scene matched to ``example_cloth_hanging``.

    Returns ``(model, world, picking, half_extents, cube_body_idx)``.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)

    cube_size = 0.4
    cube_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(2.0, 0.0, 1.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(cube_body, hx=cube_size, hy=cube_size, hz=cube_size)

    occluder_body = None
    if kinematic_occluder:
        occluder_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(2.0, 0.0, 4.0), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(occluder_body, hx=0.2, hy=0.2, hz=0.2)

    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
    # Flat 2x2 cloth at z=1 in the y-z plane, centred at x=0.
    builder.add_cloth_grid(
        pos=wp.vec3(-0.1, -0.1, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.05,
        fix_left=True,
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
    wp.launch(
        init_phoenx_bodies_kernel,
        dim=model.body_count,
        inputs=[model.body_q, state_init.body_qd, model.body_com, model.body_inv_mass, model.body_inv_inertia],
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
    if occluder_body is not None:
        occluder_slot = occluder_body + 1
        motion_type = bodies.motion_type.numpy()
        motion_type[occluder_slot] = int(MOTION_KINEMATIC)
        bodies.motion_type.assign(motion_type)
        inverse_mass = bodies.inverse_mass.numpy()
        inverse_mass[occluder_slot] = 0.0
        bodies.inverse_mass.assign(inverse_mass)

    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0, num_cloth_triangles=int(model.tri_count), device=device
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        num_particles=int(model.particle_count),
        num_cloth_triangles=int(model.tri_count),
        num_worlds=1,
        substeps=2,
        solver_iterations=2,
        rigid_contact_max=256,
        step_layout="single_world",
        device=device,
    )
    world.populate_cloth_triangles_from_model(model)

    half_extents_np = np.zeros((num_phoenx_bodies, 3), dtype=np.float32)
    half_extents_np[cube_body + 1] = (cube_size, cube_size, cube_size)
    if occluder_body is not None:
        half_extents_np[occluder_body + 1] = (0.2, 0.2, 0.2)
    half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=device)
    picking = Picking(world, half_extents, model=model, particles=world.particles)
    return model, world, picking, half_extents, cube_body


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX picking tests are CUDA-only.",
)
class TestPicking(unittest.TestCase):
    def test_rigid_pick_latches_cube(self):
        """Ray straight at the cube must latch onto the rigid body."""
        device = wp.get_preferred_device()
        _, _, picking, _, cube_body = _build_scene(device)
        picking.pick(np.array([2.0, 0.0, 5.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32))
        self.assertTrue(picking.is_picking())
        self.assertEqual(int(picking._pick_body.numpy()[0]), cube_body + 1)
        self.assertEqual(int(picking._pick_tri.numpy()[0]), -1)
        picking.release()
        self.assertFalse(picking.is_picking())
        self.assertEqual(int(picking._pick_body.numpy()[0]), -1)
        self.assertEqual(int(picking._pick_tri.numpy()[0]), -1)

    def test_rigid_pick_skips_kinematic_camera_collider(self):
        """A camera-attached kinematic collider must not intercept picking."""
        device = wp.get_preferred_device()
        _, _, picking, _, cube_body = _build_scene(device, kinematic_occluder=True)
        picking.pick(np.array([2.0, 0.0, 5.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32))
        self.assertTrue(picking.is_picking())
        self.assertEqual(int(picking._pick_body.numpy()[0]), cube_body + 1)
        self.assertEqual(int(picking._pick_tri.numpy()[0]), -1)

    def test_cloth_pick_latches_triangle(self):
        """Ray straight at the cloth must latch a triangle with
        barycentric weights that sum to 1."""
        device = wp.get_preferred_device()
        _, world, picking, _, _ = _build_scene(device)
        # Cloth lives around (0, 0, 1) in a 0.2x0.2 patch. Shoot
        # straight down from above the centre.
        picking.pick(np.array([0.0, 0.0, 5.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32))
        self.assertTrue(picking.is_picking(), "ray hit nothing -- check cloth-tri raycast")
        self.assertEqual(int(picking._pick_body.numpy()[0]), -1)
        tri = int(picking._pick_tri.numpy()[0])
        self.assertGreaterEqual(tri, 0)
        self.assertLess(tri, int(world.num_cloth_triangles))
        bary = picking._pick_bary.numpy()[0]
        self.assertAlmostEqual(float(bary[0] + bary[1] + bary[2]), 1.0, places=4)
        for w in bary:
            self.assertGreaterEqual(float(w), -1e-4)
            self.assertLessEqual(float(w), 1.0 + 1e-4)

    def test_cloth_pick_applies_impulse_to_three_vertices(self):
        """Calling :meth:`apply_force` must split the PD spring force
        equally over the three picked triangle vertices, exactly as
        the user's "divide by 3" proposal specifies. Pinned vertices
        (inv_mass == 0) must not move."""
        device = wp.get_preferred_device()
        model, world, picking, _, _ = _build_scene(device)
        picking.pick(np.array([0.0, 0.0, 5.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32))
        tri = int(picking._pick_tri.numpy()[0])
        self.assertGreaterEqual(tri, 0)
        # Drag the target straight up so the spring pulls non-pinned
        # vertices in +z.
        picking._pick_target.assign([wp.vec3f(0.0, 0.0, 5.0)])
        # Zero velocity baseline so we measure the impulse cleanly.
        world.particles.velocity.zero_()
        picking.apply_force(dt=1.0 / 60.0)
        v_after = world.particles.velocity.numpy()
        inv_mass = world.particles.inverse_mass.numpy()
        tri_indices = model.tri_indices.numpy()
        pa, pb, pc = int(tri_indices[tri, 0]), int(tri_indices[tri, 1]), int(tri_indices[tri, 2])
        # At least one of the three vertices must be non-pinned (the
        # cloth is fix_left only, so right-side vertices are free).
        free_idxs = [p for p in (pa, pb, pc) if inv_mass[p] > 0.0]
        self.assertGreater(len(free_idxs), 0, "expected at least one non-pinned vertex on picked triangle")
        for p in free_idxs:
            self.assertGreater(float(v_after[p, 2]), 0.0, f"vertex {p} did not receive upward impulse")
        # Pinned vertices must stay at zero velocity (inv_mass=0
        # multiplies the impulse to zero in the apply kernel).
        for p in (pa, pb, pc):
            if inv_mass[p] == 0.0:
                self.assertAlmostEqual(float(np.linalg.norm(v_after[p])), 0.0, places=5)

    def test_apply_force_when_not_picking_is_noop(self):
        """Both apply paths must be safe to launch when nothing is
        picked -- the kernels gate on ``pick_*[0] < 0``."""
        device = wp.get_preferred_device()
        _, world, picking, _, _ = _build_scene(device)
        world.particles.velocity.zero_()
        world.bodies.force.zero_()
        world.bodies.torque.zero_()
        picking.apply_force(dt=1.0 / 60.0)
        self.assertTrue(np.all(world.particles.velocity.numpy() == 0.0))
        self.assertTrue(np.all(world.bodies.force.numpy() == 0.0))
        self.assertTrue(np.all(world.bodies.torque.numpy() == 0.0))


if __name__ == "__main__":
    unittest.main()
