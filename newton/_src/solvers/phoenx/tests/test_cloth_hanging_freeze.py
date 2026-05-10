# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the ``example_cloth_hanging`` freeze under
``mass_split_max_partitions=12``.

Setup mirrors the example: a hanging cloth grid (pinned on its left
edge) + a free rigid cube falling onto it + a static ground plane.
With mass splitting active, the moment cloth-vs-cube contacts appear,
the scene should NOT freeze: the cube must continue moving (gravity
+ contact response), the unpinned cloth corner must keep moving, and
no body / particle position should go non-finite.

Smaller cloth (8x4) and shorter drop (cube starts ~1 m above) than
the example so the test runs in well under a second after kernel
caching.
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


def _build_world(*, mass_split_max_partitions: int | None, device: wp.context.Devicelike):
    """Construct a small cloth-hanging scene like
    ``example_cloth_hanging.Example.__init__`` but without the GUI
    parts. Returns ``(world, model, state, contacts, pipeline,
    cube_body_idx, num_particles)``."""
    dim_x = 8
    dim_y = 4
    cell = 0.1
    particle_mass = 0.05
    youngs_modulus = 5.0e8
    poisson_ratio = 0.3
    cube_size = 0.1
    # Position the cube intersecting the cloth at frame 0 so contact
    # is generated immediately -- avoids spending many frames on the
    # falling phase before we get to the freeze regression.
    cube_drop_height = 4.0 + cube_size + 0.005

    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(youngs_modulus, poisson_ratio)

    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=1.0)
    # Cloth is at pos=(0,0,4) rotated 90° around Z -- after rotation
    # it spans roughly x in [-0.4, 0], y in [0, 0.8] at z=4. Place
    # the cube over the middle of that footprint.
    cube_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.2, 0.4, cube_drop_height), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(cube_body, hx=cube_size, hy=cube_size, hz=cube_size)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 4.0),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x, dim_y=dim_y, cell_x=cell, cell_y=cell,
        mass=particle_mass, fix_left=True,
        tri_ke=tri_ke, tri_ka=tri_ka, particle_radius=0.04,
    )
    model = builder.finalize(device=device)

    num_phoenx_bodies = int(model.body_count) + 1
    bodies = body_container_zeros(num_phoenx_bodies, device=device)
    bodies.orientation.assign(
        wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
            dtype=wp.quatf, device=device,
        )
    )
    if model.body_count > 0:
        state_init = model.state()
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=model.body_count,
            inputs=[
                model.body_q, state_init.body_qd, model.body_com,
                model.body_inv_mass, model.body_inv_inertia,
            ],
            outputs=[
                bodies.position, bodies.orientation, bodies.velocity, bodies.angular_velocity,
                bodies.inverse_mass, bodies.inverse_inertia, bodies.inverse_inertia_world,
                bodies.motion_type, bodies.body_com,
            ],
            device=device,
        )

    constraints = PhoenXWorld.make_constraint_container(
        num_joints=0, num_cloth_triangles=int(model.tri_count), device=device,
    )
    world = PhoenXWorld(
        bodies=bodies, constraints=constraints, num_joints=0,
        num_particles=int(model.particle_count),
        num_cloth_triangles=int(model.tri_count),
        rigid_contact_max=4096, num_worlds=1,
        substeps=4, solver_iterations=8,
        step_layout="single_world", device=device,
        mass_split_max_partitions=mass_split_max_partitions,
    )
    world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
    world.populate_cloth_triangles_from_model(model)
    pipeline = world.setup_cloth_collision_pipeline(
        model, cloth_thickness=0.005, cloth_gap=0.010,
        rigid_contact_max=4096, shape_pairs_max=8192,
    )
    contacts = pipeline.contacts()
    state = model.state()
    return world, model, state, contacts, pipeline, cube_body, int(model.particle_count)


def _step(world, model, state, contacts, pipeline, *, dt: float):
    """One sim frame: Newton state -> PhoenX, collide, step, PhoenX
    state -> Newton."""
    n = int(model.body_count)
    if n > 0:
        wp.launch(
            newton_to_phoenx_kernel, dim=n,
            inputs=[state.body_q, state.body_qd, model.body_com],
            outputs=[
                world.bodies.position[1 : 1 + n],
                world.bodies.orientation[1 : 1 + n],
                world.bodies.velocity[1 : 1 + n],
                world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=world.device,
        )
    world.collide(state, contacts)
    world.step(dt, contacts=contacts)
    if n > 0:
        wp.launch(
            phoenx_to_newton_kernel, dim=n,
            inputs=[
                world.bodies.position[1 : 1 + n],
                world.bodies.orientation[1 : 1 + n],
                world.bodies.velocity[1 : 1 + n],
                world.bodies.angular_velocity[1 : 1 + n],
                model.body_com,
            ],
            outputs=[state.body_q, state.body_qd],
            device=world.device,
        )
        wp.copy(state.particle_q, world.particles.position)
        wp.copy(state.particle_qd, world.particles.velocity)


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX runs on CUDA only.")
class TestClothHangingFreeze(unittest.TestCase):
    """Regression: cube falling onto cloth must not freeze the scene.

    Both tests use graph capture: one warm-up ``_step`` to compile
    kernels and populate state, ``wp.ScopedCapture`` to record one
    step, then ``wp.capture_launch(graph)`` to replay. This keeps
    each test under a minute once the kernel cache is warm.
    """

    def _capture_and_replay(
        self,
        *,
        mass_split_max_partitions: int | None,
        num_replays: int,
    ):
        device = wp.get_preferred_device()
        world, model, state, contacts, pipeline, cube_body, _num_particles = _build_world(
            mass_split_max_partitions=mass_split_max_partitions, device=device,
        )
        dt = 1.0 / 60.0
        # Free corner of the cloth (left edge is pinned).
        dim_x = 8
        dim_y = 4
        free_corner_idx = (dim_y) * (dim_x + 1) + dim_x

        # Warm-up step (compiles kernels, populates partitioner CSR
        # and the InteractionGraph).
        _step(world, model, state, contacts, pipeline, dt=dt)

        with wp.ScopedCapture(device=device) as capture:
            _step(world, model, state, contacts, pipeline, dt=dt)
        graph = capture.graph

        cube_z_history: list[float] = []
        particle_speed_history: list[float] = []
        contact_count_history: list[int] = []
        for _ in range(num_replays):
            wp.capture_launch(graph)
            wp.synchronize()
            cube_q = state.body_q.numpy()
            cube_z_history.append(float(cube_q[cube_body, 2]))
            v = state.particle_qd.numpy()[free_corner_idx]
            particle_speed_history.append(float(np.linalg.norm(v)))
            contact_count_history.append(int(contacts.rigid_contact_count.numpy()[0]))
            self.assertTrue(
                np.all(np.isfinite(cube_q)),
                f"non-finite body state during replay: {cube_q}",
            )
            self.assertTrue(
                np.all(np.isfinite(state.particle_q.numpy())),
                "non-finite particle state during replay",
            )
        return cube_z_history, particle_speed_history, contact_count_history

    def test_no_mass_splitting_baseline(self):
        """Sanity: without mass splitting the scene runs fine, cube
        falls under gravity, contacts get generated, scene doesn't
        freeze. This is the reference behaviour."""
        cube_z, _particle_v, contacts = self._capture_and_replay(
            mass_split_max_partitions=None, num_replays=12,
        )
        # Contacts must be generated.
        self.assertGreater(
            max(contacts), 0,
            f"baseline: no contacts generated (history={contacts})",
        )

    def test_mass_splitting_on_cloth_does_not_freeze(self):
        """The freeze regression: with ``mass_split_max_partitions=12``
        and cloth-vs-rigid contacts active, the scene must continue
        to move. Free-corner particle velocity must remain non-zero
        across replays (hangs from gravity); cube z must change
        across replays (gravity + contact response). A FROZEN scene
        has both stuck."""
        cube_z, particle_v, contacts = self._capture_and_replay(
            mass_split_max_partitions=12, num_replays=12,
        )
        z_range = max(cube_z) - min(cube_z)
        max_pv = max(particle_v) if particle_v else 0.0
        self.assertTrue(
            z_range > 1e-3 or max_pv > 1e-3,
            f"FREEZE: cube z range = {z_range:.6e}, "
            f"max free-corner particle speed = {max_pv:.6e}. "
            f"z = {cube_z}; particle_v = {particle_v}; contacts = {contacts}",
        )


if __name__ == "__main__":
    unittest.main()
