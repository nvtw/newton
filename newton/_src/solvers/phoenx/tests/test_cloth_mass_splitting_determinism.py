# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact determinism with mass splitting enabled.

Two independent ``PhoenXWorld`` instances built from the same recipe
must produce bit-identical state after every captured step. PhoenX's
existing :mod:`test_determinism` covers the rigid-only path; this file
adds the cloth + mass-splitting overlay -- the path that
``test_cloth_mass_splitting`` showed was non-deterministic until the
overflow ordering was fixed (commit ``9ff79365``).

Run on CUDA only and inside a captured graph because: (a) mass
splitting requires CUDA, (b) the production solver runs under
``wp.ScopedCapture``, and (c) the overflow-ordering bug we're guarding
against only shows up across kernel boundaries the scheduler
reorders.
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

# Scene constants. Same shape as test_cloth_mass_splitting's regression
# scene, picked so the cube-vs-cloth contact set genuinely overflows
# beyond the 12-colour cap -- otherwise mass splitting's overflow path
# stays empty and the determinism test wouldn't reach the code that was
# previously non-deterministic.
_CLOTH_DIM_X = 16
_CLOTH_DIM_Y = 8
_CELL = 0.1
_CLOTH_Z = 2.0
_CUBE_DROP_DZ = 1.0
_CUBE_HALF_SIDE = 0.2
_PARTICLE_MASS = 0.05
_CUBE_MASS = 1.0
_SUBSTEPS = 4
_SOLVER_ITERATIONS = 16
_N_FRAMES = 12  # enough for cube to impact cloth (~frame 8 at dz=1m)
_DT = 1.0 / 60.0


class _ClothMassSplittingScene:
    """Self-contained cube-on-cloth scene with mass splitting enabled.

    Constructor builds the model, populates PhoenX, captures a single
    ``step()`` into a CUDA graph, and exposes :meth:`step` which
    replays the graph.
    """

    def __init__(self, device: wp.Device):
        self.device = device

        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=0.0)
        self._cube_body = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, _CLOTH_Z + _CUBE_DROP_DZ),
                q=wp.quat_identity(),
            ),
            mass=_CUBE_MASS,
        )
        builder.add_shape_box(self._cube_body, hx=_CUBE_HALF_SIDE, hy=_CUBE_HALF_SIDE, hz=_CUBE_HALF_SIDE)

        cloth_origin = wp.vec3(-_CLOTH_DIM_X * _CELL * 0.5, -_CLOTH_DIM_Y * _CELL * 0.5, _CLOTH_Z)
        builder.add_cloth_grid(
            pos=cloth_origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=_CLOTH_DIM_X,
            dim_y=_CLOTH_DIM_Y,
            cell_x=_CELL,
            cell_y=_CELL,
            mass=_PARTICLE_MASS,
            fix_left=True,
            fix_right=True,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.04,
        )
        self.model = builder.finalize(device=device)

        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=device,
            )
        )
        state_init = self.model.state()
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=int(self.model.body_count),
            inputs=[
                self.model.body_q,
                state_init.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
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
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(self.model.tri_count),
            device=device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=int(self.model.tri_count),
            num_worlds=1,
            substeps=_SUBSTEPS,
            solver_iterations=_SOLVER_ITERATIONS,
            rigid_contact_max=4096,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=12,
            device=device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)
        self.pipeline = self.world.setup_cloth_collision_pipeline(
            self.model, cloth_thickness=0.005, cloth_gap=0.010, rigid_contact_max=4096
        )
        self.contacts = self.pipeline.contacts()
        self.state = self.model.state()

        # Warm-up step (un-captured) so kernel cache + first-substep
        # state are populated before graph capture.
        self._simulate_one_frame()

        with wp.ScopedCapture(device=device) as cap:
            self._simulate_one_frame()
        self._graph = cap.graph

    def _sync_newton_to_phoenx(self) -> None:
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def _simulate_one_frame(self) -> None:
        self._sync_newton_to_phoenx()
        self.world.collide(self.state, self.contacts)
        self.world.step(_DT, contacts=self.contacts)
        self._sync_phoenx_to_newton()

    def step(self) -> None:
        wp.capture_launch(self._graph)


def _snapshot(scene: _ClothMassSplittingScene) -> dict[str, np.ndarray]:
    """Device-to-host copy of every solver-written state buffer."""
    return {
        "particle_pos": scene.world.particles.position.numpy().copy(),
        "particle_vel": scene.world.particles.velocity.numpy().copy(),
        "body_pos": scene.bodies.position.numpy().copy(),
        "body_vel": scene.bodies.velocity.numpy().copy(),
        "body_ang_vel": scene.bodies.angular_velocity.numpy().copy(),
        "body_orient": scene.bodies.orientation.numpy().copy(),
    }


def _assert_bit_exact(case: unittest.TestCase, ref, dup, frame: int) -> None:
    for field, a in ref.items():
        b = dup[field]
        if np.array_equal(a, b):
            continue
        max_diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        flat_diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
        worst = int(np.argmax(flat_diff))
        case.fail(
            f"frame {frame}: {field!r} diverged between two scenes -- "
            f"max |delta|={max_diff:.3e}, flat_idx={worst} (shape={a.shape})"
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Mass-splitting determinism test runs on CUDA only (mass splitting + graph capture).",
)
class TestClothMassSplittingDeterminism(unittest.TestCase):
    """Two independently-built cloth + mass-splitting scenes must
    produce bit-identical state after every captured step."""

    def test_cube_on_cloth_bit_exact(self) -> None:
        device = wp.get_preferred_device()
        ref = _ClothMassSplittingScene(device)
        dup = _ClothMassSplittingScene(device)

        # Post-warm-up + post-capture state must already match: if it
        # doesn't, the builder is non-deterministic (rare but worth
        # catching here so it doesn't get blamed on the step loop).
        wp.synchronize_device(device)
        _assert_bit_exact(self, _snapshot(ref), _snapshot(dup), frame=0)

        for f in range(1, _N_FRAMES + 1):
            ref.step()
            dup.step()
            wp.synchronize_device(device)
            _assert_bit_exact(self, _snapshot(ref), _snapshot(dup), frame=f)


if __name__ == "__main__":
    wp.init()
    unittest.main()
