# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX dragon soft-body demo.

Drops Matthias Müller's "Ten Minute Physics" dragon
(:mod:`~newton._src.solvers.phoenx.examples.dragon_softbody_data`)
onto a static ground plane and simulates it as a physically-based
tetrahedral soft body. The dragon's 1234-vertex / 3840-tet mesh is
fed through :meth:`newton.ModelBuilder.add_soft_mesh`, so the
constraints are the same corotational FEM rows used by the rest of
PhoenX (:func:`~newton._src.solvers.phoenx.constraints.\
constraint_soft_tetrahedron.soft_tet_lame_from_youngs_poisson`):

* Volumetric (1st Lame, ``k_lambda``) — penalises ``det(F) - 1``.
* Deviatoric / shear (2nd Lame, ``k_mu``) — penalises the
  symmetric deviatoric part of the deformation gradient ``F``
  in a co-rotated frame extracted via the per-tet polar
  decomposition.

This is *not* the "edges-as-springs + per-tet volume" approximation
from the upstream WebGL demo; it's a continuum-mechanics FEM
formulation driven by Young's modulus and Poisson's ratio.

Run::

    python -m newton._src.solvers.phoenx.examples.example_dragon_softbody
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.examples import dragon_softbody_data as dragon_data
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.timer import print_column_timings

#: Tonge mass splitting (matches :mod:`example_soft_body_drop`).
ENABLE_MASS_SPLITTING: bool = True
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 12

#: Per-column wall-clock profiling. See
#: :mod:`example_soft_body_drop` for the trade-offs; cheap to leave
#: on at this scene's scale.
ENABLE_COLUMN_TIMERS: bool = True


class Example:
    """A single dragon dropped onto a ground plane."""

    def __init__(
        self,
        viewer,
        args=None,
        drop_height: float = 1.5,
        scale: float = 1.0,
        youngs_modulus: float = 5.0e7,
        poisson_ratio: float = 0.3,
        density: float = 500.0,
        soft_body_thickness: float = 0.005,
        soft_body_gap: float = 0.010,
    ):
        if drop_height <= 0.0:
            raise ValueError(f"drop_height must be positive, got {drop_height}")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.viewer = viewer
        self.device = wp.get_device()
        self.drop_height = float(drop_height)
        self.scale = float(scale)

        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Match example_soft_body_drop: small Jitter2-style step
        # schedule (5 substeps x 4 PGS iterations) so material
        # constants in Pa transfer over from the C# reference.
        self.sim_substeps = 5
        self.solver_iterations = 4

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.k_lambda, self.k_mu = soft_tet_lame_from_youngs_poisson(self.youngs_modulus, self.poisson_ratio)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=0.0)

        # The source mesh is +Y up; Newton is +Z up. Rotate by +90 deg
        # around X so the dragon's spine aligns with +Z. add_soft_mesh
        # applies ``pos + rot * (scale * vert)`` per vertex.
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, self.drop_height),
            rot=rot,
            scale=self.scale,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=dragon_data.TET_VERTS.tolist(),
            indices=dragon_data.TET_IDS.reshape(-1).tolist(),
            density=density,
            k_mu=self.k_mu,
            k_lambda=self.k_lambda,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

        self.model = builder.finalize(device=self.device)

        # PhoenX bodies: slot 0 reserved as static world anchor. The
        # ground plane lives at shape_body=-1 so no dynamic rigid
        # body slot is needed.
        num_phoenx_bodies = max(1, int(self.model.body_count) + 1)
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            )
        )
        if self.model.body_count > 0:
            self.state_init = self.model.state()
            wp.launch(
                init_phoenx_bodies_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.model.body_q,
                    self.state_init.body_qd,
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
                device=self.device,
            )
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            # The dragon's 3840 tets generate many simultaneous
            # ground contacts when it lands flat-out; size the
            # contact pools to comfortably exceed the worst case so
            # the narrow-phase doesn't drop overflow contacts.
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            rigid_contact_max=4 * int(self.model.tet_count),
            step_layout="single_world",
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            partitioner_algorithm="greedy",
            enable_column_timers=ENABLE_COLUMN_TIMERS,
            device=self.device,
        )
        self._frame_index = 0
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            soft_body_thickness=soft_body_thickness,
            soft_body_gap=soft_body_gap,
            rigid_contact_max=2 * int(self.model.tet_count),
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()

        self.viewer.set_model(self.model)
        # Frame the dragon side-on. Its rotated bbox spans roughly
        # x in [-1.1, 1.1] * scale and z in [drop_height + 0.45,
        # drop_height + 2.0] * scale; pull the camera back along +x.
        bbox_min = dragon_data.TET_VERTS.min(0)
        bbox_max = dragon_data.TET_VERTS.max(0)
        half_height = 0.5 * self.scale * (bbox_max[1] - bbox_min[1])
        center_z = self.drop_height + self.scale * 0.5 * (bbox_min[1] + bbox_max[1])
        cam_distance = max(4.0, 3.0 * self.scale * (bbox_max[0] - bbox_min[0]), 2.0 * half_height)
        self.viewer.set_camera(
            pos=wp.vec3(cam_distance, 0.0, center_z),
            pitch=-15.0,
            yaw=180.0,
        )

        self._capture()

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
        if n > 0:
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

    def _capture(self) -> None:
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        self._sync_newton_to_phoenx()
        self.world.collide(self.state, self.contacts)
        self.world.step(self.frame_dt, contacts=self.contacts)
        self._sync_phoenx_to_newton()

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt
        self._frame_index += 1
        print_column_timings(self.world, self._frame_index, label="dragon_softbody")

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")
        if positions[:, 2].min() < -5.0:
            raise RuntimeError(f"a dragon particle escaped downward (min z = {positions[:, 2].min():.4f})")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--drop-height", type=float, default=1.5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--youngs-modulus", type=float, default=5.0e12)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--soft-body-thickness", type=float, default=0.005)
    parser.add_argument("--soft-body-gap", type=float, default=0.010)
    viewer, args = newton.examples.init(parser)
    viewer._paused = True
    example = Example(
        viewer,
        args,
        drop_height=args.drop_height,
        scale=args.scale,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        density=args.density,
        soft_body_thickness=args.soft_body_thickness,
        soft_body_gap=args.soft_body_gap,
    )
    newton.examples.run(example, args)
