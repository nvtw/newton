# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX cloth + rigid-cube drop demo.

A square cloth pinned at its four corners catches a small rigid
cube falling from above. Demonstrates the end-to-end two-way
rigid-vs-triangle contact path through PhoenX, driven against the
:class:`PhoenXWorld` *internal* API (no :class:`SolverPhoenX`
adapter):

* Mesh built via :meth:`~newton.ModelBuilder.add_cloth_grid`. The
  four corner particles have their inverse mass zeroed and
  ``ParticleFlags.ACTIVE`` cleared so the cloth iterate leaves
  them alone.
* A free-floating rigid box is added with density-derived mass /
  inertia.
* :meth:`PhoenXWorld.setup_cloth_collision_pipeline` builds the
  unified rigid + virtual-shape :class:`CollisionPipeline` (NXN
  broad phase, share-vertex filter, suffix metadata stamped) and
  returns the pipeline + its :class:`Contacts` buffer.
* Per frame: mirror PhoenX body state into a Newton ``State``,
  copy the world's particle positions over, and call
  :meth:`PhoenXWorld.step` with ``external_aabb_state=state``.
  The world refreshes the cloth-triangle AABB suffix and runs
  ``CollisionPipeline.collide_with_external_aabbs`` itself
  before ingesting the contacts.

Run::

    python -m newton._src.solvers.phoenx.examples.example_cloth_rigid_drop
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
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


class Example:
    """A cloth pinned at four corners catching a falling rigid cube."""

    def __init__(
        self,
        viewer,
        args=None,
        width: int = 24,
        height: int = 24,
        youngs_modulus: float = 5.0e8,
        poisson_ratio: float = 0.3,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Cloth + cube need a denser substep budget than the bare
        # cloth demo because the rigid contact constraint shares the
        # PGS sweep with the cloth iterate.
        self.sim_substeps = 8
        self.solver_iterations = 12

        self.dim_x = width
        self.dim_y = height
        self.cell = 0.05
        self.particle_mass = 0.01
        self.cloth_z = 1.0  # height of the pinned corner plane

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.tri_ka, self.tri_ke = cloth_lame_from_youngs_poisson_plane_stress(
            self.youngs_modulus, self.poisson_ratio
        )

        # ---- Build the Newton model -----------------------------------
        # Cloth lies flat in the XY plane, centred on the origin; the
        # rigid cube spawns above the centre and falls onto it.
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * self.dim_x * self.cell, -0.5 * self.dim_y * self.cell, self.cloth_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.particle_mass,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            particle_radius=0.5 * self.cell,
        )
        # Pin the four corners. ``add_cloth_grid`` only supports
        # edge-fixing via ``fix_*`` so we patch the per-particle mass /
        # flags directly. Particle layout is row-major
        # ``(dim_x + 1) * (dim_y + 1)``.
        nx = self.dim_x + 1
        self.corner_indices = (
            0,
            self.dim_x,
            self.dim_y * nx,
            self.dim_y * nx + self.dim_x,
        )
        for c in self.corner_indices:
            builder.particle_mass[c] = 0.0
            builder.particle_flags[c] = builder.particle_flags[c] & ~ParticleFlags.ACTIVE

        # Free-floating rigid cube spawned above the cloth.
        self.cube_he = 0.10
        self.cube_spawn_z = self.cloth_z + 0.4
        self.cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.cube_spawn_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(
            self.cube_body,
            hx=self.cube_he,
            hy=self.cube_he,
            hz=self.cube_he,
            cfg=newton.ModelBuilder.ShapeConfig(density=60.0, mu=0.6),
        )
        builder.gravity = -9.81

        self.model = builder.finalize(device=self.device)

        # ``add_body`` queues the free-joint pose in ``joint_q``; FK
        # propagates it into ``body_q`` so :func:`_init_phoenx_bodies_kernel`
        # seeds the cube at the spawn pose instead of the URDF-rest
        # origin.
        if int(self.model.body_count) > 0 and int(self.model.joint_count) > 0:
            tmp_state = self.model.state()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, tmp_state)
            self.model.body_q.assign(tmp_state.body_q)
            self.model.body_qd.assign(tmp_state.body_qd)

        # ---- Build the PhoenX body container --------------------------
        # Slot 0 is the static world anchor; Newton bodies occupy
        # slots ``[1, body_count + 1)``.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        # Seed every slot's orientation to identity (``body_container_zeros``
        # leaves it as a non-unit zero quaternion that the inertia
        # refresh would blow up on).
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        wp.launch(
            _init_phoenx_bodies_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_q,
                self.model.body_qd,
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

        # ---- Build the PhoenXWorld -----------------------------------
        # The constraint container holds ONLY cloth triangles
        # (``num_joints = 0``); rigid contacts live in a separate
        # contact-column container sized via ``rigid_contact_max``,
        # which we'll learn from the pipeline below.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(self.model.tri_count),
            device=self.device,
        )

        # We have to know ``rigid_contact_max`` *before* building
        # :class:`PhoenXWorld`, but the cloth-aware pipeline is what
        # sets it. Build the pipeline first by calling the world's
        # setup helper on a temporary placeholder world... not great.
        # Instead, pre-allocate the pipeline through a dummy
        # instantiation: use ``setup_cloth_collision_pipeline`` after
        # constructing the world with a generous default; the pipeline
        # is independent of the contact-column count.
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=int(self.model.tri_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            gravity=(0.0, 0.0, -9.81),
            # Cloth-rigid scenes are dominated by cloth-triangle vs
            # cube contacts; ``2 * dim_x * dim_y`` is plenty for a
            # single small cube on a sub-half-metre grid.
            rigid_contact_max=max(1024, 4 * self.dim_x * self.dim_y),
            step_layout="single_world",
            device=self.device,
        )
        self.world.populate_cloth_triangles_from_model(self.model)

        # ---- Wire the unified rigid + cloth-triangle pipeline --------
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model, cloth_margin=0.005
        )
        self.contacts = self.collision_pipeline.contacts()

        # Newton's ``shape_body`` uses -1 for the world anchor; PhoenX
        # uses slot 0. Shift accordingly so contact ingest finds the
        # right body slot per shape.
        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # ---- Newton state for the renderer ---------------------------
        # We mirror PhoenX body state -> ``state.body_q`` / ``body_qd``
        # and copy ``world.particles.position`` -> ``state.particle_q``
        # once per frame. The same state is forwarded to ``world.step``
        # as ``external_aabb_state`` so the cloth-aware collide path
        # reads the freshest pose / particle positions.
        self.state = self.model.state()
        # Snapshot initial particle positions for ``test_final``.
        self._initial_particle_q = self.model.particle_q.numpy().copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 1.6), pitch=-15.0, yaw=135.0)

        self._capture()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _sync_phoenx_to_newton(self) -> None:
        """Mirror PhoenX body state into the Newton ``State``.

        The state is what the viewer renders and what ``world.step``
        reads from when ``external_aabb_state`` is set (it expects
        Newton's body-origin transform convention, which is what the
        kernel reconstructs from the COM-centred PhoenX layout).
        """
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
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

    def _simulate_one_frame(self) -> None:
        """One physics frame: mirror state, then step.

        Order matters: the cloth-aware collide path inside
        ``world.step`` reads ``state.body_q`` and ``state.particle_q``,
        so we sync those first. The step then advances PhoenX's body
        container; we re-sync before the next frame's collide.
        """
        self._sync_phoenx_to_newton()
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            external_aabb_state=self.state,
        )

    def _capture(self) -> None:
        """Capture per-frame ``simulate`` into a CUDA graph."""
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt

    # ------------------------------------------------------------------
    # Viewer + tests
    # ------------------------------------------------------------------

    def _final_state(self):
        """Mirror PhoenX state into Newton state once for assertions /
        rendering after a captured-graph step. The graph captures the
        sync inside ``_simulate_one_frame`` so this is a redundant
        no-op except on the very first frame; we run it
        unconditionally for clarity."""
        self._sync_phoenx_to_newton()
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def test_final(self) -> None:
        """After the example finishes:

        * All particle / body positions are finite.
        * The four pinned corners haven't drifted.
        * The cube has fallen below its spawn height.
        * The cube hasn't tunnelled through (a few cube heights below
          the pinned cloth plane is still acceptable to allow for sag
          under load).
        """
        self._final_state()
        positions = self.state.particle_q.numpy()
        body_q = self.state.body_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")
        if not np.all(np.isfinite(body_q)):
            raise RuntimeError("non-finite body transform in final state")

        pinned_drift = np.linalg.norm(
            positions[list(self.corner_indices)] - self._initial_particle_q[list(self.corner_indices)],
            axis=1,
        )
        if pinned_drift.max() > 1.0e-3:
            raise RuntimeError(f"pinned corner drifted: max={pinned_drift.max():.4f} m")

        cube_z = float(body_q[self.cube_body, 2])
        if cube_z >= self.cube_spawn_z - 1.0e-3:
            raise RuntimeError(
                f"cube did not fall (z={cube_z:.4f} m, spawn={self.cube_spawn_z:.4f} m)"
            )

        floor = self.cloth_z - 3.0 * self.cube_he
        if cube_z < floor:
            raise RuntimeError(
                f"cube fell through cloth (z={cube_z:.4f} m, floor={floor:.4f} m)"
            )

    def render(self) -> None:
        self._final_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--width", type=int, default=24, help="Cloth resolution along x")
    parser.add_argument("--height", type=int, default=24, help="Cloth resolution along y")
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=5.0e8,
        help="Young's modulus E [Pa]. Higher = less stretchy.",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.3,
        help="Poisson ratio nu [-]. Higher = stronger area preservation.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        width=args.width,
        height=args.height,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
    )
    newton.examples.run(example, args)
