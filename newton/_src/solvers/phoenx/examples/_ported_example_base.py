# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared base class for the PhoenX-side ports of Box2D / JitterPhysics2 demos.

Every ported demo plugs into the same per-frame pipeline:

1. :class:`newton.ModelBuilder` -> :class:`newton.Model`
2. :class:`newton.CollisionPipeline` for contacts
3. PhoenX :class:`BodyContainer` seeded from the model
4. :class:`PhoenXWorld` step loop captured into a CUDA graph

The boilerplate is identical across demos -- only the body / shape /
joint construction and the camera differ. Subclasses implement
:meth:`build_scene` to populate a :class:`newton.ModelBuilder` and
:meth:`configure_camera` to position the view; the base does the rest.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

__all__ = [
    "PortedExample",
    "default_box_half_extents",
    "default_capsule_half_extents",
    "default_sphere_half_extents",
]


def default_box_half_extents(hx: float, hy: float, hz: float) -> tuple[float, float, float]:
    return (float(hx), float(hy), float(hz))


def default_sphere_half_extents(radius: float) -> tuple[float, float, float]:
    return (float(radius), float(radius), float(radius))


def default_capsule_half_extents(radius: float, half_height: float) -> tuple[float, float, float]:
    """OBB half-extents for picking: radius x radius x (radius + half_height) along the
    capsule axis (assumed +z in body frame)."""
    return (float(radius), float(radius), float(radius + half_height))


class PortedExample:
    """Boilerplate for a single-world PhoenX scene built via Newton.

    Subclasses implement:

    * :meth:`build_scene(builder)` -- populate ``builder`` with bodies,
      shapes, joints. Return a list of pick-OBB half-extents per Newton
      body (length ``builder.body_count``); pass ``None`` to skip
      picking on a body.
    * :meth:`configure_camera(viewer)` -- ``viewer.set_camera(...)``.
    * Override :attr:`fps`, :attr:`sim_substeps`, :attr:`solver_iterations`,
      :attr:`velocity_iterations`, :attr:`gravity`, :attr:`step_layout`,
      :attr:`broad_phase`, :attr:`shape_pairs_max`, :attr:`default_friction`,
      :attr:`default_restitution` as needed.
    """

    fps: int = 60
    sim_substeps: int = 4
    solver_iterations: int = 8
    velocity_iterations: int = 1
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    step_layout: str = "multi_world"
    broad_phase: str = "nxn"
    shape_pairs_max: int | None = None
    default_friction: float = 0.5
    default_restitution: float = 0.0

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self._build()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def build_scene(self, builder: newton.ModelBuilder) -> list[tuple[float, float, float] | None]:
        """Populate ``builder``; return per-body pick half-extents."""
        raise NotImplementedError

    def configure_camera(self, viewer) -> None:
        """Default camera: 6 m back along +x, 2 m up, looking at origin."""
        viewer.set_camera(pos=wp.vec3(6.0, 0.0, 2.0), pitch=-15.0, yaw=180.0)

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self._builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0,
            mu=self.default_friction,
            restitution=self.default_restitution,
        )
        self._builder.default_shape_cfg = cfg
        pick_extents = self.build_scene(self._builder)

        self.model = self._builder.finalize()

        if self.shape_pairs_max is not None:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                contact_matching=PHOENX_CONTACT_MATCHING,
                broad_phase=self.broad_phase,
                shape_pairs_max=self.shape_pairs_max,
            )
        else:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                contact_matching=PHOENX_CONTACT_MATCHING,
                broad_phase=self.broad_phase,
            )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        if self.model.body_count > 0:
            wp.launch(
                init_phoenx_bodies_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.model.body_q,
                    self.state.body_qd,
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

        # Joint-only constraint container is a placeholder; contact
        # constraints live in PhoenXWorld's contact column container.
        self.constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=self.device)

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=self.gravity,
            rigid_contact_max=rigid_contact_max,
            default_friction=self.default_friction,
            step_layout=self.step_layout,
            device=self.device,
        )

        self.viewer.set_model(self.model)
        self.configure_camera(self.viewer)

        # Picking: per-body OBB half-extents. Bodies the subclass marks
        # ``None`` get a zero extent so picking ignores them.
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx, he in enumerate(pick_extents):
            if he is None:
                continue
            half_extents_np[newton_idx + 1] = he
        self._pick_half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._pick_half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._sync_newton_to_phoenx()
        self.model.collide(self.state, contacts=self.contacts, collision_pipeline=self.collision_pipeline)
        self.picking.apply_force()
        self.world.step(dt=self.frame_dt, contacts=self.contacts, shape_body=self._shape_body)
        self._sync_phoenx_to_newton()

    def _sync_newton_to_phoenx(self) -> None:
        n = self.model.body_count
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
        n = self.model.body_count
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

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()


def run_ported_example(example_factory: Callable[[object, object], PortedExample]) -> None:
    """Standard ``__main__`` entry point: ``newton.examples.init`` ->
    factory -> ``newton.examples.run``. Subclasses use this to keep the
    entry-point boilerplate one line."""
    viewer, args = newton.examples.init()
    example = example_factory(viewer, args)
    newton.examples.run(example, args)
