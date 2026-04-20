# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the Jitter-demo port (``example_demo_*``).

The Newton <-> Jitter integration pattern is identical across every
demo: build a scene through :class:`newton.ModelBuilder`, drive
collision detection through :class:`newton.CollisionPipeline`, mirror
the Newton bodies into a :class:`WorldBuilder`, and finally shuttle
pose / velocity between the two state containers every frame. This
module factors the boilerplate out of the individual demos so each
``example_demo_XX.py`` stays focused on the *scene* (shape layout,
joints, cameras, asserts) rather than on repeated plumbing.

Newton up-axis is +Z (default ``ModelBuilder``). Gravity is therefore
``(0, 0, -9.81)`` here -- the Jitter C# reference uses +Y-up so all
ported scenes have their y/z swapped on import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.body import MOTION_DYNAMIC, MOTION_STATIC
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import World
from newton._src.solvers.jitter.world_builder import (
    JointHandle,
    RigidBodyDescriptor,
    WorldBuilder,
)

__all__ = [
    "WORLD_BODY",
    "DemoExample",
    "build_jitter_world_from_model",
    "jitter_to_newton_kernel",
    "newton_to_jitter_kernel",
]


# Sentinel for :meth:`DemoExample.add_joint` when a joint should be
# anchored to Jitter's static world body (Jitter body 0). Pass this in
# place of a Newton body index to connect a joint to ground.
WORLD_BODY = -1


# ---------------------------------------------------------------------------
# State sync kernels (Newton <-> Jitter)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def newton_to_jitter_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # out
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
):
    """Push Newton's body state into Jitter's :class:`BodyContainer`.

    Newton stores the body-origin transform in ``body_q`` and the COM
    as a body-local offset in ``body_com``. Jitter's
    ``BodyContainer.position`` is the COM in world space, so we rotate
    ``body_com`` into world and add it to ``body_q``'s translation.
    Orientation is shared verbatim. ``body_qd`` is
    ``spatial_vector(linear_COM_velocity, angular_velocity)`` in world
    frame so the copy is direct.
    """
    i = wp.tid()
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[i] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[i] = rot
    qd = body_qd[i]
    velocity[i] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[i] = wp.vec3f(qd[3], qd[4], qd[5])


@wp.kernel(enable_backward=False)
def jitter_to_newton_kernel(
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    # out
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Write Jitter's body state back into Newton's ``body_q`` / ``body_qd``.

    Reverse of :func:`newton_to_jitter_kernel`: Jitter's ``position``
    is the COM in world space, Newton's ``body_q`` is the body-origin
    transform, so we subtract the rotated COM offset.
    """
    i = wp.tid()
    rot = orientation[i]
    com_world = wp.quat_rotate(rot, body_com[i])
    pos_body = position[i] - com_world
    body_q[i] = wp.transform(pos_body, rot)
    body_qd[i] = wp.spatial_vector(velocity[i], angular_velocity[i])


# ---------------------------------------------------------------------------
# Body mirror (Newton Model -> Jitter WorldBuilder)
# ---------------------------------------------------------------------------


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy equivalent of ``wp.quat_rotate`` with xyzw layout.

    Kept local so the host-side finalize path doesn't reach into a
    shared utils module; the whole demo stack stays import-light.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def build_jitter_world_from_model(
    model: newton.Model,
) -> tuple[WorldBuilder, dict[int, int]]:
    """Mirror Newton's body set into a fresh :class:`WorldBuilder`.

    Jitter body 0 is the auto-created static world anchor, so Newton
    body ``i`` lands at Jitter body ``i + 1``; the returned map carries
    that shift so joints / pickers can translate Newton indices to
    Jitter's.

    Returns:
        The populated builder (nothing finalized yet) and the
        ``newton_index -> jitter_index`` map.
    """
    body_q_np = model.body_q.numpy()
    body_inv_mass_np = (
        model.body_inv_mass.numpy() if model.body_inv_mass is not None else None
    )
    body_inv_inertia_np = (
        model.body_inv_inertia.numpy()
        if model.body_inv_inertia is not None
        else None
    )
    body_com_np = model.body_com.numpy()

    builder = WorldBuilder()
    newton_to_jitter: dict[int, int] = {}

    for i in range(model.body_count):
        inv_m = float(body_inv_mass_np[i]) if body_inv_mass_np is not None else 0.0
        inv_I = (
            body_inv_inertia_np[i]
            if body_inv_inertia_np is not None
            else np.zeros((3, 3), dtype=np.float32)
        )
        is_static = inv_m <= 0.0
        pose = body_q_np[i]
        origin_pos = pose[:3]
        rot = pose[3:7]  # xyzw
        com_local = body_com_np[i]
        com_world = origin_pos + _quat_rotate_np(rot, com_local)

        if is_static:
            zero = np.zeros((3, 3), dtype=np.float32)
            desc = RigidBodyDescriptor(
                position=tuple(com_world.tolist()),
                orientation=tuple(rot.tolist()),
                motion_type=int(MOTION_STATIC),
                inverse_mass=0.0,
                inverse_inertia=tuple(tuple(float(x) for x in r) for r in zero),
                affected_by_gravity=False,
            )
        else:
            desc = RigidBodyDescriptor(
                position=tuple(com_world.tolist()),
                orientation=tuple(rot.tolist()),
                motion_type=int(MOTION_DYNAMIC),
                inverse_mass=inv_m,
                inverse_inertia=tuple(tuple(float(x) for x in r) for r in inv_I),
                affected_by_gravity=True,
            )
        newton_to_jitter[i] = builder.add_body(desc)

    return builder, newton_to_jitter


# ---------------------------------------------------------------------------
# Reusable demo base class
# ---------------------------------------------------------------------------


@dataclass
class DemoConfig:
    """Per-demo knobs a subclass sets before :meth:`DemoExample.finish_setup`.

    Central so every demo looks the same at the top of ``__init__``.
    """

    #: Window title / label; also used for log lines.
    title: str = "Jitter Demo"
    #: Camera pose (world space, Newton +Z-up convention).
    camera_pos: tuple[float, float, float] = (10.0, 10.0, 5.0)
    camera_pitch: float = -20.0
    camera_yaw: float = -45.0
    #: Integration step settings.
    fps: int = 60
    substeps: int = 4
    solver_iterations: int = 8
    #: Gravity vector (Newton +Z-up).
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


class DemoExample:
    """Base class for the Jitter-port demo scenes.

    Subclasses:

    1. set :attr:`config` (a :class:`DemoConfig`),
    2. populate ``self.model_builder`` (a :class:`newton.ModelBuilder`)
       with their scene inside :meth:`build_scene`,
    3. optionally populate ``self.joints`` (world-space anchor pairs
       mirrored through :meth:`WorldBuilder.add_joint`),
    4. call :meth:`finish_setup` at the end of their ``__init__``.

    Rendering uses ``viewer.log_state`` so all Newton shape types
    (box, sphere, capsule, cylinder, plane) render through whatever
    path the Newton viewer already supports -- no per-demo
    ``log_shapes`` boilerplate.

    Graph capture (CUDA) and picking are set up automatically. Each
    subclass can override :meth:`test_final` to assert a settled state.
    """

    def __init__(self, viewer, args, config: DemoConfig):
        self.viewer = viewer
        self.args = args
        self.config = config
        self.device = wp.get_device()

        self.fps = int(config.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(config.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_iterations = int(config.solver_iterations)
        self._frame: int = 0

        # Per-demo scratch -- subclasses fill these out.
        self.model_builder: newton.ModelBuilder = newton.ModelBuilder()
        self.pending_joints: list[dict] = []
        # Newton body indices that picking should treat as pickable
        # (half-extent / radius for the ray cube test). Populated by
        # :meth:`register_body_extent` so :meth:`finish_setup` can
        # build Jitter's half-extents array.
        self._newton_body_extents: dict[int, tuple[float, float, float]] = {}

    # ------------------------------------------------------------------
    # Hooks subclasses override
    # ------------------------------------------------------------------

    def build_scene(self) -> None:
        """Populate :attr:`model_builder` and :attr:`pending_joints`.

        Override in each demo. When this returns, ``self.model_builder``
        must be in a finalise-able state and ``self.pending_joints``
        must contain plain dicts forwarded to
        :meth:`WorldBuilder.add_joint` (with Newton body indices --
        these get shifted to Jitter indices in :meth:`finish_setup`).
        """
        raise NotImplementedError

    def test_final(self) -> None:
        """Optional: assert the settled scene state.

        Default is a no-op. Override to check that stacks stayed
        stacked, motors turned, etc.
        """
        return None

    # ------------------------------------------------------------------
    # Shared scene-construction helpers
    # ------------------------------------------------------------------

    def register_body_extent(
        self, newton_body: int, extent: tuple[float, float, float]
    ) -> None:
        """Record a bounding half-extent (or bounding sphere triple) for
        picking. Entries with any component ``<= 0`` are treated as
        unpickable; matches the world-anchor convention."""
        self._newton_body_extents[int(newton_body)] = (
            float(extent[0]),
            float(extent[1]),
            float(extent[2]),
        )

    def add_joint(self, **kwargs) -> None:
        """Queue a joint for :class:`WorldBuilder.add_joint`.

        Uses Newton body indices; :meth:`finish_setup` translates them
        to Jitter indices after :meth:`build_jitter_world_from_model`
        runs.
        """
        self.pending_joints.append(dict(kwargs))

    # ------------------------------------------------------------------
    # finish_setup: finalize Newton model + Jitter world, wire viewer
    # ------------------------------------------------------------------

    def finish_setup(self) -> None:
        """Finalise the Newton model, build the Jitter world, capture."""
        self.build_scene()
        self.model = self.model_builder.finalize()
        print(
            f"[{self.config.title}] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count}"
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=True
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        builder, newton_to_jitter = build_jitter_world_from_model(self.model)
        # Translate queued joints from Newton indices to Jitter
        # indices and install them in the builder.
        self._joint_handles: list[JointHandle] = []
        for jspec in self.pending_joints:
            jargs = dict(jspec)
            # ``WORLD_BODY`` (= -1) maps to Jitter body 0 (the static
            # world anchor created by :class:`WorldBuilder.__init__`);
            # any other index is a Newton body we've already mirrored.
            b1 = int(jargs["body1"])
            b2 = int(jargs["body2"])
            jargs["body1"] = (
                int(builder.world_body) if b1 == WORLD_BODY else newton_to_jitter[b1]
            )
            jargs["body2"] = (
                int(builder.world_body) if b2 == WORLD_BODY else newton_to_jitter[b2]
            )
            self._joint_handles.append(builder.add_joint(**jargs))

        # Budget: each Newton contact column holds up to 6 contact
        # points, so ``ceil(rigid_contact_max / 6)`` is a tight
        # worst case. Sized floor of 16 avoids tiny grids on trivial
        # scenes where the contact column count would otherwise
        # shrink below a warp.
        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        num_shapes = int(self.model.shape_count)

        self.world: World = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            gravity=self.config.gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            device=self.device,
        )
        self._newton_to_jitter = newton_to_jitter

        # Shape -> Jitter body map. Newton's ``shape_body`` holds
        # Newton body indices (or -1 for the world); shift by +1 to
        # match Jitter's body layout (Jitter body 0 is the static
        # world anchor).
        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        self._sync_newton_to_jitter()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(*self.config.camera_pos),
            pitch=self.config.camera_pitch,
            yaw=self.config.camera_yaw,
        )

        # Picking: one half-extent triple per Jitter body, body 0
        # marked (0,0,0) so ray-casts ignore the world anchor.
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx, extent in self._newton_body_extents.items():
            jitter_idx = self._newton_to_jitter[newton_idx]
            half_extents_np[jitter_idx] = extent
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = JitterPicking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # CUDA graph capture for the whole per-frame step.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        # Diagnostic -- subclasses can flip to False if they want
        # quieter output.
        self._log_colors_every = 60

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._sync_newton_to_jitter()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            picking=self.picking,
        )
        self._sync_jitter_to_newton()

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_jitter_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_jitter_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            jitter_to_newton_kernel,
            dim=n,
            inputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
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
        self._frame += 1
        if self._log_colors_every > 0 and self._frame % self._log_colors_every == 0:
            print(
                f"[{self.config.title}] frame {self._frame:5d}  "
                f"colors used by PGS: {self.world.num_colors_used()}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Newton-style helpers used by assertions
    # ------------------------------------------------------------------

    def jitter_body_state(
        self, newton_body: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position, velocity)`` for ``newton_body`` as NumPy
        arrays, copying from Jitter's :class:`BodyContainer`. Intended
        for use inside :meth:`test_final`.
        """
        j = self._newton_to_jitter[int(newton_body)]
        return (
            self.world.bodies.position.numpy()[j],
            self.world.bodies.velocity.numpy()[j],
        )

    @staticmethod
    def default_parser():
        return newton.examples.create_parser()
