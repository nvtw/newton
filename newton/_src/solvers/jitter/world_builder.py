# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side builder that assembles a :class:`World` from plain Python.

Construction is split in two phases:

1. *Append* phase -- ``add_static_body`` / ``add_dynamic_body`` /
   ``add_ball_socket`` / ... record per-body / per-constraint
   *descriptors* in plain Python lists. Nothing touches the GPU.
2. *Finalize* phase -- :meth:`WorldBuilder.finalize` allocates a
   :class:`BodyContainer` and a :class:`ConstraintContainer` sized
   exactly to the appended counts, packs every descriptor into the SoA
   body arrays via NumPy, computes the per-constraint-type cid ranges,
   uploads per-type batch arrays, and launches one initialise kernel
   per type to fill the column-major constraint storage. Returns a
   ready-to-step :class:`World`.

This keeps the user-facing API trivial (regular Python objects, no
Warp types until the very end) while still hitting the GPU in batched
launches at finalize time.

Body 0 is always the static world body (created in ``__init__``); user
descriptor indices start at 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.jitter.constraint_ball_socket import (
    BS_DWORDS,
    ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.jitter.solver_jitter import World

__all__ = [
    "BallSocketDescriptor",
    "RigidBodyDescriptor",
    "WorldBuilder",
]


_IDENTITY_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass
class RigidBodyDescriptor:
    """Plain-Python description of one rigid body.

    Mirrors the writable subset of Jitter2's ``RigidBody`` constructor
    arguments. Defaults match Jitter (no damping, gravity on, identity
    inertia) so a bare ``RigidBodyDescriptor()`` produces a
    static-by-default body sitting at the origin.

    Units (where applicable):
        * ``position`` [m]
        * ``inverse_mass`` [1/kg]
        * ``inverse_inertia`` [1/(kg m^2)] in the *body* frame; the
          solver rotates this into world space every step.
        * ``velocity`` [m/s]
        * ``angular_velocity`` [rad/s]
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    motion_type: int = int(MOTION_STATIC)
    inverse_mass: float = 0.0
    inverse_inertia: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = _IDENTITY_INERTIA
    linear_damping: float = 1.0
    angular_damping: float = 1.0
    affected_by_gravity: bool = True
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BallSocketDescriptor:
    """Plain-Python description of one ball-and-socket constraint.

    The ``anchor`` is a single point in *world* space; finalize() asks
    each body to express it in its own local frame (mirrors
    ``BallSocket.Initialize``).
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]


class WorldBuilder:
    """Append bodies and constraints, then materialise a :class:`World`.

    Usage::

        b = WorldBuilder()
        anchor = b.add_static_body()                # body 0 (auto)
        body_a = b.add_dynamic_body(position=(0, 0, 0), inverse_mass=1.0)
        body_b = b.add_dynamic_body(position=(1, 0, 0), inverse_mass=1.0)
        b.add_ball_socket(anchor, body_a, anchor=(0, 0, 0))
        b.add_ball_socket(body_a, body_b, anchor=(0.5, 0, 0))
        world = b.finalize(substeps=1, solver_iterations=8)

    The builder is single-use; call :meth:`finalize` exactly once.
    """

    def __init__(self):
        # Body 0 is the static world body; user calls to add_*_body
        # start at index 1. We seed it with the canonical defaults
        # (origin, identity rotation, MOTION_STATIC, no inertia).
        self._bodies: list[RigidBodyDescriptor] = [RigidBodyDescriptor()]
        self._ball_sockets: list[BallSocketDescriptor] = []

    # ------------------------------------------------------------------
    # Body API
    # ------------------------------------------------------------------

    @property
    def world_body(self) -> int:
        """Index of the auto-created static world body."""
        return 0

    def add_body(self, descriptor: RigidBodyDescriptor) -> int:
        """Append a fully-formed descriptor and return its body index."""
        self._bodies.append(descriptor)
        return len(self._bodies) - 1

    def add_static_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        """Append a static body (zero inverse mass / inertia, no integration)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_STATIC),
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
            )
        )

    def add_kinematic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Append a kinematic body: integrates its user-set velocities but
        ignores forces/contacts (matches Jitter ``MotionType.Kinematic``)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_KINEMATIC),
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
                velocity=velocity,
                angular_velocity=angular_velocity,
            )
        )

    def add_dynamic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        inverse_mass: float = 1.0,
        inverse_inertia: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = _IDENTITY_INERTIA,
        linear_damping: float = 1.0,
        angular_damping: float = 1.0,
        affected_by_gravity: bool = True,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Append a fully dynamic body. ``inverse_inertia`` is in the
        *body* frame; the solver rotates it to world space each step."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_DYNAMIC),
                inverse_mass=inverse_mass,
                inverse_inertia=inverse_inertia,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
                affected_by_gravity=affected_by_gravity,
                velocity=velocity,
                angular_velocity=angular_velocity,
            )
        )

    # ------------------------------------------------------------------
    # Constraint API
    # ------------------------------------------------------------------

    def add_ball_socket(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
    ) -> int:
        """Append a ball-and-socket constraint and return its (per-type)
        index. Both ``body1`` and ``body2`` must be valid body indices
        (i.e. returned from one of the ``add_*_body`` methods or 0 for
        the static world body)."""
        self._validate_body(body1)
        self._validate_body(body2)
        self._ball_sockets.append(BallSocketDescriptor(body1, body2, anchor))
        return len(self._ball_sockets) - 1

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(
        self,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_relaxations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        device: wp.context.Devicelike = None,
    ) -> World:
        """Allocate GPU storage and build a ready-to-step :class:`World`.

        The current registered descriptor lists are consumed in place
        (cleared after success). Calling :meth:`finalize` twice produces
        an empty world the second time -- clone the builder first if
        that's not what you want.

        Args:
            substeps: Forwarded to :class:`World`. Number of substeps
                per :meth:`World.step` call.
            solver_iterations: PGS iterations per substep.
            velocity_relaxations: Relaxation passes (currently a stub
                inside ``World``).
            gravity: World gravity vector [m/s^2].
            device: Warp device. ``None`` selects the current default
                device (typically ``cuda:0``).
        """
        device = wp.get_device(device)

        bodies = self._build_body_container(device)
        constraints, cid_ranges = self._build_constraint_container(bodies, device)

        world = World(
            bodies=bodies,
            constraints=constraints,
            num_ball_sockets=cid_ranges["ball_socket_count"],
            ball_socket_cid_offset=cid_ranges["ball_socket_offset"],
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_relaxations=velocity_relaxations,
            gravity=gravity,
            device=device,
        )

        # Reset internal state so the builder can't accidentally leak
        # references into a second finalize call.
        self._bodies = [RigidBodyDescriptor()]
        self._ball_sockets = []
        return world

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_body(self, idx: int) -> None:
        if not (0 <= idx < len(self._bodies)):
            raise IndexError(
                f"body index {idx} out of range [0, {len(self._bodies)})"
            )

    def _build_body_container(self, device: wp.context.Device) -> BodyContainer:
        """Pack the descriptor list into a :class:`BodyContainer` via NumPy.

        Builds one NumPy array per field and uploads it. This is one
        host-to-device transfer per field, which dwarfs anything we
        could save by interleaving -- finalize() is a one-shot setup
        path, not a hot loop.
        """
        n = len(self._bodies)
        positions = np.zeros((n, 3), dtype=np.float32)
        orientations = np.zeros((n, 4), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        angular_velocities = np.zeros((n, 3), dtype=np.float32)
        inverse_inertia = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_inertia_world = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_mass = np.zeros(n, dtype=np.float32)
        linear_damping = np.ones(n, dtype=np.float32)
        angular_damping = np.ones(n, dtype=np.float32)
        affected_by_gravity = np.ones(n, dtype=np.int32)
        motion_type = np.full(n, int(MOTION_STATIC), dtype=np.int32)

        for i, b in enumerate(self._bodies):
            positions[i] = b.position
            orientations[i] = b.orientation
            velocities[i] = b.velocity
            angular_velocities[i] = b.angular_velocity
            inverse_inertia[i] = b.inverse_inertia
            # Seed inverse_inertia_world with the body-frame value; the
            # first _update_bodies launch will rotate it into world
            # space using the actual orientation. Without this seed the
            # very first prepare_for_iteration sees zero effective mass.
            inverse_inertia_world[i] = b.inverse_inertia
            inverse_mass[i] = b.inverse_mass
            linear_damping[i] = b.linear_damping
            angular_damping[i] = b.angular_damping
            affected_by_gravity[i] = 1 if b.affected_by_gravity else 0
            motion_type[i] = int(b.motion_type)

        c = BodyContainer()
        c.position = wp.array(positions, dtype=wp.vec3f, device=device)
        c.velocity = wp.array(velocities, dtype=wp.vec3f, device=device)
        c.angular_velocity = wp.array(angular_velocities, dtype=wp.vec3f, device=device)
        c.orientation = wp.array(orientations, dtype=wp.quatf, device=device)
        c.inverse_inertia_world = wp.array(inverse_inertia_world, dtype=wp.mat33f, device=device)
        c.inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33f, device=device)
        c.inverse_mass = wp.array(inverse_mass, dtype=wp.float32, device=device)
        c.force = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.torque = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.delta_velocity = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.delta_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.linear_damping = wp.array(linear_damping, dtype=wp.float32, device=device)
        c.angular_damping = wp.array(angular_damping, dtype=wp.float32, device=device)
        c.affected_by_gravity = wp.array(affected_by_gravity, dtype=wp.int32, device=device)
        c.motion_type = wp.array(motion_type, dtype=wp.int32, device=device)
        return c

    def _build_constraint_container(
        self,
        bodies: BodyContainer,
        device: wp.context.Device,
    ) -> tuple[ConstraintContainer, dict]:
        """Allocate the shared :class:`ConstraintContainer` and pack every
        constraint type into its own contiguous cid range.

        Returns the container and a dict of ``{type}_offset / {type}_count``
        cid ranges so :meth:`finalize` can pass the right slice arguments
        to :class:`World`.
        """
        # Per-type counts. Today only ball-socket; when more types land
        # add their counts/offsets here and bump num_dwords accordingly.
        n_ball = len(self._ball_sockets)

        total = n_ball
        # max() over registered type dword counts; today only one.
        per_constraint_dwords = BS_DWORDS

        # Always allocate at least 1 column so the wp.array2d shape is
        # non-degenerate; the kernels gate on cid bounds anyway.
        container = constraint_container_zeros(
            num_constraints=max(1, total),
            num_dwords=per_constraint_dwords,
            device=device,
        )

        ball_socket_offset = 0
        if n_ball > 0:
            self._init_ball_sockets(container, bodies, ball_socket_offset, device)

        return container, {
            "ball_socket_offset": ball_socket_offset,
            "ball_socket_count": n_ball,
        }

    def _init_ball_sockets(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._ball_sockets)

        body1 = np.asarray([d.body1 for d in self._ball_sockets], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._ball_sockets], dtype=np.int32)
        anchor = np.asarray([d.anchor for d in self._ball_sockets], dtype=np.float32)

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)

        wp.launch(
            ball_socket_initialize_kernel,
            dim=n,
            inputs=[constraints, bodies, int(cid_offset), body1_d, body2_d, anchor_d],
            device=device,
        )
