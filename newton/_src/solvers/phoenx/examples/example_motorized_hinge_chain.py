# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Motorized Hinge Chain
#
# PhoenX variant of :mod:`example_motorized_hinge_chain`. Identical
# geometry -- ``NUM_CUBES`` unit cubes rotated 45 degrees about +z
# so they hang in a diamond column along -y -- but every joint is an
# :data:`~newton._src.solvers.phoenx.world_builder.JointMode.REVOLUTE`
# actuated double-ball-socket solved by :class:`PhoenXWorld`.
#
# The shared unified-joint schema lets the PhoenX solver reuse the
# jitter solver's fast-path dispatcher and graph-colouring machinery
# for joints + contacts in one CSR sweep. All we need to do on the
# PhoenX side is:
#
#   * Size the :class:`ConstraintContainer` for ``NUM_HINGES`` joint
#     columns + any contact column capacity (zero contacts here
#     because the chain never touches itself / the ground),
#   * Populate the joint columns once via
#     :meth:`PhoenXWorld.initialize_actuated_double_ball_socket_joints`,
#   * Populate the body container directly (no ``WorldBuilder`` --
#     PhoenX takes raw containers).
#
# The chain is driven by the module-level ``DRIVE_MODE`` + its
# companion target, same knobs as the jitter version. Default is a
# zero-velocity motor so the chain hangs in equilibrium.
#
# Picking is wired through :class:`Picking`, which binds to the
# body container directly and therefore works unchanged against
# :class:`PhoenXWorld`.
#
# Run:  python -m newton._src.solvers.phoenx.examples.example_motorized_hinge_chain
###########################################################################

from __future__ import annotations

import enum
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    body_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_phoenx import (
    PhoenXWorld,
    pack_body_xforms_kernel,
)
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode


class JointKind(enum.Enum):
    """Which unified-joint mode to build the chain with.

    :class:`PhoenXWorld` only supports the actuated double-ball-socket
    joint (see :mod:`solver_phoenx`), so unlike the jitter version
    we stop at a single revolute mode -- the three
    :data:`JointMode.REVOLUTE` / :data:`PRISMATIC` / :data:`BALL_SOCKET`
    values of that joint are the only choices.
    """

    ACTUATED_DOUBLE_BALL_SOCKET = "actuated_double_ball_socket"


JOINT_KIND = JointKind.ACTUATED_DOUBLE_BALL_SOCKET

# How the motor drives the free axial spin.
#   * :attr:`DriveMode.OFF`      -- free-spin axis, no motor.
#   * :attr:`DriveMode.VELOCITY` -- tracks ``TARGET_VELOCITY`` [rad/s].
#   * :attr:`DriveMode.POSITION` -- pulls the axial angle towards
#     ``TARGET_ANGLE`` [rad] with a critically-damped soft spring.
DRIVE_MODE = DriveMode.VELOCITY

NUM_CUBES = 250
HALF_EXTENT = 0.05
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body at slot 0
NUM_HINGES = NUM_CUBES  # 1 world->cube0 + (N-1) cube_{k-1}->cube_k

# Identity body-frame inverse inertia; matches the jitter scene.
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# 45-degree rotation about +z (xyzw). Puts diagonal cube corners on the
# world y axis at distance ``h*sqrt(2)``. The body-frame +z axis stays
# aligned with world +z, so the four vertical cube edges (the ones
# parallel to body +z) stay on world +z -- that's the hinge axis the
# revolute joint uses between two stacked cubes.
_DIAGONAL_HALF = HALF_EXTENT * math.sqrt(2.0)
_HALF_ANGLE = math.pi / 8.0  # half of 45 degrees
_DIAGONAL_QUAT = (0.0, 0.0, math.sin(_HALF_ANGLE), math.cos(_HALF_ANGLE))

# Motor torque cap [N*m] -- generous so the PD drive can hold its
# target against PGS jitter even on the top joint (which carries the
# full chain weight).
_MOTOR_MAX_FORCE = 50.0

# Position-drive soft-spring knobs: 4 Hz critically-damped angular
# spring on unit-inertia cubes (``omega = 2*pi*hertz``, ``zeta = 1``).
#   kp = I * omega^2
#   kd = 2*I*zeta*omega
_HERTZ_DRIVE = 4.0
_DAMPING_RATIO_DRIVE = 1.0
_STIFFNESS_DRIVE = (2.0 * math.pi * _HERTZ_DRIVE) ** 2
_DAMPING_DRIVE = 2.0 * _DAMPING_RATIO_DRIVE * (2.0 * math.pi * _HERTZ_DRIVE)

# Per-joint velocity-drive setpoint [rad/s].
#   * 0.0  -> the motor fights any relative spin; chain stays still.
#   * non-zero -> each hinge spins the two cubes about the chain axis.
TARGET_VELOCITY = 0.0

# Per-joint position-drive setpoint [rad]. 0.0 matches the rest pose
# (every cube spawns with ``_DIAGONAL_QUAT``), so a zero target holds
# the initial orientations. Non-zero values compound down the chain
# and visibly coil it.
TARGET_ANGLE = 0.0


def _populate_chain_bodies(
    bodies,
    device: wp.context.Device,
) -> None:
    """Fill :class:`BodyContainer` slots with the chain's initial state.

    Slot 0: static world anchor (default-initialised -- mass / inertia
    already zero, motion type already :data:`MOTION_STATIC`).
    Slots 1..NUM_BODIES: dynamic cubes in a diamond column along -y.
    Each cube has unit mass and identity body-frame inertia; the
    rotation stays fixed at ``_DIAGONAL_QUAT`` so the shared-corner
    anchors sit on the world y axis.
    """
    positions = np.zeros((NUM_BODIES, 3), dtype=np.float32)
    orientations = np.zeros((NUM_BODIES, 4), dtype=np.float32)
    # World anchor: identity orientation.
    orientations[0] = (0.0, 0.0, 0.0, 1.0)
    for j in range(NUM_CUBES):
        positions[j + 1] = (0.0, -(2 * j + 1) * _DIAGONAL_HALF, 0.0)
        orientations[j + 1] = _DIAGONAL_QUAT
    bodies.position.assign(positions)
    bodies.orientation.assign(orientations)

    inv_mass_np = np.zeros(NUM_BODIES, dtype=np.float32)
    inv_mass_np[1:] = 1.0  # unit mass on every dynamic cube
    bodies.inverse_mass.assign(inv_mass_np)

    # Body-frame inertia: identity -> inverse identity.
    inv_inertia_np = np.zeros((NUM_BODIES, 3, 3), dtype=np.float32)
    eye = np.array(_INV_INERTIA, dtype=np.float32)
    for j in range(1, NUM_BODIES):
        inv_inertia_np[j] = eye
    bodies.inverse_inertia.assign(inv_inertia_np)
    # World-space inertia starts at the same value because every cube
    # is rotated purely about +z and ``eye`` is rotation-invariant.
    bodies.inverse_inertia_world.assign(inv_inertia_np)

    motion = np.full(NUM_BODIES, int(MOTION_STATIC), dtype=np.int32)
    motion[1:] = int(MOTION_DYNAMIC)
    bodies.motion_type.assign(motion)


def _build_joint_arrays(
    device: wp.context.Device,
) -> dict[str, wp.array]:
    """Assemble the per-joint descriptor arrays the init kernel needs.

    Every joint connects anchor corner (world y axis) of
    consecutive diamond cubes, hinge axis +z, with the motor set up
    per ``DRIVE_MODE``. Anchor 1 sits at ``z - HALF_EXTENT`` and
    anchor 2 at ``z + HALF_EXTENT`` so the implicit hinge axis
    ``anchor2 - anchor1`` aligns with world +z.
    """
    body1 = np.zeros(NUM_HINGES, dtype=np.int32)
    body2 = np.zeros(NUM_HINGES, dtype=np.int32)
    anchor1 = np.zeros((NUM_HINGES, 3), dtype=np.float32)
    anchor2 = np.zeros((NUM_HINGES, 3), dtype=np.float32)
    # Slot 0 is the static world anchor; the first cube is slot 1.
    world_slot = 0
    for k in range(NUM_HINGES):
        body1[k] = world_slot if k == 0 else (k)  # cube_{k-1} -> slot k
        body2[k] = k + 1  # cube_k -> slot k+1
        y = -k * 2.0 * _DIAGONAL_HALF
        anchor1[k] = (0.0, y, -HALF_EXTENT)
        anchor2[k] = (0.0, y, HALF_EXTENT)

    target = np.full(NUM_HINGES, float(TARGET_ANGLE), dtype=np.float32)
    target_velocity = np.full(NUM_HINGES, float(TARGET_VELOCITY), dtype=np.float32)
    max_force_drive = np.full(NUM_HINGES, float(_MOTOR_MAX_FORCE), dtype=np.float32)
    stiffness_drive = np.full(NUM_HINGES, float(_STIFFNESS_DRIVE), dtype=np.float32)
    damping_drive = np.full(NUM_HINGES, float(_DAMPING_DRIVE), dtype=np.float32)

    # No angle limit (min > max disables the limit row).
    min_value = np.full(NUM_HINGES, 1.0, dtype=np.float32)
    max_value = np.full(NUM_HINGES, -1.0, dtype=np.float32)

    joint_mode = np.full(NUM_HINGES, int(JointMode.REVOLUTE), dtype=np.int32)
    drive_mode = np.full(NUM_HINGES, int(DRIVE_MODE), dtype=np.int32)

    # Positional block soft-constraint knobs. The jitter WorldBuilder
    # defaults these to ``DEFAULT_HERTZ_LINEAR = 1e9`` (i.e. a
    # maximally-stiff lock that behaves as a rigid joint in single
    # precision) / ``DEFAULT_DAMPING_RATIO = 1.0``. Using a low hertz
    # (e.g. 60 Hz) here would turn each joint into a soft spring --
    # fine for one joint, but 50 soft springs in series stretch
    # catastrophically under the full chain weight (each joint sags
    # by ``weight / kp`` and that adds up along the chain). Matching
    # the jitter default keeps every joint rigid-enough that the
    # top-link load propagates straight through to the bottom.
    hertz = np.full(NUM_HINGES, float(DEFAULT_HERTZ_LINEAR), dtype=np.float32)
    damping_ratio = np.full(NUM_HINGES, float(DEFAULT_DAMPING_RATIO), dtype=np.float32)
    hertz_limit = np.full(NUM_HINGES, float(DEFAULT_HERTZ_LINEAR), dtype=np.float32)
    damping_ratio_limit = np.full(NUM_HINGES, float(DEFAULT_DAMPING_RATIO), dtype=np.float32)
    stiffness_limit = np.zeros(NUM_HINGES, dtype=np.float32)
    damping_limit = np.zeros(NUM_HINGES, dtype=np.float32)

    return {
        "body1": wp.array(body1, dtype=wp.int32, device=device),
        "body2": wp.array(body2, dtype=wp.int32, device=device),
        "anchor1": wp.array(anchor1, dtype=wp.vec3f, device=device),
        "anchor2": wp.array(anchor2, dtype=wp.vec3f, device=device),
        "hertz": wp.array(hertz, dtype=wp.float32, device=device),
        "damping_ratio": wp.array(damping_ratio, dtype=wp.float32, device=device),
        "joint_mode": wp.array(joint_mode, dtype=wp.int32, device=device),
        "drive_mode": wp.array(drive_mode, dtype=wp.int32, device=device),
        "target": wp.array(target, dtype=wp.float32, device=device),
        "target_velocity": wp.array(target_velocity, dtype=wp.float32, device=device),
        "max_force_drive": wp.array(max_force_drive, dtype=wp.float32, device=device),
        "stiffness_drive": wp.array(stiffness_drive, dtype=wp.float32, device=device),
        "damping_drive": wp.array(damping_drive, dtype=wp.float32, device=device),
        "min_value": wp.array(min_value, dtype=wp.float32, device=device),
        "max_value": wp.array(max_value, dtype=wp.float32, device=device),
        "hertz_limit": wp.array(hertz_limit, dtype=wp.float32, device=device),
        "damping_ratio_limit": wp.array(damping_ratio_limit, dtype=wp.float32, device=device),
        "stiffness_limit": wp.array(stiffness_limit, dtype=wp.float32, device=device),
        "damping_limit": wp.array(damping_limit, dtype=wp.float32, device=device),
    }


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 50
        self.solver_iterations = 4

        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # ---- Body container ------------------------------------------
        # Raw :class:`BodyContainer` -- PhoenX has no ``WorldBuilder``
        # since it does not construct joints through an abstraction
        # layer; direct population of the SoA is the one and only
        # API. Slot 0 is the static world anchor (default-initialised
        # static), slots 1..NUM_CUBES are the diamond cubes.
        self.bodies = body_container_zeros(NUM_BODIES, device=self.device)
        _populate_chain_bodies(self.bodies, self.device)

        # ---- Constraint container with reserved joint slots ----------
        # Zero contact capacity -- the chain never produces contacts
        # (it hangs in free space, no ground plane, no inter-cube
        # penetration). The factory picks ``ADBS_DWORDS`` width since
        # we asked for joints.
        max_contact_columns = 0
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=NUM_HINGES,
            max_contact_columns=max_contact_columns,
            device=self.device,
        )

        # ---- Solver ---------------------------------------------------
        # No contacts in this scene, so ``max_contact_columns = 0`` --
        # the solver's contact ingest paths stay dormant and the
        # partitioner only sees joint elements.
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=1,
            gravity=(0.0, 0.0, -9.81),  # match the jitter variant's -y gravity
            max_contact_columns=max_contact_columns,
            rigid_contact_max=0,
            num_joints=NUM_HINGES,
            device=self.device,
        )

        # ---- Joint initialisation ------------------------------------
        # One-shot launch of the shared init kernel over cids
        # ``[0, NUM_HINGES)``. The dispatcher (fast-path tail kernels
        # in :mod:`solver_phoenx_kernels`) handles everything beyond
        # this point.
        joint_arrays = _build_joint_arrays(self.device)
        self.world.initialize_actuated_double_ball_socket_joints(**joint_arrays)

        # ---- Rendering scratch ---------------------------------------
        self._xforms = wp.zeros(NUM_BODIES, dtype=wp.transform, device=self.device)

        # ---- Picking --------------------------------------------------
        # Half-extents per body in body-local frame; (0, 0, 0) marks
        # the world anchor as non-pickable.
        half_extents_np = np.zeros((NUM_BODIES, 3), dtype=np.float32)
        half_extents_np[1:] = HALF_EXTENT
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # ---- CUDA graph capture --------------------------------------
        self.capture()

    def capture(self) -> None:
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        # Picking PD force is accumulated into ``bodies.force`` once
        # per frame, before the step. The solver's
        # ``_phoenx_apply_external_forces_kernel`` picks it up each
        # substep and ``_clear_forces`` zeroes it for the next frame.
        self.picking.apply_force()
        # Chain scene has no contacts -> pass ``contacts=None`` so the
        # solver skips the ingest pipeline entirely.
        self.world.step(dt=self.frame_dt, contacts=None, shape_body=None)

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=NUM_BODIES,
            inputs=[self.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_shapes(
            "/world/cubes",
            newton.GeoType.BOX,
            (HALF_EXTENT, HALF_EXTENT, HALF_EXTENT),
            self._xforms[1:],
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        """After settling the chain must still be hanging from the
        world anchor -- no body may have drifted absurdly far
        (catches solver blow-ups) and no NaNs may have appeared.
        """
        positions = self.bodies.position.numpy()
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), f"body {i} produced non-finite position"
            assert positions[i, 1] > -10.0 * NUM_CUBES, f"body {i} fell unreasonably far ({positions[i, 1]})"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
