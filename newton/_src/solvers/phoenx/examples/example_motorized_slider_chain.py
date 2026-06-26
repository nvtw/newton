# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Motorized Slider Chain
#
# Prismatic-joint sibling of :mod:`example_motorized_hinge_chain`. Same
# cantilever layout -- ``NUM_CUBES`` unit cubes in a column along world
# -y, gravity along -z -- but every joint is a
# :data:`~newton._src.solvers.phoenx.world_builder.JointMode.PRISMATIC`
# slider whose slide axis points along world +x.
#
# A prismatic joint locks all 3 relative rotations plus the 2 transverse
# translations, leaving only the slide along its axis free. The slide
# axis (+x) is perpendicular to both the chain (-y) and gravity (-z), so
# gravity never drives the free DoF and the chain stays positionally
# pinned along its length -- the cantilever is held up entirely by the
# locked transverse + rotational rows, exactly as the hinge chain hangs
# from its swing locks. The free-end -z droop is therefore a direct
# convergence probe for the slider's positional lock.
#
# This exists to evaluate prismatic convergence under the unified D6
# formulation, exactly as the hinge chain evaluates revolute.
#
# Run:  python -m newton._src.solvers.phoenx.examples.example_motorized_slider_chain
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    body_container_zeros,
    inertia_sym6_pack_np,
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

# Drive on the free slide axis. OFF leaves the chain to hold its length
# by warm-start + inertia (the transverse/rotational locks still hold the
# cantilever); POSITION pulls each slider toward ``TARGET`` [m].
DRIVE_MODE = DriveMode.OFF

NUM_CUBES = 250
HALF_EXTENT = 0.05
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor at slot 0
NUM_JOINTS = NUM_CUBES

# Centre-to-centre spacing along world -y. The slider anchors are offset
# along the same axis so the implicit slide direction ``anchor2 - anchor1``
# runs along the chain.
PITCH = 2.0 * HALF_EXTENT
_ANCHOR_OFFSET = HALF_EXTENT

_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Position-drive soft-spring knobs (used only when DRIVE_MODE==POSITION):
# 4 Hz critically-damped linear spring on unit-mass cubes.
_HERTZ_DRIVE = 4.0
_STIFFNESS_DRIVE = (2.0 * math.pi * _HERTZ_DRIVE) ** 2
_DAMPING_DRIVE = 2.0 * (2.0 * math.pi * _HERTZ_DRIVE)
_MOTOR_MAX_FORCE = 50.0
TARGET = 0.0
TARGET_VELOCITY = 0.0


def _populate_chain_bodies(bodies) -> None:
    positions = np.zeros((NUM_BODIES, 3), dtype=np.float32)
    orientations = np.zeros((NUM_BODIES, 4), dtype=np.float32)
    orientations[:] = (0.0, 0.0, 0.0, 1.0)  # identity for every body
    for j in range(NUM_CUBES):
        positions[j + 1] = (0.0, -(j + 0.5) * PITCH, 0.0)
    bodies.position.assign(positions)
    bodies.orientation.assign(orientations)

    inv_mass_np = np.zeros(NUM_BODIES, dtype=np.float32)
    inv_mass_np[1:] = 1.0
    bodies.inverse_mass.assign(inv_mass_np)

    inv_inertia_np = np.zeros((NUM_BODIES, 3, 3), dtype=np.float32)
    eye = np.array(_INV_INERTIA, dtype=np.float32)
    for j in range(1, NUM_BODIES):
        inv_inertia_np[j] = eye
    bodies.inverse_inertia.assign(inv_inertia_np)
    bodies.inverse_inertia_world.assign(inertia_sym6_pack_np(inv_inertia_np))

    motion = np.full(NUM_BODIES, int(MOTION_STATIC), dtype=np.int32)
    motion[1:] = int(MOTION_DYNAMIC)
    bodies.motion_type.assign(motion)


def _build_joint_arrays(device: wp.context.Device) -> dict[str, wp.array]:
    body1 = np.zeros(NUM_JOINTS, dtype=np.int32)
    body2 = np.zeros(NUM_JOINTS, dtype=np.int32)
    anchor1 = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    anchor2 = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for k in range(NUM_JOINTS):
        body1[k] = 0 if k == 0 else k  # link_{k-1} -> slot k (slot 0 = world)
        body2[k] = k + 1
        # Boundary between link k-1 and link k sits at y = -k * PITCH.
        # Offset the two anchors along world +x so the slide axis is +x
        # (perpendicular to the chain and to gravity).
        y = -k * PITCH
        anchor1[k] = (-_ANCHOR_OFFSET, y, 0.0)
        anchor2[k] = (_ANCHOR_OFFSET, y, 0.0)

    target = np.full(NUM_JOINTS, float(TARGET), dtype=np.float32)
    target_velocity = np.full(NUM_JOINTS, float(TARGET_VELOCITY), dtype=np.float32)
    max_force_drive = np.full(NUM_JOINTS, float(_MOTOR_MAX_FORCE), dtype=np.float32)
    stiffness_drive = np.full(NUM_JOINTS, float(_STIFFNESS_DRIVE), dtype=np.float32)
    damping_drive = np.full(NUM_JOINTS, float(_DAMPING_DRIVE), dtype=np.float32)

    # No slide limit (min > max disables the limit row).
    min_value = np.full(NUM_JOINTS, 1.0, dtype=np.float32)
    max_value = np.full(NUM_JOINTS, -1.0, dtype=np.float32)

    joint_mode = np.full(NUM_JOINTS, int(JointMode.PRISMATIC), dtype=np.int32)
    drive_mode = np.full(NUM_JOINTS, int(DRIVE_MODE), dtype=np.int32)

    # Match the hinge chain: maximally-stiff positional lock so the
    # transverse/rotational rows behave as a rigid joint in fp32.
    hertz = np.full(NUM_JOINTS, float(DEFAULT_HERTZ_LINEAR), dtype=np.float32)
    damping_ratio = np.full(NUM_JOINTS, float(DEFAULT_DAMPING_RATIO), dtype=np.float32)
    hertz_limit = np.full(NUM_JOINTS, float(DEFAULT_HERTZ_LINEAR), dtype=np.float32)
    damping_ratio_limit = np.full(NUM_JOINTS, float(DEFAULT_DAMPING_RATIO), dtype=np.float32)
    stiffness_limit = np.zeros(NUM_JOINTS, dtype=np.float32)
    damping_limit = np.zeros(NUM_JOINTS, dtype=np.float32)

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

        self.bodies = body_container_zeros(NUM_BODIES, device=self.device)
        _populate_chain_bodies(self.bodies)

        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=NUM_JOINTS,
            device=self.device,
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=1,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=0,
            num_joints=NUM_JOINTS,
            device=self.device,
        )

        joint_arrays = _build_joint_arrays(self.device)
        self.world.initialize_actuated_double_ball_socket_joints(**joint_arrays)

        self._xforms = wp.zeros(NUM_BODIES, dtype=wp.transform, device=self.device)

        half_extents_np = np.zeros((NUM_BODIES, 3), dtype=np.float32)
        half_extents_np[1:] = HALF_EXTENT
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        self.capture()

    def capture(self) -> None:
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        self.picking.apply_force()
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

    def _tip_sag(self) -> tuple[float, float]:
        """Return ``(tip_sag, max_sag)`` of the slider cantilever in metres.

        The slide axis runs along the chain (-y) and gravity is along -z,
        so gravity never drives the free slide DoF. Any -z droop is
        residual error in the slider's locked transverse + rotational
        rows, the prismatic analog of the hinge chain's swing-lock droop.
        """
        positions = self.bodies.position.numpy()
        z = positions[1:NUM_BODIES, 2]
        return float(-z[-1]), float(-np.min(z))

    def test_final(self) -> None:
        positions = self.bodies.position.numpy()
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), f"body {i} produced non-finite position"
            assert positions[i, 1] > -10.0 * NUM_CUBES, f"body {i} drifted unreasonably far ({positions[i, 1]})"

        tip_sag, max_sag = self._tip_sag()
        chain_len = NUM_CUBES * PITCH
        print(
            f"[slider_chain] tip_sag={tip_sag * 1e3:.1f} mm  max_sag={max_sag * 1e3:.1f} mm "
            f"({tip_sag / chain_len * 100.0:.2f}% of {chain_len:.1f} m chain)"
        )
        # A converged slider cantilever barely droops; decoupled per-anchor
        # blocks let it collapse. Guard inside the converged margin.
        assert tip_sag < 0.4, f"free end drooped {tip_sag * 1e3:.1f} mm -- slider lock under-converged"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
