# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Motorized Hinge Chain
#
# A 100-link, maximal-coordinate hinge mechanism used to stress PhoenX's
# mechanism-wide direct equality solver. The links are ordinary Newton
# bodies and joints. ModelBuilder's required articulation annotation records
# topology only; explicit maximal mode keeps it out of the reduced backend.
# SolverPhoenX discovers the connected body-joint graph, builds one RCM-ordered
# block-Cholesky system, and leaves no bilateral joint rows in PGS.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_motorized_hinge_chain
###########################################################################

from __future__ import annotations

import enum
import math

import numpy as np
import warp as wp

import newton
import newton.examples


class BodyShape(enum.Enum):
    """Select the visible geometry used for every chain link."""

    CUBE = "cube"
    CAPSULE = "capsule"


BODY_SHAPE = BodyShape.CAPSULE
DENSITY = 1000.0
CAPSULE_LENGTH = 0.10
CAPSULE_DIAMETER = 0.05
_CAPSULE_RADIUS = 0.5 * CAPSULE_DIAMETER
_CAPSULE_HALF_HEIGHT = 0.5 * CAPSULE_LENGTH

NUM_LINKS = 100
HALF_EXTENT = 0.05

_DIAGONAL_HALF = HALF_EXTENT * math.sqrt(2.0)
_HALF_ANGLE = math.pi / 8.0
_DIAGONAL_QUAT = (0.0, 0.0, math.sin(_HALF_ANGLE), math.cos(_HALF_ANGLE))
_CAPSULE_QUAT = (math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0))

_MOTOR_MAX_FORCE = 50.0
_HERTZ_DRIVE = 4.0
_STIFFNESS_DRIVE = (2.0 * math.pi * _HERTZ_DRIVE) ** 2
_DAMPING_DRIVE = 2.0 * (2.0 * math.pi * _HERTZ_DRIVE)
TARGET_VELOCITY = 0.0


def _link_layout() -> tuple[float, tuple[float, float, float, float]]:
    """Return the link pitch and initial body orientation."""
    if BODY_SHAPE is BodyShape.CUBE:
        return 2.0 * _DIAGONAL_HALF, _DIAGONAL_QUAT
    if BODY_SHAPE is BodyShape.CAPSULE:
        return CAPSULE_LENGTH, _CAPSULE_QUAT
    raise ValueError(f"unsupported body shape: {BODY_SHAPE!r}")


def _quat_rotate(q: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    """Rotate a vector by an xyzw quaternion."""
    qv = np.asarray(q[:3], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + float(q[3]) * t + np.cross(qv, t)


def _local_joint_frame(
    body_position: np.ndarray,
    body_orientation: tuple[float, float, float, float],
    anchor_position: np.ndarray,
) -> wp.transform:
    """Express a world-aligned hinge frame in a body's local frame."""
    inverse_orientation = (
        -body_orientation[0],
        -body_orientation[1],
        -body_orientation[2],
        body_orientation[3],
    )
    local_position = _quat_rotate(inverse_orientation, anchor_position - body_position)
    return wp.transform(wp.vec3(*local_position), wp.quat(*inverse_orientation))


def _build_model(num_links: int = NUM_LINKS) -> newton.Model:
    """Build one full-coordinate connected hinge mechanism."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY)
    pitch, link_orientation = _link_layout()

    bodies: list[int] = []
    body_positions: list[np.ndarray] = []
    for index in range(num_links):
        position = np.asarray((0.0, -(index + 0.5) * pitch, 0.0), dtype=np.float64)
        body = builder.add_link(
            xform=wp.transform(wp.vec3(*position), wp.quat(*link_orientation)),
            label=f"link_{index}",
        )
        if BODY_SHAPE is BodyShape.CAPSULE:
            builder.add_shape_capsule(
                body,
                radius=_CAPSULE_RADIUS,
                half_height=_CAPSULE_HALF_HEIGHT,
                cfg=shape_cfg,
            )
        else:
            builder.add_shape_box(body, hx=HALF_EXTENT, hy=HALF_EXTENT, hz=HALF_EXTENT, cfg=shape_cfg)
        bodies.append(body)
        body_positions.append(position)

    joints: list[int] = []
    for index, child in enumerate(bodies):
        anchor_position = np.asarray((0.0, -index * pitch, 0.0), dtype=np.float64)
        if index == 0:
            parent = -1
            parent_xform = wp.transform(wp.vec3(*anchor_position), wp.quat_identity())
        else:
            parent = bodies[index - 1]
            parent_xform = _local_joint_frame(body_positions[index - 1], link_orientation, anchor_position)
        child_xform = _local_joint_frame(body_positions[index], link_orientation, anchor_position)
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.Z,
            target_vel=TARGET_VELOCITY,
            target_ke=_STIFFNESS_DRIVE,
            target_kd=_DAMPING_DRIVE,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            effort_limit=_MOTOR_MAX_FORCE,
            limit_lower=-math.inf,
            limit_upper=math.inf,
            collision_filter_parent=True,
            label=f"hinge_{index}",
        )
        joints.append(joint)

    # ModelBuilder requires this topology annotation. SolverPhoenX's explicit
    # maximal mode still discovers and solves it as a full-coordinate mechanism.
    builder.add_articulation(joints)
    return builder.finalize()


class Example:
    """Simulate a long full-coordinate hinge mechanism with direct equalities."""

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        self.model = _build_model()
        self.state = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            articulation_mode="maximal",
        )
        direct = self.solver._direct_equality_system
        if direct is None or not direct.enabled:
            raise RuntimeError("hinge mechanism was not assigned to the direct equality solver")
        expected_dimension = 6 * NUM_LINKS
        if direct.topology.dimensions != (expected_dimension,):
            raise RuntimeError(f"expected one {expected_dimension}-row mechanism, got {direct.topology.dimensions}")
        if not self.solver.world._joint_pgs_all_disabled:
            raise RuntimeError("bilateral hinge or drive rows unexpectedly remain in PGS")

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.0, -10.0, 5.0), pitch=-18.0, yaw=140.0)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        """Advance one rendered frame."""
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.solver.step(self.state, self.state, self.control, None, self.frame_dt)

    def step(self) -> None:
        """Advance the captured or eager simulation."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the current body state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def _tip_sag(self) -> tuple[float, float]:
        """Return free-end and maximum vertical constraint error [m]."""
        positions = self.state.body_q.numpy()[:, :3]
        z = positions[:NUM_LINKS, 2]
        return float(-z[-1]), float(-np.min(z))

    def test_final(self) -> None:
        """Verify the direct mechanism remains finite and nearly straight."""
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()
        assert np.isfinite(body_q).all(), "hinge chain produced non-finite poses"
        assert np.isfinite(body_qd).all(), "hinge chain produced non-finite velocities"

        tip_sag, max_sag = self._tip_sag()
        chain_length = NUM_LINKS * _link_layout()[0]
        print(
            f"[direct_hinge_chain] tip_sag={tip_sag * 1.0e3:.3f} mm "
            f"max_sag={max_sag * 1.0e3:.3f} mm "
            f"({tip_sag / chain_length * 100.0:.5f}% of {chain_length:.1f} m chain)"
        )
        maximum_sag = 0.015 * chain_length
        assert abs(tip_sag) < maximum_sag, (
            f"direct hinge-chain tip error is {tip_sag * 1.0e3:.3f} mm "
            f"({tip_sag / chain_length * 100.0:.3f}% of chain length)"
        )
        assert max_sag < maximum_sag, "direct hinge chain bent beyond 1.5% of its length"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
