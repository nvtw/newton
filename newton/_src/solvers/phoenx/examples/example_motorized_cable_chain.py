# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Motorized Cable Chain
#
# A 100-link maximal-coordinate cable mechanism. The links are ordinary
# Newton bodies and cable joints; SolverPhoenX discovers their connected
# graph, builds one RCM-ordered direct system, and leaves no bilateral rows
# in PGS. The first link is fixed to the world and the remaining cable joints
# use stiff stretch, shear, and bend energies while leaving twist free.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_motorized_cable_chain
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

DENSITY = 1000.0
CAPSULE_LENGTH = 0.10
CAPSULE_DIAMETER = 0.05
CAPSULE_RADIUS = 0.5 * CAPSULE_DIAMETER
CAPSULE_HALF_HEIGHT = 0.5 * CAPSULE_LENGTH
NUM_LINKS = 100

STRETCH_STIFFNESS = 1.0e9
STRETCH_DAMPING = 0.0
BEND_STIFFNESS = 1.0e9
BEND_DAMPING = 0.0
TWIST_STIFFNESS = 0.0
TWIST_DAMPING = 0.0

# Body-local +Z points along world -Y, the cable material tangent.
_LINK_ORIENTATION = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)


def _build_model(num_links: int = NUM_LINKS) -> newton.Model:
    """Build one full-coordinate connected cable mechanism."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY)

    bodies: list[int] = []
    for index in range(num_links):
        body = builder.add_link(
            xform=wp.transform(
                wp.vec3(0.0, -(index + 0.5) * CAPSULE_LENGTH, 0.0),
                _LINK_ORIENTATION,
            ),
            label=f"link_{index}",
        )
        builder.add_shape_capsule(
            body,
            radius=CAPSULE_RADIUS,
            half_height=CAPSULE_HALF_HEIGHT,
            cfg=shape_cfg,
        )
        bodies.append(body)

    joints: list[int] = []
    root = builder.add_joint_fixed(
        parent=-1,
        child=bodies[0],
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), _LINK_ORIENTATION),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -CAPSULE_HALF_HEIGHT), wp.quat_identity()),
        label="cable_root",
    )
    joints.append(root)

    for index in range(1, num_links):
        joint = builder.add_joint_cable(
            parent=bodies[index - 1],
            child=bodies[index],
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, CAPSULE_HALF_HEIGHT), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -CAPSULE_HALF_HEIGHT), wp.quat_identity()),
            stretch_stiffness=STRETCH_STIFFNESS,
            stretch_damping=STRETCH_DAMPING,
            shear_stiffness=STRETCH_STIFFNESS,
            shear_damping=STRETCH_DAMPING,
            bend_stiffness=BEND_STIFFNESS,
            bend_damping=BEND_DAMPING,
            twist_stiffness=TWIST_STIFFNESS,
            twist_damping=TWIST_DAMPING,
            collision_filter_parent=True,
            label=f"cable_{index}",
        )
        joints.append(joint)

    # This annotation records topology. Explicit maximal mode keeps the
    # mechanism in full coordinates and lets PhoenX discover it automatically.
    builder.add_articulation(joints)
    return builder.finalize()


class Example:
    """Simulate a long full-coordinate cable mechanism with direct rows."""

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        self.model = _build_model()
        self.state = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            articulation_mode="maximal",
        )

        direct = self.solver._direct_equality_system
        if direct is None or not direct.enabled:
            raise RuntimeError("cable mechanism was not assigned to the direct equality solver")
        expected_dimension = 6 * NUM_LINKS
        if direct.topology.dimensions != (expected_dimension,):
            raise RuntimeError(f"expected one {expected_dimension}-row mechanism, got {direct.topology.dimensions}")
        if not self.solver.world._joint_pgs_all_disabled:
            raise RuntimeError("bilateral cable rows unexpectedly remain in PGS")

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

    def test_final(self) -> None:
        """Verify the direct cable remains finite with bounded bending."""
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()
        assert np.isfinite(body_q).all(), "cable chain produced non-finite poses"
        assert np.isfinite(body_qd).all(), "cable chain produced non-finite velocities"

        z = body_q[:NUM_LINKS, 2]
        tip_sag = float(-z[-1])
        max_sag = float(-np.min(z))
        chain_length = NUM_LINKS * CAPSULE_LENGTH
        print(
            f"[direct_cable_chain] tip_sag={tip_sag * 1.0e3:.3f} mm "
            f"max_sag={max_sag * 1.0e3:.3f} mm "
            f"({tip_sag / chain_length * 100.0:.5f}% of {chain_length:.1f} m chain)"
        )
        maximum_sag = 0.015 * chain_length
        assert abs(tip_sag) < maximum_sag, (
            f"direct cable-chain tip error is {tip_sag * 1.0e3:.3f} mm "
            f"({tip_sag / chain_length * 100.0:.3f}% of chain length)"
        )
        assert max_sag < maximum_sag, "direct cable chain bent beyond 1.5% of its length"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
