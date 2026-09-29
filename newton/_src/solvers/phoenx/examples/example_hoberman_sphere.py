# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Hoberman Sphere
#
# A 240-strut, full-coordinate looped mechanism built from neutral Python
# body and joint data. SolverPhoenX discovers its connected revolute
# graph, precomputes one RCM-ordered sparse direct system, and leaves no
# bilateral rows in PGS. The authored overlapping visual struts remain
# non-colliding, and zero gravity lets the sphere coast with a rigid-body spin.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_hoberman_sphere
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.examples import hoberman_sphere_data

SPIN_RATE_RAD_S = 0.1


def _transform(values):
    return wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:]))


def make_hoberman_builder(*, collidable: bool = False, gravity=(0.0, 0.0, 0.0)) -> newton.ModelBuilder:
    """Build one jointed sphere from neutral body, joint, and tile data."""
    builder = newton.ModelBuilder(gravity=gravity)
    for label, pose, com, inertia, mass in hoberman_sphere_data.BODIES:
        builder.add_link(
            xform=_transform(pose),
            com=com,
            inertia=inertia,
            mass=mass,
            label=label,
            lock_inertia=True,
        )

    for (
        joint_type_name,
        label,
        parent,
        child,
        parent_frame,
        child_frame,
        axes,
        coordinates,
    ) in hoberman_sphere_data.JOINTS:
        joint_type = newton.JointType[joint_type_name]
        if joint_type == newton.JointType.FREE:
            linear_axes, angular_axes = axes[:3], axes[3:]
        else:
            linear_axes, angular_axes = (), axes
        joint_index = builder.add_joint(
            joint_type,
            parent,
            child,
            linear_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in linear_axes],
            angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in angular_axes],
            label=label,
            parent_xform=_transform(parent_frame),
            child_xform=_transform(child_frame),
        )
        q_start = builder.joint_q_start[joint_index]
        builder.joint_q[q_start : q_start + len(coordinates)] = coordinates
    # The remaining 181 joints close loops and are deliberately outside the tree.
    builder.add_articulation(list(range(hoberman_sphere_data.ARTICULATION_JOINT_COUNT)))

    tile_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=collidable,
        has_particle_collision=False,
        collision_group=-2 if collidable else 0,
        gap=0.01,
    )
    for body, pose, half_extents, color in hoberman_sphere_data.TILES:
        builder.add_shape_box(
            body=body,
            xform=_transform(pose),
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            cfg=tile_cfg,
            color=color,
        )
    return builder


class Example:
    """Simulate a looped Hoberman mechanism with direct equalities."""

    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        builder = make_hoberman_builder()
        builder.color()
        self.model = builder.finalize(skip_validation_joints=True)

        self.state = self.model.state()
        self.state.body_q.assign(self.model.body_q)
        body_q = self.model.body_q.numpy()
        angular_velocity = np.array([0.0, 0.0, SPIN_RATE_RAD_S], dtype=np.float32)
        body_qd = np.zeros((self.model.body_count, 6), dtype=np.float32)
        body_qd[:, :3] = np.cross(angular_velocity, body_q[:, :3])
        body_qd[:, 3:] = angular_velocity
        self.state.body_qd.assign(body_qd)
        self.control = self.model.control()

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            joint_mode="maximal_direct",
        )
        direct = self.solver._direct_equality_system
        if direct is None or not direct.enabled:
            raise RuntimeError("Hoberman joints were not assigned to the direct equality solver")
        if len(direct.topology.dimensions) != 1:
            raise RuntimeError(f"expected one connected Hoberman mechanism, got {direct.topology.dimensions}")
        if direct.solver.block_size != 32:
            raise RuntimeError(
                f"expected 32-row panels for the loop-dense Hoberman mechanism, got {direct.solver.block_size}"
            )
        if not self.solver.world._joint_pgs_all_disabled:
            raise RuntimeError("bilateral Hoberman rows unexpectedly remain in PGS")
        print(
            f"[PhoenX Hoberman] bodies={self.model.body_count} "
            f"joints={self.model.joint_count} direct_rows={direct.topology.dimensions[0]}"
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, -6.0, 2.0), pitch=-15.0, yaw=135.0)

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
        """Render the current Newton body state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify the looped direct mechanism remains finite and bounded."""
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()
        assert np.isfinite(body_q).all(), "Hoberman sphere produced non-finite poses"
        assert np.isfinite(body_qd).all(), "Hoberman sphere produced non-finite velocities"

        maximum_radius = float(np.linalg.norm(body_q[:, :3], axis=1).max())
        print(f"[direct_hoberman] maximum_radius={maximum_radius:.4f} m")
        assert maximum_radius < 8.0, f"Hoberman sphere escaped its stability envelope: {maximum_radius:.3f} m"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
