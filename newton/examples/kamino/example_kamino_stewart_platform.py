# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Animate a tower of twenty 10 cm Stewart platforms using Kamino DVI.

Each plate is the moving platform of one stage and the base of the next.
Universal joints attach both ends of each hydraulic actuator. A passive axial
swivel in each cylinder prevents the universal joints from binding in torsion.
Only piston extensions are driven; all plate poses emerge from the dynamics.
The choreography blends waves, corkscrews, bows, and traveling curls over 20 s.

Command: python -m newton.examples kamino_stewart_platform
"""

import colorsys
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverKamino


@wp.func
def stage_pose(stage: int, stage_count: int, time: float) -> wp.transform:
    # Blend four phrases continuously: serpentine waves, corkscrews, a broad
    # bow, and a traveling curl. Every phrase starts at mid-stroke when t=0.
    fade = 1.0 - wp.exp(-0.35 * time * time)
    height = float(stage) / float(wp.max(stage_count - 1, 1))
    cycle = 0.31415927 * time
    weights = wp.vec4(
        wp.pow(wp.max(wp.cos(cycle), 0.0), 4.0),
        wp.pow(wp.max(wp.sin(cycle), 0.0), 4.0),
        wp.pow(wp.max(-wp.cos(cycle), 0.0), 4.0),
        wp.pow(wp.max(-wp.sin(cycle), 0.0), 4.0),
    )
    weights /= weights[0] + weights[1] + weights[2] + weights[3]
    phase = 0.85 * time - 0.35 * float(stage)
    spiral = 1.3 * time - 0.65 * float(stage)
    bow = wp.sin(wp.pi * height)
    pulse_center = 0.5 + 0.45 * wp.sin(0.65 * time)
    pulse = wp.exp(-wp.pow((height - pulse_center) / 0.18, 2.0))
    roll = wp.dot(
        weights,
        wp.vec4(
            0.12 * wp.sin(phase),
            0.11 * wp.sin(spiral),
            0.065 * bow * wp.sin(0.7 * time),
            0.16 * pulse * wp.sin(0.8 * time),
        ),
    )
    pitch = wp.dot(
        weights,
        wp.vec4(
            0.12 * wp.cos(phase * 0.83),
            0.11 * wp.cos(spiral),
            0.065 * bow * wp.cos(0.7 * time),
            0.16 * pulse * wp.cos(0.8 * time),
        ),
    )
    twist = wp.dot(
        weights,
        wp.vec4(
            0.045 * wp.sin(0.6 * time + 0.31 * float(stage)),
            0.13 * wp.sin(0.9 * time - 0.3 * float(stage)),
            0.10 * bow * wp.sin(0.8 * time),
            0.12 * pulse * wp.sin(1.5 * time),
        ),
    )
    rotation = wp.quat_rpy(
        fade * roll,
        fade * pitch,
        fade * twist,
    )
    position = wp.vec3(
        0.002 * fade * wp.sin(phase * 0.7),
        0.002 * fade * wp.cos(phase),
        0.1
        + fade
        * (
            0.007 * (1.0 - weights[2]) * wp.sin(1.1 * time - 0.65 * float(stage))
            + 0.011 * weights[2] * wp.sin(1.2 * time)
        ),
    )
    return wp.transform(position, rotation)


@wp.kernel
def update_actuators(
    time: wp.array[float],
    dt: float,
    stage_count: int,
    anchors_base: wp.array[wp.vec3],
    anchors_top: wp.array[wp.vec3],
    lengths_rest: wp.array[float],
    actuator_dofs: wp.array[int],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    actuator = wp.tid()
    stage = actuator / 6
    pose = stage_pose(stage, stage_count, time[0])
    pose_next = stage_pose(stage, stage_count, time[0] + dt)
    length = wp.length(wp.transform_point(pose, anchors_top[actuator]) - anchors_base[actuator])
    length_next = wp.length(wp.transform_point(pose_next, anchors_top[actuator]) - anchors_base[actuator])
    dof = actuator_dofs[actuator]
    target_q[dof] = length - lengths_rest[actuator]
    target_qd[dof] = (length_next - length) / dt


@wp.kernel
def advance_time(time: wp.array[float], dt: float):
    time[0] += dt


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.viewer = viewer
        self.stage_count = args.stage_count if args else 20
        if self.stage_count < 1:
            raise ValueError("--stage-count must be positive")
        self.frame_dt = 1.0 / 60.0
        # Implicit actuation remains stable at four substeps; twelve spends most
        # of the frame repeating the tower's sparse constraint factorization.
        self.sim_substeps = getattr(args, "substeps", 4) if args else 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.time = wp.zeros(1, dtype=float)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        # The mechanism is attached to ground. Contact between the small, closely
        # spaced fittings is excluded, as in the Kamino test-mechanism example.
        builder.default_shape_cfg.has_shape_collision = False
        builder.default_shape_cfg.density = 1600.0
        builder.add_ground_plane(color=(0.18, 0.21, 0.25))
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((0.0, 0.0, 0.005), wp.quat_identity()),
            radius=0.075,
            half_height=0.005,
            color=(0.22, 0.26, 0.30),
        )

        def passive(axis):
            return newton.ModelBuilder.JointDofConfig(axis=axis, actuator_mode=newton.JointTargetMode.NONE)

        joints_tree = []
        closures = []
        self.plates = []
        anchors_base, anchors_top, lengths_rest, actuator_dofs = [], [], [], []

        def add_plate(level):
            body = builder.add_link(
                xform=wp.transform((0.0, 0.0, 0.012 + 0.1 * level), wp.quat_identity()),
                label=f"plate_{level}",
            )
            color = colorsys.hsv_to_rgb(0.52 + 0.30 * level / self.stage_count, 0.65, 0.85)
            builder.add_shape_cylinder(body, radius=0.05, half_height=0.002, color=color)
            builder.add_shape_cylinder(body, radius=0.047, half_height=0.0024, color=(0.12, 0.16, 0.20))
            self.plates.append(body)
            return body

        base = add_plate(0)
        joints_tree.append(builder.add_joint_fixed(-1, base, parent_xform=builder.body_q[base], label="ground_anchor"))
        for stage in range(self.stage_count):
            top = add_plate(stage + 1)
            for leg in range(6):
                sector = 2.0 * math.pi * (leg // 2) / 3.0
                sign = -1.0 if leg % 2 == 0 else 1.0
                angle_base = sector + sign * math.radians(12.0)
                angle_top = sector + sign * math.radians(48.0)
                a = wp.vec3(0.0425 * math.cos(angle_base), 0.0425 * math.sin(angle_base), 0.004)
                b = wp.vec3(0.0425 * math.cos(angle_top), 0.0425 * math.sin(angle_top), -0.004)
                delta = b + wp.vec3(0.0, 0.0, 0.1) - a
                length = wp.length(delta)
                rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), delta / length)
                origin_base = wp.transform_point(builder.body_q[base], a)
                origin_top = wp.transform_point(builder.body_q[top], b)
                barrel = builder.add_link(xform=wp.transform(origin_base, rotation), label=f"barrel_{stage}_{leg}")
                rod = builder.add_link(xform=wp.transform(origin_top, rotation), label=f"rod_{stage}_{leg}")
                builder.add_shape_cylinder(
                    barrel,
                    xform=wp.transform((0.0, 0.0, 0.039), wp.quat_identity()),
                    radius=0.0032,
                    half_height=0.035,
                    color=(0.16, 0.23, 0.29),
                )
                for z in (0.008, 0.070):
                    builder.add_shape_cylinder(
                        barrel,
                        xform=wp.transform((0.0, 0.0, z), wp.quat_identity()),
                        radius=0.0038,
                        half_height=0.002,
                        color=(0.9, 0.52, 0.16),
                    )
                builder.add_shape_cylinder(
                    rod,
                    xform=wp.transform((0.0, 0.0, -0.04), wp.quat_identity()),
                    radius=0.0017,
                    half_height=0.036,
                    color=(0.75, 0.82, 0.88),
                )
                # Cross pins make the two orthogonal universal-joint axes visible.
                for body in (barrel, rod):
                    for axis in (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)):
                        builder.add_shape_cylinder(
                            body,
                            xform=wp.transform(wp.vec3(), wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis)),
                            radius=0.0018,
                            half_height=0.0042,
                            color=(0.9, 0.65, 0.24),
                        )
                lower = builder.add_joint_d6(
                    base,
                    barrel,
                    parent_xform=wp.transform(a, rotation),
                    angular_axes=[passive(newton.Axis.X), passive(newton.Axis.Y)],
                    label=f"universal_lower_{stage}_{leg}",
                )
                # A cylindrical joint combines driven sliding with passive rod spin.
                # Six U-P-U legs without that spin would overconstrain a Stewart platform.
                actuator = builder.add_joint_d6(
                    barrel,
                    rod,
                    parent_xform=wp.transform((0.0, 0.0, length), wp.quat_identity()),
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(
                            axis=newton.Axis.Z,
                            limit_lower=-0.025,
                            limit_upper=0.025,
                            target_ke=300000.0,
                            target_kd=150.0,
                            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                        )
                    ],
                    angular_axes=[passive(newton.Axis.Z)],
                    label=f"piston_{stage}_{leg}",
                )
                joints_tree.extend((lower, actuator))
                upper_args = {
                    "parent": rod,
                    "child": top,
                    "child_xform": wp.transform(b, rotation),
                    "angular_axes": [passive(newton.Axis.X), passive(newton.Axis.Y)],
                    "label": f"universal_upper_{stage}_{leg}",
                }
                if leg == 0:
                    joints_tree.append(builder.add_joint_d6(**upper_args))
                else:
                    closures.append(upper_args)
                anchors_base.append(a)
                anchors_top.append(b)
                lengths_rest.append(length)
                actuator_dofs.append(builder.joint_qd_start[actuator])
            base = top

        # The remaining upper universal joints close the six parallel leg loops.
        builder.add_articulation(joints_tree, label="stewart_tower")
        for upper_args in closures:
            builder.add_joint_d6(**upper_args)
        self.model = builder.finalize(skip_validation_joints=True)
        self.anchors_base = wp.array(anchors_base, dtype=wp.vec3)
        self.anchors_top = wp.array(anchors_top, dtype=wp.vec3)
        self.lengths_rest = wp.array(lengths_rest, dtype=float)
        self.actuator_dofs = wp.array(actuator_dofs, dtype=int)
        self.actuator_dofs_host = np.array(actuator_dofs)

        config = SolverKamino.Config.from_model(
            self.model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True
        )
        config.use_fk_solver = False
        config.use_collision_detector = False
        config.constraints.alpha = 0.1
        config.constraints.beta = 0.011
        config.constraints.gamma = 0.015
        config.dynamics.preconditioning = False
        config.dynamics.linear_solver_type = "CR"
        config.dynamics.linear_solver_kwargs = {"maxiter": 9}
        config.dvi.bilateral_solver_type = "LLTBRCM"
        config.dvi.bilateral_solver_kwargs = {"parallel_factorization": True}
        config.dvi.tolerance = 1e-5
        config.dvi.regularization = 1e-8
        config.dvi.max_alternating_iterations = 4
        config.dvi.inequality_sweeps_per_iteration = 3
        config.dvi.bilateral_solve_interval = 1
        self.solver = SolverKamino(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.5, -2.6, 0.15 + 0.05 * self.stage_count), pitch=-3.0, yaw=120.0)
        self.graph = None
        if self.model.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                update_actuators,
                dim=6 * self.stage_count,
                inputs=[
                    self.time,
                    self.sim_dt,
                    self.stage_count,
                    self.anchors_base,
                    self.anchors_top,
                    self.lengths_rest,
                    self.actuator_dofs,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.time, self.sim_dt])

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Check ground anchoring, loop closure, stroke limits, and actuator tracking."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite body poses"
        np.testing.assert_allclose(poses[self.plates[0], :3], (0.0, 0.0, 0.012), atol=2e-4)
        np.testing.assert_allclose(poses[self.plates[0], 3:], (0.0, 0.0, 0.0, 1.0), atol=2e-4)
        parent = self.model.joint_parent.numpy()
        child = self.model.joint_child.numpy()
        xp = self.model.joint_X_p.numpy()
        xc = self.model.joint_X_c.numpy()
        for joint, label in enumerate(self.model.joint_label):
            if not label.startswith("universal_"):
                continue
            a = wp.transform_point(wp.transform(*poses[parent[joint]]), wp.vec3(*xp[joint, :3]))
            b = wp.transform_point(wp.transform(*poses[child[joint]]), wp.vec3(*xc[joint, :3]))
            assert wp.length(a - b) < 0.001, f"Universal joint opened: {label}"
        q = self.state_0.joint_q.numpy()[self.actuator_dofs_host]
        target = self.control.joint_target_q.numpy()[self.actuator_dofs_host]
        assert np.max(np.abs(q)) < 0.026, "Piston exceeded its stroke"
        np.testing.assert_allclose(q, target, atol=0.002)
        # Check each stage: the traveling curl deliberately lowers the tip, and
        # small actuator compliance accumulates along the twenty-stage tower.
        for stage in range(self.stage_count):
            relative = wp.transform_multiply(
                wp.transform_inverse(wp.transform(*poses[self.plates[stage]])),
                wp.transform(*poses[self.plates[stage + 1]]),
            )
            desired = stage_pose(stage, self.stage_count, self.sim_time)
            np.testing.assert_allclose(np.array(relative)[:3], np.array(desired)[:3], atol=0.003)
            rotation_error = wp.quat_inverse(wp.transform_get_rotation(desired)) * wp.transform_get_rotation(relative)
            assert wp.length(wp.vec3(rotation_error[0], rotation_error[1], rotation_error[2])) < 0.035, (
                f"Platform {stage} did not follow the commanded tilt"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--stage-count", type=int, default=20, help="Number of stacked Stewart platforms.")
        parser.add_argument("--substeps", type=int, choices=(2, 4, 6, 12), default=4, help="Physics steps per frame.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
