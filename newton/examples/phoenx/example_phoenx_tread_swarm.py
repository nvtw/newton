# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drive a swarm of thirty small tracked robots in a swirling flower with PhoenX.

Each 14 cm robot rides on two closed tread loops of 22 hinged links. Two smooth
rollers per side carry each loop and their velocity-controlled motors drive
it, and a slippery skid carries each top run; links, rollers, skids and the
floor touch only through contacts. The run uses 7 substeps of 2 iterations with
two collision updates per frame (120 Hz). A boxy hull
carries a sensor head on a short mast and two antenna stubs, painted in a
different cheerful color per robot.

The robots start on three rings inside a 1.5 m circle. A controller that runs
on the GPU steers each robot along its ring: the inner and outer rings circle
the centre counterclockwise, the middle ring clockwise, and every ring swells
and shrinks in a slowly turning three-petal wave. Robots steer away from close
neighbours, slow down behind one ahead and turn back when near the edge. Skid
steering (one tread faster than the other) turns them. Motor targets ramp over
0.4 s, so the robots never jerk.

Success criteria (``--test``): every robot stays upright for the whole run,
every tread hinge stays closed (< 1 mm) and every tread link stays on its
rollers (< 6 mm sideways, < 25 mm from the loop). At the end all robots are
inside the circle (1.5 m plus 0.1 m margin) and, after the 1 s start delay,
each robot has moved at 50 mm/s or more on average over the last 2 s.

Commands:
    python -m newton.examples phoenx_tread_swarm
    python -m newton.examples phoenx_tread_swarm --viewer null --num-frames 600 --test
    python -m newton.examples phoenx_tread_swarm --direct-contacts
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX

# Tread loop geometry [m]: link pitch, hinge circle radius of a 4-link half wrap
# (45 degrees per link), roller spacing (seven pitches), shoe thickness and width.
PITCH = 0.011
HINGE_RADIUS = PITCH / (2.0 * np.sin(np.pi / 8.0))
CHORD_RADIUS = HINGE_RADIUS * np.cos(np.pi / 8.0)
ROLLER_SPACING = 7.0 * PITCH
SHOE_THICKNESS = 0.003
SHOE_WIDTH = 0.020
# Roller rim radius [m]: a little below the chord radius, so the loop has play.
RIM_RADIUS = 0.01318
RIM_HALF_WIDTH = 0.004
# Inner guide teeth on each link straddle the rims [m].
GUIDE_HALF_WIDTH = 0.00075
GUIDE_OFFSET = RIM_HALF_WIDTH + 0.0008 + GUIDE_HALF_WIDTH
GUIDE_HEIGHT = 0.004
# Lateral tread centre [m] and roller axle height above the floor [m] (1 mm drop).
TREAD_Y = 0.046
AXLE_HEIGHT = CHORD_RADIUS + SHOE_THICKNESS + 0.001
LINK_MASS = 0.0025  # [kg]
# Rubber pads on the links, sticky rims standing in for sprocket teeth, floor.
SHOE_FRICTION = 0.8
RIM_FRICTION = 2.0
FLOOR_FRICTION = 0.5
# Tread hinge pin friction [N m] and damping [N m s/rad].
HINGE_FRICTION = 1.0e-4
HINGE_DAMPING = 2.0e-5
# Geared motor at every roller: speed-loop gain [N m s/rad], torque limit [N m],
# reflected rotor inertia [kg m^2] and the time from standstill to full speed [s].
MOTOR_GAIN = 0.01
MOTOR_TORQUE = 0.04
MOTOR_ARMATURE = 1.0e-5
RAMP_TIME = 0.4

# Swarm layout: (ring radius [m], robots, lane wave amplitude [m], direction).
RINGS = ((0.55, 6, 0.08, 1.0), (0.85, 10, 0.11, -1.0), (1.15, 14, 0.14, 1.0))
ARENA_RADIUS = 1.5

# Controller constants: lane wave, steering, separation and speeds.
PETALS = wp.constant(3.0)
WAVE_RATE = wp.constant(0.45)  # [rad/s]
RADIAL_GAIN = wp.constant(4.0)  # [1/m]
EDGE_RADIUS = wp.constant(1.3)  # [m]
EDGE_GAIN = wp.constant(12.0)  # [1/m]
SEPARATION_RADIUS = wp.constant(0.32)  # [m]
SEPARATION_GAIN = wp.constant(2.5)
STEER_GAIN = wp.constant(2.5)  # [1/rad]
SWIRL_RATE = wp.constant(0.22)  # [rad/s], lane speed per lane radius
MIN_SPEED = wp.constant(0.1)  # [m/s]
MAX_SPEED = wp.constant(0.24)  # [m/s]
TURN_SPEED = wp.constant(0.1)  # [m/s], tread speed difference at full steer
START_DELAY = wp.constant(1.0)  # [s]
ROLL_RADIUS = wp.constant(float(RIM_RADIUS))
RAMP_SECONDS = wp.constant(float(RAMP_TIME))
MAX_ROLLER_SPEED = wp.constant(float((0.24 + 0.1) / RIM_RADIUS))  # [rad/s]


@wp.func
def _planar(v: wp.vec3):
    return wp.vec2(v[0], v[1])


@wp.kernel(enable_backward=False)
def _steer_swarm(
    body_q: wp.array[wp.transform],
    chassis: wp.array[wp.int32],
    lane_radius: wp.array[wp.float32],
    lane_amplitude: wp.array[wp.float32],
    direction: wp.array[wp.float32],
    time: wp.array[wp.float32],
    drive_dofs: wp.array2d[wp.int32],
    frame_dt: float,
    applied: wp.array[wp.vec2],
    target_qd: wp.array[wp.float32],
):
    i = wp.tid()
    t = time[0]
    pose = body_q[chassis[i]]
    position = _planar(wp.transform_get_translation(pose))
    forward = wp.normalize(_planar(wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3(1.0, 0.0, 0.0))))
    radius = wp.length(position)
    radial = position / wp.max(radius, 1.0e-3)
    tangent = wp.vec2(-radial[1], radial[0]) * direction[i]
    # Lane radius of this robot: a slowly turning three-petal wave.
    angle = wp.atan2(position[1], position[0])
    goal_radius = lane_radius[i] + lane_amplitude[i] * wp.sin(PETALS * angle - WAVE_RATE * t)
    desired = tangent + wp.clamp(RADIAL_GAIN * (goal_radius - radius), -1.5, 1.5) * radial
    if radius > EDGE_RADIUS:
        desired -= EDGE_GAIN * (radius - EDGE_RADIUS) * radial
    # Separation from close neighbours; slow down behind a robot ahead.
    slow = float(1.0)
    for j in range(chassis.shape[0]):
        if j != i:
            offset = position - _planar(wp.transform_get_translation(body_q[chassis[j]]))
            distance = wp.length(offset)
            if distance < SEPARATION_RADIUS and distance > 1.0e-4:
                away = offset / distance
                desired += SEPARATION_GAIN * (SEPARATION_RADIUS - distance) / SEPARATION_RADIUS * away
                if -wp.dot(away, forward) > 0.5:
                    slow = wp.min(slow, wp.clamp((distance - 0.17) / 0.12, 0.0, 1.0))
    heading = wp.atan2(forward[1], forward[0])
    error = wp.atan2(desired[1], desired[0]) - heading
    error = wp.atan2(wp.sin(error), wp.cos(error))
    steer = wp.clamp(STEER_GAIN * error, -1.0, 1.0)
    start = wp.clamp(t - START_DELAY, 0.0, 1.0)
    speed = wp.clamp(SWIRL_RATE * goal_radius, MIN_SPEED, MAX_SPEED) * (1.0 - 0.6 * wp.abs(steer)) * slow
    goal = start / ROLL_RADIUS * wp.vec2(speed - steer * TURN_SPEED, speed + steer * TURN_SPEED)
    # Motors ramp from standstill to full speed over RAMP_TIME.
    limit = MAX_ROLLER_SPEED * frame_dt / RAMP_SECONDS
    previous = applied[i]
    command = previous + wp.vec2(
        wp.clamp(goal[0] - previous[0], -limit, limit), wp.clamp(goal[1] - previous[1], -limit, limit)
    )
    applied[i] = command
    target_qd[drive_dofs[i, 0]] = command[0]
    target_qd[drive_dofs[i, 1]] = command[0]
    target_qd[drive_dofs[i, 2]] = command[1]
    target_qd[drive_dofs[i, 3]] = command[1]


@wp.kernel(enable_backward=False)
def _advance_time(time: wp.array[wp.float32], dt: float):
    time[0] = time[0] + dt


def _rot_y(angle):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle))


def _tread_hinges():
    """Hinge points (x, z) of one tread loop in the robot frame, in loop order.

    The bottom run starts under the rear roller and runs forward; four links
    wrap each roller at 45 degrees apiece, the middle one of each wrap sitting
    on the straight runs, so every chord touches the rim at its midpoint.
    """
    front, rear = 0.5 * ROLLER_SPACING, -0.5 * ROLLER_SPACING
    hinges = [
        (rear + HINGE_RADIUS * np.cos(np.radians(247.5)) + k * PITCH, AXLE_HEIGHT - CHORD_RADIUS) for k in range(9)
    ]
    hinges += [(front + HINGE_RADIUS * np.cos(np.radians(a)), AXLE_HEIGHT + HINGE_RADIUS * np.sin(np.radians(a))) for a in (-22.5, 22.5, 67.5)]  # fmt: skip
    hinges += [(hinges[-1][0] - k * PITCH, AXLE_HEIGHT + CHORD_RADIUS) for k in range(1, 9)]
    hinges += [(rear + HINGE_RADIUS * np.cos(np.radians(a)), AXLE_HEIGHT + HINGE_RADIUS * np.sin(np.radians(a))) for a in (157.5, 202.5)]  # fmt: skip
    return np.asarray(hinges)


def _hsv(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i]


def add_tread_robot(builder, pose, color, label):
    """Add one tracked robot at ``pose`` (a planar transform) and return its parts.

    Returns the chassis body, the four driven roller joints (left front, left
    rear, right front, right rear) and the tread hinge joints.
    """
    accent = (1.0, 0.85, 0.25)
    dark = (0.12, 0.12, 0.14)
    visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    hull_cfg = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.4, gap=0.002)
    chassis = builder.add_link(xform=pose, label=f"{label}/chassis")
    # Hull with deck and bumpers; the deck overhangs the treads so neighbours bump
    # hull to hull rather than tread to tread.
    builder.add_shape_box(chassis, xform=wp.transform((0.0, 0.0, 0.037), wp.quat_identity()), hx=0.056, hy=0.030, hz=0.021, cfg=hull_cfg, color=color, label=f"{label}/hull")  # fmt: skip
    builder.add_shape_box(chassis, xform=wp.transform((0.0, 0.0, 0.0475), wp.quat_identity()), hx=0.068, hy=0.060, hz=0.0045, cfg=hull_cfg, color=accent, label=f"{label}/deck")  # fmt: skip
    # Hidden battery pack low in the hull.
    builder.add_shape_box(chassis, xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()), hx=0.04, hy=0.022, hz=0.008, cfg=newton.ModelBuilder.ShapeConfig(density=7000.0, has_shape_collision=False, is_visible=False), label=f"{label}/battery")  # fmt: skip
    builder.add_shape_box(chassis, xform=wp.transform((0.0, 0.0, 0.0595), wp.quat_identity()), hx=0.045, hy=0.026, hz=0.0075, cfg=visual, color=color, label=f"{label}/cabin")  # fmt: skip
    builder.add_shape_box(chassis, xform=wp.transform((0.069, 0.0, 0.034), wp.quat_identity()), hx=0.006, hy=0.032, hz=0.006, cfg=visual, color=dark, label=f"{label}/bumper")  # fmt: skip
    # Sensor head on a short mast, with a lens and two antenna stubs.
    builder.add_shape_cylinder(chassis, xform=wp.transform((0.02, 0.0, 0.08), wp.quat_identity()), radius=0.004, half_height=0.014, cfg=visual, color=dark, label=f"{label}/mast")  # fmt: skip
    head = wp.transform((0.024, 0.0, 0.104), wp.quat_identity())
    builder.add_shape_box(chassis, xform=head, hx=0.016, hy=0.022, hz=0.012, cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.4, gap=0.002), color=(0.95, 0.95, 0.92), label=f"{label}/head")  # fmt: skip
    lens = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)
    for k, y in enumerate((-0.0095, 0.0095)):
        builder.add_shape_cylinder(chassis, xform=wp.transform((0.0405, y, 0.105), lens), radius=0.0065, half_height=0.0015, cfg=visual, color=dark, label=f"{label}/lens{k}")  # fmt: skip
        builder.add_shape_sphere(chassis, xform=wp.transform((0.042, y + 0.002, 0.107), wp.quat_identity()), radius=0.0018, cfg=visual, color=(0.4, 0.9, 1.0), label=f"{label}/glint{k}")  # fmt: skip
        tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.sign(y) * -0.35))
        stub = wp.transform_point(wp.transform((0.018, y, 0.116), tilt), wp.vec3(0.0, 0.0, 0.010))
        builder.add_shape_capsule(chassis, xform=wp.transform(stub, tilt), radius=0.0015, half_height=0.008, cfg=visual, color=dark, label=f"{label}/antenna{k}")  # fmt: skip
        tip = wp.transform_point(wp.transform((0.018, y, 0.116), tilt), wp.vec3(0.0, 0.0, 0.021))
        builder.add_shape_sphere(chassis, xform=wp.transform(tip, wp.quat_identity()), radius=0.004, cfg=visual, color=accent if k else color, label=f"{label}/antenna_tip{k}")  # fmt: skip

    link_cfg = newton.ModelBuilder.ShapeConfig(
        density=LINK_MASS / ((PITCH - 0.0006) * SHOE_WIDTH * SHOE_THICKNESS), mu=SHOE_FRICTION, gap=0.0005
    )
    guide_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3, gap=0.0005, is_visible=False)
    rim_cfg = newton.ModelBuilder.ShapeConfig(density=1500.0, mu=RIM_FRICTION, gap=0.0005)
    axle = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
    hinges = _tread_hinges()
    flip = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
    drives, tread_joints = [], []
    for side, y in (("left", TREAD_Y), ("right", -TREAD_Y)):
        # A slippery skid between the guide teeth carries the top run, so the
        # loose loop does not sag between the rollers.
        builder.add_shape_box(chassis, xform=wp.transform((0.0, y, AXLE_HEIGHT + CHORD_RADIUS - 0.0032), wp.quat_identity()), hx=0.02, hy=RIM_HALF_WIDTH - 0.0005, hz=0.003, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, gap=0.0005), color=(0.12, 0.12, 0.14), label=f"{label}/skid_{side}")  # fmt: skip
        for name, x in (("front", 0.5 * ROLLER_SPACING), ("rear", -0.5 * ROLLER_SPACING)):
            local = wp.transform((x, y, AXLE_HEIGHT), wp.quat_identity())
            roller = builder.add_link(xform=pose * local, label=f"{label}/roller_{side}_{name}")
            builder.add_shape_cylinder(roller, xform=wp.transform((0.0, 0.0, 0.0), axle), radius=RIM_RADIUS, half_height=RIM_HALF_WIDTH, cfg=rim_cfg, color=(1.0, 0.5, 0.15), label=f"{label}/rim_{side}_{name}")  # fmt: skip
            builder.add_shape_cylinder(roller, xform=wp.transform((0.0, 0.0, 0.0), axle), radius=0.004, half_height=0.0105, cfg=visual, color=dark, label=f"{label}/hub_{side}_{name}")  # fmt: skip
            drives.append(
                builder.add_joint_revolute(
                    chassis,
                    roller,
                    parent_xform=local,
                    child_xform=wp.transform_identity(),
                    axis=(0.0, 1.0, 0.0),
                    target_ke=0.0,
                    target_kd=MOTOR_GAIN,
                    effort_limit=MOTOR_TORQUE,
                    armature=MOTOR_ARMATURE,
                    actuator_mode=newton.JointTargetMode.VELOCITY,
                    collision_filter_parent=True,
                    label=f"{label}/drive_{side}_{name}",
                )
            )
        # Links: local x runs from this hinge to the next, local z points out of the loop.
        links, rotations = [], []
        for k in range(len(hinges)):
            a, b = hinges[k], hinges[(k + 1) % len(hinges)]
            rotation = _rot_y(-np.arctan2(b[1] - a[1], b[0] - a[0])) * flip
            link = builder.add_link(
                xform=pose * wp.transform((a[0], y, a[1]), rotation), label=f"{label}/{side}_link{k}"
            )
            builder.add_shape_box(link, xform=wp.transform((0.5 * PITCH, 0.0, 0.5 * SHOE_THICKNESS), wp.quat_identity()), hx=0.5 * PITCH - 0.0003, hy=0.5 * SHOE_WIDTH, hz=0.5 * SHOE_THICKNESS, cfg=link_cfg, color=(0.2, 0.2, 0.22) if k % 2 else (0.3, 0.3, 0.33), label=f"{label}/{side}_shoe{k}")  # fmt: skip
            for g, offset in enumerate((-GUIDE_OFFSET, GUIDE_OFFSET)):
                builder.add_shape_box(link, xform=wp.transform((0.5 * PITCH, offset, -0.5 * GUIDE_HEIGHT), wp.quat_identity()), hx=0.25 * PITCH, hy=GUIDE_HALF_WIDTH, hz=0.5 * GUIDE_HEIGHT, cfg=guide_cfg, label=f"{label}/{side}_guide{k}_{g}")  # fmt: skip
            links.append(link)
            rotations.append(rotation)
        for k in range(len(links)):
            n = (k + 1) % len(links)
            tread_joints.append(
                builder.add_joint_revolute(
                    links[k],
                    links[n],
                    parent_xform=wp.transform((PITCH, 0.0, 0.0), wp.quat_identity()),
                    # Zero joint angle in the assembled pose.
                    child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_inverse(rotations[n]) * rotations[k]),
                    axis=(0.0, 1.0, 0.0),
                    friction=HINGE_FRICTION,
                    damping=HINGE_DAMPING,
                    collision_filter_parent=True,
                    label=f"{label}/{side}_hinge{k}",
                )
            )
    return chassis, drives, tread_joints


def build_swarm(builder, count):
    """Add ``count`` robots on their rings; return chassis bodies, drive joints and lane parameters."""
    robots = []
    index = 0
    for ring, (radius, ring_count, amplitude, direction) in enumerate(RINGS):
        for k in range(ring_count):
            if index == count:
                return robots
            angle = 2.0 * np.pi * (k + 0.5 * ring) / ring_count
            heading = angle + direction * 0.5 * np.pi
            pose = wp.transform(
                (radius * np.cos(angle), radius * np.sin(angle), 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(heading)),
            )
            color = _hsv((index * 0.618034) % 1.0, 0.65, 0.95)
            chassis, drives, hinges = add_tread_robot(builder, pose, color, f"robot{index}")
            robots.append((chassis, drives, hinges, radius, amplitude, direction))
            index += 1
    return robots


class Example:
    """Thirty tracked robots swirl in a breathing flower pattern."""

    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        if not 1 <= args.robots <= sum(ring[1] for ring in RINGS):
            raise ValueError("--robots must be between 1 and 30")
        robots = build_swarm(builder, args.robots)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=FLOOR_FRICTION, gap=0.0005))
        # Painted dots on the floor mark the arena circle.
        for k in range(48):
            angle = 2.0 * np.pi * k / 48
            builder.add_shape_cylinder(
                -1,
                xform=wp.transform(
                    (ARENA_RADIUS * np.cos(angle), ARENA_RADIUS * np.sin(angle), 0.0), wp.quat_identity()
                ),
                radius=0.03,
                half_height=0.0008,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
                color=(1.0, 0.8, 0.2) if k % 2 else (0.2, 0.6, 1.0),
                label=f"arena_dot{k}",
            )
        self.model = builder.finalize(skip_validation_joints=True)
        self.state = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_qd.zero_()
        device = self.model.device

        qd_start = self.model.joint_qd_start.numpy()
        self.chassis_ids = np.array([r[0] for r in robots], dtype=np.int32)
        self.hinge_joints = np.concatenate([r[2] for r in robots])
        self.num_robots = len(robots)
        self.chassis = wp.array(self.chassis_ids, dtype=wp.int32, device=device)
        self.drive_dofs = wp.array([[qd_start[j] for j in r[1]] for r in robots], dtype=wp.int32, device=device)
        self.lane_radius = wp.array([r[3] for r in robots], dtype=wp.float32, device=device)
        self.lane_amplitude = wp.array([r[4] for r in robots], dtype=wp.float32, device=device)
        self.direction = wp.array([r[5] for r in robots], dtype=wp.float32, device=device)
        self.applied = wp.zeros(self.num_robots, dtype=wp.vec2, device=device)
        self.time = wp.zeros(1, dtype=wp.float32, device=device)

        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching="latest",
            speculative_contact_gap_max=0.001,
            speculative_contact_velocity_filter=False,
            contact_reduction_voxel_depth=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            # Tread hinges and motors form exact mechanisms. Their contacts stay in the
            # regular contact sweeps by default: twice the frame rate for about 5 % more
            # tread slip than the exact mechanism contact response (--direct-contacts).
            joint_mode="maximal_direct",
            direct_contact_mask=None if args.direct_contacts else np.zeros(self.model.joint_count, dtype=bool),
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
            velocity_iterations=1,
            direct_joint_projection_passes=1,
            parallel_contact_prepare=False,
            enable_body_pair_grouping=True,
            mass_splitting=True,
            mass_splitting_batch_size=2,
            max_colored_partitions=args.max_colored_partitions,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -2.6, 1.8), pitch=-35.0, yaw=90.0)

        # Test bookkeeping (sampled on the host in test mode only).
        poses = self.state.body_q.numpy()
        self._hinge_parent = self.model.joint_parent.numpy()[self.hinge_joints]
        self._hinge_child = self.model.joint_child.numpy()[self.hinge_joints]
        self._hinge_parent_xform = self.model.joint_X_p.numpy()[self.hinge_joints]
        self._hinge_child_xform = self.model.joint_X_c.numpy()[self.hinge_joints]
        body_robot = np.full(self.model.body_count, -1)
        labels = self.model.body_label
        for i, label in enumerate(labels):
            body_robot[i] = int(label.split("/")[0].removeprefix("robot"))
        self._link_ids = np.array([i for i, label in enumerate(labels) if "_link" in label])
        self._link_robot = body_robot[self._link_ids]
        self._link_side = np.array([TREAD_Y if "/left_" in labels[i] else -TREAD_Y for i in self._link_ids])
        self.trajectory = [(0.0, poses[self.chassis_ids, :3].copy())]
        self.min_up = 1.0
        self.max_hinge_gap = 0.0
        self.max_lateral = 0.0
        self.max_loop_distance = 0.0

        self.graph = None
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        overlap = self.overlap_simulation_render and self.viewer.supports_simulation_render_overlap
        self._render_states = (self.model.state(), self.model.state()) if overlap else None
        self._render_done = [wp.Event(device) if device.is_cuda else None for _ in range(2)]
        self._render_index = 0
        self._render_prepared = False
        self._render_time = 0.0

    def simulate(self):
        wp.launch(
            _steer_swarm,
            dim=self.num_robots,
            inputs=[
                self.state.body_q,
                self.chassis,
                self.lane_radius,
                self.lane_amplitude,
                self.direction,
                self.time,
                self.drive_dofs,
                self.frame_dt,
            ],
            outputs=[self.applied, self.control.joint_target_qd],
        )
        dt = self.frame_dt / self.args.contact_updates
        for _ in range(self.args.contact_updates):
            self.pipeline.collide(self.state, self.contacts, dt=dt)
            self.state.clear_forces()
            self.viewer.apply_forces(self.state)
            self.solver.step(self.state, self.state, self.control, self.contacts, dt)
        wp.launch(_advance_time, dim=1, inputs=[self.time, self.frame_dt])

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def prepare_render_state(self):
        """Snapshot body poses before the next asynchronous physics step."""
        self._render_index = 1 - self._render_index
        done = self._render_done[self._render_index]
        if done is not None:
            wp.wait_event(done)
        wp.copy(self._render_states[self._render_index].body_q, self.state.body_q)
        self._render_time = self.sim_time
        self._render_prepared = True

    def render(self):
        prepared = self._render_prepared
        state = self._render_states[self._render_index] if prepared else self.state
        self.viewer.begin_frame(self._render_time if prepared else self.sim_time)
        self.viewer.log_state(state)
        if not prepared:
            self.viewer.log_contacts(self.contacts, state)
        else:
            done = self._render_done[self._render_index]
            if done is not None:
                wp.record_event(done)
        self.viewer.end_frame()
        self._render_prepared = False

    def _sample(self):
        poses = self.state.body_q.numpy()
        chassis = poses[self.chassis_ids]
        quat = chassis[:, 3:]
        # z component of each chassis's up axis.
        up_z = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
        self.min_up = min(self.min_up, float(up_z.min()))

        def apply(pose, point):
            q = pose[:, 3:]
            cross = 2.0 * np.cross(q[:, :3], point)
            return pose[:, :3] + point + q[:, 3:4] * cross + np.cross(q[:, :3], cross)

        def frame(pose, xform):
            # World position of a joint anchor (translation part only).
            return apply(pose, xform[:, :3])

        gap = np.linalg.norm(
            frame(poses[self._hinge_parent], self._hinge_parent_xform)
            - frame(poses[self._hinge_child], self._hinge_child_xform),
            axis=1,
        )
        self.max_hinge_gap = max(self.max_hinge_gap, float(gap.max()))
        # Link centres in their chassis frame: sideways offset from the tread
        # plane and distance from the loop's rest path.
        owner = chassis[self._link_robot]
        delta = poses[self._link_ids, :3] - owner[:, :3]
        q = owner[:, 3:] * np.array((-1.0, -1.0, -1.0, 1.0))
        cross = 2.0 * np.cross(q[:, :3], delta)
        local = delta + q[:, 3:4] * cross + np.cross(q[:, :3], cross)
        self.max_lateral = max(self.max_lateral, float(np.abs(local[:, 1] - self._link_side).max()))
        # Distance of each hinge from the stadium the loop runs on.
        x = np.clip(local[:, 0], -0.5 * ROLLER_SPACING, 0.5 * ROLLER_SPACING)
        off = np.hypot(local[:, 0] - x, local[:, 2] - AXLE_HEIGHT) - HINGE_RADIUS
        self.max_loop_distance = max(self.max_loop_distance, float(np.abs(off).max()))
        self.trajectory.append((self.sim_time, chassis[:, :3].copy()))

    def test_post_step(self):
        if round(self.sim_time / self.frame_dt) % 15 == 0:
            self._sample()

    def test_final(self):
        self._sample()
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all() and np.isfinite(self.state.body_qd.numpy()).all()
        positions = poses[self.chassis_ids, :3]
        radius = np.linalg.norm(positions[:, :2], axis=1)
        speeds = None
        if self.sim_time >= 3.0:
            start = next(p for t, p in self.trajectory if t >= self.sim_time - 2.0 - 1.0e-6)
            speeds = np.linalg.norm(positions[:, :2] - start[:, :2], axis=1) / 2.0
        print(
            f"Tread swarm: min up {self.min_up:.3f}, max radius {radius.max():.3f} m, "
            f"max hinge gap {self.max_hinge_gap * 1e3:.2f} mm, max lateral {self.max_lateral * 1e3:.2f} mm, "
            f"max loop distance {self.max_loop_distance * 1e3:.2f} mm"
            + (f", speed min/mean {speeds.min():.3f}/{speeds.mean():.3f} m/s" if speeds is not None else ""),
            flush=True,
        )
        assert self.min_up > 0.8, f"A robot tipped over (up z {self.min_up:.3f})"
        assert radius.max() < ARENA_RADIUS + 0.1, f"A robot left the arena ({radius.max():.3f} m)"
        assert self.max_hinge_gap < 0.001, f"A tread hinge opened ({self.max_hinge_gap} m)"
        assert self.max_lateral < 0.006, f"A tread slid off its rollers ({self.max_lateral} m)"
        assert self.max_loop_distance < 0.025, f"A tread left its loop ({self.max_loop_distance} m)"
        if speeds is not None:
            assert speeds.min() > 0.05, f"A robot stalled ({speeds.min():.3f} m/s)"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix")
        parser.add_argument("--robots", type=int, default=30, help="Number of robots (at most 30).")
        parser.add_argument(
            "--direct-contacts",
            action="store_true",
            help="Solve tread contacts with the exact mechanism response (less slip, about half the frame rate).",
        )
        parser.add_argument("--substeps", type=int, default=7, help="Substeps per contact update.")
        parser.add_argument("--iterations", type=int, default=2, help="Solver iterations per substep.")
        parser.add_argument("--contact-updates", type=int, default=2, help="Collision updates per frame.")
        parser.add_argument(
            "--max-colored-partitions", type=int, default=4, help="True contact colors before overflow mass splitting."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
