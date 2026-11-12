# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run a mixed robot fleet between the shelves of one warehouse world with PhoenX.

A fleet of Unitree G1 and H1 humanoids, ANYmal quadrupeds, Kiva-style drive units, and
forklifts stands in the aisles between rows of shelving pods and pallets. Every robot is its
own articulation, held in its standing pose by position drives, and is solved in reduced
coordinates. Everything else is a free rigid body or an ordinary hinge or slider: pods loaded
with packages, pallets stacked with cartons, loose packages on the robots' feet, mesh totes
with signed distance fields, hinged gates, and drawers. The gates and drawers are solved in
reduced coordinates too, next to the robots; ``--mechanism-solver regular`` leaves them to
the solver's regular joint rows. With ``--runtime-robots`` further humanoids arrive at a dock
beside the aisles and leave again while the simulation runs, created and removed on the device.
With ``--walking-anymals`` the fleet is that many ANYmals which a walking policy steers around
the aisles instead: forward in one lane, back in the other, turning at both ends. One network
evaluation per frame serves all of them, inside the same CUDA graph as the step, at the
policy's rate of 50 Hz.
With ``--sorting-humanoids`` the warehouse becomes a sorting hall: two lines of G1 humanoids face
each other across every conveyor belt, and each sweeps its right hand over the belt about every
two seconds, pushing the packages it catches off the far side. A hand sweeps the same strip of the
belt as the hand of the humanoid opposite, so the two meet over the belt and push packages back
and forth; their periods differ slightly, so the pairs that meet change over time. Packages
appear at the start of every belt and are removed once they drop off its sides or its end, all
from device code inside the graph. Drive units carry totes up and down the aisles between the
lines, brushing past the humanoids' heels.

The scene exercises what one large world mixes: robots that touch free bodies (a package on a
humanoid's foot, a pod carried by a drive unit, a loaded pallet on a forklift), free bodies
shared by two robots (a long package across the feet of two humanoids), robots that touch each
other, sleeping stacks next to awake robots, and jointed mechanisms of the building. All
robots and bodies share one world, and collision detection and stepping run in one CUDA
graph.

Success criteria:
- All body poses stay finite and nothing falls through the floor.
- Every robot stays upright: humanoids and quadrupeds keep their root height, drive units
  and forklifts stay level on their wheels.
- Stacked cartons and shelved packages stay within a few centimeters of where they started.
- A meaningful fraction of the free bodies falls asleep (outside the sorting hall).
- The gates and drawers stay on their joints.
- Every humanoid that arrives at the dock stands, and no arrival or departure fails.
- In the sorting hall, the humanoids push packages off their belts, facing humanoids touch each
  other, the drive units shuttle their totes along the aisles, and no package is lost or left
  over: every package put on a belt is still on it or was removed.
- Walking ANYmals stay within 0.4 m of their loop and cover it at 0.7 times the commanded
  1 m/s or more on average, none at less than 0.4 times.

Runs:
- `python -m newton.examples phoenx_warehouse`
- `python -m newton.examples phoenx_warehouse --robots 1024`
  - The same warehouse with eight times the fleet and storage.
- `python -m newton.examples phoenx_warehouse --runtime-robots 8`
  - Eight more humanoids come and go at the dock, one every half second.
- `python -m newton.examples phoenx_warehouse --walking-anymals 1024`
  - A thousand policy-driven ANYmals walk laps through the aisles.
- `python -m newton.examples phoenx_warehouse --sorting-humanoids 1024`
  - A thousand humanoids sort packages off 16 conveyor belts.
- `python -m newton.examples phoenx_warehouse --viewer null --timing-frames 600`
  - Prints the scene counts and the milliseconds per captured step.
"""

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.solvers import SolverPhoenX, phoenx

CONTACT_GAP = 0.005  # [m]; the default gap of 0.1 m pairs every shelf with all its packages
REST_GAP = 0.0005  # [m], initial clearance between resting bodies
FRICTION = 0.75

# Fleet mix; the remainder of an uneven split goes to the types with the largest fractions.
FLEET_MIX = (("g1", 0.40), ("h1", 0.20), ("anymal", 0.15), ("kiva", 0.15), ("forklift", 0.10))

HAND_OVERLAP = 0.004  # [m] by which the hand colliders of two touching humanoids overlap

SLOT_PITCH = 2.8  # [m] between robots along an aisle
DOCK_DISTANCE = 3.0  # [m] from the first aisle's corner to the first dock
DOCK_PITCH = 1.2  # [m] between docks
DOCK_PERIOD = 30  # frames between two arrivals or departures
AISLE_WIDTH = 3.2  # [m] clear width between storage rows
STORAGE_PITCH = 1.3  # [m] between pods along a storage row
STORAGE_DEPTH = 1.1  # [m] of one line of pods across a storage row

# Unitree G1 standing pose [rad] and drive gains, in actuator order.
G1_JOINT_POS = (
    *(-0.10, 0.0, 0.0, 0.30, -0.20, 0.0),
    *(-0.10, 0.0, 0.0, 0.30, -0.20, 0.0),
    *(0.0, 0.0, 0.0),
    *(0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0),
    *(0.20, -0.20, 0.0, 1.28, 0.0, 0.0, 0.0),
)
G1_KP = (
    *(100.0, 100.0, 100.0, 150.0, 40.0, 40.0),
    *(100.0, 100.0, 100.0, 150.0, 40.0, 40.0),
    *(75.0, 75.0, 75.0),
    *(75.0, 75.0, 75.0, 75.0, 2.0, 2.0, 2.0),
    *(75.0, 75.0, 75.0, 75.0, 2.0, 2.0, 2.0),
)
G1_KD = (
    *(4.0, 4.0, 4.0, 6.0, 3.0, 2.2),
    *(4.0, 4.0, 4.0, 6.0, 3.0, 2.2),
    *(4.0, 4.0, 4.0),
    *(4.0, 4.0, 4.0, 4.0, 0.4, 0.4, 0.4),
    *(4.0, 4.0, 4.0, 4.0, 0.4, 0.4, 0.4),
)
G1_GAIN_SCALE = 8.0  # holds the pose without a balance policy
G1_ROOT_HEIGHT = 0.785  # [m]
G1_FOOT_CENTER = (0.04, 0.0, -0.029)  # [m], in the ankle roll link
G1_FOOT_HALF_EXTENTS = (0.09, 0.03, 0.008)  # [m]
G1_FOOT_FRICTION = 0.6

# Unitree H1 standing pose [rad] and drive gains, in actuator order.
H1_JOINT_POS = (
    *(0.0, 0.0, -0.28, 0.79, -0.52),
    *(0.0, 0.0, -0.28, 0.79, -0.52),
    0.0,
    *(0.28, 0.0, 0.0, 0.52),
    *(0.28, 0.0, 0.0, 0.52),
)
# The ankles are stiffer than the training gain of 20 N*m/rad: without a balance policy they
# must outweigh gravity's pull on the standing robot, about 460 N*m/rad.
H1_KP = (
    *(150.0, 150.0, 200.0, 200.0, 100.0),
    *(150.0, 150.0, 200.0, 200.0, 100.0),
    200.0,
    *(40.0, 40.0, 40.0, 40.0),
    *(40.0, 40.0, 40.0, 40.0),
)
H1_KD = (*(5.0, 5.0, 5.0, 5.0, 10.0), *(5.0, 5.0, 5.0, 5.0, 10.0), 5.0, *(10.0,) * 8)
H1_GAIN_SCALE = 8.0
H1_ROOT_HEIGHT = 0.98  # [m]
H1_FOOT_CENTER = (0.05, 0.0, -0.02952094)  # [m], in the ankle link
H1_FOOT_HALF_EXTENTS = (0.15, 0.015, 0.04047906)  # [m]

ANYMAL_ROOT_HEIGHT = 0.55  # [m]
ANYMAL_JOINT_POS = {
    **{"LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8},
    **{"RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8},
    **{"LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8},
    **{"RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8},
}
ANYMAL_KP = 600.0  # four times the training gain, so the legs barely sag
ANYMAL_KD = 20.0

# Walking ANYmals: the gains and rate their policy was trained with, and the loop they walk.
WALK_KP = 150.0
WALK_KD = 5.0
WALK_FPS = 50  # policy rate [Hz]
WALK_ACTION_SCALE = 0.5  # [rad] per unit of policy output
WALK_SPEED = 1.0  # commanded on the straights [m/s]
WALK_TURN_SPEED = 0.6  # commanded while facing across the loop [m/s]
WALK_YAW_RATE_MAX = 1.0  # [rad/s], the largest turn command the policy was trained with
LANE_OFFSET = 0.8  # [m] of the two lanes from the middle of their aisle

# Sorting lines: two lines of humanoids face each other across a belt and push passing packages
# off it with their right hands; each hand sweeps the same strip of the belt as the opposite one.
SORT_ROBOTS_PER_LINE = 32  # on each side of a belt
SORT_ROBOT_PITCH = 0.9  # [m] between humanoids along a line
SORT_LINE_PITCH = 2.0  # [m] between belts
SORT_BELT_NEAR = 0.14  # [m] near edge of the belt in front of a humanoid's root
SORT_BELT_FAR = 0.32  # [m] far edge, beyond which pushed packages drop to the floor
SORT_FACING = SORT_BELT_NEAR + SORT_BELT_FAR  # [m] between the roots of two facing humanoids
SORT_PERIOD_SPREAD = 0.1  # relative spread of the humanoids' periods, so the pairs that meet change
SORT_BELT_TOP = 0.70  # [m]
SORT_BELT_SPEED = 0.25  # [m/s]
SORT_PACKAGE_HALF_EXTENTS = wp.constant(wp.vec3(0.06, 0.08, 0.06))  # [m], across and along the belt, and up
SORT_PACKAGE_DENSITY = 200.0  # [kg/m^3]
SORT_SPAWN_CLEARANCE = 0.4  # [m] a package moves along its belt before the next one appears
SORT_DROP_HEIGHT = 0.4  # [m]; below it a package has left its belt and is removed
SORT_PERIOD = 2.0  # [s] of one push of a humanoid's arm
# Right shoulder pitch and elbow [rad] that sweep the hand forward at 0.80 m height, over the belt.
# A drive unit shuttles a tote up and down every aisle between the sorting lines. The aisles are
# too narrow for two units to pass, so each has one.
SORT_KIVAS_PER_AISLE = 1
SORT_KIVA_SPEED = 0.6  # [m/s], spread by up to 30% between units
KIVA_WHEEL_RADIUS = 0.08  # [m]
KIVA_WHEEL_TRACK = 0.31  # [m] from the chassis' middle to each wheel
KIVA_DRIVE_KD = 40.0  # [N*m*s/rad] of a wheel's velocity drive
SORT_ARM_SWEEP = ((0.4, 0.15), (0.2, 0.3), (0.0, 0.5), (-0.2, 0.75), (-0.4, 1.05), (-0.6, 1.4))

KIVA_ROOT_HEIGHT = 0.12  # chassis center [m]
KIVA_TOP = 0.25  # top of the lift plate [m]
FORKLIFT_ROOT_HEIGHT = 0.45  # chassis center [m]
FORK_REACH = 0.72  # fork carriage in front of the chassis center [m]
FORK_LENGTH = 1.0  # [m]
FORK_THICKNESS = 0.04  # [m]
FORK_EMPTY_HEIGHT = 0.10  # underside of the tines above the floor [m]
FORK_LOADED_HEIGHT = 0.35  # [m]

POD_SHELF_TOPS = (0.26, 0.815, 1.365)  # [m] above the pod's feet
POD_UNDERSIDE = 0.23  # underside of the lowest shelf, where a drive unit lifts the pod [m]
PACKAGE_HALF_EXTENTS = ((0.16, 0.12, 0.09), (0.12, 0.16, 0.07), (0.13, 0.13, 0.12), (0.18, 0.10, 0.06))
PALLET_HALF_EXTENTS = (0.5, 0.4, 0.07)
CARTON_HALF_EXTENTS = (0.24, 0.19, 0.14)

# Lowest root height [m] at which a robot of each type still counts as upright.
ROOT_HEIGHT_MIN = {"g1": 0.55, "h1": 0.70, "anymal": 0.35, "kiva": 0.07, "forklift": 0.35}

COLORS = {
    "robot": (0.30, 0.33, 0.38),
    "kiva": (0.95, 0.45, 0.10),
    "forklift": (0.95, 0.75, 0.10),
    "steel": (0.25, 0.27, 0.30),
    "pod": (0.16, 0.36, 0.62),
    "pallet": (0.70, 0.55, 0.36),
    "carton": (0.78, 0.62, 0.42),
    "package": (0.66, 0.50, 0.32),
    "tote": (0.20, 0.55, 0.50),
    "frame": (0.55, 0.57, 0.60),
    "gate": (0.85, 0.20, 0.15),
}


def _fleet_counts(robots: int) -> dict[str, int]:
    """Split ``robots`` over the fleet mix by largest remainder."""
    shares = [(name, fraction * robots) for name, fraction in FLEET_MIX]
    counts = {name: int(share) for name, share in shares}
    by_remainder = sorted(shares, key=lambda item: item[1] - int(item[1]), reverse=True)
    for name, _ in by_remainder[: robots - sum(counts.values())]:
        counts[name] += 1
    return counts


def _shape_cfg(density: float = 200.0, mu: float = FRICTION, visible: bool = True) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=density, mu=mu, gap=CONTACT_GAP, is_visible=visible)


def _new_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_shape_cfg.mu = FRICTION
    builder.default_shape_cfg.gap = CONTACT_GAP
    return builder


def _position(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> wp.transform:
    return wp.transform(wp.vec3(x, y, z), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))


def _body_named(builder: newton.ModelBuilder, name: str) -> int:
    matches = [body for body, label in enumerate(builder.body_label) if label.rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one body named {name!r}, found {len(matches)}")
    return matches[0]


def _standing_poses(builder: newton.ModelBuilder) -> np.ndarray:
    """Return the body poses of a robot prototype in its initial joint configuration."""
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state.body_q.numpy()


def _set_position_drives(builder: newton.ModelBuilder, positions, stiffness, damping) -> None:
    """Hold the actuated joints behind a free root at ``positions``."""
    for index, position in enumerate(positions):
        builder.joint_q[7 + index] = position
        builder.joint_target_q[7 + index] = position
        builder.joint_target_ke[6 + index] = stiffness[index]
        builder.joint_target_kd[6 + index] = damping[index]
        builder.joint_target_mode[6 + index] = int(newton.JointTargetMode.POSITION)


def _add_contact_proxy(
    builder: newton.ModelBuilder, body: int, center, half_extents=None, radius=None, visible: bool = True
) -> None:
    """Add a massless collider to an imported link that has none."""
    cfg = _shape_cfg(density=0.0, visible=visible)
    xform = wp.transform(wp.vec3(*center), wp.quat_identity())
    if radius is not None:
        builder.add_shape_sphere(body, xform=xform, radius=radius, cfg=cfg, color=COLORS["robot"])
    else:
        hx, hy, hz = half_extents
        builder.add_shape_box(body, xform=xform, hx=hx, hy=hy, hz=hz, cfg=cfg, color=COLORS["robot"])


def _keep_meshes_as_visuals(builder: newton.ModelBuilder) -> None:
    """Stop imported meshes from colliding: the robots collide with primitives only."""
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for shape, shape_type in enumerate(builder.shape_type):
        if shape_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~collide


def build_g1(visual_meshes: bool = False) -> newton.ModelBuilder:
    """Unitree G1 with primitive colliders, flat foot boxes, and stiff standing drives."""
    builder = _new_builder()
    asset = newton.utils.download_asset("unitree_g1")
    builder.add_mjcf(
        str(asset / "mjcf" / "g1_29dof.xml"),
        floating=None,
        enable_self_collisions=False,
        ignore_names=("floor", "ground"),
        parse_visuals=visual_meshes,
        parse_meshes=visual_meshes,
    )
    _keep_meshes_as_visuals(builder)
    imported_shapes = builder.shape_count
    visible = not visual_meshes  # the meshes show the robot; the colliders would cover them
    foot_cfg = _shape_cfg(density=1000.0, mu=G1_FOOT_FRICTION, visible=visible)
    foot_xform = wp.transform(wp.vec3(*G1_FOOT_CENTER), wp.quat_identity())
    hx, hy, hz = G1_FOOT_HALF_EXTENTS
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    feet = []
    for side in ("left", "right"):
        # The four spheres under each ankle roll and slide; a flat box stands.
        for shape, label in enumerate(builder.shape_label[:imported_shapes]):
            if f"{side}_ankle_roll_link_geom_" in label:
                builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~collide
        foot = builder.add_shape_box(
            _body_named(builder, f"{side}_ankle_roll_link"),
            xform=foot_xform,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=foot_cfg,
            color=COLORS["robot"],
        )
        for shape in range(imported_shapes):
            builder.add_shape_collision_filter_pair(foot, shape)
        feet.append(foot)
    builder.add_shape_collision_filter_pair(feet[0], feet[1])
    # The imported model has no torso or hand colliders: neighbors and packages need some.
    _add_contact_proxy(
        builder, _body_named(builder, "torso_link"), (0.0, 0.0, 0.13), half_extents=(0.07, 0.10, 0.13), visible=visible
    )
    for side in ("left", "right"):
        _add_contact_proxy(
            builder, _body_named(builder, f"{side}_wrist_yaw_link"), (0.0, 0.0, 0.0), radius=0.045, visible=visible
        )
    builder.joint_q[:3] = [0.0, 0.0, G1_ROOT_HEIGHT]
    _set_position_drives(
        builder,
        G1_JOINT_POS,
        [G1_GAIN_SCALE * gain for gain in G1_KP],
        [G1_GAIN_SCALE * gain for gain in G1_KD],
    )
    return builder


def build_h1(visual_meshes: bool = False) -> newton.ModelBuilder:
    """Unitree H1 with primitive colliders, foot boxes, and standing drives."""
    builder = _new_builder()
    asset = newton.utils.download_asset("unitree_h1")
    builder.add_mjcf(
        str(asset / "mjcf" / "h1.xml"),
        up_axis="Z",
        parse_meshes=visual_meshes,
        parse_visuals=visual_meshes,
        ignore_names=("floor", "ground"),
        enable_self_collisions=False,
        parse_mujoco_options=False,
    )
    _keep_meshes_as_visuals(builder)
    foot_xform = wp.transform(wp.vec3(*H1_FOOT_CENTER), wp.quat_identity())
    hx, hy, hz = H1_FOOT_HALF_EXTENTS
    for side in ("left", "right"):
        builder.add_shape_box(
            _body_named(builder, f"{side}_ankle_link"),
            xform=foot_xform,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=_shape_cfg(density=1000.0, visible=not visual_meshes),
            color=COLORS["robot"],
        )
        _add_contact_proxy(
            builder, _body_named(builder, f"{side}_elbow_link"), (0.0, 0.0, 0.0), radius=0.06, visible=not visual_meshes
        )
    _add_contact_proxy(
        builder,
        _body_named(builder, "torso_link"),
        (0.0, 0.0, 0.25),
        half_extents=(0.09, 0.13, 0.22),
        visible=not visual_meshes,
    )
    builder.joint_q[:7] = [0.0, 0.0, H1_ROOT_HEIGHT, 0.0, 0.0, 0.0, 1.0]
    _set_position_drives(
        builder,
        H1_JOINT_POS,
        [H1_GAIN_SCALE * gain for gain in H1_KP],
        [H1_GAIN_SCALE * gain for gain in H1_KD],
    )
    return builder


def build_anymal(walking: bool = False) -> newton.ModelBuilder:
    """ANYmal C standing on its position-controlled legs, or with the gains of its walking policy."""
    builder = _new_builder()
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1)
    asset = newton.utils.download_asset("anybotics_anymal_c")
    builder.add_urdf(
        str(asset / "urdf" / "anymal.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, ANYMAL_ROOT_HEIGHT), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    names = [label.rsplit("/", 1)[-1] for label in builder.joint_label[1:]]
    positions = [ANYMAL_JOINT_POS[name] for name in names]
    stiffness, damping = (WALK_KP, WALK_KD) if walking else (ANYMAL_KP, ANYMAL_KD)
    _set_position_drives(builder, positions, [stiffness] * len(names), [damping] * len(names))
    return builder


def _add_wheel(
    builder: newton.ModelBuilder,
    chassis: int,
    center,
    radius: float,
    half_width: float,
    label: str,
    driven: bool = False,
):
    """Add a wheel on a revolute joint about the chassis' lateral axis, braked or ``driven`` at a target rate."""
    chassis_position = wp.transform_get_translation(builder.body_q[chassis])
    wheel = builder.add_link(xform=wp.transform(chassis_position + wp.vec3(*center), wp.quat_identity()), label=label)
    builder.add_shape_cylinder(
        wheel,
        xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)),
        radius=radius,
        half_height=half_width,
        cfg=_shape_cfg(density=800.0, mu=1.0),
        color=COLORS["steel"],
    )
    builder.add_joint_revolute(
        chassis,
        wheel,
        parent_xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=newton.Axis.Y,
        target_q=0.0,
        target_ke=0.0 if driven else 400.0,
        target_kd=KIVA_DRIVE_KD if driven else 40.0,
        armature=0.01,
        actuator_mode=newton.JointTargetMode.VELOCITY if driven else newton.JointTargetMode.POSITION,
        label=f"{label}_axle",
    )


def build_kiva(driven: bool = False) -> newton.ModelBuilder:
    """Drive unit: a low chassis on two wheels and two passive caster balls.

    The wheels are braked, or ``driven`` by velocity drives that steer the unit.
    """
    builder = _new_builder()
    chassis = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, KIVA_ROOT_HEIGHT), wp.quat_identity()), label="kiva"
    )
    root = builder.add_joint_free(chassis, label="kiva_root")
    builder.add_shape_box(chassis, hx=0.36, hy=0.28, hz=0.07, cfg=_shape_cfg(density=2000.0), color=COLORS["kiva"])
    plate_half_height = 0.03
    builder.add_shape_box(
        chassis,
        xform=wp.transform(wp.vec3(0.0, 0.0, KIVA_TOP - plate_half_height - KIVA_ROOT_HEIGHT), wp.quat_identity()),
        hx=0.25,
        hy=0.25,
        hz=plate_half_height,
        cfg=_shape_cfg(density=1000.0),
        color=COLORS["steel"],
    )
    caster_radius = 0.035
    # Driven wheels need the load for traction: like spring-loaded casters, rigid ones stand
    # clear of the floor, and the unit tips onto one only when it pitches.
    caster_clearance = 0.004 if driven else 0.0
    for x in (-0.30, 0.30):
        builder.add_shape_sphere(
            chassis,
            xform=wp.transform(
                wp.vec3(x, 0.0, caster_radius + caster_clearance - KIVA_ROOT_HEIGHT), wp.quat_identity()
            ),
            radius=caster_radius,
            cfg=_shape_cfg(density=1000.0, mu=0.1),
            color=COLORS["steel"],
        )
    wheel_radius = KIVA_WHEEL_RADIUS
    for side, y in (("left", KIVA_WHEEL_TRACK), ("right", -KIVA_WHEEL_TRACK)):
        _add_wheel(
            builder, chassis, (0.0, y, wheel_radius - KIVA_ROOT_HEIGHT), wheel_radius, 0.025, f"{side}_wheel", driven
        )
    builder.add_articulation(list(range(root, builder.joint_count)), label="kiva")
    return builder


def build_forklift(fork_height: float) -> newton.ModelBuilder:
    """Forklift: a heavy chassis on four wheels with a fork carriage sliding up its mast."""
    builder = _new_builder()
    z0 = FORKLIFT_ROOT_HEIGHT
    chassis = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()), label="forklift")
    root = builder.add_joint_free(chassis, label="forklift_root")
    builder.add_shape_box(chassis, hx=0.55, hy=0.40, hz=0.25, cfg=_shape_cfg(density=1500.0), color=COLORS["forklift"])
    light = _shape_cfg(density=300.0)
    for x in (-0.45, 0.25):  # overhead guard posts
        for y in (-0.35, 0.35):
            builder.add_shape_box(
                chassis,
                xform=wp.transform(wp.vec3(x, y, 0.25 + 0.5), wp.quat_identity()),
                hx=0.03,
                hy=0.03,
                hz=0.5,
                cfg=light,
                color=COLORS["steel"],
            )
    builder.add_shape_box(
        chassis,
        xform=wp.transform(wp.vec3(-0.10, 0.0, 1.27), wp.quat_identity()),
        hx=0.42,
        hy=0.40,
        hz=0.02,
        cfg=light,
        color=COLORS["forklift"],
    )
    for y in (-0.30, 0.30):  # mast uprights
        builder.add_shape_box(
            chassis,
            xform=wp.transform(wp.vec3(0.63, y, 1.0 - z0), wp.quat_identity()),
            hx=0.04,
            hy=0.04,
            hz=0.9,
            cfg=_shape_cfg(density=1000.0),
            color=COLORS["steel"],
        )
    wheel_radius = 0.2
    for name, x, y in (("front_left", 0.38, 0.48), ("front_right", 0.38, -0.48)):
        _add_wheel(builder, chassis, (x, y, wheel_radius - z0), wheel_radius, 0.07, f"{name}_wheel")
    for name, x, y in (("rear_left", -0.38, 0.48), ("rear_right", -0.38, -0.48)):
        _add_wheel(builder, chassis, (x, y, wheel_radius - z0), wheel_radius, 0.07, f"{name}_wheel")
    # The carriage origin lies in the underside of the tines: the joint coordinate is their height.
    carriage = builder.add_link(
        xform=wp.transform(wp.vec3(FORK_REACH, 0.0, fork_height), wp.quat_identity()), label="fork"
    )
    steel = _shape_cfg(density=2000.0)
    builder.add_shape_box(
        carriage,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        hx=0.025,
        hy=0.35,
        hz=0.30,
        cfg=steel,
        color=COLORS["steel"],
    )
    for y in (-0.2, 0.2):
        builder.add_shape_box(
            carriage,
            xform=wp.transform(wp.vec3(0.025 + 0.5 * FORK_LENGTH, y, 0.5 * FORK_THICKNESS), wp.quat_identity()),
            hx=0.5 * FORK_LENGTH,
            hy=0.05,
            hz=0.5 * FORK_THICKNESS,
            cfg=steel,
            color=COLORS["steel"],
        )
    builder.add_joint_prismatic(
        chassis,
        carriage,
        parent_xform=wp.transform(wp.vec3(FORK_REACH, 0.0, -z0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=newton.Axis.Z,
        target_q=fork_height,
        target_ke=4.0e5,
        target_kd=1.0e4,
        limit_lower=0.02,
        limit_upper=1.6,
        actuator_mode=newton.JointTargetMode.POSITION,
        label="fork_lift",
    )
    builder.joint_q[-1] = fork_height
    builder.add_articulation(list(range(root, builder.joint_count)), label="forklift")
    return builder


def _frustum_mesh(half_extents, taper: float = 0.8) -> newton.Mesh:
    """Tote: a box whose bottom is ``taper`` times its top, with a signed distance field."""
    hx, hy, hz = half_extents
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    vertices = [(taper * hx * x, taper * hy * y, -hz) for x, y in corners] + [(hx * x, hy * y, hz) for x, y in corners]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for a in range(4):
        b = (a + 1) % 4
        faces += [(a, b, b + 4), (a, b + 4, a + 4)]
    mesh = newton.Mesh(vertices=np.array(vertices, dtype=np.float32), indices=np.array(faces, dtype=np.int32).ravel())
    mesh.build_sdf(max_resolution=32)
    return mesh


def _ramp_mesh(length: float, half_width: float, height: float) -> newton.Mesh:
    """Loading ramp rising along +x, with a signed distance field."""
    w = half_width
    vertices = [(0, -w, 0), (length, -w, 0), (length, w, 0), (0, w, 0), (length, -w, height), (length, w, height)]
    faces = [(0, 2, 1), (0, 3, 2), (1, 2, 5), (1, 5, 4), (0, 4, 5), (0, 5, 3), (0, 1, 4), (3, 5, 2)]
    mesh = newton.Mesh(vertices=np.array(vertices, dtype=np.float32), indices=np.array(faces, dtype=np.int32).ravel())
    mesh.build_sdf(max_resolution=64)
    return mesh


def _add_box_body(builder, center, half_extents, color, density=200.0, rotation=None) -> int:
    body = builder.add_body(xform=wp.transform(wp.vec3(*center), rotation or wp.quat_identity()))
    hx, hy, hz = half_extents
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=_shape_cfg(density=density), color=color)
    return body


def _add_tote_body(builder, center, mesh: newton.Mesh, rotation=None) -> int:
    body = builder.add_body(xform=wp.transform(wp.vec3(*center), rotation or wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=mesh, cfg=_shape_cfg(), color=COLORS["tote"])
    return body


def build_pod(rng: np.random.Generator, tote: newton.Mesh) -> newton.ModelBuilder:
    """Shelving pod with packages on its three shelves; body 0 is the pod."""
    builder = _new_builder()
    pod = builder.add_body(xform=wp.transform_identity())
    frame = _shape_cfg(density=300.0)
    for x in (-0.46, 0.46):
        for y in (-0.46, 0.46):
            builder.add_shape_box(
                pod,
                xform=wp.transform(wp.vec3(x, y, 0.88), wp.quat_identity()),
                hx=0.02,
                hy=0.02,
                hz=0.88,
                cfg=frame,
                color=COLORS["steel"],
            )
    for top in POD_SHELF_TOPS:
        builder.add_shape_box(
            pod,
            xform=wp.transform(wp.vec3(0.0, 0.0, top - 0.015), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.015,
            cfg=frame,
            color=COLORS["pod"],
        )
    for level, top in enumerate(POD_SHELF_TOPS):
        for x in (-0.23, 0.23):
            for y in (-0.23, 0.23):
                if rng.random() > 0.75:
                    continue
                half = PACKAGE_HALF_EXTENTS[int(rng.integers(len(PACKAGE_HALF_EXTENTS)))]
                shade = float(rng.uniform(0.85, 1.15))
                color = tuple(min(1.0, shade * c) for c in COLORS["package"])
                if level == 2 and rng.random() < 0.3:
                    _add_tote_body(builder, (x, y, top + 0.08 + REST_GAP), tote)
                    continue
                _add_box_body(builder, (x, y, top + half[2] + REST_GAP), half, color)
                if rng.random() < 0.25:  # a small package on top of this one
                    z = top + 2.0 * half[2] + 0.05 + 2.0 * REST_GAP
                    _add_box_body(builder, (x, y, z), (0.10, 0.08, 0.05), color)
    return builder


def build_pallet(layers: int) -> newton.ModelBuilder:
    """Pallet with ``layers`` layers of four cartons; body 0 is the pallet."""
    builder = _new_builder()
    hx, hy, hz = PALLET_HALF_EXTENTS
    _add_box_body(builder, (0.0, 0.0, hz + REST_GAP), PALLET_HALF_EXTENTS, COLORS["pallet"], density=180.0)
    cz = CARTON_HALF_EXTENTS[2]
    for layer in range(layers):
        z = 2.0 * hz + (2 * layer + 1) * cz + (layer + 2) * REST_GAP
        shade = 1.0 - 0.06 * layer
        color = tuple(shade * c for c in COLORS["carton"])
        for x in (-0.5 * hx, 0.5 * hx):
            for y in (-0.5 * hy, 0.5 * hy):
                _add_box_body(builder, (x, y, z), CARTON_HALF_EXTENTS, color, density=150.0)
    return builder


@dataclass
class Scene:
    """Builder of the warehouse and the indices the example reports and checks."""

    builder: newton.ModelBuilder
    counts: dict[str, int]
    #: Root body of every robot, per type.
    robot_roots: dict[str, list[int]] = field(default_factory=dict)
    #: Articulations solved in reduced coordinates: the robots.
    robot_articulations: list[int] = field(default_factory=list)
    #: Bodies that are not part of a robot or a mechanism.
    free_bodies: list[int] = field(default_factory=list)
    #: Shelved packages and stacked cartons, which must stay where they are.
    settled_bodies: list[int] = field(default_factory=list)
    #: Joints of the gates and drawers.
    mechanism_joints: list[int] = field(default_factory=list)
    #: Articulations of the gates and drawers, one joint each.
    mechanism_articulations: list[int] = field(default_factory=list)
    #: Root poses at which humanoids arrive at runtime.
    docks: list[wp.transform] = field(default_factory=list)
    #: Middle [m] of the aisle each walking ANYmal loops in, in the order of their roots.
    walker_aisles: list[float] = field(default_factory=list)
    #: Ends [m] of the straight part of the walkers' loops along the aisles.
    walk_span: tuple[float, float] = (0.0, 0.0)
    #: Sorting lines: where packages appear on every belt, the static belt meshes, and the
    #: ``y`` [m] at which all belts end.
    sort_spawns: list[tuple[float, float, float]] = field(default_factory=list)
    sort_belts: list[newton.Mesh] = field(default_factory=list)
    sort_end: float = 0.0
    #: Middle [m] of the aisle each drive unit shuttles in, in the order of their roots, and the
    #: ends [m] of its shuttle along the aisle.
    kiva_aisles: list[float] = field(default_factory=list)
    kiva_span: tuple[float, float] = (0.0, 0.0)


class _SceneAssembler:
    """Places prototypes into one builder and records what the copies are."""

    def __init__(self, counts: dict[str, int]):
        self.builder = _new_builder()
        self.builder.rigid_gap = CONTACT_GAP
        self.builder.add_ground_plane()
        self.scene = Scene(builder=self.builder, counts=counts)
        self.scene.robot_roots = {name: [] for name in counts}

    def place_robots(self, name: str, prototype: newton.ModelBuilder, xforms: list) -> None:
        if not xforms:
            return
        body_start = self.builder.body_count
        articulation_start = self.builder.articulation_count
        self.builder.replicate(prototype, len(xforms), xforms=xforms, separate_worlds=False)
        self.scene.robot_roots[name] += range(body_start, self.builder.body_count, prototype.body_count)
        self.scene.robot_articulations += range(articulation_start, self.builder.articulation_count)

    def place_bodies(self, prototype: newton.ModelBuilder, xforms: list, settled: bool = True) -> None:
        """Place copies of a prototype of free bodies whose body 0 carries the others."""
        if not xforms:
            return
        body_start = self.builder.body_count
        self.builder.replicate(prototype, len(xforms), xforms=xforms, separate_worlds=False)
        self.scene.free_bodies += range(body_start, self.builder.body_count)
        if settled:
            carried = np.arange(body_start, self.builder.body_count).reshape(len(xforms), prototype.body_count)[:, 1:]
            self.scene.settled_bodies += carried.ravel().tolist()

    def add_free_box(self, center, half_extents, color, rotation=None) -> None:
        self.scene.free_bodies.append(_add_box_body(self.builder, center, half_extents, color, rotation=rotation))

    def add_free_tote(self, center, mesh, rotation=None) -> None:
        self.scene.free_bodies.append(_add_tote_body(self.builder, center, mesh, rotation=rotation))

    def add_gate(self, x: float, y: float) -> None:
        """Gate across an aisle: a panel hinged on one of two posts, pulled shut by a spring."""
        builder = self.builder
        frame = _shape_cfg()
        for side in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, y + side * 1.3, 0.5), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.5,
                cfg=frame,
                color=COLORS["frame"],
            )
        hinge = wp.vec3(x, y + 1.2, 0.55)
        panel = builder.add_link(xform=wp.transform(hinge, wp.quat_identity()), label="gate")
        builder.add_shape_box(
            panel,
            xform=wp.transform(wp.vec3(0.0, -1.2, 0.0), wp.quat_identity()),
            hx=0.02,
            hy=1.15,
            hz=0.35,
            cfg=_shape_cfg(density=300.0),
            color=COLORS["gate"],
        )
        joint = builder.add_joint_revolute(
            -1,
            panel,
            parent_xform=wp.transform(hinge, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=newton.Axis.Z,
            target_q=0.0,
            target_ke=60.0,
            target_kd=60.0,
            limit_lower=-1.6,
            limit_upper=1.6,
            actuator_mode=newton.JointTargetMode.POSITION,
            label="gate_hinge",
        )
        builder.joint_q[builder.joint_q_start[joint]] = 0.5  # starts ajar and swings shut
        builder.add_articulation([joint], label="gate")
        self.scene.mechanism_joints.append(joint)
        self.scene.mechanism_articulations.append(builder.articulation_count - 1)

    def add_drawer_cabinet(self, x: float, y: float) -> None:
        """Static cabinet whose drawer slides on a prismatic joint and holds a package."""
        builder = self.builder
        frame = _shape_cfg()
        parts = (
            ((0.0, 0.0, 0.125), (0.30, 0.40, 0.125)),  # plinth
            ((0.0, 0.0, 0.26), (0.30, 0.40, 0.01)),
            ((0.0, 0.0, 0.50), (0.30, 0.40, 0.01)),
            ((0.0, -0.39, 0.38), (0.30, 0.01, 0.11)),
            ((0.0, 0.39, 0.38), (0.30, 0.01, 0.11)),
            ((0.29, 0.0, 0.38), (0.01, 0.38, 0.11)),
        )
        for center, (hx, hy, hz) in parts:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x + center[0], y + center[1], center[2]), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=frame,
                color=COLORS["frame"],
            )
        opening = 0.25
        closed = wp.vec3(x, y, 0.38)
        drawer = builder.add_link(
            xform=wp.transform(closed - wp.vec3(opening, 0.0, 0.0), wp.quat_identity()), label="drawer"
        )
        tray = (
            ((0.0, 0.0, -0.08), (0.27, 0.36, 0.01)),
            ((-0.26, 0.0, 0.0), (0.01, 0.36, 0.09)),
            ((0.26, 0.0, 0.0), (0.01, 0.36, 0.09)),
            ((0.0, -0.35, 0.0), (0.27, 0.01, 0.09)),
            ((0.0, 0.35, 0.0), (0.27, 0.01, 0.09)),
        )
        for center, (hx, hy, hz) in tray:
            builder.add_shape_box(
                drawer,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=_shape_cfg(density=400.0),
                color=COLORS["pod"],
            )
        joint = builder.add_joint_prismatic(
            -1,
            drawer,
            parent_xform=wp.transform(closed, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=newton.Axis.X,
            target_q=-opening,
            target_ke=300.0,
            target_kd=60.0,
            limit_lower=-0.4,
            limit_upper=0.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            label="drawer_slide",
        )
        builder.joint_q[builder.joint_q_start[joint]] = -opening
        builder.add_articulation([joint], label="drawer")
        self.scene.mechanism_joints.append(joint)
        self.scene.mechanism_articulations.append(builder.articulation_count - 1)
        self.add_free_box((x - opening, y, 0.31 + 0.05 + REST_GAP), (0.12, 0.14, 0.05), COLORS["package"])

    def add_ramp(self, x: float, y: float, ramp: newton.Mesh, slope: float, tote: newton.Mesh) -> None:
        """Static mesh ramp with a package and a tote resting on its slope."""
        self.builder.add_shape_mesh(
            -1,
            xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity()),
            mesh=ramp,
            cfg=_shape_cfg(),
            color=COLORS["frame"],
        )
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -slope)
        normal = np.array([-math.sin(slope), 0.0, math.cos(slope)])
        for along, lateral, height, is_tote in ((0.8, -0.3, 0.08, True), (1.3, 0.3, 0.07, False)):
            center = np.array([x + along, y + lateral, along * math.tan(slope)]) + (height + REST_GAP) * normal
            if is_tote:
                self.add_free_tote(center, tote, rotation=tilt)
            else:
                self.add_free_box(center, (0.12, 0.16, height), COLORS["package"], rotation=tilt)


def _belt_mesh(center, half_extents) -> newton.Mesh:
    """Static box mesh whose surface can move, as a conveyor belt."""
    box = newton.Mesh.create_box(
        *half_extents, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
    )
    return newton.Mesh(
        np.asarray(box.vertices) + np.asarray(center, dtype=np.float32),
        np.asarray(box.indices),
        compute_inertia=False,
        enable_surface_velocity=True,
    )


def build_sorting_hall(humanoids: int, robot_meshes: bool = False) -> Scene:
    """Belts with a line of G1 humanoids on each side, the far line offset by half a pitch.

    A right hand is about 0.2 m beside its root, so with the lines half a pitch (0.45 m) apart
    along the belt, a humanoid's right hand sweeps about 0.05 m beside the right hand of the
    humanoid opposite.
    """
    per_belt = 2 * SORT_ROBOTS_PER_LINE
    belts = math.ceil(humanoids / per_belt)
    assembler = _SceneAssembler({"g1": humanoids, "kiva": belts * SORT_KIVAS_PER_AISLE})
    length = (SORT_ROBOTS_PER_LINE + 0.5) * SORT_ROBOT_PITCH
    scene = assembler.scene
    scene.sort_end = length
    width = SORT_BELT_FAR - SORT_BELT_NEAR
    xforms = []
    for belt_index in range(belts):
        x = belt_index * SORT_LINE_PITCH
        for k in range(min(per_belt, humanoids - belt_index * per_belt)):
            side, slot = divmod(k, SORT_ROBOTS_PER_LINE)
            if side == 0:
                xforms.append(_position(x, (slot + 0.5) * SORT_ROBOT_PITCH))
            else:
                xforms.append(_position(x + SORT_FACING, slot * SORT_ROBOT_PITCH, yaw=math.pi))
        start = -1.0
        belt = _belt_mesh(
            (x + 0.5 * (SORT_BELT_NEAR + SORT_BELT_FAR), 0.5 * (start + length), 0.5 * SORT_BELT_TOP),
            (0.5 * width, 0.5 * (length - start), 0.5 * SORT_BELT_TOP),
        )
        assembler.builder.add_shape_mesh(-1, mesh=belt, cfg=_shape_cfg(), color=COLORS["frame"], label="belt")
        scene.sort_belts.append(belt)
        # Packages appear at the belt's start, in its middle, where the hands of both sides reach.
        scene.sort_spawns.append((x + 0.5 * SORT_FACING, start + 0.3, SORT_BELT_TOP))
    assembler.place_robots("g1", build_g1(robot_meshes), xforms)
    # Drive units in the aisle beyond every belt's far line, alternately heading up and down it.
    aisle_offset = 0.5 * (SORT_FACING + SORT_LINE_PITCH)
    scene.kiva_span = (0.8, length - 0.8)
    kiva_xforms = []
    tote = _frustum_mesh((0.16, 0.12, 0.08))
    for belt_index in range(belts):
        aisle = belt_index * SORT_LINE_PITCH + aisle_offset
        for unit in range(SORT_KIVAS_PER_AISLE):
            heading = 1.0 if unit % 2 == 0 else -1.0
            y = scene.kiva_span[0] + (unit + 0.5) / SORT_KIVAS_PER_AISLE * (scene.kiva_span[1] - scene.kiva_span[0])
            x = aisle
            kiva_xforms.append(_position(x, y, yaw=heading * 0.5 * math.pi))
            scene.kiva_aisles.append(aisle)
            assembler.add_free_tote((x, y, KIVA_TOP + 0.08 + REST_GAP), tote)
    assembler.place_robots("kiva", build_kiva(driven=True), kiva_xforms)
    capacity = belts * math.ceil((length + 2.0) / SORT_SPAWN_CLEARANCE)
    assembler.builder.reserve_dynamic_capacity(bodies=capacity, shapes=capacity)
    for shape in range(assembler.builder.shape_count):
        assembler.builder.shape_gap[shape] = CONTACT_GAP
    return scene


def build_scene(
    robots: int = 128,
    storage_scale: float = 1.0,
    seed: int = 0,
    robot_meshes: bool = False,
    runtime_robots: int = 0,
    walking_anymals: int = 0,
) -> Scene:
    """Build the warehouse: robots in aisles between rows of pods and pallets.

    ``robot_meshes`` shows the humanoids' visual meshes; they never collide. ``runtime_robots``
    reserves that many G1 humanoids, which arrive at a dock beside the aisles at runtime.
    ``walking_anymals`` replaces the standing fleet by that many ANYmals in two lanes per aisle.
    """
    if robots < 1:
        raise ValueError("The warehouse needs at least one robot")
    if storage_scale < 0.0:
        raise ValueError("storage_scale must not be negative")
    rng = np.random.default_rng(seed)
    counts = {"anymal": walking_anymals} if walking_anymals > 0 else _fleet_counts(robots)
    assembler = _SceneAssembler(counts)

    g1, h1 = build_g1(robot_meshes), build_h1(robot_meshes)
    g1_pose, h1_pose = _standing_poses(g1), _standing_poses(h1)
    tote = _frustum_mesh((0.16, 0.12, 0.08))

    # Interleave the types along the aisles, then turn some neighbors of one type into pairs
    # that stand side by side in one slot: "shared" pairs carry one long package on their feet,
    # "touching" pairs stand hand to hand.
    order = sorted(
        (float(k + 0.5) / count, rank, name) for rank, (name, count) in enumerate(counts.items()) for k in range(count)
    )
    units: list[tuple[str, str]] = []
    seen = dict.fromkeys(counts, 0)
    partners: set[tuple[str, int]] = set()
    for _, _, name in order:
        index = seen[name]
        seen[name] += 1
        if (name, index) in partners:
            continue  # stands in its partner's slot
        kind = "single"
        if index + 1 < counts[name]:
            if name == "g1" and index % 12 == 2:
                kind = "shared"
            elif (name == "g1" and index % 12 == 8) or (name == "h1" and index % 10 == 3):
                kind = "touching"
        if kind != "single":
            partners.add((name, index + 1))
        units.append((name, kind))

    depth = max(1, math.ceil(storage_scale))
    aisle_pitch = AISLE_WIDTH + depth * STORAGE_DEPTH
    lanes = 2 if walking_anymals > 0 else 1  # per aisle
    columns = max(1, math.ceil(math.sqrt(len(units) * aisle_pitch / (lanes * SLOT_PITCH))))
    aisles = math.ceil(len(units) / (lanes * columns))
    length = columns * SLOT_PITCH
    assembler.scene.walk_span = (0.5 * SLOT_PITCH, length - 0.5 * SLOT_PITCH)

    # Lateral reach of a humanoid's hand collider, for pairs standing hand to hand.
    g1_hand = g1_pose[_body_named(g1, "left_wrist_yaw_link"), 1] + 0.045
    h1_hand = h1_pose[_body_named(h1, "left_elbow_link"), 1] + 0.06
    g1_foot = g1_pose[_body_named(g1, "left_ankle_roll_link"), :3] + np.array(G1_FOOT_CENTER)
    h1_foot = h1_pose[_body_named(h1, "left_ankle_link"), :3] + np.array(H1_FOOT_CENTER)
    shared_distance = 0.9

    xforms: dict[str, list] = {name: [] for name in ("g1", "h1", "anymal", "kiva", "forklift_empty", "forklift_loaded")}
    carried_pods, carried_pallets, kiva_cartons = [], [], []
    humanoids = 0
    for slot, (name, kind) in enumerate(units):
        x = (slot % columns + 0.5) * SLOT_PITCH
        y = (slot // columns) * aisle_pitch
        if walking_anymals > 0:
            # The lane right of the aisle's middle runs along +x, the left one back.
            aisle, back = divmod(slot // columns, 2)
            y = aisle * aisle_pitch
            xforms[name].append(_position(x, y + (2.0 * back - 1.0) * LANE_OFFSET, yaw=back * math.pi))
            assembler.scene.walker_aisles.append(y)
            if back:  # loose stock between the lanes
                if slot % 2:
                    assembler.add_free_tote((x, y, 0.08 + REST_GAP), tote)
                else:
                    assembler.add_free_box((x, y, 0.06 + REST_GAP), (0.15, 0.12, 0.06), COLORS["package"])
        elif name in ("g1", "h1"):
            foot, hand = (g1_foot, g1_hand) if name == "g1" else (h1_foot, h1_hand)
            foot_half = G1_FOOT_HALF_EXTENTS if name == "g1" else H1_FOOT_HALF_EXTENTS
            foot_top = foot[2] + foot_half[2]
            toe = foot[0] + foot_half[0]
            if kind == "single":
                offsets = (0.0,)
            elif kind == "shared":
                offsets = (-0.5 * shared_distance, 0.5 * shared_distance)
            else:
                offsets = (-hand + 0.5 * HAND_OVERLAP, hand - 0.5 * HAND_OVERLAP)
            for offset in offsets:
                xforms[name].append(_position(x, y + offset))
                # A flat package with one end on a toe and the other on the floor.
                side = 1.0 if humanoids % 2 == 0 else -1.0
                humanoids += 1
                if kind == "shared":
                    continue
                half = (0.09, 0.07, 0.03) if name == "g1" else (0.14, 0.08, 0.03)
                center = (x + toe - 0.04 + half[0], y + offset + side * foot[1], foot_top + half[2] + 0.004)
                assembler.add_free_box(center, half, COLORS["package"])
            if kind == "shared":
                # One long package across the toes of both robots.
                half = (0.05, 0.5 * shared_distance + foot[1] + 0.06, 0.03)
                assembler.add_free_box((x + toe - half[0], y, foot_top + half[2] + 0.003), half, COLORS["carton"])
        elif name == "anymal":
            xforms[name].append(_position(x, y))
            if len(xforms[name]) % 2:
                assembler.add_free_tote((x + 0.75, y + 0.2, 0.08 + REST_GAP), tote)
            else:  # a package dropped onto its back
                assembler.add_free_box((x, y, ANYMAL_ROOT_HEIGHT + 0.30), (0.15, 0.12, 0.06), COLORS["package"])
        elif name == "kiva":
            xforms[name].append(_position(x, y))
            if len(xforms[name]) % 2:
                carried_pods.append(_position(x, y, KIVA_TOP - POD_UNDERSIDE + REST_GAP))
            else:
                kiva_cartons.append((x, y, KIVA_TOP + CARTON_HALF_EXTENTS[2] + REST_GAP))
        else:
            loaded = (len(xforms["forklift_empty"]) + len(xforms["forklift_loaded"])) % 2 == 0
            rear = x - 0.6  # the forks reach far ahead of the chassis: center the whole truck
            xforms["forklift_loaded" if loaded else "forklift_empty"].append(_position(rear, y))
            fork_center = rear + FORK_REACH + 0.045 + 0.5 * FORK_LENGTH
            if loaded:
                carried_pallets.append(_position(fork_center, y, FORK_LOADED_HEIGHT + FORK_THICKNESS))
            else:
                assembler.add_free_box(
                    (fork_center + 0.9, y - 0.5, 0.09 + REST_GAP), (0.16, 0.12, 0.09), COLORS["package"]
                )

    assembler.place_robots("g1", g1, xforms["g1"])
    assembler.place_robots("h1", h1, xforms["h1"])
    if xforms["anymal"]:
        assembler.place_robots("anymal", build_anymal(walking=walking_anymals > 0), xforms["anymal"])
    assembler.place_robots("kiva", build_kiva(), xforms["kiva"])
    assembler.place_robots("forklift", build_forklift(FORK_EMPTY_HEIGHT), xforms["forklift_empty"])
    assembler.place_robots("forklift", build_forklift(FORK_LOADED_HEIGHT), xforms["forklift_loaded"])
    for center in kiva_cartons:
        assembler.add_free_box(center, CARTON_HALF_EXTENTS, COLORS["carton"])

    # Storage rows on both sides of every aisle: mostly pods, every fifth place a pallet.
    pod_variants = [build_pod(rng, tote) for _ in range(8)]
    pallet_variants = [build_pallet(layers) for layers in (2, 3, 4)]
    pod_xforms: list[list] = [[] for _ in pod_variants]
    pallet_xforms: list[list] = [[] for _ in pallet_variants]
    for index, xform in enumerate(carried_pods):
        pod_xforms[index % len(pod_variants)].append(xform)
    pallet_xforms[0] += carried_pallets
    places = max(1, int(length / STORAGE_PITCH))
    fill = storage_scale / depth
    for row in range(aisles + 1):
        for line in range(depth):
            y = (row - 0.5) * aisle_pitch + (line - 0.5 * (depth - 1)) * STORAGE_DEPTH
            for place in range(places):
                if rng.random() >= fill:
                    continue
                x = (place + 0.5) * length / places
                if place % 5 == 4:
                    pallet_xforms[int(rng.integers(len(pallet_variants)))].append(_position(x, y, yaw=0.5 * math.pi))
                else:
                    pod_xforms[int(rng.integers(len(pod_variants)))].append(_position(x, y))
    for prototype, placements in zip(pod_variants, pod_xforms, strict=True):
        assembler.place_bodies(prototype, placements)
    for prototype, placements in zip(pallet_variants, pallet_xforms, strict=True):
        assembler.place_bodies(prototype, placements)

    # Mechanisms and the mesh ramps close the aisles at both ends.
    slope = math.atan2(0.3, 2.0)
    ramp = _ramp_mesh(2.0, 0.6, 0.3)
    for aisle in range(aisles):
        y = aisle * aisle_pitch
        assembler.add_gate(-1.5, y)
        if aisle % 2 == 0:
            assembler.add_drawer_cabinet(length + 1.5, y)
        else:
            assembler.add_ramp(length + 0.8, y, ramp, slope, tote)

    builder = assembler.builder
    for shape in range(builder.shape_count):
        builder.shape_gap[shape] = CONTACT_GAP  # imported sites and colliders keep their own otherwise
    if runtime_robots > 0:
        # Instances take their poses from the template's bodies: put them where its joints do.
        g1.body_q[:] = [wp.transform(*pose) for pose in g1_pose]
        g1.shape_gap[:] = [CONTACT_GAP] * g1.shape_count
        builder.reserve_dynamic_capacity(
            shapes=runtime_robots * g1.shape_count,
            articulations=[runtime_robots],
            articulation_templates=[g1],
            articulation_coordinates="reduced",
        )
        assembler.scene.docks = [
            wp.transform(
                wp.vec3(-DOCK_DISTANCE, -DOCK_DISTANCE - DOCK_PITCH * dock, G1_ROOT_HEIGHT), wp.quat_identity()
            )
            for dock in range(runtime_robots)
        ]
    return assembler.scene


@wp.kernel(enable_backward=False)
def _cycle_dock_kernel(
    dyn: phoenx.DynamicsData,
    docks: wp.array[wp.transform],
    period: int,
    frame: wp.array[int],
    occupant: wp.array[int],
):
    """Every ``period`` frames the next dock's humanoid leaves, or one arrives at the empty dock."""
    current = frame[0]
    frame[0] = current + 1
    if current % period != 0:
        return
    dock = (current // period) % docks.shape[0]
    if occupant[dock] >= 0:
        phoenx.remove_articulation(dyn, occupant[dock])
        occupant[dock] = -1
    else:
        occupant[dock] = phoenx.add_articulation(dyn, 0, docks[dock], wp.spatial_vector(), 0)


@wp.kernel(enable_backward=False)
def _walker_observation_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    aisle: wp.array[float],
    span: wp.vec2,
    default_q: wp.array[float],
    policy_joint: wp.array[int],
    dt: float,
    previous_action: wp.array2d[float],
    observation: wp.array2d[float],
    progress: wp.array[float],
):
    """Steer every walker around its aisle's loop and write its policy observation."""
    walker = wp.tid()
    a = q_start[walker]
    b = qd_start[walker]
    q = wp.quat(joint_q[a + 3], joint_q[a + 4], joint_q[a + 5], joint_q[a + 6])
    velocity = wp.vec3(joint_qd[b], joint_qd[b + 1], joint_qd[b + 2])

    # The loop is a stadium, walked counterclockwise: all points at the lane offset from the
    # segment between the span's ends in the aisle's middle.
    x = joint_q[a]
    outward = wp.vec2(x - wp.clamp(x, span[0], span[1]), joint_q[a + 1] - aisle[walker])
    distance = wp.length(outward)
    outward = wp.where(distance > 1.0e-3, outward / distance, wp.vec2(0.0, -1.0))
    along = wp.vec2(-outward[1], outward[0])
    inside = LANE_OFFSET - distance  # [m] by which the walker is inside the loop
    goal = along + inside * outward  # heads back to the loop over about a meter
    forward = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    heading_error = wp.atan2(forward[0] * goal[1] - forward[1] * goal[0], forward[0] * goal[0] + forward[1] * goal[1])
    command = wp.vec3(
        wp.lerp(WALK_SPEED, WALK_TURN_SPEED, wp.min(wp.abs(heading_error), 1.0)),
        wp.clamp(inside * (forward[0] * outward[1] - forward[1] * outward[0]), -0.3, 0.3),
        wp.clamp(2.0 * heading_error, -WALK_YAW_RATE_MAX, WALK_YAW_RATE_MAX),
    )
    progress[walker] += dt * (velocity[0] * along[0] + velocity[1] * along[1])

    linear = wp.quat_rotate_inv(q, velocity)
    angular = wp.quat_rotate_inv(q, wp.vec3(joint_qd[b + 3], joint_qd[b + 4], joint_qd[b + 5]))
    gravity = wp.quat_rotate_inv(q, wp.vec3(0.0, 0.0, -1.0))
    for k in range(3):
        observation[walker, k] = linear[k]
        observation[walker, 3 + k] = angular[k]
        observation[walker, 6 + k] = gravity[k]
        observation[walker, 9 + k] = command[k]
    for k in range(12):
        joint = policy_joint[k]
        observation[walker, 12 + k] = joint_q[a + 7 + joint] - default_q[joint]
        observation[walker, 24 + k] = joint_qd[b + 6 + joint]
        observation[walker, 36 + k] = previous_action[walker, k]


@wp.kernel(enable_backward=False)
def _walker_target_kernel(
    action: wp.array2d[float],
    q_start: wp.array[int],
    default_q: wp.array[float],
    policy_index: wp.array[int],
    joint_target_q: wp.array[float],
    previous_action: wp.array2d[float],
):
    """Turn the policy's actions into the joint position targets of the walkers' drives."""
    walker, joint = wp.tid()
    joint_target_q[q_start[walker] + 7 + joint] = (
        default_q[joint] + WALK_ACTION_SCALE * action[walker, policy_index[joint]]
    )
    previous_action[walker, joint] = action[walker, joint]


@wp.func
def _sort_push(phase: float) -> float:
    """Progress [0, 1] of a hand's forward push over one period: out, back, and a pause."""
    s = wp.where(phase < 0.3, phase / 0.3, wp.where(phase < 0.6, (0.6 - phase) / 0.3, 0.0))
    return s * s * (3.0 - 2.0 * s)


@wp.kernel(enable_backward=False)
def _sorter_target_kernel(
    q_start: wp.array[int],
    shoulder: int,
    elbow: int,
    sweep: wp.array[wp.vec2],
    frame: wp.array[int],
    frame_dt: float,
    joint_target_q: wp.array[float],
):
    """Drive every sorter's right arm through its push, each humanoid at its own phase."""
    sorter = wp.tid()
    time = float(frame[0]) * frame_dt
    spread = 0.754878 * float(sorter)
    period = SORT_PERIOD * (1.0 + SORT_PERIOD_SPREAD * (spread - wp.floor(spread)))
    phase = time / period + 0.618034 * float(sorter)
    along = _sort_push(phase - wp.floor(phase)) * float(sweep.shape[0] - 1)
    k = wp.min(int(along), sweep.shape[0] - 2)
    target = wp.lerp(sweep[k], sweep[k + 1], along - float(k))
    joint_target_q[q_start[sorter] + shoulder] = target[0]
    joint_target_q[q_start[sorter] + elbow] = target[1]


@wp.kernel(enable_backward=False)
def _kiva_drive_kernel(
    body_q: wp.array[wp.transform],
    roots: wp.array[int],
    qd_start: wp.array[int],
    aisle: wp.array[float],
    span: wp.vec2,
    heading: wp.array[float],
    joint_target_qd: wp.array[float],
):
    """Steer every drive unit along the middle of its aisle, turning around at the shuttle's ends."""
    unit = wp.tid()
    pose = body_q[roots[unit]]
    p = wp.transform_get_translation(pose)
    forward = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3(1.0, 0.0, 0.0))
    direction = heading[unit]
    if direction > 0.0 and p[1] > span[1]:
        direction = -1.0
    elif direction < 0.0 and p[1] < span[0]:
        direction = 1.0
    heading[unit] = direction
    # Head along the aisle, steering back to its middle.
    goal = wp.vec2(wp.clamp(2.0 * (aisle[unit] - p[0]), -0.6, 0.6), direction)
    error = wp.atan2(forward[0] * goal[1] - forward[1] * goal[0], forward[0] * goal[0] + forward[1] * goal[1])
    spread = 0.618034 * float(unit)
    speed = SORT_KIVA_SPEED * (1.0 - 0.3 * (spread - wp.floor(spread)))
    forward_speed = speed * wp.max(wp.cos(error), 0.0) ** 2.0
    turn_rate = wp.clamp(2.5 * error, -1.5, 1.5)
    start = qd_start[unit]
    joint_target_qd[start + 6] = (forward_speed - turn_rate * KIVA_WHEEL_TRACK) / KIVA_WHEEL_RADIUS
    joint_target_qd[start + 7] = (forward_speed + turn_rate * KIVA_WHEEL_TRACK) / KIVA_WHEEL_RADIUS


@wp.kernel(enable_backward=False)
def _sort_spawn_kernel(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    spawns: wp.array[wp.vec3],
    spawn_body: wp.array[int],
    spawned: wp.array[int],
):
    """Put a package on every belt whose previous package has moved clear of the start."""
    line = wp.tid()
    origin = spawns[line]
    last = spawn_body[line]
    if last >= 0 and phoenx.is_body_active(dyn, last):
        if wp.length(wp.transform_get_translation(body_q[last]) - origin) < SORT_SPAWN_CLEARANCE:
            return
    h = SORT_PACKAGE_HALF_EXTENTS
    mass = SORT_PACKAGE_DENSITY * 8.0 * h[0] * h[1] * h[2]
    desc = phoenx.body_desc(
        wp.transform(origin + wp.vec3(0.0, 0.0, h[2] + REST_GAP), wp.quat_identity()),
        mass,
        wp.diag(mass / 3.0 * wp.vec3(h[1] * h[1] + h[2] * h[2], h[0] * h[0] + h[2] * h[2], h[0] * h[0] + h[1] * h[1])),
    )
    desc.velocity = wp.spatial_vector(wp.vec3(0.0, SORT_BELT_SPEED, 0.0), wp.vec3(0.0))
    body = phoenx.add_body(dyn, desc)
    if body < 0:
        return
    shape = phoenx.shape_desc(int(newton.GeoType.BOX), h)
    shape.gap = CONTACT_GAP
    shape.mu = FRICTION
    phoenx.add_shape(dyn, body, shape)
    spawn_body[line] = body
    wp.atomic_add(spawned, 0, 1)


@wp.kernel(enable_backward=False)
def _sort_retire_kernel(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    body_start: int,
    belt_end: float,
    frame: wp.array[int],
    retired: wp.array[int],
):
    """Remove packages that left their belt: pushed off its side (sorted) or off its end."""
    slot = wp.tid()
    if slot == 0:
        frame[0] = frame[0] + 1
    body = body_start + slot
    if not phoenx.is_body_active(dyn, body):
        return
    p = wp.transform_get_translation(body_q[body])
    if p[2] > SORT_DROP_HEIGHT:
        return
    if phoenx.remove_body(dyn, body):
        wp.atomic_add(retired, wp.where(p[1] < belt_end, 0, 1), 1)


@wp.kernel(enable_backward=False)
def _sort_instances_kernel(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    body_start: int,
    xforms: wp.array[wp.transform],
):
    """Poses of the live packages; free rows are parked out of view."""
    slot = wp.tid()
    body = body_start + slot
    xforms[slot] = wp.transform(wp.vec3(0.0, 0.0, -1000.0), wp.quat_identity())
    if phoenx.is_body_active(dyn, body):
        xforms[slot] = body_q[body]


class Example:
    """One warehouse world with reduced-coordinate robots among sleeping free bodies."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        walkers = int(getattr(args, "walking_anymals", 0))
        sorters = int(getattr(args, "sorting_humanoids", 0))
        self.frame_dt = 1.0 / WALK_FPS if walkers > 0 else 1.0 / 60.0
        self.sim_time = 0.0
        self.timing_frames = int(getattr(args, "timing_frames", 0))
        self.timing_warmup = int(getattr(args, "timing_warmup", 120))
        robots = int(getattr(args, "robots", 128))
        print(f"[PhoenX Warehouse] building {walkers or sorters or robots} robots...", flush=True)
        robot_meshes = getattr(args, "robot_meshes", None)
        if robot_meshes is None:
            # Meshes are for looking at: they only add shapes to a headless run.
            robot_meshes = not isinstance(viewer, newton.viewer.ViewerNull)
        if sorters > 0:
            self.scene = build_sorting_hall(sorters, robot_meshes=bool(robot_meshes))
        else:
            self.scene = build_scene(
                robots=robots,
                storage_scale=float(getattr(args, "storage_scale", 1.0)),
                seed=int(getattr(args, "seed", 0)),
                robot_meshes=bool(robot_meshes),
                runtime_robots=int(getattr(args, "runtime_robots", 0)),
                walking_anymals=walkers,
            )
        self.model = self.scene.builder.finalize()
        for belt in self.scene.sort_belts:
            belt.mesh.velocities.fill_(wp.vec3(0.0, SORT_BELT_SPEED, 0.0))
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control()
        # The solver builds the collision pipeline: it picks the broad phase for one large
        # world and filters pairs of sleeping bodies. Gates and drawers ride along with the
        # robots in reduced coordinates: as regular joints they need solves of their own between
        # the contact sweeps (4.5 against 4.0 ms per step at 1024 robots).
        reduced = list(self.scene.robot_articulations)
        if getattr(args, "mechanism_solver", "reduced") == "reduced":
            reduced += self.scene.mechanism_articulations
        self.solver = SolverPhoenX(
            self.model,
            joint_mode="reduced",
            reduced_articulations=reduced,
            sleeping_velocity_threshold=0.05,
            substeps=int(getattr(args, "substeps", 4)),
            solver_iterations=int(getattr(args, "iterations", 4)),
            velocity_iterations=1,
            # Most of the stock rests and sleeps: at its peak, while everything settles at once,
            # the scene has about two contacts per shape. The automatic capacity budgets a
            # dense pile around every shape, and each kernel that walks the contact buffer
            # pays for its length.
            rigid_contact_max=8 * int(self.model.shape_count),
        )
        self.contacts = self.model._collision_pipeline.contacts()
        self.docks = wp.array(self.scene.docks, dtype=wp.transform) if self.scene.docks else None
        self.dock_frame = wp.zeros(1, dtype=int)
        self.dock_occupant = wp.full(max(1, len(self.scene.docks)), -1, dtype=int)
        self.initial_poses = self.state.body_q.numpy().copy()
        self.free_bodies = np.asarray(self.scene.free_bodies, dtype=np.int64)
        self.settled_bodies = np.asarray(self.scene.settled_bodies, dtype=np.int64)
        self.policy = None
        if walkers > 0:
            self._init_walkers()
        self.sorting = sorters > 0
        if self.sorting:
            self._init_sorters()

        viewer.set_model(self.model)
        # Look down the first aisles from behind their gates.
        viewer.set_camera(pos=wp.vec3(-4.5, 2.6, 3.4), pitch=-17.0, yaw=12.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        print(f"[PhoenX Warehouse] {self.describe()}", flush=True)

    def _init_walkers(self):
        """Load the walking policy and the buffers one evaluation for all walkers needs."""
        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415

        from newton.examples.robot.example_robot_anymal_c_walk import lab_to_mujoco, mujoco_to_lab  # noqa: PLC0415

        model = self.model
        roots = self.scene.robot_roots["anymal"]
        joint_of_body = np.zeros(model.body_count, dtype=np.int64)
        joint_of_body[model.joint_child.numpy()] = np.arange(model.joint_count)
        q_start = model.joint_q_start.numpy()[joint_of_body[roots]]
        self.walk_q_start = wp.array(q_start, dtype=int)
        self.walk_qd_start = wp.array(model.joint_qd_start.numpy()[joint_of_body[roots]], dtype=int)
        self.walk_aisle = wp.array(self.scene.walker_aisles, dtype=float)
        self.walk_default_q = wp.array(model.joint_q.numpy()[q_start[0] + 7 : q_start[0] + 19], dtype=float)
        self.walk_policy_joint = wp.array(lab_to_mujoco, dtype=int)
        self.walk_policy_index = wp.array(mujoco_to_lab, dtype=int)
        self.walk_observation = wp.zeros((len(roots), 48), dtype=float)
        self.walk_action = wp.zeros((len(roots), 12), dtype=float)
        #: Distance [m] every walker has covered along its loop.
        self.walk_progress = wp.zeros(len(roots), dtype=float)
        asset = newton.utils.download_asset("anybotics_anymal_c")
        self.policy = OnnxRuntime(str(asset / "rl_policies" / "anymal_walking_policy_physx.onnx"), device=model.device)
        self.policy_input = self.policy.inputs[0].name
        self.policy_output = self.policy.outputs[0].name
        # The runtime allocates its buffers on the first call, which must not be a captured one.
        self.policy({self.policy_input: self.walk_observation})

    def _init_sorters(self):
        """Index the sorters' arm drives and allocate the package bookkeeping."""
        model = self.model
        roots = self.scene.robot_roots["g1"]
        joint_of_body = np.zeros(model.body_count, dtype=np.int64)
        joint_of_body[model.joint_child.numpy()] = np.arange(model.joint_count)
        q_start = model.joint_q_start.numpy()
        root_joint = int(joint_of_body[roots[0]])
        labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]

        def offset(name: str) -> int:
            joint = next(j for j in range(root_joint, model.joint_count) if labels[j] == name)
            return int(q_start[joint] - q_start[root_joint])

        self.sort_shoulder = offset("right_shoulder_pitch_joint")
        self.sort_elbow = offset("right_elbow_joint")
        self.sort_q_start = wp.array(q_start[joint_of_body[roots]], dtype=int)
        self.sort_sweep = wp.array(SORT_ARM_SWEEP, dtype=wp.vec2)
        self.sort_spawns = wp.array(self.scene.sort_spawns, dtype=wp.vec3)
        self.sort_spawn_body = wp.full(len(self.scene.sort_spawns), -1, dtype=int)
        self.sort_body_start = int(model._dynamic_capacity.body_start)
        self.sort_capacity = model.body_count - self.sort_body_start
        self.sort_frame = wp.zeros(1, dtype=int)
        #: Packages put on the belts.
        self.sort_spawned = wp.zeros(1, dtype=int)
        #: Packages removed after the humanoids pushed them off a belt's side, and off its end.
        self.sort_retired = wp.zeros(2, dtype=int)
        self.sort_xforms = wp.zeros(self.sort_capacity, dtype=wp.transform)
        # Which robot each body belongs to, to count the contacts between robots.
        articulation_start = model.articulation_start.numpy()
        joint_child = model.joint_child.numpy()
        self.body_robot = np.full(model.body_count, -1, dtype=np.int64)
        for robot, articulation in enumerate(self.scene.robot_articulations):
            joints = slice(articulation_start[articulation], articulation_start[articulation + 1])
            self.body_robot[joint_child[joints]] = robot
        #: Steps in which two robots touched, counted by the test.
        self.robot_touch_steps = 0
        kivas = self.scene.robot_roots["kiva"]
        self.kiva_roots = wp.array(kivas, dtype=int)
        self.kiva_qd_start = wp.array(model.joint_qd_start.numpy()[joint_of_body[kivas]], dtype=int)
        self.kiva_aisle = wp.array(self.scene.kiva_aisles, dtype=float)
        poses = self.state.body_q.numpy()[kivas]
        # Units start heading along +y (yaw +90 deg) or -y.
        self.kiva_heading = wp.array(np.sign(poses[:, 5] * poses[:, 6]).astype(np.float32), dtype=float)
        #: Where every drive unit started, for the test.
        self.kiva_start = poses[:, :2].copy()

    def _sort(self):
        """Spawn packages and set the arm targets for this frame."""
        dyn = self.solver.dynamics.data
        wp.launch(
            _sort_spawn_kernel,
            dim=self.sort_spawn_body.shape[0],
            inputs=[dyn, self.state.body_q, self.sort_spawns, self.sort_spawn_body, self.sort_spawned],
        )
        wp.launch(
            _kiva_drive_kernel,
            dim=self.kiva_roots.shape[0],
            inputs=[
                self.state.body_q,
                self.kiva_roots,
                self.kiva_qd_start,
                self.kiva_aisle,
                wp.vec2(*self.scene.kiva_span),
                self.kiva_heading,
                self.control.joint_target_qd,
            ],
        )
        wp.launch(
            _sorter_target_kernel,
            dim=self.sort_q_start.shape[0],
            inputs=[
                self.sort_q_start,
                self.sort_shoulder,
                self.sort_elbow,
                self.sort_sweep,
                self.sort_frame,
                self.frame_dt,
                self.control.joint_target_q,
            ],
        )

    def _retire_packages(self):
        wp.launch(
            _sort_retire_kernel,
            dim=self.sort_capacity,
            inputs=[
                self.solver.dynamics.data,
                self.state.body_q,
                self.sort_body_start,
                float(self.scene.sort_end),
                self.sort_frame,
                self.sort_retired,
            ],
        )

    def _steer_walkers(self):
        """Evaluate the policy once for all walkers and set their joint targets."""
        wp.launch(
            _walker_observation_kernel,
            dim=self.walk_progress.shape[0],
            inputs=[
                self.state.joint_q,
                self.state.joint_qd,
                self.walk_q_start,
                self.walk_qd_start,
                self.walk_aisle,
                wp.vec2(*self.scene.walk_span),
                self.walk_default_q,
                self.walk_policy_joint,
                self.frame_dt,
                self.walk_action,
                self.walk_observation,
                self.walk_progress,
            ],
        )
        action = self.policy({self.policy_input: self.walk_observation})[self.policy_output]
        wp.launch(
            _walker_target_kernel,
            dim=self.walk_action.shape,
            inputs=[
                action,
                self.walk_q_start,
                self.walk_default_q,
                self.walk_policy_index,
                self.control.joint_target_q,
                self.walk_action,
            ],
        )

    def simulate(self):
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        if self.policy is not None:
            self._steer_walkers()
        if self.docks is not None:
            wp.launch(
                _cycle_dock_kernel,
                dim=1,
                inputs=[self.solver.dynamics.data, self.docks, DOCK_PERIOD, self.dock_frame, self.dock_occupant],
            )
        if self.sorting:
            self._sort()
        self.solver.collide(self.state, self.contacts)
        self.solver.step(self.state, self.state, self.control, self.contacts, self.frame_dt)
        if self.sorting:
            self._retire_packages()

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.sorting:
            wp.launch(
                _sort_instances_kernel,
                dim=self.sort_capacity,
                inputs=[self.solver.dynamics.data, self.state.body_q, self.sort_body_start, self.sort_xforms],
            )
            self.viewer.log_shapes(
                "/packages",
                newton.GeoType.BOX,
                tuple(float(h) for h in SORT_PACKAGE_HALF_EXTENTS),
                self.sort_xforms,
                wp.full(self.sort_capacity, wp.vec3(*COLORS["package"]), dtype=wp.vec3),
            )
        self.viewer.end_frame()

    def asleep_mask(self) -> np.ndarray:
        """Return which model bodies the solver put to sleep."""
        return self.solver.world.bodies.island_root.numpy()[: self.model.body_count] >= 0

    def describe(self) -> str:
        """Summarize the scene and the solver configuration."""
        fleet = " ".join(f"{name}={len(roots)}" for name, roots in self.scene.robot_roots.items())
        reduced = self.solver._reduced_articulation
        fused = reduced is not None and reduced.fused_substep is not None
        return (
            f"robots: {fleet} | bodies={self.model.body_count} free_bodies={len(self.free_bodies)} "
            f"shapes={self.model.shape_count} joints={self.model.joint_count} "
            f"mechanism_joints={len(self.scene.mechanism_joints)} | fused_substep={fused} "
            f"broad_phase={type(self.model._collision_pipeline.broad_phase).__name__}"
        )

    def report(self, step_ms: float | None = None) -> None:
        """Print the counts a benchmark of this scene is read against."""
        asleep = self.asleep_mask()
        free_asleep = int(asleep[self.free_bodies].sum())
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        reduced = self.solver._reduced_articulation
        cross = "n/a"
        block_system = getattr(reduced, "contact_block_system", None)
        if block_system is not None and getattr(block_system, "cross_column_count", None) is not None:
            cross = str(int(block_system.cross_column_count.numpy()[0]))
        timing = "" if step_ms is None else f"step={step_ms:.3f} ms ({1000.0 / step_ms:.1f} steps/s) | "
        if self.sorting:
            sorted_off, passed = self.sort_retired.numpy()
            timing += (
                f"packages spawned={int(self.sort_spawned.numpy()[0])} sorted={int(sorted_off)} passed={int(passed)} "
                f"robot_contacts={self.robot_contacts()} | "
            )
        print(
            f"[PhoenX Warehouse] {timing}contacts={contact_count}/{self.contacts.rigid_contact_max} "
            f"asleep={int(asleep.sum())}/{self.model.body_count} "
            f"free_asleep={free_asleep}/{len(self.free_bodies)} cross_columns={cross}",
            flush=True,
        )

    def run_timing(self) -> None:
        """Time the captured step after a warm-up and print the benchmark line."""
        for _ in range(self.timing_warmup):
            self.step()
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(self.timing_frames):
            self.step()
        wp.synchronize()
        self.report(1.0e3 * (time.perf_counter() - start) / self.timing_frames)

    def _robot_heights(self, poses: np.ndarray) -> dict[str, np.ndarray]:
        heights = {name: poses[roots, 2] for name, roots in self.scene.robot_roots.items() if roots}
        if self.docks is not None:
            # The humanoids at the dock: the root is an instance's first body.
            data = self.solver.dynamics.data
            assert int(data.error_count.numpy()[0]) == 0, "a humanoid could not arrive at or leave the dock"
            occupants = self.dock_occupant.numpy()
            roots = data.instance_body_start.numpy()[occupants[occupants >= 0]]
            if len(roots):
                heights["g1"] = np.concatenate([heights.get("g1", np.empty(0, dtype=poses.dtype)), poses[roots, 2]])
        return heights

    def robot_contacts(self) -> int:
        """Count the contacts between two different robots of the sorting hall."""
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        shape_body = self.model.shape_body.numpy()
        robots = []
        for shapes in (self.contacts.rigid_contact_shape0, self.contacts.rigid_contact_shape1):
            body = shape_body[shapes.numpy()[:count]]
            robots.append(np.where(body >= 0, self.body_robot[np.clip(body, 0, None)], -1))
        return int(np.sum((robots[0] >= 0) & (robots[1] >= 0) & (robots[0] != robots[1])))

    def test_post_step(self):
        """Stop at the first frame that loses a robot or contact capacity."""
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        assert count < self.contacts.rigid_contact_max, "contact capacity was exhausted"
        if self.sorting and self.robot_contacts() > 0:
            self.robot_touch_steps += 1
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all(), f"body poses are non-finite at t={self.sim_time:.2f} s"
        for name, heights in self._robot_heights(poses).items():
            lowest = float(heights.min())
            assert lowest > ROOT_HEIGHT_MIN[name], (
                f"a {name} fell: root height {lowest:.3f} m at t={self.sim_time:.2f} s"
            )

    def test_final(self):
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all(), "body poses are non-finite"
        assert np.isfinite(self.state.body_qd.numpy()).all(), "body velocities are non-finite"
        for name, heights in self._robot_heights(poses).items():
            lowest = float(heights.min())
            print(f"[PhoenX Warehouse] {name} root height {lowest:.3f}..{float(heights.max()):.3f} m", flush=True)
            assert lowest > ROOT_HEIGHT_MIN[name], f"a {name} fell: root height {lowest:.3f} m"
        # Wheeled robots must also still stand on their wheels.
        for name in ("kiva", "forklift"):
            roots = self.scene.robot_roots.get(name)
            if roots:
                q = poses[roots, 3:7]
                up_z = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
                assert float(up_z.min()) > 0.98, f"a {name} tipped: up axis z {float(up_z.min()):.3f}"
        if self.policy is not None:
            # Walkers keep to their loop and cover it at close to the commanded speed, less
            # the second they take to get going.
            roots = self.scene.robot_roots["anymal"]
            x, y = poses[roots, 0], poses[roots, 1] - np.asarray(self.scene.walker_aisles)
            off_loop = np.abs(np.hypot(x - np.clip(x, *self.scene.walk_span), y) - LANE_OFFSET)
            progress = self.walk_progress.numpy()
            expected = WALK_SPEED * max(0.0, self.sim_time - 1.0)
            print(
                f"[PhoenX Warehouse] walkers covered {float(progress.min()):.2f}..{float(progress.max()):.2f} m "
                f"in {self.sim_time:.1f} s, at most {float(off_loop.max()):.2f} m off their loop",
                flush=True,
            )
            assert float(off_loop.max()) < 0.4, f"a walker left its loop by {float(off_loop.max()):.2f} m"
            assert float(progress.mean()) >= 0.7 * expected, f"the walkers covered {float(progress.mean()):.2f} m"
            assert float(progress.min()) >= 0.4 * expected, f"a walker covered only {float(progress.min()):.2f} m"

        if self.sorting:
            assert int(self.solver.dynamics.error_count.numpy()[0]) == 0, "a package could not be added or removed"
            spawned = int(self.sort_spawned.numpy()[0])
            sorted_off, passed = (int(v) for v in self.sort_retired.numpy())
            # Runtime rows in use: added this step (1) or active (2).
            live = int(np.isin(self.solver.dynamics.bodies.data.state.numpy()[: self.sort_capacity], (1, 2)).sum())
            print(
                f"[PhoenX Warehouse] packages: {spawned} spawned, {sorted_off} sorted, {passed} passed, {live} on belts",
                flush=True,
            )
            assert spawned == sorted_off + passed + live, "a package went missing"
            # The first package reaches the first humanoid after about five seconds.
            print(f"[PhoenX Warehouse] robots touched each other in {self.robot_touch_steps} steps", flush=True)
            kivas = np.asarray(self.scene.robot_roots["kiva"], dtype=np.int64)
            travelled = np.linalg.norm(poses[kivas, :2] - self.kiva_start, axis=1)
            print(f"[PhoenX Warehouse] drive units moved {travelled.min():.2f}..{travelled.max():.2f} m", flush=True)
            if self.sim_time > 10.0:
                assert sorted_off > 0, "no humanoid pushed a package off its belt"
                assert self.robot_touch_steps > 0, "no two robots touched"
                assert float(travelled.min()) > 1.0, f"a drive unit moved only {travelled.min():.2f} m"

        if len(self.settled_bodies):
            moved = np.linalg.norm(poses[self.settled_bodies, :3] - self.initial_poses[self.settled_bodies, :3], axis=1)
            print(f"[PhoenX Warehouse] stored goods moved at most {float(moved.max()):.4f} m", flush=True)
            assert float(moved.max()) < 0.05, f"a stored package or carton moved {float(moved.max()):.3f} m"
        if len(self.free_bodies):
            assert float(poses[self.free_bodies, 2].min()) > -0.05, "a free body fell through the floor"
            if not self.sorting:  # the sorting hall's totes ride on moving drive units
                asleep = float(self.asleep_mask()[self.free_bodies].mean())
                assert asleep > 0.3, f"only {100.0 * asleep:.0f}% of the free bodies fell asleep"

        # Mechanisms hang on the world: their joint anchors must coincide, and a drawer must
        # not have turned in its slide.
        joints = np.asarray(self.scene.mechanism_joints, dtype=np.int64)
        if len(joints):
            children = self.model.joint_child.numpy()[joints]
            parent_anchor = self.model.joint_X_p.numpy()[joints, :3]
            child_pose = poses[children]
            offset = child_pose[:, :3] - parent_anchor
            axes = self.model.joint_axis.numpy()[self.model.joint_qd_start.numpy()[joints]]
            prismatic = self.model.joint_type.numpy()[joints] == int(newton.JointType.PRISMATIC)
            # A drawer may travel along its axis; nothing else may leave its anchor.
            offset[prismatic] -= np.sum(offset[prismatic] * axes[prismatic], axis=1, keepdims=True) * axes[prismatic]
            error = float(np.linalg.norm(offset, axis=1).max())
            turned = float(np.abs(child_pose[prismatic, 3:6]).max(initial=0.0))
            tilted = float(np.abs(child_pose[~prismatic, 3:5]).max(initial=0.0))
            print(f"[PhoenX Warehouse] mechanism anchor error {error:.5f} m", flush=True)
            assert error < 0.005, f"a mechanism left its joint by {error:.4f} m"
            assert turned < 0.01 and tilted < 0.01, "a mechanism rotated off its joint axis"
        self.report()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--robots", type=int, default=128, help="Number of robots in the fleet.")
    parser.add_argument("--storage-scale", type=float, default=1.0, help="Scale of the number of pods and pallets.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the storage layout.")
    parser.add_argument(
        "--runtime-robots", type=int, default=0, help="Humanoids that arrive at the dock and leave again at runtime."
    )
    parser.add_argument(
        "--walking-anymals",
        type=int,
        default=0,
        help="Replace the standing fleet by this many ANYmals that a policy walks around the aisles.",
    )
    parser.add_argument(
        "--robot-meshes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the humanoids' visual meshes. Defaults to on unless the viewer is null.",
    )
    parser.add_argument(
        "--mechanism-solver",
        choices=("reduced", "regular"),
        default="reduced",
        help="Solve the gates and drawers in reduced coordinates with the robots, or as regular joints.",
    )
    parser.add_argument(
        "--sorting-humanoids",
        type=int,
        default=0,
        help="Replace the warehouse by lines of this many G1 humanoids that push packages off belts.",
    )
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--timing-frames", type=int, default=0, help="Time this many captured steps, print the result, and exit."
    )
    parser.add_argument("--timing-warmup", type=int, default=120, help="Steps to run before timing.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if example.timing_frames > 0:
        example.run_timing()
    else:
        newton.examples.run(example, args)
