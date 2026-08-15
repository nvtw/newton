# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate RacerX USD physics with PhoenX and render the composed OptiX scene.

Newton imports standard UsdPhysics rigid bodies and colliders, while
SchemaResolverPhysx retains vendor-specific attributes for diagnostics. OptiX
keeps the complete visual hierarchy; dynamic body poses are mapped back to that
hierarchy by their exact USD paths.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd

try:
    from .racerx_track import add_track_barriers, build_track_layout
except ImportError:
    from racerx_track import add_track_barriers, build_track_layout

RACERX_USD_PATHS = {
    "a3": Path("/home/twidmer/Documents/Meshes/RacerX/Collected_A3_physics/A3_physics.usda"),
    "b3": Path("/home/twidmer/Documents/Meshes/RacerX/Collected_B3_physics/B3_physics.usda"),
    "c3": Path("/home/twidmer/Documents/Meshes/RacerX/Collected_C3_physics/C3_physics.usda"),
}
DEFAULT_RC_MODEL = "a3"
DEFAULT_USD_PATH = RACERX_USD_PATHS[DEFAULT_RC_MODEL]

# Easy-to-find example switches. Command-line flags can still override them.
LOAD_USD_ENVIRONMENT = False
LOAD_USD_EMISSIVE_MATERIALS = False
ENABLE_PHYSICS = True
ENABLE_CHASE_CAMERA = True
ENABLE_TRACK = True
ENABLE_MASS_SPLITTING = True
PRINT_PHYSICS_DATA = True
START_PAUSED = True
USD_MAX_TEXTURE_SIZE = 4096
MAX_CONTACT_GAP = 0.01
OPTIX_DLSS_QUALITY = "quality"
GROUND_HEIGHT = 0.0
GROUND_VISUAL_SIZE = 1000.0
SIM_SUBSTEPS = 4
C3_SIM_SUBSTEPS = 4
SOLVER_ITERATIONS = 6
CHASE_CAMERA_DISTANCE = 0.45
CHASE_CAMERA_HEIGHT = 0.18
CHASE_CAMERA_LOOK_AHEAD = 0.12
CHASE_CAMERA_TARGET_HEIGHT = 0.05
CHASE_CAMERA_RESPONSE = 10.0
CHASE_CAMERA_FOV = 55.0
VEHICLE_CONTACT_GAP = 0.001
DRIVE_SPEED = 140.0
DRIVE_ACCELERATION = 140.0
DRIVE_DECELERATION = 280.0
DRIVE_DAMPING = 0.35
DRIVE_TORQUE_LIMIT = 3.0
SUSPENSION_STIFFNESS_SCALE = 0.16
C3_REAR_SUSPENSION_STIFFNESS_MULTIPLIER = 10.0
C3_WHEEL_FRICTION = 0.5
LOOPED_VEHICLE_VARIANTS = frozenset(("b3", "c3"))
A3_SUSPENSION_STIFFNESS_MULTIPLIER = 6.25
B3_SUSPENSION_STIFFNESS_MULTIPLIER = 6.25
C3_FRONT_SUSPENSION_STIFFNESS_MULTIPLIER = 16.0
STEERING_TRAVEL = 0.00125
STEERING_LIMIT = 0.0015
C3_STEERING_TRAVEL = 0.00475
C3_STEERING_LIMIT = 0.0057
C3_STEERING_STIFFNESS = 3072000.0
C3_STEERING_DAMPING = 1680.0
C3_STEERING_FORCE_LIMIT = 3840.0
STEERING_RATE = 0.015
STEERING_STIFFNESS = 64000.0
STEERING_DAMPING = 240.0
STEERING_FORCE_LIMIT = 320.0
WHEEL_FRICTION = 1.2
TRACK_HALF_WIDTH = 0.32
TRACK_BARRIER_SPACING = 0.32
TRACK_BARRIER_HALF_EXTENTS = (0.05, 0.05, 0.05)
TRACK_BARRIER_DENSITY = 450.0
TRACK_ROAD_HALF_THICKNESS = 0.002


_VEHICLE_CORNERS = (
    ("FR", "front", "right"),
    ("FL", "front", "left"),
    ("RR", "rear", "right"),
    ("RL", "rear", "left"),
)


@dataclass(frozen=True)
class _VehicleParts:
    """Store model-specific bodies, joints, and collider shapes."""

    wheel_joints: tuple[int, int, int, int]
    wheel_shapes: tuple[int, int, int, int]
    wheel_shape_paths: tuple[str, str, str, str]
    steering_joint: int
    chassis_body: int
    chassis_body_path: str
    variant: str = "a3"


NON_PHYSICAL_RIGID_BODY_PATHS = {
    "/World/FX_wheel_dust_flow_01": "VFX helper has no collider",
    "/World/FlowCandleWind": "flow helper has no collider",
}

_RENDERER_FROM_PHYSICS = np.array(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)


@wp.kernel
def _gather_body_transforms(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[wp.int32],
    transforms: wp.array[wp.transform],
):
    """Gather track-barrier poses without a host readback."""
    index = wp.tid()
    transforms[index] = body_q[body_indices[index]]


@wp.kernel
def _update_vehicle_controls(
    drive_input: wp.array[float],
    wheel_dofs: wp.array[wp.int32],
    steering_target_index: wp.int32,
    frame_dt: float,
    steering_travel: float,
    wheel_speed_command: wp.array[float],
    steering_command: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
):
    target_speed = drive_input[0] * DRIVE_SPEED
    braking = target_speed == 0.0 or wheel_speed_command[0] * target_speed < 0.0
    acceleration = DRIVE_DECELERATION if braking else DRIVE_ACCELERATION
    max_speed_delta = acceleration * frame_dt
    speed_delta = wp.clamp(target_speed - wheel_speed_command[0], -max_speed_delta, max_speed_delta)
    wheel_speed_command[0] += speed_delta
    for wheel_index in range(wheel_dofs.shape[0]):
        joint_target_qd[wheel_dofs[wheel_index]] = wheel_speed_command[0]

    steering_target = drive_input[1] * steering_travel
    max_steering_delta = STEERING_RATE * frame_dt
    steering_delta = wp.clamp(
        steering_target - steering_command[0],
        -max_steering_delta,
        max_steering_delta,
    )
    steering_command[0] += steering_delta
    joint_target_q[steering_target_index] = steering_command[0]


@wp.kernel
def _update_chase_camera_device(
    body_q: wp.array[wp.transform],
    chassis_body: wp.int32,
    frame_dt: float,
    snap: wp.int32,
    initialized: wp.array[wp.int32],
    camera_forwards: wp.array[wp.vec3],
    camera_positions: wp.array[wp.vec3],
    camera_targets: wp.array[wp.vec3],
):
    pose = body_q[chassis_body]
    position = wp.transform_get_translation(pose)
    forward = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3(1.0, 0.0, 0.0))
    forward = wp.vec3(forward[0], forward[1], 0.0)
    if wp.length(forward) <= 1.0e-6:
        forward = wp.vec3(1.0, 0.0, 0.0)
    else:
        forward = wp.normalize(forward)
    if snap != 0 or initialized[0] == 0:
        camera_forwards[0] = forward
        initialized[0] = 1
    else:
        blend = 1.0 - wp.exp(-CHASE_CAMERA_RESPONSE * frame_dt)
        camera_forwards[0] = wp.normalize(camera_forwards[0] + blend * (forward - camera_forwards[0]))
    camera_positions[0] = position - CHASE_CAMERA_DISTANCE * camera_forwards[0] + wp.vec3(0.0, 0.0, CHASE_CAMERA_HEIGHT)
    camera_targets[0] = (
        position + CHASE_CAMERA_LOOK_AHEAD * camera_forwards[0] + wp.vec3(0.0, 0.0, CHASE_CAMERA_TARGET_HEIGHT)
    )


@wp.kernel
def _compute_dynamic_local_transforms(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[wp.int32],
    parent_world_inverses: wp.array[wp.mat44],
    geometry_offsets: wp.array[wp.mat44],
    renderer_from_physics: wp.array[wp.mat44],
    physics_from_renderer: wp.array[wp.mat44],
    local_transforms: wp.array[wp.mat44],
):
    binding_index = wp.tid()
    rigid_physics = wp.transform_to_matrix(body_q[body_indices[binding_index]])
    rigid_renderer = renderer_from_physics[0] * rigid_physics * physics_from_renderer[0]
    local_transforms[binding_index] = (
        parent_world_inverses[binding_index] * rigid_renderer * geometry_offsets[binding_index]
    )


def _relevant_physics_properties(prim) -> list:
    """Return authored standard and vendor physics properties on a prim."""
    return [
        prop
        for prop in prim.GetAuthoredProperties()
        if prop.GetName().startswith(("physics:", "physx")) or prop.GetName() == "physicsVelocityScale"
    ]


def _physics_prims(stage) -> list:
    """Return composed prims carrying relevant USD physics data."""
    from pxr import UsdPhysics

    records = []
    for prim in stage.TraverseAll():
        schemas = [
            str(schema) for schema in prim.GetAppliedSchemas() if "Physics" in str(schema) or "Physx" in str(schema)
        ]
        properties = _relevant_physics_properties(prim)
        if schemas or properties or prim.IsA(UsdPhysics.Scene):
            records.append((prim, schemas, properties))
    return records


def _property_value(prim, prop):
    """Read an authored USD attribute or relationship for diagnostics."""
    attribute = prim.GetAttribute(prop.GetName())
    if attribute:
        return attribute.Get()
    relationship = prim.GetRelationship(prop.GetName())
    return [str(path) for path in relationship.GetTargets()]


def _print_physics_inventory(stage) -> None:
    """Print every composed physics prim and its authored physics properties."""
    records = _physics_prims(stage)
    schema_counts = Counter(schema for _prim, schemas, _props in records for schema in schemas)
    errors = list(stage.GetCompositionErrors())

    print(
        "[RacerX USD physics] "
        f"prims={len(records)} schemas={dict(sorted(schema_counts.items()))} "
        f"composition_errors={len(errors)}"
    )
    for error in errors:
        message = error.GetMessage() if hasattr(error, "GetMessage") else str(error)
        print(f"[RacerX USD composition error] {message}")

    for prim, schemas, properties in records:
        print(f"[RacerX USD prim] path={prim.GetPath()} type={prim.GetTypeName() or '<none>'} schemas={schemas}")
        for prop in properties:
            print(f"  {prop.GetName()} = {_property_value(prim, prop)!r}")


def _physics_ignore_paths(stage) -> dict[str, str]:
    """Find trigger-only and incomplete collider prims that Newton cannot simulate."""
    from pxr import UsdGeom, UsdPhysics

    ignored = dict(NON_PHYSICAL_RIGID_BODY_PATHS)
    for prim in stage.TraverseAll():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if any(prop.GetName().startswith("physxTrigger:") for prop in prim.GetAuthoredProperties()):
            ignored[path] = "PhysX trigger scripts are not rigid collision geometry"
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            if (
                mesh.GetPointsAttr().Get() is None
                or mesh.GetFaceVertexIndicesAttr().Get() is None
                or mesh.GetFaceVertexCountsAttr().Get() is None
            ):
                ignored[path] = "collider mesh has no composed topology"
    return ignored


def _normalize_stage_units_for_newton(stage) -> float:
    """Author a session-layer root scale so Newton imports the stage in meters."""
    from pxr import Gf, UsdGeom

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if math.isclose(meters_per_unit, 1.0):
        return meters_per_unit

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("A non-meter USD stage needs a default prim for unit normalization")

    stage.SetEditTarget(stage.GetSessionLayer())
    root = UsdGeom.Xformable(default_prim)
    scale_op = root.AddScaleOp(
        UsdGeom.XformOp.PrecisionDouble,
        "newtonStageUnits",
    )
    scale_op.Set(Gf.Vec3d(meters_per_unit))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return meters_per_unit


def _scale_shape_contact_gaps(builder, scale: float) -> None:
    """Convert imported contact gaps to meters and cap their detection shell."""
    builder.shape_gap[:] = [min(gap * scale, MAX_CONTACT_GAP, VEHICLE_CONTACT_GAP) for gap in builder.shape_gap]


def _scale_linear_joint_units(builder, scale: float) -> None:
    """Convert imported linear joint coordinates from stage units to meters."""
    if math.isclose(scale, 1.0):
        return
    for joint_index, (linear_count, _angular_count) in enumerate(builder.joint_dof_dim):
        dof_start = builder.joint_qd_start[joint_index]
        coord_start = builder.joint_q_start[joint_index]
        for axis_offset in range(linear_count):
            dof = dof_start + axis_offset
            coord = coord_start + axis_offset
            builder.joint_limit_lower[dof] *= scale
            builder.joint_limit_upper[dof] *= scale
            builder.joint_target_q[dof] *= scale
            builder.joint_target_qd[dof] *= scale
            builder.joint_velocity_limit[dof] *= scale
            builder.joint_q[coord] *= scale
            builder.joint_qd[dof] *= scale


def _find_unique_vehicle_path(paths, description: str, predicate) -> str:
    """Find one semantically identified vehicle path."""
    matches = [path for path in paths if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one RacerX {description}, found {matches}")
    return matches[0]


def _resolve_vehicle_parts(builder, result) -> _VehicleParts:
    """Discover model-specific vehicle parts from semantic USD names."""
    path_joint_map = result["path_joint_map"]
    path_body_map = result["path_body_map"]
    path_shape_map = result["path_shape_map"]
    body_path = {int(body): path for path, body in path_body_map.items()}

    wheel_joints = []
    wheel_shapes = []
    wheel_shape_paths = []
    for corner, longitudinal, lateral in _VEHICLE_CORNERS:
        joint_path = _find_unique_vehicle_path(
            path_joint_map,
            f"{corner} wheel joint",
            lambda path, corner=corner: f"/{corner}/" in path and path.endswith("/Hinge_Wheel_WheelLinkage"),
        )
        joint = int(path_joint_map[joint_path])
        wheel_body_path = _find_unique_vehicle_path(
            path_body_map,
            f"{corner} wheel body",
            lambda path, longitudinal=longitudinal, lateral=lateral: (
                "wheel" in path.rsplit("/", 1)[-1].lower()
                and longitudinal in path.rsplit("/", 1)[-1].lower()
                and lateral in path.rsplit("/", 1)[-1].lower()
                and "link" not in path.rsplit("/", 1)[-1].lower()
                and "connection" not in path.rsplit("/", 1)[-1].lower()
            ),
        )
        wheel_body = int(path_body_map[wheel_body_path])
        endpoints = (int(builder.joint_parent[joint]), int(builder.joint_child[joint]))
        if wheel_body not in endpoints:
            endpoint_paths = tuple(body_path.get(body, "<world>") for body in endpoints)
            raise RuntimeError(f"RacerX {corner} wheel joint connects {endpoint_paths}, not {wheel_body_path}")

        shape_candidates = [
            (path, int(shape))
            for path, shape in path_shape_map.items()
            if int(builder.shape_body[int(shape)]) == wheel_body
            and builder.shape_type[int(shape)] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
        ]
        if len(shape_candidates) != 1:
            raise RuntimeError(f"Expected one mesh collider for RacerX {corner} wheel, found {shape_candidates}")
        shape_path, shape = shape_candidates[0]
        wheel_joints.append(joint)
        wheel_shapes.append(shape)
        wheel_shape_paths.append(shape_path)

    steering_path = _find_unique_vehicle_path(
        path_joint_map,
        "steering drive",
        lambda path: path.endswith("/Steering/Steering_Link_Drive"),
    )
    steering_joint = int(path_joint_map[steering_path])
    steering_bodies = (int(builder.joint_parent[steering_joint]), int(builder.joint_child[steering_joint]))
    dynamic_steering_bodies = [body for body in steering_bodies if body >= 0]
    if len(dynamic_steering_bodies) != 2:
        raise RuntimeError(f"Expected two dynamic RacerX steering bodies, found {steering_bodies}")
    chassis_body = max(dynamic_steering_bodies, key=lambda body: float(builder.body_mass[body]))

    chassis_name = body_path[chassis_body].rsplit("/", 1)[-1].lower()
    if "_c1_" in chassis_name:
        variant = "c3"
    elif "_b1_" in chassis_name:
        variant = "b3"
    else:
        variant = "a3"

    return _VehicleParts(
        wheel_joints=tuple(wheel_joints),
        wheel_shapes=tuple(wheel_shapes),
        wheel_shape_paths=tuple(wheel_shape_paths),
        steering_joint=steering_joint,
        variant=variant,
        chassis_body=chassis_body,
        chassis_body_path=body_path[chassis_body],
    )


def _replace_wheel_mesh_colliders(builder, parts: _VehicleParts) -> None:
    """Place one axle-aligned cylinder collider at each rendered wheel."""
    cylinder_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
    for path, joint, shape in zip(parts.wheel_shape_paths, parts.wheel_joints, parts.wheel_shapes, strict=True):
        if builder.shape_type[shape] not in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            raise RuntimeError(f"Expected a mesh-backed RacerX wheel collider at {path}")
        joint_axis = np.asarray(builder.joint_axis[int(builder.joint_qd_start[joint])], dtype=np.float32)
        if (
            abs(float(joint_axis[1])) < 1.0 - 1.0e-5
            or max(abs(float(joint_axis[0])), abs(float(joint_axis[2]))) > 1.0e-5
        ):
            raise RuntimeError(f"Expected a local Y axle for RacerX wheel {path}, found {joint_axis}")

        source = builder.shape_source[shape]
        vertices = np.asarray(source.vertices, dtype=np.float32)
        scale = np.asarray(builder.shape_scale[shape], dtype=np.float32)
        scaled_vertices = vertices * scale
        lower = scaled_vertices.min(axis=0)
        upper = scaled_vertices.max(axis=0)
        center = 0.5 * (lower + upper)
        extent = upper - lower
        radius = 0.5 * max(extent[0], extent[2])
        half_width = 0.5 * extent[1]

        cylinder_offset = wp.transform(wp.vec3(*center), cylinder_rotation)
        builder.shape_transform[shape] = builder.shape_transform[shape] * cylinder_offset
        builder.shape_type[shape] = newton.GeoType.CYLINDER
        builder.shape_source[shape] = None
        builder.shape_scale[shape] = wp.vec3(radius, half_width, 0.0)
        builder.shape_material_mu[shape] = C3_WHEEL_FRICTION if parts.variant == "c3" else WHEEL_FRICTION


def _configure_vehicle_ground_friction(builder, parts: _VehicleParts) -> None:
    """Keep underside contacts slippery while preserving tire grip."""
    if parts.variant not in LOOPED_VEHICLE_VARIANTS:
        return
    wheel_shapes = set(parts.wheel_shapes)
    for shape in range(builder.shape_count):
        if shape not in wheel_shapes:
            builder.shape_material_mu[shape] = 0.0


def _filter_vehicle_ground_contacts(builder, parts: _VehicleParts, ground_shape: int) -> None:
    """Keep the generated ground in contact with only the vehicle wheels."""
    if parts.variant not in LOOPED_VEHICLE_VARIANTS:
        return
    wheel_shapes = set(parts.wheel_shapes)
    for shape in range(ground_shape):
        if shape not in wheel_shapes:
            builder.add_shape_collision_filter_pair(shape, ground_shape)


def _configure_vehicle_joints(builder, result, parts: _VehicleParts) -> tuple[list[int], int, int]:
    """Configure wheel velocity drives and a PhoenX-compatible steering slider."""
    wheel_dofs = []
    for joint_index in parts.wheel_joints:
        dof = int(builder.joint_qd_start[joint_index])
        builder.joint_target_mode[dof] = newton.JointTargetMode.VELOCITY
        builder.joint_target_ke[dof] = 0.0
        builder.joint_target_kd[dof] = DRIVE_DAMPING
        builder.joint_effort_limit[dof] = DRIVE_TORQUE_LIMIT
        wheel_dofs.append(dof)

    suspension_joints = {
        int(joint_index): path
        for path, joint_index in result["path_joint_map"].items()
        if path.endswith("/Slider_Suspension")
    }
    # C3 authors a much softer front axle and carries its motor over the rear.
    # Tune each axle independently while retaining the authored damping ratio.
    for joint_index, path in suspension_joints.items():
        stiffness_scale = SUSPENSION_STIFFNESS_SCALE
        if parts.variant == "a3":
            stiffness_scale *= A3_SUSPENSION_STIFFNESS_MULTIPLIER
        if parts.variant == "b3":
            stiffness_scale *= B3_SUSPENSION_STIFFNESS_MULTIPLIER
        if parts.variant == "c3":
            if "/RR/" in path or "/RL/" in path:
                stiffness_scale *= C3_REAR_SUSPENSION_STIFFNESS_MULTIPLIER
            else:
                stiffness_scale *= C3_FRONT_SUSPENSION_STIFFNESS_MULTIPLIER
        dof = int(builder.joint_qd_start[joint_index])
        builder.joint_target_ke[dof] *= stiffness_scale
        builder.joint_target_kd[dof] *= math.sqrt(stiffness_scale)

    steering_joint = parts.steering_joint
    steering_dof_start = int(builder.joint_qd_start[steering_joint])
    linear_count, _angular_count = builder.joint_dof_dim[steering_joint]
    if linear_count != 2:
        raise RuntimeError(f"Expected two RacerX steering translations, found {linear_count}")

    if parts.variant == "b3":
        suspension_dofs = [int(builder.joint_qd_start[joint]) for joint in suspension_joints]
        stiffness = max(builder.joint_target_ke[dof] for dof in suspension_dofs)
        damping = max(builder.joint_target_kd[dof] for dof in suspension_dofs)
        for dof in suspension_dofs:
            builder.joint_target_ke[dof] = stiffness
            builder.joint_target_kd[dof] = damping

    # The source leaves both X/Y translations finite, but PhoenX currently
    # supports finite bounds after reducing D6 to a one-axis prismatic joint.
    # X has no authored drive, so lock it and retain the driven Y coordinate.
    builder.joint_limit_lower[steering_dof_start] = 1.0
    builder.joint_limit_upper[steering_dof_start] = -1.0
    steering_dof = steering_dof_start + 1
    steering_limit = C3_STEERING_LIMIT if parts.variant == "c3" else STEERING_LIMIT
    builder.joint_limit_lower[steering_dof] = -steering_limit
    builder.joint_limit_upper[steering_dof] = steering_limit
    builder.joint_target_mode[steering_dof] = newton.JointTargetMode.POSITION
    builder.joint_target_ke[steering_dof] = C3_STEERING_STIFFNESS if parts.variant == "c3" else STEERING_STIFFNESS
    builder.joint_target_kd[steering_dof] = C3_STEERING_DAMPING if parts.variant == "c3" else STEERING_DAMPING
    builder.joint_effort_limit[steering_dof] = (
        C3_STEERING_FORCE_LIMIT if parts.variant == "c3" else STEERING_FORCE_LIMIT
    )
    return wheel_dofs, steering_joint, steering_dof


def _add_closed_loop_joint_metadata(builder) -> None:
    """Register imported closed-loop joints for maximal-coordinate validation."""
    # RC stages may contain partial articulation metadata that cannot be mixed
    # monotonically with the loop joints. Rebuild a canonical singleton layout
    # without inventing an invalid reduced-coordinate tree; PhoenX still solves
    # the connected equality rows as one maximal mechanism.
    builder.articulation_start.clear()
    builder.articulation_end.clear()
    builder.articulation_label.clear()
    builder.articulation_world.clear()
    builder.joint_articulation[:] = [-1] * len(builder.joint_articulation)
    for joint_index, label in enumerate(builder.joint_label):
        builder.add_articulation([joint_index], label=label)


def _create_collision_pipeline(model):
    """Create a pipeline that ignores contacts between immovable props."""
    return newton.CollisionPipeline(
        model,
        broad_phase="sap",
        contact_matching="sticky",
        include_static_kinematic_pairs=False,
    )


def _select_authored_camera(viewer, requested_path: str | None = None) -> str:
    """Select the same authored overview-camera convention as the OTK USD example."""
    from pxr import UsdGeom, UsdRender

    usd_scene = viewer.usd_scene
    stage = usd_scene.stage
    camera_paths = []
    if requested_path:
        camera_paths.append(requested_path)
    else:
        settings = UsdRender.Settings.GetStageRenderSettings(stage)
        if settings:
            camera_paths.extend(str(path) for path in settings.GetCameraRel().GetTargets())
        custom_data = stage.GetPseudoRoot().GetMetadata("customLayerData") or {}
        bound_camera = custom_data.get("cameraSettings", {}).get("boundCamera")
        if bound_camera:
            camera_paths.append(str(bound_camera))

        unsuitable = ("follow", "velocity", "physics", "collision", "debug")
        authored = [
            str(prim.GetPath())
            for prim in stage.TraverseAll()
            if prim.IsA(UsdGeom.Camera)
            and str(UsdGeom.Imageable(prim).ComputeVisibility()) != "invisible"
            and not any(token in prim.GetName().lower() for token in unsuitable)
        ]
        camera_paths.extend(
            sorted(
                authored,
                key=lambda path: (
                    0 if "overview" in Path(path).name.lower() else 1,
                    path,
                ),
            )
        )

    for camera_path in dict.fromkeys(camera_paths):
        prim = stage.GetPrimAtPath(camera_path)
        handle = usd_scene.get_transform(camera_path)
        if not prim or handle is None or not prim.IsA(UsdGeom.Camera):
            continue
        camera = UsdGeom.Camera(prim)
        if str(camera.GetProjectionAttr().Get()) != "perspective":
            continue
        focal_length = float(camera.GetFocalLengthAttr().Get() or 0.0)
        aperture = float(camera.GetVerticalApertureAttr().Get() or 0.0)
        if focal_length <= 0.0 or aperture <= 0.0:
            continue

        world = usd_scene.get_world_transform(handle)
        position = world[:3, 3]
        target = position - world[:3, 2]
        fov = math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))
        viewer.set_camera_look_at(
            position,
            target,
            fov=float(np.clip(fov, 5.0, 120.0)),
            renderer_space=True,
        )
        return camera_path

    if requested_path:
        raise ValueError(f"USD camera is missing, invisible, or unsupported: {requested_path}")
    car_candidates = [prim for prim in stage.Traverse() if prim.GetName().lower().startswith("rc_car")]
    if len(car_candidates) != 1:
        raise RuntimeError(
            f"Expected one RacerX vehicle root, found {[str(prim.GetPath()) for prim in car_candidates]}"
        )
    car_prim = car_candidates[0]
    bounds = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_]).ComputeWorldBound(car_prim).ComputeAlignedRange()
    lower = np.asarray(bounds.GetMin(), dtype=np.float32)
    upper = np.asarray(bounds.GetMax(), dtype=np.float32)
    center = 0.5 * (lower + upper)
    radius = max(float(np.linalg.norm(upper - lower)), 0.25)
    position = center + radius * np.array((1.2, -1.2, 0.65), dtype=np.float32)
    position_renderer = (_RENDERER_FROM_PHYSICS @ np.append(position, 1.0))[:3]
    target_renderer = (_RENDERER_FROM_PHYSICS @ np.append(center, 1.0))[:3]
    viewer.set_camera_look_at(
        position_renderer,
        target_renderer,
        fov=45.0,
        renderer_space=True,
    )
    return "<generated RacerX overview>"


def _pose_matrix(pose) -> np.ndarray:
    """Convert one Warp transform NumPy row to a homogeneous matrix."""
    px, py, pz, x, y, z, w = (float(value) for value in pose)
    rotation = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (px, py, pz)
    return matrix


def _chase_camera_targets(pose) -> tuple[np.ndarray, np.ndarray]:
    """Return a level chase-camera eye and target from one chassis pose."""
    chassis_world = _pose_matrix(pose)
    position = chassis_world[:3, 3]
    forward = chassis_world[:3, 0].copy()
    forward[2] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-6:
        forward = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    else:
        forward /= forward_norm
    eye = position - CHASE_CAMERA_DISTANCE * forward
    eye[2] += CHASE_CAMERA_HEIGHT
    target = position + CHASE_CAMERA_LOOK_AHEAD * forward
    target[2] += CHASE_CAMERA_TARGET_HEIGHT
    return eye, target


def _nearest_transform_parent(usd_scene, path: str):
    """Return the nearest transformable ancestor handle for a USD path."""
    prim = usd_scene.get_prim(path).GetParent()
    while prim:
        handle = usd_scene.get_transform(str(prim.GetPath()))
        if handle is not None:
            return handle
        prim = prim.GetParent()
    return None


class Example:
    """Run imported RacerX rigid-body physics with PhoenX and OptiX."""

    def __init__(self, viewer, args):
        if not all(
            hasattr(viewer, method)
            for method in ("load_scene_from_usd", "set_camera_look_at", "set_camera_look_at_device")
        ):
            raise RuntimeError("The RacerX USD example requires the latest --viewer optix")

        self.viewer = viewer
        self.device = wp.get_device()
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0

        usd_path = Path(args.usd_path).expanduser().resolve()
        if not usd_path.is_file():
            raise FileNotFoundError(f"RacerX USD stage not found: {usd_path}")

        if not viewer.load_scene_from_usd(
            str(usd_path),
            max_texture_size=args.usd_max_texture_size,
            enable_emissive_materials=args.usd_emissive_materials,
            load_usd_environment=args.usd_environment,
            usd_environment_scale=args.usd_environment_scale,
        ):
            raise RuntimeError(f"OptiX failed to load USD stage: {usd_path}")
        self._usd_visuals_visible = True

        stage = viewer.usd_scene.stage
        if args.print_physics_data:
            _print_physics_inventory(stage)

        self.physics_enabled = bool(args.physics)
        self.chase_camera_enabled = bool(args.chase_camera and self.physics_enabled)
        self.physics_result = None
        self._dynamic_bindings = []
        if self.physics_enabled:
            self._build_physics(stage)
        else:
            self.state = None
            self.graph = None

        camera_path = _select_authored_camera(viewer, args.usd_camera)
        if self.chase_camera_enabled:
            camera_path = "<RacerX chase camera>"
        print(
            f"[PhoenX RacerX USD] loaded {usd_path} "
            f"({viewer.usd_scene.transform_count} retained transforms, camera={camera_path})"
        )

    def _build_physics(self, stage) -> None:
        ignored = _physics_ignore_paths(stage)
        authored_unit = _normalize_stage_units_for_newton(stage)

        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        result = builder.add_usd(
            stage,
            only_load_enabled_rigid_bodies=True,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
            ignore_paths=list(ignored),
            schema_resolvers=[newton.usd.SchemaResolverPhysx()],
            ignore_composition_errors=True,
        )
        vehicle_parts = _resolve_vehicle_parts(builder, result)
        self._vehicle_variant = vehicle_parts.variant
        self._looped_vehicle = vehicle_parts.variant in LOOPED_VEHICLE_VARIANTS
        self._steering_travel = C3_STEERING_TRAVEL if vehicle_parts.variant == "c3" else STEERING_TRAVEL
        self._sim_substeps = C3_SIM_SUBSTEPS if vehicle_parts.variant == "c3" else SIM_SUBSTEPS
        _scale_shape_contact_gaps(builder, authored_unit)
        _replace_wheel_mesh_colliders(builder, vehicle_parts)
        _configure_vehicle_ground_friction(builder, vehicle_parts)
        # The source PhysicsScene stores 981 in cm/s^2; stage-root scaling
        # does not transform scalar physics attributes.
        builder.gravity = wp.vec3(0.0, 0.0, -9.81)
        _scale_linear_joint_units(builder, authored_unit)
        self._wheel_dofs, self._steering_joint, self._steering_dof = _configure_vehicle_joints(
            builder, result, vehicle_parts
        )
        self._steering_target_index = builder.joint_q_start[self._steering_joint] + 1
        _add_closed_loop_joint_metadata(builder)
        ground_shape = builder.add_ground_plane(
            height=GROUND_HEIGHT,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.9, gap=VEHICLE_CONTACT_GAP),
        )
        _filter_vehicle_ground_contacts(builder, vehicle_parts, ground_shape)
        self._track_layout = None
        self._track_body_indices = []
        if ENABLE_TRACK:
            self._track_layout = build_track_layout(
                spacing=TRACK_BARRIER_SPACING,
                half_width=TRACK_HALF_WIDTH,
                barrier_half_height=TRACK_BARRIER_HALF_EXTENTS[2],
                road_height=GROUND_HEIGHT + TRACK_ROAD_HALF_THICKNESS,
            )
            self._track_body_indices = add_track_barriers(
                builder,
                self._track_layout,
                half_extents=TRACK_BARRIER_HALF_EXTENTS,
                density=TRACK_BARRIER_DENSITY,
                contact_gap=VEHICLE_CONTACT_GAP,
            )

        self.physics_result = result
        self._chassis_body = vehicle_parts.chassis_body
        self._builder = builder

        for path, reason in sorted(ignored.items()):
            print(f"[RacerX USD skipped] path={path} reason={reason}")

        dynamic_paths = []
        kinematic_paths = []
        for path, body_index in sorted(result["path_body_map"].items()):
            if int(builder.body_flags[body_index]) & int(newton.BodyFlags.KINEMATIC):
                motion = "kinematic"
                kinematic_paths.append(path)
            else:
                motion = "dynamic"
                dynamic_paths.append(path)
            print(f"[RacerX USD body] id={body_index} motion={motion} path={path}")

        for path, shape_index in sorted(result["path_shape_map"].items()):
            body_index = int(builder.shape_body[shape_index])
            print(f"[RacerX USD collider] id={shape_index} body={body_index} path={path}")

        for namespace, paths in sorted(result["schema_attrs"].items()):
            for path, attributes in sorted(paths.items()):
                print(f"[RacerX USD {namespace}] path={path} attributes={attributes}")

        print(
            "[RacerX USD import] "
            f"authored_meters_per_unit={authored_unit:g} bodies={builder.body_count} "
            f"dynamic={len(dynamic_paths)} kinematic={len(kinematic_paths)} "
            f"colliders={builder.shape_count} joints={builder.joint_count} "
            f"filtered_pairs={len(builder.shape_collision_filter_pairs)}"
        )
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
        print("[RacerX controls] I/K throttle, J/L steering, release throttle to brake")

        self.model = builder.finalize(skip_shape_contact_pairs=True)
        self.collision_pipeline = _create_collision_pipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.state = self.model.state()
        self.state.body_q.assign(self.model.body_q)
        self.state.body_qd.assign(self.model.body_qd)
        self.initial_state = self.model.state()
        self.initial_state.assign(self.state)
        self.control = self.model.control()
        self._drive_input_host = np.zeros(2, dtype=np.float32)
        self._drive_input_device = wp.zeros(2, dtype=float, device=self.device)
        self._wheel_dofs_device = wp.array(self._wheel_dofs, dtype=wp.int32, device=self.device)
        self._wheel_speed_command = wp.zeros(1, dtype=float, device=self.device)
        self._steering_command = wp.zeros(1, dtype=float, device=self.device)
        if self.chase_camera_enabled:
            self._chase_camera_initialized = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._chase_camera_positions = wp.empty(1, dtype=wp.vec3, device=self.device)
            self._chase_camera_forwards = wp.empty(1, dtype=wp.vec3, device=self.device)
            self._chase_camera_targets = wp.empty(1, dtype=wp.vec3, device=self.device)
            self._update_chase_camera(snap=True)
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            collision_pipeline=self.collision_pipeline,
            substeps=self._sim_substeps,
            solver_iterations=SOLVER_ITERATIONS,
            velocity_iterations=1,
            default_friction=0.9,
            friction_combine_mode="min" if self._looped_vehicle else "average",
            mass_splitting=ENABLE_MASS_SPLITTING,
            mass_splitting_unrolled=ENABLE_MASS_SPLITTING,
            step_layout="single_world" if ENABLE_MASS_SPLITTING else "multi_world",
            articulation_mode="maximal",
        )

        self.viewer.set_model(self.model)
        if self.chase_camera_enabled:
            self.viewer.set_camera_look_at_device(
                self._chase_camera_positions,
                self._chase_camera_targets,
                fov=CHASE_CAMERA_FOV,
            )
        self._build_dynamic_bindings(dynamic_paths)

        self._ground_visual_xforms = wp.array(
            [wp.transform((0.0, 0.0, GROUND_HEIGHT), wp.quat_identity())],
            dtype=wp.transform,
            device=self.device,
        )
        self._ground_visual_colors = wp.array([wp.vec3(0.7, 0.7, 0.7)], dtype=wp.vec3, device=self.device)
        self._ground_visual_materials = wp.array([wp.vec4(0.8, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)
        if self._track_layout is not None:
            barrier_count = len(self._track_body_indices)
            road_count = len(self._track_layout.road_poses)
            self._track_body_indices_device = wp.array(
                self._track_body_indices,
                dtype=wp.int32,
                device=self.device,
            )
            self._track_barrier_xforms = wp.empty(barrier_count, dtype=wp.transform, device=self.device)
            self._track_barrier_colors = wp.array(
                self._track_layout.barrier_colors,
                dtype=wp.vec3,
                device=self.device,
            )
            self._track_barrier_materials = wp.full(
                barrier_count,
                wp.vec4(0.65, 0.15, 0.0, 0.0),
                dtype=wp.vec4,
                device=self.device,
            )
            self._track_road_xforms = wp.array(
                self._track_layout.road_poses,
                dtype=wp.transform,
                device=self.device,
            )
            self._track_road_colors = wp.full(
                road_count,
                wp.vec3(0.055, 0.06, 0.07),
                dtype=wp.vec3,
                device=self.device,
            )
            self._track_road_materials = wp.full(
                road_count,
                wp.vec4(0.9, 0.05, 0.0, 0.0),
                dtype=wp.vec4,
                device=self.device,
            )
            print(
                f"[RacerX track] length={self._track_layout.length:.1f} m "
                f"barriers={barrier_count} road_segments={road_count}"
            )

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_frame()
            self.graph = capture.graph

    def _build_dynamic_bindings(self, dynamic_paths: list[str]) -> None:
        usd_scene = self.viewer.usd_scene
        world_device = usd_scene.world_transforms_device
        render_worlds = (
            world_device.numpy()
            if world_device is not None
            else np.stack([usd_scene.get_world_transform(handle) for handle in usd_scene.transforms])
        )
        physics_poses = self.state.body_q.numpy()
        physics_from_renderer = np.linalg.inv(_RENDERER_FROM_PHYSICS)

        for path in dynamic_paths:
            handle = usd_scene.get_transform(path)
            if handle is None:
                print(f"[RacerX USD transform] no render transform for dynamic body {path}")
                continue
            body_index = int(self.physics_result["path_body_map"][path])
            rigid_physics = _pose_matrix(physics_poses[body_index])
            rigid_renderer = _RENDERER_FROM_PHYSICS @ rigid_physics @ physics_from_renderer
            original_world = render_worlds[handle.index]
            geometry_offset = np.linalg.inv(rigid_renderer) @ original_world

            parent = _nearest_transform_parent(usd_scene, path)
            parent_world = np.eye(4, dtype=np.float32) if parent is None else render_worlds[parent.index]
            parent_world_inverse = np.linalg.inv(parent_world)
            self._dynamic_bindings.append((body_index, handle, parent_world_inverse, geometry_offset))
            print(
                f"[RacerX USD transform] body={body_index} handle={handle.index} "
                f"path={path} parent={None if parent is None else parent.path}"
            )

        if self._dynamic_bindings:
            body_indices, handles, parent_inverses, geometry_offsets = zip(*self._dynamic_bindings, strict=True)
            self._dynamic_body_indices = wp.array(body_indices, dtype=wp.int32, device=self.device)
            self._dynamic_handle_indices = wp.array(
                [handle.index for handle in handles], dtype=wp.int32, device=self.device
            )
            self._dynamic_transform_count = wp.array(
                [len(self._dynamic_bindings)],
                dtype=wp.int32,
                device=self.device,
            )
            self._dynamic_parent_inverses = wp.array(parent_inverses, dtype=wp.mat44, device=self.device)
            self._dynamic_geometry_offsets = wp.array(geometry_offsets, dtype=wp.mat44, device=self.device)
            self._renderer_from_physics = wp.array([_RENDERER_FROM_PHYSICS], dtype=wp.mat44, device=self.device)
            self._physics_from_renderer = wp.array([physics_from_renderer], dtype=wp.mat44, device=self.device)
            self._dynamic_local_transforms = wp.empty(len(self._dynamic_bindings), dtype=wp.mat44, device=self.device)

    def _key_down(self, key: str) -> bool:
        """Return whether an interactive drive key is held."""
        return bool(hasattr(self.viewer, "is_key_down") and self.viewer.is_key_down(key))

    def _update_drive_input(self) -> None:
        """Upload the two host-polled inputs consumed by the captured control kernel."""
        self._drive_input_host[0] = float(self._key_down("i")) - float(self._key_down("k"))
        self._drive_input_host[1] = float(self._key_down("j")) - float(self._key_down("l"))
        self._drive_input_device.assign(self._drive_input_host)

    def _apply_drive_controls(self) -> None:
        """Ramp and apply vehicle commands entirely on the simulation device."""
        wp.launch(
            _update_vehicle_controls,
            dim=1,
            inputs=[
                self._drive_input_device,
                self._wheel_dofs_device,
                self._steering_target_index,
                self.frame_dt,
                self._steering_travel,
                self._wheel_speed_command,
                self._steering_command,
                self.control.joint_target_q,
                self.control.joint_target_qd,
            ],
            device=self.device,
        )

    def _update_chase_camera(self, *, snap: bool = False) -> None:
        """Update the graph-written chase camera without host readback."""
        wp.launch(
            _update_chase_camera_device,
            dim=1,
            inputs=[
                self.state.body_q,
                self._chassis_body,
                self.frame_dt,
                int(snap),
                self._chase_camera_initialized,
            ],
            outputs=[
                self._chase_camera_forwards,
                self._chase_camera_positions,
                self._chase_camera_targets,
            ],
            device=self.device,
        )

    def _simulate_frame(self) -> None:
        """Apply controls and simulate one graph-capturable frame."""
        self._apply_drive_controls()
        self._simulate()
        if self.chase_camera_enabled:
            self._update_chase_camera()

    def _simulate(self) -> None:
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.collision_pipeline.collide(self.state, self.contacts)
        self.solver.step(
            self.state,
            self.state,
            self.control,
            self.contacts,
            self.frame_dt,
        )

    def step(self) -> None:
        """Advance the imported rigid bodies with PhoenX."""
        if not self.physics_enabled:
            self.sim_time += self.frame_dt
            return
        self._update_drive_input()
        if self.graph is None:
            self._simulate_frame()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def reset_in_place(self) -> None:
        """Restore simulation state without rebuilding the retained USD scene."""
        self.sim_time = 0.0
        if not self.physics_enabled:
            return
        self.state.assign(self.initial_state)
        self.state.clear_forces()
        self.collision_pipeline.reset_contact_matching()
        self._drive_input_host.fill(0.0)
        self._drive_input_device.zero_()
        self._wheel_speed_command.zero_()
        self._steering_command.zero_()
        self._apply_drive_controls()
        self._sync_dynamic_render_transforms()

        if self.chase_camera_enabled:
            self._chase_camera_initialized.zero_()
            self._update_chase_camera(snap=True)

    def _sync_dynamic_render_transforms(self) -> None:
        if not self._dynamic_bindings:
            return
        wp.launch(
            _compute_dynamic_local_transforms,
            dim=len(self._dynamic_bindings),
            inputs=[
                self.state.body_q,
                self._dynamic_body_indices,
                self._dynamic_parent_inverses,
                self._dynamic_geometry_offsets,
                self._renderer_from_physics,
                self._physics_from_renderer,
            ],
            outputs=[self._dynamic_local_transforms],
            device=self.device,
        )
        self.viewer.usd_scene.update_local_transforms_device(
            self._dynamic_transform_count,
            self._dynamic_handle_indices,
            self._dynamic_local_transforms,
        )

    def render(self) -> None:
        """Render the retained USD hierarchy after applying PhoenX poses."""
        usd_visuals_visible = bool(self.viewer.show_visual)
        if usd_visuals_visible != self._usd_visuals_visible:
            self.viewer.usd_scene.set_visible(usd_visuals_visible)
            self._usd_visuals_visible = usd_visuals_visible
        if self.physics_enabled:
            self._sync_dynamic_render_transforms()
            if self._track_layout is not None:
                wp.launch(
                    _gather_body_transforms,
                    dim=len(self._track_body_indices),
                    inputs=[self.state.body_q, self._track_body_indices_device],
                    outputs=[self._track_barrier_xforms],
                    device=self.device,
                )
        self.viewer.begin_frame(self.sim_time)
        if self.physics_enabled:
            self.viewer.log_state(self.state)
            self.viewer.log_contacts(self.contacts, self.state)
            self.viewer.log_shapes(
                "/racerx/ground",
                newton.GeoType.PLANE,
                (GROUND_VISUAL_SIZE, GROUND_VISUAL_SIZE),
                self._ground_visual_xforms,
                colors=self._ground_visual_colors,
                materials=self._ground_visual_materials,
                hidden=not self.viewer.show_ground,
            )
            if self._track_layout is not None:
                self.viewer.log_shapes(
                    "/racerx/track/road",
                    newton.GeoType.BOX,
                    (0.55 * TRACK_BARRIER_SPACING, TRACK_HALF_WIDTH, TRACK_ROAD_HALF_THICKNESS),
                    self._track_road_xforms,
                    colors=self._track_road_colors,
                    materials=self._track_road_materials,
                    hidden=not self.viewer.show_visual,
                )
                self.viewer.log_shapes(
                    "/racerx/track/barriers",
                    newton.GeoType.BOX,
                    TRACK_BARRIER_HALF_EXTENTS,
                    self._track_barrier_xforms,
                    colors=self._track_barrier_colors,
                    materials=self._track_barrier_materials,
                    hidden=not self.viewer.show_visual,
                )
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify the imported physics and retained rendering hierarchy."""
        assert self.viewer.usd_scene is not None
        assert self.viewer.usd_scene.transform_count > 0
        if self.physics_enabled:
            assert self.physics_result is not None
            assert len(self.physics_result["path_body_map"]) > 0
            assert len(self._dynamic_bindings) > 0
            assert np.isfinite(self.state.body_q.numpy()).all()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--rc-model",
        choices=tuple(RACERX_USD_PATHS),
        default=DEFAULT_RC_MODEL,
        help="Bundled RacerX model to load when --usd-path is not specified.",
    )
    parser.add_argument(
        "--usd-path",
        type=str,
        default=None,
        help="Composed USD stage to render and simulate; overrides --rc-model.",
    )
    parser.add_argument(
        "--usd-max-texture-size",
        type=int,
        default=USD_MAX_TEXTURE_SIZE,
        help="Maximum source texture dimension before adaptive atlas fitting; use 0 for no per-texture cap.",
    )
    parser.add_argument(
        "--usd-environment",
        action=argparse.BooleanOptionalAction,
        default=LOAD_USD_ENVIRONMENT,
        help="Load a supported USD DomeLight texture into the OptiX environment.",
    )
    parser.add_argument(
        "--usd-emissive-materials",
        action=argparse.BooleanOptionalAction,
        default=LOAD_USD_EMISSIVE_MATERIALS,
        help="Preserve emissive USD material inputs instead of loading them with zero emission.",
    )
    parser.add_argument(
        "--usd-environment-scale",
        type=float,
        default=1.0,
        help="Brightness multiplier for the USD DomeLight environment texture.",
    )
    parser.add_argument(
        "--usd-camera",
        type=str,
        default=None,
        help="Authored perspective camera path; defaults to the overview camera.",
    )
    parser.add_argument(
        "--chase-camera",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_CHASE_CAMERA,
        help="Fly a game-style camera behind the simulated chassis.",
    )
    parser.add_argument(
        "--physics",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_PHYSICS,
        help="Import UsdPhysics/PhysX data and simulate it with PhoenX.",
    )
    parser.add_argument(
        "--print-physics-data",
        action=argparse.BooleanOptionalAction,
        default=PRINT_PHYSICS_DATA,
        help="Print all authored physics schemas, properties, and import mappings.",
    )
    parser.set_defaults(viewer="optix", paused=START_PAUSED, optix_dlss_quality=OPTIX_DLSS_QUALITY)
    viewer, args = newton.examples.init(parser)
    if args.usd_path is None:
        args.usd_path = str(RACERX_USD_PATHS[args.rc_model])
    if args.usd_max_texture_size == 0:
        args.usd_max_texture_size = None
    example = Example(viewer, args)
    newton.examples.run(example, args)
