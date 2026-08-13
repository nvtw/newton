# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate Marbles USD physics with PhoenX and render the composed OptiX scene.

Newton imports standard UsdPhysics rigid bodies and colliders, while
SchemaResolverPhysx retains vendor-specific attributes for diagnostics. OptiX
keeps the complete visual hierarchy; dynamic body poses are mapped back to that
hierarchy by their exact USD paths.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd

DEFAULT_USD_PATH = Path("/home/twidmer/Documents/Meshes/Marbles/Marbles_Assets_with_physics.usd")

# Easy-to-find example switches. Command-line flags can still override them.
LOAD_USD_ENVIRONMENT = False
ENABLE_PHYSICS = True
PRINT_PHYSICS_DATA = True

_RENDERER_FROM_PHYSICS = np.array(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
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
        "[Marbles USD physics] "
        f"prims={len(records)} schemas={dict(sorted(schema_counts.items()))} "
        f"composition_errors={len(errors)}"
    )
    for error in errors:
        message = error.GetMessage() if hasattr(error, "GetMessage") else str(error)
        print(f"[Marbles USD composition error] {message}")

    for prim, schemas, properties in records:
        print(f"[Marbles USD prim] path={prim.GetPath()} type={prim.GetTypeName() or '<none>'} schemas={schemas}")
        for prop in properties:
            print(f"  {prop.GetName()} = {_property_value(prim, prop)!r}")


def _physics_ignore_paths(stage) -> dict[str, str]:
    """Find trigger-only and incomplete collider prims that Newton cannot simulate."""
    from pxr import UsdGeom, UsdPhysics

    ignored = {}
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
    raise RuntimeError("The USD stage has no suitable authored perspective camera")


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
    """Run imported Marbles rigid-body physics with PhoenX and OptiX."""

    def __init__(self, viewer, args):
        if not hasattr(viewer, "load_scene_from_usd") or not hasattr(viewer, "set_camera_look_at"):
            raise RuntimeError("The Marbles USD example requires the latest --viewer optix")

        self.viewer = viewer
        self.device = wp.get_device()
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0

        usd_path = Path(args.usd_path).expanduser().resolve()
        if not usd_path.is_file():
            raise FileNotFoundError(f"Marbles USD stage not found: {usd_path}")

        if not viewer.load_scene_from_usd(
            str(usd_path),
            max_texture_size=args.usd_max_texture_size,
            load_usd_environment=args.usd_environment,
            usd_environment_scale=args.usd_environment_scale,
        ):
            raise RuntimeError(f"OptiX failed to load USD stage: {usd_path}")

        stage = viewer.usd_scene.stage
        if args.print_physics_data:
            _print_physics_inventory(stage)

        self.physics_enabled = bool(args.physics)
        self.physics_result = None
        self._dynamic_bindings = []
        if self.physics_enabled:
            self._build_physics(stage)
        else:
            self.state = None
            self.graph = None

        camera_path = _select_authored_camera(viewer, args.usd_camera)
        print(
            f"[PhoenX Marbles USD] loaded {usd_path} "
            f"({viewer.usd_scene.transform_count} retained transforms, camera={camera_path})"
        )

    def _build_physics(self, stage) -> None:
        ignored = _physics_ignore_paths(stage)
        authored_unit = _normalize_stage_units_for_newton(stage)

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
        self.physics_result = result
        self._builder = builder

        for path, reason in sorted(ignored.items()):
            print(f"[Marbles USD skipped] path={path} reason={reason}")

        dynamic_paths = []
        kinematic_paths = []
        for path, body_index in sorted(result["path_body_map"].items()):
            if int(builder.body_flags[body_index]) & int(newton.BodyFlags.KINEMATIC):
                motion = "kinematic"
                kinematic_paths.append(path)
            else:
                motion = "dynamic"
                dynamic_paths.append(path)
            print(f"[Marbles USD body] id={body_index} motion={motion} path={path}")

        for path, shape_index in sorted(result["path_shape_map"].items()):
            body_index = int(builder.shape_body[shape_index])
            print(f"[Marbles USD collider] id={shape_index} body={body_index} path={path}")

        for namespace, paths in sorted(result["schema_attrs"].items()):
            for path, attributes in sorted(paths.items()):
                print(f"[Marbles USD {namespace}] path={path} attributes={attributes}")

        print(
            "[Marbles USD import] "
            f"authored_meters_per_unit={authored_unit:g} bodies={builder.body_count} "
            f"dynamic={len(dynamic_paths)} kinematic={len(kinematic_paths)} "
            f"colliders={builder.shape_count} joints={builder.joint_count}"
        )

        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching="sticky",
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)
        self.control = self.model.control()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            collision_pipeline=self.collision_pipeline,
            substeps=4,
            solver_iterations=8,
            velocity_iterations=1,
            default_friction=0.5,
            step_layout="multi_world",
            articulation_mode="maximal",
        )

        self.viewer.set_model(self.model)
        self._build_dynamic_bindings(dynamic_paths)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
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
                print(f"[Marbles USD transform] no render transform for dynamic body {path}")
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
                f"[Marbles USD transform] body={body_index} handle={handle.index} "
                f"path={path} parent={None if parent is None else parent.path}"
            )

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
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def _sync_dynamic_render_transforms(self) -> None:
        if not self._dynamic_bindings:
            return
        poses = self.state.body_q.numpy()
        physics_from_renderer = np.linalg.inv(_RENDERER_FROM_PHYSICS)
        handles = []
        local_matrices = []
        for body_index, handle, parent_inverse, geometry_offset in self._dynamic_bindings:
            rigid_physics = _pose_matrix(poses[body_index])
            rigid_renderer = _RENDERER_FROM_PHYSICS @ rigid_physics @ physics_from_renderer
            handles.append(handle)
            local_matrices.append(parent_inverse @ rigid_renderer @ geometry_offset)
        self.viewer.usd_scene.update_local_transforms(
            handles,
            np.asarray(local_matrices, dtype=np.float32),
        )

    def render(self) -> None:
        """Render the retained USD hierarchy after applying PhoenX poses."""
        if self.physics_enabled:
            self._sync_dynamic_render_transforms()
        self.viewer.begin_frame(self.sim_time)
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
        "--usd-path",
        type=str,
        default=str(DEFAULT_USD_PATH),
        help="Composed USD stage to render and simulate.",
    )
    parser.add_argument(
        "--usd-max-texture-size",
        type=int,
        default=1024,
        help="Maximum loaded texture dimension; use 0 for the source resolution.",
    )
    parser.add_argument(
        "--usd-environment",
        action=argparse.BooleanOptionalAction,
        default=LOAD_USD_ENVIRONMENT,
        help="Load a supported USD DomeLight texture into the OptiX environment.",
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
    parser.set_defaults(viewer="optix")
    viewer, args = newton.examples.init(parser)
    if args.usd_max_texture_size == 0:
        args.usd_max_texture_size = None
    example = Example(viewer, args)
    newton.examples.run(example, args)
