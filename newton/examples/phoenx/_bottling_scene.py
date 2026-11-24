# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read the local Berlin plant's composed geometry and physics into a builder.

This is an example-specific adapter. It does not run Kit, execute embedded
scripts, or change the source layers. OpenUSD resolves references and variants;
Newton receives shared meshes, primitive colliders, rigid bodies, and joints.
"""

import itertools
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.usd
from newton._src.solvers.phoenx.materials import CombineMode
from newton.examples.phoenx._bottling_dispenser import disable_owned_proxies, prepare_dispensers
from newton.examples.phoenx._bottling_fleet import add_rack_shelf
from newton.examples.phoenx._bottling_process import (
    IFM_RECEIVING_BELT,
    PROCESS_BOTTLES,
    add_filling_bottles,
    add_filling_transport,
    filling_route,
    open_pool_passage,
    replace_process_proxies,
    wet_end_layout,
)
from newton.examples.phoenx._bottling_wrap import build_wrapper, correct_wrapper_frame, mesh_components
from newton.solvers import SolverPhoenX


def _value(prim, name, default):
    value = prim.GetAttribute(name).Get()
    return default if value is None else value


def _quat(value):
    return wp.quat(*value.GetImaginary(), value.GetReal())


#: Bottle lanes and packer inlets carry bottles on belts whose top is at this height [m].
_BELT_TOP = 0.95
# The source collides each bottle with 21-sided hulls. Their flat facets lock
# against each other where round glass rolls apart, so the body above the
# base ring collides as the round profile those hulls approximate: (shape,
# radius, bottom, top) in m, from the source's Hull_01..Hull_07 rings; the
# shoulder cone's top is its apex. The base ring hull Hull_00 is kept.
_BOTTLE_PROFILE = (
    ("cylinder", 0.0305, 0.016, 0.1212),
    ("cone", 0.0304, 0.1212, 0.2016),
    ("cylinder", 0.016, 0.1593, 0.2297),
)


def add_round_bottle(builder, body, collision_root, *, xform=None, **kw):
    """Collide a bottle above its base ring as the source's round profile.

    ``xform`` places the bottle's base centre in the body frame; ``kw`` holds
    the remaining shape arguments, such as the glass's ``cfg``.
    """
    pose = wp.transform_identity() if xform is None else xform
    for shape, radius, bottom, top in _BOTTLE_PROFILE:
        center = wp.transform_multiply(pose, wp.transform((0.0, 0.0, 0.5 * (bottom + top)), wp.quat_identity()))
        add = builder.add_shape_cylinder if shape == "cylinder" else builder.add_shape_cone
        label = f"{collision_root}/Round_{shape}_{1000.0 * bottom:.0f}mm/collision"
        add(body, xform=center, radius=radius, half_height=0.5 * (top - bottom), label=label, **kw)


#: Underside of the lower guide-rail band above the belt; the bottle base ring is 16 mm high.
_RAIL_CLEARANCE = 0.022
#: Belt-to-belt transfer gaps up to this length [m] receive a static dead plate.
_MAX_TRANSFER_GAP = 0.05


@dataclass
class PlantScene:
    """Builder and source-to-model maps used by the line controller and checks."""

    builder: newton.ModelBuilder
    bodies: dict = field(default_factory=dict)
    joints: dict = field(default_factory=dict)
    belts: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    bottle_bodies: list = field(default_factory=list)
    case_bodies: list = field(default_factory=list)
    body_mass: dict = field(default_factory=dict)
    joint_frames: list = field(default_factory=list)
    self_collision_scopes: list = field(default_factory=list)
    articulations: list = field(default_factory=list)
    frames: dict = field(default_factory=dict)
    collision_groups: list = field(default_factory=list)
    #: Reconstructed moving shapes that join the source's dynamic line group.
    line_dynamic_shapes: set = field(default_factory=set)
    line_dynamic_group: int = -1

    def collision_masks(self, labels):
        """Map the source collections to compact per-shape collision masks."""
        from pxr import Sdf  # noqa: PLC0415 - OpenUSD is optional.

        membership = np.zeros(len(labels), dtype=np.uint64)
        filtered = np.zeros(len(labels), dtype=np.uint64)
        # The adapter appends /collision or /visual to source geometry paths.
        paths = [Sdf.Path(label).GetParentPath() for label in labels]
        for index, (query, excluded) in enumerate(self.collision_groups):
            included = np.fromiter((query.IsPathIncluded(path) for path in paths), dtype=bool, count=len(paths))
            membership[included] |= np.uint64(1 << index)
            filtered[included] |= np.uint64(excluded)
        if self.line_dynamic_group >= 0 and self.line_dynamic_shapes:
            # The source wrapper is static, so its collections never list the
            # driven reconstruction; vehicle keep-outs must ignore it like
            # every other dynamic machine.
            moving = np.fromiter((label in self.line_dynamic_shapes for label in labels), dtype=bool, count=len(labels))
            membership[moving] |= np.uint64(1 << self.line_dynamic_group)
            filtered[moving] |= np.uint64(self.collision_groups[self.line_dynamic_group][1])
        return membership, filtered


def _mesh_normals(mesh):
    """Keep valid authored normals, rebuilding missing or invalid normals."""
    vertices = np.asarray(mesh.vertices)
    if mesh.normals is not None:
        normals = np.asarray(mesh.normals)
        if normals.shape == vertices.shape and np.isfinite(normals).all():
            return normals
    faces = np.asarray(mesh.indices).reshape(-1, 3)
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    return normals


def _prepare_wrapper_inventory(stage):
    """Start with empty wrapper ports, matching the source live scenario."""
    from pxr import Usd  # noqa: PLC0415 - OpenUSD is optional.

    retired = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name in ("WL_U", "WL_D"):
            path = f"/Plant/WrapLine/Lines/{name}/Loads"
            prim = stage.GetPrimAtPath(path)
            if prim:
                # The source scenario's create_pool() removes these layout
                # stand-ins before runtime inventory owns the empty ports.
                prim.SetActive(False)
                retired.append(path)
    disable_owned_proxies(stage, retired)


def _prepare_wrapper_proxy_ownership(stage):
    """Use the reconstructed frame and authored belts around the wrapper opening."""
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415 - OpenUSD is optional.

    covered = []
    for name in ("WL_U", "WL_D"):
        line = f"/Plant/WrapLine/Lines/{name}"
        wrapper = stage.GetPrimAtPath(line + "/Wrapper")
        if not wrapper or not any(prim.IsA(UsdGeom.Mesh) for prim in wrapper.GetChildren()):
            continue
        covered.append(str(wrapper.GetPath()))
        zones = [stage.GetPrimAtPath(f"{line}/Physics/Conveyors/Z{i}") for i in range(1, 6)]
        if all(
            zone
            and any(
                part.IsA(UsdGeom.Gprim)
                and part.HasAPI(UsdPhysics.CollisionAPI)
                and _value(part, "physics:collisionEnabled", True)
                for part in Usd.PrimRange(zone)
            )
            for zone in zones
        ):
            # The coarse navigation union fills the space under the rotary
            # arm. Its wrapper and conveyor owners already supply colliders.
            covered.append(line + "/Conveyor")
    disable_owned_proxies(stage, covered, scope="/Plant/Physics/NavProxies")


def _warehouse_supply_template(stage, asset_root, name, prototype, empty_path, pallet_mass):
    """Combine the shipping material with an A08 pallet, preserving their individual masses."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415 - OpenUSD is optional.

    # The warehouse prototypes use different package dimensions from the
    # handling assets. Keep the source package mass/material and fit its
    # box to the visible stack above the forkable pallet.
    source = asset_root / f"assets/living_plant/{name}/{name}.usda"
    if not source.is_file():
        raise FileNotFoundError(f"Missing warehouse payload asset: {source}")
    path = "/_Warehouse" + prototype.GetName()
    unit = stage.CreateClassPrim(path)
    unit.GetReferences().AddInternalReference(empty_path)
    cargo = stage.OverridePrim(path + "/Payload")
    cargo.GetReferences().AddReference(str(source.resolve()))
    cargo.GetVariantSet("physics").SetVariantSelection("static")
    mass = UsdPhysics.MassAPI(cargo).GetMassAttr().Get()
    visual_bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    extent = visual_bounds.ComputeRelativeBound(prototype.GetChild("Geo"), prototype.GetChild("Pallet"))
    extent = extent.ComputeAlignedRange()
    size = extent.GetSize()
    box = stage.GetPrimAtPath(path + "/Payload/Collision/box_0")
    box.GetAttribute("xformOp:translate").Set(extent.GetMidpoint())
    box.GetAttribute("xformOp:scale").Set(Gf.Vec3f(size))
    UsdPhysics.MassAPI.Apply(box).CreateDensityAttr(mass / np.prod(size))
    UsdPhysics.MassAPI(unit).CreateMassAttr(pallet_mass + mass)
    return path


def _prepare_warehouse_inventory(stage, asset_root):
    """Recover the authored forkable floor and rack stock without changing its placements."""
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415 - OpenUSD is optional.

    restored = []
    for name in (
        "Warehouse/W1_InspectionHold",
        "Warehouse/W4_EmptyPallets",
        "PackingHall/P2_CartonSupply",
        "PackingHall/P4_Cleaning",
    ):
        zone = stage.GetPrimAtPath("/Plant/LivingPlant/" + name)
        if not zone:
            continue
        for prop in zone.GetChildren():
            asset = _value(prop, "living:asset", "")
            physics = prop.GetVariantSet("physics")
            if name.startswith("Warehouse/") and asset not in ("pallet", "pallet_board_broken"):
                continue
            if "dynamic" in physics.GetVariantNames():
                physics.SetVariantSelection("dynamic")
                if asset == "pallet":
                    prop.GetVariantSet("collision").SetVariantSelection("boxes")
                elif asset in ("carton_blanks_load", "slip_sheet_load"):
                    # The strap extends under the deck through the board gaps. The
                    # solid package starts at local z=0, directly on the pallet.
                    prop.SetInstanceable(False)
                    box = prop.GetChild("Collision").GetChild("box_0")
                    translation = box.GetAttribute("xformOp:translate")
                    scale = box.GetAttribute("xformOp:scale")
                    center, size = translation.Get(), scale.Get()
                    top = center[2] + 0.5 * size[2]
                    if center[2] < 0.5 * size[2]:
                        center[2], size[2] = 0.5 * top, top
                        translation.Set(center)
                        scale.Set(size)
                restored.append(str(prop.GetPath()))
    disable_owned_proxies(stage, restored, scope="/Plant/Physics/NavProxies")

    instancer = UsdGeom.PointInstancer.Get(stage, "/Plant/Logistics/PalletLoads")
    if not instancer:
        return {}, 0.0
    asset = asset_root / "assets/pallet_load/pallet_load.usda"
    if not asset.is_file():
        raise FileNotFoundError(f"Missing warehouse unit-load asset: {asset}")
    template_path = "/_RackUnitLoad"
    template = stage.CreateClassPrim(template_path)
    template.GetReferences().AddReference(str(asset.resolve()))
    # Render the original instanced stock on the moving body, including its wrap and labels.
    stage.OverridePrim(template_path + "/Load").SetActive(False)
    UsdGeom.Imageable(template).MakeInvisible()
    pallet = stage.GetPrimAtPath(template_path + "/Pallet/Geometry/sm_pallet_block_a08_obj_00")
    pallet_mass = UsdPhysics.MassAPI(pallet).GetMassAttr().Get()
    load_mass = UsdPhysics.MassAPI(template).GetMassAttr().Get() - pallet_mass
    cache = UsdGeom.XformCache()
    for root, mass in ((pallet, pallet_mass), (stage.GetPrimAtPath(template_path + "/Collision"), load_mass)):
        # Distribute the component's authored mass over its solid boxes. Equal density
        # for wood and cases would put the combined centre of mass about 4 cm too high.
        boxes = [
            p
            for p in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate)
            if p.IsA(UsdGeom.Cube) and p.HasAPI(UsdPhysics.CollisionAPI) and _value(p, "physics:collisionEnabled", True)
        ]
        volume = sum(
            abs(cache.GetLocalToWorldTransform(p).GetDeterminant()) * UsdGeom.Cube(p).GetSizeAttr().Get() ** 3
            for p in boxes
        )
        for box in boxes:
            UsdPhysics.MassAPI.Apply(box).CreateDensityAttr(mass / volume)

    empty_path = "/_RackEmptyPallet"
    empty = stage.CreateClassPrim(empty_path)
    empty.GetReferences().AddInternalReference(template_path)
    stage.OverridePrim(empty_path + "/Collision").SetActive(False)
    UsdPhysics.MassAPI(empty).CreateMassAttr(pallet_mass)
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["guide"])
    racks = stage.GetPrimAtPath("/Plant/Physics/Racks")
    bounds = [bounds_cache.ComputeWorldBound(row).ComputeAlignedRange() for row in racks.GetChildren()] if racks else []
    supply_assets = {
        "CartonBlanks": "carton_blanks_load",
        "CrownCartons": "crown_carton_load",
        "FilmRolls": "stretch_film_load",
        "TierSheets": "slip_sheet_load",
    }
    templates = {"FinishedGoods": template_path}
    for kind in (
        "LoadFullOpen",
        "LoadPartial",
        "EmptyPallet",
        "WeatheredEmptyPallet",
        "DirtyEmptyPallet",
        "BrokenEmptyPallet",
    ):
        templates[kind] = empty_path
    targets = instancer.GetPrototypesRel().GetTargets()
    transforms = instancer.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
    world = cache.GetLocalToWorldTransform(instancer.GetPrim())
    stock = {}
    floor_positions = []
    for index, (identifier, proto, matrix) in enumerate(
        zip(instancer.GetIdsAttr().Get(), instancer.GetProtoIndicesAttr().Get(), transforms, strict=True)
    ):
        target = targets[proto]
        placement = matrix * world
        rack = any(row.Contains(placement.ExtractTranslation()) for row in bounds)
        if not rack and target.name not in templates and target.name not in supply_assets:
            continue
        label = f"load_{identifier:05d}"
        proxy = stage.GetPrimAtPath("/Plant/Physics/Loads/" + label)
        if not proxy or _value(proxy, "plant:kind", "") != target.name:
            raise ValueError(f"No matching static stock collider for warehouse load {identifier}")
        # The unit-load asset uses its pallet frame. Warehouse prototypes rotate
        # that pallet inside their own frame before the instancer places it.
        prototype = stage.GetPrimAtPath(target)
        pallet_frame = cache.ComputeRelativeTransform(prototype.GetChild("Pallet"), prototype)[0]
        if target.name not in templates:
            if target.name not in supply_assets:
                raise ValueError(f"Unsupported warehouse stock: {target}")
            templates[target.name] = _warehouse_supply_template(
                stage, asset_root, supply_assets[target.name], prototype, empty_path, pallet_mass
            )
        stock_root = "/Plant/Sim/RackStock/" if rack else "/Plant/Sim/FloorStock/"
        body = UsdGeom.Xform.Define(stage, stock_root + label)
        body.GetPrim().GetReferences().AddInternalReference(templates[target.name])
        parent_inverse = cache.GetLocalToWorldTransform(body.GetPrim().GetParent()).GetInverse()
        body.AddTransformOp().Set(pallet_frame * placement * parent_inverse)
        proxy.SetActive(False)
        stock[index] = str(body.GetPath())
        if not rack:
            floor_positions.append(placement.ExtractTranslation())
    # Each packaging store has a single solid block representing the whole
    # stockpile. Its individual packages and pallets now supply that collision.
    equipment = stage.GetPrimAtPath("/Plant/Physics/Equipment")
    if equipment:
        for proxy in equipment.GetChildren():
            if _value(proxy, "plant:kind", "") == "packaging_store":
                extent = bounds_cache.ComputeWorldBound(proxy).ComputeAlignedRange()
                if any(extent.Contains(position) for position in floor_positions):
                    proxy.SetActive(False)
    # The source forklift unit carries two layers of 17 filled cases.
    return stock, load_mass / 34


def _prepare_case_inventory(stage, cartons_per_packer):
    """Extend finite carton stock on explicit offsite staging platforms.

    Preserve authored cases and their positions. Additional reserves reference
    the same open carton, including its four flap joints, on separated static
    supports outside the source plant bounds. This does not implement physical
    delivery from staging to the existing case supply entrance.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415 - OpenUSD is optional.

    if cartons_per_packer is not None and cartons_per_packer < 1:
        raise ValueError("cartons_per_packer must be positive")
    root = stage.GetPrimAtPath("/Plant/Sim/Product/Cases")
    if not root:
        return {}, []
    existing = {}
    for prim in root.GetChildren():
        cell, _, name = prim.GetName().rpartition("_")
        if name == "C0" or (name.startswith("E") and name[1:].isdigit()):
            existing.setdefault(cell, []).append(name)
    bounds = None
    inventory, staging = {}, []
    for cell, names in sorted(existing.items()):
        if "C0" not in names:
            raise ValueError(f"{cell}: initial C0 carton is missing")
        reserves = sorted((name for name in names if name != "C0"), key=lambda name: int(name[1:]))
        requested = (
            ["C0", *reserves]
            if cartons_per_packer is None
            else ["C0", *(f"E{i:02}" for i in range(cartons_per_packer - 1))]
        )
        inventory[cell] = requested
        for name in names:
            if name not in requested:
                stage.GetPrimAtPath(root.GetPath().AppendChild(f"{cell}_{name}")).SetActive(False)
        additions = [name for name in requested if name not in names]
        if not additions:
            continue
        template = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{cell}_E00"))
        if not template:
            raise ValueError(f"{cell}: cannot extend stock without an authored E00 carton")
        if bounds is None:
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
            plant_range = bounds.ComputeWorldBound(stage.GetPrimAtPath("/Plant")).ComputeAlignedRange()
            next_x = float(plant_range.GetMax()[0]) + 1.0
        template_range = bounds.ComputeWorldBound(template).ComputeAlignedRange()
        size = np.asarray(template_range.GetSize())
        pitch = size[:2] + 0.1
        columns = min(7, len(additions))
        rows = math.ceil(len(additions) / columns)
        extent = np.array([columns, rows]) * pitch
        corner = np.array([next_x, float(plant_range.GetMin()[1])])
        platform_path = f"/Plant/Sim/CaseStaging/{cell}"
        platform = UsdGeom.Cube.Define(stage, platform_path)
        platform.CreateSizeAttr(2.0)
        platform.AddTranslateOp().Set(Gf.Vec3d(*(corner + extent * 0.5), 0.3))
        platform.AddScaleOp().Set(Gf.Vec3d(*(extent * 0.5), 0.3))
        UsdPhysics.CollisionAPI.Apply(platform.GetPrim())
        template_center = np.asarray(template_range.GetMidpoint())
        for index, name in enumerate(additions):
            path = root.GetPath().AppendChild(f"{cell}_{name}")
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddInternalReference(template.GetPath())
            target = corner + (np.array([index % columns, index // columns]) + 0.5) * pitch
            shift = np.array([*(target - template_center[:2]), 0.6005 - template_range.GetMin()[2]])
            # Source cases have a translate then rotate stack, with no rotated
            # ancestor; overriding that translation preserves all local joints.
            translate = prim.GetAttribute("xformOp:translate")
            if not translate or not translate.HasAuthoredValueOpinion():
                raise ValueError(f"{cell}: reserve template has no authored translation")
            translate.Set(Gf.Vec3d(*(np.asarray(translate.Get()) + shift)))
        staging.append({"cell": cell, "platform": platform_path, "cartons": additions})
        next_x += float(extent[0]) + 1.0
    return inventory, staging


def load_plant(
    asset_root: Path,
    *,
    cutaway=True,
    visuals=True,
    motor_off=False,
    cartons_per_packer=None,
    filling_room_physics=False,
) -> PlantScene:
    """Build the full line from local USD, retaining its authored SI properties."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415 - OpenUSD is optional.

    started = time.perf_counter()
    path = asset_root / "scenarios/line_sim/line_sim.usda"
    if not path.is_file():
        raise FileNotFoundError(
            f"No running-line stage at {path}. Download the documented 2.33 GB simulation subset and pass --asset-root."
        )
    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())
    fleet = asset_root / "scenarios/logistics/logistics_fleet.usda"
    if fleet.is_file():
        stage.GetSessionLayer().subLayerPaths.append(str(fleet.resolve()))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("The Berlin plant adapter expects its metre-scaled, Z-up source stage")
    plant = stage.GetPrimAtPath("/Plant")
    if cutaway:
        plant.GetVariantSet("view").SetVariantSelection("cutaway")
    _prepare_wrapper_inventory(stage)
    _prepare_wrapper_proxy_ownership(stage)
    stock_instances, rack_case_mass = _prepare_warehouse_inventory(stage, asset_root)
    prepare_dispensers(stage)
    case_inventory, case_staging = _prepare_case_inventory(stage, cartons_per_packer)
    cache = UsdGeom.XformCache()
    process_path = filling_route(stage, cache) if filling_room_physics else None
    if process_path is not None:
        replace_process_proxies(stage, cache, process_path[0, 2])
        layout = wet_end_layout(stage, cache, process_path)
        open_pool_passage(stage, process_path, layout["pool"][1][-1])
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    scene = PlantScene(newton.ModelBuilder())
    builder = scene.builder
    SolverPhoenX.register_custom_attributes(builder)
    # Carrier-box walls authored for convex decomposition, as PhysX cooks them.
    decomposed = []
    collision_table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(stage)
    collision_groups = collision_table.GetGroups()
    if len(collision_groups) > 64:
        raise ValueError("The plant adapter supports at most 64 USD collision groups")
    for group in collision_groups:
        source_group = UsdPhysics.CollisionGroup(stage.GetPrimAtPath(group))
        if source_group.GetInvertFilteredGroupsAttr().Get():
            raise ValueError("The plant adapter does not support inverted collision groups")
        query = source_group.GetCollidersCollectionAPI().ComputeMembershipQuery()
        excluded = sum(
            1 << i for i, other in enumerate(collision_groups) if not collision_table.IsCollisionEnabled(group, other)
        )
        scene.collision_groups.append((query, excluded))
        if group.name == "line_dynamic":
            scene.line_dynamic_group = len(scene.collision_groups) - 1
    for prim in prims:
        if _value(prim, "physxArticulation:enabledSelfCollisions", None) is False:
            root = prim
            if prim.IsA(UsdPhysics.Joint):
                # A fixed-root articulation may put its joint in a separate
                # folder. Its root body identifies the enclosing mechanism.
                targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
                root = stage.GetPrimAtPath(targets[0]).GetParent() if targets else prim.GetParent()
            scene.self_collision_scopes.append(str(root.GetPath()))
    scene.metadata = dict(stage.GetRootLayer().customLayerData)
    scene.metadata["case_inventory"] = case_inventory
    scene.metadata["dynamic_rack_loads"] = sum(
        path.startswith("/Plant/Sim/RackStock/") for path in stock_instances.values()
    )
    if case_staging:
        scene.metadata["case_staging"] = case_staging
        scene.metadata["case_supply_limitation"] = (
            "Additional finite stock rests on offsite staging platforms. The existing controller transfers "
            "reserve poses to the case inlet; physical upstream delivery remains unimplemented."
        )
    for path in ("/Plant/Line/Packers", "/Plant/Line/Palletizers", "/Plant/Line"):
        parent = stage.GetPrimAtPath(path)
        if parent:
            for prim in parent.GetChildren():
                scene.frames[str(prim.GetPath())] = np.asarray(cache.GetLocalToWorldTransform(prim)).copy()
    matrices = {}
    parked = set()
    body_prims = {}
    nominal = {}
    wrapper_belt_zones = {}
    for line in ("WL_U", "WL_D"):
        sidecar = asset_root / f"plant/lines_1x8/wrapline/{line}.json"
        if not sidecar.is_file():
            continue
        for zone, spec in json.loads(sidecar.read_text())["zones"].items():
            label = spec["prim"]
            wrapper_belt_zones[label] = {"line": line, "zone": zone, "speed_mps": spec["speed_mps"]}
            if not motor_off:
                nominal[label] = (tuple(np.asarray(spec["local_axis"]) * spec["speed_mps"]), (0.0, 0.0, 0.0))
    scene.metadata["wrapper_belt_zones"] = wrapper_belt_zones
    vendor = asset_root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
    if not motor_off:
        interface = json.loads((vendor / "ifm_v01/interface.json").read_text())
        for belt in interface["belts"]:
            speed = interface["speeds_mps"][belt["speed_key"]]
            linear, angular = (speed, 0.0, 0.0), (0.0, 0.0, 0.0)
            if belt["kind"] == "angular":
                linear = (0.0, 0.0, 0.0)
                angular = (0.0, 0.0, belt.get("omega_sign", 1.0) * speed / belt["radius"])
            nominal[belt["prim"].replace(interface["root"], "/Plant/Line/IFM", 1)] = (linear, angular)
        interface = json.loads((vendor / "ctp_v01/interface.json").read_text())
        for module in interface["modules"].values():
            for belt in module["belts"]:
                speed = belt["speed"]
                skew = math.radians(interface["zones"][belt["zone"]].get("skew_deg", 0.0))
                linear, angular = (speed * math.cos(skew), speed * math.sin(skew), 0.0), (0.0, 0.0, 0.0)
                if belt["kind"] == "angular":
                    linear = (0.0, 0.0, 0.0)
                    angular = (0.0, 0.0, belt.get("omega_sign", 1.0) * speed / belt["radius"])
                nominal["/Plant/Line/CTP" + belt["prim"]] = (linear, angular)

    def transform(matrix):
        parts = Gf.Transform(matrix)
        return wp.transform(parts.GetTranslation(), _quat(parts.GetRotation().GetQuat())), np.array(parts.GetScale())

    for prim in prims:
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not _value(prim, "physics:rigidBodyEnabled", True):
            continue
        label = str(prim.GetPath())
        matrix = cache.GetLocalToWorldTransform(prim)
        pose, scale = transform(matrix)
        if label.startswith("/Plant/Sim/Product/") and pose.p[2] < -2.0:
            parked.add(label)
            continue
        mass = float(_value(prim, "physics:mass", 0.0))
        inertia = np.asarray(_value(prim, "physics:diagonalInertia", (0.0, 0.0, 0.0)))
        com = np.asarray(_value(prim, "physics:centerOfMass", (0.0, 0.0, 0.0)))
        if not np.isfinite(com).all():
            com = np.zeros(3)
        axes = _quat(_value(prim, "physics:principalAxes", Gf.Quatf(1.0)))
        rotation = np.asarray(wp.quat_to_matrix(axes)).reshape(3, 3)
        tensor = rotation @ np.diag(inertia) @ rotation.T
        explicit = mass > 0 and np.all(inertia > 0)
        kinematic = bool(_value(prim, "physics:kinematicEnabled", False))
        if kinematic and not explicit:
            # Kinematic conveyor surfaces have infinite effective mass. Their
            # open triangle sheets have no volume from which to infer inertia.
            tensor = np.eye(3)
            explicit = True
        body = builder.add_link(
            xform=pose,
            mass=(mass or 1.0) if explicit else 0.0,
            inertia=wp.mat33(tensor) if explicit else None,
            com=wp.vec3(com * scale),
            lock_inertia=bool(explicit),
            is_kinematic=kinematic,
            label=label,
        )
        scene.bodies[label] = body
        scene.body_mass[body] = mass
        matrices[body] = Gf.Matrix4d(1.0)
        matrices[body].SetRotate(Gf.Quatd(pose.q[3], Gf.Vec3d(*pose.q[:3])))
        matrices[body].SetTranslateOnly(Gf.Vec3d(*pose.p))
        body_prims[body] = prim
        if label.startswith("/Plant/Sim/Product/Bottles/"):
            scene.bottle_bodies.append(body)
        elif label.startswith("/Plant/Sim/Product/Cases/"):
            scene.case_bodies.append(body)

    texture_files = {}
    if visuals:
        for file in asset_root.rglob("*"):
            if file.is_file():
                texture_files.setdefault(file.name, file)
    atlases = {}

    def resolve_texture(mesh):
        texture = mesh.texture
        if not isinstance(texture, str) or Path(texture).is_file():
            return
        name = Path(texture).name
        if "<UDIM>" not in name:
            if name in texture_files:
                mesh.texture = str(texture_files[name])
            return
        if name not in atlases:
            from PIL import Image  # noqa: PLC0415 - Decode only when UDIM textures are used.

            before, after = name.split("<UDIM>")
            tiles = {}
            for filename, file in texture_files.items():
                number = filename[len(before) : -len(after)] if after else filename[len(before) :]
                if filename.startswith(before) and filename.endswith(after) and number.isdigit():
                    tiles[int(number)] = file
            if not tiles:
                raise FileNotFoundError(f"No local tiles for source texture {texture}")
            coordinates = {tile: ((tile - 1001) % 10, (tile - 1001) // 10) for tile in tiles}
            columns = max(u for u, _ in coordinates.values()) + 1
            rows = max(v for _, v in coordinates.values()) + 1
            with Image.open(next(iter(tiles.values()))) as image:
                width, height = image.size
            atlas = np.zeros((rows * height, columns * width, 4), dtype=np.uint8)
            for tile, file in tiles.items():
                u, v = coordinates[tile]
                y, x = (rows - v - 1) * height, u * width
                with Image.open(file) as image:
                    if image.size != (width, height):
                        raise ValueError(f"Inconsistent UDIM tile size: {file}")
                    atlas[y : y + height, x : x + width] = np.asarray(image.convert("RGBA"))
            atlases[name] = atlas, columns, rows
        mesh.texture, columns, rows = atlases[name]
        matrix = np.asarray(mesh.texture_transform).copy()
        matrix[0] /= columns
        matrix[1] /= rows
        mesh.texture_transform = matrix

    meshes = {}
    owner_cache = {}

    def owner(prim):
        label = str(prim.GetPath())
        if label not in owner_cache:
            if label in scene.bodies:
                result = scene.bodies[label]
            elif label in parked:
                result = -2
            elif prim.IsPseudoRoot():
                result = -1
            else:
                result = owner(prim.GetParent())
            owner_cache[label] = result
        return owner_cache[label]

    def moving_mesh(mesh, velocity):
        mesh = newton.Mesh(mesh.vertices, mesh.indices, compute_inertia=False, enable_surface_velocity=True)
        linear, angular, shape_pose, scale, belt_label = velocity
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(shape_pose[3:]))).reshape(3, 3)
        points = np.asarray(mesh.vertices) * np.asarray(scale) @ rotation.T + np.asarray(shape_pose[:3])
        # Newton rotates and scales vertex velocities. Undo the shape transform
        # so the authored speed remains in metres per second in the body frame.
        speeds = np.asarray(linear) + np.cross(np.asarray(angular), points)
        speeds = (speeds @ rotation) / np.asarray(scale)
        scene.belts.append((mesh, speeds.astype(np.float32), belt_label))
        return mesh

    def mesh_for(prim, *, visual, velocity=None):
        points = prim.GetAttribute("points")
        stack = points.GetPropertyStack()
        source = (stack[0].layer.identifier, str(stack[0].path)) if stack else str(prim.GetPath())
        # Instance proxies have distinct paths but share the same authored geometry.
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0] if visual else None
        material_key = None
        if material:
            material_stack = material.GetPrim().GetPrimStack()
            material_key = tuple((s.layer.identifier, str(s.path)) for s in material_stack)
        key = (source, visual, material_key, velocity)
        if key not in meshes:
            mesh = newton.usd.get_mesh(
                prim,
                load_normals=visual,
                load_uvs=visual,
                compute_inertia=False,
                load_visual_materials=visual,
            )
            if visual:
                resolve_texture(mesh)
            if visual and (
                mesh.normals is None or mesh.normals.shape != mesh.vertices.shape or not np.isfinite(mesh.normals).all()
            ):
                mesh = newton.Mesh(
                    mesh.vertices,
                    mesh.indices,
                    normals=_mesh_normals(mesh),
                    uvs=mesh.uvs,
                    color=mesh.color,
                    roughness=mesh.roughness,
                    metallic=mesh.metallic,
                    texture=mesh.texture,
                    texture_transform=mesh.texture_transform,
                    opacity=mesh.opacity,
                    compute_inertia=False,
                )
            if not visual and velocity is None:
                # Several source actuators bake metre-scale offsets into tiny
                # collider vertices. Integrating their moments about zero loses
                # enough precision to produce negative central inertias. Compute
                # moments near the solid and then translate only its COM back.
                vertices = np.asarray(mesh.vertices)
                center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
                centered = newton.Mesh(vertices - center, mesh.indices)
                mesh.has_inertia = True
                mesh.mass = centered.mass
                mesh.com = wp.vec3(np.asarray(centered.com) + center)
                mesh.inertia = centered.inertia
            if velocity is not None:
                if _value(prim, "physics:approximation", "none") == "convexHull":
                    # Cook the authored hull before assigning its velocity field.
                    mesh = mesh.compute_convex_hull()
                mesh = moving_mesh(mesh, velocity)
            meshes[key] = mesh
        return meshes[key]

    def add_geometry(prim, matrix, body, *, visual, collision):
        local = matrix * matrices[body].GetInverse() if body >= 0 else matrix
        pose, scale = transform(local)
        cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0 if collision else 0.0,
            gap=max(0.0, float(_value(prim, "physxCollision:contactOffset", 0.001))),
            margin=float(_value(prim, "physxCollision:restOffset", 0.0)),
            has_shape_collision=collision,
            has_particle_collision=False,
            is_visible=visual,
        )
        custom_attributes = None
        if collision:
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
            if material:
                cfg.mu = float(_value(material.GetPrim(), "physics:dynamicFriction", 0.5))
                cfg.mu_static = max(cfg.mu, float(_value(material.GetPrim(), "physics:staticFriction", cfg.mu)))
                cfg.restitution = float(_value(material.GetPrim(), "physics:restitution", 0.0))
                mode = material.GetPrim().GetAttribute("physxMaterial:restitutionCombineMode")
                if mode and mode.HasAuthoredValueOpinion():
                    custom_attributes = {"phoenx:restitution_combine_mode": int(CombineMode[mode.Get().upper()])}
                cfg.density = float(_value(material.GetPrim(), "physics:density", 1000.0)) or 1000.0
        kind = prim.GetTypeName()
        kw = {"xform": pose, "cfg": cfg, "label": str(prim.GetPath()) + ("/visual" if visual else "/collision")}
        if custom_attributes is not None:
            kw["custom_attributes"] = custom_attributes
        path = str(prim.GetPath())
        if collision and path.startswith("/Plant/Sim/Product/Bottles/") and "/Collision/Hull_" in path:
            if body == scene.bottle_bodies[0]:
                # Photo-eyes trace the source's hulls, on which its PLCs were tuned.
                hull = mesh_for(prim, visual=False, velocity=None)
                scene.metadata.setdefault("bottle_sensor_hulls", []).append((hull, pose, scale))
            if not path.endswith("/Hull_00"):
                if path.endswith("/Hull_01"):
                    del kw["label"]
                    add_round_bottle(builder, body, path.rsplit("/", 1)[0], **kw)
                return
        color = _value(prim, "primvars:displayColor", None)
        if visual and color is not None and len(color):
            kw["color"] = tuple(color[0])
        velocity = None
        if collision and body >= 0:
            parent = body_prims[body]
            enabled = parent.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled")
            # A stopped conveyor still needs a velocity field for its PLC to
            # start it later; the authored enable flag controls its initial speed.
            if enabled:
                label = str(parent.GetPath())
                linear, angular = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                if label in wrapper_belt_zones and not motor_off:
                    # Wrapper USD starts with zero speed. Its sidecar supplies
                    # the operating field; separate controller channels keep it stopped.
                    linear, angular = nominal[label]
                elif enabled.Get() and not motor_off:
                    if label in nominal:
                        linear, angular = nominal[label]
                    else:
                        linear = _value(parent, "physxSurfaceVelocity:surfaceVelocity", linear)
                        angular = np.radians(_value(parent, "physxSurfaceVelocity:surfaceAngularVelocity", angular))
                        body_pose, body_scale = transform(cache.GetLocalToWorldTransform(parent))
                        if _value(parent, "physxSurfaceVelocity:surfaceVelocityLocalSpace", True):
                            linear = np.asarray(linear) * body_scale
                        else:
                            rotation = np.asarray(wp.quat_to_matrix(body_pose.q)).reshape(3, 3)
                            linear, angular = np.asarray(linear) @ rotation, np.asarray(angular) @ rotation
                    if label.startswith("/Plant/Line/BottleLanes/"):
                        linear = (float(_value(parent, "bottle_lanes:speedNominal", 0.2)), 0.0, 0.0)
                    if "/Packers/" in label and label.endswith(("/Belts/IN_A", "/Belts/IN_B")):
                        linear = (0.2 if label.endswith("IN_A") else 0.4, 0.0, 0.0)
                velocity = (tuple(linear), tuple(angular), tuple(pose.p) + tuple(pose.q), tuple(scale), label)
        if kind == "Mesh":
            mesh = mesh_for(prim, visual=visual, velocity=velocity)
            approximation = _value(prim, "physics:approximation", "none")
            if collision and approximation == "convexHull" and velocity is None and len(mesh.vertices) == 8:
                # Avoid float32 midpoint rounding rejecting exact box corners.
                points = np.asarray(mesh.vertices, dtype=np.float64)
                center = (points.min(axis=0) + points.max(axis=0)) / 2.0
                half = (points.max(axis=0) - points.min(axis=0)) / 2.0
                if np.all(half > 0.0) and np.allclose(np.abs(points - center), half, rtol=0.0, atol=1e-8):
                    # These are exactly boxes, rather than approximations of
                    # the source hulls; use their analytic contact path.
                    kw["xform"] = wp.transform_multiply(pose, wp.transform(center * scale, wp.quat_identity()))
                    h = np.abs(scale) * half
                    builder.add_shape_box(body, hx=h[0], hy=h[1], hz=h[2], **kw)
                    return
            add = (
                builder.add_shape_convex_hull
                if collision and approximation == "convexHull" and velocity is None
                else builder.add_shape_mesh
            )
            if collision and approximation == "convexDecomposition" and "/Geom/Walls/" in kw["label"] and body >= 0:
                decomposed.append(builder.shape_count)
            add(body, mesh=mesh, scale=scale, **kw)
        elif kind == "Cube":
            density = _value(prim, "physics:density", 0.0)
            if collision and density > 0.0:
                cfg.density = density
            half = float(_value(prim, "size", 2.0)) / 2.0
            if velocity is not None:
                mesh = moving_mesh(newton.Mesh.create_box(half, compute_inertia=False), velocity)
                builder.add_shape_mesh(body, mesh=mesh, scale=scale, **kw)
            else:
                half = half * np.abs(scale)
                label = kw["label"]
                if collision and ("/Inlet/Colliders/" in label or ("/BottleLanes/" in label and "/Guides" in label)):
                    pose = kw["xform"]
                    if float(pose.p[2]) - half[2] - _BELT_TOP < _RAIL_CLEARANCE:
                        # The source's lower rail band starts 18 mm above the belt, 2 mm
                        # above the bottle's 16 mm base ring. A rocking bottle's rim slid
                        # under it and was levered over; raise the band to clear the ring.
                        lift = _RAIL_CLEARANCE - (float(pose.p[2]) - half[2] - _BELT_TOP)
                        kw["xform"] = wp.transform(wp.vec3(pose.p[0], pose.p[1], pose.p[2] + lift), pose.q)
                        scene.metadata.setdefault("rail_band_lift_m", lift)
                builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], **kw)
        elif kind == "Sphere":
            builder.add_shape_sphere(body, radius=float(_value(prim, "radius", 1.0)) * max(abs(scale)), **kw)
        elif kind == "Plane":
            axis = "XYZ".index(_value(prim, "axis", "Z"))
            if axis != 2:
                rotation = wp.quat_from_axis_angle(
                    wp.vec3(0.0, 1.0, 0.0) if axis == 0 else wp.vec3(1.0, 0.0, 0.0),
                    math.pi / 2.0 if axis == 0 else -math.pi / 2.0,
                )
                kw["xform"] = wp.transform(pose.p, pose.q * rotation)
            builder.add_shape_plane(body=body, width=0.0, length=0.0, **kw)
        elif kind in ("Cylinder", "Capsule", "Cone"):
            axis = _value(prim, "axis", "Z")
            i = "XYZ".index(axis)
            radius = float(_value(prim, "radius", 1.0)) * max(abs(scale[j]) for j in range(3) if j != i)
            half_height = float(_value(prim, "height", 2.0)) * abs(scale[i]) / 2.0
            add = {
                "Cylinder": builder.add_shape_cylinder,
                "Capsule": builder.add_shape_capsule,
                "Cone": builder.add_shape_cone,
            }[kind]
            if i != 2:
                rotation = (
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0)
                    if i == 0
                    else wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)
                )
                kw["xform"] = wp.transform(pose.p, pose.q * rotation)
            add(body, radius=radius, half_height=half_height, **kw)
        else:
            raise ValueError(f"Unsupported geometry {kind} at {prim.GetPath()}")

    wrappers = {}
    wrapper_proxies = {}
    geometry_types = {"Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"}
    instancers = [UsdGeom.PointInstancer(p) for p in prims if p.IsA(UsdGeom.PointInstancer)]
    prototypes = {path for instancer in instancers for path in instancer.GetPrototypesRel().GetTargets()}
    for prim in prims:
        if prim.GetTypeName() not in geometry_types:
            continue
        # Walk this prim's ancestors instead of scanning every instancer
        # prototype for every geometry prim in the factory.
        if not prototypes.isdisjoint(prim.GetPath().GetPrefixes()):
            continue
        body = owner(prim)
        if body == -2:
            continue
        imageable = UsdGeom.Imageable(prim)
        visible = (
            visuals
            and imageable.ComputeVisibility() != "invisible"
            and imageable.ComputePurpose() not in ("guide", "proxy")
        )
        collision = prim.HasAPI(UsdPhysics.CollisionAPI) and _value(prim, "physics:collisionEnabled", True)
        matrix = cache.GetLocalToWorldTransform(prim)
        parent_path = str(prim.GetParent().GetPath())
        if (
            prim.GetTypeName() == "Mesh"
            and parent_path.startswith("/Plant/WrapLine/Lines/")
            and parent_path.endswith("/Wrapper")
        ):
            if collision:
                raise ValueError(f"Wrapper visual mesh unexpectedly has authored collision: {prim.GetPath()}")
            mesh = mesh_for(prim, visual=visuals)
            world = np.asarray(matrix)
            for part in mesh_components(mesh):
                vertices = part.vertices @ world[:3, :3] + world[3, :3]
                component = part.copy(vertices=vertices)
                if part.normals is not None:
                    normals = part.normals @ np.linalg.inv(world[:3, :3]).T
                    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                    component.normals[:] = normals / np.maximum(lengths, 1.0e-12)
                wrappers.setdefault(parent_path, []).append((prim.GetName(), component))
            continue
        if (
            collision
            and parent_path.startswith("/Plant/WrapLine/Lines/")
            and parent_path.endswith("/Physics/Static")
            and prim.GetName() in ("portal_da", "portal_df", "portal_ua", "portal_uf")
        ):
            # These four coarse posts are covered by the reconstructed frame
            # and feet. Defer them until their replacement is known to exist.
            wrapper = parent_path.removesuffix("/Physics/Static") + "/Wrapper"
            wrapper_proxies.setdefault(wrapper, []).append((prim, matrix, body))
            collision = False
        if collision:
            add_geometry(prim, matrix, body, visual=False, collision=True)
        if visible:
            add_geometry(prim, matrix, body, visual=True, collision=False)

    def prototype_meshes(root):
        """Bake a prototype once per material, retaining the placement instancing."""
        inverse = cache.GetLocalToWorldTransform(root).GetInverse()
        batches = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            kind = prim.GetTypeName()
            if kind not in geometry_types:
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy"):
                continue
            if kind == "Mesh":
                mesh = mesh_for(prim, visual=True)
            elif kind == "Cube":
                h = float(_value(prim, "size", 2.0)) / 2.0
                mesh = newton.Mesh.create_box(h, h, h, compute_inertia=False)
            elif kind == "Sphere":
                mesh = newton.Mesh.create_sphere(float(_value(prim, "radius", 1.0)), compute_inertia=False)
            else:
                create = {
                    "Cylinder": newton.Mesh.create_cylinder,
                    "Capsule": newton.Mesh.create_capsule,
                    "Cone": newton.Mesh.create_cone,
                }[kind]
                mesh = create(
                    float(_value(prim, "radius", 1.0)),
                    float(_value(prim, "height", 2.0)) / 2.0,
                    up_axis=newton.Axis("XYZ".index(_value(prim, "axis", "Z"))),
                    compute_inertia=False,
                )
            color = _value(prim, "primvars:displayColor", None)
            color = tuple(color[0]) if color is not None and len(color) else tuple(mesh.color or (0.65, 0.65, 0.65))
            key = (color, mesh.roughness, mesh.metallic, mesh.texture_hash, mesh.texture_transform, mesh.opacity)
            if key not in batches:
                batches[key] = ([], [], [], [], mesh)
            positions, faces, normals, uvs, _ = batches[key]
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
            linear = matrix[:3, :3]
            vertices = np.asarray(mesh.vertices) @ linear + matrix[3, :3]
            indices = np.asarray(mesh.indices).reshape(-1, 3)
            if np.linalg.det(linear) < 0:
                indices = indices[:, ::-1]
            offset = sum(len(p) for p in positions)
            positions.append(vertices)
            faces.append(indices + offset)
            n = _mesh_normals(mesh) @ np.linalg.inv(linear).T
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
            normals.append(n)
            uvs.append(np.zeros((len(vertices), 2)) if mesh.uvs is None else mesh.uvs)
        result = []
        for key, (positions, faces, normals, uvs, original) in batches.items():
            result.append(
                newton.Mesh(
                    np.concatenate(positions),
                    np.concatenate(faces).ravel(),
                    normals=np.concatenate(normals),
                    uvs=np.concatenate(uvs),
                    compute_inertia=False,
                    color=key[0],
                    roughness=original.roughness,
                    metallic=original.metallic,
                    texture=original.texture,
                    texture_transform=original.texture_transform,
                    opacity=original.opacity,
                )
            )
        return result

    prototype_cache = {}
    visual_cfg = builder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)

    def add_instance_visuals(prim, body, pose, scale, label):
        if prim not in prototype_cache:
            prototype_cache[prim] = prototype_meshes(prim)
        for mesh in prototype_cache[prim]:
            builder.add_shape_mesh(body, mesh=mesh, xform=pose, scale=scale, cfg=visual_cfg, label=label)

    # Open loads can shed individual cartons. Each closed carton uses its
    # outside dimensions and its share of the source filled unit-load mass.
    case_dims = None
    case_layouts = {}
    if process_path is not None:
        receiving_belt = nominal.get(IFM_RECEIVING_BELT)
        inlet_speed = 0.1 if receiving_belt is None else float(np.linalg.norm(receiving_belt[0]))
        add_filling_transport(scene, process_path, motor_off=motor_off, layout=layout, inlet_speed=inlet_speed)
        add_filling_bottles(scene, process_path, motor_off=motor_off, layout=layout)
    for instancer in instancers:
        if process_path is not None and str(instancer.GetPath()) == PROCESS_BOTTLES:
            continue
        stock = stock_instances if str(instancer.GetPath()) == "/Plant/Logistics/PalletLoads" else {}
        visible = visuals and instancer.ComputeVisibility() != "invisible"
        if not visible and not stock:
            continue
        targets = instancer.GetPrototypesRel().GetTargets()
        indices = instancer.GetProtoIndicesAttr().Get()
        transforms = instancer.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
        world = cache.GetLocalToWorldTransform(instancer.GetPrim())
        label = str(instancer.GetPath()) + "/instance"
        for instance, (index, matrix) in enumerate(zip(indices, transforms, strict=True)):
            path = stock.get(instance)
            body = scene.bodies[path] if path else -1
            placement = matrix * world
            target = targets[index]
            prototype = stage.GetPrimAtPath(target)
            if path and target.name in ("LoadFullOpen", "LoadPartial"):
                if case_dims is None:
                    case_dims = np.asarray(
                        json.loads((asset_root / "assets/beerbox/dimensions.json").read_text())["outside_m"]
                    )
                    carton_cfg = builder.ShapeConfig(
                        density=rack_case_mass / np.prod(case_dims), gap=0.002, is_visible=False
                    )
                if target not in case_layouts:
                    case_layouts[target] = [
                        (case, cache.ComputeRelativeTransform(case, prototype)[0])
                        for case in prototype.GetChild("Cases").GetChildren()
                    ]
                for case, local in case_layouts[target]:
                    pose, scale = transform(local * placement)
                    case_path = f"{path}/Cases/{case.GetName()}"
                    case_body = builder.add_link(xform=pose, label=case_path)
                    scene.bodies[case_path] = case_body
                    half = 0.5 * case_dims * scale
                    builder.add_shape_box(
                        case_body,
                        xform=wp.transform((0, 0, half[2]), wp.quat_identity()),
                        hx=half[0],
                        hy=half[1],
                        hz=half[2],
                        cfg=carton_cfg,
                        label=case_path + "/collision",
                    )
                    if visible:
                        add_instance_visuals(case, case_body, wp.transform_identity(), scale, label)
                if visible:
                    pallet = prototype.GetChild("Pallet")
                    local = cache.ComputeRelativeTransform(pallet, prototype)[0]
                    pose, scale = transform(local * placement * matrices[body].GetInverse())
                    add_instance_visuals(pallet, body, pose, scale, label)
            elif visible:
                pose, scale = transform(placement * matrices[body].GetInverse() if path else placement)
                add_instance_visuals(prototype, body, pose, scale, label)

    plans = []
    gripper_templates = [p.GetPath() for p in prims if p.GetTypeName() == "IsaacSurfaceGripper"]
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint) or not _value(prim, "physics:jointEnabled", True):
            continue
        # Kit's surface-gripper extension disables these attachment templates
        # while open and retargets them on grasp. Our runtime creates the grasp
        # constraints against the actual bottles instead.
        if any(prim.GetPath().HasPrefix(path) for path in gripper_templates):
            continue
        ids, frames = [], []
        for side in (0, 1):
            targets = prim.GetRelationship(f"physics:body{side}").GetTargets()
            target = str(targets[0]) if targets else None
            body = scene.bodies.get(target, -1)
            position = _value(prim, f"physics:localPos{side}", (0.0, 0.0, 0.0))
            rotation = _quat(_value(prim, f"physics:localRot{side}", Gf.Quatf(1.0)))
            if target and body < 0:
                target_prim = stage.GetPrimAtPath(target)
                if not target_prim:
                    raise ValueError(f"Joint {prim.GetPath()} references an unavailable frame: {target}")
                local = Gf.Matrix4d(1.0)
                local.SetRotate(Gf.Quatd(rotation[3], Gf.Vec3d(*rotation[:3])))
                local.SetTranslateOnly(Gf.Vec3d(*position))
                world = local * cache.GetLocalToWorldTransform(target_prim)
                position = world.ExtractTranslation()
                rotation = _quat(world.ExtractRotationQuat())
            ids.append(body)
            frames.append(wp.transform(position, rotation))
        if ids[1] < 0:
            if ids[0] < 0:
                continue
            ids.reverse()
            frames.reverse()
        plans.append((prim, ids, frames))

    def add_source_joint(prim, ids, frames):
        kw = {"parent_xform": frames[0], "child_xform": frames[1], "label": str(prim.GetPath())}
        kind = prim.GetTypeName()
        if kind == "PhysicsFixedJoint":
            joint = builder.add_joint_fixed(*ids, **kw)
        elif kind in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            angular = kind == "PhysicsRevoluteJoint"
            unit = math.pi / 180.0 if angular else 1.0
            gain_unit = 1.0 / unit if angular else 1.0
            drive = "drive:" + ("angular" if angular else "linear") + ":physics:"
            stiffness = float(_value(prim, drive + "stiffness", 0.0)) * gain_unit
            damping = float(_value(prim, drive + "damping", 0.0)) * gain_unit
            add = builder.add_joint_revolute if angular else builder.add_joint_prismatic
            joint = add(
                *ids,
                axis="XYZ".index(_value(prim, "physics:axis", "X")),
                limit_lower=float(_value(prim, "physics:lowerLimit", -math.inf)) * unit,
                limit_upper=float(_value(prim, "physics:upperLimit", math.inf)) * unit,
                target_q=float(_value(prim, drive + "targetPosition", 0.0)) * unit,
                target_qd=float(_value(prim, drive + "targetVelocity", 0.0)) * unit,
                target_ke=stiffness,
                target_kd=damping,
                effort_limit=float(_value(prim, drive + "maxForce", math.inf)),
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY
                if stiffness
                else newton.JointTargetMode.VELOCITY,
                **kw,
            )
        elif kind == "PhysicsJoint":
            axes = [[], []]
            for angular, names in enumerate((("transX", "transY", "transZ"), ("rotX", "rotY", "rotZ"))):
                unit = math.pi / 180.0 if angular else 1.0
                for axis, name in enumerate(names):
                    lo = float(_value(prim, f"limit:{name}:physics:low", -math.inf))
                    hi = float(_value(prim, f"limit:{name}:physics:high", math.inf))
                    if lo > hi:
                        continue
                    drive = f"drive:{name}:physics:"
                    stiffness = float(_value(prim, drive + "stiffness", 0.0)) / unit
                    axes[angular].append(
                        newton.ModelBuilder.JointDofConfig(
                            axis=axis,
                            limit_lower=lo * unit,
                            limit_upper=hi * unit,
                            target_q=float(_value(prim, drive + "targetPosition", 0.0)) * unit,
                            target_qd=float(_value(prim, drive + "targetVelocity", 0.0)) * unit,
                            target_ke=stiffness,
                            target_kd=float(_value(prim, drive + "damping", 0.0)) / unit,
                            effort_limit=float(_value(prim, drive + "maxForce", math.inf)),
                            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY
                            if stiffness
                            else newton.JointTargetMode.VELOCITY,
                        )
                    )
            joint = builder.add_joint_d6(*ids, linear_axes=axes[0], angular_axes=axes[1], **kw)
        else:
            raise ValueError(f"Unsupported joint {kind}: {prim.GetPath()}")
        scene.joints[str(prim.GetPath())] = joint
        scene.joint_frames.append((ids, frames, kind))
        return joint

    # Build each spanning tree contiguously. Keep the authored spring loops as
    # maximal constraints; trees can then use PhoenX's reduced-coordinate path.
    roots = list(range(builder.body_count))

    def find(body):
        while roots[body] != body:
            roots[body] = roots[roots[body]]
            body = roots[body]
        return body

    tree, constraints, incoming = [], [], set()
    for plan in plans:
        prim, (parent, child), _ = plan
        loop = child in incoming or (parent >= 0 and find(parent) == find(child))
        if loop or _value(prim, "physics:excludeFromArticulation", False):
            constraints.append(plan)
            continue
        incoming.add(child)
        tree.append(plan)
        if parent >= 0:
            roots[find(child)] = find(parent)
    components = {}
    for plan in tree:
        components.setdefault(find(plan[1][1]), []).append(plan)
    articulated = set()
    articulation_bodies = {}
    for component in components.values():
        children = {p[1][1] for p in component}
        parents = {p[1][0] for p in component if p[1][0] >= 0}
        floating = parents - children
        added = {-1}
        joints = []
        for body in sorted(floating):
            joints.append(builder.add_joint_free(child=body, label=builder.body_label[body] + "/floating_root"))
            added.add(body)
        remaining = list(component)
        while remaining:
            ready = [p for p in remaining if p[1][0] in added]
            if not ready:
                raise ValueError("The source articulation tree has no root")
            for plan in ready:
                joints.append(add_source_joint(*plan))
                added.add(plan[1][1])
                remaining.remove(plan)
        scene.articulations.append(builder.articulation_count)
        articulation_bodies[builder.articulation_count] = added - {-1}
        builder.add_articulation(joints)
        articulated.update(added - {-1})
    constraint_bodies = {b for _, bodies, _ in constraints for b in bodies if b >= 0}
    for body in sorted(constraint_bodies - articulated):
        joint = builder.add_joint_free(child=body, label=builder.body_label[body] + "/floating_root")
        builder.add_articulation([joint])
    with warnings.catch_warnings():
        # Multiple attachment springs between the same two gripper bodies are
        # intentional in the source. The maximal solver keeps every spring.
        warnings.filterwarnings("ignore", message="Adding a .*Parallel joints.*", category=newton.NewtonWarning)
        for plan in constraints:
            add_source_joint(*plan)
    unsupported = {body for prim, ids, _ in constraints if prim.GetTypeName() == "PhysicsJoint" for body in ids}
    scene.articulations = [a for a in scene.articulations if articulation_bodies[a].isdisjoint(unsupported)]

    for label, components in wrappers.items():
        corrected, correction = correct_wrapper_frame(components, label=label)
        bodies, joints, _ = build_wrapper(builder, corrected, label=label, visuals=visuals)
        scene.line_dynamic_shapes.update(
            builder.shape_label[shape]
            for shape in range(builder.shape_count)
            if builder.shape_body[shape] in bodies
            and builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        )
        correction["replaced_collision_proxies"] = [
            str(prim.GetPath()) for prim, _, _ in wrapper_proxies.get(label, [])
        ]
        scene.metadata.setdefault("wrapper_frame_corrections", {})[label] = correction
        scene.articulations.append(builder.articulation_count - 1)
        for body in bodies:
            scene.bodies[builder.body_label[body]] = body
            scene.body_mass[body] = builder.body_mass[body]
        for joint, kind in zip(joints, ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"), strict=True):
            scene.joints[builder.joint_label[joint]] = joint
            scene.joint_frames.append(
                (
                    [builder.joint_parent[joint], builder.joint_child[joint]],
                    [builder.joint_X_p[joint], builder.joint_X_c[joint]],
                    kind,
                )
            )
    for label, proxies in wrapper_proxies.items():
        if label not in wrappers:
            for prim, matrix, body in proxies:
                add_geometry(prim, matrix, body, visual=False, collision=True)
    if wrappers:
        scene.metadata["wrapper_mass_estimates"] = {
            "metal_shell_thickness_m": 0.003,
            "steel_density_kg_m3": 7850.0,
            "aluminium_density_kg_m3": 2700.0,
            "film_density_kg_m3": 920.0,
            "rubber_density_kg_m3": 1100.0,
            "rotor_effort_limit_nm": 1000.0,
            "lift_effort_limit_n": 3000.0,
            "lift_servo_frequency_hz": 10.0,
            "source_has_authored_wrapper_dynamics": False,
        }

    _bridge_transfers(builder, scene)
    if decomposed:
        # Case end walls are flat 3.2 mm plates with a hand hole that nothing
        # reaches through. Their bounding box keeps both faces and the
        # thickness; a convex decomposition of the hole needs eight slabs per
        # wall, whose contacts with guides and neighbours lengthen every
        # filled case's serial contact chain. Robot links keep their
        # triangle surfaces. Parts add no mass.
        builder.approximate_meshes("bounding_box", shape_indices=decomposed, raise_on_failure=True)
    for body, mass in scene.body_mass.items():
        if not builder.body_lock_inertia[body] and mass > 0 and builder.body_mass[body] > 0:
            ratio = mass / builder.body_mass[body]
            builder.body_inertia[body] = builder.body_inertia[body] * ratio
            builder.body_mass[body] = mass
            builder.body_inv_mass[body] = 1.0 / mass
            builder.body_inv_inertia[body] = wp.mat33(
                np.linalg.inv(np.asarray(builder.body_inertia[body]).reshape(3, 3))
            )
    if fleet.is_file():
        add_rack_shelf(builder)
    print(
        f"Berlin plant: {len(scene.bodies)} bodies, {len(scene.bottle_bodies)} live bottles, "
        f"{len(scene.joints)} joints, {builder.shape_count} shapes, {len(meshes)} shared meshes "
        f"loaded in {time.perf_counter() - started:.1f} s"
    )
    return scene


def _bridge_transfers(builder, scene):
    """Add static dead plates across the bottle path's belt-to-belt transfer gaps.

    Each lane chain hands bottles to the packer's IN_A belt, which hands them to IN_B,
    across 20 mm gaps. A real transfer bridges such a gap with a dead plate; without one,
    a slow bottle's base rim dropped into the gap at the next belt's square end and tipped.
    """
    plates = []
    for name in ("PU1", "PU2", "PU3", "PU4", "PD1", "PD2", "PD3", "PD4"):
        prefixes = (f"/Plant/Line/BottleLanes/{name}/Physics/Chain", f"/Plant/Line/Packers/{name}/Inlet/Belts/")
        spans = []
        for shape, label in enumerate(builder.shape_label):
            if not label.startswith(prefixes) or not builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            mesh = builder.shape_source[shape]
            if mesh is None:
                continue
            body = builder.shape_body[shape]
            pose = builder.shape_transform[shape]
            if body >= 0:
                pose = wp.transform_multiply(builder.body_q[body], pose)
            points = np.asarray(mesh.vertices) * np.asarray(builder.shape_scale[shape])
            rotation = np.asarray(wp.quat_to_matrix(pose.q)).reshape(3, 3)
            points = points @ rotation.T + np.asarray(pose.p)
            spans.append((points.min(axis=0), points.max(axis=0)))
        spans.sort(key=lambda span: span[0][1])
        for (lo0, hi0), (lo1, hi1) in itertools.pairwise(spans):
            gap = lo1[1] - hi0[1]
            if 0.0 < gap <= _MAX_TRANSFER_GAP and abs(hi0[2] - hi1[2]) < 1e-3:
                lo, hi = np.maximum(lo0[0], lo1[0]), np.minimum(hi0[0], hi1[0])
                center = ((lo + hi) / 2, (hi0[1] + lo1[1]) / 2, hi0[2] - 0.005)
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(center, wp.quat_identity()),
                    hx=(hi - lo) / 2,
                    hy=gap / 2,
                    hz=0.005,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.15, mu_static=0.18, gap=0.002, margin=0.0),
                    label=f"/Plant/Line/Packers/{name}/Inlet/DeadPlate_{len(plates)}/collision",
                )
                plates.append((name, float(gap)))
    scene.metadata["transfer_dead_plates"] = plates
