# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recover existing dispenser stock and its stationary separator contacts."""

import numpy as np

import newton.usd
from newton.examples.phoenx._bottling_wrap import mesh_components


def disable_owned_proxies(stage, covered_roots, *, scope="/Plant/Physics/NavProxies/Replaceable"):
    """Retire coarse proxies only when every represented owner is covered."""
    from pxr import Usd  # noqa: PLC0415 - OpenUSD is optional.

    proxies = stage.GetPrimAtPath(scope)
    if not covered_roots or not proxies:
        return
    prefixes = tuple(path.rstrip("/") + "/" for path in covered_roots)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for proxy in Usd.PrimRange(proxies):
            owner = proxy.GetAttribute("plant:owner").Get()
            owners = [path.strip() for path in owner.split(";") if path.strip()] if owner else []
            if owners and all(path.startswith(prefixes) or path in covered_roots for path in owners):
                proxy.SetActive(False)


def prepare_dispensers(stage):
    """Physicalize all authored magazine pallets and their four support fingers.

    Select each pallet's authored dynamic variant and compound collision.
    Overrides live only in the session layer; visual geometry and authored
    initial poses are preserved. Return the magazine body paths.

    Support contacts reuse the station tower's physics material when authored;
    otherwise the loader's default contact material applies. Preserve the exact
    finger triangles, including any nonconvex detail. Remove a coarse navigation
    proxy only when all its owners have been replaced by physical stock.

    Fingers remain stationary in their source pose, so stock initially settles
    onto them. This supplies initial stock/support physics, not a dispenser
    controller: lowering, separation, presentation and refill remain missing.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415 - OpenUSD is optional.

    root = stage.GetPrimAtPath("/Plant/WrapLine/Dispensers")
    if not root:
        return []
    cache = UsdGeom.XformCache()
    pallets, fingers = [], []
    for dispenser in root.GetChildren():
        magazine = dispenser.GetChild("Magazine")
        if not magazine:
            continue
        for pallet in magazine.GetChildren():
            if "dynamic" not in pallet.GetVariantSet("physics").GetVariantNames():
                raise ValueError(f"{pallet.GetPath()}: source stock has no dynamic physics variant")
            if "boxes" not in pallet.GetVariantSet("collision").GetVariantNames():
                raise ValueError(f"{pallet.GetPath()}: source stock has no compound collision variant")
            pallets.append(str(pallet.GetPath()))
        steel = stage.GetPrimAtPath(str(dispenser.GetPath()) + "/Frame/steel")
        if not steel:
            raise ValueError(f"{dispenser.GetPath()}: separator geometry is missing")
        mesh = newton.usd.get_mesh(steel, load_visual_materials=False, compute_inertia=False)
        physics = dispenser.GetChild("Physics")
        parent = physics if physics else dispenser
        transform = np.asarray(
            cache.GetLocalToWorldTransform(steel) * cache.GetLocalToWorldTransform(parent).GetInverse()
        )
        selected = []
        for part in mesh_components(mesh):
            vertices = part.vertices @ transform[:3, :3] + transform[3, :3]
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            if np.allclose(upper - lower, (0.25, 0.12, 0.05), atol=1.0e-4):
                selected.append((vertices, part.indices))
        if len(selected) != 4:
            raise ValueError(f"{steel.GetPath()}: expected four source separator fingers")
        tower = stage.GetPrimAtPath(str(dispenser.GetPath()) + "/Physics/Static/tower_l")
        material = UsdShade.MaterialBindingAPI(tower).ComputeBoundMaterial("physics")[0] if tower else None
        if material and not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            material = None
        fingers.append((str(dispenser.GetPath()), selected, material))

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for pallet_path in pallets:
            pallet = stage.GetPrimAtPath(pallet_path)
            pallet.GetVariantSet("physics").SetVariantSelection("dynamic")
            pallet.GetVariantSet("collision").SetVariantSelection("boxes")
        for dispenser_path, components, material in fingers:
            for i, (vertices, indices) in enumerate(components):
                finger = UsdGeom.Mesh.Define(stage, f"{dispenser_path}/Physics/SeparatorFinger{i}")
                finger.CreatePointsAttr(vertices.tolist())
                finger.CreateFaceVertexIndicesAttr(indices.tolist())
                finger.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
                finger.CreateVisibilityAttr("invisible")
                finger.CreatePurposeAttr("guide")
                UsdPhysics.CollisionAPI.Apply(finger.GetPrim())
                if material:
                    UsdShade.MaterialBindingAPI.Apply(finger.GetPrim()).Bind(material, materialPurpose="physics")
    bodies = [path + "/Geometry/sm_pallet_block_a08_obj_00" for path in pallets]
    disable_owned_proxies(stage, bodies)
    return bodies
