# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Restore the palletizer area's empty-pallet stores and magazines, and run their dispensers."""

import json

import numpy as np
import warp as wp

import newton.usd
from newton.examples.phoenx._bottling_wrap import mesh_components

_RETRACT = 0.15  # m, clears the separator fingers' 0.12 m reach into the pallet pockets
_PLATE_GAP = 0.01  # m, between the push-out plate and the column's rear face


@wp.kernel
def kinematic_strokes(
    bodies: wp.array[int], rest: wp.array[wp.transform], strokes: wp.array[wp.vec3], body_q: wp.array[wp.transform]
):
    """Set each kinematic carriage's target pose: its rest pose moved by its stroke in the plant frame."""
    i = wp.tid()
    body_q[bodies[i]] = wp.transform(
        wp.transform_get_translation(rest[i]) + strokes[i], wp.transform_get_rotation(rest[i])
    )


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


def _restore_pallet(pallet):
    for name, choice in (("physics", "dynamic"), ("collision", "boxes")):
        variants = pallet.GetVariantSet(name)
        if choice not in variants.GetVariantNames():
            raise ValueError(f"{pallet.GetPath()}: source pallet has no {name}={choice} variant")
        variants.SetVariantSelection(choice)


def prepare_dispensers(stage):
    """Restore the floor stores, magazine pallets and four support fingers per dispenser.

    Select each pallet's authored dynamic variant and compound collision.
    Overrides live only in the session layer; visual geometry and authored
    initial poses are preserved. Return the magazine body paths.

    Support contacts reuse the station tower's physics material when authored;
    otherwise the loader's default contact material applies. Preserve the exact
    finger triangles, including any nonconvex detail. Remove a coarse navigation
    proxy only when all its owners have been replaced by physical stock.

    The source's four separator fingers become two kinematic carriages, one
    per side, and its push-out plate a kinematic plate, all in their source
    poses; :class:`DispenserController` strokes them. Stock initially settles
    onto the fingers.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415 - OpenUSD is optional.

    stores = stage.GetPrimAtPath("/Plant/WrapLine/Stores")
    if stores:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for store in stores.GetChildren():
                for bay in store.GetChildren():
                    if not bay.GetName().startswith("bay_") or not bay.GetChildren():
                        continue
                    for pallet in bay.GetChildren():
                        _restore_pallet(pallet)
                    # The same-named cube represents the entire stockpile.
                    # The plant floor underneath supplies the actual support.
                    proxy = stage.GetPrimAtPath(store.GetPath().AppendPath("Physics/" + bay.GetName()))
                    if proxy and proxy.HasAPI(UsdPhysics.CollisionAPI):
                        proxy.SetActive(False)

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

    def carriage(path):
        # A kinematic machine part, authored at the identity so its pose is its stroke.
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(True)
        return path

    def collider(prim, material):
        prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
        prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("guide")
        UsdPhysics.CollisionAPI.Apply(prim)
        if material:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for pallet_path in pallets:
            _restore_pallet(stage.GetPrimAtPath(pallet_path))
        for dispenser_path, components, material in fingers:
            # The separator's two finger pairs enter the column's side pockets
            # from either side; the push-out plate behind the column moves a
            # lowered pallet forward into the pick opening.
            ordered = sorted(components, key=lambda component: component[0][:, 0].mean())
            for side, pair in (("L", ordered[:2]), ("R", ordered[2:])):
                root = carriage(f"{dispenser_path}/Physics/Separator{side}")
                for i, (vertices, indices) in enumerate(pair):
                    finger = UsdGeom.Mesh.Define(stage, f"{root}/Finger{i}")
                    finger.CreatePointsAttr(vertices.tolist())
                    finger.CreateFaceVertexIndicesAttr(indices.tolist())
                    finger.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
                    collider(finger.GetPrim(), material)
            plate = stage.GetPrimAtPath(f"{dispenser_path}/Physics/Static/push_out")
            if not plate:
                raise ValueError(f"{dispenser_path}: push-out plate is missing")
            matrix = (
                cache.GetLocalToWorldTransform(plate)
                * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{dispenser_path}/Physics")).GetInverse()
            )
            root = carriage(f"{dispenser_path}/Physics/PushOut")
            cube = UsdGeom.Cube.Define(stage, f"{root}/plate")
            cube.CreateSizeAttr(plate.GetAttribute("size").Get() or 2.0)
            cube.MakeMatrixXform().Set(matrix)
            collider(cube.GetPrim(), material)
            plate.SetActive(False)
    bodies = [path + "/Geometry/sm_pallet_block_a08_obj_00" for path in pallets]
    disable_owned_proxies(stage, bodies)
    return bodies


def dispenser_unit(spec):
    """Return a dispenser's strokes and dispense sequence from its source specification (``EP_*.json``)."""
    pick, load = (np.asarray(spec["ports"][port]["xy"]) for port in ("pick", "load"))
    heights = spec["magazine"]["pallet_z"]
    forward = (pick - load) / np.linalg.norm(pick - load)
    drop, pitch = heights[1] - heights[0], heights[2] - heights[1]
    return {
        "pick": pick,
        "started": None,
        "duration": spec["timers_s"]["dispense_s"],
        # The finger carriages retract across the fork axis, the left one (lower x) to -x.
        "side": np.abs((forward[1], forward[0])) * _RETRACT,
        "push": forward * (spec["magazine"]["column_behind_pick_m"] + _PLATE_GAP),
        # (duration share, column height [m], fingers retracted, push-out stroke)
        "phases": (
            (4.0, -drop, 0.0, 0.0),  # lower until the column's bottom pallet stands on the floor
            (1.5, -drop, 1.0, 0.0),  # release it
            (2.0, pitch - drop, 1.0, 0.0),  # rise to the next pallet's pockets
            (1.5, pitch - drop, 0.0, 0.0),  # clamp the next pallet
            (3.0, 0.0, 0.0, 0.0),  # lift the column clear of the released pallet
            (1.5, 0.0, 0.0, 1.0),  # push the released pallet forward into the pick opening
            (1.5, 0.0, 0.0, 0.0),  # return the push-out
        ),
    }


def dispense_strokes(unit, age):
    """Return the left and right finger carriages' and the push-out's strokes ``age`` seconds into a dispense."""
    height, retract, push = np.zeros(3)
    if age is not None:
        scale = unit["duration"] / sum(phase[0] for phase in unit["phases"])
        start, previous = 0.0, np.zeros(3)
        for share, *target in unit["phases"]:
            end = start + share * scale
            if age < end:
                t = max(0.0, age - start) / (end - start)
                previous = previous + (np.asarray(target) - previous) * (t * t * (3.0 - 2.0 * t))
                break
            start, previous = end, np.asarray(target)
        height, retract, push = previous
    side = unit["side"] * retract
    return [(-side[0], -side[1], height), (side[0], side[1], height), (*(unit["push"] * push), 0.0)]


class DispenserController:
    """Run the source's empty-pallet dispensers (``plant/lines_1x8/wrapline/EP_*.json``).

    A magazine presents one pallet at its pick opening and holds the column
    behind it on the separator fingers. Once a vehicle has lifted the
    presented pallet and withdrawn (``amr_pick_done``), the dispenser lowers
    the column until its bottom pallet stands on the floor, releases it,
    clamps the next pallet, lifts the column clear and pushes the released
    pallet forward into the pick opening, all within the source's
    ``dispense_s``.

    The source refills an empty magazine with a store stack of 11 at its load
    port (role R). A magazine of 15 serves its palletizer for more than an hour
    of production; that refill is not run here, so an empty magazine stops.
    """

    def __init__(self, scene, state, asset_root, fleet, link):
        self.state, self.fleet, self.link = state, fleet, link
        root = "/Plant/WrapLine/Dispensers/"
        names = sorted(
            label[len(root) :].split("/", 1)[0]
            for label in scene.bodies
            if label.startswith(root) and label.endswith("/Physics/PushOut")
        )
        self.units, bodies = [], []
        for name in names:
            unit = dispenser_unit(json.loads((asset_root / f"plant/lines_1x8/wrapline/{name}.json").read_text()))
            unit["port"] = f"{name}_pick"
            # The pallets in the column above the one presented at pick.
            unit["column"] = sum(label.startswith(f"{root}{name}/Magazine/") for label in scene.bodies) - 1
            self.units.append(unit)
            bodies.extend(
                scene.bodies[f"{root}{name}/Physics/{part}"] for part in ("SeparatorL", "SeparatorR", "PushOut")
            )
        self.bodies = wp.array(bodies, dtype=int, device=state.body_q.device)
        self.rest = wp.array(state.body_q.numpy()[bodies], dtype=wp.transform, device=state.body_q.device)
        self.strokes = wp.zeros(len(bodies), dtype=wp.vec3, device=state.body_q.device)

    def before_step(self, time, data):
        """Start a dispense once the presented pallet has left with its vehicle; queue the carriage strokes."""
        pallets = data["body_q"][self.fleet.pallet_bodies]
        strokes = []
        for unit in self.units:
            if unit["started"] is not None and time - unit["started"] >= unit["duration"]:
                unit["started"] = None
            if unit["started"] is None and unit["column"] > 0:
                offset = pallets[:, :3] - np.array([*unit["pick"], 0.0])
                present = np.any((np.linalg.norm(offset[:, :2], axis=1) < 0.6) & (offset[:, 2] < 0.2))
                picking = any(
                    mission.source == unit["port"] and mission.phase in ("to_pick", "pick")
                    for mission in self.fleet.missions.values()
                )
                if not present and not picking:
                    unit["started"] = time
                    unit["column"] -= 1
            age = None if unit["started"] is None else time - unit["started"]
            strokes += dispense_strokes(unit, age)
        self.link.write(self.strokes, np.asarray(strokes, dtype=np.float32))

    def simulate(self):
        """Apply the queued carriage strokes; graph-capturable."""
        wp.launch(
            kinematic_strokes, dim=len(self.bodies), inputs=[self.bodies, self.rest, self.strokes, self.state.body_q]
        )
