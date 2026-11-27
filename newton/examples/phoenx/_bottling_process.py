# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical wet-end bottle transport for the Berlin plant's process layout.

Single files run at the speed that carries the plant's rated output at the
admission pitch. The pasteuriser infeed and outfeed tables and the
accumulation table before the IFM carry packed beds across their source mat
tops. The pasteuriser tunnel carries its bed between its tables at the
tables' width. As in the source, a straight conveyor between parallel rails
carries the outfeed chute's stream through the dryer to the accumulation
table; no single filer stands between them.
"""

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_infeed import (
    BOTTLES_PER_MINUTE,
    SINGLE_FILE_PITCH,
    add_bottle_shapes,
    bottle_shape_templates,
)

PROCESS_BOTTLES = "/Plant/Process/Product/L1/Bottles"
_MACHINES = "/Plant/Process/Machines/L1"
_INLET_PLATE = "/Plant/Line/IFM/Modules/IFM_A/Colliders/DeadPlates/InletPlate"
# m/s, carries the plant's rated output in single file at the admission pitch.
_BELT_SPEED = BOTTLES_PER_MINUTE / 60.0 * SINGLE_FILE_PITCH
_GUIDE_WIDTH = 0.08  # m, at the working faces of the driven bottle guides.
_POOL_MAT = "/Plant/Blockout/Pool/mattop"
_WET_END_MAT = "/Plant/Blockout/WetEnd/mattop"
_WET_END_RAIL = "/Plant/Blockout/WetEnd/rail_green"
_TABLES = {
    "infeed": f"{_MACHINES}/L1_pasteuriser_table_in/conveyors_mattop",
    "outfeed": f"{_MACHINES}/L1_pasteuriser_table_out/conveyors_mattop",
    "pool": _POOL_MAT,
}
_PACK_PITCH = 0.063  # m, centre spacing of the source's mass-packed table placements
_PACK_AREA = 0.5 * math.sqrt(3.0) * _PACK_PITCH**2  # m^2 of deck per bottle in a packed bed
_POOL_STOCK = 1000  # bottles held by the accumulation table in production
# The source authors no pasteuriser rating. Tunnel pasteurisers are sized
# above the filler, commonly by about 10 %, so the bed never limits the line.
PASTEURISER_CAPACITY = 1.1
# The source's guide rails (the IFM's StubRails, the inlet's Phys_uhmw) are UHMW.
GUIDE_FRICTION = (0.08, 0.1)  # dynamic and static friction of the process guides and table guards
_CHUTE_MARGIN = 2.0  # outfeed funnel and chute speed over mass flow, for the looser flowing stream
# m above the deck, from the IFM's RailsOuter colliders that continue the neck.
_GUARD_BANDS = ((0.018, 0.043), (0.088, 0.113))
# m, deck width beyond each guard line. The guards start above the 16 mm heel, so a
# bottle pressed against a guard passes its heel under the guard line; a deck ending
# on that line let a packed bottle's heel slip off its edge and the bottle drop 16 mm.
_DECK_UNDER_GUARD = 0.02
_EYE = (0.065, 0.19)  # m above the deck, body and neck beams of the IFM's inlet eye
IFM_RECEIVING_BELT = "/Plant/Line/IFM/Modules/IFM_A/Belts/S2"


def _ports(stage, cache, name):
    prim = stage.GetPrimAtPath(f"{_MACHINES}/L1_{name}")
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
    return {
        key: np.asarray(point) @ matrix[:3, :3] + matrix[3, :3]
        for key, point in zip(
            prim.GetAttribute("process:portNames").Get(), prim.GetAttribute("process:ports").Get(), strict=True
        )
    }


def _arc(center, radius, start, end, direction):
    sweep = direction * ((direction * (end - start)) % (2.0 * math.pi))
    angles = np.linspace(start, start + sweep, max(2, math.ceil(abs(sweep) * radius / 0.04) + 1))
    return np.asarray(center) + radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)


def _rotary_route(ports, wheels):
    """Connect successive pitch circles at their shared tangent points."""
    origin = ports["in"][:2]
    main_axis = (ports["out"][:2] - origin) / np.linalg.norm(ports["out"][:2] - origin)
    left = np.array([-main_axis[1], main_axis[0]])
    centers = np.array([origin + x * main_axis + y * left for x, y, _ in wheels])
    port_angle = math.atan2(-left[1], -left[0])
    turns = np.r_[port_angle, np.arctan2(np.diff(centers[:, 1]), np.diff(centers[:, 0])), port_angle]
    arcs = []
    for i, (center, (_, _, radius)) in enumerate(zip(centers, wheels, strict=True)):
        start = turns[i] + (math.pi if i else 0.0)
        arc = _arc(center, radius, start, turns[i + 1], (-1) ** i)
        arcs.append(arc)
    return np.vstack((origin, *arcs, ports["out"][:2]))


def filling_route(stage, cache):
    """Read transfer ports and reconstruct the authored bottle pitch circles.

    The wet-end layer supplies no transport body or joint schemas. Its bottle
    placements give the pitch circles of the rinser, filler and crowner. The
    visible transfer-wheel centres and radii complete the tangent connections.
    Wheel coordinates are metres from each machine's inlet toward its outlet
    and to the left of that direction.
    """
    if not stage.GetPrimAtPath(PROCESS_BOTTLES):
        return None
    depalletiser = _ports(stage, cache, "depalletiser")["out"]
    inspection_in = _ports(stage, cache, "empty_bottle_inspector")
    rinser = _ports(stage, cache, "rinser")
    filler = _ports(stage, cache, "filler_crowner")
    inspection_out = _ports(stage, cache, "fill_level_inspector")
    pasteuriser = _ports(stage, cache, "pasteuriser")
    dryer = _ports(stage, cache, "dryer")
    rinser_path = _rotary_route(rinser, ((0.5475, 0.2, 0.2), (1.2, 0.4636, 0.50375), (1.8525, 0.2, 0.2)))
    filler_path = _rotary_route(
        filler,
        (
            (0.5512, 0.25, 0.25),
            (1.6164, 0.865, 0.98),
            (2.6816, 0.25, 0.25),
            (3.5652, 0.7602, 0.77),
            (4.4488, 0.25, 0.25),
        ),
    )
    from pxr import Usd, UsdGeom  # noqa: PLC0415 - USD is optional.

    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        useExtentsHint=False,
        ignoreVisibility=True,
    )
    wet_end = bounds.ComputeWorldBound(stage.GetPrimAtPath("/Plant/Blockout/WetEnd/mattop")).ComputeAlignedRange()
    pool = bounds.ComputeWorldBound(stage.GetPrimAtPath(_POOL_MAT)).ComputeAlignedRange()
    outlet_x = pool.GetMax()[0]
    inlet_plate = stage.GetPrimAtPath(_INLET_PLATE)
    if inlet_plate:
        plate = bounds.ComputeWorldBound(inlet_plate).ComputeAlignedRange()
        outlet_x = min(outlet_x, plate.GetMax()[0])
    # The return and IFM entry have no named ports in this layer. The return
    # belt is 0.415 m wide; its centre and the pool outfeed are measured from
    # the authored bent return belt and the outlet edge in the scene.
    corner_x = wet_end.GetMin()[0] + 0.2075
    y_in, y_out = inspection_out["out"][1], pasteuriser["in"][1]
    bend = 0.16
    high_turn = _arc((corner_x + bend, y_in - bend), bend, math.pi / 2, math.pi, 1)
    low_turn = _arc((corner_x + bend, y_out + bend), bend, math.pi, 1.5 * math.pi, 1)
    xy = np.vstack(
        (
            depalletiser[:2] + np.array([0.35, 0.0]),
            inspection_in["in"][:2],
            inspection_in["out"][:2],
            rinser_path,
            filler_path,
            inspection_out["in"][:2],
            inspection_out["out"][:2],
            high_turn,
            low_turn,
            pasteuriser["in"][:2],
            pasteuriser["out"][:2],
            dryer["out"][:2],
            (outlet_x, y_out),
        )
    )
    # The source measurements are rounded by USD. Join the two representations
    # of each tangent point before evaluating corridor normals.
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.001]
    xy = xy[keep]
    return np.column_stack((xy, np.full(len(xy), depalletiser[2])))


def _strip(first, second):
    vertices = np.stack((first, second), axis=1).reshape(-1, 3)
    i = np.arange(0, len(vertices) - 2, 2, dtype=np.int32)
    faces = np.column_stack((i, i + 2, i + 1, i + 1, i + 2, i + 3))
    return vertices, faces.ravel()


def _points(stage, cache, path):
    """World vertices of a source mesh."""
    from pxr import UsdGeom  # noqa: PLC0415 - USD is optional.

    prim = stage.GetPrimAtPath(path)
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
    return np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()) @ matrix[:3, :3] + matrix[3, :3]


def _deck_points(stage, cache, path):
    """World vertices of a source mat top's conveying plane."""
    points = _points(stage, cache, path)
    return points[np.isclose(points[:, 2], points[:, 2].max(), atol=1.0e-4)]


def _outline(points, center):
    """Stations ``x`` and half-widths of a mat top that runs straight along +x about ``center``."""
    x = np.unique(np.round(points[:, 0], 4))
    half = np.array([np.abs(points[np.isclose(points[:, 0], xi, atol=1.0e-4), 1] - center).max() for xi in x])
    return x, half


def wet_end_layout(stage, cache, path, pasteuriser_scale=1.0):
    """Read the mass-flow tables on the route's final straight from the source mat tops.

    Returns the outlines (stations and half-widths) of the pasteuriser infeed
    and outfeed tables, with their chutes and funnels, and of the accumulation
    table with its railed neck; the start of the dryer on the outfeed
    conveyor; and that conveyor's rails, as the half-width of their inner
    faces and their band above the deck.

    ``pasteuriser_scale`` widens the pasteuriser bed by that factor; see
    :func:`widen_pasteuriser`.
    """
    center, deck = path[-1, 1:]
    layout = {name: _outline(_deck_points(stage, cache, prim), center) for name, prim in _TABLES.items()}
    if pasteuriser_scale != 1.0:
        # The chutes keep the width of the single file and the outfeed rails.
        # The funnels keep their source angle and lengthen with the bed's
        # width: a steeper outfeed funnel lets the packed bed arch across it.
        chute, bed = layout["infeed"][1].min(), layout["infeed"][1].max()
        stretch = (pasteuriser_scale * bed - chute) / (bed - chute)
        for name, sign in (("infeed", 1.0), ("outfeed", -1.0)):
            x, half = layout[name]
            narrow = half == chute
            neck = sign * (sign * x[narrow]).max()
            end = neck + stretch * (sign * (sign * x[~narrow]).min() - neck)
            layout[name] = (
                np.where(narrow, x, sign * np.maximum(sign * x, sign * end)),
                np.where(narrow, half, pasteuriser_scale * bed),
            )
    # A straight conveyor continues the outfeed table through the dryer to
    # the accumulation table; its first mat top ends at the dryer.
    start = layout["outfeed"][0][-1]
    conveyors = _deck_points(stage, cache, _WET_END_MAT)
    on_line = (np.abs(conveyors[:, 1] - center) < 0.5) & (conveyors[:, 0] > start + 1.0e-3)
    layout["dryer"] = conveyors[on_line, 0].min()
    rails = _points(stage, cache, _WET_END_RAIL)
    rails = rails[(np.abs(rails[:, 1] - center) < 0.5) & (rails[:, 0] > start - 1.0e-3)]
    layout["rails"] = (np.abs(rails[:, 1] - center).min(), (rails[:, 2].min() - deck, rails[:, 2].max() - deck))
    return layout


def widen_pasteuriser(stage, cache, layout, center, pasteuriser_scale):
    """Widen the pasteuriser's visual housing and tables to a bed widened by :func:`wet_end_layout`.

    Each machine grows across the line by the width added to the bed, so the
    housing keeps its source clearance around the bed. The source authors the
    machines as instances; their shared prototypes stay unchanged. Utility
    colliders that end at the housing's side walls, such as its hot-water
    pipes, end at the widened walls.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415 - USD is optional.

    added = 2.0 * layout["infeed"][1].max() * (1.0 - 1.0 / pasteuriser_scale)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
    housing = stage.GetPrimAtPath(f"{_MACHINES}/L1_pasteuriser")
    walls = bounds.ComputeWorldBound(housing).ComputeAlignedRange()
    for root in (_MACHINES, _MACHINES.replace("/Machines/", "/Detail/Machines/")):
        for name in ("pasteuriser", "pasteuriser_table_in", "pasteuriser_table_out"):
            prim = stage.GetPrimAtPath(f"{root}/L1_{name}")
            if not prim:
                continue
            extent = bounds.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()[1]
            # Scale across the line about its centre, in world coordinates.
            widen = Gf.Matrix4d().SetScale(Gf.Vec3d(1.0, (extent + added) / extent, 1.0))
            about = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, center, 0.0))
            world = cache.GetLocalToWorldTransform(prim) * about.GetInverse() * widen * about
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(world * cache.GetParentToWorldTransform(prim).GetInverse())
    cache.Clear()
    bounds.Clear()
    widened = bounds.ComputeWorldBound(housing).ComputeAlignedRange()
    low, high = np.asarray(widened.GetMin()), np.asarray(widened.GetMax())
    walls = (walls.GetMin()[1], walls.GetMax()[1])
    nav = stage.GetPrimAtPath("/Plant/Physics/NavProxies")
    for prim in nav.GetChildren() if nav else ():
        translate, scale = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:scale")
        if not (translate and scale):
            continue
        center_point, size = np.asarray(translate.Get()), np.asarray(scale.Get())
        lo, hi = center_point - 0.5 * size, center_point + 0.5 * size
        if not (low[0] < lo[0] and hi[0] < high[0] and lo[2] < high[2]):
            continue
        if abs(hi[1] - walls[0]) < 0.1:
            hi[1] = min(hi[1], low[1])
        elif abs(lo[1] - walls[1]) < 0.1:
            lo[1] = max(lo[1], high[1])
        else:
            continue
        if hi[1] <= lo[1]:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            continue
        translate.Set(tuple(0.5 * (lo + hi)))
        scale.Set(tuple(hi - lo))


def _mass_flow_speed(width):
    """Belt speed [m/s] that carries the rated output as a packed bed of the given width [m]."""
    return BOTTLES_PER_MINUTE / 60.0 * _PACK_AREA / width


def _packed_field(outline, center, count=None, clearance=0.0):
    """Hexagonally packed stock within an outline, filled from its downstream end backwards."""
    x_stations, half = outline
    radius = 0.5 * _PACK_PITCH
    rows = []
    x = x_stations[-1] - radius - clearance
    while x - radius >= x_stations[0] and (count is None or sum(map(len, rows)) < count):
        # The narrower of the row's edges bounds it where the outline steps in.
        width = min(np.interp(x - radius, x_stations, half), np.interp(x + radius, x_stations, half)) - radius
        # Keep one lattice across the outline steps.
        y = (
            np.arange(-math.ceil(width / _PACK_PITCH), math.ceil(width / _PACK_PITCH) + 1) + 0.5 * (len(rows) % 2)
        ) * _PACK_PITCH
        y = y[np.abs(y) <= width + 1.0e-9]
        rows.append(np.column_stack((np.full_like(y, x), center + y)))
        x -= _PACK_PITCH * math.sqrt(3.0) / 2.0
    return np.vstack(rows)[:count]


def pasteuriser_outline(layout):
    """Join the infeed table, the tunnel deck at the tables' width, and the outfeed table."""
    infeed, outfeed = layout["infeed"], layout["outfeed"]
    return np.r_[infeed[0], outfeed[0]], np.r_[infeed[1], outfeed[1]]


def _add_table(scene, name, outline, center, speed, *, inflow=None, bands=_GUARD_BANDS, motor_off, deck_cfg, guard_cfg):
    """Add a straight mass-flow table along +x with guards along its outline.

    ``outline`` holds stations and half-widths. ``speed`` maps stations to
    belt speeds [m/s]. ``inflow`` is the half-width of a narrower opening in
    the upstream end; without it, that end stays open at the table's width.
    The downstream end stays open into the next conveyor. ``bands`` are the
    guards' heights above the deck. Returns the deck's label.
    """
    builder = scene.builder
    x_stations, half_stations = outline
    # Subdivide the deck so its surface velocity follows the speed profile.
    x = np.unique(np.r_[x_stations, np.arange(x_stations[0], x_stations[-1], 0.1)])
    half = np.interp(x, x_stations, half_stations)
    y, z = center[1], center[2]
    label = f"/Plant/Process/Transport/{name}"
    vertices, indices = _strip(
        np.column_stack((x, y - half - _DECK_UNDER_GUARD, np.full_like(x, z))),
        np.column_stack((x, y + half + _DECK_UNDER_GUARD, np.full_like(x, z))),
    )
    deck = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=not motor_off)
    builder.add_shape_mesh(-1, mesh=deck, cfg=deck_cfg, label=label + "/Deck/collision")
    if not motor_off:
        velocity = np.zeros((len(vertices), 3), dtype=np.float32)
        velocity[:, 0] = np.repeat(speed(x), 2)
        scene.belts.append((deck, velocity, label + "/Deck"))
    # Each guard keeps the table on its right in traversal order.
    edge = np.column_stack((x, half))
    if inflow is not None:
        edge = np.vstack(((x[0], inflow), edge))
    for side, sign, corners in (("left", 1.0, edge), ("right", -1.0, edge[::-1])):
        line = np.column_stack((corners[:, 0], y + sign * corners[:, 1], np.full(len(corners), z)))
        for band, (low, high) in enumerate(bands):
            vertices, indices = _strip(line + np.array((0.0, 0.0, low)), line + np.array((0.0, 0.0, high)))
            guard = newton.Mesh(vertices, indices, compute_inertia=False)
            builder.add_shape_mesh(-1, mesh=guard, cfg=guard_cfg, label=f"{label}/Guard_{side}_{band}/collision")
    return label + "/Deck"


def _straight(path, length, x):
    """Arc length of ``x`` on the route's final straight along +x."""
    return length[-1] - (path[-1, 0] - x)


def add_filling_transport(scene, path, *, motor_off, layout=None, inlet_speed=None):
    """Create contact-driven transport under the source process visuals.

    With ``layout`` (see :func:`wet_end_layout`), the route's final straight
    carries packed beds through the pasteuriser and over the accumulation
    table, which runs with the IFM's infeed drive at ``inlet_speed``.
    """
    builder = scene.builder
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    sections = [np.linspace(a, b, max(2, math.ceil((b - a) / 0.4) + 1))[:-1] for a, b in pairwise(length)]
    samples = np.concatenate([*sections, length[-1:]])
    runs = [(0.0, length[-1])]
    if layout is not None:
        start = np.flatnonzero(path[:, 0] < layout["infeed"][0][0])[-1]
        if np.ptp(path[start:, 1]) > 1.0e-6 or np.any(np.diff(path[start:, 0]) <= 0.0):
            raise ValueError("The filling route must cross the wet-end tables on one straight run along +x")
        # A single file feeds the pasteuriser infeed table.
        runs = [(0.0, _straight(path, length, layout["infeed"][0][0]))]
    belt_cfg = builder.ShapeConfig(density=0.0, mu=0.18, mu_static=0.2, gap=0.001, is_visible=False)
    rail_cfg = builder.ShapeConfig(
        density=0.0, mu=GUIDE_FRICTION[0], mu_static=GUIDE_FRICTION[1], gap=0.001, is_visible=False
    )
    guide_width = _GUIDE_WIDTH / 2.0
    deck_width = guide_width + 0.02
    # Guide the bottle near its centre of mass and leave the heel free as it
    # rolls over the curved sections. The source straight guides start 80 mm
    # above the deck.
    guide_base, guide_top = np.array([0.0, 0.0, 0.07]), np.array([0.0, 0.0, 0.098])
    section = 0
    zones = []
    for begin_length, end_length in runs:
        zone = []
        zones.append(zone)
        inner = samples[(begin_length + 1.0e-4 < samples) & (samples < end_length - 1.0e-4)]
        run = np.r_[begin_length, inner, end_length]
        points = np.column_stack([np.interp(run, length, column) for column in path.T])
        tangent = np.gradient(points, run, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        if begin_length == 0.0:
            # Leave room for the entire bottle base on the physical entrance deck.
            inlet = points[0] + 0.15 * tangent[0] + np.array([0.0, 0.0, 0.001])
            yaw = math.atan2(tangent[0, 1], tangent[0, 0])
            scene.metadata["process_inlet"] = (
                wp.transform(inlet, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)),
                tangent[0] * _BELT_SPEED,
            )
        left = np.column_stack((-tangent[:, 1], tangent[:, 0], np.zeros(len(points))))
        # Use short meshes to keep broad-phase bounds local along the return loop.
        ends = np.unique(np.r_[0, np.searchsorted(run, np.arange(run[0] + 2.0, run[-1], 2.0)), len(run) - 1])
        for begin, stop in pairwise(ends):
            region = slice(begin, stop + 1)
            label = f"/Plant/Process/Transport/Section_{section:02}"
            section += 1
            center = points[region]
            sides = left[region]
            vertices, indices = _strip(center - deck_width * sides, center + deck_width * sides)
            belt = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=not motor_off)
            builder.add_shape_mesh(-1, mesh=belt, cfg=belt_cfg, label=label + "/Deck/collision")
            if not motor_off:
                velocity = np.repeat(tangent[region] * _BELT_SPEED, 2, axis=0).astype(np.float32)
                scene.belts.append((belt, velocity, label + "/Deck"))
                zone.append(label + "/Deck")
            for side in (-1, 1):
                edge = center + side * guide_width * sides
                bottom = edge + guide_base
                top = edge + guide_top
                vertices, indices = _strip(top, bottom) if side < 0 else _strip(bottom, top)
                rail = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=not motor_off)
                rail_label = f"{label}/Guide_{'left' if side > 0 else 'right'}"
                builder.add_shape_mesh(-1, mesh=rail, cfg=rail_cfg, label=rail_label + "/collision")
                if not motor_off:
                    scene.belts.append((rail, velocity, rail_label))
                    zone.append(rail_label)
    if layout is None:
        return
    outlet = path[-1]
    cfg = {"motor_off": motor_off, "deck_cfg": belt_cfg, "guard_cfg": rail_cfg}
    # The pasteuriser's infeed table, tunnel deck and outfeed table form one
    # bed, which runs at PASTEURISER_CAPACITY times the rated output. Its
    # chutes and funnels run at twice mass flow: the streams that fill and
    # drain the bed pack at about 55 % of a standing bed.
    outline = pasteuriser_outline(layout)
    body, funnel = layout["infeed"][0][2], layout["outfeed"][0][1]
    chute, pool = outline[0][-1], layout["pool"]

    def pasteuriser_speed(x):
        # Chutes and funnels carry a loose stream; the bed between them is packed.
        margin = np.where((body <= x) & (x < funnel), PASTEURISER_CAPACITY, _CHUTE_MARGIN)
        return np.minimum(margin * _mass_flow_speed(2.0 * np.interp(x, *outline)), _BELT_SPEED)

    bed = _add_table(scene, "Pasteuriser", outline, outlet, pasteuriser_speed, inflow=guide_width, **cfg)
    # The source continues the chute with one straight conveyor through the
    # dryer to the accumulation table, between parallel rails in one band; it
    # authors no single filer. The conveyor runs on at the chute's speed, so
    # the stream widens into it without a speed step and stays loose. It only
    # ever widens, from the chute to the rails and the rails to the table, so
    # no bottles can arch across it.
    rail, band = layout["rails"]
    chute_speed = pasteuriser_speed(np.array([chute]))[0]
    conveyor = _add_table(
        scene,
        "Outfeed",
        (np.array([chute, pool[0][0]]), np.array([rail, rail])),
        outlet,
        lambda x: np.full_like(x, chute_speed),
        bands=(band,),
        **cfg,
    )
    table = _add_table(
        scene, "AccumulationTable", pool, outlet, lambda x: np.full_like(x, inlet_speed), inflow=rail, **cfg
    )

    # Each feed stops while a standing queue holds the eye at the entrance of
    # what it feeds, like the source IFM's "may feed" permit: the body beam
    # stays blocked and the neck beam, which flickers between the necks of a
    # moving file, stays unchanged. Eyes sit on the receiving belts, so a
    # stopped feed cannot hold its own eye. A full section therefore stops its
    # feed instead of letting the belts slip under a queue that presses into
    # it. The accumulation table feeds the IFM at the IFM's own infeed speed
    # and follows its drive M1, so it never packs its field into the infeed
    # faster than the infeed carries it, and it waits for the IFM's permit eye
    # PE-A1, so it never presses its field into a full infeed.
    def eye(x, half):
        start = (x, outlet[1] - half - 0.01, outlet[2])
        return [((start[0], start[1], start[2] + height), (0.0, 1.0, 0.0), 2.0 * half + 0.02) for height in _EYE]

    scene.metadata["process_zones"] = {
        "infeed_line": (zones[0], eye(outline[0][0] + 0.15, outline[1][0])),
        "pasteuriser": ([bed], eye(chute + 0.15, rail)),
        "outfeed": ([conveyor], eye(pool[0][0] + 0.1, rail)),
        "pool": ([table], None),
    }
    scene.metadata["process_handoff"] = (outlet, outlet[0] - pool[0][0], pool[1].max() + 0.03)


def add_filling_bottles(scene, path, *, motor_off=False, layout=None):
    """Seed production stock with the same material and collision geometry as the IFM.

    Single files carry stock at the rated admission pitch. With ``layout``, a
    packed bed fills the pasteuriser from its infeed funnel to its outfeed
    funnel, a single file stands on the outfeed conveyor from the dryer on, as
    the source places it, and the accumulation table holds a packed field
    against its IFM outlet.
    """
    builder = scene.builder
    source = scene.bottle_bodies[0]
    shapes = bottle_shape_templates(builder, source)
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    # Start the line in production, so the tables receive the rated output from
    # the start rather than after the transit time. Without tables, leave a
    # short clear run into the live IFM so its original stock can start moving.
    runs = [(0.35, length[-1] - 1.5)]
    if layout is not None:
        runs = [(0.35, _straight(path, length, layout["infeed"][0][0]) - _PACK_PITCH)]
    samples = np.concatenate([np.arange(begin, end, SINGLE_FILE_PITCH) for begin, end in runs])
    centers = np.column_stack([np.interp(samples, length, column) for column in path.T])
    tangents = np.gradient(path, length, axis=0)
    direction = np.column_stack([np.interp(samples, length, axis) for axis in tangents.T])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    velocities = direction * _BELT_SPEED
    if layout is not None:
        x, half = pasteuriser_outline(layout)
        # The chutes stay clear; the funnels and everything between are full.
        bed = (x[1:-1], half[1:-1])
        dryer = np.arange(layout["dryer"], layout["pool"][0][0] - _PACK_PITCH, SINGLE_FILE_PITCH)
        stock = np.vstack(
            (
                _packed_field(bed, path[-1, 1]),
                np.column_stack((dryer, np.full_like(dryer, path[-1, 1]))),
                _packed_field(layout["pool"], path[-1, 1], _POOL_STOCK, clearance=_PACK_PITCH),
            )
        )
        centers = np.vstack((centers, np.column_stack((stock, np.full(len(stock), path[-1, 2])))))
        velocities = np.vstack((velocities, np.zeros((len(stock), 3))))
    for index, position in enumerate(centers):
        position[2] += 0.001
        label = f"/Plant/Sim/Product/Bottles/WetEnd_{index:05d}"
        body = builder.add_link(
            xform=wp.transform(position, wp.quat_identity()),
            com=builder.body_com[source],
            inertia=builder.body_inertia[source],
            mass=builder.body_mass[source],
            lock_inertia=True,
            label=label,
        )
        scene.bodies[label] = body
        scene.bottle_bodies.append(body)
        if not motor_off:
            builder.body_qd[body] = wp.spatial_vector(*velocities[index], 0.0, 0.0, 0.0)
        add_bottle_shapes(builder, body, label, shapes)
    scene.metadata["process_bottles"] = len(centers)


def _shorten_box(prim, top):
    scale = np.asarray(prim.GetAttribute("xformOp:scale").Get())
    translation = np.asarray(prim.GetAttribute("xformOp:translate").Get())
    bottom = translation[2] - 0.5 * scale[2]
    if bottom < top < translation[2] + 0.5 * scale[2]:
        scale[2] = top - bottom
        translation[2] = 0.5 * (bottom + top)
        prim.GetAttribute("xformOp:scale").Set(tuple(scale))
        prim.GetAttribute("xformOp:translate").Set(tuple(translation))


def replace_process_proxies(stage, cache, belt_top):
    """Keep static machine supports below the physical moving belt."""
    top = belt_top - 0.04
    for prim in stage.GetPrimAtPath("/Plant/Physics/Equipment").GetChildren():
        item = prim.GetAttribute("plant:item").Get()
        kind = prim.GetAttribute("plant:kind").Get()
        if item == "bottle_pool" or (item and item.startswith("L1_") and kind != "reject_table"):
            _shorten_box(prim, top)
    for prim in stage.GetPrimAtPath("/Plant/Physics/Conveyors").GetChildren():
        if prim.GetName().startswith("L1_bottles_"):
            _shorten_box(prim, top)
    plate = stage.GetPrimAtPath(_INLET_PLATE)
    if plate:
        # Recess the fixed plate under the driven nose at the pool outlet.
        # This keeps the sill as physical support below the belt surface.
        translation = plate.GetAttribute("xformOp:translate")
        x, y, z = translation.Get()
        translation.Set((x, y, z - 0.005))
    cache.Clear()


def open_pool_passage(stage, path, half_width):
    """Leave the source doorway frame and provide a physical conveyor opening."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415 - USD is optional.

    # The vehicle proxy merges a doorway with the conveyor and spans the
    # otherwise unobstructed transport path. Keep its four surrounding pieces.
    start, end = path[-2:]
    width = half_width + 0.015
    left, right = end[1] - width, end[1] + width
    deck, header = end[2] - 0.02, end[2] + 0.27
    nav = stage.GetPrimAtPath("/Plant/Physics/NavProxies")
    if not nav:
        return
    for prim in nav.GetChildren():
        owner = prim.GetAttribute("plant:owner").Get() or ""
        if "/Plant/Blockout/Pool/mattop" not in owner:
            continue
        center = np.asarray(prim.GetAttribute("xformOp:translate").Get())
        scale = np.asarray(prim.GetAttribute("xformOp:scale").Get())
        lo, hi = center - 0.5 * scale, center + 0.5 * scale
        if not (start[0] <= lo[0] < hi[0] <= end[0] and lo[1] < right and left < hi[1]):
            continue
        if hi[2] <= end[2] + 0.005:
            # Some pool proxies duplicate the belt top exactly. Retain their
            # volume below the deck so it does not also carry stationary friction.
            _shorten_box(prim, deck)
            continue
        if not (lo[2] < deck < header < hi[2] and lo[1] < left < right < hi[1]):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        pieces = (
            ("LeftJamb", lo, np.array([hi[0], left, hi[2]])),
            ("RightJamb", np.array([lo[0], right, lo[2]]), hi),
            ("Sill", np.array([lo[0], left, lo[2]]), np.array([hi[0], right, deck])),
            ("Header", np.array([lo[0], left, header]), np.array([hi[0], right, hi[2]])),
        )
        for name, minimum, maximum in pieces:
            cube = UsdGeom.Cube.Define(stage, nav.GetPath().AppendChild(f"{prim.GetName()}_{name}"))
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(tuple(0.5 * (maximum + minimum)))
            cube.AddScaleOp().Set(tuple(maximum - minimum))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
