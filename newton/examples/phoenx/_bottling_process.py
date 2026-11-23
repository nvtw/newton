# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical wet-end bottle transport for the Berlin plant's process layout."""

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
_POOL_LENGTH = 5.3425  # m, from the source table's start to the IFM transfer plate.
_POOL_HALF_WIDTH = 0.245  # m, across six native IFM strands.
_POOL_SLOWDOWN = 1.8025  # m, from the crossbelt drive section to the transfer plate.
_POOL_RUNOUT = 0.6  # m, at the receiving strand speed before crossing the seam.
_POOL_UPPER_GUIDE = 4.00  # m, engage the cap before the crossbelt ramp.
POOL_SWEEP_PERIOD = 6.4  # s, for the oscillating crossbelt.
IFM_ENTRY_SPEED_RATIO = 1.4


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
    pool = bounds.ComputeWorldBound(stage.GetPrimAtPath("/Plant/Blockout/Pool/mattop")).ComputeAlignedRange()
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


def _ease(x):
    return x * x * (3.0 - 2.0 * x)


def _transport_speed(remaining, inlet_speed):
    if inlet_speed is None:
        return np.full_like(remaining, _BELT_SPEED)
    fraction = np.clip((remaining - _POOL_RUNOUT) / (_POOL_SLOWDOWN - _POOL_RUNOUT), 0.0, 1.0)
    return inlet_speed + (_BELT_SPEED - inlet_speed) * _ease(fraction)


def add_filling_transport(scene, path, *, motor_off, inlet_speed=None):
    """Create contact-driven transport under the source process visuals."""
    builder = scene.builder
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    sections = [np.linspace(a, b, max(2, math.ceil((b - a) / 0.4) + 1))[:-1] for a, b in pairwise(length)]
    samples = np.concatenate([*sections, length[-1:]])
    if inlet_speed is not None:
        # The accumulation table is straight. Refine its contact mesh where
        # the belt widens and the crossbelt speed changes along the deck.
        pool_start = max(0.0, length[-1] - _POOL_LENGTH)
        pool_samples = np.linspace(pool_start, length[-1], 60)
        samples = np.concatenate([samples[samples < pool_start - 1.0e-4], pool_samples])
    points = np.column_stack([np.interp(samples, length, column) for column in path.T])
    tangent = np.gradient(points, samples, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    remaining = length[-1] - samples
    speed = _transport_speed(remaining, inlet_speed)
    guide_width = np.full_like(samples, _GUIDE_WIDTH / 2.0)
    deck_width = guide_width + 0.02
    lateral = np.zeros_like(samples)
    if inlet_speed is not None:
        guide_width = np.interp(
            remaining, (0, 3.7925, 4.7925), (_POOL_HALF_WIDTH, _POOL_HALF_WIDTH, _GUIDE_WIDTH / 2.0)
        )
        deck_width = np.minimum(guide_width + 0.02, _POOL_HALF_WIDTH)
        # The short transverse drive spreads a single file onto all six
        # strands. The final mat slows to the receiving strand speed.
        lateral = 0.088 * np.interp(remaining, (0.0, 1.8025, 2.0025, 3.0125, 3.2125, _POOL_LENGTH), (0, 0, 1, 1, 0, 0))
    # Leave room for the entire bottle base on the physical entrance deck.
    inlet = points[0] + 0.15 * tangent[0] + np.array([0.0, 0.0, 0.001])
    yaw = math.atan2(tangent[0, 1], tangent[0, 0])
    scene.metadata["process_inlet"] = (
        wp.transform(inlet, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)),
        tangent[0] * speed[0],
    )
    left = np.column_stack((-tangent[:, 1], tangent[:, 0], np.zeros(len(points))))
    # Use short meshes to keep broad-phase bounds local along the return loop.
    ends = np.unique(np.r_[0, np.searchsorted(samples, np.arange(2.0, length[-1], 2.0)), len(samples) - 1])
    belt_cfg = builder.ShapeConfig(density=0.0, mu=0.18, mu_static=0.2, gap=0.001, is_visible=False)
    pool_cfg = builder.ShapeConfig(density=0.0, mu=0.35, mu_static=0.45, gap=0.001, is_visible=False)
    rail_cfg = builder.ShapeConfig(density=0.0, mu=0.30, mu_static=0.35, gap=0.001, is_visible=False)
    # The elastomer upper belt yields as a rocking cap first meets it.
    keeper_cfg = builder.ShapeConfig(
        density=0.0, ke=1800.0, kd=12.0, mu=0.15, mu_static=0.18, gap=0.001, is_visible=False
    )
    pool_crossbelt = {}
    # Guide the bottle near its centre of mass and leave the heel free as it
    # rolls over the curved sections. The source straight guides start 80 mm
    # above the deck.
    guide_base, guide_top = np.array([0.0, 0.0, 0.07]), np.array([0.0, 0.0, 0.098])
    for section, (begin, end) in enumerate(pairwise(ends)):
        region = slice(begin, end + 1)
        label = f"/Plant/Process/Transport/Section_{section:02}"
        center = points[region]
        sides = left[region]
        deck_edges = deck_width[region, None] * sides
        vertices, indices = _strip(center - deck_edges, center + deck_edges)
        belt = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=not motor_off)
        on_pool = inlet_speed is not None and _POOL_SLOWDOWN < remaining[begin] and remaining[end] <= _POOL_LENGTH
        builder.add_shape_mesh(-1, mesh=belt, cfg=pool_cfg if on_pool else belt_cfg, label=label + "/Deck/collision")
        if not motor_off:
            velocity = np.repeat(tangent[region] * speed[region, None], 2, axis=0).astype(np.float32)
            deck_label = label + "/Deck"
            scene.belts.append((belt, velocity, deck_label))
            if np.any(lateral[region]):
                pool_crossbelt[deck_label] = np.repeat(sides * lateral[region, None], 2, axis=0).astype(np.float32)
        for side in (-1, 1):
            edge = center + side * guide_width[region, None] * sides
            bottom = edge + guide_base
            top = edge + guide_top
            vertices, indices = _strip(top, bottom) if side < 0 else _strip(bottom, top)
            rail = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=not motor_off)
            rail_label = f"{label}/Guide_{'left' if side > 0 else 'right'}"
            builder.add_shape_mesh(-1, mesh=rail, cfg=rail_cfg, label=rail_label + "/collision")
            if not motor_off:
                scene.belts.append((rail, velocity, rail_label))
    if inlet_speed is not None:
        scene.metadata["process_handoff"] = (points[-1], _POOL_UPPER_GUIDE, _POOL_HALF_WIDTH + 0.03)
        # The upper belt engages before the end of the crossbelt. Match the
        # lower mat's local velocity until the bottles reach the IFM.
        region = remaining <= _POOL_UPPER_GUIDE
        start = remaining[region][0]
        entry = np.clip(1.0 - (start - remaining[region]) / 0.25, 0.0, 1.0)
        cap = points[region].copy()
        cap[:, 2] += 0.230 + 0.012 * _ease(entry)
        cap_edges = deck_width[region, None] * left[region]
        # The receiving strands support the entire heel before release.
        downstream = np.linspace(0.0, 0.60, 17)[1:]
        extension = cap[-1] + downstream[:, None] * tangent[-1]
        release = np.clip((downstream - 0.30) / 0.30, 0.0, 1.0)
        extension[:, 2] += 0.012 * _ease(release)
        cap = np.vstack((cap, extension))
        cap_edges = np.vstack((cap_edges, np.tile(cap_edges[-1], (len(downstream), 1))))
        vertices, indices = _strip(cap + cap_edges, cap - cap_edges)
        roof = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=not motor_off)
        roof_label = "/Plant/Process/Transport/UpperGuide"
        builder.add_shape_mesh(-1, mesh=roof, cfg=keeper_cfg, label=roof_label + "/collision")
        if not motor_off:
            longitudinal = tangent[region] * speed[region, None]
            longitudinal = np.vstack((longitudinal, np.tile(longitudinal[-1], (len(downstream), 1))))
            scene.belts.append((roof, np.repeat(longitudinal, 2, axis=0).astype(np.float32), roof_label))
            cross = left[region] * lateral[region, None]
            cross = np.vstack((cross, np.zeros_like(extension)))
            pool_crossbelt[roof_label] = np.repeat(cross, 2, axis=0).astype(np.float32)
    scene.metadata["process_pool_crossbelt"] = pool_crossbelt


def add_filling_bottles(scene, path, *, motor_off=False, inlet_speed=None):
    """Seed live stock with the same material and collision geometry as the IFM."""
    builder = scene.builder
    source = scene.bottle_bodies[0]
    shapes = bottle_shape_templates(builder, source)
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    # Start the line in production: single-file stock at the rated admission
    # pitch, so the accumulation table receives the rated output from the start
    # rather than after the transit time. Leave a short clear run into the live
    # IFM so its original stock can start moving before the handoff.
    samples = np.arange(0.35, length[-1] - 1.5, SINGLE_FILE_PITCH)
    centers = np.column_stack([np.interp(samples, length, column) for column in path.T])
    tangents = np.gradient(path, length, axis=0)
    direction = np.column_stack([np.interp(samples, length, axis) for axis in tangents.T])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    velocities = direction * _transport_speed(length[-1] - samples, inlet_speed)[:, None]
    for index, position in enumerate(centers):
        position[2] += 0.001
        label = f"/Plant/Sim/Product/Bottles/WetEnd_{index:04d}"
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
    scene.metadata["process_bottles"] = len(samples)


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


def open_pool_passage(stage, path):
    """Leave the source doorway frame and provide a physical conveyor opening."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415 - USD is optional.

    # The vehicle proxy merges a doorway with the conveyor and spans the
    # otherwise unobstructed transport path. Keep its four surrounding pieces.
    start, end = path[-2:]
    width = _POOL_HALF_WIDTH + 0.015
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
