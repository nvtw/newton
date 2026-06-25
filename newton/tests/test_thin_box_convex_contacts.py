# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for convex/primitive contacts against extreme thin boxes.

A table is commonly modelled as a thin box primitive with one tiny half-extent
and two large ones (e.g. ``(10, 10, 0.1)`` -- a 100:1 aspect ratio). When a much
smaller shape contacts such a box away from its center, the box's geometric
center is far from the contact region, so the MPR/GJK center-to-center initial
search direction points sideways. MPR then converges on a *side* face and reports
a bogus deep penetration with a near-horizontal (or flipped) normal, instead of
the correct shallow top-face contact.

:func:`~newton._src.geometry.mpr.geometric_center` fixes this by seeding ``v0``
from the closest point on the box to the partner's center whenever that center
lies outside the box (the resting/shallow regime), keeping ``v0`` laterally
aligned with the partner so the initial ray is ~perpendicular to the contact
face.

These tests place each shape type near the edge of an extreme thin box and assert
the contact normal points along +Z (out of the top face) with a distance that
tracks the placed separation -- across separated, shallow, and penetrating poses.
Without the fix, the convex/cylinder/cone/ellipsoid cases report a side-face
normal (``|nz| ~ 0``) and ~100x-too-deep penetration.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Extreme thin table with all-different half-extents (non-uniform, ~80:1 thinnest
# aspect). Top face at z = TABLE_TOP.
TABLE_HALF = (8.0, 6.0, 0.1)
TABLE_TOP = 0.1
GAP = 0.1  # large gap so contacts are found well before penetration
EDGE_X = 7.7  # near the +X edge (table edge at x = 8), well inside the footprint

# Convex hull whose local origin is HEAVILY offset from its geometry (~7 m) --
# exercises the AABB-center interior point together with the large-flat-box v0
# projection: a side-face flip here would mean the offset is mishandled.
_CUBE = np.array(
    [[sx, sy, sz] for sx in (-0.08, 0.08) for sy in (-0.08, 0.08) for sz in (-0.08, 0.08)],
    dtype=np.float32,
)
_HULL_OFFSET = np.array([5.0, -4.0, 3.0], dtype=np.float32)
_HULL_VERTS = _CUBE + _HULL_OFFSET
_HULL_INDICES = np.array(
    [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3],
    dtype=np.int32,
)

# placed separation of the shape's lowest point relative to the table top [m]
SEPARATIONS = (0.05, 0.0, -0.005, -0.01)

NORMAL_Z_MIN = 0.9  # contact normal must be within ~26 deg of +Z (no side/flip)
DIST_TOL = 1.0e-3  # reported signed distance must track the placed separation


def _quat_rotate(q, v):
    qv = np.asarray(q[:3], np.float64)
    t = 2.0 * np.cross(qv, v)
    return np.asarray(v, np.float64) + float(q[3]) * t + np.cross(qv, t)


def _add_shape(builder, body, kind):
    """Add a shape of ``kind``; return (shape_idx, lowest_offset, lateral_offset)."""
    if kind == "sphere":
        return builder.add_shape_sphere(body, radius=0.1, label=kind), 0.1, (0.0, 0.0)
    if kind == "box":
        return builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label=kind), 0.1, (0.0, 0.0)
    if kind == "ellipsoid":
        return builder.add_shape_ellipsoid(body, rx=0.15, ry=0.1, rz=0.08, label=kind), 0.08, (0.0, 0.0)
    if kind == "capsule":  # lying flat (axis X): lowest point = radius below center
        return builder.add_shape_capsule(body, radius=0.05, half_height=0.15, label=kind), 0.05, (0.0, 0.0)
    if kind == "cylinder":  # lying flat (axis X)
        return builder.add_shape_cylinder(body, radius=0.1, half_height=0.15, label=kind), 0.1, (0.0, 0.0)
    if kind == "cone":  # axis Z
        return builder.add_shape_cone(body, radius=0.12, half_height=0.15, label=kind), 0.15, (0.0, 0.0)
    if kind == "convex":
        mesh = newton.Mesh(_HULL_VERTS.copy(), _HULL_INDICES.copy(), compute_inertia=False, is_solid=True)
        shape = builder.add_shape_convex_hull(body, mesh=mesh, label=kind)
        # geometry is offset from the body origin by _HULL_OFFSET; report it so the
        # caller can place the hull's geometry (not its origin) over the table.
        return shape, 0.08, (float(_HULL_OFFSET[0]), float(_HULL_OFFSET[1]))
    raise ValueError(kind)


def _lying_flat(kind):
    return kind in ("capsule", "cylinder")


def _build(device, kind, bottom_z, x=EDGE_X):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = GAP
    builder.default_shape_cfg.is_hydroelastic = False

    table = builder.add_body(xform=((0, 0, 0), (0, 0, 0, 1)), is_kinematic=True, label="table")
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi * 0.5) if _lying_flat(kind) else wp.quat_identity()
    body = builder.add_body(xform=((0, 0, 0), tuple(q)), is_kinematic=True, label="obj")

    ts = builder.add_shape_box(table, hx=TABLE_HALF[0], hy=TABLE_HALF[1], hz=TABLE_HALF[2], label="table")
    os, low, (off_x, off_y) = _add_shape(builder, body, kind)

    # Place the body so the shape's geometry sits at (x, 0) with its lowest point at
    # bottom_z. For the off-origin hull, subtract its lateral/vertical geometry
    # offset; the hull's AABB half-height equals ``low``.
    if kind == "convex":
        pos = wp.vec3(x - off_x, -off_y, bottom_z - float(_HULL_OFFSET[2]) + low)
    else:
        pos = wp.vec3(x, 0.0, bottom_z + low)
    builder.body_q[body] = wp.transform(pos, q)

    flags = int(newton.ShapeFlags.VISIBLE) | int(newton.ShapeFlags.COLLIDE_SHAPES)
    for s in (ts, os):
        builder.shape_flags[s] = int(builder.shape_flags[s]) | flags

    model = builder.finalize(device=device)
    return model, model.state(), ts, os


def _pair_contacts(contacts, model, state, ts, os):
    """Return (min_normal_z, [signed_gaps]) for table<->object contacts (table->object +Z)."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    n = min(n, contacts.rigid_contact_max)
    if n == 0:
        return None, []
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    nm = contacts.rigid_contact_normal.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    m0 = contacts.rigid_contact_margin0.numpy()[:n]
    m1 = contacts.rigid_contact_margin1.numpy()[:n]
    body_q = state.body_q.numpy()
    shape_body = model.shape_body.numpy()

    def to_world(shape, local):
        b = int(shape_body[shape])
        if b < 0:
            return np.asarray(local, np.float64)
        tr = body_q[b]
        return np.asarray(tr[:3], np.float64) + _quat_rotate(tr[3:], local)

    nzs, gaps = [], []
    for i in range(n):
        a, b = int(s0[i]), int(s1[i])
        if {a, b} != {ts, os}:
            continue
        nz = float(nm[i][2]) if a == ts else float(-nm[i][2])  # table -> object should be +Z
        gap = float(np.dot(nm[i], to_world(b, p1[i]) - to_world(a, p0[i])) - m0[i] - m1[i])
        nzs.append(nz)
        gaps.append(gap)
    if not nzs:
        return None, []
    return min(nzs), gaps


def test_thin_box_contact_normals(test, device):
    """Every shape on an extreme thin box near its edge must report a +Z top-face
    contact whose distance tracks the placed separation -- never a side-face flip."""
    # NOTE: box-box overhanging the edge at penetration remains a known MPR
    # limitation (handled by the analytical box-box path, not the v0 seed); the
    # near-edge, on-footprint box case validated here is unaffected.
    shapes = ["sphere", "box", "ellipsoid", "capsule", "cylinder", "cone", "convex"]
    with test.subTest(device=str(device)):
        failures = []
        for kind in shapes:
            for sep in SEPARATIONS:
                model, state, ts, os = _build(device, kind, TABLE_TOP + sep)
                pipeline = newton.CollisionPipeline(model, reduce_contacts=False, broad_phase="sap")
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                min_nz, gaps = _pair_contacts(contacts, model, state, ts, os)
                if min_nz is None:
                    failures.append(f"{kind} sep={sep * 1000:+.0f}mm: no contact within gap")
                    continue
                if min_nz < NORMAL_Z_MIN:
                    failures.append(f"{kind} sep={sep * 1000:+.0f}mm: normal_z={min_nz:+.3f} (side/flip)")
                if not any(abs(g - sep) < DIST_TOL for g in gaps):
                    dmm = [round(g * 1e3, 1) for g in gaps]
                    failures.append(f"{kind} sep={sep * 1000:+.0f}mm: dist {dmm}mm != {sep * 1e3:.0f}mm")
        test.assertEqual(len(failures), 0, "\n" + "\n".join(failures))


def test_box_box_overhang(test, device):
    """A box overhanging the thin table's edge while penetrating must still report a
    +Z top-face contact. This is the corner-contact case the analytical box-box path
    handles (the v0 seed alone resolves an edge contact ambiguously -> side flip)."""
    overhang_x = TABLE_HALF[0] * 1.01  # box center just past the +X edge
    with test.subTest(device=str(device)):
        failures = []
        for sep in (0.0, -0.005, -0.01):
            model, state, ts, os = _build(device, "box", TABLE_TOP + sep, x=overhang_x)
            pipeline = newton.CollisionPipeline(model, reduce_contacts=False, broad_phase="sap")
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            min_nz, _gaps = _pair_contacts(contacts, model, state, ts, os)
            if min_nz is None:
                failures.append(f"sep={sep * 1000:+.0f}mm: no contact")
            elif min_nz < NORMAL_Z_MIN:
                failures.append(f"sep={sep * 1000:+.0f}mm: normal_z={min_nz:+.3f} (side flip)")
        test.assertEqual(len(failures), 0, "\n" + "\n".join(failures))


class TestThinBoxConvexContacts(unittest.TestCase):
    pass


add_function_test(
    TestThinBoxConvexContacts,
    "test_thin_box_contact_normals",
    test_thin_box_contact_normals,
    devices=get_test_devices(),
)
add_function_test(
    TestThinBoxConvexContacts,
    "test_box_box_overhang",
    test_box_box_overhang,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
