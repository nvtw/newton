# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared material birth on a certified connected planar triangle union.

This changes only per-point material history. Original point cones, impulses,
contact witnesses, response, and solve ordering are retained.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer


@wp.struct
class Domain:
    triangles: wp.array2d[wp.vec3]
    normal: wp.vec3
    origin: wp.vec3
    mesh_shape: int
    plane_shapes: wp.array[int]
    active: wp.array[int]
    birth: wp.array[float]
    previous_member: wp.array[int]
    member: wp.array[int]
    previous_count: wp.array[int]
    other_shape: wp.array[int]
    stats: wp.array[int]


@wp.func
def contains(domain: Domain, point: wp.vec3):
    eps = wp.float32(8.0 * 1.1920928955078125e-7)
    bound = eps * wp.dot(wp.abs(point) + wp.abs(domain.origin), wp.abs(domain.normal))
    if wp.abs(wp.dot(point - domain.origin, domain.normal)) > bound:
        return False
    inside = wp.bool(False)
    for tri in range(domain.triangles.shape[0]):
        accepted = wp.bool(True)
        for j in range(3):
            a = domain.triangles[tri, j]
            b = domain.triangles[tri, (j + 1) % 3]
            edge = b - a
            delta = point - a
            signed = wp.dot(wp.cross(edge, delta), domain.normal)
            edge_error = eps * wp.length(wp.abs(a) + wp.abs(b))
            delta_error = eps * wp.length(wp.abs(point) + wp.abs(a))
            error = (
                eps * wp.length(edge) * wp.length(delta) + edge_error * wp.length(delta) + delta_error * wp.length(edge)
            )
            if signed < -error:
                accepted = False
        if accepted:
            inside = True
            break
    return inside


@wp.kernel
def update(domain: Domain, bodies: BodyContainer, contacts: ContactViews, cc: ContactContainer, count: wp.array[int]):
    # One certified domain; serial ownership avoids simultaneous birth writes.
    n = wp.min(count[0], domain.member.shape[0])
    first = int(-1)
    other = int(-1)
    conflict = wp.bool(False)
    for k in range(n):
        domain.member[k] = 0
        sa = contacts.rigid_contact_shape0[k]
        sb = contacts.rigid_contact_shape1[k]
        point = contacts.rigid_contact_point0[k]
        plane = sb
        if sb == domain.mesh_shape:
            point = contacts.rigid_contact_point1[k]
            plane = sa
        if (sa == domain.mesh_shape or sb == domain.mesh_shape) and domain.plane_shapes[plane] != 0:
            if contains(domain, point):
                domain.member[k] = 1
                if first < 0:
                    first = k
                    other = plane
                elif other != plane:
                    conflict = True
    broken = wp.bool(False)
    # Projection may mark an unloaded speculative point as broken. It has no
    # material friction state to invalidate, so only a loaded sliding point
    # breaks the patch. These are FP32 impulse comparisons, not damping.
    for k in range(domain.previous_count[0]):
        if domain.previous_member[k] != 0:
            normal_load = wp.abs(cc.prev_impulses[0, k])
            tangential_load = wp.sqrt(
                cc.prev_impulses[1, k] * cc.prev_impulses[1, k]
                + cc.prev_impulses[2, k] * cc.prev_impulses[2, k]
            )
            if (cc.prev_lambdas[12, k] != 0.0
                    and normal_load > wp.float32(1.0e-7)
                    and tangential_load > wp.float32(1.0e-8)):
                broken = True
    domain.stats[0] += 1
    if first < 0 or conflict:
        domain.active[0] = 0
        domain.stats[1] += 1
    else:
        sa = contacts.rigid_contact_shape0[first]
        sb = contacts.rigid_contact_shape1[first]
        b0 = contacts.shape_body[sa]
        b1 = contacts.shape_body[sb]
        p0 = bodies.position[b0]
        p1 = bodies.position[b1]
        q0 = bodies.orientation[b0]
        q1 = bodies.orientation[b1]
        mesh_body = b0
        plane_body = b1
        plane_point = contacts.rigid_contact_point1[first]
        normal = (
            cc.lambdas[0, first] * wp.vec3(1.0, 0.0, 0.0)
            + cc.lambdas[1, first] * wp.vec3(0.0, 1.0, 0.0)
            + cc.lambdas[2, first] * wp.vec3(0.0, 0.0, 1.0)
        )
        if sb == domain.mesh_shape:
            mesh_body = b1
            plane_body = b0
            plane_point = contacts.rigid_contact_point0[first]
            normal = -normal
        plane_world = bodies.position[plane_body] + wp.quat_rotate(
            bodies.orientation[plane_body], plane_point - bodies.body_com[plane_body]
        )
        mesh_normal = wp.quat_rotate(bodies.orientation[mesh_body], domain.normal)
        coplanar = wp.dot(mesh_normal, normal) >= wp.float32(0.999)
        margin = contacts.rigid_contact_margin0[first] + contacts.rigid_contact_margin1[first]
        for triangle in range(domain.triangles.shape[0]):
            for vertex in range(3):
                local = domain.triangles[triangle, vertex]
                point = bodies.position[mesh_body] + wp.quat_rotate(
                    bodies.orientation[mesh_body], local - bodies.body_com[mesh_body]
                )
                gap = wp.dot(plane_world - point, normal) - margin
                # Include the operands before COM subtraction and quaternion
                # rotation, not only their small final world-space result.
                error = wp.float32(8.0 * 1.1920928955078125e-7) * (
                    wp.dot(wp.abs(bodies.position[mesh_body]) + wp.abs(bodies.position[plane_body]), wp.abs(normal))
                    + wp.length(wp.abs(local) + wp.abs(bodies.body_com[mesh_body]))
                    + wp.length(wp.abs(plane_point) + wp.abs(bodies.body_com[plane_body]))
                    + wp.abs(margin)
                )
                if wp.abs(gap) > error:
                    coplanar = False
        # Coplanarity certifies a new material reference. Once active, small
        # witness motion and FP32 transform error must not rebirth the patch;
        # membership and the explicit friction-break flag govern continuity.
        if not coplanar and domain.active[0] == 0:
            domain.stats[2] += 1
        else:
            if broken or domain.other_shape[0] != other:
                domain.active[0] = 0
                domain.stats[3] += 1
            if domain.active[0] == 0:
                for j in range(3):
                    domain.birth[j] = p0[j]
                    domain.birth[3 + j] = p1[j]
                for j in range(4):
                    domain.birth[6 + j] = q0[j]
                    domain.birth[10 + j] = q1[j]
                domain.active[0] = 1
                domain.other_shape[0] = other
                domain.stats[4] += 1
            else:
                domain.stats[5] += 1
            p0b = wp.vec3(domain.birth[0], domain.birth[1], domain.birth[2])
            p1b = wp.vec3(domain.birth[3], domain.birth[4], domain.birth[5])
            q0b = wp.quat(domain.birth[6], domain.birth[7], domain.birth[8], domain.birth[9])
            q1b = wp.quat(domain.birth[10], domain.birth[11], domain.birth[12], domain.birth[13])
            for k in range(n):
                if domain.member[k] != 0:
                    # New quadrature witness samples the same already touching
                    # material domain; retain its ORIGINAL physical force point.
                    witness0 = wp.quat_rotate(q0, contacts.rigid_contact_point0[k] - bodies.body_com[b0])
                    witness1 = wp.quat_rotate(q1, contacts.rigid_contact_point1[k] - bodies.body_com[b1])
                    # Same common physical point as native birth, evaluated in
                    # body1-relative coordinates to avoid world cancellation.
                    lever1 = wp.float32(0.5) * (
                        (p0 - p1)
                        + witness0
                        + witness1
                        + (contacts.rigid_contact_margin0[k] - contacts.rigid_contact_margin1[k])
                        * contacts.rigid_contact_normal[k]
                    )
                    a1 = bodies.body_com[b1] + wp.quat_rotate_inv(q1, lever1)
                    for j in range(3):
                        cc.lambdas[9 + j, k] = a1[j]
                    relative_birth = (p1b - p0b) + wp.quat_rotate(q1b, a1 - bodies.body_com[b1])
                    a0 = bodies.body_com[b0] + wp.quat_rotate_inv(q0b, relative_birth)
                    for j in range(3):
                        cc.lambdas[6 + j, k] = a0[j]
                    for j in range(14):
                        cc.lambdas[13 + j, k] = domain.birth[j]
                    domain.stats[6] += 1
    for k in range(n):
        domain.previous_member[k] = domain.member[k]
    domain.previous_count[0] = n


def install(solver, model, output, certificate):
    """Bind an explicitly certified source-domain asset to matching model geometry."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    import newton

    path = Path(certificate)
    data = np.load(path)
    triangles = data["triangles_621"].astype(np.float32)
    normal = data["normal_621"].astype(np.float32)
    origin = data["origin_621"].astype(np.float32)
    unique = np.unique(triangles.reshape(-1, 3), axis=0)
    scales = model.shape_scale.numpy()
    transforms = model.shape_transform.numpy()
    candidates = []
    tolerance = float(16 * np.finfo(np.float32).eps * np.max(abs(unique)))
    for i, mesh in enumerate(model.shape_source):
        if mesh is None or not hasattr(mesh, "vertices"):
            continue
        points = np.asarray(mesh.vertices) * scales[i]
        points = Rotation.from_quat(transforms[i, 3:]).apply(points) + transforms[i, :3]
        if len(points) >= len(unique) and np.max(cKDTree(points).query(unique)[0]) <= tolerance:
            candidates.append(i)
    assert len(candidates) == 1, ("Domain must identify exactly one source mesh", candidates)
    w = solver.world
    cc = w._contact_container
    domain = Domain()
    domain.triangles = wp.array(triangles, dtype=wp.vec3, device=cc.lambdas.device)
    domain.normal = wp.vec3(*normal)
    domain.origin = wp.vec3(*origin)
    domain.mesh_shape = candidates[0]
    plane = (model.shape_type.numpy() == int(newton.GeoType.PLANE)) & (model.shape_body.numpy() < 0)
    domain.plane_shapes = wp.array(plane.astype(np.int32), dtype=int, device=cc.lambdas.device)
    for name in ("active", "previous_count", "other_shape"):
        setattr(domain, name, wp.zeros(1, dtype=int, device=cc.lambdas.device))
    domain.birth = wp.zeros(14, dtype=float, device=cc.lambdas.device)
    domain.member = wp.zeros(cc.lambdas.shape[1], dtype=int, device=cc.lambdas.device)
    domain.previous_member = wp.zeros_like(domain.member)
    domain.stats = wp.zeros(7, dtype=int, device=cc.lambdas.device)
    old = w._ingest_and_warmstart_contacts

    def ingest(*args, **kwargs):
        result = old(*args, **kwargs)
        wp.launch(
            update, 1, inputs=[domain, w.bodies, w._contact_views, cc, w._cc_valid_count], device=cc.lambdas.device
        )
        return result

    w._ingest_and_warmstart_contacts = ingest

    def finish():
        Path(output).with_suffix(".certified_patch.json").write_text(
            json.dumps(
                {
                    "certificate": str(path),
                    "mesh_shape": domain.mesh_shape,
                    "stats_fields": [
                        "updates",
                        "no_domain_or_conflict",
                        "not_whole_domain_touching",
                        "slid_or_changed_plane",
                        "births",
                        "retained",
                        "point_references_assigned",
                    ],
                    "stats": domain.stats.numpy().tolist(),
                    "scope": __doc__,
                    "coplanarity": "Scale-aware FP32 arithmetic bound, not contact-offset tolerance",
                },
                indent=2,
            )
        )

    return finish
