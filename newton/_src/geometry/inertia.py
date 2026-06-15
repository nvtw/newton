# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for computing rigid body inertia properties."""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from .types import (
    GeoType,
    Heightfield,
    Mesh,
    Vec3,
)

# Relative tolerance for eigenvalue positivity checks.  An eigenvalue is
# considered "near-zero" only when it is smaller than this fraction of the
# largest eigenvalue.  This prevents spurious inflation of physically correct
# but small inertia values (e.g. lightweight gripper pads).
_INERTIA_REL_TOL = 1.0e-6

# Absolute floor for the eigenvalue check when max_eigenvalue itself is ~0
# (degenerate tensor).  Must be well below the smallest physically meaningful
# eigenvalue we want to preserve (order ~1e-7 for lightweight gripper pads).
_INERTIA_ABS_FLOOR = 1.0e-10

# Absolute value used when an eigenvalue correction *is* triggered.  This
# keeps the corrected tensor well-conditioned (e.g. singular inertia [0,0,0]
# becomes [1e-6, 1e-6, 1e-6]).
_INERTIA_ABS_ADJUSTMENT = 1.0e-6

# Match numpy's default np.allclose() tolerances when deciding whether a
# nearly-symmetric tensor should be treated as unchanged.
_INERTIA_SYMMETRY_RTOL = 1.0e-5
_INERTIA_SYMMETRY_ATOL = 1.0e-8


def compute_inertia_sphere(density: float, radius: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid sphere

    Args:
        density: The sphere density [kg/m³]
        radius: The sphere radius [m]

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """

    v = 4.0 / 3.0 * wp.pi * radius * radius * radius

    m = density * v
    Ia = 2.0 / 5.0 * m * radius * radius

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_inertia_sphere_from_mass(mass: float, radius: float) -> wp.mat33:
    """Helper to compute 3x3 inertia matrix of a solid box with given mass and half-extents.

    Args:
        mass: The box mass [kg]
        hx: The box half-extent along the x-axis [m]
        hy: The box half-extent along the y-axis [m]
        hz: The box half-extent along the z-axis [m]

    Returns:

        A 3x3 inertia matrix with inertia specified around the center of mass
    """
    Ia = 0.4 * mass * radius * radius
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])
    return I


def compute_inertia_capsule(density: float, radius: float, half_height: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid capsule extending along the z-axis

    Args:
        density: The capsule density [kg/m³]
        radius: The capsule radius [m]
        half_height: Half-length of the cylindrical section (excluding hemispherical caps) [m]

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """

    h = 2.0 * half_height  # full height of the cylindrical section

    ms = density * (4.0 / 3.0) * wp.pi * radius * radius * radius
    mc = density * wp.pi * radius * radius * h

    # total mass
    m = ms + mc

    # adapted from ODE
    Ia = mc * (0.25 * radius * radius + (1.0 / 12.0) * h * h) + ms * (
        0.4 * radius * radius + 0.375 * radius * h + 0.25 * h * h
    )
    Ib = (mc * 0.5 + ms * 0.4) * radius * radius

    # For Z-axis orientation: I_xx = I_yy = Ia, I_zz = Ib
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])

    return (m, wp.vec3(), I)


def compute_inertia_cylinder(density: float, radius: float, half_height: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid cylinder extending along the z-axis

    Args:
        density: The cylinder density [kg/m³]
        radius: The cylinder radius [m]
        half_height: The half-height of the cylinder along the z-axis [m]

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """

    h = 2.0 * half_height  # full height

    m = density * wp.pi * radius * radius * h

    Ia = 1 / 12 * m * (3 * radius * radius + h * h)
    Ib = 1 / 2 * m * radius * radius

    # For Z-axis orientation: I_xx = I_yy = Ia, I_zz = Ib
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])

    return (m, wp.vec3(), I)


def compute_inertia_cone(density: float, radius: float, half_height: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid cone extending along the z-axis

    Args:
        density: The cone density [kg/m³]
        radius: The cone base radius [m]
        half_height: The half-height of the cone (distance from geometric center to base or apex) [m]

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """

    h = 2.0 * half_height  # full height

    m = density * wp.pi * radius * radius * h / 3.0

    # Center of mass is at -h/4 from the geometric center
    # Since the cone has base at -h/2 and apex at +h/2, the COM is 1/4 of the height from base toward apex
    com = wp.vec3(0.0, 0.0, -h / 4.0)

    # Inertia about the center of mass
    Ia = 3 / 20 * m * radius * radius + 3 / 80 * m * h * h
    Ib = 3 / 10 * m * radius * radius

    # For Z-axis orientation: I_xx = I_yy = Ia, I_zz = Ib
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])

    return (m, com, I)


def compute_inertia_ellipsoid(density: float, rx: float, ry: float, rz: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid ellipsoid

    The ellipsoid is centered at the origin with semi-axes rx, ry, rz along the x, y, z axes respectively.

    Args:
        density: The ellipsoid density [kg/m³]
        rx: The semi-axis along the x-axis [m]
        ry: The semi-axis along the y-axis [m]
        rz: The semi-axis along the z-axis [m]

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """
    # Volume of ellipsoid: V = (4/3) * pi * rx * ry * rz
    v = (4.0 / 3.0) * wp.pi * rx * ry * rz
    m = density * v

    # Inertia tensor for a solid ellipsoid about its center of mass:
    # Ixx = (1/5) * m * (ry² + rz²)
    # Iyy = (1/5) * m * (rx² + rz²)
    # Izz = (1/5) * m * (rx² + ry²)
    Ixx = (1.0 / 5.0) * m * (ry * ry + rz * rz)
    Iyy = (1.0 / 5.0) * m * (rx * rx + rz * rz)
    Izz = (1.0 / 5.0) * m * (rx * rx + ry * ry)

    I = wp.mat33([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])

    return (m, wp.vec3(), I)


def compute_inertia_triangle_prism(
    density: float,
    edge_ab: float,
    c_y: float,
    c_z: float,
    margin: float,
) -> tuple[float, wp.vec3, wp.mat33]:
    """Mass and centroidal inertia of a thin triangular prism.

    The canonical triangle has vertex A at the local origin, vertex B at
    ``(0, 0, edge_ab)`` along the local +Z axis, and vertex C at
    ``(0, c_y, c_z)`` in the local YZ plane. The prism is the triangle
    extruded symmetrically along the local X axis by ``±margin``, so it
    spans ``x ∈ [-margin, +margin]`` with full thickness ``t = 2·margin``.

    Args:
        density: Material density [kg/m³].
        edge_ab [m]: Length ``|AB|`` of the first edge (local +Z direction).
        c_y [m]: Vertex C's local Y coordinate.
        c_z [m]: Vertex C's local Z coordinate.
        margin [m]: Half-thickness of the prism along the local X axis;
            the full thickness is ``2 * margin``. Must be non-negative.

    Returns:
        A tuple of ``(mass, com, inertia)`` where ``com`` is in the local
        frame and ``inertia`` is expressed about ``com``. When the
        thickness or triangle area is zero, returns
        ``(0.0, centroid, mat33())`` (zero inertia, but a sensible COM
        offset so external mass / inertia overrides resolve correctly).
    """
    L = float(edge_ab)
    cy = float(c_y)
    cz = float(c_z)
    t = 2.0 * float(margin)

    # Centroid of the triangle in the local frame:
    # mean of (0,0,0), (0,0,L), (0,c_y,c_z).
    com_y = cy / 3.0
    com_z = (L + cz) / 3.0
    com = wp.vec3(0.0, com_y, com_z)

    area = 0.5 * abs(L * cy)  # half magnitude of AB cross AC

    if t <= 0.0 or area == 0.0 or density <= 0.0:
        return 0.0, com, wp.mat33()

    mass = density * area * t

    # Vertex offsets relative to centroid (q_i = p_i - com), local frame.
    # Vertex A = (0, 0, 0), B = (0, 0, L), C = (0, c_y, c_z).
    qa_y, qa_z = -com_y, -com_z
    qb_y, qb_z = -com_y, L - com_z
    qc_y, qc_z = cy - com_y, cz - com_z

    # Triangle planar second moments about the centroid, normalised by area:
    #   I^(2D)_ab / A = (1/12) Σ q_i^a q_i^b.
    # (Standard result, e.g. Wikipedia "List of second moments of area".)
    s_yy = qa_y * qa_y + qb_y * qb_y + qc_y * qc_y
    s_zz = qa_z * qa_z + qb_z * qb_z + qc_z * qc_z
    s_yz = qa_y * qa_z + qb_y * qb_z + qc_y * qc_z

    inv12 = 1.0 / 12.0
    iyy_over_area = inv12 * s_yy
    izz_over_area = inv12 * s_zz
    iyz_over_area = inv12 * s_yz

    # 3D inertia of the prism about its centroid. The prism is symmetric
    # in x ∈ [-t/2, t/2] so I_xy = I_xz = 0 and the x-thickness contributes
    # m·t²/12 to I_yy and I_zz only.
    t2_over_12 = (t * t) * inv12
    Ixx = mass * (iyy_over_area + izz_over_area)
    Iyy = mass * (t2_over_12 + izz_over_area)
    Izz = mass * (t2_over_12 + iyy_over_area)
    Iyz = -mass * iyz_over_area  # tensor convention: -∫ y·z dm

    inertia = wp.mat33(
        [
            [Ixx, 0.0, 0.0],
            [0.0, Iyy, Iyz],
            [0.0, Iyz, Izz],
        ]
    )
    return mass, com, inertia


def compute_inertia_tetrahedron(
    density: float,
    edge_ab: float,
    c_y: float,
    c_z: float,
    d_x: float,
    d_y: float,
    d_z: float,
) -> tuple[float, wp.vec3, wp.mat33]:
    """Mass and centroidal inertia of a uniform-density solid tetrahedron.

    The canonical tetrahedron has vertices

    - ``A = (0, 0, 0)``
    - ``B = (0, 0, edge_ab)``
    - ``C = (0, c_y, c_z)``
    - ``D = (d_x, d_y, d_z)``

    in the local frame.

    Mass is exactly ``density * |edge_ab * c_y * d_x| / 6`` (the absolute
    determinant ``|AB . (AC x AD)|`` equals ``|edge_ab * c_y * d_x|``
    after expanding the canonical-frame components). The centre of mass
    is the four-vertex centroid.

    The inertia tensor about the centroid is computed in closed form
    using `Tonon (2005), Explicit Exact Formulas for the 3-D Tetrahedron
    Inertia Tensor in Terms of its Vertex Coordinates
    <https://thescipub.com/abstract/jmssp.2005.8.11>`_, then shifted
    from the local origin to the centroid via the parallel-axis
    theorem. The inertia tensor is symmetric and full (off-diagonal
    products of inertia in general are non-zero unless ``D`` happens
    to lie on a symmetry axis).

    Args:
        density [kg/m³]: Material density.
        edge_ab [m]: Length ``|AB|`` of the canonical first edge along
            local +Z.
        c_y [m]: Vertex C's local Y coordinate.
        c_z [m]: Vertex C's local Z coordinate.
        d_x [m]: Vertex D's local X coordinate.
        d_y [m]: Vertex D's local Y coordinate.
        d_z [m]: Vertex D's local Z coordinate.

    Returns:
        ``(mass, com, inertia)``. When the volume is zero (degenerate /
        coplanar tet) or ``density <= 0``, returns
        ``(0.0, centroid, mat33())``.
    """
    e = float(edge_ab)
    cy = float(c_y)
    cz = float(c_z)
    dx = float(d_x)
    dy = float(d_y)
    dz = float(d_z)

    # Centroid (COM for uniform density) is the mean of the four vertices.
    com_x = dx * 0.25
    com_y = (cy + dy) * 0.25
    com_z = (e + cz + dz) * 0.25
    com = wp.vec3(com_x, com_y, com_z)

    # Volume = | AB . (AC x AD) | / 6 = | e * cy * dx | / 6.
    vol = abs(e * cy * dx) / 6.0
    if vol == 0.0 or density <= 0.0:
        return 0.0, com, wp.mat33()

    mass = density * vol
    det_abs = 6.0 * vol  # |DET(J)| in Tonon's notation

    # Tonon (2005) eq. (9a-f) collapse to:
    #     a  = ρ |DET(J)| ((Sy^2 + Sy2) + (Sz^2 + Sz2)) / 120
    #     b  = ρ |DET(J)| ((Sx^2 + Sx2) + (Sz^2 + Sz2)) / 120
    #     c  = ρ |DET(J)| ((Sx^2 + Sx2) + (Sy^2 + Sy2)) / 120
    #     a' = ρ |DET(J)| (Syz + Sy*Sz) / 120
    #     b' = ρ |DET(J)| (Sxz + Sx*Sz) / 120
    #     c' = ρ |DET(J)| (Sxy + Sx*Sy) / 120
    # where Sx  = sum_i x_i, Sx2 = sum_i x_i^2, Sxy = sum_i x_i y_i, etc.
    # (Derived from the Tonon expansion: the sum-of-pairs structure
    # ``y_i^2 + y_i y_j + y_j^2 + ...`` rearranges into
    # ``(sum_i y_i)^2 + sum_i y_i^2`` summed and divided by 2.)
    #
    # In the canonical local frame:
    #   Sx  = dx,             Sx2 = dx^2
    #   Sy  = cy + dy,        Sy2 = cy^2 + dy^2
    #   Sz  = e + cz + dz,    Sz2 = e^2 + cz^2 + dz^2
    #   Sxy = dx * dy,        Sxz = dx * dz,    Syz = cy * cz + dy * dz
    Sx = dx
    Sy = cy + dy
    Sz = e + cz + dz
    Sx2 = dx * dx
    Sy2 = cy * cy + dy * dy
    Sz2 = e * e + cz * cz + dz * dz
    Sxy = dx * dy
    Sxz = dx * dz
    Syz = cy * cz + dy * dz

    coeff = density * det_abs / 120.0
    # Inertia tensor about the origin (Tonon eq. 2 with diagonal =
    # moments-of-inertia and off-diagonal = -product-of-inertia).
    a_o = coeff * ((Sy * Sy + Sy2) + (Sz * Sz + Sz2))
    b_o = coeff * ((Sx * Sx + Sx2) + (Sz * Sz + Sz2))
    c_o = coeff * ((Sx * Sx + Sx2) + (Sy * Sy + Sy2))
    apr_o = coeff * (Syz + Sy * Sz)
    bpr_o = coeff * (Sxz + Sx * Sz)
    cpr_o = coeff * (Sxy + Sx * Sy)

    # Parallel-axis shift: I_origin = I_centroid + m * (||g||^2 I3 - g g^T)
    # so I_centroid = I_origin - m * (||g||^2 I3 - g g^T).
    g2 = com_x * com_x + com_y * com_y + com_z * com_z
    a_c = a_o - mass * (g2 - com_x * com_x)  # = a_o - m * (com_y^2 + com_z^2)
    b_c = b_o - mass * (g2 - com_y * com_y)
    c_c = c_o - mass * (g2 - com_z * com_z)
    apr_c = apr_o - mass * (com_y * com_z)
    bpr_c = bpr_o - mass * (com_x * com_z)
    cpr_c = cpr_o - mass * (com_x * com_y)

    inertia = wp.mat33(
        [
            [a_c, -cpr_c, -bpr_c],
            [-cpr_c, b_c, -apr_c],
            [-bpr_c, -apr_c, c_c],
        ]
    )
    return mass, com, inertia


def compute_inertia_box_from_mass(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    """Helper to compute 3x3 inertia matrix of a solid box with given mass and half-extents.

    Args:
        mass: The box mass [kg]
        hx: The box half-extent along the x-axis [m]
        hy: The box half-extent along the y-axis [m]
        hz: The box half-extent along the z-axis [m]

    Returns:

        A 3x3 inertia matrix with inertia specified around the center of mass
    """
    Ia = 1.0 / 3.0 * mass * (hy * hy + hz * hz)
    Ib = 1.0 / 3.0 * mass * (hx * hx + hz * hz)
    Ic = 1.0 / 3.0 * mass * (hx * hx + hy * hy)

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

    return I


def compute_inertia_box(density: float, hx: float, hy: float, hz: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid box

    Args:
        density: The box density [kg/m³]
        hx: The box half-extent along the x-axis [m]
        hy: The box half-extent along the y-axis [m]
        hz: The box half-extent along the z-axis [m]

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """

    v = 8.0 * hx * hy * hz
    m = density * v
    I = compute_inertia_box_from_mass(m, hx, hy, hz)

    return (m, wp.vec3(), I)


@wp.func
def triangle_inertia(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
):
    vol = wp.dot(v0, wp.cross(v1, v2)) / 6.0  # tetra volume (0,v0,v1,v2)
    first = vol * (v0 + v1 + v2) / 4.0  # first-order integral

    # second-order integral (symmetric)
    o00, o11, o22 = wp.outer(v0, v0), wp.outer(v1, v1), wp.outer(v2, v2)
    o01, o02, o12 = wp.outer(v0, v1), wp.outer(v0, v2), wp.outer(v1, v2)
    o01t, o02t, o12t = wp.transpose(o01), wp.transpose(o02), wp.transpose(o12)

    second = (vol / 10.0) * (o00 + o11 + o22)
    second += (vol / 20.0) * (o01 + o01t + o02 + o02t + o12 + o12t)

    return vol, first, second


#: Number of scalar components reduced per triangle: 1 (volume) + 3 (first
#: moment) + 9 (second moment, 3x3 row-major). Lowering the mesh-inertia
#: reduction into 13 scalar prefix-scans makes it bit-exact reproducible
#: across runs — ``wp.atomic_add`` does floating-point sums in
#: scheduling-dependent order, and float addition is not associative.
_MESH_INERTIA_NUM_SCALARS: int = 13


@wp.kernel
def compute_solid_mesh_inertia(
    indices: wp.array[int],
    vertices: wp.array[wp.vec3],
    # outputs: per-triangle scratch, shape (13, num_tris). Component layout:
    # [0]=volume, [1:4]=first, [4:13]=second (row-major 3x3).
    per_tri: wp.array2d[float],
):
    i = wp.tid()
    p = vertices[indices[i * 3 + 0]]
    q = vertices[indices[i * 3 + 1]]
    r = vertices[indices[i * 3 + 2]]

    v, f, s = triangle_inertia(p, q, r)
    per_tri[0, i] = v
    per_tri[1, i] = f[0]
    per_tri[2, i] = f[1]
    per_tri[3, i] = f[2]
    per_tri[4, i] = s[0, 0]
    per_tri[5, i] = s[0, 1]
    per_tri[6, i] = s[0, 2]
    per_tri[7, i] = s[1, 0]
    per_tri[8, i] = s[1, 1]
    per_tri[9, i] = s[1, 2]
    per_tri[10, i] = s[2, 0]
    per_tri[11, i] = s[2, 1]
    per_tri[12, i] = s[2, 2]


@wp.kernel
def compute_hollow_mesh_inertia(
    indices: wp.array[int],
    vertices: wp.array[wp.vec3],
    thickness: wp.array[float],
    # outputs: per-triangle scratch, shape (13, num_tris). Layout matches
    # :func:`compute_solid_mesh_inertia`.
    per_tri: wp.array2d[float],
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    normal = -wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    # wedge vertices
    vi0 = vi - ti
    vi1 = vi + ti
    vj0 = vj - tj
    vj1 = vj + tj
    vk0 = vk - tk
    vk1 = vk + tk

    v_total = float(0.0)
    f_total = wp.vec3(0.0)
    s_total = wp.mat33(0.0)

    v, f, s = triangle_inertia(vi0, vj0, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vk1, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vj1, vk1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vi1, vj1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vi0, vi1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj1, vi1, vk1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vi1, vi0, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vi1, vk0, vk1)
    v_total += v
    f_total += f
    s_total += s

    per_tri[0, tid] = v_total
    per_tri[1, tid] = f_total[0]
    per_tri[2, tid] = f_total[1]
    per_tri[3, tid] = f_total[2]
    per_tri[4, tid] = s_total[0, 0]
    per_tri[5, tid] = s_total[0, 1]
    per_tri[6, tid] = s_total[0, 2]
    per_tri[7, tid] = s_total[1, 0]
    per_tri[8, tid] = s_total[1, 1]
    per_tri[9, tid] = s_total[1, 2]
    per_tri[10, tid] = s_total[2, 0]
    per_tri[11, tid] = s_total[2, 1]
    per_tri[12, tid] = s_total[2, 2]


def _deterministic_sum_per_triangle(per_tri: wp.array) -> tuple[float, wp.vec3, wp.mat33]:
    """Reduce a ``(13, num_tris)`` per-triangle scratch buffer to scalar
    totals via per-component inclusive prefix scan.

    ``wp.utils.array_scan`` runs deterministically (single sequential
    summation tree) so the result is bit-exact reproducible across runs,
    in contrast to ``wp.atomic_add`` where ordering depends on GPU
    scheduling. The last entry of each scanned row holds the sum.
    """
    num_tris = per_tri.shape[1]
    scratch = wp.zeros(num_tris, dtype=float, device=per_tri.device)
    totals = np.empty(_MESH_INERTIA_NUM_SCALARS, dtype=np.float32)
    for c in range(_MESH_INERTIA_NUM_SCALARS):
        wp.utils.array_scan(per_tri[c], scratch, inclusive=True)
        totals[c] = float(scratch.numpy()[-1])
    V_tot = float(totals[0])
    F_tot = wp.vec3(float(totals[1]), float(totals[2]), float(totals[3]))
    S_tot = wp.mat33(
        float(totals[4]),
        float(totals[5]),
        float(totals[6]),
        float(totals[7]),
        float(totals[8]),
        float(totals[9]),
        float(totals[10]),
        float(totals[11]),
        float(totals[12]),
    )
    return V_tot, F_tot, S_tot


def compute_inertia_mesh(
    density: float,
    vertices: list[Vec3] | np.ndarray,
    indices: list[int] | np.ndarray,
    is_solid: bool = True,
    thickness: list[float] | float = 0.001,
) -> tuple[float, wp.vec3, wp.mat33, float]:
    """
    Compute the mass, center of mass, inertia, and volume of a triangular mesh.

    Args:
        density: The density of the mesh material.
        vertices: A list of vertex positions (3D coordinates).
        indices: A list of triangle indices (each triangle is defined by 3 vertex indices).
        is_solid: If True, compute inertia for a solid mesh; if False, for a hollow mesh using the given thickness.
        thickness: Thickness of the mesh if it is hollow. Can be a single value or a list of values for each vertex.

    Returns:
        A tuple containing:
            - mass: The mass of the mesh.
            - com: The center of mass (3D coordinates).
            - I: The inertia tensor (3x3 matrix).
            - volume: The signed volume of the mesh.
    """

    indices = np.array(indices).flatten()
    num_tris = len(indices) // 3

    wp_vertices = wp.array(vertices, dtype=wp.vec3)
    wp_indices = wp.array(indices, dtype=int)

    # Per-triangle scratch fed into a deterministic prefix-scan reduction
    # instead of ``atomic_add``. See :data:`_MESH_INERTIA_NUM_SCALARS`.
    per_tri = wp.zeros((_MESH_INERTIA_NUM_SCALARS, max(num_tris, 1)), dtype=float)

    if is_solid:
        wp.launch(
            kernel=compute_solid_mesh_inertia,
            dim=num_tris,
            inputs=[
                wp_indices,
                wp_vertices,
            ],
            outputs=[per_tri],
        )
    else:
        if isinstance(thickness, float):
            thickness = [thickness] * len(vertices)
        wp.launch(
            kernel=compute_hollow_mesh_inertia,
            dim=num_tris,
            inputs=[
                wp_indices,
                wp_vertices,
                wp.array(thickness, dtype=float),
            ],
            outputs=[per_tri],
        )

    V_tot, F_tot_v3, S_tot_m33 = _deterministic_sum_per_triangle(per_tri)
    F_tot = np.array([F_tot_v3[0], F_tot_v3[1], F_tot_v3[2]])
    S_tot = np.array(
        [
            [S_tot_m33[0, 0], S_tot_m33[0, 1], S_tot_m33[0, 2]],
            [S_tot_m33[1, 0], S_tot_m33[1, 1], S_tot_m33[1, 2]],
            [S_tot_m33[2, 0], S_tot_m33[2, 1], S_tot_m33[2, 2]],
        ]
    )

    # If the winding is inward, flip signs
    if V_tot < 0:
        V_tot = -V_tot
        F_tot = -F_tot
        S_tot = -S_tot

    mass = density * V_tot
    if V_tot > 0.0:
        com = F_tot / V_tot
    else:
        com = F_tot

    S_tot *= density  # include density
    I_origin = np.trace(S_tot) * np.eye(3) - S_tot  # inertia about origin
    r = com
    I_com = I_origin - mass * ((r @ r) * np.eye(3) - np.outer(r, r))

    return mass, wp.vec3(*com), wp.mat33(*I_com), V_tot


@wp.func
def transform_inertia(mass: float, inertia: wp.mat33, offset: wp.vec3, quat: wp.quat) -> wp.mat33:
    """
    Compute a rigid body's inertia tensor expressed in a new coordinate frame.

    The transformation applies (1) a rotation by quaternion ``quat`` and
    (2) a parallel-axis shift by vector ``offset`` (Steiner's theorem).
    Let ``R`` be the rotation matrix corresponding to ``quat``. The returned
    inertia tensor :math:`\\mathbf{I}'` is

    .. math::

        \\mathbf{I}' \\,=\\, \\mathbf{R}\\,\\mathbf{I}\\,\\mathbf{R}^\\top
        \\, + \\, m\\big(\\lVert\\mathbf{p}\\rVert^2\\,\\mathbf{I}_3
        \\, - \\, \\mathbf{p}\\,\\mathbf{p}^\\top\\big),

    where :math:`\\mathbf{I}_3` is the :math:`3\\times3` identity matrix.

    Args:
        mass: Mass of the rigid body.
        inertia: Inertia tensor expressed in the body's local frame, relative
            to its center of mass.
        offset: Position vector from the new frame's origin to the body's
            center of mass.
        quat: Orientation of the body relative to the new frame, expressed
            as a quaternion.

    Returns:
        wp.mat33: The transformed inertia tensor expressed in the new frame.
    """

    R = wp.quat_to_matrix(quat)

    # Steiner's theorem
    return R @ inertia @ wp.transpose(R) + mass * (
        wp.dot(offset, offset) * wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - wp.outer(offset, offset)
    )


def compute_inertia_shape(
    type: int,
    scale: Vec3,
    src: Mesh | Heightfield | None,
    density: float,
    is_solid: bool = True,
    thickness: list[float] | float = 0.001,
) -> tuple[float, wp.vec3, wp.mat33]:
    """Computes the mass, center of mass and 3x3 inertia tensor of a shape

    Args:
        type: The type of shape (GeoType.SPHERE, GeoType.BOX, etc.)
        scale: The scale of the shape
        src: The source shape (Mesh or Heightfield)
        density: The density of the shape
        is_solid: Whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)

    Returns:
        The mass, center of mass and 3x3 inertia tensor of the shape
    """
    if density == 0.0 or type == GeoType.PLANE:
        # Zero density means fixed; PLANE is a thin shell with zero volume
        # so it contributes no mass or rotational inertia.
        return 0.0, wp.vec3(), wp.mat33()

    if type == GeoType.TRIANGLE:
        # Treat the triangle as a thin prism extruded along its local +X
        # axis (the canonical face normal direction) by ±margin, so the
        # full prism thickness equals ``2 * margin``. The ``thickness``
        # argument is reused as that half-thickness -- the builder passes
        # ``cfg.margin`` here, which is also the per-shape collision
        # margin used for contact generation, so the inertia matches the
        # solid the contact surface implicitly defines.
        margin = float(thickness) if isinstance(thickness, (int, float)) else 0.0
        return compute_inertia_triangle_prism(
            density,
            edge_ab=scale[0],
            c_y=scale[1],
            c_z=scale[2],
            margin=margin,
        )

    if type == GeoType.TETRAHEDRON:
        # ``src`` carries the 4th vertex ``D = (d_x, d_y, d_z)`` (a
        # ``_TetrahedronVertexD`` from :mod:`newton._src.sim.builder`).
        # Duck-type so this module doesn't import from ``sim``.
        if src is None or not all(hasattr(src, attr) for attr in ("d_x", "d_y", "d_z")):
            raise ValueError(
                "compute_inertia_shape: GeoType.TETRAHEDRON requires "
                "``src`` to carry the 4th vertex (``d_x``, ``d_y``, "
                "``d_z`` attributes). The builder normally supplies a "
                "``_TetrahedronVertexD`` automatically via "
                "``add_shape_tetrahedron``."
            )
        return compute_inertia_tetrahedron(
            density,
            edge_ab=scale[0],
            c_y=scale[1],
            c_z=scale[2],
            d_x=float(src.d_x),
            d_y=float(src.d_y),
            d_z=float(src.d_z),
        )

    if type == GeoType.SPHERE:
        solid = compute_inertia_sphere(density, scale[0])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow sphere geom"
            hollow = compute_inertia_sphere(density, scale[0] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.BOX:
        # scale stores half-extents (hx, hy, hz)
        solid = compute_inertia_box(density, scale[0], scale[1], scale[2])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow box geom"
            hollow = compute_inertia_box(density, scale[0] - thickness, scale[1] - thickness, scale[2] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.CAPSULE:
        # scale[0] = radius, scale[1] = half_height
        solid = compute_inertia_capsule(density, scale[0], scale[1])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow capsule geom"
            hollow = compute_inertia_capsule(density, scale[0] - thickness, scale[1] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.CYLINDER:
        # scale[0] = radius, scale[1] = half_height
        solid = compute_inertia_cylinder(density, scale[0], scale[1])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow cylinder geom"
            hollow = compute_inertia_cylinder(density, scale[0] - thickness, scale[1] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.CONE:
        # scale[0] = radius, scale[1] = half_height
        solid = compute_inertia_cone(density, scale[0], scale[1])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow cone geom"
            hollow = compute_inertia_cone(density, scale[0] - thickness, scale[1] - thickness)
            m_shell = solid[0] - hollow[0]
            if m_shell <= 0.0:
                raise ValueError(
                    f"Hollow cone shell has non-positive mass ({m_shell:.6g}). "
                    f"The thickness ({thickness}) must be smaller than both the "
                    f"radius ({scale[0]}) and half_height ({scale[1]})."
                )
            # Cones have non-zero COM so outer and inner cones have different COMs;
            # compute the shell COM as the weighted difference, then shift both
            # inertia tensors to the shell COM before subtracting (parallel-axis theorem).
            com_s = np.array(solid[1])
            com_i = np.array(hollow[1])
            com_shell = (solid[0] * com_s - hollow[0] * com_i) / m_shell

            def _shift_inertia(mass, I_mat, com_from, com_to):
                d = com_to - np.array(com_from)
                return np.array(I_mat).reshape(3, 3) + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

            I_shell = _shift_inertia(solid[0], solid[2], com_s, com_shell) - _shift_inertia(
                hollow[0], hollow[2], com_i, com_shell
            )
            return m_shell, wp.vec3(*com_shell), wp.mat33(*I_shell.flatten())
    elif type == GeoType.ELLIPSOID:
        # scale stores semi-axes (rx, ry, rz)
        solid = compute_inertia_ellipsoid(density, scale[0], scale[1], scale[2])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow ellipsoid geom"
            hollow = compute_inertia_ellipsoid(
                density, scale[0] - thickness, scale[1] - thickness, scale[2] - thickness
            )
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.HFIELD or type == GeoType.GAUSSIAN:
        # Heightfields are always static terrain; Gaussians are render-only (zero mass, zero inertia)
        return 0.0, wp.vec3(), wp.mat33()
    elif type == GeoType.MESH or type == GeoType.CONVEX_MESH:
        assert src is not None, "src must be provided for mesh or convex hull shapes"
        if src.has_inertia and src.mass > 0.0 and src.is_solid == is_solid:
            m, c, I = src.mass, src.com, src.inertia
            scale = wp.vec3(scale)
            sx, sy, sz = scale

            # Mass scales with absolute volume — mirrored geometry (det(scale) < 0)
            # has the same volume as the original, so use |sx*sy*sz| here. The
            # signed scale is still applied below to mirror the COM and to flip
            # the appropriate inertia products. Without abs(), negative-scale
            # mesh/convex-hull shapes would receive negative mass, which
            # ``verify_and_correct_inertia`` then clamps to zero, making the
            # body effectively static.
            mass_ratio = abs(sx * sy * sz) * density
            m_new = m * mass_ratio

            c_new = wp.cw_mul(c, scale)

            Ixx = I[0, 0] * (sy**2 + sz**2) / 2 * mass_ratio
            Iyy = I[1, 1] * (sx**2 + sz**2) / 2 * mass_ratio
            Izz = I[2, 2] * (sx**2 + sy**2) / 2 * mass_ratio
            # Products of inertia pick up the sign of the corresponding scale
            # pair, which is the correct mirror behavior under a single-axis
            # sign flip.
            Ixy = I[0, 1] * sx * sy * mass_ratio
            Ixz = I[0, 2] * sx * sz * mass_ratio
            Iyz = I[1, 2] * sy * sz * mass_ratio

            I_new = wp.mat33([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

            return m_new, c_new, I_new
        elif type == GeoType.MESH or type == GeoType.CONVEX_MESH:
            assert isinstance(src, Mesh), "src must be a Mesh for mesh or convex hull shapes"
            # fall back to computing inertia from mesh geometry
            vertices = np.array(src.vertices) * np.array(scale)
            m, c, I, _vol = compute_inertia_mesh(density, vertices, src.indices, is_solid, thickness)
            return m, c, I
    raise ValueError(f"Unsupported shape type: {type}")


def verify_and_correct_inertia(
    mass: float,
    inertia: wp.mat33,
    balance_inertia: bool = True,
    bound_mass: float | None = None,
    bound_inertia: float | None = None,
    body_label: str | None = None,
) -> tuple[float, wp.mat33, bool]:
    """Verify and correct inertia values similar to MuJoCo's balanceinertia compiler setting.

    This function checks for invalid inertia values and corrects them if needed. It performs
    the following checks and corrections:
    1. Ensures mass is non-negative (and bounded if specified)
    2. Ensures inertia diagonal elements are non-negative (and bounded if specified)
    3. Ensures inertia matrix satisfies triangle inequality (principal moments satisfy Ixx + Iyy >= Izz etc.)
    4. Optionally balances inertia to satisfy the triangle inequality exactly

    Eigenvalue positivity is checked using a relative threshold
    (``_INERTIA_REL_TOL * max_eigenvalue``) so that lightweight components with
    small but physically valid inertia are not spuriously inflated.  When
    correction *is* needed, the adjustment uses a small absolute floor
    (``_INERTIA_ABS_ADJUSTMENT``) to keep the result well-conditioned.

    Args:
        mass: The mass of the body [kg].
        inertia: The 3x3 inertia tensor [kg*m^2].
        balance_inertia: If True, adjust inertia to exactly satisfy triangle inequality (like MuJoCo's balanceinertia).
        bound_mass: If specified, clamp mass to be at least this value [kg].
        bound_inertia: If specified, clamp inertia diagonal elements to be at least this value [kg*m^2].
        body_label: Optional label/name of the body for more informative warnings.

    Returns:
        A tuple of (corrected_mass, corrected_inertia, was_corrected) where was_corrected
        indicates if any corrections were made
    """
    was_corrected = False
    corrected_mass = mass
    inertia_array = np.array(inertia).reshape(3, 3)
    corrected_inertia = inertia_array.copy()

    # Format body identifier for warnings
    body_id = f" for body '{body_label}'" if body_label else ""

    # Check for NaN/Inf in mass or inertia
    if not np.isfinite(mass) or not np.all(np.isfinite(inertia_array)):
        warnings.warn(
            f"NaN/Inf detected in mass or inertia{body_id}, zeroing out mass and inertia",
            stacklevel=2,
        )
        return 0.0, wp.mat33(np.zeros((3, 3))), True

    # Check and correct mass
    if mass < 0:
        warnings.warn(f"Negative mass {mass} detected{body_id}, setting to 0", stacklevel=2)
        corrected_mass = 0.0
        was_corrected = True
    elif bound_mass is not None and mass < bound_mass and mass > 0:
        warnings.warn(f"Mass {mass} is below bound {bound_mass}{body_id}, clamping", stacklevel=2)
        corrected_mass = bound_mass
        was_corrected = True

    # For zero mass, inertia should also be zero
    if corrected_mass == 0.0:
        if np.any(inertia_array != 0):
            warnings.warn(f"Zero mass body{body_id} should have zero inertia, correcting", stacklevel=2)
            corrected_inertia = np.zeros((3, 3))
            was_corrected = True
        return corrected_mass, wp.mat33(corrected_inertia), was_corrected

    # Unconditionally symmetrize inertia matrix (idempotent for symmetric tensors)
    symmetrized = (inertia_array + inertia_array.T) / 2
    if not np.allclose(
        inertia_array,
        symmetrized,
        rtol=_INERTIA_SYMMETRY_RTOL,
        atol=_INERTIA_SYMMETRY_ATOL,
    ):
        warnings.warn(f"Inertia matrix{body_id} is not symmetric, making it symmetric", stacklevel=2)
        was_corrected = True
    corrected_inertia = symmetrized

    # Compute eigenvalues (principal moments) for validation
    try:
        eigenvalues = np.linalg.eigvals(corrected_inertia)

        # Check for negative or near-zero eigenvalues (ensure positive-definite).
        # The threshold is relative to the largest eigenvalue so that small but
        # physically valid inertia (lightweight components) is not inflated.
        max_eig = np.max(eigenvalues)
        eig_threshold = max(_INERTIA_REL_TOL * max_eig, _INERTIA_ABS_FLOOR)
        if np.any(eigenvalues < eig_threshold):
            warnings.warn(
                f"Eigenvalues below threshold detected{body_id}: {eigenvalues}, correcting inertia",
                stacklevel=2,
            )
            # Make positive definite by adjusting eigenvalues
            min_eig = np.min(eigenvalues)
            adjustment = eig_threshold - min_eig + _INERTIA_ABS_ADJUSTMENT
            corrected_inertia += np.eye(3) * adjustment
            eigenvalues += adjustment
            was_corrected = True

        # Apply inertia bounds to eigenvalues if specified
        if bound_inertia is not None:
            min_eig = np.min(eigenvalues)
            if min_eig < bound_inertia:
                warnings.warn(
                    f"Minimum eigenvalue {min_eig} is below bound {bound_inertia}{body_id}, adjusting", stacklevel=2
                )
                adjustment = bound_inertia - min_eig
                corrected_inertia += np.eye(3) * adjustment
                eigenvalues += adjustment
                was_corrected = True

        # Sort eigenvalues to get principal moments
        principal_moments = np.sort(eigenvalues)
        I1, I2, I3 = principal_moments

        # Check triangle inequality on principal moments
        # For a physically valid inertia tensor: I1 + I2 >= I3 (with tolerance)
        # Use float32 machine epsilon scaled by I3 as numerical noise floor.
        tri_tol = max(np.finfo(np.float32).eps * I3, _INERTIA_ABS_FLOOR)
        has_violations = I1 + I2 < I3 - tri_tol

    except np.linalg.LinAlgError:
        warnings.warn(f"Failed to compute eigenvalues for inertia tensor{body_id}, making it diagonal", stacklevel=2)
        was_corrected = True
        # Fallback: use diagonal elements
        trace = np.trace(corrected_inertia)
        if trace <= 0:
            trace = _INERTIA_ABS_ADJUSTMENT
        corrected_inertia = np.eye(3) * (trace / 3.0)
        has_violations = False
        principal_moments = [trace / 3.0, trace / 3.0, trace / 3.0]
        eigenvalues = np.array(principal_moments)

    if has_violations:
        warnings.warn(
            f"Inertia tensor{body_id} violates triangle inequality with principal moments ({I1:.6f}, {I2:.6f}, {I3:.6f})",
            stacklevel=2,
        )

        if balance_inertia:
            # For non-diagonal matrices, we need to adjust while preserving the rotation
            deficit = I3 - I1 - I2
            if deficit > 0:
                # Simple approach: add scalar to all eigenvalues to ensure validity
                # This preserves eigenvectors exactly
                # We need: (I1 + a) + (I2 + a) >= I3 + a
                # Which simplifies to: I1 + I2 + a >= I3
                # So: a >= I3 - I1 - I2 = deficit
                adjustment = deficit + _INERTIA_ABS_ADJUSTMENT

                # Add scalar*I to shift all eigenvalues equally
                corrected_inertia = corrected_inertia + np.eye(3) * adjustment
                was_corrected = True

                # Update principal moments
                new_I1 = I1 + adjustment
                new_I2 = I2 + adjustment
                new_I3 = I3 + adjustment

                warnings.warn(
                    f"Balanced principal moments{body_id} from ({I1:.6f}, {I2:.6f}, {I3:.6f}) to "
                    f"({new_I1:.6f}, {new_I2:.6f}, {new_I3:.6f})",
                    stacklevel=2,
                )

    # Final check: ensure the corrected inertia matrix is positive definite
    if has_violations and balance_inertia:
        # Need to recompute after balancing since we modified the matrix
        try:
            eigenvalues = np.linalg.eigvals(corrected_inertia)
        except np.linalg.LinAlgError:
            warnings.warn(f"Failed to compute eigenvalues of inertia matrix{body_id}", stacklevel=2)
            eigenvalues = np.array([0.0, 0.0, 0.0])

    # Check final eigenvalues
    if np.any(eigenvalues <= 0) or np.any(~np.isfinite(eigenvalues)):
        warnings.warn(
            f"Corrected inertia matrix{body_id} is not positive definite, this should not happen", stacklevel=2
        )
        # As a last resort, make it positive definite by adding a small value to diagonal.
        min_eigenvalue = (
            np.min(eigenvalues[np.isfinite(eigenvalues)])
            if np.any(np.isfinite(eigenvalues))
            else -_INERTIA_ABS_ADJUSTMENT
        )
        epsilon = abs(min_eigenvalue) + _INERTIA_ABS_ADJUSTMENT
        corrected_inertia[0, 0] += epsilon
        corrected_inertia[1, 1] += epsilon
        corrected_inertia[2, 2] += epsilon
        was_corrected = True

    return corrected_mass, wp.mat33(corrected_inertia), was_corrected


@wp.kernel(enable_backward=False, module="unique")
def validate_and_correct_inertia_kernel(
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    balance_inertia: wp.bool,
    bound_mass: wp.float32,
    bound_inertia: wp.float32,
    correction_count: wp.array[wp.int32],  # Output: atomic counter of corrected bodies
):
    """Warp kernel for parallel inertia validation and correction.

    This kernel performs basic validation and correction but doesn't support:
    - Warning messages (handled by caller)
    - Complex iterative balancing (falls back to simple correction)
    """
    tid = wp.tid()

    mass = body_mass[tid]
    inertia = body_inertia[tid]
    original_inertia = inertia
    was_corrected = False

    # Detect NaN/Inf in mass or any inertia coefficient and zero out
    if (
        not wp.isfinite(mass)
        or not wp.isfinite(inertia[0, 0])
        or not wp.isfinite(inertia[0, 1])
        or not wp.isfinite(inertia[0, 2])
        or not wp.isfinite(inertia[1, 0])
        or not wp.isfinite(inertia[1, 1])
        or not wp.isfinite(inertia[1, 2])
        or not wp.isfinite(inertia[2, 0])
        or not wp.isfinite(inertia[2, 1])
        or not wp.isfinite(inertia[2, 2])
    ):
        mass = 0.0
        inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        was_corrected = True

    # Check for negative mass
    if mass < 0.0:
        mass = 0.0
        was_corrected = True

    # Apply mass bound (only to positive mass; zero mass means static/fixed body)
    if bound_mass > 0.0 and mass < bound_mass and mass > 0.0:
        mass = bound_mass
        was_corrected = True

    # For zero mass, inertia should be zero
    if mass == 0.0:
        was_corrected = was_corrected or (wp.ddot(inertia, inertia) > 0.0)
        inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        # Symmetrize inertia matrix: (I + I^T) / 2
        sym01 = (inertia[0, 1] + inertia[1, 0]) * 0.5
        sym02 = (inertia[0, 2] + inertia[2, 0]) * 0.5
        sym12 = (inertia[1, 2] + inertia[2, 1]) * 0.5
        sym = wp.mat33(
            inertia[0, 0],
            sym01,
            sym02,
            sym01,
            inertia[1, 1],
            sym12,
            sym02,
            sym12,
            inertia[2, 2],
        )

        tol01 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym01)
        tol02 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym02)
        tol12 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym12)
        if (
            wp.abs(inertia[0, 1] - sym01) > tol01
            or wp.abs(inertia[1, 0] - sym01) > tol01
            or wp.abs(inertia[0, 2] - sym02) > tol02
            or wp.abs(inertia[2, 0] - sym02) > tol02
            or wp.abs(inertia[1, 2] - sym12) > tol12
            or wp.abs(inertia[2, 1] - sym12) > tol12
        ):
            was_corrected = True
        inertia = sym

        # Use eigendecomposition for proper validation
        _eigvecs, eigvals = wp.eig3(inertia)

        # Sort eigenvalues to get principal moments (I1 <= I2 <= I3)
        I1, I2, I3 = eigvals[0], eigvals[1], eigvals[2]
        if I1 > I2:
            I1, I2 = I2, I1
        if I2 > I3:
            I2, I3 = I3, I2
            if I1 > I2:
                I1, I2 = I2, I1

        # Check for negative or near-zero eigenvalues (ensure positive-definite).
        # Use a relative threshold so lightweight components are not inflated.
        eig_threshold = wp.max(1.0e-6 * I3, 1.0e-10)
        if I1 < eig_threshold:
            adjustment = eig_threshold - I1 + 1.0e-6
            # Add scalar to all eigenvalues
            I1 += adjustment
            I2 += adjustment
            I3 += adjustment
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

        # Apply eigenvalue bounds
        if bound_inertia > 0.0 and I1 < bound_inertia:
            adjustment = bound_inertia - I1
            I1 += adjustment
            I2 += adjustment
            I3 += adjustment
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

        # Check triangle inequality: I1 + I2 >= I3 (with tolerance)
        tri_tol = wp.max(1.1920929e-7 * I3, 1.0e-10)  # float32 eps * I3
        if balance_inertia and (I1 + I2 < I3 - tri_tol):
            deficit = I3 - I1 - I2
            adjustment = deficit + 1.0e-6
            # Add scalar*I to fix triangle inequality
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

    output_inertia = inertia if was_corrected else original_inertia

    # Write back corrected values
    body_mass[tid] = mass
    body_inertia[tid] = output_inertia

    # Update inverse mass
    if mass > 0.0:
        body_inv_mass[tid] = 1.0 / mass
    else:
        body_inv_mass[tid] = 0.0

    # Update inverse inertia
    if mass > 0.0:
        body_inv_inertia[tid] = wp.inverse(output_inertia)
    else:
        body_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if was_corrected:
        wp.atomic_add(correction_count, 0, 1)
