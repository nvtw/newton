# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the unified rigid / triangle contact iterate.

The contact iterate / prepare reads each endpoint via
:func:`endpoint_load` and writes via :func:`endpoint_apply_impulse`,
so the same kernel handles RR, RT, and TT pairs. These tests exercise
each path through a direct kernel launch (no full solver / collision
pipeline) to keep them fast and focused.

CUDA + graph-capture-only per repo policy.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    BodyContainer,
    body_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_iterate_at,
    contact_set_contact_count,
    contact_set_contact_first,
    contact_set_endpoint_idx1,
    contact_set_endpoint_idx2,
    contact_set_endpoint_kind1,
    contact_set_endpoint_kind2,
    contact_set_friction,
    contact_set_friction_dynamic,
    contact_views_make,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    constraint_bodies_make,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_local_p0,
    cc_set_local_p1,
    cc_set_normal,
    cc_set_tangent1,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    ENDPOINT_KIND_RIGID,
    ENDPOINT_KIND_TRIANGLE,
)
from newton._src.solvers.phoenx.constraints.contact_ingest import (
    _contact_warmstart_gather_kernel,
)
from newton._src.solvers.phoenx.helpers.data_packing import (
    reinterpret_int_as_float,
)
from newton._src.solvers.phoenx.particle import (
    ParticleContainer,
    particle_container_zeros,
)


@wp.kernel(enable_backward=False)
def _seed_column_kernel(
    cols: ContactColumnContainer,
    cid: wp.int32,
    kind1: wp.int32,
    idx1: wp.int32,
    kind2: wp.int32,
    idx2: wp.int32,
    contact_first: wp.int32,
    contact_count: wp.int32,
    mu_s: wp.float32,
    mu_k: wp.float32,
):
    """Stamp endpoint kind/idx + range + friction onto one column."""
    if wp.tid() != 0:
        return
    contact_set_endpoint_kind1(cols, cid, kind1)
    contact_set_endpoint_idx1(cols, cid, idx1)
    contact_set_endpoint_kind2(cols, cid, kind2)
    contact_set_endpoint_idx2(cols, cid, idx2)
    contact_set_contact_first(cols, cid, contact_first)
    contact_set_contact_count(cols, cid, contact_count)
    contact_set_friction(cols, cid, mu_s)
    contact_set_friction_dynamic(cols, cid, mu_k)


@wp.kernel(enable_backward=False)
def _seed_contact_kernel(
    cc: ContactContainer,
    k: wp.int32,
    n: wp.vec3f,
    t1: wp.vec3f,
    local_p0: wp.vec3f,
    local_p1: wp.vec3f,
):
    """Stamp normal / tangent / local_p anchors on one contact slot."""
    if wp.tid() != 0:
        return
    cc_set_normal(cc, k, n)
    cc_set_tangent1(cc, k, t1)
    cc_set_local_p0(cc, k, local_p0)
    cc_set_local_p1(cc, k, local_p1)


@wp.kernel(enable_backward=False)
def _seed_eff_kernel(cc: ContactContainer, k: wp.int32, eff: wp.float32):
    """Stamp effective masses (n + 2 tangents) onto one contact slot."""
    if wp.tid() != 0:
        return
    cc_set_eff_n(cc, k, eff)
    cc_set_eff_t1(cc, k, eff)
    cc_set_eff_t2(cc, k, eff)


@wp.kernel(enable_backward=False)
def _run_iterate_kernel(
    cols: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    """Single-thread driver for one contact column iterate sweep."""
    if wp.tid() != 0:
        return
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at(cols, cid, wp.int32(0), bodies, body_pair, idt, cc, contacts, use_bias, particles, tri_indices)


def _make_contact_views_from_arrays(
    point0: wp.array,
    point1: wp.array,
    normal: wp.array,
    margin0: wp.array,
    margin1: wp.array,
    shape0: wp.array,
    shape1: wp.array,
    rigid_contact_count: wp.array,
    shape_body: wp.array,
    device: wp.DeviceLike,
) -> ContactViews:
    """Stitch a :class:`ContactViews` from raw arrays.

    Soft-contact / match / sort-key arrays use length-0 sentinels;
    the iterate path under test only reads ``point0/1``, ``normal``,
    ``margin0/1``, and the shape arrays (via the prepare; iterate
    itself only reads ``margin0/1``)."""
    return contact_views_make(
        rigid_contact_count=rigid_contact_count,
        rigid_contact_point0=point0,
        rigid_contact_point1=point1,
        rigid_contact_normal=normal,
        rigid_contact_shape0=shape0,
        rigid_contact_shape1=shape1,
        rigid_contact_match_index=wp.zeros(1, dtype=wp.int32, device=device),
        rigid_contact_margin0=margin0,
        rigid_contact_margin1=margin1,
        shape_body=shape_body,
        rigid_contact_stiffness=wp.zeros(0, dtype=wp.float32, device=device),
        rigid_contact_damping=wp.zeros(0, dtype=wp.float32, device=device),
        rigid_contact_friction=wp.zeros(0, dtype=wp.float32, device=device),
    )


@unittest.skipUnless(wp.is_cuda_available(), "Unified contact tests require CUDA")
class TestUnifiedContactRT(unittest.TestCase):
    """Rigid-vs-triangle contact: a falling rigid sphere strikes a
    pinned cloth triangle. After one iterate sweep, the sphere's
    downward velocity must reverse (or zero out) and the triangle
    vertices must receive an upward impulse."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_rigid_endpoint_velocity_flips(self) -> None:
        device = self.device
        with wp.ScopedDevice(device):
            with wp.ScopedCapture(force_module_load=False) as capture:
                # Two-body store (slot 0 = static anchor, slot 1 =
                # dynamic falling sphere). Triangle endpoint sits in
                # particle land.
                bodies = body_container_zeros(2, device=device)
                bodies.position.assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
                bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32))
                bodies.velocity.assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32))
                bodies.angular_velocity.assign(np.zeros((2, 3), dtype=np.float32))
                bodies.body_com.assign(np.zeros((2, 3), dtype=np.float32))
                bodies.inverse_mass.assign(np.array([0.0, 1.0], dtype=np.float32))
                bodies.inverse_inertia_world.assign(
                    np.array(
                        [
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        ],
                        dtype=np.float32,
                    )
                )
                motion = np.array([MOTION_STATIC, MOTION_DYNAMIC], dtype=np.int32)
                bodies.motion_type.assign(motion)

                # Three-particle triangle (vertices A, B, C). All
                # pinned (inverse_mass = 0) so the rigid body owns
                # the whole effective mass.
                particles = particle_container_zeros(3, device=device)
                tri_pos = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                )
                particles.position.assign(tri_pos)
                particles.velocity.assign(np.zeros((3, 3), dtype=np.float32))
                particles.inverse_mass.assign(np.zeros(3, dtype=np.float32))

                tri_indices = wp.array([wp.vec4i(0, 1, 2, -1)], dtype=wp.vec4i, device=device)

                # One contact at the triangle centroid, normal +z,
                # rigid sphere endpoint above. ``local_p0`` for the
                # triangle is the barycentric weight (1/3, 1/3, 1/3);
                # ``local_p1`` for the rigid endpoint is the body-
                # local anchor (sphere centre, here the body origin).
                rigid_contact_max = 1
                point0 = wp.array([wp.vec3f(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)], dtype=wp.vec3f, device=device)
                point1 = wp.array([wp.vec3f(0.0, 0.0, 0.0)], dtype=wp.vec3f, device=device)
                normal = wp.array([wp.vec3f(0.0, 0.0, 1.0)], dtype=wp.vec3f, device=device)
                margin0 = wp.zeros(1, dtype=wp.float32, device=device)
                margin1 = wp.zeros(1, dtype=wp.float32, device=device)
                shape0 = wp.zeros(1, dtype=wp.int32, device=device)
                shape1 = wp.ones(1, dtype=wp.int32, device=device)
                rigid_contact_count = wp.array([1], dtype=wp.int32, device=device)
                shape_body = wp.array([-1, 1], dtype=wp.int32, device=device)

                cc = contact_container_zeros(rigid_contact_max, device=device)
                cols = contact_column_container_zeros(1, device=device)

                # Endpoint 1 = TRIANGLE (idx 0); endpoint 2 = RIGID
                # (body 1). Friction zero so we isolate the normal row.
                wp.launch(
                    _seed_column_kernel,
                    dim=1,
                    inputs=[
                        cols,
                        wp.int32(0),
                        ENDPOINT_KIND_TRIANGLE,
                        wp.int32(0),
                        ENDPOINT_KIND_RIGID,
                        wp.int32(1),
                        wp.int32(0),
                        wp.int32(1),
                        wp.float32(0.0),
                        wp.float32(0.0),
                    ],
                    device=device,
                )
                wp.launch(
                    _seed_contact_kernel,
                    dim=1,
                    inputs=[
                        cc,
                        wp.int32(0),
                        wp.vec3f(0.0, 0.0, 1.0),
                        wp.vec3f(1.0, 0.0, 0.0),
                        wp.vec3f(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                        wp.vec3f(0.0, 0.0, 0.0),
                    ],
                    device=device,
                )

                contacts = _make_contact_views_from_arrays(
                    point0,
                    point1,
                    normal,
                    margin0,
                    margin1,
                    shape0,
                    shape1,
                    rigid_contact_count,
                    shape_body,
                    device,
                )

                # Stamp eff_n / eff_t1 / eff_t2 directly so the test
                # doesn't depend on the prepare's ingest plumbing.
                # eff_n=1 means the rigid sphere (inv_mass=1, point on
                # its own COM with r=0 lever arm) sees a unit normal
                # row -- the triangle endpoint contributes zero (all
                # vertices pinned), so combined eff_n collapses to the
                # rigid term.
                wp.launch(
                    _seed_eff_kernel,
                    dim=1,
                    inputs=[cc, wp.int32(0), wp.float32(1.0)],
                    device=device,
                )

                wp.launch(
                    _run_iterate_kernel,
                    dim=1,
                    inputs=[
                        cols,
                        wp.int32(0),
                        bodies,
                        wp.int32(0),
                        wp.int32(1),
                        wp.float32(60.0),
                        cc,
                        contacts,
                        False,  # use_bias=False -> rigid PGS, zero bias
                        particles,
                        tri_indices,
                    ],
                    device=device,
                )
            graph = capture.graph
            wp.capture_launch(graph)
            wp.synchronize_device(device)

            # Sphere's downward velocity (-1) should be cancelled or
            # reversed by the normal row. With eff_n=1 and zero bias,
            # ``d_lam_n = -eff_n * jv_n = +1`` (positive impulse along
            # +n on endpoint 2), so v_z goes from -1 to 0.
            v_after = bodies.velocity.numpy()
            np.testing.assert_allclose(
                v_after[1, 2],
                0.0,
                atol=1e-5,
                err_msg=f"sphere v_z should be cancelled by normal impulse, got {v_after[1]}",
            )
            # Triangle vertices are pinned (inv_mass=0) so velocities
            # stay zero (impulse is dropped on inert vertices).
            np.testing.assert_allclose(
                particles.velocity.numpy(),
                0.0,
                atol=1e-7,
                err_msg="pinned triangle vertices should stay still",
            )


@unittest.skipUnless(wp.is_cuda_available(), "Unified contact tests require CUDA")
class TestUnifiedContactTT(unittest.TestCase):
    """Triangle-vs-triangle smoke: two free triangles in head-on
    contact. After one iterate sweep, equal-and-opposite barycentric
    impulses must land on the two triangles' vertices and momentum
    must be conserved (sum of vertex velocity * mass)."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_momentum_conserved(self) -> None:
        device = self.device
        with wp.ScopedDevice(device):
            with wp.ScopedCapture(force_module_load=False) as capture:
                # Single static body slot (unused; both endpoints are
                # triangles).
                bodies = body_container_zeros(1, device=device)
                bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
                bodies.motion_type.assign(np.array([MOTION_STATIC], dtype=np.int32))

                # Two triangles, six particles total. Triangle 0 =
                # (0,1,2) moving in +z; Triangle 1 = (3,4,5) moving in
                # -z. All inverse masses = 1.
                particles = particle_container_zeros(6, device=device)
                pos = np.array(
                    [
                        # Triangle 0 (below z=0)
                        [0.0, 0.0, -0.1],
                        [1.0, 0.0, -0.1],
                        [0.0, 1.0, -0.1],
                        # Triangle 1 (above z=0)
                        [0.0, 0.0, 0.1],
                        [1.0, 0.0, 0.1],
                        [0.0, 1.0, 0.1],
                    ],
                    dtype=np.float32,
                )
                particles.position.assign(pos)
                vel = np.array(
                    [
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ],
                    dtype=np.float32,
                )
                particles.velocity.assign(vel)
                particles.inverse_mass.assign(np.ones(6, dtype=np.float32))

                tri_indices = wp.array(
                    [wp.vec4i(0, 1, 2, -1), wp.vec4i(3, 4, 5, -1)],
                    dtype=wp.vec4i,
                    device=device,
                )

                rigid_contact_max = 1
                bary = wp.vec3f(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
                point0 = wp.array([bary], dtype=wp.vec3f, device=device)
                point1 = wp.array([bary], dtype=wp.vec3f, device=device)
                normal = wp.array([wp.vec3f(0.0, 0.0, 1.0)], dtype=wp.vec3f, device=device)
                margin0 = wp.zeros(1, dtype=wp.float32, device=device)
                margin1 = wp.zeros(1, dtype=wp.float32, device=device)
                shape0 = wp.zeros(1, dtype=wp.int32, device=device)
                shape1 = wp.ones(1, dtype=wp.int32, device=device)
                rigid_contact_count = wp.array([1], dtype=wp.int32, device=device)
                shape_body = wp.array([-1, -1], dtype=wp.int32, device=device)

                cc = contact_container_zeros(rigid_contact_max, device=device)
                cols = contact_column_container_zeros(1, device=device)

                wp.launch(
                    _seed_column_kernel,
                    dim=1,
                    inputs=[
                        cols,
                        wp.int32(0),
                        ENDPOINT_KIND_TRIANGLE,
                        wp.int32(0),
                        ENDPOINT_KIND_TRIANGLE,
                        wp.int32(1),
                        wp.int32(0),
                        wp.int32(1),
                        wp.float32(0.0),
                        wp.float32(0.0),
                    ],
                    device=device,
                )
                wp.launch(
                    _seed_contact_kernel,
                    dim=1,
                    inputs=[
                        cc,
                        wp.int32(0),
                        wp.vec3f(0.0, 0.0, 1.0),
                        wp.vec3f(1.0, 0.0, 0.0),
                        bary,
                        bary,
                    ],
                    device=device,
                )

                # ``inv_m_eff = sum(w_i^2 * inv_m_i) = 3 * (1/9) * 1 =
                # 1/3`` per triangle endpoint, so combined eff_n =
                # 1 / (1/3 + 1/3) = 1.5.
                wp.launch(
                    _seed_eff_kernel,
                    dim=1,
                    inputs=[cc, wp.int32(0), wp.float32(1.5)],
                    device=device,
                )

                contacts = _make_contact_views_from_arrays(
                    point0,
                    point1,
                    normal,
                    margin0,
                    margin1,
                    shape0,
                    shape1,
                    rigid_contact_count,
                    shape_body,
                    device,
                )
                wp.launch(
                    _run_iterate_kernel,
                    dim=1,
                    inputs=[
                        cols,
                        wp.int32(0),
                        bodies,
                        wp.int32(0),
                        wp.int32(0),
                        wp.float32(60.0),
                        cc,
                        contacts,
                        False,
                        particles,
                        tri_indices,
                    ],
                    device=device,
                )
            graph = capture.graph
            wp.capture_launch(graph)
            wp.synchronize_device(device)

            v_after = particles.velocity.numpy()
            # Total z-momentum = sum(v_i * m_i) where m_i = 1.
            # Initially: 3 * 1 + 3 * (-1) = 0. Should stay 0.
            total_pz = float(np.sum(v_after[:, 2]))
            self.assertAlmostEqual(
                total_pz,
                0.0,
                places=4,
                msg=f"TT contact violated z-momentum conservation: {total_pz}",
            )
            # Triangle 0 (was moving +z) must have decelerated;
            # triangle 1 (was moving -z) must have decelerated.
            v0_avg = float(np.mean(v_after[:3, 2]))
            v1_avg = float(np.mean(v_after[3:, 2]))
            self.assertLess(v0_avg, 1.0, "triangle 0 didn't decelerate")
            self.assertGreater(v1_avg, -1.0, "triangle 1 didn't decelerate")


@wp.kernel(enable_backward=False)
def _seed_prev_lambdas_kernel(
    prev_buf: wp.array2d[wp.float32],
    k: wp.int32,
    lam_n: wp.float32,
    lam_t1: wp.float32,
    lam_t2: wp.float32,
    n: wp.vec3f,
    t1: wp.vec3f,
    local_p0: wp.vec3f,
    local_p1: wp.vec3f,
    kind1: wp.int32,
    idx1: wp.int32,
    kind2: wp.int32,
    idx2: wp.int32,
):
    """Stamp a complete persistent record directly into ``prev_lambdas``.

    Bypasses the public ``cc_set_*`` accessors (which target
    ``cc.lambdas``) and writes the prev buffer directly. Layout
    mirrors :class:`ContactContainer` -- see ``_CC_OFF_*`` constants
    in :mod:`contact_container`. The four endpoint tags are bit-cast
    via :func:`reinterpret_int_as_float` like the production
    accessors.
    """
    if wp.tid() != 0:
        return
    prev_buf[0, k] = lam_n
    prev_buf[1, k] = lam_t1
    prev_buf[2, k] = lam_t2
    prev_buf[3, k] = n[0]
    prev_buf[4, k] = n[1]
    prev_buf[5, k] = n[2]
    prev_buf[6, k] = t1[0]
    prev_buf[7, k] = t1[1]
    prev_buf[8, k] = t1[2]
    prev_buf[9, k] = local_p0[0]
    prev_buf[10, k] = local_p0[1]
    prev_buf[11, k] = local_p0[2]
    prev_buf[12, k] = local_p1[0]
    prev_buf[13, k] = local_p1[1]
    prev_buf[14, k] = local_p1[2]
    prev_buf[15, k] = reinterpret_int_as_float(kind1)
    prev_buf[16, k] = reinterpret_int_as_float(idx1)
    prev_buf[17, k] = reinterpret_int_as_float(kind2)
    prev_buf[18, k] = reinterpret_int_as_float(idx2)


@wp.kernel(enable_backward=False)
def _read_cc_lambdas_kernel(
    cc: ContactContainer,
    k: wp.int32,
    out_lam_n: wp.array[wp.float32],
    out_lam_t1: wp.array[wp.float32],
    out_lam_t2: wp.array[wp.float32],
    out_local_p0: wp.array[wp.vec3f],
    out_local_p1: wp.array[wp.vec3f],
    out_normal: wp.array[wp.vec3f],
    out_tangent1: wp.array[wp.vec3f],
):
    """Mirror ``cc.lambdas[:, k]`` into 1-element output arrays."""
    if wp.tid() != 0:
        return
    out_lam_n[0] = cc_get_normal_lambda(cc, k)
    out_lam_t1[0] = cc_get_tangent1_lambda(cc, k)
    out_lam_t2[0] = cc_get_tangent2_lambda(cc, k)
    out_local_p0[0] = cc_get_local_p0(cc, k)
    out_local_p1[0] = cc_get_local_p1(cc, k)
    out_normal[0] = cc_get_normal(cc, k)
    out_tangent1[0] = cc_get_tangent1(cc, k)


@unittest.skipUnless(wp.is_cuda_available(), "Unified contact tests require CUDA")
class TestUnifiedContactWarmStart(unittest.TestCase):
    """Warm-start carry-forward across frames for RT and TT pairs.

    The gather kernel is invoked directly with a hand-seeded
    ``prev_lambdas`` slot; the assertion is that on a "matched" prev
    contact (``rigid_contact_match_index[k] = 0``,
    ``prev_cid_of_contact[0] >= 0``) the persistent impulses
    (``lam_n``, ``lam_t1``, ``lam_t2``) and the prev anchors carry
    forward into ``cc.lambdas`` -- i.e. the slot is **not**
    re-initialised to zero for triangle-touching pairs.
    """

    def setUp(self) -> None:
        self.device = wp.get_device()

    def _build_minimal_rt_scene(self) -> dict:
        """One RT pair: rigid body 1 vs triangle 0. ``num_rigid_shapes
        = 1`` so shape 0 is rigid and shape 1 is the triangle."""
        device = self.device
        bodies = body_container_zeros(2, device=device)
        bodies.position.assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32))
        bodies.velocity.assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32))
        bodies.body_com.assign(np.zeros((2, 3), dtype=np.float32))
        bodies.inverse_mass.assign(np.array([0.0, 1.0], dtype=np.float32))
        bodies.motion_type.assign(np.array([MOTION_STATIC, MOTION_DYNAMIC], dtype=np.int32))

        particles = particle_container_zeros(3, device=device)
        particles.position.assign(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
        particles.velocity.assign(np.zeros((3, 3), dtype=np.float32))
        particles.inverse_mass.assign(np.zeros(3, dtype=np.float32))

        tri_indices = wp.array([wp.vec4i(0, 1, 2, -1)], dtype=wp.vec4i, device=device)

        # Pair (shape 0 = rigid body 1, shape 1 = triangle 0).
        # ``num_rigid_shapes = 1`` => ``sa = 0`` is rigid, ``sb = 1``
        # is triangle (idx = 0).
        rigid_contact_max = 1
        bary = wp.vec3f(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        # Rigid endpoint: body-local origin-frame anchor (sphere
        # bottom-pole, body 1's local frame).
        rigid_local_anchor = wp.vec3f(0.0, 0.0, 0.0)
        point0 = wp.array([rigid_local_anchor], dtype=wp.vec3f, device=device)
        # Triangle endpoint: world-space contact point (the gather
        # converts this to bary against the current particle pos).
        point1 = wp.array([wp.vec3f(1.0 / 3.0, 1.0 / 3.0, 0.0)], dtype=wp.vec3f, device=device)
        normal = wp.array([wp.vec3f(0.0, 0.0, 1.0)], dtype=wp.vec3f, device=device)
        margin0 = wp.zeros(1, dtype=wp.float32, device=device)
        margin1 = wp.zeros(1, dtype=wp.float32, device=device)
        shape0 = wp.zeros(1, dtype=wp.int32, device=device)  # rigid shape 0
        shape1 = wp.ones(1, dtype=wp.int32, device=device)  # triangle shape 1
        rigid_contact_count = wp.array([1], dtype=wp.int32, device=device)
        # ``shape_body[0] = 1`` -> rigid shape 0 maps to body 1.
        shape_body = wp.array([1], dtype=wp.int32, device=device)

        cc = contact_container_zeros(rigid_contact_max, device=device)
        match_index = wp.zeros(1, dtype=wp.int32, device=device)
        empty_f = wp.zeros(0, dtype=wp.float32, device=device)
        contacts = contact_views_make(
            rigid_contact_count=rigid_contact_count,
            rigid_contact_point0=point0,
            rigid_contact_point1=point1,
            rigid_contact_normal=normal,
            rigid_contact_shape0=shape0,
            rigid_contact_shape1=shape1,
            rigid_contact_match_index=match_index,
            rigid_contact_margin0=margin0,
            rigid_contact_margin1=margin1,
            shape_body=shape_body,
            rigid_contact_stiffness=empty_f,
            rigid_contact_damping=empty_f,
            rigid_contact_friction=empty_f,
        )
        # Hold all backing arrays in the dict so they stay alive for
        # the duration of the test (Warp structs don't keep Python
        # refs to their array fields).
        return {
            "device": device,
            "bodies": bodies,
            "particles": particles,
            "tri_indices": tri_indices,
            "cc": cc,
            "contacts": contacts,
            "bary": bary,
            "rigid_local_anchor": rigid_local_anchor,
            "_keepalive": [
                point0,
                point1,
                normal,
                margin0,
                margin1,
                shape0,
                shape1,
                rigid_contact_count,
                shape_body,
                match_index,
                empty_f,
            ],
        }

    def _build_gather_scratch(self, ctx: dict) -> dict:
        """Allocate gather-kernel scratch arrays. Returned dict must
        be kept alive across the captured launch so the underlying
        device memory survives until ``capture_launch`` runs."""
        device = ctx["device"]
        return {
            "pair_source_idx": wp.array([0], dtype=wp.int32, device=device),
            "pair_first": wp.array([0], dtype=wp.int32, device=device),
            "pair_count": wp.array([1], dtype=wp.int32, device=device),
            "pair_shape_a": wp.array([0], dtype=wp.int32, device=device),
            "pair_shape_b": wp.array([1], dtype=wp.int32, device=device),
            "rigid_contact_match_index": wp.array([0], dtype=wp.int32, device=device),
            "prev_cid_of_contact": wp.array([0], dtype=wp.int32, device=device),
            "num_contact_columns": wp.array([1], dtype=wp.int32, device=device),
        }

    def _launch_gather(self, ctx: dict, scratch: dict) -> None:
        """Launch the gather kernel for one RT pair."""
        wp.launch(
            _contact_warmstart_gather_kernel,
            dim=1,
            inputs=[
                scratch["pair_source_idx"],
                scratch["pair_first"],
                scratch["pair_count"],
                scratch["pair_shape_a"],
                scratch["pair_shape_b"],
                scratch["rigid_contact_match_index"],
                scratch["prev_cid_of_contact"],
                scratch["num_contact_columns"],
                ctx["bodies"],
                ctx["contacts"],
                ctx["cc"],
                ctx["particles"],
                ctx["tri_indices"],
                wp.int32(1),  # num_rigid_shapes
            ],
            device=ctx["device"],
        )

    def test_rt_carry_forward_lambdas_and_anchors(self) -> None:
        """RT pair with a prev-frame match must carry impulses + prev
        anchors forward (the prev triangle bary anchor follows the
        triangle, the rigid anchor stays body-local)."""
        device = self.device
        with wp.ScopedDevice(device):
            ctx = self._build_minimal_rt_scene()
            cc = ctx["cc"]
            bary = ctx["bary"]
            rigid_local_anchor = ctx["rigid_local_anchor"]

            out_lam_n = wp.zeros(1, dtype=wp.float32, device=device)
            out_lam_t1 = wp.zeros(1, dtype=wp.float32, device=device)
            out_lam_t2 = wp.zeros(1, dtype=wp.float32, device=device)
            out_local_p0 = wp.zeros(1, dtype=wp.vec3f, device=device)
            out_local_p1 = wp.zeros(1, dtype=wp.vec3f, device=device)
            out_normal = wp.zeros(1, dtype=wp.vec3f, device=device)
            out_tangent1 = wp.zeros(1, dtype=wp.vec3f, device=device)

            # Seed prev_lambdas eagerly (outside capture) so the
            # gather captured graph reads stable initial state. The
            # gather + read remain inside the captured graph -- this
            # is the production launch pattern (the seed is the
            # equivalent of "last frame's iterate output", which is
            # already settled before this frame's capture).
            wp.launch(
                _seed_prev_lambdas_kernel,
                dim=1,
                inputs=[
                    cc.prev_lambdas,
                    wp.int32(0),
                    wp.float32(2.5),  # lam_n
                    wp.float32(0.7),  # lam_t1
                    wp.float32(-0.3),  # lam_t2
                    wp.vec3f(0.0, 0.0, 1.0),
                    wp.vec3f(1.0, 0.0, 0.0),
                    rigid_local_anchor,
                    bary,
                    ENDPOINT_KIND_RIGID,
                    wp.int32(1),
                    ENDPOINT_KIND_TRIANGLE,
                    wp.int32(0),
                ],
                device=device,
            )
            # Allocate gather scratch outside capture; keep refs alive
            # for the duration of the captured launch.
            gather_scratch = self._build_gather_scratch(ctx)

            with wp.ScopedCapture() as capture:
                self._launch_gather(ctx, gather_scratch)
                wp.launch(
                    _read_cc_lambdas_kernel,
                    dim=1,
                    inputs=[
                        cc,
                        wp.int32(0),
                        out_lam_n,
                        out_lam_t1,
                        out_lam_t2,
                        out_local_p0,
                        out_local_p1,
                        out_normal,
                        out_tangent1,
                    ],
                    device=device,
                )
            graph = capture.graph
            wp.capture_launch(graph)
            wp.synchronize_device(device)

            # Impulses must be carried forward bit-for-bit.
            np.testing.assert_allclose(out_lam_n.numpy()[0], 2.5, atol=1e-7)
            np.testing.assert_allclose(out_lam_t1.numpy()[0], 0.7, atol=1e-7)
            np.testing.assert_allclose(out_lam_t2.numpy()[0], -0.3, atol=1e-7)
            # Anchors must be the prev anchors (rigid: body-local;
            # triangle: barycentric weights). Prev penetration ==
            # fresh penetration (stationary scene) so the carry-prev
            # branch wins.
            np.testing.assert_allclose(out_local_p0.numpy()[0], np.asarray(rigid_local_anchor), atol=1e-6)
            np.testing.assert_allclose(
                out_local_p1.numpy()[0],
                np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                atol=1e-6,
            )
            # Frozen contact frame -- normal and tangent1 carry too.
            np.testing.assert_allclose(out_normal.numpy()[0], [0.0, 0.0, 1.0], atol=1e-6)
            np.testing.assert_allclose(out_tangent1.numpy()[0], [1.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
