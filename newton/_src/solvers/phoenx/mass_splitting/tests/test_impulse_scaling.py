# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical equivalence test: split iterate at ``inv_factor=1``
(every body in a single partition) must produce body velocities
bit-equal to the unsplit ``contact_iterate_at`` after the
broadcast / iterate / average / write_back round-trip.

Mirrors the C# invariant: with all bodies in 1 partition,
``invFactor = 1`` everywhere and the mass-splitting pipeline
reduces to standard PGS. Any deviation here means the rigid
contact split path is broken regardless of how many partitions
the scene actually uses.

Test plan: a single 2-body rigid contact at known velocities,
known mass / inertia / lever-arms. Drive it through (a) the
unsplit ``contact_iterate_at`` against a body store, and (b) the
split path against an InteractionGraph with one (body, 0) entry
per body (inv_factor = 1). Compare body velocities; require
match to fp tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintBodies,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_iterate_at,
    contact_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraph,
    InteractionGraphData,
)
from newton._src.solvers.phoenx.mass_splitting.iterate_contact_split import (
    contact_iterate_at_split,
)
from newton._src.solvers.phoenx.mass_splitting.setup_kernels import (
    broadcast_to_copy_states_unified_kernel,
    copy_state_into_unified_kernel,
)
from newton._src.solvers.phoenx.particle import ParticleContainer, particle_container_zeros


@wp.kernel(enable_backward=False)
def _drive_unsplit(
    constraints: ContactColumnContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    views: ContactViews,
    idt: wp.float32,
    use_bias: wp.bool,
):
    body_pair = ConstraintBodies()
    body_pair.b1 = wp.int32(0)
    body_pair.b2 = wp.int32(1)
    contact_prepare_for_iteration_at(
        constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views,
    )
    contact_iterate_at(
        constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views, use_bias,
    )


@wp.kernel(enable_backward=False)
def _drive_split(
    constraints: ContactColumnContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    views: ContactViews,
    idt: wp.float32,
    use_bias: wp.bool,
    graph: InteractionGraphData,
    cid_to_partition_constraint_id: wp.array[wp.int32],
):
    body_pair = ConstraintBodies()
    body_pair.b1 = wp.int32(0)
    body_pair.b2 = wp.int32(1)
    # Reuse the unsplit prepare against the body store. Broadcast
    # then snapshots the post-warm-start body velocity into the
    # copy states.
    contact_prepare_for_iteration_at(
        constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views,
    )
    # The split iterate reads / writes through copy states.
    contact_iterate_at_split(
        constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views, use_bias, graph,
        cid_to_partition_constraint_id,
    )


def _make_two_body_scene(device):
    """Two unit-mass cubes at z=0 and z=1.5 with downward velocity
    on the upper body so the contact is in closing motion."""
    bodies = body_container_zeros(2, device=device)
    inv_mass = np.array([1.0, 1.0], dtype=np.float32)
    bodies.inverse_mass.assign(inv_mass)
    bodies.inverse_inertia_world.assign(
        wp.from_numpy(
            np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            dtype=wp.mat33f, device=device,
        )
    )
    bodies.inverse_inertia.assign(
        wp.from_numpy(
            np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            dtype=wp.mat33f, device=device,
        )
    )
    bodies.position.assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=np.float32))
    bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    # Body 1 (upper) moving downward at -2 m/s.
    bodies.velocity.assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]], dtype=np.float32))
    bodies.angular_velocity.assign(np.zeros((2, 3), dtype=np.float32))
    bodies.body_com.assign(np.zeros((2, 3), dtype=np.float32))
    return bodies


def _make_one_contact(device):
    """Constraint container with one shape pair; contact container
    with one normal contact at the centre between the bodies."""
    constraints = contact_column_container_zeros(1, device=device)
    cc = contact_container_zeros(8, device=device)
    # Wire the cid header: contact_count = 1, contact_first = 0,
    # body1 = 0, body2 = 1, friction = 0.
    constraints_np = constraints.data.numpy().copy()
    # The dword layout is opaque; populate via the public setters.
    from newton._src.solvers.phoenx.constraints.constraint_contact import (  # noqa: PLC0415
        contact_set_body1, contact_set_body2,
        contact_set_contact_count, contact_set_contact_first,
        contact_set_friction, contact_set_friction_dynamic,
    )

    @wp.kernel(enable_backward=False)
    def _populate_cid(cc: ContactColumnContainer):
        contact_set_body1(cc, wp.int32(0), wp.int32(0))
        contact_set_body2(cc, wp.int32(0), wp.int32(1))
        contact_set_contact_count(cc, wp.int32(0), wp.int32(1))
        contact_set_contact_first(cc, wp.int32(0), wp.int32(0))
        contact_set_friction(cc, wp.int32(0), wp.float32(0.0))
        contact_set_friction_dynamic(cc, wp.int32(0), wp.float32(0.0))

    wp.launch(_populate_cid, dim=1, inputs=[constraints], device=device)

    # ContactViews: one contact at the midpoint between the bodies,
    # normal pointing from body 0 to body 1 (+Z), zero margins, no
    # match_index logic.
    views = ContactViews()
    n_contacts = 1
    views.rigid_contact_count = wp.array([n_contacts], dtype=wp.int32, device=device)
    views.rigid_contact_normal = wp.array(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.vec3f, device=device)
    # Local contact points: midpoint at z=0.75 in world; for body 0 (at z=0) that's local_p0 = (0,0,0.75).
    # For body 1 (at z=1.5) local_p1 = (0,0,-0.75).
    views.rigid_contact_point0 = wp.array(np.array([[0.0, 0.0, 0.75]], dtype=np.float32), dtype=wp.vec3f, device=device)
    views.rigid_contact_point1 = wp.array(np.array([[0.0, 0.0, -0.75]], dtype=np.float32), dtype=wp.vec3f, device=device)
    views.rigid_contact_margin0 = wp.array(np.array([0.0], dtype=np.float32), dtype=wp.float32, device=device)
    views.rigid_contact_margin1 = wp.array(np.array([0.0], dtype=np.float32), dtype=wp.float32, device=device)
    views.rigid_contact_shape0 = wp.zeros(n_contacts, dtype=wp.int32, device=device)
    views.rigid_contact_shape1 = wp.zeros(n_contacts, dtype=wp.int32, device=device)
    views.rigid_contact_match_index = wp.full((n_contacts,), -1, dtype=wp.int32, device=device)
    views.shape_body = wp.array([0, 1], dtype=wp.int32, device=device)
    # Sentinels for soft-contact arrays.
    views.rigid_contact_stiffness = wp.zeros(0, dtype=wp.float32, device=device)
    views.rigid_contact_damping = wp.zeros(0, dtype=wp.float32, device=device)
    views.rigid_contact_friction = wp.zeros(0, dtype=wp.float32, device=device)
    return constraints, cc, views


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX kernels run on CUDA only.")
class TestImpulseScalingMatchesUnsplit(unittest.TestCase):
    """Split iterate at inv_factor=1 must equal the unsplit iterate."""

    def test_split_at_inv_factor_one_matches_unsplit(self):
        device = wp.get_preferred_device()

        # Reference run: unsplit iterate against body store.
        bodies_ref = _make_two_body_scene(device)
        constraints_ref, cc_ref, views_ref = _make_one_contact(device)
        idt = wp.float32(60.0)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _drive_unsplit, dim=1,
                inputs=[constraints_ref, bodies_ref, cc_ref, views_ref, idt, wp.bool(True)],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()
        ref_v = bodies_ref.velocity.numpy().copy()
        ref_w = bodies_ref.angular_velocity.numpy().copy()

        # Split run: same inputs, but iterate routes through
        # InteractionGraph copies. Build a graph with each body
        # registered at (body, 0) so inv_factor=1 for both.
        bodies_split = _make_two_body_scene(device)
        constraints_split, cc_split, views_split = _make_one_contact(device)
        graph = InteractionGraph(max_rigid_bodies=2, max_interactions=2, device=device)
        graph.add_entry(0, 0)
        graph.add_entry(1, 0)
        graph.build()
        # cid 0 -> partition_constraint_id 0 (regular).
        cid_to_pcid = wp.zeros(1, dtype=wp.int32, device=device)
        # Sentinel particle SoA (rigid scene; not used by split iterate
        # for rigid-only bodies, but the broadcast / write_back kernels
        # take a particle container regardless).
        particles = particle_container_zeros(1, device=device)
        # Drive: prepare + broadcast + iterate (split) + write_back.
        with wp.ScopedCapture(device=device) as capture:
            # Prepare against body store (same as unsplit). The
            # subsequent broadcast snapshots the post-prepare body
            # state into the per-(body, partition) copies.
            @wp.kernel(enable_backward=False)
            def _drive_prep_only(
                constraints: ContactColumnContainer,
                bodies: BodyContainer,
                cc: ContactContainer,
                views: ContactViews,
                idt: wp.float32,
            ):
                body_pair = ConstraintBodies()
                body_pair.b1 = wp.int32(0)
                body_pair.b2 = wp.int32(1)
                contact_prepare_for_iteration_at(
                    constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views,
                )
            wp.launch(
                _drive_prep_only, dim=1,
                inputs=[constraints_split, bodies_split, cc_split, views_split, idt],
                device=device,
            )
            wp.launch(
                broadcast_to_copy_states_unified_kernel,
                dim=2,  # 2 bodies, 0 particles
                inputs=[
                    graph.data,
                    bodies_split.position, bodies_split.orientation,
                    bodies_split.velocity, bodies_split.angular_velocity,
                    particles, wp.int32(2), wp.float32(1.0 / 60.0),
                ],
                device=device,
            )

            @wp.kernel(enable_backward=False)
            def _drive_iter_only(
                constraints: ContactColumnContainer,
                bodies: BodyContainer,
                cc: ContactContainer,
                views: ContactViews,
                idt: wp.float32,
                use_bias: wp.bool,
                graph: InteractionGraphData,
                cid_to_pcid: wp.array[wp.int32],
            ):
                body_pair = ConstraintBodies()
                body_pair.b1 = wp.int32(0)
                body_pair.b2 = wp.int32(1)
                contact_iterate_at_split(
                    constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views, wp.bool(True), graph,
                    cid_to_pcid,
                )
            wp.launch(
                _drive_iter_only, dim=1,
                inputs=[constraints_split, bodies_split, cc_split, views_split, idt, wp.bool(True), graph.data, cid_to_pcid],
                device=device,
            )
            wp.launch(
                copy_state_into_unified_kernel,
                dim=2,
                inputs=[
                    graph.data,
                    bodies_split.position, bodies_split.orientation,
                    bodies_split.velocity, bodies_split.angular_velocity,
                    particles, wp.int32(2), wp.float32(60.0),
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        split_v = bodies_split.velocity.numpy()
        split_w = bodies_split.angular_velocity.numpy()

        np.testing.assert_allclose(
            split_v, ref_v, rtol=1e-4, atol=1e-5,
            err_msg=f"split velocity != unsplit. ref={ref_v}, split={split_v}",
        )
        np.testing.assert_allclose(
            split_w, ref_w, rtol=1e-4, atol=1e-5,
            err_msg=f"split angular velocity != unsplit. ref={ref_w}, split={split_w}",
        )


if __name__ == "__main__":
    unittest.main()
