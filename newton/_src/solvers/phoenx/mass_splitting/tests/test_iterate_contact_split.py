# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile + launch smoke tests for the split-aware contact iterate
``@wp.func`` shipped in
:mod:`newton._src.solvers.phoenx.mass_splitting.iterate_contact_split`.

These tests intentionally exercise only the *contract* of the
function -- it must (a) compile inside a Warp kernel, (b) link
against the matching ``read_state`` / ``write_state`` /
``InteractionGraphData`` types, and (c) be a no-op when the contact
column is empty. Full numerical equivalence with the unsplit
:func:`~newton._src.solvers.phoenx.constraints.constraint_contact.contact_iterate_at`
is verified in Phase C.2b once the kernel-factory axis lands and a
real ``PhoenXWorld`` step can drive both paths side by side.
"""

from __future__ import annotations

import unittest

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintBodies,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
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
    contact_iterate_split,
)


@wp.kernel(enable_backward=False)
def _drive_contact_iterate_at_split(
    constraints: ContactColumnContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    views: ContactViews,
    graph: InteractionGraphData,
    cid_to_partition_constraint_id: wp.array[wp.int32],
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Single-thread wrapper that calls :func:`contact_iterate_at_split`.

    Used purely as a Warp-codegen smoke check: a kernel that
    references the ``@wp.func`` at all forces Warp to compile it,
    so a successful launch implies (a) the imports resolved, (b)
    the function signature is consistent with its callers, and (c)
    the static-body fallback path inside ``read_state`` works
    against an empty ``InteractionGraphData``.
    """
    body_pair = ConstraintBodies()
    body_pair.b1 = wp.int32(0)
    body_pair.b2 = wp.int32(1)
    contact_iterate_at_split(
        constraints, wp.int32(0), wp.int32(0), bodies, body_pair, idt, cc, views, use_bias, graph,
        cid_to_partition_constraint_id,
    )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX kernels run on CUDA only.")
class TestContactIterateSplitCompile(unittest.TestCase):
    """Compile + launch checks for the split iterate ``@wp.func``."""

    def test_compiles_and_runs_against_empty_contact_column(self):
        """An empty contact column (``contact_count == 0``) must
        early-exit without writing anything; the call still forces
        Warp to compile the full function body so any signature
        mismatch would surface here."""
        device = wp.get_preferred_device()
        bodies = body_container_zeros(2, device=device)
        # Identity orientations -- the function reads them.
        bodies.orientation.assign(
            wp.from_numpy(
                __import__("numpy").array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype="float32"),
                dtype=wp.quatf, device=device,
            )
        )
        # Non-zero inverse mass so the body is treated as dynamic.
        bodies.inverse_mass.assign(wp.from_numpy(
            __import__("numpy").array([1.0, 1.0], dtype="float32"),
            dtype=wp.float32, device=device,
        ))
        constraints = contact_column_container_zeros(1, device=device)
        cc = contact_container_zeros(1, device=device)
        # Sentinel ContactViews -- length-zero arrays for everything
        # the function might read; ``contact_count == 0`` means none
        # of these slots are ever indexed.
        views = ContactViews()
        sentinel_vec3 = wp.zeros(0, dtype=wp.vec3f, device=device)
        sentinel_int = wp.zeros(0, dtype=wp.int32, device=device)
        sentinel_float = wp.zeros(0, dtype=wp.float32, device=device)
        views.rigid_contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        views.rigid_contact_point0 = sentinel_vec3
        views.rigid_contact_point1 = sentinel_vec3
        views.rigid_contact_normal = sentinel_vec3
        views.rigid_contact_shape0 = sentinel_int
        views.rigid_contact_shape1 = sentinel_int
        views.rigid_contact_match_index = sentinel_int
        views.rigid_contact_margin0 = sentinel_float
        views.rigid_contact_margin1 = sentinel_float
        views.shape_body = sentinel_int
        views.rigid_contact_stiffness = sentinel_float
        views.rigid_contact_damping = sentinel_float
        views.rigid_contact_friction = sentinel_float

        graph = InteractionGraph(max_rigid_bodies=2, max_interactions=2, device=device)
        # No entries registered -> ``read_state`` falls back to the
        # body store; ``inv_factor == 0`` flags static and the
        # function is contractually a no-op for impulses on those
        # bodies. Combined with empty ``contact_count`` the call is
        # entirely a no-op -- which is exactly what we want for a
        # codegen smoke check.
        graph.build()

        cid_to_pcid = wp.zeros(8, dtype=wp.int32, device=device)
        wp.launch(
            kernel=_drive_contact_iterate_at_split,
            dim=1,
            inputs=[constraints, bodies, cc, views, graph.data, cid_to_pcid, wp.float32(60.0), wp.bool(True)],
            device=device,
        )
        wp.synchronize()

    def test_convenience_wrapper_imports(self):
        """The top-level :func:`contact_iterate_split` wrapper exists
        and is referenceable -- catches accidental rename / removal."""
        self.assertTrue(callable(contact_iterate_split))


if __name__ == "__main__":
    unittest.main()
