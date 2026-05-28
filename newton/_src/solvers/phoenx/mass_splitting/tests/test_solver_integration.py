# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the mass-splitting integration in
:class:`PhoenXWorld`.

Sub-step 6b verifies the build path: with ``mass_splitting=True``, the
per-step substep loop builds the (body, partition_key) interaction
graph after coloring and writes it into the copy-state container.
The broadcast / average / writeback wiring lands in 6c.

For the build to produce entries, the world needs at least one active
constraint that involves a non-static body. We construct a hand-built
two-body scene with a single joint that exercises the partitioner +
emit path.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _make_minimal_world(*, mass_splitting: bool, device) -> PhoenXWorld:
    """2-body, no joint, no contact world. step() runs but the
    interaction graph stays empty — exercise the no-op path."""
    bodies = body_container_zeros(2, device=device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
    return PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        rigid_contact_max=4,
        mass_splitting=mass_splitting,
        max_colored_partitions=12,
        step_layout="single_world",
        device=device,
    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only per feedback_phoenx_tests_capture_only.",
)
class TestMassSplittingSolverWiring(unittest.TestCase):
    def test_step_runs_with_mass_splitting_enabled(self):
        # Bare-bones smoke test: step() must not crash when
        # ``mass_splitting=True`` even with zero active constraints.
        # The build kernel still launches (single-world path) but
        # ``num_active_constraints == 0`` makes the emit a no-op.
        device = wp.get_preferred_device()
        world = _make_minimal_world(mass_splitting=True, device=device)
        world.step(0.01)
        wp.synchronize_device(device)
        # No constraints → no pairs emitted → graph stays disabled
        # (``highest_index_in_use[0] == 0``).
        self.assertEqual(int(world._copy_state.highest_index_in_use.numpy()[0]), 0)

    def test_step_runs_under_graph_capture(self):
        # Load-bearing capture-safety assertion: every kernel launched
        # by the new ``_rebuild_mass_splitting_graph`` path must work
        # inside ``wp.ScopedCapture`` + ``wp.capture_launch``. See
        # ``feedback_phoenx_tests_capture_only.md``.
        device = wp.get_preferred_device()
        world = _make_minimal_world(mass_splitting=True, device=device)
        # Warm-up.
        world.step(0.01)
        with wp.ScopedCapture(device=device) as capture:
            world.step(0.01)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

    def test_build_populates_copy_state_for_hand_built_elements(self):
        # Bypass contact ingest: stuff three "fake" constraint elements
        # directly into the partitioner's element buffer, color them,
        # and run the mass-splitting build. Assert each non-static body
        # endpoint shows up exactly once in the copy state with
        # ``inv_factor == 1`` (one slot per body since each lives in a
        # single colour bucket).
        device = wp.get_preferred_device()
        bodies = body_container_zeros(4, device=device)
        # Mark body 0 static (inverse_mass=0); bodies 1, 2, 3 are dynamic.
        bodies.inverse_mass.assign(np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32))
        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            rigid_contact_max=4,
            mass_splitting=True,
            max_colored_partitions=12,
            step_layout="single_world",
            device=device,
        )

        # Inject three hand-built elements directly into the partitioner
        # input. Each element has 2 non-static body endpoints; element
        # 0 conflicts with element 1 via body 1, so the coloring will
        # put them in different buckets.
        max_bodies = int(MAX_BODIES)
        struct_dtype = np.dtype(
            {"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 4 * max_bodies}
        )
        elems_np = np.zeros(world._constraint_capacity, dtype=struct_dtype)
        elems_np["bodies"][:] = -1
        elems_np["bodies"][0, 0] = 1
        elems_np["bodies"][0, 1] = 2
        elems_np["bodies"][1, 0] = 1
        elems_np["bodies"][1, 1] = 3
        elems_np["bodies"][2, 0] = 2
        elems_np["bodies"][2, 1] = 3
        world._elements.assign(wp.from_numpy(elems_np, dtype=ElementInteractionData, device=device))
        world._num_active_constraints.assign(np.array([3], dtype=np.int32))

        # Run the partitioner + mass-splitting build by hand (skip
        # step()'s contact ingest path).
        world._partitioner.reset(world._elements, world._num_active_constraints)
        world._partitioner.build_csr_greedy_with_jp_fallback()
        world._rebuild_mass_splitting_graph()
        wp.synchronize_device(device)

        # With max_colored_partitions=12 and only 3 elements, coloring
        # fits trivially in 3 colours -- no overflow. Every body lives
        # in exactly one colour bucket -> one (body, partition_key=0)
        # slot per body.
        section_end = world._copy_state.section_end.numpy()
        # Body 0 is static -> 0 slots. Bodies 1, 2, 3 -> 1 slot each.
        self.assertEqual(section_end[0], 0)
        self.assertEqual(section_end[1] - section_end[0], 1)
        self.assertEqual(section_end[2] - section_end[1], 1)
        self.assertEqual(section_end[3] - section_end[2], 1)
        # 3 slots total (1 per dynamic body).
        self.assertEqual(int(world._copy_state.highest_index_in_use.numpy()[0]), 3)
        # Every slot's partition_key is 0 (no overflow).
        partition_list = world._copy_state.partition_list.numpy()[:3]
        np.testing.assert_array_equal(partition_list, [0, 0, 0])

    def test_overflow_bucket_produces_multiple_slots_per_body(self):
        # Force coloring into the overflow bucket by setting
        # ``max_colored_partitions=1``: every adjacent element after
        # the first one lands in colour 1 (overflow). A hub body
        # touching three elements ends up with multiple slots when
        # ``mass_splitting_batch_size=1`` (strongest splitting).
        device = wp.get_preferred_device()
        bodies = body_container_zeros(5, device=device)
        bodies.inverse_mass.assign(np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            rigid_contact_max=8,
            mass_splitting=True,
            max_colored_partitions=1,  # 1 regular colour + 1 overflow
            mass_splitting_batch_size=1,  # no batching -> 1 slot per overflow contact
            step_layout="single_world",
            device=device,
        )

        max_bodies = int(MAX_BODIES)
        struct_dtype = np.dtype(
            {"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 4 * max_bodies}
        )
        elems_np = np.zeros(world._constraint_capacity, dtype=struct_dtype)
        elems_np["bodies"][:] = -1
        # Body 1 is the hub; three elements share it.
        elems_np["bodies"][0, 0] = 1
        elems_np["bodies"][0, 1] = 2
        elems_np["bodies"][1, 0] = 1
        elems_np["bodies"][1, 1] = 3
        elems_np["bodies"][2, 0] = 1
        elems_np["bodies"][2, 1] = 4
        world._elements.assign(wp.from_numpy(elems_np, dtype=ElementInteractionData, device=device))
        world._num_active_constraints.assign(np.array([3], dtype=np.int32))

        world._partitioner.reset(world._elements, world._num_active_constraints)
        world._partitioner.build_csr_greedy_with_jp_fallback()
        world._rebuild_mass_splitting_graph()
        wp.synchronize_device(device)

        # One element colours at 0 (regular), the other two land in
        # colour 1 (overflow). Body 1 (the hub) participates in all
        # three, so it gets >= 2 slots (1 for the regular bucket, then
        # one per overflow slot it touches with batch_size=1).
        section_end = world._copy_state.section_end.numpy()
        body1_slot_count = section_end[1] - section_end[0]
        self.assertGreaterEqual(body1_slot_count, 2)
        # Each non-hub body (2, 3, 4) participates in exactly one
        # element, so they get exactly 1 slot each.
        self.assertEqual(section_end[2] - section_end[1], 1)
        self.assertEqual(section_end[3] - section_end[2], 1)
        self.assertEqual(section_end[4] - section_end[3], 1)
        # Static body has 0 slots.
        self.assertEqual(section_end[0], 0)

    def test_disabled_path_unchanged(self):
        # ``mass_splitting=False`` (default) must not even launch the
        # build kernel; ``_copy_state.highest_index_in_use`` stays at
        # the sentinel zero and ``_interaction_graph_scratch.num_pairs``
        # is never written to.
        device = wp.get_preferred_device()
        world = _make_minimal_world(mass_splitting=False, device=device)
        world.step(0.01)
        self.assertEqual(int(world._copy_state.highest_index_in_use.numpy()[0]), 0)
        self.assertEqual(int(world._interaction_graph_scratch.num_pairs.numpy()[0]), 0)


def _build_box_stack_scene(num_boxes: int, mass_splitting: bool, device):
    """Build a small free-falling pile that produces real contact
    columns and exercises the iterate path. Returns the
    :class:`PhoenXWorld`, a Newton ``State``, ``Contacts``, and the
    ``shape_body`` map.

    The scene is rigid-only (no joints, no cloth) so it's compatible
    with the current mass-splitting guards.
    """
    import newton  # noqa: PLC0415
    from newton._src.solvers.phoenx.body import body_container_zeros  # noqa: PLC0415
    from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING  # noqa: PLC0415

    mb = newton.ModelBuilder()
    mb.default_shape_cfg.gap = 0.05
    # Ground plane (static).
    mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
    # Stacked boxes (slightly offset so contacts are non-degenerate).
    half_ext = (0.5, 0.5, 0.5)
    body_ids: list[int] = []
    for i in range(num_boxes):
        z = 0.55 + 1.05 * i
        body = mb.add_body(xform=wp.transform(p=wp.vec3(0.01 * i, 0.0, z), q=wp.quat_identity()))
        mb.add_shape_box(body, hx=half_ext[0], hy=half_ext[1], hz=half_ext[2])
        body_ids.append(body)
    model = mb.finalize(device=device)
    state = model.state()
    collision_pipeline = newton.CollisionPipeline(model, contact_matching=PHOENX_CONTACT_MATCHING)
    contacts = collision_pipeline.contacts()

    num_bodies_phx = model.body_count + 1  # +1 for static-world slot 0
    bodies = body_container_zeros(num_bodies_phx, device=device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        rigid_contact_max=int(contacts.rigid_contact_max),
        gravity=(0.0, 0.0, -9.81),
        substeps=4,
        solver_iterations=4,
        velocity_iterations=1,
        mass_splitting=mass_splitting,
        max_colored_partitions=12,
        step_layout="single_world",
        device=device,
    )

    shape_body_np = model.shape_body.numpy()
    shape_body_phx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
    shape_body = wp.array(shape_body_phx, dtype=wp.int32, device=device)
    return world, state, contacts, model, collision_pipeline, shape_body, body_ids


def _sync_newton_to_phoenx(model, state, bodies, device):
    from newton._src.solvers.phoenx.examples.example_common import (  # noqa: PLC0415
        newton_to_phoenx_kernel,
    )

    n = model.body_count
    if n == 0:
        return
    wp.launch(
        newton_to_phoenx_kernel,
        dim=n,
        inputs=[state.body_q, state.body_qd, model.body_com],
        outputs=[
            bodies.position[1 : 1 + n],
            bodies.orientation[1 : 1 + n],
            bodies.velocity[1 : 1 + n],
            bodies.angular_velocity[1 : 1 + n],
        ],
        device=device,
    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only.",
)
class TestMassSplittingPhysicsEquivalence(unittest.TestCase):
    """When the constraint graph fits in K=12 colours (no overflow), the
    slot-aware path with ``mass_splitting=True`` must produce the SAME
    physics as ``mass_splitting=False``.

    Mass splitting only meaningfully changes the impulse algebra in the
    overflow bucket (``inv_factor > 1``). For colour buckets ``0..K-1``,
    ``parallel_id=0`` and every body has exactly one slot, so
    ``inv_factor=1`` and the slot helpers fall through to identity
    arithmetic. The round-trip through ``broadcast → iterate via slots
    → writeback`` is mathematically identity (modulo float precision).

    These tests use real Newton scenes (gravity-driven box stacks) so
    they exercise the full contact ingest → coloring → build →
    broadcast → solve → writeback pipeline.
    """

    def _run_n_frames(self, world, state, contacts, model, collision_pipeline, shape_body, n_frames, dt):
        for _ in range(n_frames):
            _sync_newton_to_phoenx(model, state, world.bodies, world.device)
            model.collide(state, contacts=contacts, collision_pipeline=collision_pipeline)
            world.step(dt=dt, contacts=contacts, shape_body=shape_body)
        return world.bodies.position.numpy().copy(), world.bodies.velocity.numpy().copy()

    def test_single_box_on_plane_matches_disabled(self):
        device = wp.get_preferred_device()
        # mass_splitting=False reference.
        w0, s0, c0, m0, cp0, sb0, _ = _build_box_stack_scene(1, mass_splitting=False, device=device)
        pos0, vel0 = self._run_n_frames(w0, s0, c0, m0, cp0, sb0, 10, 1.0 / 60.0)
        # mass_splitting=True with no overflow → same physics.
        w1, s1, c1, m1, cp1, sb1, _ = _build_box_stack_scene(1, mass_splitting=True, device=device)
        pos1, vel1 = self._run_n_frames(w1, s1, c1, m1, cp1, sb1, 10, 1.0 / 60.0)
        # Slot 1 = the dynamic box.
        np.testing.assert_allclose(pos1[1], pos0[1], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(vel1[1], vel0[1], rtol=1e-4, atol=1e-4)

    def test_two_box_stack_matches_disabled(self):
        device = wp.get_preferred_device()
        w0, s0, c0, m0, cp0, sb0, _ = _build_box_stack_scene(2, mass_splitting=False, device=device)
        pos0, vel0 = self._run_n_frames(w0, s0, c0, m0, cp0, sb0, 30, 1.0 / 60.0)
        w1, s1, c1, m1, cp1, sb1, _ = _build_box_stack_scene(2, mass_splitting=True, device=device)
        pos1, vel1 = self._run_n_frames(w1, s1, c1, m1, cp1, sb1, 30, 1.0 / 60.0)
        # Both bodies (slots 1, 2).
        for b in (1, 2):
            np.testing.assert_allclose(pos1[b], pos0[b], rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(vel1[b], vel0[b], rtol=1e-3, atol=1e-3)

    def test_single_box_with_overflow_still_settles(self):
        # Force the overflow path by setting ``max_colored_partitions=1``:
        # every contact-bucket past the first goes to the overflow
        # bucket. With a single box on a plane there's only one contact
        # column so coloring fits in 1 colour and nothing overflows --
        # but every body has exactly 1 slot, exercising the slot-aware
        # iterate path with ``inv_factor=1``.
        device = wp.get_preferred_device()
        w0, s0, c0, m0, cp0, sb0, _ = _build_box_stack_scene(1, mass_splitting=False, device=device)
        pos0, _ = self._run_n_frames(w0, s0, c0, m0, cp0, sb0, 30, 1.0 / 60.0)
        # mass_splitting=True with K=1 still routes through slots.
        device = wp.get_preferred_device()
        import newton  # noqa: PLC0415
        from newton._src.solvers.phoenx.body import body_container_zeros  # noqa: PLC0415
        from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING  # noqa: PLC0415

        mb = newton.ModelBuilder()
        mb.default_shape_cfg.gap = 0.05
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
        body = mb.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.55), q=wp.quat_identity()))
        mb.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        model = mb.finalize(device=device)
        state = model.state()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching=PHOENX_CONTACT_MATCHING)
        contacts = collision_pipeline.contacts()
        bodies = body_container_zeros(model.body_count + 1, device=device)
        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        w1 = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            rigid_contact_max=int(contacts.rigid_contact_max),
            gravity=(0.0, 0.0, -9.81),
            substeps=4,
            solver_iterations=4,
            velocity_iterations=1,
            mass_splitting=True,
            max_colored_partitions=1,  # Force overflow more aggressively.
            step_layout="single_world",
            device=device,
        )
        shape_body_np = model.shape_body.numpy()
        shape_body_phx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        sb1 = wp.array(shape_body_phx, dtype=wp.int32, device=device)
        pos1, _ = self._run_n_frames(w1, state, contacts, model, collision_pipeline, sb1, 30, 1.0 / 60.0)
        # Both should have the box resting on the plane (z ≈ 0.5).
        # We don't require byte-equivalent positions (the iterate
        # algebra inside the overflow bucket scales by inv_factor); we
        # just require the box hasn't tunneled through the plane nor
        # exploded upward.
        self.assertGreater(float(pos1[1][2]), 0.45)  # not below the plane
        self.assertLess(float(pos1[1][2]), 0.6)  # not floating
        # Sanity vs disabled.
        self.assertAlmostEqual(float(pos1[1][2]), float(pos0[1][2]), delta=0.05)

    def test_pendulum_joint_matches_disabled(self):
        # Two-body pendulum: parent body welded to world via a fixed
        # joint, child body hanging from a revolute joint. With
        # mass_splitting=True (K=12, no overflow), the joint iterate
        # routes through the slot helpers with ``inv_factor=1``, so the
        # math is identity vs ``mass_splitting=False`` (modulo float
        # ordering). Covers the joint refactor end-to-end.
        device = wp.get_preferred_device()
        import newton  # noqa: PLC0415

        def _build(mass_splitting: bool):
            mb = newton.ModelBuilder()
            mb.add_ground_plane()
            hx = hy = 0.1
            hz = 0.4
            parent = mb.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity()))
            child = mb.add_link(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0 - 2.0 * hz), q=wp.quat_identity()),
                label="pend_child",
            )
            mb.add_shape_box(parent, hx=hx, hy=hy, hz=hz)
            mb.add_shape_box(child, hx=hx, hy=hy, hz=hz)
            j_fix = mb.add_joint_fixed(
                parent=-1,
                child=parent,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            )
            j_rev = mb.add_joint_revolute(
                parent=parent,
                child=child,
                axis=wp.vec3(1.0, 0.0, 0.0),
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -hz), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, hz), q=wp.quat_identity()),
            )
            mb.add_articulation([j_fix, j_rev])
            mb.joint_q[-1] = 0.3
            mb.color()
            model = mb.finalize(device=device)
            solver = newton.solvers.SolverPhoenX(
                model,
                substeps=4,
                solver_iterations=8,
                velocity_iterations=1,
                step_layout="single_world",
                mass_splitting=mass_splitting,
                max_colored_partitions=12,
            )
            return model, solver

        model0, solver0 = _build(mass_splitting=False)
        model1, solver1 = _build(mass_splitting=True)
        state0 = model0.state()
        state1 = model1.state()
        contacts0 = newton.CollisionPipeline(model0).contacts() if False else None
        contacts1 = newton.CollisionPipeline(model1).contacts() if False else None
        for _ in range(30):
            solver0.step(state0, state0, control=None, contacts=contacts0, dt=1.0 / 60.0)
            solver1.step(state1, state1, control=None, contacts=contacts1, dt=1.0 / 60.0)
        q0 = state0.body_q.numpy()
        q1 = state1.body_q.numpy()
        np.testing.assert_allclose(q1, q0, rtol=1e-3, atol=1e-3)

    def test_cloth_grid_matches_disabled(self):
        # Small cloth grid falling onto a static box. With
        # mass_splitting=True (K=12, no overflow) every cloth-triangle
        # endpoint has exactly one slot so ``inv_factor=1`` and the
        # prepare-time mass scaling is identity. End-state particle
        # positions must match the ``mass_splitting=False`` reference.
        device = wp.get_preferred_device()
        import newton  # noqa: PLC0415
        from newton._src.solvers.phoenx.body import body_container_zeros  # noqa: PLC0415
        from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (  # noqa: PLC0415
            cloth_lame_from_youngs_poisson_plane_stress,
        )

        def _build(mass_splitting: bool):
            mb = newton.ModelBuilder()
            mb.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                hx=0.5,
                hy=0.5,
                hz=0.1,
            )
            tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
            mb.add_cloth_grid(
                pos=wp.vec3(-0.3, -0.3, 0.3),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=4,
                dim_y=4,
                cell_x=0.15,
                cell_y=0.15,
                mass=0.05,
                fix_left=False,
                tri_ke=tri_ke,
                tri_ka=tri_ka,
                particle_radius=0.04,
            )
            model = mb.finalize(device=device)
            bodies = body_container_zeros(max(1, int(model.body_count)), device=device)
            constraints = PhoenXWorld.make_constraint_container(
                num_joints=0,
                num_cloth_triangles=int(model.tri_count),
                device=device,
            )
            world = PhoenXWorld(
                bodies=bodies,
                constraints=constraints,
                num_joints=0,
                num_particles=int(model.particle_count),
                num_cloth_triangles=int(model.tri_count),
                rigid_contact_max=2048,
                num_worlds=1,
                substeps=4,
                solver_iterations=8,
                step_layout="single_world",
                device=device,
                mass_splitting=mass_splitting,
                max_colored_partitions=12,
            )
            world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
            world.populate_cloth_triangles_from_model(model)
            pipeline = world.setup_cloth_collision_pipeline(
                model,
                cloth_thickness=0.005,
                cloth_gap=0.010,
                rigid_contact_max=2048,
            )
            state = model.state()
            contacts = pipeline.contacts()
            return world, state, contacts

        w0, s0, c0 = _build(mass_splitting=False)
        w1, s1, c1 = _build(mass_splitting=True)
        for _ in range(30):
            w0.collide(s0, c0)
            w0.step(1.0 / 60.0, contacts=c0)
            w1.collide(s1, c1)
            w1.step(1.0 / 60.0, contacts=c1)
        p0 = w0.particles.position.numpy()
        p1 = w1.particles.position.numpy()
        # Cloth particles should land at the same height (within float
        # ordering noise); the cloth-triangle iterate's inv_mass scaling
        # is identity for inv_factor=1.
        np.testing.assert_allclose(p1, p0, rtol=1e-3, atol=1e-3)

    def test_box_stack_under_graph_capture_with_mass_splitting(self):
        # Load-bearing capture-safety check: the full pipeline (ingest →
        # color → mass-splitting build → broadcast → solve → writeback →
        # integrate) must run inside a captured CUDA graph and re-launch
        # deterministically across multiple frames.
        device = wp.get_preferred_device()
        world, state, contacts, model, cp, shape_body, _ = _build_box_stack_scene(2, mass_splitting=True, device=device)

        def _frame():
            _sync_newton_to_phoenx(model, state, world.bodies, world.device)
            model.collide(state, contacts=contacts, collision_pipeline=cp)
            world.step(dt=1.0 / 60.0, contacts=contacts, shape_body=shape_body)

        # Warm-up.
        _frame()
        with wp.ScopedCapture(device=device) as capture:
            _frame()
        # Run several frames of the captured graph; assertion is just
        # that no kernel crashes and bodies fall.
        for _ in range(5):
            wp.capture_launch(capture.graph)
        pos = world.bodies.position.numpy()
        # Bodies should have moved downward from their initial z (>= 0.55).
        self.assertLess(float(pos[1][2]), 1.0)
        self.assertLess(float(pos[2][2]), 2.0)


if __name__ == "__main__":
    unittest.main()
