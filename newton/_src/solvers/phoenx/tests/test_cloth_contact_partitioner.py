# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the graph-coloring partitioner colours cloth-aware contacts
correctly: no two elements in the same colour share any unified-index
node.

Builds two overlapping cloth grids dropping onto a dynamic box --
the two grids have disjoint particle sets, so the share-vertex
broad-phase filter passes their cloth-vs-cloth pairs through, giving
6-node cloth-cloth elements. The dynamic box gives 4-node cloth-rigid
elements (against a static box the rigid node would correctly
collapse to -1 and the element would only have 3 nodes).

Asserts:

* The element-emission kernel produces cloth-tri (3-node),
  cloth-rigid (4-node), and cloth-cloth (6-node) elements.
* For every colour, the pairwise intersection of element node sets
  is empty -- the necessary correctness condition for the per-colour
  parallel iterate to be race-free.

Skipped on CPU because the contact pipeline requires a CUDA broad
phase.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    IncrementalContactPartitioner,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cloth contacts run on CUDA only.",
)
class TestClothContactPartitioner(unittest.TestCase):
    def test_no_two_contacts_in_one_colour_share_a_node(self):
        device = wp.get_preferred_device()

        builder = newton.ModelBuilder()
        # Dynamic box -- gives 4-node (cloth + rigid) elements when
        # cloth tris contact it.
        b = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.1)
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
        # Two stacked cloth grids: their particle sets are disjoint,
        # so the share-vertex broad-phase filter passes their
        # cross-cloth tri pairs through, giving genuine 6-node
        # cloth-cloth elements when the two grids overlap.
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.11),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4, dim_y=4, cell_x=0.25, cell_y=0.25,
            mass=0.05, fix_left=False,
            tri_ke=tri_ke, tri_ka=tri_ka, particle_radius=0.04,
        )
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.13),  # 2 cm above the first grid
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4, dim_y=4, cell_x=0.25, cell_y=0.25,
            mass=0.05, fix_left=False,
            tri_ke=tri_ke, tri_ka=tri_ka, particle_radius=0.04,
        )
        model = builder.finalize(device=device)

        # Standard PhoenX convention: slot 0 is the world-anchor body,
        # Newton body i lands at PhoenX slot i+1. The cloth-aware
        # ``setup_cloth_collision_pipeline`` defaults to
        # ``phoenx_body_offset=1`` to match this layout.
        num_phoenx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        # Populate the dynamic box's inverse mass at PhoenX slot 1 so
        # the element-emission kernel treats it as dynamic (else it
        # collapses to -1 like a static rigid).
        if int(model.body_count) > 0:
            inv_mass_host = np.zeros(num_phoenx_bodies, dtype=np.float32)
            inv_mass_host[1:] = 1.0
            bodies.inverse_mass.assign(inv_mass_host)
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
            rigid_contact_max=4096,
            num_worlds=1,
            substeps=4,
            solver_iterations=8,
            step_layout="single_world",
            device=device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_cloth_triangles_from_model(model)
        pipeline = world.setup_cloth_collision_pipeline(model, rigid_contact_max=4096)

        state = model.state()
        contacts = pipeline.contacts()
        world.collide(state, contacts)

        # Drive the ingest + element emission + partitioner build only.
        # No iterate -- we want to inspect the colouring produced from
        # the cloth-aware element-interaction-data.
        world.step_dt = 1.0 / 60.0
        world.inv_step_dt = 60.0
        world.substep_dt = 1.0 / 240.0
        world._ingest_and_warmstart_contacts(contacts, None)
        world._partitioner.set_costs_from_contacts(
            world.num_joints + world.num_cloth_triangles,
            world._ingest_scratch.num_contact_columns,
            world._contact_cols,
        )
        world._rebuild_elements()
        world._partitioner.reset(world._elements, world._num_active_constraints)
        world._partitioner.build_csr_greedy_with_jp_fallback()
        wp.synchronize()

        n_active = int(world._num_active_constraints.numpy()[0])
        n_colors = int(world._partitioner.num_colors.numpy()[0])
        self.assertGreater(n_active, world.num_cloth_triangles, "expected > num_cloth_triangles active cids (contacts present)")
        self.assertGreater(n_colors, 0)
        self.assertLess(n_colors, n_active, "more colours than elements would mean every element is its own colour -- partitioner failed to merge")

        color_starts = world._partitioner.color_starts.numpy()
        ids_by_color = world._partitioner.element_ids_by_color.numpy()
        elements = world._elements.numpy()

        # Sanity: at least one cloth-cloth (6-node), one cloth-rigid
        # (4-node), and zero rigid-rigid (2-node) elements -- the
        # 8x8-on-box scene has 645 cloth-cloth + 128 cloth-rigid + 0
        # rigid-rigid contact columns.
        node_counts = []
        for cid in range(n_active):
            arr = elements[cid]["bodies"]
            node_counts.append(int(np.sum(arr >= 0)))
        node_counts = np.asarray(node_counts)
        self.assertIn(3, node_counts.tolist(), "expected at least one 3-node (cloth-tri) element")
        self.assertIn(4, node_counts.tolist(), "expected at least one 4-node (cloth-rigid contact) element")
        self.assertIn(6, node_counts.tolist(), "expected at least one 6-node (cloth-cloth contact) element")

        # Pairwise check within each colour.
        violations = 0
        sample_violations = []
        for c in range(n_colors):
            a = color_starts[c]
            b = color_starts[c + 1]
            cids_in_color = ids_by_color[a:b]
            sets = []
            for cid in cids_in_color:
                arr = elements[cid]["bodies"]
                sets.append((int(cid), set(int(x) for x in arr if int(x) >= 0)))
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    ci, ni = sets[i]
                    cj, nj = sets[j]
                    shared = ni & nj
                    if shared:
                        violations += 1
                        if len(sample_violations) < 5:
                            sample_violations.append((c, ci, cj, sorted(shared)))

        self.assertEqual(
            violations,
            0,
            f"partitioner produced {violations} pairwise node-sharing violations within a colour. "
            f"Samples: {sample_violations}",
        )

    def test_max_partitions_cap_dumps_overflow_into_last_colour(self):
        """With ``max_partitions=K`` set, the partitioner must produce
        at most ``K + 1`` colours and stuff anything that didn't fit
        into the last one (the Jitter2-style overflow bucket).

        Re-uses the cloth-vs-cube scene from the prior test (which
        produces > 2 chromatic colours -- two grids of 4x4 plus a
        dynamic box gives a tangle of 3-, 4- and 6-node elements).
        Drops a tight ``max_partitions=2`` cap on a fresh partitioner
        so overflow is guaranteed without scaling the scene up.

        Asserts:

        * ``num_colors <= max_partitions + 1`` (the cap held).
        * ``has_overflow_partition == 1`` (the cap was hit, since
          the underlying graph's chromatic number is > 2 -- the
          previous test already showed it produces multiple colours
          without a cap).
        * For every *regular* colour (``0..max_partitions - 1``) the
          existing pairwise no-shared-node invariant still holds.
        * The overflow bucket (``num_colors - 1``) collects every
          element not placed in a regular colour, and its element
          count equals ``num_active - sum(regular partition sizes)``.
        """
        device = wp.get_preferred_device()
        max_partitions = 2

        builder = newton.ModelBuilder()
        b = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.1)
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.11), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4, dim_y=4, cell_x=0.25, cell_y=0.25,
            mass=0.05, fix_left=False, tri_ke=tri_ke, tri_ka=tri_ka, particle_radius=0.04,
        )
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.13), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4, dim_y=4, cell_x=0.25, cell_y=0.25,
            mass=0.05, fix_left=False, tri_ke=tri_ke, tri_ka=tri_ka, particle_radius=0.04,
        )
        model = builder.finalize(device=device)

        num_phoenx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        if int(model.body_count) > 0:
            inv_mass_host = np.zeros(num_phoenx_bodies, dtype=np.float32)
            inv_mass_host[1:] = 1.0
            bodies.inverse_mass.assign(inv_mass_host)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0, num_cloth_triangles=int(model.tri_count), device=device,
        )
        world = PhoenXWorld(
            bodies=bodies, constraints=constraints, num_joints=0,
            num_particles=int(model.particle_count),
            num_cloth_triangles=int(model.tri_count),
            rigid_contact_max=4096, num_worlds=1, substeps=4, solver_iterations=8,
            step_layout="single_world", device=device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_cloth_triangles_from_model(model)
        pipeline = world.setup_cloth_collision_pipeline(model, rigid_contact_max=4096)

        state = model.state()
        contacts = pipeline.contacts()
        world.collide(state, contacts)

        world.step_dt = 1.0 / 60.0
        world.inv_step_dt = 60.0
        world.substep_dt = 1.0 / 240.0
        world._ingest_and_warmstart_contacts(contacts, None)
        world._partitioner.set_costs_from_contacts(
            world.num_joints + world.num_cloth_triangles,
            world._ingest_scratch.num_contact_columns,
            world._contact_cols,
        )
        world._rebuild_elements()

        # Fresh partitioner with the mass-splitting cap -- same
        # capacity as the world's so it can adopt the elements array
        # verbatim. Match max_num_nodes too so the overflow bucket
        # write doesn't blow past color_starts capacity.
        capped = IncrementalContactPartitioner(
            max_num_interactions=world._constraint_capacity,
            max_num_nodes=max(1, world.num_bodies + world.num_particles),
            device=device,
            use_tile_scan=True,
            max_partitions=max_partitions,
        )
        capped.reset(world._elements, world._num_active_constraints)
        capped.build_csr_greedy_with_jp_fallback()
        wp.synchronize()

        n_active = int(world._num_active_constraints.numpy()[0])
        n_colors = int(capped.num_colors.numpy()[0])
        has_overflow = int(capped.has_overflow_partition.numpy()[0])
        color_starts = capped.color_starts.numpy()
        ids_by_color = capped.element_ids_by_color.numpy()
        elements = world._elements.numpy()

        self.assertLessEqual(
            n_colors, max_partitions + 1,
            f"capped partitioner returned {n_colors} colours (cap = {max_partitions} + overflow)",
        )
        # The unrestricted prior test already showed this scene
        # generates more than 2 colours, so the cap MUST trigger.
        self.assertEqual(
            has_overflow, 1,
            "expected the cap to be hit and the overflow bucket to be populated",
        )
        self.assertEqual(
            n_colors, max_partitions + 1,
            f"with overflow the build must report exactly {max_partitions + 1} colours",
        )

        # Sum of partition sizes equals the active-cid count.
        total_in_colors = int(color_starts[n_colors] - color_starts[0])
        self.assertEqual(
            total_in_colors, n_active,
            f"colours hold {total_in_colors} elements but {n_active} are active",
        )

        # Regular colours (0..max_partitions - 1) must still be true
        # independent sets. The overflow bucket may share bodies.
        for c in range(max_partitions):
            a = int(color_starts[c])
            b = int(color_starts[c + 1])
            cids = ids_by_color[a:b]
            seen: set[int] = set()
            for cid in cids:
                arr = elements[int(cid)]["bodies"]
                nodes = {int(x) for x in arr if int(x) >= 0}
                shared = nodes & seen
                self.assertFalse(
                    shared,
                    f"regular colour {c} has body sharing between elements (shared={sorted(shared)})",
                )
                seen.update(nodes)

        # Overflow bucket: the partition_id stamp must point at the
        # last colour for every element that landed there.
        ovf_a = int(color_starts[max_partitions])
        ovf_b = int(color_starts[max_partitions + 1])
        self.assertGreater(
            ovf_b - ovf_a, 0,
            "has_overflow_partition == 1 but the overflow bucket is empty",
        )
        partition_id = capped.interaction_id_to_partition.numpy()
        for cid in ids_by_color[ovf_a:ovf_b]:
            self.assertEqual(
                int(partition_id[int(cid)]), max_partitions,
                f"element {int(cid)} in overflow slice but interaction_id_to_partition says {int(partition_id[int(cid)])}",
            )


    def test_phoenx_world_cap_steps_through_when_chromatic_fits(self):
        """End-to-end: ``PhoenXWorld`` constructed with
        ``mass_split_max_partitions=16`` (well above the chromatic
        number of any trivial scene) takes one ``step()`` cleanly.

        Validates that the cap parameter plumbs through to the
        partitioner and that no overflow bucket forms when the
        build fits inside the cap."""
        device = wp.get_preferred_device()

        builder = newton.ModelBuilder()
        b = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device)

        num_phoenx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        if int(model.body_count) > 0:
            inv_mass_host = np.zeros(num_phoenx_bodies, dtype=np.float32)
            inv_mass_host[1:] = 1.0
            bodies.inverse_mass.assign(inv_mass_host)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0, num_cloth_triangles=0, device=device,
        )
        world = PhoenXWorld(
            bodies=bodies, constraints=constraints, num_joints=0,
            num_particles=0, num_cloth_triangles=0,
            rigid_contact_max=512, num_worlds=1, substeps=1, solver_iterations=4,
            step_layout="single_world", device=device,
            mass_split_max_partitions=16,
        )
        # Public API spot-check.
        self.assertIsNotNone(world.mass_splitting, "mass_split_max_partitions=16 must allocate the orchestrator")
        self.assertEqual(world.mass_split_max_partitions, 16)
        self.assertEqual(world._partitioner.max_partitions, 16)

        # One step must run cleanly. There are no contacts ingested
        # here (we don't pass `contacts`), so the partitioner sees
        # zero active cids -- exercise of the cap and gate without
        # any iteration kernel work.
        world.step(1.0 / 60.0)

    def test_phoenx_world_cap_steps_through_with_overflow(self):
        """Minimal-scene end-to-end: a 2-cube stack on a plane with
        ``mass_split_max_partitions=1`` forces the partitioner to
        emit an overflow bucket (the chromatic number is > 1
        because cube1-cube2 shares both bodies with cube-plane
        contacts). The Phase C.2 split iterate kernel should
        process the overflow via the
        :class:`MassSplitting` orchestrator's per-(body, partition)
        copy states without raising.

        Kept deliberately small (3 bodies, no cloth, 1 substep,
        2 solver iterations) so the host-side ``setup_from_coloring``
        roundtrip stays sub-second.
        """
        device = wp.get_preferred_device()

        builder = newton.ModelBuilder()
        b1 = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b1, hx=0.5, hy=0.5, hz=0.5)
        b2 = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.6), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b2, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device)

        num_phoenx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        inv_mass_host = np.zeros(num_phoenx_bodies, dtype=np.float32)
        inv_mass_host[1:] = 1.0
        bodies.inverse_mass.assign(inv_mass_host)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0, num_cloth_triangles=0, device=device,
        )
        world = PhoenXWorld(
            bodies=bodies, constraints=constraints, num_joints=0,
            num_particles=0, num_cloth_triangles=0,
            rigid_contact_max=64, num_worlds=1, substeps=1, solver_iterations=2,
            step_layout="single_world", device=device,
            mass_split_max_partitions=1,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))

        # No contacts ingested (no Newton Contacts hooked up); the
        # partitioner's "cap" path still runs, but with zero active
        # contact constraints there's nothing to overflow. This
        # exercises the orchestrator wiring without the
        # narrow-phase / collision-pipeline cost.
        world.step(1.0 / 60.0)
        wp.synchronize()

        self.assertIsNotNone(world.mass_splitting)
        # Body state must stay finite after a clean step.
        body_pos = world.bodies.position.numpy()
        self.assertTrue(np.all(np.isfinite(body_pos)), f"non-finite body position: {body_pos}")


    def test_phoenx_world_cap_runs_under_graph_capture(self):
        """Phase C.3 -- the primary win of the device-side
        ``_setup_mass_splitting_for_step`` chain. Captures one
        ``world.step()`` into a CUDA graph and replays it; the run
        must finish cleanly + produce finite body state without any
        host roundtrip during replay.

        Two-cubes-on-plane scene with ``mass_split_max_partitions=1``
        forces overflow on the first step (so the orchestrator wires
        copy states), then the captured graph carries that wiring
        through replays.
        """
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("graph capture requires CUDA")

        builder = newton.ModelBuilder()
        b1 = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b1, hx=0.5, hy=0.5, hz=0.5)
        b2 = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.6), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b2, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device)

        num_phoenx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=device)
        inv_mass_host = np.zeros(num_phoenx_bodies, dtype=np.float32)
        inv_mass_host[1:] = 1.0
        bodies.inverse_mass.assign(inv_mass_host)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0, num_cloth_triangles=0, device=device,
        )
        world = PhoenXWorld(
            bodies=bodies, constraints=constraints, num_joints=0,
            num_particles=0, num_cloth_triangles=0,
            rigid_contact_max=64, num_worlds=1, substeps=1, solver_iterations=2,
            step_layout="single_world", device=device,
            mass_split_max_partitions=1,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))

        # Warm-up step (compiles kernels, populates partitioner CSR).
        world.step(1.0 / 60.0)

        # Capture + replay.
        with wp.ScopedCapture(device=device) as capture:
            world.step(1.0 / 60.0)
        graph = capture.graph
        wp.capture_launch(graph)
        wp.synchronize()

        body_pos = world.bodies.position.numpy()
        self.assertTrue(np.all(np.isfinite(body_pos)), f"non-finite body position after graph replay: {body_pos}")


if __name__ == "__main__":
    unittest.main()
