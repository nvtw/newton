# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the graph-coloring partitioner colours cloth-aware contacts
correctly: no two elements in the same colour share any unified-index
node.

Builds an 8x8 cloth grid resting on a dynamic box (so cloth-rigid
contacts produce 4-node elements; against a *static* rigid the
partitioner correctly drops the static-anchor node and the element
collapses to 3 nodes), runs the contact ingest + element emission +
partitioner.build_csr, then asserts:

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
        # cloth tris contact it; against a body=-1 static box the
        # rigid node would correctly collapse to -1 and the element
        # would only have 3 nodes (just the cloth triangle).
        b = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.1)
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 0.11),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=8, dim_y=8, cell_x=0.125, cell_y=0.125,
            mass=0.05, fix_left=False,
            tri_ke=tri_ke, tri_ka=tri_ka, particle_radius=0.04,
        )
        model = builder.finalize(device=device)

        bodies = body_container_zeros(max(1, int(model.body_count)), device=device)
        # Populate the box body's inverse mass so the element-emission
        # kernel treats it as dynamic (else it collapses to -1 like a
        # static rigid). The actual value doesn't matter for the
        # colouring test -- we just need it strictly positive.
        if int(model.body_count) > 0:
            bodies.inverse_mass.assign(np.full(int(model.body_count), 1.0, dtype=np.float32))
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


if __name__ == "__main__":
    unittest.main()
