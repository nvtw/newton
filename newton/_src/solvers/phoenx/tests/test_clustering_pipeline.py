# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the ``ClusteringPipeline`` composition + the
``enable_clustering`` opt-in on ``SolverPhoenX``.

Covers two things:

1. Direct ``ClusteringPipeline.build()`` produces the same outputs as
   chaining ``ConstraintClusterBuilder.build_clusters`` +
   ``SupernodalElements.build`` by hand (just verifies the
   composition).
2. ``SolverPhoenX(enable_clustering=True)`` runs end-to-end on a
   minimal joint scene without crashing and surfaces a non-``None``
   ``num_clusters`` in ``step_report()``.

Invocation: per repo memory, run via

    uv run --extra dev -m newton.tests -k test_clustering_pipeline
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.clustering import (
    ClusteringPipeline,
    ConstraintClusterBuilder,
    SupernodalElements,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)


def _make_elements(bodies_np: np.ndarray, device) -> wp.array:
    """Pack ``(N, MAX_BODIES)`` int32 into a ``wp.array[ElementInteractionData]``."""
    n = bodies_np.shape[0]
    max_bodies = int(MAX_BODIES)
    struct_dtype = np.dtype(
        {
            "names": ["bodies"],
            "formats": [(np.int32, max_bodies)],
            "offsets": [0],
            "itemsize": 4 * max_bodies,
        }
    )
    arr = np.zeros(n, dtype=struct_dtype)
    arr["bodies"] = bodies_np
    return wp.from_numpy(arr, dtype=ElementInteractionData, device=device)


class TestClusteringPipelineComposition(unittest.TestCase):
    """``ClusteringPipeline`` must produce the same outputs as the manual
    ``ConstraintClusterBuilder`` + ``SupernodalElements`` chain."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_pipeline_matches_manual_chain(self) -> None:
        device = wp.get_preferred_device()

        rng = np.random.default_rng(42)
        n_elements = 80
        n_bodies = 30
        rows = []
        max_bodies = int(MAX_BODIES)
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            rows.append(row)
        bodies_np = np.full((n_elements, max_bodies), -1, dtype=np.int32)
        for i, row in enumerate(rows):
            bodies_np[i, : len(row)] = row

        elements = _make_elements(bodies_np, device)
        num_elements = wp.array([n_elements], dtype=wp.int32, device=device)

        # Manual chain.
        cb = ConstraintClusterBuilder(
            max_num_interactions=n_elements,
            max_num_nodes=n_bodies,
            device=device,
            seed=0,
        )
        cb.build_clusters(elements, num_elements)
        se = SupernodalElements(max_num_clusters=n_elements, device=device)
        se.build(cb.cluster_members, cb.num_clusters, elements)
        wp.synchronize_device(device)
        manual_num = int(cb.num_clusters.numpy()[0])
        manual_members = cb.cluster_members.numpy().copy()
        manual_supernodal = se.elements.numpy().copy()
        manual_member_counts = se.member_counts.numpy().copy()

        # Pipeline.
        pipeline = ClusteringPipeline(
            max_num_interactions=n_elements,
            max_num_nodes=n_bodies,
            device=device,
            seed=0,
        )
        pipeline.build(elements, num_elements)
        wp.synchronize_device(device)

        self.assertEqual(int(pipeline.num_clusters.numpy()[0]), manual_num)
        self.assertTrue(np.array_equal(pipeline.cluster_members.numpy(), manual_members))
        self.assertTrue(np.array_equal(pipeline.supernodal_elements.numpy(), manual_supernodal))
        self.assertTrue(np.array_equal(pipeline.supernodal_member_counts.numpy(), manual_member_counts))


def _build_two_link_chain_model() -> newton.Model:
    """Minimal two-joint scene: ground -> link0 -> link1 via fixed +
    revolute joints. Enough constraint payload to exercise the
    clustering pipeline inside the captured step graph."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    a = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
    b = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.5, 1.0), q=wp.quat_identity()), label="b")
    builder.add_shape_box(a, hx=0.05, hy=0.05, hz=0.1)
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.1)
    j_fixed = builder.add_joint_fixed(
        parent=-1,
        child=a,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j_rev = builder.add_joint_revolute(
        parent=a,
        child=b,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.25, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, -0.25, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation([j_fixed, j_rev])
    return builder.finalize()


class TestSolverPhoenXClusteringFlag(unittest.TestCase):
    """End-to-end: ``SolverPhoenX(enable_clustering=True)`` must run
    captured steps without crashing and surface ``num_clusters`` in
    :meth:`step_report`."""

    @unittest.skipUnless(
        wp.get_preferred_device().is_cuda,
        "PhoenX clustering integration runs on CUDA only (graph-capture path).",
    )
    def test_flag_runs_step_and_surfaces_num_clusters(self) -> None:
        model = _build_two_link_chain_model()
        # ``step_layout="single_world"`` is currently required for the
        # cluster-aware dispatch path (initial scope).
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=2,
            solver_iterations=4,
            velocity_iterations=0,
            step_layout="single_world",
            enable_clustering=True,
        )

        s0 = model.state()
        s1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        sim_dt = 1.0 / 60.0
        # A handful of frames so we exercise both the warm-up capture
        # and the captured-graph replay path.
        for _ in range(3):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, sim_dt)
            s0, s1 = s1, s0

        # The clustering output lives on the wrapped PhoenXWorld
        # (SolverPhoenX.world). step_report() proxies through.
        world = solver.world
        self.assertTrue(world.enable_clustering)
        self.assertIsNotNone(world._clustering)

        report = world.step_report()
        self.assertIsNotNone(
            report.num_clusters,
            msg="step_report.num_clusters should be populated when enable_clustering=True",
        )
        # 1 fixed joint + 1 revolute = 2 constraints; clusters should be
        # 1..2 (both joints can fit in one cluster if they don't share
        # bodies, but here they share body ``a`` so they end up as two
        # singletons unless the body-cap allows the merge).
        self.assertGreaterEqual(int(report.num_clusters), 1)
        self.assertLessEqual(int(report.num_clusters), int(report.num_active_constraints))

        # In cluster-aware mode the main partitioner colours the
        # supernodal graph, so ``num_colors`` is the cluster-level
        # chromatic bound and must be sane (>= 1 with at least one
        # cluster, and not larger than the active cluster count).
        self.assertGreaterEqual(int(report.num_colors), 1)
        self.assertLessEqual(int(report.num_colors), int(report.num_clusters))


if __name__ == "__main__":
    wp.init()
    unittest.main()
