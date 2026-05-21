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

import math
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


def _build_dense_joint_model(num_spokes: int = 6) -> newton.Model:
    """Build a hub-and-spoke articulation: one fixed-to-world hub body
    plus ``num_spokes`` peripheral bodies connected to the hub by
    revolute joints. Every revolute joint shares the hub body, so the
    constraint graph is a clique-on-hub -- the kind of dense pattern
    clustering is designed to compress (each K=4 cluster absorbs up
    to 3 spoke joints + still respects the 8-body union cap because
    every joint touches only ``{hub, spoke}``).

    Resulting constraint count: ``1 (fixed) + num_spokes (revolute)``.
    Body union per joint: 2 (hub + one spoke).
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=-1.0)
    hub_z = 2.0

    hub = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.0, hub_z), q=wp.quat_identity()), label="hub")
    builder.add_shape_box(hub, hx=0.2, hy=0.2, hz=0.2)
    j_fixed = builder.add_joint_fixed(
        parent=-1,
        child=hub,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, hub_z), q=wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )

    joints = [j_fixed]
    spoke_radius = 0.5
    for i in range(num_spokes):
        angle = (2.0 * math.pi * i) / max(1, num_spokes)
        sx = spoke_radius * math.cos(angle)
        sy = spoke_radius * math.sin(angle)
        spoke = builder.add_link(
            xform=wp.transform(p=wp.vec3(sx, sy, hub_z), q=wp.quat_identity()),
            label=f"spoke_{i}",
        )
        builder.add_shape_box(spoke, hx=0.1, hy=0.05, hz=0.05)
        j_rev = builder.add_joint_revolute(
            parent=hub,
            child=spoke,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(sx * 0.5, sy * 0.5, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-sx * 0.5, -sy * 0.5, 0.0), q=wp.quat_identity()),
        )
        joints.append(j_rev)

    builder.add_articulation(joints)
    return builder.finalize()


def _run_joint_scene_capture(
    model: newton.Model,
    *,
    enable_clustering: bool,
    frames: int,
) -> dict:
    """Run a small captured-graph joint sim. Returns final state
    summary + step_report metrics from the last frame."""
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=2,
        solver_iterations=4,
        velocity_iterations=0,
        step_layout="single_world",
        mass_splitting=False,  # gated out for the cluster-aware path
        enable_clustering=enable_clustering,
    )

    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    device = wp.get_device()
    fps = 60
    dt = 1.0 / fps

    def _pair():
        for s_in, s_out in ((s0, s1), (s1, s0)):
            s_in.clear_forces()
            model.collide(s_in, contacts)
            solver.step(s_in, s_out, control, contacts, dt)

    # Warm-up two frames before capturing the graph (absorbs JIT +
    # lazy allocations so the captured replay is steady-state).
    _pair()

    with wp.ScopedCapture(device=device) as capture:
        _pair()
    graph = capture.graph

    # Already advanced 4 frames (2 warm + 2 captured). Replay the rest
    # through the captured graph so the cluster pipeline runs inside
    # graph capture, not eager.
    advanced = 4
    remaining = max(0, frames - advanced)
    for _ in range(remaining // 2):
        wp.capture_launch(graph)
    for _ in range(remaining % 2):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0

    wp.synchronize_device(device)

    report = solver.world.step_report()
    body_q = s0.body_q.numpy().copy()
    return {
        "solver": solver,
        "body_q": body_q,
        "num_clusters": report.num_clusters,
        "num_colors": int(report.num_colors),
        "num_active_constraints": int(report.num_active_constraints),
        "max_body_degree": int(report.max_body_degree),
    }


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX clustering correctness tests run on CUDA only (graph-capture path).",
)
class TestClusteringJointScenePhysics(unittest.TestCase):
    """End-to-end validation: a captured-graph hub-and-spoke joint sim
    with ``enable_clustering=True`` must produce sane physics, a
    measurable clustering compaction, determinism across runs, and
    end-state agreement with the clustering-off baseline."""

    def test_joint_scene_with_clustering_runs_and_compacts(self) -> None:
        model = _build_dense_joint_model(num_spokes=6)
        result = _run_joint_scene_capture(model, enable_clustering=True, frames=8)

        # Cluster output must be populated when clustering is on.
        self.assertIsNotNone(result["num_clusters"])
        n_clusters = int(result["num_clusters"])
        n_active = result["num_active_constraints"]
        self.assertGreater(
            n_clusters,
            0,
            msg=(
                f"cluster builder produced 0 clusters; "
                f"num_active_constraints={n_active}, "
                f"num_colors={result['num_colors']}, "
                f"max_body_degree={result['max_body_degree']}"
            ),
        )
        # Clustering must actually compact -- the hub-and-spoke topology
        # has every revolute joint sharing the hub body, so the cluster
        # builder should absorb groups of joints into K=4 supernodes.
        # 1 fixed + 6 revolute = 7 active constraints; clusters should
        # be strictly fewer.
        self.assertLess(
            n_clusters,
            n_active,
            msg=(
                f"clustering produced {n_clusters} clusters from "
                f"{n_active} active constraints -- expected at least "
                "some compaction on a hub-and-spoke scene"
            ),
        )

        # In cluster-aware mode, ``num_colors`` is the supernodal
        # chromatic bound. It must be sane and (by construction)
        # bounded by the cluster count.
        self.assertGreaterEqual(result["num_colors"], 1)
        self.assertLessEqual(result["num_colors"], n_clusters)

        # Body poses must remain finite.
        q = result["body_q"]
        self.assertTrue(
            np.all(np.isfinite(q)),
            msg="non-finite body position after captured-graph cluster-aware sim",
        )
        # No body should have escaped to infinity. The hub is anchored
        # at z=2; spokes orbit at radius ~0.5; ground plane at z=-1.
        # A loose bounding box catches divergent solves without flaking
        # on transient overshoot.
        for axis, lo, hi in (("x", -3.0, 3.0), ("y", -3.0, 3.0), ("z", -2.0, 5.0)):
            vals = q[:, "xyz".index(axis)]
            self.assertGreater(float(vals.min()), lo, msg=f"body escaped past {axis} >= {lo}")
            self.assertLess(float(vals.max()), hi, msg=f"body escaped past {axis} <= {hi}")

    def test_joint_scene_clustering_determinism(self) -> None:
        """Two independent captured-graph runs with the same model and
        flag must produce bit-identical body states."""
        model = _build_dense_joint_model(num_spokes=6)
        r1 = _run_joint_scene_capture(model, enable_clustering=True, frames=6)
        model2 = _build_dense_joint_model(num_spokes=6)
        r2 = _run_joint_scene_capture(model2, enable_clustering=True, frames=6)

        self.assertEqual(r1["num_clusters"], r2["num_clusters"])
        self.assertEqual(r1["num_colors"], r2["num_colors"])
        self.assertTrue(
            np.array_equal(r1["body_q"], r2["body_q"]),
            msg=(
                "body positions differ between two cluster-aware runs of the same "
                f"scene; max abs diff = "
                f"{float(np.max(np.abs(r1['body_q'] - r2['body_q']))):.6e}"
            ),
        )

    def test_joint_scene_clustering_vs_baseline_equivalence(self) -> None:
        """Clustering-on and clustering-off must both produce stable
        physics on the same scene. Trajectories diverge slightly
        because the colour ordering differs, but end states under
        the joint constraints should stay close -- joints are hard
        positional constraints, not soft penalties, so the converged
        pose is colour-order-invariant within the PGS slack."""
        model_off = _build_dense_joint_model(num_spokes=6)
        r_off = _run_joint_scene_capture(model_off, enable_clustering=False, frames=8)
        model_on = _build_dense_joint_model(num_spokes=6)
        r_on = _run_joint_scene_capture(model_on, enable_clustering=True, frames=8)

        # Both paths must produce finite states.
        for label, q in (("baseline", r_off["body_q"]), ("clustered", r_on["body_q"])):
            self.assertTrue(np.all(np.isfinite(q)), msg=f"{label}: non-finite body position")

        # Per-body position drift between the two paths must stay
        # small. Joints are hard constraints, so the steady-state
        # geometry must agree closely. 5 cm tolerance is generous for
        # the PGS noise.
        max_drift = float(np.max(np.linalg.norm(r_off["body_q"][:, :3] - r_on["body_q"][:, :3], axis=1)))
        self.assertLess(
            max_drift,
            0.05,
            msg=(
                f"clustered vs baseline max body drift = {max_drift:.4f} m "
                "(expected < 0.05). One of the paths is diverging."
            ),
        )

        # The clustering path must report a real cluster count.
        self.assertIsNotNone(r_on["num_clusters"])
        self.assertGreater(int(r_on["num_clusters"]), 0)
        # And the baseline must report ``num_clusters=None`` since the
        # clustering pipeline is not allocated.
        self.assertIsNone(r_off["num_clusters"])


if __name__ == "__main__":
    wp.init()
    unittest.main()
