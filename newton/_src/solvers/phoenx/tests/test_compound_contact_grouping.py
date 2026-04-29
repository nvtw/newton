# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compound-body contact grouping regression.

Builds a body with multiple collision shapes (a small box cluster) and
a static multi-shape platform. Without the grouping optimisation each
``(cluster_shape, platform_shape)`` pair becomes its own contact
column sharing the same body pair, forcing the graph colourer to
spend up to ``cluster_shapes * platform_shapes`` colours on what is
physically one body-body interaction.

Tests:

1. Solver opts the scene into body-pair grouping (compound detection
   at construction).
2. Single-shape scenes opt out (zero-overhead path).
3. Stepping the scene a few frames produces no NaN / divergence.
4. The colour count after settling is at most as low as the
   non-compound equivalent run with one merged shape per body.

Runs on CUDA only -- the PhoenX path is GPU-only by design.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _build_compound_scene(
    *,
    body_a_shapes: int = 4,
    body_b_shapes: int = 4,
):
    """Two bodies stacked above the ground; each body has multiple
    box shapes laid out side-by-side along its local +x axis. The
    bottom body's bottom shapes touch the floor on the first frame;
    the top body's bottom shapes touch the bottom body's top.
    """
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)

    half_extent = 0.05
    spacing = 2.05 * half_extent
    box_cfg = mb.ShapeConfig(density=1000.0)

    # Static ground.
    mb.add_ground_plane()

    # Body B: closer to the ground.
    body_b = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-2, 0, 0), (0, 1.0e-2, 0), (0, 0, 1.0e-2)),
    )
    j_b = mb.add_joint_free(parent=-1, child=body_b)
    for j in range(body_b_shapes):
        offset_x = (j - 0.5 * (body_b_shapes - 1)) * spacing
        mb.add_shape_box(
            body_b,
            xform=wp.transform(p=wp.vec3(offset_x, 0.0, 0.0), q=wp.quat_identity()),
            hx=half_extent,
            hy=half_extent,
            hz=half_extent,
            cfg=box_cfg,
        )

    # Body A: above body B.
    body_a = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.20), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-2, 0, 0), (0, 1.0e-2, 0), (0, 0, 1.0e-2)),
    )
    j_a = mb.add_joint_free(parent=-1, child=body_a)
    for i in range(body_a_shapes):
        offset_x = (i - 0.5 * (body_a_shapes - 1)) * spacing
        mb.add_shape_box(
            body_a,
            xform=wp.transform(p=wp.vec3(offset_x, 0.0, 0.0), q=wp.quat_identity()),
            hx=half_extent,
            hy=half_extent,
            hz=half_extent,
            cfg=box_cfg,
        )

    mb.add_articulation([j_b])
    mb.add_articulation([j_a])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _build_single_shape_scene():
    """Mirror of :func:`_build_compound_scene` but with one big box per
    body (single-shape baseline)."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)

    half_extent = 0.05
    box_cfg = mb.ShapeConfig(density=1000.0)
    mb.add_ground_plane()

    body_b = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-2, 0, 0), (0, 1.0e-2, 0), (0, 0, 1.0e-2)),
    )
    j_b = mb.add_joint_free(parent=-1, child=body_b)
    mb.add_shape_box(body_b, hx=4 * half_extent, hy=half_extent, hz=half_extent, cfg=box_cfg)

    body_a = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.20), q=wp.quat_identity()),
        mass=1.0,
        inertia=((1.0e-2, 0, 0), (0, 1.0e-2, 0), (0, 0, 1.0e-2)),
    )
    j_a = mb.add_joint_free(parent=-1, child=body_a)
    mb.add_shape_box(body_a, hx=4 * half_extent, hy=half_extent, hz=half_extent, cfg=box_cfg)

    mb.add_articulation([j_b])
    mb.add_articulation([j_a])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _make_solver(model):
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=8,
        velocity_iterations=1,
    )


def _step_n(model, solver, n_frames: int, dt: float) -> None:
    """Advance ``n_frames`` steps eagerly (test runs short, capture
    overhead not worth it). Returns final body_q for inspection."""
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = model.contacts() if model.shape_count > 0 else None
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(n_frames):
        s0.clear_forces()
        if contacts is not None:
            model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    return s0


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX compound-grouping test requires CUDA.",
)
class TestCompoundContactGrouping(unittest.TestCase):
    """Compound-body contact grouping correctness + colour-count check."""

    def test_compound_scene_opts_in(self) -> None:
        """Multi-shape body layout must trigger compound detection in
        ``SolverPhoenX``, which routes ingest through the body-pair
        grouping path."""
        model = _build_compound_scene()
        solver = _make_solver(model)
        self.assertTrue(
            solver.world._enable_body_pair_grouping,
            "compound-body scene should opt into body-pair grouping",
        )
        self.assertIsNotNone(
            solver.world._ingest_scratch.body_pair_keys,
            "ingest scratch should have allocated body-pair sort buffers",
        )

    def test_single_shape_scene_opts_out(self) -> None:
        """One shape per body keeps compound detection off; the
        scratch arrays for sorting are not allocated (zero overhead)."""
        model = _build_single_shape_scene()
        solver = _make_solver(model)
        self.assertFalse(
            solver.world._enable_body_pair_grouping,
            "single-shape scene should not opt into body-pair grouping",
        )
        self.assertIsNone(
            solver.world._ingest_scratch.body_pair_keys,
            "single-shape scratch should not allocate body-pair sort buffers",
        )

    def test_compound_scene_steps_without_nan(self) -> None:
        """Step the compound scene for ~0.5 s; assert finite poses and
        no body has escaped the simulation envelope. Catches gather /
        sort / inverse-perm bugs that would corrupt warm-start state."""
        model = _build_compound_scene()
        solver = _make_solver(model)
        s = _step_n(model, solver, n_frames=120, dt=1.0 / 240.0)
        body_q = s.body_q.numpy()
        self.assertTrue(np.isfinite(body_q).all(), msg="non-finite body_q after compound step")
        positions = body_q[:, :3]
        max_xy = float(np.max(np.abs(positions[:, :2])))
        self.assertLess(
            max_xy,
            5.0,
            msg=f"compound bodies escaped envelope: max |xy| = {max_xy:.3f} m",
        )

    def test_compound_color_count_at_or_below_baseline(self) -> None:
        """Grouped colouring should not exceed the single-shape
        baseline's colour count (with a small slop allowance for
        greedy tie-breaking). The naive (no-grouping) compound
        colouring would need ``shape_pairs ~= 4 * 4 + 4 + 4 = 24``
        colours; the grouped path collapses to the body-pair lower
        bound (a handful of colours)."""
        compound_model = _build_compound_scene()
        single_model = _build_single_shape_scene()
        compound_solver = _make_solver(compound_model)
        single_solver = _make_solver(single_model)

        _step_n(compound_model, compound_solver, n_frames=60, dt=1.0 / 240.0)
        _step_n(single_model, single_solver, n_frames=60, dt=1.0 / 240.0)

        compound_colors = int(compound_solver.world.step_report().num_colors)
        single_colors = int(single_solver.world.step_report().num_colors)
        # Allow +1 slop: greedy colouring may pick a slightly higher
        # colour for a tie-break case.
        self.assertLessEqual(
            compound_colors,
            single_colors + 1,
            msg=(
                f"grouped colour count {compound_colors} > single-shape "
                f"baseline {single_colors} + 1; body-pair grouping should "
                "match (or beat) the body-pair graph chromatic number."
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
