# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construction-time safety checks for :class:`PhoenXWorld`.

The solver runs a small ``_assert_invariants`` pass at the end of
``__init__`` that validates every caller-supplied container has the
shape the kernels expect. The audit that prompted this safety net
turned up four bypass-the-factory call sites that allocated a
constraint container with the wrong shape and silently produced
out-of-range reads in the per-step kernels; this test asserts that
the invariants actually fire when a mis-shaped container is passed.
"""

from __future__ import annotations

import unittest

import warp as wp

if not wp.get_preferred_device().is_cuda:
    raise unittest.SkipTest("PhoenX tests require CUDA")

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import CONTACT_DWORDS
from newton._src.solvers.phoenx.constraints.constraint_container import (
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import MAX_BODIES
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode


def _make_kwargs(
    num_bodies: int = 1,
    num_joints: int = 0,
    rigid_contact_max: int = 0,
    max_contact_columns: int | None = None,
    num_particles: int = 0,
    num_cloth_triangles: int = 0,
    num_cloth_bending: int = 0,
    num_soft_tetrahedra: int = 0,
    num_soft_hexahedra: int = 0,
) -> dict:
    """Build a minimal (bodies, constraints, ...) kwarg dict."""
    device = wp.get_device()
    bodies = body_container_zeros(num_bodies, device=device)
    constraints = PhoenXWorld.make_constraint_container(
        num_joints=num_joints,
        device=device,
        num_cloth_triangles=num_cloth_triangles,
        num_soft_tetrahedra=num_soft_tetrahedra,
        num_cloth_bending=num_cloth_bending,
        num_soft_hexahedra=num_soft_hexahedra,
    )
    return {
        "bodies": bodies,
        "constraints": constraints,
        "num_joints": num_joints,
        "rigid_contact_max": rigid_contact_max,
        "max_contact_columns": max_contact_columns,
        "num_particles": num_particles,
        "num_cloth_triangles": num_cloth_triangles,
        "num_cloth_bending": num_cloth_bending,
        "num_soft_tetrahedra": num_soft_tetrahedra,
        "num_soft_hexahedra": num_soft_hexahedra,
        "device": device,
    }


class TestInvariants(unittest.TestCase):
    """``_assert_invariants`` traps four classes of caller mistake."""

    def test_correct_construction_accepted(self) -> None:
        """A correctly-built world constructs without raising."""
        PhoenXWorld(**_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=10))

    def test_contact_columns_can_be_sized_independently(self) -> None:
        """Per-column storage can use a tighter pair budget than per-contact state."""
        w = PhoenXWorld(**_make_kwargs(num_bodies=2, num_joints=1, rigid_contact_max=128, max_contact_columns=12))
        self.assertEqual(w.rigid_contact_max, 128)
        self.assertEqual(w.max_contact_columns, 12)
        self.assertEqual(w._contact_container.lambdas.shape[1], 128)
        self.assertEqual(w._contact_cols.data.shape[1], 12)
        self.assertEqual(w._constraint_capacity, w.num_joints + 12)

    def test_invalid_contact_column_capacity_raises(self) -> None:
        """Column-capacity mistakes should fail at construction."""
        with self.assertRaisesRegex(ValueError, "max_contact_columns must be >= 1"):
            PhoenXWorld(**_make_kwargs(num_bodies=2, rigid_contact_max=10, max_contact_columns=0))
        with self.assertRaisesRegex(ValueError, "max_contact_columns must be None or 0"):
            PhoenXWorld(**_make_kwargs(num_bodies=2, rigid_contact_max=0, max_contact_columns=1))

    def test_undersized_constraint_container_raises(self) -> None:
        """Allocating ``ConstraintContainer`` directly with a mismatched
        column count -- the bypass-the-factory mistake the audit
        caught -- must fail loudly."""
        kw = _make_kwargs(num_bodies=1, num_joints=2)
        # ``num_joints=2`` -> factory would emit (ADBS_DWORDS, 2). Pass
        # a too-small container instead.
        kw["constraints"] = constraint_container_zeros(
            num_constraints=1, num_dwords=CONTACT_DWORDS, device=wp.get_device()
        )
        with self.assertRaisesRegex(AssertionError, r"ConstraintContainer\.data has shape"):
            PhoenXWorld(**kw)

    def test_oversized_constraint_container_raises(self) -> None:
        """The exact bug the audit found: a no-joints scene allocates
        a ``rigid_contact_max``-sized constraint container instead of
        the correct 1-row placeholder."""
        kw = _make_kwargs(num_bodies=1, num_joints=0, rigid_contact_max=4200)
        # Mimic the pre-fix _PhoenXScene allocation.
        kw["constraints"] = constraint_container_zeros(
            num_constraints=4200, num_dwords=CONTACT_DWORDS, device=wp.get_device()
        )
        with self.assertRaisesRegex(
            AssertionError, rf"ConstraintContainer\.data has shape \({int(CONTACT_DWORDS)}, 4200\)"
        ):
            PhoenXWorld(**kw)


class TestPrepareRefreshStride(unittest.TestCase):
    """Construction-time checks for cached prepare-data refresh."""

    def test_single_world_stride_schedule(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            step_layout="single_world",
            prepare_refresh_stride=2,
        )
        self.assertEqual(w.prepare_refresh_stride, 2)
        w._current_substep_index = 0
        self.assertTrue(w._refresh_prepare_this_substep())
        w._current_substep_index = 1
        self.assertFalse(w._refresh_prepare_this_substep())
        w._current_substep_index = 2
        self.assertTrue(w._refresh_prepare_this_substep())

    def test_rejects_non_positive_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "prepare_refresh_stride must be >= 1"):
            PhoenXWorld(**_make_kwargs(num_bodies=2, rigid_contact_max=1), prepare_refresh_stride=0)

    def test_default_stride_uses_auto_policy(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            substeps=8,
        )
        self.assertEqual(w.prepare_refresh_stride, 3)
        self.assertEqual(w._prepare_refresh_stride_policy, "auto")

    def test_auto_stride_uses_substep_count(self) -> None:
        low = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            substeps=3,
            prepare_refresh_stride="auto",
        )
        moderate = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            substeps=4,
            prepare_refresh_stride="auto",
        )
        high = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            substeps=8,
            prepare_refresh_stride="auto",
        )
        self.assertEqual(low.prepare_refresh_stride, 1)
        self.assertEqual(moderate.prepare_refresh_stride, 1)
        self.assertEqual(high.prepare_refresh_stride, 3)
        self.assertEqual(high._prepare_refresh_stride_policy, "auto")

    def test_auto_stride_falls_back_when_unsupported(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, rigid_contact_max=1, num_particles=4, num_cloth_triangles=1),
            substeps=20,
            prepare_refresh_stride="auto",
        )
        self.assertEqual(w.prepare_refresh_stride, 1)

    def test_rejects_unknown_stride_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "prepare_refresh_stride"):
            PhoenXWorld(**_make_kwargs(num_bodies=2, rigid_contact_max=1), prepare_refresh_stride="sometimes")

    def test_accepts_contact_stride_three(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            step_layout="single_world",
            prepare_refresh_stride=3,
        )
        self.assertEqual(w.prepare_refresh_stride, 3)
        for substep, expected in ((0, True), (1, False), (2, False), (3, True)):
            with self.subTest(substep=substep):
                w._current_substep_index = substep
                self.assertEqual(w._refresh_prepare_this_substep(), expected)

    def test_rejects_contact_stride_above_three(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "contact worlds"):
            PhoenXWorld(
                **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
                step_layout="single_world",
                prepare_refresh_stride=4,
            )

    def test_accepts_joint_only_stride_four(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=1, rigid_contact_max=0),
            step_layout="single_world",
            prepare_refresh_stride=4,
        )
        self.assertEqual(w.prepare_refresh_stride, 4)
        for substep, expected in ((0, True), (1, False), (2, False), (3, False), (4, True)):
            with self.subTest(substep=substep):
                w._current_substep_index = substep
                self.assertEqual(w._refresh_prepare_this_substep(), expected)

    def test_multi_world_stride_schedule(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
            step_layout="multi_world",
            prepare_refresh_stride=2,
        )
        self.assertEqual(w.prepare_refresh_stride, 2)
        w._current_substep_index = 0
        self.assertTrue(w._refresh_prepare_this_substep())
        w._current_substep_index = 1
        self.assertFalse(w._refresh_prepare_this_substep())

    def test_rejects_deformable_stride(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "rigid contact/joint worlds"):
            PhoenXWorld(
                **_make_kwargs(num_bodies=2, rigid_contact_max=1, num_particles=4, num_cloth_triangles=1),
                step_layout="single_world",
                prepare_refresh_stride=2,
            )

    def test_accepts_non_revolute_joint_stride(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=1, rigid_contact_max=1),
            step_layout="single_world",
            prepare_refresh_stride=2,
        )
        device = wp.get_device()

        def _f(v: float) -> wp.array:
            return wp.array([v], dtype=wp.float32, device=device)

        def _i(v: int) -> wp.array:
            return wp.array([v], dtype=wp.int32, device=device)

        def _v(v: tuple[float, float, float]) -> wp.array:
            return wp.array([v], dtype=wp.vec3f, device=device)

        w.initialize_actuated_double_ball_socket_joints(
            body1=_i(0),
            body2=_i(1),
            anchor1=_v((0.0, 0.0, 0.0)),
            anchor2=_v((1.0, 0.0, 0.0)),
            hertz=_f(60.0),
            damping_ratio=_f(1.0),
            joint_mode=_i(int(JointMode.PRISMATIC)),
            drive_mode=_i(int(DriveMode.OFF)),
            target=_f(0.0),
            target_velocity=_f(0.0),
            max_force_drive=_f(0.0),
            stiffness_drive=_f(0.0),
            damping_drive=_f(0.0),
            min_value=_f(1.0),
            max_value=_f(-1.0),
            hertz_limit=_f(60.0),
            damping_ratio_limit=_f(1.0),
            stiffness_limit=_f(0.0),
            damping_limit=_f(0.0),
        )
        self.assertFalse(w._use_revolute_specialization)

    def test_rejects_mass_splitting_stride(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "deformables, mass splitting, or sleeping"):
            PhoenXWorld(
                **_make_kwargs(num_bodies=2, rigid_contact_max=1),
                step_layout="single_world",
                mass_splitting=True,
                prepare_refresh_stride=2,
            )

    def test_rejects_sleeping_stride(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "deformables, mass splitting, or sleeping"):
            PhoenXWorld(
                **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
                step_layout="single_world",
                sleeping_velocity_threshold=0.1,
                prepare_refresh_stride=2,
            )


class TestMassSplittingConfig(unittest.TestCase):
    """Mass-splitting config plumbs through to the partitioner and
    allocates the copy-state / interaction-graph scratch."""

    def test_disabled_by_default_uses_sentinel_containers(self) -> None:
        w = PhoenXWorld(**_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=10))
        self.assertFalse(w.mass_splitting_enabled)
        self.assertIsNone(w.max_colored_partitions)
        # Sentinel allocations: capacity-1, num_nodes-1. highest_index_in_use
        # stays at zero so the broadcast / average / writeback kernels
        # short-circuit when called.
        self.assertEqual(w._copy_state.position.shape[0], 1)
        self.assertEqual(w._copy_state.section_end.shape[0], 1)
        self.assertEqual(int(w._copy_state.highest_index_in_use.numpy()[0]), 0)

    def test_enabled_allocates_capacity_sized_buffers(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=4, num_joints=2, rigid_contact_max=128),
            mass_splitting=True,
            max_colored_partitions=12,
            step_layout="single_world",
        )
        self.assertTrue(w.mass_splitting_enabled)
        self.assertEqual(w.max_colored_partitions, 12)
        # Rigid-only worlds only emit the two body endpoints per
        # contact/joint, so copy-state scratch should not be inflated
        # to the deformable-contact upper bound.
        expected_capacity = 2 * 2 + 128 * 2
        self.assertEqual(w._copy_state.position.shape[0], expected_capacity)
        self.assertEqual(w._copy_state.section_end.shape[0], w.num_bodies + w.num_particles)
        self.assertEqual(int(w._copy_state.highest_index_in_use.numpy()[0]), 0)

    def test_enabled_uses_wide_contact_capacity_with_particles(self) -> None:
        w = PhoenXWorld(
            **_make_kwargs(
                num_bodies=4,
                num_joints=1,
                rigid_contact_max=7,
                num_particles=16,
                num_cloth_triangles=2,
                num_cloth_bending=3,
                num_soft_tetrahedra=5,
                num_soft_hexahedra=1,
            ),
            mass_splitting=True,
            max_colored_partitions=12,
            step_layout="single_world",
        )
        max_bodies = int(MAX_BODIES)
        expected_capacity = 1 * 2 + 2 * 3 + 3 * 4 + 5 * 4 + 1 * max_bodies + 7 * max_bodies
        self.assertEqual(w._copy_state.position.shape[0], expected_capacity)
        self.assertEqual(w._copy_state.section_end.shape[0], w.num_bodies + w.num_particles)
        self.assertEqual(int(w._copy_state.highest_index_in_use.numpy()[0]), 0)

    def test_rejects_overflow_cap_above_greedy_limit(self) -> None:
        # The partitioner raises if max_colored_partitions exceeds the
        # int64 forbidden-mask budget.
        with self.assertRaises(ValueError):
            PhoenXWorld(
                **_make_kwargs(num_bodies=1, num_joints=0, rigid_contact_max=1),
                mass_splitting=True,
                max_colored_partitions=64,
                step_layout="single_world",
            )

    def test_accepts_mass_splitting_with_joints(self) -> None:
        # Joint constraint kernels now route through the slot-aware
        # helpers, so ``mass_splitting=True`` no longer rejects
        # ``num_joints > 0``.
        w = PhoenXWorld(
            **_make_kwargs(num_bodies=2, num_joints=1, rigid_contact_max=1),
            mass_splitting=True,
            step_layout="single_world",
        )
        self.assertTrue(w.mass_splitting_enabled)

    def test_rejects_mass_splitting_with_multi_world_layout(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "step_layout='single_world'"):
            PhoenXWorld(
                **_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=1),
                mass_splitting=True,
                step_layout="multi_world",
            )


if __name__ == "__main__":
    unittest.main()
