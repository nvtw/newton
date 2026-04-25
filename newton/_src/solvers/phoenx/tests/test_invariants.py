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

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import CONTACT_DWORDS
from newton._src.solvers.phoenx.constraints.constraint_container import (
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _make_kwargs(num_bodies: int = 1, num_joints: int = 0, rigid_contact_max: int = 0) -> dict:
    """Build a minimal (bodies, constraints, ...) kwarg dict."""
    device = wp.get_device()
    bodies = body_container_zeros(num_bodies, device=device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=num_joints, device=device)
    return {
        "bodies": bodies,
        "constraints": constraints,
        "num_joints": num_joints,
        "rigid_contact_max": rigid_contact_max,
        "device": device,
    }


class TestInvariants(unittest.TestCase):
    """``_assert_invariants`` traps four classes of caller mistake."""

    def test_correct_construction_accepted(self) -> None:
        """A correctly-built world constructs without raising."""
        PhoenXWorld(**_make_kwargs(num_bodies=2, num_joints=0, rigid_contact_max=10))

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
        with self.assertRaisesRegex(AssertionError, r"ConstraintContainer\.data has shape \(7, 4200\)"):
            PhoenXWorld(**kw)


if __name__ == "__main__":
    unittest.main()
