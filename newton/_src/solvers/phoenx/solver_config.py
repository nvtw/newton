# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Central knobs for the PhoenX solver."""

from __future__ import annotations

from typing import Literal

import warp as wp

#: Contact-matching mode for :class:`newton.CollisionPipeline`.
#: ``"sticky"`` (default) overwrites matched contacts' anchors with previous-frame
#: values to prevent geometry jitter; ``"latest"`` keeps this frame's geometry
#: (diagnostic). ``"disabled"`` is unsupported — the ingest path requires the match index.
PHOENX_CONTACT_MATCHING: Literal["sticky", "latest"] = "sticky"

#: Fixed-count inner unroll inside every ``wp.capture_while`` body. Amortises
#: graph-edge-traversal cost; safe because kernels early-exit once their
#: termination predicate hits zero.
NUM_INNER_WHILE_ITERATIONS: int = 8

#: Max colour size eligible for the single-block fused-tail path. Once a sweep's
#: remaining colours are all <= this, control hands off from the persistent
#: multi-block grid to a single 1D-block kernel that walks the rest with
#: ``__syncthreads`` between colours. ``0`` disables. Must be <= FUSE_TAIL_BLOCK_DIM.
FUSE_TAIL_MAX_COLOR_SIZE: int = 256

#: Block width of the fused-tail kernel. Must be >= :data:`FUSE_TAIL_MAX_COLOR_SIZE`.
FUSE_TAIL_BLOCK_DIM: int = 256

#: Use JP-MIS + greedy-colour partitioner instead of round-equals-colour JP.
#: 2-3x fewer colours on dense graphs; bounded at 64 colours total
#: (single-int64 forbidden mask). Falls back to descriptive error if exceeded.
PHOENX_USE_GREEDY_COLORING: bool = True

# Per-row Nyquist headroom multipliers. Strict implicit-Euler bound is
# k <= 1 / (M_inv * dt^2) (N=1); the constants let each PD row request N on top.
# Capped at _PD_NYQUIST_HEADROOM_MAX (10) inside pd_coefficients.

PHOENX_BOOST_REVOLUTE_DRIVE = wp.constant(wp.float32(10.0))
PHOENX_BOOST_REVOLUTE_LIMIT = wp.constant(wp.float32(10.0))
PHOENX_BOOST_PRISMATIC_DRIVE = wp.constant(wp.float32(10.0))
PHOENX_BOOST_PRISMATIC_LIMIT = wp.constant(wp.float32(10.0))
PHOENX_BOOST_CABLE_BEND = wp.constant(wp.float32(10.0))
PHOENX_BOOST_CABLE_TWIST = wp.constant(wp.float32(10.0))
#: Strict (N=1) for soft-contact normal PD row; contacts already self-limit
#: via effective_gap clamping. Box2D-style normal path uses its own omega cap.
PHOENX_BOOST_CONTACT_NORMAL = wp.constant(wp.float32(1.0))

#: Slip velocity at which the Coulomb-friction row saturates [m/s for
#: prismatic, rad/s for revolute]. PhoenX's friction row is a saturated
#: soft constraint with regularization ``gamma`` chosen so that the
#: equivalent slip at the saturation impulse equals this constant:
#: ``gamma = PHOENX_FRICTION_SLIP_VELOCITY / (μ * dt)``. A smaller slip
#: velocity yields sharper Coulomb behaviour (closer to a hard stick);
#: a larger value smooths the transition into viscous-blended friction.
#: ``1 mm/s`` (or ``1 mrad/s``) is small enough to be invisible in robot
#: control loops but large enough to keep the regularization well above
#: float32 noise at typical dt values. Matches the role of MuJoCo's
#: ``solref`` regularization on ``dof_frictionloss``.
PHOENX_FRICTION_SLIP_VELOCITY = wp.constant(wp.float32(1.0e-3))


__all__ = [
    "FUSE_TAIL_BLOCK_DIM",
    "FUSE_TAIL_MAX_COLOR_SIZE",
    "NUM_INNER_WHILE_ITERATIONS",
    "PHOENX_BOOST_CABLE_BEND",
    "PHOENX_BOOST_CABLE_TWIST",
    "PHOENX_BOOST_CONTACT_NORMAL",
    "PHOENX_BOOST_PRISMATIC_DRIVE",
    "PHOENX_BOOST_PRISMATIC_LIMIT",
    "PHOENX_BOOST_REVOLUTE_DRIVE",
    "PHOENX_BOOST_REVOLUTE_LIMIT",
    "PHOENX_CONTACT_MATCHING",
    "PHOENX_FRICTION_SLIP_VELOCITY",
    "PHOENX_USE_GREEDY_COLORING",
]
