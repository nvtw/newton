# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Central knobs for the PhoenX solver.

All PhoenX examples, tests, and internal solver code read their
tuning constants from this module.

Knobs
-----

``PHOENX_CONTACT_MATCHING`` (``Literal["sticky", "latest"]``)
    Contact-matching mode for :class:`newton.CollisionPipeline`.
    Both modes populate ``Contacts.rigid_contact_match_index`` (what
    the warm-start gather consumes); they differ in what happens to
    matched contacts' geometry:

    * ``"sticky"`` (required for good stacking): after matching, the
      anchors (``point0/1``, ``offset0/1``) and world-frame ``normal``
      are overwritten with the previous-frame values. No
      frame-to-frame geometry jitter leaks into the Jacobian.
    * ``"latest"`` (fallback / diagnostic): match index still
      populated but this frame's narrow-phase geometry is kept. Used
      to isolate whether a regression is caused by the anchor pinning.

    Do NOT set to ``"disabled"`` -- the ingest path raises without
    a match index.

``NUM_INNER_WHILE_ITERATIONS`` (``int``)
    Fixed-count inner unroll inside every ``wp.capture_while`` body.
    Each outer iteration pays a graph-edge-traversal cost that
    dominates once the body is short (per-colour PGS sweep, a
    tail JP round); bundling N copies amortises the cost by N.
    Safe because the kernels early-exit once their termination
    predicate (``color_cursor``, ``num_remaining``) hits zero --
    tail iterations are no-ops. Shipped default ``8``.

``FUSE_TAIL_MAX_COLOR_SIZE`` (``int``)
    Max colour size (in cids) eligible for the single-block fused-
    tail path of the single-world PGS sweep. Once a sweep's
    remaining colours are all <= this value, control hands off from
    the persistent multi-block grid to a single 1D-block fused
    kernel that walks the rest back-to-back using ``__syncthreads``
    between colours instead of kernel boundaries; saves
    O(remaining-colours) kernel launches per sweep.

    Correctness: a "small" colour fits inside one block of width
    :data:`FUSE_TAIL_BLOCK_DIM`, so each cid owns a distinct lane
    and ``__syncthreads`` orders writes between colours. The
    persistent-grid kernels hand off by early-exit (without
    decrementing ``color_cursor``) when the current colour's size is
    already <= this threshold.

    ``0`` disables the path. Default :data:`FUSE_TAIL_BLOCK_DIM`
    (256); parity asserted to 1e-6 by
    :mod:`newton._src.solvers.phoenx.tests.test_tail_fuse`.

``FUSE_TAIL_BLOCK_DIM`` (``int``)
    Block width of the fused tail kernel. Must be
    >= :data:`FUSE_TAIL_MAX_COLOR_SIZE`.

Usage::

    from newton._src.solvers.phoenx.solver_config import (
        FUSE_TAIL_BLOCK_DIM,
        FUSE_TAIL_MAX_COLOR_SIZE,
        NUM_INNER_WHILE_ITERATIONS,
        PHOENX_CONTACT_MATCHING,
    )

    pipeline = newton.CollisionPipeline(model, contact_matching=PHOENX_CONTACT_MATCHING)
"""

from __future__ import annotations

from typing import Literal

import warp as wp

#: Contact-matching mode fed into :class:`newton.CollisionPipeline`.
#: See the module docstring; ``"sticky"`` is the production default.
PHOENX_CONTACT_MATCHING: Literal["sticky", "latest"] = "sticky"

#: Unroll count for every ``wp.capture_while`` body in PhoenX. See
#: the module docstring for the early-exit contract.
NUM_INNER_WHILE_ITERATIONS: int = 8

#: Max colour size (in cids) eligible for the single-block fused-tail
#: path of the single-world PGS sweep. ``0`` disables; default
#: :data:`FUSE_TAIL_BLOCK_DIM`. See the module docstring for the
#: hand-off mechanism and correctness contract.
FUSE_TAIL_MAX_COLOR_SIZE: int = 256

#: Block width of the fused-tail kernel.
#: Must be >= :data:`FUSE_TAIL_MAX_COLOR_SIZE`.
FUSE_TAIL_BLOCK_DIM: int = 256

#: Use the JP-MIS + greedy-colour partitioner
#: (:meth:`IncrementalContactPartitioner.build_csr_greedy`) instead of
#: the round-equals-colour JP path. Greedy gives 2-3x fewer colours on
#: dense contact graphs (Kapla tower: 78 → 28 = lower bound) at
#: comparable build time, in exchange for a max-colour-size that's
#: typically ~2x larger -- the colour sweep parallelism is already
#: GPU-saturated on big colours, so the trade favours fewer rounds.
#: Bounded at 64 colours total (single-int64 forbidden mask); falls
#: back to a descriptive error if a graph wants more, in which case
#: flip this off to recover the round-based JP.
PHOENX_USE_GREEDY_COLORING: bool = True

# ---------------------------------------------------------------------------
# Per-row Nyquist headroom multipliers (compile-time)
# ---------------------------------------------------------------------------
#
# Every PD-style soft constraint clamps user-supplied stiffness at the
# substep's Nyquist limit so requesting "more spring than the timestep
# can resolve" produces the stiffest resolvable lock instead of
# aliasing. The strict implicit-Euler bound is
#
#     k <= 1 / (M_inv * dt^2)         <=>     N = 1
#
# beyond which the soft-PD's eff_mass collapses to ``1 / M_inv`` and
# its bias spikes to ``C / dt``. The constants below let each PD-style
# row request a multiplier ``N`` on that bound; the helper
# (:func:`pd_coefficients` and the inline cable clamps) caps each row
# at ``_PD_NYQUIST_HEADROOM_MAX`` (currently ``10``) so requests above
# the global ceiling are silently truncated.
#
# Defaults: ``1`` (strict) for revolute / prismatic drive and limit;
# ``10`` for cable bend / twist (matches the previous BEAM behaviour
# the cable mode was promoted from). Tune per-row at compile time --
# changing one of these constants triggers a kernel-cache miss and
# rebuild but otherwise needs no API changes.

#: Headroom on the revolute joint's PD drive row. Default ``10``: the
#: drive is a single-axis scalar PD row, well-conditioned enough to
#: tolerate the same headroom as the cable bend / twist rows. Aligns
#: with the Box2D-soft limit's ``omega <= pi/dt`` convention
#: (effective ``N ~ pi^2 ~ 9.87``) so the two limit formulations
#: behave consistently.
PHOENX_BOOST_REVOLUTE_DRIVE = wp.constant(wp.float32(10.0))
#: Headroom on the revolute joint's PD limit row (PD path only;
#: Box2D-soft limits use the omega-cap convention).
PHOENX_BOOST_REVOLUTE_LIMIT = wp.constant(wp.float32(10.0))
#: Headroom on the prismatic joint's PD drive row. Same rationale
#: as :data:`PHOENX_BOOST_REVOLUTE_DRIVE`.
PHOENX_BOOST_PRISMATIC_DRIVE = wp.constant(wp.float32(10.0))
#: Headroom on the prismatic joint's PD limit row.
PHOENX_BOOST_PRISMATIC_LIMIT = wp.constant(wp.float32(10.0))
#: Headroom on the cable joint's bend (anchor-2 tangent 2-row PD)
#: rows. Default ``10``: cable bends are visually "soft" springs, so
#: chains routinely request stiff gains relative to dt; the headroom
#: keeps them inside the resolvable band rather than saturating the
#: cap.
PHOENX_BOOST_CABLE_BEND = wp.constant(wp.float32(10.0))
#: Headroom on the cable joint's twist (anchor-3 scalar 1-row PD)
#: row. Same default rationale as the bend rows.
PHOENX_BOOST_CABLE_TWIST = wp.constant(wp.float32(10.0))
#: Headroom on the soft-contact normal PD row (used when contacts opt
#: in to the implicit-Euler PD path via positive ``stiffness`` /
#: ``damping`` material parameters; the Box2D-style normal path uses
#: its own omega-cap convention and is unaffected). Default ``1``
#: (strict): contacts already self-limit via ``effective_gap``
#: clamping and a stiff impact regime, so extra headroom is rarely
#: useful.
PHOENX_BOOST_CONTACT_NORMAL = wp.constant(wp.float32(1.0))


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
    "PHOENX_USE_GREEDY_COLORING",
]
