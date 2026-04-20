# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-stability regression test: settle an N-layer box pyramid.

This is the first end-to-end test for the Jitter solver's persistent
contact path. It exercises:

* :class:`CollisionPipeline` ingest with ``contact_matching=True``,
* the ``(contact_first, contact_count)`` column packing in
  :class:`ConstraintContainer`,
* frame-to-frame warm-starting via ``rigid_contact_match_index`` and
  the double-buffered :class:`ContactContainer`,
* the unified PGS dispatcher (contacts and joints share the loop),
* and the Baumgarte / speculative bias that keeps resting contacts
  stable without hopping or sinking.

Any regression in the contact prepare/iterate path, ingest ordering,
warm-start gather, or bias formulation collapses the stack and the
test's per-cube position / velocity budget catches it.

The scene itself is driven by :class:`Example` from
:mod:`newton._src.solvers.jitter.example_pyramid` so the test and
the interactive example stay bit-for-bit in sync.
"""

from __future__ import annotations

import types
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.example_pyramid import Example


class _HeadlessViewer:
    """Minimal viewer stub so :class:`Example` can be instantiated in tests.

    Only the methods :class:`Example.__init__` calls (``set_model`` /
    ``set_camera``) and the render-path no-ops need to exist; the
    test never calls ``render()``.
    """

    def set_model(self, *_a, **_kw) -> None: ...
    def set_camera(self, *_a, **_kw) -> None: ...
    def begin_frame(self, *_a, **_kw) -> None: ...
    def end_frame(self, *_a, **_kw) -> None: ...
    def log_state(self, *_a, **_kw) -> None: ...
    def log_contacts(self, *_a, **_kw) -> None: ...


def _run_pyramid(layers: int, frames: int) -> Example:
    """Build and settle an ``layers``-tall pyramid for ``frames`` frames."""
    viewer = _HeadlessViewer()
    args = types.SimpleNamespace(layers=layers)
    ex = Example(viewer, args)
    for _ in range(frames):
        ex.step()
    return ex


class TestPyramidSettle(unittest.TestCase):
    """End-to-end settle test for the Jitter contact solver.

    A ``layers``-row pyramid starts ~1 mm above its resting height in
    each slot. After ``SETTLE_FRAMES`` simulation frames at 60 fps /
    4 substeps, every cube's :meth:`Example.test_final` budget
    (10 cm position slack, 0.5 m/s velocity slack) must pass. Tested
    for a tiny (3-layer) and a realistic (10-layer) tower.
    """

    # Big frame counts on purpose: the warm-start path needs several
    # seconds of simulated time to damp out the initial transient
    # when the whole stack is freshly dropped.
    SETTLE_FRAMES = 180  # 3 s @ 60 fps

    @classmethod
    def setUpClass(cls):
        # Warp picks the default device; keep it explicit so CI logs
        # show us which backend was actually exercised.
        cls.device = wp.get_preferred_device()

    def test_3_layer_pyramid(self):
        """A 3-layer pyramid (6 cubes) must come to rest."""
        ex = _run_pyramid(layers=3, frames=self.SETTLE_FRAMES)
        ex.test_final()

    def test_10_layer_pyramid(self):
        """A 10-layer pyramid (55 cubes) must come to rest.

        This is the user-facing stability bar: if a 10-tall stack
        holds together, the contact solver's warm-start and friction
        paths are doing their job.
        """
        ex = _run_pyramid(layers=10, frames=self.SETTLE_FRAMES)
        ex.test_final()

    def test_settled_pyramid_force_report(self):
        """Contact forces on a settled 5-layer pyramid must satisfy
        static equilibrium for every body individually.

        After 3 s of settle the stack is at rest. Newton's second law
        at zero acceleration requires, for every dynamic body ``b``:

            F_up(b) - F_down(b) - m_b * g == 0

        where

            F_up(b)   = sum of contact-normal Fz from pairs where b
                        is ``body2`` (things below pushing ``b`` up)
            F_down(b) = sum of contact-normal Fz from pairs where b
                        is ``body1`` (what ``b`` pushes down onto
                        things below -- equivalently the reaction b
                        feels from its upper neighbours is the
                        opposite sign, so we count the +Fz applied to
                        those neighbours here)

        This is the strongest per-body check possible for a
        multi-body stack because it does *not* depend on how PGS
        distributes load between symmetric siblings (that
        distribution is mathematically indeterminate for aligned,
        rigid cubes and changes with warm-start / iteration count).
        Only the *per-body* sums are uniquely determined, and those
        are what we pin.

        The test also cross-checks:
          * the top cube has ``F_down == 0`` (nothing rests on it),
          * every vertical contact has non-negative normal force
            (Signorini),
          * the per-pair summary kernel agrees with the per-contact
            kernel (consistency between the two report APIs).
        """
        g = 9.81
        ex = _run_pyramid(layers=5, frames=self.SETTLE_FRAMES)

        # Confirm the stack is actually at rest -- otherwise F = m*a
        # rather than F = m*g and the balance below is meaningless.
        velocities = ex.world.bodies.velocity.numpy()
        v_max = float(np.linalg.norm(velocities[1:], axis=1).max())
        self.assertLess(v_max, 1e-2, f"stack hasn't settled (max |v|={v_max:.4f})")

        inv_m = ex.world.bodies.inverse_mass.numpy()
        mass = np.where(inv_m > 0.0, 1.0 / np.maximum(inv_m, 1e-30), 0.0)
        weight = mass * g  # indexed by Jitter body id

        # ---- per-contact-column wrenches --------------------------------
        n_cols = ex.world.max_contact_columns
        pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=ex.device)
        pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=ex.device)
        pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=ex.device)
        pair_count = wp.zeros(n_cols, dtype=wp.int32, device=ex.device)
        ex.world.gather_contact_pair_wrenches(pair_w, pair_b1, pair_b2, pair_count)

        pw = pair_w.numpy()[:, :3]
        b1_np = pair_b1.numpy()
        b2_np = pair_b2.numpy()
        active = pair_count.numpy() > 0
        self.assertGreater(int(active.sum()), 0, "no active contact pairs")

        # ---- per-body vertical balance ----------------------------------
        # Residual budget: warm-start lag + one PGS sweep's worth of
        # slack. Empirically < 5 N on a 9810 N weight -- we grant
        # 50 N (0.5% of one cube's weight).
        BALANCE_TOL = 50.0

        residuals = []
        for b in range(1, ex.world.num_bodies):
            sel_up = active & (b2_np == b)
            sel_dn = active & (b1_np == b)
            f_up = float(pw[sel_up, 2].sum())
            f_dn = float(pw[sel_dn, 2].sum())
            net = f_up - f_dn
            resid = net - float(weight[b])
            residuals.append(resid)
            self.assertLess(
                abs(resid),
                BALANCE_TOL,
                msg=(
                    f"body {b}: vertical imbalance {resid:.3f} N "
                    f"(F_up={f_up:.1f}, F_down={f_dn:.1f}, "
                    f"m*g={weight[b]:.1f})"
                ),
            )

        # ---- apex cube has nothing resting on it -----------------------
        # In a 5-layer pyramid body 15 is the top cube. Nothing sits
        # above it, so its F_down must be zero.
        apex = ex.world.num_bodies - 1
        sel_apex_dn = active & (b1_np == apex)
        self.assertAlmostEqual(
            float(pw[sel_apex_dn, 2].sum()),
            0.0,
            delta=1.0,
            msg="apex cube should have zero downward contact force",
        )

        # ---- Signorini: no vertical contact may pull -------------------
        per = wp.zeros(ex.world.rigid_contact_max, dtype=wp.spatial_vector, device=ex.device)
        ex.world.gather_contact_wrenches(per)
        contacts = ex.contacts
        n_active = int(contacts.rigid_contact_count.numpy()[0])
        nrm = contacts.rigid_contact_normal.numpy()[:n_active]
        f_all = per.numpy()[:n_active, :3]
        # Vertical contacts (box-on-box or box-on-plane) have |nz|
        # close to 1. For them the reported Fz (= lambda_n * idt,
        # times the +z normal) must be non-negative.
        vert = np.abs(nrm[:, 2]) > 0.9
        if vert.any():
            self.assertGreaterEqual(
                float(f_all[vert, 2].min()),
                -1.0,
                msg="vertical contact reported pulling (negative normal)",
            )

        # ---- cross-check per-pair vs per-contact totals ----------------
        total_pair = pw[active].sum(axis=0)
        total_percontact = f_all.sum(axis=0)
        np.testing.assert_allclose(
            total_pair,
            total_percontact,
            rtol=1e-3,
            atol=1e-1,
            err_msg="per-pair and per-contact totals disagree",
        )
        self.assertEqual(int(pair_count.numpy().sum()), n_active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
