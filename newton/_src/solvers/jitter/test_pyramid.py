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

    def test_ground_reaction_balances_weight(self):
        """Total vertical ground reaction must equal the stack's weight.

        At rest the sum of all contact forces the plane applies to the
        cubes must cancel gravity exactly, modulo PGS discretisation
        error. We check the Z component of the sum of per-contact
        forces reported by :meth:`World.gather_contact_wrenches` for
        every contact whose ``shape0`` is the plane (shape id 0 in
        this scene).

        Also cross-checks that the per-pair summary kernel
        (:meth:`World.gather_contact_pair_wrenches`) agrees with the
        per-contact kernel: summing both should yield the same total.
        """
        g = 9.81
        ex = _run_pyramid(layers=5, frames=self.SETTLE_FRAMES)

        # ---- total weight we expect to see pushed back by the ground ----
        inv_mass = ex.world.bodies.inverse_mass.numpy()
        # Jitter body 0 is the static anchor (inv_mass == 0). Skip it.
        masses = np.where(inv_mass[1:] > 0.0, 1.0 / np.maximum(inv_mass[1:], 1e-30), 0.0)
        total_weight = float(masses.sum() * g)
        self.assertGreater(total_weight, 0.0)

        # ---- per-contact wrenches --------------------------------------
        per_contact = wp.zeros(
            ex.world.rigid_contact_max,
            dtype=wp.spatial_vector,
            device=ex.device,
        )
        ex.world.gather_contact_wrenches(per_contact)
        wrenches = per_contact.numpy()
        # ``wp.spatial_vector`` is packed as 6 floats; top 3 are force,
        # bottom 3 are torque (the convention we chose in
        # ``gather_constraint_wrenches``).
        forces = wrenches[:, :3]

        # Which contacts touch the ground plane? The plane is shape 0.
        contacts = ex.contacts
        n_active = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(n_active, 0, "no active contacts after settle")
        shape0 = contacts.rigid_contact_shape0.numpy()[:n_active]
        shape1 = contacts.rigid_contact_shape1.numpy()[:n_active]
        ground_mask = np.zeros(forces.shape[0], dtype=bool)
        for k in range(n_active):
            if shape0[k] == 0 or shape1[k] == 0:
                ground_mask[k] = True

        ground_force = forces[ground_mask].sum(axis=0)
        # The ground pushes up (+z) on body 2 of each contact. Body 2
        # is the *cube* for every ground contact we collected (shape 0
        # is the plane, so it is always shape0/body 1), so the +Z
        # component should be ~total_weight.
        self.assertGreater(
            float(ground_force[2]),
            0.5 * total_weight,
            msg=f"ground reaction too low: {ground_force} vs weight {total_weight}",
        )
        self.assertLess(
            abs(float(ground_force[2]) - total_weight) / total_weight,
            0.10,
            msg=(
                f"ground reaction Z ({ground_force[2]:.3f} N) does not "
                f"match weight ({total_weight:.3f} N)"
            ),
        )

        # ---- per-pair summary should agree -----------------------------
        n_cols = ex.world.max_contact_columns
        pair_wrenches = wp.zeros(n_cols, dtype=wp.spatial_vector, device=ex.device)
        pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=ex.device)
        pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=ex.device)
        pair_count = wp.zeros(n_cols, dtype=wp.int32, device=ex.device)
        ex.world.gather_contact_pair_wrenches(
            pair_wrenches, pair_b1, pair_b2, pair_count
        )
        pw = pair_wrenches.numpy()[:, :3]
        pc = pair_count.numpy()
        # Totals across all *active* columns must equal the total
        # over all per-contact entries, within float slop.
        total_pair_force = pw[pc > 0].sum(axis=0)
        total_per_contact_force = forces.sum(axis=0)
        np.testing.assert_allclose(
            total_pair_force,
            total_per_contact_force,
            rtol=1e-3,
            atol=1e-2,
            err_msg="per-pair and per-contact totals disagree",
        )

        # Active columns' contact counts should add up to the active
        # contacts reported by the CollisionPipeline.
        self.assertEqual(int(pc.sum()), n_active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
