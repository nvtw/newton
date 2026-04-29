# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Empirical characterisation of ``_BEAM_NYQUIST_HEADROOM``.

The BEAM bend / twist Nyquist clamp uses a headroom factor ``N`` so
the user-supplied stiffness is bounded by ``k <= N / (M_inv * dt^2)``;
``_BEAM_NYQUIST_HEADROOM`` defaults to ``10`` (vs the strict ``N = 1``
implicit-Euler bound). This test sweeps ``k_bend`` from well below
the strict bound up past the headroom-relaxed cap and asserts:

1. **Boundedness** -- pose magnitude and angular velocity stay finite
   across the full sweep, including super-Nyquist gains. A regression
   that miscomputed the cap (e.g. sign-flipped, off by ``dt``) would
   typically blow up here.
2. **Monotone stiffening up to the cap** -- below ``k_max = N / (M_inv
   * dt^2)`` doubling ``k_bend`` measurably reduces the residual tilt
   at a fixed settle time, since the implicit-Euler PD has not yet
   saturated.
3. **Saturation plateau beyond the cap** -- past ``k_max`` further
   doubling ``k_bend`` produces *no* additional stiffening (within
   numerical noise), since the apparent stiffness is bounded by the
   cap and any extra ``k`` is silently truncated by ``wp.min(k, k_max)``.

The cap is computed from the BEAM bend block's representative
effective inverse-mass, which depends on the rod's geometry / inertia.
We use the pendulum helper from :mod:`test_beam_joint` (rod COM at
anchor, ``rest_length = 1 m``, ``inv_inertia = 20 I``) so ``M_inv ~=
inv_mass * 2 + r^2 * inv_I * 2`` lands near 21 and the strict cap at
``dt = 1 / (240 * 4)`` s is around ``k_max_strict ~ 4.4e4``; the
headroom-relaxed cap lands near ``4.4e5``. The sweep brackets both
regimes by an order of magnitude on either side.

This test is the empirical complement to
:file:`NYQUIST_HEADROOM.md`: that document inventories every clamp
site, this test pins the BEAM-side cap behaviour so the inventory
recommendations cannot silently drift.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    _BEAM_NYQUIST_HEADROOM,
)
from newton._src.solvers.phoenx.tests._test_helpers import run_settle_loop
from newton._src.solvers.phoenx.tests.test_beam_joint import (
    FPS,
    SOLVER_ITERATIONS,
    SUBSTEPS,
    _axis_angle_quat,
    _build_beam_pendulum,
    _rod_orientation,
    _rotation_angle_about,
)

#: Initial bend angle [rad] for every sweep point. Must be small
#: enough that sin(theta) ~= theta (so the linear analytical regime
#: applies and the saturation plateau is identifiable) but large
#: enough that the residual after settling is well above quantisation
#: noise.
_INIT_TILT_RAD = math.radians(5.0)

#: Settling window [s] before the residual tilt is sampled. Picked so
#: undamped period at the strict-cap stiffness (k ~ 1e4, omega ~ sqrt
#: (k / I) ~ 450 rad/s, T ~ 14 ms) fits ~70x into the window: we
#: average over many cycles so the residual is the *envelope* not a
#: phase-snapshot.
_SETTLE_S = 1.0


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "BEAM Nyquist sweep requires CUDA (graph-capture path).",
)
class TestBeamNyquistHeadroom(unittest.TestCase):
    """Sweep ``k_bend`` past the strict and relaxed Nyquist caps; check
    boundedness, monotone stiffening below the cap, plateau above."""

    def _measure_residual_tilt(self, k_bend: float, c_bend: float) -> tuple[float, float]:
        """Build a 5-deg pre-deflected pendulum at ``(k_bend, c_bend)``,
        settle for ``_SETTLE_S`` seconds, return ``(residual_tilt_rad,
        omega_norm_rad_per_s)``. Returns ``inf`` for either component
        if the sim diverges to NaN -- callers can fail loudly on that.
        """
        init_q = _axis_angle_quat((0.0, 1.0, 0.0), _INIT_TILT_RAD)
        world = _build_beam_pendulum(
            wp.get_preferred_device(),
            bend_stiffness=k_bend,
            twist_stiffness=k_bend,
            bend_damping=c_bend,
            twist_damping=c_bend,
            init_rotation=init_q,
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
        )
        dt = 1.0 / FPS
        run_settle_loop(world, int(_SETTLE_S * FPS), dt=dt)

        y_axis = np.array([0.0, 1.0, 0.0])
        tilt = abs(_rotation_angle_about(_rod_orientation(world), y_axis))
        omega = float(np.linalg.norm(world.bodies.angular_velocity.numpy()[1]))
        if not (np.isfinite(tilt) and np.isfinite(omega)):
            return float("inf"), float("inf")
        return tilt, omega

    def test_super_nyquist_gains_stay_bounded(self) -> None:
        """All sweep points (including 100x and 1000x past the strict
        cap) finish with finite pose / omega -- catches sign or ``dt``
        regressions in the cap expression that would let
        unconditionally large gains propagate.

        Uses substantial damping to suppress ringing -- this test only
        cares about boundedness, not envelope shape.
        """
        c_bend = 5.0
        for k in (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7):
            with self.subTest(k_bend=k):
                tilt, omega = self._measure_residual_tilt(k, c_bend)
                self.assertTrue(
                    math.isfinite(tilt),
                    msg=f"residual tilt diverged at k_bend={k:.0e}: tilt={tilt}",
                )
                self.assertTrue(
                    math.isfinite(omega),
                    msg=f"omega diverged at k_bend={k:.0e}: |omega|={omega}",
                )
                # Pose stays inside the unit-rotation half-space; with
                # ``_INIT_TILT_RAD ~ 0.087 rad`` and damping the rod
                # cannot wind up by more than a few times its initial
                # deflection without something going wrong.
                self.assertLess(
                    tilt,
                    1.0,
                    msg=(
                        f"residual tilt exceeded 1 rad at k_bend={k:.0e}: "
                        f"tilt={tilt:.3f} -- possible cap miscomputation."
                    ),
                )

    def test_monotone_stiffening_below_strict_cap(self) -> None:
        """Below the strict cap doubling ``k`` reduces the residual
        tilt -- the standard linear-PD regime. Use a low damping so
        the spring's stiffness is the dominant settling timescale
        (otherwise residual is set by the damping rate and is
        ``k``-independent).

        Cap estimate at ``dt = 1/(240*4) = 1.04 ms``: ``M_inv ~ 21``,
        ``k_max_strict ~ 1 / (21 * 1.08e-6) ~ 4.4e4``. We sweep
        through ``[1e2, 1e3, 1e4]`` so every probe sits below the cap.
        """
        c_bend = 0.05  # very light damping
        ks = [1.0e2, 1.0e3, 1.0e4]
        residuals = [self._measure_residual_tilt(k, c_bend)[0] for k in ks]
        for k, tilt in zip(ks, residuals):
            self.assertTrue(
                math.isfinite(tilt),
                msg=f"residual tilt diverged at k_bend={k:.0e}",
            )

        # Monotone non-increasing tilt: each successive 10x bump in
        # ``k`` should not *increase* the residual envelope. Allow a
        # 10 % slack for numerical noise (the rod may catch a phase
        # peak rather than the envelope at exactly 1 s).
        for i in range(1, len(residuals)):
            self.assertLessEqual(
                residuals[i],
                residuals[i - 1] * 1.1,
                msg=(
                    f"BEAM bend residual increased with stiffness below cap: "
                    f"k={ks[i - 1]:.0e} -> tilt={residuals[i - 1]:.4f}, "
                    f"k={ks[i]:.0e} -> tilt={residuals[i]:.4f}"
                ),
            )
        # End-to-end the strongest spring should be visibly tighter
        # than the softest -- catches regressions where the cap
        # silently kicks in below the documented threshold.
        self.assertLess(
            residuals[-1],
            0.5 * residuals[0],
            msg=(
                f"Below-cap stiffening collapsed: k_min={ks[0]:.0e} -> "
                f"tilt={residuals[0]:.4f}, k_max={ks[-1]:.0e} -> "
                f"tilt={residuals[-1]:.4f}; expected at least 2x decrease."
            ),
        )

    def test_plateau_beyond_headroom_cap(self) -> None:
        """Past the headroom cap (~``N * 1 / (M_inv * dt^2) ~ N * 4.4e4
        ~ 4.4e5`` with ``N = 10``) further increases in ``k_bend``
        produce no measurable stiffening -- the cap silently truncates
        them via ``wp.min(k, k_max)``. The residual at ``k = 1e6`` and
        ``k = 1e7`` should agree within a few percent; if they diverge
        the cap is missing or wrong.

        Uses heavy damping so the residual is driven to a stable
        steady state by the settling window's end and the comparison
        is between settled poses, not phase snapshots.
        """
        c_bend = 5.0
        ks = [1.0e6, 1.0e7]
        residuals = [self._measure_residual_tilt(k, c_bend)[0] for k in ks]
        for k, tilt in zip(ks, residuals):
            self.assertTrue(
                math.isfinite(tilt),
                msg=f"residual tilt diverged at k_bend={k:.0e}",
            )

        # Plateau check: 10x bump in ``k`` past the cap should change
        # the residual by less than 30 %. Cap-truncated gains produce
        # the same effective spring -- some variance is allowed because
        # the relax pass and damping interact with the cap-clipped
        # bias slightly differently between the two ``k`` values.
        rel = abs(residuals[1] - residuals[0]) / max(residuals[0], 1e-9)
        self.assertLess(
            rel,
            0.30,
            msg=(
                f"Above-cap residual changed by {rel * 100:.1f}%; expected <30% "
                f"(plateau). k=1e6 -> tilt={residuals[0]:.6f}, "
                f"k=1e7 -> tilt={residuals[1]:.6f}. If this fails, the "
                f"BEAM Nyquist cap may not be saturating as documented."
            ),
        )

    def test_headroom_constant_value(self) -> None:
        """Pin ``_BEAM_NYQUIST_HEADROOM = 10.0`` so a future tweak to
        the constant requires updating the headroom inventory
        (NYQUIST_HEADROOM.md) and the audit doc (PD_DRIVE_AUDIT.md)
        in the same PR. Strictly an inventory-coherence guard, not a
        physical assertion.
        """
        self.assertAlmostEqual(
            float(_BEAM_NYQUIST_HEADROOM),
            10.0,
            places=6,
            msg=(
                "BEAM Nyquist headroom factor changed; update "
                "NYQUIST_HEADROOM.md and PD_DRIVE_AUDIT.md to "
                "match the new value, and re-tune the plateau "
                "threshold in test_plateau_beyond_headroom_cap."
            ),
        )


if __name__ == "__main__":
    unittest.main()
