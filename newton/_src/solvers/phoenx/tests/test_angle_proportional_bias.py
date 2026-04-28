# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``angle_proportional_chord_bias``.

The helper converts a chord-length displacement into an
angle-proportional velocity bias for a soft point-anchor row. A
naive ``chord * idt`` bias produces a restoring torque that's
linear in ``theta`` only at small angles (because the chord itself
saturates as ``2*L*sin(theta/2)``). The helper recovers ``theta``
from the chord and returns ``theta * L * idt``, so the resulting
restoring torque ``tau ~ k * L^2 * theta`` stays linear across the
full angle range.

These tests poke the helper directly (no joint, no PGS solver) at a
fixed lever arm and a fixed substep ``dt``, to verify the math is
right before it gets wired into the cable joint redesign.
"""

import math
import unittest

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_container import (
    angle_proportional_chord_bias,
)


@wp.kernel
def _evaluate_bias_kernel(
    chords: wp.array[wp.float32],
    lever_arm: wp.float32,
    idt: wp.float32,
    out: wp.array[wp.float32],
):
    tid = wp.tid()
    out[tid] = angle_proportional_chord_bias(chords[tid], lever_arm, idt)


def _eval_bias(chord_values, lever_arm: float, idt: float):
    n = len(chord_values)
    chords = wp.array(chord_values, dtype=wp.float32)
    out = wp.zeros(n, dtype=wp.float32)
    wp.launch(
        _evaluate_bias_kernel,
        dim=n,
        inputs=[chords, wp.float32(lever_arm), wp.float32(idt)],
        outputs=[out],
    )
    return out.numpy().tolist()


class TestAngleProportionalChordBias(unittest.TestCase):
    """Check that the bias is ``theta * L * idt`` (not ``chord * idt``).

    For each test angle, set up the corresponding chord value
    ``chord = 2 * L * sin(theta/2)`` and check the helper returns
    ``theta * L * idt``.
    """

    def setUp(self) -> None:
        wp.init()
        self.L = 0.05  # 5 cm lever arm (typical cable link half-length)
        self.dt = 1.0 / 6000.0  # 100 substeps at 60 Hz, like the bend example
        self.idt = 1.0 / self.dt

    def _chord_for_angle(self, theta: float) -> float:
        return 2.0 * self.L * math.sin(0.5 * theta)

    def _expected_bias(self, theta: float) -> float:
        return theta * self.L * self.idt

    def test_bias_recovers_theta_at_canonical_angles(self) -> None:
        """At theta in {10, 45, 90, 135} deg the helper output must
        equal ``theta * L * idt`` to within float32 round-trip slop.
        """
        thetas = [math.radians(d) for d in (10.0, 45.0, 90.0, 135.0)]
        chords = [self._chord_for_angle(t) for t in thetas]
        biases = _eval_bias(chords, self.L, self.idt)

        for theta, bias_got in zip(thetas, biases):
            bias_expected = self._expected_bias(theta)
            self.assertAlmostEqual(
                bias_got,
                bias_expected,
                delta=max(abs(bias_expected) * 1e-5, 1e-3),
                msg=(
                    f"theta={math.degrees(theta):.1f} deg: "
                    f"bias={bias_got:.4f} expected {bias_expected:.4f} "
                    f"(naive chord*idt would give {self._chord_for_angle(theta) * self.idt:.4f})"
                ),
            )

    def test_bias_is_linear_in_theta(self) -> None:
        """``bias / theta`` must be the same constant ``L * idt``
        across all angles -- the whole point of the helper.
        """
        thetas = [math.radians(d) for d in (5.0, 30.0, 60.0, 90.0, 120.0, 150.0)]
        chords = [self._chord_for_angle(t) for t in thetas]
        biases = _eval_bias(chords, self.L, self.idt)
        ratios = [b / t for b, t in zip(biases, thetas)]

        target = self.L * self.idt
        for theta, ratio in zip(thetas, ratios):
            self.assertAlmostEqual(
                ratio,
                target,
                delta=target * 5e-5,
                msg=(
                    f"theta={math.degrees(theta):.1f} deg: "
                    f"bias/theta={ratio:.4f} expected {target:.4f} "
                    f"(naive chord-bias would give "
                    f"{self._chord_for_angle(theta) * self.idt / theta:.4f})"
                ),
            )

    def test_bias_sign_follows_chord_sign(self) -> None:
        """A negative chord (rotation in the opposite direction)
        must produce a negative bias of equal magnitude to the
        positive case.
        """
        theta = math.radians(45.0)
        chord_pos = self._chord_for_angle(theta)
        biases = _eval_bias([chord_pos, -chord_pos], self.L, self.idt)
        self.assertAlmostEqual(biases[0], -biases[1], delta=1e-6)
        self.assertGreater(biases[0], 0.0)
        self.assertLess(biases[1], 0.0)

    def test_bias_zero_for_zero_chord(self) -> None:
        """Rest pose => zero displacement => zero bias."""
        biases = _eval_bias([0.0], self.L, self.idt)
        self.assertEqual(biases[0], 0.0)

    def test_bias_zero_lever_arm_returns_zero(self) -> None:
        """Defensive: degenerate ``L = 0`` returns 0 without
        ``inf``/``nan``."""
        biases = _eval_bias([0.01], 0.0, self.idt)
        self.assertEqual(biases[0], 0.0)

    def test_naive_chord_bias_is_wrong_at_large_theta(self) -> None:
        """Sanity: the naive ``chord * idt`` bias *would* be wrong
        at 90 deg by ~10 % and at 135 deg by ~21 %. Confirms the
        test fixture is meaningful (the helper result genuinely
        differs from the naive formula at those angles).
        """
        for theta_deg, expected_pct_error in ((90.0, 9.5), (135.0, 21.0)):
            theta = math.radians(theta_deg)
            chord = self._chord_for_angle(theta)
            naive = chord * self.idt
            true_bias = theta * self.L * self.idt
            err_pct = 100.0 * abs(naive - true_bias) / true_bias
            self.assertGreater(
                err_pct,
                expected_pct_error - 1.0,
                msg=(
                    f"theta={theta_deg} deg: naive chord*idt only differs "
                    f"from theta*L*idt by {err_pct:.1f}% -- expected "
                    f">~{expected_pct_error}%"
                ),
            )


if __name__ == "__main__":
    unittest.main()
