# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the full-revolution angle tracker.

Validates :func:`math_helpers.extract_rotation_angle`,
:func:`math_helpers.revolution_tracker_update`, and
:func:`math_helpers.revolution_tracker_angle` directly via a
thin driver kernel on CPU (no jitter solver needed).

Cases:

* Single substep, no wrap -> counter stays at 0, cumulative angle
  equals the quaternion angle.
* Sweep of ``theta`` values against the closed-form Jolt / PhoenX
  reference (PhoenX quat.cuh:703).
* Full forward revolution in 12 substeps -> counter ends at +1, total
  angle ~= +2*pi.
* Full backward revolution in 12 substeps -> counter ends at -1, total
  angle ~= -2*pi.
* Multi-revolution sweep (+3 turns and -3 turns).
* Random walk well inside one branch -> never triggers a wrap even
  when the raw quaternion angle crosses zero multiple times.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.math_helpers import (
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)


@wp.kernel
def _sweep_tracker_kernel(
    quats: wp.array[wp.quatf],
    axis: wp.vec3f,
    counters_out: wp.array[wp.int32],
    prev_angle_out: wp.array[wp.float32],
    cumulative_out: wp.array[wp.float32],
    raw_angle_out: wp.array[wp.float32],
):
    """Sequentially drive the tracker over ``quats`` from a single thread.

    The tracker is inherently sequential (each step depends on the
    previous step's persistent state), so we pin the walk to thread 0
    and let the other "threads" be no-ops. This matches how the
    constraint solver uses it in practice: one-thread-per-constraint,
    state carried through scratch storage.
    """
    tid = wp.tid()
    if tid != 0:
        return

    counter = wp.int32(0)
    prev = wp.float32(0.0)
    n = quats.shape[0]
    for i in range(n):
        raw = extract_rotation_angle(quats[i], axis)
        counter, prev = revolution_tracker_update(raw, counter, prev)
        cumulative_out[i] = revolution_tracker_angle(counter, prev)
        raw_angle_out[i] = raw
        counters_out[i] = counter
        prev_angle_out[i] = prev


def _axis_angle_quat(axis_np: np.ndarray, angle: float) -> wp.quatf:
    """Build a unit quaternion from axis + angle using Warp's convention
    ``(x, y, z, w)``. Matches what a Warp body's orientation carries.
    """
    half = 0.5 * angle
    s = math.sin(half)
    c = math.cos(half)
    return wp.quatf(
        float(axis_np[0] * s),
        float(axis_np[1] * s),
        float(axis_np[2] * s),
        float(c),
    )


def _run(axis_np: np.ndarray, angles: list[float]):
    """Drive the tracker kernel on CPU for the given angle sequence.

    Returns ``(cumulative, counters, raw_in_branch)`` as NumPy arrays
    of length ``len(angles)``.
    """
    device = wp.get_device("cpu")
    quats = [_axis_angle_quat(axis_np, a) for a in angles]
    quats_wp = wp.array(quats, dtype=wp.quatf, device=device)
    n = len(angles)
    cumulative = wp.zeros(n, dtype=wp.float32, device=device)
    counters = wp.zeros(n, dtype=wp.int32, device=device)
    prev = wp.zeros(n, dtype=wp.float32, device=device)
    raw = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(
        _sweep_tracker_kernel,
        dim=1,
        inputs=[
            quats_wp,
            wp.vec3f(float(axis_np[0]), float(axis_np[1]), float(axis_np[2])),
            counters,
            prev,
            cumulative,
            raw,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return cumulative.numpy(), counters.numpy(), raw.numpy()


class TestExtractRotationAngle(unittest.TestCase):
    """Closed-form check of :func:`extract_rotation_angle` in its
    principal branch ``(-pi, pi]``."""

    def test_principal_branch(self):
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        angles_in = np.linspace(-math.pi + 0.05, math.pi - 0.05, 21).tolist()
        _, _, raw = _run(axis, angles_in)
        np.testing.assert_allclose(raw, angles_in, atol=1e-5)

    def test_pi_boundary_produces_consistent_magnitude(self):
        """At ``theta = +/- pi`` the quaternion's ``w`` component is
        mathematically zero but in float32 it's a tiny residual of
        either sign. The PhoenX closed form (``atan((x*s)/w)``) then
        produces ``+/- pi`` depending on the sign of ``w``; both
        branches agree in magnitude and map onto the same
        half-rotation, which is all we need for the revolution-tracker
        delta test (a jump of ``+/- 2*pi`` between consecutive steps
        never occurs across this boundary because the wrapped substep
        has to reach the *other* branch first via :func:`atan`, which
        always returns a value strictly inside ``(-pi, pi)`` for
        nonzero ``w``).
        """
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _, _, raw_pos = _run(axis, [math.pi])
        _, _, raw_neg = _run(axis, [-math.pi])
        self.assertAlmostEqual(abs(float(raw_pos[0])), math.pi, places=5)
        self.assertAlmostEqual(abs(float(raw_neg[0])), math.pi, places=5)


class TestRevolutionTracker(unittest.TestCase):
    """End-to-end walks through the tracker -- forward, backward, and
    idling in-branch."""

    AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def test_no_wrap_in_branch(self):
        angles = np.linspace(-1.0, 1.0, 11).tolist()
        cumulative, counters, _ = _run(self.AXIS, angles)
        np.testing.assert_array_equal(counters, np.zeros_like(counters))
        np.testing.assert_allclose(cumulative, angles, atol=1e-5)

    def test_forward_revolution(self):
        """Twelve equal steps of ``+pi/6`` sweep ``+2*pi`` without
        tripping either branch (the quaternion double-cover wraps at
        each substep but the tracker unwraps it back)."""
        step = math.pi / 6.0
        angles = [(i + 1) * step for i in range(12)]
        cumulative, counters, _ = _run(self.AXIS, angles)
        self.assertEqual(int(counters[-1]), 1)
        self.assertAlmostEqual(float(cumulative[-1]), 12.0 * step, places=5)
        # Cumulative must be monotonically non-decreasing along a pure
        # forward sweep.
        diffs = np.diff(cumulative)
        self.assertTrue(bool(np.all(diffs >= -1e-5)))

    def test_backward_revolution(self):
        step = math.pi / 6.0
        angles = [-(i + 1) * step for i in range(12)]
        cumulative, counters, _ = _run(self.AXIS, angles)
        self.assertEqual(int(counters[-1]), -1)
        self.assertAlmostEqual(float(cumulative[-1]), -12.0 * step, places=5)
        diffs = np.diff(cumulative)
        self.assertTrue(bool(np.all(diffs <= 1e-5)))

    def test_multi_revolution_forward(self):
        """Three full forward turns across 36 substeps -> counter = +3,
        cumulative = +6*pi."""
        step = math.pi / 6.0
        angles = [(i + 1) * step for i in range(36)]
        cumulative, counters, _ = _run(self.AXIS, angles)
        self.assertEqual(int(counters[-1]), 3)
        self.assertAlmostEqual(float(cumulative[-1]), 36.0 * step, places=5)

    def test_multi_revolution_backward(self):
        step = math.pi / 6.0
        angles = [-(i + 1) * step for i in range(36)]
        cumulative, counters, _ = _run(self.AXIS, angles)
        self.assertEqual(int(counters[-1]), -3)
        self.assertAlmostEqual(float(cumulative[-1]), -36.0 * step, places=5)

    def test_inbranch_idle_stays_on_zero_count(self):
        """Random walk with per-step jumps well under ``pi`` -> counter
        stays at zero even across dozens of back-and-forth crossings of
        the branch origin."""
        rng = np.random.default_rng(0xC0FFEE)
        # Walk in the principal branch, steps of at most ~0.3 rad so
        # we never get within ``pi`` of the wrap boundary.
        raw = 0.0
        angles: list[float] = []
        for _ in range(50):
            raw += float(rng.uniform(-0.3, 0.3))
            # Keep raw inside (-pi+0.2, pi-0.2) with soft reflections.
            if raw > math.pi - 0.2:
                raw = math.pi - 0.2
            elif raw < -math.pi + 0.2:
                raw = -math.pi + 0.2
            angles.append(raw)
        cumulative, counters, _ = _run(self.AXIS, angles)
        np.testing.assert_array_equal(counters, np.zeros_like(counters))
        np.testing.assert_allclose(cumulative, angles, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
