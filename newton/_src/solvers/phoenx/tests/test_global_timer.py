# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-side timing via the CUDA ``%globaltimer`` special register.

``%globaltimer`` is a 64-bit nanosecond counter exposed by PTX that
ticks at wall-clock rate and is shared across the whole device, so two
reads in the same thread bracket a real elapsed-time window. Wrapping
the inline PTX in :func:`warp.func_native` lets a Warp kernel sample
``start`` / ``end`` cheaply and write the difference back to host
memory.

This test exercises that pattern end-to-end:

* A kernel reads ``%globaltimer`` before and after a fixed busy-spin
  workload and writes the elapsed nanoseconds to an output array.
* We assert the elapsed value is strictly positive, monotonically
  larger than a no-op baseline, and bounded above by a generous
  ceiling (so a runaway timer or sign-flip would fail loudly).

CUDA only -- inline PTX has no CPU fallback.
"""

from __future__ import annotations

import unittest

import warp as wp

from newton._src.solvers.phoenx.timer import read_global_timer_ns as _read_global_timer_ns


@wp.kernel(enable_backward=False)
def _measure_busy_spin_kernel(
    spin_iters: wp.int32,
    elapsed_ns: wp.array[wp.uint64],
    sink: wp.array[wp.int32],
):
    """Single-thread kernel: sample ``%globaltimer`` before/after a
    fixed busy-spin workload. ``sink`` exists so the optimiser cannot
    hoist the spin loop away.
    """
    tid = wp.tid()
    if tid != 0:
        return

    start = _read_global_timer_ns()

    acc = wp.int32(0)
    for i in range(spin_iters):
        acc = acc + i * 3 + 1
    sink[0] = acc

    end = _read_global_timer_ns()

    elapsed_ns[0] = end - start


@wp.kernel(enable_backward=False)
def _measure_noop_kernel(
    elapsed_ns: wp.array[wp.uint64],
):
    """Baseline: back-to-back ``%globaltimer`` reads with no work
    between them. The diff bounds the intrinsic's own overhead.
    """
    tid = wp.tid()
    if tid != 0:
        return

    start = _read_global_timer_ns()
    end = _read_global_timer_ns()
    elapsed_ns[0] = end - start


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Global-timer tests require CUDA (inline PTX has no CPU fallback).",
)
class TestGlobalTimerNative(unittest.TestCase):
    """Verify ``%globaltimer`` via :func:`wp.func_native` returns sane
    elapsed nanoseconds from a Warp kernel."""

    def setUp(self) -> None:
        self.device = wp.get_preferred_device()

    def _launch_busy_spin(self, spin_iters: int) -> int:
        elapsed = wp.zeros(1, dtype=wp.uint64, device=self.device)
        sink = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _measure_busy_spin_kernel,
            dim=1,
            inputs=[spin_iters, elapsed, sink],
            device=self.device,
        )
        return int(elapsed.numpy()[0])

    def _launch_noop(self) -> int:
        elapsed = wp.zeros(1, dtype=wp.uint64, device=self.device)
        wp.launch(
            _measure_noop_kernel,
            dim=1,
            inputs=[elapsed],
            device=self.device,
        )
        return int(elapsed.numpy()[0])

    def test_busy_spin_elapsed_is_positive_and_bounded(self) -> None:
        """A 100k-iter busy spin must register a strictly positive,
        finite elapsed time. The upper bound (1 s) is intentionally
        generous so a slow CI machine cannot flake; the goal here is
        sign + magnitude sanity, not perf characterisation."""
        elapsed_ns = self._launch_busy_spin(spin_iters=100_000)
        self.assertGreater(
            elapsed_ns,
            0,
            msg=f"globaltimer diff must be > 0 ns, got {elapsed_ns}",
        )
        one_second_ns = 1_000_000_000
        self.assertLess(
            elapsed_ns,
            one_second_ns,
            msg=f"globaltimer diff implausibly large: {elapsed_ns} ns",
        )

    def test_busy_spin_dominates_noop_baseline(self) -> None:
        """A real workload's elapsed window must exceed a back-to-back
        no-op pair, which bounds the intrinsic's own overhead."""
        noop_ns = self._launch_noop()
        busy_ns = self._launch_busy_spin(spin_iters=100_000)
        self.assertGreater(
            busy_ns,
            noop_ns,
            msg=(f"busy-spin elapsed ({busy_ns} ns) must exceed no-op baseline ({noop_ns} ns)"),
        )

    def test_more_work_takes_more_time(self) -> None:
        """Monotonicity: 10x the iterations must register more elapsed
        nanoseconds. Compares medians of 5 runs to absorb scheduler
        jitter without inflating the iteration budget."""
        small_runs = sorted(self._launch_busy_spin(1_000) for _ in range(5))
        large_runs = sorted(self._launch_busy_spin(10_000) for _ in range(5))
        small_median = small_runs[len(small_runs) // 2]
        large_median = large_runs[len(large_runs) // 2]
        self.assertGreater(
            large_median,
            small_median,
            msg=(f"10x workload should take longer: small_median={small_median} ns, large_median={large_median} ns"),
        )


if __name__ == "__main__":
    unittest.main()
