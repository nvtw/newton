# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU wall-clock timer helpers built on CUDA's ``%globaltimer`` register.

``%globaltimer`` is a 64-bit nanosecond counter shared across the whole
device; two reads in the same thread bracket a real elapsed-time
window suitable for per-constraint profiling inside a Warp kernel.

GPU-only -- inline PTX has no CPU fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

__all__ = [
    "elapsed_us",
    "print_column_timings",
    "read_global_timer_ns",
]


@wp.func_native(
    """
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
    """
)
def read_global_timer_ns() -> wp.uint64:
    """Read CUDA ``%globaltimer`` (nanoseconds since device boot)."""
    ...


@wp.func
def elapsed_us(start_ns: wp.uint64, end_ns: wp.uint64) -> wp.float32:
    """Convert a ``(start, end)`` ``%globaltimer`` pair to microseconds."""
    delta_ns = wp.int64(end_ns - start_ns)
    return wp.float32(delta_ns) * wp.float32(1.0e-3)


def print_column_timings(world: PhoenXWorld, frame: int, label: str = "phoenx") -> None:
    """Dump the per-type ``time_us`` totals from the most recent step.

    Reads :meth:`PhoenXWorld.step_report` (D2H), then prints absolute
    microseconds + share-of-total for the five dispatch buckets so it
    is obvious where the substep loop is spending wall-clock time.

    Requires the world to have been constructed with
    ``enable_column_timers=True``; no-ops otherwise.

    Args:
        world: Solver instance whose last :meth:`PhoenXWorld.step` is
            being inspected.
        frame: Caller-supplied frame index, included in the log prefix.
        label: Short scene tag for the log prefix; defaults to
            ``"phoenx"``.
    """
    if not getattr(world, "enable_column_timers", False):
        return
    report = world.step_report()
    if report.time_us_total_joints is None:
        return
    j = report.time_us_total_joints
    t = report.time_us_total_cloth_triangles or 0.0
    b = report.time_us_total_cloth_bending or 0.0
    s = report.time_us_total_soft_tetrahedra or 0.0
    c = report.time_us_total_contacts or 0.0
    total = j + t + b + s + c
    if total <= 0.0:
        return

    def pct(v: float) -> float:
        return 100.0 * v / total

    print(
        f"[{label} column timers, frame={frame}] total={total:.1f}us | "
        f"soft_tet={s:.1f}us ({pct(s):.1f}%) "
        f"contacts={c:.1f}us ({pct(c):.1f}%) "
        f"joints={j:.1f}us ({pct(j):.1f}%) "
        f"cloth_tri={t:.1f}us ({pct(t):.1f}%) "
        f"cloth_bend={b:.1f}us ({pct(b):.1f}%)"
    )
