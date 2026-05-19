# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmark for the unified ADBS joint dispatcher.

Times per-substep PhoenX step() on scenes that stress each
:data:`JOINT_MODE_*` branch. Not part of the unittest suite -- run
directly:

.. code-block:: bash

    uv run --extra dev python -m newton._src.solvers.phoenx.tests._bench_adbs

The script warms the Warp module cache, runs a timed sweep of
``(num_joints, steps)`` pairs per mode, and prints mean frame time +
FPS. Uses :class:`warp.ScopedTimer` with ``synchronize=True`` so the
host-side elapsed time reflects the full GPU-side step.
"""

from __future__ import annotations

import time

import warp as wp

from newton._src.solvers.phoenx.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

FPS = 240
SUBSTEPS = 4
SOLVER_ITERATIONS = 16


def _chain_builder(
    mode: JointMode,
    num_links: int,
    *,
    drive_mode: DriveMode = DriveMode.OFF,
    bend_stiffness: float = 0.0,
    twist_stiffness: float = 0.0,
    stiffness_drive: float = 0.0,
    damping_drive: float = 0.0,
):
    """Build a linear chain of ``num_links`` bodies connected by
    joints in the requested ``mode``. Returns a finalized
    :class:`PhoenXWorld`. The chain is pinned to the world at the
    first link and hangs under no gravity so the joint rows get fully
    exercised without contact traffic."""
    b = WorldBuilder()
    world = b.world_body
    prev = world
    for i in range(num_links):
        body = b.add_dynamic_body(
            position=(0.0, 0.0, -float(i + 1) * 0.1),
            orientation=(0.0, 0.0, 0.0, 1.0),
            inverse_mass=1.0,
            inverse_inertia=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
            affected_by_gravity=False,
        )
        anchor1 = (0.0, 0.0, -float(i) * 0.1)
        anchor2 = (0.0, 0.0, -float(i) * 0.1 - 0.05)
        kwargs = {}
        if mode is JointMode.BALL_SOCKET:
            b.add_joint(body1=prev, body2=body, anchor1=anchor1, mode=mode)
        elif mode is JointMode.CABLE:
            b.add_joint(
                body1=prev,
                body2=body,
                anchor1=anchor1,
                anchor2=anchor2,
                mode=mode,
                bend_stiffness=bend_stiffness,
                twist_stiffness=twist_stiffness,
                bend_damping=0.01,
                twist_damping=0.01,
            )
        elif mode is JointMode.REVOLUTE:
            b.add_joint(
                body1=prev,
                body2=body,
                anchor1=anchor1,
                anchor2=anchor2,
                mode=mode,
                drive_mode=drive_mode,
                stiffness_drive=stiffness_drive,
                damping_drive=damping_drive,
                max_force_drive=1000.0,
                **kwargs,
            )
        elif mode is JointMode.PRISMATIC:
            b.add_joint(
                body1=prev,
                body2=body,
                anchor1=anchor1,
                anchor2=anchor2,
                mode=mode,
                drive_mode=drive_mode,
                stiffness_drive=stiffness_drive,
                damping_drive=damping_drive,
                max_force_drive=1000.0,
            )
        elif mode is JointMode.FIXED:
            b.add_joint(
                body1=prev,
                body2=body,
                anchor1=anchor1,
                anchor2=anchor2,
                mode=mode,
            )
        prev = body
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, 0.0),
        device=wp.get_preferred_device(),
    )


def _warmup(world, frames: int = 3) -> None:
    dt = 1.0 / FPS
    for _ in range(frames):
        world.step(dt)
    wp.synchronize_device()


def _timed_step(world, frames: int) -> float:
    """Return mean wall-clock frame time [ms] over ``frames`` steps
    (with CUDA sync to capture the full GPU pipeline)."""
    dt = 1.0 / FPS
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(frames):
        world.step(dt)
    wp.synchronize_device()
    t1 = time.perf_counter()
    return (t1 - t0) / frames * 1000.0


def _bench(mode: JointMode, num_links: int, frames: int, **kwargs) -> float:
    world = _chain_builder(mode, num_links, **kwargs)
    _warmup(world)
    return _timed_step(world, frames)


def main() -> None:
    wp.init()
    device = wp.get_preferred_device()
    if not device.is_cuda:
        print("ADBS micro-benchmark requires CUDA.")
        return
    print(f"Device: {device}")
    print()

    # (mode_label, JointMode, kwargs)
    cases = [
        ("BALL_SOCKET", JointMode.BALL_SOCKET, {}),
        ("REVOLUTE (no drive)", JointMode.REVOLUTE, {}),
        (
            "REVOLUTE (PD drive)",
            JointMode.REVOLUTE,
            {
                "drive_mode": DriveMode.POSITION,
                "stiffness_drive": 100.0,
                "damping_drive": 5.0,
            },
        ),
        ("PRISMATIC (no drive)", JointMode.PRISMATIC, {}),
        (
            "PRISMATIC (PD drive)",
            JointMode.PRISMATIC,
            {
                "drive_mode": DriveMode.POSITION,
                "stiffness_drive": 100.0,
                "damping_drive": 5.0,
            },
        ),
        ("FIXED", JointMode.FIXED, {}),
        (
            "CABLE",
            JointMode.CABLE,
            {
                "bend_stiffness": 50.0,
                "twist_stiffness": 50.0,
            },
        ),
    ]

    num_links = 256
    frames = 50
    print(f"chain of {num_links} joints, {frames} frames per measurement")
    print(f"  substeps={SUBSTEPS}, solver_iterations={SOLVER_ITERATIONS}")
    print()
    print(f"{'mode':<26} {'mean ms/frame':>14} {'FPS':>10}")
    print("-" * 54)
    for label, mode, kwargs in cases:
        ms = _bench(mode, num_links, frames, **kwargs)
        fps = 1000.0 / ms
        print(f"{label:<26} {ms:>14.3f} {fps:>10.1f}")


if __name__ == "__main__":
    main()
