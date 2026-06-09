# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Invariant checks on PhoenX's CUDA launch geometry.

The solver's perf model assumes every main constraint-solve kernel
launches a padded integer-warp grid: dynamic auto mode reserves one
warp per world, while fixed ``threads_per_world`` may pack multiple
worlds into a warp. Each world walks its per-color CSR bucket via
``tid % tpw`` / ``tid / tpw``. Breaking this invariant -- e.g. by
dispatching ``dim=num_worlds``, or using a non-warp block size -- would
silently serialise the solve or run the kernels with wrong thread-block
conventions.

These tests hook :func:`warp.launch` for the duration of a single
:meth:`PhoenXWorld.step` call, capture ``(kernel_name, dim,
block_dim)`` tuples for every main-solve kernel, and assert the
invariant. Runs on CUDA only (the kernels are GPU-only).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _STRAGGLER_BLOCK_DIM,
    get_fast_tail_kernel,
)
from newton._src.solvers.phoenx.tests.test_multi_world import _build_n_pendulums


def _main_solve_kernel_launch_bounds(world) -> dict[str, int]:
    base_fast_tail_kw = {
        "revolute_only": bool(world._use_revolute_specialization),
        "has_sleeping": bool(world._sleeping_enabled),
        "enable_column_timers": world.enable_column_timers,
    }
    bounds = {}
    for fixed_tpw in world._fast_tail_auto_fixed_choices():
        fast_tail_kw = {**base_fast_tail_kw, "fixed_tpw": fixed_tpw, "guard_tpw": world._tpw_auto}
        launch_bound = fixed_tpw if fixed_tpw > 0 else world._tpw_launch_bound
        bounds[get_fast_tail_kernel(kind="prepare_plus_iterate", **fast_tail_kw).key] = launch_bound
        bounds[get_fast_tail_kernel(kind="relax", **fast_tail_kw).key] = launch_bound
    return bounds


def _collect_launches(world, *, step_dt: float = 1.0 / 60.0) -> list[dict]:
    """Run one ``world.step(step_dt)`` while capturing every
    :func:`wp.launch` call's kernel + dim + block_dim.

    Returns the captured list of dicts. The patched ``wp.launch``
    still forwards to the real implementation so the step produces
    its normal GPU-side effects; we only observe the launch shape.
    """
    captured: list[dict] = []
    real_launch = wp.launch

    def spy(*args, **kwargs):
        # First positional is the kernel; ``dim`` / ``block_dim``
        # can be either positional or keyword.
        kernel = args[0] if args else kwargs.get("kernel")
        dim = args[1] if len(args) > 1 and "dim" not in kwargs else kwargs.get("dim")
        block_dim = kwargs.get("block_dim")
        if block_dim is None:
            # Block dim is also positional in some signatures; pull
            # the default from Warp's actual ``launch`` binding if
            # absent from both.
            block_dim = 256
        captured.append(
            {
                "kernel": getattr(kernel, "key", str(kernel)),
                "dim": (int(dim) if not isinstance(dim, list | tuple) else tuple(int(d) for d in dim)),
                "block_dim": int(block_dim),
            }
        )
        return real_launch(*args, **kwargs)

    with patch("warp.launch", spy), patch("newton._src.solvers.phoenx.solver_phoenx.wp.launch", spy):
        world.step(dt=step_dt, contacts=None, shape_body=None)
    return captured


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX launch-shape tests require CUDA")
class TestPhoenXFastTailLaunchGeometry(unittest.TestCase):
    """Warp-aligned launch geometry across every main-solve kernel."""

    def _assert_fast_tail_geometry(self, captured: list[dict], world) -> None:
        # Filter to just the kernels this invariant covers.
        main_bounds = _main_solve_kernel_launch_bounds(world)
        main = [c for c in captured if c["kernel"] in main_bounds]
        self.assertGreater(
            len(main),
            0,
            msg=("did not observe any main-solve kernel launch during step() -- test plumbing broken"),
        )
        expected_block_dim = world._fast_tail_block_dim()
        wpb = expected_block_dim // int(_STRAGGLER_BLOCK_DIM)
        for c in main:
            launch_bound = main_bounds[c["kernel"]]
            raw_dim = world.num_worlds * int(launch_bound)
            expected_dim = ((raw_dim + expected_block_dim - 1) // expected_block_dim) * expected_block_dim
            kernel_name = c["kernel"]
            block_dim = c["block_dim"]
            dim = c["dim"]
            self.assertEqual(
                block_dim,
                expected_block_dim,
                msg=(
                    f"{kernel_name}: block_dim={block_dim} != "
                    f"{expected_block_dim} (_STRAGGLER_BLOCK_DIM={_STRAGGLER_BLOCK_DIM}"
                    f" x wpb={wpb})"
                ),
            )
            self.assertEqual(
                dim,
                expected_dim,
                msg=(
                    f"{kernel_name}: dim={dim} != padded num_worlds "
                    f"({world.num_worlds}) * tpw launch bound ({launch_bound}) "
                    f"rounded up to multiple of {expected_block_dim} = {expected_dim}"
                ),
            )

    def test_one_world(self) -> None:
        """Single-world baseline: block_dim = 32 * wpb, dim padded up."""
        world, _ = _build_n_pendulums(num_worlds=1)
        captured = _collect_launches(world)
        self._assert_fast_tail_geometry(captured, world)

    def test_eight_worlds(self) -> None:
        """Every main-solve kernel uses the heuristic world-group block size."""
        world, _ = _build_n_pendulums(num_worlds=8)
        captured = _collect_launches(world)
        self._assert_fast_tail_geometry(captured, world)

    def test_sixtyfour_worlds(self) -> None:
        """Scales cleanly at 64 worlds too (catches a hypothetical
        max_blocks clamp regression)."""
        world, _ = _build_n_pendulums(num_worlds=64)
        captured = _collect_launches(world)
        self._assert_fast_tail_geometry(captured, world)

    def test_all_main_kernels_observed(self) -> None:
        """Fused prepare+iterate and relax must both appear under the
        default configuration. Missing one would mean a kernel got
        renamed or the dispatcher skipped it."""
        world, _ = _build_n_pendulums(num_worlds=4)
        captured = _collect_launches(world)
        kernels_seen = {c["kernel"] for c in captured}
        # Prepare + iterate always run when there are active
        # constraints. Relax runs when velocity_iterations > 0 (our
        # test harness uses velocity_iterations=1).
        expected = set(_main_solve_kernel_launch_bounds(world))
        self.assertTrue(
            expected.issubset(kernels_seen),
            msg=("fused prepare+iterate and relax kernels should both run when constraints are active"),
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX block-world capture tests require CUDA")
class TestPhoenXBlockWorldGraphCapture(unittest.TestCase):
    """The opt-in block-world scheduler must be fixed before capture."""

    def test_forced_block_world_capture_replay(self) -> None:
        world, _ = _build_n_pendulums(num_worlds=4)
        world._multi_world_scheduler = "block_world"
        world._multi_world_block_dim = 64

        world.step(dt=1.0 / 60.0, contacts=None, shape_body=None)
        wp.synchronize_device(world.device)

        with wp.ScopedCapture(device=world.device) as capture:
            world.step(dt=1.0 / 60.0, contacts=None, shape_body=None)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(world.device)

        self.assertTrue(np.isfinite(world.bodies.position.numpy()).all())


if __name__ == "__main__":
    unittest.main()
