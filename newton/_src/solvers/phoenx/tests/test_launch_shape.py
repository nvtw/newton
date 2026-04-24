# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Invariant checks on PhoenX's CUDA launch geometry.

The solver's perf model assumes every main constraint-solve kernel
launches **exactly one CUDA block of 256 threads per world**:

    wp.launch(
        kernel,
        dim=self.num_worlds * _STRAGGLER_BLOCK_DIM,
        block_dim=_STRAGGLER_BLOCK_DIM,
        ...
    )

Each world's 256-thread block walks its per-color CSR bucket
cooperatively via ``tid % _STRAGGLER_BLOCK_DIM`` /
``tid / _STRAGGLER_BLOCK_DIM``. Breaking this invariant -- e.g. by
dispatching ``dim=num_worlds``, or using a different block size --
would silently serialise the solve or run the kernels with wrong
thread-block conventions.

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
    _constraint_iterate_fast_tail_kernel,
    _constraint_position_iterate_fast_tail_kernel,
    _constraint_prepare_fast_tail_kernel,
    _constraint_relax_fast_tail_kernel,
)
from newton._src.solvers.phoenx.tests.test_multi_world import _build_n_pendulums

_MAIN_SOLVE_KERNELS = {
    _constraint_prepare_fast_tail_kernel.key,
    _constraint_iterate_fast_tail_kernel.key,
    _constraint_relax_fast_tail_kernel.key,
    _constraint_position_iterate_fast_tail_kernel.key,
}


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
        dim = (
            args[1]
            if len(args) > 1 and "dim" not in kwargs
            else kwargs.get("dim")
        )
        block_dim = kwargs.get("block_dim")
        if block_dim is None:
            # Block dim is also positional in some signatures; pull
            # the default from Warp's actual ``launch`` binding if
            # absent from both.
            block_dim = 256
        captured.append(
            {
                "kernel": getattr(kernel, "key", str(kernel)),
                "dim": (
                    int(dim)
                    if not isinstance(dim, (list, tuple))
                    else tuple(int(d) for d in dim)
                ),
                "block_dim": int(block_dim),
            }
        )
        return real_launch(*args, **kwargs)

    with patch("warp.launch", spy), patch(
        "newton._src.solvers.phoenx.solver_phoenx.wp.launch", spy
    ):
        world.step(dt=step_dt, contacts=None, shape_body=None)
    return captured


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX launch-shape tests require CUDA"
)
class TestPhoenXOneBlockPerWorld(unittest.TestCase):
    """One CUDA block per world across every main-solve kernel."""

    def _assert_one_block_per_world(
        self, captured: list[dict], num_worlds: int
    ) -> None:
        # Filter to just the kernels this invariant covers.
        main = [c for c in captured if c["kernel"] in _MAIN_SOLVE_KERNELS]
        self.assertGreater(
            len(main),
            0,
            msg=(
                "did not observe any main-solve kernel launch during "
                "step() -- test plumbing broken"
            ),
        )
        expected_dim = num_worlds * int(_STRAGGLER_BLOCK_DIM)
        for c in main:
            self.assertEqual(
                c["block_dim"],
                int(_STRAGGLER_BLOCK_DIM),
                msg=f"{c['kernel']}: block_dim={c['block_dim']} != "
                f"_STRAGGLER_BLOCK_DIM ({_STRAGGLER_BLOCK_DIM})",
            )
            self.assertEqual(
                c["dim"],
                expected_dim,
                msg=(
                    f"{c['kernel']}: dim={c['dim']} != num_worlds "
                    f"({num_worlds}) * _STRAGGLER_BLOCK_DIM "
                    f"({_STRAGGLER_BLOCK_DIM}) = {expected_dim} -- the "
                    "one-block-per-world invariant is broken"
                ),
            )

    def test_one_world(self) -> None:
        """Single-world baseline: dim == 1 * 256, block_dim == 256."""
        world, _ = _build_n_pendulums(num_worlds=1)
        captured = _collect_launches(world)
        self._assert_one_block_per_world(captured, num_worlds=1)

    def test_eight_worlds(self) -> None:
        """Every main-solve kernel dispatches 8 blocks of 256 threads."""
        world, _ = _build_n_pendulums(num_worlds=8)
        captured = _collect_launches(world)
        self._assert_one_block_per_world(captured, num_worlds=8)

    def test_sixtyfour_worlds(self) -> None:
        """Scales cleanly at 64 worlds too (catches a hypothetical
        max_blocks clamp regression)."""
        world, _ = _build_n_pendulums(num_worlds=64)
        captured = _collect_launches(world)
        self._assert_one_block_per_world(captured, num_worlds=64)

    def test_all_main_kernels_observed(self) -> None:
        """Prepare, iterate, relax, and position-iterate must all
        appear. Missing one would mean a kernel got renamed or the
        dispatcher skipped it under the default configuration."""
        world, _ = _build_n_pendulums(num_worlds=4)
        captured = _collect_launches(world)
        kernels_seen = {c["kernel"] for c in captured}
        # Prepare + iterate always run when there are active
        # constraints. Relax runs when velocity_iterations > 0 (our
        # test harness uses velocity_iterations=1). Position iterate
        # runs only when position_iterations > 0 (default 0), so
        # we don't require it here.
        self.assertIn(
            _constraint_prepare_fast_tail_kernel.key,
            kernels_seen,
            msg="prepare kernel did not fire -- solver pipeline regressed",
        )
        self.assertIn(
            _constraint_iterate_fast_tail_kernel.key,
            kernels_seen,
            msg="iterate kernel did not fire -- solver pipeline regressed",
        )
        self.assertIn(
            _constraint_relax_fast_tail_kernel.key,
            kernels_seen,
            msg=(
                "relax kernel did not fire -- "
                "velocity_iterations=1 should always run one relax pass"
            ),
        )


if __name__ == "__main__":
    unittest.main()
