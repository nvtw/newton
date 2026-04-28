# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Invariant checks on PhoenX's CUDA launch geometry.

The multi-world solver's perf model assumes the substep mega-kernel
launches **exactly one CUDA block per world**:

    wp.launch_tiled(
        _phoenx_substep_megakernel,
        dim=[num_worlds],
        block_dim=_PHOENX_SUBSTEP_BLOCK_DIM,
        ...
    )

Each world's block grid-strides over both its bodies (via the
body-per-world CSR) and its constraint elements (via the per-world
colour CSR). The whole substep loop -- forces, solve, integrate,
relax, inertia, kinematic, damping -- runs in this single launch.

Breaking this invariant -- e.g. dispatching ``dim=num_worlds * lanes``
or splitting the substep loop back across many launches -- would
silently regress throughput. These tests hook
:func:`warp.launch_tiled` for the duration of a single
:meth:`PhoenXWorld.step` call, capture ``(kernel_name, dim,
block_dim)`` tuples, and assert the invariant. CUDA only.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import warp as wp

from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _PHOENX_SUBSTEP_BLOCK_DIM,
    _phoenx_substep_megakernel,
)
from newton._src.solvers.phoenx.tests.test_multi_world import _build_n_pendulums


def _collect_launches(world, *, step_dt: float = 1.0 / 60.0) -> list[dict]:
    """Run one ``world.step(step_dt)`` while capturing every
    :func:`wp.launch` and :func:`wp.launch_tiled` call's kernel +
    dim + block_dim.

    Returns the captured list of dicts. The patched launchers still
    forward to the real implementations so the step produces its
    normal GPU-side effects; we only observe the launch shape.
    """
    captured: list[dict] = []
    real_launch = wp.launch
    real_launch_tiled = wp.launch_tiled

    def _record(kernel, dim, block_dim, kind):
        captured.append(
            {
                "kernel": getattr(kernel, "key", str(kernel)),
                "dim": (int(dim) if not isinstance(dim, (list, tuple)) else tuple(int(d) for d in dim)),
                "block_dim": int(block_dim),
                "kind": kind,
            }
        )

    def spy_launch(*args, **kwargs):
        kernel = args[0] if args else kwargs.get("kernel")
        dim = args[1] if len(args) > 1 and "dim" not in kwargs else kwargs.get("dim")
        block_dim = kwargs.get("block_dim", 256)
        _record(kernel, dim, block_dim, "launch")
        return real_launch(*args, **kwargs)

    def spy_launch_tiled(*args, **kwargs):
        kernel = args[0] if args else kwargs.get("kernel")
        dim = args[1] if len(args) > 1 and "dim" not in kwargs else kwargs.get("dim")
        block_dim = kwargs.get("block_dim", 256)
        _record(kernel, dim, block_dim, "launch_tiled")
        return real_launch_tiled(*args, **kwargs)

    with (
        patch("warp.launch", spy_launch),
        patch("warp.launch_tiled", spy_launch_tiled),
        patch("newton._src.solvers.phoenx.solver_phoenx.wp.launch", spy_launch),
        patch("newton._src.solvers.phoenx.solver_phoenx.wp.launch_tiled", spy_launch_tiled),
    ):
        world.step(dt=step_dt, contacts=None, shape_body=None)
    return captured


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX launch-shape tests require CUDA")
class TestPhoenXOneBlockPerWorld(unittest.TestCase):
    """The multi-world substep mega-kernel must launch exactly one
    block of :data:`_PHOENX_SUBSTEP_BLOCK_DIM` threads per world."""

    def _assert_megakernel_shape(self, captured: list[dict], num_worlds: int) -> None:
        mega = [c for c in captured if c["kernel"] == _phoenx_substep_megakernel.key]
        self.assertEqual(
            len(mega),
            1,
            msg=f"expected exactly one mega-kernel launch per step, got {len(mega)}",
        )
        c = mega[0]
        self.assertEqual(
            c["block_dim"],
            int(_PHOENX_SUBSTEP_BLOCK_DIM),
            msg=f"mega-kernel block_dim={c['block_dim']} != _PHOENX_SUBSTEP_BLOCK_DIM={_PHOENX_SUBSTEP_BLOCK_DIM}",
        )
        # ``launch_tiled`` carries the world count as ``dim=[num_worlds]``.
        expected_dim = (num_worlds,)
        self.assertEqual(
            c["dim"],
            expected_dim,
            msg=f"mega-kernel dim={c['dim']} != one-block-per-world {expected_dim}",
        )
        self.assertEqual(
            c["kind"],
            "launch_tiled",
            msg="mega-kernel must dispatch via wp.launch_tiled (block-indexed)",
        )

    def test_one_world(self) -> None:
        world, _ = _build_n_pendulums(num_worlds=1)
        captured = _collect_launches(world)
        self._assert_megakernel_shape(captured, num_worlds=1)

    def test_eight_worlds(self) -> None:
        world, _ = _build_n_pendulums(num_worlds=8)
        captured = _collect_launches(world)
        self._assert_megakernel_shape(captured, num_worlds=8)

    def test_sixtyfour_worlds(self) -> None:
        world, _ = _build_n_pendulums(num_worlds=64)
        captured = _collect_launches(world)
        self._assert_megakernel_shape(captured, num_worlds=64)

    def test_substep_loop_is_one_launch(self) -> None:
        """At ``substeps > 1`` the entire loop must still resolve to
        exactly one mega-kernel launch -- the loop runs *inside* the
        kernel."""
        world, _ = _build_n_pendulums(num_worlds=4)
        # ``_build_n_pendulums`` hard-codes substeps=1 via a module
        # constant; widen the loop on this instance to exercise the
        # in-kernel substep walk. The mega-kernel reads ``substeps``
        # off ``self`` at launch time.
        world.substeps = 4
        captured = _collect_launches(world)
        mega_launches = [c for c in captured if c["kernel"] == _phoenx_substep_megakernel.key]
        self.assertEqual(
            len(mega_launches),
            1,
            msg=(
                f"expected exactly one mega-kernel launch even at substeps=4, "
                f"got {len(mega_launches)} -- substep loop should not fan out across launches"
            ),
        )


if __name__ == "__main__":
    unittest.main()
