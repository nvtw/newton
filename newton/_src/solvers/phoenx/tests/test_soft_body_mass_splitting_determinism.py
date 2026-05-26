# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact determinism for soft body + mass splitting.

Mirrors :mod:`test_cloth_mass_splitting_determinism` but for soft-tet
constraints. Two independently-built soft-cube-on-ground scenes,
both with mass splitting ON, must produce bit-identical particle and
body state after every captured step.

The soft-tet iterate path is a different code path from cloth-tri --
4-body XPBD on volume / shear strain instead of 3-body cloth elasticity
-- so a fix that only stabilises the cloth determinism could regress
soft-tet without this test. Built on top of the
``_SoftCubeMassSplittingScene`` helper used by the momentum test.

CUDA only, graph-captured.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_soft_body_mass_splitting_momentum import (
    _SoftCubeMassSplittingScene,
)

_N_FRAMES = 15  # impact happens around frame 6-8 at the test's drop height


def _snapshot(scene: _SoftCubeMassSplittingScene) -> dict[str, np.ndarray]:
    """Device-to-host copy of every solver-written buffer that a
    bit-exact regression would diverge on."""
    return {
        "particle_pos": scene.world.particles.position.numpy().copy(),
        "particle_vel": scene.world.particles.velocity.numpy().copy(),
        "particle_access_mode": scene.world.particles.access_mode.numpy().copy(),
        "body_pos": scene.bodies.position.numpy().copy(),
        "body_vel": scene.bodies.velocity.numpy().copy(),
    }


def _assert_bit_exact(case: unittest.TestCase, ref, dup, frame: int) -> None:
    for field, a in ref.items():
        b = dup[field]
        if np.array_equal(a, b):
            continue
        max_diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        flat_diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
        worst = int(np.argmax(flat_diff))
        case.fail(
            f"frame {frame}: {field!r} diverged between two scenes -- "
            f"max |delta|={max_diff:.3e}, flat_idx={worst} (shape={a.shape})"
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Soft-body mass-splitting determinism runs on CUDA only.",
)
class TestSoftBodyMassSplittingDeterminism(unittest.TestCase):
    """Two independently-built soft-tet + mass-splitting scenes must
    produce bit-identical state after every captured step."""

    def test_soft_cube_bit_exact(self) -> None:
        device = wp.get_preferred_device()
        ref = _SoftCubeMassSplittingScene(device)
        dup = _SoftCubeMassSplittingScene(device)

        # Post-warm-up / post-capture state must already match. If the
        # builder is non-deterministic the step-loop comparison would
        # blame the solver, which would be wrong.
        wp.synchronize_device(device)
        _assert_bit_exact(self, _snapshot(ref), _snapshot(dup), frame=0)

        for f in range(1, _N_FRAMES + 1):
            ref.step()
            dup.step()
            wp.synchronize_device(device)
            _assert_bit_exact(self, _snapshot(ref), _snapshot(dup), frame=f)


if __name__ == "__main__":
    wp.init()
    unittest.main()
