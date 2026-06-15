# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parity test for the single-world tail-fused PGS path.

The fused-tail path (see :data:`newton._src.solvers.phoenx.solver_config.
FUSE_TAIL_MAX_COLOR_SIZE`) is a pure optimisation: it swaps the
per-colour kernel-launch boundary at the tail of each single-world PGS
sweep for a single fused kernel that uses ``__syncthreads`` between
colours.  Because the graph-coloring invariant already forbids two
same-colour cids from touching the same body, the switch must not
change any body state.

This test runs the same three-layer mini-tower scene twice -- once
with the fused-tail path disabled and once with it enabled at the
shipped ``FUSE_TAIL_BLOCK_DIM`` -- and asserts that every body's
final position / orientation / linear velocity / angular velocity
matches to a tight 1e-6 tolerance.  The mini-tower has a
contact-dominated colour histogram with a long tail of small colours,
so the fused-tail path sees real work; if the sync barrier, the
hand-off predicate in the head kernel, or the cursor resume in the
fused kernel were wrong, the tower wobble would diverge well outside
the tolerance within the 60-frame window.

Parity here is stronger than needed for correctness (``1e-6`` on
float32 state is near bit-exact) but is the right guarantee for a
purely algebraic refactor.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver_config import FUSE_TAIL_BLOCK_DIM
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


def _build_mini_tower_scene(fuse_threshold: int) -> _PhoenXScene:
    """Build the three-layer circular tower from
    :meth:`test_stacking.test_mini_circular_tower`, pinned to the
    single-world step layout the fused-tail path targets.

    ``fuse_threshold`` overrides the solver's
    :attr:`PhoenXWorld._fuse_threshold` *after* construction but
    *before* the CUDA graph is captured on the first
    :meth:`_PhoenXScene.step` call, so the value bakes into the
    captured graph.
    """
    scene = _PhoenXScene(
        fps=120,
        substeps=20,
        solver_iterations=3,
        velocity_iterations=1,
        step_layout="single_world",
    )
    scene.add_ground_plane()

    tower_height_layers = 3
    boxes_per_ring = 6
    plank_hx = 1.5
    plank_hy = 0.1
    plank_hz = 0.5
    ring_radius = 3.5
    full_rotation_step = 2.0 * math.pi / boxes_per_ring
    half_rotation_step = 0.5 * full_rotation_step

    orientation_rad = 0.0
    for e in range(tower_height_layers):
        orientation_rad += half_rotation_step
        for _ in range(boxes_per_ring):
            cos_o = math.cos(orientation_rad)
            sin_o = math.sin(orientation_rad)
            local_y = ring_radius
            world_x = -sin_o * local_y
            world_y = cos_o * local_y
            world_z = plank_hz + e * 2.0 * plank_hz
            half = 0.5 * orientation_rad
            quat = (0.0, 0.0, math.sin(half), math.cos(half))
            scene.add_box(
                position=(float(world_x), float(world_y), float(world_z)),
                half_extents=(plank_hx, plank_hy, plank_hz),
                orientation=quat,
            )
            orientation_rad += full_rotation_step

    scene.finalize()

    scene.world._fuse_threshold = int(fuse_threshold)
    scene.world._fuse_tail_block_dim = int(FUSE_TAIL_BLOCK_DIM)
    return scene


def _collect_body_state(scene: _PhoenXScene) -> dict[str, np.ndarray]:
    """Snapshot the full rigid-body state as NumPy arrays, skipping
    the static world anchor in slot 0 so the comparison only
    considers dynamic bodies."""
    return {
        "position": scene.bodies.position.numpy()[1:].copy(),
        "orientation": scene.bodies.orientation.numpy()[1:].copy(),
        "velocity": scene.bodies.velocity.numpy()[1:].copy(),
        "angular_velocity": scene.bodies.angular_velocity.numpy()[1:].copy(),
    }


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX solver requires CUDA for graph-captured stepping")
class TestPhoenXTailFuseParity(unittest.TestCase):
    """End-to-end bit-similar parity between the fused-tail-disabled
    (``FUSE_TAIL_MAX_COLOR_SIZE = 0``) and fused-tail-enabled
    (``FUSE_TAIL_MAX_COLOR_SIZE = FUSE_TAIL_BLOCK_DIM``) single-world
    PGS paths.
    """

    def test_mini_tower_parity(self) -> None:
        """Same 60 frames of a 3-ring, 6-plank circular tower, run
        with and without the fused-tail path.

        ``atol = 1e-6`` is within float32 precision for this scene;
        the fused path is a pure kernel-boundary refactor and should
        produce identical arithmetic.  If the head / tail hand-off or
        the inter-colour sync is broken, the tower geometry would
        diverge long before this tolerance.
        """
        num_frames = 60

        scene_off = _build_mini_tower_scene(fuse_threshold=0)
        for _ in range(num_frames):
            scene_off.step()
        state_off = _collect_body_state(scene_off)

        scene_on = _build_mini_tower_scene(fuse_threshold=FUSE_TAIL_BLOCK_DIM)
        for _ in range(num_frames):
            scene_on.step()
        state_on = _collect_body_state(scene_on)

        for key in ("position", "orientation", "velocity", "angular_velocity"):
            np.testing.assert_allclose(
                state_on[key],
                state_off[key],
                atol=1e-6,
                rtol=0.0,
                err_msg=f"tail-fuse parity mismatch on {key}",
            )


if __name__ == "__main__":
    unittest.main()
