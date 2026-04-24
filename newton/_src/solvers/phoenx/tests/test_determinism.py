# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact determinism tests for :class:`PhoenXWorld`.

Two independent :class:`_PhoenXScene` instances built from the same
construction recipe must produce bit-identical body state after the
same number of steps. Newton's :class:`CollisionPipeline` is
deterministic when ``contact_matching != DISABLED`` (implied by
:data:`PHOENX_CONTACT_MATCHING`) and the Jones-Plassmann partitioner
seeds its priorities with a fixed seed, so every downstream solver
step should also be deterministic.

Scenes scale from trivial (a single free-falling box) to progressively
more contact-heavy pyramids. If one scene passes but a larger one
fails, the failure tells us roughly where in the contact / solver
pipeline non-determinism sneaks in.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


def _build_free_fall_scene() -> _PhoenXScene:
    """Two boxes falling in free space -- no contacts, pure gravity +
    velocity integration. The cheapest determinism smoke test."""
    scene = _PhoenXScene(fps=60, substeps=4, solver_iterations=4)
    scene.add_box(position=(0.0, 0.0, 2.0), half_extents=(0.1, 0.1, 0.1))
    scene.add_box(position=(0.3, 0.0, 2.0), half_extents=(0.1, 0.1, 0.1))
    scene.finalize()
    return scene


def _build_box_on_plane_scene() -> _PhoenXScene:
    """Single unit box settling on a ground plane -- the simplest
    non-trivial contact scene. One shape pair, a handful of contact
    points."""
    scene = _PhoenXScene(fps=60, substeps=8, solver_iterations=8)
    scene.add_ground_plane()
    scene.add_box(position=(0.0, 0.0, 0.3), half_extents=(0.1, 0.1, 0.1))
    scene.finalize()
    return scene


def _build_pyramid_scene(layers: int) -> _PhoenXScene:
    """A ``layers``-layer square pyramid on a ground plane.

    Layer 0 (bottom) has ``layers * layers`` boxes; each layer above
    has one fewer box per side. Total box count is
    ``sum(k^2 for k in 1..layers)`` -- 1, 5, 14, 30, 55 for 1..5
    layers. Adjacent boxes touch on a single face + one side each, so
    every layer's settled contact graph has many same-body-pair
    columns and exercises the Jones-Plassmann colouring under load.
    """
    scene = _PhoenXScene(fps=60, substeps=10, solver_iterations=8)
    scene.add_ground_plane()

    half_extent = 0.1
    edge = 2.0 * half_extent
    # Small gap so bodies don't spawn intersecting (hard contacts on
    # initial penetration can inject nondeterministic ordering from the
    # narrow phase's write-cursor atomics; a clean spawn avoids that
    # altogether and keeps the scene focused on the solver itself).
    spawn_gap = 0.002

    for k in range(layers):
        side = layers - k
        base_z = (k + 0.5) * edge + k * spawn_gap
        # Centre each layer in world XY.
        offset = -0.5 * (side - 1) * edge
        for row in range(side):
            for col in range(side):
                x = offset + col * edge
                y = offset + row * edge
                scene.add_box(
                    position=(x, y, base_z),
                    half_extents=(half_extent, half_extent, half_extent),
                )
    scene.finalize()
    return scene


def _snapshot(scene: _PhoenXScene) -> dict[str, np.ndarray]:
    """Device-to-host copy of every body-state buffer the solver
    writes. Returns a dict so the failing assertion can name which
    field diverged first."""
    b = scene.world.bodies
    return {
        "position": b.position.numpy().copy(),
        "orientation": b.orientation.numpy().copy(),
        "velocity": b.velocity.numpy().copy(),
        "angular_velocity": b.angular_velocity.numpy().copy(),
        "inverse_inertia_world": b.inverse_inertia_world.numpy().copy(),
    }


def _assert_bit_exact(
    case: unittest.TestCase,
    ref: dict[str, np.ndarray],
    dup: dict[str, np.ndarray],
    scene_name: str,
    frame: int,
) -> None:
    """Fail with a pointed message if any field differs by a single
    bit. Reports the per-field max abs diff + the worst-offender body
    index so the failure says where to look instead of just *that* it
    diverged."""
    for field, a in ref.items():
        b = dup[field]
        if np.array_equal(a, b):
            continue
        max_diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        # ``argmax`` on a flattened diff array gives the linear index
        # of the worst offender; unravel to a body index + component.
        flat_diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
        worst = int(np.argmax(flat_diff))
        case.fail(
            f"{scene_name}: frame {frame} {field!r} diverged between runs -- "
            f"max |delta|={max_diff:.3e}, flat_index={worst} "
            f"(shape={a.shape})"
        )


def _run_and_compare(
    case: unittest.TestCase,
    build: callable,
    *,
    scene_name: str,
    frames: int,
) -> None:
    """Build the scene twice, step both ``frames`` times, and check
    that every writable body-state field matches bit-exactly after
    each step.

    Per-frame checks (not just final state) narrow down which step
    first diverges when a future regression breaks determinism.
    """
    ref_scene = build()
    dup_scene = build()

    # Initial states must already match before any stepping; if they
    # don't, the builder itself is nondeterministic (e.g. a host-side
    # random seed drifted), and blaming the solver would be wrong.
    _assert_bit_exact(
        case, _snapshot(ref_scene), _snapshot(dup_scene), scene_name, frame=0
    )

    for f in range(1, frames + 1):
        ref_scene.step()
        dup_scene.step()
        _assert_bit_exact(
            case, _snapshot(ref_scene), _snapshot(dup_scene), scene_name, frame=f
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX determinism tests run on CUDA only (graph capture + deterministic "
    "collision pipeline require CUDA).",
)
class TestPhoenXDeterminism(unittest.TestCase):
    """Two solver instances from the same scene recipe must produce
    bit-identical body state every step."""

    def test_free_fall_no_contacts(self) -> None:
        """Smallest test: two boxes in free space. If this fails, a
        non-contact kernel (integrate-forces / integrate-gravity /
        integrate-positions / update-inertia) is nondeterministic."""
        _run_and_compare(
            self, _build_free_fall_scene, scene_name="free_fall", frames=30
        )

    def test_single_box_on_plane(self) -> None:
        """Next-smallest: one box settling on the ground plane.
        Exercises single-pair contact ingest + PGS; if this fails but
        free-fall passes, the culprit is in ingest / contact prepare /
        contact iterate."""
        _run_and_compare(
            self,
            _build_box_on_plane_scene,
            scene_name="box_on_plane",
            frames=60,
        )

    def test_pyramid_2_layers(self) -> None:
        """5 boxes (1 top + 2x2 base)."""
        _run_and_compare(
            self,
            lambda: _build_pyramid_scene(2),
            scene_name="pyramid_2",
            frames=60,
        )

    def test_pyramid_3_layers(self) -> None:
        """14 boxes (1 + 2x2 + 3x3). First scene with a nontrivial
        graph-colouring problem -- several contacts share bodies."""
        _run_and_compare(
            self,
            lambda: _build_pyramid_scene(3),
            scene_name="pyramid_3",
            frames=60,
        )

    def test_pyramid_4_layers(self) -> None:
        """30 boxes (1 + 4 + 9 + 16). Dense colouring + deep stack."""
        _run_and_compare(
            self,
            lambda: _build_pyramid_scene(4),
            scene_name="pyramid_4",
            frames=60,
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
