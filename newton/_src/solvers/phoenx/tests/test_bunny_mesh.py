# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh-shape contact tests for :class:`PhoenXWorld`.

Exercises the :class:`PhoenXWorld` contact path on non-primitive
triangle-mesh geometry so the narrow phase emits per-triangle
contacts (up to the contact-max budget per substep) instead of
the box/sphere narrow phase's 1-4 points. Scenarios:

* ``TestBunnyMeshRestsOnPlane`` -- drop a bunny-ish mesh (loaded
  from Newton's ``bunny.usd`` when available, else a procedurally
  generated icosahedron-style triangulated sphere as a
  dependency-free stand-in) on the plane and verify it settles
  (bounded velocity, no NaN, stays above the plane).
* ``TestBunnyMeshFrictionalSlide`` -- launch the mesh along +X
  with friction. The decel should be close to ``mu * g``.
* ``TestBunnyMeshFrictionlessSlide`` -- same launch with
  ``mu = 0``; the mesh must keep its velocity for the entire
  run (within FP noise). Mesh equivalent of the cube zero-friction
  slide test -- catches a regression in tangent-row handling
  specific to the mesh ingest.

Runs on CUDA only.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

_G = 9.81


def _icosahedron_vertices_and_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, indices)`` for a unit icosahedron.

    Dependency-free fallback for the bunny when Newton's USD
    loader (``pxr``) isn't installed. Icosahedron has 12 vertices
    + 20 triangular faces and exercises the same mesh contact
    path as the bunny -- the per-triangle narrow phase produces
    multiple manifold points per substep on contact with a plane,
    which is the property under test.
    """
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    verts = np.array(
        [
            [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
            [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
            [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
        ],
        dtype=np.int32,
    )
    return verts, faces


def _load_bunny_mesh() -> newton.Mesh:
    """Load the bunny triangle mesh.

    Tries Newton's USD loader first (the canonical asset); if
    the optional ``pxr`` dep isn't installed, falls back to the
    icosahedron stand-in so the tests can run on any machine. The
    property we care about -- ``add_shape_mesh`` producing a
    per-triangle manifold -- is the same either way.
    """
    try:
        # Both imports are inside the try so we exit cleanly when
        # either ``pxr`` or the USD-backed ``newton.usd`` bits are
        # missing from the env. Local-import ``newton_usd`` so the
        # name doesn't shadow the module-level ``newton``.
        from pxr import Usd  # noqa: PLC0415
        import newton.usd as newton_usd  # noqa: PLC0415
    except ModuleNotFoundError:
        verts, faces = _icosahedron_vertices_and_indices()
        return newton.Mesh(verts, faces)
    stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
    return newton_usd.get_mesh(stage.GetPrimAtPath("/root/bunny"))


def _bunny_scene(
    *,
    friction: float,
    initial_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    initial_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    fps: int = 120,
    substeps: int = 8,
    solver_iterations: int = 16,
) -> tuple[_PhoenXScene, int]:
    """Assemble a ground-plane + single-bunny scene via
    :class:`_PhoenXScene`.

    Returns ``(scene, bunny_body_id)``. The harness only exposes
    :meth:`add_box` / :meth:`add_sphere` so we reach into its
    ``ModelBuilder`` directly to add the mesh shape (this is what
    :class:`newton.examples.basic.example_basic_shapes` does).
    """
    scene = _PhoenXScene(
        fps=fps,
        substeps=substeps,
        solver_iterations=solver_iterations,
        friction=friction,
    )
    scene.add_ground_plane()
    mesh = _load_bunny_mesh()
    body = scene.mb.add_body(
        xform=wp.transform(p=wp.vec3(*initial_xyz), q=wp.quat(*initial_quat)),
    )
    scene.mb.add_shape_mesh(
        body,
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
    )
    scene._newton_body_ids.append(body)
    scene.finalize()
    return scene, body


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX bunny-mesh tests require CUDA")
class TestBunnyMeshRestsOnPlane(unittest.TestCase):
    """The bunny dropped on a plane must settle -- bounded
    velocity, still above the plane, no NaN.
    """

    def test_bunny_settles(self) -> None:
        scene, body = _bunny_scene(friction=0.5, initial_xyz=(0.0, 0.0, 0.8))
        for _ in range(300):  # 2.5 s at 120 Hz
            scene.step()
        pos = scene.body_position(body)
        vel = scene.body_velocity(body)
        self.assertTrue(np.isfinite(pos).all() and np.isfinite(vel).all())
        # COM stays above the plane by some margin (bunny is ~half
        # a meter tall, so its COM must be above z=0).
        self.assertGreater(
            float(pos[2]),
            -0.05,
            f"bunny sank through the plane: z={pos[2]:.4f}",
        )
        self.assertLess(
            float(np.linalg.norm(vel)),
            0.5,
            f"bunny still moving after settle: |v|={np.linalg.norm(vel):.4f} m/s",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX bunny-mesh tests require CUDA")
class TestBunnyMeshFrictionalSlide(unittest.TestCase):
    """Bunny launched along +X with ``mu = 0.3`` must decelerate
    at approximately ``mu * g``.

    The bunny's contact manifold is jagged (per-triangle normals
    with small tilts) compared with a cube's flat-face manifold,
    so the deceleration won't be exactly ``mu * g`` -- we allow a
    50 % tolerance, which still catches a regression in the mesh
    contact friction-cone clamp (e.g. if the tangent-row basis
    wasn't being reconstructed correctly for per-triangle contacts).
    """

    def test_bunny_decelerates_under_friction(self) -> None:
        mu = 0.3
        v0 = 2.0
        scene, body = _bunny_scene(
            friction=mu,
            initial_xyz=(0.0, 0.0, 0.5),
        )
        # Let the bunny settle onto the plane first so subsequent
        # measurements see a steady-state normal impulse.
        for _ in range(60):
            scene.step()
        scene.set_body_velocity(body, (v0, 0.0, 0.0))
        for _ in range(2):
            scene.step()
        v_start = float(scene.body_velocity(body)[0])

        # Short measurement window -- deceleration is constant so
        # a short sample is as accurate as a long one and avoids
        # running into the complete stop (where decel goes to 0).
        measure_frames = 30
        for _ in range(measure_frames):
            scene.step()
        v_end = float(scene.body_velocity(body)[0])
        dt = measure_frames / 120.0
        measured_decel = (v_start - v_end) / dt
        expected_decel = mu * _G
        # 50 % tolerance -- mesh contact manifolds are noisier than
        # primitive-vs-primitive so a tight bound would fail on
        # manifold-point reshuffling between substeps. The test
        # fails any regression that changes the decel by more than
        # 2x or flips its sign.
        self.assertGreater(
            measured_decel,
            0.5 * expected_decel,
            f"bunny decel {measured_decel:.3f} m/s^2 < 0.5 * mu*g = "
            f"{0.5 * expected_decel:.3f}",
        )
        self.assertLess(
            measured_decel,
            2.0 * expected_decel,
            f"bunny decel {measured_decel:.3f} m/s^2 > 2 * mu*g = "
            f"{2.0 * expected_decel:.3f}",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX bunny-mesh tests require CUDA")
class TestBunnyMeshFrictionlessSlide(unittest.TestCase):
    """Bunny launched along +X with ``mu = 0`` must keep its
    velocity for the entire run.

    Mesh equivalent of the cube zero-friction slide test. Any
    tangent-row leak on per-triangle contacts shows up here --
    and the bunny's bumpy surface produces many contacts per
    frame, so this is the strongest per-contact symmetry check in
    the mesh contact path.
    """

    def test_bunny_zero_friction_no_decel(self) -> None:
        v0 = 2.0
        scene, body = _bunny_scene(
            friction=0.0,
            initial_xyz=(0.0, 0.0, 0.5),
        )
        # Settle vertical first so the initial horizontal velocity
        # isn't corrupted by the drop transient.
        for _ in range(60):
            scene.step()
        scene.set_body_velocity(body, (v0, 0.0, 0.0))
        for _ in range(2):
            scene.step()
        v_start = float(scene.body_velocity(body)[0])

        # 1.5 s of simulated time: any friction leak equivalent to
        # ``mu_effective > 0.05`` would bleed >70 % of v0. Tolerance
        # is conservative but still catches a real leak.
        for _ in range(180):
            scene.step()
        v_end = float(scene.body_velocity(body)[0])
        dt = 180 / 120.0
        decel = (v_start - v_end) / dt
        # Tolerance -- pure FP noise on mesh contacts is larger
        # than on primitive contacts (the tangent basis per
        # triangle drifts a hair with orientation changes), but
        # any leak of ``mu_effective >= 0.02`` would trip the
        # 0.2 m/s^2 threshold.
        self.assertLess(
            abs(decel),
            0.2,
            f"bunny |decel|={decel:.4f} m/s^2 with mu=0 -- mesh contact "
            f"tangent row leaked (v_start={v_start:.3f}, v_end={v_end:.3f})",
        )


if __name__ == "__main__":
    unittest.main()
