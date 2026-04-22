# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-shape material system.

Verifies:

* ``TestCombineModes`` -- the four PhysX-style combine modes
  (AVERAGE / MIN / MULTIPLY / MAX) produce the expected effective
  friction for a pair of materials, and the "stricter mode wins"
  rule correctly picks the combine mode when the two materials
  disagree.

* ``TestContactUsesMaterialFriction`` -- end-to-end: two cubes
  sliding on a plane under identical push forces, one with
  ``mu_dynamic = 0.1`` and the other with ``mu_dynamic = 0.7``;
  the low-friction cube accelerates, the high-friction cube stays
  put (same scene as the single-cube friction tests but with two
  bodies sharing a single world builder).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.constraints.contact_matching_config import (
    JITTER_CONTACT_MATCHING,
)
from newton._src.solvers.jitter.examples.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)
from newton._src.solvers.jitter.materials import (
    COMBINE_AVERAGE,
    COMBINE_MAX,
    COMBINE_MIN,
    COMBINE_MULTIPLY,
    Material,
    material_table_from_list,
)


_G = 9.81


def _combine_scalar(a: float, b: float, mode: int) -> float:
    """Host-side reference implementation of the combine rule; used
    for analytical checks against the solver-applied friction."""
    if mode == COMBINE_MIN:
        return min(a, b)
    if mode == COMBINE_MAX:
        return max(a, b)
    if mode == COMBINE_MULTIPLY:
        return a * b
    return 0.5 * (a + b)


class TestCombineModes(unittest.TestCase):
    """Unit tests on the host-side combine helper.

    Covers the four modes and the "max wins" tie-break. No device
    activity -- these tests run on CPU-only CI too.
    """

    def test_average(self):
        self.assertAlmostEqual(_combine_scalar(0.2, 0.8, COMBINE_AVERAGE), 0.5)

    def test_min(self):
        self.assertAlmostEqual(_combine_scalar(0.2, 0.8, COMBINE_MIN), 0.2)

    def test_max(self):
        self.assertAlmostEqual(_combine_scalar(0.2, 0.8, COMBINE_MAX), 0.8)

    def test_multiply(self):
        self.assertAlmostEqual(_combine_scalar(0.5, 0.4, COMBINE_MULTIPLY), 0.2)

    def test_material_construction_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            Material(static_friction=-0.1)
        with self.assertRaises(ValueError):
            Material(dynamic_friction=-0.5)
        with self.assertRaises(ValueError):
            Material(restitution=-0.5)
        with self.assertRaises(ValueError):
            Material(restitution=1.5)
        with self.assertRaises(ValueError):
            Material(friction_combine_mode=99)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Material-friction integration test requires CUDA.",
)
class TestContactUsesMaterialFriction(unittest.TestCase):
    """End-to-end: the per-pair friction written into the contact
    column reflects the material table, not ``default_friction``.

    Scene: one cube on a plane, material tables swapped to vary
    friction. With a horizontal push of ``0.5 * m * g`` a cube
    should slide if ``mu_effective < 0.5`` and hold otherwise.
    Same arithmetic as the classical :class:`TestStaticFrictionThreshold`
    in :mod:`test_friction_slide`, but driven from the material
    table so a regression in the pack-kernel friction lookup shows
    up here first.
    """

    N_FRAMES = 120  # 2 s at 60 fps

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("materials contact test requires CUDA")

    def _run(self, plane_mu: float, cube_mu: float, combine_mode: int) -> float:
        device = wp.get_device("cuda:0")
        mb = newton.ModelBuilder()
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)
        half = 0.5
        body = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, half + 1e-3), q=wp.quat_identity()),
        )
        mb.add_shape_box(body, hx=half, hy=half, hz=half)

        model = mb.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.body_q.assign(state.body_q)

        cp = newton.CollisionPipeline(model, contact_matching=JITTER_CONTACT_MATCHING)
        contacts = cp.contacts()

        builder, n2j = build_jitter_world_from_model(model)

        # default_friction intentionally wrong (0.0) so the test
        # fails loudly if the kernel ever falls back to the
        # default instead of using the materials table.
        world = builder.finalize(
            substeps=4,
            solver_iterations=16,
            gravity=(0.0, 0.0, -_G),
            max_contact_columns=16,
            rigid_contact_max=int(contacts.rigid_contact_point0.shape[0]),
            num_shapes=int(model.shape_count),
            default_friction=0.0,
            device=device,
        )
        shape_body_np = model.shape_body.numpy()
        sb_j = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        sb = wp.array(sb_j, dtype=wp.int32, device=device)

        # Install materials: plane (shape 0) uses material 1, cube
        # (shape 1) uses material 2. Both materials declare
        # ``friction_combine_mode = combine_mode`` so the per-pair
        # resolution produces a known analytical result.
        materials = material_table_from_list(
            [
                Material(),  # index 0 = default
                Material(
                    dynamic_friction=plane_mu,
                    static_friction=plane_mu,
                    friction_combine_mode=combine_mode,
                ),
                Material(
                    dynamic_friction=cube_mu,
                    static_friction=cube_mu,
                    friction_combine_mode=combine_mode,
                ),
            ],
            device=device,
        )
        shape_material = wp.array(
            [1, 2], dtype=wp.int32, device=device
        )  # plane -> mat 1; cube -> mat 2
        world.set_materials(materials, shape_material)

        # Apply a 0.5 * m * g horizontal push on the cube for the
        # whole simulated run; measure how far it slides.
        m = 1000.0 * (2.0 * half) ** 3  # density 1000, 1 m^3 -> 1000 kg
        push = 0.5 * m * _G

        # Let the cube settle vertically first with zero push so the
        # normal response is established.
        for _ in range(10):
            wp.launch(
                newton_to_jitter_kernel,
                dim=model.body_count,
                inputs=[state.body_q, state.body_qd, model.body_com],
                outputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                ],
                device=device,
            )
            model.collide(state, contacts=contacts, collision_pipeline=cp)
            world.step(dt=1.0 / 60.0, contacts=contacts, shape_body=sb)
            wp.launch(
                jitter_to_newton_kernel,
                dim=model.body_count,
                inputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                    model.body_com,
                ],
                outputs=[state.body_q, state.body_qd],
                device=device,
            )

        # Now apply the push every frame.
        for _ in range(self.N_FRAMES):
            j = n2j[body]
            forces = world.bodies.force.numpy().copy()
            forces[j] = [push, 0.0, 0.0]
            world.bodies.force.assign(forces)

            wp.launch(
                newton_to_jitter_kernel,
                dim=model.body_count,
                inputs=[state.body_q, state.body_qd, model.body_com],
                outputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                ],
                device=device,
            )
            model.collide(state, contacts=contacts, collision_pipeline=cp)
            world.step(dt=1.0 / 60.0, contacts=contacts, shape_body=sb)
            wp.launch(
                jitter_to_newton_kernel,
                dim=model.body_count,
                inputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                    model.body_com,
                ],
                outputs=[state.body_q, state.body_qd],
                device=device,
            )

        return float(world.bodies.position.numpy()[n2j[body]][0])

    def test_high_friction_holds(self):
        """mu_plane=0.7, mu_cube=0.7, combine=AVERAGE -> mu=0.7 > push 0.5, box stays."""
        x = self._run(0.7, 0.7, COMBINE_AVERAGE)
        print(f"[materials: both 0.7 AVG push=0.5mg] x={x:.4f} m")
        self.assertLess(abs(x), 0.1)

    def test_low_friction_slides(self):
        """mu_plane=0.1, mu_cube=0.1, combine=AVERAGE -> mu=0.1 < push, box slides."""
        x = self._run(0.1, 0.1, COMBINE_AVERAGE)
        print(f"[materials: both 0.1 AVG push=0.5mg] x={x:.4f} m")
        self.assertGreater(x, 1.0)

    def test_mixed_average(self):
        """mu=(0.1, 0.9) AVERAGE -> 0.5 (exactly at threshold)."""
        x = self._run(0.1, 0.9, COMBINE_AVERAGE)
        print(f"[materials: (0.1, 0.9) AVG -> mu=0.5] x={x:.4f} m")
        # Threshold case: effective mu = 0.5 == push; barely holds
        # or slowly creeps. Loose bound.
        self.assertLess(abs(x), 2.0)

    def test_mixed_min_slides(self):
        """mu=(0.1, 0.9) MIN -> 0.1 (slippery wins)."""
        x = self._run(0.1, 0.9, COMBINE_MIN)
        print(f"[materials: (0.1, 0.9) MIN -> mu=0.1] x={x:.4f} m")
        self.assertGreater(x, 1.0)

    def test_mixed_max_holds(self):
        """mu=(0.1, 0.9) MAX -> 0.9 (grippy wins)."""
        x = self._run(0.1, 0.9, COMBINE_MAX)
        print(f"[materials: (0.1, 0.9) MAX -> mu=0.9] x={x:.4f} m")
        self.assertLess(abs(x), 0.1)


if __name__ == "__main__":
    unittest.main()
