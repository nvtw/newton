# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resting-stack stability of PhoenX contacts under few solver sweeps.

A short Kapla wall of planks standing on their 8 mm edges, solved like the
Kapla arena (4 substeps, 4 iterations, 1 relax iteration). With the contact
spring at the Nyquist limit the unconverged sweeps feed penetration
corrections into the planks' rocking modes, and the resting wall jitters at
about 1 cm/s; a stable resting state stays below 1 mm/s.

A separate CPU fixture checks heavy contents in a light container. Correct
mean support forces alone do not detect its persistent internal oscillation.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene
from newton.solvers import SolverPhoenX

#: Kapla plank half extents [m]: 8 mm thick, 100 mm long, 25 mm tall.
_PLANK = (0.004, 0.05, 0.0125)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact tests require CUDA graph capture")
class TestContactRestStability(unittest.TestCase):
    def test_edge_standing_plank_wall_stays_at_rest(self) -> None:
        """Six layers of crossing on-edge planks must not start rocking."""
        scene = _PhoenXScene(
            fps=120,
            substeps=4,
            solver_iterations=4,
            velocity_iterations=1,
            friction=0.5,
            step_layout="single_world",
        )
        scene.add_ground_plane()
        along_x = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        ids = []
        for layer in range(6):
            z = _PLANK[2] * (2 * layer + 1)
            if layer % 2 == 0:
                # Three rows of two planks along x.
                for y in (-0.046, 0.0, 0.046):
                    for x in (-0.05, 0.05):
                        ids.append(scene.add_box((x, y, z), _PLANK, orientation=along_x))
            else:
                # Four planks across the rows.
                for x in (-0.075, -0.025, 0.025, 0.075):
                    ids.append(scene.add_box((x, 0.0, z), _PLANK))
        scene.finalize()
        initial = np.array([scene.body_position(i) for i in ids])

        speeds = []
        for frame in range(240):
            scene.step()
            if frame >= 60 and frame % 20 == 0:
                velocity = np.array([scene.body_velocity(i) for i in ids])
                speeds.append(np.linalg.norm(velocity, axis=1))
        speeds = np.array(speeds)
        final = np.array([scene.body_position(i) for i in ids])

        self.assertLess(float(np.max(np.median(speeds, axis=1))), 1.0e-3)
        self.assertLess(float(np.max(speeds)), 5.0e-3)
        self.assertLess(float(np.max(np.linalg.norm(final - initial, axis=1))), 1.0e-3)


class TestPackedContainerRest(unittest.TestCase):
    def test_heavy_contents_settle_in_light_container(self):
        """Free cylinders in a movable light box must stop oscillating."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            length, width, height, thickness = 0.286, 0.216, 0.261, 0.0032
            case_mass = 0.42
            volume = length * width * thickness + 2 * thickness * (width + length - 2 * thickness) * (
                height - thickness
            )
            case_cfg = builder.ShapeConfig(
                density=case_mass / volume, mu=0.4, mu_static=0.45, gap=0.002, margin=0, restitution=0
            )
            case = builder.add_body(xform=wp.transform((0, 0, 0.001), wp.quat_identity()))
            builder.add_shape_box(
                case,
                xform=wp.transform((0, 0, thickness / 2), wp.quat_identity()),
                hx=length / 2,
                hy=width / 2,
                hz=thickness / 2,
                cfg=case_cfg,
            )
            # Disjoint shell pieces give the actual uniform-density COM and inertia.
            for axis in (0, 1):
                for sign in (-1, 1):
                    center = [0, 0, (height + thickness) / 2]
                    half = [length / 2, width / 2, (height - thickness) / 2]
                    center[axis] = sign * (half[axis] - thickness / 2)
                    half[axis] = thickness / 2
                    if axis == 1:
                        half[0] -= thickness
                    builder.add_shape_box(
                        case,
                        xform=wp.transform(center, wp.quat_identity()),
                        hx=half[0],
                        hy=half[1],
                        hz=half[2],
                        cfg=case_cfg,
                    )
            mass, radius, cylinder_height = 0.57, 0.0304, 0.25
            inertia = np.diag([mass * (3 * radius**2 + cylinder_height**2) / 12] * 2 + [mass * radius**2 / 2])
            cfg = builder.ShapeConfig(density=0, mu=0.4, mu_static=0.45, gap=0.002, margin=0, restitution=0)
            contents = []
            for x in range(4):
                for y in range(3):
                    position = ((x - 1.5) * 0.0687, (y - 1) * 0.0687, 0.001 + thickness + cylinder_height / 2 + 0.0001)
                    body = builder.add_body(
                        xform=wp.transform(position, wp.quat_identity()), mass=mass, inertia=wp.mat33(inertia)
                    )
                    builder.add_shape_cylinder(body, radius=radius, half_height=cylinder_height / 2, cfg=cfg)
                    contents.append(body)
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize(device="cpu")
            np.testing.assert_allclose(model.body_mass.numpy()[case], case_mass, rtol=1.0e-5)
            case_com = model.body_com.numpy()[case]
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(
                model, contact_matching="sticky", rigid_contact_max=4096, speculative_contact_gap_max=0.005
            )
            contacts = pipeline.contacts()
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                solver_iterations=24,
                temporal_iterations=True,
                substeps=1,
                velocity_iterations=1,
                step_layout="single_world",
                contact_hertz=80,
                enable_body_pair_grouping=True,
                sleeping_velocity_threshold=0,
            )
            positions, speeds = [], []
            for step in range(360):
                state.clear_forces()
                pipeline.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
                if step < 120:
                    continue
                poses, velocities = state.body_q.numpy(), state.body_qd.numpy()
                offset = poses[contents, :3] - poses[case, :3]
                rotation = poses[case, 3:]
                com_offset = np.asarray(wp.quat_rotate(wp.quat(rotation), wp.vec3(case_com)))
                relative_velocity = (
                    velocities[contents, :3]
                    - velocities[case, :3]
                    - np.cross(velocities[case, 3:], offset - com_offset)
                )
                inverse = rotation * np.array([-1, -1, -1, 1])
                local = offset + 2 * np.cross(inverse[:3], np.cross(inverse[:3], offset) + inverse[3] * offset)
                positions.append(local)
                speeds.append(np.sum(relative_velocity**2, axis=1))
            positions = np.asarray(positions)
            self.assertLess(float(np.sqrt(np.mean(speeds))), 1.0e-3, "persistent motion inside the resting container")
            self.assertLess(float(np.linalg.norm(positions - positions[0], axis=2).max()), 5.0e-4)


if __name__ == "__main__":
    unittest.main()
