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
from newton.examples.phoenx._bottling_control import _attach_sealed_flaps, _seal_cases
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

    def test_pressed_cylinder_field_comes_to_rest(self) -> None:
        """A packed field of standing bottles pressed into a corner must settle.

        Constant sideways gravity presses 144 bottle-sized cylinders into the
        corner of a pen. They touch along their sides, so every contact must
        support them against rocking along the whole side. Contacts collapsing
        to single points that move along the sides kept the field rocking.
        """
        radius, half_height, mass = 0.0305, 0.0606, 0.57
        builder = newton.ModelBuilder()
        glass = builder.ShapeConfig(density=0.0, mu=0.16, mu_static=0.2, gap=0.002)
        steel = builder.ShapeConfig(density=0.0, mu=0.18, mu_static=0.2, gap=0.001)
        pitch, count = 2.0 * radius, 12
        width = (count + 0.5) * pitch
        builder.add_shape_box(
            -1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.05, cfg=steel
        )
        builder.add_shape_box(
            -1, xform=wp.transform((0.01, 0.0, 0.06), wp.quat_identity()), hx=0.01, hy=1.0, hz=0.06, cfg=steel
        )
        for side in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform((-0.5, side * (0.5 * width + 0.01), 0.06), wp.quat_identity()),
                hx=0.5,
                hy=0.01,
                hz=0.06,
                cfg=steel,
            )
        bodies = []
        for row in range(count):
            for column in range(count):
                x = -radius - 0.001 - row * pitch * math.sqrt(0.75)
                y = -0.5 * width + radius + 0.001 + (column + 0.5 * (row % 2)) * pitch
                body = builder.add_body(
                    xform=wp.transform((x, y, 0.0005), wp.quat_identity()),
                    mass=mass,
                    com=wp.vec3(0.0, 0.0, 0.0834),
                    inertia=wp.mat33(np.diag((0.00184, 0.00184, 0.000261))),
                    lock_inertia=True,
                )
                builder.add_shape_cylinder(
                    body,
                    xform=wp.transform((0.0, 0.0, half_height), wp.quat_identity()),
                    radius=radius,
                    half_height=half_height,
                    cfg=glass,
                )
                bodies.append(body)
        model = builder.finalize("cuda:0")
        # Press the field into the pen's corner, below the bottles' tipping ratio.
        model.gravity.assign(np.array([[0.3 * 9.81, 0.0, -9.81]], dtype=np.float32))
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky", speculative_contact_gap_max=0.005)
        solver = SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            substeps=3,
            solver_iterations=4,
            temporal_iterations=True,
            contact_hertz=80.0,
            velocity_iterations=1,
            step_layout="single_world",
        )
        state, contacts = model.state(), pipeline.contacts()
        dt = 1.0 / 120.0
        with wp.ScopedCapture() as capture:
            pipeline.collide(state, contacts, dt=dt)
            solver.step(state, state, None, contacts, dt)
        energies = []
        for frame in range(720):
            wp.capture_launch(capture.graph)
            if frame >= 360 and frame % 6 == 0:
                qd = state.body_qd.numpy()[bodies]
                energies.append(
                    0.5 * mass * np.sum(qd[:, :3] ** 2)
                    + 0.5 * 0.00184 * np.sum(qd[:, 3:5] ** 2)
                    + 0.5 * 0.000261 * np.sum(qd[:, 5] ** 2)
                )
        # A settled field keeps about 1 uJ; the rocking field kept 0.1 mJ.
        self.assertLess(float(np.mean(energies)), 2.0e-5)


class TestPackedContainerRest(unittest.TestCase):
    def test_heavy_contents_settle_in_light_container(self):
        """Free cylinders in a movable light box must stop oscillating."""
        self._check_packed_rest("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Overflow-only dispatch requires CUDA")
    def test_temporal_overflow_contents_settle_cuda(self):
        """Keep a light packed carton stable with twelve colors and real overflow."""
        self._check_packed_rest("cuda:0", overflow_colors=12)

    def _check_packed_rest(self, device, *, overflow_colors=None):
        with wp.ScopedDevice(device):
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
            model = builder.finalize(device=device)
            np.testing.assert_allclose(model.body_mass.numpy()[case], case_mass, rtol=1.0e-5)
            case_com = model.body_com.numpy()[case]
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(
                model, contact_matching="sticky", rigid_contact_max=4096, speculative_contact_gap_max=0.005
            )
            contacts = pipeline.contacts()
            options = {}
            if overflow_colors is not None:
                options = {
                    "mass_splitting": True,
                    "mass_splitting_overflow_only": True,
                    "mass_splitting_unrolled": True,
                    "mass_splitting_batch_size": 1,
                    "max_colored_partitions": overflow_colors,
                    "partitioner_algorithm": "endpoint_owner",
                }
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
                **options,
            )
            graph = None
            if model.device.is_cuda:
                with wp.ScopedCapture() as capture:
                    state.clear_forces()
                    pipeline.collide(state, contacts, dt=1 / 120)
                    solver.step(state, state, control, contacts, 1 / 120)
                graph = capture.graph
            positions, speeds = [], []
            for step in range(360):
                if graph is None:
                    state.clear_forces()
                    pipeline.collide(state, contacts, dt=1 / 120)
                    solver.step(state, state, control, contacts, 1 / 120)
                else:
                    wp.capture_launch(graph)
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
            if overflow_colors is not None:
                self.assertGreater(solver.step_report().overflow_size, 0)
                starts = solver.world._partitioner.color_starts.numpy()
                self.assertGreater(int(starts[overflow_colors] - starts[0]), 0)


def _stack_bottle_hull(zs, radii, n=21):
    vertices = [
        (r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n), z)
        for z, r in zip(zs, radii, strict=True)
        for k in range(n)
    ]
    indices = []
    for ring in range(len(zs) - 1):
        for k in range(n):
            a = ring * n + k
            b = ring * n + (k + 1) % n
            c = (ring + 1) * n + k
            d = (ring + 1) * n + (k + 1) % n
            indices.extend([a, b, c, b, d, c])
    for k in range(1, n - 1):
        indices.extend([0, k + 1, k, (len(zs) - 1) * n, (len(zs) - 1) * n + k, (len(zs) - 1) * n + k + 1])
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32), compute_inertia=False
    )


_STACK_BOTTLE_PROFILES = [
    ([0, 0.0015, 0.016], [0.025605, 0.028304, 0.030505]),
    ([0.016, 0.12125, 0.12725], [0.030505, 0.030410, 0.029434]),
    ([0.12725, 0.13275, 0.15075], [0.029434, 0.027541, 0.019202]),
    ([0.151, 0.161, 0.17125], [0.019179, 0.018575, 0.017726]),
    ([0.1715, 0.1845, 0.21025], [0.017702, 0.016251, 0.012738]),
    ([0.21025, 0.2125, 0.2135], [0.012738, 0.014204, 0.014273]),
    ([0.21375, 0.2265, 0.2285], [0.014256, 0.013138, 0.011153]),
    ([0.2234, 0.2297], [0.01605, 0.013803]),
]


class TestSealedCartonStack(unittest.TestCase):
    def test_two_filled_cartons_retain_contents_and_tape_bonds(self):
        """A sealed loaded carton must support another without breaking its fixed bonds.

        Geometry is procedural; dimensions, masses and inertias represent the factory
        carton. Thin flaps use local hinges until the production sealing kernel retires
        them and creates runtime tape joints. Each bottle is free and uses eight convex radial sections.
        Sleeping is disabled and collision runs at 120 Hz with 24 temporal sweeps.
        """
        with wp.ScopedDevice("cpu"):
            b = newton.ModelBuilder()
            length, width, height, t = 0.28575, 0.2159, 0.26035, 0.003175
            cfg = b.ShapeConfig(density=0, mu=0.4, mu_static=0.45, gap=0.002, margin=0, restitution=0)
            groups = []
            hinges = []
            bottle_meshes = [_stack_bottle_hull(zs, radii) for zs, radii in _STACK_BOTTLE_PROFILES]
            for layer in range(2):
                z = 0.001 + layer * (height + 0.00018)
                root = b.add_link(
                    xform=wp.transform((0, 0, z), wp.quat_identity()),
                    mass=0.35,
                    com=wp.vec3(0, 0, 0.102318),
                    inertia=wp.mat33(np.diag([0.00434844, 0.00555476, 0.00514938])),
                    lock_inertia=True,
                )
                for sign in (-1, 1):
                    b.add_shape_box(
                        root,
                        xform=wp.transform((0, sign * 0.05435, t / 2), wp.quat_identity()),
                        hx=0.1392,
                        hy=0.0536,
                        hz=t / 2,
                        cfg=cfg,
                    )
                    b.add_shape_box(
                        root,
                        xform=wp.transform((sign * 0.0876875, 0, 1.5 * t), wp.quat_identity()),
                        hx=0.0520125,
                        hy=0.103775,
                        hz=t / 2,
                        cfg=cfg,
                    )
                    b.add_shape_box(
                        root,
                        xform=wp.transform((0, sign * 0.0351896, 0.1285875), wp.quat_identity()),
                        hx=0.1392,
                        hy=0.00079375,
                        hz=0.1222375,
                        cfg=cfg,
                    )
                for x in (-0.0702469, 0, 0.0702469):
                    for y in (-0.0701292, 0, 0.0701292):
                        b.add_shape_box(
                            root,
                            xform=wp.transform((x, y, 0.1285875), wp.quat_identity()),
                            hx=0.00079375,
                            hy=0.0343958 if y == 0 else 0.0341458,
                            hz=0.1222375,
                            cfg=cfg,
                        )
                for axis in (0, 1):
                    for sign in (-1, 1):
                        p = [0, 0, height / 2 if axis == 0 else 0.130175]
                        half = [length / 2, width / 2, height / 2 if axis == 0 else 0.127]
                        p[axis] = sign * (half[axis] - t / 2)
                        half[axis] = t / 2
                        if axis == 1:
                            half[0] -= t
                        if axis == 0:
                            # The factory decomposes these end walls into convex slabs.
                            mesh = newton.Mesh.create_box(*half, duplicate_vertices=False, compute_inertia=False)
                            b.add_shape_convex_hull(root, xform=wp.transform(p, wp.quat_identity()), mesh=mesh, cfg=cfg)
                        else:
                            b.add_shape_box(
                                root,
                                xform=wp.transform(p, wp.quat_identity()),
                                hx=half[0],
                                hy=half[1],
                                hz=half[2],
                                cfg=cfg,
                            )
                ids = [root]
                joints = [b.add_joint_free(child=root)]
                for axis in (1, 0):
                    for sign in (1, -1):
                        major = axis == 1
                        half = [0.1392, 0.0536, t / 2] if major else [0.0520125, 0.103775, t / 2]
                        p = [0, 0, height - t]
                        p[axis] = sign * ([length, width][axis] / 2 - t)
                        local = [0, 0, t / 2 if major else -t / 2]
                        local[axis] = -sign * (0.050425 if major else 0.0520125)
                        mass = 0.02 if major else 0.015
                        inertia = (
                            mass
                            / 3
                            * np.diag(
                                [half[1] ** 2 + half[2] ** 2, half[0] ** 2 + half[2] ** 2, half[0] ** 2 + half[1] ** 2]
                            )
                        )
                        flap = b.add_link(
                            xform=wp.transform((p[0], p[1], z + p[2]), wp.quat_identity()),
                            mass=mass,
                            com=wp.vec3(local),
                            inertia=wp.mat33(inertia),
                            lock_inertia=True,
                        )
                        b.add_shape_box(
                            flap,
                            xform=wp.transform(local, wp.quat_identity()),
                            hx=half[0],
                            hy=half[1],
                            hz=half[2],
                            cfg=cfg,
                        )
                        ids.append(flap)
                        hinge_axis = (1, 0, 0) if major else (0, 1, 0)
                        angle = (-sign * np.deg2rad(118)) if major else (sign * np.deg2rad(100))
                        lo, hi = (-np.deg2rad(150), -angle) if angle < 0 else (-angle, np.deg2rad(150))
                        hinge = b.add_joint_revolute(
                            root,
                            flap,
                            parent_xform=wp.transform(p, wp.quat_from_axis_angle(wp.vec3(hinge_axis), float(angle))),
                            axis=hinge_axis,
                            target_ke=0.0458366,
                            target_kd=0.0040107,
                            limit_lower=lo,
                            limit_upper=hi,
                        )
                        joints.append(hinge)
                        hinges.append(hinge)
                b.add_articulation(joints)
                mass = 0.57
                for x in range(4):
                    for y in range(3):
                        body = b.add_link(
                            xform=wp.transform(((x - 1.5) * 0.0703, (y - 1) * 0.0703, z + 0.0067), wp.quat_identity()),
                            mass=mass,
                            com=wp.vec3(0, 0, 0.083439),
                            inertia=wp.mat33(np.diag([0.00184135, 0.00184135, 0.00026124])),
                            lock_inertia=True,
                        )
                        for mesh in bottle_meshes:
                            b.add_shape_convex_hull(
                                body,
                                mesh=mesh,
                                cfg=b.ShapeConfig(
                                    density=0, mu=0.16, mu_static=0.2, gap=0.002, margin=0, restitution=0.05
                                ),
                            )
                        b.add_articulation([b.add_joint_free(child=body)])
                        ids.append(body)
                # Match the source carton self-collision scope: only its shell
                # and flaps are excluded, never any of the free bottles.
                shell_shapes = [i for i, body in enumerate(b.shape_body) if body in ids[:5]]
                for index, first in enumerate(shell_shapes):
                    for second in shell_shapes[index + 1 :]:
                        if b.shape_body[first] != b.shape_body[second]:
                            b.add_shape_collision_filter_pair(first, second)
                groups.append(ids)
            b.add_ground_plane(cfg=cfg)
            b.reserve_dynamic_capacity(joints=8)
            model = b.finalize()
            state = model.state()
            newton.eval_ik(model, state, model.joint_q, model.joint_qd)
            solver = SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                contact_hertz=80,
                velocity_iterations=1,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                direct_joint_mask=np.zeros(model.joint_count, dtype=bool),
                enable_body_pair_grouping=True,
                sleeping_velocity_threshold=0,
            )
            state = model.state()
            control = model.control()
            contacts = solver._collision_pipeline.contacts()
            q = model.joint_q.numpy()
            targets = control.joint_target_q.numpy()
            qs = model.joint_q_start.numpy()
            ts = model.joint_target_q_start.numpy()
            for j in hinges:
                targets[ts[j]] = q[qs[j]]
            control.joint_target_q.assign(targets)
            bodies = wp.array(np.asarray(groups)[:, :5], dtype=int)
            joint_ids = wp.full((2, 4), -1, dtype=int)
            requests = wp.ones(2, dtype=int)
            sealed = wp.zeros(2, dtype=int)
            hinge_ids = wp.array(np.asarray(hinges).reshape(2, 4), dtype=int)
            wp.launch(
                _seal_cases,
                dim=2,
                inputs=[
                    solver.dynamics.data,
                    state.body_q,
                    state.body_qd,
                    bodies,
                    requests,
                    sealed,
                    joint_ids,
                    hinge_ids,
                    model.joint_enabled,
                ],
            )
            np.testing.assert_array_equal(sealed.numpy(), [1, 1])
            # As in the factory, sealed flap shapes move onto their case body.
            collide = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
            shape_body = model.shape_body.numpy()
            flap_shapes = [
                [int(shape) for flap in ids[1:5] for shape in np.flatnonzero((shape_body == flap) & collide)]
                for ids in groups
            ]
            wp.launch(
                _attach_sealed_flaps,
                dim=2,
                inputs=[
                    state.body_q,
                    bodies,
                    sealed,
                    wp.array(flap_shapes, dtype=int),
                    wp.zeros(2, dtype=int),
                    model.shape_body,
                    model.shape_transform,
                    solver._shape_body,
                ],
            )
            np.testing.assert_array_equal(
                model.shape_body.numpy()[np.ravel(flap_shapes)], np.repeat(bodies.numpy()[:, 0], 4)
            )
            self.assertFalse(model.joint_enabled.numpy()[hinges].any())
            self.assertTrue(np.all(joint_ids.numpy() >= 0))
            initial = state.body_q.numpy().copy()
            resting = []
            for step in range(240):
                state.clear_forces()
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
                poses = state.body_q.numpy()
                self.assertTrue(np.isfinite(poses).all(), f"nonfinite stacked carton at step {step}")
                self.assertTrue(np.isfinite(state.body_qd.numpy()).all())
                local_positions = []
                for ids in groups:
                    inverse = wp.transform_inverse(wp.transform(poses[ids[0], :3], poses[ids[0], 3:]))
                    initial_inverse = wp.transform_inverse(wp.transform(initial[ids[0], :3], initial[ids[0], 3:]))
                    for flap in ids[1:5]:
                        before = wp.transform_multiply(
                            initial_inverse, wp.transform(initial[flap, :3], initial[flap, 3:])
                        )
                        after = wp.transform_multiply(inverse, wp.transform(poses[flap, :3], poses[flap, 3:]))
                        self.assertLess(
                            np.linalg.norm(np.asarray(before.p) - np.asarray(after.p)),
                            0.001,
                            f"fixed tape bond stretches under stack load at step {step}",
                        )
                        cosine = abs(float(np.dot(np.asarray(before.q), np.asarray(after.q))))
                        self.assertLess(
                            2 * math.acos(min(1.0, cosine)),
                            math.radians(1),
                            f"sealed flap rotates under stack load at step {step}",
                        )
                    for bottle in ids[5:]:
                        center = wp.transform_point(
                            wp.transform(poses[bottle, :3], poses[bottle, 3:]), wp.vec3(0, 0, 0.083439)
                        )
                        local = np.asarray(wp.transform_point(inverse, center))
                        self.assertLess(abs(local[0]), length / 2)
                        self.assertLess(abs(local[1]), width / 2)
                        self.assertGreater(local[2], 0)
                        self.assertLess(local[2], height)
                        local_positions.append(local)
                if step >= 120:
                    resting.append(local_positions)
            for ids in groups:
                root = ids[0]
                self.assertLess(
                    np.max(np.abs(poses[root, :3] - initial[root, :3])), 0.005, "cartons leave their stacked positions"
                )
                cosine = abs(float(np.dot(poses[root, 3:], initial[root, 3:])))
                self.assertLess(2 * math.acos(min(1.0, cosine)), math.radians(3), "stacked carton tips over")
            resting = np.asarray(resting)
            self.assertLess(
                np.linalg.norm(resting - resting[0], axis=2).max(), 0.0005, "contents keep moving after settling"
            )
            # Position-derived velocity avoids end-of-substep velocity sampling artifacts.
            self.assertLess(np.sqrt(np.mean(np.sum((np.diff(resting, axis=0) * 120) ** 2, axis=2))), 0.005)


if __name__ == "__main__":
    unittest.main()
