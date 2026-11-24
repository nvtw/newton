# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic low-iteration Coulomb stopping, force balance, and contact release."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_FRICTION, _OFF_FRICTION_DYNAMIC
from newton.solvers import SolverBase, SolverPhoenX


class TestFrictionSliding(unittest.TestCase):
    def test_stopping_force_balance_and_release(self):
        """Stop sliding cubes by Coulomb friction and release without adhesive forces.

        Cover kinetic/static coefficient separation, several initial speeds,
        diagonal slip, and an external horizontal force. Cubes remain below
        the tipping threshold; their normal-force center must balance friction
        torque. Reuse settled contacts for release to expose stale impulses.
        """
        gravity, half, inertia = 9.81, 0.25, 1.0 / 24.0
        mu = np.array([0.2, 0.2, 0.2, 0.5])
        mu_static = np.array([0.2, 0.5, 0.2, 0.5])
        speeds = np.array([0.5, 1.0, 2.0, 1.0])
        directions = np.array([[1, 0], [1, 0], [0.6, 0.8], [1, 0]])
        applied = np.array([0.0, 0.5, 0.0, 0.0])
        deceleration = mu * gravity - applied
        expected_times = speeds / deceleration
        expected_distances = speeds**2 / (2 * deceleration)

        def run(dt):
            builder = newton.ModelBuilder()
            for i in range(len(mu)):
                cfg = builder.ShapeConfig(density=0, mu=mu[i], mu_static=mu_static[i], margin=0, gap=0.001)
                body = builder.add_body(
                    xform=wp.transform((0, 3 * i, half + 0.0001), wp.quat_identity()),
                    mass=1.0,
                    inertia=wp.mat33(np.eye(3) * inertia),
                )
                builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((0, 3 * i, -0.05), wp.quat_identity()),
                    hx=4,
                    hy=1.2,
                    hz=0.05,
                    cfg=cfg,
                )
            model = builder.finalize(device="cpu")
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
            contacts = pipeline.contacts()
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                substeps=1,
                solver_iterations=4,
                velocity_iterations=1,
                sleeping_velocity_threshold=0.0,
            )
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            state, control = model.state(), model.control()
            shape_body = model.shape_body.numpy()

            def step(forces=None):
                state.clear_forces()
                if forces is not None:
                    state.body_f.assign(forces)
                pipeline.collide(state, contacts, dt=dt)
                solver.step(state, state, control, contacts, dt, observables=observables)
                pose, velocity = state.body_q.numpy().copy(), state.body_qd.numpy().copy()
                count = int(contacts.rigid_contact_count.numpy()[0])
                wrench = observables.contact_f.numpy()[:count]
                force = wrench[:, :3]
                normal = contacts.rigid_contact_normal.numpy()[:count]
                body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
                body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
                normal_force = -np.sum(force * normal, axis=1)
                tangent = force + normal_force[:, None] * normal
                self.assertGreaterEqual(float(normal_force.min(initial=0)), -1.0e-5)
                self.assertLessEqual(
                    float(
                        (np.linalg.norm(tangent, axis=1) - mu_static[np.maximum(body0, body1)] * normal_force).max(
                            initial=0
                        )
                    ),
                    2.0e-4,
                )
                owners = np.concatenate((body0, body1))
                active = owners >= 0
                owners = owners[active]
                forces_on_body = np.concatenate((force, -force))[active]
                local = np.concatenate(
                    (contacts.rigid_contact_point0.numpy()[:count], contacts.rigid_contact_point1.numpy()[:count])
                )[active]

                def world_lever(points, bodies):
                    # Body origins are the COMs; static points are already in world space.
                    quat = np.where(bodies[:, None] >= 0, pose[bodies, 3:], [0.0, 0.0, 0.0, 1.0])
                    return points + 2 * np.cross(quat[:, :3], np.cross(quat[:, :3], points) + quat[:, 3:] * points)

                lever = world_lever(local, owners)
                # The reported torque about body0 adds the column's torsional and
                # rolling friction moment to the point force's lever moment.
                lever0 = world_lever(contacts.rigid_contact_point0.numpy()[:count], body0)
                free = wrench[:, 3:] - np.cross(lever0, force)
                total, torque = np.zeros((4, 3)), np.zeros((4, 3))
                np.add.at(total, owners, forces_on_body)
                np.add.at(torque, owners, np.cross(lever, forces_on_body) + np.concatenate((free, -free))[active])
                return pose, velocity, total, torque

            for _ in range(round(0.5 / dt)):
                step()
            # Check the actual contact coefficients, not just the requested
            # shape configuration: the public bridge must preserve mu_s != mu_k.
            count = int(contacts.rigid_contact_count.numpy()[0])
            body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
            body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
            slots = solver.world._cid_of_contact_cur.numpy()[:count] - solver.world._contact_offset
            active = slots >= 0
            self.assertTrue(np.any(active))
            owners = np.maximum(body0, body1)[active]
            columns = solver.world._contact_cols.data.numpy()
            np.testing.assert_allclose(columns[int(_OFF_FRICTION), slots[active]], mu_static[owners])
            np.testing.assert_allclose(columns[int(_OFF_FRICTION_DYNAMIC), slots[active]], mu[owners])
            initial = state.body_q.numpy().copy()
            velocity = np.zeros((4, 6), dtype=np.float32)
            velocity[:, :2] = speeds[:, None] * directions
            state.body_qd.assign(velocity)
            forces = np.zeros_like(velocity)
            forces[:, :2] = applied[:, None] * directions
            samples = [step(forces) for _ in range(math.ceil((max(expected_times) + 0.6) / dt))]
            poses, velocities, contact_forces, torques = (np.array(values) for values in zip(*samples, strict=True))
            times = dt * (np.arange(len(samples)) + 1)
            projected = np.einsum("tij,ij->ti", velocities[:, :, :2], directions)
            distance = np.einsum("tij,ij->ti", poses[:, :, :2] - initial[None, :, :2], directions)

            previous = np.concatenate((velocity[None], velocities[:-1]))
            measured_force = (velocities[:, :, :3] - previous[:, :, :3]) / dt
            measured_force[:, :, 2] += gravity
            np.testing.assert_allclose(contact_forces + forces[None, :, :3], measured_force, atol=2.0e-3)
            np.testing.assert_allclose(torques, inertia * (velocities[:, :, 3:] - previous[:, :, 3:]) / dt, atol=3.0e-3)
            stopping_errors = np.empty(4)
            for i in range(4):
                with self.subTest(case=i):
                    sliding = (times > 2 * dt) & (times < expected_times[i] - 2 * dt)
                    expected_force = np.array(
                        [-mu[i] * gravity * directions[i, 0], -mu[i] * gravity * directions[i, 1], gravity]
                    )
                    np.testing.assert_allclose(
                        contact_forces[sliding, i],
                        np.broadcast_to(expected_force, contact_forces[sliding, i].shape),
                        atol=0.03,
                    )
                    # A balanced cube needs a normal-force center offset
                    # mu * half toward its direction of travel. Allow
                    # 1 mm of pressure-center error at four iterations.
                    np.testing.assert_allclose(torques[sliding, i], 0, atol=gravity * 0.001)
                    self.assertLess(float(np.linalg.norm(velocities[sliding, i, 3:], axis=1).max()), 0.05)
                    stop = np.flatnonzero(projected[:, i] <= 0.1 * deceleration[i] * dt)
                    self.assertGreater(len(stop), 0)
                    self.assertAlmostEqual(times[stop[0]], expected_times[i], delta=2 * dt)
                    # Semi-implicit position integration has O(v0 * dt)
                    # bias; keep that explicit instead of a percent fudge.
                    self.assertAlmostEqual(distance[stop[0], i], expected_distances[i], delta=speeds[i] * dt + 1.0e-4)
                    stopping_errors[i] = abs(distance[stop[0], i] - expected_distances[i])
                    rest = times > expected_times[i] + 0.1
                    self.assertLess(float(np.ptp(distance[rest, i])), 0.001)
                    self.assertLess(abs(float(distance[-1, i] - distance[stop[0], i])), 0.0025)
                    energy = (
                        0.5 * np.sum(velocities[:, i, :3] ** 2, axis=1)
                        + 0.5 * inertia * np.sum(velocities[:, i, 3:] ** 2, axis=1)
                        + gravity * (poses[:, i, 2] - initial[i, 2])
                        - applied[i] * distance[:, i]
                    )
                    # Permit work equivalent to 10 micrometres of vertical
                    # contact correction, not macroscopic energy growth.
                    self.assertLess(float(np.diff(energy).max()), gravity * 1.0e-5)
                    self.assertLess(float(energy.max()), 0.5 * speeds[i] ** 2)

            # Release from the established contact history using force,
            # not a pose edit or collision disable. No normal/tangent pull.
            forces.fill(0)
            forces[:, 0], forces[:, 2] = 1.0, 2 * gravity
            last_velocity = velocities[-1, :, :3]
            for _ in range(round(0.2 / dt)):
                _, released_velocity, contact_force, _ = step(forces)
                np.testing.assert_allclose(contact_force, 0, atol=1.0e-5)
                np.testing.assert_allclose(
                    (released_velocity[:, :3] - last_velocity) / dt,
                    np.broadcast_to((1.0, 0.0, gravity), (4, 3)),
                    atol=1.0e-3,
                )
                last_velocity = released_velocity[:, :3]
            return stopping_errors

        errors = []
        for dt in (1.0 / 120.0, 1.0 / 240.0):
            with self.subTest(dt=dt), wp.ScopedDevice("cpu"):
                errors.append(run(dt))
        if len(errors) == 2:
            # Halving h halves the leading Euler position error. The final
            # stopping step contributes only an O(a * h^2) remainder.
            self.assertTrue(np.all(errors[1] <= 0.5 * errors[0] + 2 * deceleration / 120**2 + 1.0e-5))


if __name__ == "__main__":
    unittest.main()
