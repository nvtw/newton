# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate temporal contact integration, materials, friction and force accounting."""

from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverPhoenX


class TestTemporalIterationPhysics(unittest.TestCase):
    """Temporal iterations change the contact time step, not the physics."""

    @staticmethod
    def _solver(model, temporal, pipeline=None, iterations=8):
        return SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            substeps=1,
            solver_iterations=iterations,
            velocity_iterations=1,
            step_layout="single_world",
            joint_mode="maximal_pgs",
            contact_hertz=80,
            sleeping_velocity_threshold=0,
            temporal_iterations=temporal,
        )

    def test_free_fall_matches_analytic(self):
        """A falling body integrates the average of its contact substeps' velocities."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0, 0, 10.0), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
            # A resting box elsewhere keeps the contact substeps running.
            rest = builder.add_body(xform=wp.transform((5, 0, 0.1), wp.quat_identity()))
            builder.add_shape_box(rest, hx=0.1, hy=0.1, hz=0.1)
            builder.add_ground_plane()
            model = builder.finalize(device="cpu")
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=64)
            contacts = pipeline.contacts()
            solver = self._solver(model, True, pipeline)
            for _ in range(120):
                pipeline.collide(state, contacts, dt=1 / 240)
                solver.step(state, state, control, contacts, 1 / 240)
            t = 0.5
            self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), -9.81 * t, delta=1.0e-3)
            drop = 10.0 - float(state.body_q.numpy()[0, 2])
            # Semi-implicit Euler at 1/240 s overshoots by dt/t = 0.8%; the substep average by 0.1%.
            self.assertAlmostEqual(drop / (0.5 * 9.81 * t * t), 1.0, delta=2.0e-3)

    def test_kinematic_mount_isolates_local_drive(self):
        """A prescribed support transmits no impulse between its weld and its slider."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            mount = builder.add_link(is_kinematic=True)
            slider = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
            base = builder.add_joint_fixed(-1, mount)
            drive = builder.add_joint_prismatic(mount, slider, axis=newton.Axis.Z, target_ke=1000, target_kd=100)
            builder.add_articulation([base, drive])
            model = builder.finalize()
            solver = self._solver(model, True)
            state = model.state()
            control = model.control()
            for _ in range(240):
                solver.step(state, state, control, None, 1 / 120)
            np.testing.assert_array_equal(solver._direct_equality_system._microstep_joint.numpy(), [False, True])
            self.assertAlmostEqual(float(state.body_q.numpy()[slider, 2]), -9.81 / 1000, delta=2e-5)
            self.assertLess(abs(float(state.body_qd.numpy()[slider, 2])), 1e-4)

    def test_resting_contact_force_balances_weight(self):
        """Balance a resting body weight with accumulated contact force."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=builder.ShapeConfig(density=1000))
            builder.add_ground_plane()
            model = builder.finalize(device="cpu")
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=64)
            contacts = pipeline.contacts()
            solver = self._solver(model, True, pipeline)
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            forces = []
            for step in range(240):
                state.clear_forces()
                pipeline.collide(state, contacts, dt=1 / 240)
                solver.step(state, state, control, contacts, 1 / 240, observables=observables)
                if step >= 120:
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    forces.append(abs(float(observables.contact_f.numpy()[:count, 2].sum())))
            weight = float(model.body_mass.numpy()[0]) * 9.81
            self.assertAlmostEqual(float(np.mean(forces)) / weight, 1.0, delta=1.0e-4)
            self.assertLess(abs(float(state.body_q.numpy()[0, 2]) - 0.1), 1.0e-3)

    def test_pendulum_energy_matches_iterations(self):
        """A swinging revolute pendulum keeps the energy behavior of plain iterations."""

        def energy_history(temporal):
            with wp.ScopedDevice("cpu"):
                builder = newton.ModelBuilder()
                link = builder.add_link(xform=wp.transform((0.5, 0, 2.0), wp.quat_identity()))
                builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05, cfg=builder.ShapeConfig(density=1000))
                joint = builder.add_joint_revolute(
                    -1,
                    link,
                    parent_xform=wp.transform((0, 0, 2.0), wp.quat_identity()),
                    child_xform=wp.transform((-0.5, 0, 0), wp.quat_identity()),
                    axis=(0, 1, 0),
                )
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                state, control = model.state(), model.control()
                solver = self._solver(model, temporal)
                mass = float(model.body_mass.numpy()[0])
                inertia = model.body_inertia.numpy()[0]
                energies = []
                for _ in range(1200):
                    solver.step(state, state, control, None, 1 / 240)
                    q, qd = state.body_q.numpy()[0], state.body_qd.numpy()[0]
                    kinetic = 0.5 * mass * float(qd[:3] @ qd[:3]) + 0.5 * float(qd[3:] @ (inertia @ qd[3:]))
                    energies.append(kinetic + mass * 9.81 * float(q[2]))
                return np.asarray(energies)

        reference = energy_history(False)
        temporal = energy_history(True)
        initial = abs(reference[0])
        self.assertLess(float(np.max(np.abs(temporal - reference))) / initial, 1.0e-3)
        self.assertLessEqual(float(temporal[-1] - temporal[0]), 1.0e-6 * initial)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
class TestTemporalLayouts(unittest.TestCase):
    """Every dispatcher runs the substep's contact substeps."""

    LAYOUTS: ClassVar[dict[str, dict]] = {
        "multi_world_fast_tail": {"step_layout": "multi_world", "multi_world_scheduler": "fast_tail"},
        "multi_world_block_world": {"step_layout": "multi_world", "multi_world_scheduler": "block_world"},
        "multi_world_mass_splitting": {"step_layout": "multi_world", "mass_splitting": True},
        "single_world_mass_splitting": {"step_layout": "single_world", "mass_splitting": True},
        "single_world_mass_splitting_unrolled": {
            "step_layout": "single_world",
            "mass_splitting": True,
            "mass_splitting_unrolled": True,
        },
    }

    def test_forces_and_contacts_in_every_layout(self):
        """A free sphere receives all of gravity and a resting box's reported support balances its weight."""
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(density=1000)
            for _ in range(2):
                builder.begin_world()
                box = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
                builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                sphere = builder.add_body(xform=wp.transform((2, 0, 10.0), wp.quat_identity()))
                builder.add_shape_sphere(sphere, radius=0.1, cfg=cfg)
                builder.end_world()
            builder.add_ground_plane()
            model = builder.finalize()
            boxes, spheres = [0, 2], [1, 3]
            weight = float(model.body_mass.numpy()[0]) * 9.81
            dt, frames = 1 / 120, 60
            for name, layout in self.LAYOUTS.items():
                with self.subTest(layout=name):
                    pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=64)
                    contacts = pipeline.contacts()
                    solver = SolverPhoenX(
                        model,
                        collision_pipeline=pipeline,
                        substeps=2,
                        solver_iterations=8,
                        contact_hertz=60,
                        sleeping_velocity_threshold=0,
                        temporal_iterations=True,
                        **layout,
                    )
                    self.assertEqual(solver.world.step_layout, layout["step_layout"])
                    state, control = model.state(), model.control()
                    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)

                    def step(
                        state=state,
                        control=control,
                        contacts=contacts,
                        pipeline=pipeline,
                        solver=solver,
                        observables=observables,
                    ):
                        pipeline.collide(state, contacts, dt=dt)
                        solver.step(state, state, control, contacts, dt, observables=observables)

                    step()
                    with wp.ScopedCapture() as capture:
                        step()
                    for _ in range(frames - 1):
                        wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(state.body_qd.numpy()[spheres, 2], -9.81 * frames * dt, atol=1.0e-3)
                    np.testing.assert_allclose(state.body_q.numpy()[boxes, 2], 0.1, atol=1.0e-3)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    shape_body = model.shape_body.numpy()
                    body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
                    body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
                    force_z = observables.contact_f.numpy()[:count, 2]
                    support = [abs(float(force_z[body0 == b].sum() - force_z[body1 == b].sum())) for b in boxes]
                    np.testing.assert_allclose(support, weight, rtol=2.0e-3)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
class TestTemporalStiffMaterial(unittest.TestCase):
    def test_stiff_box_rests_when_substeps_reuse_rows(self):
        """A box on four stiff undamped corner springs keeps its resting pose.

        Substeps that reuse prepared rows advanced the gaps by the integrated motion on
        top of the motion the contact substeps had tracked, so the next substep saw
        twice the penetration and the box was thrown off (7.6 m/s after two seconds).
        """
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(density=2000, ke=1.0e8, kd=0.0, mu=0.7)
            body = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.2, hy=0.1, hz=0.1, cfg=cfg)
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize()
            solver = SolverPhoenX(
                model,
                step_layout="single_world",
                substeps=8,
                solver_iterations=8,
                temporal_iterations=True,
                sleeping_velocity_threshold=0,
            )
            self.assertGreater(solver.world.prepare_refresh_stride, 1)
            state, control = model.state(), model.control()
            contacts = model._collision_pipeline.contacts()

            def step():
                solver.collide(state, contacts, dt=1 / 60)
                solver.step(state, state, control, contacts, 1 / 60)

            step()
            with wp.ScopedCapture() as capture:
                step()
            for _ in range(118):
                wp.capture_launch(capture.graph)
            # The undamped springs ring at a few millimetres per second.
            self.assertLess(float(np.abs(state.body_qd.numpy()).max()), 0.02)
            self.assertAlmostEqual(float(state.body_q.numpy()[0, 2]), 0.1, delta=1.0e-4)


class _FirstSweepRecorded(Exception):
    """Stop after observing the real solver's first temporal sweep."""


class TestTemporalContactFriction(unittest.TestCase):
    @staticmethod
    def _first_impulse(gap: float, temporal: bool) -> np.ndarray:
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, 0.1 + gap), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.004),
            )
            cfg = builder.ShapeConfig(density=0, mu=0.5, mu_static=0.5, gap=0.004, margin=0, restitution=0)
            builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize(device="cpu")
            state = model.state()
            state.body_qd.assign(np.array([[1.0, 0.0, -8.0, 0.0, 0.0, 0.0]], dtype=np.float32))
            pipeline = newton.CollisionPipeline(
                model, contact_matching="sticky", rigid_contact_max=64, speculative_contact_gap_max=0.005
            )
            contacts = pipeline.contacts()
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                substeps=1,
                solver_iterations=16 if temporal else 1,
                velocity_iterations=0,
                step_layout="single_world",
                contact_hertz=80,
                sleeping_velocity_threshold=0,
                temporal_iterations=temporal,
            )
            saved = []
            if temporal:

                def record_first_sweep(last):
                    saved.append(solver.world._contact_container.impulses.numpy()[:, 0])
                    raise _FirstSweepRecorded

                solver.world._temporal_end_sweep = record_first_sweep
            dt = 1 / 120 if temporal else 1 / 1920
            pipeline.collide(state, contacts, dt=dt)
            try:
                solver.step(state, state, model.control(), contacts, dt)
            except _FirstSweepRecorded:
                pass
            if int(contacts.rigid_contact_count.numpy()[0]) != 1:
                raise AssertionError("Expected exactly one sphere-plane contact")
            return saved[0] if temporal else solver.world._contact_container.impulses.numpy()[:, 0]

    def test_speculative_friction_uses_physical_gap(self):
        """Match one ordinary contact step inside and outside the 2 mm friction range."""
        for gap in (0.00005, 0.001, 0.003):
            with self.subTest(gap=gap):
                ordinary = self._first_impulse(gap, False)
                temporal = self._first_impulse(gap, True)
                self.assertGreater(float(ordinary[0]), 0.0)
                np.testing.assert_allclose(temporal, ordinary, rtol=1.0e-6, atol=1.0e-7)
                if gap < 0.002:
                    self.assertGreater(float(np.linalg.norm(temporal[1:])), 0.1)
                else:
                    np.testing.assert_array_equal(temporal[1:], 0.0)


class TestTemporalContactMaterial(unittest.TestCase):
    def _check_material(self, device: str, stiffness: float, damping: float, velocity: float, grouped: bool) -> None:
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, 0.099), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.004),
            )
            cfg = builder.ShapeConfig(density=0, mu=0, mu_static=0, gap=0.002, margin=0, restitution=0)
            builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize()
            state = model.state()
            state.body_qd.assign(np.array([[0.0, 0.0, velocity, 0.0, 0.0, 0.0]], dtype=np.float32))
            pipeline = newton.CollisionPipeline(
                model, contact_matching="sticky", rigid_contact_max=16, speculative_contact_gap_max=0.005
            )
            contacts = newton.Contacts(
                16, 0, device=model.device, per_contact_shape_properties=True, contact_matching=True
            )
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                substeps=1,
                solver_iterations=16,
                velocity_iterations=0,
                step_layout="single_world",
                contact_hertz=80,
                sleeping_velocity_threshold=0,
                temporal_iterations=True,
                enable_body_pair_grouping=grouped,
            )
            dt = 1.0 / 120
            pipeline.collide(state, contacts, dt=dt)
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(damping)
            samples = []
            world = solver.world
            original_end_sweep = world._temporal_end_sweep

            def record_sweep(last: bool) -> None:
                self.assertEqual(int(world._cc_valid_count.numpy()[0]), 1)
                cc = world._contact_container
                samples.append((float(cc.gap.numpy()[0]), float(cc.impulses.numpy()[0, 0])))
                original_end_sweep(last)

            world._temporal_end_sweep = record_sweep
            solver.step(state, state, model.control(), contacts, dt)

        self.assertEqual(len(samples), 16)
        h = dt / 16
        gap = samples[0][0]
        if damping == 0.0:
            # Zero damping means none was set: the unit-mass row damps its spring, far
            # below the Nyquist cap, at the default damping ratio of one.
            damping = 2.0 * np.sqrt(stiffness)
        # For one unit-mass normal row, backward Euler has a closed form.
        # The impulse is unilateral: separating contact cannot pull the body.
        for actual_gap, actual_impulse in samples:
            impulse = max(
                0.0,
                h * (-stiffness * gap - (damping + h * stiffness) * velocity) / (1.0 + h * (damping + h * stiffness)),
            )
            self.assertAlmostEqual(actual_gap, gap, delta=2.0e-9)
            self.assertAlmostEqual(actual_impulse, impulse, delta=1.0e-7)
            self.assertGreaterEqual(actual_impulse, 0.0)
            velocity += impulse
            gap += h * velocity

    def test_implicit_material_response_cpu(self) -> None:
        for stiffness, damping, velocity in (
            (10000.0, 100.0, -0.1),
            (10000.0, 0.0, -0.1),
            (0.0, 100.0, -0.1),
            (10000.0, 100.0, 1.0),
        ):
            for grouped in (False, True):
                with self.subTest(stiffness=stiffness, damping=damping, velocity=velocity, grouped=grouped):
                    self._check_material("cpu", stiffness, damping, velocity, grouped)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_implicit_material_response_cuda(self) -> None:
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                self._check_material("cuda:0", 10000.0, 100.0, -0.1, grouped)


class TestTemporalContactObservables(unittest.TestCase):
    def test_ordinary_jointed_contacts_integrate_sweep_motion(self):
        """Integrate observed sweep motion when jointed contacts are not solved by the late direct pass."""
        for mode in ("maximal_pgs", "maximal_direct"):
            with self.subTest(mode=mode), wp.ScopedDevice("cpu"):
                builder = newton.ModelBuilder()
                body = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.1)))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.Z,
                    parent_xform=wp.transform(wp.vec3(0, 0, 0.1)),
                    target_ke=0,
                    target_kd=0,
                    limit_lower=-1e6,
                    limit_upper=1e6,
                )
                builder.add_articulation([joint])
                builder.add_ground_plane()
                model = builder.finalize()
                solver = SolverPhoenX(
                    model,
                    joint_mode=mode,
                    direct_contact_mask=np.zeros(1, dtype=bool) if mode == "maximal_direct" else None,
                    step_layout="single_world",
                    substeps=1,
                    solver_iterations=24,
                    temporal_iterations=True,
                    contact_hertz=80,
                )
                state, control = model.state(), model.control()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                contacts = model._collision_pipeline.contacts()
                dt = 1 / 120
                for _ in range(30):
                    previous_z = float(state.body_q.numpy()[body, 2])
                    solver.collide(state, contacts, dt=dt)
                    solver.step(state, state, control, contacts, dt)
                    world = solver.world
                    # Each ordinary-contact sweep advances dt / n. Their
                    # displacement must survive the outer position update.
                    displacement = float(world._temporal_velocity_sum.numpy()[body + 1, 2])
                    displacement *= dt / world._temporal_sweep_index
                    self.assertAlmostEqual(float(state.body_q.numpy()[body, 2]), previous_z + displacement, delta=1e-7)

    def test_owned_support_force_prevents_position_drift(self):
        """Keep a free hinged pair resting when contact reactions balance its weight."""
        self._check_owned_support_force("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_owned_support_force_balances_momentum_cuda(self):
        """Balance GPU support forces against weight and measured momentum change."""
        self._check_owned_support_force("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_owned_support_force_with_temporal_overflow_cuda(self):
        """Balance exact-joint reactions alongside ordinary overflow contacts."""
        self._check_owned_support_force("cuda:0", overflow=True)

    def _check_owned_support_force(self, device, *, overflow=False):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            body = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.1)))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            other = builder.add_link(xform=wp.transform(wp.vec3(0.4, 0, 0.1)))
            builder.add_shape_box(other, hx=0.1, hy=0.1, hz=0.1)
            root = builder.add_joint_free(child=body)
            joint = builder.add_joint_revolute(
                body,
                other,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.2, 0, 0)),
                child_xform=wp.transform(wp.vec3(-0.2, 0, 0)),
            )
            builder.add_articulation([root, joint])
            if overflow:
                ordinary = builder.add_body(xform=wp.transform((2.0, 0, 0.1)))
                builder.add_shape_box(ordinary, hx=0.1, hy=0.1, hz=0.1)
            ground = builder.add_ground_plane()
            model = builder.finalize()
            options = {}
            if overflow:
                options = {
                    "mass_splitting": True,
                    "mass_splitting_overflow_only": True,
                    "mass_splitting_unrolled": True,
                    "mass_splitting_batch_size": 1,
                    "max_colored_partitions": 0,
                    "direct_joint_mask": np.ones(model.joint_count, dtype=bool),
                    "direct_contact_mask": np.ones(model.joint_count, dtype=bool),
                }
            solver = SolverPhoenX(
                model,
                joint_mode="maximal_pgs" if overflow else "maximal_direct",
                step_layout="single_world",
                temporal_iterations=True,
                substeps=1,
                solver_iterations=24,
                contact_hertz=80,
                **options,
            )
            state, control = model.state(), model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            contacts = model._collision_pipeline.contacts()
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            masses = model.body_mass.numpy().astype(np.float64)
            weight = float(masses.sum()) * 9.81
            forces = []
            expected_forces = []
            for frame in range(120):
                previous_momentum = float(masses @ state.body_qd.numpy()[:, 2]) if frame >= 90 else 0.0
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120, observables=observables)
                if frame >= 90:
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                    force_z = observables.contact_f.numpy()[:count, 2]
                    support = np.concatenate((force_z[shape1 == ground], -force_z[shape0 == ground]))
                    self.assertTrue(np.all(support >= -1.0e-6), "The floor must not pull on the bodies")
                    forces.append(float(support.sum()))
                    momentum = float(masses @ state.body_qd.numpy()[:, 2])
                    expected_forces.append(weight + (momentum - previous_momentum) * 120)
            # CUDA can still have a small settling acceleration in this window;
            # comparing its force to weight alone would misdiagnose that motion.
            np.testing.assert_allclose(forces, expected_forces, atol=0.001, rtol=0.0)
            if overflow:
                self.assertTrue(solver.world._direct_equality_system.enabled)
                self.assertIsNotNone(solver.world._direct_contact_response)
                self.assertGreater(solver.step_report().overflow_size, 0)
            if device == "cpu":
                self.assertAlmostEqual(np.mean(forces), weight, delta=0.01)
            np.testing.assert_allclose(state.body_q.numpy()[:, 2], 0.1, atol=0.001)
            self.assertLess(float(np.abs(state.body_qd.numpy()[:, 2]).max()), 0.005)

    def test_owned_contacts_accumulate_in_every_sweep(self):
        """Count owned impulses in every temporal sweep, which solves them as a contact substep."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.7, gap=0.002)
        links = []
        for x in (0.0, 0.4):
            body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.2, hy=0.1, hz=0.1, cfg=cfg)
            links.append(body)
        root = builder.add_joint_free(parent=-1, child=links[0])
        hinge = builder.add_joint_revolute(
            parent=links[0],
            child=links[1],
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([root, hinge])
        box = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="maximal_direct",
            step_layout="single_world",
            temporal_iterations=True,
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
        )
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        control = model.control()
        observables = solver.observables([solver.ObservableKind.CONTACT_F], contacts=contacts)
        world = solver.world
        original = world._accumulate_contact_impulses
        calls = []

        def record_accumulation(current_contacts, **kwargs):
            before = world._contact_impulse_sum.numpy().copy()
            original(current_contacts, **kwargs)
            count = int(current_contacts.rigid_contact_count.numpy()[0])
            ids = world._cid_of_contact_cur.numpy()[:count] - world._contact_offset
            valid = ids >= 0
            owned = np.zeros(count, dtype=bool)
            owned[valid] = world._contact_cols.articulation_owner.numpy()[ids[valid]] >= 0
            delta = world._contact_impulse_sum.numpy()[:count] - before[:count]
            calls.append((float(np.linalg.norm(delta[owned])), float(np.linalg.norm(delta[valid & ~owned]))))

        saw_owned_intermediate = False
        saw_ordinary_intermediate = False
        shape_body = model.shape_body.numpy()
        box_mass = float(model.body_mass.numpy()[box])
        dt = 1.0 / 240.0
        with patch.object(world, "_accumulate_contact_impulses", record_accumulation):
            for _ in range(16):
                calls.clear()
                previous_velocity = float(state.body_qd.numpy()[box, 2])
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, state, control, contacts, dt, observables=observables)
                self.assertTrue(world._regular_pgs_active_this_step)
                self.assertEqual(len(calls), 4)
                for owned, ordinary in calls[:-1]:
                    saw_owned_intermediate |= owned > 0.0
                    saw_ordinary_intermediate |= ordinary > 0.0
                count = int(contacts.rigid_contact_count.numpy()[0])
                body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
                body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
                force_z = observables.contact_f.numpy()[:count, 2]
                reported_force = float(force_z[body0 == box].sum() - force_z[body1 == box].sum())
                velocity = float(state.body_qd.numpy()[box, 2])
                expected_force = box_mass * (9.81 + (velocity - previous_velocity) / dt)
                self.assertAlmostEqual(reported_force, expected_force, delta=1.0e-4)
        self.assertTrue(saw_owned_intermediate, "Owned contact substeps must accumulate their impulses")
        self.assertTrue(saw_ordinary_intermediate, "Ordinary temporal contact impulses must still accumulate")


if __name__ == "__main__":
    unittest.main()
