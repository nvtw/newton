# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.benchmarks.bench_g1_drive_convergence import (
    _SETTINGS as DRIVE_CONVERGENCE_SETTINGS,
)
from newton._src.solvers.phoenx.benchmarks.bench_g1_drive_convergence import (
    _parse_args as parse_drive_convergence_args,
)
from newton._src.solvers.phoenx.benchmarks.bench_g1_train import _parse_args as parse_train_args
from newton._src.solvers.phoenx.benchmarks.bench_g1_train import _summarize_measured_history
from newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate import (
    _make_parser as make_train_to_gate_parser,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog import (
    _g1_env_config,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog import (
    build_arg_parser as build_leapfrog_arg_parser,
)
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.g1 import ACTION_DIM_G1, EnvG1PhoenX


def _stats(elapsed: float) -> SimpleNamespace:
    return SimpleNamespace(rollout_seconds=elapsed, update_seconds=0.0)


class TestBenchG1Train(unittest.TestCase):
    def test_training_script_defaults_are_the_production_recipe(self):
        with mock.patch("sys.argv", ["bench_g1_train"]):
            args = parse_train_args()

        self.assertEqual(args.world_count, g1_recipe.WORLD_COUNT)
        self.assertEqual(args.sim_substeps, g1_recipe.SIM_SUBSTEPS)
        self.assertEqual(args.solver_iterations, g1_recipe.SOLVER_ITERATIONS)
        self.assertEqual(args.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)
        self.assertEqual(args.articulation_mode, g1_recipe.ARTICULATION_MODE)
        self.assertEqual(args.reduced_articulation_path, g1_recipe.REDUCED_ARTICULATION_PATH)
        self.assertEqual(args.actuation_model, g1_recipe.ACTUATION_MODEL)
        self.assertEqual(args.contact_geometry, g1_recipe.CONTACT_GEOMETRY)

    def test_train_to_gate_does_not_perturb_training_by_default(self):
        args = make_train_to_gate_parser().parse_args([])

        self.assertFalse(args.reset_env_between_chunks)
        self.assertEqual(args.late_replay_start_samples, args.angular_fine_tune_start_samples)

    def test_graph_leapfrog_excludes_final_drain(self):
        measured, env_sps, excluded_drain = _summarize_measured_history(
            [_stats(4.0), _stats(2.0), _stats(0.1)],
            warmup_iterations=1,
            execution_mode="graph_leapfrog",
            samples_per_interval=100,
        )

        self.assertEqual(measured, [_stats(2.0)])
        self.assertEqual(env_sps, 50.0)
        self.assertTrue(excluded_drain)

    def test_rejects_graph_run_with_only_drain_after_warmup(self):
        with self.assertRaisesRegex(ValueError, "no complete measured training intervals"):
            _summarize_measured_history(
                [_stats(4.0), _stats(0.1)],
                warmup_iterations=1,
                execution_mode="graph_leapfrog",
                samples_per_interval=100,
            )

    def test_uses_aggregate_elapsed_time(self):
        _, env_sps, excluded_drain = _summarize_measured_history(
            [_stats(1.0), _stats(3.0)],
            warmup_iterations=0,
            execution_mode="eager",
            samples_per_interval=100,
        )

        self.assertEqual(env_sps, 50.0)
        self.assertFalse(excluded_drain)

    def test_leapfrog_benchmark_inherits_production_environment(self):
        args = build_leapfrog_arg_parser().parse_args([])
        config = _g1_env_config(args)

        self.assertEqual(config.articulation_mode, g1_recipe.ARTICULATION_MODE)
        self.assertEqual(config.reduced_articulation_path, g1_recipe.REDUCED_ARTICULATION_PATH)
        self.assertEqual(config.contact_geometry, g1_recipe.CONTACT_GEOMETRY)
        self.assertEqual(config.contact_friction_model, g1_recipe.CONTACT_FRICTION_MODEL)
        self.assertEqual(config.actuation_model, g1_recipe.ACTUATION_MODEL)
        self.assertEqual(config.observation_mode, g1_recipe.OBSERVATION_MODE)

    def test_drive_convergence_benchmark_inherits_production_environment(self):
        with mock.patch("sys.argv", ["bench_g1_drive_convergence"]):
            args = parse_drive_convergence_args()
        setting = DRIVE_CONVERGENCE_SETTINGS["rl_current"]

        self.assertEqual(setting.sim_substeps, g1_recipe.SIM_SUBSTEPS)
        self.assertEqual(setting.solver_iterations, g1_recipe.SOLVER_ITERATIONS)
        self.assertEqual(setting.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)
        self.assertEqual(args.actuation_model, g1_recipe.ACTUATION_MODEL)
        self.assertEqual(args.articulation_mode, g1_recipe.ARTICULATION_MODE)


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "G1 training dynamics validation requires CUDA.")
class TestBenchG1TrainAnalyticalDynamics(unittest.TestCase):
    def test_production_g1_drive_torques_match_clamped_pd_analytically(self):
        config = g1_recipe.default_g1_env_config(
            world_count=1,
            max_episode_steps=0,
            auto_reset=False,
            randomize_commands_on_reset=False,
            reset_noise=0.0,
        )
        env = EnvG1PhoenX(config, device=wp.get_preferred_device())

        self.assertEqual(env.config.frame_dt, g1_recipe.FRAME_DT)
        self.assertEqual(env.config.sim_substeps, g1_recipe.SIM_SUBSTEPS)
        self.assertEqual(env.config.solver_iterations, g1_recipe.SOLVER_ITERATIONS)
        self.assertEqual(env.config.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)
        self.assertEqual(env.config.articulation_mode, g1_recipe.ARTICULATION_MODE)
        self.assertEqual(env.config.actuation_model, g1_recipe.ACTUATION_MODEL)
        self.assertEqual(env.config.contact_friction_model, g1_recipe.CONTACT_FRICTION_MODEL)
        self.assertEqual(env.solver.world.substeps, 1)
        self.assertEqual(env.solver.world.solver_iterations, g1_recipe.SOLVER_ITERATIONS)
        self.assertEqual(env.solver.world.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)

        q = env.state_0.joint_q.numpy().reshape(1, env.coord_stride)
        qd = env.state_0.joint_qd.numpy().reshape(1, env.dof_stride)
        q[:, 7 : 7 + ACTION_DIM_G1] = env.default_joint_pos.numpy() + np.linspace(
            -3.0, 3.0, ACTION_DIM_G1, dtype=np.float32
        )
        qd[:, 6 : 6 + ACTION_DIM_G1] = np.linspace(-2.0, 2.0, ACTION_DIM_G1, dtype=np.float32)
        env.state_0.joint_q.assign(q.reshape(-1))
        env.state_0.joint_qd.assign(qd.reshape(-1))

        target = env.control.joint_target_q.numpy()
        if env.model.use_coord_layout_targets:
            target = target.reshape(1, env.coord_stride)
            target[:, 7 : 7 + ACTION_DIM_G1] = env.default_joint_pos.numpy()
        else:
            target = target.reshape(1, env.dof_stride)
            target[:, 6 : 6 + ACTION_DIM_G1] = env.default_joint_pos.numpy()
        env.control.joint_target_q.assign(target.reshape(-1))
        env._gather_actuator_force(scatter_joint_f=True)

        joint_q = q[:, 7 : 7 + ACTION_DIM_G1]
        joint_qd = qd[:, 6 : 6 + ACTION_DIM_G1]
        expected_force = env.actuator_force_kp.numpy()[None, :] * (env.default_joint_pos.numpy()[None, :] - joint_q)
        expected_force -= env.actuator_force_kd.numpy()[None, :] * joint_qd
        expected_force = np.clip(
            expected_force,
            env.actuator_force_lower.numpy()[None, :],
            env.actuator_force_upper.numpy()[None, :],
        )
        expected_joint_f = expected_force - env.passive_damping.numpy()[None, :] * joint_qd

        self.assertTrue(np.any(expected_force == env.actuator_force_lower.numpy()[None, :]))
        self.assertTrue(np.any(expected_force == env.actuator_force_upper.numpy()[None, :]))
        self.assertGreater(float(np.max(np.abs(expected_joint_f - expected_force))), 0.0)
        np.testing.assert_allclose(env.actuator_force.numpy(), expected_force, rtol=1.0e-6, atol=1.0e-6)
        joint_f = env.control.joint_f.numpy().reshape(1, env.dof_stride)
        np.testing.assert_allclose(joint_f[:, :6], 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            joint_f[:, 6 : 6 + ACTION_DIM_G1],
            expected_joint_f,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_production_timestep_matches_constant_torque_solution(self):
        inertia = 0.5
        torque = 1.25
        q_initial = 0.1
        qd_initial = -0.2

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0), up_axis=newton.Axis.Z)
        body = builder.add_link(
            mass=2.0,
            inertia=((0.3, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, inertia)),
        )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            actuator_mode=newton.JointTargetMode.EFFORT,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_preferred_device())
        state_0 = model.state()
        state_1 = model.state()
        state_0.joint_q.assign(np.array([q_initial], dtype=np.float32))
        state_0.joint_qd.assign(np.array([qd_initial], dtype=np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        control.joint_f.assign(np.array([torque], dtype=np.float32))
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=g1_recipe.SOLVER_ITERATIONS,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            articulation_mode=g1_recipe.ARTICULATION_MODE,
            reduced_articulation_path=g1_recipe.REDUCED_ARTICULATION_PATH,
        )

        substeps = g1_recipe.SIM_SUBSTEPS
        substep_dt = g1_recipe.FRAME_DT / substeps
        for substep in range(substeps):
            state_0.clear_forces()
            solver.step(
                state_0,
                state_1,
                control,
                None,
                substep_dt,
                state_is_continuation=substep > 0,
                state_kinematics_valid=True,
            )
            state_0, state_1 = state_1, state_0

        acceleration = torque / inertia
        expected_qd = qd_initial + acceleration * g1_recipe.FRAME_DT
        expected_q = (
            q_initial + qd_initial * g1_recipe.FRAME_DT + acceleration * substep_dt**2 * substeps * (substeps + 1) / 2.0
        )
        forward_euler_q = (
            q_initial + qd_initial * g1_recipe.FRAME_DT + acceleration * substep_dt**2 * substeps * (substeps - 1) / 2.0
        )
        self.assertGreater(abs(expected_q - forward_euler_q), 1.0e-4)
        np.testing.assert_allclose(state_0.joint_qd.numpy(), [expected_qd], rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(state_0.joint_q.numpy(), [expected_q], rtol=1.0e-6, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
