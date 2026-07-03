.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.rl
=========

Warp-only reinforcement learning utilities for Newton.

.. py:module:: newton.rl
.. currentmodule:: newton.rl

.. toctree::
   :hidden:

   newton_rl_reward_functions

.. rubric:: Submodules

- :doc:`newton.rl.reward_functions <newton_rl_reward_functions>`

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   Adam
   BatchSAC
   BufferReplaySAC
   BufferRollout
   ConfigEnvAntPhoenX
   ConfigEnvAnymalPhoenX
   ConfigEnvDrLegsPhoenX
   ConfigEnvG1PhoenX
   ConfigEnvGo2PhoenX
   ConfigEnvH1PhoenX
   ConfigEnvHumanoidPhoenX
   ConfigEvaluateAnymalPPO
   ConfigEvaluateG1GatePPO
   ConfigEvaluateG1PPO
   ConfigEvaluateG1TargetPPO
   ConfigPBT
   ConfigPPO
   ConfigSAC
   ConfigTrainAnymalPPO
   ConfigTrainG1PPO
   EnvAntPhoenX
   EnvAnymalPhoenX
   EnvDrLegsPhoenX
   EnvG1PhoenX
   EnvGo2PhoenX
   EnvH1PhoenX
   EnvHumanoidPhoenX
   EnvPPO
   GaussianActor
   GenerationResult
   HparamSpec
   MirrorMapPPO
   Muon
   PufferMinGRUNet
   ResultEvaluateAnymalPPO
   ResultEvaluateG1GatePPO
   ResultEvaluateG1PPO
   ResultEvaluateG1TargetPPO
   ResultPBT
   ResultTrainAnymalPPO
   ResultTrainG1PPO
   StatsEvaluateAnymalTargetPPO
   StatsEvaluateG1GateCommandPPO
   StatsEvaluateG1GatePPO
   StatsEvaluateG1PPO
   StatsEvaluateG1TargetPPO
   StatsPPOUpdate
   StatsSACUpdate
   StatsTrainAnymalPPO
   StatsTrainG1PPO
   TrainerPPO
   TrainerSAC
   WarpMLP
   WorkerState

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   anymal_mirror_map_ppo
   capture_env_steps
   collect_ppo_rollout
   collect_ppo_rollout_seed_counter
   default_anymal_hparam_specs
   default_g1_hparam_specs
   drop_ppo_checkpoint_inputs
   evaluate_anymal_ppo
   evaluate_g1_gate_ppo
   evaluate_g1_ppo
   evaluate_g1_target_ppo
   g1_mirror_map_ppo
   insert_ppo_checkpoint_inputs
   load_ppo_checkpoint
   make_seed_counter
   population_based_train
   population_based_train_anymal
   population_based_train_g1
   resize_ppo_checkpoint_inputs
   save_ppo_checkpoint
   train_anymal_ppo
   train_g1_ppo

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``ACTION_DIM_ANT``
     - ``8``
   * - ``ACTION_DIM_ANYMAL``
     - ``12``
   * - ``ACTION_DIM_DR_LEGS``
     - ``12``
   * - ``ACTION_DIM_G1``
     - ``29``
   * - ``ACTION_DIM_H1``
     - ``19``
   * - ``ACTION_DIM_HUMANOID``
     - ``21``
   * - ``ACTION_OBS_OFFSET_ANYMAL``
     - ``37``
   * - ``COMMAND_DIM_ANYMAL``
     - ``4``
   * - ``COMMAND_OBS_OFFSET_ANYMAL``
     - ``9``
   * - ``JOINT_POS_OBS_OFFSET_ANYMAL``
     - ``13``
   * - ``JOINT_VEL_OBS_OFFSET_ANYMAL``
     - ``25``
   * - ``OBS_DIM_ANT``
     - ``34``
   * - ``OBS_DIM_ANYMAL``
     - ``49``
   * - ``OBS_DIM_DR_LEGS_HOLD``
     - ``42``
   * - ``OBS_DIM_DR_LEGS_WALK``
     - ``47``
   * - ``OBS_DIM_G1``
     - ``98``
   * - ``OBS_DIM_G1_ISAACLAB_FLAT``
     - ``99``
   * - ``OBS_DIM_G1_NANOG1``
     - ``98``
   * - ``OBS_DIM_H1``
     - ``69``
   * - ``OBS_DIM_HUMANOID``
     - ``75``
