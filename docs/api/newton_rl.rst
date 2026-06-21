.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.rl
=========

Warp-only reinforcement learning utilities for Newton.

.. py:module:: newton.rl
.. currentmodule:: newton.rl

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   Adam
   BatchSAC
   BufferReplaySAC
   BufferRollout
   ConfigEnvAnymalPhoenX
   ConfigEnvG1PhoenX
   ConfigEvaluateAnymalPPO
   ConfigEvaluateG1GatePPO
   ConfigEvaluateG1PPO
   ConfigPPO
   ConfigSAC
   ConfigTrainAnymalPPO
   ConfigTrainG1PPO
   EnvAnymalPhoenX
   EnvG1PhoenX
   EnvPPO
   GaussianActor
   MirrorMapPPO
   Muon
   PufferMinGRUNet
   ResultEvaluateAnymalPPO
   ResultEvaluateG1GatePPO
   ResultEvaluateG1PPO
   ResultTrainAnymalPPO
   ResultTrainG1PPO
   StatsEvaluateAnymalTargetPPO
   StatsEvaluateG1GateCommandPPO
   StatsEvaluateG1GatePPO
   StatsEvaluateG1PPO
   StatsPPOUpdate
   StatsSACUpdate
   StatsTrainAnymalPPO
   StatsTrainG1PPO
   TrainerPPO
   TrainerSAC
   WarpMLP

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   capture_env_steps
   collect_ppo_rollout
   evaluate_anymal_ppo
   evaluate_g1_gate_ppo
   evaluate_g1_ppo
   g1_mirror_map_ppo
   load_ppo_checkpoint
   save_ppo_checkpoint
   train_anymal_ppo
   train_g1_ppo

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``ACTION_DIM_ANYMAL``
     - ``12``
   * - ``ACTION_DIM_G1``
     - ``29``
   * - ``OBS_DIM_ANYMAL``
     - ``48``
   * - ``OBS_DIM_G1``
     - ``98``
