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
   ConfigPPO
   ConfigSAC
   ConfigTrainAnymalPPO
   EnvAnymalPhoenX
   GaussianActor
   ResultTrainAnymalPPO
   StatsPPOUpdate
   StatsSACUpdate
   StatsTrainAnymalPPO
   TrainerPPO
   TrainerSAC
   WarpMLP

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   train_anymal_ppo

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``ACTION_DIM_ANYMAL``
     - ``12``
   * - ``OBS_DIM_ANYMAL``
     - ``48``
