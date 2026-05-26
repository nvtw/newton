# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tonge-style mass-splitting primitives for PhoenX.

Direct port of the C# PhoenX ``MassSplitting`` subtree
(``experimentalsim/PhoenX/src/PhoenX/MassSplitting/``). Provides three
self-contained building blocks that compose with the PhoenX PGS pipeline:

* :class:`CopyStateContainer` — per-(node, partition_copy) TinyRigidState
  slots. One flat SoA buffer that holds both rigid-body and particle
  copies under the unified body-or-particle index space.

* :class:`InteractionGraphScratch` plus :func:`build_interaction_graph` —
  on-device builder that walks the active constraints and stamps each
  (unified_node_id, partition_key) pair into ``CopyStateContainer``'s
  ``section_end`` / ``partition_list`` arrays. Sorted + deduplicated
  inside a captured CUDA graph.

* :func:`launch_broadcast_rigid_to_copy_states`,
  :func:`launch_average_and_broadcast`,
  :func:`launch_copy_state_into_rigids` — the substep-loop kernels that
  fan body / particle state into copies, average copies after a PGS
  sweep, and write the averaged result back.

This first milestone exposes the data plane only; the solver loop in
``solver_phoenx.py`` is unchanged. Step 6 will wire these primitives in.
"""

from __future__ import annotations

from newton._src.solvers.phoenx.mass_splitting.access import (
    get_state_index,
    read_angular_velocity_unified,
    read_orientation_unified,
    read_position_unified,
    read_velocity_unified,
    set_access_mode_unified,
    slot_synchronize_to_velocity_level,
    write_angular_velocity_unified,
    write_orientation_unified,
    write_position_unified,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import (
    CopyStateContainer,
    copy_state_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphScratch,
    build_interaction_graph,
    interaction_graph_scratch_zeros,
    record_all_interactions_kernel,
)
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    launch_average_and_broadcast,
    launch_average_and_broadcast_grouped,
    launch_average_and_broadcast_rigid_velocity,
    launch_broadcast_rigid_to_copy_states,
    launch_copy_state_into_rigids,
)
from newton._src.solvers.phoenx.mass_splitting.slot_cache import (
    build_constraint_slot_cache,
)

__all__ = [
    "CopyStateContainer",
    "InteractionGraphScratch",
    "build_constraint_slot_cache",
    "build_interaction_graph",
    "copy_state_container_zeros",
    "get_state_index",
    "interaction_graph_scratch_zeros",
    "launch_average_and_broadcast",
    "launch_average_and_broadcast_grouped",
    "launch_average_and_broadcast_rigid_velocity",
    "launch_broadcast_rigid_to_copy_states",
    "launch_copy_state_into_rigids",
    "read_angular_velocity_unified",
    "read_orientation_unified",
    "read_position_unified",
    "read_velocity_unified",
    "record_all_interactions_kernel",
    "set_access_mode_unified",
    "slot_synchronize_to_velocity_level",
    "write_angular_velocity_unified",
    "write_orientation_unified",
    "write_position_unified",
    "write_velocity_unified",
]
