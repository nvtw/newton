# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tonge 2012 mass splitting -- isolated port from PhoenX C#.

See :doc:`README.md` (in this directory) for the design overview
and the per-file mapping to the C# source.

The public entry point is :class:`MassSplitting`; the rest of the
public symbols are here for advanced users (custom integrators,
unit tests, or callers that want to skip the orchestrator and
launch the kernels themselves).
"""

from __future__ import annotations

from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraph,
    InteractionGraphData,
    graph_get_rigid_state_index,
    graph_get_state,
    graph_set_state,
    graph_state_section,
)
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    average_and_broadcast_kernel,
    broadcast_rigid_to_copy_states_kernel,
    copy_state_into_rigids_kernel,
)
from newton._src.solvers.phoenx.mass_splitting.mass_splitting import MassSplitting
from newton._src.solvers.phoenx.mass_splitting.partitions import (
    ContactPartitions,
    ContactPartitionsData,
    partitions_get_partition_end,
    partitions_get_partition_size,
    partitions_get_partition_start,
)
from newton._src.solvers.phoenx.mass_splitting.read_state import (
    read_state,
    write_state,
)
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_NONE,
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_STATIC_BODY,
    ACCESS_MODE_VELOCITY_LEVEL,
    TinyRigidState,
    tiny_rigid_state_from_body,
    tiny_rigid_state_set_access_mode,
    tiny_rigid_state_synchronize,
    tiny_rigid_state_write_back,
)

__all__ = [
    "ACCESS_MODE_NONE",
    "ACCESS_MODE_POSITION_LEVEL",
    "ACCESS_MODE_STATIC_BODY",
    "ACCESS_MODE_VELOCITY_LEVEL",
    "ContactPartitions",
    "ContactPartitionsData",
    "InteractionGraph",
    "InteractionGraphData",
    "MassSplitting",
    "TinyRigidState",
    "average_and_broadcast_kernel",
    "broadcast_rigid_to_copy_states_kernel",
    "copy_state_into_rigids_kernel",
    "graph_get_rigid_state_index",
    "graph_get_state",
    "graph_set_state",
    "graph_state_section",
    "partitions_get_partition_end",
    "partitions_get_partition_size",
    "partitions_get_partition_start",
    "read_state",
    "tiny_rigid_state_from_body",
    "tiny_rigid_state_set_access_mode",
    "tiny_rigid_state_synchronize",
    "tiny_rigid_state_write_back",
    "write_state",
]
