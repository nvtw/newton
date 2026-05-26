# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common API for graph-coloring partitioner implementations.

The PhoenX single-world solve consumes a coloring of the active
constraint graph: partitions of the constraint set such that no two
constraints in the same partition share a body / particle node. Each
substep's PGS sweep then processes partitions one colour at a time
(within a colour, every constraint can run in parallel).

Partitioner implementations follow this contract: allocate scratch
in the ctor, accept the per-step constraint elements via
:meth:`reset`, insert their kernels into the current Warp stream via
:meth:`build_csr`, and expose the resulting CSR via the output
properties. The PhoenX solver selects between implementations at
construction; the surface is intentionally minimal so new algorithms
can land without touching the solver.

To add a new implementation:

1. Write a class implementing :class:`ContactPartitioner` in its own
   ``phoenx/graph_coloring/<name>.py`` file.
2. Reuse the adjacency-build kernels from
   :mod:`graph_coloring_common` if helpful; the coloring step itself
   is the lever.
3. Register it in :func:`make_partitioner` and add a tag to
   :data:`PHOENX_PARTITIONER_ALGORITHM` in ``solver_config.py``.
4. Bench head-to-head via
   :mod:`benchmarks.bench_graph_coloring_algorithms`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import warp as wp


@runtime_checkable
class ContactPartitioner(Protocol):
    """Single-world graph-coloring partitioner.

    The solver guarantees this call sequence:

    * Once at construction: allocate all scratch sized to
      ``max_num_interactions``.
    * Per simulator step, in order:

      a. :meth:`set_costs_from_contacts` (host-side, may inspect
         contact column counts to bias JP priorities).
      b. :meth:`reset` (point at this step's element buffer + count).
      c. :meth:`build_csr` (insert coloring kernels into the captured
         Warp graph).
      d. Per PGS sweep: :meth:`begin_sweep` (resets ``color_cursor``).

    Implementations are free to ignore ``set_costs_from_contacts`` if
    they don't bias on per-element cost.
    """

    # --- Output CSR (must be allocated in ctor; populated by build_csr) ---

    #: Per-element constraint id, grouped by colour. ``element_ids_by_color
    #: [color_starts[c]:color_starts[c+1]]`` is the slice for colour ``c``.
    element_ids_by_color: wp.array

    #: Exclusive prefix sum of per-colour element counts. Length
    #: ``MAX_COLORS + 1``. ``color_starts[num_colors[0]]`` is the total
    #: active count.
    color_starts: wp.array

    #: Colour count from the most recent build. Length 1.
    num_colors: wp.array

    #: Reverse lookup ``eid -> colour``. Length ``max_num_interactions``.
    interaction_id_to_partition: wp.array

    #: Sweep-time colour countdown initialised by :meth:`begin_sweep`,
    #: decremented by the PGS sweep kernels. Length 1.
    color_cursor: wp.array

    # --- Lifecycle ---

    def reset(
        self,
        elements: wp.array,
        num_elements: wp.array,
    ) -> None:
        """Point the partitioner at this step's input buffers. Cheap;
        does not launch kernels."""
        ...

    def set_costs_from_contacts(
        self,
        num_cids: int,
        num_contact_columns: wp.array,
        contact_cols,
    ) -> None:
        """Optional per-step cost-array refresh. Used by implementations
        that bias JP priorities by per-element contact count."""
        ...

    def build_csr(self) -> None:
        """Insert the coloring kernels into the current Warp stream /
        captured graph. Populates :attr:`element_ids_by_color`,
        :attr:`color_starts`, :attr:`num_colors`, and
        :attr:`interaction_id_to_partition`."""
        ...

    def begin_sweep(self) -> None:
        """Reset :attr:`color_cursor` for a new PGS sweep
        (``color_cursor[0] = num_colors[0]``)."""
        ...


__all__ = ["ContactPartitioner"]
