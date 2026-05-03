# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Top-level orchestrator that wires the mass-splitting components.

Mirrors the C# ``MassSplitting`` class in
``PhoenX/MassSplitting/MassSplitting.cs:11-225``.

A :class:`MassSplitting` owns a :class:`ContactPartitions` and a
:class:`InteractionGraph` and exposes one ergonomic API for the
caller:

* ``setup(...)`` -- one-time per-scene wiring (allocate buffers,
  partition, register interaction-graph entries, build).
* ``broadcast(...)``, ``average(...)``, ``write_back(...)`` --
  per-substep launches against a body store.

Intentionally *not* yet connected to :class:`SolverPhoenX`. Callers
hand it plain Warp arrays (``body_position``, ``body_orientation``,
``body_velocity``, ``body_angular_velocity``); the eventual
integration step in :class:`SolverPhoenX.step` will pull these out
of :class:`~newton._src.solvers.phoenx.body.BodyContainer` at the
launch site.

## Setup paths

Either build path on :class:`ContactPartitions` works:

1. **Self-built MIS partitioning** -- host-side greedy on
   :meth:`ContactPartitions.build`. Useful for tests / small scenes
   that don't already have a graph coloring.

2. **Adopt an existing Newton coloring** -- pass the existing
   ``(element_ids_by_color, color_starts, num_colors)`` arrays
   from :mod:`newton._src.solvers.phoenx.graph_coloring` to
   :meth:`ContactPartitions.wrap_color_arrays`. Integration target
   so the per-step graph-coloring work isn't duplicated.

## Velocity-level vs position-level passes

The same orchestrator handles both. C# PhoenX runs constraints in
either regime by passing an ``accessMode`` argument to
``ReadState``; this port mirrors that exactly via the
:func:`~newton._src.solvers.phoenx.mass_splitting.read_state.read_state`
``new_access_mode`` parameter. The infrastructure -- ``TinyRigidState``,
broadcast, write-back -- is unchanged between regimes; only what
the constraint kernel does between read and write changes:

* **Velocity-level**: read with ``ACCESS_MODE_VELOCITY_LEVEL``,
  mutate ``state.velocity`` / ``state.angular_velocity``, write
  back. Box2D / Jitter sequential-impulse style.
* **Position-level**: read with ``ACCESS_MODE_POSITION_LEVEL``,
  mutate ``state.position`` / ``state.orientation``, write back.
  XPBD projection style. The
  :func:`~newton._src.solvers.phoenx.mass_splitting.state.tiny_rigid_state_set_access_mode`
  helper inside ``read_state`` integrates pose forward when
  flipping into the position regime; the
  :func:`~newton._src.solvers.phoenx.mass_splitting.kernels.copy_state_into_rigids_kernel`
  runs the XPBD finite-difference recovery
  ``v = (state.position - body_position) * inv_dt`` at write-back.

Mixing the two regimes inside a single substep is allowed: each
constraint picks its own ``new_access_mode``; the synchronize
helper handles the regime change lazily. See ``README.md`` for the
recipe and the round-trip test in ``tests/test_isolated.py``.

.. note::
    Production today uses a weaker variant of this contract -- per-
    constraint ``ACCESS_MODE`` metadata + a constraint-owns-handoff
    finite-diff at iterate exit, without ``TinyRigidState``. See
    :file:`newton/_src/solvers/phoenx/CONSTRAINT_ACCESS_MODE.md`. The
    tag values match so a future mass-splitting integration drops in
    unchanged.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.mass_splitting.interaction_graph import InteractionGraph
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    average_and_broadcast_kernel,
    broadcast_rigid_to_copy_states_kernel,
    copy_state_into_rigids_kernel,
)
from newton._src.solvers.phoenx.mass_splitting.partitions import ContactPartitions

__all__ = [
    "MassSplitting",
]


class MassSplitting:
    """Top-level mass-splitting container.

    Owns and orchestrates the partitioner, the interaction graph,
    and the per-substep launches. Lifecycle:

    1. ``ms = MassSplitting(num_bodies, num_constraints, max_partitions)``
    2. *one of:*

       * ``ms.setup_from_graph(constraint_bodies)`` -- run the
         host-side greedy MIS partitioner on the constraint graph.
       * ``ms.setup_from_coloring(element_ids_by_color, color_starts,
         num_colors, constraint_bodies)`` -- adopt an existing
         coloring (e.g. from
         :mod:`newton._src.solvers.phoenx.graph_coloring`).
    3. Each substep:

       * ``ms.broadcast(body_state, dt)`` -- before the first
         iterate.
       * Iterate sweeps go here (the constraint kernels are *not*
         in this module yet -- they need to consume
         :func:`~.read_state.read_state` /
         :func:`~.read_state.write_state`).
       * ``ms.average(body_state, inv_dt)`` -- after each iterate
         sweep.
       * ``ms.write_back(body_state, inv_dt)`` -- once at the end
         of the substep.

    Attributes:
        num_bodies: Body count addressable by the interaction graph.
        num_constraints: Constraint capacity for the partitioner.
        max_partitions: Independent-set partition cap.
        partitions: :class:`ContactPartitions` instance.
        graph: :class:`InteractionGraph` instance.
    """

    def __init__(
        self,
        num_bodies: int,
        num_constraints: int,
        max_partitions: int = 12,
        device: wp.context.Devicelike = None,
    ) -> None:
        self.num_bodies = int(num_bodies)
        self.num_constraints = int(num_constraints)
        self.max_partitions = int(max_partitions)
        self.device = device
        # Capacity of the interaction-graph buffers: every
        # (body, partition) pair the constraint touches counts as
        # one interaction. Worst case: every constraint touches
        # ``max_partitions`` distinct bodies (it can't, but bound
        # the upper limit by the constraint count * the average
        # body-count-per-constraint). For typical 1- or 2-body
        # constraints, ``num_constraints * 2`` is enough; we add
        # a 25% safety margin so callers don't have to fiddle.
        max_interactions = max(num_bodies, num_constraints * 2 + num_constraints // 4)
        self.partitions = ContactPartitions(
            max_num_rigids=self.num_bodies,
            max_num_constraints=self.num_constraints,
            max_num_partitions=self.max_partitions,
            device=self.device,
        )
        self.graph = InteractionGraph(
            max_rigid_bodies=self.num_bodies,
            max_interactions=max_interactions,
            device=self.device,
        )
        self._setup_complete: bool = False

    # ------------------------------------------------------------------
    # Setup paths -- pick one.
    # ------------------------------------------------------------------

    def setup_from_graph(self, constraint_bodies) -> None:
        """Run the host-side greedy MIS partitioner.

        Args:
            constraint_bodies: iterable of iterables; element ``i``
                lists the body IDs that constraint ``i`` touches
                (1 for unilateral, 2 for typical pair constraints).
        """
        constraints_list = [list(b) for b in constraint_bodies]
        self.partitions.build(constraints_list)
        self._register_interaction_graph_entries(constraints_list)
        self.graph.build()
        self._setup_complete = True

    def setup_from_coloring(
        self,
        element_ids_by_color: wp.array,
        color_starts: wp.array,
        num_colors,
        constraint_bodies,
    ) -> None:
        """Adopt an existing graph coloring as the partition table.

        The ``element_ids_by_color`` / ``color_starts`` arrays are
        the same shape Newton's
        :mod:`newton._src.solvers.phoenx.graph_coloring` produces.
        ``constraint_bodies`` is still required because the
        :class:`InteractionGraph` build needs to know which bodies
        each constraint touches; the partitioning is *not* re-run.
        """
        self.partitions.wrap_color_arrays(
            element_ids_by_color,
            color_starts,
            num_colors,
            constraint_bodies=constraint_bodies,
        )
        self._register_interaction_graph_entries(constraint_bodies)
        self.graph.build()
        self._setup_complete = True

    def _register_interaction_graph_entries(self, constraint_bodies) -> None:
        """Push every ``(rigid, constraint)`` edge into the graph
        builder. The partitioner's per-partition assignment is
        looked up via :class:`ContactPartitions.iter_partition_constraints`
        so the graph's ``partition_list`` ends up sorted *by
        partition index* per body (the binary-search invariant in
        :func:`~.interaction_graph.graph_get_rigid_state_index`)."""
        constraint_bodies_list = [list(b) for b in constraint_bodies]
        # Walk partitions in ascending order; for each constraint
        # in the partition register one entry per touched body.
        # ``constraint_id`` in the InteractionGraph's partition_list
        # is the *partition index*, not the global constraint ID --
        # the read_state binary search is keyed by partition index
        # because that's what the iterate kernel knows.
        partition_count = self.partitions.num_partitions
        for partition_index in range(partition_count):
            for cid in self.partitions.iter_partition_constraints(partition_index):
                if cid < 0 or cid >= len(constraint_bodies_list):
                    continue
                for body_id in constraint_bodies_list[cid]:
                    self.graph.add_entry(int(body_id), int(partition_index))
        # Additional partition: also has a parallel-id (= max_partitions)
        # so iterate kernels operating on it can still read_state.
        if self.partitions.has_additional_partition:
            partition_index = self.partitions.max_num_partitions
            for cid in self.partitions.iter_partition_constraints(self.partitions.num_partitions):
                if cid < 0 or cid >= len(constraint_bodies_list):
                    continue
                for body_id in constraint_bodies_list[cid]:
                    self.graph.add_entry(int(body_id), int(partition_index))

    # ------------------------------------------------------------------
    # Per-substep launches.
    # ------------------------------------------------------------------

    def broadcast(
        self,
        body_position: wp.array,
        body_orientation: wp.array,
        body_velocity: wp.array,
        body_angular_velocity: wp.array,
        dt: float,
    ) -> None:
        """Initialise every ``tiny_states`` slot from the body store.

        Run once at the start of each substep, before any iterate
        sweeps. Mirrors C#
        ``MassSplitting.BroadcastRigidToCopyStates``
        (``MassSplitting.cs:148-165``).
        """
        self._require_setup()
        wp.launch(
            kernel=broadcast_rigid_to_copy_states_kernel,
            dim=self.num_bodies,
            inputs=[
                self.graph.data,
                body_position,
                body_orientation,
                body_velocity,
                body_angular_velocity,
                float(dt),
            ],
            device=self.device,
        )

    def average(
        self,
        body_position: wp.array,
        body_orientation: wp.array,
        inv_dt: float,
    ) -> None:
        """Run the per-iteration consensus pass.

        Replaces every per-partition copy with the body's averaged
        velocity / angular velocity. Mirrors
        :func:`~.kernels.average_and_broadcast_kernel`. Run after
        every iterate sweep (and also after the prepare pass in the
        C# reference).
        """
        self._require_setup()
        wp.launch(
            kernel=average_and_broadcast_kernel,
            dim=self.num_bodies,
            inputs=[
                self.graph.data,
                body_position,
                body_orientation,
                float(inv_dt),
            ],
            device=self.device,
        )

    def write_back(
        self,
        body_position: wp.array,
        body_orientation: wp.array,
        body_velocity: wp.array,
        body_angular_velocity: wp.array,
        inv_dt: float,
    ) -> None:
        """End-of-substep write-back from ``tiny_states`` to body store.

        Reads slot 0 of each body's section (which
        :meth:`average` made equal to all the other slots) and
        scatters into the SoA body velocity arrays. Mirrors C#
        ``MassSplitting.CopyStateIntoRigids``
        (``MassSplitting.cs:185-201``).
        """
        self._require_setup()
        wp.launch(
            kernel=copy_state_into_rigids_kernel,
            dim=self.num_bodies,
            inputs=[
                self.graph.data,
                body_position,
                body_orientation,
                body_velocity,
                body_angular_velocity,
                float(inv_dt),
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Internal.
    # ------------------------------------------------------------------

    def _require_setup(self) -> None:
        if not self._setup_complete:
            raise RuntimeError(
                "MassSplitting.setup_from_graph(...) or setup_from_coloring(...) "
                "must be called before broadcast / average / write_back"
            )
