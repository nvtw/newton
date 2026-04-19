# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Top-level Warp port of Jitter2's ``World.Step`` driver loop.

This is the skeleton that mirrors ``World.Step`` (Jitter2/Dynamics/World.Step.cs).
The real per-phase logic still lives in C# / will be ported incrementally; for
now each phase is an empty member function on :class:`World` so the control
flow / sub-stepping structure is in place and can be filled in piece by piece.

Scope of this skeleton:
    * Multiple constraint types (ball-socket, hinge-angle, angular-motor,
      contacts when they land); the solver dispatches per-cid via the
      type tag at the front of every column (see
      :mod:`solver_jitter_kernels._constraint_*` kernels), so the
      :class:`World` driver itself is type-agnostic.
    * Regular solve mode only -- the deterministic / islands branch from
      C# is intentionally dropped (can be reintroduced later if needed).
    * No callbacks (PreStep / PostStep / PreSubStep / PostSubStep).
    * No tracing, no asserts, no SetTime instrumentation.

Construction: the recommended path is to build the world via
:class:`WorldBuilder` (see :mod:`world_builder`) which populates the
:class:`BodyContainer` and :class:`ConstraintContainer` and hands the
assembled state to :meth:`World.__init__`. Direct construction is
supported for advanced callers that want to manage their own SoA arrays.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import ConstraintContainer
from newton._src.solvers.jitter.graph_coloring_common import (
    ElementInteractionData,
)
from newton._src.solvers.jitter.graph_coloring_incremental import (
    IncrementalContactPartitioner,
)
from newton._src.solvers.jitter.solver_jitter_kernels import (
    _constraint_gather_wrenches_kernel,
    _constraint_iterate_kernel,
    _constraint_prepare_for_iteration_kernel,
    _constraints_to_elements_kernel,
    _integrate_forces_kernel,
    _integrate_velocities_kernel,
    _update_bodies_kernel,
    pack_body_xforms_kernel,
)

# pack_body_xforms_kernel is re-exported here so existing callers
# (examples, viewer integration) keep importing it from solver_jitter.
__all__ = ["World", "pack_body_xforms_kernel"]


class World:
    """Warp port of Jitter2's ``World`` driver.

    Holds the simulation's :class:`BodyContainer` and a single shared
    :class:`ConstraintContainer` (column-major-by-cid dword storage).
    The driver is *type-agnostic*: it never names a specific constraint
    kind, instead launching the unified
    :func:`_constraint_prepare_for_iteration_kernel` /
    :func:`_constraint_iterate_kernel` dispatchers that route per-cid on
    the type tag at the front of every column. Exposes a :meth:`step`
    method that mirrors ``World.Step`` from Jitter2.

    Constructed either directly (advanced) or, more commonly, through
    :meth:`WorldBuilder.finalize`.
    """

    def __init__(
        self,
        bodies: BodyContainer,
        constraints: ConstraintContainer,
        num_constraints: int,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_relaxations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        device: wp.context.Devicelike = None,
    ):
        """Take ownership of pre-built containers.

        Args:
            bodies: Pre-allocated and populated :class:`BodyContainer`.
                Body 0 is conventionally the static world body
                (created automatically by :class:`WorldBuilder`).
            constraints: Pre-allocated and populated shared
                :class:`ConstraintContainer`. Each column's type is
                self-describing via its first dword (``constraint_type``
                tag); the solver dispatches purely from that.
            num_constraints: Total number of *active* constraint columns
                (across all types). Must equal
                ``constraints.data.shape[1]`` unless the container has
                trailing unused capacity.
            substeps: Number of sub-steps per :meth:`step` call (mirrors
                ``World.substeps``).
            solver_iterations: PGS iterations inside ``SolveVelocities``.
            velocity_relaxations: Relaxation passes inside ``RelaxVelocities``.
            gravity: World gravity vector applied in
                :meth:`_foreach_active_body` to all bodies that have
                ``affected_by_gravity != 0``.
            device: Warp device to allocate the partitioner / element
                view on. If ``None``, derived from ``bodies.position``.
        """
        if device is None:
            self.device = bodies.position.device
        else:
            self.device = wp.get_device(device)

        self.bodies: BodyContainer = bodies
        self.constraints: ConstraintContainer = constraints

        self.num_bodies: int = int(bodies.position.shape[0])
        self.num_constraints: int = int(num_constraints)

        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.velocity_relaxations = int(velocity_relaxations)
        self.gravity = wp.vec3f(float(gravity[0]), float(gravity[1]), float(gravity[2]))

        # Mirror World.stepDt / invStepDt / substepDt -- populated by step().
        self.step_dt: float = 0.0
        self.inv_step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # ----- Coloring / partitioning infrastructure ------------------
        # The partitioner consumes a flat ElementInteractionData array.
        # We rebuild it once per step() so future contact / motor types
        # whose topology varies frame-to-frame can be added without any
        # dirty-flag plumbing.
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            max(1, self.num_constraints), dtype=ElementInteractionData, device=self.device
        )
        self._num_active_constraints: wp.array[int] = wp.array(
            [self.num_constraints], dtype=wp.int32, device=self.device
        )
        # tile-scan path keeps wp.capture_while bodies allocation-free.
        self._partitioner = IncrementalContactPartitioner(
            max_num_interactions=max(1, self.num_constraints),
            max_num_nodes=max(1, self.num_bodies),
            device=self.device,
            use_tile_scan=True,
        )

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the world by ``dt`` seconds.

        Direct port of ``World.Step`` (Jitter2/Dynamics/World.Step.cs:1-137).
        Tracing, asserts, SetTime and the multi-threaded thread-pool wake/sleep
        signalling are all dropped -- on Warp, parallelism is implicit in the
        kernel launches.
        """
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        if dt < 1e-7:
            return

        self.step_dt = dt
        self.inv_step_dt = 1.0 / dt
        self.substep_dt = dt / self.substeps

        # Narrow phase + arbiter bookkeeping + deactivation (pre-solve).
        self._narrow_phase()
        self._handle_deferred_arbiters()
        self._check_deactivation()

        # Reorder contacts (regular solve mode only).
        self._reorder_contacts()

        # Rebuild the partitioner's element view from the current
        # constraint set every step. Cheap (one kernel per type, sized
        # by that type's count) and lets future contact / motor types
        # vary their topology freely between steps.
        self._rebuild_elements()

        # Sub-stepping: integrate forces, solve, integrate velocities, relax.
        for _ in range(self.substeps):
            self._integrate_forces()
            self._solve_velocities(self.solver_iterations)
            self._integrate_velocities()
            self._relax_velocities(self.velocity_relaxations)

        # Post-solve bookkeeping.
        self._remove_broken_arbiters()
        self._update_contacts()
        self._foreach_active_body()
        self._broad_phase_update()

    # ------------------------------------------------------------------
    # Phase stubs -- to be filled in as the port progresses
    # ------------------------------------------------------------------

    def _narrow_phase(self) -> None:
        """Mirrors ``DynamicTree.EnumerateOverlaps(detect, ...)``."""

    def _handle_deferred_arbiters(self) -> None:
        """Mirrors ``HandleDeferredArbiters``."""

    def _check_deactivation(self) -> None:
        """Mirrors ``CheckDeactivation``."""

    def _reorder_contacts(self) -> None:
        """Mirrors ``ReorderContacts`` (regular solve mode)."""

    def _integrate_forces(self) -> None:
        """Mirrors per-substep ``IntegrateForces``: add cached deltas.

        The actual force/torque/gravity assembly happens once per *step*
        in :meth:`_foreach_active_body` -- this kernel just adds the
        cached ``delta_velocity`` / ``delta_angular_velocity`` to the
        live velocity, matching Jitter's two-stage split.
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _integrate_forces_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    def _solve_velocities(self, iterations: int) -> None:
        """Mirrors ``SolveVelocities``: PGS over all constraints.

        Drives the :class:`IncrementalContactPartitioner` via
        ``wp.capture_while`` -- one ``capture_while`` per sweep walks
        the partitioner's ``num_remaining`` counter down to zero,
        launching the unified
        :func:`_constraint_prepare_for_iteration_kernel` /
        :func:`_constraint_iterate_kernel` dispatcher once per
        partition. Each thread reads the per-cid type tag and routes to
        the right ``wp.func`` -- the solver itself is type-agnostic.

        ``capture_while`` works in both modes: standalone it captures
        and launches its own little graph; under an outer
        ``wp.ScopedCapture`` it just adds a conditional-while node to
        the parent capture. Either way, the caller controls the graph
        lifetime -- we never capture or cache anything inside
        :class:`World`.
        """
        if self.num_constraints == 0:
            return

        idt = wp.float32(1.0 / self.substep_dt)

        # PrepareForIteration: one sweep over all partitions.
        self._partitioner.reset(self._elements, self._num_active_constraints)
        wp.capture_while(
            self._partitioner.num_remaining,
            self._capture_partition_sweep,
            kernel=_constraint_prepare_for_iteration_kernel,
            idt=idt,
        )

        # Iterate: ``iterations`` sweeps over all partitions.
        for _ in range(iterations):
            self._partitioner.reset(self._elements, self._num_active_constraints)
            wp.capture_while(
                self._partitioner.num_remaining,
                self._capture_partition_sweep,
                kernel=_constraint_iterate_kernel,
                idt=idt,
            )

    def gather_constraint_wrenches(self, out: wp.array) -> None:
        """Write per-constraint world-frame wrenches into ``out``.

        Each ``out[cid]`` is a :class:`wp.spatial_vector` whose
        ``spatial_top`` is the world-frame force [N] and whose
        ``spatial_bottom`` is the world-frame torque [N·m] that
        constraint exerts on its ``body2``. For purely-angular
        constraints (hinge, motor) the force is zero; for the
        ball-socket the torque is the moment of the anchor force about
        body2's COM.

        Force/torque are derived from each constraint's
        ``accumulated_impulse`` divided by ``substep_dt`` -- i.e. the
        average wrench the constraint applied during the most recent
        substep. ``substep_dt`` is whichever value :meth:`step` last
        set; calling this method before the first :meth:`step` returns
        zeros (no impulse history yet).

        Args:
            out: Output buffer of dtype :class:`wp.spatial_vector` and
                length at least :attr:`num_constraints`. Must live on
                :attr:`device`.
        """
        if self.num_constraints == 0:
            return
        if self.substep_dt <= 0.0:
            out.zero_()
            return
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            _constraint_gather_wrenches_kernel,
            dim=self.num_constraints,
            inputs=[
                self.constraints,
                self.bodies,
                wp.int32(self.num_constraints),
                idt,
                out,
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Coloring helpers (private)
    # ------------------------------------------------------------------

    def _rebuild_elements(self) -> None:
        """Project every constraint's body pair into ``_elements``.

        Launched once per :meth:`step`. The projection is type-agnostic:
        :func:`_constraints_to_elements_kernel` reads body1/body2 from
        the fixed-offset constraint header without dispatching on type,
        so a single launch covers ball-sockets, hinges, motors, future
        contacts, etc.
        """
        if self.num_constraints == 0:
            return
        wp.launch(
            _constraints_to_elements_kernel,
            dim=self.num_constraints,
            inputs=[
                self.constraints,
                self._num_active_constraints,
                self._elements,
            ],
            device=self.device,
        )

    def _capture_partition_sweep(self, kernel=None, idt: wp.float32 = 0.0) -> None:
        """Body of the ``wp.capture_while`` loop in :meth:`_solve_velocities`.

        Per partition we (1) advance the partitioner one colour to publish
        a fresh independent set in
        ``partitioner.partition_element_ids[:partition_count[0]]``, then
        (2) launch the unified dispatch ``kernel`` over that set. The
        launch is sized by the constraint capacity so the dim is a
        Python-side constant; threads beyond ``partition_count[0]``
        early-out inside the kernel.
        """
        self._partitioner.launch()
        wp.launch(
            kernel,
            dim=self.num_constraints,
            inputs=[
                self.constraints,
                self.bodies,
                idt,
                self._partitioner.partition_element_ids,
                self._partitioner.partition_count,
            ],
            device=self.device,
        )

    def _integrate_velocities(self) -> None:
        """Mirrors ``IntegrateVelocities``: x += v*dt, q += 0.5*omega*q*dt."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _integrate_velocities_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _relax_velocities(self, relaxations: int) -> None:
        """Mirrors ``RelaxVelocities``: extra PGS passes with relaxed bias."""

    def _remove_broken_arbiters(self) -> None:
        """Mirrors ``RemoveBrokenArbiters``."""

    def _update_contacts(self) -> None:
        """Mirrors ``UpdateContacts``: refresh contact manifolds post-step."""

    def _foreach_active_body(self) -> None:
        """Mirrors ``ForeachActiveBody`` -> ``UpdateBodies``.

        Runs once per step (not per substep) and matches Jitter's
        ``RigidBody.Update``: applies damping, builds the per-substep
        ``delta_velocity`` / ``delta_angular_velocity`` from accumulated
        force / torque + gravity, zeros the force/torque accumulators,
        and refreshes ``inverse_inertia_world`` from the current
        orientation. The substep loop's :meth:`_integrate_forces` then
        consumes the cached deltas.

        Note: the very first ``step()`` after construction sees zero
        deltas (the user hasn't called this yet), so the first substep
        of the first step has no gravity. This matches Jitter's
        behaviour exactly (``Update`` runs at the *end* of ``Step`` so
        the deltas are prepared for the *next* call).
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _update_bodies_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self.gravity, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _broad_phase_update(self) -> None:
        """Mirrors ``DynamicTree.Update``: refit/rebuild the broad-phase tree."""
