# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Top-level Warp port of Jitter2's ``World.Step`` driver loop.

This is the skeleton that mirrors ``World.Step`` (Jitter2/Dynamics/World.Step.cs).
The real per-phase logic still lives in C# / will be ported incrementally; for
now each phase is an empty member function on :class:`World` so the control
flow / sub-stepping structure is in place and can be filled in piece by piece.

Scope of this skeleton:
    * Single constraint type: :class:`BallSocketData`.
    * Regular solve mode only -- the deterministic / islands branch from C# is
      intentionally dropped (can be reintroduced later if needed).
    * No callbacks (PreStep / PostStep / PreSubStep / PostSubStep).
    * No tracing, no asserts, no SetTime instrumentation.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer, body_container_zeros
from newton._src.solvers.jitter.constraints import (
    BallSocketData,
    ball_socket_iterate_kernel,
    ball_socket_prepare_for_iteration_kernel,
)
from newton._src.solvers.jitter.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)
from newton._src.solvers.jitter.graph_coloring_incremental import (
    IncrementalContactPartitioner,
)


@wp.kernel
def _ball_socket_to_element_kernel(
    constraints: wp.array[BallSocketData],
    num_constraints: wp.array[int],
    elements: wp.array[ElementInteractionData],
):
    """Project each :class:`BallSocketData` into the partitioner's
    :class:`ElementInteractionData` view: only the two body indices matter,
    the remaining slots are filled with ``-1``."""
    tid = wp.tid()
    if tid >= num_constraints[0]:
        return
    c = constraints[tid]
    elements[tid] = element_interaction_data_make(c.body1, c.body2, -1, -1, -1, -1, -1, -1)


@wp.kernel
def _integrate_forces_kernel(
    bodies: BodyContainer,
    gravity: wp.vec3f,
    dt: wp.float32,
):
    """Mirrors Jitter2's IntegrateForces: ``v += g * dt`` for dynamic bodies.

    Static bodies (``inverse_mass == 0``) are skipped so the world-anchor
    body stays put.
    """
    i = wp.tid()
    if bodies.inverse_mass[i] == 0.0:
        return
    bodies.velocity[i] = bodies.velocity[i] + gravity * dt


@wp.kernel
def _integrate_velocities_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Mirrors Jitter2's IntegrateVelocities: advance position + orientation.

    Static bodies (``inverse_mass == 0``) are skipped so they remain fixed.
    Orientation update uses the standard quaternion derivative
    ``q' = 0.5 * omega_quat * q``; the result is renormalised so floating
    point drift over many sub-steps doesn't blow up the rotation.
    """
    i = wp.tid()
    if bodies.inverse_mass[i] == 0.0:
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt

    omega = bodies.angular_velocity[i]
    q = bodies.orientation[i]
    # omega_quat = (omega.x, omega.y, omega.z, 0)
    omega_quat = wp.quatf(omega[0], omega[1], omega[2], 0.0)
    dq = wp.mul(omega_quat, q) * 0.5
    new_q = wp.quatf(q[0] + dq[0] * dt, q[1] + dq[1] * dt, q[2] + dq[2] * dt, q[3] + dq[3] * dt)
    bodies.orientation[i] = wp.normalize(new_q)


@wp.kernel
def pack_body_xforms_kernel(
    bodies: BodyContainer,
    xforms: wp.array[wp.transform],
):
    """Pack ``(position, orientation)`` from a :class:`BodyContainer` into a
    flat :class:`wp.transform` array suitable for ``viewer.log_shapes``.
    Exposed at module scope so examples can render a Jitter ``World`` with
    the standard Newton viewer without writing their own kernel."""
    i = wp.tid()
    xforms[i] = wp.transform(bodies.position[i], bodies.orientation[i])


class World:
    """Warp port of Jitter2's ``World`` driver.

    Holds the simulation's :class:`BodyContainer` and a flat array of
    :class:`BallSocketData` constraints, and exposes a :meth:`step` method
    that mirrors ``World.Step`` from Jitter2.
    """

    def __init__(
        self,
        num_bodies: int,
        num_ball_sockets: int,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_relaxations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        device: wp.context.Devicelike = None,
    ):
        """Pre-allocate body / constraint containers.

        Args:
            num_bodies: Capacity of the :class:`BodyContainer`.
            num_ball_sockets: Capacity of the ball-socket constraint array.
            substeps: Number of sub-steps per :meth:`step` call (mirrors
                ``World.substeps``).
            solver_iterations: PGS iterations inside ``SolveVelocities``.
            velocity_relaxations: Relaxation passes inside ``RelaxVelocities``.
            device: Warp device to allocate on.
        """
        self.device = wp.get_device(device)

        self.bodies: BodyContainer = body_container_zeros(num_bodies, device=self.device)
        self.ball_sockets: wp.array[BallSocketData] = wp.zeros(
            num_ball_sockets, dtype=BallSocketData, device=self.device
        )

        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.velocity_relaxations = int(velocity_relaxations)
        self.num_bodies = int(num_bodies)
        self.num_ball_sockets = int(num_ball_sockets)
        self.gravity = wp.vec3f(float(gravity[0]), float(gravity[1]), float(gravity[2]))

        # Mirror World.stepDt / invStepDt / substepDt -- populated by step().
        self.step_dt: float = 0.0
        self.inv_step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # ----- Coloring / partitioning infrastructure ------------------
        # The partitioner consumes a flat ElementInteractionData array; we
        # rebuild it from ``ball_sockets`` whenever the constraint topology
        # changes. The "active count" is held in a 1-element device array so
        # the partitioner sees the same source of truth (no host sync).
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            num_ball_sockets, dtype=ElementInteractionData, device=self.device
        )
        self._num_active_constraints: wp.array[int] = wp.array(
            [num_ball_sockets], dtype=wp.int32, device=self.device
        )
        # tile-scan path keeps wp.capture_while bodies allocation-free.
        self._partitioner = IncrementalContactPartitioner(
            max_num_interactions=num_ball_sockets,
            max_num_nodes=num_bodies,
            device=self.device,
            use_tile_scan=True,
        )
        # Set on construction; callers will flip this back to True when they
        # mutate constraint topology so _solve_velocities rebuilds elements.
        self._elements_dirty: bool = True

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
        """Mirrors ``IntegrateForces``: applies gravity / external forces.

        Currently only constant gravity is applied; per-body external
        forces / torques will hook in here later.
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _integrate_forces_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self.gravity, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _solve_velocities(self, iterations: int) -> None:
        """Mirrors ``SolveVelocities``: PGS over all constraints.

        Drives the :class:`IncrementalContactPartitioner` via
        ``wp.capture_while`` -- one ``capture_while`` per sweep walks the
        partitioner's ``num_remaining`` counter down to zero, launching
        the supplied indirect ball-socket kernel once per partition.

        ``capture_while`` works in both modes: standalone it captures and
        launches its own little graph; under an outer ``wp.ScopedCapture``
        it just adds a conditional-while node to the parent capture.
        Either way, the caller controls the graph lifetime -- we never
        capture or cache anything inside :class:`World`.
        """
        if self.num_ball_sockets == 0:
            return

        if self._elements_dirty:
            self._rebuild_elements()
            self._elements_dirty = False

        idt = wp.float32(1.0 / self.substep_dt)

        # PrepareForIteration: one sweep over all partitions.
        self._partitioner.reset(self._elements, self._num_active_constraints)
        wp.capture_while(
            self._partitioner.num_remaining,
            self._capture_partition_sweep,
            kernel=ball_socket_prepare_for_iteration_kernel,
            idt=idt,
        )

        # Iterate: ``iterations`` sweeps over all partitions.
        for _ in range(iterations):
            self._partitioner.reset(self._elements, self._num_active_constraints)
            wp.capture_while(
                self._partitioner.num_remaining,
                self._capture_partition_sweep,
                kernel=ball_socket_iterate_kernel,
                idt=idt,
            )

    # ------------------------------------------------------------------
    # Coloring helpers (private)
    # ------------------------------------------------------------------

    def _rebuild_elements(self) -> None:
        """Project the current ``ball_sockets`` array into ``_elements`` so
        the partitioner sees an up-to-date adjacency view. Cheap (one
        kernel launch sized by ``num_ball_sockets``) so we just rerun it
        whenever the dirty flag is set."""
        wp.launch(
            _ball_socket_to_element_kernel,
            dim=self.num_ball_sockets,
            inputs=[self.ball_sockets, self._num_active_constraints, self._elements],
            device=self.device,
        )

    def _capture_partition_sweep(self, kernel=None, idt: wp.float32 = 0.0) -> None:
        """Body of the ``wp.capture_while`` loop in :meth:`_solve_velocities`.

        Per partition we (1) advance the partitioner one colour to publish
        a fresh independent set in
        ``partitioner.partition_element_ids[:partition_count[0]]``, then
        (2) launch the supplied indirect ball-socket ``kernel`` over that
        set. The launch is sized by the constraint capacity so the dim is
        a Python-side constant; threads beyond ``partition_count[0]``
        early-out inside the kernel.
        """
        self._partitioner.launch()
        wp.launch(
            kernel,
            dim=self.num_ball_sockets,
            inputs=[
                self.ball_sockets,
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
        """Mirrors ``ForeachActiveBody``: per-body bookkeeping pass."""

    def _broad_phase_update(self) -> None:
        """Mirrors ``DynamicTree.Update``: refit/rebuild the broad-phase tree."""
