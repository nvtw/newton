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

from collections.abc import Iterable

import numpy as np
import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_contact import (
    ContactViews,
    contact_pair_wrench_kernel,
    contact_per_contact_wrench_kernel,
    contact_views_make,
)
from newton._src.solvers.jitter.constraint_container import ConstraintContainer
from newton._src.solvers.jitter.contact_container import (
    ContactContainer,
    contact_container_swap_prev_current,
    contact_container_zeros,
)
from newton._src.solvers.jitter.contact_ingest import (
    IngestScratch,
    gather_contact_warmstart,
    ingest_contacts,
    stamp_forward_contact_map,
)
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
    _constraint_relax_kernel,
    _constraints_to_elements_kernel,
    _integrate_forces_kernel,
    _integrate_velocities_kernel,
    _update_bodies_kernel,
    pack_body_xforms_kernel,
)

# pack_body_xforms_kernel is re-exported here so existing callers
# (examples, viewer integration) keep importing it from solver_jitter.
__all__ = ["World", "pack_body_xforms_kernel"]


@wp.kernel(enable_backward=False)
def _sync_num_active_constraints_kernel(
    num_contact_columns: wp.array[wp.int32],
    joint_constraint_count: wp.int32,
    # out
    num_active_constraints: wp.array[wp.int32],
):
    """Fuse device-held ``num_contact_columns`` into ``num_active_constraints``.

    One-thread kernel: reads the latest contact-column count written
    by the ingest pipeline and writes
    ``num_active_constraints[0] = joint_constraint_count +
    num_contact_columns[0]``. Stays on-device so
    :meth:`World._ingest_and_warmstart_contacts` can update the
    partitioner / dispatcher gate without a host readback (which
    would break graph capture).
    """
    tid = wp.tid()
    if tid != 0:
        return
    num_active_constraints[0] = joint_constraint_count + num_contact_columns[0]


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
        max_contact_columns: int = 0,
        rigid_contact_max: int = 0,
        num_shapes: int = 0,
        joint_constraint_count: int | None = None,
        collision_filter_pairs: Iterable[tuple[int, int]] | None = None,
        device: wp.context.Devicelike = None,
    ):
        """Take ownership of pre-built containers.

        Args:
            bodies: Pre-allocated and populated :class:`BodyContainer`.
            constraints: Pre-allocated shared :class:`ConstraintContainer`.
            num_constraints: Joint-only active constraint count at
                construction. Grown each :meth:`step` to include the
                contact columns the ingest pipeline emits.
            substeps: Sub-steps per :meth:`step`.
            solver_iterations: PGS iterations per substep.
            velocity_relaxations: Box2D v3 TGS-soft relax passes per
                substep (``use_bias=False`` PGS sweeps that shed the
                positional drift velocity without re-introducing it).
                Default ``1`` matches Box2D v3.
            gravity: World gravity [m/s^2].
            max_contact_columns: Hard cap on
                :data:`CONSTRAINT_TYPE_CONTACT` columns per step. Sizes
                the trailing cid range, persistent
                :class:`ContactContainer` and ingest scratch. ``0``
                disables contact code paths.
            rigid_contact_max: Upper bound on contacts in the upstream
                Newton :class:`Contacts` buffer. Defaults to
                ``max_contact_columns * 6`` if ``0``.
            num_shapes: Total Newton shapes; used to pack the
                ``(shape_a, shape_b)`` RLE key into int32. Must satisfy
                ``num_shapes * num_shapes < 2**31``.
            joint_constraint_count: First cid reserved for contacts.
                Defaults to ``num_constraints``.
            collision_filter_pairs: Optional iterable of
                ``(body_a, body_b)`` pairs whose contacts must be
                ignored by the ingest pipeline (Jitter2's
                ``IgnoreCollisionBetweenFilter``). Each pair is stored
                in canonical order ``(min, max)`` and consulted by the
                ingest kernel via binary search; filtered contacts are
                dropped before any constraint column is allocated. May
                be updated at runtime via :meth:`set_collision_filter_pairs`.
            device: Warp device. Defaults to ``bodies.position.device``.
        """
        if device is None:
            self.device = bodies.position.device
        else:
            self.device = wp.get_device(device)

        self.bodies: BodyContainer = bodies
        self.constraints: ConstraintContainer = constraints

        self.num_bodies: int = int(bodies.position.shape[0])
        self.num_constraints: int = int(num_constraints)
        self.joint_constraint_count: int = (
            int(joint_constraint_count) if joint_constraint_count is not None else int(num_constraints)
        )
        self.max_contact_columns: int = int(max_contact_columns)
        self.rigid_contact_max: int = int(rigid_contact_max)
        if self.max_contact_columns > 0 and self.rigid_contact_max == 0:
            self.rigid_contact_max = self.max_contact_columns * 6
        self.num_shapes: int = int(num_shapes)

        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.velocity_relaxations = int(velocity_relaxations)
        self.gravity = wp.vec3f(float(gravity[0]), float(gravity[1]), float(gravity[2]))

        self.step_dt: float = 0.0
        self.inv_step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # Total container capacity: joint cids + reserved contact
        # trailing range. Kernels launched at this dim gate via
        # _num_active_constraints[0].
        self._constraint_capacity: int = self.joint_constraint_count + self.max_contact_columns
        if self._constraint_capacity <= 0:
            self._constraint_capacity = max(1, self.num_constraints)

        # ----- Coloring / partitioning infrastructure ------------------
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            max(1, self._constraint_capacity), dtype=ElementInteractionData, device=self.device
        )
        self._num_active_constraints: wp.array[int] = wp.array(
            [self.num_constraints], dtype=wp.int32, device=self.device
        )
        self._partitioner = IncrementalContactPartitioner(
            max_num_interactions=max(1, self._constraint_capacity),
            max_num_nodes=max(1, self.num_bodies),
            device=self.device,
            use_tile_scan=True,
        )

        # ----- Contact infrastructure (optional) -----------------------
        if self.max_contact_columns > 0:
            self._contact_container: ContactContainer = contact_container_zeros(
                self.max_contact_columns, device=self.device
            )
            self._ingest_scratch: IngestScratch | None = IngestScratch(
                rigid_contact_max=self.rigid_contact_max,
                max_contact_columns=self.max_contact_columns,
                device=self.device,
            )
            self._slot_of_contact_cur = wp.full(
                self.rigid_contact_max, -1, dtype=wp.int32, device=self.device
            )
            self._cid_of_contact_cur = wp.full(
                self.rigid_contact_max, -1, dtype=wp.int32, device=self.device
            )
            self._slot_of_contact_prev = wp.full(
                self.rigid_contact_max, -1, dtype=wp.int32, device=self.device
            )
            self._cid_of_contact_prev = wp.full(
                self.rigid_contact_max, -1, dtype=wp.int32, device=self.device
            )
        else:
            self._contact_container = contact_container_zeros(1, device=self.device)
            self._ingest_scratch = None
            self._slot_of_contact_cur = None
            self._cid_of_contact_cur = None
            self._slot_of_contact_prev = None
            self._cid_of_contact_prev = None

        self._contact_views: ContactViews | None = None
        self._contact_views_placeholder: ContactViews = self._make_placeholder_contact_views()

        # ----- Pairwise contact filter ----------------------------
        # Sorted int64 array of packed ``min(a,b) * num_bodies + max(a,b)``
        # keys consulted by the contact-ingest key kernel. A size-1
        # sentinel array (``[INT64_MAX]``) stands in when no filters
        # are registered so the kernel signature stays constant and
        # launches always have a valid pointer to bind. See
        # :meth:`set_collision_filter_pairs`.
        self._collision_filter_keys: wp.array[wp.int64]
        self._set_collision_filter_pairs_impl(collision_filter_pairs or ())

    # ------------------------------------------------------------------
    # Placeholder ContactViews helper
    # ------------------------------------------------------------------

    def _make_placeholder_contact_views(self) -> ContactViews:
        """Build a 1-element dummy :class:`ContactViews` for no-contact steps.

        The unified dispatcher kernels take ``ContactViews`` as a
        required argument; when the caller steps without a
        :class:`Contacts` buffer (contacts disabled), we still need
        something type-correct to hand in. The placeholder's arrays
        are all size-1 and never read (the contact branch of the
        dispatcher is only taken when a
        :data:`CONSTRAINT_TYPE_CONTACT` cid exists, which is only
        true after a real ``step(dt, contacts)`` call).
        """
        dummy_int = wp.zeros(1, dtype=wp.int32, device=self.device)
        dummy_vec3 = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        dummy_float = wp.zeros(1, dtype=wp.float32, device=self.device)
        return contact_views_make(
            rigid_contact_count=dummy_int,
            rigid_contact_point0=dummy_vec3,
            rigid_contact_point1=dummy_vec3,
            rigid_contact_normal=dummy_vec3,
            rigid_contact_shape0=dummy_int,
            rigid_contact_shape1=dummy_int,
            rigid_contact_match_index=dummy_int,
            rigid_contact_margin0=dummy_float,
            rigid_contact_margin1=dummy_float,
            shape_body=dummy_int,
        )

    # ------------------------------------------------------------------
    # Pairwise contact filter
    # ------------------------------------------------------------------

    def set_collision_filter_pairs(
        self, pairs: Iterable[tuple[int, int]]
    ) -> None:
        """Replace the registered body-pair contact filter.

        Each pair ``(a, b)`` instructs :meth:`step`'s contact ingest
        to drop every upstream contact whose resolved
        ``(shape_body[shape_a], shape_body[shape_b])`` matches in
        either order -- no constraint column is allocated, no
        warm-start is gathered, and the dispatcher never sees the
        contact. Intended for self-collision suppression between
        jointed limbs (Jitter2's ``IgnoreCollisionBetweenFilter``).

        Pairs are stored canonically as ``(min, max)``; duplicates and
        either argument order are idempotent. A self-pair ``(b, b)``
        is rejected with :class:`ValueError`.

        Not graph-capture-safe by itself (reallocates a device array),
        but idempotent with :meth:`step`: call it outside any
        :func:`wp.ScopedCapture` block whenever the filter set needs
        to change. The kernel that consumes the filter launches at a
        fixed dim (``rigid_contact_max``), so changing the filter list
        does not require re-capturing the graph *if* the new list
        fits in the pre-allocated array (the impl reallocates
        unconditionally -- callers that do want capture compatibility
        should set the filter once at world construction via
        :meth:`WorldBuilder.add_collision_filter_pair`).

        Args:
            pairs: Iterable of ``(body_a, body_b)`` tuples. Empty
                iterable clears the filter.
        """
        self._set_collision_filter_pairs_impl(pairs)

    def _set_collision_filter_pairs_impl(
        self, pairs: Iterable[tuple[int, int]]
    ) -> None:
        """Pack + upload the sorted filter key array.

        Packs each canonicalised ``(a, b)`` into an int64 key
        ``a * num_bodies + b`` and uploads the sorted array to the
        device. The ingest kernel consults this array via binary
        search. The always-allocated size-1 sentinel array (holding
        ``INT64_MAX``) avoids a null-pointer special case in the
        kernel when no filters are registered.
        """
        nb = int(self.num_bodies)
        packed: list[int] = []
        seen: set[tuple[int, int]] = set()
        for a, b in pairs:
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                raise ValueError(
                    f"collision filter pair must have two distinct bodies "
                    f"(got both = {a_i})"
                )
            if not (0 <= a_i < nb and 0 <= b_i < nb):
                raise IndexError(
                    f"collision filter pair ({a_i}, {b_i}) out of range "
                    f"[0, {nb}) for this World's body count"
                )
            lo = min(a_i, b_i)
            hi = max(a_i, b_i)
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            packed.append(lo * nb + hi)

        # Sorted for binary search. Include a trailing sentinel so the
        # kernel can always dereference ``keys[0]`` safely even when
        # the user registered no filters. The sentinel (INT64_MAX)
        # never collides with a real packed body pair.
        packed.sort()
        if not packed:
            arr = np.asarray([np.iinfo(np.int64).max], dtype=np.int64)
        else:
            arr = np.asarray(packed, dtype=np.int64)

        self._collision_filter_keys = wp.array(arr, dtype=wp.int64, device=self.device)
        self._collision_filter_count = int(len(packed))

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def step(self, dt: float, contacts=None, shape_body=None, picking=None) -> None:
        """Advance the world by ``dt`` seconds.

        Direct port of ``World.Step`` (Jitter2/Dynamics/World.Step.cs:1-137).
        Tracing, asserts, SetTime and the multi-threaded thread-pool wake/sleep
        signalling are all dropped -- on Warp, parallelism is implicit in the
        kernel launches.

        The internal control flow is:

        1. Ingest contacts, rebuild the element view, **build the
           graph coloring in CSR form once**
           (:meth:`IncrementalContactPartitioner.build_csr`).
        2. Substep loop (``self.substeps`` iterations). Each substep:
           apply picking forces (if any), integrate forces, run the
           PGS velocity solve (``self.solver_iterations`` iterations
           reusing the shared CSR coloring), integrate velocities,
           relax.

        Coloring runs once per step rather than once per substep, and
        is reused across every PGS iteration. This is valid because
        the constraint graph's topology is frozen inside a step
        (contacts are ingested before the loop, joints are static);
        only the accumulated impulses change across iterations, and
        those do not affect adjacency.

        Args:
            dt: Time step in seconds. Non-positive values no-op.
            contacts: Optional Newton :class:`Contacts` buffer produced
                by an external collision pipeline. **Must** have been
                built with a non-disabled ``contact_matching`` mode
                (``"sticky"`` is strongly recommended -- and required
                for stable stacking -- as it pins matched contacts'
                anchors/normals frame-to-frame in addition to populating
                ``rigid_contact_match_index`` for the warm-start gather). When present, its sorted active prefix is
                ingested into :data:`CONSTRAINT_TYPE_CONTACT` columns
                and processed by the unified PGS loop side-by-side
                with the joint constraints, following PhoenX's
                "contacts are constraints" pattern. When ``None``,
                the contact paths are skipped (and any previously
                packed contact columns are cleared).
            shape_body: ``model.shape_body`` mapping (shape id ->
                body id). Required when ``contacts`` is provided.
            picking: Optional :class:`~newton._src.solvers.jitter.picking.JitterPicking`
                instance. When provided, its
                :meth:`~newton._src.solvers.jitter.picking.JitterPicking.apply_force`
                is invoked at the start of every internal substep so
                the spring stays stiff regardless of
                :attr:`substeps`. Left at ``None`` for headless tests
                and examples that don't support mouse picking.
        """
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        if dt < 1e-7:
            return

        self.step_dt = dt
        self.inv_step_dt = 1.0 / dt
        self.substep_dt = dt / self.substeps

        # ---- Contact ingest (replaces the old narrow-phase stub) ----
        # The caller owns contact detection (a Newton CollisionPipeline
        # or any other source). We just consume the sorted, matched
        # buffer and materialise contact columns into our shared
        # constraint container.
        self._ingest_and_warmstart_contacts(contacts, shape_body)

        # Handle deferred arbiters + deactivation (no-ops still).
        self._handle_deferred_arbiters()
        self._check_deactivation()
        self._reorder_contacts()

        # Rebuild the partitioner's element view -- covers both joints
        # and newly ingested contacts in a single launch (the header is
        # type-agnostic).
        self._rebuild_elements()

        # Build the CSR graph coloring once for the entire step. The
        # adjacency + colouring are reused by every substep + every
        # PGS iteration below. ``reset`` rebuilds the adjacency (the
        # active constraint set changed through contact ingest);
        # ``build_csr`` drives JP to completion and materialises the
        # flat element_ids_by_color / color_starts / num_colors
        # layout the sweep kernels consume.
        if self._constraint_capacity > 0:
            self._partitioner.reset(self._elements, self._num_active_constraints)
            self._partitioner.build_csr()

        # Sub-stepping: picking, integrate forces, solve, integrate
        # velocities, relax. Picking is applied first so its spring
        # wrench is summed into the force accumulators before the
        # force integration consumes them.
        for _ in range(self.substeps):
            if picking is not None:
                picking.apply_force()
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

    def _ingest_and_warmstart_contacts(self, contacts, shape_body) -> None:
        """Materialise contact columns + gather warm-start from prev frame.

        Three phases:

        1. If ``contacts`` is ``None`` or contact support is disabled,
           clear the contact cid range (set ``num_active_constraints``
           back to the joint count) and early-out. The placeholder
           :class:`ContactViews` covers the dispatcher's unused
           argument slot.
        2. Validate that ``contacts`` was built with a non-disabled
           ``contact_matching`` mode (required for warm-starting; use
           ``"sticky"`` for stable stacking) -- raise at step-time
           rather than leak garbage match indices into the gather
           kernel.
        3. Swap prev/current :class:`ContactContainer` buffers (pointer
           swap, no device copy), build the per-step
           :class:`ContactViews`, and drive ingest + warm-start +
           forward-map stamp. Finally update
           ``_num_active_constraints[0]`` so the partitioner and
           dispatchers see
           ``joint_constraint_count + num_contact_columns`` cids.

        Graph-capture safe: the only host readback is the trailing
        ``num_contact_columns.numpy()`` call that syncs
        ``num_active_constraints``, which happens *outside* any
        captured region (it's right here in the Python driver). The
        ingest + gather + stamp kernels are fully device-driven.
        """
        if contacts is None or self.max_contact_columns == 0 or self._ingest_scratch is None:
            # No contacts this step: active cid count is just the joints.
            self._num_active_constraints.fill_(self.joint_constraint_count)
            self._contact_views = None
            self._ingest_scratch_last_n_contacts = 0
            return

        if getattr(contacts, "contact_matching", False) is False:
            raise ValueError(
                "Jitter solver requires the Contacts buffer to be built with "
                "a non-disabled contact_matching mode (needed for persistent "
                'warm-starting; use contact_matching="sticky" for stable '
                "stacking)."
            )
        if shape_body is None:
            raise ValueError(
                "step(dt, contacts=...) requires shape_body=model.shape_body to "
                "resolve contact shape ids to rigid-body ids."
            )

        # Build the per-step views. Cheap (pure struct pack).
        self._contact_views = contact_views_make(
            rigid_contact_count=contacts.rigid_contact_count,
            rigid_contact_point0=contacts.rigid_contact_point0,
            rigid_contact_point1=contacts.rigid_contact_point1,
            rigid_contact_normal=contacts.rigid_contact_normal,
            rigid_contact_shape0=contacts.rigid_contact_shape0,
            rigid_contact_shape1=contacts.rigid_contact_shape1,
            rigid_contact_match_index=contacts.rigid_contact_match_index,
            rigid_contact_margin0=contacts.rigid_contact_margin0,
            rigid_contact_margin1=contacts.rigid_contact_margin1,
            shape_body=shape_body,
        )

        # ---- Swap lambda buffers & forward maps ----
        # After this, cc.prev_lambdas holds last step's finished
        # lambdas; cc.lambdas is scratch for this step. Same for
        # (slot_of_contact, cid_of_contact) -- prev holds last step's
        # map, cur is the clean slate the stamp kernel will fill.
        contact_container_swap_prev_current(self._contact_container)
        self._slot_of_contact_cur, self._slot_of_contact_prev = (
            self._slot_of_contact_prev,
            self._slot_of_contact_cur,
        )
        self._cid_of_contact_cur, self._cid_of_contact_prev = (
            self._cid_of_contact_prev,
            self._cid_of_contact_cur,
        )

        # ---- Ingest this step's contact columns ----
        ingest_contacts(
            contacts=contacts,
            shape_body=shape_body,
            num_shapes=self.num_shapes,
            constraints=self.constraints,
            scratch=self._ingest_scratch,
            cid_base=self.joint_constraint_count,
            max_contact_columns=self.max_contact_columns,
            default_friction=0.5,
            device=self.device,
            num_bodies=self.num_bodies,
            filter_keys=self._collision_filter_keys,
            filter_count=self._collision_filter_count,
        )

        # ---- Warm-start lambdas from the prev frame's state ----
        gather_contact_warmstart(
            cid_base=self.joint_constraint_count,
            scratch=self._ingest_scratch,
            rigid_contact_match_index=contacts.rigid_contact_match_index,
            prev_slot_of_contact=self._slot_of_contact_prev,
            prev_cid_of_contact=self._cid_of_contact_prev,
            cc=self._contact_container,
            device=self.device,
        )

        # ---- Stamp this frame's forward map for next frame's gather ----
        stamp_forward_contact_map(
            rigid_contact_max=self.rigid_contact_max,
            cid_base=self.joint_constraint_count,
            scratch=self._ingest_scratch,
            slot_of_contact=self._slot_of_contact_cur,
            cid_of_contact=self._cid_of_contact_cur,
            device=self.device,
        )

        # ---- Update the active-constraint counter ----
        # Host-side readback of num_contact_columns[0] is OK here: we
        # are *outside* any captured region. (When the user wraps
        # step() under their own wp.ScopedCapture, they should size
        # max_contact_columns conservatively and accept the cap as
        # the worst-case active count; a graph-captured step should
        # drive the partitioner via the device-held counter directly
        # -- see _sync_num_active_constraints_kernel below.)
        self._sync_num_active_constraints()

    def _sync_num_active_constraints(self) -> None:
        """Device-only update of ``_num_active_constraints[0]``.

        Uses a tiny 1-thread kernel so nothing leaves the device.
        Called from the non-capture ingest path; under graph capture
        it still works -- the launch is size-1 and its inputs are
        device arrays.
        """
        wp.launch(
            _sync_num_active_constraints_kernel,
            dim=1,
            inputs=[
                self._ingest_scratch.num_contact_columns,
                wp.int32(self.joint_constraint_count),
            ],
            outputs=[self._num_active_constraints],
            device=self.device,
        )

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

        Consumes the CSR coloring produced once per :meth:`step` by
        :meth:`IncrementalContactPartitioner.build_csr`. Each sweep --
        one prepare + ``iterations`` iterates -- walks the CSR colour
        by colour via ``wp.capture_while(color_cursor, ...)``:

        * :meth:`~IncrementalContactPartitioner.begin_sweep` resets
          the device-side ``color_cursor`` to ``num_colors``.
        * The body kernel processes one colour's CSR slice and
          decrements ``color_cursor`` by 1 before returning.
        * Capture-while exits when the cursor hits 0.

        Contact constraints and joint constraints take the same path:
        the per-cid type tag in the column header drives per-thread
        dispatch inside the kernel (PhoenX's "contacts are
        constraints" pattern). The dispatchers take an extra
        :class:`ContactContainer` + :class:`ContactViews` argument;
        when no contacts are active we pass the placeholder views,
        which works because the contact branch is never reached
        (no CONSTRAINT_TYPE_CONTACT cid exists).
        """
        if self._constraint_capacity == 0:
            return

        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder

        # PrepareForIteration: one sweep over all colours.
        self._partitioner.begin_sweep()
        wp.capture_while(
            self._partitioner.color_cursor,
            self._capture_partition_sweep,
            kernel=_constraint_prepare_for_iteration_kernel,
            idt=idt,
            contact_views=contact_views,
        )

        # Iterate: ``iterations`` sweeps over all colours. The CSR is
        # unchanged across iterations (graph topology is frozen for
        # the whole step) -- we only need to reset the per-sweep
        # ``color_cursor`` before each pass.
        for _ in range(iterations):
            self._partitioner.begin_sweep()
            wp.capture_while(
                self._partitioner.color_cursor,
                self._capture_partition_sweep,
                kernel=_constraint_iterate_kernel,
                idt=idt,
                contact_views=contact_views,
            )

    def gather_constraint_wrenches(self, out: wp.array, contacts=None, shape_body=None) -> None:
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
        if self._constraint_capacity == 0:
            return
        if self.substep_dt <= 0.0:
            out.zero_()
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        wp.launch(
            _constraint_gather_wrenches_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self.bodies,
                wp.int32(self._constraint_capacity),
                idt,
                self._contact_container,
                contact_views,
            ],
            outputs=[out],
            device=self.device,
        )

    def num_colors_used(self) -> int:
        """Number of graph colors used by the last PGS partitioning sweep.

        Diagnostic read-only accessor intended for sim inspection,
        logging, and tuning: it answers "how many sequential partitions
        did the PGS dispatcher fan out over in the most recent substep?".
        Low values mean the constraint graph is loosely coupled and the
        solver's per-iteration cost is nearly proportional to
        ``num_constraints / num_colors``; high values indicate a densely
        connected graph (stacks, grasped objects, long rigid loops) that
        serialises more of the PGS work.

        Graph-capture-safe by design: this performs a single device-to-
        host int32 copy via ``.numpy()``, which implicitly synchronises
        and cannot be captured into a CUDA graph. Call it *outside* any
        :func:`wp.ScopedCapture` / :func:`wp.capture_launch` block --
        typically once per frame from the host for logging. The copy
        is a ~10 us round-trip; budget accordingly if you plan to poll
        on every substep.

        The value reflects the coloring :meth:`step` built for the most
        recent call -- one ``build_csr`` pass now runs once per step
        and is reused across every substep + every PGS iteration, so
        the returned number is the *shared* colour count for the whole
        step. Calling this before the first :meth:`step` returns ``0``.

        Returns:
            Number of colors (>=0) used to partition the active
            constraint graph in the last :meth:`step`.
        """
        return int(self._partitioner.num_colors.numpy()[0])

    def gather_contact_wrenches(self, out: wp.array) -> None:
        """Per-individual-contact force + torque from the last substep.

        Writes one :class:`wp.spatial_vector` per rigid contact index
        ``k`` used by the upstream :class:`Contacts` buffer -- i.e.
        ``out[k]`` corresponds 1:1 with
        ``contacts.rigid_contact_normal[k]`` /
        ``rigid_contact_point0[k]`` / ``rigid_contact_shape0[k]`` /
        etc. Inactive indices (``k >= rigid_contact_count[0]`` or
        slots masked out of the packed column) keep whatever was in
        the buffer on entry, so the output must be zeroed first to
        get a clean per-contact slice.

        Each entry is the wrench body 2 (``contacts.rigid_contact_
        shape1[k]``'s body) "felt" from this one contact during the
        most recent substep:
            ``spatial_top``    = force [N]  (along n + tangents)
            ``spatial_bottom`` = torque [N·m] (``r2 x force``)

        Force/torque are derived from the per-slot warm-started
        impulse divided by ``substep_dt``; calling this before the
        first :meth:`step` returns zeros (all impulses are zero).

        Args:
            out: :class:`wp.array` of ``wp.spatial_vector`` of length
                at least :attr:`rigid_contact_max`, on :attr:`device`.

        Graph-capture safe: launches one kernel at a fixed dim equal
        to the contact cid capacity; nothing reads back to the host.
        """
        if self.max_contact_columns == 0:
            out.zero_()
            return
        out.zero_()
        if self.substep_dt <= 0.0 or self._contact_views is None:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            contact_per_contact_wrench_kernel,
            dim=self.max_contact_columns,
            inputs=[
                self.constraints,
                self._contact_container,
                self._contact_views,
                wp.int32(self.joint_constraint_count),
                wp.int32(self._constraint_capacity),
                idt,
            ],
            outputs=[out],
            device=self.device,
        )

    def gather_contact_pair_wrenches(
        self,
        wrenches: wp.array,
        body1: wp.array,
        body2: wp.array,
        contact_count: wp.array,
    ) -> None:
        """Per-contact-column summary wrench and metadata.

        Sums the per-slot wrench across the (up to 6) active contacts
        in each packed column -- what body 2 "felt" from this group
        of contacts during the last substep:
            ``wrenches[i]`` = :class:`wp.spatial_vector(force, torque)`
            ``body1[i]``    = Jitter body index of the pair's body 1
                              (``-1`` for an inactive slot)
            ``body2[i]``    = Jitter body index of body 2 (``-1`` for
                              an inactive slot)
            ``contact_count[i]`` = number of active contacts folded
                                   into this entry

        A single shape pair with > 6 contacts spans multiple adjacent
        columns; each column reports its own partial sum. Aggregating
        "by pair" then reduces to summing consecutive entries with
        matching ``(body1, body2)``.

        All four output arrays must be length
        :attr:`max_contact_columns` and live on :attr:`device`.

        Graph-capture safe: single fixed-dim launch, no host readbacks.
        """
        if self.max_contact_columns == 0:
            return
        if self.substep_dt <= 0.0 or self._contact_views is None:
            wrenches.zero_()
            body1.fill_(-1)
            body2.fill_(-1)
            contact_count.zero_()
            return
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            contact_pair_wrench_kernel,
            dim=self.max_contact_columns,
            inputs=[
                self.constraints,
                self._contact_container,
                self._contact_views,
                wp.int32(self.joint_constraint_count),
                wp.int32(self._constraint_capacity),
                idt,
            ],
            outputs=[wrenches, body1, body2, contact_count],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Coloring helpers (private)
    # ------------------------------------------------------------------

    def _rebuild_elements(self) -> None:
        """Project every active constraint's body pair into ``_elements``.

        Launched once per :meth:`step`. The projection is type-agnostic:
        :func:`_constraints_to_elements_kernel` reads body1/body2 from
        the fixed-offset constraint header without dispatching on
        type, so a single launch covers joints + contacts uniformly.
        Launch dim is the full capacity; threads beyond the active
        count (device-held in ``_num_active_constraints``) early-out.
        """
        if self._constraint_capacity == 0:
            return
        wp.launch(
            _constraints_to_elements_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self.bodies,
                self._num_active_constraints,
                self._elements,
            ],
            device=self.device,
        )

    def _capture_partition_sweep(
        self,
        kernel=None,
        idt: wp.float32 = 0.0,
        contact_views: ContactViews | None = None,
    ) -> None:
        """Body of the ``wp.capture_while`` loop in :meth:`_solve_velocities`.

        One colour per iteration of the enclosing ``capture_while``.
        The constraint dispatch ``kernel`` reads the current colour's
        CSR slice (via ``num_colors - color_cursor``), runs its
        element work, and decrements ``color_cursor`` by 1 so the
        next iteration processes the following colour. No
        partitioner work happens here -- the CSR is prebuilt once per
        :meth:`step`.

        Launch dim stays at the full constraint capacity (Python-side
        constant, required by ``capture_while``); threads beyond the
        current colour's size early-out inside the kernel.
        """
        views = contact_views if contact_views is not None else self._contact_views_placeholder
        wp.launch(
            kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self.bodies,
                idt,
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.num_colors,
                self._partitioner.color_cursor,
                self._contact_container,
                views,
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
        """Box2D v3 TGS-soft relaxation sub-step.

        After the main solve has driven all constraints to
        ``Jv + bias = 0``, each body's velocity contains a
        positional-drift correction component. The relax pass runs
        ``relaxations`` more PGS sweeps *without* the positional bias
        so the lock rows converge to plain ``Jv = 0`` and shed any
        residual position-error velocity. This is what prevents the
        "soft-anchor impulse leak" that otherwise forces joints with
        a COM-offset body to run at
        :data:`~newton._src.solvers.jitter.constraint_container.DEFAULT_HERTZ_LINEAR`
        ``= 0`` to keep drive / limit forces from bleeding into the
        anchor. See :func:`_constraint_relax_kernel` for the
        per-constraint dispatch and
        :mod:`~newton._src.solvers.jitter.constraint_double_ball_socket`
        for the per-joint ``use_bias`` handling.
        """
        if self._constraint_capacity == 0:
            return
        if relaxations <= 0:
            return

        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder

        for _ in range(relaxations):
            self._partitioner.begin_sweep()
            wp.capture_while(
                self._partitioner.color_cursor,
                self._capture_partition_sweep,
                kernel=_constraint_relax_kernel,
                idt=idt,
                contact_views=contact_views,
            )

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
