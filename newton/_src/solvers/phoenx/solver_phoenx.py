# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of PhoenX's ``Scene.Step`` driver -- contacts + actuated
double-ball-socket joints.

Supports :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` (ball-socket
/ revolute / prismatic with optional PD drives and limits) and
:data:`CONSTRAINT_TYPE_CONTACT`. Newton's :class:`CollisionPipeline`
produces contacts; this solver consumes them.

Per-step flow mirrors ``World.Step.cs``:
    1. Ingest contacts.
    2. Rebuild element view + Jones-Plassmann colouring once.
    3. Substep loop: integrate forces + gravity, main PGS solve
       (bias=True), integrate positions, position iterate, relax
       (bias=False).
    4. Damping + rotated-inertia refresh.
    5. Clear force accumulators.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_KINEMATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    CONTACT_DWORDS,
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_pair_wrench_kernel,
    contact_per_contact_error_kernel,
    contact_per_contact_wrench_kernel,
    contact_views_make,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    contact_container_swap_prev_current,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_ingest import (
    IngestScratch,
    gather_contact_warmstart,
    ingest_contacts,
    stamp_forward_contact_map,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    MAX_COLORS,
    IncrementalContactPartitioner,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int
from newton._src.solvers.phoenx.materials import MaterialData
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _PER_WORLD_COLORING_BLOCK_DIM,
    _STRAGGLER_BLOCK_DIM,
    _build_scatter_keys_kernel,
    _choose_fast_tail_worlds_per_block,
    _constraint_gather_errors_kernel,
    _constraint_gather_wrenches_kernel,
    _constraint_iterate_singleworld_kernel,
    _constraint_prepare_plus_iterate_fast_tail_kernel,
    _constraint_prepare_singleworld_kernel,
    _constraint_relax_fast_tail_kernel,
    _constraint_relax_singleworld_kernel,
    _constraints_to_elements_kernel,
    _count_elements_per_world_kernel,
    _integrate_velocities_kernel,
    _kinematic_interpolate_substep_kernel,
    _kinematic_prepare_step_kernel,
    _per_world_jp_coloring_kernel,
    _phoenx_apply_forces_and_gravity_kernel,
    _phoenx_clear_forces_kernel,
    _phoenx_update_inertia_kernel,
    _pick_threads_per_world_kernel,
    _reduce_total_colours_kernel,
    _set_kinematic_pose_batch_kernel,
    _sync_num_active_constraints_kernel,
    pack_body_xforms_kernel,
)

__all__ = [
    "DEFAULT_SHAPE_GAP",
    "PhoenXWorld",
    "pack_body_xforms_kernel",
]


#: Default contact-detection gap [m] for shapes in PhoenX scenes.
#: Generous (5 cm) so contacts emit a few frames before impact;
#: PhoenX's speculative branch decelerates closing bodies while they
#: still have a gap, which is easier to stabilise than penetration
#: recovery. Scene-scale-sensitive: override for MEMS / vehicle scales.
DEFAULT_SHAPE_GAP: float = 0.05


def _build_gravity_array(gravity, num_worlds: int, device) -> wp.array[wp.vec3f]:
    """Coerce ``gravity`` into a ``wp.array[wp.vec3f]`` of length
    ``num_worlds``. Accepts a single 3-tuple (broadcast) or an
    iterable of ``num_worlds`` 3-tuples."""
    g_list = list(gravity) if hasattr(gravity, "__iter__") else None
    if g_list is not None and len(g_list) == 3 and not hasattr(g_list[0], "__iter__"):
        vec = (float(g_list[0]), float(g_list[1]), float(g_list[2]))
        data = [vec] * num_worlds
    else:
        if g_list is None:
            raise ValueError("gravity must be a tuple or an iterable of tuples")
        if len(g_list) != num_worlds:
            raise ValueError(f"gravity length {len(g_list)} != num_worlds {num_worlds}")
        data = []
        for row in g_list:
            r = tuple(row)
            if len(r) != 3:
                raise ValueError(f"per-world gravity entry must be 3 components; got {len(r)}")
            data.append((float(r[0]), float(r[1]), float(r[2])))
    arr_np = np.asarray(data, dtype=np.float32)
    return wp.array(arr_np, dtype=wp.vec3f, device=device)


class PhoenXWorld:
    """PhoenX solver driver. Owns a :class:`BodyContainer`, a
    :class:`ConstraintContainer`, and per-step contact ingest /
    warm-start state. :meth:`step` advances every rigid body by ``dt``
    seconds.

    Joint columns occupy cids ``[0, num_joints)``; contact columns
    occupy ``[num_joints, num_joints + max_contact_columns)``. Only
    :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` joints are
    supported; call :meth:`initialize_actuated_double_ball_socket_joints`
    to populate them before the first :meth:`step`.
    """

    def __init__(
        self,
        bodies: BodyContainer,
        constraints: ConstraintContainer,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 1,
        gravity: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        rigid_contact_max: int = 0,
        num_joints: int = 0,
        collision_filter_pairs: Iterable[tuple[int, int]] | None = None,
        default_friction: float = 0.5,
        num_worlds: int = 1,
        step_layout: str = "multi_world",
        threads_per_world: int | str = "auto",
        device: wp.context.Devicelike = None,
    ):
        """Take ownership of pre-built body and constraint containers.

        Args:
            bodies: Rigid body SoA. Caller sets initial pose, mass,
                inertia; solver reads/writes velocity, position,
                orientation, forces, rotated inertia.
            constraints: Shared container. Joints at cids
                ``[0, num_joints)``; contacts at
                ``[num_joints, num_joints + max_contact_columns)``.
                Per-constraint dword width must be at least
                ``max(CONTACT_DWORDS, ADBS_DWORDS)`` when
                ``num_joints > 0``. :meth:`make_constraint_container`
                handles the sizing.
            substeps: Substeps per :meth:`step` call.
            solver_iterations: PGS iterations per substep (main solve,
                bias=True).
            velocity_iterations: TGS-soft relax sweeps per substep
                (bias=False). Defaults to ``1``; ``0`` recovers raw
                PhoenX but accumulates Baumgarte drift on tall stacks.
            gravity: Constant world-space gravity [m/s^2]. A 3-tuple
                broadcasts; an iterable of 3-tuples gives each world
                its own vector.
            rigid_contact_max: Upper bound on the Newton ``Contacts``
                buffer's rigid-contact range. Sizes the contact-column
                capacity 1:1 (one column per ``(shape_a, shape_b)``
                pair); ``0`` disables contact paths entirely.
            num_joints: Actuated-DBS joint columns reserved at
                ``[0, num_joints)``. Populate via
                :meth:`initialize_actuated_double_ball_socket_joints`
                before the first step.
            collision_filter_pairs: Optional ``(body_a, body_b)`` pairs
                whose contacts are dropped on ingest.
            default_friction: Friction used when no per-shape material
                is registered (see :meth:`set_materials`).
            num_worlds: Number of independent sub-worlds. Gating uses
                ``BodyContainer.world_id``.
            step_layout: Solve dispatch strategy. ``"multi_world"``
                (default) is the per-world fast-tail path: one warp
                per world, scales beyond ~256 worlds. ``"single_world"``
                drives the global Jones-Plassmann colouring with
                per-colour grid launches via ``wp.capture_while``.
                Each colour is a full grid launch (``dim =
                constraint_capacity``, default 256-thread blocks),
                so every SM on the device picks up work for the
                colour -- not a single-block sweep. Wins when the
                scene is one (or a few) very big world(s); accepts
                ``num_worlds > 1`` but loses the per-world parallelism
                of the default path.
            threads_per_world: Effective threads-per-world for the
                multi-world fast-tail kernels. ``"auto"`` (default)
                picks per-step on the GPU from the colour-size
                histogram; ``32`` matches the legacy one-warp-per-world
                layout; ``16`` packs two worlds per warp (wins on
                sparse-colour scenes -- e.g. h1_flat -- once the
                fleet is large enough to keep SM occupancy up); ``8``
                packs four worlds per warp (rarely wins outright,
                gated tight in the auto picker). The launch grid is
                always ``num_worlds * 32`` lanes; reducing tpw just
                early-exits the surplus, so this knob is graph-capture
                safe and does not need re-construction to retune.
            device: Warp device. Defaults to ``bodies.position.device``.
        """
        if device is None:
            self.device = bodies.position.device
        else:
            self.device = wp.get_device(device)

        self.bodies: BodyContainer = bodies
        self.constraints: ConstraintContainer = constraints

        self.num_bodies: int = int(bodies.position.shape[0])
        # Count kinematic bodies once at construction so the per-substep
        # kinematic kernels can short-circuit when no body is scripted.
        # ``motion_type`` is written by the builder and never mutated by
        # the solver, so a single host count at init stays accurate.
        if self.num_bodies > 0:
            mt = bodies.motion_type.numpy()
            self._num_kinematic_bodies: int = int((mt == int(MOTION_KINEMATIC)).sum())
        else:
            self._num_kinematic_bodies = 0
        # One contact column covers an entire ``(shape_a, shape_b)``
        # pair regardless of contact count, so the column count equals
        # ``rigid_contact_max`` 1:1. The ``max(1, ...)`` keeps the
        # contact-free path sizing-safe (zero-length wp.array2d isn't
        # legal).
        self.rigid_contact_max: int = int(rigid_contact_max)
        self.max_contact_columns: int = max(1, self.rigid_contact_max) if self.rigid_contact_max > 0 else 0
        self.num_joints: int = int(num_joints)
        if self.num_joints < 0:
            raise ValueError(f"num_joints must be >= 0 (got {self.num_joints})")

        self.substeps = int(substeps)
        if self.substeps <= 0:
            raise ValueError(f"substeps must be >= 1 (got {self.substeps})")
        self.solver_iterations = int(solver_iterations)
        if self.solver_iterations < 1:
            raise ValueError(f"solver_iterations must be >= 1 (got {self.solver_iterations})")
        self.velocity_iterations = int(velocity_iterations)
        if self.velocity_iterations < 0:
            raise ValueError(f"velocity_iterations must be >= 0 (got {self.velocity_iterations})")

        self.num_worlds: int = int(num_worlds)
        if self.num_worlds <= 0:
            raise ValueError(f"num_worlds must be >= 1 (got {self.num_worlds})")
        if step_layout not in ("multi_world", "single_world"):
            raise ValueError(f"step_layout must be 'multi_world' or 'single_world' (got {step_layout!r})")
        self.step_layout: str = step_layout
        # Threads-per-world. ``"auto"`` lets the GPU picker decide every
        # step from colour stats; an int forces a fixed value (validated
        # against the {8, 16, 32} set the kernels have been tuned for).
        # ``_tpw_choice`` is a 1-element GPU buffer that the fast-tail
        # kernels read at the top of every launch -- see
        # :func:`_pick_threads_per_world_kernel` for the heuristic.
        if isinstance(threads_per_world, str):
            if threads_per_world != "auto":
                raise ValueError(f"threads_per_world must be 'auto' or one of (8, 16, 32) (got {threads_per_world!r})")
            # Host-side fast-path: the picker's saturation gate for
            # tpw=16 is "num_worlds >= 8 * sm_count". Below that, the
            # picker would always emit tpw=32, so we skip the per-step
            # launch entirely and pin to 32. Avoids ~10us/step of
            # picker overhead on small fleets where it would never
            # pay off anyway.
            _sm = getattr(self.device, "sm_count", 0) or 1
            if self.num_worlds < 8 * _sm:
                self._tpw_auto: bool = False
                initial_tpw = _STRAGGLER_BLOCK_DIM
            else:
                self._tpw_auto = True
                initial_tpw = _STRAGGLER_BLOCK_DIM
        else:
            tpw_int = int(threads_per_world)
            if tpw_int not in (8, 16, 32):
                raise ValueError(f"threads_per_world must be 'auto' or one of (8, 16, 32) (got {tpw_int})")
            self._tpw_auto = False
            initial_tpw = tpw_int
        self._tpw_choice: wp.array[wp.int32] = wp.array([initial_tpw], dtype=wp.int32, device=self.device)
        # Scratch for the picker's parallel colour-count reduction.
        # Reset to 0 each step before the reduction kernel runs.
        self._tpw_total_colours: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.gravity: wp.array[wp.vec3f] = _build_gravity_array(gravity, self.num_worlds, self.device)
        self.default_friction = float(default_friction)

        # ----- Step time bookkeeping -----
        self.step_dt: float = 0.0
        self.inv_step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # Joint cids at ``[0, num_joints)``; contact cids follow.
        self._constraint_capacity: int = max(1, self.num_joints + self.max_contact_columns)

        # ----- Partitioner + per-world CSR buffers -----
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            self._constraint_capacity, dtype=ElementInteractionData, device=self.device
        )
        # Joints are the only active cids until the first ingest.
        self._num_active_constraints: wp.array[int] = wp.array([self.num_joints], dtype=wp.int32, device=self.device)
        self._partitioner = IncrementalContactPartitioner(
            max_num_interactions=self._constraint_capacity,
            max_num_nodes=max(1, self.num_bodies),
            device=self.device,
            use_tile_scan=True,
            # Contact cids occupy ``[num_joints, num_joints + max_contact_columns)``.
            # Biasing their JP priorities upward clusters contacts into earlier
            # colours and joints into later colours, cutting warp divergence
            # in the fast-tail iterate kernel (contacts and joints take
            # different ``constraint_iterate`` branches).
            contact_cid_start=self.num_joints,
        )

        cap = self._constraint_capacity
        nw = self.num_worlds
        self._world_element_ids_by_color: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._world_color_starts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_csr_offsets: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        # Shifted staging buffer for the cross-world prefix scan. Sized
        # ``num_worlds + 1`` so the inclusive scan input/output lengths
        # match and the scan result lands directly in
        # ``world_csr_offsets``.
        self._world_totals_shifted: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        self._world_num_colors: wp.array[wp.int32] = wp.zeros(nw, dtype=wp.int32, device=self.device)

        # ----- Per-world coloring scratch (for parallel JP path) -----
        #
        # The per-world coloring path parallelises the JP MIS loop over
        # worlds (one block per world); it's the only multi_world
        # coloring strategy now. Single-world layouts route through
        # ``partitioner.build_csr`` instead -- see the dispatch in
        # :meth:`step`.
        self._per_world_element_count: wp.array[wp.int32] = wp.zeros(nw, dtype=wp.int32, device=self.device)
        # Exclusive prefix of per-world counts; sized nw+1 so
        # ``world_element_offsets[w+1] - world_element_offsets[w]`` is
        # always a legal read.
        self._per_world_element_offsets: wp.array[wp.int32] = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        # Flat buffer that holds each world's active cids contiguously
        # (bucketed by world_id of ``bodies[0]``). Sized to ``2 * cap``
        # because the deterministic scatter uses
        # :func:`wp.utils.radix_sort_pairs`, which needs a ping-pong
        # second half. Only the first ``cap`` entries (the sort
        # output) are read by the JP coloring; the second half is
        # scratch.
        self._per_world_elements: wp.array[wp.int32] = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        # Per-cid world-id keys for the scatter sort. Same ping-pong
        # sizing as ``_per_world_elements``.
        self._per_world_scatter_keys: wp.array[wp.int32] = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        # Per-element colour assignment (1-based; 0 = unassigned). This
        # is the primary output of the per-world JP kernel and also
        # doubles as the scratch the kernel clears at the start of each
        # step.
        self._per_world_assigned: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._world_color_counts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_color_cursor: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )

        # ----- Contact infrastructure -----
        # Contact state lives in TWO separate containers, both sized for
        # just contacts (not joints), so the joint-wide constraint
        # buffer doesn't allocate padding for contact columns.
        #
        #   :class:`ContactContainer`        -- keyed by contact index
        #     ``k`` (into the rigid_contact_max buffer). Holds per-
        #     contact warm-start (lambdas), prev-step lambdas, and
        #     per-substep derived scratch (r1/r2, eff_n, bias, ...).
        #
        #   :class:`ContactColumnContainer`  -- keyed by local contact
        #     column cid ``[0, max_contact_columns)``. Holds the 7-dword
        #     column header (type, body1, body2, friction,
        #     friction_dynamic, contact_first, contact_count). Split
        #     out so ``ConstraintContainer`` stays joint-sized (154
        #     dwords) without allocating 147 padding dwords per contact
        #     column -- cut ~1.3 GB at h1_flat 4096 worlds.
        if self.max_contact_columns > 0:
            self._contact_container: ContactContainer = contact_container_zeros(
                self.rigid_contact_max, device=self.device
            )
            self._contact_cols: ContactColumnContainer = contact_column_container_zeros(
                self.max_contact_columns, device=self.device
            )
            self._ingest_scratch: IngestScratch | None = IngestScratch(
                rigid_contact_max=self.rigid_contact_max,
                max_contact_columns=self.max_contact_columns,
                device=self.device,
            )
            # Forward map stamps cid (no slot); prev-frame state is
            # keyed by the contact's sorted-buffer index ``k``.
            self._cid_of_contact_cur = wp.full(self.rigid_contact_max, -1, dtype=wp.int32, device=self.device)
            self._cid_of_contact_prev = wp.full(self.rigid_contact_max, -1, dtype=wp.int32, device=self.device)
        else:
            self._contact_container = contact_container_zeros(1, device=self.device)
            self._contact_cols = contact_column_container_zeros(1, device=self.device)
            self._ingest_scratch = None
            self._cid_of_contact_cur = None
            self._cid_of_contact_prev = None

        self._contact_views: ContactViews | None = None
        self._contact_views_placeholder: ContactViews = self._make_placeholder_contact_views()

        # ----- Pairwise contact filter (packed int64 keys) -----
        self._collision_filter_keys: wp.array[wp.int64]
        self._set_collision_filter_pairs_impl(collision_filter_pairs or ())

        # ----- Optional material table -----
        self._shape_material: wp.array[wp.int32] | None = None
        self._materials: wp.array[MaterialData] | None = None
        # Optional internally-stored shape_body. Set by
        # :class:`WorldBuilder` via :meth:`set_shape_body` when shapes
        # were declared through the builder API. When set, ``step()``
        # reads from here if the caller doesn't pass ``shape_body``
        # explicitly (Newton-Model callers still pass it through).
        self._shape_body_internal: wp.array[wp.int32] | None = None
        # Lazy zero-length sentinel substituted for optional
        # per-contact stiffness / damping / friction arrays when the
        # user's :class:`~newton.Contacts` didn't allocate them.
        # Allocated on first step that needs it.
        self._soft_contact_sentinel: wp.array[wp.float32] | None = None

    # ------------------------------------------------------------------
    # Material system / collision filters / placeholder contact views
    # ------------------------------------------------------------------

    def set_materials(
        self,
        materials: wp.array | None,
        shape_material: wp.array | None,
    ) -> None:
        """Install per-shape friction materials. ``materials`` is a
        ``wp.array[MaterialData]``; ``shape_material`` is an
        ``wp.array[int32]`` mapping shape index -> material index."""
        self._materials = materials
        self._shape_material = shape_material

    def set_shape_body(self, shape_body: wp.array | None) -> None:
        """Install the shape -> body map used by contact ingest.

        Populated by :class:`WorldBuilder` after shapes are declared;
        once set, :meth:`step` doesn't need ``shape_body=...`` from
        the caller. Pass ``None`` to clear (e.g. when switching to a
        Newton-Model-driven flow that supplies the array per step).
        """
        self._shape_body_internal = shape_body

    # ------------------------------------------------------------------
    # Kinematic pose scripting
    # ------------------------------------------------------------------

    def set_kinematic_pose(
        self,
        body: int,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """Script a kinematic body's end-of-next-step pose.

        The next :meth:`step` call will: (1) snapshot the body's
        current pose as the lerp / slerp origin, (2) infer the linear
        and angular velocity needed to land on the new pose by the
        end of the step, and (3) lerp the origin and slerp the
        orientation across substeps so contacts see smooth motion.

        Args:
            body: Body index. Must be a kinematic body (added via
                :meth:`WorldBuilder.add_kinematic_body` or a Newton
                body with :data:`~newton.BodyFlags.KINEMATIC` set).
                Calling on a dynamic or static body raises
                ``ValueError``.
            position: Target origin in world frame [m].
            orientation: Target orientation as a unit quaternion
                ``(x, y, z, w)``.

        Batched calls with hundreds of kinematic bodies should use
        :meth:`set_kinematic_poses_batch` -- this method does a
        single-element host->device roundtrip per call.
        """
        # Host-side motion-type validation. Fetching one int per call
        # is not free, but it trades a GPU sync for a clear error when
        # users misaddress bodies, which matches the "hard to screw up"
        # ethos of the WorldBuilder API.
        if body < 0 or body >= self.num_bodies:
            raise IndexError(f"set_kinematic_pose: body index {body} out of range [0, {self.num_bodies})")
        mt = int(self.bodies.motion_type.numpy()[body])
        if mt != int(MOTION_KINEMATIC):
            raise ValueError(
                f"set_kinematic_pose(body={body}): body motion_type is "
                f"{mt} (not MOTION_KINEMATIC={int(MOTION_KINEMATIC)}). "
                "Only kinematic bodies can be pose-scripted."
            )
        body_arr = wp.array([int(body)], dtype=wp.int32, device=self.device)
        pos_arr = wp.array([tuple(float(c) for c in position)], dtype=wp.vec3f, device=self.device)
        orient_arr = wp.array(
            [tuple(float(c) for c in orientation)],
            dtype=wp.quatf,
            device=self.device,
        )
        wp.launch(
            _set_kinematic_pose_batch_kernel,
            dim=1,
            inputs=[self.bodies, body_arr, pos_arr, orient_arr],
            device=self.device,
        )

    def set_kinematic_poses_batch(
        self,
        body_ids: wp.array,
        positions: wp.array,
        orientations: wp.array,
    ) -> None:
        """Batched variant of :meth:`set_kinematic_pose`.

        Args:
            body_ids: ``wp.array[int32]`` of kinematic body indices.
            positions: ``wp.array[wp.vec3f]``, parallel to ``body_ids``.
            orientations: ``wp.array[wp.quatf]``, parallel to ``body_ids``.

        Non-kinematic entries in ``body_ids`` are silently ignored by
        the kernel -- the caller is expected to pre-filter on the
        host if strict validation matters. This matches the Newton
        adapter, which pushes *all* body indices through the same
        kernel and lets the motion-type check gate.
        """
        n = int(body_ids.shape[0])
        if n == 0:
            return
        if positions.shape[0] != n or orientations.shape[0] != n:
            raise ValueError(
                "set_kinematic_poses_batch: body_ids, positions, orientations "
                f"must share length (got {n}, {positions.shape[0]}, "
                f"{orientations.shape[0]})"
            )
        wp.launch(
            _set_kinematic_pose_batch_kernel,
            dim=n,
            inputs=[self.bodies, body_ids, positions, orientations],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Joint initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def required_constraint_dwords(num_joints: int) -> int:
        """Dword width of the joint-only :class:`ConstraintContainer`.

        Contacts now live in a dedicated narrow
        :class:`ContactColumnContainer`; the
        :class:`ConstraintContainer` is sized strictly for joints and
        only needs the 154-dword ADBS header when any joints are
        reserved (otherwise the minimal ``CONTACT_DWORDS = 7`` is
        enough to satisfy the shared header contract on the unused
        single-row allocation).
        """
        if int(num_joints) > 0:
            return int(ADBS_DWORDS)
        return int(CONTACT_DWORDS)

    @staticmethod
    def make_constraint_container(
        num_joints: int,
        max_contact_columns: int,
        device: wp.context.Devicelike = None,
    ) -> ConstraintContainer:
        """Factory for a correctly-sized joint-only
        :class:`ConstraintContainer`.

        Capacity is ``num_joints`` (not ``num_joints +
        max_contact_columns`` as before): contact columns moved to
        :class:`ContactColumnContainer` to avoid allocating the wide
        ADBS header for every contact slot. ``max_contact_columns``
        is kept in the signature for API compatibility but ignored.
        """
        _ = max_contact_columns  # reserved for API compat; contacts live elsewhere
        cap = max(1, int(num_joints))
        return constraint_container_zeros(
            num_constraints=cap,
            num_dwords=PhoenXWorld.required_constraint_dwords(num_joints),
            device=device,
        )

    def initialize_actuated_double_ball_socket_joints(
        self,
        body1: wp.array,
        body2: wp.array,
        anchor1: wp.array,
        anchor2: wp.array,
        hertz: wp.array,
        damping_ratio: wp.array,
        joint_mode: wp.array,
        drive_mode: wp.array,
        target: wp.array,
        target_velocity: wp.array,
        max_force_drive: wp.array,
        stiffness_drive: wp.array,
        damping_drive: wp.array,
        min_value: wp.array,
        max_value: wp.array,
        hertz_limit: wp.array,
        damping_ratio_limit: wp.array,
        stiffness_limit: wp.array,
        damping_limit: wp.array,
    ) -> None:
        """Pack ``num_joints`` actuated-DBS joint columns.

        One-shot launch over cids ``[0, num_joints)``. All input
        arrays must be length ``num_joints``, live on ``device``, and
        match the kernel's dtypes (see the kernel signature in
        :mod:`constraint_actuated_double_ball_socket`). Call once,
        after :meth:`__init__`, before the first :meth:`step`.

        Args:
            body1, body2: ``wp.array[int32]`` per-joint body indices.
            anchor1, anchor2: ``wp.array[vec3f]`` world-space anchors
                at init. Line from ``anchor1`` to ``anchor2`` defines
                the joint axis for revolute / prismatic modes.
            hertz, damping_ratio: Positional-block soft-constraint
                frequency [Hz] and damping ratio.
            joint_mode: Per-joint
                :class:`JointMode` enum (``wp.array[int32]``).
            drive_mode: Per-joint
                :class:`DriveMode` enum.
            target, target_velocity: PD setpoints [rad / rad/s for
                revolute, m / m/s for prismatic].
            max_force_drive, stiffness_drive, damping_drive: Drive
                impulse cap and PD gains. ``stiffness_drive ==
                damping_drive == 0`` disables the drive row.
            min_value, max_value: Limit window (``min > max``
                disables).
            hertz_limit, damping_ratio_limit: Soft-constraint limit
                knobs (used when both PD gains are zero).
            stiffness_limit, damping_limit: Limit PD gains (SI). Any
                strictly positive value selects the PD formulation.
        """
        if self.num_joints <= 0:
            return
        wp.launch(
            actuated_double_ball_socket_initialize_kernel,
            dim=self.num_joints,
            inputs=[
                self.constraints,
                self.bodies,
                wp.int32(0),  # cid_offset -- joints start at cid 0
                body1,
                body2,
                anchor1,
                anchor2,
                hertz,
                damping_ratio,
                joint_mode,
                drive_mode,
                target,
                target_velocity,
                max_force_drive,
                stiffness_drive,
                damping_drive,
                min_value,
                max_value,
                hertz_limit,
                damping_ratio_limit,
                stiffness_limit,
                damping_limit,
            ],
            device=self.device,
        )

    def set_collision_filter_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Replace the registered body-pair contact filter. Pairs are
        canonicalised ``(min, max)`` and deduped; self-pairs rejected;
        contact lookup is a device-side binary search."""
        self._set_collision_filter_pairs_impl(pairs)

    def _set_collision_filter_pairs_impl(self, pairs: Iterable[tuple[int, int]]) -> None:
        nb = int(self.num_bodies)
        packed: list[int] = []
        seen: set[tuple[int, int]] = set()
        for a, b in pairs:
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                raise ValueError(f"collision filter pair must have two distinct bodies (got both = {a_i})")
            if not (0 <= a_i < nb and 0 <= b_i < nb):
                raise IndexError(
                    f"collision filter pair ({a_i}, {b_i}) out of range [0, {nb}) for this World's body count"
                )
            lo = min(a_i, b_i)
            hi = max(a_i, b_i)
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            packed.append(lo * nb + hi)

        packed.sort()
        if not packed:
            arr = np.asarray([np.iinfo(np.int64).max], dtype=np.int64)
        else:
            arr = np.asarray(packed, dtype=np.int64)

        self._collision_filter_keys = wp.array(arr, dtype=wp.int64, device=self.device)
        self._collision_filter_count = int(len(packed))

    def _make_placeholder_contact_views(self) -> ContactViews:
        """Size-1 dummy :class:`ContactViews` for contact-free steps.
        Never actually read because the contact branch only fires for
        cids tagged :data:`CONSTRAINT_TYPE_CONTACT`."""
        dummy_int = wp.zeros(1, dtype=wp.int32, device=self.device)
        dummy_vec3 = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        dummy_float = wp.zeros(1, dtype=wp.float32, device=self.device)
        # Soft-contact stiffness / damping / friction arrays can be
        # sentinel length 0 when the caller's ``Contacts`` didn't
        # allocate them. The prepare kernel gates on the array
        # length per-contact.
        sentinel_float = wp.zeros(0, dtype=wp.float32, device=self.device)
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
            rigid_contact_stiffness=sentinel_float,
            rigid_contact_damping=sentinel_float,
            rigid_contact_friction=sentinel_float,
        )

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def step(self, dt: float, contacts=None, shape_body=None, picking=None) -> None:
        """Advance the world by ``dt`` seconds.

        Phases: ingest contacts -> rebuild elements + JP colouring ->
        substep loop (forces, gravity, main solve, integrate
        positions, position iterate, relax) -> damping + rotated
        inertia -> clear forces.

        Args:
            dt: Time step [s]. Non-positive values no-op.
            contacts: Newton :class:`Contacts` buffer. Must have been
                built with a non-disabled ``contact_matching`` mode
                (``"sticky"`` recommended for stable stacking).
                ``None`` runs a free-fall step with no contact
                constraints.
            shape_body: ``model.shape_body`` (shape id -> body id).
                Required when ``contacts`` is provided.
            picking: Optional :class:`Picking` instance. When
                provided, its ``apply_force()`` is called at the
                start of every substep so the pick spring stays stiff
                regardless of :attr:`substeps`. Callers driving
                picking themselves should leave this ``None``.
        """
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        if dt < 1e-7:
            return

        self.step_dt = dt
        self.inv_step_dt = 1.0 / dt
        self.substep_dt = dt / self.substeps

        self._ingest_and_warmstart_contacts(contacts, shape_body)

        self._rebuild_elements()
        if self._constraint_capacity > 0:
            self._partitioner.reset(self._elements, self._num_active_constraints)
            if self.step_layout == "single_world":
                # Single-world step path needs only the global CSR;
                # the per-world bucketing scaffolding is unused.
                self._partitioner.build_csr()
            else:
                # Multi-world: parallel per-world JP coloring. The
                # adjacency build is still global (per-body CSR of
                # element neighbours), but the MIS loop runs one
                # block per world and is the only multi_world
                # strategy now.
                self._build_per_world_coloring()

            # Per-step adaptive threads-per-world pick. Reads the freshly
            # built per-world colour stats and writes ``_tpw_choice[0]``;
            # the fast-tail kernels read it at every launch in the
            # substep loop below. Skipped when the user pinned a static
            # tpw at construction or when the single-world layout is in
            # use (its kernels don't read ``_tpw_choice``).
            if self._tpw_auto and self.step_layout != "single_world":
                self._pick_tpw()

        # Once-per-step kinematic prepare: snapshot ``position`` ->
        # ``position_prev``, resolve this step's pose target
        # (user-scripted or auto from constant velocity), and infer
        # the linear / angular velocity the solver exposes to contacts.
        # Dynamic and static bodies are a no-op.
        self._kinematic_prepare_step()

        # Substep loop ordering: solve with bias ON, integrate, position
        # iterate, then relax with bias OFF. Running relax before
        # integrate would throw away the positional bias's contribution
        # to penetration recovery. Kinematic bodies are slotted into
        # their lerp/slerp-interpolated pose at each substep's tail so
        # contacts prepared at the *next* substep see smooth motion.
        inv_n = 1.0 / float(self.substeps)
        for k in range(self.substeps):
            if picking is not None:
                picking.apply_force()
            self._integrate_forces_and_gravity()
            if self.step_layout == "single_world":
                self._solve_main_singleworld()
                self._integrate_positions()
                self._relax_velocities_singleworld()
            else:
                self._solve_main()
                self._integrate_positions()
                self._relax_velocities()
            alpha = float(k + 1) * inv_n
            self._kinematic_interpolate_substep(alpha)

        self._update_inertia()
        self._clear_forces()

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _ingest_and_warmstart_contacts(self, contacts, shape_body) -> None:
        """Translate Newton's ``Contacts`` buffer into contact columns.

        No contacts this step -> fall back to the joint-only active
        count. Otherwise swap prev/current per-cid state, run
        ingest -> warm-start -> forward-map stamp, then fuse joint +
        contact counts on-device.
        """
        if contacts is None or self.max_contact_columns == 0 or self._ingest_scratch is None:
            self._num_active_constraints.fill_(self.num_joints)
            self._contact_views = None
            return

        if getattr(contacts, "contact_matching", False) is False:
            raise ValueError(
                "PhoenX solver requires the Contacts buffer to be built with "
                "a non-disabled contact_matching mode (needed for persistent "
                'warm-starting; use contact_matching="sticky" for stable '
                "stacking)."
            )
        if shape_body is None:
            shape_body = self._shape_body_internal
        if shape_body is None:
            raise ValueError(
                "step(dt, contacts=...) requires shape_body to resolve contact "
                "shape ids to rigid-body ids. Pass it explicitly (e.g. "
                "``model.shape_body``), or register shapes via "
                "``WorldBuilder.add_shape_*`` so it gets installed at finalize."
            )

        # Soft-contact per-contact arrays are optional on Newton's
        # ``Contacts``: only allocated when
        # ``per_contact_shape_properties=True``. When missing we pass
        # a length-0 sentinel so the Warp kernels' per-contact
        # ``array.shape[0] > k`` check short-circuits cleanly.
        if self._soft_contact_sentinel is None:
            self._soft_contact_sentinel = wp.zeros(0, dtype=wp.float32, device=self.device)
        contact_stiffness = (
            contacts.rigid_contact_stiffness
            if getattr(contacts, "rigid_contact_stiffness", None) is not None
            else self._soft_contact_sentinel
        )
        contact_damping = (
            contacts.rigid_contact_damping
            if getattr(contacts, "rigid_contact_damping", None) is not None
            else self._soft_contact_sentinel
        )
        contact_friction = (
            contacts.rigid_contact_friction
            if getattr(contacts, "rigid_contact_friction", None) is not None
            else self._soft_contact_sentinel
        )

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
            rigid_contact_stiffness=contact_stiffness,
            rigid_contact_damping=contact_damping,
            rigid_contact_friction=contact_friction,
        )

        # Swap lambda buffers + forward map (pointer swap, O(1)).
        contact_container_swap_prev_current(self._contact_container)
        self._cid_of_contact_cur, self._cid_of_contact_prev = (
            self._cid_of_contact_prev,
            self._cid_of_contact_cur,
        )

        ingest_contacts(
            contacts=contacts,
            shape_body=shape_body,
            contact_cols=self._contact_cols,
            scratch=self._ingest_scratch,
            max_contact_columns=self.max_contact_columns,
            default_friction=self.default_friction,
            device=self.device,
            num_bodies=self.num_bodies,
            filter_keys=self._collision_filter_keys,
            filter_count=self._collision_filter_count,
            shape_material=self._shape_material,
            materials=self._materials,
        )

        gather_contact_warmstart(
            cid_base=self.num_joints,
            scratch=self._ingest_scratch,
            rigid_contact_match_index=contacts.rigid_contact_match_index,
            prev_cid_of_contact=self._cid_of_contact_prev,
            bodies=self.bodies,
            contacts=self._contact_views,
            cc=self._contact_container,
            device=self.device,
        )

        stamp_forward_contact_map(
            rigid_contact_max=self.rigid_contact_max,
            cid_base=self.num_joints,
            scratch=self._ingest_scratch,
            cid_of_contact=self._cid_of_contact_cur,
            device=self.device,
        )

        self._sync_num_active_constraints()

    def _sync_num_active_constraints(self) -> None:
        """Fuse ``num_joints + num_contact_columns`` into
        ``_num_active_constraints`` on-device (graph-capture safe)."""
        wp.launch(
            _sync_num_active_constraints_kernel,
            dim=1,
            inputs=[
                self._ingest_scratch.num_contact_columns,
                wp.int32(self.num_joints),
            ],
            outputs=[self._num_active_constraints],
            device=self.device,
        )

    def _rebuild_elements(self) -> None:
        """Project every active constraint into the partitioner's
        element view. Type-agnostic launch at
        ``dim = constraint_capacity``; threads beyond the device-held
        ``num_active_constraints`` early-out."""
        if self._constraint_capacity == 0:
            return
        wp.launch(
            _constraints_to_elements_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                self._num_active_constraints,
                wp.int32(self.num_joints),
                self._elements,
            ],
            device=self.device,
        )

    def _build_per_world_coloring(self) -> None:
        """Parallel per-world Jones-Plassmann coloring.

        1. :func:`_count_elements_per_world_kernel` -- atomic count of
           active cids per world. The atomics are commutative so the
           counts themselves are bit-deterministic regardless of GPU
           thread scheduling.
        2. Inclusive scan of the counts -> ``per_world_element_offsets``.
        3. :func:`_build_scatter_keys_kernel` + stable
           ``radix_sort_pairs`` -- bucket cids into
           ``per_world_elements`` by world id. Stable sort keeps the
           cid order within each bucket fixed, replacing an
           atomic-cursor scatter that produced a thread-scheduling-
           dependent order.
        4. :func:`_per_world_jp_coloring_kernel` -- one block per
           world, runs the full JP MIS loop on its bucket and emits
           directly into ``world_element_ids_by_color`` /
           ``world_color_starts`` / ``world_num_colors``. Output slot
           assignment within a colour uses
           ``wp.tile_scan_exclusive`` (also deterministic) instead of
           the older atomic cursor.

        The adjacency CSR from ``partitioner.reset`` is reused
        (worlds have disjoint body sets after static-null-out, so
        each element's neighbours all live in the same world).
        """
        nw = self.num_worlds
        cap = self._constraint_capacity

        # Phase 1: count elements per world (and seed the shifted-offset
        # buffer so the inclusive scan below produces exclusive offsets
        # + trailing total).
        self._per_world_element_count.zero_()
        self._world_totals_shifted.zero_()
        wp.launch(
            _count_elements_per_world_kernel,
            dim=cap,
            inputs=[self._elements, self._num_active_constraints, self.bodies],
            outputs=[self._per_world_element_count, self._world_totals_shifted],
            device=self.device,
        )

        # Phase 2: inclusive scan of shifted counts -> per_world_element_offsets[0]=0,
        # [w+1] = sum(counts[0..w]). Deterministic.
        wp.utils.array_scan(self._world_totals_shifted, self._per_world_element_offsets, inclusive=True)

        # Phase 3: deterministic scatter via stable sort by world id.
        # ``_build_scatter_keys_kernel`` populates per-cid (key=world_id,
        # value=cid) pairs (with INT32_MAX keys masking inactive
        # cids); ``sort_variable_length_int`` runs a stable
        # ``radix_sort_pairs`` so that the value array becomes the
        # per-world bucketed cid stream, ordered by cid within each
        # world. This replaces an atomic-cursor scatter whose order
        # depended on GPU thread scheduling -- the sort is a function
        # of (world_id, cid) and therefore bit-deterministic across
        # runs.
        wp.launch(
            _build_scatter_keys_kernel,
            dim=cap,
            inputs=[self._elements, self._num_active_constraints, self.bodies, wp.int32(cap)],
            outputs=[self._per_world_scatter_keys, self._per_world_elements],
            device=self.device,
        )
        sort_variable_length_int(
            self._per_world_scatter_keys,
            self._per_world_elements,
            self._num_active_constraints,
        )

        # Phase 4: per-world JP coloring. One block per world.
        self._per_world_assigned.zero_()
        # The output CSR base offsets (world_csr_offsets) match the
        # per-world element offsets (every active element belongs to
        # some colour in some world, so offsets[w] identifies both the
        # bucket start and the start of world w's colour sequence).
        wp.copy(self._world_csr_offsets, self._per_world_element_offsets)
        # ``launch_tiled`` returns (block_idx, lane) from ``wp.tid()`` --
        # exactly the (world, lane) pair the kernel expects. Plain
        # ``launch`` would return a global thread index instead, which
        # would silently mis-address every block-scoped read/write.
        # Biased priorities: contact cids get ``contact_bias`` added on
        # the fly so they outrank every joint cid in JP. Clusters
        # contacts into earlier colours / joints into later colours,
        # reducing warp divergence in the constraint iterate kernel.
        # Setting ``contact_cid_start = capacity`` disables the bias
        # (joint-free or contact-free cases).
        if 0 < self.num_joints < self._constraint_capacity:
            contact_cid_start = self.num_joints
            contact_bias = self._constraint_capacity
        else:
            contact_cid_start = self._constraint_capacity
            contact_bias = 0
        wp.launch_tiled(
            _per_world_jp_coloring_kernel,
            dim=[nw],
            inputs=[
                self._per_world_element_offsets,
                self._per_world_element_count,
                self._per_world_elements,
                self._elements,
                self._partitioner._adjacency_section_end_indices,
                self._partitioner._vertex_to_adjacent_elements,
                self._partitioner._random_values,
                int(MAX_COLORS),
                wp.int32(contact_cid_start),
                wp.int32(contact_bias),
            ],
            outputs=[
                self._per_world_assigned,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_num_colors,
            ],
            block_dim=_PER_WORLD_COLORING_BLOCK_DIM,
            device=self.device,
        )

    def _integrate_forces_and_gravity(self) -> None:
        """Apply per-body force / torque accumulators AND gravity in one
        per-substep kernel launch (replaces two sequential launches with
        identical dim / gating)."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_apply_forces_and_gravity_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self.gravity, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _solve_main(self) -> None:
        """Main PGS solve per substep: prepare + ``solver_iterations``
        iterate sweeps with bias ON.

        Launches the fused prepare+iterate kernel so the per-world
        setup (``world_id``, ``n_colors``, ``world_base``) is computed
        once and the one kernel-launch boundary between prepare and
        iterate is removed.
        """
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        wp.launch(
            _constraint_prepare_plus_iterate_fast_tail_kernel,
            dim=self._fast_tail_launch_dim(),
            block_dim=self._fast_tail_block_dim(),
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                idt,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(self.solver_iterations),
                wp.int32(self.num_worlds),
                wp.int32(self.num_joints),
                self._tpw_choice,
            ],
            device=self.device,
        )

    def _relax_velocities(self) -> None:
        """TGS-soft relax sweeps with bias OFF. Removes the drift
        velocity the positional bias injected during the main solve."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        self._launch_fast_iter(
            _constraint_relax_fast_tail_kernel,
            self.velocity_iterations,
            idt,
            contact_views,
        )

    # ------------------------------------------------------------------
    # Single-world dispatch (capture-while over the global colour CSR)
    # ------------------------------------------------------------------

    def _capture_singleworld_sweep(self, kernel, **kw) -> None:
        """``wp.capture_while`` body: launch ``kernel`` for one colour.

        Decrements ``color_cursor`` inside the kernel; the capture-while
        exits when the cursor hits 0.
        """
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        idt = kw.get("idt", wp.float32(0.0))
        wp.launch(
            kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                idt,
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.num_colors,
                self._partitioner.color_cursor,
                self._contact_container,
                contact_views,
                wp.int32(self.num_joints),
            ],
            device=self.device,
        )

    def _solve_main_singleworld(self) -> None:
        """Single-world prepare + main PGS iterate path.

        One ``wp.capture_while`` for prepare, then ``solver_iterations``
        more capture-while loops for the bias-on iterate. Each launch
        uses the entire device on one colour, then the cursor advances.
        """
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)

        # Prepare-for-iteration sweep.
        self._partitioner.begin_sweep()
        wp.capture_while(
            self._partitioner.color_cursor,
            self._capture_singleworld_sweep,
            kernel=_constraint_prepare_singleworld_kernel,
            idt=idt,
        )

        # Main iterate sweeps (bias ON).
        for _ in range(self.solver_iterations):
            self._partitioner.begin_sweep()
            wp.capture_while(
                self._partitioner.color_cursor,
                self._capture_singleworld_sweep,
                kernel=_constraint_iterate_singleworld_kernel,
                idt=idt,
            )

    def _relax_velocities_singleworld(self) -> None:
        """Single-world TGS-soft relax sweeps (bias OFF)."""
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        for _ in range(self.velocity_iterations):
            self._partitioner.begin_sweep()
            wp.capture_while(
                self._partitioner.color_cursor,
                self._capture_singleworld_sweep,
                kernel=_constraint_relax_singleworld_kernel,
                idt=idt,
            )

    def _fast_tail_block_dim(self) -> int:
        """Resolve the fast-tail block size for this solver instance.

        Always ``_STRAGGLER_BLOCK_DIM * wpb`` so the block holds an
        integer number of warps (``__syncwarp`` correctness). ``wpb``
        depends on ``num_worlds`` -- see
        :func:`_choose_fast_tail_worlds_per_block`. The block_dim is
        independent of the adaptive ``threads_per_world`` choice; the
        latter only affects how many lanes inside each warp do real
        work.
        """
        return _STRAGGLER_BLOCK_DIM * _choose_fast_tail_worlds_per_block(self.num_worlds)

    def _fast_tail_launch_dim(self) -> int:
        """Launch dim for the fast-tail kernels, padded up to the block
        size so we can use a block_dim wider than a single warp.

        Sized for the maximum threads-per-world (``_STRAGGLER_BLOCK_DIM
        = 32``); when the adaptive picker drops the effective tpw to 16
        or 8, the surplus lanes resolve a ``world_id`` past
        ``num_worlds`` and early-exit. Keeping the launch shape fixed
        is what lets the per-step picker live inside the captured CUDA
        graph -- changing ``dim`` would require re-capturing.
        """
        block_dim = self._fast_tail_block_dim()
        raw = self.num_worlds * _STRAGGLER_BLOCK_DIM
        return ((raw + block_dim - 1) // block_dim) * block_dim

    def _pick_tpw(self) -> None:
        """Run the per-step threads-per-world picker on the GPU.

        Two-kernel pipeline so the cost stays ~constant in
        ``num_worlds``: a parallel atomic reduction over
        ``_world_num_colors`` totals the colour count, then a 1-thread
        kernel reads the precomputed total cid count
        (``_world_csr_offsets[num_worlds]``) plus the device's SM count
        and writes the chosen tpw to ``_tpw_choice[0]``. No host sync;
        the whole pipeline lands in the captured CUDA graph.
        """
        sm_count = getattr(self.device, "sm_count", 0) or 0
        self._tpw_total_colours.zero_()
        wp.launch(
            _reduce_total_colours_kernel,
            dim=self.num_worlds,
            inputs=[self._world_num_colors, wp.int32(self.num_worlds)],
            outputs=[self._tpw_total_colours],
            device=self.device,
        )
        wp.launch(
            _pick_threads_per_world_kernel,
            dim=1,
            inputs=[
                self._world_csr_offsets,
                self._tpw_total_colours,
                wp.int32(self.num_worlds),
                wp.int32(sm_count),
            ],
            outputs=[self._tpw_choice],
            device=self.device,
        )

    def _launch_fast_iter(
        self,
        kernel,
        num_iterations: int,
        idt: wp.float32,
        contact_views: ContactViews,
    ) -> None:
        """Launch an iterate / relax kernel that runs
        ``num_iterations`` sweeps internally, one warp per world."""
        wp.launch(
            kernel,
            dim=self._fast_tail_launch_dim(),
            block_dim=self._fast_tail_block_dim(),
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                idt,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(num_iterations),
                wp.int32(self.num_worlds),
                wp.int32(self.num_joints),
                self._tpw_choice,
            ],
            device=self.device,
        )

    def _integrate_positions(self) -> None:
        """``x += v * dt`` and ``q = dq(w * dt) * q`` for dynamic
        bodies. Static and kinematic bodies are skipped -- kinematic
        pose advances via
        :meth:`_kinematic_interpolate_substep`. Axis-angle quaternion
        form keeps unit norm over many substeps."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _integrate_velocities_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _kinematic_prepare_step(self) -> None:
        """Once-per-step kinematic prepare: snapshot prev pose,
        resolve target, infer velocity. No-op when no kinematic bodies
        are present in the model -- the kernel's per-thread guard
        would early-return on every thread anyway, so we can skip
        launching the grid entirely and save ~4us."""
        if self.num_bodies == 0 or self._num_kinematic_bodies == 0:
            return
        wp.launch(
            _kinematic_prepare_step_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.step_dt)],
            device=self.device,
        )

    def _kinematic_interpolate_substep(self, alpha: float) -> None:
        """Per-substep kinematic pose update via lerp / slerp. Skipped
        when no kinematic bodies exist (see :meth:`_kinematic_prepare_step`
        rationale); at substeps=4 this is ~16us / step reclaimed in
        the common dynamic-only case."""
        if self.num_bodies == 0 or self._num_kinematic_bodies == 0:
            return
        wp.launch(
            _kinematic_interpolate_substep_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(alpha)],
            device=self.device,
        )

    def _update_inertia(self) -> None:
        """Apply damping and rebuild ``inverse_inertia_world`` from
        the final orientation. Once per step, after the substep loop."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_update_inertia_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    def _clear_forces(self) -> None:
        """Zero per-body force/torque accumulators so the next
        :meth:`step` starts with an empty external load."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_clear_forces_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_constraints(self) -> int:
        """Total allocated cid capacity (joints + contact columns).
        Use this to size output arrays for
        :meth:`gather_constraint_wrenches` /
        :meth:`gather_constraint_errors`."""
        return self._constraint_capacity

    def gather_constraint_wrenches(self, out: wp.array) -> None:
        """Per-cid world-frame wrench on ``body2`` averaged over the
        last substep. ``out`` is
        ``wp.array[wp.spatial_vector]`` of length :attr:`num_constraints`."""
        if self._constraint_capacity == 0:
            return
        out.zero_()
        if self.substep_dt <= 0.0:
            return
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        idt = wp.float32(1.0 / self.substep_dt)
        wp.launch(
            _constraint_gather_wrenches_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                wp.int32(self._constraint_capacity),
                wp.int32(self.num_joints),
                idt,
                self._contact_container,
                contact_views,
            ],
            outputs=[out],
            device=self.device,
        )

    def gather_constraint_errors(self, out: wp.array) -> None:
        """Per-cid position-level constraint residual. ``out`` is
        ``wp.array[wp.spatial_vector]`` of length :attr:`num_constraints`."""
        if self._constraint_capacity == 0:
            return
        out.zero_()
        wp.launch(
            _constraint_gather_errors_kernel,
            dim=self._constraint_capacity,
            inputs=[
                self.constraints,
                self._contact_cols,
                self.bodies,
                wp.int32(self._constraint_capacity),
                wp.int32(self.num_joints),
            ],
            outputs=[out],
            device=self.device,
        )

    def num_colors_used(self) -> int:
        """Number of graph colours the last PGS colouring used.
        Performs a device-to-host copy -- do not call inside a
        :func:`wp.ScopedCapture` region."""
        return int(self._partitioner.num_colors.numpy()[0])

    def gather_contact_wrenches(self, out: wp.array) -> None:
        """Per-individual-contact wrench (force + torque) from the
        last substep. ``out`` is ``wp.array[wp.spatial_vector]`` of
        length :attr:`rigid_contact_max`."""
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
                self._contact_cols,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(self.max_contact_columns),
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
        """Per-contact-column wrench summary."""
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
                self._contact_cols,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(self.max_contact_columns),
                idt,
            ],
            outputs=[wrenches, body1, body2, contact_count],
            device=self.device,
        )

    def gather_contact_errors(self, out: wp.array) -> None:
        """Per-individual-contact position-level residual."""
        if self.max_contact_columns == 0:
            out.zero_()
            return
        out.zero_()
        if self._contact_views is None:
            return
        wp.launch(
            contact_per_contact_error_kernel,
            dim=self.max_contact_columns,
            inputs=[
                self._contact_cols,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(self.max_contact_columns),
            ],
            outputs=[out],
            device=self.device,
        )
