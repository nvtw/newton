# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of PhoenX's ``Scene.Step`` driver loop, contacts-only.

Translation of ``PhoenX/src/PhoenX/World.Step.cs`` to Python, adapted
to Newton's :class:`~newton._src.solvers.jitter` contact ingest
pipeline and the upstream :class:`~newton._src.solvers.Contacts`
buffer produced by Newton's ``CollisionPipeline`` (the equivalent of
PhoenX's ``ContactManager``).

Scope of this port:
    * Contacts-only -- joints are not supported. Every constraint
      column in the shared :class:`ConstraintContainer` carries the
      :data:`CONSTRAINT_TYPE_CONTACT` tag, so the dispatcher's
      joint branch never fires. Use :mod:`solver_jitter` when joints
      are required.
    * Full graph coloring via the shared
      :class:`IncrementalContactPartitioner` (Jones-Plassmann), the
      same coloring :mod:`solver_jitter` uses. One colouring pass per
      :meth:`PhoenXWorld.step` is reused across every substep and
      every PGS iteration.
    * No mass splitting -- dropped from the C# pipeline; a single
      coloured PGS sweep per iteration converges contact-only
      workloads without the per-body copy-state averaging PhoenX
      runs for dense constraint graphs.
    * Contact patches up to 6 contacts per pair (vs. the C# PhoenX
      4-slot ``ContactManifold``). The underlying
      :class:`ContactContainer` already stores 6 slots per cid; no
      translation surgery is needed to enable the extra capacity.
    * Newton's :class:`CollisionPipeline` produces the contacts;
      this solver never runs its own broad or narrow phase
      (``BroadPhase.FindPairs2`` / ``ContactManager.DetectCollisions``
      from ``World.Step.cs`` map to calls the caller makes *before*
      invoking :meth:`PhoenXWorld.step`).
    * No articulations, no islands, no deactivation, no
      ``KernelGraph`` capture path, no per-frame timing -- intentional
      simplifications that keep the port focused on the solver loop.

Step structure mirrors ``World.Step.cs``:

1. Ingest contacts from the Newton ``Contacts`` buffer (replaces
   ``ContactManager.DetectCollisions`` / ``SortKeys``).
2. Rebuild the per-cid element view and run the graph colouring
   (``ConstraintGraph.UpdateInteractingElements`` /
   ``BuildPartitions``).
3. Substep loop. Each substep:
    a. ``StoreBeginOfSubstepVelocity`` on the final substep (skipped
       in this port -- :class:`BodyContainer` has no acceleration
       fields).
    b. ``IntegrateForces``: apply per-body ``force`` / ``torque``
       accumulators to velocity.
    c. ``IntegrateGravity``: ``v += g * dt`` for gravity-affected
       dynamic bodies.
    d. ``SolveSubstep``: PGS prepare + iterate (``use_bias=True``)
       sweeps, then the ``VelocityIterations`` relax sweeps
       (``use_bias=False``).
    e. ``Integrate``: advance position + orientation.
4. ``UpdateInertia``: apply per-body damping + refresh
   ``inverse_inertia_world`` from the post-step orientation.
5. Clear per-body force/torque accumulators for next step.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.jitter.constraints.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_contact import (
    CONTACT_DWORDS,
    ContactViews,
    contact_pair_wrench_kernel,
    contact_per_contact_error_kernel,
    contact_per_contact_wrench_kernel,
    contact_views_make,
)
from newton._src.solvers.jitter.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.jitter.constraints.contact_container import (
    ContactContainer,
    contact_container_swap_prev_current,
    contact_container_zeros,
)
from newton._src.solvers.jitter.constraints.contact_ingest import (
    IngestScratch,
    gather_contact_warmstart,
    ingest_contacts,
    stamp_forward_contact_map,
)
from newton._src.solvers.jitter.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
)
from newton._src.solvers.jitter.graph_coloring.graph_coloring_incremental import (
    MAX_COLORS,
    IncrementalContactPartitioner,
)
from newton._src.solvers.jitter.materials import MaterialData
from newton._src.solvers.jitter.solver_jitter_kernels import (
    _STRAGGLER_BLOCK_DIM,
    _constraint_iterate_fast_tail_kernel,
    _constraint_position_iterate_fast_tail_kernel,
    _constraint_prepare_fast_tail_kernel,
    _constraint_relax_fast_tail_kernel,
    _constraints_to_elements_kernel,
    _integrate_velocities_kernel,
    _world_csr_count_kernel,
    _world_csr_prefix_offsets_kernel,
    _world_csr_scan_kernel,
    _world_csr_scatter_kernel,
    pack_body_xforms_kernel,
)

__all__ = [
    "DEFAULT_SHAPE_GAP",
    "PhoenXWorld",
    "make_phoenx_shape_cfg",
    "pack_body_xforms_kernel",
]


#: Default contact-detection gap [m] for shapes in PhoenX scenes. The
#: value is deliberately generous -- 5 cm is large relative to
#: everyday objects but gives the broad phase a comfortable lead so
#: contacts are emitted a few frames *before* impact. PhoenX's
#: speculative-approach branch then decelerates closing bodies while
#: they still have a gap, which is much easier to stabilise than
#: correcting penetration after the fact.
#:
#: Set on ``ModelBuilder.default_shape_cfg.gap`` via
#: :func:`make_phoenx_shape_cfg` (or directly) so every dynamic shape
#: in a PhoenX scene picks it up without having to plumb the constant
#: through every ``add_shape_*`` call site.
#:
#: Scale-sensitivity note: this is a world-space length. Scenes that
#: want a different characteristic length (e.g. millimetre-scale MEMS
#: or kilometre-scale vehicles) should override the default rather
#: than rely on it.
DEFAULT_SHAPE_GAP: float = 0.05


def make_phoenx_shape_cfg(**overrides):
    """Return a ``ModelBuilder.ShapeConfig`` with PhoenX defaults.

    Presets ``gap = DEFAULT_SHAPE_GAP`` so contacts are detected a
    few cm ahead of impact -- PhoenX's speculative branch handles
    those cleanly and it avoids the penetration-resolution
    transients that hurt stability. All other fields fall through
    to Newton's defaults; any keyword passed to ``overrides``
    wins over the PhoenX default.
    """
    import newton as _newton  # local import: avoids top-level cycle

    cfg = _newton.ModelBuilder.ShapeConfig(gap=DEFAULT_SHAPE_GAP)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@wp.kernel(enable_backward=False)
def _sync_num_active_constraints_kernel(
    num_contact_columns: wp.array[wp.int32],
    joint_constraint_count: wp.int32,
    # out
    num_active_constraints: wp.array[wp.int32],
):
    """Fuse the ingest pipeline's device-held ``num_contact_columns``
    into ``num_active_constraints = num_joints + num_contact_columns``
    on-device.

    Single-thread kernel so the active-count update stays inside a
    graph capture. Copies the jitter port's helper verbatim.
    """
    tid = wp.tid()
    if tid != 0:
        return
    num_active_constraints[0] = joint_constraint_count + num_contact_columns[0]


# ---------------------------------------------------------------------------
# PhoenX-specific per-substep / per-step body kernels
# ---------------------------------------------------------------------------
#
# These mirror the C# kernels in ``SolverManager`` /
# ``CudaKernels/Solver/SolverKernels.cs``. The Jitter port's
# ``_update_bodies_kernel`` fuses damping + force delta + gravity + the
# inertia refresh and runs it once per *step*; PhoenX's C# pipeline
# splits those concerns across separate per-substep kernels, which is
# what we reproduce here so the public step-loop structure matches
# ``World.Step.cs`` one-for-one.


@wp.kernel(enable_backward=False)
def _phoenx_apply_external_forces_kernel(
    bodies: BodyContainer,
    substep_dt: wp.float32,
):
    """Port of :func:`SolverKernels.ApplyExternalForcesKernel`
    (without the sparse ``ForceAndTorque`` buffer).

    Adds the per-body ``force`` / ``torque`` accumulators to the
    linear / angular velocity. Runs every substep so user-scripted
    time-varying forces take effect as the substep cadence advances;
    for a constant external load the net effect matches Jitter's
    once-per-step cached-delta split.

    Static and zero-inverse-mass bodies are skipped so we never
    pollute their velocity state. The force accumulators are *not*
    zeroed here -- :func:`_phoenx_clear_forces_kernel` does that
    once at the end of :meth:`PhoenXWorld.step` so every substep
    sees the same external load.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    if bodies.inverse_mass[i] == 0.0:
        return
    inv_mass = bodies.inverse_mass[i]
    inv_inertia_world = bodies.inverse_inertia_world[i]
    f = bodies.force[i]
    t = bodies.torque[i]
    bodies.velocity[i] = bodies.velocity[i] + f * (inv_mass * substep_dt)
    bodies.angular_velocity[i] = bodies.angular_velocity[i] + (inv_inertia_world * t) * substep_dt


@wp.kernel(enable_backward=False)
def _phoenx_integrate_gravity_kernel(
    bodies: BodyContainer,
    gravity: wp.array[wp.vec3f],
    substep_dt: wp.float32,
):
    """Port of :func:`SolverKernels.IntegrateGravityKernel`.

    Adds ``gravity[world_id] * substep_dt`` to the linear velocity
    of every dynamic, gravity-affected body every substep.

    The gyroscopic force path is intentionally dropped -- the C#
    implementation gates it behind an ``ApplyGyroscopicForces``
    flag that is off for the default scenes this port targets, and
    adding it would pull in the per-substep inverse computation
    of :attr:`inverse_inertia_world` which is unnecessary overhead
    for contact-only workloads.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    if bodies.inverse_mass[i] == 0.0:
        return
    if bodies.affected_by_gravity[i] == 0:
        return
    w = bodies.world_id[i]
    bodies.velocity[i] = bodies.velocity[i] + gravity[w] * substep_dt


@wp.kernel(enable_backward=False)
def _phoenx_update_inertia_kernel(
    bodies: BodyContainer,
):
    """Port of :func:`SolverKernels.UpdateInertiaKernel`.

    Runs once per step, after the substep loop completes. For every
    dynamic body:

    * Multiplies the linear / angular velocity by the per-body damping
      multiplier (PhoenX's ``linearDampingMultiplier`` /
      ``angularDampingMultiplier``). The multipliers come straight
      from :class:`BodyContainer` and are typically set from a
      time-constant via :math:`m = \\exp(-k \\cdot dt)` at
      construction.
    * Refreshes ``inverse_inertia_world = R * inverse_inertia * R^T``
      from the final orientation so the next step's effective-mass
      computation sees the rotated inertia.

    Static bodies keep their zeroed mass / inertia by construction
    and are skipped.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    bodies.velocity[i] = bodies.velocity[i] * bodies.linear_damping[i]
    bodies.angular_velocity[i] = bodies.angular_velocity[i] * bodies.angular_damping[i]
    r = wp.quat_to_matrix(bodies.orientation[i])
    bodies.inverse_inertia_world[i] = r * bodies.inverse_inertia[i] * wp.transpose(r)


@wp.kernel(enable_backward=False)
def _phoenx_clear_forces_kernel(
    bodies: BodyContainer,
):
    """Zero the per-body ``force`` / ``torque`` accumulators.

    PhoenX's C# pipeline lets the user zero the sparse
    ``ForceAndTorque`` storage explicitly between steps; with
    Newton's dense per-body accumulators we do it here so every
    :meth:`PhoenXWorld.step` starts with an empty external-load
    state. Run once per step, after the substep loop, so the
    in-loop :func:`_phoenx_apply_external_forces_kernel` launches
    all see the same external force across substeps.
    """
    i = wp.tid()
    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Driver: PhoenXWorld
# ---------------------------------------------------------------------------


def _build_gravity_array(
    gravity, num_worlds: int, device
) -> wp.array[wp.vec3f]:
    """Coerce the user's ``gravity`` argument into a length-``num_worlds``
    :class:`wp.array[wp.vec3f]`.

    Accepts either a single 3-tuple (broadcast to every world) or an
    iterable of ``num_worlds`` 3-tuples (one per world), matching the
    helper in :mod:`solver_jitter`.
    """
    g_list = list(gravity) if hasattr(gravity, "__iter__") else None
    if g_list is not None and len(g_list) == 3 and not hasattr(g_list[0], "__iter__"):
        vec = (float(g_list[0]), float(g_list[1]), float(g_list[2]))
        data = [vec] * num_worlds
    else:
        if g_list is None:
            raise ValueError("gravity must be a tuple or an iterable of tuples")
        if len(g_list) != num_worlds:
            raise ValueError(
                f"gravity length {len(g_list)} != num_worlds {num_worlds}"
            )
        data = []
        for row in g_list:
            r = tuple(row)
            if len(r) != 3:
                raise ValueError(
                    f"per-world gravity entry must be 3 components; got {len(r)}"
                )
            data.append((float(r[0]), float(r[1]), float(r[2])))
    arr_np = np.asarray(data, dtype=np.float32)
    return wp.array(arr_np, dtype=wp.vec3f, device=device)


class PhoenXWorld:
    """Warp port of PhoenX's ``Scene.Step`` driver, contacts + optional
    actuated double-ball-socket joints.

    The solver owns a :class:`BodyContainer`, a shared
    :class:`ConstraintContainer`, and the per-step contact ingest /
    warm-start infrastructure. Calling :meth:`step` advances every
    rigid body by ``dt`` seconds using the same phased loop as
    ``PhoenX/src/PhoenX/World.Step.cs``.

    Construction mirrors :class:`~newton._src.solvers.jitter.World`
    for the contact-plus-unified-joint subset. The only supported
    joint type is
    :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET`, which
    materialises ball-socket / revolute / prismatic joints with
    optional PD drives and limits (see
    :meth:`initialize_actuated_double_ball_socket_joints` and
    :class:`~newton._src.solvers.jitter.world_builder.JointMode`).
    Direct construction is supported; a dedicated builder is not
    provided because the solver only needs the initial body + joint
    state.
    """

    def __init__(
        self,
        bodies: BodyContainer,
        constraints: ConstraintContainer,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 0,
        position_iterations: int = 0,
        gravity: tuple[float, float, float]
        | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        max_contact_columns: int = 0,
        rigid_contact_max: int = 0,
        num_shapes: int = 0,
        num_joints: int = 0,
        collision_filter_pairs: Iterable[tuple[int, int]] | None = None,
        default_friction: float = 0.5,
        num_worlds: int = 1,
        device: wp.context.Devicelike = None,
    ):
        """Take ownership of a pre-built body and constraint container.

        Args:
            bodies: :class:`BodyContainer` with the rigid bodies to
                simulate. The caller is responsible for initial pose,
                mass, and inertia; the solver only reads / writes
                velocity, position, orientation, forces and the
                rotated inertia.
            constraints: Shared :class:`ConstraintContainer`. Joint
                columns (when ``num_joints > 0``) live at cids
                ``[0, num_joints)``; contact columns occupy cids
                ``[num_joints, num_joints + max_contact_columns)``.
                The container's per-constraint dword width must be at
                least ``max(CONTACT_DWORDS, ADBS_DWORDS)`` if joints
                are used; :meth:`make_constraint_container` handles
                the sizing for callers that don't want to compute
                that themselves.
            substeps: Number of substeps per :meth:`step` call.
            solver_iterations: PGS iterations per substep, matching
                PhoenX's ``SolverManager.Iterations``.
            velocity_iterations: Box2D-v3-style TGS-soft relaxation
                sweeps (``use_bias=False``) per substep, matching
                PhoenX's ``SolverManager.VelocityIterations``.
                Defaults to ``0`` (PhoenX's default); raise to 1 to
                shed positional-bias drift velocity the way
                :mod:`solver_jitter` does by default.
            position_iterations: Extra XPBD-style position correction
                sweeps for contact tangent drift. Not present in the
                C# original -- carried over from
                :mod:`solver_jitter`'s fast-path because the 6-slot
                contact constraint reuses the same
                :func:`contact_position_iterate` kernel. Set to ``0``
                for a straight PhoenX translation.
            gravity: Constant world-space gravity (m/s^2). A 3-tuple
                broadcasts to every world; pass an iterable of
                3-tuples to give each world its own vector.
            max_contact_columns: Upper bound on contact cid columns
                per step. Sizes the persistent
                :class:`ContactContainer` and ingest scratch. ``0``
                disables contact code paths (degenerate case: no
                work to do).
            rigid_contact_max: Upper bound on the rigid-contact index
                range in the Newton ``Contacts`` buffer. Defaults to
                ``max_contact_columns * 6`` if ``0``.
            num_shapes: Total shape count in the upstream model; used
                to pack ``(shape_a, shape_b)`` into a 32-bit key for
                contact matching. Must satisfy
                ``num_shapes * num_shapes < 2**31``.
            num_joints: Number of actuated-double-ball-socket joint
                columns reserved at the head of ``constraints`` (cids
                ``[0, num_joints)``). The caller is responsible for
                populating these columns via
                :meth:`initialize_actuated_double_ball_socket_joints`
                before the first :meth:`step`. Leaves ``0`` for a
                contacts-only scene.
            collision_filter_pairs: Optional iterable of
                ``(body_a, body_b)`` pairs whose contacts must be
                dropped on ingest. Pairs are canonicalised to
                ``(min, max)`` and deduped; self-pairs are rejected.
            default_friction: Friction coefficient used when no
                per-shape material is registered (see
                :meth:`set_materials`). Carries the same semantics as
                :mod:`solver_jitter`.
            num_worlds: Number of independent sub-worlds sharing this
                solver (used by multi-env rollouts). Each world's
                bodies are gated by ``BodyContainer.world_id``; the
                graph colouring is global and re-bucketed per world
                so single-world scenes pay no extra cost.
            device: Warp device. Defaults to
                ``bodies.position.device``.
        """
        if device is None:
            self.device = bodies.position.device
        else:
            self.device = wp.get_device(device)

        self.bodies: BodyContainer = bodies
        self.constraints: ConstraintContainer = constraints

        self.num_bodies: int = int(bodies.position.shape[0])
        self.max_contact_columns: int = int(max_contact_columns)
        self.rigid_contact_max: int = int(rigid_contact_max)
        if self.max_contact_columns > 0 and self.rigid_contact_max == 0:
            # 6 = ContactContainer.MAX_SLOTS -- we support up to 6
            # contacts per pair (the PhoenX C# original caps at 4).
            self.rigid_contact_max = self.max_contact_columns * 6
        self.num_shapes: int = int(num_shapes)
        self.num_joints: int = int(num_joints)
        if self.num_joints < 0:
            raise ValueError(f"num_joints must be >= 0 (got {self.num_joints})")

        self.substeps = int(substeps)
        if self.substeps <= 0:
            raise ValueError(f"substeps must be >= 1 (got {self.substeps})")
        self.solver_iterations = int(solver_iterations)
        if self.solver_iterations < 1:
            raise ValueError(
                f"solver_iterations must be >= 1 (got {self.solver_iterations})"
            )
        self.velocity_iterations = int(velocity_iterations)
        if self.velocity_iterations < 0:
            raise ValueError(
                f"velocity_iterations must be >= 0 (got {self.velocity_iterations})"
            )
        self.position_iterations = int(position_iterations)

        self.num_worlds: int = int(num_worlds)
        if self.num_worlds <= 0:
            raise ValueError(f"num_worlds must be >= 1 (got {self.num_worlds})")
        self.gravity: wp.array[wp.vec3f] = _build_gravity_array(
            gravity, self.num_worlds, self.device
        )
        self.default_friction = float(default_friction)

        # ----- Step time bookkeeping -----
        self.step_dt: float = 0.0
        self.inv_step_dt: float = 0.0
        self.substep_dt: float = 0.0

        # Joint columns occupy ``[0, num_joints)``; contact columns
        # follow at ``[num_joints, num_joints + max_contact_columns)``.
        # ``_constraint_capacity`` is the launch dimension for every
        # partitioner / dispatcher kernel and bounds
        # ``num_active_constraints`` from above.
        self._constraint_capacity: int = max(
            1, self.num_joints + self.max_contact_columns
        )

        # ----- Partitioner + per-world CSR buffers -----
        self._elements: wp.array[ElementInteractionData] = wp.zeros(
            self._constraint_capacity, dtype=ElementInteractionData, device=self.device
        )
        # Joints are the only cids active before the first ingest;
        # the contact range stays empty until the first step's
        # ``_ingest_and_warmstart_contacts`` bumps the count.
        self._num_active_constraints: wp.array[int] = wp.array(
            [self.num_joints], dtype=wp.int32, device=self.device
        )
        self._partitioner = IncrementalContactPartitioner(
            max_num_interactions=self._constraint_capacity,
            max_num_nodes=max(1, self.num_bodies),
            device=self.device,
            use_tile_scan=True,
        )

        cap = self._constraint_capacity
        nw = self.num_worlds
        self._world_element_ids_by_color: wp.array[wp.int32] = wp.zeros(
            cap, dtype=wp.int32, device=self.device
        )
        self._world_color_starts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_csr_offsets: wp.array[wp.int32] = wp.zeros(
            nw + 1, dtype=wp.int32, device=self.device
        )
        self._world_num_colors: wp.array[wp.int32] = wp.zeros(
            nw, dtype=wp.int32, device=self.device
        )
        self._world_color_counts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_color_cursor: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )

        # ----- Contact infrastructure -----
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

        # ----- Pairwise contact filter (packed int64 keys) -----
        self._collision_filter_keys: wp.array[wp.int64]
        self._set_collision_filter_pairs_impl(collision_filter_pairs or ())

        # ----- Optional material table -----
        self._shape_material: wp.array[wp.int32] | None = None
        self._materials: wp.array[MaterialData] | None = None

    # ------------------------------------------------------------------
    # Material system / collision filters / placeholder contact views
    # ------------------------------------------------------------------

    def set_materials(
        self,
        materials: wp.array | None,
        shape_material: wp.array | None,
    ) -> None:
        """Install per-shape materials for contact friction.

        Equivalent to the material hook in :mod:`solver_jitter`; see
        that module for the array shape and combine-mode conventions.
        """
        self._materials = materials
        self._shape_material = shape_material

    # ------------------------------------------------------------------
    # Joint initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def required_constraint_dwords(num_joints: int) -> int:
        """Return the minimum per-constraint dword width the
        :class:`ConstraintContainer` handed to :class:`PhoenXWorld`
        must carry.

        ``CONTACT_DWORDS`` if the scene is contacts-only;
        ``max(CONTACT_DWORDS, ADBS_DWORDS)`` when any joint columns
        are reserved. Callers that roll their own container can use
        this instead of importing the two dword constants directly.
        """
        if int(num_joints) > 0:
            return max(int(CONTACT_DWORDS), int(ADBS_DWORDS))
        return int(CONTACT_DWORDS)

    @staticmethod
    def make_constraint_container(
        num_joints: int,
        max_contact_columns: int,
        device: wp.context.Devicelike = None,
    ) -> ConstraintContainer:
        """Factory for a correctly-sized :class:`ConstraintContainer`.

        Produces a zero-initialised container with capacity
        ``num_joints + max_contact_columns`` and the per-constraint
        dword width the solver requires (see
        :meth:`required_constraint_dwords`). Use this from the
        caller's scene-construction code; then pass the resulting
        container to :class:`PhoenXWorld.__init__` and call
        :meth:`initialize_actuated_double_ball_socket_joints` before
        the first step.
        """
        cap = max(1, int(num_joints) + int(max_contact_columns))
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
        """Pack ``num_joints`` actuated double-ball-socket joint columns.

        One-shot launch of
        :func:`actuated_double_ball_socket_initialize_kernel` over
        cids ``[0, num_joints)``. All input arrays must be length
        ``num_joints``, live on ``device``, and carry the dtypes the
        kernel expects (see the kernel signature in
        :mod:`constraint_actuated_double_ball_socket`). The solver's
        fast-path dispatcher already routes
        :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` through
        :func:`actuated_double_ball_socket_prepare_for_iteration` /
        :func:`actuated_double_ball_socket_iterate`, so no further
        wiring is required after this call returns.

        Call exactly once, after :meth:`__init__` and before the
        first :meth:`step`. ``num_joints == 0`` is a no-op.

        Args:
            body1, body2: Body indices for each joint's two bodies.
                ``wp.array[wp.int32]`` of length ``num_joints``.
            anchor1, anchor2: World-space joint anchors at joint
                finalisation time. ``wp.array[wp.vec3f]``. The line
                from ``anchor1`` to ``anchor2`` is the joint axis
                for revolute / prismatic modes.
            hertz, damping_ratio: Positional Schur block soft-
                constraint frequency [Hz] and dimensionless damping
                ratio.
            joint_mode: One of
                :data:`~newton._src.solvers.jitter.world_builder.JointMode`
                enum values per joint. ``wp.array[wp.int32]``.
            drive_mode: One of
                :data:`~newton._src.solvers.jitter.world_builder.DriveMode`
                enum values per joint. ``wp.array[wp.int32]``.
            target, target_velocity: Per-joint position / velocity
                drive setpoints ([rad, rad/s] for revolute,
                [m, m/s] for prismatic).
            max_force_drive, stiffness_drive, damping_drive: Drive
                impulse cap and PD gains. ``stiffness_drive ==
                damping_drive == 0`` turns the drive row off.
            min_value, max_value: Limit window (``min > max``
                disables the limit row).
            hertz_limit, damping_ratio_limit: Limit soft-constraint
                knobs (used when both PD gains are zero).
            stiffness_limit, damping_limit: Limit PD gains (absolute
                SI). Either strictly positive selects the PD
                formulation over the ``(hertz_limit,
                damping_ratio_limit)`` one.
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

    def set_collision_filter_pairs(
        self, pairs: Iterable[tuple[int, int]]
    ) -> None:
        """Replace the registered body-pair contact filter.

        See :meth:`solver_jitter.World.set_collision_filter_pairs` for
        the semantic contract (self-pair rejection, canonicalisation,
        binary-search-based kernel lookup).
        """
        self._set_collision_filter_pairs_impl(pairs)

    def _set_collision_filter_pairs_impl(
        self, pairs: Iterable[tuple[int, int]]
    ) -> None:
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

        packed.sort()
        if not packed:
            arr = np.asarray([np.iinfo(np.int64).max], dtype=np.int64)
        else:
            arr = np.asarray(packed, dtype=np.int64)

        self._collision_filter_keys = wp.array(arr, dtype=wp.int64, device=self.device)
        self._collision_filter_count = int(len(packed))

    def _make_placeholder_contact_views(self) -> ContactViews:
        """Size-1 dummy :class:`ContactViews` for no-contact steps.

        The dispatcher kernels require a valid :class:`ContactViews`
        argument even when no contacts are active. The placeholder is
        never actually read because the contact branch only fires for
        cids whose type is :data:`CONSTRAINT_TYPE_CONTACT`, and no
        such cid exists when the active-constraint counter is zero.
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
    # Public driver
    # ------------------------------------------------------------------

    def step(self, dt: float, contacts=None, shape_body=None) -> None:
        """Advance the world by ``dt`` seconds.

        Direct port of ``Scene.Step`` (``PhoenX/src/PhoenX/World.Step.cs``).
        The phases below map one-for-one onto the C# original; only the
        collision-detection branch is inverted (we ingest contacts
        Newton's :class:`CollisionPipeline` produced upstream, rather
        than running our own narrow phase).

        Per-step flow:

        1. Contact ingest + warm-start (replaces
           ``ContactManager.DetectCollisions`` / ``SortKeys`` /
           ``ExportImpulseModels``). Updates the device-held
           ``num_active_constraints`` counter.
        2. ``UpdateInteractingElements`` + ``BuildPartitions``: map
           every active constraint into the partitioner and run
           Jones-Plassmann graph colouring once. The CSR is reused
           across every substep + every PGS iteration because the
           graph topology is frozen for the whole step (contacts are
           ingested before the loop, and this port has no joints).
        3. Substep loop (``self.substeps`` iterations). Each substep
           runs ``IntegrateForces`` -> ``IntegrateGravity`` ->
           ``SolveSubstep`` -> ``Integrate`` in that order, matching
           the C# sequence.
        4. ``UpdateInertia``: damping + rotated-inertia refresh, once
           per step after all substeps have advanced the pose.
        5. Force accumulators are cleared so the next ``step`` starts
           with a zeroed external load.

        Args:
            dt: Time step in seconds. Zero or negative values no-op.
            contacts: Newton :class:`Contacts` buffer produced by the
                upstream :class:`CollisionPipeline`. Must be built
                with a non-disabled ``contact_matching`` mode; use
                ``"sticky"`` for stable stacking (the warm-start
                gather relies on persistent match indices). Pass
                ``None`` to run a "free-fall" step with no contact
                constraints.
            shape_body: ``model.shape_body`` mapping (shape id ->
                body id). Required whenever ``contacts`` is provided.
        """
        if dt < 0.0:
            raise ValueError("Time step cannot be negative.")
        if dt < 1e-7:
            # C# :``if(dt == 0.0f) return;`` -- match the "nothing
            # to do" early-out and avoid division-by-zero downstream.
            return

        self.step_dt = dt
        self.inv_step_dt = 1.0 / dt
        self.substep_dt = dt / self.substeps

        # ---- Phase 1: ingest Newton contacts ----
        # Replaces ``BroadPhase.FindPairs2`` + ``ContactManager
        # .DetectCollisions`` + ``SortKeys`` from the C# pipeline.
        self._ingest_and_warmstart_contacts(contacts, shape_body)

        # ---- Phase 2: rebuild elements + graph colouring ----
        # ``UpdateInteractingElements`` (type-agnostic projection of
        # every active constraint's body pair) and ``BuildPartitions``
        # (Jones-Plassmann colouring) from the C#
        # ``ConstraintGraph`` pipeline. Coloring is reused across every
        # substep + PGS iteration.
        self._rebuild_elements()
        if self._constraint_capacity > 0:
            self._partitioner.reset(self._elements, self._num_active_constraints)
            self._partitioner.build_csr()
            self._build_world_csr()

        # ---- Phase 3: substep loop ----
        # ``for (int i = 0; i < numSubsteps; i++)`` in World.Step.cs
        # with Box2D-v3 / jitter's TGS-soft ordering (solve ->
        # integrate_positions -> relax). The ordering diverges from
        # the strict C# read "prepare + iterate + velocity_iterate +
        # integrate" because without mass splitting the correct
        # place for the ``use_bias=False`` relax pass is *after*
        # position integration, not before it. Running relax before
        # integrate throws away the positional bias's contribution
        # to position correction -- which is critical for tall
        # stacks and long joint chains where the top constraint
        # carries order-of-the-full-load forces and needs each
        # substep's position update to make dent in the penetration
        # / drift.
        for _ in range(self.substeps):
            self._integrate_forces()
            self._integrate_gravity()
            # Main PGS solve with positional bias ON; produces
            # velocities that include the bias correction.
            self._solve_main()
            # Advance positions using the biased velocity so the
            # bias actually translates into reduced penetration /
            # joint drift.
            self._integrate_positions()
            # Optional XPBD contact tangent-drift passes (contacts
            # only).
            self._position_iterate()
            # Relax the positional-bias-induced drift velocity so
            # the next substep's solve starts from a clean ``Jv =
            # 0`` state -- Box2D v3 TGS-soft relax pass.
            self._relax_velocities()

        # ---- Phase 4: per-step bookkeeping ----
        # ``Solver.UpdateInertia`` from the tail of ``Scene.Step``.
        # Rotates the body-frame inertia into the post-step
        # orientation and applies linear/angular damping.
        self._update_inertia()

        # Clear the force accumulators so the next call to
        # ``step()`` starts with a zeroed external load (PhoenX does
        # this implicitly via the sparse force buffer's
        # ``Clear()``).
        self._clear_forces()

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _ingest_and_warmstart_contacts(self, contacts, shape_body) -> None:
        """Translate Newton's ``Contacts`` buffer into contact columns.

        Replaces the broad+narrow phase scope of ``World.Step.cs``.
        Three sub-phases:

        1. Fast-reject when contacts are disabled or absent: zero the
           active-constraint counter and early-out.
        2. Swap prev/current per-cid lambdas + forward maps (pointer
           swap, no device copy) so the warm-start gather reads the
           previous step's finished impulses.
        3. Run :func:`ingest_contacts` ->
           :func:`gather_contact_warmstart` ->
           :func:`stamp_forward_contact_map`, then sync the
           device-held ``num_active_constraints`` counter. Because
           ``max_contact_columns`` sits at the head of the cid range
           (we reserve no joint range), ``num_active_constraints``
           equals the number of contact columns emitted.
        """
        if contacts is None or self.max_contact_columns == 0 or self._ingest_scratch is None:
            # No contacts this step: fall back to the joint-only active
            # count. Joints live at cids ``[0, num_joints)`` and
            # :meth:`initialize_actuated_double_ball_socket_joints`
            # has already populated them.
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
            raise ValueError(
                "step(dt, contacts=...) requires shape_body=model.shape_body to "
                "resolve contact shape ids to rigid-body ids."
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
        )

        # Swap lambda buffers + forward maps (pointer swap, O(1)).
        contact_container_swap_prev_current(self._contact_container)
        self._slot_of_contact_cur, self._slot_of_contact_prev = (
            self._slot_of_contact_prev,
            self._slot_of_contact_cur,
        )
        self._cid_of_contact_cur, self._cid_of_contact_prev = (
            self._cid_of_contact_prev,
            self._cid_of_contact_cur,
        )

        # Materialise this step's contact columns. ``cid_base = 0``
        # Contact columns are written starting at ``cid_base =
        # num_joints`` so joint columns at ``[0, num_joints)`` stay
        # untouched. The sync kernel below then picks up the ingest
        # pipeline's ``num_contact_columns`` and adds the joint
        # count on-device.
        ingest_contacts(
            contacts=contacts,
            shape_body=shape_body,
            num_shapes=self.num_shapes,
            constraints=self.constraints,
            scratch=self._ingest_scratch,
            cid_base=self.num_joints,
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
            prev_slot_of_contact=self._slot_of_contact_prev,
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
            slot_of_contact=self._slot_of_contact_cur,
            cid_of_contact=self._cid_of_contact_cur,
            device=self.device,
        )

        # Fuse joint count + ingested contact column count into
        # ``_num_active_constraints`` on-device so the partitioner /
        # dispatcher see the right extent without a host readback.
        self._sync_num_active_constraints()

    def _sync_num_active_constraints(self) -> None:
        """Fuse joint count + contact column count on-device.

        Launches :func:`_sync_num_active_constraints_kernel` with
        ``joint_constraint_count = self.num_joints``; the kernel
        writes ``num_active_constraints[0] = num_joints +
        num_contact_columns[0]``. Kept out of :meth:`step` so the
        ingest path can be unit-tested in isolation, and graph-
        capture safe (one-thread launch, all device arrays).
        """
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
        :class:`ElementInteractionData` view.

        Type-agnostic launch at ``dim = constraint_capacity``; threads
        beyond the device-held ``num_active_constraints`` early-out.
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

    def _build_world_csr(self) -> None:
        """Bucket the partitioner's global CSR into per-world slices.

        Identical to :meth:`solver_jitter.World._build_world_csr` --
        the colouring is global (single JP pass across every active
        constraint) and the per-world dispatcher expects its input
        pre-partitioned by world. Cheap because every launch is
        fully device-side and proportional to the capacity.
        """
        self._world_color_counts.zero_()
        cap = self._constraint_capacity
        wp.launch(
            _world_csr_count_kernel,
            dim=cap,
            inputs=[
                self.bodies,
                self._elements,
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.num_colors,
            ],
            outputs=[self._world_color_counts],
            device=self.device,
        )
        wp.launch(
            _world_csr_scan_kernel,
            dim=self.num_worlds,
            inputs=[
                self._world_color_counts,
                self._partitioner.num_colors,
            ],
            outputs=[self._world_color_starts, self._world_num_colors],
            device=self.device,
        )
        wp.launch(
            _world_csr_prefix_offsets_kernel,
            dim=1,
            inputs=[
                self._world_color_starts,
                self._partitioner.num_colors,
                wp.int32(self.num_worlds),
            ],
            outputs=[self._world_csr_offsets],
            device=self.device,
        )
        self._world_color_cursor.zero_()
        wp.launch(
            _world_csr_scatter_kernel,
            dim=cap,
            inputs=[
                self.bodies,
                self._elements,
                self._partitioner.element_ids_by_color,
                self._partitioner.color_starts,
                self._partitioner.num_colors,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_color_cursor,
            ],
            outputs=[self._world_element_ids_by_color],
            device=self.device,
        )

    def _integrate_forces(self) -> None:
        """Port of :func:`SolverManager.IntegrateForces`.

        Applies the per-body ``force`` / ``torque`` accumulators to
        velocity. Runs every substep so the velocity increment is
        consistent with PhoenX's split (Newton's Jitter port caches
        the delta once per step; we pay the per-substep cost here to
        match the C# sequence).
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_apply_external_forces_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _integrate_gravity(self) -> None:
        """Port of :func:`SolverManager.IntegrateGravity`.

        ``v += gravity * substep_dt`` for every dynamic, gravity-affected
        body. Runs every substep; each world reads its own gravity via
        ``bodies.world_id``.
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_integrate_gravity_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self.gravity, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _solve_main(self) -> None:
        """Per-substep main PGS solve: prepare + iterate(bias=True).

        Equivalent to PhoenX C#'s ``SolveSubstep`` body *minus* the
        ``velocity_iterations`` (relax) loop. With mass splitting
        dropped (per the port scope), the relax pass has to run
        *after* :meth:`_integrate_positions` to avoid throwing away
        the positional bias's contribution to penetration recovery;
        :meth:`_relax_velocities` handles that below.

        Prepare runs once per substep (effective masses + warm-start
        scatter); iterate runs ``solver_iterations`` times on the
        same colouring. Both use the shared fast-path tail kernels
        so joints and contacts sweep in the same launches.
        """
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = (
            self._contact_views
            if self._contact_views is not None
            else self._contact_views_placeholder
        )
        self._launch_fast_prepare(idt, contact_views)
        self._launch_fast_iter(
            _constraint_iterate_fast_tail_kernel,
            self.solver_iterations,
            idt,
            contact_views,
        )

    def _position_iterate(self) -> None:
        """XPBD-style position iteration for contact tangent drift.

        Contacts-only (joints have no XPBD position path). No-op
        when ``position_iterations == 0``. Runs on the fast-path
        dispatcher, same CSR as the main solve.
        """
        if self._constraint_capacity == 0 or self.position_iterations <= 0:
            return
        wp.launch(
            _constraint_position_iterate_fast_tail_kernel,
            dim=self.num_worlds * _STRAGGLER_BLOCK_DIM,
            block_dim=_STRAGGLER_BLOCK_DIM,
            inputs=[
                self.constraints,
                self.bodies,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                wp.int32(self.position_iterations),
            ],
            device=self.device,
        )

    def _relax_velocities(self) -> None:
        """Box2D v3 TGS-soft relax pass: iterate with bias=False.

        Runs *after* position integration so it removes the drift
        velocity the positional bias injected during the main solve
        without having first thrown that bias away. This is the
        same order :mod:`solver_jitter`'s step uses. No-op when
        ``velocity_iterations == 0``.
        """
        if self._constraint_capacity == 0 or self.velocity_iterations <= 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = (
            self._contact_views
            if self._contact_views is not None
            else self._contact_views_placeholder
        )
        self._launch_fast_iter(
            _constraint_relax_fast_tail_kernel,
            self.velocity_iterations,
            idt,
            contact_views,
        )

    def _launch_fast_prepare(
        self,
        idt: wp.float32,
        contact_views: ContactViews,
    ) -> None:
        """Launch the fast-path prepare kernel, one block per world.

        Prepare runs once per substep (effective masses + warm-start
        scatter). The single-block-per-world dispatcher walks the
        world's colours serially with ``__syncthreads`` between them
        and a block-stride lane loop within each colour, so a single
        launch covers every world's full colouring.
        """
        wp.launch(
            _constraint_prepare_fast_tail_kernel,
            dim=self.num_worlds * _STRAGGLER_BLOCK_DIM,
            block_dim=_STRAGGLER_BLOCK_DIM,
            inputs=[
                self.constraints,
                self.bodies,
                idt,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
            ],
            device=self.device,
        )

    def _launch_fast_iter(
        self,
        kernel,
        num_iterations: int,
        idt: wp.float32,
        contact_views: ContactViews,
    ) -> None:
        """Launch an iterate / relax kernel that loops ``num_iterations``
        sweeps internally, one block per world.

        One launch replaces ``num_iterations`` separate launches --
        the kernel's outer loop is driven by the scalar
        ``num_iterations`` argument, the per-colour serialisation is
        in the middle loop, and block-stride lanes cover the inner
        loop. Matches :meth:`solver_jitter.World._launch_single_block_iter`
        for the contact-only dispatcher variant.
        """
        wp.launch(
            kernel,
            dim=self.num_worlds * _STRAGGLER_BLOCK_DIM,
            block_dim=_STRAGGLER_BLOCK_DIM,
            inputs=[
                self.constraints,
                self.bodies,
                idt,
                self._world_element_ids_by_color,
                self._world_color_starts,
                self._world_csr_offsets,
                self._world_num_colors,
                self._contact_container,
                contact_views,
                wp.int32(num_iterations),
            ],
            device=self.device,
        )

    def _integrate_positions(self) -> None:
        """Port of :func:`SolverManager.Integrate`.

        ``x += v * dt`` and ``q = rotation_quaternion(omega * dt) * q``
        for every non-static body (kinematic bodies advance too, to
        match PhoenX's behaviour for user-scripted velocities). The
        axis-angle form keeps the quaternion unit-norm across many
        substeps.
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _integrate_velocities_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _update_inertia(self) -> None:
        """Port of :func:`SolverManager.UpdateInertia`.

        Applies linear/angular damping and rebuilds
        ``inverse_inertia_world`` from the final orientation. Runs
        once per step, after the substep loop completes.
        """
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_update_inertia_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies],
            device=self.device,
        )

    def _clear_forces(self) -> None:
        """Zero the per-body force/torque accumulators.

        Runs at the tail of :meth:`step` so every call to ``step``
        starts with an empty external-load state.
        """
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

    def num_colors_used(self) -> int:
        """Number of graph colours used by the last PGS colouring.

        Diagnostic accessor: performs a device-to-host copy via
        ``.numpy()`` and is therefore not safe inside a
        :func:`wp.ScopedCapture` region. Returns ``0`` before the
        first :meth:`step` call.
        """
        return int(self._partitioner.num_colors.numpy()[0])

    def gather_contact_wrenches(self, out: wp.array) -> None:
        """Per-individual-contact wrench (force + torque) from the
        last substep. See :meth:`solver_jitter.World.gather_contact_wrenches`
        for the full indexing contract -- behaviour is identical because
        this solver reuses the same :class:`ContactContainer` storage
        layout.
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
                wp.int32(0),
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
        """Per-contact-column wrench summary; see
        :meth:`solver_jitter.World.gather_contact_pair_wrenches`."""
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
                wp.int32(0),
                wp.int32(self._constraint_capacity),
                idt,
            ],
            outputs=[wrenches, body1, body2, contact_count],
            device=self.device,
        )

    def gather_contact_errors(self, out: wp.array) -> None:
        """Per-individual-contact position-level residual. See
        :meth:`solver_jitter.World.gather_contact_errors` for the
        layout; behaviour is identical."""
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
                self.constraints,
                self.bodies,
                self._contact_container,
                self._contact_views,
                wp.int32(0),
                wp.int32(self._constraint_capacity),
            ],
            outputs=[out],
            device=self.device,
        )
