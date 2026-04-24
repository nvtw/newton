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
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    CONTACT_DWORDS,
    ContactViews,
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
from newton._src.solvers.phoenx.materials import MaterialData
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _STRAGGLER_BLOCK_DIM,
    _constraint_gather_errors_kernel,
    _constraint_gather_wrenches_kernel,
    _constraint_iterate_fast_tail_kernel,
    _constraint_position_iterate_fast_tail_kernel,
    _constraint_prepare_fast_tail_kernel,
    _constraint_relax_fast_tail_kernel,
    _constraints_to_elements_kernel,
    _integrate_velocities_kernel,
    _kinematic_interpolate_substep_kernel,
    _kinematic_prepare_step_kernel,
    _set_kinematic_pose_batch_kernel,
    _world_csr_count_kernel,
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


#: Default contact-detection gap [m] for shapes in PhoenX scenes.
#: Generous (5 cm) so contacts emit a few frames before impact;
#: PhoenX's speculative branch decelerates closing bodies while they
#: still have a gap, which is easier to stabilise than penetration
#: recovery. Scene-scale-sensitive: override for MEMS / vehicle scales.
DEFAULT_SHAPE_GAP: float = 0.05


def make_phoenx_shape_cfg(**overrides):
    """Return a ``ModelBuilder.ShapeConfig`` with ``gap = DEFAULT_SHAPE_GAP``.
    ``overrides`` win over the PhoenX default."""
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
    """``num_active_constraints = num_joints + num_contact_columns``,
    on-device. Single-thread; safe inside graph capture."""
    tid = wp.tid()
    if tid != 0:
        return
    num_active_constraints[0] = joint_constraint_count + num_contact_columns[0]


@wp.kernel(enable_backward=False)
def _phoenx_apply_external_forces_kernel(
    bodies: BodyContainer,
    substep_dt: wp.float32,
):
    """``v += f / m * dt``, ``w += I^-1 * tau * dt`` for dynamic
    bodies. Force accumulators are NOT cleared here --
    :func:`_phoenx_clear_forces_kernel` runs once at the end of
    :meth:`PhoenXWorld.step`."""
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
    """``v += gravity[world_id] * dt`` for dynamic gravity-affected bodies."""
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
    """Apply linear/angular damping and refresh
    ``inverse_inertia_world = R * I^-1 * R^T`` from the current
    orientation. Once per step, after the substep loop."""
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
    """Zero per-body force/torque accumulators. Runs once at the end
    of :meth:`PhoenXWorld.step`."""
    i = wp.tid()
    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)


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
        position_iterations: int = 0,
        gravity: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        max_contact_columns: int = 0,
        rigid_contact_max: int = 0,
        num_shapes: int = 0,
        num_joints: int = 0,
        collision_filter_pairs: Iterable[tuple[int, int]] | None = None,
        default_friction: float = 0.5,
        num_worlds: int = 1,
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
            position_iterations: Extra XPBD contact tangent-drift
                sweeps per substep. Set to ``0`` for straight PhoenX.
            gravity: Constant world-space gravity [m/s^2]. A 3-tuple
                broadcasts; an iterable of 3-tuples gives each world
                its own vector.
            max_contact_columns: Contact cid capacity per step. ``0``
                disables contact paths.
            rigid_contact_max: Upper bound on the Newton ``Contacts``
                buffer's rigid-contact range. Defaults to
                ``max_contact_columns * 6``.
            num_shapes: Total shape count. Must satisfy
                ``num_shapes * num_shapes < 2**31``.
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
            device: Warp device. Defaults to ``bodies.position.device``.
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
            # One column covers a shape pair's multi-contact cluster,
            # so a 6x multiplier is a conservative default.
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
            raise ValueError(f"solver_iterations must be >= 1 (got {self.solver_iterations})")
        self.velocity_iterations = int(velocity_iterations)
        if self.velocity_iterations < 0:
            raise ValueError(f"velocity_iterations must be >= 0 (got {self.velocity_iterations})")
        self.position_iterations = int(position_iterations)

        self.num_worlds: int = int(num_worlds)
        if self.num_worlds <= 0:
            raise ValueError(f"num_worlds must be >= 1 (got {self.num_worlds})")
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
        self._world_color_counts: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )
        self._world_color_cursor: wp.array2d[wp.int32] = wp.zeros(
            (nw, MAX_COLORS + 1), dtype=wp.int32, device=self.device
        )

        # ----- Contact infrastructure -----
        # ContactContainer is keyed by contact index ``k``, so it
        # sizes to ``rigid_contact_max``.
        if self.max_contact_columns > 0:
            self._contact_container: ContactContainer = contact_container_zeros(
                self.rigid_contact_max, device=self.device
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
        """Minimum per-constraint dword width the
        :class:`ConstraintContainer` must carry.
        ``CONTACT_DWORDS`` if contacts-only; ``max(CONTACT_DWORDS,
        ADBS_DWORDS)`` when any joints are reserved."""
        if int(num_joints) > 0:
            return max(int(CONTACT_DWORDS), int(ADBS_DWORDS))
        return int(CONTACT_DWORDS)

    @staticmethod
    def make_constraint_container(
        num_joints: int,
        max_contact_columns: int,
        device: wp.context.Devicelike = None,
    ) -> ConstraintContainer:
        """Factory for a correctly-sized :class:`ConstraintContainer`
        with capacity ``num_joints + max_contact_columns`` and the
        required dword width (see
        :meth:`required_constraint_dwords`)."""
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
            self._partitioner.build_csr()
            self._build_world_csr()

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
            self._integrate_forces()
            self._integrate_gravity()
            self._solve_main()
            self._integrate_positions()
            self._position_iterate()
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
                self.bodies,
                self._num_active_constraints,
                self._elements,
            ],
            device=self.device,
        )

    def _build_world_csr(self) -> None:
        """Bucket the partitioner's global CSR into per-world slices.
        Fully device-side; proportional to the capacity."""
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
            outputs=[self._world_color_starts, self._world_num_colors, self._world_totals_shifted],
            device=self.device,
        )
        # Cross-world prefix: inclusive scan of the shifted per-world
        # totals produces ``world_csr_offsets[0] = 0`` and
        # ``world_csr_offsets[w + 1] = sum(totals[0..w])``. Replaces a
        # single-thread serial scan that grew to 206 us at 1024 worlds.
        wp.utils.array_scan(self._world_totals_shifted, self._world_csr_offsets, inclusive=True)
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
        """Apply per-body force / torque accumulators to velocity.
        Runs every substep."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_apply_external_forces_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _integrate_gravity(self) -> None:
        """``v += gravity[world_id] * dt`` per substep."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _phoenx_integrate_gravity_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, self.gravity, wp.float32(self.substep_dt)],
            device=self.device,
        )

    def _solve_main(self) -> None:
        """Main PGS solve per substep: prepare + ``solver_iterations``
        iterate sweeps with bias ON."""
        if self._constraint_capacity == 0:
            return
        idt = wp.float32(1.0 / self.substep_dt)
        contact_views = self._contact_views if self._contact_views is not None else self._contact_views_placeholder
        self._launch_fast_prepare(idt, contact_views)
        self._launch_fast_iter(
            _constraint_iterate_fast_tail_kernel,
            self.solver_iterations,
            idt,
            contact_views,
        )

    def _position_iterate(self) -> None:
        """XPBD-style contact tangent-drift sweeps. Contacts-only;
        joints have no XPBD path."""
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

    def _launch_fast_prepare(
        self,
        idt: wp.float32,
        contact_views: ContactViews,
    ) -> None:
        """Launch the prepare kernel, one block per world."""
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
        """Launch an iterate / relax kernel that runs
        ``num_iterations`` sweeps internally, one block per world."""
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
        resolve target, infer velocity."""
        if self.num_bodies == 0:
            return
        wp.launch(
            _kinematic_prepare_step_kernel,
            dim=self.num_bodies,
            inputs=[self.bodies, wp.float32(self.step_dt)],
            device=self.device,
        )

    def _kinematic_interpolate_substep(self, alpha: float) -> None:
        """Per-substep kinematic pose update via lerp / slerp."""
        if self.num_bodies == 0:
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
                self.bodies,
                wp.int32(self._constraint_capacity),
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
                self.bodies,
                wp.int32(self._constraint_capacity),
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
        length :attr:`max_contact_columns * 6`."""
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
