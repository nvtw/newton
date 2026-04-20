# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the Jitter-demo port (``example_demo_*``).

The Newton <-> Jitter integration pattern is identical across every
demo: build a scene through :class:`newton.ModelBuilder`, drive
collision detection through :class:`newton.CollisionPipeline`, mirror
the Newton bodies into a :class:`WorldBuilder`, and finally shuttle
pose / velocity between the two state containers every frame. This
module factors the boilerplate out of the individual demos so each
``example_demo_XX.py`` stays focused on the *scene* (shape layout,
joints, cameras, asserts) rather than on repeated plumbing.

Newton up-axis is +Z (default ``ModelBuilder``). Gravity is therefore
``(0, 0, -9.81)`` here -- the Jitter C# reference uses +Y-up so all
ported scenes have their y/z swapped on import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.body import MOTION_DYNAMIC, MOTION_STATIC
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import World
from newton._src.solvers.jitter.world_builder import (
    DriveMode,
    JointHandle,
    JointMode,
    RigidBodyDescriptor,
    WorldBuilder,
)

__all__ = [
    "WORLD_BODY",
    "DemoExample",
    "build_jitter_joints_from_model",
    "build_jitter_world_from_model",
    "jitter_to_newton_kernel",
    "newton_to_jitter_kernel",
]


# Sentinel for :meth:`DemoExample.add_joint` when a joint should be
# anchored to Jitter's static world body (Jitter body 0). Pass this in
# place of a Newton body index to connect a joint to ground.
WORLD_BODY = -1


# ---------------------------------------------------------------------------
# State sync kernels (Newton <-> Jitter)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def newton_to_jitter_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # out
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
):
    """Push Newton's body state into Jitter's :class:`BodyContainer`.

    Newton stores the body-origin transform in ``body_q`` and the COM
    as a body-local offset in ``body_com``. Jitter's
    ``BodyContainer.position`` is the COM in world space, so we rotate
    ``body_com`` into world and add it to ``body_q``'s translation.
    Orientation is shared verbatim. ``body_qd`` is
    ``spatial_vector(linear_COM_velocity, angular_velocity)`` in world
    frame so the copy is direct.
    """
    i = wp.tid()
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[i] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[i] = rot
    qd = body_qd[i]
    velocity[i] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[i] = wp.vec3f(qd[3], qd[4], qd[5])


@wp.kernel(enable_backward=False)
def jitter_to_newton_kernel(
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    # out
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Write Jitter's body state back into Newton's ``body_q`` / ``body_qd``.

    Reverse of :func:`newton_to_jitter_kernel`: Jitter's ``position``
    is the COM in world space, Newton's ``body_q`` is the body-origin
    transform, so we subtract the rotated COM offset.
    """
    i = wp.tid()
    rot = orientation[i]
    com_world = wp.quat_rotate(rot, body_com[i])
    pos_body = position[i] - com_world
    body_q[i] = wp.transform(pos_body, rot)
    body_qd[i] = wp.spatial_vector(velocity[i], angular_velocity[i])


# ---------------------------------------------------------------------------
# Body mirror (Newton Model -> Jitter WorldBuilder)
# ---------------------------------------------------------------------------


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy equivalent of ``wp.quat_rotate`` with xyzw layout.

    Kept local so the host-side finalize path doesn't reach into a
    shared utils module; the whole demo stack stays import-light.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def build_jitter_world_from_model(
    model: newton.Model,
) -> tuple[WorldBuilder, dict[int, int]]:
    """Mirror Newton's body set into a fresh :class:`WorldBuilder`.

    Jitter body 0 is the auto-created static world anchor, so Newton
    body ``i`` lands at Jitter body ``i + 1``; the returned map carries
    that shift so joints / pickers can translate Newton indices to
    Jitter's.

    Returns:
        The populated builder (nothing finalized yet) and the
        ``newton_index -> jitter_index`` map.
    """
    body_q_np = model.body_q.numpy()
    body_inv_mass_np = (
        model.body_inv_mass.numpy() if model.body_inv_mass is not None else None
    )
    body_inv_inertia_np = (
        model.body_inv_inertia.numpy()
        if model.body_inv_inertia is not None
        else None
    )
    body_com_np = model.body_com.numpy()

    builder = WorldBuilder()
    newton_to_jitter: dict[int, int] = {}

    for i in range(model.body_count):
        inv_m = float(body_inv_mass_np[i]) if body_inv_mass_np is not None else 0.0
        inv_I = (
            body_inv_inertia_np[i]
            if body_inv_inertia_np is not None
            else np.zeros((3, 3), dtype=np.float32)
        )
        is_static = inv_m <= 0.0
        pose = body_q_np[i]
        origin_pos = pose[:3]
        rot = pose[3:7]  # xyzw
        com_local = body_com_np[i]
        com_world = origin_pos + _quat_rotate_np(rot, com_local)

        if is_static:
            zero = np.zeros((3, 3), dtype=np.float32)
            desc = RigidBodyDescriptor(
                position=tuple(com_world.tolist()),
                orientation=tuple(rot.tolist()),
                motion_type=int(MOTION_STATIC),
                inverse_mass=0.0,
                inverse_inertia=tuple(tuple(float(x) for x in r) for r in zero),
                affected_by_gravity=False,
            )
        else:
            desc = RigidBodyDescriptor(
                position=tuple(com_world.tolist()),
                orientation=tuple(rot.tolist()),
                motion_type=int(MOTION_DYNAMIC),
                inverse_mass=inv_m,
                inverse_inertia=tuple(tuple(float(x) for x in r) for r in inv_I),
                affected_by_gravity=True,
            )
        newton_to_jitter[i] = builder.add_body(desc)

    return builder, newton_to_jitter


# ---------------------------------------------------------------------------
# Joint mirror (Newton Model -> Jitter WorldBuilder)
# ---------------------------------------------------------------------------


def _transform_point_np(xform: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a Newton ``wp.transform`` (stored as 7 floats: px,py,pz,qx,qy,qz,qw)
    to a 3-vector: ``xform.p + rotate(xform.q, p)``. Host-side NumPy so
    joint mirroring stays on the CPU.
    """
    pos = xform[:3]
    rot = xform[3:7]
    return pos + _quat_rotate_np(rot, p)


def _compose_transform_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two Newton transforms stored as 7-floats (xyzw)."""
    ap = a[:3]
    aq = a[3:7]
    bp = b[:3]
    bq = b[3:7]
    out_p = ap + _quat_rotate_np(aq, bp)
    # Quat multiply (xyzw).
    ax, ay, az, aw = float(aq[0]), float(aq[1]), float(aq[2]), float(aq[3])
    bx, by, bz, bw = float(bq[0]), float(bq[1]), float(bq[2]), float(bq[3])
    out_q = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )
    return np.concatenate([out_p, out_q])


def build_jitter_joints_from_model(
    model: newton.Model,
    state: newton.State,
    builder: WorldBuilder,
    newton_to_jitter: dict[int, int],
    *,
    drive_mode: DriveMode = DriveMode.OFF,
    max_force_drive: float = 0.0,
    apply_joint_targets: bool = True,
    skip_fixed: bool = True,
) -> list[JointHandle]:
    """Translate every articulation joint in ``model`` into Jitter joints.

    Designed for scenes whose joints live on the
    :class:`~newton.ModelBuilder` side (e.g. ``builder.add_urdf`` or
    ``builder.add_mjcf``) rather than being queued manually via
    :meth:`DemoExample.add_joint`. Intended to be called from a
    subclass's :meth:`DemoExample.on_jitter_builder_ready` hook --
    by then the Newton :class:`~newton.Model` is finalised, its
    :class:`~newton.State` is filled by ``eval_fk``, and the Jitter
    :class:`WorldBuilder` knows about every Newton body.

    Mapping rules:

    * ``FREE``: skipped. Jitter has no free-joint constraint -- the
      child body is already an unconstrained dynamic rigid body, so
      no builder entry is needed.
    * ``REVOLUTE``: installed as :attr:`JointMode.REVOLUTE` with the
      axis taken from ``model.joint_axis[joint_qd_start[j]]`` (parent
      frame, rotated into world) and the shared anchor taken from
      ``body_q[child] * joint_X_c[j]``. Limits are copied from
      ``joint_limit_lower/upper``; ``drive_mode`` + ``target`` can
      optionally drive the hinge towards ``joint_target_pos``.
    * ``FIXED``: skipped by default (URDF scenes commonly use fixed
      joints only to weld visual-only links that Newton may have
      already merged into the parent body; we don't emit a weld).
      Pass ``skip_fixed=False`` to raise instead.
    * ``BALL``, ``PRISMATIC``, ``D6``, ``DISTANCE``, ``CABLE``: not
      yet mapped -- raises :class:`NotImplementedError` for visibility.

    Args:
        model: Finalised Newton :class:`~newton.Model`.
        state: Newton :class:`~newton.State` with ``body_q`` populated
            (typically the result of :func:`newton.eval_fk`). World-space
            anchors are derived from it.
        builder: Jitter :class:`WorldBuilder` mirroring ``model``'s
            bodies (produced by :func:`build_jitter_world_from_model`).
        newton_to_jitter: Newton body index -> Jitter body index map
            from :func:`build_jitter_world_from_model`. Newton bodies
            with ``joint_parent == -1`` map to Jitter's world anchor.
        drive_mode: Actuator mode for revolute joints. Defaults to
            :attr:`DriveMode.OFF` (passive hinges matching URDF's
            unactuated physics). Use :attr:`DriveMode.POSITION` to
            turn on a PD drive toward ``joint_target_pos``.
        max_force_drive: Per-substep torque cap for the drive [N*m].
            Ignored when ``drive_mode`` is :attr:`DriveMode.OFF`.
        apply_joint_targets: When ``True`` (default) and ``drive_mode``
            is :attr:`DriveMode.POSITION`, use ``model.joint_target_pos``
            as the per-joint setpoint. When ``False``, setpoint stays
            at 0 for every joint.
        skip_fixed: When ``True`` (default) ``FIXED`` joints are
            silently skipped; ``False`` raises :class:`NotImplementedError`
            so the caller can decide what to do.

    Returns:
        List of :class:`JointHandle` for every installed joint, in
        iteration order over ``model.joint_type``.
    """
    body_q_np = state.body_q.numpy()
    joint_type_np = model.joint_type.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_X_p_np = model.joint_X_p.numpy()
    joint_X_c_np = model.joint_X_c.numpy()
    joint_axis_np = model.joint_axis.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    joint_limit_lower_np = (
        model.joint_limit_lower.numpy()
        if model.joint_limit_lower is not None
        else None
    )
    joint_limit_upper_np = (
        model.joint_limit_upper.numpy()
        if model.joint_limit_upper is not None
        else None
    )
    joint_target_pos_np = (
        model.joint_target_pos.numpy()
        if model.joint_target_pos is not None
        else None
    )

    def _to_jitter(newton_idx: int) -> int:
        if newton_idx < 0:
            return int(builder.world_body)
        return int(newton_to_jitter[int(newton_idx)])

    from newton import JointType  # local: avoid circular imports at module import

    handles: list[JointHandle] = []
    n = int(model.joint_count)
    for j in range(n):
        jt = int(joint_type_np[j])
        if jt == int(JointType.FREE):
            continue
        if jt == int(JointType.FIXED):
            if skip_fixed:
                continue
            raise NotImplementedError(
                f"build_jitter_joints_from_model: JointType.FIXED at joint {j} "
                "is not yet mapped to a Jitter weld."
            )
        if jt != int(JointType.REVOLUTE):
            raise NotImplementedError(
                f"build_jitter_joints_from_model: joint {j} has JointType "
                f"{JointType(jt).name}; only REVOLUTE / FREE / FIXED are "
                "currently supported."
            )

        parent = int(joint_parent_np[j])
        child = int(joint_child_np[j])
        X_c = np.asarray(joint_X_c_np[j], dtype=np.float32).reshape(-1)
        X_p = np.asarray(joint_X_p_np[j], dtype=np.float32).reshape(-1)

        # World-space anchor: use the child-side transform so we get
        # the anchor exactly where the child body sits (the two anchors
        # coincide at rest, but floating-point drift can make the
        # parent-side point a hair off).
        body_pose_c = np.asarray(body_q_np[child], dtype=np.float32).reshape(-1)
        anchor_world = _transform_point_np(body_pose_c, X_c[:3])

        # Hinge axis lives in the parent anchor frame. Rotate it to
        # world via (body_q[parent] * joint_X_p).rotation, or -- if
        # parent is the world -- straight through X_p.
        qd_start = int(joint_qd_start_np[j])
        axis_local = np.asarray(joint_axis_np[qd_start], dtype=np.float32)
        if parent < 0:
            parent_anchor_world = X_p
        else:
            body_pose_p = np.asarray(
                body_q_np[parent], dtype=np.float32
            ).reshape(-1)
            parent_anchor_world = _compose_transform_np(body_pose_p, X_p)
        axis_world = _quat_rotate_np(parent_anchor_world[3:7], axis_local)
        # Normalise (URDF axes are already unit but belt-and-braces).
        n_axis = float(np.linalg.norm(axis_world))
        if n_axis > 1.0e-8:
            axis_world = axis_world / n_axis

        anchor1 = tuple(float(x) for x in anchor_world)
        anchor2 = (
            anchor1[0] + float(axis_world[0]),
            anchor1[1] + float(axis_world[1]),
            anchor1[2] + float(axis_world[2]),
        )

        lo = (
            float(joint_limit_lower_np[qd_start])
            if joint_limit_lower_np is not None
            else 0.0
        )
        hi = (
            float(joint_limit_upper_np[qd_start])
            if joint_limit_upper_np is not None
            else 0.0
        )
        target = 0.0
        if (
            apply_joint_targets
            and drive_mode is DriveMode.POSITION
            and joint_target_pos_np is not None
        ):
            target = float(joint_target_pos_np[qd_start])

        handles.append(
            builder.add_joint(
                body1=_to_jitter(parent),
                body2=_to_jitter(child),
                anchor1=anchor1,
                anchor2=anchor2,
                mode=JointMode.REVOLUTE,
                drive_mode=drive_mode,
                target=target,
                max_force_drive=max_force_drive,
                min_value=lo,
                max_value=hi,
            )
        )
    return handles


# ---------------------------------------------------------------------------
# Reusable demo base class
# ---------------------------------------------------------------------------


@dataclass
class DemoConfig:
    """Per-demo knobs a subclass sets before :meth:`DemoExample.finish_setup`.

    Central so every demo looks the same at the top of ``__init__``.
    """

    #: Window title / label; also used for log lines.
    title: str = "Jitter Demo"
    #: Camera pose (world space, Newton +Z-up convention).
    camera_pos: tuple[float, float, float] = (10.0, 10.0, 5.0)
    camera_pitch: float = -20.0
    camera_yaw: float = -45.0
    #: Integration step settings.
    fps: int = 60
    substeps: int = 4
    solver_iterations: int = 8
    #: Gravity vector (Newton +Z-up).
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


class DemoExample:
    """Base class for the Jitter-port demo scenes.

    Subclasses:

    1. set :attr:`config` (a :class:`DemoConfig`),
    2. populate ``self.model_builder`` (a :class:`newton.ModelBuilder`)
       with their scene inside :meth:`build_scene`,
    3. optionally populate ``self.joints`` (world-space anchor pairs
       mirrored through :meth:`WorldBuilder.add_joint`),
    4. call :meth:`finish_setup` at the end of their ``__init__``.

    Rendering uses ``viewer.log_state`` so all Newton shape types
    (box, sphere, capsule, cylinder, plane) render through whatever
    path the Newton viewer already supports -- no per-demo
    ``log_shapes`` boilerplate.

    Graph capture (CUDA) and picking are set up automatically. Each
    subclass can override :meth:`test_final` to assert a settled state.
    """

    def __init__(self, viewer, args, config: DemoConfig):
        self.viewer = viewer
        self.args = args
        self.config = config
        self.device = wp.get_device()

        self.fps = int(config.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(config.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_iterations = int(config.solver_iterations)
        self._frame: int = 0

        # Per-demo scratch -- subclasses fill these out.
        self.model_builder: newton.ModelBuilder = newton.ModelBuilder()
        self.pending_joints: list[dict] = []
        # Newton body indices that picking should treat as pickable
        # (half-extent / radius for the ray cube test). Populated by
        # :meth:`register_body_extent` so :meth:`finish_setup` can
        # build Jitter's half-extents array.
        self._newton_body_extents: dict[int, tuple[float, float, float]] = {}
        # Queued body-pair contact filters (Newton body indices).
        # Mirrored twice at :meth:`finish_setup`: once into the Newton
        # :class:`ModelBuilder` as shape-pair filters (so the broad
        # phase never emits the contacts in the first place), and
        # once into the Jitter :class:`WorldBuilder` as body-pair
        # filters (so anything the broad phase *does* emit still gets
        # dropped on the Jitter side -- defense in depth).
        self._pending_collision_filters: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Hooks subclasses override
    # ------------------------------------------------------------------

    def build_scene(self) -> None:
        """Populate :attr:`model_builder` and :attr:`pending_joints`.

        Override in each demo. When this returns, ``self.model_builder``
        must be in a finalise-able state and ``self.pending_joints``
        must contain plain dicts forwarded to
        :meth:`WorldBuilder.add_joint` (with Newton body indices --
        these get shifted to Jitter indices in :meth:`finish_setup`).
        """
        raise NotImplementedError

    def test_final(self) -> None:
        """Optional: assert the settled scene state.

        Default is a no-op. Override to check that stacks stayed
        stacked, motors turned, etc.
        """
        return None

    def on_jitter_builder_ready(
        self,
        builder: WorldBuilder,
        newton_to_jitter: dict[int, int],
    ) -> None:
        """Optional: customise the Jitter :class:`WorldBuilder`.

        Called from :meth:`finish_setup` right after the builder has
        mirrored every Newton body and every queued
        :meth:`pending_joints` / :meth:`add_collision_filter_pair` has
        been installed, but *before* :meth:`WorldBuilder.finalize`.
        Default is a no-op.

        The typical use is mirroring Newton-side articulations (URDF,
        MJCF, etc.) into the Jitter builder via
        :func:`build_jitter_joints_from_model` -- by this point
        ``self.model`` / ``self.state`` are finalised so world-space
        anchors can be derived from them.
        """
        return None

    # ------------------------------------------------------------------
    # Shared scene-construction helpers
    # ------------------------------------------------------------------

    def register_body_extent(
        self, newton_body: int, extent: tuple[float, float, float]
    ) -> None:
        """Record a bounding half-extent (or bounding sphere triple) for
        picking. Entries with any component ``<= 0`` are treated as
        unpickable; matches the world-anchor convention."""
        self._newton_body_extents[int(newton_body)] = (
            float(extent[0]),
            float(extent[1]),
            float(extent[2]),
        )

    def add_joint(self, **kwargs) -> None:
        """Queue a joint for :class:`WorldBuilder.add_joint`.

        Uses Newton body indices; :meth:`finish_setup` translates them
        to Jitter indices after :meth:`build_jitter_world_from_model`
        runs.
        """
        self.pending_joints.append(dict(kwargs))

    def add_collision_filter_pair(
        self, newton_body_a: int, newton_body_b: int
    ) -> None:
        """Ignore contacts between two Newton bodies.

        Queues a body-pair collision filter using *Newton* body
        indices (matches :meth:`add_joint`'s convention). Pass
        :data:`WORLD_BODY` for the static world anchor. At
        :meth:`finish_setup` time each queued pair is installed in
        two places:

        1. The Newton :class:`ModelBuilder` via
           :meth:`~newton.ModelBuilder.add_shape_collision_filter_pair`
           for every shape attached to body A crossed with every
           shape attached to body B. The collision pipeline skips
           these pairs in the broad phase, so filtered contacts
           never enter the :class:`Contacts` buffer at all.
        2. The Jitter :class:`WorldBuilder` via
           :meth:`~newton._src.solvers.jitter.world_builder.WorldBuilder.add_collision_filter_pair`
           using the translated Jitter body indices. This is
           defence in depth: any contact the broad phase still
           produces (e.g. because of a custom broad-phase mode
           that doesn't honour the shape filter) is still dropped
           on the Jitter side before a constraint column is
           allocated.

        Typical use is to suppress self-collision between jointed
        limbs of a ragdoll -- adjacent limbs overlap at the joint
        anchor and the spurious contacts fight the joint's
        positional constraint, which makes the whole articulation
        jitter.

        Idempotent: duplicate or argument-order-reversed registrations
        collapse to one canonical ``(min, max)`` pair.

        Args:
            newton_body_a: First Newton body index (or
                :data:`WORLD_BODY` for the static world anchor).
            newton_body_b: Second Newton body index (or
                :data:`WORLD_BODY`).
        """
        self._pending_collision_filters.append(
            (int(newton_body_a), int(newton_body_b))
        )

    # ------------------------------------------------------------------
    # finish_setup: finalize Newton model + Jitter world, wire viewer
    # ------------------------------------------------------------------

    def finish_setup(self) -> None:
        """Finalise the Newton model, build the Jitter world, capture."""
        self.build_scene()

        # Expand each queued body-pair filter into the Cartesian
        # product of the two bodies' shapes and register them with
        # the ModelBuilder *before* finalize(): the builder bakes
        # ``shape_collision_filter_pairs`` into ``shape_contact_pairs``
        # at finalize time, so any filter added after will be ignored
        # by the broad phase. We still keep the list to mirror into
        # the Jitter WorldBuilder below.
        for newton_a, newton_b in self._pending_collision_filters:
            shapes_a = self.model_builder.body_shapes.get(newton_a, [])
            shapes_b = self.model_builder.body_shapes.get(newton_b, [])
            for sa in shapes_a:
                for sb in shapes_b:
                    if sa == sb:
                        continue
                    self.model_builder.add_shape_collision_filter_pair(sa, sb)

        self.model = self.model_builder.finalize()
        print(
            f"[{self.config.title}] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count}"
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=True
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # Sync the FK-derived body poses back into ``model.body_q`` so
        # :func:`build_jitter_world_from_model` spawns Jitter bodies
        # at the same world transforms Newton's state is about to
        # start from. For scenes built entirely through direct
        # ``add_body(xform=...)`` calls this is a no-op (both arrays
        # already agree); for articulations loaded via ``add_urdf`` /
        # ``add_mjcf`` or overridden ``joint_q``, ``eval_fk`` changes
        # the state without touching ``model.body_q``, so the Jitter
        # mirror would otherwise spawn links at the zero-pose rest
        # configuration and the articulation joints would explosively
        # pull them together.
        self.model.body_q.assign(self.state.body_q)

        builder, newton_to_jitter = build_jitter_world_from_model(self.model)

        def _to_jitter_body(newton_idx: int) -> int:
            """Translate a Newton body index (or :data:`WORLD_BODY`)
            into the matching Jitter body index."""
            if newton_idx == WORLD_BODY:
                return int(builder.world_body)
            return newton_to_jitter[int(newton_idx)]

        # Translate queued joints from Newton indices to Jitter
        # indices and install them in the builder.
        self._joint_handles: list[JointHandle] = []
        for jspec in self.pending_joints:
            jargs = dict(jspec)
            jargs["body1"] = _to_jitter_body(int(jargs["body1"]))
            jargs["body2"] = _to_jitter_body(int(jargs["body2"]))
            self._joint_handles.append(builder.add_joint(**jargs))

        # Subclass hook: the Newton model is finalised, its state is
        # filled by eval_fk, the Jitter builder knows about every
        # body, and ``pending_joints`` has been installed -- a
        # convenient spot to mirror extra joints from the Newton
        # :class:`ModelBuilder` (e.g. via
        # :func:`build_jitter_joints_from_model` for URDF-loaded
        # articulations) without the subclass having to reimplement
        # the body translation map.
        self.on_jitter_builder_ready(builder, newton_to_jitter)

        # Mirror queued body-pair collision filters onto the Jitter
        # builder. The same pairs were expanded to shape-level
        # filters on the Newton ModelBuilder above; duplicating on
        # the Jitter side is cheap and defends against any code path
        # that bypasses the broad-phase shape filter.
        for newton_a, newton_b in self._pending_collision_filters:
            ja = _to_jitter_body(newton_a)
            jb = _to_jitter_body(newton_b)
            if ja == jb:
                continue
            builder.add_collision_filter_pair(ja, jb)

        # Budget: each Newton contact column holds up to 6 contact
        # points, so ``ceil(rigid_contact_max / 6)`` is a tight
        # worst case. Sized floor of 16 avoids tiny grids on trivial
        # scenes where the contact column count would otherwise
        # shrink below a warp.
        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        num_shapes = int(self.model.shape_count)

        self.world: World = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            gravity=self.config.gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            device=self.device,
        )
        self._newton_to_jitter = newton_to_jitter

        # Shape -> Jitter body map. Newton's ``shape_body`` holds
        # Newton body indices (or -1 for the world); shift by +1 to
        # match Jitter's body layout (Jitter body 0 is the static
        # world anchor).
        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        self._sync_newton_to_jitter()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(*self.config.camera_pos),
            pitch=self.config.camera_pitch,
            yaw=self.config.camera_yaw,
        )

        # Picking: one half-extent triple per Jitter body, body 0
        # marked (0,0,0) so ray-casts ignore the world anchor.
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx, extent in self._newton_body_extents.items():
            jitter_idx = self._newton_to_jitter[newton_idx]
            half_extents_np[jitter_idx] = extent
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = JitterPicking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # CUDA graph capture for the whole per-frame step.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        # Diagnostic -- subclasses can flip to False if they want
        # quieter output.
        self._log_colors_every = 60

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._sync_newton_to_jitter()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            picking=self.picking,
        )
        self._sync_jitter_to_newton()

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_jitter_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_jitter_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            jitter_to_newton_kernel,
            dim=n,
            inputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame += 1
        if self._log_colors_every > 0 and self._frame % self._log_colors_every == 0:
            print(
                f"[{self.config.title}] frame {self._frame:5d}  "
                f"colors used by PGS: {self.world.num_colors_used()}"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Newton-style helpers used by assertions
    # ------------------------------------------------------------------

    def jitter_body_state(
        self, newton_body: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position, velocity)`` for ``newton_body`` as NumPy
        arrays, copying from Jitter's :class:`BodyContainer`. Intended
        for use inside :meth:`test_final`.
        """
        j = self._newton_to_jitter[int(newton_body)]
        return (
            self.world.bodies.position.numpy()[j],
            self.world.bodies.velocity.numpy()[j],
        )

    @staticmethod
    def default_parser():
        return newton.examples.create_parser()
