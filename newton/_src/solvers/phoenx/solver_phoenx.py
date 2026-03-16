# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhoenX solver state: body/contact stores, warm starting, PGS solving.

This module ties together the column-major storage
(:class:`~newton._src.solvers.phoenx.data_base.HandleStore` /
:class:`~newton._src.solvers.phoenx.data_base.DataStore`),
:class:`~newton._src.solvers.phoenx.warm_start.WarmStarter`,
:class:`~newton._src.solvers.phoenx.maximal_independent_set.GraphColoring`,
and PGS contact-solver kernels into a single :class:`SolverState`
that interacts with Newton's :class:`CollisionPipeline`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .constraints import (
    JOINT_BALL_SOCKET,
    JOINT_DISTANCE_LIMIT,
    JOINT_FIXED,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    ConstraintKernels,
    JointSchema,
)
from .data_base import DataStore, HandleStore
from .contacts import (
    ContactKernels,
    build_bundle_elements_kernel,
    clear_contact_count_kernel,
    count_contacts_per_body_kernel,
    import_contacts_kernel,
)
from .contacts_xpbd import (
    ContactKernelsXPBD,
    derive_velocities_kernel,
    predict_positions_kernel,
    store_ref_state_kernel,
)
from .kernels import (
    add_int32_kernel,
    integrate_positions_kernel,
    integrate_velocities_kernel,
    update_world_inertia_kernel,
)
from .maximal_independent_set import GraphColoring
from .schemas import BODY_FLAG_STATIC, ContactPointSchema, RigidBodySchema
from .warm_start import WarmStarter

MAX_BODIES_PER_ELEMENT = 8


class SolverState:
    """Central state container for the PhoenX rigid-body solver.

    Holds a :class:`HandleStore` for bodies, a :class:`DataStore` for
    contacts (rebuilt every frame), a :class:`WarmStarter` for
    cross-frame impulse transfer, and a :class:`GraphColoring` for
    partitioned parallel PGS solving.

    Args:
        body_capacity: maximum number of rigid bodies.
        contact_capacity: maximum number of contact points per frame.
        shape_count: number of shapes (for the shape-to-body map).
        device: Warp device string or object.
        default_friction: Coulomb friction coefficient used when importing contacts.
        max_colors: maximum number of colour partitions for graph coloring.
    """

    def __init__(
        self,
        body_capacity: int,
        contact_capacity: int,
        shape_count: int,
        device: wp.context.Device | str | None = None,
        default_friction: float = 0.5,
        max_colors: int = 12,
        joint_capacity: int = 0,
        contact_mode: str = "pgs",
    ):
        self.device = wp.get_device(device)
        d = self.device

        self.body_store = HandleStore(RigidBodySchema, body_capacity, device=d)
        self.contact_store = DataStore(ContactPointSchema, contact_capacity, device=d)
        self.warm_starter = WarmStarter(contact_capacity, device=d)

        total_elements = contact_capacity + joint_capacity
        self.graph_coloring = GraphColoring(
            max_elements=total_elements,
            max_nodes=body_capacity,
            max_colors=max_colors,
            device=d,
        )

        self.shape_body = wp.full(shape_count, -1, dtype=wp.int32, device=d)
        self.shape_count = shape_count
        self.default_friction = default_friction

        self._elements = wp.zeros((total_elements, MAX_BODIES_PER_ELEMENT), dtype=wp.int32, device=d)

        # Mass splitting: per-body contact count (Tonge et al. 2012)
        self._contact_count_per_body = wp.zeros(body_capacity, dtype=wp.int32, device=d)

        # Pre-allocated device counter for total element count (bundles + joints).
        # Must NOT be re-allocated per frame — that breaks CUDA graph capture.
        self._total_element_count = wp.zeros(1, dtype=wp.int32, device=d)

        # Maximum number of bundles (worst case: every contact is its own bundle)
        self.capacity_bundles = contact_capacity

        # Contact kernels (bake column offsets at construction time)
        self.contact_mode = contact_mode
        if contact_mode == "xpbd":
            self._contact_kernels = ContactKernelsXPBD(self.contact_store, self.body_store)
            # Reference state arrays for XPBD position-level solving
            self._ref_position = wp.zeros(body_capacity, dtype=wp.vec3, device=d)
            self._ref_orientation = wp.zeros(body_capacity, dtype=wp.quat, device=d)
        else:
            self._contact_kernels = ContactKernels(self.contact_store, self.body_store)

        # Joint storage
        self.joint_capacity = joint_capacity
        if joint_capacity > 0:
            self.joint_store = DataStore(JointSchema, joint_capacity, device=d)
            self._constraint_kernels = ConstraintKernels(self.joint_store, self.body_store)
        else:
            self.joint_store = None
            self._constraint_kernels = None
        self._joint_count = 0

    # -- body management (host-side) ----------------------------------------

    def add_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        inverse_mass: float = 1.0,
        inverse_inertia_local: np.ndarray | None = None,
        linear_damping: float = 1.0,
        angular_damping: float = 1.0,
        is_static: bool = False,
    ) -> int:
        """Allocate a rigid body and initialise its columns.

        Returns:
            Handle ID (non-negative), or ``-1`` if at capacity.
        """
        handle = self.body_store.allocate()
        if handle < 0:
            return -1

        row = int(self.body_store.handle_to_index.numpy()[handle])
        bs = self.body_store

        def _write_col(name, value, dtype):
            col = bs.column_of(name).numpy()
            col[row] = value
            bs.column_of(name).assign(wp.array(col, dtype=dtype, device=self.device))

        _write_col("position", np.array(position, dtype=np.float32), wp.vec3)
        _write_col("orientation", np.array(orientation, dtype=np.float32), wp.quat)
        _write_col("velocity", np.array(velocity, dtype=np.float32), wp.vec3)
        _write_col("angular_velocity", np.array(angular_velocity, dtype=np.float32), wp.vec3)

        mass_col = bs.column_of("inverse_mass").numpy()
        mass_col[row] = 0.0 if is_static else inverse_mass
        bs.column_of("inverse_mass").assign(wp.array(mass_col, dtype=wp.float32, device=self.device))

        if inverse_inertia_local is not None:
            inv_i = np.array(inverse_inertia_local, dtype=np.float32).reshape(3, 3)
        else:
            inv_i = np.zeros((3, 3), dtype=np.float32) if is_static else np.eye(3, dtype=np.float32) * inverse_mass
        inertia_col = bs.column_of("inverse_inertia_local").numpy()
        inertia_col[row] = inv_i
        bs.column_of("inverse_inertia_local").assign(wp.array(inertia_col, dtype=wp.mat33, device=self.device))

        damp_l = bs.column_of("linear_damping").numpy()
        damp_l[row] = linear_damping
        bs.column_of("linear_damping").assign(wp.array(damp_l, dtype=wp.float32, device=self.device))

        damp_a = bs.column_of("angular_damping").numpy()
        damp_a[row] = angular_damping
        bs.column_of("angular_damping").assign(wp.array(damp_a, dtype=wp.float32, device=self.device))

        flags_col = bs.column_of("flags").numpy()
        flags_col[row] = BODY_FLAG_STATIC if is_static else 0
        bs.column_of("flags").assign(wp.array(flags_col, dtype=wp.int32, device=self.device))

        return handle

    def set_shape_body(self, shape_index: int, body_handle: int):
        """Map a Newton shape index to a PhoenX body storage row.

        Args:
            shape_index: index into Newton's shape arrays.
            body_handle: handle returned by :meth:`add_body`.
        """
        row = int(self.body_store.handle_to_index.numpy()[body_handle])
        sb = self.shape_body.numpy()
        sb[shape_index] = row
        self.shape_body.assign(wp.array(sb, dtype=wp.int32, device=self.device))

    # -- joint management (host-side) ----------------------------------------

    def add_joint_ball_socket(
        self,
        body_handle0: int,
        body_handle1: int,
        anchor_world: tuple[float, float, float],
    ) -> int:
        """Add a ball-socket joint between two bodies.

        Args:
            body_handle0: handle returned by :meth:`add_body`.
            body_handle1: handle returned by :meth:`add_body`.
            anchor_world: world-space anchor point [m].

        Returns:
            Joint index, or -1 if at capacity.
        """
        return self._add_joint(JOINT_BALL_SOCKET, body_handle0, body_handle1, anchor_world, axis_world=(0.0, 0.0, 1.0))

    def add_joint_revolute(
        self,
        body_handle0: int,
        body_handle1: int,
        anchor_world: tuple[float, float, float],
        axis_world: tuple[float, float, float],
        angle_min: float = -1.0e7,
        angle_max: float = 1.0e7,
    ) -> int:
        """Add a revolute (hinge) joint between two bodies.

        Args:
            body_handle0: handle returned by :meth:`add_body`.
            body_handle1: handle returned by :meth:`add_body`.
            anchor_world: world-space anchor point [m].
            axis_world: world-space hinge axis (will be normalized).
            angle_min: lower angle limit [rad].
            angle_max: upper angle limit [rad].

        Returns:
            Joint index, or -1 if at capacity.
        """
        ji = self._add_joint(JOINT_REVOLUTE, body_handle0, body_handle1, anchor_world, axis_world)
        if ji >= 0:
            js = self.joint_store
            a_min = js.column_of("angle_min").numpy()
            a_max_arr = js.column_of("angle_max").numpy()
            a_min[ji] = angle_min
            a_max_arr[ji] = angle_max
            js.column_of("angle_min").assign(wp.array(a_min, dtype=wp.float32, device=self.device))
            js.column_of("angle_max").assign(wp.array(a_max_arr, dtype=wp.float32, device=self.device))
        return ji

    def add_joint_fixed(
        self,
        body_handle0: int,
        body_handle1: int,
        anchor_world: tuple[float, float, float],
    ) -> int:
        """Add a fixed joint (weld) between two bodies.

        Args:
            body_handle0: handle returned by :meth:`add_body`.
            body_handle1: handle returned by :meth:`add_body`.
            anchor_world: world-space anchor point [m].

        Returns:
            Joint index, or -1 if at capacity.
        """
        return self._add_joint(JOINT_FIXED, body_handle0, body_handle1, anchor_world, axis_world=(0.0, 0.0, 1.0))

    def add_joint_prismatic(
        self,
        body_handle0: int,
        body_handle1: int,
        anchor_world: tuple[float, float, float],
        axis_world: tuple[float, float, float],
        slide_min: float = -1.0e7,
        slide_max: float = 1.0e7,
    ) -> int:
        """Add a prismatic (slider) joint between two bodies.

        The joint constrains body 1 to slide along *axis_world* relative to
        body 0, while preventing rotation and lateral translation.

        Args:
            body_handle0: handle returned by :meth:`add_body`.
            body_handle1: handle returned by :meth:`add_body`.
            anchor_world: world-space anchor point [m].
            axis_world: world-space slide axis (will be normalized).
            slide_min: lower slide limit [m].
            slide_max: upper slide limit [m].

        Returns:
            Joint index, or -1 if at capacity.
        """
        ji = self._add_joint(JOINT_PRISMATIC, body_handle0, body_handle1, anchor_world, axis_world)
        if ji >= 0:
            js = self.joint_store
            a_min = js.column_of("angle_min").numpy()
            a_max_arr = js.column_of("angle_max").numpy()
            a_min[ji] = slide_min
            a_max_arr[ji] = slide_max
            js.column_of("angle_min").assign(wp.array(a_min, dtype=wp.float32, device=self.device))
            js.column_of("angle_max").assign(wp.array(a_max_arr, dtype=wp.float32, device=self.device))
        return ji

    def add_joint_distance_limit(
        self,
        body_handle0: int,
        body_handle1: int,
        anchor0_world: tuple[float, float, float],
        anchor1_world: tuple[float, float, float],
        limit_min: float = 0.0,
        limit_max: float = 0.0,
        stiffness: float = 0.0,
        damping: float = 0.0,
    ) -> int:
        """Add a distance limit (optionally spring-loaded) between two bodies.

        Constrains the distance between two anchor points to stay within
        ``[rest + limit_min, rest + limit_max]`` where *rest* is the
        initial distance between the anchors.  When ``stiffness > 0`` the
        constraint acts as a damped spring (Erin Catto soft constraints).

        Args:
            body_handle0: handle returned by :meth:`add_body`.
            body_handle1: handle returned by :meth:`add_body`.
            anchor0_world: world-space attachment point on body 0 [m].
            anchor1_world: world-space attachment point on body 1 [m].
            limit_min: minimum distance offset from rest [m] (negative = allow compression).
            limit_max: maximum distance offset from rest [m] (positive = allow extension).
            stiffness: spring stiffness k [N/m].  0 = hard constraint.
            damping: spring damping c [N s/m].  0 = undamped.

        Returns:
            Joint index, or -1 if at capacity.
        """
        # Use _add_joint with axis=(0,0,1) placeholder (unused for distance limit)
        ji = self._add_joint(
            JOINT_DISTANCE_LIMIT,
            body_handle0,
            body_handle1,
            anchor0_world,  # stored as local_anchor0 after transform
            axis_world=(0.0, 0.0, 1.0),
        )
        if ji < 0:
            return -1

        js = self.joint_store
        d = self.device

        # _add_joint stores local_anchor0 from anchor0_world.
        # We also need local_anchor1 from anchor1_world.
        bs = self.body_store
        h2i = bs.handle_to_index.numpy()
        row1 = int(h2i[body_handle1])
        pos1 = bs.column_of("position").numpy()[row1]
        orient1 = bs.column_of("orientation").numpy()[row1]

        # Quaternion inverse-rotate (anchor1_world - pos1) into body1 local frame
        a1 = np.array(anchor1_world, dtype=np.float32)

        def _quat_conj(q):
            return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

        def _quat_mul(a, b):
            return np.array([
                a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
            ], dtype=np.float32)

        def _quat_inv_rotate(q, v):
            qc = _quat_conj(q)
            qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
            return _quat_mul(_quat_mul(qc, qv), q)[:3]

        local_anchor1 = _quat_inv_rotate(orient1, a1 - pos1)

        col = js.column_of("local_anchor1").numpy()
        col[ji] = local_anchor1
        js.column_of("local_anchor1").assign(wp.array(col, dtype=wp.vec3, device=d))

        # Compute rest distance from initial anchor positions
        row0 = int(h2i[body_handle0])
        pos0 = bs.column_of("position").numpy()[row0]
        orient0 = bs.column_of("orientation").numpy()[row0]

        def _quat_rotate(q, v):
            qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
            return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[:3]

        w0 = pos0 + _quat_rotate(orient0, js.column_of("local_anchor0").numpy()[ji])
        w1 = pos1 + _quat_rotate(orient1, local_anchor1)
        rest_distance = float(np.linalg.norm(w1 - w0))

        # Write distance-limit-specific fields via flat buffer offsets,
        # since these fields don't exist in the JointSchema alias
        # (RevoluteJointData).  Use schema_col_base to find the offset.
        from .constraints import DistanceLimitJointData, schema_col_base

        cap = js.capacity
        data_np = js.data.numpy()

        def _write_dl(field_name, val):
            offset = schema_col_base(DistanceLimitJointData, cap, field_name)
            data_np[offset + ji] = float(val)

        _write_dl("rest_distance", rest_distance)
        _write_dl("limit_min", limit_min)
        _write_dl("limit_max", limit_max)

        # Pre-compute spring softness and bias factor (Erin Catto GDC 2011)
        # softness = 1 / (dt * (c + dt * k)),  bias_factor = dt * k / (c + dt * k)
        # We use 1/480 as the default substep dt (60 FPS / 8 substeps).
        sim_dt = 1.0 / 480.0
        if stiffness > 0.0:
            soft = 1.0 / (sim_dt * (damping + sim_dt * stiffness))
            bf = sim_dt * stiffness * soft
        else:
            soft = 0.0
            bf = 0.2  # hard constraint: standard Baumgarte factor
        _write_dl("softness", soft)
        _write_dl("bias_factor", bf)
        _write_dl("dl_accumulated_impulse", 0.0)

        js.data.assign(wp.array(data_np, dtype=wp.float32, device=d))

        return ji

    # -- drive configuration ------------------------------------------------

    DRIVE_OFF = 0
    DRIVE_POSITION = 1
    DRIVE_VELOCITY = 2

    def set_joint_drive(
        self,
        joint_index: int,
        mode: int,
        target: float,
        stiffness: float = 100.0,
        damping: float = 10.0,
        max_force: float = 1.0e6,
    ):
        """Configure a motor/drive on a joint.

        For revolute joints the drive acts on the hinge angle. For prismatic
        joints the drive acts on the slide displacement.

        Position drive (``mode=DRIVE_POSITION``): implicit PD controller
        targeting *target* angle/position with given stiffness and damping.

        Velocity drive (``mode=DRIVE_VELOCITY``): targets *target* angular/
        linear velocity with given damping (stiffness is ignored).

        Args:
            joint_index: index returned by :meth:`add_joint_revolute` etc.
            mode: ``DRIVE_OFF`` (0), ``DRIVE_POSITION`` (1), or ``DRIVE_VELOCITY`` (2).
            target: target angle [rad] or target velocity [rad/s].
            stiffness: spring stiffness for position drive [N m/rad or N/m].
            damping: damping coefficient [N m s/rad or N s/m].
            max_force: maximum drive torque/force [N m or N].
        """
        if self.joint_store is None or joint_index < 0 or joint_index >= self._joint_count:
            return
        js = self.joint_store
        d = self.device

        def _write(name, val, dtype):
            col = js.column_of(name).numpy()
            col[joint_index] = val
            js.column_of(name).assign(wp.array(col, dtype=dtype, device=d))

        _write("drive_mode", mode, wp.int32)
        _write("drive_target", target, wp.float32)
        _write("drive_stiffness", stiffness, wp.float32)
        _write("drive_damping", damping, wp.float32)
        _write("drive_max_force", max_force, wp.float32)

    def _add_joint(
        self,
        joint_type: int,
        body_handle0: int,
        body_handle1: int,
        anchor_world: tuple[float, float, float],
        axis_world: tuple[float, float, float],
    ) -> int:
        """Internal: add a joint of given type."""
        if self.joint_store is None or self._joint_count >= self.joint_capacity:
            return -1

        ji = self._joint_count
        self._joint_count += 1

        js = self.joint_store
        bs = self.body_store
        d = self.device

        h2i = bs.handle_to_index.numpy()
        row0 = int(h2i[body_handle0])
        row1 = int(h2i[body_handle1])

        # Read body positions and orientations
        pos = bs.column_of("position").numpy()
        orient = bs.column_of("orientation").numpy()
        p0 = pos[row0]
        p1 = pos[row1]
        q0 = orient[row0]
        q1 = orient[row1]

        anchor = np.array(anchor_world, dtype=np.float32)
        axis = np.array(axis_world, dtype=np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        # Convert to body-local frames using pure numpy quaternion math.
        # Warp quaternion layout: (x, y, z, w).
        def _quat_conj(q):
            return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

        def _quat_mul(a, b):
            # Hamilton product, (x,y,z,w) layout
            return np.array(
                [
                    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
                ],
                dtype=np.float32,
            )

        def _quat_rotate(q, v):
            # Rotate vector v by quaternion q: q * (0,v) * conj(q)
            qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
            return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[:3]

        def _quat_inv_rotate(q, v):
            return _quat_rotate(_quat_conj(q), v)

        local_anchor0 = _quat_inv_rotate(q0, anchor - p0)
        local_anchor1 = _quat_inv_rotate(q1, anchor - p1)
        local_axis0 = _quat_inv_rotate(q0, axis)
        local_axis1 = _quat_inv_rotate(q1, axis)

        # Inverse initial relative orientation: (q0^-1 * q1)^-1
        q0_inv = _quat_conj(q0)
        rel = _quat_mul(q0_inv, q1)
        inv_rel_q = _quat_conj(rel)

        def _write_col(name, value, dtype):
            col = js.column_of(name).numpy()
            col[ji] = value
            js.column_of(name).assign(wp.array(col, dtype=dtype, device=d))

        _write_col("joint_type", joint_type, wp.int32)
        _write_col("body0", row0, wp.int32)
        _write_col("body1", row1, wp.int32)
        _write_col("local_anchor0", local_anchor0, wp.vec3)
        _write_col("local_anchor1", local_anchor1, wp.vec3)
        _write_col("local_axis0", local_axis0, wp.vec3)
        _write_col("local_axis1", local_axis1, wp.vec3)
        _write_col("inv_initial_orientation", inv_rel_q, wp.quat)

        # Update count
        cnt = js.count.numpy()
        cnt[0] = self._joint_count
        js.count.assign(wp.array(cnt, dtype=wp.int32, device=d))

        return ji

    # -- contact import -----------------------------------------------------

    def import_contacts(self, contacts):
        """Import rigid contacts from a Newton :class:`Contacts` object.

        Copies contact data into the internal :class:`DataStore`, resolves
        body indices via :attr:`shape_body`, and runs the warm-starter
        pipeline (key computation, sorting, impulse transfer).

        Args:
            contacts: a :class:`newton.sim.Contacts` instance filled by
                :meth:`CollisionPipeline.collide`.
        """
        d = self.device
        cap = self.contact_store.capacity

        wp.copy(self.contact_store.count, contacts.rigid_contact_count)

        cs = self.contact_store
        wp.launch(
            import_contacts_kernel,
            dim=cap,
            inputs=[
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.contact_store.count,
                self.shape_body,
                self.default_friction,
                cs.column_of("shape0"),
                cs.column_of("shape1"),
                cs.column_of("body0"),
                cs.column_of("body1"),
                cs.column_of("normal"),
                cs.column_of("offset0"),
                cs.column_of("offset1"),
                cs.column_of("margin0"),
                cs.column_of("margin1"),
                cs.column_of("friction"),
            ],
            device=d,
        )

        # Warm starting pipeline
        self.warm_starter.import_keys(
            cs.column_of("shape0"),
            cs.column_of("shape1"),
            self.contact_store.count,
            offset0=cs.column_of("offset0"),
        )
        self.warm_starter.sort()
        self.warm_starter.build_bundles()
        self.warm_starter.transfer_impulses(
            cs.column_of("accumulated_normal_impulse"),
            cs.column_of("accumulated_tangent_impulse1"),
            cs.column_of("accumulated_tangent_impulse2"),
        )

    # -- partitioning -------------------------------------------------------

    def _partition_contacts(self):
        """Build the element array and run graph coloring on bundles + joints."""
        d = self.device
        cs = self.contact_store
        ws = self.warm_starter

        # Build elements from bundles (one element per bundle)
        wp.launch(
            build_bundle_elements_kernel,
            dim=self.capacity_bundles,
            inputs=[
                ws.bundle_starts,
                ws.curr_indices,
                cs.column_of("body0"),
                cs.column_of("body1"),
                self._elements,
                ws.bundle_count,
            ],
            device=d,
        )

        # Append joint elements right after bundles (contiguous)
        if self._constraint_kernels is not None and self._joint_count > 0:
            ck = self._constraint_kernels
            wp.launch(
                ck.build_elements,
                dim=self.joint_capacity,
                inputs=[ck.joint_data, self._elements, ws.bundle_count, ck.joint_count],
                device=d,
            )

        # Total element count = bundles + joints (device-side, no host sync)
        if self._constraint_kernels is not None and self._joint_count > 0:
            wp.launch(
                add_int32_kernel,
                dim=1,
                inputs=[ws.bundle_count, self.joint_store.count, self._total_element_count],
                device=d,
            )
        else:
            wp.copy(self._total_element_count, ws.bundle_count)

        self.graph_coloring.color(
            elements=self._elements,
            element_count=self._total_element_count,
            node_count=self.body_store.count,
        )

    # -- integration --------------------------------------------------------

    def integrate_velocities(self, gravity: tuple[float, float, float], dt: float):
        """Apply gravity to dynamic body velocities."""
        d = self.device
        bs = self.body_store
        cap = bs.capacity
        wp.launch(
            integrate_velocities_kernel,
            dim=cap,
            inputs=[
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("inverse_mass"),
                bs.column_of("flags"),
                wp.vec3(*gravity),
                dt,
                bs.count,
            ],
            device=d,
        )

    def integrate_positions(self, dt: float):
        """Integrate positions and orientations using semi-implicit Euler."""
        d = self.device
        bs = self.body_store
        cap = bs.capacity
        wp.launch(
            integrate_positions_kernel,
            dim=cap,
            inputs=[
                bs.column_of("position"),
                bs.column_of("orientation"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("flags"),
                dt,
                bs.count,
            ],
            device=d,
        )

    def update_world_inertia(self):
        """Recompute world-frame inverse inertia and apply per-frame damping.

        Damping is applied here (once per frame) matching C# PhoenX
        ``UpdateInertiaKernel``, not per substep.
        """
        d = self.device
        bs = self.body_store
        cap = bs.capacity
        wp.launch(
            update_world_inertia_kernel,
            dim=cap,
            inputs=[
                bs.column_of("orientation"),
                bs.column_of("inverse_inertia_local"),
                bs.column_of("inverse_inertia_world"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("linear_damping"),
                bs.column_of("angular_damping"),
                bs.column_of("flags"),
                bs.count,
            ],
            device=d,
        )

    def export_impulses(self):
        """Snapshot solved contact impulses for next-frame warm starting."""
        cs = self.contact_store
        self.warm_starter.export_impulses(
            cs.column_of("accumulated_normal_impulse"),
            cs.column_of("accumulated_tangent_impulse1"),
            cs.column_of("accumulated_tangent_impulse2"),
            src_offset0=cs.column_of("offset0"),
        )

    # -- partitioned PGS solve helpers --------------------------------------

    def _launch_prepare(self, partition_slot: int, inv_dt: float, dim: int = 0):
        """Launch bundled contact prepare kernel for one partition slot."""
        ck = self._contact_kernels
        gc = self.graph_coloring
        ws = self.warm_starter
        wp.launch(
            ck.prepare_bundled,
            dim=dim if dim > 0 else self.capacity_bundles,
            inputs=[
                ck.contact_data,
                ck.body_data,
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                ws.bundle_starts,
                ws.bundle_count,
                ws.curr_indices,
                self._contact_count_per_body,
                inv_dt,
            ],
            device=self.device,
        )

    def _launch_solve(self, partition_slot: int, use_bias: int, dim: int = 0):
        """Launch bundled contact solve kernel for one partition slot."""
        ck = self._contact_kernels
        gc = self.graph_coloring
        ws = self.warm_starter
        wp.launch(
            ck.solve_bundled,
            dim=dim if dim > 0 else self.capacity_bundles,
            inputs=[
                ck.contact_data,
                ck.body_data,
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                ws.bundle_starts,
                ws.bundle_count,
                ws.curr_indices,
                use_bias,
            ],
            device=self.device,
        )

    # -- constraint launch helpers ------------------------------------------

    def _launch_prepare_constraints(self, partition_slot: int, inv_dt: float, dim: int = 0):
        """Launch constraint prepare kernel for one partition slot."""
        if self._constraint_kernels is None or self._joint_count == 0:
            return
        ck = self._constraint_kernels
        gc = self.graph_coloring
        ws = self.warm_starter
        wp.launch(
            ck.prepare,
            dim=dim if dim > 0 else self.joint_capacity,
            inputs=[
                ck.joint_data,
                ck.body_data,
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                ws.bundle_count,
                ck.joint_count,
                inv_dt,
            ],
            device=self.device,
        )

    def _launch_solve_constraints(self, partition_slot: int, use_bias: int, dim: int = 0):
        """Launch constraint solve kernel for one partition slot."""
        if self._constraint_kernels is None or self._joint_count == 0:
            return
        ck = self._constraint_kernels
        gc = self.graph_coloring
        ws = self.warm_starter
        wp.launch(
            ck.solve,
            dim=dim if dim > 0 else self.joint_capacity,
            inputs=[
                ck.joint_data,
                ck.body_data,
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                ws.bundle_count,
                ck.joint_count,
                use_bias,
            ],
            device=self.device,
        )

    # -- full step ----------------------------------------------------------

    def step(
        self,
        dt: float,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        num_iterations: int = 8,
        num_velocity_iterations: int = 0,
    ):
        """Run one substep with partitioned PGS contact solving.

        The partition loop uses a fixed upper bound
        (``max_colors + 1``) so that the entire method is free of
        GPU-to-CPU synchronisation and can be captured into a CUDA
        graph.

        Pipeline (per substep):

        1. ``integrate_velocities(gravity, dt)``
        2. ``_partition_contacts()``
        3. Count contacts per body (mass splitting)
        4. ``prepare_contacts_kernel`` (per partition slot)
        5. Position iterations (with bias)
        6. Velocity iterations (no bias)
        7. ``integrate_positions(dt)``

        Call :meth:`update_world_inertia` once per frame (before the
        substep loop) to recompute world-space inertia and apply
        per-frame damping, matching the C# PhoenX pipeline.

        Args:
            dt: time step [s].
            gravity: gravity vector [m/s^2].
            num_iterations: PGS position iterations (with Baumgarte bias).
            num_velocity_iterations: PGS velocity iterations (no bias).
        """
        inv_dt = 1.0 / dt if dt > 0.0 else 0.0
        d = self.device
        bs = self.body_store
        cs = self.contact_store
        is_xpbd = self.contact_mode == "xpbd"

        self.integrate_velocities(gravity, dt)

        if is_xpbd:
            # Store reference state and predict positions from velocities
            wp.launch(
                store_ref_state_kernel,
                dim=bs.capacity,
                inputs=[
                    bs.column_of("position"),
                    bs.column_of("orientation"),
                    self._ref_position,
                    self._ref_orientation,
                    bs.count,
                ],
                device=d,
            )
            wp.launch(
                predict_positions_kernel,
                dim=bs.capacity,
                inputs=[
                    bs.column_of("position"),
                    bs.column_of("orientation"),
                    bs.column_of("velocity"),
                    bs.column_of("angular_velocity"),
                    bs.column_of("flags"),
                    dt,
                    bs.count,
                ],
                device=d,
            )

        self._partition_contacts()

        # Mass splitting: count how many contacts reference each body
        wp.launch(
            clear_contact_count_kernel,
            dim=bs.capacity,
            inputs=[self._contact_count_per_body, bs.count],
            device=d,
        )
        wp.launch(
            count_contacts_per_body_kernel,
            dim=cs.capacity,
            inputs=[
                cs.column_of("body0"),
                cs.column_of("body1"),
                cs.count,
                self._contact_count_per_body,
            ],
            device=d,
        )

        max_slots = self.graph_coloring.max_colors + 1

        # Loop over all partition slots with full capacity — kernels
        # early-exit for out-of-range threads, keeping this free of
        # GPU-to-CPU sync so the entire method is graph-capturable.
        for p in range(max_slots):
            self._launch_prepare(p, inv_dt)
            self._launch_prepare_constraints(p, inv_dt)

        for _ in range(num_iterations):
            for p in range(max_slots):
                self._launch_solve(p, 1)
                self._launch_solve_constraints(p, 1)

        for _ in range(num_velocity_iterations):
            for p in range(max_slots):
                self._launch_solve(p, 0)
                self._launch_solve_constraints(p, 0)

        if is_xpbd:
            # Derive velocities from position change, skip integrate_positions
            # (positions are already at their final values after XPBD solving)
            wp.launch(
                derive_velocities_kernel,
                dim=bs.capacity,
                inputs=[
                    bs.column_of("position"),
                    bs.column_of("orientation"),
                    self._ref_position,
                    self._ref_orientation,
                    bs.column_of("velocity"),
                    bs.column_of("angular_velocity"),
                    bs.column_of("flags"),
                    inv_dt,
                    bs.count,
                ],
                device=d,
            )
        else:
            self.integrate_positions(dt)

    # -- external impulse application ---------------------------------------

    def apply_body_impulse(
        self,
        body_row: int,
        impulse_world: tuple[float, float, float],
        point_world: tuple[float, float, float],
        dt: float,
    ):
        """Apply an impulse to a body, modifying its linear and angular velocity.

        The impulse is applied at a world-space point, producing both linear
        and angular velocity changes::

            v += impulse * inv_mass
            w += inv_inertia_world * cross(r, impulse)

        where ``r = point_world - position``.

        Args:
            body_row: storage row index for the body.
            impulse_world: impulse vector in world frame [N*s].
            point_world: world-space application point [m].
            dt: time step (unused, kept for API symmetry).
        """
        bs = self.body_store
        inv_mass = bs.column_of("inverse_mass").numpy()[body_row]
        if inv_mass <= 0.0:
            return
        inv_inertia = bs.column_of("inverse_inertia_world").numpy()[body_row]
        pos = bs.column_of("position").numpy()[body_row]

        imp = np.array(impulse_world, dtype=np.float32)
        r = np.array(point_world, dtype=np.float32) - pos

        # Linear velocity change
        vel = bs.column_of("velocity").numpy()
        vel[body_row] += imp * inv_mass
        bs.column_of("velocity").assign(wp.array(vel, dtype=wp.vec3, device=self.device))

        # Angular velocity change: w += I_inv * (r x impulse)
        torque_impulse = np.cross(r, imp)
        ang_vel = bs.column_of("angular_velocity").numpy()
        ang_vel[body_row] += inv_inertia @ torque_impulse
        bs.column_of("angular_velocity").assign(wp.array(ang_vel, dtype=wp.vec3, device=self.device))
