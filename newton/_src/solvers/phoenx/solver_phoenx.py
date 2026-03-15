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
    JOINT_FIXED,
    JOINT_REVOLUTE,
    JointSchema,
    build_joint_elements_kernel,
    prepare_constraints_kernel,
    solve_constraints_kernel,
)
from .data_base import DataStore, HandleStore
from .kernels import (
    build_elements_kernel,
    clear_contact_count_kernel,
    count_contacts_per_body_kernel,
    import_contacts_kernel,
    integrate_positions_kernel,
    integrate_velocities_kernel,
    prepare_contacts_kernel,
    solve_contacts_kernel,
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
        max_colors: int = 16,
        joint_capacity: int = 0,
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

        # Joint storage
        self.joint_capacity = joint_capacity
        if joint_capacity > 0:
            self.joint_store = DataStore(JointSchema, joint_capacity, device=d)
        else:
            self.joint_store = None
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

        # Convert to body-local frames using quaternion inverse rotation
        from scipy.spatial.transform import Rotation

        r0 = Rotation.from_quat([q0[0], q0[1], q0[2], q0[3]])
        r1 = Rotation.from_quat([q1[0], q1[1], q1[2], q1[3]])
        local_anchor0 = r0.inv().apply(anchor - p0).astype(np.float32)
        local_anchor1 = r1.inv().apply(anchor - p1).astype(np.float32)
        local_axis0 = r0.inv().apply(axis).astype(np.float32)
        local_axis1 = r1.inv().apply(axis).astype(np.float32)

        # Inverse initial relative orientation: (q0^-1 * q1)^-1
        q0_inv = r0.inv()
        rel = q0_inv * r1
        inv_rel = rel.inv()
        inv_rel_q = inv_rel.as_quat().astype(np.float32)  # [x,y,z,w]

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
        )
        self.warm_starter.sort()
        self.warm_starter.transfer_impulses(
            cs.column_of("accumulated_normal_impulse"),
            cs.column_of("accumulated_tangent_impulse1"),
            cs.column_of("accumulated_tangent_impulse2"),
        )

    # -- partitioning -------------------------------------------------------

    def _partition_contacts(self):
        """Build the element array and run graph coloring on contacts + joints."""
        d = self.device
        cs = self.contact_store
        cap = cs.capacity

        wp.launch(
            build_elements_kernel,
            dim=cap,
            inputs=[
                cs.column_of("body0"),
                cs.column_of("body1"),
                self._elements,
                cs.count,
            ],
            device=d,
        )

        # Append joint elements right after contacts (contiguous)
        if self.joint_store is not None and self._joint_count > 0:
            js = self.joint_store
            wp.launch(
                build_joint_elements_kernel,
                dim=js.capacity,
                inputs=[
                    js.column_of("body0"),
                    js.column_of("body1"),
                    self._elements,
                    js.count,
                    cs.count,
                ],
                device=d,
            )

        # Total element count = contacts + joints (device-side)
        self._total_element_count = wp.zeros(1, dtype=wp.int32, device=d)
        if self.joint_store is not None and self._joint_count > 0:
            nc = cs.count.numpy()[0]
            nj = self.joint_store.count.numpy()[0]
            total = nc + nj
            self._total_element_count.assign(wp.array([total], dtype=wp.int32, device=d))
        else:
            wp.copy(self._total_element_count, cs.count)

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
        )

    # -- partitioned PGS solve helpers --------------------------------------

    def _launch_prepare(self, partition_slot: int, inv_dt: float):
        """Launch :func:`prepare_contacts_kernel` for one partition slot."""
        d = self.device
        cs = self.contact_store
        bs = self.body_store
        gc = self.graph_coloring

        wp.launch(
            prepare_contacts_kernel,
            dim=cs.capacity,
            inputs=[
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                cs.column_of("normal"),
                cs.column_of("offset0"),
                cs.column_of("offset1"),
                cs.column_of("body0"),
                cs.column_of("body1"),
                cs.column_of("accumulated_normal_impulse"),
                cs.column_of("accumulated_tangent_impulse1"),
                cs.column_of("accumulated_tangent_impulse2"),
                cs.column_of("tangent1"),
                cs.column_of("rel_pos_world0"),
                cs.column_of("rel_pos_world1"),
                cs.column_of("effective_mass_n"),
                cs.column_of("effective_mass_t1"),
                cs.column_of("effective_mass_t2"),
                cs.column_of("bias"),
                bs.column_of("position"),
                bs.column_of("orientation"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("inverse_mass"),
                bs.column_of("inverse_inertia_world"),
                bs.column_of("flags"),
                self._contact_count_per_body,
                inv_dt,
            ],
            device=d,
        )

    def _launch_solve(self, partition_slot: int, use_bias: int):
        """Launch :func:`solve_contacts_kernel` for one partition slot."""
        d = self.device
        cs = self.contact_store
        bs = self.body_store
        gc = self.graph_coloring

        wp.launch(
            solve_contacts_kernel,
            dim=cs.capacity,
            inputs=[
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                cs.column_of("normal"),
                cs.column_of("tangent1"),
                cs.column_of("body0"),
                cs.column_of("body1"),
                cs.column_of("accumulated_normal_impulse"),
                cs.column_of("accumulated_tangent_impulse1"),
                cs.column_of("accumulated_tangent_impulse2"),
                cs.column_of("rel_pos_world0"),
                cs.column_of("rel_pos_world1"),
                cs.column_of("effective_mass_n"),
                cs.column_of("effective_mass_t1"),
                cs.column_of("effective_mass_t2"),
                cs.column_of("bias"),
                cs.column_of("friction"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("inverse_mass"),
                bs.column_of("inverse_inertia_world"),
                bs.column_of("flags"),
                use_bias,
            ],
            device=d,
        )

    # -- constraint launch helpers ------------------------------------------

    def _launch_prepare_constraints(self, partition_slot: int):
        """Launch :func:`prepare_constraints_kernel` for one partition slot."""
        if self.joint_store is None or self._joint_count == 0:
            return
        d = self.device
        js = self.joint_store
        bs = self.body_store
        gc = self.graph_coloring
        # Use actual contact count (cached from partition step)
        nc = self._cached_contact_count

        wp.launch(
            prepare_constraints_kernel,
            dim=js.capacity,
            inputs=[
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                nc,
                js.column_of("joint_type"),
                js.column_of("body0"),
                js.column_of("body1"),
                js.column_of("local_anchor0"),
                js.column_of("local_anchor1"),
                js.column_of("local_axis0"),
                js.column_of("local_axis1"),
                js.column_of("inv_initial_orientation"),
                js.column_of("point_lambda"),
                js.column_of("hinge_lambda_x"),
                js.column_of("hinge_lambda_y"),
                js.column_of("angle_lambda"),
                js.column_of("point_eff_mass"),
                js.column_of("hinge_b2xa1"),
                js.column_of("hinge_c2xa1"),
                js.column_of("hinge_eff_mass_00"),
                js.column_of("hinge_eff_mass_01"),
                js.column_of("hinge_eff_mass_10"),
                js.column_of("hinge_eff_mass_11"),
                js.column_of("angle_eff_mass"),
                js.column_of("angle_axis"),
                js.column_of("rw0"),
                js.column_of("rw1"),
                js.column_of("angle_min"),
                js.column_of("angle_max"),
                js.column_of("angle_current"),
                bs.column_of("position"),
                bs.column_of("orientation"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("inverse_mass"),
                bs.column_of("inverse_inertia_world"),
                bs.column_of("flags"),
                js.count,
            ],
            device=d,
        )

    def _launch_solve_constraints(self, partition_slot: int, use_bias: int):
        """Launch :func:`solve_constraints_kernel` for one partition slot."""
        if self.joint_store is None or self._joint_count == 0:
            return
        d = self.device
        js = self.joint_store
        bs = self.body_store
        gc = self.graph_coloring
        nc = self._cached_contact_count

        wp.launch(
            solve_constraints_kernel,
            dim=js.capacity,
            inputs=[
                gc.partition_data,
                gc.partition_ends,
                partition_slot,
                nc,
                js.column_of("joint_type"),
                js.column_of("body0"),
                js.column_of("body1"),
                js.column_of("point_lambda"),
                js.column_of("hinge_lambda_x"),
                js.column_of("hinge_lambda_y"),
                js.column_of("angle_lambda"),
                js.column_of("point_eff_mass"),
                js.column_of("hinge_b2xa1"),
                js.column_of("hinge_c2xa1"),
                js.column_of("hinge_eff_mass_00"),
                js.column_of("hinge_eff_mass_01"),
                js.column_of("hinge_eff_mass_10"),
                js.column_of("hinge_eff_mass_11"),
                js.column_of("angle_eff_mass"),
                js.column_of("angle_axis"),
                js.column_of("rw0"),
                js.column_of("rw1"),
                js.column_of("angle_min"),
                js.column_of("angle_max"),
                js.column_of("angle_current"),
                bs.column_of("position"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                bs.column_of("inverse_mass"),
                bs.column_of("inverse_inertia_world"),
                bs.column_of("flags"),
                js.count,
                use_bias,
            ],
            device=d,
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

        # Cache contact count for constraint kernels (host sync once)
        if self.joint_store is not None and self._joint_count > 0:
            self._cached_contact_count = int(cs.count.numpy()[0])
        else:
            self._cached_contact_count = 0

        self.integrate_velocities(gravity, dt)

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

        for p in range(max_slots):
            self._launch_prepare(p, inv_dt)
            self._launch_prepare_constraints(p)

        for _ in range(num_iterations):
            for p in range(max_slots):
                self._launch_solve(p, 1)
                self._launch_solve_constraints(p, 1)

        for _ in range(num_velocity_iterations):
            for p in range(max_slots):
                self._launch_solve(p, 0)
                self._launch_solve_constraints(p, 0)

        self.integrate_positions(dt)
