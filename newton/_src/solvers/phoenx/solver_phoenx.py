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
    ):
        self.device = wp.get_device(device)
        d = self.device

        self.body_store = HandleStore(RigidBodySchema, body_capacity, device=d)
        self.contact_store = DataStore(ContactPointSchema, contact_capacity, device=d)
        self.warm_starter = WarmStarter(contact_capacity, device=d)
        self.graph_coloring = GraphColoring(
            max_elements=contact_capacity,
            max_nodes=body_capacity,
            max_colors=max_colors,
            device=d,
        )

        self.shape_body = wp.full(shape_count, -1, dtype=wp.int32, device=d)
        self.shape_count = shape_count
        self.default_friction = default_friction

        self._elements = wp.zeros(
            (contact_capacity, MAX_BODIES_PER_ELEMENT), dtype=wp.int32, device=d
        )

        # Mass splitting: per-body contact count (Tonge et al. 2012)
        self._contact_count_per_body = wp.zeros(body_capacity, dtype=wp.int32, device=d)

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
        bs.column_of("inverse_inertia_local").assign(
            wp.array(inertia_col, dtype=wp.mat33, device=self.device)
        )

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
        """Build the element array and run graph coloring on current contacts."""
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

        self.graph_coloring.color(
            elements=self._elements,
            element_count=cs.count,
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
                bs.column_of("linear_damping"),
                bs.column_of("angular_damping"),
                bs.column_of("flags"),
                dt,
                bs.count,
            ],
            device=d,
        )

    def update_world_inertia(self):
        """Recompute world-frame inverse inertia from body orientations."""
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

    # -- full step ----------------------------------------------------------

    def step(
        self,
        dt: float,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        num_iterations: int = 8,
        num_velocity_iterations: int = 0,
    ):
        """Run one simulation step with partitioned PGS contact solving.

        The partition loop uses a fixed upper bound
        (``max_colors + 1``) so that the entire method is free of
        GPU-to-CPU synchronisation and can be captured into a CUDA
        graph.

        Pipeline:

        1. ``update_world_inertia()``
        2. ``integrate_velocities(gravity, dt)``
        3. ``_partition_contacts()``
        4. Count contacts per body (mass splitting)
        5. ``prepare_contacts_kernel`` (per partition slot)
        6. Position iterations (with bias)
        7. Velocity iterations (no bias)
        8. ``integrate_positions(dt)``

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

        self.update_world_inertia()
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

        for _ in range(num_iterations):
            for p in range(max_slots):
                self._launch_solve(p, 1)

        for _ in range(num_velocity_iterations):
            for p in range(max_slots):
                self._launch_solve(p, 0)

        self.integrate_positions(dt)
