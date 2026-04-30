# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constraint-side ``read_state`` / ``write_state`` wrappers.

When mass splitting is active, every constraint kernel that today
reads ``bodies.velocity[b]`` directly must instead route through
this wrapper so it picks up the right per-partition copy. The
wrapper also returns the ``inv_factor`` (= partition count) the
constraint must scale its impulse by to stay momentum-conserving.

Mirrors ``ConstraintHelper::ReadState`` /
``ConstraintHelper::WriteState`` in
``PhoenX/CudaKernels/Constraints/ConstraintHelper.cuh:153-233``.

## Static-body fallback

For bodies that don't appear in the interaction graph
(``rigid_body_index >= highest_index_in_use[0]`` or no entry was
registered),
:func:`~newton._src.solvers.phoenx.mass_splitting.interaction_graph.graph_get_rigid_state_index`
returns ``state_index = -1, inv_factor = 0``. ``read_state`` then
synthesises a :class:`~.state.TinyRigidState` from the body store
directly, and ``write_state`` is a no-op. Constraints carrying a
static body should treat ``inv_factor == 0`` as "this body is
infinitely massive" and skip the impulse on it (same convention
as PhoenX's ``VALIDATION_ASSERT(x1.invFactor != 0 ||
x2.invFactor != 0)`` in the C# constraint kernels).

## Wiring (later, not in this port)

A constraint kernel that today does::

    v1 = bodies.velocity[b1]
    w1 = bodies.angular_velocity[b1]
    ...  # solve, mutate v1 / w1 in registers
    bodies.velocity[b1] = v1
    bodies.angular_velocity[b1] = w1

becomes::

    state1, inv_factor1, id1 = read_state(graph, cid, b1, ...)
    ...  # solve, scale impulse by 1/inv_factor1, mutate state1
    write_state(graph, id1, state1)

with the SoA body fields supplied as fallback args so the wrapper
can synthesise a state for static bodies in one place.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphData,
    graph_get_rigid_state_index,
    graph_get_state,
    graph_set_state,
)
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_STATIC_BODY,
    TinyRigidState,
    tiny_rigid_state_set_access_mode,
)

__all__ = [
    "read_state",
    "write_state",
]


_ACCESS_MODE_STATIC_BODY_C = wp.constant(wp.int32(ACCESS_MODE_STATIC_BODY))


@wp.func
def read_state(
    graph: InteractionGraphData,
    constraint_index: wp.int32,
    rigid_body_index: wp.int32,
    body_position: wp.vec3f,
    body_orientation: wp.quatf,
    body_velocity: wp.vec3f,
    body_angular_velocity: wp.vec3f,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Fetch the velocity-state copy for ``(constraint, body)``.

    Returns ``(state, inv_factor, state_index)``:

    * ``state`` is the :class:`~.state.TinyRigidState` to mutate.
      For non-static bodies it's read from
      ``graph.tiny_states[state_index]`` and synced to the
      requested ``new_access_mode``. For static bodies it's
      synthesised from the body-store args with ``access_mode =
      ACCESS_MODE_STATIC_BODY``.
    * ``inv_factor`` is the count of partitions the body
      participates in. For static bodies it's ``0`` and the caller
      MUST treat the body as infinitely massive (skip impulse).
    * ``state_index`` is the slot in ``graph.tiny_states`` for
      :func:`write_state` -- ``-1`` means static and write_state
      becomes a no-op.

    Mirrors C# ``ReadState`` (ConstraintHelper.cuh:153-169, plus the
    overload at 187-212). The C# ``AccessMode`` regime switch is
    plumbed through ``new_access_mode``; pass
    :data:`~.state.ACCESS_MODE_VELOCITY_LEVEL` for an iterate sweep
    or :data:`~.state.ACCESS_MODE_POSITION_LEVEL` for a
    position-level prepare. Note (matching the C# TODO at line
    166-167): the access-mode change applies only to the returned
    *copy*; the underlying ``tiny_states`` slot remains in whatever
    mode it was in until :func:`write_state` puts it back.
    """
    state_index, inv_factor = graph_get_rigid_state_index(
        graph,
        constraint_index,
        rigid_body_index,
    )
    if state_index < wp.int32(0):
        # Static / unregistered body: synthesise the state from the
        # body store, mark static, return inv_factor=0. Position /
        # orientation pass through; velocity / angular velocity are
        # the body's *current* values (zero for true static bodies,
        # finite for kinematic ones the user wants to expose to
        # constraints but not include in mass splitting).
        state = TinyRigidState()
        state.position = body_position
        state.orientation = body_orientation
        state.velocity = body_velocity
        state.angular_velocity = body_angular_velocity
        state.access_mode = _ACCESS_MODE_STATIC_BODY_C
        return state, wp.int32(0), wp.int32(-1)
    state = graph_get_state(graph, state_index)
    state = tiny_rigid_state_set_access_mode(
        state,
        new_access_mode,
        body_position,
        body_orientation,
        inv_dt,
    )
    return state, inv_factor, state_index


@wp.func
def write_state(
    graph: InteractionGraphData,
    state_index: wp.int32,
    state: TinyRigidState,
):
    """Write a (possibly mutated) state back into ``tiny_states``.

    Mirrors C# ``WriteState`` (ConstraintHelper.cuh:214-233). When
    ``state_index < 0`` (static / unregistered body) the call is a
    no-op -- :func:`read_state` set the index to -1 and the constraint
    kernel was already supposed to skip the impulse on that side.
    """
    graph_set_state(graph, state_index, state)
