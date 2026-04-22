# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels and ``@wp.func`` helpers for the Jitter solver.

Kept separate from :mod:`solver_jitter` so the driver class file stays
readable; this module is pure GPU code (no Python control flow). Symbols
are re-exported from :mod:`solver_jitter` so callers don't need to know
about the split.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.jitter.constraints.constraint_actuated_double_ball_socket import (
    actuated_double_ball_socket_iterate,
    actuated_double_ball_socket_prepare_for_iteration,
    actuated_double_ball_socket_world_error,
    actuated_double_ball_socket_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_angular_limit import (
    angular_limit_iterate,
    angular_limit_prepare_for_iteration,
    angular_limit_world_error,
    angular_limit_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_angular_motor import (
    angular_motor_iterate,
    angular_motor_prepare_for_iteration,
    angular_motor_world_error,
    angular_motor_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_ball_socket import (
    ball_socket_iterate,
    ball_socket_prepare_for_iteration,
    ball_socket_world_error,
    ball_socket_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_contact import (
    ContactViews,
    contact_iterate,
    contact_prepare_for_iteration,
    contact_world_error,
    contact_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET,
    CONSTRAINT_TYPE_ANGULAR_LIMIT,
    CONSTRAINT_TYPE_ANGULAR_MOTOR,
    CONSTRAINT_TYPE_BALL_SOCKET,
    CONSTRAINT_TYPE_CONTACT,
    CONSTRAINT_TYPE_D6,
    CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET,
    CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC,
    CONSTRAINT_TYPE_HINGE_ANGLE,
    CONSTRAINT_TYPE_HINGE_JOINT,
    CONSTRAINT_TYPE_LINEAR_LIMIT,
    CONSTRAINT_TYPE_LINEAR_MOTOR,
    CONSTRAINT_TYPE_PRISMATIC,
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
    constraint_get_type,
)
from newton._src.solvers.jitter.constraints.constraint_d6 import (
    d6_iterate,
    d6_prepare_for_iteration,
    d6_world_error,
    d6_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_double_ball_socket import (
    double_ball_socket_iterate,
    double_ball_socket_prepare_for_iteration,
    double_ball_socket_world_error,
    double_ball_socket_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_double_ball_socket_prismatic import (
    double_ball_socket_prismatic_iterate,
    double_ball_socket_prismatic_prepare_for_iteration,
    double_ball_socket_prismatic_world_error,
    double_ball_socket_prismatic_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_hinge_angle import (
    hinge_angle_iterate,
    hinge_angle_prepare_for_iteration,
    hinge_angle_world_error,
    hinge_angle_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_hinge_joint import (
    hinge_joint_iterate,
    hinge_joint_prepare_for_iteration,
    hinge_joint_world_error,
    hinge_joint_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_linear_limit import (
    linear_limit_iterate,
    linear_limit_prepare_for_iteration,
    linear_limit_world_error,
    linear_limit_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_linear_motor import (
    linear_motor_iterate,
    linear_motor_prepare_for_iteration,
    linear_motor_world_error,
    linear_motor_world_wrench,
)
from newton._src.solvers.jitter.constraints.constraint_prismatic import (
    prismatic_iterate,
    prismatic_prepare_for_iteration,
    prismatic_world_error,
    prismatic_world_wrench,
)
from newton._src.solvers.jitter.constraints.contact_container import ContactContainer
from newton._src.solvers.jitter.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)

__all__ = [
    "_STRAGGLER_BLOCK_DIM",
    "_constraint_gather_errors_kernel",
    "_constraint_gather_wrenches_kernel",
    "_constraint_iterate_fast_tail_kernel",
    "_constraint_iterate_kernel",
    "_constraint_prepare_fast_tail_kernel",
    "_constraint_prepare_for_iteration_kernel",
    "_constraint_relax_fast_tail_kernel",
    "_constraint_relax_kernel",
    "_constraints_to_elements_kernel",
    "_integrate_forces_kernel",
    "_integrate_velocities_kernel",
    "_rotation_quaternion",
    "_update_bodies_kernel",
    "pack_body_xforms_kernel",
]


# ---------------------------------------------------------------------------
# Unified constraint dispatch
# ---------------------------------------------------------------------------
#
# The solver no longer has per-type prepare/iterate kernels. Instead one
# pair of *type-agnostic* dispatcher kernels reads the constraint_type
# tag at the front of every column and routes to the correct ``wp.func``
# via an ``if/elif`` cascade. Each branch compiles to a tight inlined
# call (Warp inlines ``wp.func`` calls aggressively); the cascade adds
# one int compare per branch in the worst case which is far cheaper
# than the per-launch overhead of having one kernel per type.
#
# The build-time wiring is therefore:
#
#   * solver -> ``_constraint_prepare_for_iteration_kernel`` /
#               ``_constraint_iterate_kernel``
#   * dispatch kernel -> ``constraint_get_type(c, cid)`` -> per-type ``wp.func``
#
# Adding a new constraint type means: write its ``*_prepare_for_iteration``
# / ``*_iterate`` ``wp.func`` (same ``constraints, cid, bodies, idt``
# signature), add a ``CONSTRAINT_TYPE_FOO`` tag in
# :mod:`constraint_container`, stamp it in the type's init kernel, and
# add one ``elif`` branch to each dispatcher below. No change to
# :class:`World` or :class:`WorldBuilder` needed.


@wp.kernel(enable_backward=False)
def _constraint_prepare_for_iteration_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Dispatch the per-type ``prepare_for_iteration`` ``wp.func`` for
    each cid in the *current* colour of the CSR coloring layout.

    Reads its work list from the shared CSR built once per ``step()``
    by :meth:`IncrementalContactPartitioner.build_csr`:

    * ``cursor = color_cursor[0]`` -- countdown from ``num_colors[0]``
      to 0, decremented by this kernel as each colour completes.
    * ``c = num_colors[0] - cursor`` -- index of the colour being
      processed.
    * ``start = color_starts[c]``, ``end = color_starts[c + 1]`` --
      half-open range of ``element_ids_by_color`` belonging to
      colour ``c``.

    Per-thread ``tid`` picks up ``cid = element_ids_by_color[start + tid]``
    when ``tid < end - start``; threads beyond that early-out. The
    partitioner guarantees that no two cids in the same colour share
    a body, so the per-thread RMW of the body container is race-free
    regardless of which constraint type each thread happens to
    dispatch.

    The launch size is the constraint *capacity* (Python-side
    constant) so the kernel is legal inside ``wp.capture_while``;
    actual work per launch is bounded by the largest colour's
    element count (usually << capacity).

    After the work is done, thread 0 decrements ``color_cursor[0]``.
    This is race-free because every thread reads ``cursor`` once at
    the top into a per-thread register, and the only write happens
    from thread 0 after every thread has finished its element work;
    subsequent reads of ``color_cursor`` only occur in the next
    captured kernel launch (after the implicit kernel-boundary sync).

    Follows PhoenX's "contacts are constraints" pattern: the contact
    branch takes the additional
    :class:`~newton._src.solvers.jitter.constraints.contact_container.ContactContainer`
    and :class:`~newton._src.solvers.jitter.constraints.constraint_contact.ContactViews`
    arguments; joint-only branches ignore them. The two extra struct
    arguments are essentially free for non-contact dispatch because
    Warp passes structs by reference.
    """
    tid = wp.tid()

    # Snapshot the sweep cursor once. Every thread participates so the
    # implicit per-launch state is kept local to this grid; the only
    # writer of ``color_cursor`` is thread 0 below.
    cursor = color_cursor[0]
    n_colors = num_colors[0]
    c = n_colors - cursor

    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start

    if tid < count:
        cid = element_ids_by_color[start + tid]

        t = constraint_get_type(constraints, cid)
        if t == CONSTRAINT_TYPE_BALL_SOCKET:
            ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
            hinge_angle_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
            angular_motor_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_LINEAR_MOTOR:
            linear_motor_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_ANGULAR_LIMIT:
            angular_limit_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_LINEAR_LIMIT:
            linear_limit_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_HINGE_JOINT:
            hinge_joint_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET:
            double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC:
            double_ball_socket_prismatic_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
            actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_PRISMATIC:
            prismatic_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_D6:
            d6_prepare_for_iteration(constraints, cid, bodies, idt)
        elif t == CONSTRAINT_TYPE_CONTACT:
            contact_prepare_for_iteration(constraints, cid, bodies, idt, cc, contacts)

    # Advance the per-sweep cursor by one colour. Thread 0 only; see
    # the docstring for the race-freedom argument.
    if tid == 0:
        color_cursor[0] = cursor - 1


@wp.kernel(enable_backward=False)
def _constraint_iterate_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Dispatch the per-type ``iterate`` ``wp.func`` for each cid in the
    current CSR colour with positional bias ON (main solve pass). See
    :func:`_constraint_prepare_for_iteration_kernel` for the launch
    contract and the cursor-decrement semantics, and
    :func:`_constraint_relax_kernel` for the relaxation counterpart.
    """
    tid = wp.tid()

    cursor = color_cursor[0]
    n_colors = num_colors[0]
    c = n_colors - cursor

    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start

    if tid < count:
        cid = element_ids_by_color[start + tid]

        t = constraint_get_type(constraints, cid)
        if t == CONSTRAINT_TYPE_BALL_SOCKET:
            ball_socket_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
            hinge_angle_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
            angular_motor_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_LINEAR_MOTOR:
            linear_motor_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_ANGULAR_LIMIT:
            angular_limit_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_LINEAR_LIMIT:
            linear_limit_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_HINGE_JOINT:
            hinge_joint_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET:
            double_ball_socket_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC:
            double_ball_socket_prismatic_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
            actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_PRISMATIC:
            prismatic_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_D6:
            d6_iterate(constraints, cid, bodies, idt, True)
        elif t == CONSTRAINT_TYPE_CONTACT:
            contact_iterate(constraints, cid, bodies, idt, cc, contacts, True)

    if tid == 0:
        color_cursor[0] = cursor - 1


@wp.kernel(enable_backward=False)
def _constraint_relax_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Box2D v3 TGS-soft relaxation pass: dispatch each cid's iterate
    with positional bias OFF.

    Functionally identical to :func:`_constraint_iterate_kernel`
    except every sub-iterate is called with ``use_bias=False`` so the
    rigid-lock rows enforce ``Jv = 0`` without re-injecting
    position-error velocity. This is the "relax" sub-step of the
    Box2D v3 substep loop and is what prevents the soft-anchor
    impulse-leak artefact that forced the rigid-default in
    :data:`~newton._src.solvers.jitter.constraints.constraint_container.DEFAULT_HERTZ_LINEAR`.
    """
    tid = wp.tid()

    cursor = color_cursor[0]
    n_colors = num_colors[0]
    c = n_colors - cursor

    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start

    if tid < count:
        cid = element_ids_by_color[start + tid]

        t = constraint_get_type(constraints, cid)
        if t == CONSTRAINT_TYPE_BALL_SOCKET:
            ball_socket_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
            hinge_angle_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
            angular_motor_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_LINEAR_MOTOR:
            linear_motor_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_ANGULAR_LIMIT:
            angular_limit_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_LINEAR_LIMIT:
            linear_limit_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_HINGE_JOINT:
            hinge_joint_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET:
            double_ball_socket_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC:
            double_ball_socket_prismatic_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
            actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_PRISMATIC:
            prismatic_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_D6:
            d6_iterate(constraints, cid, bodies, idt, False)
        elif t == CONSTRAINT_TYPE_CONTACT:
            contact_iterate(constraints, cid, bodies, idt, cc, contacts, False)

    if tid == 0:
        color_cursor[0] = cursor - 1


# ---------------------------------------------------------------------------
# Fast-path dispatchers: actuated double-ball-socket joints + contact only
# ---------------------------------------------------------------------------
#
# The "full" dispatchers above branch through a dozen joint types. In
# practice the vast majority of Jitter scenes -- including everything
# built by :class:`WorldBuilder.add_joint` today -- route all of their
# joints through :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET`
# (revolute / prismatic / ball-socket with optional drives + limits) +
# contacts. For those scenes the other ten branches are dead code that
# still takes register budget, hurts instruction-cache locality, and
# slows per-thread dispatch because the final ``elif`` has to chase a
# chain of compares.
#
# The fast-path kernels below ship only the two branches we actually
# use. ``World.__init__`` takes an ``enable_all_constraints`` boolean
# that flips between the full and fast variants at record time -- the
# selected kernel is the one captured into the CUDA graph, so there's
# no runtime branch overhead either way.


# ---------------------------------------------------------------------------
# Fast-path unified single-block dispatchers
# ---------------------------------------------------------------------------
#
# The three ``_fast_tail`` kernels below replace the two-launch
# (per-colour multi-block + single-block tail) scheme previously used by
# the fast dispatcher. Each runs in a single thread block and walks the
# *entire* remaining colour range internally: outer ``while cursor > 0``
# loop with ``__syncthreads`` between colours so body-velocity writes
# propagate, and a block-stride ``while base < count`` loop inside each
# colour so colours larger than ``block_dim`` are still covered. One
# launch per sweep replaces ``num_colors`` launches; the cursor is
# zeroed on exit so the outer ``wp.capture_while`` terminates
# immediately.
#
# The full dispatchers (``_constraint_prepare_for_iteration_kernel``
# etc.) are still required for ``enable_all_constraints=True`` scenes
# that wire constraint types beyond
# :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` +
# :data:`CONSTRAINT_TYPE_CONTACT`; they follow the legacy per-colour
# multi-block pattern.


_STRAGGLER_BLOCK_DIM: int = 256


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.kernel(enable_backward=False)
def _constraint_iterate_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_iterations: wp.int32,
):
    """Single-block main-solve dispatcher: runs ``num_iterations``
    PGS sweeps in one kernel launch.

    Launched with ``dim=block_dim, 1 block``. Internally:

    * Outer loop runs ``num_iterations`` times (Python-side
      ``solver_iterations`` passed in as a kernel scalar). This
      replaces the Python ``for _ in range(iterations): wp.launch(...)``
      loop -- one launch per sweep becomes one launch for the entire
      solve.
    * Middle loop walks every colour (``0 .. num_colors``) with a
      ``__syncthreads`` between colours so body-velocity writes
      from colour ``c`` propagate to colour ``c+1``. The final
      sync of iteration ``i`` also guards the first colour of
      iteration ``i+1``, so no extra iteration-boundary sync is
      needed.
    * Inner block-stride loop covers colours larger than
      ``block_dim``; each thread processes ``ceil(count / block_dim)``
      cids.

    Prepare is kept in a separate kernel (see
    :func:`_constraint_prepare_fast_tail_kernel`) even though it
    could be fused -- an earlier version inlined the prepare phase
    here and the register-pressure bump pushed occupancy down
    enough to slow the iterate phase by ~15 %. Separate kernels
    give each one a tighter register budget and better occupancy.
    """
    tid = wp.tid()

    n_colors = num_colors[0]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = color_starts[c]
            end = color_starts[c + 1]
            count = end - start

            base = tid
            while base < count:
                cid = element_ids_by_color[start + base]
                t = constraint_get_type(constraints, cid)
                if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
                    actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, True)
                elif t == CONSTRAINT_TYPE_CONTACT:
                    contact_iterate(constraints, cid, bodies, idt, cc, contacts, True)
                base += _STRAGGLER_BLOCK_DIM

            _sync_threads()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraint_prepare_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Single-block prepare dispatcher: one sweep over all colours.

    Prepare is a once-per-substep pass (compute effective masses,
    velocity bias, apply warm-start impulse) so no outer iteration
    loop is needed -- see :func:`_constraint_iterate_fast_tail_kernel`
    for the block-stride + per-colour sync pattern.
    """
    tid = wp.tid()

    n_colors = num_colors[0]

    c = wp.int32(0)
    while c < n_colors:
        start = color_starts[c]
        end = color_starts[c + 1]
        count = end - start

        base = tid
        while base < count:
            cid = element_ids_by_color[start + base]
            t = constraint_get_type(constraints, cid)
            if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
                actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
            elif t == CONSTRAINT_TYPE_CONTACT:
                contact_prepare_for_iteration(constraints, cid, bodies, idt, cc, contacts)
            base += _STRAGGLER_BLOCK_DIM

        _sync_threads()
        c += 1


@wp.kernel(enable_backward=False)
def _constraint_relax_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_iterations: wp.int32,
):
    """Single-block relax dispatcher: runs ``num_iterations`` relax
    sweeps (use_bias=False) in one kernel launch.

    See :func:`_constraint_iterate_fast_tail_kernel` for the outer
    iteration / colour-sync pattern. Uses ``use_bias=False`` to match
    the Box2D v3 TGS-soft relax semantics -- no positional bias, no
    speculative separation, just ``Jv = 0``.
    """
    tid = wp.tid()

    n_colors = num_colors[0]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = color_starts[c]
            end = color_starts[c + 1]
            count = end - start

            base = tid
            while base < count:
                cid = element_ids_by_color[start + base]
                t = constraint_get_type(constraints, cid)
                if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
                    actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, False)
                elif t == CONSTRAINT_TYPE_CONTACT:
                    contact_iterate(constraints, cid, bodies, idt, cc, contacts, False)
                base += _STRAGGLER_BLOCK_DIM

            _sync_threads()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraints_to_elements_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
):
    """Project every active constraint into the partitioner's
    :class:`ElementInteractionData` view, regardless of type.

    Only the two body indices are needed by the graph colourer; the
    remaining slots are filled with ``-1``. This is the type-agnostic
    successor to the previous per-type element-projection kernels:
    because the constraint header (constraint_type, body1, body2) lives
    at fixed dword offsets across every schema, a single kernel can
    pull body1/body2 with :func:`constraint_get_body1` /
    :func:`constraint_get_body2` without dispatching on type.

    Static / kinematic bodies (``inverse_mass == 0``) are replaced with
    ``-1`` so the graph colourer ignores them. This mirrors PhoenX's
    ``AddIfNotKinematicOrStatic`` and is the reason e.g. N contacts
    against the same ground plane can share a single colour: the
    ground body is never surfaced as a dependency.

    Launched once per :meth:`World.step` so contact constraints (which
    arrive next) can rebuild their element rows from scratch each
    frame without coordinating with the joint constraints."""
    tid = wp.tid()
    if tid >= num_constraints[0]:
        return
    b1 = constraint_get_body1(constraints, tid)
    b2 = constraint_get_body2(constraints, tid)
    if b1 >= 0 and bodies.inverse_mass[b1] == 0.0:
        b1 = -1
    if b2 >= 0 and bodies.inverse_mass[b2] == 0.0:
        b2 = -1
    # Compact: non-negative IDs must come first so the adjacency loop
    # (which stops on the first -1) doesn't miss a dynamic body when
    # the static one happens to sit in slot 0.
    if b1 < 0 and b2 >= 0:
        b1 = b2
        b2 = -1
    elements[tid] = element_interaction_data_make(b1, b2, -1, -1, -1, -1, -1, -1)


@wp.kernel(enable_backward=False)
def _constraint_gather_wrenches_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    out: wp.array[wp.spatial_vector],
):
    """Dispatch the per-type ``world_wrench`` ``wp.func`` for every
    constraint and write the result into ``out[cid]``.

    ``out`` carries the *world-frame* wrench applied by the constraint
    on its ``body2``: ``spatial_top = force [N]``,
    ``spatial_bottom = torque [N·m]``. ``idt`` is ``1 / substep_dt`` so
    the warm-started impulse is reported as the average force the
    constraint exerted during the most recent substep.

    No partitioning here -- each thread writes a unique slot, no body
    state is mutated, so a flat one-thread-per-cid launch is enough.
    """
    cid = wp.tid()
    if cid >= num_constraints:
        return

    t = constraint_get_type(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    if t == CONSTRAINT_TYPE_BALL_SOCKET:
        force, torque = ball_socket_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
        force, torque = hinge_angle_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
        force, torque = angular_motor_world_wrench(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_LINEAR_MOTOR:
        force, torque = linear_motor_world_wrench(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_ANGULAR_LIMIT:
        force, torque = angular_limit_world_wrench(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_LINEAR_LIMIT:
        force, torque = linear_limit_world_wrench(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_HINGE_JOINT:
        force, torque = hinge_joint_world_wrench(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET:
        force, torque = double_ball_socket_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC:
        force, torque = double_ball_socket_prismatic_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
        force, torque = actuated_double_ball_socket_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_PRISMATIC:
        force, torque = prismatic_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_D6:
        force, torque = d6_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_CONTACT:
        force, torque = contact_world_wrench(constraints, cid, idt, cc, contacts)

    out[cid] = wp.spatial_vector(force, torque)


@wp.kernel(enable_backward=False)
def _constraint_gather_errors_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    # out
    out: wp.array[wp.spatial_vector],
):
    """Dispatch the per-type ``world_error`` ``wp.func`` for every
    constraint and write the result into ``out[cid]``.

    ``out`` carries the position-level constraint residual packed into
    a :class:`wp.spatial_vector`: ``spatial_top`` = linear residual
    [m], ``spatial_bottom`` = angular residual [rad, half-sin
    approximation where applicable]. Per-type layout is documented on
    each ``*_world_error_at`` ``wp.func``; inactive limit rows return
    zero and the contact branch returns zero at the cid level
    (per-slot residuals are surfaced separately via
    :func:`contact_per_contact_error_kernel`).

    Computed on the fly from current body pose + persisted per-type
    state (local anchors, rest quaternions, revolution tracker, etc.)
    -- no additional storage. One thread per cid, no partitioning
    required (pure read, no body mutation), which also makes this
    kernel safe to call before the first :meth:`World.step`."""
    cid = wp.tid()
    if cid >= num_constraints:
        return

    t = constraint_get_type(constraints, cid)
    zero = wp.spatial_vector(
        wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0)
    )
    err = zero
    if t == CONSTRAINT_TYPE_BALL_SOCKET:
        err = ball_socket_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
        err = hinge_angle_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
        err = angular_motor_world_error(constraints, cid)
    elif t == CONSTRAINT_TYPE_LINEAR_MOTOR:
        err = linear_motor_world_error(constraints, cid)
    elif t == CONSTRAINT_TYPE_ANGULAR_LIMIT:
        err = angular_limit_world_error(constraints, cid)
    elif t == CONSTRAINT_TYPE_LINEAR_LIMIT:
        err = linear_limit_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_HINGE_JOINT:
        err = hinge_joint_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET:
        err = double_ball_socket_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC:
        err = double_ball_socket_prismatic_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
        err = actuated_double_ball_socket_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_PRISMATIC:
        err = prismatic_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_D6:
        err = d6_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_CONTACT:
        err = contact_world_error(constraints, cid)

    out[cid] = err


@wp.func
def _rotation_quaternion(omega: wp.vec3f, dt: wp.float32) -> wp.quatf:
    """Build the rotation quaternion for ``omega * dt`` (axis-angle form).

    Mirrors Jitter2's ``MathHelper.RotationQuaternion``. The axis-angle
    form keeps unit norm by construction, which is significantly more
    stable across many sub-steps than the linearised
    ``q' = 0.5 * (omega, 0) * q`` derivative -- the linearised form
    grows the quaternion magnitude every step and relies on renormalising
    each frame to compensate.
    """
    omega_len = wp.length(omega)
    theta = omega_len * dt
    if theta < 1.0e-9:
        return wp.quatf(0.0, 0.0, 0.0, 1.0)
    half = theta * 0.5
    # axis * sin(half) = omega / |omega| * sin(half)
    s = wp.sin(half) / omega_len
    return wp.quatf(omega[0] * s, omega[1] * s, omega[2] * s, wp.cos(half))


@wp.kernel(enable_backward=False)
def _update_bodies_kernel(
    bodies: BodyContainer,
    gravity: wp.vec3f,
    substep_dt: wp.float32,
):
    """Mirrors Jitter2's ``RigidBody.Update`` (called once per *step*).

    For each dynamic body:
      * Apply per-body damping to ``velocity`` / ``angular_velocity``.
      * Build ``delta_velocity`` and ``delta_angular_velocity`` from the
        accumulated ``force`` / ``torque`` plus optional gravity, scaled
        by ``substep_dt``. The substep loop's per-substep
        ``_integrate_forces`` then just adds these cached deltas (Jitter
        splits the work this way so the per-substep path is a single
        vector add).
      * Zero ``force`` / ``torque`` so the next step starts clean.
      * Refresh ``inverse_inertia_world = R * inverse_inertia * R^T`` from
        the current orientation so the constraint solver's effective-mass
        terms see the rotated inertia.

    Static bodies are skipped (their inertia / mass are already zero by
    construction).
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return

    v = bodies.velocity[i] * bodies.linear_damping[i]
    w = bodies.angular_velocity[i] * bodies.angular_damping[i]
    bodies.velocity[i] = v
    bodies.angular_velocity[i] = w

    inv_mass = bodies.inverse_mass[i]
    inv_inertia_world = bodies.inverse_inertia_world[i]
    f = bodies.force[i]
    t = bodies.torque[i]

    dv = f * (inv_mass * substep_dt)
    dw = (inv_inertia_world * t) * substep_dt
    if bodies.affected_by_gravity[i] != 0:
        dv = dv + gravity * substep_dt
    bodies.delta_velocity[i] = dv
    bodies.delta_angular_velocity[i] = dw

    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)

    # InverseInertiaWorld = R * inverse_inertia * R^T, with R from the
    # current orientation. We use wp.quat_to_matrix so we don't depend on
    # any helper outside warp.
    r = wp.quat_to_matrix(bodies.orientation[i])
    bodies.inverse_inertia_world[i] = r * bodies.inverse_inertia[i] * wp.transpose(r)


@wp.kernel(enable_backward=False)
def _integrate_forces_kernel(bodies: BodyContainer):
    """Mirrors Jitter2's per-substep ``IntegrateForces``: just add the
    cached deltas built once per step in :func:`_update_bodies_kernel`.

    Static bodies are skipped. Kinematic bodies are skipped as well --
    Jitter's ``IntegrateForces`` only advances dynamic bodies' velocities;
    kinematic velocities are user-scripted and only feed
    ``IntegrateVelocities``.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    bodies.velocity[i] = bodies.velocity[i] + bodies.delta_velocity[i]
    bodies.angular_velocity[i] = bodies.angular_velocity[i] + bodies.delta_angular_velocity[i]


@wp.kernel(enable_backward=False)
def _integrate_velocities_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Mirrors Jitter2's IntegrateVelocities: advance position + orientation.

    Static bodies are skipped (kinematic bodies *do* advance, since their
    user-scripted velocities are meaningful; matches Jitter, which gates
    on ``MotionType != Static``).
    Orientation update uses the axis-angle rotation quaternion form (see
    :func:`_rotation_quaternion`) to stay numerically stable for large
    angular velocities.
    """
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_STATIC:
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt

    q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
    bodies.orientation[i] = wp.normalize(q_rot * bodies.orientation[i])


@wp.kernel(enable_backward=False)
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
