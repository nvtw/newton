# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""``PositionPass`` -- lightweight owner of the snapshot/sync buffers.

Pairs the :func:`snapshot_pose_kernel` /
:func:`sync_position_to_velocity_kernel` primitives with the small
amount of host-side state they need (the pre-pass pose buffers,
allocated lazily). This is what callers usually want -- a single
object that knows how to fence a position-level constraint pass.

Two ergonomic shapes:

* **Explicit -- ``snapshot()`` then ``sync_to_velocity()``.** Use
  when the caller is running its own iterate loop and just wants
  the bookend kernels.
* **All-in-one -- ``run(...)``.** Pass an iterate callable + an
  iteration count; ``PositionPass`` runs the snapshot, calls the
  iterate the right number of times, then syncs. Cheaper to read at
  the call site when the iterate function is well-encapsulated
  (typical XPBD setup).

Either way the orchestrator allocates the pre-pass buffers on first
``snapshot()``; nothing is allocated when the pass is never used.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from newton._src.solvers.phoenx.position_level.snapshot import (
    snapshot_pose_kernel,
    sync_position_to_velocity_kernel,
)

__all__ = [
    "PositionPass",
]


class PositionPass:
    """Owns the pre-pass pose buffers + drives the snapshot / sync kernels.

    One :class:`PositionPass` per body store. Sized at construction
    so the buffers can be allocated up-front (avoiding any
    per-substep allocation that would invalidate a captured graph).

    Attributes:
        num_bodies: Body count -- buffer length.
        device: Warp device the buffers live on.
        position_pre_pass: ``wp.array[wp.vec3f]`` of length
            ``num_bodies`` -- start-of-pass position. Allocated
            lazily on first ``snapshot()``; ``None`` until then so
            scenes that never call into the pass pay zero memory.
        orientation_pre_pass: ``wp.array[wp.quatf]`` of length
            ``num_bodies`` -- start-of-pass orientation, same
            lifetime as ``position_pre_pass``.
    """

    def __init__(
        self,
        num_bodies: int,
        device: wp.context.Devicelike = None,
    ) -> None:
        if num_bodies <= 0:
            raise ValueError(f"num_bodies must be > 0 (got {num_bodies})")
        self.num_bodies = int(num_bodies)
        self.device = device
        self.position_pre_pass: wp.array | None = None
        self.orientation_pre_pass: wp.array | None = None

    # ------------------------------------------------------------------
    # Lifetime.
    # ------------------------------------------------------------------

    def _ensure_buffers(self) -> None:
        """Allocate ``position_pre_pass`` / ``orientation_pre_pass``
        on first use.

        Lazy so a :class:`PhoenXWorld` that constructs
        :class:`PositionPass` up-front-but-never-uses-it (e.g. a
        scene where the user later toggles ``position_iterations``
        on) doesn't pay the buffer cost. Once allocated the buffers
        stick around for the lifetime of the pass -- the captured
        graph holds device-pointer references, so freeing them
        mid-simulation invalidates the graph.
        """
        if self.position_pre_pass is None:
            self.position_pre_pass = wp.zeros(self.num_bodies, dtype=wp.vec3f, device=self.device)
        if self.orientation_pre_pass is None:
            self.orientation_pre_pass = wp.zeros(self.num_bodies, dtype=wp.quatf, device=self.device)

    # ------------------------------------------------------------------
    # Per-substep API.
    # ------------------------------------------------------------------

    def snapshot(
        self,
        body_position: wp.array,
        body_orientation: wp.array,
    ) -> None:
        """Capture the current pose into the pre-pass buffers.

        Run **once** at the start of the position-level pass. The
        caller's iterate function then mutates ``body_position`` /
        ``body_orientation`` directly; :meth:`sync_to_velocity`
        recovers velocity from the delta against this snapshot.
        """
        self._ensure_buffers()
        wp.launch(
            kernel=snapshot_pose_kernel,
            dim=self.num_bodies,
            inputs=[
                body_position,
                body_orientation,
                self.position_pre_pass,
                self.orientation_pre_pass,
            ],
            device=self.device,
        )

    def sync_to_velocity(
        self,
        body_position: wp.array,
        body_orientation: wp.array,
        body_velocity: wp.array,
        body_angular_velocity: wp.array,
        inv_dt: float,
    ) -> None:
        """Recover velocity / angular velocity from the pose delta.

        Run **once** at the end of the position-level pass.
        Overwrites ``body_velocity`` / ``body_angular_velocity``
        with the XPBD finite-difference values
        ``(position - position_pre_pass) * inv_dt`` and the
        log-map of the orientation delta. See
        :func:`sync_position_to_velocity_kernel` for the math.

        Raises:
            RuntimeError: if :meth:`snapshot` hasn't been called
                yet (the pre-pass buffers don't exist).
        """
        if self.position_pre_pass is None or self.orientation_pre_pass is None:
            raise RuntimeError(
                "PositionPass.snapshot(...) must be called before sync_to_velocity(...) -- pre-pass buffers don't exist"
            )
        wp.launch(
            kernel=sync_position_to_velocity_kernel,
            dim=self.num_bodies,
            inputs=[
                body_position,
                body_orientation,
                self.position_pre_pass,
                self.orientation_pre_pass,
                body_velocity,
                body_angular_velocity,
                float(inv_dt),
            ],
            device=self.device,
        )

    def run(
        self,
        body_position: wp.array,
        body_orientation: wp.array,
        body_velocity: wp.array,
        body_angular_velocity: wp.array,
        iterate: Callable[[int], None],
        num_iterations: int,
        inv_dt: float,
    ) -> None:
        """Run a full position-level pass: snapshot, iterate, sync.

        Args:
            body_position, body_orientation,
            body_velocity, body_angular_velocity: SoA body store
                arrays; ``body_position`` / ``body_orientation`` are
                read **and** written across the iterate callable;
                ``body_velocity`` / ``body_angular_velocity`` are
                only written by :meth:`sync_to_velocity`.
            iterate: ``Callable[[int], None]``. Called
                ``num_iterations`` times with the iteration index
                (``0..num_iterations-1``). The callable is expected
                to mutate ``body_position`` / ``body_orientation``
                in place via XPBD-style projections; this
                orchestrator does not pass them in -- the callable
                closes over them.
            num_iterations: How many position-level iterations to
                run. ``0`` is a no-op (still snapshots + syncs, but
                with no projection in between -- recovered
                velocities equal the input velocities modulo
                float32 noise).
            inv_dt: ``1 / substep_dt``; the XPBD finite difference
                rate.

        Convenience wrapper for the common case. Equivalent to::

            self.snapshot(body_position, body_orientation)
            for i in range(num_iterations):
                iterate(i)
            self.sync_to_velocity(body_position, body_orientation, body_velocity, body_angular_velocity, inv_dt)
        """
        self.snapshot(body_position, body_orientation)
        for i in range(int(num_iterations)):
            iterate(i)
        self.sync_to_velocity(
            body_position,
            body_orientation,
            body_velocity,
            body_angular_velocity,
            inv_dt,
        )
