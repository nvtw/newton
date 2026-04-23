# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Self-contained picking helper for Jitter examples.

Mirrors :class:`newton._src.viewer.picking.Picking`'s public surface
(``pick`` / ``update`` / ``release`` / ``is_picking``) but operates
directly on a :class:`World` (i.e. on :class:`BodyContainer`) instead
of on a :class:`newton.Model` / :class:`newton.State`. This lets the
Jitter sample apps wire the standard GL-viewer mouse handlers without
having to construct a Newton ``Model`` they will never simulate.

How it plugs in
---------------

The GL viewer keeps per-event callback lists on its underlying
``opengl`` renderer; the *viewer's* own handlers are appended first
during :meth:`ViewerGL.__init__` and silently no-op when
``viewer.picking is None`` (no model). We therefore
:func:`register_with_viewer_gl` *additional* callbacks that talk to a
:class:`JitterPicking` instance; this avoids touching ``viewer.picking``
or ``viewer._last_state`` at all.

How it picks
------------

* Bodies are abstracted to per-body axis-aligned half-extents in their
  *body* frame, so we get OBB-style picking after composing with each
  body's world transform. Sphere/capsule support is left as a follow-up
  -- the sample only needs boxes.
* A fixed-size ``wp.array`` device buffer carries all picking state
  (one int + a few vec3s). All operations are kernel launches so the
  whole flow stays graph-capture-friendly if the example wants to
  capture the per-frame pick application.
* Force model matches Newton's own picking kernel (PD spring on the
  picked point + clamp to multiple of g * mass). Output is *added* to
  :attr:`BodyContainer.force` / :attr:`BodyContainer.torque` so it
  composes with gravity and any user-applied forces.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.raycast import _spinlock_acquire, _spinlock_release
from newton._src.solvers.jitter.body import BodyContainer

__all__ = [
    "JitterPicking",
    "register_with_viewer_gl",
]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
#
# All state lives in length-1 device arrays so reads/writes don't need a
# host sync. The host -> device pieces (ray start/dir, stiffness, ...) are
# passed as plain kernel arguments because they only change per mouse event
# (not per simulation step) so the trip through ``wp.launch`` is fine.


@wp.kernel(enable_backward=False)
def _raycast_obb_kernel(
    bodies: BodyContainer,
    half_extents: wp.array[wp.vec3f],
    ray_start: wp.vec3f,
    ray_dir: wp.vec3f,
    # Outputs (length-1 arrays, atomically updated)
    out_dist: wp.array[wp.float32],
    out_body: wp.array[wp.int32],
    out_local_hit: wp.array[wp.vec3f],
    lock: wp.array[wp.int32],
):
    """Per-body OBB raycast.

    Each thread tests one body. A body whose ``half_extents`` has any
    component <= 0 is skipped (used to mark non-pickable bodies like the
    world anchor at index 0).

    On hit, the thread takes a spinlock and updates the shared min-dist
    output if its hit is closer. We use a spinlock instead of a CAS
    because we also need to atomically update the body index and local
    hit point alongside the distance.
    """
    bid = wp.tid()

    he = half_extents[bid]
    if he[0] <= 0.0 or he[1] <= 0.0 or he[2] <= 0.0:
        return

    pos = bodies.position[bid]
    rot = bodies.orientation[bid]

    # Transform the ray into the body's local frame: rotate by inverse
    # orientation and translate by negated position.
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, ray_start - pos)
    d_local = wp.quat_rotate(inv_rot, ray_dir)

    # Slab test against the AABB [-he, +he]. ``inv_d`` may overflow when
    # ``d_local[k]`` is zero, but that just makes the corresponding pair
    # of slab planes never bound this axis (one becomes +inf, the other
    # -inf), which is the right behaviour for parallel rays.
    t_min = wp.float32(-1.0e30)
    t_max = wp.float32(1.0e30)

    for k in range(3):
        if d_local[k] != 0.0:
            inv_d = 1.0 / d_local[k]
            t1 = (-he[k] - p_local[k]) * inv_d
            t2 = (he[k] - p_local[k]) * inv_d
            if t1 > t2:
                tmp = t1
                t1 = t2
                t2 = tmp
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
        else:
            # Ray parallel to the slab; only hits if origin is inside.
            if p_local[k] < -he[k] or p_local[k] > he[k]:
                return

    if t_max < t_min or t_max < 0.0:
        return

    # Use the entry point if the ray origin is outside, else the exit.
    t_hit = t_min
    if t_hit < 0.0:
        t_hit = t_max

    local_hit = p_local + d_local * t_hit

    # Spinlock-protected min-reduction across all hitting threads.
    # Reuses Newton's existing raycast spinlock helpers so we get the
    # same well-tested CAS + atomic_exch pattern.
    if t_hit < out_dist[0]:
        _spinlock_acquire(lock)
        old_min = wp.atomic_min(out_dist, 0, t_hit)
        if t_hit <= old_min:
            out_body[0] = bid
            out_local_hit[0] = local_hit
        _spinlock_release(lock)


@wp.kernel(enable_backward=False)
def _update_pick_target_kernel(
    ray_start: wp.vec3f,
    ray_dir: wp.vec3f,
    pick_dist: wp.array[wp.float32],
    out_target: wp.array[wp.vec3f],
):
    """Reproject the mouse ray to the picked point's depth.

    Matches Newton's behaviour: the target tracks the mouse cursor at
    constant distance from the camera, equal to the distance at which
    the body was originally hit. That feels right interactively because
    near objects move less than far ones for the same mouse delta."""
    out_target[0] = ray_start + ray_dir * pick_dist[0]


@wp.kernel(enable_backward=False)
def _apply_pick_force_kernel(
    bodies: BodyContainer,
    pick_body: wp.array[wp.int32],
    pick_local: wp.array[wp.vec3f],
    pick_target: wp.array[wp.vec3f],
    stiffness: wp.float32,
    damping: wp.float32,
    max_acc_g: wp.float32,
    inv_mass_floor: wp.float32,
):
    """PD spring-damper from the picked point to the mouse target.

    Mirrors :func:`newton._src.viewer.kernels.apply_picking_force_kernel`
    but reads/writes the Jitter ``BodyContainer`` directly and skips
    the kinematic-flag check (we mark the world anchor non-pickable via
    its zero half-extents instead).

    ``inv_mass_floor`` lets the caller cap the effective mass so the
    force clamp is meaningful for very-light or massless picked bodies
    (mirrors the ``effective_mass`` table in Newton's picking)."""
    bid = pick_body[0]
    if bid < 0:
        return

    pos = bodies.position[bid]
    rot = bodies.orientation[bid]

    # World-space pick attach point (the same body-local point we
    # latched at pick time, transformed by the body's *current* pose).
    world_attach = pos + wp.quat_rotate(rot, pick_local[0])

    target = pick_target[0]

    # Velocity of the attach point: linear COM velocity + omega x r.
    inv_m = bodies.inverse_mass[bid]
    v_com = bodies.velocity[bid]
    omega = bodies.angular_velocity[bid]
    r = world_attach - pos  # offset from COM to attach point in world frame
    v_attach = v_com + wp.cross(omega, r)

    # PD force. Match Newton's "force_multiplier = 10 + mass" by
    # converting back from inv_mass: a body with inv_m == 0 (static) is
    # filtered out above by the bid < 0 / pickability checks, so we can
    # treat 0 here as "1 kg surrogate" without harm.
    if inv_m > 0.0:
        mass = 1.0 / inv_m
    else:
        mass = 1.0
    force_multiplier = 10.0 + mass

    f = force_multiplier * (stiffness * (target - world_attach) - damping * v_attach)

    # Clamp |f| to ``max_acc_g * g * effective_mass`` so light objects
    # near stiff contacts don't fly off. ``inv_mass_floor`` lets the
    # caller pretend a chain has more mass than its picked link does.
    eff_inv_m = wp.min(inv_m, inv_mass_floor)
    if eff_inv_m > 0.0:
        eff_mass = 1.0 / eff_inv_m
    else:
        eff_mass = 1.0
    max_force = max_acc_g * 9.81 * eff_mass
    fmag = wp.length(f)
    if fmag > max_force:
        f = f * (max_force / fmag)

    torque = wp.cross(r, f)

    # Add to the Jitter accumulators (gravity / user code may have
    # already written here this substep).
    wp.atomic_add(bodies.force, bid, f)
    wp.atomic_add(bodies.torque, bid, torque)


# ---------------------------------------------------------------------------
# JitterPicking
# ---------------------------------------------------------------------------


class JitterPicking:
    """Holds the per-pick device state and exposes Newton-Picking-shaped
    methods (:meth:`pick`, :meth:`update`, :meth:`release`,
    :meth:`is_picking`).

    The example owns a single :class:`JitterPicking`, registers the
    mouse callbacks via :func:`register_with_viewer_gl`, and calls
    :meth:`apply_force` from inside its simulate loop right before each
    ``world.step(dt)``.

    Example::

        picking = JitterPicking(world, half_extents)
        register_with_viewer_gl(viewer, picking)

        # ... inside simulate() ...
        picking.apply_force()
        world.step(dt)
    """

    def __init__(
        self,
        world,
        half_extents: wp.array,
        *,
        stiffness: float = 50.0,
        damping: float = 5.0,
        max_acceleration: float = 5.0,
    ) -> None:
        """Create a picking helper bound to ``world``.

        Args:
            world: The :class:`World` whose :class:`BodyContainer` will
                be raycast against / forced.
            half_extents: ``wp.array[wp.vec3f]`` of length
                ``world.num_bodies``, giving each body's half-extents in
                its local frame. Bodies with any component <= 0 are
                non-pickable (used to suppress the world anchor at
                index 0).
            stiffness, damping: PD spring parameters.
            max_acceleration: Force clamp expressed as a multiple of g
                (``9.81 m/s^2``); same convention as Newton's picking.
        """
        self.world = world
        self.half_extents = half_extents
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.max_acceleration = float(max_acceleration)

        device = world.bodies.position.device
        self.device = device

        # Length-1 device buffers carry all the picking state.
        self._pick_body = wp.full(1, value=-1, dtype=wp.int32, device=device)
        self._pick_local = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._pick_target = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._pick_dist = wp.zeros(1, dtype=wp.float32, device=device)

        # Scratch for the raycast reduction.
        self._scratch_dist = wp.zeros(1, dtype=wp.float32, device=device)
        self._scratch_body = wp.full(1, value=-1, dtype=wp.int32, device=device)
        self._scratch_local = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._scratch_lock = wp.zeros(1, dtype=wp.int32, device=device)

        self._is_picking = False

    # --- Newton-shaped surface ----------------------------------------

    def is_picking(self) -> bool:
        return self._is_picking

    def release(self) -> None:
        """Drop the current pick. Safe to call when not picking."""
        self._pick_body.fill_(-1)
        self._is_picking = False

    def pick(self, ray_start, ray_dir) -> None:
        """Cast a world-space ray and latch onto the closest hit body.

        Mirrors the right-click handler of Newton's GL viewer: prime the
        scratch buffers, run the per-body raycast, then on hit copy the
        winner into the persistent pick state and seed
        ``pick_target = ray_start + ray_dir * dist``.
        """
        # Reset reduction scratch.
        self._scratch_dist.fill_(1.0e30)
        self._scratch_body.fill_(-1)
        self._scratch_local.zero_()
        self._scratch_lock.zero_()

        rs = wp.vec3f(float(ray_start[0]), float(ray_start[1]), float(ray_start[2]))
        rd = wp.vec3f(float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2]))

        wp.launch(
            _raycast_obb_kernel,
            dim=self.world.num_bodies,
            inputs=[
                self.world.bodies,
                self.half_extents,
                rs,
                rd,
                self._scratch_dist,
                self._scratch_body,
                self._scratch_local,
                self._scratch_lock,
            ],
            device=self.device,
        )

        # Need a sync to know whether we actually hit something. This
        # only runs on the right-click event (not in the per-frame hot
        # path), so the sync is cheap.
        wp.synchronize_device(self.device)
        body_idx = int(self._scratch_body.numpy()[0])
        if body_idx < 0:
            return

        dist = float(self._scratch_dist.numpy()[0])

        self._pick_body.assign([body_idx])
        self._pick_local.assign([self._scratch_local.numpy()[0]])
        self._pick_dist.assign([dist])
        # Seed target at the current mouse-ray depth so the first frame
        # of dragging doesn't yank the body.
        self._pick_target.assign(
            [
                wp.vec3f(
                    float(ray_start[0] + ray_dir[0] * dist),
                    float(ray_start[1] + ray_dir[1] * dist),
                    float(ray_start[2] + ray_dir[2] * dist),
                ),
            ]
        )
        self._is_picking = True

    def update(self, ray_start, ray_dir) -> None:
        """Re-project the mouse ray to the latched depth.

        Called from the mouse-drag callback. Pure kernel launch; no
        sync, so this is cheap to call every drag event."""
        if not self._is_picking:
            return
        rs = wp.vec3f(float(ray_start[0]), float(ray_start[1]), float(ray_start[2]))
        rd = wp.vec3f(float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2]))
        wp.launch(
            _update_pick_target_kernel,
            dim=1,
            inputs=[rs, rd, self._pick_dist, self._pick_target],
            device=self.device,
        )

    # --- Per-step force application -----------------------------------

    def apply_force(self, inv_mass_floor: float = 1.0) -> None:
        """Add the current picking force to the body force/torque
        accumulators.

        Always launches the kernel (even when not picking) so the call
        is graph-capture safe -- the kernel itself short-circuits on
        ``pick_body[0] < 0``.

        Args:
            inv_mass_floor: Lower bound on the inverse mass used for
                the force clamp. ``1.0`` means "treat anything lighter
                than 1 kg as 1 kg for clamping purposes", which keeps
                light bodies controllable without runaway force on
                stiff-contact divergence."""
        wp.launch(
            _apply_pick_force_kernel,
            dim=1,
            inputs=[
                self.world.bodies,
                self._pick_body,
                self._pick_local,
                self._pick_target,
                wp.float32(self.stiffness),
                wp.float32(self.damping),
                wp.float32(self.max_acceleration),
                wp.float32(inv_mass_floor),
            ],
            device=self.device,
        )


# ---------------------------------------------------------------------------
# GL viewer wiring
# ---------------------------------------------------------------------------


def register_with_viewer_gl(viewer, picking: JitterPicking) -> None:
    """Hook ``picking`` into a :class:`ViewerGL`'s mouse callbacks.

    Right mouse button picks; right-drag updates the target; release
    drops the pick. We do not touch ``viewer.picking`` -- the viewer's
    own handler will see ``viewer.picking is None`` and silently no-op,
    after which our handler runs and does the real work.

    No-op for non-GL viewers (file / null / rerun / viser). They have
    no interactive surface to wire up.
    """
    renderer = getattr(viewer, "renderer", None)
    register_press = getattr(renderer, "register_mouse_press", None) if renderer else None
    if register_press is None:
        return  # Not a GL viewer, nothing to wire.

    try:
        import pyglet
    except ImportError:
        return

    right = pyglet.window.mouse.RIGHT

    def on_press(x, y, button, modifiers):
        if button != right:
            return
        fb_x, fb_y = viewer._to_framebuffer_coords(x, y)
        ray_start, ray_dir = viewer.camera.get_world_ray(fb_x, fb_y)
        picking.pick(ray_start, ray_dir)

    def on_release(x, y, button, modifiers):
        if button == right:
            picking.release()

    def on_drag(x, y, dx, dy, buttons, modifiers):
        if not (buttons & right):
            return
        fb_x, fb_y = viewer._to_framebuffer_coords(x, y)
        ray_start, ray_dir = viewer.camera.get_world_ray(fb_x, fb_y)
        picking.update(ray_start, ray_dir)

    renderer.register_mouse_press(on_press)
    renderer.register_mouse_release(on_release)
    renderer.register_mouse_drag(on_drag)
