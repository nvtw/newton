# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Self-contained picking helper for :class:`PhoenXWorld` examples.

Per-body OBB raycast, then a PD spring on the picked point clamped to a multiple
of ``g * mass``. State lives in fixed-size device buffers (graph-capture-friendly).
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.raycast import _spinlock_acquire, _spinlock_release
from newton._src.solvers.phoenx.body import BodyContainer

__all__ = [
    "Picking",
    "register_with_viewer_gl",
]


@wp.kernel(enable_backward=False)
def _raycast_obb_kernel(
    bodies: BodyContainer,
    half_extents: wp.array[wp.vec3f],
    ray_start: wp.vec3f,
    ray_dir: wp.vec3f,
    out_dist: wp.array[wp.float32],
    out_body: wp.array[wp.int32],
    out_local_hit: wp.array[wp.vec3f],
    lock: wp.array[wp.int32],
):
    """Per-body OBB raycast. Bodies with non-positive half_extents are skipped
    (marker for non-pickable bodies like the world anchor)."""
    bid = wp.tid()

    he = half_extents[bid]
    if he[0] <= 0.0 or he[1] <= 0.0 or he[2] <= 0.0:
        return

    pos = bodies.position[bid]
    rot = bodies.orientation[bid]

    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, ray_start - pos)
    d_local = wp.quat_rotate(inv_rot, ray_dir)

    # Slab test against the AABB [-he, +he].
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

    # Entry point if origin is outside, else exit.
    t_hit = t_min
    if t_hit < 0.0:
        t_hit = t_max

    local_hit = p_local + d_local * t_hit

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
    """Reproject the mouse ray to the picked point's depth (constant from camera)."""
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
    ``inv_mass_floor`` caps the effective mass so the force clamp stays meaningful
    for very-light bodies."""
    bid = pick_body[0]
    if bid < 0:
        return

    pos = bodies.position[bid]
    rot = bodies.orientation[bid]

    world_attach = pos + wp.quat_rotate(rot, pick_local[0])

    target = pick_target[0]

    inv_m = bodies.inverse_mass[bid]
    v_com = bodies.velocity[bid]
    omega = bodies.angular_velocity[bid]
    r = world_attach - pos
    v_attach = v_com + wp.cross(omega, r)

    if inv_m > 0.0:
        mass = 1.0 / inv_m
    else:
        mass = 1.0
    force_multiplier = 10.0 + mass

    f = force_multiplier * (stiffness * (target - world_attach) - damping * v_attach)

    # Clamp |f| to max_acc_g * g * effective_mass.
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

    wp.atomic_add(bodies.force, bid, f)
    wp.atomic_add(bodies.torque, bid, torque)


class Picking:
    """Per-pick device state with Newton-Picking-shaped methods.

    Example::

        picking = Picking(world, half_extents)
        register_with_viewer_gl(viewer, picking)
        # inside simulate():
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
        """Bind picking to ``world``. ``half_extents`` is per-body local-frame
        half-extents (vec3f, length ``world.num_bodies``); non-positive components
        mark the body non-pickable. ``max_acceleration`` is a multiple of g."""
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
        """Cast a world-space ray and latch onto the closest hit body."""
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

        # Sync only fires on the right-click event, not the hot path.
        wp.synchronize_device(self.device)
        body_idx = int(self._scratch_body.numpy()[0])
        if body_idx < 0:
            return

        dist = float(self._scratch_dist.numpy()[0])

        self._pick_body.assign([body_idx])
        self._pick_local.assign([self._scratch_local.numpy()[0]])
        self._pick_dist.assign([dist])
        # Seed target at current mouse-ray depth so the first drag doesn't yank.
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
        """Re-project the mouse ray to the latched depth (no sync)."""
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

    def apply_force(self, inv_mass_floor: float = 1.0) -> None:
        """Add the picking force to body force/torque accumulators. Always launches
        (graph-capture safe); kernel short-circuits when not picking.
        ``inv_mass_floor=1.0`` treats sub-1-kg bodies as 1 kg for the clamp."""
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


def register_with_viewer_gl(viewer, picking: Picking) -> None:
    """Hook ``picking`` into a :class:`ViewerGL`'s mouse callbacks.
    Right-click picks, right-drag updates, release drops. No-op for non-GL viewers."""
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
