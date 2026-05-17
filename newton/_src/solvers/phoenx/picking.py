# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Self-contained picking helper for :class:`PhoenXWorld` examples.

Per-body OBB raycast, then a PD spring on the picked point clamped to a multiple
of ``g * mass``. State lives in fixed-size device buffers (graph-capture-friendly).
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.raycast import (
    _ray_intersect_triangle_mt,
    _spinlock_acquire,
    _spinlock_release,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "Picking",
    "register_with_viewer_gl",
]


@wp.kernel(enable_backward=False)
def _raycast_cloth_tri_kernel(
    particles: ParticleContainer,
    tri_indices: wp.array2d[wp.int32],
    num_tris: wp.int32,
    ray_start: wp.vec3f,
    ray_dir: wp.vec3f,
    out_dist: wp.array[wp.float32],
    out_tri: wp.array[wp.int32],
    out_bary: wp.array[wp.vec3f],
    out_body: wp.array[wp.int32],
    lock: wp.array[wp.int32],
):
    """Per-cloth-triangle Möller-Trumbore raycast against the current
    particle positions. Writes the closest hit's ``(t, tri_idx,
    barycentric)`` if it beats ``out_dist[0]``.

    ``out_bary`` stores ``(alpha, beta, gamma)`` with ``alpha`` paired
    to ``tri_indices[t, 0]``, ``beta`` to ``[t, 1]``, ``gamma`` to
    ``[t, 2]``; the three sum to 1 by construction.
    """
    t = wp.tid()
    if t >= num_tris:
        return
    pa = tri_indices[t, 0]
    pb = tri_indices[t, 1]
    pc = tri_indices[t, 2]
    a = particles.position[pa]
    b = particles.position[pb]
    c = particles.position[pc]

    # Möller-Trumbore returns ``(t_hit, geometric_normal)``. The MT
    # algorithm computes ``u, v`` internally; we re-derive the
    # barycentric weights from the hit point so we don't have to fork
    # the shared helper.
    t_hit, _n = _ray_intersect_triangle_mt(ray_start, ray_dir, a, b, c)
    if t_hit < 0.0:
        return
    if t_hit >= out_dist[0]:
        return

    # Hit point + barycentric weights. Use the most-stable axis for
    # the 2D back-projection (skip the smallest normal component).
    hit = ray_start + ray_dir * t_hit
    e1 = b - a
    e2 = c - a
    n = wp.cross(e1, e2)
    n2 = wp.dot(n, n)
    if n2 < wp.float32(1.0e-30):
        return
    inv_n2 = wp.float32(1.0) / n2
    rp = hit - a
    # ``beta = ((rp x e2) . n) / n^2``, ``gamma = ((e1 x rp) . n) / n^2``.
    beta = wp.dot(wp.cross(rp, e2), n) * inv_n2
    gamma = wp.dot(wp.cross(e1, rp), n) * inv_n2
    alpha = wp.float32(1.0) - beta - gamma

    _spinlock_acquire(lock)
    old_min = wp.atomic_min(out_dist, 0, t_hit)
    if t_hit <= old_min:
        out_tri[0] = t
        out_bary[0] = wp.vec3f(alpha, beta, gamma)
        out_body[0] = wp.int32(-1)
    _spinlock_release(lock)


@wp.kernel(enable_backward=False)
def _apply_pick_force_cloth_kernel(
    particles: ParticleContainer,
    tri_indices: wp.array2d[wp.int32],
    pick_tri: wp.array[wp.int32],
    pick_bary: wp.array[wp.vec3f],
    pick_target: wp.array[wp.vec3f],
    stiffness: wp.float32,
    damping: wp.float32,
    max_acc_g: wp.float32,
    dt: wp.float32,
):
    """PD spring-damper from the picked cloth point to the mouse
    target, applied as a velocity impulse to the three triangle
    vertices.

    Per-frame impulse instead of a ``particles.force`` accumulator:
    ``ParticleContainer`` has no force field (unlike ``BodyContainer``)
    so the impulse is written directly into ``particles.velocity``
    once per frame. The XPBD substep loop then advects the new
    velocity through the cloth constraints.

    The total force is split equally between the three triangle
    vertices (the user's "divide by 3" proposal): each vertex
    receives ``F/3`` scaled by its own ``inv_mass`` and ``dt``.
    Pinned particles (``inv_mass == 0``) absorb their share with
    no motion -- matches the cloth's pinning semantics.
    """
    tri = pick_tri[0]
    if tri < 0:
        return

    pa = tri_indices[tri, 0]
    pb = tri_indices[tri, 1]
    pc = tri_indices[tri, 2]
    bary = pick_bary[0]

    xa = particles.position[pa]
    xb = particles.position[pb]
    xc = particles.position[pc]
    va = particles.velocity[pa]
    vb = particles.velocity[pb]
    vc = particles.velocity[pc]

    # World-space hit point + its rate-of-change, both barycentric-
    # interpolated. Damping uses the velocity at the hit point.
    world_attach = bary[0] * xa + bary[1] * xb + bary[2] * xc
    v_attach = bary[0] * va + bary[1] * vb + bary[2] * vc

    target = pick_target[0]
    f = stiffness * (target - world_attach) - damping * v_attach

    # Clamp |f| against an acceleration cap. Reference mass is the
    # triangle's total (sum of vertex masses), using ``inv_mass``
    # reciprocals; pinned vertices contribute infinity-mass and the
    # cap effectively becomes ``g * (sum of non-pinned masses)``.
    inv_ma = particles.inverse_mass[pa]
    inv_mb = particles.inverse_mass[pb]
    inv_mc = particles.inverse_mass[pc]
    mass_a = wp.float32(0.0)
    mass_b = wp.float32(0.0)
    mass_c = wp.float32(0.0)
    if inv_ma > wp.float32(0.0):
        mass_a = wp.float32(1.0) / inv_ma
    if inv_mb > wp.float32(0.0):
        mass_b = wp.float32(1.0) / inv_mb
    if inv_mc > wp.float32(0.0):
        mass_c = wp.float32(1.0) / inv_mc
    eff_mass = mass_a + mass_b + mass_c
    if eff_mass < wp.float32(1.0e-6):
        # All three vertices pinned -- no impulse to apply.
        return
    max_force = max_acc_g * wp.float32(9.81) * eff_mass
    fmag = wp.length(f)
    if fmag > max_force:
        f = f * (max_force / fmag)

    # Equal split (per the user's proposal): each vertex receives
    # F/3. Per-vertex velocity impulse: ``dv = (F/3) * inv_mass * dt``;
    # pinned vertices (``inv_mass == 0``) silently skip via the
    # multiply, no extra branching needed.
    f_third = f * wp.float32(1.0 / 3.0)
    particles.velocity[pa] = va + (f_third * inv_ma) * dt
    particles.velocity[pb] = vb + (f_third * inv_mb) * dt
    particles.velocity[pc] = vc + (f_third * inv_mc) * dt


@wp.kernel(enable_backward=False)
def _raycast_obb_kernel(
    bodies: BodyContainer,
    half_extents: wp.array[wp.vec3f],
    ray_start: wp.vec3f,
    ray_dir: wp.vec3f,
    out_dist: wp.array[wp.float32],
    out_body: wp.array[wp.int32],
    out_local_hit: wp.array[wp.vec3f],
    out_tri: wp.array[wp.int32],
    lock: wp.array[wp.int32],
):
    """Per-body OBB raycast. Bodies with non-positive half_extents are skipped
    (marker for non-pickable bodies like the world anchor).

    Shares ``out_dist`` + ``lock`` with :func:`_raycast_cloth_tri_kernel`
    so the two raycasts produce a single closest hit across both
    pickable kinds. On commit the rigid path invalidates
    ``out_tri[0]`` to signal "rigid won"; the cloth path mirrors this
    for ``out_body[0]``.
    """
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
            out_tri[0] = wp.int32(-1)
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

    Supports two pickable target kinds:

    * Rigid bodies via a per-body OBB raycast (``half_extents``).
    * Cloth triangles via per-triangle Möller-Trumbore raycast against
      current particle positions (pass ``model`` + ``particles`` to
      enable; both default to ``None`` for backwards compatibility
      with rigid-only callers).

    ``pick()`` runs both raycasts and latches the closest hit. The
    apply-force kernels for the two kinds are completely independent
    (each gates on its own state ``< 0``), so the picking object
    handles whichever target the user grabbed without per-call
    plumbing in the host loop.

    Example::

        picking = Picking(world, half_extents, model=model, particles=world.particles)
        register_with_viewer_gl(viewer, picking)
        # inside simulate():
        picking.apply_force(dt=frame_dt)
        world.step(dt=frame_dt)
    """

    def __init__(
        self,
        world,
        half_extents: wp.array,
        *,
        stiffness: float = 50.0,
        damping: float = 5.0,
        max_acceleration: float = 5.0,
        model=None,
        particles: ParticleContainer | None = None,
    ) -> None:
        """Bind picking to ``world``. ``half_extents`` is per-body local-frame
        half-extents (vec3f, length ``world.num_bodies``); non-positive components
        mark the body non-pickable. ``max_acceleration`` is a multiple of g.

        Cloth picking is enabled when both ``model`` (with
        ``tri_count > 0``) and ``particles`` are passed; otherwise the
        cloth raycast / force-application kernels are skipped and the
        helper degrades to rigid-only picking.
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

        # Cloth picking state. ``_pick_tri[0] < 0`` means "not picking
        # cloth"; the apply-cloth-force kernel short-circuits then.
        self._particles = particles
        self._cloth_tri_indices = None
        self._cloth_num_tris = 0
        if model is not None and particles is not None and int(getattr(model, "tri_count", 0)) > 0:
            self._cloth_tri_indices = model.tri_indices
            self._cloth_num_tris = int(model.tri_count)
        self._pick_tri = wp.full(1, value=-1, dtype=wp.int32, device=device)
        self._pick_bary = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._scratch_tri = wp.full(1, value=-1, dtype=wp.int32, device=device)
        self._scratch_bary = wp.zeros(1, dtype=wp.vec3f, device=device)

        self._is_picking = False

    # --- Newton-shaped surface ----------------------------------------

    def is_picking(self) -> bool:
        return self._is_picking

    def release(self) -> None:
        """Drop the current pick. Safe to call when not picking."""
        self._pick_body.fill_(-1)
        self._pick_tri.fill_(-1)
        self._is_picking = False

    def pick(self, ray_start, ray_dir) -> None:
        """Cast a world-space ray and latch onto the closest hit.

        Runs the rigid OBB raycast and (if cloth picking is enabled)
        the cloth-triangle raycast against a shared ``out_dist[0]``
        accumulator, so the winning target -- rigid body or cloth
        triangle -- is whichever lies closest along the ray. The
        latching ``_pick_body[0]`` and ``_pick_tri[0]`` slots are
        mutually exclusive: at most one is non-``-1`` after a single
        :meth:`pick` call.
        """
        self._scratch_dist.fill_(1.0e30)
        self._scratch_body.fill_(-1)
        self._scratch_local.zero_()
        self._scratch_lock.zero_()
        self._scratch_tri.fill_(-1)
        self._scratch_bary.zero_()

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
                self._scratch_tri,
                self._scratch_lock,
            ],
            device=self.device,
        )

        if self._cloth_tri_indices is not None and self._cloth_num_tris > 0:
            wp.launch(
                _raycast_cloth_tri_kernel,
                dim=self._cloth_num_tris,
                inputs=[
                    self._particles,
                    self._cloth_tri_indices,
                    wp.int32(self._cloth_num_tris),
                    rs,
                    rd,
                    self._scratch_dist,
                    self._scratch_tri,
                    self._scratch_bary,
                    self._scratch_body,
                    self._scratch_lock,
                ],
                device=self.device,
            )

        # Sync only fires on the right-click event, not the hot path.
        wp.synchronize_device(self.device)
        body_idx = int(self._scratch_body.numpy()[0])
        tri_idx = int(self._scratch_tri.numpy()[0])
        # By construction, at most one of (body_idx, tri_idx) is
        # ``>= 0`` after the two raycasts: each kernel's commit
        # invalidates the other slot when it wins the shared
        # ``out_dist`` min.
        if body_idx < 0 and tri_idx < 0:
            return

        dist = float(self._scratch_dist.numpy()[0])
        target_seed = wp.vec3f(
            float(ray_start[0] + ray_dir[0] * dist),
            float(ray_start[1] + ray_dir[1] * dist),
            float(ray_start[2] + ray_dir[2] * dist),
        )

        if body_idx >= 0:
            self._pick_body.assign([body_idx])
            self._pick_local.assign([self._scratch_local.numpy()[0]])
            self._pick_tri.fill_(-1)
        else:
            self._pick_tri.assign([tri_idx])
            self._pick_bary.assign([self._scratch_bary.numpy()[0]])
            self._pick_body.fill_(-1)

        self._pick_dist.assign([dist])
        # Seed target at current mouse-ray depth so the first drag doesn't yank.
        self._pick_target.assign([target_seed])
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

    def apply_force(self, inv_mass_floor: float = 1.0, dt: float = 1.0 / 60.0) -> None:
        """Add the picking force/impulse to the picked target.

        Rigid path: PD spring on the picked body's local hit point is
        atomic-added to ``bodies.force`` and ``bodies.torque``. The
        solver's per-substep ``apply_external_forces`` kernel consumes
        these and the per-step force clear zeroes them.

        Cloth path: PD spring on the picked-triangle hit point is
        applied as a per-frame velocity impulse to the three triangle
        vertices (split equally per the user's "divide by 3"
        proposal). ``ParticleContainer`` has no force accumulator so
        the impulse goes straight into ``particles.velocity`` here.
        ``dt`` should match the host ``frame_dt`` so the impulse
        magnitude matches a continuous force.

        Both kernels gate on their own state ``< 0`` and are safe to
        launch unconditionally inside a captured graph.
        """
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
        if self._cloth_tri_indices is not None and self._cloth_num_tris > 0:
            wp.launch(
                _apply_pick_force_cloth_kernel,
                dim=1,
                inputs=[
                    self._particles,
                    self._cloth_tri_indices,
                    self._pick_tri,
                    self._pick_bary,
                    self._pick_target,
                    wp.float32(self.stiffness),
                    wp.float32(self.damping),
                    wp.float32(self.max_acceleration),
                    wp.float32(dt),
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
