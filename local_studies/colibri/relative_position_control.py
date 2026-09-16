# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local FP32 translation accumulation control; canonical rotation is untouched.

Call install(PhoenXWorld) before construction. Buffers allocate in __init__,
reset in each world.step, and are device-written during CUDA graph replay.
Scope: maximal rigid bodies, no alternate position-level projection.
"""

import functools

import warp as wp


@wp.kernel(enable_backward=False)
def reset(position: wp.array[wp.vec3f], birth: wp.array[wp.vec3f], delta: wp.array[wp.vec3f]):
    i = wp.tid()
    birth[i] = position[i]
    delta[i] = wp.vec3f(0.0)


@wp.kernel(enable_backward=False)
def accumulate(
    velocity: wp.array[wp.vec3f],
    motion_type: wp.array[wp.int32],
    island_root: wp.array[wp.int32],
    delta: wp.array[wp.vec3f],
    dt: wp.float32,
    dynamic_type: wp.int32,
):
    i = wp.tid()
    if motion_type[i] == dynamic_type and island_root[i] < 0:
        delta[i] = delta[i] + velocity[i] * dt


@wp.kernel(enable_backward=False)
def publish(
    position: wp.array[wp.vec3f],
    birth: wp.array[wp.vec3f],
    delta: wp.array[wp.vec3f],
    motion_type: wp.array[wp.int32],
    island_root: wp.array[wp.int32],
    dynamic_type: wp.int32,
):
    i = wp.tid()
    if motion_type[i] == dynamic_type and island_root[i] < 0:
        position[i] = birth[i] + delta[i]


def install(world_class):
    """Wrap only local process methods; never rewrite canonical solver source."""
    from newton._src.solvers.phoenx.body import MOTION_DYNAMIC  # noqa: PLC0415

    assert not getattr(world_class, "_relative_position_control", False)
    original_init = world_class.__init__
    original_step = world_class.step
    original_integrate = world_class._integrate_positions

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        assert self._reduced_articulation is None, "Maximal-only diagnostic"
        assert getattr(self, "_maximal_tree_projector", None) is None or getattr(
            self, "_direct_tree_contacts", False
        ), "Independent position projection is incompatible with this diagnostic"
        self._relative_birth = wp.empty(self.num_bodies, dtype=wp.vec3f, device=self.device)
        self._relative_delta = wp.empty_like(self._relative_birth)

    @functools.wraps(original_step)
    def step(self, *args, **kwargs):
        if self.num_bodies:
            wp.launch(
                reset,
                dim=self.num_bodies,
                inputs=[self.bodies.position, self._relative_birth, self._relative_delta],
                device=self.device,
            )
        return original_step(self, *args, **kwargs)

    @functools.wraps(original_integrate)
    def integrate(self):
        if self.num_bodies == 0 or not self._has_maximal_dynamic_bodies:
            return original_integrate(self)
        wp.launch(
            accumulate,
            dim=self.num_bodies,
            inputs=[
                self.bodies.velocity,
                self.bodies.motion_type,
                self.bodies.island_root,
                self._relative_delta,
                wp.float32(self.substep_dt),
                MOTION_DYNAMIC,
            ],
            device=self.device,
        )
        # Original angular integration uses no position input. Its temporary
        # x += v*h is overwritten before subsequent geometry/contact consumers.
        original_integrate(self)
        wp.launch(
            publish,
            dim=self.num_bodies,
            inputs=[
                self.bodies.position,
                self._relative_birth,
                self._relative_delta,
                self.bodies.motion_type,
                self.bodies.island_root,
                MOTION_DYNAMIC,
            ],
            device=self.device,
        )

    world_class.__init__ = initialize
    world_class.step = step
    world_class._integrate_positions = integrate
    world_class._relative_position_control = True
