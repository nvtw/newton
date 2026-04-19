# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Body Chain
#
# Ten unit cubes arranged in a zig-zag along +x and connected with ball
# joints at *diagonal* corners. Cube j's centre is placed at
#   (2j+1)*h, +h if j even else -h, 0
# so consecutive cubes share exactly one corner: the joint between cube
# k-1 and cube k sits on the centerline at (2k*h, 0, 0), which is the
# +x/-y (or +x/+y) corner of cube k-1 and the opposite -x/+y (or -x/-y)
# corner of cube k. Cube 0's bottom-left-back corner (-h, -h, 0) is
# anchored to a static "world" body at the origin so the chain hangs in
# mid-air. Gravity (-y) pulls everything down and the joints keep the
# zig-zag together.
#
# Constructed via :class:`WorldBuilder` -- the recommended high-level
# API. The builder packs descriptors into the SoA body container and
# the column-major constraint container, then hands an assembled
# :class:`World` back ready to step.
#
# Run:  python -m newton._src.solvers.jitter.example_body_chain
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel
from newton._src.solvers.jitter.world_builder import WorldBuilder

NUM_CUBES = 10
HALF_EXTENT = 0.5  # unit cube -> half-extent 0.5 on each axis
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body
NUM_BALL_SOCKETS = NUM_CUBES  # 1 world->cube0 + 9 cube_i->cube_(i+1)

# Identity body-frame inverse inertia. A unit cube of mass 1 actually has
# I = m*s^2/6 * I_3 = 1/6 * I_3 so I^-1 = 6 * I_3, but identity keeps
# the demo numbers round and is fine for an unstressed visual test.
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # ---- Build the Jitter World via WorldBuilder ------------------
        b = WorldBuilder()
        world_body = b.world_body  # body 0, auto-created static anchor

        # Cube j sits at x = (2j+1)*h on +x and alternates y = +h (j even)
        # / -h (j odd), so consecutive cubes meet at exactly one diagonal
        # corner on the y=0 line.
        cube_ids: list[int] = []
        for j in range(NUM_CUBES):
            sign = 1.0 if j % 2 == 0 else -1.0
            cube_ids.append(
                b.add_dynamic_body(
                    position=((2 * j + 1) * HALF_EXTENT, sign * HALF_EXTENT, 0.0),
                    inverse_mass=1.0,
                    inverse_inertia=_INV_INERTIA,
                )
            )

        # Joint k anchors:
        #   * k = 0 : world (origin) <-> cube 0's (-h, -h, 0) corner at (0,0,0)
        #   * k>=1 : cube k-1 <-> cube k at (2*k*h, 0, 0) -- their shared
        #            corner on the centerline.
        for k in range(NUM_BALL_SOCKETS):
            body_a = world_body if k == 0 else cube_ids[k - 1]
            body_b = cube_ids[k]
            anchor = (0.0, 0.0, 0.0) if k == 0 else (2.0 * k * HALF_EXTENT, 0.0, 0.0)
            b.add_ball_socket(body_a, body_b, anchor)

        # substeps=1 here because we drive substepping ourselves from
        # simulate() (matches the basic-pendulum pattern).
        self.world = b.finalize(
            substeps=1,
            solver_iterations=8,
            device=self.device,
        )

        # ---- Rendering scratch ---------------------------------------
        # Reused per frame so we don't allocate inside render().
        self._xforms = wp.zeros(NUM_BODIES, dtype=wp.transform, device=self.device)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.world.step(self.sim_dt)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            pack_body_xforms_kernel,
            dim=NUM_BODIES,
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )

        self.viewer.begin_frame(self.sim_time)
        # Render the dynamic cubes only (skip the static world body at idx 0).
        self.viewer.log_shapes(
            "/world/cubes",
            newton.GeoType.BOX,
            (HALF_EXTENT, HALF_EXTENT, HALF_EXTENT),
            self._xforms[1:],
        )
        self.viewer.end_frame()

    def test_final(self):
        # After the run the chain should have swung from the world anchor
        # but no body should have fallen below y = -10 (a generous lower
        # bound that catches outright explosions / NaNs).
        positions = self.world.bodies.position.numpy()
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), f"body {i} produced non-finite position"
            assert positions[i, 1] > -10.0, f"body {i} fell unreasonably far ({positions[i, 1]})"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
