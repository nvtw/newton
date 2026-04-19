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
# Run:  python -m newton._src.solvers.jitter.example_body_chain
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.body import MOTION_DYNAMIC, RigidBodyData
from newton._src.solvers.jitter.constraints import BallSocketData, ball_socket_initialize
from newton._src.solvers.jitter.solver_jitter import World, pack_body_xforms_kernel

# Layout: 10 dynamic cubes + 1 static world body at index 0.
NUM_CUBES = 10
HALF_EXTENT = 0.5  # unit cube -> half-extent 0.5 on each axis
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body
NUM_BALL_SOCKETS = NUM_CUBES  # 1 world->cube0 + 9 cube_i->cube_(i+1)


class Example:
    def __init__(self, viewer, args):
        # Standard sim cadence used by the basic examples.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # ---- Build the Jitter World ----------------------------------
        # substeps=1 here because we drive substepping ourselves from
        # simulate() (matches the basic-pendulum pattern).
        self.world = World(
            num_bodies=NUM_BODIES,
            num_ball_sockets=NUM_BALL_SOCKETS,
            substeps=1,
            solver_iterations=8,
            device=self.device,
        )

        # Body 0 is the static world anchor (motion_type = STATIC, default
        # from body_container_zeros, so we leave it as-is). Cubes 1..N are
        # dynamic: unit mass, identity body-frame inverse inertia (a unit
        # cube of mass 1 has I = m * s^2 / 6 * I_3 = 1/6 * I_3, so
        # I^-1 = 6 * I_3; using identity here keeps the demo's numbers
        # round and is fine since it's an unstressed visual demo).
        positions = np.zeros((NUM_BODIES, 3), dtype=np.float32)
        orientations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (NUM_BODIES, 1))
        inverse_mass = np.zeros(NUM_BODIES, dtype=np.float32)
        inverse_inertia_body = np.zeros((NUM_BODIES, 3, 3), dtype=np.float32)
        motion_type = np.zeros(NUM_BODIES, dtype=np.int32)  # 0 = MOTION_STATIC

        # Body i = cube (i-1). Cube j sits at x = (2j+1)*h on the +x axis
        # and alternates y = +h (j even) / -h (j odd), so consecutive
        # cubes meet at exactly one diagonal corner on the y=0 line.
        for i in range(1, NUM_BODIES):
            cube_idx = i - 1
            sign = 1.0 if cube_idx % 2 == 0 else -1.0
            positions[i, 0] = (2 * cube_idx + 1) * HALF_EXTENT  # 0.5, 1.5, 2.5, ...
            positions[i, 1] = sign * HALF_EXTENT
            inverse_mass[i] = 1.0
            inverse_inertia_body[i] = np.eye(3, dtype=np.float32)
            motion_type[i] = int(MOTION_DYNAMIC)

        self.world.bodies.position.assign(positions)
        self.world.bodies.orientation.assign(orientations)
        self.world.bodies.inverse_mass.assign(inverse_mass)
        # inverse_inertia (body frame) is the persistent source of truth;
        # _update_bodies_kernel rebuilds inverse_inertia_world from it
        # each step. We seed inverse_inertia_world with the same value so
        # the very first step (which runs before _update_bodies has a
        # chance to fire) sees a non-zero world inertia.
        self.world.bodies.inverse_inertia.assign(inverse_inertia_body)
        self.world.bodies.inverse_inertia_world.assign(inverse_inertia_body)
        self.world.bodies.motion_type.assign(motion_type)

        # ---- Build the ball-socket joints ----------------------------
        # Joint k connects body k and body k+1 (body 0 = static world).
        # Anchor convention:
        #   * k = 0:  world (origin) ↔ cube 0's bottom-left-back corner
        #             at world position (0, 0, 0) -- this is cube 0's
        #             (-h, -h, 0) corner since cube 0 sits at (h, +h, 0).
        #   * k ≥ 1:  cube (k-1) ↔ cube k. The two cubes are placed so
        #             their shared corner sits on the centerline at
        #             world position (2k*h, 0, 0). For cube k-1 (centred
        #             at sign_{k-1}*h on y) this is its (+h, -sign_{k-1}*h, 0)
        #             corner, and for cube k (sign_k = -sign_{k-1}) it's
        #             the opposite (-h, -sign_k*h, 0) = (-h, +sign_{k-1}*h, 0)
        #             corner -- diagonal corners of two diagonally
        #             adjacent cubes.
        constraints_np = np.zeros(NUM_BALL_SOCKETS, dtype=BallSocketData.numpy_dtype())
        for k in range(NUM_BALL_SOCKETS):
            body_a = k  # 0 = world, 1..NUM_CUBES-1 = cubes
            body_b = k + 1  # 1..NUM_CUBES = cubes

            pos_a = wp.vec3f(*positions[body_a])
            pos_b = wp.vec3f(*positions[body_b])
            rba = RigidBodyData()
            rba.position = pos_a
            rba.orientation = wp.quatf(0, 0, 0, 1)
            rbb = RigidBodyData()
            rbb.position = pos_b
            rbb.orientation = wp.quatf(0, 0, 0, 1)

            if k == 0:
                anchor = wp.vec3f(0.0, 0.0, 0.0)
            else:
                anchor = wp.vec3f(2.0 * k * HALF_EXTENT, 0.0, 0.0)

            data = BallSocketData()
            data.body1 = int(body_a)
            data.body2 = int(body_b)
            data = ball_socket_initialize(data, rba, rbb, anchor)
            constraints_np[k] = data.numpy_value()

        self.world.ball_sockets.assign(constraints_np)

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
        # Pack body (position, orientation) into wp.transform for the viewer.
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
        # After one second the chain should have swung from the world anchor
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
