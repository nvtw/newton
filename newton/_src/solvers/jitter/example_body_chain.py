# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Body Chain
#
# Ten unit cubes laid out face-to-face along +x and connected with ball
# joints in a zig-zag pattern: cube i's top-right-front corner is glued to
# cube (i+1)'s bottom-left-back corner. Cube 0's bottom-left-back corner is
# attached to a static "world" body so the chain hangs in mid-air. Gravity
# (-y) pulls everything down and the joints keep the chain together.
#
# Run:  python -m newton._src.solvers.jitter.example_body_chain
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.body import RigidBodyData
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

        # Body 0 is the static world anchor (inverse_mass = 0 -> the
        # integration / constraint kernels both skip it).
        positions = np.zeros((NUM_BODIES, 3), dtype=np.float32)
        orientations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (NUM_BODIES, 1))
        inverse_mass = np.zeros(NUM_BODIES, dtype=np.float32)
        # World inverse-inertia for unit-mass unit cubes is the identity *
        # 1 / (m * (2*hx)^2 / 6) = 6 (for m=1, side=1). Using 1.0 is a
        # decent visual default; constraints don't care about the exact
        # value beyond non-zero.
        inverse_inertia = np.zeros((NUM_BODIES, 3, 3), dtype=np.float32)

        # Cube i (i = 1..NUM_CUBES) sits at x = (2*i - 1) * HALF_EXTENT
        # so the cubes touch face-to-face along +x and the world anchor
        # (origin) coincides with cube 0's -x face center.
        for i in range(1, NUM_BODIES):
            cube_idx = i - 1
            positions[i, 0] = (2 * cube_idx + 1) * HALF_EXTENT  # 0.5, 1.5, 2.5, ...
            inverse_mass[i] = 1.0
            inverse_inertia[i] = np.eye(3, dtype=np.float32)  # 1.0 on the diagonal

        self.world.bodies.position.assign(positions)
        self.world.bodies.orientation.assign(orientations)
        self.world.bodies.inverse_mass.assign(inverse_mass)
        self.world.bodies.inverse_inertia_world.assign(inverse_inertia)

        # ---- Build the ball-socket joints ----------------------------
        # Constraint k anchors a corner shared between body k and body k+1
        # (using 1-based body indices, so the world body is body 0).
        # Anchor convention: cube i's top-right-front corner
        # (+hx, +hy, +hz) coincides with cube (i+1)'s bottom-left-back
        # corner (-hx, -hy, -hz) when both are positioned along +x with
        # 2*HALF_EXTENT spacing. For the world->cube0 joint the anchor is
        # cube 0's bottom-left-back corner (i.e. world origin offset
        # (-hx, -hy, -hz) from cube 0).
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

            # Shared anchor: world-space point where the corners meet.
            # For k=0 (world->cube0): anchor at cube 0's -x/-y/-z corner
            #     = pos_b + (-hx, -hy, -hz).
            # For k>=1 (cube_(k-1)->cube_k): anchor at cube (k-1)'s
            #     +hx/+hy/+hz corner = pos_a + (+hx, +hy, +hz),
            # which equals cube k's -hx/-hy/-hz corner because the cubes
            # are spaced by 2*hx along x and aligned in y/z.
            if k == 0:
                anchor = wp.vec3f(
                    float(positions[body_b, 0]) - HALF_EXTENT,
                    float(positions[body_b, 1]) - HALF_EXTENT,
                    float(positions[body_b, 2]) - HALF_EXTENT,
                )
            else:
                anchor = wp.vec3f(
                    float(positions[body_a, 0]) + HALF_EXTENT,
                    float(positions[body_a, 1]) + HALF_EXTENT,
                    float(positions[body_a, 2]) + HALF_EXTENT,
                )

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
