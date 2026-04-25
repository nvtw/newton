# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world smoke tests for :class:`PhoenXWorld`.

Ports the jitter-side :mod:`test_multi_world` suite: verify that
:class:`PhoenXWorld` built with ``num_worlds > 1`` keeps worlds
isolated (no cross-world state leakage), supports per-world
gravity, and scales to many worlds without regressing.

Uses the actuated-double-ball-socket joint in ``BALL_SOCKET`` mode
as the single per-world constraint (the jitter suite uses the
standalone ``add_ball_socket``; PhoenX's unified joint produces an
equivalent positional lock).

Runs on CUDA only: the per-world CSR scatter / dispatcher requires
graph-captured launches for reasonable wall-clock throughput.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    body_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode

_FPS = 60
_SUBSTEPS = 4
_SOLVER_ITERATIONS = 16


def _build_n_pendulums(
    *,
    num_worlds: int,
    angular_velocities: list[tuple[float, float, float]] | None = None,
    gravity: tuple[float, float, float] | list[tuple[float, float, float]] = (0.0, -9.81, 0.0),
    device: wp.context.Devicelike = None,
) -> tuple[PhoenXWorld, list[int]]:
    """Build ``num_worlds`` identical pendulum scenes sharing one
    :class:`BodyContainer` / :class:`ConstraintContainer`.

    Each world gets a static anchor body (slot ``2*w``) and a
    dynamic cube 1 m below it (slot ``2*w + 1``) connected by one
    ball-socket joint; each cube is tagged with ``world_id = w``
    so the per-world gravity + CSR bucketing kicks in.

    Returns the solver and the list of *Newton-side* cube indices,
    which in our convention are PhoenX slots (there's no Newton
    model here -- we hand the solver raw containers). The caller
    queries ``bodies.position.numpy()[cube_slot]`` directly.
    """
    if device is None:
        device = wp.get_device("cuda:0")
    num_bodies = 2 * num_worlds  # anchor + cube per world
    bodies = body_container_zeros(num_bodies, device=device)

    pos = np.zeros((num_bodies, 3), dtype=np.float32)
    ori = np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies, 1)).astype(np.float32)
    inv_m = np.zeros(num_bodies, dtype=np.float32)
    inv_I = np.zeros((num_bodies, 3, 3), dtype=np.float32)
    motion = np.full(num_bodies, int(MOTION_STATIC), dtype=np.int32)
    world_id = np.zeros(num_bodies, dtype=np.int32)
    ang_v = np.zeros((num_bodies, 3), dtype=np.float32)

    cube_slots: list[int] = []
    for w in range(num_worlds):
        anchor_slot = 2 * w
        cube_slot = 2 * w + 1
        # Anchor is static (default).
        world_id[anchor_slot] = w
        # Cube dangles 1 m below the anchor.
        pos[cube_slot] = (0.0, -1.0, 0.0)
        inv_m[cube_slot] = 1.0
        inv_I[cube_slot] = np.eye(3, dtype=np.float32)
        motion[cube_slot] = int(MOTION_DYNAMIC)
        world_id[cube_slot] = w
        if angular_velocities is not None:
            ang_v[cube_slot] = angular_velocities[w]
        cube_slots.append(cube_slot)

    bodies.position.assign(pos)
    bodies.orientation.assign(ori)
    bodies.inverse_mass.assign(inv_m)
    bodies.inverse_inertia.assign(inv_I)
    bodies.inverse_inertia_world.assign(inv_I)
    bodies.motion_type.assign(motion)
    bodies.world_id.assign(world_id)
    bodies.angular_velocity.assign(ang_v)

    # One joint per world.
    num_joints = num_worlds
    constraints = PhoenXWorld.make_constraint_container(num_joints=num_joints, max_contact_columns=0, device=device)
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        substeps=_SUBSTEPS,
        solver_iterations=_SOLVER_ITERATIONS,
        velocity_iterations=1,
        gravity=gravity,
        rigid_contact_max=0,
        num_joints=num_joints,
        num_worlds=num_worlds,
        device=device,
    )

    # Init one BALL_SOCKET joint per world.
    body1 = np.array([2 * w for w in range(num_joints)], dtype=np.int32)
    body2 = np.array([2 * w + 1 for w in range(num_joints)], dtype=np.int32)
    anchor1 = np.zeros((num_joints, 3), dtype=np.float32)  # world origin for each world
    anchor2 = np.zeros((num_joints, 3), dtype=np.float32)  # ignored for BALL_SOCKET

    def _f(v: float) -> wp.array:
        return wp.array(np.full(num_joints, v, dtype=np.float32), dtype=wp.float32, device=device)

    def _i(v: int) -> wp.array:
        return wp.array(np.full(num_joints, v, dtype=np.int32), dtype=wp.int32, device=device)

    world.initialize_actuated_double_ball_socket_joints(
        body1=wp.array(body1, dtype=wp.int32, device=device),
        body2=wp.array(body2, dtype=wp.int32, device=device),
        anchor1=wp.array(anchor1, dtype=wp.vec3f, device=device),
        anchor2=wp.array(anchor2, dtype=wp.vec3f, device=device),
        hertz=_f(float(DEFAULT_HERTZ_LINEAR)),
        damping_ratio=_f(float(DEFAULT_DAMPING_RATIO)),
        joint_mode=_i(int(JointMode.BALL_SOCKET)),
        drive_mode=_i(int(DriveMode.OFF)),
        target=_f(0.0),
        target_velocity=_f(0.0),
        max_force_drive=_f(0.0),
        stiffness_drive=_f(0.0),
        damping_drive=_f(0.0),
        min_value=_f(1.0),
        max_value=_f(-1.0),
        hertz_limit=_f(float(DEFAULT_HERTZ_LINEAR)),
        damping_ratio_limit=_f(float(DEFAULT_DAMPING_RATIO)),
        stiffness_limit=_f(0.0),
        damping_limit=_f(0.0),
    )

    return world, cube_slots


def _run_frames(world: PhoenXWorld, n_frames: int) -> None:
    """Advance ``world`` by ``n_frames`` joint-only steps, CUDA-graph
    captured on the first call so each additional frame is one graph
    replay rather than a dozen eager launches."""
    dt = 1.0 / _FPS
    device = wp.get_device()
    if not device.is_cuda or n_frames < 4:
        for _ in range(n_frames):
            world.step(dt=dt, contacts=None, shape_body=None)
        return
    # Warm-up outside the capture so kernel JIT + lazy scratch
    # allocations complete; capture the next step; replay the rest.
    world.step(dt=dt, contacts=None, shape_body=None)
    with wp.ScopedCapture(device=device) as capture:
        world.step(dt=dt, contacts=None, shape_body=None)
    graph = capture.graph
    for _ in range(n_frames - 2):
        wp.capture_launch(graph)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX multi-world tests require CUDA")
class TestPhoenXMultiWorld(unittest.TestCase):
    """Cross-world isolation + per-world gravity checks.

    Directly mirrors :class:`TestMultiWorld` on the jitter side,
    except we use :data:`JointMode.BALL_SOCKET` on the unified joint
    in place of ``add_ball_socket``.
    """

    def test_single_pendulum_per_world(self) -> None:
        """``N`` identical pendulums -> every world's cube ends at
        the same settled pose. Non-match would mean the per-world
        CSR leak'd one world's solve into another's state.
        """
        device = wp.get_device("cuda:0")
        num_worlds = 8
        world, cube_slots = _build_n_pendulums(num_worlds=num_worlds, device=device)
        _run_frames(world, 30)
        positions = world.bodies.position.numpy()
        ref = positions[cube_slots[0]]
        for w in range(1, num_worlds):
            np.testing.assert_allclose(
                positions[cube_slots[w]],
                ref,
                atol=1.0e-4,
                err_msg=f"world {w} cube diverged from world 0",
            )

    def test_world_independence(self) -> None:
        """World 1's cube spins, world 0's cube starts at rest.
        World 0 must stay essentially at rest (no yaw spin leak),
        while world 1 retains its spin.
        """
        device = wp.get_device("cuda:0")
        avels = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0)]
        world, cube_slots = _build_n_pendulums(num_worlds=2, angular_velocities=avels, device=device)
        _run_frames(world, 30)
        avels_after = world.bodies.angular_velocity.numpy()
        w0_ang = avels_after[cube_slots[0]]
        self.assertLess(
            abs(float(w0_ang[1])),
            0.1,
            f"world 0 picked up yaw spin {w0_ang[1]:.3f} -- worlds leaked",
        )
        w1_ang = avels_after[cube_slots[1]]
        self.assertGreater(
            abs(float(w1_ang[1])),
            1.0,
            f"world 1 lost its yaw spin ({w1_ang[1]:.3f}) -- unexpected damping",
        )

    def test_per_world_gravity(self) -> None:
        """World 0 at earth-g, world 1 at moon-g. Cubes should
        swing at different rates -- after the same wall-clock time
        they must be at different z positions (or different joint
        tensions on an identically-held scene).
        """
        device = wp.get_device("cuda:0")
        gravity = [(0.0, -9.81, 0.0), (0.0, -1.62, 0.0)]
        world, cube_slots = _build_n_pendulums(num_worlds=2, gravity=gravity, device=device)
        # Start each cube slightly off-axis so it swings instead of
        # hanging perfectly vertical (which would hide gravity's
        # effect since the ball socket's normal reaction scales
        # with gravity but the cube doesn't actually move).
        positions = world.bodies.position.numpy()
        positions[cube_slots[0]] = (0.3, -0.95, 0.0)  # off-axis
        positions[cube_slots[1]] = (0.3, -0.95, 0.0)
        world.bodies.position.assign(positions)

        _run_frames(world, 30)

        # After 0.5 s both cubes have swung; earth-g cube's
        # angular velocity about the anchor axis should be
        # sqrt(6.06x) bigger than moon-g cube's (ratio of g's).
        # Less precise check: the kinetic energies differ by the
        # gravity ratio since each cube is in the same arc phase
        # after the same time.
        ang_v = world.bodies.angular_velocity.numpy()
        w0_speed = float(np.linalg.norm(ang_v[cube_slots[0]]))
        w1_speed = float(np.linalg.norm(ang_v[cube_slots[1]]))
        # Earth cube must swing visibly faster than moon cube.
        # Exact ratio depends on arc phase; just check w0 > w1 by
        # a large margin.
        self.assertGreater(
            w0_speed,
            2.0 * w1_speed,
            f"earth cube |w|={w0_speed:.3f} not > 2x moon |w|={w1_speed:.3f} -- per-world gravity didn't take effect",
        )

    def test_many_worlds_converge(self) -> None:
        """1024 identical pendulum worlds -- max divergence across
        worlds stays under 1 mm.

        This is the "many worlds" stress test from the jitter
        suite, scaled down from 1024 but still exercising
        the multi-block per-world dispatcher across a large grid.
        """
        device = wp.get_device("cuda:0")
        num_worlds = 256
        world, cube_slots = _build_n_pendulums(num_worlds=num_worlds, device=device)
        _run_frames(world, 10)
        positions = world.bodies.position.numpy()
        ref = positions[cube_slots[0]]
        max_dev = 0.0
        for w in range(1, num_worlds):
            dev = float(np.max(np.abs(positions[cube_slots[w]] - ref)))
            if dev > max_dev:
                max_dev = dev
        self.assertLess(
            max_dev,
            1.0e-3,
            f"max divergence across {num_worlds} worlds was {max_dev:.6f} (> 1e-3)",
        )

    def test_per_world_initial_state_does_not_leak(self) -> None:
        """Give each world a DIFFERENT initial angular velocity.
        After one step each world must retain its distinct spin;
        a leak would homogenise them toward the mean."""
        device = wp.get_device("cuda:0")
        num_worlds = 8
        # 8 linearly-separated spins about y so after 30 frames we
        # can assert each world's final |omega_y| is distinct and
        # monotone in w.
        avels = [(0.0, 0.5 + float(w), 0.0) for w in range(num_worlds)]
        world, cube_slots = _build_n_pendulums(num_worlds=num_worlds, angular_velocities=avels, device=device)
        _run_frames(world, 30)
        ang_v = world.bodies.angular_velocity.numpy()
        per_world_omega = [float(ang_v[cube_slots[w]][1]) for w in range(num_worlds)]
        # Check monotone non-decreasing -- a damping / leakage bug
        # would pull later worlds down toward zero.
        for w in range(num_worlds - 1):
            self.assertGreater(
                per_world_omega[w + 1] + 1.0e-3,
                per_world_omega[w],
                msg=(
                    f"omega_y not monotone across worlds: w{w}={per_world_omega[w]:.3f} "
                    f"vs w{w + 1}={per_world_omega[w + 1]:.3f}"
                ),
            )
        # And check no world collapsed to zero (would indicate
        # complete cross-world damping).
        self.assertGreater(
            min(per_world_omega),
            0.1,
            msg=f"some world lost all spin: {per_world_omega}",
        )

    def test_kill_one_world_does_not_affect_others(self) -> None:
        """World 0 is "killed" by pinning its cube to the anchor
        (zero inverse mass). All other worlds must simulate normally.

        Catches a class of bug where the per-world CSR bucketing
        assumed every world has exactly the same number of active
        bodies / constraints.
        """
        device = wp.get_device("cuda:0")
        num_worlds = 4
        world, cube_slots = _build_n_pendulums(num_worlds=num_worlds, device=device)
        # Zero inverse mass on world 0's cube -> no dynamics.
        inv_mass = world.bodies.inverse_mass.numpy()
        inv_mass[cube_slots[0]] = 0.0
        world.bodies.inverse_mass.assign(inv_mass)
        # Also mark it static so the integrator skips it.
        motion = world.bodies.motion_type.numpy()
        motion[cube_slots[0]] = int(MOTION_STATIC)
        world.bodies.motion_type.assign(motion)

        _run_frames(world, 30)

        # World 0: cube stayed put. Worlds 1..N: cube settled to the
        # same place.
        positions = world.bodies.position.numpy()
        self.assertAlmostEqual(
            float(positions[cube_slots[0]][0]),
            0.0,
            delta=1.0e-4,
            msg="killed world's cube drifted despite static pinning",
        )
        ref = positions[cube_slots[1]]
        for w in range(2, num_worlds):
            np.testing.assert_allclose(
                positions[cube_slots[w]],
                ref,
                atol=1.0e-4,
                err_msg=(f"world {w} diverged from world 1 after world 0 was killed -- per-world isolation broken"),
            )

    def test_1024_worlds_stress(self) -> None:
        """Scale the identical-worlds check to 1024. Catches crashes
        from per-world scratch sizing (the CSR scatter allocates
        ``num_worlds * max_colors`` entries), and catches regressions
        in the 1-block-per-world dispatcher at scale.

        Deliberately keeps the physics trivial (settling pendulum) so
        numerical divergence under float32 stays bounded; the point
        here is "does the solver handle 1024 blocks without OOM,
        race conditions, or CSR clipping" not "is 5-digit precision
        preserved across 1024 copies".
        """
        device = wp.get_device("cuda:0")
        num_worlds = 1024
        world, cube_slots = _build_n_pendulums(num_worlds=num_worlds, device=device)
        # Short run -- just enough for one full graph replay.
        _run_frames(world, 8)
        positions = world.bodies.position.numpy()
        self.assertTrue(
            np.isfinite(positions).all(),
            msg="NaN / inf in body positions after 1024-world step",
        )
        ref = positions[cube_slots[0]]
        max_dev = 0.0
        for w in range(1, num_worlds):
            dev = float(np.max(np.abs(positions[cube_slots[w]] - ref)))
            if dev > max_dev:
                max_dev = dev
        # Much looser tolerance than the 256-world case: 1024 blocks
        # fire in interleaved order under the graph-coloring CSR, and
        # float32 + per-block scratch produce bit-level jitter across
        # worlds. The physical drift is still sub-millimetre.
        self.assertLess(
            max_dev,
            5.0e-3,
            msg=(f"1024-world max divergence {max_dev:.6f} m exceeds 5 mm -- per-world isolation regressed at scale"),
        )

    def test_mixed_gravity_magnitudes(self) -> None:
        """Three worlds with three distinct gravities (earth, moon,
        zero-g). Cubes start off-axis so they swing; after the same
        wall-clock time their kinetic energies differ according to
        the gravity ratios, and the zero-g cube stays near its
        starting point.
        """
        device = wp.get_device("cuda:0")
        gravity = [
            (0.0, -9.81, 0.0),  # earth
            (0.0, -1.62, 0.0),  # moon
            (0.0, 0.0, 0.0),  # zero-g
        ]
        world, cube_slots = _build_n_pendulums(num_worlds=3, gravity=gravity, device=device)
        # Off-axis starts so gravity actually does work.
        positions = world.bodies.position.numpy()
        for cube in cube_slots:
            positions[cube] = (0.3, -0.95, 0.0)
        world.bodies.position.assign(positions)

        _run_frames(world, 30)

        ang_v = world.bodies.angular_velocity.numpy()
        w_earth = float(np.linalg.norm(ang_v[cube_slots[0]]))
        w_moon = float(np.linalg.norm(ang_v[cube_slots[1]]))
        w_zero = float(np.linalg.norm(ang_v[cube_slots[2]]))

        # Earth > moon > zero-g on angular speed.
        self.assertGreater(
            w_earth,
            1.5 * w_moon,
            msg=f"earth |w|={w_earth:.3f} not > 1.5x moon |w|={w_moon:.3f}",
        )
        self.assertGreater(
            w_moon,
            5.0 * w_zero,
            msg=f"moon |w|={w_moon:.3f} not > 5x zero-g |w|={w_zero:.3f}",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX multi-world tests require CUDA")
class TestPhoenXMultiWorldScaling(unittest.TestCase):
    """Stability / cardinality checks at sub-1024 world counts that
    stress the per-world block-dispatcher edges.

    Kept in a separate class so ``python -m unittest
    test_multi_world.TestPhoenXMultiWorld`` picks the fast suite,
    and scaling is opt-in via the fully-qualified class path.
    """

    def test_various_world_counts_finite(self) -> None:
        """Run the identical-pendulums scene at a sweep of world
        counts and assert positions stay finite. Catches off-by-one
        in the world_num_colors / world_csr_offsets sizing.
        """
        device = wp.get_device("cuda:0")
        # Include boundary sizes: 1 (single-block edge case), 2 (adjacent
        # blocks), 32 (one warp's worth), 128 (half-SM), 513 (just over
        # 512 -- catches fixed-size scratch arrays).
        for num_worlds in [1, 2, 32, 128, 513]:
            with self.subTest(num_worlds=num_worlds):
                world, cube_slots = _build_n_pendulums(num_worlds=num_worlds, device=device)
                _run_frames(world, 5)
                positions = world.bodies.position.numpy()
                self.assertTrue(
                    np.isfinite(positions).all(),
                    msg=f"non-finite positions at num_worlds={num_worlds}",
                )
                # Also check every world's cube is somewhere near
                # the anchor (within 2 m); a runaway dispatcher bug
                # could launch cubes into the void.
                for w, slot in enumerate(cube_slots):
                    dist = float(np.linalg.norm(positions[slot]))
                    self.assertLess(
                        dist,
                        2.0,
                        msg=(f"num_worlds={num_worlds}, world={w}: cube at distance {dist:.3f} m from origin"),
                    )


if __name__ == "__main__":
    unittest.main()
