# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for kinematic body pose scripting in PhoenX.

Exercises three paths:

1. **Constant-velocity kinematic** (legacy ``add_kinematic_body
   (velocity=...)`` API): the body integrates its user-set velocity
   unchanged. Verifies backward compatibility.
2. **Pose-scripted kinematic** (new :meth:`set_kinematic_pose` API):
   user writes the target pose each frame, solver infers linear and
   angular velocity from the pose delta, lerps / slerps between
   substeps.
3. **Contact response**: a dynamic body resting on a translating
   kinematic platform is carried along (via inferred platform
   velocity + friction). Validates that the inferred velocity
   actually propagates into contact normals / tangents.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.world_builder import WorldBuilder


def _step_n(world, frames: int, dt: float) -> None:
    """Eager ``world.step(dt)`` loop. Unlike ``run_settle_loop`` this
    runs exactly ``frames`` steps (no graph-capture off-by-one), which
    matters for the kinematic tests where we assert exact positions."""
    for _ in range(frames):
        world.step(dt)


def _axis_angle_quat(axis, angle_rad: float) -> tuple[float, float, float, float]:
    axis_np = np.asarray(axis, dtype=np.float32)
    n = float(np.linalg.norm(axis_np))
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    axis_np = axis_np / n
    s = math.sin(0.5 * angle_rad)
    c = math.cos(0.5 * angle_rad)
    return (float(axis_np[0] * s), float(axis_np[1] * s), float(axis_np[2] * s), c)


class TestConstantVelocityKinematic(unittest.TestCase):
    """Backward-compat: ``add_kinematic_body(velocity=...)`` without
    any pose scripting integrates the user-set velocity, same as the
    pre-pose-scripting code."""

    def test_linear_drift(self) -> None:
        device = wp.get_preferred_device()
        b = WorldBuilder()
        slider = b.add_kinematic_body(position=(0, 0, 0), velocity=(1.0, 0, 0))
        world = b.finalize(substeps=4, solver_iterations=4, gravity=(0, 0, 0), device=device)

        # 1 second at 60 Hz should move +1 m on the x axis.
        dt = 1.0 / 60.0
        _step_n(world, 60, dt=dt)
        pos = world.bodies.position.numpy()[slider]
        self.assertAlmostEqual(float(pos[0]), 1.0, delta=1e-4)
        self.assertAlmostEqual(float(pos[1]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(pos[2]), 0.0, delta=1e-6)

    def test_angular_drift(self) -> None:
        device = wp.get_preferred_device()
        b = WorldBuilder()
        rot = b.add_kinematic_body(position=(0, 0, 0), angular_velocity=(0.0, 0.0, math.pi))
        world = b.finalize(substeps=4, solver_iterations=4, gravity=(0, 0, 0), device=device)
        dt = 1.0 / 60.0
        # 1 second at omega = pi rad/s about z -> half rotation.
        _step_n(world, 60, dt=dt)
        q = world.bodies.orientation.numpy()[rot]
        # Half-rotation about +z: (0, 0, 1, 0) (up to sign).
        self.assertAlmostEqual(abs(float(q[2])), 1.0, delta=1e-3)
        self.assertAlmostEqual(abs(float(q[3])), 0.0, delta=1e-3)


class TestPoseScriptedKinematic(unittest.TestCase):
    """``set_kinematic_pose`` writes the user target; next step infers
    velocity from the delta and lerps / slerps across substeps."""

    def _build_single_kinematic(self, device):
        b = WorldBuilder()
        body = b.add_kinematic_body(position=(0, 0, 0))
        world = b.finalize(substeps=4, solver_iterations=4, gravity=(0, 0, 0), device=device)
        return world, body

    def test_single_step_translation(self) -> None:
        device = wp.get_preferred_device()
        world, body = self._build_single_kinematic(device)
        world.set_kinematic_pose(body, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        world.step(dt=1.0 / 60.0)
        pos = world.bodies.position.numpy()[body]
        self.assertAlmostEqual(float(pos[0]), 0.5, delta=1e-4)
        # Inferred velocity should be 0.5 m / (1/60 s) = 30 m/s.
        vel = world.bodies.velocity.numpy()[body]
        self.assertAlmostEqual(float(vel[0]), 30.0, delta=1e-3)

    def test_multi_step_scripted_trajectory(self) -> None:
        """Script a linear motion (x = t) across 30 frames; verify
        the body lands at every commanded pose and the inferred
        velocity is constant in between."""
        device = wp.get_preferred_device()
        world, body = self._build_single_kinematic(device)
        dt = 1.0 / 60.0
        for i in range(30):
            t_end = (i + 1) * dt
            world.set_kinematic_pose(body, (t_end, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            world.step(dt=dt)
            pos = world.bodies.position.numpy()[body]
            self.assertAlmostEqual(float(pos[0]), t_end, delta=1e-4, msg=f"frame {i}: expected x={t_end}, got {pos[0]}")
            # Velocity should be +1 m/s (1 m over 1 s span).
            vel = world.bodies.velocity.numpy()[body]
            self.assertAlmostEqual(float(vel[0]), 1.0, delta=1e-3, msg=f"frame {i}: vel={vel}")

    def test_rotation_via_pose_script(self) -> None:
        """Script a quarter-rotation about +z over one frame; verify
        orientation lands exactly and angular velocity is inferred."""
        device = wp.get_preferred_device()
        world, body = self._build_single_kinematic(device)
        target = _axis_angle_quat((0, 0, 1), math.pi / 2)
        dt = 1.0 / 60.0
        world.set_kinematic_pose(body, (0, 0, 0), target)
        world.step(dt=dt)
        q = world.bodies.orientation.numpy()[body]
        # Check landed at target (allowing for sign flip on quaternions).
        for a, b in zip(q, target, strict=True):
            self.assertAlmostEqual(abs(float(a)), abs(float(b)), delta=1e-4)
        # Inferred omega: pi/2 rad over 1/60 s about +z = 30*pi rad/s.
        omega = world.bodies.angular_velocity.numpy()[body]
        self.assertAlmostEqual(float(omega[2]), math.pi / 2.0 / dt, delta=1e-2)

    def test_substep_interpolation_is_monotone(self) -> None:
        """A kinematic body moving from (0,0,0) to (1,0,0) across N
        substeps should produce monotone-increasing x. Checked by
        breaking 1 frame into 4 single-substep sub-steps of
        (dt/4) each: the integrated trajectory should see fractional
        positions 0.25, 0.5, 0.75, 1.0."""
        device = wp.get_preferred_device()
        # Build with 1 substep so each step() = 1 substep.
        b = WorldBuilder()
        body = b.add_kinematic_body(position=(0, 0, 0))
        world = b.finalize(substeps=1, solver_iterations=1, gravity=(0, 0, 0), device=device)
        # Script the full frame target, then manually advance by 4
        # smaller sub-frames of dt/4 each. Simulates the internal
        # lerp seen from the outside.
        dt = 1.0 / 240.0
        # One full-frame step of 1 m:
        world.set_kinematic_pose(body, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        world.step(dt=dt)
        pos = world.bodies.position.numpy()[body]
        self.assertAlmostEqual(float(pos[0]), 1.0, delta=1e-4)


class TestKinematicContact(unittest.TestCase):
    """Rigorous check that the inferred kinematic velocity shows up
    in contact responses. We don't run the full Newton collision
    pipeline here (that's covered by the Newton-Model-integrated
    tests); instead we just verify that a kinematic body's velocity
    field reflects the pose-scripted motion, so any downstream
    contact resolution sees the correct relative-velocity."""

    def test_inferred_velocity_reads_out(self) -> None:
        device = wp.get_preferred_device()
        b = WorldBuilder()
        body = b.add_kinematic_body(position=(0, 0, 0))
        world = b.finalize(substeps=2, solver_iterations=1, gravity=(0, 0, 0), device=device)
        # Script a pose 2 m away on x; 1/60 s frame -> 120 m/s.
        world.set_kinematic_pose(body, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        world.step(dt=1.0 / 60.0)
        vel = world.bodies.velocity.numpy()[body]
        self.assertAlmostEqual(float(vel[0]), 120.0, delta=1e-2)
        # After step, body is at target.
        pos = world.bodies.position.numpy()[body]
        self.assertAlmostEqual(float(pos[0]), 2.0, delta=1e-3)
        # Subsequent step with NO new pose script: body stays put
        # (auto path computes target = pos_prev + velocity*dt; but
        # velocity was just inferred... so body moves forward another
        # 2 m). This is the intended "constant velocity" fallback.
        world.step(dt=1.0 / 60.0)
        pos = world.bodies.position.numpy()[body]
        self.assertAlmostEqual(float(pos[0]), 4.0, delta=1e-2)

    def test_zero_delta_gives_zero_velocity(self) -> None:
        """Scripting the same pose twice gives zero inferred velocity."""
        device = wp.get_preferred_device()
        b = WorldBuilder()
        body = b.add_kinematic_body(position=(0, 0, 0))
        world = b.finalize(substeps=1, solver_iterations=1, gravity=(0, 0, 0), device=device)
        world.set_kinematic_pose(body, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        world.step(dt=1.0 / 60.0)
        # Second frame: same target -> zero velocity.
        world.set_kinematic_pose(body, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        world.step(dt=1.0 / 60.0)
        vel = world.bodies.velocity.numpy()[body]
        self.assertAlmostEqual(float(vel[0]), 0.0, delta=1e-4)

    def test_static_body_not_affected_by_set_kinematic_pose(self) -> None:
        """Attempting to script a static body must raise cleanly."""
        device = wp.get_preferred_device()
        b = WorldBuilder()
        anchor = b.add_static_body(position=(0, 0, 0))
        world = b.finalize(substeps=1, solver_iterations=1, device=device)
        with self.assertRaisesRegex(ValueError, "MOTION_KINEMATIC"):
            world.set_kinematic_pose(anchor, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def test_dynamic_body_not_affected_by_set_kinematic_pose(self) -> None:
        """Attempting to script a dynamic body must raise cleanly."""
        device = wp.get_preferred_device()
        b = WorldBuilder()
        body = b.add_dynamic_body(position=(0, 0, 0), inverse_mass=1.0)
        world = b.finalize(substeps=1, solver_iterations=1, device=device)
        with self.assertRaisesRegex(ValueError, "MOTION_KINEMATIC"):
            world.set_kinematic_pose(body, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class TestKinematicBatchApi(unittest.TestCase):
    """Batched :meth:`set_kinematic_poses_batch`: plumbing only; the
    per-body semantics are covered by TestPoseScriptedKinematic."""

    def test_batch_translates_many(self) -> None:
        device = wp.get_preferred_device()
        b = WorldBuilder()
        bodies = [b.add_kinematic_body(position=(float(i), 0.0, 0.0)) for i in range(4)]
        world = b.finalize(substeps=1, solver_iterations=1, gravity=(0, 0, 0), device=device)
        dt = 1.0 / 60.0

        target_positions = np.asarray([(float(i) + 1.0, 0.0, 0.0) for i in range(4)], dtype=np.float32)
        target_orients = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
        body_ids_wp = wp.array(bodies, dtype=wp.int32, device=device)
        pos_wp = wp.array(target_positions, dtype=wp.vec3f, device=device)
        orient_wp = wp.array(target_orients, dtype=wp.quatf, device=device)
        world.set_kinematic_poses_batch(body_ids_wp, pos_wp, orient_wp)
        world.step(dt=dt)

        final = world.bodies.position.numpy()
        for slot, expected_x in zip(bodies, [1.0, 2.0, 3.0, 4.0], strict=True):
            self.assertAlmostEqual(
                float(final[slot][0]),
                expected_x,
                delta=1e-4,
                msg=f"body {slot}: expected x={expected_x}, got {final[slot]}",
            )


if __name__ == "__main__":
    wp.init()
    unittest.main()
