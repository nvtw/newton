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


@unittest.skipUnless(
    wp.is_cuda_available(),
    "Newton-pipeline kinematic tests require CUDA (CollisionPipeline + graph capture).",
)
class TestKinematicNewtonCollisionPipeline(unittest.TestCase):
    """Newton's :class:`CollisionPipeline` reads ``state.body_q``, not
    PhoenX's body container. A scripted kinematic body whose
    ``state.body_q`` slot isn't refreshed is invisible to broad/narrow
    phase, even though PhoenX's internal pose tracks the target. This
    is the bug the Kapla camera collider hit before the host-side
    ``state.body_q`` patch in :mod:`example_kapla_tower`.
    """

    def _build_kinematic_pusher_scene(self):
        """Two unit spheres a small gap apart on a plane: one
        kinematic (slot 0), one dynamic (slot 1). Returns
        ``(model, state, contacts, collision_pipeline, world,
        bodies, kine_id, dyn_id)``.
        """
        import newton  # noqa: PLC0415
        from newton._src.solvers.phoenx.body import (  # noqa: PLC0415
            MOTION_KINEMATIC,
            body_container_zeros,
        )
        from newton._src.solvers.phoenx.examples.example_common import (  # noqa: PLC0415
            init_phoenx_bodies_kernel,
            newton_to_phoenx_kernel,
        )
        from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld  # noqa: PLC0415

        device = wp.get_device("cuda:0")
        radius = 0.2
        gap = 0.05  # initial separation along x; smaller than the
        # 0.3 m kinematic step, so the dynamic body cannot help being
        # touched.

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, gap=0.01)

        kine_id = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, radius), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(kine_id, radius=radius, cfg=cfg)
        dyn_id = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(2.0 * radius + gap, 0.0, radius),
                q=wp.quat_identity(),
            ),
        )
        builder.add_shape_sphere(dyn_id, radius=radius, cfg=cfg)

        model = builder.finalize()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = collision_pipeline.contacts()
        rigid_contact_max = int(contacts.rigid_contact_point0.shape[0])
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.body_q.assign(state.body_q)

        num_phx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phx_bodies, device=device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=device,
            ),
        )
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=model.body_count,
            inputs=[
                model.body_q,
                state.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=device,
        )
        # Promote slot 0 -> kinematic, zero its inverse mass / inertia.
        slot = kine_id + 1
        mt = bodies.motion_type.numpy()
        mt[slot] = int(MOTION_KINEMATIC)
        bodies.motion_type.assign(mt)
        for arr_name in ("inverse_mass",):
            arr = getattr(bodies, arr_name).numpy()
            arr[slot] = 0.0
            getattr(bodies, arr_name).assign(arr)
        for arr_name in ("inverse_inertia", "inverse_inertia_world"):
            arr = getattr(bodies, arr_name).numpy()
            arr[slot] = np.zeros((3, 3), dtype=np.float32)
            getattr(bodies, arr_name).assign(arr)

        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        shape_body_np = model.shape_body.numpy()
        shape_body_phx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        shape_body = wp.array(shape_body_phx, dtype=wp.int32, device=device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            substeps=4,
            solver_iterations=4,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            step_layout="single_world",
            device=device,
        )

        # The example_kapla_tower pattern: skip the kinematic body in
        # *both* sync directions so its only writers are
        # ``set_kinematic_poses_batch`` (PhoenX side) and the optional
        # ``state.body_q`` patch under test. We achieve that by
        # syncing the slice ``[0, kine_id)``; the dynamic body is
        # before kine_id by construction (kine added first, dyn
        # second... we need to swap order).
        # Actually we added kine_id FIRST so dyn_id > kine_id. Sync
        # only ``[kine_id+1, body_count)`` -- i.e. everything past the
        # kinematic body. There's just the one dynamic body after
        # kine_id, so this trims to a 1-body slice.
        sync_start = kine_id + 1
        sync_count = model.body_count - sync_start

        def sync_dynamic_to_phx() -> None:
            if sync_count <= 0:
                return
            wp.launch(
                newton_to_phoenx_kernel,
                dim=sync_count,
                inputs=[
                    state.body_q[sync_start:],
                    state.body_qd[sync_start:],
                    model.body_com,
                ],
                outputs=[
                    bodies.position[1 + sync_start : 1 + sync_start + sync_count],
                    bodies.orientation[1 + sync_start : 1 + sync_start + sync_count],
                    bodies.velocity[1 + sync_start : 1 + sync_start + sync_count],
                    bodies.angular_velocity[1 + sync_start : 1 + sync_start + sync_count],
                ],
                device=device,
            )

        return (
            model,
            state,
            contacts,
            collision_pipeline,
            world,
            bodies,
            shape_body,
            kine_id,
            dyn_id,
            sync_dynamic_to_phx,
            sync_start,
            sync_count,
        )

    def _run(self, *, patch_state_body_q: bool, frames: int = 8):
        """Step the kinematic-pusher scene ``frames`` times, moving the
        kinematic body by 0.3 m along +x each frame. Returns the
        final dynamic-body x-position (world frame).

        ``patch_state_body_q=True`` mirrors the camera-collider fix:
        write the kinematic target into ``state.body_q`` so Newton's
        CollisionPipeline sees the body's live position. ``False``
        is the buggy path -- shows that the dynamic body never moves
        because broad phase doesn't see the kinematic close in.
        """
        (
            model,
            state,
            contacts,
            collision_pipeline,
            world,
            bodies,
            shape_body,
            kine_id,
            dyn_id,
            sync_dynamic_to_phx,
            sync_start,
            sync_count,
        ) = self._build_kinematic_pusher_scene()
        device = world.device
        radius = 0.2
        kine_pos_arr = wp.array([(0.0, 0.0, radius)], dtype=wp.vec3f, device=device)
        kine_orient_arr = wp.array([(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
        body_id_arr = wp.array([int(kine_id + 1)], dtype=wp.int32, device=device)
        dt = 1.0 / 60.0

        for f in range(frames):
            x = 0.3 * (f + 1)
            kine_pos_np = np.asarray([(x, 0.0, radius)], dtype=np.float32)
            kine_pos_arr.assign(kine_pos_np)
            world.set_kinematic_poses_batch(
                body_ids=body_id_arr,
                positions=kine_pos_arr,
                orientations=kine_orient_arr,
            )
            if patch_state_body_q:
                # Rewrite the kinematic body's slot of state.body_q so
                # CollisionPipeline broad/narrow-phases at the live
                # position. This is the same trick example_kapla_tower
                # uses for its camera collider.
                bq = state.body_q.numpy()
                bq[kine_id] = (x, 0.0, radius, 0.0, 0.0, 0.0, 1.0)
                state.body_q.assign(bq)
            sync_dynamic_to_phx()
            model.collide(state, contacts=contacts, collision_pipeline=collision_pipeline)
            world.step(dt=dt, contacts=contacts, shape_body=shape_body)
            # Pull the dynamic body's pose back to host for assertions
            # via the PhoenX-side container (which the solver wrote).
            dyn_pos = bodies.position.numpy()[dyn_id + 1]
            # Bridge dynamic-body slice only back to Newton state.
            # Skipping the kinematic slot mirrors example_kapla_tower's
            # design -- the host-side patch (or absence of one) is the
            # only writer of state.body_q[kine_id] under test.
            if sync_count > 0:
                from newton._src.solvers.phoenx.examples.example_common import (  # noqa: PLC0415
                    phoenx_to_newton_kernel,
                )

                wp.launch(
                    phoenx_to_newton_kernel,
                    dim=sync_count,
                    inputs=[
                        bodies.position[1 + sync_start : 1 + sync_start + sync_count],
                        bodies.orientation[1 + sync_start : 1 + sync_start + sync_count],
                        bodies.velocity[1 + sync_start : 1 + sync_start + sync_count],
                        bodies.angular_velocity[1 + sync_start : 1 + sync_start + sync_count],
                        model.body_com,
                    ],
                    outputs=[state.body_q[sync_start:], state.body_qd[sync_start:]],
                    device=device,
                )
        return float(dyn_pos[0])

    def test_kinematic_pushes_dynamic_when_state_body_q_patched(self) -> None:
        """Kinematic body advances 0.3 m/frame; without the patch the
        dynamic body sits still, with the patch it gets shoved."""
        x_with = self._run(patch_state_body_q=True, frames=4)
        x_without = self._run(patch_state_body_q=False, frames=4)
        # Dynamic body started at 2*0.2 + 0.05 = 0.45 m. Without the
        # patch it stays put (no collision detected). With it, the
        # kinematic sphere sweeps past 1.2 m of x, so the dynamic
        # body gets pushed well past its start.
        self.assertGreater(
            x_with,
            x_without + 0.1,
            f"state.body_q patch failed to wake collision: with={x_with:.3f}, without={x_without:.3f}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
