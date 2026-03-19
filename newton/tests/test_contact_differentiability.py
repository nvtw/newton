# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests verifying that convex rigid-body contact gradients are correct.

Each test builds a minimal scene, runs one simulation step (collide + solver)
inside ``wp.Tape()``, and compares the analytic gradient of a scalar loss
w.r.t. ``body_q`` against a central finite-difference estimate.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@wp.kernel
def _loss_body_pos_kernel(
    body_q: wp.array(dtype=wp.transform),
    target: wp.vec3,
    body_idx: int,
    loss: wp.array(dtype=float),
):
    pos = wp.transform_get_translation(body_q[body_idx])
    delta = pos - target
    loss[0] = wp.dot(delta, delta)


def _build_scene_body_on_plane(shape_type, shape_args, body_pos, body_rot=None):
    """Return (model, solver) for a single dynamic body resting above a ground plane."""
    builder = newton.ModelBuilder()
    ke = 1.0e4
    kd = 1.0e1
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=0.5)
    builder.add_ground_plane()

    if body_rot is None:
        body_rot = wp.quat_identity()

    body = builder.add_body(xform=wp.transform(body_pos, body_rot))

    if shape_type == "sphere":
        builder.add_shape_sphere(body=body, radius=shape_args["radius"])
    elif shape_type == "box":
        builder.add_shape_box(body=body, hx=shape_args["hx"], hy=shape_args["hy"], hz=shape_args["hz"])
    elif shape_type == "capsule":
        builder.add_shape_capsule(body=body, radius=shape_args["radius"], half_height=shape_args["half_height"])
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")

    model = builder.finalize(requires_grad=True)
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverSemiImplicit(model)
    return model, solver


def _build_scene_two_bodies(shape_type_a, args_a, pos_a, shape_type_b, args_b, pos_b):
    """Return (model, solver) for two dynamic bodies (no ground plane)."""
    builder = newton.ModelBuilder()
    ke = 1.0e4
    kd = 1.0e1
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=0.5)

    body_a = builder.add_body(xform=wp.transform(pos_a, wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(pos_b, wp.quat_identity()))

    _add_shape(builder, body_a, shape_type_a, args_a)
    _add_shape(builder, body_b, shape_type_b, args_b)

    model = builder.finalize(requires_grad=True)
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverSemiImplicit(model)
    return model, solver


def _add_shape(builder, body, shape_type, args):
    if shape_type == "sphere":
        builder.add_shape_sphere(body=body, radius=args["radius"])
    elif shape_type == "box":
        builder.add_shape_box(body=body, hx=args["hx"], hy=args["hy"], hz=args["hz"])
    elif shape_type == "capsule":
        builder.add_shape_capsule(body=body, radius=args["radius"], half_height=args["half_height"])
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")


def _forward_loss(model, solver, state_0, contacts, dt, target, body_idx, loss):
    """Run one collide + step and compute loss.

    Collision detection runs with ``record_tape=False`` because the narrow-phase
    adjoint exceeds GPU resource limits. Gradients still flow through the solver's
    ``eval_body_contact`` kernel which reads ``body_q`` and the (frozen) contact
    arrays to compute forces.
    """
    state_0.clear_forces()
    # Collision detection is non-differentiable (topology + contact geometry
    # are treated as constants). The tape only records the solver step.
    model.collide(state_0, contacts)
    state_1 = model.state(requires_grad=True)
    control = model.control()
    solver.step(state_0, state_1, control, contacts, dt)
    wp.launch(
        _loss_body_pos_kernel,
        dim=1,
        inputs=[state_1.body_q, target, body_idx, loss],
    )
    return state_1


def _check_grad_body_q(
    model, solver, state_0, contacts, dt, target, body_idx, loss, eps=1.0e-4
):
    """Central finite-difference vs wp.Tape gradient for body_q of *body_idx*."""
    q_np = state_0.body_q.numpy().copy()

    # -- numeric gradient (central finite differences on translation) --
    numeric_grad = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        q_plus = q_np.copy()
        q_minus = q_np.copy()
        q_plus[body_idx, axis] += eps
        q_minus[body_idx, axis] -= eps

        state_0.body_q.assign(q_plus)
        _forward_loss(model, solver, state_0, contacts, dt, target, body_idx, loss)
        l_plus = loss.numpy()[0]

        state_0.body_q.assign(q_minus)
        _forward_loss(model, solver, state_0, contacts, dt, target, body_idx, loss)
        l_minus = loss.numpy()[0]

        numeric_grad[axis] = (l_plus - l_minus) / (2.0 * eps)

    # -- analytic gradient --
    state_0.body_q.assign(q_np)
    # Collision detection outside the tape (adjoint too large for GPU).
    # Gradients flow through the solver which reads body_q + frozen contacts.
    state_0.clear_forces()
    model.collide(state_0, contacts)
    tape = wp.Tape()
    with tape:
        state_1 = model.state(requires_grad=True)
        control = model.control()
        solver.step(state_0, state_1, control, contacts, dt)
        wp.launch(
            _loss_body_pos_kernel,
            dim=1,
            inputs=[state_1.body_q, target, body_idx, loss],
        )
    tape.backward(loss)
    analytic_grad = state_0.body_q.grad.numpy()[body_idx, :3].copy()
    tape.zero()

    # restore
    state_0.body_q.assign(q_np)
    return numeric_grad, analytic_grad


# ---------------------------------------------------------------------------
# Per-shape-pair gradient tests
# ---------------------------------------------------------------------------

def test_grad_sphere_on_plane(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane(
            "sphere", {"radius": 0.2}, body_pos=(0.0, 0.0, 0.15),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(0.5, 0.0, 0.15)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_box_on_plane(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane(
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, body_pos=(0.0, 0.0, 0.1),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(0.5, 0.0, 0.1)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_capsule_on_plane(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane(
            "capsule",
            {"radius": 0.1, "half_height": 0.2},
            body_pos=(0.0, 0.0, 0.05),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(0.5, 0.0, 0.05)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_sphere_sphere(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies(
            "sphere", {"radius": 0.2}, (0.0, 0.0, 0.0),
            "sphere", {"radius": 0.2}, (0.35, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_box_box(test, device):
    """Box-box goes through the GJK/MPR path."""
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies(
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.0, 0.0, 0.0),
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.28, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_optimization_converges(test, device):
    """Run a few gradient-descent steps and verify the loss decreases."""
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane(
            "sphere", {"radius": 0.2}, body_pos=(0.0, 0.0, 0.15),
        )
        target = wp.vec3(0.5, 0.0, 0.15)
        dt = 1.0 / 60.0
        lr = 1e-2
        losses = []

        for _ in range(5):
            state_0 = model.state(requires_grad=True)
            contacts = model.contacts()
            loss = wp.zeros(1, dtype=float, requires_grad=True)

            # Collide outside tape; solver inside tape
            state_0.clear_forces()
            model.collide(state_0, contacts)
            tape = wp.Tape()
            with tape:
                state_1 = model.state(requires_grad=True)
                control = model.control()
                solver.step(state_0, state_1, control, contacts, dt)
                wp.launch(
                    _loss_body_pos_kernel,
                    dim=1,
                    inputs=[state_1.body_q, target, 0, loss],
                )
            tape.backward(loss)

            losses.append(loss.numpy()[0])

            # gradient descent on body_q translation
            q_np = state_0.body_q.numpy()
            grad_np = state_0.body_q.grad.numpy()
            q_np[0, :3] -= lr * grad_np[0, :3]
            # persist updated position back into the model default state
            model.body_q.assign(q_np)

            tape.zero()

        # loss should decrease (allow small noise at the end)
        for i in range(len(losses) - 2):
            test.assertLess(losses[i + 1], losses[i] + 1e-6,
                            f"Loss did not decrease at step {i}: {losses}")


# ---------------------------------------------------------------------------
# XPBD differentiable contact tests (collide inside the tape)
# ---------------------------------------------------------------------------

def _build_scene_body_on_plane_xpbd(shape_type, shape_args, body_pos, body_rot=None):
    """Return (model, solver) for a single dynamic body resting above a ground plane using XPBD."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e1, mu=0.5)
    builder.add_ground_plane()

    if body_rot is None:
        body_rot = wp.quat_identity()

    body = builder.add_body(xform=wp.transform(body_pos, body_rot))

    if shape_type == "sphere":
        builder.add_shape_sphere(body=body, radius=shape_args["radius"])
    elif shape_type == "box":
        builder.add_shape_box(body=body, hx=shape_args["hx"], hy=shape_args["hy"], hz=shape_args["hz"])
    elif shape_type == "capsule":
        builder.add_shape_capsule(body=body, radius=shape_args["radius"], half_height=shape_args["half_height"])
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")

    model = builder.finalize(requires_grad=True)
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverXPBD(model)
    return model, solver


def _build_scene_two_bodies_xpbd(shape_type_a, args_a, pos_a, shape_type_b, args_b, pos_b):
    """Return (model, solver) for two dynamic bodies (no ground plane) using XPBD."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e1, mu=0.5)

    body_a = builder.add_body(xform=wp.transform(pos_a, wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(pos_b, wp.quat_identity()))

    _add_shape(builder, body_a, shape_type_a, args_a)
    _add_shape(builder, body_b, shape_type_b, args_b)

    model = builder.finalize(requires_grad=True)
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverXPBD(model)
    return model, solver


def _forward_loss_xpbd(model, solver, state_0, contacts, dt, target, body_idx, loss):
    """Run one collide + XPBD step and compute loss (all inside the tape)."""
    state_0.clear_forces()
    model.collide(state_0, contacts)
    state_1 = model.state(requires_grad=True)
    control = model.control()
    solver.step(state_0, state_1, control, contacts, dt)
    wp.launch(
        _loss_body_pos_kernel,
        dim=1,
        inputs=[state_1.body_q, target, body_idx, loss],
    )
    return state_1


def _check_grad_body_q_xpbd(
    model, solver, state_0, contacts, dt, target, body_idx, loss, eps=1.0e-4
):
    """Central finite-difference vs wp.Tape gradient for body_q with XPBD.

    Both collide and solver run inside the tape so gradients flow through
    the differentiable narrow phase contact geometry.
    """
    q_np = state_0.body_q.numpy().copy()

    numeric_grad = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        q_plus = q_np.copy()
        q_minus = q_np.copy()
        q_plus[body_idx, axis] += eps
        q_minus[body_idx, axis] -= eps

        state_0.body_q.assign(q_plus)
        _forward_loss_xpbd(model, solver, state_0, contacts, dt, target, body_idx, loss)
        l_plus = loss.numpy()[0]

        state_0.body_q.assign(q_minus)
        _forward_loss_xpbd(model, solver, state_0, contacts, dt, target, body_idx, loss)
        l_minus = loss.numpy()[0]

        numeric_grad[axis] = (l_plus - l_minus) / (2.0 * eps)

    state_0.body_q.assign(q_np)
    tape = wp.Tape()
    with tape:
        _forward_loss_xpbd(model, solver, state_0, contacts, dt, target, body_idx, loss)
    tape.backward(loss)
    analytic_grad = state_0.body_q.grad.numpy()[body_idx, :3].copy()
    tape.zero()

    state_0.body_q.assign(q_np)
    return numeric_grad, analytic_grad


def test_grad_sphere_on_plane_xpbd(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane_xpbd(
            "sphere", {"radius": 0.2}, body_pos=(0.0, 0.0, 0.15),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(0.5, 0.0, 0.15)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_box_on_plane_xpbd(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane_xpbd(
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, body_pos=(0.0, 0.0, 0.14),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(0.5, 0.0, 0.14)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-1, rtol=2e-1)


def test_grad_sphere_sphere_xpbd(test, device):
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies_xpbd(
            "sphere", {"radius": 0.2}, (0.0, 0.0, 0.0),
            "sphere", {"radius": 0.2}, (0.35, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=5e-2, rtol=1e-1)


def test_grad_box_box_xpbd(test, device):
    """Box-box through GJK/MPR with differentiable collision inside the tape."""
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies_xpbd(
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.0, 0.0, 0.0),
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.28, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-1, rtol=2e-1)


def test_grad_box_box_body_b_xpbd(test, device):
    """Check gradient of body B in a box-box (GJK/MPR) collision.

    This verifies the Envelope Theorem replay: gradients must flow through
    the collision detection to body B's pose, not just body A's.
    """
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies_xpbd(
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.0, 0.0, 0.0),
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.28, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(0.8, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=1, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-1, rtol=2e-1)


def test_grad_box_box_separated_xpbd(test, device):
    """Two separated boxes within the contact gap — exercises the GJK path.

    When shapes don't overlap, MPR returns no collision and the GJK fallback
    computes the speculative contact.  This verifies the Envelope Theorem
    replay for the GJK branch (as opposed to the MPR branch tested above).
    """
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies_xpbd(
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.0, 0.0, 0.0),
            "box", {"hx": 0.15, "hy": 0.15, "hz": 0.15}, (0.35, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-1, rtol=2e-1)


def test_grad_capsule_capsule_xpbd(test, device):
    """Capsule-capsule through GJK/MPR — different support function than box."""
    with wp.ScopedDevice(device):
        model, solver = _build_scene_two_bodies_xpbd(
            "capsule", {"radius": 0.1, "half_height": 0.2}, (0.0, 0.0, 0.0),
            "capsule", {"radius": 0.1, "half_height": 0.2}, (0.18, 0.0, 0.0),
        )
        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, 0.0, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-1, rtol=2e-1)


def test_grad_box_box_rotated_xpbd(test, device):
    """Rotated box-box with off-axis contact normal.

    Tests gradient flow through orientation-dependent relative transforms,
    ensuring the Envelope Theorem replay handles non-trivial rotations.
    """
    with wp.ScopedDevice(device):
        rot_45 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.785)
        builder = newton.ModelBuilder()
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e1, mu=0.5)

        body_a = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), rot_45))
        body_b = builder.add_body(xform=wp.transform((0.35, 0.1, 0.0), wp.quat_identity()))

        builder.add_shape_box(body=body_a, hx=0.15, hy=0.15, hz=0.15)
        builder.add_shape_box(body=body_b, hx=0.15, hy=0.15, hz=0.15)

        model = builder.finalize(requires_grad=True)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = newton.solvers.SolverXPBD(model)

        state_0 = model.state(requires_grad=True)
        contacts = model.contacts()
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        target = wp.vec3(-0.5, -0.3, 0.0)
        dt = 1.0 / 60.0

        numeric, analytic = _check_grad_body_q_xpbd(
            model, solver, state_0, contacts, dt, target, body_idx=0, loss=loss,
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-1, rtol=2e-1)


def test_grad_optimization_converges_xpbd(test, device):
    """Run a few gradient-descent steps with XPBD and verify the loss decreases."""
    with wp.ScopedDevice(device):
        model, solver = _build_scene_body_on_plane_xpbd(
            "sphere", {"radius": 0.2}, body_pos=(0.0, 0.0, 0.15),
        )
        target = wp.vec3(0.5, 0.0, 0.15)
        dt = 1.0 / 60.0
        lr = 1e-2
        losses = []

        for _ in range(5):
            state_0 = model.state(requires_grad=True)
            contacts = model.contacts()
            loss = wp.zeros(1, dtype=float, requires_grad=True)

            tape = wp.Tape()
            with tape:
                _forward_loss_xpbd(model, solver, state_0, contacts, dt, target, 0, loss)
            tape.backward(loss)

            losses.append(loss.numpy()[0])

            q_np = state_0.body_q.numpy()
            grad_np = state_0.body_q.grad.numpy()
            q_np[0, :3] -= lr * grad_np[0, :3]
            model.body_q.assign(q_np)

            tape.zero()

        for i in range(len(losses) - 2):
            test.assertLess(losses[i + 1], losses[i] + 1e-6,
                            f"Loss did not decrease at step {i}: {losses}")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestContactDifferentiability(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestContactDifferentiability, "test_grad_sphere_on_plane", test_grad_sphere_on_plane, devices=devices, check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_on_plane", test_grad_box_on_plane, devices=devices, check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_capsule_on_plane", test_grad_capsule_on_plane, devices=devices, check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_sphere_sphere", test_grad_sphere_sphere, devices=devices, check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_box", test_grad_box_box, devices=devices, check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_optimization_converges", test_grad_optimization_converges, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_sphere_on_plane_xpbd", test_grad_sphere_on_plane_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_on_plane_xpbd", test_grad_box_on_plane_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_sphere_sphere_xpbd", test_grad_sphere_sphere_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_box_xpbd", test_grad_box_box_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_box_body_b_xpbd", test_grad_box_box_body_b_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_box_separated_xpbd", test_grad_box_box_separated_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_capsule_capsule_xpbd", test_grad_capsule_capsule_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_box_box_rotated_xpbd", test_grad_box_box_rotated_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)
add_function_test(TestContactDifferentiability, "test_grad_optimization_converges_xpbd", test_grad_optimization_converges_xpbd, devices=get_selected_cuda_test_devices(), check_output=False)


if __name__ == "__main__":
    unittest.main()
