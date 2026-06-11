# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression for high-friction humanoid contact relax stability."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.examples.example_humanoid import DEFAULT_CONTACT_GAP, Example
from newton._src.viewer.kernels import PickingState
from newton._src.viewer.picking import Picking


class _NullViewer:
    def set_model(self, model) -> None:
        self.model = model

    def set_world_offsets(self, offsets) -> None:
        self.offsets = offsets

    def set_camera(self, **_kwargs) -> None:
        pass

    def apply_forces(self, state) -> None:
        pass


def _body_com_by_world(model, state, world_count: int) -> np.ndarray:
    body_count = int(model.body_count)
    bodies_per_world = body_count // int(world_count)
    q = state.body_q.numpy()[:, :3].reshape(world_count, bodies_per_world, 3)
    mass = model.body_mass.numpy().reshape(world_count, bodies_per_world)
    return (q * mass[:, :, None]).sum(axis=1) / mass.sum(axis=1)[:, None]


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX humanoid contact test requires CUDA")
class TestPhoenXHumanoidContactRelax(unittest.TestCase):
    def test_example_default_gap_limits_passive_drop_slide(self) -> None:
        args = SimpleNamespace(
            world_count=8,
            show_contacts=False,
            seed=123,
            stress_random_poses=False,
            joint_random_range=1.0,
            root_height=1.4,
            self_collisions=False,
            contact_gap=DEFAULT_CONTACT_GAP,
            solver_substeps=10,
            solver_iterations=8,
            velocity_iterations=1,
            prepare_refresh_stride=1,
        )
        example = Example(_NullViewer(), args)
        np.testing.assert_allclose(example.model.shape_gap.numpy(), DEFAULT_CONTACT_GAP, atol=1.0e-7)

        com0 = _body_com_by_world(example.model, example.state_0, args.world_count)
        for _ in range(180):
            example.step()

        q = example.state_0.body_q.numpy()
        qd = example.state_0.body_qd.numpy()
        self.assertTrue(np.isfinite(q).all() and np.isfinite(qd).all())

        com = _body_com_by_world(example.model, example.state_0, args.world_count)
        drift = np.linalg.norm(com[:, :2] - com0[:, :2], axis=1)
        self.assertLess(float(drift.mean()), 0.75)
        self.assertLess(float(drift.max()), 1.1)

    def test_speculative_contact_relax_does_not_launch_humanoid(self) -> None:
        builder_humanoid = newton.ModelBuilder()
        builder_humanoid.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=["floor", "ground"],
            up_axis="Z",
            parse_sites=False,
            enable_self_collisions=False,
        )
        builder_humanoid.joint_q[:7] = [0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0]

        builder = newton.ModelBuilder()
        builder.add_world(builder_humanoid)
        builder.add_ground_plane()
        model = builder.finalize()

        state = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=50,
            solver_iterations=50,
            velocity_iterations=4,
            prepare_refresh_stride="auto",
        )
        contacts = model.contacts()

        for frame in range(80):
            model.collide(state, contacts)
            state.clear_forces()
            solver.step(state, state, control, contacts, 1.0 / 60.0)

            q = state.body_q.numpy()
            qd = state.body_qd.numpy()
            finite = np.isfinite(q).all() and np.isfinite(qd).all()
            max_qd = float(np.nanmax(np.abs(qd)))
            self.assertTrue(finite, f"non-finite humanoid state at frame {frame}")
            self.assertLess(max_qd, 200.0, f"humanoid launched at frame {frame}: max |qd|={max_qd:.3f}")

    def test_picking_wrench_does_not_nan_humanoid_joints(self) -> None:
        builder_humanoid = newton.ModelBuilder()
        builder_humanoid.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=["floor", "ground"],
            up_axis="Z",
            parse_sites=False,
            enable_self_collisions=False,
        )
        builder_humanoid.joint_q[:7] = [0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0]

        builder = newton.ModelBuilder()
        builder.add_world(builder_humanoid)
        model = builder.finalize()

        state = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=50,
            solver_iterations=50,
            velocity_iterations=1,
            prepare_refresh_stride="auto",
        )
        picking = Picking(model)

        body = 5
        q = state.body_q.numpy()
        local = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hit = q[body, :3]
        pick_state = np.empty(1, dtype=PickingState.numpy_dtype())
        pick_state[0]["picked_point_local"] = local
        pick_state[0]["picked_point_world"] = hit
        pick_state[0]["picking_target_world"] = hit + np.array([2.0, 0.0, 1.0], dtype=np.float32)
        pick_state[0]["pick_stiffness"] = 50.0
        pick_state[0]["pick_damping"] = 5.0
        pick_state[0]["pick_max_acceleration"] = 1.0
        picking.pick_state.assign(pick_state)
        picking.pick_body.assign(np.array([body], dtype=np.int32))
        picking.picking_active = True

        for frame in range(80):
            if frame % 10 == 0:
                target = pick_state[0]["picking_target_world"]
                target[1] = np.float32(np.sin(frame * 0.05) * 1.5)
                pick_state[0]["picking_target_world"] = target
                picking.pick_state.assign(pick_state)

            state.clear_forces()
            picking._apply_picking_force(state)
            solver.step(state, state, control, None, 1.0 / 60.0)

            q = state.body_q.numpy()
            qd = state.body_qd.numpy()
            finite = np.isfinite(q).all() and np.isfinite(qd).all()
            self.assertTrue(finite, f"non-finite humanoid pick state at frame {frame}")
            self.assertLess(float(np.max(np.abs(qd))), 1000.0)


if __name__ == "__main__":
    unittest.main()
