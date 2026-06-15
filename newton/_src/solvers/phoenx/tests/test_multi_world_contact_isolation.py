# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for PhoenX multi-world contact isolation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.viewer.kernels import PickingState
from newton._src.viewer.picking import Picking
from newton.examples.robot.example_robot_h1 import Example


class _PickViewer:
    def __init__(self) -> None:
        self.picking: Picking | None = None

    def set_model(self, model) -> None:
        self.picking = Picking(
            model,
            pick_stiffness=80.0,
            pick_damping=8.0,
            pick_max_acceleration=12.0,
        )

    def set_world_offsets(self, _offsets) -> None:
        pass

    def set_camera(self, **_kwargs) -> None:
        pass

    def apply_forces(self, state) -> None:
        assert self.picking is not None
        self.picking._apply_picking_force(state)

    def activate_pick(
        self,
        state,
        *,
        body: int = 5,
        delta: tuple[float, float, float] = (2.0, 0.0, 1.0),
    ) -> None:
        assert self.picking is not None
        body_q = state.body_q.numpy()
        hit = body_q[body, :3].astype(np.float32)

        pick_state = np.empty(1, dtype=PickingState.numpy_dtype())
        pick_state[0]["picked_point_local"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pick_state[0]["picked_point_world"] = hit
        pick_state[0]["picking_target_world"] = hit + np.array(delta, dtype=np.float32)
        pick_state[0]["pick_stiffness"] = 80.0
        pick_state[0]["pick_damping"] = 8.0
        pick_state[0]["pick_max_acceleration"] = 12.0

        self.picking.pick_state.assign(pick_state)
        self.picking.pick_body.assign(np.array([body], dtype=np.int32))
        self.picking.picking_active = True

    def release_pick(self) -> None:
        assert self.picking is not None
        self.picking.release()


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX multi-world contact isolation requires CUDA")
class TestPhoenXMultiWorldContactIsolation(unittest.TestCase):
    def test_reused_contact_generation_warmstart_is_world_local(self) -> None:
        args = SimpleNamespace(world_count=8, solver="phoenx", use_mujoco_contacts=False)
        baseline_viewer = _PickViewer()
        picked_viewer = _PickViewer()
        baseline = Example(baseline_viewer, args)
        picked = Example(picked_viewer, args)

        picked_viewer.activate_pick(picked.state_0)

        body_world = baseline.model.body_world.numpy()
        higher_worlds = body_world > 0
        first_world = body_world <= 0

        max_first_world_delta = 0.0
        max_higher_world_delta = 0.0
        baseline_contact_count = None
        min_picked_contact_count = None

        for frame in range(20):
            if frame == 6:
                picked_viewer.activate_pick(picked.state_0, delta=(-2.0, 1.0, 0.5))
            if frame == 12:
                picked_viewer.release_pick()

            baseline.step()
            picked.step()

            baseline_count = int(baseline.contacts.rigid_contact_count.numpy()[0])
            picked_count = int(picked.contacts.rigid_contact_count.numpy()[0])
            if baseline_contact_count is None:
                baseline_contact_count = baseline_count
                min_picked_contact_count = picked_count
            else:
                min_picked_contact_count = min(min_picked_contact_count, picked_count)

            baseline_pos = baseline.state_0.body_q.numpy()[:, :3]
            picked_pos = picked.state_0.body_q.numpy()[:, :3]

            max_first_world_delta = max(
                max_first_world_delta,
                float(np.max(np.abs(baseline_pos[first_world] - picked_pos[first_world]))),
            )
            max_higher_world_delta = max(
                max_higher_world_delta,
                float(np.max(np.abs(baseline_pos[higher_worlds] - picked_pos[higher_worlds]))),
            )

        assert baseline_contact_count is not None
        assert min_picked_contact_count is not None
        self.assertLess(min_picked_contact_count, baseline_contact_count)
        self.assertGreater(max_first_world_delta, 0.1)
        self.assertLessEqual(max_higher_world_delta, 1.0e-7)


if __name__ == "__main__":
    unittest.main()
