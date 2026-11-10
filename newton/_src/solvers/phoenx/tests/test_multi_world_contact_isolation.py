# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for PhoenX multi-world contact isolation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.viewer.picking import Picking
from newton.examples.robot.example_robot_h1 import Example


class _PickViewer:
    def __init__(self) -> None:
        self.picking: Picking | None = None

    def set_model(self, model) -> None:
        # All worlds overlap without viewer offsets, so only world 0 may be hit.
        first_world = wp.array(np.arange(model.world_count) == 0, dtype=int, device=model.device)
        self.picking = Picking(
            model,
            pick_stiffness=80.0,
            pick_damping=8.0,
            pick_max_acceleration=12.0,
            visible_worlds_mask=first_world,
        )

    def set_world_offsets(self, _offsets) -> None:
        pass

    def set_camera(self, **_kwargs) -> None:
        pass

    def apply_forces(self, state) -> None:
        assert self.picking is not None
        self.picking.apply_force(state)

    def activate_pick(
        self,
        state,
        *,
        body: int = 5,
        delta: tuple[float, float, float] = (2.0, 0.0, 1.0),
    ) -> None:
        """Grab ``body`` of world 0 with a ray and pull the grab point by ``delta`` [m]."""
        assert self.picking is not None
        origin = state.body_q.numpy()[body, :3]
        # Shoot through the body origin from whichever side is unobstructed.
        for direction in np.vstack((np.eye(3), -np.eye(3))).astype(np.float32):
            self.picking.pick(state, wp.vec3f(*(origin - 5.0 * direction)), wp.vec3f(*direction))
            if self.picking.pick_body == body:
                break
        assert self.picking.pick_body == body, f"ray picked body {self.picking.pick_body}, expected {body}"
        self.picking.pick_target = wp.vec3(*(np.array(self.picking.pick_point) + np.array(delta)))

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
