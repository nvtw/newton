# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise Colibri gear contacts at the authored collision update rate."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples
from newton.examples.kamino.example_kamino_colibri import Example as AssemblyChecks
from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Phoenx Colibri requires CUDA")
class TestColibriContacts(unittest.TestCase):
    def _make_example(self, num_worlds=1, fix_base=False):
        assets = Path(newton.examples.get_asset_directory()) / "colibri"
        if not assets.is_dir():
            self.skipTest("Colibri mesh assets are not installed")
        args = Example.create_parser().parse_args(
            [
                "--num-worlds",
                str(num_worlds),
                "--contact-updates-per-frame",
                "2",
                "--substeps",
                "24",
                "--counterweight-density-scale",
                "1.0",
            ]
        )
        args.fix_base = fix_base
        return Example(ViewerNull(), args)

    def test_replicated_fixed_roots_match_ordered_sweep(self):
        """Match the ordered reference for world-anchor joints and a partial grid."""
        example = self._make_example(3, fix_base=True)
        reference = self._make_example(3, fix_base=True)
        reference.solver.world._temporal_sweep_worlds = 1
        with wp.ScopedCapture(device=reference.model.device) as capture:
            reference.simulate()
        reference.graph = capture.graph
        # The existing strict fixed-base pose tolerance fails in both paths.
        # Preserve that diagnostic; this checks scheduling parity instead.
        for _ in range(60):
            example.step()
            reference.step()
            for name in ("body_q", "body_qd"):
                np.testing.assert_array_equal(
                    getattr(example.state_0, name).numpy().view(np.uint32),
                    getattr(reference.state_0, name).numpy().view(np.uint32),
                )

    def test_replicated_worlds_keep_contacts_local(self):
        """Replicate at the origin, separate only in the viewer, and audit every world."""
        example = self._make_example(4)
        self.assertEqual(example.model.world_count, 4)
        q = example.initial_q.reshape(4, -1, 7)
        for world in range(1, 4):
            np.testing.assert_array_equal(q[world], q[0])
        offsets = example.viewer.world_offsets.numpy()
        self.assertEqual(len(np.unique(offsets[:, 0])), 2)
        self.assertEqual(len(np.unique(offsets[:, 1])), 2)
        worlds = example.model.shape_world.numpy()
        for _ in range(60):
            example.step()
            example.test_post_step()
            contacts = example.contacts
            count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0)
            self.assertLess(count, contacts.rigid_contact_max)
            first = worlds[contacts.rigid_contact_shape0.numpy()[:count]]
            second = worlds[contacts.rigid_contact_shape1.numpy()[:count]]
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(np.unique(first), np.arange(4))

    def test_contact_visualization_uses_snapshot(self):
        """Generate contact arrows from a snapshot after live buffers change."""
        example = self._make_example(4)
        example.step()
        example._render_states = (example.model.state(), example.model.state())
        example.viewer.show_contacts = True
        example.viewer.show_contact_disks = False
        example.viewer.show_contact_forces = False
        example.prepare_render_state()
        count = int(example.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        example.contacts.rigid_contact_count.zero_()
        with patch.object(example.viewer, "log_arrows") as arrows:
            example.render()
        normals = [call for call in arrows.call_args_list if call.args[0] == "/contacts/normals"]
        self.assertEqual(len(normals), 1)
        starts, ends = normals[0].args[1:3]
        self.assertIsNotNone(starts)
        self.assertEqual(len(starts), count)
        self.assertTrue(np.isfinite(starts.numpy()).all())
        self.assertTrue(np.isfinite(ends.numpy()).all())
        self.assertTrue(np.any(starts.numpy() != ends.numpy()))

    def test_gear_pins_at_120_hz(self):
        """Keep the gear engaged during the initial passive-frame motion."""
        example = self._make_example()
        # Sticky normal geometry previously crossed 1 mm after 0.47 seconds.
        # Check the whole trajectory: final depths can hide a skipped pin.
        for _ in range(60):
            example.step()
            example.test_post_step()

    def test_gear_pins_for_one_minute_at_120_hz(self):
        """Reject intermittent pin crossings without accepting a stalled mechanism."""
        example = self._make_example()
        for _ in range(3600):
            example.step()
            AssemblyChecks.test_post_step(example)
            example._test_drive_tracking()
            depth, labels, _gear_depth = example._measure_contact_penetration()
            self.assertLess(
                depth,
                0.001,
                f"Penetration at {example.sim_time:.6f} s between {labels}: {depth:.6f} m",
            )
        # Support stationarity remains a separate, currently failing acceptance
        # check. Do not disable or relax it in the example to pass this test.


if __name__ == "__main__":
    unittest.main()
