# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise the contact-only chain fan at the requested even link count."""

import unittest

import numpy as np
import warp as wp

from newton.examples.phoenx.example_phoenx_chain_fan import Example
from newton.viewer import ViewerNull


class TestChainFan(unittest.TestCase):
    def test_forty_links_carry_their_pendants(self):
        """Swing four 40-link SDF chains without losing a link or pendant."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")
        args = Example.create_parser().parse_args(["--viewer", "null", "--chains", "4", "--links", "40"])
        scene = Example(ViewerNull(), args)
        self.assertEqual(scene.model.body_count, 4 * 41)
        self.assertTrue(all(len(chain) == 41 for chain in scene.chain_bodies))
        scene._check_pendants()
        self.assertFalse(scene.unlinked_pairs())
        self.assertGreater(np.linalg.norm(scene.state_0.body_qd.numpy()[:, :3]), 0.1)
        initial_z = scene.state_0.body_q.numpy()[:, 2].copy()
        for _ in range(180):
            scene.step()
            scene.test_post_step()
        scene.test_final()
        # Start at the beginning of the outward motion, where displacement is
        # still zero, and verify that graph replay advances the anchor targets.
        initial_anchors = scene.state_0.body_q.numpy()[scene.anchor_index, :3].copy()
        scene.sim_time = 6.0
        for _ in range(60):
            scene.step()
            scene.test_post_step()
        displacement = scene.state_0.body_q.numpy()[scene.anchor_index, :3] - initial_anchors
        np.testing.assert_allclose(np.linalg.norm(displacement[:, :2], axis=1), 0.0279296875, atol=2.0e-6)
        scene._check_anchors()
        self.assertTrue(np.isfinite(scene.state_0.body_qd.numpy()).all())
        pendants = [chain[-1] for chain in scene.chain_bodies]
        self.assertTrue(np.all(scene.state_0.body_q.numpy()[pendants, 2] < initial_z[pendants] - 0.25))
        self.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
        # Moving the last link away must invalidate both geometric attachments.
        poses = scene.state_0.body_q.numpy()
        poses[scene.chain_bodies[0][-2], 0] += 1.0
        scene.state_0.body_q.assign(poses)
        self.assertIn((0, 38), scene.unlinked_pairs())
        with self.assertRaisesRegex(AssertionError, "pendant"):
            scene._check_pendants()


if __name__ == "__main__":
    unittest.main(verbosity=2)
