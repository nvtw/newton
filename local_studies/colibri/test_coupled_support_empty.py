"""CPU edge-case regression for a native phase with no eligible contacts."""

import importlib
import os
import unittest
import numpy as np

reference = importlib.import_module(
    os.environ.get("COUPLED_REFERENCE_MODULE", "local_studies.colibri.coupled_support_online")
)


class TestEmptyCoupledContacts(unittest.TestCase):
    def test_empty_outer(self):
        """An empty contact system returns no impulses without optimization."""
        lam, report = reference.solve_online(np.zeros((0, 0)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        self.assertEqual(lam.shape, (0,))
        self.assertTrue(report["accepted"])
        self.assertEqual(report["stages"], [])

    def test_joint_only_relax_preserves_drive(self):
        """Separated contacts remain fixed while the actual hinge and drive solve."""
        saved = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
        d = {k.split(".", 1)[1]: saved[k].copy() for k in saved.files if k.startswith("relax_solved.")}
        d["derived"][3, :] = 1.0
        a = reference.assemble_snapshot(d, "relax", float(saved["dt"][0]), int(saved["num_joints"][0]))
        self.assertEqual(a["C"].shape, (0, 12))
        self.assertEqual(a["A"].shape, (0, 0))
        self.assertEqual(a["rhs"].shape, (0,))
        self.assertEqual(len(a["points"]), 0)
        self.assertTrue(np.any(a["diagonal"] > 0))
        joint = np.linalg.solve(a["K"], a["targets"] - a["B"] @ a["free"])
        v = a["free"] + a["W"] @ a["B"].T @ joint
        np.testing.assert_allclose(a["B"] @ v + a["diagonal"] * joint, a["targets"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(v, a["velocity"] + a["W"] @ a["B"].T @ (joint - a["old_joint"]), atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
