# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the local BikeTransmission example assets."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.phoenx.bike_transmission_scene import SCENE
from newton.examples.phoenx.example_phoenx_bike_transmission import (
    ASSETS,
    CHAIN_JOINT_FRICTION,
    DEFAULT_CADENCE_RPM,
    DEFAULT_REAR_LOAD_DAMPING,
    DERAILLEUR_DAMPING_SCALE,
    DERAILLEUR_PRELOAD_SCALE,
    DRIVETRAIN_CONTACT_FRICTION,
    Example,
)
from newton.viewer import ViewerNull


class TestBikeTransmission(unittest.TestCase):
    def test_loaded_chain_transmits_power_at_bicycle_cadence(self):
        """Keep the loaded 60 rpm drivetrain engaged and laterally bounded."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")
        if not (ASSETS / SCENE["shapes"][0]["mesh"]).is_file():
            self.skipTest("BikeTransmission meshes are local copyrighted assets")

        args = SimpleNamespace(
            cadence_rpm=DEFAULT_CADENCE_RPM,
            chain_joint_friction=CHAIN_JOINT_FRICTION,
            contact_chunk_size=64,
            derailleur_damping_scale=DERAILLEUR_DAMPING_SCALE,
            derailleur_preload_scale=DERAILLEUR_PRELOAD_SCALE,
            iterations=4,
            motor_off=False,
            rear_load_damping=DEFAULT_REAR_LOAD_DAMPING,
            sdf_resolution=0,
            sdf_voxel_depth_contacts=False,
            solver_stats=False,
            substeps=8,
        )
        example = Example(ViewerNull(), args)
        for _ in range(300):
            example.step()

        example.test_final()
        metrics = example.drivetrain_metrics()
        chain_y = example.state.body_q.numpy()[example.chain_bodies, 1]
        self.assertTrue(np.isfinite(tuple(metrics.values())).all())
        self.assertGreater(metrics["rear_load_power_w"], 3.0)
        self.assertGreater(metrics["speed_ratio"], 2.5)
        self.assertLess(metrics["speed_ratio"], 4.5)
        self.assertLess(float(np.ptp(chain_y)), 0.005)
        self.assertTrue(np.allclose(example.model.shape_material_mu.numpy(), DRIVETRAIN_CONTACT_FRICTION))


if __name__ == "__main__":
    unittest.main()
